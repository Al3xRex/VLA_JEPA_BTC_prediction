from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import math
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_WORLD_NAMES = ("structure", "environment", "edges", "movement", "liquidation")


@dataclass(frozen=True)
class EMAMomentumSchedule:
    """Cosine EMA schedule parameterized in optimizer-step units.

    ``initial_half_life_steps`` is easier to calibrate to a finite training run
    than copying a momentum constant from a much longer pretraining schedule.
    A half-life of ``n`` means that, at the start of training, the cumulative
    coefficient on a target parameter value is halved after ``n`` updates when
    the online value is held fixed.
    """

    total_optimizer_steps: int
    initial_half_life_steps: float
    final_momentum: float = 1.0

    def __post_init__(self) -> None:
        if int(self.total_optimizer_steps) <= 0:
            raise ValueError("total_optimizer_steps must be positive.")
        if not math.isfinite(float(self.initial_half_life_steps)) or float(
            self.initial_half_life_steps
        ) <= 0.0:
            raise ValueError("initial_half_life_steps must be finite and positive.")
        if not math.isfinite(float(self.final_momentum)) or not 0.0 <= float(
            self.final_momentum
        ) <= 1.0:
            raise ValueError("final_momentum must be finite and lie in [0, 1].")
        if float(self.final_momentum) < self.initial_momentum:
            raise ValueError("final_momentum cannot be smaller than initial_momentum.")

    @property
    def initial_momentum(self) -> float:
        return float(0.5 ** (1.0 / float(self.initial_half_life_steps)))

    def momentum_at(self, optimizer_step: int) -> float:
        """Return momentum for a zero-based optimizer step."""

        step = int(optimizer_step)
        total = int(self.total_optimizer_steps)
        if step < 0 or step >= total:
            raise ValueError(f"optimizer_step must be in [0, {total - 1}].")
        progress = 0.0 if total == 1 else step / float(total - 1)
        cosine_weight = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.initial_momentum + (
            float(self.final_momentum) - self.initial_momentum
        ) * cosine_weight


@dataclass
class WorldJEPAOutput:
    """Typed output for one world's causal JEPA encoder.

    ``padding_mask`` follows the PyTorch convention: ``True`` means that a
    timestep is padding and must not be attended to. ``feature_mask`` uses the
    opposite convention: ``True`` means that a feature was observed.
    """

    context_tokens: torch.Tensor
    context_summary: torch.Tensor
    predicted_target: torch.Tensor
    target_latent: torch.Tensor | None
    regime_logits: torch.Tensor
    regime_probabilities: torch.Tensor
    uncertainty: torch.Tensor
    valid_context_fraction: torch.Tensor
    disagreement: torch.Tensor
    state_token: torch.Tensor
    regime_token: torch.Tensor
    target_valid_mask: torch.Tensor | None
    diagnostics: dict[str, torch.Tensor]

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self) -> Iterator[str]:
        return (field.name for field in fields(self))


def _sinusoidal_positions(length: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    """Return deterministic positions without imposing a maximum sequence length."""

    if length <= 0:
        return reference.new_zeros((1, 0, width))
    compute_dtype = torch.float32 if reference.dtype in {torch.float16, torch.bfloat16} else reference.dtype
    position = torch.arange(length, device=reference.device, dtype=compute_dtype).unsqueeze(1)
    even_width = (width + 1) // 2
    divisor = torch.exp(
        torch.arange(even_width, device=reference.device, dtype=compute_dtype)
        * (-math.log(10_000.0) / max(width, 1))
        * 2.0
    )
    phase = position * divisor.unsqueeze(0)
    encoded = torch.zeros((length, width), device=reference.device, dtype=compute_dtype)
    encoded[:, 0::2] = torch.sin(phase[:, : encoded[:, 0::2].shape[1]])
    if width > 1:
        encoded[:, 1::2] = torch.cos(phase[:, : encoded[:, 1::2].shape[1]])
    return encoded.to(dtype=reference.dtype).unsqueeze(0)


def _normalize_padding_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.zeros((batch_size, time_steps), dtype=torch.bool, device=device)
    normalized = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if normalized.ndim == 1:
        if normalized.shape[0] != time_steps:
            raise ValueError(f"Expected padding_mask [{time_steps}], got {tuple(normalized.shape)}.")
        normalized = normalized.unsqueeze(0).expand(batch_size, -1)
    elif normalized.ndim == 2:
        if normalized.shape == (1, time_steps):
            normalized = normalized.expand(batch_size, -1)
        elif normalized.shape != (batch_size, time_steps):
            raise ValueError(
                f"Expected padding_mask [{batch_size}, {time_steps}], got {tuple(normalized.shape)}."
            )
    else:
        raise ValueError("padding_mask must have shape [T] or [B, T].")
    return normalized


def _normalize_feature_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    batch_size, time_steps, feature_count = x.shape
    finite = torch.isfinite(x)
    if mask is None:
        return finite
    normalized = torch.as_tensor(mask, device=x.device, dtype=torch.bool)
    if normalized.ndim == 1:
        if normalized.shape[0] != feature_count:
            raise ValueError(f"Expected feature_mask [{feature_count}], got {tuple(normalized.shape)}.")
        normalized = normalized.view(1, 1, feature_count)
    elif normalized.ndim == 2:
        if normalized.shape == (batch_size, feature_count):
            normalized = normalized.unsqueeze(1)
        elif normalized.shape == (time_steps, feature_count):
            normalized = normalized.unsqueeze(0)
        else:
            raise ValueError(
                "A 2D feature_mask must have shape [B, F] or [T, F]; "
                f"got {tuple(normalized.shape)}."
            )
    elif normalized.ndim != 3:
        raise ValueError("feature_mask must have shape [F], [B, F], [T, F], or [B, T, F].")
    try:
        normalized = normalized.expand(batch_size, time_steps, feature_count)
    except RuntimeError as error:
        raise ValueError(
            f"feature_mask {tuple(normalized.shape)} cannot broadcast to {tuple(x.shape)}."
        ) from error
    return normalized & finite


def _normalize_ids(
    values: torch.Tensor | int | None,
    *,
    batch_size: int,
    upper_bound: int,
    default: int,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        result = torch.full((batch_size,), int(default), dtype=torch.long, device=device)
    else:
        result = torch.as_tensor(values, dtype=torch.long, device=device)
        if result.ndim == 0:
            result = result.expand(batch_size)
        elif result.ndim == 1 and result.shape[0] == 1:
            result = result.expand(batch_size)
        elif result.ndim != 1 or result.shape[0] != batch_size:
            raise ValueError(f"Expected ids with shape [{batch_size}], got {tuple(result.shape)}.")
    return result.clamp(0, max(int(upper_bound) - 1, 0))


class _CausalWorldBackbone(nn.Module):
    """Masked causal encoder with explicit state and regime tokens.

    State and regime tokens are appended after the temporal sequence. Under a
    causal attention mask they can summarize every prior valid timestep while
    temporal token ``t`` can never inspect token ``t + 1``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_worlds: int,
        num_states: int,
        num_regimes: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.world_embedding = nn.Embedding(max(int(num_worlds), 1), self.hidden_dim)
        self.state_embedding = nn.Embedding(max(int(num_states), 1), self.hidden_dim)
        self.regime_embedding = nn.Embedding(max(int(num_regimes), 1), self.hidden_dim)
        self.state_token_parameter = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.regime_token_parameter = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.token_projection = nn.Linear(self.hidden_dim, self.latent_dim)
        self.summary_projection = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
        feature_mask: torch.Tensor | None,
        world_ids: torch.Tensor,
        state_ids: torch.Tensor,
        regime_ids: torch.Tensor,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, F], got {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[-1]}.")
        batch_size, time_steps, _ = x.shape
        observed = _normalize_feature_mask(x, feature_mask)
        time_padding = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            time_steps=time_steps,
            device=x.device,
        )
        observed = observed & ~time_padding.unsqueeze(-1)
        # An all-missing timestep must not become an attention key merely because
        # its numeric placeholder happens to be zero.
        time_padding = time_padding | ~observed.any(dim=-1)
        observed = observed & ~time_padding.unsqueeze(-1)
        clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        clean = torch.where(observed, clean, torch.zeros_like(clean))
        temporal = self.input_projection(torch.cat([clean, observed.to(dtype=x.dtype)], dim=-1))
        temporal = temporal + _sinusoidal_positions(time_steps, self.hidden_dim, temporal)

        world_context = self.world_embedding(world_ids).unsqueeze(1)
        state_token = self.state_token_parameter.expand(batch_size, -1, -1)
        state_token = state_token + world_context + self.state_embedding(state_ids).unsqueeze(1)
        regime_token = self.regime_token_parameter.expand(batch_size, -1, -1)
        regime_token = regime_token + world_context + self.regime_embedding(regime_ids).unsqueeze(1)
        sequence = torch.cat([temporal, state_token, regime_token], dim=1)

        special_padding = torch.zeros((batch_size, 2), dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([time_padding, special_padding], dim=1)
        sequence_length = time_steps + 2
        causal_mask = (
            torch.triu(
                torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            if causal
            else None
        )
        encoded = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        encoded = self.output_norm(encoded)
        temporal_encoded = self.token_projection(encoded[:, :time_steps, :])
        temporal_encoded = torch.where(
            (~time_padding).unsqueeze(-1),
            temporal_encoded,
            torch.zeros_like(temporal_encoded),
        )
        state_encoded = self.summary_projection(encoded[:, time_steps, :])
        regime_encoded = self.summary_projection(encoded[:, time_steps + 1, :])
        valid_context_fraction = (~time_padding).to(dtype=x.dtype).mean(dim=1)
        valid_capacity = (~time_padding).sum(dim=1).clamp_min(1) * self.input_dim
        valid_feature_fraction = observed.sum(dim=(1, 2)).to(dtype=x.dtype) / valid_capacity.to(dtype=x.dtype)
        return (
            temporal_encoded,
            state_encoded,
            regime_encoded,
            time_padding,
            valid_context_fraction,
            valid_feature_fraction,
        )


class _ContinuousHorizonEncoder(nn.Module):
    def __init__(self, output_dim: int, fourier_bands: int = 8) -> None:
        super().__init__()
        self.fourier_bands = max(int(fourier_bands), 1)
        frequencies = torch.pow(2.0, torch.arange(self.fourier_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=True)
        feature_dim = 3 + 2 * self.fourier_bands
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, horizons: torch.Tensor) -> torch.Tensor:
        horizons = horizons.to(dtype=self.frequencies.dtype).clamp_min(0.0)
        log_horizon = torch.log1p(horizons)
        scaled = log_horizon.unsqueeze(-1) * self.frequencies.to(device=horizons.device)
        features = torch.cat(
            [
                log_horizon.unsqueeze(-1),
                torch.sqrt(horizons + 1.0).unsqueeze(-1),
                (1.0 / (horizons + 1.0)).unsqueeze(-1),
                torch.sin(scaled),
                torch.cos(scaled),
            ],
            dim=-1,
        )
        projection_dtype = self.projection[0].weight.dtype
        return self.projection(features.to(dtype=projection_dtype))


class WorldJEPAEncoder(nn.Module):
    """A causal, mask-aware JEPA for one of an arbitrary set of worlds.

    The online/context encoder is optimized by gradient descent. The target
    encoder is always evaluation-only and is updated explicitly through
    :meth:`update_target_encoder` using exponential moving averages.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        feature_dim: int | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        horizon_dim: int = 32,
        num_regimes: int = 8,
        num_states: int = 32,
        world_names: Sequence[str] = DEFAULT_WORLD_NAMES,
        world_name: str | None = None,
        ema_momentum: float = 0.996,
        # Config-file aliases.  Keeping them explicit makes
        # ``WorldJEPAEncoder(input_dim=..., **config.encoder)`` safe.
        model_dimension: int | None = None,
        latent_dimension: int | None = None,
        attention_heads: int | None = None,
        transformer_layers: int | None = None,
        ema_decay: float | None = None,
        feature_mask_probability: float = 0.0,
        time_mask_probability: float = 0.0,
        teacher_target_normalization: str = "layernorm",
        teacher_target_normalization_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        resolved_input_dim = input_dim if input_dim is not None else feature_dim
        if resolved_input_dim is None:
            raise ValueError("input_dim (or feature_dim) is required.")
        hidden_dim = int(model_dimension) if model_dimension is not None else int(hidden_dim)
        latent_dim = int(latent_dimension) if latent_dimension is not None else int(latent_dim)
        num_heads = int(attention_heads) if attention_heads is not None else int(num_heads)
        num_layers = int(transformer_layers) if transformer_layers is not None else int(num_layers)
        ema_momentum = float(ema_decay) if ema_decay is not None else float(ema_momentum)
        names = tuple(str(name) for name in world_names)
        if not names:
            raise ValueError("world_names cannot be empty.")
        if len(set(names)) != len(names):
            raise ValueError("world_names must be unique.")
        if world_name is not None and str(world_name) not in names:
            names = (*names, str(world_name))
        self.input_dim = int(resolved_input_dim)
        self.latent_dim = int(latent_dim)
        self.num_regimes = int(num_regimes)
        self.num_states = int(num_states)
        self.world_names = names
        self.world_to_id = {name: idx for idx, name in enumerate(names)}
        self.world_name = str(world_name) if world_name is not None else names[0]
        self.default_world_id = self.world_to_id[self.world_name]
        self.ema_momentum = float(ema_momentum)
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError("ema_momentum must be in [0, 1].")
        self.feature_mask_probability = float(feature_mask_probability)
        self.time_mask_probability = float(time_mask_probability)
        if not 0.0 <= self.feature_mask_probability < 1.0:
            raise ValueError("feature_mask_probability must be in [0, 1).")
        if not 0.0 <= self.time_mask_probability < 1.0:
            raise ValueError("time_mask_probability must be in [0, 1).")
        self.teacher_target_normalization = str(teacher_target_normalization).lower()
        if self.teacher_target_normalization not in {"layernorm", "l2", "none"}:
            raise ValueError(
                "teacher_target_normalization must be one of 'layernorm', 'l2', or 'none'."
            )
        self.teacher_target_normalization_eps = float(teacher_target_normalization_eps)
        if (
            not math.isfinite(self.teacher_target_normalization_eps)
            or self.teacher_target_normalization_eps <= 0.0
        ):
            raise ValueError("teacher_target_normalization_eps must be finite and positive.")

        backbone_kwargs = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": float(dropout),
            "num_worlds": len(self.world_names),
            "num_states": self.num_states,
            "num_regimes": self.num_regimes,
        }
        self.context_encoder = _CausalWorldBackbone(**backbone_kwargs)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.horizon_encoder = _ContinuousHorizonEncoder(int(horizon_dim))
        predictor_input = self.latent_dim * 2 + int(horizon_dim)
        self.predictor_trunk = nn.Sequential(
            nn.Linear(predictor_input, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.target_head = nn.Linear(int(hidden_dim), self.latent_dim)
        self.uncertainty_head = nn.Linear(int(hidden_dim), 1)
        self.regime_head = nn.Linear(self.latent_dim, self.num_regimes)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        encoder_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> "WorldJEPAEncoder":
        """Construct directly from the ``encoder`` section of world_jepa.json."""

        return cls(input_dim=input_dim, **dict(encoder_config), **kwargs)

    def _freeze_target_encoder(self) -> None:
        self.target_encoder.eval()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "WorldJEPAEncoder":
        super().train(mode)
        # ``Module.train`` recursively toggles children, so restore the target
        # branch after every mode change.
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """EMA-update the stop-gradient target encoder from the online encoder."""

        decay = self.ema_momentum if momentum is None else float(momentum)
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1].")
        online_parameters = dict(self.context_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            online = online_parameters[name]
            target_parameter.mul_(decay).add_(online, alpha=1.0 - decay)
        online_buffers = dict(self.context_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            online = online_buffers[name]
            if target_buffer.dtype.is_floating_point:
                target_buffer.mul_(decay).add_(online, alpha=1.0 - decay)
            else:
                target_buffer.copy_(online)
        self._freeze_target_encoder()

    @torch.no_grad()
    def update_target_encoder_for_step(
        self,
        optimizer_step: int,
        schedule: EMAMomentumSchedule,
    ) -> float:
        """Update the target encoder with a step-calibrated schedule.

        Returns the applied momentum so callers can persist it in training
        diagnostics. The optimizer must be stepped before invoking this method.
        """

        if not isinstance(schedule, EMAMomentumSchedule):
            raise TypeError("schedule must be an EMAMomentumSchedule.")
        momentum = schedule.momentum_at(optimizer_step)
        self.update_target_encoder(momentum=momentum)
        return momentum

    def _normalize_teacher_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.teacher_target_normalization == "none":
            return targets
        if self.teacher_target_normalization == "l2":
            return F.normalize(
                targets,
                p=2.0,
                dim=-1,
                eps=self.teacher_target_normalization_eps,
            )
        return F.layer_norm(
            targets,
            (self.latent_dim,),
            eps=self.teacher_target_normalization_eps,
        )

    def _normalize_horizons(self, horizons: torch.Tensor | Sequence[float], batch_size: int, device: torch.device) -> torch.Tensor:
        values = torch.as_tensor(horizons, dtype=torch.float32, device=device)
        if values.ndim == 0:
            values = values.reshape(1, 1).expand(batch_size, 1)
        elif values.ndim == 1:
            values = values.unsqueeze(0).expand(batch_size, -1)
        elif values.ndim == 2:
            if values.shape[0] == 1:
                values = values.expand(batch_size, -1)
            elif values.shape[0] != batch_size:
                raise ValueError(f"Expected horizons [H] or [{batch_size}, H], got {tuple(values.shape)}.")
        else:
            raise ValueError("horizons must have shape [H] or [B, H].")
        if not torch.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("horizons must be finite and strictly positive.")
        return values

    def _prepare_context_masks(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
        feature_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = x.shape
        effective_padding = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            time_steps=time_steps,
            device=x.device,
        )
        if time_mask is not None:
            included = torch.as_tensor(time_mask, dtype=torch.bool, device=x.device)
            if included.ndim == 1:
                included = included.unsqueeze(0).expand(batch_size, -1)
            elif included.ndim == 2 and included.shape[0] == 1:
                included = included.expand(batch_size, -1)
            if included.shape != (batch_size, time_steps):
                raise ValueError("time_mask must have shape [T] or [B, T], where True means observed.")
            effective_padding = effective_padding | ~included
        effective_features = _normalize_feature_mask(x, feature_mask)
        effective_features = effective_features & ~effective_padding.unsqueeze(-1)

        if self.training and self.time_mask_probability > 0.0:
            valid_before_mask = ~effective_padding
            dropped = (
                torch.rand((batch_size, time_steps), device=x.device) < self.time_mask_probability
            ) & valid_before_mask
            # Never mask every valid timestep in a sample: the most recent
            # originally valid row remains available as an anchor.
            for batch_index in range(batch_size):
                valid_positions = torch.nonzero(valid_before_mask[batch_index], as_tuple=False).flatten()
                if len(valid_positions) and bool(dropped[batch_index, valid_positions].all()):
                    dropped[batch_index, valid_positions[-1]] = False
            effective_padding = effective_padding | dropped
            effective_features = effective_features & ~effective_padding.unsqueeze(-1)
        if self.training and self.feature_mask_probability > 0.0:
            observed_before_random_mask = effective_features.clone()
            keep = torch.rand_like(effective_features, dtype=torch.float32) >= self.feature_mask_probability
            effective_features = effective_features & keep
            for batch_index in range(batch_size):
                if bool(observed_before_random_mask[batch_index].any()) and not bool(
                    effective_features[batch_index].any()
                ):
                    observed_times = torch.nonzero(
                        observed_before_random_mask[batch_index].any(dim=-1),
                        as_tuple=False,
                    ).flatten()
                    latest = int(observed_times[-1])
                    effective_features[batch_index, latest] = observed_before_random_mask[batch_index, latest]
        return effective_padding, effective_features

    def forward(
        self,
        x: torch.Tensor,
        horizons: torch.Tensor | Sequence[float],
        *,
        padding_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
        target_x: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
        target_time_mask: torch.Tensor | None = None,
        target_feature_mask: torch.Tensor | None = None,
        world_ids: torch.Tensor | int | None = None,
        state_ids: torch.Tensor | int | None = None,
        regime_ids: torch.Tensor | int | None = None,
    ) -> WorldJEPAOutput:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, F], got {tuple(x.shape)}.")
        batch_size = int(x.shape[0])
        horizon_values = self._normalize_horizons(horizons, batch_size, x.device)
        horizon_count = int(horizon_values.shape[1])
        normalized_world_ids = _normalize_ids(
            world_ids,
            batch_size=batch_size,
            upper_bound=len(self.world_names),
            default=self.default_world_id,
            device=x.device,
        )
        normalized_state_ids = _normalize_ids(
            state_ids,
            batch_size=batch_size,
            upper_bound=self.num_states,
            default=0,
            device=x.device,
        )
        normalized_regime_ids = _normalize_ids(
            regime_ids,
            batch_size=batch_size,
            upper_bound=self.num_regimes,
            default=0,
            device=x.device,
        )
        effective_padding, effective_feature_mask = self._prepare_context_masks(
            x,
            padding_mask=padding_mask,
            time_mask=time_mask,
            feature_mask=feature_mask,
        )

        (
            context_tokens,
            context_summary,
            regime_token,
            normalized_padding,
            valid_context_fraction,
            valid_feature_fraction,
        ) = self.context_encoder(
            x,
            padding_mask=effective_padding,
            feature_mask=effective_feature_mask,
            world_ids=normalized_world_ids,
            state_ids=normalized_state_ids,
            regime_ids=normalized_regime_ids,
        )
        horizon_features = self.horizon_encoder(horizon_values).to(dtype=context_summary.dtype)
        expanded_summary = context_summary.unsqueeze(1).expand(-1, horizon_count, -1)
        expanded_regime = regime_token.unsqueeze(1).expand(-1, horizon_count, -1)
        predictor_features = self.predictor_trunk(
            torch.cat([expanded_summary, expanded_regime, horizon_features], dim=-1)
        )
        predicted_target = self.target_head(predictor_features)
        predicted_variance = F.softplus(self.uncertainty_head(predictor_features).squeeze(-1)) + 1e-6
        uncertainty = torch.sqrt(predicted_variance)

        target_latent: torch.Tensor | None = None
        target_valid_mask: torch.Tensor | None = None
        if target_x is not None:
            if target_x.ndim == 3:
                if target_x.shape[:2] != (batch_size, horizon_count):
                    raise ValueError(
                        f"Expected point target_x [{batch_size}, {horizon_count}, F], got {tuple(target_x.shape)}."
                    )
                target_blocks = target_x.unsqueeze(2)
            elif target_x.ndim == 4:
                if target_x.shape[:2] != (batch_size, horizon_count):
                    raise ValueError(
                        f"Expected block target_x [{batch_size}, {horizon_count}, L, F], got {tuple(target_x.shape)}."
                    )
                target_blocks = target_x
            else:
                raise ValueError(
                    f"Expected target_x [B, H, F] or [B, H, L, F], got {tuple(target_x.shape)}."
                )
            if target_blocks.shape[-1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} target features, got {target_blocks.shape[-1]}.")
            block_length = int(target_blocks.shape[2])

            if target_padding_mask is None:
                block_padding = torch.zeros(
                    (batch_size, horizon_count, block_length),
                    dtype=torch.bool,
                    device=x.device,
                )
            else:
                block_padding = torch.as_tensor(target_padding_mask, dtype=torch.bool, device=x.device)
                if block_padding.ndim == 1 and block_padding.shape[0] == horizon_count:
                    block_padding = block_padding.view(1, horizon_count, 1).expand(batch_size, -1, block_length)
                elif block_padding.ndim == 2:
                    if block_padding.shape == (batch_size, horizon_count):
                        block_padding = block_padding.unsqueeze(-1).expand(-1, -1, block_length)
                    elif block_padding.shape == (horizon_count, block_length):
                        block_padding = block_padding.unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        raise ValueError("2D target_padding_mask must have shape [B, H] or [H, L].")
                elif block_padding.ndim == 3:
                    if block_padding.shape == (1, horizon_count, block_length):
                        block_padding = block_padding.expand(batch_size, -1, -1)
                    elif block_padding.shape != (batch_size, horizon_count, block_length):
                        raise ValueError("target_padding_mask must have shape [B, H, L].")
                else:
                    raise ValueError("target_padding_mask must have shape [H], [B, H], [H, L], or [B, H, L].")
            if target_time_mask is not None:
                included = torch.as_tensor(target_time_mask, dtype=torch.bool, device=x.device)
                if included.ndim == 2 and included.shape == (batch_size, horizon_count):
                    included = included.unsqueeze(-1).expand(-1, -1, block_length)
                elif included.ndim == 3 and included.shape == (1, horizon_count, block_length):
                    included = included.expand(batch_size, -1, -1)
                if included.shape != (batch_size, horizon_count, block_length):
                    raise ValueError("target_time_mask must have shape [B, H] or [B, H, L].")
                block_padding = block_padding | ~included

            flat_targets = target_blocks.reshape(batch_size * horizon_count, block_length, self.input_dim)
            flat_padding = block_padding.reshape(batch_size * horizon_count, block_length)
            if target_feature_mask is None:
                flat_feature_mask = None
            else:
                block_feature_mask = torch.as_tensor(target_feature_mask, dtype=torch.bool, device=x.device)
                if target_x.ndim == 3 and block_feature_mask.shape == target_x.shape:
                    block_feature_mask = block_feature_mask.unsqueeze(2)
                if block_feature_mask.shape == (1, horizon_count, block_length, self.input_dim):
                    block_feature_mask = block_feature_mask.expand(batch_size, -1, -1, -1)
                if block_feature_mask.shape != target_blocks.shape:
                    raise ValueError(
                        f"target_feature_mask must match target_x; got {tuple(block_feature_mask.shape)} "
                        f"for normalized target shape {tuple(target_blocks.shape)}."
                    )
                flat_feature_mask = block_feature_mask.reshape(
                    batch_size * horizon_count,
                    block_length,
                    self.input_dim,
                )
            expanded_world_ids = normalized_world_ids.unsqueeze(1).expand(-1, horizon_count).reshape(-1)
            expanded_state_ids = normalized_state_ids.unsqueeze(1).expand(-1, horizon_count).reshape(-1)
            expanded_regime_ids = normalized_regime_ids.unsqueeze(1).expand(-1, horizon_count).reshape(-1)
            with torch.no_grad():
                _, target_summaries, _, effective_target_padding, *_ = self.target_encoder(
                    flat_targets,
                    padding_mask=flat_padding,
                    feature_mask=flat_feature_mask,
                    world_ids=expanded_world_ids,
                    state_ids=expanded_state_ids,
                    regime_ids=expanded_regime_ids,
                    # The teacher sees the complete target block. Besides
                    # matching JEPA's contextual-target design, bidirectional
                    # encoding prevents structurally left-padded disjoint bins
                    # from creating all-masked causal attention rows.
                    causal=False,
                )
            effective_target_padding = effective_target_padding.reshape(
                batch_size,
                horizon_count,
                block_length,
            )
            target_valid_mask = (~effective_target_padding).any(dim=-1)
            target_latent = target_summaries.reshape(batch_size, horizon_count, self.latent_dim)
            target_latent = self._normalize_teacher_targets(target_latent).detach()
            target_latent = torch.where(
                target_valid_mask.unsqueeze(-1),
                target_latent,
                torch.zeros_like(target_latent),
            )

        if target_latent is None:
            disagreement = predicted_target.std(dim=-1, unbiased=False)
            target_rmse = torch.full_like(disagreement, float("nan"))
        else:
            cosine = F.cosine_similarity(predicted_target, target_latent, dim=-1, eps=1e-8)
            disagreement = (1.0 - cosine).clamp(0.0, 2.0)
            target_rmse = torch.sqrt((predicted_target - target_latent).pow(2).mean(dim=-1) + 1e-8)
            if target_valid_mask is not None:
                disagreement = torch.where(target_valid_mask, disagreement, torch.zeros_like(disagreement))
                target_rmse = torch.where(target_valid_mask, target_rmse, torch.zeros_like(target_rmse))

        regime_logits = self.regime_head(regime_token)
        regime_probabilities = torch.softmax(regime_logits, dim=-1)
        regime_entropy = -(
            regime_probabilities * regime_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        diagnostics = {
            "valid_feature_fraction": valid_feature_fraction,
            "regime_entropy": regime_entropy,
            "predicted_variance": predicted_variance,
            "target_rmse": target_rmse,
            "context_token_norm": context_tokens.norm(dim=-1),
            "padding_mask": normalized_padding,
        }
        return WorldJEPAOutput(
            context_tokens=context_tokens,
            context_summary=context_summary,
            predicted_target=predicted_target,
            target_latent=target_latent,
            regime_logits=regime_logits,
            regime_probabilities=regime_probabilities,
            uncertainty=uncertainty,
            valid_context_fraction=valid_context_fraction,
            disagreement=disagreement,
            state_token=context_summary,
            regime_token=regime_token,
            target_valid_mask=target_valid_mask,
            diagnostics=diagnostics,
        )


# A concise alias is useful in configs while retaining the explicit public name.
WorldEncoder = WorldJEPAEncoder
