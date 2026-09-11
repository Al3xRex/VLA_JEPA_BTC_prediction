"""Context-relevant attention over independently encoded JEPA worlds.

This is the final aggregation layer for JEPA space.  It intentionally operates
on generic world context tokens rather than depending on a particular encoder,
so callers can stack ``WorldJEPAOutput.context_summary`` from heterogeneous
world models.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .continuous_horizon import ContinuousHorizonEmbedding


QUANTILE_ROLE_NAMES = ("center", "width", "tails")


@dataclass(frozen=True)
class WorldAttentionOutput:
    """Outputs and inspectable routing state for a world-attention query."""

    fused_context: Tensor
    attention_weights: Tensor
    quantiles: Tensor
    state_mixture: Tensor
    regime_mixture: Tensor
    world_gates: Tensor
    horizon_embeddings: Tensor
    router_diagnostics: dict[str, Any]

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Short alias retained for interactive inspection code."""

        return self.router_diagnostics

    @property
    def regime_state_mixture(self) -> Tensor:
        """Attention-weighted state representation before context fusion."""

        return self.state_mixture


class StructuredMonotoneQuantileHead(nn.Module):
    """Predict a center and positive outward gaps, preventing quantile crossing."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
        min_gap: float = 1e-4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        levels = tuple(float(value) for value in quantiles)
        if not levels:
            raise ValueError("at least one quantile is required")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantiles must lie strictly between zero and one")
        if any(right <= left for left, right in zip(levels, levels[1:])):
            raise ValueError("quantiles must be strictly increasing")
        if min_gap < 0.0:
            raise ValueError("min_gap cannot be negative")

        hidden_dim = int(hidden_dim or max(32, input_dim))
        self.center_index = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.5))
        self.lower_count = self.center_index
        self.upper_count = len(levels) - self.center_index - 1
        self.min_gap = float(min_gap)
        self.register_buffer("quantile_levels", torch.tensor(levels, dtype=torch.float32))

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(hidden_dim),
        )
        self.center_projection = nn.Linear(hidden_dim, 1)
        self.gap_projection = nn.Linear(hidden_dim, len(levels) - 1)
        # Start with compact, plausible intervals instead of softplus(0)
        # producing roughly 0.69 per adjacent quantile gap.
        nn.init.zeros_(self.gap_projection.weight)
        nn.init.constant_(self.gap_projection.bias, -3.0)

    def forward(self, context: Tensor) -> Tensor:
        features = self.backbone(context)
        center = self.center_projection(features)
        if self.quantile_levels.numel() == 1:
            return center

        raw_gaps = self.gap_projection(features)
        positive_gaps = F.softplus(raw_gaps) + self.min_gap
        pieces: list[Tensor] = []

        if self.lower_count:
            # Gap zero is nearest the center.  Reversing cumulative gaps emits
            # the requested lower quantiles from the most distant to nearest.
            lower_gaps = positive_gaps[..., : self.lower_count]
            lower_cumulative = torch.cumsum(lower_gaps, dim=-1)
            pieces.append(center - torch.flip(lower_cumulative, dims=(-1,)))

        pieces.append(center)

        if self.upper_count:
            upper_gaps = positive_gaps[..., self.lower_count :]
            pieces.append(center + torch.cumsum(upper_gaps, dim=-1))

        return torch.cat(pieces, dim=-1)


class SemanticRoleMonotoneQuantileHead(nn.Module):
    """Build five ordered quantiles from center-, width-, and tail-role states.

    The median is a function only of ``center_context``. The width branch
    controls the two positive gaps from q50 to q25/q75, while the tails branch
    controls the additional positive gaps from q25/q75 to q05/q95. This makes
    the semantic routing constraint structural rather than merely advisory and
    guarantees ordered quantiles for every parameter value.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
        min_gap: float = 1e-4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        levels = tuple(float(value) for value in quantiles)
        if len(levels) != 5:
            raise ValueError("semantic role routing requires exactly five quantiles")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantiles must lie strictly between zero and one")
        if any(right <= left for left, right in zip(levels, levels[1:])):
            raise ValueError("quantiles must be strictly increasing")
        if not math.isclose(levels[2], 0.5, abs_tol=1e-8):
            raise ValueError("the middle semantic-role quantile must be 0.50")
        if float(min_gap) < 0.0:
            raise ValueError("min_gap cannot be negative")

        hidden_dim = int(hidden_dim or max(32, int(input_dim)))

        def tower() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(input_dim), hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.LayerNorm(hidden_dim),
            )

        self.center_backbone = tower()
        self.width_backbone = tower()
        self.tails_backbone = tower()
        self.center_projection = nn.Linear(hidden_dim, 1)
        self.inner_gap_projection = nn.Linear(hidden_dim, 2)
        self.outer_gap_projection = nn.Linear(hidden_dim, 2)
        self.min_gap = float(min_gap)
        self.register_buffer("quantile_levels", torch.tensor(levels, dtype=torch.float32))

        for projection in (self.inner_gap_projection, self.outer_gap_projection):
            nn.init.zeros_(projection.weight)
            nn.init.constant_(projection.bias, -3.0)

    def forward(
        self,
        center_context: Tensor,
        width_context: Tensor,
        tails_context: Tensor,
    ) -> Tensor:
        if center_context.shape != width_context.shape or center_context.shape != tails_context.shape:
            raise ValueError("center, width, and tails contexts must have identical shapes")
        if center_context.ndim < 2:
            raise ValueError("semantic role contexts must have a latent feature dimension")

        center = self.center_projection(self.center_backbone(center_context))
        inner_gaps = F.softplus(
            self.inner_gap_projection(self.width_backbone(width_context))
        ) + self.min_gap
        outer_gaps = F.softplus(
            self.outer_gap_projection(self.tails_backbone(tails_context))
        ) + self.min_gap

        q25 = center - inner_gaps[..., 0:1]
        q75 = center + inner_gaps[..., 1:2]
        q05 = q25 - outer_gaps[..., 0:1]
        q95 = q75 + outer_gaps[..., 1:2]
        return torch.cat((q05, q25, center, q75, q95), dim=-1)


def _expand_world_value(
    value: Tensor | Sequence[float] | None,
    *,
    name: str,
    batch_size: int,
    world_count: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> Tensor:
    if value is None:
        return torch.full((batch_size, world_count), default, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 1:
        if tensor.shape[0] != world_count:
            raise ValueError(f"{name} must have W={world_count} entries")
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != world_count:
        raise ValueError(f"{name} must have shape [B, W] or [W]")
    if tensor.shape[0] == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size, -1)
    elif tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension must be 1 or B={batch_size}")
    return tensor


def _expand_world_mask(
    mask: Tensor | Sequence[bool] | None,
    *,
    batch_size: int,
    world_count: int,
    device: torch.device,
) -> Tensor:
    """Return an inclusion mask; ``True`` means the world may be attended."""

    if mask is None:
        return torch.ones((batch_size, world_count), device=device, dtype=torch.bool)
    tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if tensor.ndim == 1:
        if tensor.shape[0] != world_count:
            raise ValueError(f"world_mask must have W={world_count} entries")
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != world_count:
        raise ValueError("world_mask must have shape [B, W] or [W]")
    if tensor.shape[0] == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size, -1)
    elif tensor.shape[0] != batch_size:
        raise ValueError(f"world_mask batch dimension must be 1 or B={batch_size}")
    return tensor


def _expand_role_world_masks(
    masks: Mapping[str, Tensor | Sequence[bool]] | Tensor | Sequence[Sequence[bool]],
    *,
    batch_size: int,
    world_count: int,
    device: torch.device,
) -> Tensor:
    """Return role inclusion masks with shape ``[B, 3, W]``."""

    if isinstance(masks, Mapping):
        missing = [role for role in QUANTILE_ROLE_NAMES if role not in masks]
        extra = [str(role) for role in masks if str(role) not in QUANTILE_ROLE_NAMES]
        if missing or extra:
            raise ValueError(
                f"role_world_masks must contain exactly {QUANTILE_ROLE_NAMES}; "
                f"missing={missing}, extra={extra}"
            )
        expanded = torch.stack(
            [
                _expand_world_mask(
                    masks[role],
                    batch_size=batch_size,
                    world_count=world_count,
                    device=device,
                )
                for role in QUANTILE_ROLE_NAMES
            ],
            dim=1,
        )
    else:
        expanded = torch.as_tensor(masks, device=device, dtype=torch.bool)
        if expanded.ndim == 2:
            if tuple(expanded.shape) != (len(QUANTILE_ROLE_NAMES), world_count):
                raise ValueError("role_world_masks must have shape [3, W] or [B, 3, W]")
            expanded = expanded.unsqueeze(0)
        if expanded.ndim != 3 or tuple(expanded.shape[1:]) != (
            len(QUANTILE_ROLE_NAMES),
            world_count,
        ):
            raise ValueError("role_world_masks must have shape [3, W] or [B, 3, W]")
        if expanded.shape[0] == 1 and batch_size != 1:
            expanded = expanded.expand(batch_size, -1, -1)
        elif expanded.shape[0] != batch_size:
            raise ValueError(f"role_world_masks batch dimension must be 1 or B={batch_size}")

    if not expanded.any(dim=-1).all():
        raise ValueError("every semantic role must include at least one configured world")
    return expanded


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    has_world = mask.any(dim=-1, keepdim=True)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    safe_logits = torch.where(has_world, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits, dim=-1) * mask.to(logits.dtype)
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    return torch.where(
        has_world,
        probabilities / normalizer.clamp_min(torch.finfo(logits.dtype).eps),
        torch.zeros_like(probabilities),
    )


def _strict_monotone_quantile_projection(
    values: Tensor,
    *,
    minimum_gap: float = 1e-6,
) -> Tensor:
    """Project arbitrary quantile values onto a strictly ordered sequence.

    Adding a monotone residual vector to an ordered baseline does not itself
    preserve order: a larger correction to a lower quantile can still cross
    the next baseline quantile. Sorting retains the predicted support while a
    tiny cumulative offset makes ties strictly increasing.
    """

    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("quantile values must have a non-empty final dimension")
    if minimum_gap <= 0.0:
        raise ValueError("minimum_gap must be positive")
    ordered = torch.sort(values, dim=-1).values
    offsets = torch.arange(
        ordered.shape[-1],
        device=ordered.device,
        dtype=ordered.dtype,
    ) * float(minimum_gap)
    return torch.cummax(ordered - offsets, dim=-1).values + offsets


def _sparsemax(logits: Tensor, mask: Tensor) -> Tensor:
    """Dependency-free sparsemax over the final dimension."""

    has_world = mask.any(dim=-1, keepdim=True)
    valid_max = logits.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
    valid_max = torch.where(has_world, valid_max, torch.zeros_like(valid_max))
    shifted = logits - valid_max
    floor = torch.full_like(shifted, -1e4)
    shifted = torch.where(mask, shifted, floor)

    sorted_values, _ = torch.sort(shifted, dim=-1, descending=True)
    ranks = torch.arange(
        1,
        shifted.shape[-1] + 1,
        device=shifted.device,
        dtype=shifted.dtype,
    )
    view_shape = (1,) * (shifted.ndim - 1) + (shifted.shape[-1],)
    ranks = ranks.view(view_shape)
    cumulative = sorted_values.cumsum(dim=-1) - 1.0
    support = sorted_values - cumulative / ranks > 0.0
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = cumulative.gather(dim=-1, index=support_size - 1) / support_size.to(shifted.dtype)
    probabilities = torch.clamp(shifted - threshold, min=0.0) * mask.to(shifted.dtype)
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    fallback = _masked_softmax(logits, mask)
    normalized = probabilities / normalizer.clamp_min(torch.finfo(logits.dtype).eps)
    return torch.where((normalizer > 0.0) & has_world, normalized, fallback)


def _entmax15(logits: Tensor, mask: Tensor) -> Tensor:
    """Closed-form alpha=1.5 entmax, implemented using only PyTorch."""

    has_world = mask.any(dim=-1, keepdim=True)
    valid_max = logits.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
    valid_max = torch.where(has_world, valid_max, torch.zeros_like(valid_max))
    # The division by two is part of the alpha=1.5 threshold derivation.
    shifted = (logits - valid_max) / 2.0
    shifted = torch.where(mask, shifted, torch.full_like(shifted, -1e4))
    sorted_values, _ = torch.sort(shifted, dim=-1, descending=True)
    ranks = torch.arange(
        1,
        shifted.shape[-1] + 1,
        device=shifted.device,
        dtype=shifted.dtype,
    )
    view_shape = (1,) * (shifted.ndim - 1) + (shifted.shape[-1],)
    ranks = ranks.view(view_shape)
    mean = sorted_values.cumsum(dim=-1) / ranks
    mean_square = sorted_values.square().cumsum(dim=-1) / ranks
    variance_sum = ranks * (mean_square - mean.square())
    delta = (1.0 - variance_sum) / ranks
    thresholds = mean - torch.sqrt(torch.clamp(delta, min=0.0))
    support = thresholds <= sorted_values
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = thresholds.gather(dim=-1, index=support_size - 1)
    probabilities = torch.clamp(shifted - threshold, min=0.0).square()
    probabilities = probabilities * mask.to(shifted.dtype)
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    fallback = _masked_softmax(logits, mask)
    normalized = probabilities / normalizer.clamp_min(torch.finfo(logits.dtype).eps)
    return torch.where((normalizer > 0.0) & has_world, normalized, fallback)


class ContextRelevantWorldRouter(nn.Module):
    """Route horizon-conditioned queries across fresh, valid JEPA worlds.

    ``world_tokens`` may be ``[B, W, D]`` or ``[B, W, T, D]``.  Temporal
    blocks are finite-value averaged before cross-world attention. Optional
    ``world_future_tokens`` have shape ``[B, W, H, D]`` and let each horizon
    route the JEPA-predicted future state instead of reusing one summary for
    every query. A shared
    horizon grid ``[H]`` is broadcast over the batch, while ``[B, H]`` permits
    different float horizons per example.

    ``world_mask`` is an inclusion mask: ``True`` means a world is eligible.
    ``world_freshness`` is a causal age/staleness value where zero is freshest.
    Fully invalid, fully missing, explicitly masked, or non-finite-freshness
    worlds receive exactly zero attention.
    """

    _ROUTING_MODES = {"softmax", "topk", "sparsemax", "entmax15"}

    def __init__(
        self,
        token_dim: int | None = None,
        *,
        world_dim: int | None = None,
        input_dim: int | None = None,
        model_dim: int | None = None,
        market_state_dim: int | None = None,
        regime_dim: int | None = None,
        horizon_embedding_dim: int = 32,
        horizon_dim: int | None = None,
        horizon_hidden_dim: int | None = None,
        num_horizon_frequencies: int = 8,
        num_heads: int = 1,
        attention_heads: int | None = None,
        quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
        quantile_hidden_dim: int | None = None,
        routing_mode: str = "softmax",
        routing: str | None = None,
        top_k: int | None = None,
        top_k_worlds: int | None = None,
        freshness_half_life: float = 7.0,
        freshness_temperature: float | None = None,
        uncertainty_temperature: float = 1.0,
        world_dropout: float = 0.0,
        hard_top1_inference: bool = False,
        residual_to_baseline: bool = False,
        semantic_role_routing: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dimension_aliases = [value for value in (token_dim, world_dim, input_dim) if value is not None]
        if dimension_aliases and any(int(value) != int(dimension_aliases[0]) for value in dimension_aliases[1:]):
            raise ValueError("token_dim, world_dim, and input_dim aliases disagree")
        token_dim = dimension_aliases[0] if dimension_aliases else None
        if token_dim is None or token_dim <= 0:
            raise ValueError("token_dim (or world_dim/input_dim) must be positive")
        model_dim = int(model_dim or token_dim)
        if attention_heads is not None:
            if num_heads != 1 and int(attention_heads) != int(num_heads):
                raise ValueError("num_heads and attention_heads aliases disagree")
            num_heads = int(attention_heads)
        if top_k_worlds is not None:
            if top_k is not None and int(top_k_worlds) != int(top_k):
                raise ValueError("top_k and top_k_worlds aliases disagree")
            top_k = int(top_k_worlds)
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide model_dim")
        if freshness_half_life <= 0.0:
            raise ValueError("freshness_half_life must be positive")
        if freshness_temperature is not None and freshness_temperature <= 0.0:
            raise ValueError("freshness_temperature must be positive")
        if uncertainty_temperature <= 0.0:
            raise ValueError("uncertainty_temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if not 0.0 <= world_dropout < 1.0:
            raise ValueError("world_dropout must be in [0, 1)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if top_k_worlds is not None and routing is None and routing_mode == "softmax":
            routing_mode = "topk"
        routing_mode = (routing or routing_mode).lower().replace("top_k", "topk")
        if routing_mode == "entmax":
            routing_mode = "entmax15"
        if routing_mode not in self._ROUTING_MODES:
            allowed = ", ".join(sorted(self._ROUTING_MODES))
            raise ValueError(f"routing_mode must be one of: {allowed}")

        self.token_dim = int(token_dim)
        self.model_dim = model_dim
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // self.num_heads
        self.routing_mode = routing_mode
        self.top_k = top_k
        self.world_dropout = float(world_dropout)
        self.hard_top1_inference = bool(hard_top1_inference)
        self.residual_to_baseline = bool(residual_to_baseline)
        self.semantic_role_routing = bool(semantic_role_routing)
        self.regime_dim = int(regime_dim) if regime_dim is not None else None

        horizon_embedding_dim = int(horizon_dim or horizon_embedding_dim)
        self.horizon_embedding = ContinuousHorizonEmbedding(
            horizon_embedding_dim,
            hidden_dim=horizon_hidden_dim,
            num_frequencies=num_horizon_frequencies,
            dropout=dropout,
        )
        self.horizon_projection = nn.Linear(horizon_embedding_dim, model_dim)
        self.latest_world_projection = nn.Linear(self.token_dim, model_dim)
        self.regime_projection = (
            nn.Linear(self.regime_dim, self.token_dim, bias=False)
            if self.regime_dim is not None
            else None
        )
        self.market_projection = nn.Linear(int(market_state_dim or token_dim), model_dim)
        self.key_projection = nn.Linear(self.token_dim, model_dim)
        self.value_projection = nn.Linear(self.token_dim, model_dim)
        self.state_projection = nn.Linear(self.token_dim, model_dim)
        self.future_projection = nn.Linear(self.token_dim, self.token_dim)
        self.future_norm = nn.LayerNorm(self.token_dim)
        self._future_gate_unconstrained = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        nn.init.eye_(self.future_projection.weight)
        nn.init.zeros_(self.future_projection.bias)
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.attention_output = nn.Linear(model_dim, model_dim)
        self.query_norm = nn.LayerNorm(model_dim)
        self.output_norm = nn.LayerNorm(model_dim)
        self.context_gate = nn.Sequential(nn.Linear(2 * model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, 1))
        self.dropout = nn.Dropout(float(dropout))
        self.quantile_head = StructuredMonotoneQuantileHead(
            model_dim,
            hidden_dim=quantile_hidden_dim,
            quantiles=quantiles,
            dropout=dropout,
        )
        self.role_quantile_head = (
            SemanticRoleMonotoneQuantileHead(
                model_dim,
                hidden_dim=quantile_hidden_dim,
                quantiles=quantiles,
                dropout=dropout,
            )
            if self.semantic_role_routing
            else None
        )
        # A small learned gate makes the residual variant start close to its
        # causal baseline and earn larger corrections only through validation.
        self._residual_gate_unconstrained = nn.Parameter(
            torch.tensor(math.log(0.05 / 0.95), dtype=torch.float32)
        )

        initial_decay = (
            1.0 / float(freshness_temperature)
            if freshness_temperature is not None
            else math.log(2.0) / float(freshness_half_life)
        )
        inverse_softplus = math.log(math.expm1(initial_decay))
        self._freshness_decay_unconstrained = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))
        initial_uncertainty_decay = 1.0 / float(uncertainty_temperature)
        self._uncertainty_decay_unconstrained = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_uncertainty_decay)), dtype=torch.float32)
        )

    def _summarize_tokens(self, world_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if world_tokens.ndim not in (3, 4):
            raise ValueError("world_tokens must have shape [B, W, D] or [B, W, T, D]")
        if world_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"world token dimension is {world_tokens.shape[-1]}, expected {self.token_dim}"
            )
        if world_tokens.shape[0] <= 0 or world_tokens.shape[1] <= 0:
            raise ValueError("world_tokens must contain at least one batch row and world")

        dtype = self.key_projection.weight.dtype
        tokens = world_tokens.to(dtype=dtype)
        finite = torch.isfinite(tokens)
        safe = torch.where(finite, tokens, torch.zeros_like(tokens))
        if tokens.ndim == 3:
            summary = safe
            inferred_missingness = 1.0 - finite.to(dtype).mean(dim=-1)
        else:
            counts = finite.to(dtype).sum(dim=2).clamp_min(1.0)
            summary = safe.sum(dim=2) / counts
            inferred_missingness = 1.0 - finite.to(dtype).mean(dim=(2, 3))
        return summary, inferred_missingness

    @staticmethod
    def _prepare_horizons(
        horizons: Tensor | Sequence[float] | float,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        values = torch.as_tensor(horizons, device=device, dtype=torch.float32)
        if values.ndim == 0:
            values = values.reshape(1, 1).expand(batch_size, -1)
        elif values.ndim == 1:
            values = values.unsqueeze(0).expand(batch_size, -1)
        elif values.ndim == 2:
            if values.shape[0] == 1 and batch_size != 1:
                values = values.expand(batch_size, -1)
            elif values.shape[0] != batch_size:
                raise ValueError(f"horizons first dimension must be 1 or B={batch_size}")
        else:
            raise ValueError("horizons must be a scalar, [H], or [B, H]")
        if values.shape[1] == 0:
            raise ValueError("horizons must contain at least one query")
        return values

    def _route(self, logits: Tensor, mask: Tensor) -> Tensor:
        routing_mode = "topk" if self.hard_top1_inference and not self.training else self.routing_mode
        if routing_mode == "softmax":
            return _masked_softmax(logits, mask)
        if routing_mode == "topk":
            world_count = logits.shape[-1]
            requested_top_k = 1 if self.hard_top1_inference and not self.training else self.top_k
            top_k = min(requested_top_k or min(2, world_count), world_count)
            # Select one shared support across heads.  Per-head top-k followed
            # by averaging could expose as many as ``heads * k`` worlds in the
            # public mixture, defeating the promised routing budget.
            aggregate_mask = mask.any(dim=-2, keepdim=True)
            aggregate_logits = logits.mean(dim=-2, keepdim=True).masked_fill(~aggregate_mask, -torch.inf)
            indices = torch.topk(aggregate_logits, k=top_k, dim=-1).indices
            top_mask = torch.zeros_like(aggregate_mask).scatter(dim=-1, index=indices, value=True)
            top_mask = top_mask.expand_as(mask)
            return _masked_softmax(logits, mask & top_mask)
        if routing_mode == "sparsemax":
            return _sparsemax(logits, mask)
        return _entmax15(logits, mask)

    def forward(
        self,
        world_tokens: Tensor,
        horizons: Tensor | Sequence[float] | float,
        world_freshness: Tensor | Sequence[float] | None = None,
        world_validity: Tensor | Sequence[float] | None = None,
        world_missingness: Tensor | Sequence[float] | None = None,
        world_uncertainty: Tensor | Sequence[float] | None = None,
        world_mask: Tensor | Sequence[bool] | None = None,
        market_state: Tensor | None = None,
        world_regimes: Tensor | None = None,
        world_future_tokens: Tensor | None = None,
        baseline_quantiles: Tensor | None = None,
        role_world_masks: (
            Mapping[str, Tensor | Sequence[bool]]
            | Tensor
            | Sequence[Sequence[bool]]
            | None
        ) = None,
    ) -> WorldAttentionOutput:
        summary, inferred_missingness = self._summarize_tokens(world_tokens)
        batch_size, world_count, _ = summary.shape
        device, dtype = summary.device, summary.dtype
        horizon_values = self._prepare_horizons(horizons, batch_size=batch_size, device=device)
        configured_role_masks: Tensor | None = None
        if self.semantic_role_routing:
            if role_world_masks is None:
                raise ValueError(
                    "role_world_masks are required when semantic_role_routing=True"
                )
            configured_role_masks = _expand_role_world_masks(
                role_world_masks,
                batch_size=batch_size,
                world_count=world_count,
                device=device,
            )
        elif role_world_masks is not None:
            raise ValueError(
                "role_world_masks require semantic_role_routing=True"
            )

        # Keep the context-only route as a strict fallback/ablation, but when
        # the caller supplies JEPA future predictions, preserve their horizon
        # axis all the way through cross-world attention. Previously those
        # predictions were computed and then discarded by the pipeline.
        if world_future_tokens is None:
            horizon_world_states = summary.unsqueeze(1).expand(
                -1, horizon_values.shape[1], -1, -1
            )
            future_valid_fraction = torch.ones(
                (batch_size, world_count, horizon_values.shape[1]),
                device=device,
                dtype=dtype,
            )
            future_gate = summary.new_tensor(0.0)
        else:
            future = torch.as_tensor(world_future_tokens, device=device, dtype=dtype)
            if future.ndim == 3:
                future = future.unsqueeze(0)
            if future.shape == (1, world_count, horizon_values.shape[1], self.token_dim):
                future = future.expand(batch_size, -1, -1, -1)
            expected = (batch_size, world_count, horizon_values.shape[1], self.token_dim)
            if tuple(future.shape) != expected:
                raise ValueError(
                    "world_future_tokens must have shape [B, W, H, D] matching the query "
                    f"horizons; expected {expected}, got {tuple(future.shape)}"
                )
            finite_future = torch.isfinite(future)
            future_valid_fraction = finite_future.to(dtype).mean(dim=-1)
            context_by_horizon = summary.unsqueeze(2).expand_as(future)
            safe_future = torch.where(finite_future, future, context_by_horizon)
            future_delta = safe_future - context_by_horizon
            future_gate = torch.sigmoid(self._future_gate_unconstrained).to(dtype=dtype)
            fused_future = self.future_norm(
                context_by_horizon + future_gate * self.future_projection(future_delta)
            )
            horizon_world_states = fused_future.permute(0, 2, 1, 3).contiguous()

        normalized_regimes: Tensor | None = None
        if world_regimes is not None:
            normalized_regimes = torch.as_tensor(world_regimes, device=device, dtype=dtype)
            if normalized_regimes.ndim == 2:
                normalized_regimes = normalized_regimes.unsqueeze(0).expand(batch_size, -1, -1)
            if normalized_regimes.ndim != 3 or normalized_regimes.shape[:2] != (
                batch_size,
                world_count,
            ):
                raise ValueError("world_regimes must have shape [B, W, R] or [W, R]")
            normalized_regimes = torch.nan_to_num(normalized_regimes).clamp_min(0.0)
            normalized_regimes = normalized_regimes / normalized_regimes.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(dtype).eps)
            if self.regime_projection is not None:
                if normalized_regimes.shape[-1] != self.regime_dim:
                    raise ValueError(
                        f"world_regimes has R={normalized_regimes.shape[-1]}, expected {self.regime_dim}"
                    )
                summary = summary + self.regime_projection(normalized_regimes)

        freshness_raw = _expand_world_value(
            world_freshness,
            name="world_freshness",
            batch_size=batch_size,
            world_count=world_count,
            device=device,
            dtype=dtype,
            default=0.0,
        )
        freshness_finite = torch.isfinite(freshness_raw)
        freshness = torch.nan_to_num(freshness_raw, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(0.0)
        validity = _expand_world_value(
            world_validity,
            name="world_validity",
            batch_size=batch_size,
            world_count=world_count,
            device=device,
            dtype=dtype,
            default=1.0,
        )
        validity = torch.nan_to_num(validity, nan=0.0).clamp(0.0, 1.0)
        explicit_missingness = _expand_world_value(
            world_missingness,
            name="world_missingness",
            batch_size=batch_size,
            world_count=world_count,
            device=device,
            dtype=dtype,
            default=0.0,
        )
        explicit_missingness = torch.nan_to_num(explicit_missingness, nan=1.0).clamp(0.0, 1.0)
        # Union explicit metadata with missing values observed in the token.
        missingness = 1.0 - (1.0 - explicit_missingness) * (1.0 - inferred_missingness)
        uncertainty = _expand_world_value(
            world_uncertainty,
            name="world_uncertainty",
            batch_size=batch_size,
            world_count=world_count,
            device=device,
            dtype=dtype,
            default=0.0,
        )
        uncertainty = torch.nan_to_num(uncertainty, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(0.0)
        inclusion_mask = _expand_world_mask(
            world_mask,
            batch_size=batch_size,
            world_count=world_count,
            device=device,
        )
        metadata_mask = inclusion_mask & freshness_finite & (validity > 0.0) & (missingness < 1.0)
        effective_mask = metadata_mask
        if self.training and self.world_dropout > 0.0:
            dropout_keep = torch.rand_like(validity) >= self.world_dropout
            effective_mask = metadata_mask & dropout_keep
            # World dropout must never turn a usable row into an all-masked
            # row.  Keep the freshest available world as a deterministic
            # fallback for any such row.
            had_world = metadata_mask.any(dim=-1)
            lost_all = had_world & ~effective_mask.any(dim=-1)
            fallback_index = freshness.masked_fill(~metadata_mask, torch.inf).argmin(dim=-1)
            rows = torch.arange(batch_size, device=device)
            effective_mask[rows[lost_all], fallback_index[lost_all]] = True

        metadata_role_masks: Tensor | None = None
        effective_role_masks: Tensor | None = None
        if configured_role_masks is not None:
            metadata_role_masks = configured_role_masks & metadata_mask.unsqueeze(1)
            effective_role_masks = configured_role_masks & effective_mask.unsqueeze(1)
            # Global world dropout has one all-row fallback. Role routing needs
            # the same safety independently for each semantic subset, without
            # ever borrowing a world assigned to another role.
            role_had_world = metadata_role_masks.any(dim=-1)
            role_lost_all = role_had_world & ~effective_role_masks.any(dim=-1)
            if role_lost_all.any():
                role_freshness = freshness.unsqueeze(1).expand_as(metadata_role_masks)
                role_fallback = role_freshness.masked_fill(
                    ~metadata_role_masks,
                    torch.inf,
                ).argmin(dim=-1)
                lost_batch, lost_role = torch.nonzero(role_lost_all, as_tuple=True)
                effective_role_masks[
                    lost_batch,
                    lost_role,
                    role_fallback[lost_batch, lost_role],
                ] = True

        freshness_decay = F.softplus(self._freshness_decay_unconstrained).to(dtype=dtype)
        freshness_gate = torch.exp(-freshness_decay * freshness)
        uncertainty_decay = F.softplus(self._uncertainty_decay_unconstrained).to(dtype=dtype)
        uncertainty_gate = torch.exp(-uncertainty_decay * uncertainty)
        validity_gate = validity
        missingness_gate = 1.0 - missingness
        ungated_combined_gate = (
            freshness_gate * validity_gate * missingness_gate * uncertainty_gate
        )
        combined_gate = ungated_combined_gate * effective_mask.to(dtype)

        # Unless the caller supplies an explicit market state, build the query
        # from a reliability-weighted aggregate of the latest five world
        # summaries. The freshest world remains a deterministic fallback and
        # diagnostic, but no single clock defines the market context.
        masked_freshness = freshness.masked_fill(~effective_mask, torch.inf)
        latest_world_index = masked_freshness.argmin(dim=-1)
        batch_indices = torch.arange(batch_size, device=device)
        latest_state = summary[batch_indices, latest_world_index]
        has_eligible_world = effective_mask.any(dim=-1)
        latest_state = torch.where(has_eligible_world.unsqueeze(-1), latest_state, torch.zeros_like(latest_state))
        aggregate_weight = combined_gate / combined_gate.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )
        aggregate_state = torch.einsum("bw,bwd->bd", aggregate_weight, summary)
        has_reliability_mass = combined_gate.sum(dim=-1) > torch.finfo(dtype).eps
        aggregate_state = torch.where(
            has_reliability_mass.unsqueeze(-1),
            aggregate_state,
            latest_state,
        )

        if market_state is not None:
            supplied_state = market_state.to(device=device, dtype=dtype)
            if supplied_state.ndim == 3:
                supplied_state = supplied_state[:, -1, :]
            if supplied_state.ndim != 2:
                raise ValueError("market_state must have shape [B, D] or [B, T, D]")
            if supplied_state.shape[0] == 1 and batch_size != 1:
                supplied_state = supplied_state.expand(batch_size, -1)
            elif supplied_state.shape[0] != batch_size:
                raise ValueError(f"market_state batch dimension must be 1 or B={batch_size}")
            supplied_state = torch.nan_to_num(supplied_state)
            base_query = self.market_projection(supplied_state)
        else:
            base_query = self.latest_world_projection(aggregate_state)

        horizon_embeddings = self.horizon_embedding(horizon_values)
        horizon_context = self.horizon_projection(horizon_embeddings)
        query_state = self.query_norm(base_query.unsqueeze(1) + horizon_context)

        role_query_state: Tensor | None = None
        role_query_world_weights: Tensor | None = None
        role_has_eligible_world: Tensor | None = None
        role_combined_gate: Tensor | None = None
        if effective_role_masks is not None:
            role_combined_gate = (
                ungated_combined_gate.unsqueeze(1) * effective_role_masks.to(dtype)
            )
            role_query_world_weights = role_combined_gate / role_combined_gate.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(dtype).eps)
            role_masked_freshness = freshness.unsqueeze(1).expand_as(
                effective_role_masks
            ).masked_fill(~effective_role_masks, torch.inf)
            role_latest_index = role_masked_freshness.argmin(dim=-1)
            role_batch = torch.arange(batch_size, device=device).view(-1, 1).expand_as(
                role_latest_index
            )
            role_latest_state = summary[role_batch, role_latest_index]
            role_has_eligible_world = effective_role_masks.any(dim=-1)
            role_latest_state = torch.where(
                role_has_eligible_world.unsqueeze(-1),
                role_latest_state,
                torch.zeros_like(role_latest_state),
            )
            role_aggregate_state = torch.einsum(
                "brw,bwd->brd",
                role_query_world_weights,
                summary,
            )
            role_has_reliability_mass = (
                role_combined_gate.sum(dim=-1) > torch.finfo(dtype).eps
            )
            role_aggregate_state = torch.where(
                role_has_reliability_mass.unsqueeze(-1),
                role_aggregate_state,
                role_latest_state,
            )
            if market_state is not None:
                # An explicitly supplied causal market state is external to
                # the five world roles and may seed all three queries.
                role_base_query = base_query.unsqueeze(1).expand(
                    -1,
                    len(QUANTILE_ROLE_NAMES),
                    -1,
                )
            else:
                role_base_query = self.latest_world_projection(role_aggregate_state)
            role_query_state = self.query_norm(
                role_base_query.unsqueeze(1) + horizon_context.unsqueeze(2)
            )

        query = self.query_projection(query_state).reshape(
            batch_size, horizon_values.shape[1], self.num_heads, self.head_dim
        )
        keys = self.key_projection(horizon_world_states).reshape(
            batch_size,
            horizon_values.shape[1],
            world_count,
            self.num_heads,
            self.head_dim,
        )
        values = self.value_projection(horizon_world_states).reshape(
            batch_size,
            horizon_values.shape[1],
            world_count,
            self.num_heads,
            self.head_dim,
        )
        content_logits = torch.einsum("bhnd,bhwnd->bhnw", query, keys) / math.sqrt(self.head_dim)

        eps = torch.finfo(dtype).eps
        reliability_log_gate = torch.log(combined_gate.clamp_min(eps)).unsqueeze(1).unsqueeze(1)
        gated_logits = content_logits + reliability_log_gate
        attention_mask = effective_mask.unsqueeze(1).unsqueeze(1).expand_as(gated_logits)
        per_head_attention = self._route(gated_logits, attention_mask)
        attention_weights = per_head_attention.mean(dim=2)

        attended_heads = torch.einsum("bhnw,bhwnd->bhnd", per_head_attention, values)
        attended = self.attention_output(attended_heads.reshape(batch_size, horizon_values.shape[1], self.model_dim))
        context_gate = torch.sigmoid(self.context_gate(torch.cat((query_state, attended), dim=-1)))
        fused_context = self.output_norm(query_state + context_gate * self.dropout(attended))

        role_per_head_attention: Tensor | None = None
        role_attention_weights: Tensor | None = None
        role_context_gate: Tensor | None = None
        role_fused_contexts: Tensor | None = None
        role_has_route: Tensor | None = None
        global_attention_weights = attention_weights
        global_fused_context = fused_context
        if effective_role_masks is not None:
            if role_query_state is None:
                raise RuntimeError("semantic role queries were not constructed")
            role_count = len(QUANTILE_ROLE_NAMES)
            role_query = self.query_projection(role_query_state).reshape(
                batch_size,
                horizon_values.shape[1],
                role_count,
                self.num_heads,
                self.head_dim,
            )
            role_content_logits = torch.einsum(
                "bhrnd,bhwnd->bhrnw",
                role_query,
                keys,
            ) / math.sqrt(self.head_dim)
            if role_combined_gate is None:
                raise RuntimeError("semantic role reliability gates were not constructed")
            role_reliability_log_gate = torch.log(
                role_combined_gate.clamp_min(eps)
            ).unsqueeze(1).unsqueeze(3)
            role_logits = role_content_logits + role_reliability_log_gate
            role_attention_mask = effective_role_masks.unsqueeze(1).unsqueeze(3).expand_as(
                role_logits
            )
            role_per_head_attention = self._route(role_logits, role_attention_mask)
            role_attention_weights = role_per_head_attention.mean(dim=3)
            role_attended_heads = torch.einsum(
                "bhrnw,bhwnd->bhrnd",
                role_per_head_attention,
                values,
            )
            role_attended = self.attention_output(
                role_attended_heads.reshape(
                    batch_size,
                    horizon_values.shape[1],
                    role_count,
                    self.model_dim,
                )
            )
            role_context_gate = torch.sigmoid(
                self.context_gate(torch.cat((role_query_state, role_attended), dim=-1))
            )
            role_fused_contexts = self.output_norm(
                role_query_state + role_context_gate * self.dropout(role_attended)
            )
            role_has_route = role_attention_weights.sum(dim=-1) > eps
            active_role_count = role_has_route.sum(dim=-1, keepdim=True)
            attention_weights = role_attention_weights.sum(dim=2) / active_role_count.to(
                dtype
            ).clamp_min(1.0)
            role_context_weights = role_has_route.to(dtype) / active_role_count.to(
                dtype
            ).clamp_min(1.0)
            role_context_average = torch.einsum(
                "bhr,bhrd->bhd",
                role_context_weights,
                role_fused_contexts,
            )
            fused_context = torch.where(
                (active_role_count.squeeze(-1) > 0).unsqueeze(-1),
                role_context_average,
                global_fused_context,
            )

        projected_states = self.state_projection(horizon_world_states)
        state_mixture = torch.einsum("bhw,bhwd->bhd", attention_weights, projected_states)
        if role_fused_contexts is None:
            quantile_adjustment = self.quantile_head(fused_context)
        else:
            if self.role_quantile_head is None:
                raise RuntimeError("semantic role head was not constructed")
            quantile_adjustment = self.role_quantile_head(
                role_fused_contexts[:, :, 0, :],
                role_fused_contexts[:, :, 1, :],
                role_fused_contexts[:, :, 2, :],
            )
        residual_gate = torch.sigmoid(self._residual_gate_unconstrained).to(dtype=dtype)
        if self.residual_to_baseline:
            if baseline_quantiles is None:
                raise ValueError("baseline_quantiles are required when residual_to_baseline=True")
            baseline = torch.as_tensor(baseline_quantiles, device=device, dtype=dtype)
            expected_baseline = tuple(quantile_adjustment.shape)
            if tuple(baseline.shape) != expected_baseline:
                raise ValueError(
                    f"baseline_quantiles must have shape {expected_baseline}, got {tuple(baseline.shape)}"
                )
            if not torch.isfinite(baseline).all():
                raise ValueError("baseline_quantiles must be finite")
            quantile_predictions = _strict_monotone_quantile_projection(
                baseline + residual_gate * quantile_adjustment
            )
        else:
            baseline = None
            quantile_predictions = quantile_adjustment

        entropy = -(
            attention_weights * torch.log(attention_weights.clamp_min(eps))
        ).sum(dim=-1)
        public_effective_mask = (
            effective_role_masks.any(dim=1)
            if effective_role_masks is not None
            else effective_mask
        )
        diagnostics: dict[str, Any] = {
            "routing_mode": (
                "topk1" if self.hard_top1_inference and not self.training else self.routing_mode
            ),
            "horizons": horizon_values,
            "freshness": freshness,
            "freshness_decay": freshness_decay,
            "freshness_gate": freshness_gate,
            "validity_gate": validity_gate,
            "missingness": missingness,
            "missingness_gate": missingness_gate,
            "combined_world_gate": combined_gate,
            "metadata_world_mask": metadata_mask,
            "effective_world_mask": public_effective_mask,
            "global_effective_world_mask": effective_mask,
            "latest_world_index": latest_world_index,
            "query_world_weights": aggregate_weight,
            "has_eligible_world": has_eligible_world,
            "content_attention_logits": content_logits.mean(dim=2),
            "gated_attention_logits": gated_logits.mean(dim=2),
            "per_head_attention_weights": per_head_attention,
            "global_attention_weights": global_attention_weights,
            "attention_sum": attention_weights.sum(dim=-1),
            "attention_entropy": entropy,
            "active_world_count": (attention_weights > 1e-8).sum(dim=-1),
            "context_gate": context_gate,
            "future_gate": future_gate,
            "future_valid_fraction": future_valid_fraction,
            "uses_future_tokens": world_future_tokens is not None,
            "residual_to_baseline": self.residual_to_baseline,
            "residual_gate": residual_gate,
            "baseline_quantiles": baseline,
            "semantic_role_routing": self.semantic_role_routing,
            "quantile_role_names": QUANTILE_ROLE_NAMES,
            "configured_role_world_masks": configured_role_masks,
            "metadata_role_world_masks": metadata_role_masks,
            "effective_role_world_masks": effective_role_masks,
            "role_world_gates": role_combined_gate,
            "role_query_world_weights": role_query_world_weights,
            "role_has_eligible_world": role_has_eligible_world,
            "role_per_head_attention_weights": role_per_head_attention,
            "role_attention_weights": role_attention_weights,
            "role_attention_sum": (
                role_attention_weights.sum(dim=-1)
                if role_attention_weights is not None
                else None
            ),
            "role_has_route": role_has_route,
            "role_active_world_count": (
                (role_attention_weights > 1e-8).sum(dim=-1)
                if role_attention_weights is not None
                else None
            ),
            "role_context_gate": role_context_gate,
            "role_fused_contexts": role_fused_contexts,
        }
        regime_mixture = (
            torch.einsum("bhw,bwr->bhr", attention_weights, normalized_regimes)
            if normalized_regimes is not None
            else attention_weights
        )
        diagnostics["uncertainty"] = uncertainty
        diagnostics["uncertainty_decay"] = uncertainty_decay
        diagnostics["uncertainty_gate"] = uncertainty_gate
        diagnostics["world_regimes"] = normalized_regimes
        return WorldAttentionOutput(
            fused_context=fused_context,
            attention_weights=attention_weights,
            quantiles=quantile_predictions,
            state_mixture=state_mixture,
            regime_mixture=regime_mixture,
            world_gates=combined_gate,
            horizon_embeddings=horizon_embeddings,
            router_diagnostics=diagnostics,
        )


# A compact alias is useful in configuration-driven model factories.
MonotoneQuantileHead = StructuredMonotoneQuantileHead
