"""Causal split and objective utilities for the five-world JEPA trainer.

The helpers in this module deliberately keep data selection separate from model
code.  A split is built from positions, with two embargoes large enough to
contain the longest target.  Consequently a target attached to the final train
or selection row cannot land in the next fitted/evaluated segment.

Router regularisation follows the same separation of concerns: individual
queries are encouraged to be sparse and decisive, while load balance is
measured only on the *aggregate* batch marginal.  This permits a row to select
one relevant world without allowing the entire router to collapse onto that
world.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PurgedSplit:
    """A chronological train/selection/test split with explicit embargo rows.

    Positions always refer to the original ``index`` supplied to
    :func:`build_purged_train_selection_test`.  Timestamp containers preserve
    the input index type when it provides ``take`` (for example a pandas
    ``DatetimeIndex``); otherwise tuples are returned.
    """

    train_positions: np.ndarray
    selection_positions: np.ndarray
    test_positions: np.ndarray
    train_selection_purge_positions: np.ndarray
    selection_test_purge_positions: np.ndarray
    train_timestamps: Any
    selection_timestamps: Any
    test_timestamps: Any
    train_selection_purge_timestamps: Any
    selection_test_purge_timestamps: Any
    configured_purge_rows: int
    effective_purge_rows: int
    max_horizon: float
    max_horizon_rows: int

    @property
    def train(self) -> np.ndarray:
        """Short alias for ``train_positions``."""

        return self.train_positions

    @property
    def selection(self) -> np.ndarray:
        """Short alias for ``selection_positions``."""

        return self.selection_positions

    @property
    def test(self) -> np.ndarray:
        """Short alias for ``test_positions``."""

        return self.test_positions

    @property
    def train_selection_purge(self) -> np.ndarray:
        return self.train_selection_purge_positions

    @property
    def selection_test_purge(self) -> np.ndarray:
        return self.selection_test_purge_positions

    @property
    def is_target_safe(self) -> bool:
        """Whether longest-horizon endpoints remain before the next segment."""

        if not (len(self.train_positions) and len(self.selection_positions) and len(self.test_positions)):
            return False
        train_safe = (
            int(self.train_positions[-1]) + int(self.max_horizon_rows)
            < int(self.selection_positions[0])
        )
        selection_safe = (
            int(self.selection_positions[-1]) + int(self.max_horizon_rows)
            < int(self.test_positions[0])
        )
        return bool(train_safe and selection_safe)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit record of split boundaries."""

        def segment(positions: np.ndarray, timestamps: Any) -> dict[str, Any]:
            timestamp_values = list(timestamps)
            return {
                "rows": int(len(positions)),
                "position_start": int(positions[0]) if len(positions) else None,
                "position_end": int(positions[-1]) if len(positions) else None,
                "timestamp_start": str(timestamp_values[0]) if timestamp_values else None,
                "timestamp_end": str(timestamp_values[-1]) if timestamp_values else None,
            }

        return {
            "train": segment(self.train_positions, self.train_timestamps),
            "train_selection_purge": segment(
                self.train_selection_purge_positions,
                self.train_selection_purge_timestamps,
            ),
            "selection": segment(self.selection_positions, self.selection_timestamps),
            "selection_test_purge": segment(
                self.selection_test_purge_positions,
                self.selection_test_purge_timestamps,
            ),
            "test": segment(self.test_positions, self.test_timestamps),
            "configured_purge_rows": int(self.configured_purge_rows),
            "effective_purge_rows": int(self.effective_purge_rows),
            "max_horizon": float(self.max_horizon),
            "max_horizon_rows": int(self.max_horizon_rows),
            "target_safe": self.is_target_safe,
        }


def _take_index(index: Any, positions: np.ndarray) -> Any:
    take = getattr(index, "take", None)
    if callable(take):
        try:
            return take(positions)
        except (TypeError, ValueError, IndexError):
            pass
    values = list(index)
    return tuple(values[int(position)] for position in positions)


def _validate_chronological_index(index: Any) -> list[Any]:
    try:
        values = list(index)
    except TypeError as exc:
        raise TypeError("index must be a finite one-dimensional sequence") from exc
    if not values:
        raise ValueError("index cannot be empty")

    ndim = getattr(index, "ndim", 1)
    if int(ndim) != 1:
        raise ValueError("index must be one-dimensional")

    monotonic = getattr(index, "is_monotonic_increasing", None)
    has_duplicates = getattr(index, "has_duplicates", None)
    if monotonic is not None and not bool(monotonic):
        raise ValueError("index must be chronological and monotonically increasing")
    if has_duplicates is not None and bool(has_duplicates):
        raise ValueError("index timestamps must be unique")

    # A strict comparison catches duplicates and non-finite numeric values for
    # plain Python/numpy sequences that do not expose pandas-style metadata.
    if monotonic is None or has_duplicates is None:
        try:
            if any(not bool(left < right) for left, right in zip(values, values[1:])):
                raise ValueError("index must be strictly increasing with unique timestamps")
        except TypeError as exc:
            raise ValueError("index values must be chronologically orderable") from exc
    return values


def build_purged_train_selection_test(
    index: Any,
    minimum_train_rows: int,
    selection_rows: int,
    test_rows: int,
    purge_rows: int,
    max_horizon: float,
) -> PurgedSplit:
    """Build a latest-anchored target-safe chronological split.

    ``max_horizon`` is expressed in row units.  Fractional horizons are rounded
    up because a partially overlapping target is still leakage.  The effective
    purge is ``max(purge_rows, ceil(max_horizon))`` and is applied independently
    between train/selection and selection/test.  Extra history is assigned to
    training; selection and test retain their exact requested sizes.
    """

    values = _validate_chronological_index(index)
    minimum_train_rows = int(minimum_train_rows)
    selection_rows = int(selection_rows)
    test_rows = int(test_rows)
    purge_rows = int(purge_rows)
    max_horizon = float(max_horizon)
    if minimum_train_rows <= 0:
        raise ValueError("minimum_train_rows must be positive")
    if selection_rows <= 0:
        raise ValueError("selection_rows must be positive")
    if test_rows <= 0:
        raise ValueError("test_rows must be positive")
    if purge_rows < 0:
        raise ValueError("purge_rows cannot be negative")
    if not math.isfinite(max_horizon) or max_horizon <= 0.0:
        raise ValueError("max_horizon must be finite and positive")

    max_horizon_rows = int(math.ceil(max_horizon))
    effective_purge_rows = max(purge_rows, max_horizon_rows)
    required_rows = (
        minimum_train_rows
        + selection_rows
        + test_rows
        + 2 * effective_purge_rows
    )
    if len(values) < required_rows:
        raise ValueError(
            "Not enough rows for a target-safe split: "
            f"have={len(values)}, require={required_rows} "
            f"(train={minimum_train_rows}, selection={selection_rows}, "
            f"test={test_rows}, two purges={effective_purge_rows})."
        )

    test_start = len(values) - test_rows
    selection_test_purge_start = test_start - effective_purge_rows
    selection_start = selection_test_purge_start - selection_rows
    train_selection_purge_start = selection_start - effective_purge_rows

    train_positions = np.arange(0, train_selection_purge_start, dtype=np.int64)
    train_selection_purge_positions = np.arange(
        train_selection_purge_start,
        selection_start,
        dtype=np.int64,
    )
    selection_positions = np.arange(
        selection_start,
        selection_test_purge_start,
        dtype=np.int64,
    )
    selection_test_purge_positions = np.arange(
        selection_test_purge_start,
        test_start,
        dtype=np.int64,
    )
    test_positions = np.arange(test_start, len(values), dtype=np.int64)

    split = PurgedSplit(
        train_positions=train_positions,
        selection_positions=selection_positions,
        test_positions=test_positions,
        train_selection_purge_positions=train_selection_purge_positions,
        selection_test_purge_positions=selection_test_purge_positions,
        train_timestamps=_take_index(index, train_positions),
        selection_timestamps=_take_index(index, selection_positions),
        test_timestamps=_take_index(index, test_positions),
        train_selection_purge_timestamps=_take_index(index, train_selection_purge_positions),
        selection_test_purge_timestamps=_take_index(index, selection_test_purge_positions),
        configured_purge_rows=purge_rows,
        effective_purge_rows=effective_purge_rows,
        max_horizon=max_horizon,
        max_horizon_rows=max_horizon_rows,
    )
    if len(train_positions) < minimum_train_rows or not split.is_target_safe:
        # The arithmetic above is intentionally simple; retain this executable
        # invariant so later refactors cannot silently weaken the causal gate.
        raise RuntimeError("internal error: constructed split is not target-safe")
    return split


def _quantile_levels_tensor(
    levels: Sequence[float] | Tensor,
    *,
    quantile_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    values = torch.as_tensor(levels, device=device, dtype=dtype)
    if values.ndim != 1 or values.numel() != quantile_count:
        raise ValueError(f"levels must have exactly Q={quantile_count} entries")
    if not torch.isfinite(values).all() or not ((values > 0.0) & (values < 1.0)).all():
        raise ValueError("quantile levels must be finite and lie strictly between zero and one")
    if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
        raise ValueError("quantile levels must be strictly increasing in prediction order")
    return values


def _target_mask(targets: Tensor, valid_mask: Tensor | None) -> Tensor:
    mask = torch.isfinite(targets)
    if valid_mask is not None:
        supplied = torch.as_tensor(valid_mask, device=targets.device, dtype=torch.bool)
        if supplied.shape != targets.shape:
            raise ValueError("valid_mask must have shape [B, H]")
        mask = mask & supplied
    if not mask.any():
        raise ValueError("no finite valid targets were supplied")
    return mask


def quantile_pinball_loss(
    predictions: Tensor,
    targets: Tensor,
    levels: Sequence[float] | Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return mean pinball loss for ordered ``[B, H, Q]`` predictions."""

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [B, H, Q]")
    if not predictions.is_floating_point():
        raise TypeError("predictions must be floating point")
    targets = torch.as_tensor(targets, device=predictions.device, dtype=predictions.dtype)
    if targets.shape != predictions.shape[:2]:
        raise ValueError("targets must have shape [B, H] matching predictions")
    quantiles = _quantile_levels_tensor(
        levels,
        quantile_count=predictions.shape[-1],
        device=predictions.device,
        dtype=predictions.dtype,
    )
    mask = _target_mask(targets, valid_mask)
    if not torch.isfinite(predictions[mask]).all():
        raise ValueError("predictions must be finite wherever targets are valid")

    errors = targets.unsqueeze(-1) - predictions
    losses = torch.maximum(
        quantiles.view(1, 1, -1) * errors,
        (quantiles - 1.0).view(1, 1, -1) * errors,
    )
    return losses[mask].mean()


def _expand_horizons(horizons: Tensor | Sequence[float] | float, reference: Tensor) -> Tensor:
    batch_size, horizon_count = reference.shape[:2]
    values = torch.as_tensor(horizons, device=reference.device, dtype=reference.dtype)
    if values.ndim == 0:
        if horizon_count != 1:
            raise ValueError("a scalar horizon is valid only when H=1")
        values = values.reshape(1, 1).expand(batch_size, 1)
    elif values.ndim == 1:
        if values.numel() != horizon_count:
            raise ValueError(f"horizons must have H={horizon_count} entries")
        values = values.unsqueeze(0).expand(batch_size, -1)
    elif values.ndim == 2:
        if values.shape[1] != horizon_count:
            raise ValueError(f"horizons second dimension must be H={horizon_count}")
        if values.shape[0] == 1 and batch_size != 1:
            values = values.expand(batch_size, -1)
        elif values.shape[0] != batch_size:
            raise ValueError(f"horizons first dimension must be 1 or B={batch_size}")
    else:
        raise ValueError("horizons must be scalar, [H], or [B, H]")
    if not torch.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError("horizons must be finite and positive")
    return values


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("masked mean values and mask must have the same shape")
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()


def _router_eligibility(output: Any, attention: Tensor) -> Tensor:
    batch_size, _, world_count = attention.shape
    diagnostics = getattr(output, "router_diagnostics", None)
    if diagnostics is None:
        diagnostics = getattr(output, "diagnostics", None)
    effective_mask = None
    if isinstance(diagnostics, dict):
        effective_mask = diagnostics.get("effective_world_mask")
    if effective_mask is None:
        gates = getattr(output, "world_gates", None)
        if gates is not None:
            gates = torch.as_tensor(gates, device=attention.device)
            if gates.shape == (batch_size, world_count):
                effective_mask = gates > 0.0
    if effective_mask is None:
        return torch.ones((batch_size, world_count), device=attention.device, dtype=torch.bool)
    mask = torch.as_tensor(effective_mask, device=attention.device, dtype=torch.bool)
    if mask.shape != (batch_size, world_count):
        raise ValueError("effective_world_mask must have shape [B, W]")
    return mask


def router_regularized_loss(
    output: Any,
    targets: Tensor,
    horizons: Tensor | Sequence[float] | float,
    *,
    quantile_levels: Sequence[float] | Tensor = (0.05, 0.25, 0.50, 0.75, 0.95),
    valid_mask: Tensor | None = None,
    quantile_weight: float = 1.0,
    load_balance_weight: float = 0.02,
    collapse_weight: float = 0.05,
    router_entropy_weight: float = 0.01,
    horizon_smoothness_weight: float = 0.02,
    attention_smoothness_fraction: float = 0.25,
    width_monotonicity_weight: float = 0.02,
    quantile_crossing_weight: float = 0.05,
    minimum_batch_world_entropy: float = 0.65,
    maximum_world_share: float = 0.80,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Supervise quantiles while keeping sparse world routing operational.

    Load balance and collapse checks operate on the batch/horizon marginal, not
    on individual rows.  In contrast, ``router_entropy_weight`` minimizes mean
    per-query entropy, so a query may confidently select one world.  These
    opposing levels of regularisation produce sparse local decisions without a
    globally dead specialist.
    """

    predictions = getattr(output, "quantiles", None)
    attention = getattr(output, "attention_weights", None)
    if predictions is None or attention is None:
        raise TypeError("output must expose quantiles and attention_weights")
    if predictions.ndim != 3:
        raise ValueError("output.quantiles must have shape [B, H, Q]")
    if attention.ndim != 3 or attention.shape[:2] != predictions.shape[:2]:
        raise ValueError("output.attention_weights must have shape [B, H, W]")
    if attention.shape[-1] <= 0:
        raise ValueError("router must contain at least one world")
    if not torch.isfinite(attention).all() or (attention < -float(eps)).any():
        raise ValueError("attention weights must be finite and non-negative")
    attention = attention.clamp_min(0.0)

    target_tensor = torch.as_tensor(targets, device=predictions.device, dtype=predictions.dtype)
    if target_tensor.shape != predictions.shape[:2]:
        raise ValueError("targets must have shape [B, H]")
    target_valid = _target_mask(target_tensor, valid_mask)
    pinball = quantile_pinball_loss(
        predictions,
        target_tensor,
        quantile_levels,
        valid_mask=target_valid,
    )
    # Validate the level/order contract even when the caller sets quantile
    # weight to zero for an ablation.
    _quantile_levels_tensor(
        quantile_levels,
        quantile_count=predictions.shape[-1],
        device=predictions.device,
        dtype=predictions.dtype,
    )

    horizon_values = _expand_horizons(horizons, predictions)
    eligibility = _router_eligibility(output, attention)
    eligibility_by_query = eligibility.unsqueeze(1).expand_as(attention)
    if (attention.masked_select(~eligibility_by_query) > 1e-6).any():
        raise ValueError("masked worlds must have zero attention")

    attention_sum = attention.sum(dim=-1, keepdim=True)
    has_route = attention_sum.squeeze(-1) > float(eps)
    normalized_attention = torch.where(
        has_route.unsqueeze(-1),
        attention / attention_sum.clamp_min(float(eps)),
        torch.zeros_like(attention),
    )
    query_mask = target_valid & has_route
    zero = attention.sum() * 0.0

    if query_mask.any():
        weighted_attention = normalized_attention * query_mask.unsqueeze(-1)
        marginal = weighted_attention.sum(dim=(0, 1))
        marginal = marginal / marginal.sum().clamp_min(float(eps))

        availability_count = (
            eligibility_by_query.to(attention.dtype) * query_mask.unsqueeze(-1)
        ).sum(dim=(0, 1))
        availability_target = availability_count / availability_count.sum().clamp_min(float(eps))
        globally_available = availability_count > 0
        available_world_count = globally_available.sum()
        load_balance = available_world_count.to(attention.dtype) * (
            marginal - availability_target
        ).pow(2).sum()

        row_entropy = -(
            normalized_attention
            * torch.log(normalized_attention.clamp_min(float(eps)))
        ).sum(dim=-1)
        eligible_count = eligibility_by_query.sum(dim=-1)
        entropy_denominator = torch.log(eligible_count.clamp_min(2).to(attention.dtype))
        normalized_row_entropy = torch.where(
            eligible_count > 1,
            row_entropy / entropy_denominator.clamp_min(float(eps)),
            torch.zeros_like(row_entropy),
        )
        confidence_entropy = normalized_row_entropy[query_mask].mean()

        if int(available_world_count.detach().cpu()) > 1:
            marginal_entropy = -(
                marginal[globally_available]
                * torch.log(marginal[globally_available].clamp_min(float(eps)))
            ).sum()
            normalized_marginal_entropy = marginal_entropy / torch.log(
                available_world_count.to(attention.dtype)
            )
            max_share = marginal[globally_available].max()
            collapse_penalty = torch.relu(
                attention.new_tensor(float(minimum_batch_world_entropy))
                - normalized_marginal_entropy
            ).pow(2) + torch.relu(
                max_share - attention.new_tensor(float(maximum_world_share))
            ).pow(2)
        else:
            normalized_marginal_entropy = zero
            max_share = marginal.max() if marginal.numel() else zero
            collapse_penalty = zero
    else:
        marginal = torch.zeros(attention.shape[-1], device=attention.device, dtype=attention.dtype)
        load_balance = zero
        confidence_entropy = zero
        normalized_marginal_entropy = zero
        max_share = zero
        collapse_penalty = zero

    # Sort only for cross-horizon regularisation.  Forecast/target association
    # and pinball supervision remain in the caller's original query order.
    sorted_horizons, order = torch.sort(horizon_values, dim=1)
    if sorted_horizons.shape[1] > 1:
        delta_horizon = sorted_horizons[:, 1:] - sorted_horizons[:, :-1]
        if (delta_horizon <= float(eps)).any():
            raise ValueError("horizons must be unique within each sample")
        prediction_order = order.unsqueeze(-1).expand_as(predictions)
        attention_order = order.unsqueeze(-1).expand_as(normalized_attention)
        sorted_predictions = torch.gather(predictions, 1, prediction_order)
        sorted_attention = torch.gather(normalized_attention, 1, attention_order)
        sorted_valid = torch.gather(target_valid, 1, order)
        adjacent_valid = sorted_valid[:, 1:] & sorted_valid[:, :-1]

        quantile_slopes = (
            sorted_predictions[:, 1:] - sorted_predictions[:, :-1]
        ) / delta_horizon.unsqueeze(-1)
        attention_slopes = (
            sorted_attention[:, 1:] - sorted_attention[:, :-1]
        ) / delta_horizon.unsqueeze(-1)
        quantile_smoothness = _masked_mean(
            quantile_slopes.pow(2).mean(dim=-1),
            adjacent_valid,
        )
        attention_smoothness = _masked_mean(
            attention_slopes.pow(2).mean(dim=-1),
            adjacent_valid,
        )
        horizon_smoothness = (
            quantile_smoothness
            + float(attention_smoothness_fraction) * attention_smoothness
        )

        pair_count = predictions.shape[-1] // 2
        if pair_count:
            lower_indices = torch.arange(pair_count, device=predictions.device)
            upper_indices = predictions.shape[-1] - 1 - lower_indices
            interval_widths = (
                sorted_predictions.index_select(-1, upper_indices)
                - sorted_predictions.index_select(-1, lower_indices)
            )
            width_slopes = (
                interval_widths[:, 1:] - interval_widths[:, :-1]
            ) / delta_horizon.unsqueeze(-1)
            width_monotonicity = _masked_mean(
                torch.relu(-width_slopes).pow(2).mean(dim=-1),
                adjacent_valid,
            )
        else:
            width_monotonicity = zero
    else:
        quantile_smoothness = zero
        attention_smoothness = zero
        horizon_smoothness = zero
        width_monotonicity = zero

    if predictions.shape[-1] > 1:
        crossing_by_query = torch.relu(
            predictions[..., :-1] - predictions[..., 1:]
        ).pow(2).mean(dim=-1)
        quantile_crossing = _masked_mean(crossing_by_query, target_valid)
    else:
        quantile_crossing = zero

    total = (
        float(quantile_weight) * pinball
        + float(load_balance_weight) * load_balance
        + float(collapse_weight) * collapse_penalty
        + float(router_entropy_weight) * confidence_entropy
        + float(horizon_smoothness_weight) * horizon_smoothness
        + float(width_monotonicity_weight) * width_monotonicity
        + float(quantile_crossing_weight) * quantile_crossing
    )
    if not torch.isfinite(total):
        raise FloatingPointError("router loss is non-finite")

    def scalar(value: Tensor) -> float:
        return float(value.detach().cpu())

    diagnostics = {
        "loss": scalar(total),
        "pinball_loss": scalar(pinball),
        "router_load_balance_loss": scalar(load_balance),
        "load_balance_loss": scalar(load_balance),
        "router_collapse_penalty": scalar(collapse_penalty),
        "collapse_penalty": scalar(collapse_penalty),
        "router_confidence_entropy_loss": scalar(confidence_entropy),
        "confidence_entropy_loss": scalar(confidence_entropy),
        "normalized_batch_world_entropy": scalar(normalized_marginal_entropy),
        "maximum_batch_world_share": scalar(max_share),
        "quantile_horizon_smoothness_loss": scalar(quantile_smoothness),
        "attention_horizon_smoothness_loss": scalar(attention_smoothness),
        "horizon_smoothness_loss": scalar(horizon_smoothness),
        "interval_width_monotonicity_loss": scalar(width_monotonicity),
        "quantile_crossing_loss": scalar(quantile_crossing),
        "valid_target_count": float(target_valid.sum().detach().cpu()),
        "routed_query_count": float(query_mask.sum().detach().cpu()),
    }
    for world_index, share in enumerate(marginal):
        diagnostics[f"world_{world_index}_marginal_share"] = scalar(share)
    return total, diagnostics


def self_supervised_regime_loss(
    regime_probs: Tensor,
    adjacency_pairs: Tensor | Sequence[Sequence[int]] | None = None,
    *,
    confidence_weight: float = 0.25,
    confidence_hinge_weight: float = 1.0,
    balance_weight: float = 1.0,
    consistency_weight: float = 0.25,
    minimum_assignment_confidence: float = 0.55,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Learn semi-state regimes without requiring external class labels.

    Low per-sample entropy makes a regime assignment interpretable, high batch
    marginal entropy prevents every sample from choosing the same regime, and
    Jensen-Shannon consistency keeps explicitly adjacent samples stable.  The
    confidence and balance terms intentionally operate at different levels;
    together their optimum is confident assignments distributed across the
    available regime prototypes.
    """

    if regime_probs.ndim != 2 or regime_probs.shape[0] <= 0 or regime_probs.shape[1] <= 0:
        raise ValueError("regime_probs must have non-empty shape [B, R]")
    if not 0.0 < float(minimum_assignment_confidence) <= 1.0:
        raise ValueError("minimum_assignment_confidence must be in (0, 1]")
    if not regime_probs.is_floating_point():
        raise TypeError("regime_probs must be floating point")
    if not torch.isfinite(regime_probs).all() or (regime_probs < 0.0).any():
        raise ValueError("regime_probs must be finite and non-negative")
    row_mass = regime_probs.sum(dim=-1, keepdim=True)
    if (row_mass <= 0.0).any():
        raise ValueError("every regime probability row must have positive mass")
    probabilities = regime_probs / row_mass
    batch_size, regime_count = probabilities.shape
    zero = probabilities.sum() * 0.0

    if regime_count > 1:
        log_regime_count = math.log(regime_count)
        per_sample_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(float(eps)))
        ).sum(dim=-1) / log_regime_count
        confidence_loss = per_sample_entropy.mean()
        confidence_hinge = torch.relu(
            probabilities.new_tensor(float(minimum_assignment_confidence))
            - probabilities.max(dim=-1).values
        ).pow(2).mean()

        marginal = probabilities.mean(dim=0)
        # KL(marginal || uniform), divided by log(R), is zero at a balanced
        # marginal and one for a fully collapsed marginal.
        balance_loss = (
            marginal
            * torch.log((marginal * regime_count).clamp_min(float(eps)))
        ).sum() / log_regime_count
        normalized_marginal_entropy = -(
            marginal * torch.log(marginal.clamp_min(float(eps)))
        ).sum() / log_regime_count
        maximum_regime_share = marginal.max()
    else:
        confidence_loss = zero
        confidence_hinge = zero
        balance_loss = zero
        normalized_marginal_entropy = zero
        maximum_regime_share = probabilities.new_tensor(1.0)

    consistency_loss = zero
    pair_count = 0
    if adjacency_pairs is not None:
        pairs = torch.as_tensor(adjacency_pairs, device=probabilities.device, dtype=torch.long)
        if pairs.numel():
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("adjacency_pairs must have shape [P, 2]")
            if (pairs < 0).any() or (pairs >= batch_size).any():
                raise ValueError("adjacency_pairs contains an out-of-range sample index")
            left = probabilities[pairs[:, 0]]
            right = probabilities[pairs[:, 1]]
            midpoint = 0.5 * (left + right)
            left_kl = (
                left
                * (
                    torch.log(left.clamp_min(float(eps)))
                    - torch.log(midpoint.clamp_min(float(eps)))
                )
            ).sum(dim=-1)
            right_kl = (
                right
                * (
                    torch.log(right.clamp_min(float(eps)))
                    - torch.log(midpoint.clamp_min(float(eps)))
                )
            ).sum(dim=-1)
            consistency_loss = (0.5 * (left_kl + right_kl)).mean()
            pair_count = int(pairs.shape[0])

    total = (
        float(confidence_weight) * confidence_loss
        + float(confidence_hinge_weight) * confidence_hinge
        + float(balance_weight) * balance_loss
        + float(consistency_weight) * consistency_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("self-supervised regime loss is non-finite")

    def scalar(value: Tensor) -> float:
        return float(value.detach().cpu())

    diagnostics = {
        "loss": scalar(total),
        "regime_confidence_loss": scalar(confidence_loss),
        "confidence_loss": scalar(confidence_loss),
        "regime_confidence_hinge_loss": scalar(confidence_hinge),
        "confidence_hinge_loss": scalar(confidence_hinge),
        "regime_balance_loss": scalar(balance_loss),
        "batch_balance_loss": scalar(balance_loss),
        "regime_collapse_penalty": scalar(balance_loss),
        "adjacent_regime_consistency_loss": scalar(consistency_loss),
        "consistency_loss": scalar(consistency_loss),
        "normalized_batch_regime_entropy": scalar(normalized_marginal_entropy),
        "maximum_batch_regime_share": scalar(maximum_regime_share),
        "adjacency_pair_count": float(pair_count),
    }
    return total, diagnostics


__all__ = [
    "PurgedSplit",
    "build_purged_train_selection_test",
    "quantile_pinball_loss",
    "router_regularized_loss",
    "self_supervised_regime_loss",
]
