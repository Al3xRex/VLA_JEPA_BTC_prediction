"""End-to-end five-world JEPA shadow training and evaluation.

This module deliberately lives beside, rather than inside, the legacy fusion
pipeline.  ``mode=shadow`` can read the same causal source tables but writes a
separate artifact tree and cannot replace production forecasts.  Promotion is
an explicit evidence result, not a side effect of a successful training run.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dual_model_forecaster.config import load_config
from dual_model_forecaster.data import ForecastDataBundle, load_forecast_data
from dual_model_forecaster.utils import set_global_seed
from dual_model_forecaster.world_jepa.config import (
    WORLD_NAMES,
    WorldJEPAConfig,
    load_world_jepa_config,
)
from dual_model_forecaster.world_jepa.datasets import WorldJEPADataset, world_jepa_collate
from dual_model_forecaster.world_jepa.evaluation import (
    QUANTILE_COLUMNS,
    QUANTILES,
    QuantileCalibrator,
    add_forecast_losses,
    causal_ewma_baseline,
    evaluate_promotion_gates,
    fit_quantile_calibrator,
    moving_block_bootstrap_difference,
    summarize_forecasts,
    weighted_quantile,
)
from dual_model_forecaster.world_jepa.feature_audit import (
    WorldFeatureAuditConfig,
    audit_all_worlds,
    save_feature_audit_report,
)
from dual_model_forecaster.world_jepa.losses import world_jepa_loss
from dual_model_forecaster.world_jepa.preprocessing import (
    RobustWorldScaler,
    build_world_model_frame,
)
from dual_model_forecaster.world_jepa.regimes import (
    CausalRegimeCodebook,
    causal_smooth_regime_probabilities,
)
from dual_model_forecaster.world_jepa.router import (
    QUANTILE_ROLE_NAMES,
    ContextRelevantWorldRouter,
)
from dual_model_forecaster.world_jepa.training import (
    PurgedSplit,
    build_purged_train_selection_test,
    quantile_pinball_loss,
    router_regularized_loss,
    self_supervised_regime_loss,
)
from dual_model_forecaster.world_jepa.world_encoder import EMAMomentumSchedule, WorldJEPAEncoder
from compute_data.ta.pca_latent import causal_robust_standardize


PRODUCTION_QUANTILE_ROLE_WORLDS: dict[str, tuple[str, ...]] = {
    "center": ("edges", "movement"),
    "width": ("environment", "edges", "movement", "liquidation"),
    "tails": ("structure", "environment", "liquidation"),
}


def production_quantile_role_masks(
    world_names: Sequence[str] = WORLD_NAMES,
) -> dict[str, torch.Tensor]:
    """Map production world semantics to center, width, and tail masks."""

    names = tuple(str(name) for name in world_names)
    if len(set(names)) != len(names):
        raise ValueError("world_names must be unique when constructing semantic role masks")
    missing = sorted(
        {
            world
            for worlds in PRODUCTION_QUANTILE_ROLE_WORLDS.values()
            for world in worlds
            if world not in names
        }
    )
    if missing:
        raise ValueError(f"Production semantic role worlds are unavailable: {missing}")
    return {
        role: torch.tensor(
            [world in PRODUCTION_QUANTILE_ROLE_WORLDS[role] for world in names],
            dtype=torch.bool,
        )
        for role in QUANTILE_ROLE_NAMES
    }


def semantic_role_routing_metadata(
    router: ContextRelevantWorldRouter,
    world_names: Sequence[str] = WORLD_NAMES,
) -> dict[str, Any] | None:
    """Return the immutable semantic-role contract needed to replay a router."""

    if not router.semantic_role_routing:
        return None
    names = tuple(str(name) for name in world_names)
    masks = production_quantile_role_masks(names)
    return {
        "name": "production_quantile_roles_v1",
        "world_order": list(names),
        "role_worlds": {
            role: [name for name, included in zip(names, masks[role].tolist()) if included]
            for role in QUANTILE_ROLE_NAMES
        },
        "role_masks": {
            role: [bool(value) for value in masks[role].tolist()]
            for role in QUANTILE_ROLE_NAMES
        },
    }


@dataclass(frozen=True)
class RuntimeOverrides:
    """Optional bounded overrides used for smoke runs and CI."""

    world_epochs: int | None = None
    router_epochs: int | None = None
    batch_size: int | None = None
    world_stride: int = 1
    maximum_world_batches_per_epoch: int | None = None
    maximum_router_batches_per_epoch: int | None = None
    device: str | None = None
    validated_test_suite: bool = False


@dataclass
class PreparedWorld:
    name: str
    raw_columns: list[str]
    model_columns: list[str]
    channel_columns: dict[str, list[str]]
    frame: pd.DataFrame
    scaler: RobustWorldScaler


@dataclass
class EncodedWorldState:
    index: pd.Index
    tokens: np.ndarray
    future_tokens: np.ndarray
    future_horizons: np.ndarray
    regimes: np.ndarray
    uncertainty: np.ndarray
    valid_context_fraction: np.ndarray
    valid_feature_fraction: np.ndarray
    novelty: np.ndarray


@dataclass
class RouterPanel:
    index: pd.Index
    world_tokens: np.ndarray
    world_regimes: np.ndarray
    freshness: np.ndarray
    validity: np.ndarray
    missingness: np.ndarray
    uncertainty: np.ndarray
    novelty: np.ndarray
    future_tokens: np.ndarray | None
    future_horizons: np.ndarray | None
    market_state: np.ndarray | None
    baseline_quantiles: np.ndarray | None
    baseline_horizons: np.ndarray | None

    def positions_for(self, timestamps: Iterable[Any]) -> np.ndarray:
        requested = pd.Index(list(timestamps))
        positions = self.index.get_indexer(requested)
        return positions[positions >= 0].astype(np.int64)


def build_dense_return_targets(
    close: pd.Series,
    horizons: Sequence[float],
    *,
    return_type: str = "log",
) -> pd.DataFrame:
    """Build observed daily labels for a dense integer training horizon grid."""

    numeric = pd.to_numeric(close, errors="coerce").astype(float)
    targets = pd.DataFrame(index=numeric.index)
    for horizon_value in horizons:
        horizon = int(round(float(horizon_value)))
        if horizon <= 0 or not np.isclose(float(horizon_value), horizon):
            raise ValueError("Observed training labels require positive integer-day horizons.")
        shifted = numeric.shift(-horizon)
        if return_type == "log":
            values = np.log(shifted.where(shifted > 0.0)) - np.log(numeric.where(numeric > 0.0))
        elif return_type == "simple":
            values = shifted / numeric - 1.0
        else:
            raise ValueError(f"Unsupported return_type={return_type!r}.")
        targets[f"target_{horizon}d"] = values
    return targets.replace([np.inf, -np.inf], np.nan)


def _load_selected_columns(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"World feature manifest not found at {path}. Run scripts/audit_world_features.py first."
        )
    payload = json.loads(path.read_text())
    selected = payload.get("selected_columns_by_world")
    if selected is None and "worlds" in payload:
        selected = {
            world: world_payload.get("selected_columns", [])
            for world, world_payload in payload["worlds"].items()
        }
    if not isinstance(selected, dict):
        raise ValueError("Feature manifest does not contain selected_columns_by_world.")
    missing = [world for world in WORLD_NAMES if world not in selected]
    if missing:
        raise ValueError(f"Feature manifest is missing worlds: {missing}")
    resolved = {world: [str(value) for value in selected[world]] for world in WORLD_NAMES}
    empty = [world for world, columns in resolved.items() if not columns]
    if empty:
        raise ValueError(
            "Every JEPA world needs representation inputs; the feature manifest is empty for "
            f"{empty}. Re-run the audit with representation-coverage minima."
        )
    return resolved


def _prepare_worlds(
    bundle: ForecastDataBundle,
    selected: Mapping[str, Sequence[str]],
    split: PurgedSplit,
    *,
    causalize_selected_features: bool = False,
) -> dict[str, PreparedWorld]:
    train_positions = split.train_positions
    prepared: dict[str, PreparedWorld] = {}
    for world in WORLD_NAMES:
        raw_frame = bundle.buckets[world]
        columns = [str(column) for column in selected[world]]
        missing = [column for column in columns if column not in raw_frame.columns]
        if missing:
            raise KeyError(f"{world} manifest columns are absent from source data: {missing[:8]}")
        selected_frame = raw_frame.loc[:, columns]
        if causalize_selected_features:
            # Full-history PCA loadings cannot be repaired downstream, so they
            # are excluded from the causal experiment. Scalar global
            # normalizations are neutralized by a past-only robust transform.
            safe_columns = [column for column in columns if "pca" not in column.lower()]
            if not safe_columns:
                raise ValueError(f"{world} has no non-PCA inputs for the causal JEPA experiment.")
            selected_frame = selected_frame.loc[:, safe_columns].copy()
            for column in safe_columns:
                source = pd.to_numeric(selected_frame[column], errors="coerce").astype(float)
                transformed = causal_robust_standardize(
                    source,
                    min_periods=30,
                    window=730,
                )
                enough_history = source.notna().cumsum() >= 30
                selected_frame[column] = transformed.where(source.notna() & enough_history)
            columns = safe_columns
        scaler = RobustWorldScaler.fit(selected_frame.iloc[train_positions])
        model_frame, channel_columns = build_world_model_frame(selected_frame, scaler)
        prepared[world] = PreparedWorld(
            name=world,
            raw_columns=columns,
            model_columns=model_frame.columns.tolist(),
            channel_columns=channel_columns,
            frame=model_frame,
            scaler=scaler,
        )
    return prepared


def _refresh_feature_audit(
    bundle: ForecastDataBundle,
    config: WorldJEPAConfig,
) -> dict[str, Any]:
    """Re-select world inputs from the current causal source snapshot."""

    target_horizon = 7
    target_column = f"target_{target_horizon}d"
    if target_column not in bundle.targets:
        raise KeyError(f"Feature audit requires {target_column} in the loaded target grid.")
    minima = {
        "structure": 6,
        "environment": 6,
        "edges": 4,
        "movement": 8,
        "liquidation": 8,
    }
    minimum_rows = (
        int(config.splits["minimum_train_rows"])
        + int(config.splits["selection_rows"])
        + int(config.splits["test_rows"])
        + 2 * int(config.splits["purge_days"])
    )
    audit_config = WorldFeatureAuditConfig(
        target_horizon=target_horizon,
        world_target_horizons=tuple(int(value) for value in config.horizons["world_target_days"]),
        feature_budget=12,
        per_world_feature_budget={
            world: spec.feature_budget for world, spec in config.worlds.items()
        },
        minimum_world_features=4,
        per_world_minimum_features=minima,
        selection_rows=int(config.splits["selection_rows"]),
        test_rows=int(config.splits["test_rows"]),
        minimum_train_rows=int(config.splits["minimum_train_rows"]),
        purge_gap_rows=int(config.splits["purge_days"]),
        min_fold_train_rows=int(config.splits["minimum_train_rows"]),
        min_aligned_rows=minimum_rows,
    )
    audit = audit_all_worlds(
        {world: bundle.buckets[world] for world in WORLD_NAMES},
        bundle.targets[target_column].rename(target_column),
        audit_config,
    )
    save_feature_audit_report(
        audit,
        config.feature_manifest.parent,
        manifest_path=config.feature_manifest,
    )
    if not bool(audit.get("all_world_minimums_met", False)):
        raise ValueError("Current feature audit cannot populate every JEPA world minimum.")
    return audit


def _tensor_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
        if all(np.isfinite(float(row[key])) for row in rows)
    }


def _train_world_encoder(
    prepared: PreparedWorld,
    config: WorldJEPAConfig,
    split: PurgedSplit,
    runtime: RuntimeOverrides,
) -> tuple[WorldJEPAEncoder, list[dict[str, Any]]]:
    world_spec = config.worlds[prepared.name]
    training_end = int(split.train_positions[-1]) + 1
    dataset = WorldJEPADataset(
        prepared.frame.iloc[:training_end],
        world_name=prepared.name,
        horizons=config.horizons["world_target_days"],
        context_length=world_spec.context_length,
        target_block_length=world_spec.target_block_length,
        target_window_mode=str(objective_window_mode)
        if (objective_window_mode := config.objective.get("target_window_mode", "forward"))
        else "forward",
        stride=max(1, int(runtime.world_stride)),
    )
    if len(dataset) == 0:
        raise ValueError(f"{prepared.name} has no complete causal JEPA training samples.")
    batch_size = int(runtime.batch_size or config.training["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=world_jepa_collate,
    )
    device = str(runtime.device or config.training["device"])
    model = WorldJEPAEncoder.from_config(
        len(prepared.model_columns),
        config.encoder,
        world_name=prepared.name,
        num_regimes=world_spec.regime_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    epochs = int(runtime.world_epochs or config.training["world_epochs"])
    effective_batches = len(loader)
    if runtime.maximum_world_batches_per_epoch is not None:
        effective_batches = min(effective_batches, int(runtime.maximum_world_batches_per_epoch))
    total_optimizer_steps = max(1, epochs * effective_batches)
    initial_half_life = float(
        config.training.get(
            "ema_initial_half_life_steps",
            max(4.0, 0.10 * total_optimizer_steps),
        )
    )
    final_momentum = float(config.training.get("ema_final_momentum", config.encoder["ema_decay"]))
    ema_schedule = EMAMomentumSchedule(
        total_optimizer_steps=total_optimizer_steps,
        initial_half_life_steps=initial_half_life,
        final_momentum=final_momentum,
    )
    optimizer_step = 0
    history: list[dict[str, Any]] = []
    objective = config.objective
    for epoch in range(epochs):
        model.train()
        epoch_rows: list[dict[str, float]] = []
        for batch_number, raw_batch in enumerate(loader):
            if (
                runtime.maximum_world_batches_per_epoch is not None
                and batch_number >= int(runtime.maximum_world_batches_per_epoch)
            ):
                break
            batch = _tensor_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["x"],
                batch["horizons"],
                padding_mask=batch["padding_mask"],
                time_mask=batch["time_mask"],
                feature_mask=batch["feature_mask"],
                target_x=batch["target_x"],
                target_padding_mask=batch["target_padding_mask"],
                target_time_mask=batch["target_time_mask"],
                target_feature_mask=batch["target_feature_mask"],
                world_ids=batch["world_ids"],
            )
            latent_loss, latent_metrics = world_jepa_loss(
                output,
                adjacency_pairs=batch["adjacency_pairs"],
                lambda_variance=float(objective["variance_weight"]),
                lambda_covariance=float(objective["covariance_weight"]),
                lambda_temporal=float(objective["temporal_consistency_weight"]),
                lambda_uncertainty=float(objective.get("uncertainty_weight", 0.0)),
                lambda_regime=0.0,
            )
            regime_scale = max(float(objective["regime_balance_weight"]), 1e-8)
            regime_loss, regime_metrics = self_supervised_regime_loss(
                output.regime_probabilities,
                batch["adjacency_pairs"],
                confidence_weight=0.25,
                confidence_hinge_weight=1.0,
                balance_weight=1.0,
                consistency_weight=float(objective["regime_consistency_weight"]) / regime_scale,
            )
            total = (
                float(objective["latent_prediction_weight"]) * latent_loss
                + regime_scale * regime_loss
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite {prepared.name} loss at epoch={epoch + 1} "
                    f"batch={batch_number}: latent={latent_metrics}, regime={regime_metrics}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_momentum = model.update_target_encoder_for_step(optimizer_step, ema_schedule)
            optimizer_step += 1
            epoch_rows.append(
                {
                    **{f"latent_{key}": value for key, value in latent_metrics.items()},
                    **{f"regime_{key}": value for key, value in regime_metrics.items()},
                    "total_loss": float(total.detach().cpu()),
                    "ema_momentum": float(ema_momentum),
                }
            )
        if not epoch_rows:
            raise RuntimeError(f"{prepared.name} training consumed no batches.")
        history.append(
            {
                "world": prepared.name,
                "epoch": epoch + 1,
                "samples": len(dataset),
                "batches": len(epoch_rows),
                **_mean_metrics(epoch_rows),
            }
        )
    return model.eval(), history


@torch.no_grad()
def _encode_world_state(
    prepared: PreparedWorld,
    model: WorldJEPAEncoder,
    config: WorldJEPAConfig,
    runtime: RuntimeOverrides,
) -> EncodedWorldState:
    context_length = config.worlds[prepared.name].context_length
    positions = np.arange(context_length - 1, len(prepared.frame), dtype=np.int64)
    if not len(positions):
        raise ValueError(f"{prepared.name} has fewer rows than its context length.")
    batch_size = int(runtime.batch_size or config.training["batch_size"])
    device = str(runtime.device or config.training["device"])
    values = prepared.frame.to_numpy(dtype=np.float32)
    token_parts: list[np.ndarray] = []
    future_token_parts: list[np.ndarray] = []
    regime_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []
    context_fraction_parts: list[np.ndarray] = []
    feature_fraction_parts: list[np.ndarray] = []
    model.eval()
    prediction_horizons = np.asarray(
        sorted(
            {
                *[float(value) for value in config.horizons["training_days"]],
                *[float(value) for value in config.horizons["live_days"]],
                *[float(value) for value in config.horizons["evaluation_days"]],
            }
        ),
        dtype=np.float32,
    )
    horizons = torch.from_numpy(prediction_horizons).to(device=device)
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        contexts = np.stack(
            [values[position - context_length + 1 : position + 1] for position in batch_positions]
        )
        observed = np.isfinite(contexts)
        x = torch.from_numpy(np.where(observed, contexts, 0.0)).to(device)
        feature_mask = torch.from_numpy(observed).to(device)
        output = model(x, horizons, feature_mask=feature_mask)
        recent_length = min(8, output.context_tokens.shape[1])
        recent = output.context_tokens[:, -recent_length:, :].mean(dim=1)
        scale_tokens = torch.stack(
            (
                output.context_summary,
                output.regime_token,
                output.context_tokens[:, -1, :],
                recent,
            ),
            dim=1,
        )
        token_parts.append(scale_tokens.cpu().numpy().astype(np.float32))
        future_token_parts.append(output.predicted_target.cpu().numpy().astype(np.float32))
        regime_parts.append(output.regime_probabilities.cpu().numpy().astype(np.float32))
        uncertainty_parts.append(output.uncertainty.mean(dim=1).cpu().numpy().astype(np.float32))
        context_fraction_parts.append(output.valid_context_fraction.cpu().numpy().astype(np.float32))
        feature_fraction_parts.append(
            output.diagnostics["valid_feature_fraction"].cpu().numpy().astype(np.float32)
        )
    return EncodedWorldState(
        index=prepared.frame.index[positions],
        tokens=np.concatenate(token_parts, axis=0),
        future_tokens=np.concatenate(future_token_parts, axis=0),
        future_horizons=prediction_horizons,
        regimes=np.concatenate(regime_parts, axis=0),
        uncertainty=np.concatenate(uncertainty_parts, axis=0),
        valid_context_fraction=np.concatenate(context_fraction_parts, axis=0),
        valid_feature_fraction=np.concatenate(feature_fraction_parts, axis=0),
        novelty=np.zeros(len(positions), dtype=np.float32),
    )


def _assemble_router_panel(
    bundle: ForecastDataBundle,
    states: Mapping[str, EncodedWorldState],
    base_config: Mapping[str, Any],
    split: PurgedSplit,
    *,
    use_future_tokens: bool = False,
    use_market_state: bool = False,
    residual_to_baseline: bool = False,
) -> RouterPanel:
    common_index: pd.Index | None = None
    for world in WORLD_NAMES:
        common_index = states[world].index if common_index is None else common_index.intersection(states[world].index)
    if common_index is None or common_index.empty:
        raise ValueError("World encodings have no common timestamps.")
    common_index = common_index.sort_values()
    max_regimes = max(states[world].regimes.shape[1] for world in WORLD_NAMES)
    token_worlds: list[np.ndarray] = []
    regime_worlds: list[np.ndarray] = []
    uncertainty_worlds: list[np.ndarray] = []
    novelty_worlds: list[np.ndarray] = []
    context_validity_worlds: list[np.ndarray] = []
    future_worlds: list[np.ndarray] = []
    future_horizons: np.ndarray | None = None
    for world in WORLD_NAMES:
        state = states[world]
        positions = state.index.get_indexer(common_index)
        token_worlds.append(state.tokens[positions])
        if future_horizons is None:
            future_horizons = np.asarray(state.future_horizons, dtype=np.float32)
        elif not np.array_equal(future_horizons, np.asarray(state.future_horizons, dtype=np.float32)):
            raise ValueError("Every world must expose the same future-horizon grid.")
        future_worlds.append(state.future_tokens[positions])
        padded_regimes = np.zeros((len(common_index), max_regimes), dtype=np.float32)
        padded_regimes[:, : state.regimes.shape[1]] = state.regimes[positions]
        regime_worlds.append(padded_regimes)
        uncertainty_worlds.append(state.uncertainty[positions])
        novelty_worlds.append(state.novelty[positions])
        context_validity_worlds.append(
            state.valid_context_fraction[positions] * state.valid_feature_fraction[positions]
        )

    freshness_columns: list[np.ndarray] = []
    validity_columns: list[np.ndarray] = []
    missingness_columns: list[np.ndarray] = []
    for world_index, world in enumerate(WORLD_NAMES):
        half_life = float(base_config["data"]["freshness_half_life_days"][world])
        freshness_days = bundle.freshness_days[world].reindex(common_index).to_numpy(dtype=float)
        freshness = np.nan_to_num(freshness_days / max(half_life, 1e-6), nan=1e6, posinf=1e6)
        missingness = bundle.missingness[world].reindex(common_index).to_numpy(dtype=float)
        missingness = np.nan_to_num(missingness, nan=1.0).clip(0.0, 1.0)
        source_freshness = bundle.freshness_scores[world].reindex(common_index).to_numpy(dtype=float)
        source_freshness = np.nan_to_num(source_freshness, nan=0.0).clip(0.0, 1.0)
        validity = (
            source_freshness
            * (1.0 - missingness)
            * np.asarray(context_validity_worlds[world_index], dtype=float)
        ).clip(0.0, 1.0)
        freshness_columns.append(freshness.astype(np.float32))
        missingness_columns.append(missingness.astype(np.float32))
        validity_columns.append(validity.astype(np.float32))

    uncertainty = np.stack(uncertainty_worlds, axis=1).astype(np.float32)
    # A train-time model can choose arbitrary latent scale. Normalize each
    # world's uncertainty by its causal historical median so gates compare
    # relative disagreement rather than parameterization units.
    for world_index in range(uncertainty.shape[1]):
        train_mask = common_index <= pd.Timestamp(split.train_timestamps[-1])
        candidate = uncertainty[train_mask, world_index]
        finite_positive = candidate[np.isfinite(candidate) & (candidate > 1e-8)]
        scale = float(np.median(finite_positive)) if len(finite_positive) else 1.0
        uncertainty[:, world_index] = np.nan_to_num(
            uncertainty[:, world_index] / max(scale, 1e-6),
            nan=1e6,
            posinf=1e6,
        )
    market_state = (
        _build_endogenous_market_state(bundle.close, common_index, split)
        if use_market_state
        else None
    )
    baseline_horizons = (
        np.asarray(future_horizons, dtype=np.float32)
        if residual_to_baseline and future_horizons is not None
        else None
    )
    baseline_quantiles = (
        _build_causal_quantile_baseline(bundle.close, common_index, baseline_horizons)
        if baseline_horizons is not None
        else None
    )
    return RouterPanel(
        index=common_index,
        world_tokens=np.stack(token_worlds, axis=1).astype(np.float32),
        world_regimes=np.stack(regime_worlds, axis=1).astype(np.float32),
        freshness=np.stack(freshness_columns, axis=1).astype(np.float32),
        validity=np.stack(validity_columns, axis=1).astype(np.float32),
        missingness=np.stack(missingness_columns, axis=1).astype(np.float32),
        uncertainty=uncertainty,
        novelty=np.stack(novelty_worlds, axis=1).astype(np.float32),
        future_tokens=(
            np.stack(future_worlds, axis=1).astype(np.float32)
            if use_future_tokens
            else None
        ),
        future_horizons=future_horizons if use_future_tokens else None,
        market_state=market_state,
        baseline_quantiles=baseline_quantiles,
        baseline_horizons=baseline_horizons,
    )


def _build_endogenous_market_state(
    close: pd.Series,
    index: pd.Index,
    split: PurgedSplit,
) -> np.ndarray:
    """Build a compact causal BTC query and fit its scaling on outer train only."""

    numeric = pd.to_numeric(close, errors="coerce").reindex(index).astype(float)
    log_close = np.log(numeric.where(numeric > 0.0))
    one_day = log_close.diff()
    features = pd.DataFrame(
        {
            "return_1": one_day,
            "return_3": log_close.diff(3),
            "return_7": log_close.diff(7),
            "return_15": log_close.diff(15),
            "realized_vol_7": one_day.rolling(7, min_periods=3).std(),
            "realized_vol_30": one_day.rolling(30, min_periods=10).std(),
            "trend_7": log_close - log_close.rolling(7, min_periods=3).mean(),
            "trend_30": log_close - log_close.rolling(30, min_periods=10).mean(),
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan)
    train_end = pd.Timestamp(split.train_timestamps[-1])
    train = features.loc[features.index <= train_end]
    center = train.median(axis=0, skipna=True).fillna(0.0)
    lower = train.quantile(0.25).fillna(center)
    upper = train.quantile(0.75).fillna(center)
    scale = (upper - lower).replace(0.0, np.nan).fillna(1.0)
    normalized = (features - center) / scale
    return normalized.clip(-10.0, 10.0).fillna(0.0).to_numpy(dtype=np.float32)


def _future_tokens_for_horizons(
    panel: RouterPanel,
    positions: np.ndarray,
    horizons: Sequence[float],
) -> np.ndarray | None:
    """Causally interpolate precomputed JEPA predictions onto router queries."""

    if panel.future_tokens is None or panel.future_horizons is None:
        return None
    source_horizons = np.asarray(panel.future_horizons, dtype=float)
    requested = np.asarray([float(value) for value in horizons], dtype=float)
    source = panel.future_tokens[np.asarray(positions, dtype=np.int64)]
    output = np.empty(
        (source.shape[0], source.shape[1], len(requested), source.shape[-1]),
        dtype=np.float32,
    )
    for target_index, horizon in enumerate(requested):
        right = int(np.searchsorted(source_horizons, horizon, side="left"))
        if right <= 0:
            output[:, :, target_index] = source[:, :, 0]
        elif right >= len(source_horizons):
            output[:, :, target_index] = source[:, :, -1]
        elif np.isclose(source_horizons[right], horizon):
            output[:, :, target_index] = source[:, :, right]
        else:
            left = right - 1
            width = source_horizons[right] - source_horizons[left]
            weight = float((horizon - source_horizons[left]) / max(width, 1e-12))
            output[:, :, target_index] = (
                (1.0 - weight) * source[:, :, left] + weight * source[:, :, right]
            )
    return output


def _build_causal_quantile_baseline(
    close: pd.Series,
    index: pd.Index,
    horizons: np.ndarray,
    *,
    span: int = 180,
    minimum_history: int = 180,
) -> np.ndarray:
    """Return an expanding, past-only empirical distribution for residual routing."""

    numeric = pd.to_numeric(close, errors="coerce").sort_index().astype(float)
    log_close = np.log(numeric.where(numeric > 0.0))
    output = np.full((len(index), len(horizons), len(QUANTILES)), np.nan, dtype=np.float32)
    decay = np.log(2.0) / max(float(span), 1.0)
    for horizon_index, horizon_value in enumerate(horizons):
        horizon = int(round(float(horizon_value)))
        if horizon <= 0 or not np.isclose(float(horizon_value), horizon):
            raise ValueError("Residual baseline horizons must be positive integer days.")
        realized = (log_close - log_close.shift(horizon)).dropna()
        for row_index, as_of in enumerate(pd.Index(index)):
            history = realized.loc[: pd.Timestamp(as_of)].to_numpy(dtype=float)
            if len(history) < int(minimum_history):
                continue
            positions = np.arange(len(history), dtype=float)
            weights = np.exp(-decay * (positions[-1] - positions))
            output[row_index, horizon_index] = weighted_quantile(
                history,
                QUANTILES,
                weights,
            ).astype(np.float32)
    if not np.isfinite(output).all():
        missing_rows = np.flatnonzero(~np.isfinite(output).all(axis=(1, 2)))
        raise ValueError(
            "Causal residual baseline has insufficient past history at router rows: "
            f"{missing_rows[:8].tolist()}"
        )
    return output


def _baseline_quantiles_for_horizons(
    panel: RouterPanel,
    positions: np.ndarray,
    horizons: Sequence[float],
) -> np.ndarray | None:
    if panel.baseline_quantiles is None or panel.baseline_horizons is None:
        return None
    # Quantiles are interpolated across horizon exactly like future latents;
    # the final sort guards monotonicity under floating-point interpolation.
    proxy = RouterPanel(
        index=panel.index,
        world_tokens=panel.world_tokens,
        world_regimes=panel.world_regimes,
        freshness=panel.freshness,
        validity=panel.validity,
        missingness=panel.missingness,
        uncertainty=panel.uncertainty,
        novelty=panel.novelty,
        future_tokens=panel.baseline_quantiles[:, None, :, :],
        future_horizons=panel.baseline_horizons,
        market_state=panel.market_state,
        baseline_quantiles=None,
        baseline_horizons=None,
    )
    interpolated = _future_tokens_for_horizons(proxy, positions, horizons)
    if interpolated is None:
        return None
    return np.sort(interpolated[:, 0], axis=-1).astype(np.float32)


def _router_forward(
    router: ContextRelevantWorldRouter,
    panel: RouterPanel,
    positions: np.ndarray,
    horizons: Sequence[float],
    device: str,
    *,
    world_mask: torch.Tensor | None = None,
) -> Any:
    index = np.asarray(positions, dtype=np.int64)
    role_world_masks = (
        production_quantile_role_masks(WORLD_NAMES)
        if router.semantic_role_routing
        else None
    )
    return router(
        torch.from_numpy(panel.world_tokens[index]).to(device),
        torch.tensor(horizons, dtype=torch.float32, device=device),
        world_freshness=torch.from_numpy(panel.freshness[index]).to(device),
        world_validity=torch.from_numpy(panel.validity[index]).to(device),
        world_missingness=torch.from_numpy(panel.missingness[index]).to(device),
        world_uncertainty=torch.from_numpy(panel.uncertainty[index]).to(device),
        world_mask=world_mask,
        world_regimes=torch.from_numpy(panel.world_regimes[index]).to(device),
        market_state=(
            torch.from_numpy(panel.market_state[index]).to(device)
            if panel.market_state is not None
            else None
        ),
        world_future_tokens=(
            torch.from_numpy(future_tokens).to(device)
            if (future_tokens := _future_tokens_for_horizons(panel, index, horizons)) is not None
            else None
        ),
        baseline_quantiles=(
            torch.from_numpy(baseline_quantiles).to(device)
            if (
                baseline_quantiles := _baseline_quantiles_for_horizons(
                    panel,
                    index,
                    horizons,
                )
            ) is not None
            else None
        ),
        role_world_masks=role_world_masks,
    )


@torch.no_grad()
def _router_pinball(
    router: ContextRelevantWorldRouter,
    panel: RouterPanel,
    positions: np.ndarray,
    targets: pd.DataFrame,
    horizons: Sequence[float],
    device: str,
    batch_size: int,
) -> float:
    router.eval()
    losses: list[float] = []
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        output = _router_forward(router, panel, batch_positions, horizons, device)
        timestamps = panel.index[batch_positions]
        target_values = targets.reindex(timestamps).to_numpy(dtype=np.float32, copy=True)
        target_tensor = torch.from_numpy(target_values).to(device)
        loss = quantile_pinball_loss(
            output.quantiles,
            target_tensor,
            (0.05, 0.25, 0.50, 0.75, 0.95),
        )
        losses.append(float(loss.cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _train_router(
    panel: RouterPanel,
    targets: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    config: WorldJEPAConfig,
    runtime: RuntimeOverrides,
) -> tuple[ContextRelevantWorldRouter, list[dict[str, Any]]]:
    device = str(runtime.device or config.training["device"])
    latent_dim = int(config.encoder["latent_dimension"])
    router = ContextRelevantWorldRouter(
        token_dim=latent_dim,
        model_dim=latent_dim,
        market_state_dim=(panel.market_state.shape[-1] if panel.market_state is not None else latent_dim),
        regime_dim=panel.world_regimes.shape[-1],
        horizon_embedding_dim=int(config.horizons["fourier_dimension"]),
        attention_heads=int(config.router["attention_heads"]),
        routing_mode="topk",
        top_k_worlds=int(config.router["top_k_worlds"]),
        world_dropout=float(config.router["world_dropout"]),
        freshness_temperature=float(config.router["freshness_temperature"]),
        hard_top1_inference=bool(config.router["hard_top1_inference"]),
        residual_to_baseline=bool(config.router.get("residual_to_baseline", False)),
        semantic_role_routing=bool(config.router.get("semantic_role_routing", False)),
        dropout=float(config.encoder["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    horizons = [float(value) for value in config.horizons["training_days"]]
    batch_size = int(runtime.batch_size or config.training["batch_size"])
    epochs = int(runtime.router_epochs or config.training["router_epochs"])
    patience = int(config.training["patience"])
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed)
    for epoch in range(epochs):
        router.train()
        shuffled = rng.permutation(train_positions)
        epoch_rows: list[dict[str, float]] = []
        for batch_number, start in enumerate(range(0, len(shuffled), batch_size)):
            if (
                runtime.maximum_router_batches_per_epoch is not None
                and batch_number >= int(runtime.maximum_router_batches_per_epoch)
            ):
                break
            batch_positions = shuffled[start : start + batch_size]
            timestamps = panel.index[batch_positions]
            target_values = targets.reindex(timestamps).to_numpy(dtype=np.float32, copy=True)
            finite_rows = np.isfinite(target_values).all(axis=1)
            if not finite_rows.any():
                continue
            batch_positions = batch_positions[finite_rows]
            target_tensor = torch.from_numpy(target_values[finite_rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = _router_forward(router, panel, batch_positions, horizons, device)
            loss, metrics = router_regularized_loss(
                output,
                target_tensor,
                horizons,
                quantile_weight=float(config.objective["quantile_weight"]),
                load_balance_weight=float(config.router["load_balance_weight"]),
                router_entropy_weight=float(config.router["router_entropy_weight"]),
                horizon_smoothness_weight=float(config.router["horizon_smoothness_weight"]),
                width_monotonicity_weight=float(config.router["width_monotonicity_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        if not epoch_rows:
            raise RuntimeError("Router training consumed no finite batches.")
        validation_loss = _router_pinball(
            router,
            panel,
            validation_positions,
            targets,
            horizons,
            device,
            batch_size,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "batches": len(epoch_rows),
                "validation_pinball": validation_loss,
                **_mean_metrics(epoch_rows),
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Router did not produce a valid validation checkpoint.")
    router.load_state_dict(best_state)
    return router.eval(), history


@torch.no_grad()
def _forecast_frame(
    router: ContextRelevantWorldRouter,
    panel: RouterPanel,
    positions: np.ndarray,
    horizons: Sequence[float],
    targets: pd.DataFrame | None,
    device: str,
    batch_size: int,
    *,
    world_mask: torch.Tensor | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    router.eval()
    horizon_values = [float(value) for value in horizons]
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        output = _router_forward(
            router,
            panel,
            batch_positions,
            horizon_values,
            device,
            world_mask=world_mask,
        )
        quantiles = output.quantiles.cpu().numpy()
        attention = output.attention_weights.cpu().numpy()
        role_attention_tensor = output.router_diagnostics.get("role_attention_weights")
        role_attention = (
            role_attention_tensor.detach().cpu().numpy()
            if isinstance(role_attention_tensor, torch.Tensor)
            else None
        )
        regimes = output.regime_mixture.cpu().numpy()
        gates = output.world_gates.cpu().numpy()
        for batch_row, position in enumerate(batch_positions):
            timestamp = panel.index[int(position)]
            for horizon_index, horizon in enumerate(horizon_values):
                target_column = f"target_{int(round(horizon))}d"
                actual = np.nan
                if (
                    targets is not None
                    and np.isclose(horizon, round(horizon))
                    and target_column in targets.columns
                    and timestamp in targets.index
                ):
                    actual = float(targets.at[timestamp, target_column])
                row: dict[str, Any] = {
                    "as_of": pd.Timestamp(timestamp),
                    "horizon": horizon,
                    "actual_return": actual,
                    **{
                        column: float(quantiles[batch_row, horizon_index, quantile_index])
                        for quantile_index, column in enumerate(QUANTILE_COLUMNS)
                    },
                }
                row.update(
                    {
                        f"attention_{world}": float(attention[batch_row, horizon_index, world_index])
                        for world_index, world in enumerate(WORLD_NAMES)
                    }
                )
                if role_attention is not None:
                    row.update(
                        {
                            f"role_attention_{role}_{world}": float(
                                role_attention[
                                    batch_row,
                                    horizon_index,
                                    role_index,
                                    world_index,
                                ]
                            )
                            for role_index, role in enumerate(QUANTILE_ROLE_NAMES)
                            for world_index, world in enumerate(WORLD_NAMES)
                        }
                    )
                row.update(
                    {
                        f"gate_{world}": float(gates[batch_row, world_index])
                        for world_index, world in enumerate(WORLD_NAMES)
                    }
                )
                for regime_index in range(regimes.shape[-1]):
                    row[f"regime_mix_{regime_index}"] = float(
                        regimes[batch_row, horizon_index, regime_index]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _selection_subsegments(
    split: PurgedSplit,
    index: pd.Index,
    config: WorldJEPAConfig,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    selection = split.selection_positions
    validation_rows = int(config.splits["router_validation_rows"])
    calibration_rows = int(config.splits["calibration_rows"])
    purge_rows = int(config.splits["purge_days"])
    validation = selection[:validation_rows]
    calibration = selection[-calibration_rows:]
    embargo = selection[validation_rows : len(selection) - calibration_rows]
    if len(embargo) < purge_rows:
        raise ValueError("Selection subsegments do not contain the configured router/calibration purge.")
    if int(validation[-1]) + split.max_horizon_rows >= int(calibration[0]):
        raise ValueError("Router validation targets overlap the calibration segment.")
    return index[validation], index[embargo], index[calibration]


def _world_regime_frame(
    panel: RouterPanel,
    positions: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position in positions:
        timestamp = panel.index[int(position)]
        for world_index, world in enumerate(WORLD_NAMES):
            row: dict[str, Any] = {
                "as_of": pd.Timestamp(timestamp),
                "world": world,
                "freshness": float(panel.freshness[position, world_index]),
                "validity": float(panel.validity[position, world_index]),
                "missingness": float(panel.missingness[position, world_index]),
                "latent_uncertainty": float(panel.uncertainty[position, world_index]),
                "prototype_novelty": float(panel.novelty[position, world_index]),
            }
            for regime_index, probability in enumerate(panel.world_regimes[position, world_index]):
                row[f"regime_{regime_index}"] = float(probability)
            rows.append(row)
    return pd.DataFrame(rows)


def _regime_usage_diagnostics(
    panel: RouterPanel,
    positions: np.ndarray,
    config: WorldJEPAConfig,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for world_index, world in enumerate(WORLD_NAMES):
        regime_count = config.worlds[world].regime_count
        probabilities = panel.world_regimes[positions, world_index, :regime_count].astype(float)
        probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
        marginal = probabilities.mean(axis=0)
        entropy = -float(np.sum(marginal * np.log(np.clip(marginal, 1e-12, None))))
        dominant = probabilities.argmax(axis=1)
        transition_count = int(np.count_nonzero(dominant[1:] != dominant[:-1]))
        diagnostics.append(
            {
                "world": world,
                "rows": int(len(probabilities)),
                "regime_count": int(regime_count),
                "effective_regime_count": float(np.exp(entropy)),
                "mean_assignment_confidence": float(probabilities.max(axis=1).mean()),
                "maximum_marginal_regime_share": float(marginal.max()),
                "dominant_regime_transitions": transition_count,
                "dominant_regime_transitions_per_100_rows": float(
                    100.0 * transition_count / max(len(probabilities) - 1, 1)
                ),
            }
        )
    return diagnostics


def _source_content_hash(config_path: Path, feature_manifest: Path) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    paths = [config_path, feature_manifest]
    paths.extend(sorted((root / "dual_model_forecaster" / "world_jepa").glob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(root) if path.is_relative_to(root) else path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_run_markdown(
    path: Path,
    *,
    summary: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> None:
    comparison = candidate_metrics[["horizon", "wis", "q05_miss_rate"]].merge(
        baseline_metrics[["horizon", "wis"]].rename(columns={"wis": "baseline_wis"}),
        on="horizon",
        how="left",
    )
    lines = [
        "# Five-World JEPA Shadow Evaluation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Promoted: `{summary['promoted']}`",
        f"- Source through: `{summary['source_last_timestamp']}` ({summary['source_age_days']} days old)",
        f"- Content hash: `{summary['content_hash']}`",
        "",
        "The model remains separate from the legacy forecast unless every promotion gate passes. A completed training run is not evidence of forecasting edge.",
        "",
        "| Horizon | JEPA WIS | Strongest causal baseline WIS | Difference | q05 miss rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {float(row['horizon']):g} | {float(row['wis']):.6f} | "
            f"{float(row['baseline_wis']):.6f} | "
            f"{float(row['wis'] - row['baseline_wis']):+.6f} | "
            f"{float(row['q05_miss_rate']):.3f} |"
        )
    lines.extend(["", "## Promotion gates", ""])
    for key, value in summary["promotion_gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- Per-world scalers and encoders, plus the router, use the outer training segment only.",
            "- Router early stopping and quantile calibration use disjoint selection subsegments separated by an embargo.",
            "- Test outcomes are used only for the final comparison and ablations.",
            "- Fractional horizons are continuous model queries with interpolated calibration; they are not additional observed labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_scored = add_forecast_losses(candidate)
    baseline_scored = add_forecast_losses(baseline)
    paired = candidate_scored[["as_of", "horizon", "wis"]].merge(
        baseline_scored[["as_of", "horizon", "wis"]],
        on=["as_of", "horizon"],
        suffixes=("_candidate", "_baseline"),
    )
    # Forecast horizons overlap. Aggregate the paired loss to one observation
    # per as-of date before drawing contiguous date blocks; treating each
    # horizon row as an independent time step would overstate precision.
    paired = (
        paired.groupby("as_of", as_index=False)[["wis_candidate", "wis_baseline"]]
        .mean()
        .sort_values("as_of")
    )
    return moving_block_bootstrap_difference(
        paired["wis_candidate"].to_numpy(dtype=float),
        paired["wis_baseline"].to_numpy(dtype=float),
        block_length=block_length,
        samples=1000,
        seed=seed,
    )


def _save_checkpoints(
    artifact_root: Path,
    config: WorldJEPAConfig,
    prepared: Mapping[str, PreparedWorld],
    models: Mapping[str, WorldJEPAEncoder],
    regime_codebooks: Mapping[str, CausalRegimeCodebook],
    router: ContextRelevantWorldRouter,
    split: PurgedSplit,
    calibrator: QuantileCalibrator,
    content_hash: str,
) -> dict[str, str]:
    checkpoint_root = artifact_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for world in WORLD_NAMES:
        path = checkpoint_root / f"{world}_world_jepa.pt"
        torch.save(
            {
                "schema_version": 1,
                "world": world,
                "state_dict": models[world].state_dict(),
                "raw_columns": prepared[world].raw_columns,
                "model_columns": prepared[world].model_columns,
                "channel_columns": prepared[world].channel_columns,
                "scaler": prepared[world].scaler.to_dict(),
                "world_spec": config.worlds[world].__dict__,
                "regime_codebook": regime_codebooks[world].to_dict(),
                "regime_smoothing_alpha": 0.25,
                "encoder_config": config.encoder,
                "split": split.as_dict(),
                "content_hash": content_hash,
            },
            path,
        )
        paths[world] = str(path)
    router_path = checkpoint_root / "context_relevant_world_router.pt"
    semantic_role_map = semantic_role_routing_metadata(router)
    torch.save(
        {
            "schema_version": 2,
            "state_dict": router.state_dict(),
            "router_config": config.router,
            "semantic_role_map": semantic_role_map,
            "horizons": config.horizons,
            "calibrator": calibrator.adjustments,
            "split": split.as_dict(),
            "content_hash": content_hash,
        },
        router_path,
    )
    paths["router"] = str(router_path)
    return paths


def run_world_jepa_pipeline(
    config_path: str | Path = "configs/world_jepa.json",
    *,
    runtime: RuntimeOverrides | None = None,
    end_timestamp: Any | None = None,
    refresh_feature_audit: bool = True,
) -> dict[str, Any]:
    """Train, calibrate, and causally evaluate the five-world JEPA in shadow mode."""

    started = time.time()
    runtime = runtime or RuntimeOverrides()
    config = load_world_jepa_config(config_path)
    if not config.enabled or config.mode == "off":
        return {"status": "disabled", "mode": config.mode, "config": str(config.path)}
    set_global_seed(config.seed)
    base_config = load_config(config.base_config)
    bundle = load_forecast_data(base_config, end_timestamp=end_timestamp)
    training_horizons = [float(value) for value in config.horizons["training_days"]]
    split = build_purged_train_selection_test(
        bundle.close.index,
        minimum_train_rows=int(config.splits["minimum_train_rows"]),
        selection_rows=int(config.splits["selection_rows"]),
        test_rows=int(config.splits["test_rows"]),
        purge_rows=int(config.splits["purge_days"]),
        max_horizon=max(training_horizons),
    )
    feature_audit = _refresh_feature_audit(bundle, config) if refresh_feature_audit else None
    selected = _load_selected_columns(config.feature_manifest)
    prepared = _prepare_worlds(
        bundle,
        selected,
        split,
        causalize_selected_features=bool(
            config.objective.get("causalize_selected_features", False)
        ),
    )

    world_models: dict[str, WorldJEPAEncoder] = {}
    world_histories: list[dict[str, Any]] = []
    world_states: dict[str, EncodedWorldState] = {}
    regime_codebooks: dict[str, CausalRegimeCodebook] = {}
    regime_codebook_diagnostics: list[dict[str, Any]] = []
    for world in WORLD_NAMES:
        model, history = _train_world_encoder(prepared[world], config, split, runtime)
        world_models[world] = model
        world_histories.extend(history)
        state = _encode_world_state(prepared[world], model, config, runtime)
        train_state_mask = state.index <= pd.Timestamp(split.train_timestamps[-1])
        codebook = CausalRegimeCodebook.fit(
            state.tokens[train_state_mask, 0, :],
            config.worlds[world].regime_count,
            seed=config.seed,
        )
        state.regimes = causal_smooth_regime_probabilities(
            codebook.predict_proba(state.tokens[:, 0, :]),
            alpha=0.25,
        )
        state.novelty = codebook.novelty_score(state.tokens[:, 0, :])
        world_states[world] = state
        regime_codebooks[world] = codebook
        regime_codebook_diagnostics.append(
            {
                "world": world,
                "fit_segment": "outer_train_only",
                **codebook.diagnostics(state.tokens[train_state_mask, 0, :]),
            }
        )

    panel = _assemble_router_panel(
        bundle,
        world_states,
        base_config,
        split,
        use_future_tokens=bool(config.router.get("use_future_tokens", False)),
        use_market_state=bool(config.router.get("use_market_state", False)),
        residual_to_baseline=bool(config.router.get("residual_to_baseline", False)),
    )
    targets = build_dense_return_targets(
        bundle.close,
        training_horizons,
        return_type=str(base_config["data"]["return_type"]),
    )
    validation_timestamps, selection_embargo_timestamps, calibration_timestamps = _selection_subsegments(
        split,
        bundle.close.index,
        config,
    )
    train_positions = panel.positions_for(split.train_timestamps)
    validation_positions = panel.positions_for(validation_timestamps)
    calibration_positions = panel.positions_for(calibration_timestamps)
    test_positions = panel.positions_for(split.test_timestamps)
    if min(len(train_positions), len(validation_positions), len(calibration_positions), len(test_positions)) == 0:
        raise ValueError("Encoded context windows leave an empty router segment.")
    router, router_history = _train_router(
        panel,
        targets,
        train_positions,
        validation_positions,
        config,
        runtime,
    )

    device = str(runtime.device or config.training["device"])
    batch_size = int(runtime.batch_size or config.training["batch_size"])
    calibration_raw = _forecast_frame(
        router,
        panel,
        calibration_positions,
        training_horizons,
        targets,
        device,
        batch_size,
    )
    calibrator = fit_quantile_calibrator(calibration_raw)
    evaluation_horizons = [float(value) for value in config.horizons["evaluation_days"]]
    test_raw = _forecast_frame(
        router,
        panel,
        test_positions,
        evaluation_horizons,
        targets,
        device,
        batch_size,
    )
    test_calibrated = calibrator.apply(test_raw).dropna(subset=["actual_return"]).reset_index(drop=True)
    candidate_metrics = summarize_forecasts(test_calibrated, forecaster="five_world_jepa_attention")
    causal_ewma = causal_ewma_baseline(
        bundle.close,
        test_calibrated["as_of"].drop_duplicates(),
        evaluation_horizons,
    )
    causal_ewma = causal_ewma.dropna(subset=["actual_return", *QUANTILE_COLUMNS]).reset_index(drop=True)
    causal_ewma_metrics = summarize_forecasts(causal_ewma, forecaster="causal_ewma")

    # Promotion is judged against an oracle envelope of simple causal
    # baselines, not just one hand-picked comparator.
    from dual_model_forecaster.brutal_baselines import (
        EvaluationConfig,
        build_forecaster_panel,
    )

    baseline_input = test_calibrated.copy()
    baseline_input["target_timestamp"] = baseline_input["as_of"] + pd.to_timedelta(
        baseline_input["horizon"], unit="D"
    )
    baseline_input["base_close"] = baseline_input["as_of"].map(bundle.close)
    baseline_input["actual_close"] = baseline_input["target_timestamp"].map(bundle.close)
    brutal_panel, _ = build_forecaster_panel(
        baseline_input,
        bundle,
        [int(value) for value in evaluation_horizons],
        str(base_config["data"]["return_type"]),
        EvaluationConfig(
            history_min=180,
            regime_history_min=60,
            ewma_span=30,
            bootstrap_samples=0,
            include_garch=False,
            include_kalman_synthesis=False,
        ),
    )
    baseline_names = {
        "random_walk",
        "historical_vol_cone",
        "ewma_vol_cone",
        "regime_conditioned_historical",
    }
    causal_baseline_panel = brutal_panel.loc[
        brutal_panel["forecaster"].isin(baseline_names),
        ["forecaster", "as_of", "horizon", "actual_return", *QUANTILE_COLUMNS],
    ].dropna(subset=["actual_return", *QUANTILE_COLUMNS])
    causal_baseline_panel = pd.concat(
        [causal_baseline_panel, causal_ewma.assign(forecaster="causal_ewma")],
        ignore_index=True,
    )
    baseline_metric_parts = [
        summarize_forecasts(group, forecaster=str(name))
        for name, group in causal_baseline_panel.groupby("forecaster", sort=True)
    ]
    all_baseline_metrics = pd.concat(baseline_metric_parts, ignore_index=True)
    strongest_rows = (
        all_baseline_metrics.sort_values(["horizon", "wis", "forecaster"])
        .groupby("horizon", as_index=False)
        .first()
    )
    strongest_rows["forecaster"] = "strongest_causal_baseline"
    strongest_forecaster = {
        float(row["horizon"]): str(
            all_baseline_metrics.loc[
                (all_baseline_metrics["horizon"] == row["horizon"])
                & (all_baseline_metrics["wis"] == row["wis"]),
                "forecaster",
            ].iloc[0]
        )
        for _, row in strongest_rows.iterrows()
    }
    strongest_baseline_parts = [
        causal_baseline_panel.loc[
            (causal_baseline_panel["horizon"] == horizon)
            & (causal_baseline_panel["forecaster"] == forecaster)
        ]
        for horizon, forecaster in strongest_forecaster.items()
    ]
    baseline = pd.concat(strongest_baseline_parts, ignore_index=True)
    baseline_metrics = strongest_rows
    bootstrap = _paired_bootstrap(
        test_calibrated,
        baseline,
        block_length=int(np.ceil(max(evaluation_horizons))),
        seed=config.seed,
    )
    attention_columns = [f"attention_{world}" for world in WORLD_NAMES]
    role_attention_columns = [
        f"role_attention_{role}_{world}"
        for role in QUANTILE_ROLE_NAMES
        for world in WORLD_NAMES
        if f"role_attention_{role}_{world}" in test_calibrated.columns
    ]
    promotion = evaluate_promotion_gates(
        candidate_metrics,
        baseline_metrics,
        test_calibrated[["as_of", "horizon", *attention_columns]],
        config.promotion_gates,
        bootstrap=bootstrap,
        temporal_safety_passed=split.is_target_safe,
        tests_passed=bool(runtime.validated_test_suite),
    )
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    source_last = pd.Timestamp(bundle.close.index.max()).normalize()
    source_age_days = max(int((today - source_last).days), 0)
    maximum_source_age = int(config.promotion_gates.get("maximum_source_age_days", 2))
    source_fresh = bool(end_timestamp is not None or source_age_days <= maximum_source_age)
    promotion["source_data_fresh"] = source_fresh
    promotion["source_age_days"] = source_age_days
    promotion["maximum_source_age_days"] = maximum_source_age
    regime_usage = _regime_usage_diagnostics(panel, test_positions, config)
    minimum_effective_regimes = float(
        config.promotion_gates.get("minimum_effective_regime_count", 1.0)
    )
    minimum_regime_confidence = float(
        config.promotion_gates.get("minimum_mean_regime_confidence", 0.0)
    )
    regimes_operational = bool(
        all(
            float(row["effective_regime_count"]) >= minimum_effective_regimes
            and float(row["mean_assignment_confidence"]) >= minimum_regime_confidence
            for row in regime_usage
        )
    )
    promotion["regime_states_operational"] = regimes_operational
    promotion["minimum_effective_regime_count"] = minimum_effective_regimes
    promotion["minimum_mean_regime_confidence"] = minimum_regime_confidence
    promotion["passed"] = bool(promotion["passed"] and source_fresh and regimes_operational)

    ablation_rows: list[pd.DataFrame] = []
    for world_index, world in enumerate(WORLD_NAMES):
        mask = torch.ones(len(WORLD_NAMES), dtype=torch.bool, device=device)
        mask[world_index] = False
        removed = _forecast_frame(
            router,
            panel,
            test_positions,
            evaluation_horizons,
            targets,
            device,
            batch_size,
            world_mask=mask,
        )
        removed = calibrator.apply(removed).dropna(subset=["actual_return"])
        metrics = summarize_forecasts(removed, forecaster=f"without_{world}")
        metrics["ablation"] = f"without_{world}"
        ablation_rows.append(metrics)
    ablations = pd.concat(ablation_rows, ignore_index=True) if ablation_rows else pd.DataFrame()

    live_positions = np.asarray([len(panel.index) - 1], dtype=np.int64)
    live = _forecast_frame(
        router,
        panel,
        live_positions,
        config.horizons["live_days"],
        None,
        device,
        batch_size,
    )
    live = calibrator.apply(live)

    artifact_root = config.artifact_root
    report_root = config.report_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    content_hash = _source_content_hash(Path(config_path), config.feature_manifest)
    checkpoints = _save_checkpoints(
        artifact_root,
        config,
        prepared,
        world_models,
        regime_codebooks,
        router,
        split,
        calibrator,
        content_hash,
    )
    pd.DataFrame(world_histories).to_csv(report_root / "world_training_metrics.csv", index=False)
    pd.DataFrame(router_history).to_csv(report_root / "router_training_metrics.csv", index=False)
    pd.DataFrame(regime_codebook_diagnostics).to_csv(
        report_root / "regime_codebook_diagnostics.csv",
        index=False,
    )
    calibration_raw.to_csv(report_root / "calibration_raw_forecasts.csv", index=False)
    test_raw.to_csv(report_root / "test_raw_forecasts.csv", index=False)
    test_calibrated.to_csv(report_root / "test_calibrated_forecasts.csv", index=False)
    candidate_metrics.to_csv(report_root / "candidate_metrics.csv", index=False)
    causal_baseline_panel.to_csv(report_root / "causal_baseline_forecasts.csv", index=False)
    all_baseline_metrics.to_csv(report_root / "causal_baseline_metrics.csv", index=False)
    baseline.to_csv(report_root / "strongest_causal_baseline_forecasts.csv", index=False)
    baseline_metrics.to_csv(report_root / "strongest_causal_baseline_metrics.csv", index=False)
    ablations.to_csv(report_root / "world_removal_ablations.csv", index=False)
    live.to_csv(report_root / "live_shadow_forecast.csv", index=False)
    test_calibrated[
        ["as_of", "horizon", *attention_columns, *role_attention_columns]
    ].to_csv(
        report_root / "world_attention.csv",
        index=False,
    )
    _world_regime_frame(panel, test_positions).to_csv(
        report_root / "world_regime_states.csv",
        index=False,
    )
    pd.DataFrame(regime_usage).to_csv(report_root / "regime_usage_diagnostics.csv", index=False)
    leakage_audit = {
        "outer_split": split.as_dict(),
        "router_validation": {
            "rows": len(validation_timestamps),
            "start": str(validation_timestamps.min()),
            "end": str(validation_timestamps.max()),
        },
        "router_calibration_embargo": {
            "rows": len(selection_embargo_timestamps),
            "start": str(selection_embargo_timestamps.min()),
            "end": str(selection_embargo_timestamps.max()),
        },
        "calibration": {
            "rows": len(calibration_timestamps),
            "start": str(calibration_timestamps.min()),
            "end": str(calibration_timestamps.max()),
        },
        "feature_scalers_fit_on": "outer_train_only",
        "world_encoders_fit_on": "outer_train_only",
        "router_fit_on": "outer_train_only",
        "router_early_stopping": "first_selection_subsegment",
        "calibrator_fit_on": "last_selection_subsegment_after_embargo",
        "test_used_for_fitting": False,
    }
    _write_json(report_root / "leakage_audit.json", leakage_audit)
    _write_json(report_root / "block_bootstrap.json", bootstrap)
    _write_json(report_root / "promotion_gates.json", promotion)
    summary = {
        "status": "ok",
        "mode": config.mode,
        "promoted": bool(config.mode == "primary" and promotion["passed"]),
        "promotion_gates": promotion,
        "content_hash": content_hash,
        "source_last_timestamp": str(bundle.close.index.max()),
        "source_age_days": source_age_days,
        # Report the columns actually passed to each encoder. In causal mode
        # this differs from the source manifest because globally fitted PCA
        # shortcuts are removed before scaling and training.
        "feature_columns": {
            world: list(prepared[world].raw_columns) for world in WORLD_NAMES
        },
        "feature_manifest_columns": selected,
        "regime_codebook_diagnostics": regime_codebook_diagnostics,
        "regime_usage_diagnostics": regime_usage,
        "semantic_role_routing": bool(router.semantic_role_routing),
        "semantic_role_map": semantic_role_routing_metadata(router),
        "feature_audit_refreshed": bool(feature_audit is not None),
        "split": split.as_dict(),
        "router_selection_subsegments": leakage_audit,
        "candidate_metrics": candidate_metrics.to_dict(orient="records"),
        "baseline_metrics": baseline_metrics.to_dict(orient="records"),
        "bootstrap": bootstrap,
        "checkpoints": checkpoints,
        "live_shadow_forecast": str(report_root / "live_shadow_forecast.csv"),
        "report_root": str(report_root),
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json(report_root / "run_summary.json", summary)
    _write_run_markdown(
        report_root / "summary.md",
        summary=summary,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    return summary


__all__ = [
    "RuntimeOverrides",
    "build_dense_return_targets",
    "run_world_jepa_pipeline",
]
