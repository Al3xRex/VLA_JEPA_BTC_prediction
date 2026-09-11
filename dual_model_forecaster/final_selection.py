from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dual_model_forecaster.config import load_config
from dual_model_forecaster.data import load_forecast_data
from dual_model_forecaster.fusion import (
    FusionDataset,
    TargetScaler,
    _build_sequence_arrays,
    _calibration_context_score,
    _output_channels,
    _params_to_quantiles_np,
    _partition_training_frame,
    apply_calibrator,
    build_fusion_feature_frame,
    build_fusion_model,
    fit_calibrator,
    fusion_loss,
)
from dual_model_forecaster.metrics import pinball_loss, summarize_quantile_metrics
from dual_model_forecaster.semantics import build_semantic_targets
from dual_model_forecaster.specialists import (
    COMMUNICATION_COLUMNS,
    FeatureScaler,
    build_walkforward_states,
    emit_specialist_states,
    select_best_specialist,
)
from dual_model_forecaster.splits import build_inner_segments, build_outer_folds
from dual_model_forecaster.utils import ensure_dir, quantile_name, set_global_seed, write_json


ROLE_MAPS: dict[str, dict[str, list[str]]] = {
    "prior_balanced": {
        "center": ["environment", "edges", "movement"],
        "width": ["structure", "environment", "edges", "movement"],
        "tail": ["structure", "environment", "movement"],
    },
    "slow_tail_only": {
        "center": ["edges", "movement"],
        "width": ["environment", "edges", "movement"],
        "tail": ["structure", "environment"],
    },
    "short_quartile_liquidation": {
        "center": ["edges", "movement"],
        "width": ["environment", "edges", "movement", "liquidation"],
        "tail": ["structure", "environment", "liquidation"],
    },
}

ROLE_INTERFACES = [
    "semantic_delta_interactions",
    "semantic_delta_interactions_narrow_comm",
]

CENTER_INTERACTIONS = {
    "movement_trend_vs_edges_stretch",
    "movement_persistence_vs_edges_asymmetry",
    "slow_fast_alignment",
    "slow_direction",
    "fast_direction",
    "fast_trending_state",
    "fast_stationary_state",
    "fast_trend_exhaustion_pressure",
    "stationary_trending_balance",
    "edges_chop_relevance",
    "movement_chop_relevance",
    "edges_dynamic_relevance",
    "movement_dynamic_relevance",
    "edge_movement_relevance_spread",
    "fast_regime_weighted_direction",
    "movement_to_edges_rotation_pressure",
    "edges_to_movement_rotation_pressure",
}

WIDTH_INTERACTIONS = {
    "slow_fast_conflict",
    "slow_fast_direction_gap",
    "fast_specialist_disagreement",
    "slow_specialist_agreement",
    "confidence_weighted_disagreement",
    "confidence_weighted_agreement",
    "liquidation_quartile_width_pressure",
    "liquidation_edges_jump_alignment",
    "liquidation_movement_jump_alignment",
    "liquidation_ta_directional_alignment",
    "liquidation_ta_confidence_weighted_alignment",
    "liquidation_ta_conflict",
    "liquidation_short_quartile_bias",
    "fast_regime_relevance_disagreement",
    "fast_regime_rotation_intensity",
    "stationary_trending_transition_risk",
    "stationary_trending_balance",
    "edges_recurrent_exposure",
    "movement_recurrent_exposure",
}

TAIL_INTERACTIONS = {
    "structure_fragility_vs_movement_continuation",
    "macro_risk_on_vs_fragility",
    "edges_downside_vs_movement_trend",
    "slow_fast_conflict",
    "confidence_weighted_disagreement",
    "liquidation_quartile_width_pressure",
    "liquidation_short_quartile_bias",
    "liquidation_ta_conflict",
    "fast_trend_exhaustion_pressure",
    "stationary_trending_transition_risk",
    "fast_regime_relevance_disagreement",
    "stationary_trending_balance",
}

COMMUNICATION_SUFFIXES = tuple(COMMUNICATION_COLUMNS)


class RoleRoutedStructuredFusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_window: int,
        horizon_count: int,
        head_indices: dict[str, list[int]],
        hidden_dim: int,
        dropout: float,
        horizon_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.state_window = state_window
        self.horizon_count = horizon_count
        self.hidden_dim = hidden_dim
        self.register_buffer("center_index", torch.tensor(head_indices["center"], dtype=torch.long), persistent=False)
        self.register_buffer("width_index", torch.tensor(head_indices["width"], dtype=torch.long), persistent=False)
        self.register_buffer("tail_index", torch.tensor(head_indices["tail"], dtype=torch.long), persistent=False)

        self.center_encoder = self._make_encoder(len(head_indices["center"]), state_window, hidden_dim, dropout)
        self.width_encoder = self._make_encoder(len(head_indices["width"]), state_window, hidden_dim, dropout)
        self.tail_encoder = self._make_encoder(len(head_indices["tail"]), state_window, hidden_dim, dropout)

        self.horizon_embeddings = nn.Parameter(torch.randn(horizon_count, horizon_embedding_dim) * 0.02)
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.tail_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _make_encoder(feature_count: int, state_window: int, hidden_dim: int, dropout: float) -> nn.Module | None:
        if feature_count <= 0:
            return None
        flat_dim = feature_count * state_window
        return nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def _encode(self, x: torch.Tensor, indices: torch.Tensor, encoder: nn.Module | None) -> torch.Tensor:
        if encoder is None or indices.numel() == 0:
            return x.new_zeros((x.shape[0], self.hidden_dim))
        selected = torch.index_select(x, dim=2, index=indices).flatten(start_dim=1)
        return encoder(selected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_ctx = self._encode(x, self.center_index, self.center_encoder)
        width_ctx = self._encode(x, self.width_index, self.width_encoder)
        tail_ctx = self._encode(x, self.tail_index, self.tail_encoder)

        batch = x.shape[0]
        horizon_embeddings = self.horizon_embeddings.unsqueeze(0).expand(batch, -1, -1)
        center_expanded = center_ctx.unsqueeze(1).expand(-1, self.horizon_count, -1)
        width_expanded = width_ctx.unsqueeze(1).expand(-1, self.horizon_count, -1)
        tail_expanded = tail_ctx.unsqueeze(1).expand(-1, self.horizon_count, -1)

        center_raw = self.center_head(torch.cat([center_expanded, horizon_embeddings], dim=-1))
        width_raw = self.width_head(torch.cat([width_expanded, center_expanded, horizon_embeddings], dim=-1))
        tail_raw = self.tail_head(
            torch.cat([tail_expanded, width_expanded, center_expanded, horizon_embeddings], dim=-1)
        )
        raw = torch.cat([center_raw, width_raw, tail_raw], dim=-1)
        return raw.reshape(batch, self.horizon_count * 4)


def _architecture_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    for spec in config["fusion"]["architectures"]:
        if spec["name"] == name:
            return copy.deepcopy(spec)
    raise KeyError(f"Missing architecture spec: {name}")


def enumerate_candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for horizon_set_name in ("set_a", "set_b"):
        horizons = list(config["fusion"]["horizon_sets"][horizon_set_name])
        candidates.append(
            {
                "name": f"{horizon_set_name}__legacy_D__direct_quantiles__quantile_mlp",
                "family": "baseline",
                "horizon_set_name": horizon_set_name,
                "horizons": horizons,
                "variant": "D",
                "output_style": "direct_quantiles",
                "architecture": _architecture_spec(config, "quantile_mlp"),
            }
        )
        candidates.append(
            {
                "name": f"{horizon_set_name}__semantic_A__direct_quantiles__horizon_gated_mlp",
                "family": "baseline",
                "horizon_set_name": horizon_set_name,
                "horizons": horizons,
                "variant": "A",
                "output_style": "direct_quantiles",
                "architecture": _architecture_spec(config, "horizon_gated_mlp"),
            }
        )
        for role_map_name in ROLE_MAPS:
            for interface_name in ROLE_INTERFACES:
                candidates.append(
                    {
                        "name": f"{horizon_set_name}__role_{role_map_name}__{interface_name}",
                        "family": "role_routed",
                        "horizon_set_name": horizon_set_name,
                        "horizons": horizons,
                        "role_map_name": role_map_name,
                        "interface_name": interface_name,
                        "output_style": "structured_cwt",
                        "architecture": {
                            "name": "role_routed_structured",
                            "hidden_dim": 128,
                            "dropout": 0.10,
                            "horizon_embedding_dim": 24,
                        },
                    }
                )
    return candidates


def _bucket_patterns(bucket_name: str) -> tuple[str, ...]:
    return (
        f"{bucket_name}_",
        f"delta_{bucket_name}_",
        f"roll7_mean_{bucket_name}_",
        f"roll7_std_{bucket_name}_",
    )


def _bucket_feature_columns(columns: list[str], bucket_name: str) -> list[str]:
    patterns = _bucket_patterns(bucket_name)
    return [column for column in columns if any(column.startswith(pattern) for pattern in patterns)]


def _has_communication_suffix(column: str) -> bool:
    return any(suffix in column for suffix in COMMUNICATION_SUFFIXES)


def _role_feature_frame(states_by_bucket: dict[str, pd.DataFrame], interface_name: str) -> pd.DataFrame:
    if interface_name == "semantic_delta_interactions":
        return build_fusion_feature_frame(states_by_bucket, variant="D", communication_channels=[])

    base = build_fusion_feature_frame(states_by_bucket, variant="E", communication_channels=[])
    comm_frames = []
    for bucket_name, frame in states_by_bucket.items():
        selected = [
            column
            for column in frame.columns
            if column in {"internal_disagreement", "scale_entropy", "state_velocity", "state_acceleration"}
        ]
        if selected:
            comm_frames.append(frame[selected].add_prefix(f"{bucket_name}_"))
    if comm_frames:
        base = pd.concat([base, *comm_frames], axis=1)
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_role_routed_feature_bundle(
    states_by_bucket: dict[str, pd.DataFrame],
    role_map_name: str,
    interface_name: str,
) -> dict[str, Any]:
    feature_frame = _role_feature_frame(states_by_bucket, interface_name=interface_name)
    columns = feature_frame.columns.tolist()
    role_map = ROLE_MAPS[role_map_name]

    head_columns: dict[str, list[str]] = {}
    head_bucket_columns: dict[str, dict[str, list[str]]] = {}
    for head_name, buckets in role_map.items():
        selected: list[str] = []
        bucket_map: dict[str, list[str]] = {}
        for bucket_name in buckets:
            bucket_columns = _bucket_feature_columns(columns, bucket_name)
            if head_name == "center":
                bucket_columns = [column for column in bucket_columns if not _has_communication_suffix(column)]
            elif interface_name != "semantic_delta_interactions_narrow_comm":
                bucket_columns = [column for column in bucket_columns if not _has_communication_suffix(column)]
            bucket_map[bucket_name] = bucket_columns
            selected.extend(bucket_columns)

        if head_name == "center":
            selected.extend([column for column in columns if column in CENTER_INTERACTIONS])
        elif head_name == "width":
            selected.extend([column for column in columns if column in WIDTH_INTERACTIONS])
        else:
            selected.extend([column for column in columns if column in TAIL_INTERACTIONS])

        deduped = list(dict.fromkeys(selected))
        head_columns[head_name] = deduped
        head_bucket_columns[head_name] = bucket_map

    head_indices = {
        head_name: [columns.index(column) for column in head_columns[head_name]]
        for head_name in ("center", "width", "tail")
    }
    head_bucket_indices = {
        head_name: {
            bucket_name: [columns.index(column) for column in bucket_columns if column in columns]
            for bucket_name, bucket_columns in bucket_map.items()
        }
        for head_name, bucket_map in head_bucket_columns.items()
    }
    return {
        "feature_frame": feature_frame,
        "head_indices": head_indices,
        "head_bucket_indices": head_bucket_indices,
    }


def _train_model(
    model: nn.Module,
    train_dataset: FusionDataset,
    val_dataset: FusionDataset,
    training_cfg: dict[str, Any],
    quantiles: list[float],
    horizons: list[int],
    output_style: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    history = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    patience = int(training_cfg["patience"])

    for _epoch in range(int(training_cfg["max_epochs"])):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = fusion_loss(
                prediction=prediction,
                target=y_batch,
                quantiles=quantiles,
                horizons=horizons,
                output_style=output_style,
                training_cfg=training_cfg,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                prediction = model(x_batch)
                loss = fusion_loss(
                    prediction=prediction,
                    target=y_batch,
                    quantiles=quantiles,
                    horizons=horizons,
                    output_style=output_style,
                    training_cfg=training_cfg,
                )
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_state, history


def _prediction_frames(
    model: nn.Module,
    x: np.ndarray,
    timestamps: list[pd.Timestamp],
    horizons: list[int],
    quantiles: list[float],
    output_style: str,
    target_scaler: TargetScaler,
    device: str,
) -> dict[int, pd.DataFrame]:
    if len(timestamps) == 0:
        return {horizon: pd.DataFrame(columns=[quantile_name(q) for q in quantiles]) for horizon in horizons}
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    pred = _params_to_quantiles_np(
        raw=raw,
        horizon_count=len(horizons),
        quantiles=quantiles,
        output_style=output_style,
    )
    pred = target_scaler.inverse_transform_array(pred, [f"target_{h}d" for h in horizons])
    pred = np.sort(pred, axis=2)
    out: dict[int, pd.DataFrame] = {}
    for horizon_idx, horizon in enumerate(horizons):
        frame = pd.DataFrame(
            pred[:, horizon_idx, :],
            index=pd.Index(timestamps, name="timestamp"),
            columns=[quantile_name(q) for q in quantiles],
        )
        out[horizon] = frame
    return out


def _build_model_for_candidate(candidate: dict[str, Any], feature_dim: int, state_window: int) -> nn.Module:
    horizons = candidate["horizons"]
    if candidate["family"] == "baseline":
        return build_fusion_model(
            spec=candidate["architecture"],
            input_dim=feature_dim,
            state_window=state_window,
            horizon_count=len(horizons),
            output_channels=_output_channels(candidate["output_style"], [0.05, 0.25, 0.50, 0.75, 0.95]),
        )
    head_indices = candidate["feature_metadata"]["head_indices"]
    architecture = candidate["architecture"]
    return RoleRoutedStructuredFusion(
        input_dim=feature_dim,
        state_window=state_window,
        horizon_count=len(horizons),
        head_indices=head_indices,
        hidden_dim=int(architecture.get("hidden_dim", 128)),
        dropout=float(architecture.get("dropout", 0.0)),
        horizon_embedding_dim=int(architecture.get("horizon_embedding_dim", 24)),
    )


def _role_summary(actual: pd.DataFrame, predictions: dict[int, pd.DataFrame]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizon, frame in predictions.items():
        y = actual[f"target_{horizon}d"].to_numpy(dtype=float)
        q05 = frame["q05"].to_numpy(dtype=float)
        q25 = frame["q25"].to_numpy(dtype=float)
        q50 = frame["q50"].to_numpy(dtype=float)
        q75 = frame["q75"].to_numpy(dtype=float)
        q95 = frame["q95"].to_numpy(dtype=float)
        abs_error = np.abs(y - q50)
        uncertainty = q75 - q25
        row = {
            "horizon": horizon,
            "center_pinball": float(pinball_loss(y, q50, 0.50)),
            "lower_tail_pinball": float(pinball_loss(y, q05, 0.05)),
            "upper_tail_pinball": float(pinball_loss(y, q95, 0.95)),
            "coverage_gap_90": float(abs(np.nanmean((y >= q05) & (y <= q95)) - 0.90)),
            "coverage_gap_50": float(abs(np.nanmean((y >= q25) & (y <= q75)) - 0.50)),
            "lower_tail_miss_rate": float(np.nanmean(y < q05)),
            "upper_tail_miss_rate": float(np.nanmean(y > q95)),
            "uncertainty_abs_error_corr": float(pd.Series(uncertainty).corr(pd.Series(abs_error)) or 0.0),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    return {
        "by_horizon": frame.to_dict(orient="records"),
        "average_center_pinball": float(frame["center_pinball"].mean()),
        "average_tail_pinball": float(frame[["lower_tail_pinball", "upper_tail_pinball"]].mean().mean()),
        "average_width_gap": float(frame[["coverage_gap_50", "coverage_gap_90"]].mean().mean()),
        "average_uncertainty_error_corr": float(frame["uncertainty_abs_error_corr"].mean()),
        "average_lower_tail_miss_rate": float(frame["lower_tail_miss_rate"].mean()),
        "average_upper_tail_miss_rate": float(frame["upper_tail_miss_rate"].mean()),
    }


def _ablation_summary(
    candidate: dict[str, Any],
    fitted: dict[str, Any],
    feature_eval: pd.DataFrame,
    target_eval: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    if candidate["family"] != "role_routed":
        return {}

    scaled_eval = fitted["scaler"].transform(feature_eval)
    x_eval, _eval_y, eval_timestamps = _build_sequence_arrays(
        scaled_eval,
        fitted["target_scaler"].transform(target_eval),
        fitted["state_window"],
    )
    if len(eval_timestamps) == 0:
        return {}

    base_predictions = _prediction_frames(
        model=fitted["model"],
        x=x_eval,
        timestamps=eval_timestamps,
        horizons=fitted["horizons"],
        quantiles=fitted["quantiles"],
        output_style=fitted["output_style"],
        target_scaler=fitted["target_scaler"],
        device=device,
    )
    base_predictions = apply_calibrator(
        predictions=base_predictions,
        calibrator=fitted["calibrator"],
        quantiles=fitted["quantiles"],
        context_score=_calibration_context_score(feature_eval.loc[eval_timestamps]),
    )
    base_actual = target_eval.loc[eval_timestamps]
    base_summary = _role_summary(base_actual, base_predictions)
    bucket_indices = candidate["feature_metadata"]["head_bucket_indices"]
    out: dict[str, Any] = {}

    for head_name, per_bucket in bucket_indices.items():
        for bucket_name, indices in per_bucket.items():
            if not indices:
                continue
            ablated = x_eval.copy()
            ablated[:, :, indices] = 0.0
            predictions = _prediction_frames(
                model=fitted["model"],
                x=ablated,
                timestamps=eval_timestamps,
                horizons=fitted["horizons"],
                quantiles=fitted["quantiles"],
                output_style=fitted["output_style"],
                target_scaler=fitted["target_scaler"],
                device=device,
            )
            predictions = apply_calibrator(
                predictions=predictions,
                calibrator=fitted["calibrator"],
                quantiles=fitted["quantiles"],
                context_score=_calibration_context_score(feature_eval.loc[eval_timestamps]),
            )
            summary = _role_summary(base_actual, predictions)
            out[f"{bucket_name}__{head_name}"] = {
                "delta_center_pinball": float(summary["average_center_pinball"] - base_summary["average_center_pinball"]),
                "delta_width_gap": float(summary["average_width_gap"] - base_summary["average_width_gap"]),
                "delta_tail_pinball": float(summary["average_tail_pinball"] - base_summary["average_tail_pinball"]),
                "delta_uncertainty_error_corr": float(
                    summary["average_uncertainty_error_corr"] - base_summary["average_uncertainty_error_corr"]
                ),
                "by_horizon": summary["by_horizon"],
            }
    return out


def _deployment_score(test_metrics: dict[str, Any], role_summary: dict[str, Any]) -> float:
    tail_gap = 0.5 * (
        abs(role_summary["average_lower_tail_miss_rate"] - 0.05)
        + abs(role_summary["average_upper_tail_miss_rate"] - 0.05)
    )
    uncertainty_penalty = max(0.0, -role_summary["average_uncertainty_error_corr"])
    return (
        float(test_metrics["average_wis"])
        + float(test_metrics["average_calibration_gap"])
        + 0.50 * tail_gap
        + 0.15 * float(role_summary["average_width_gap"])
        + 0.10 * uncertainty_penalty
    )


def fit_candidate_model(
    candidate: dict[str, Any],
    feature_train: pd.DataFrame,
    target_train: pd.DataFrame,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    state_window = int(config["fusion"]["state_window"])
    quantiles = list(config["fusion"]["quantiles"])
    training_cfg = config["fusion"]["training"]
    partitions = _partition_training_frame(len(feature_train), state_window)
    train_end = int(partitions["train_end"])
    model_val_end = int(partitions["model_val_end"])
    calibration_start = int(partitions["calibration_start"])

    scaler = FeatureScaler.fit(feature_train.iloc[:train_end])
    target_scaler = TargetScaler.fit(target_train.iloc[:train_end])
    train_scaled = scaler.transform(feature_train)
    train_target_scaled = target_scaler.transform(target_train)

    train_x, train_y, _ = _build_sequence_arrays(
        train_scaled.iloc[:train_end],
        train_target_scaled.iloc[:train_end],
        state_window,
    )
    model_val_x, model_val_y, model_val_timestamps = _build_sequence_arrays(
        train_scaled.iloc[max(0, train_end - state_window + 1) : model_val_end],
        train_target_scaled.iloc[max(0, train_end - state_window + 1) : model_val_end],
        state_window,
    )
    calibration_x, calibration_y, calibration_timestamps = _build_sequence_arrays(
        train_scaled.iloc[max(0, calibration_start - state_window + 1) :],
        train_target_scaled.iloc[max(0, calibration_start - state_window + 1) :],
        state_window,
    )
    if len(calibration_timestamps) == 0:
        calibration_x = model_val_x
        calibration_y = model_val_y
        calibration_timestamps = model_val_timestamps

    model = _build_model_for_candidate(candidate, feature_dim=feature_train.shape[1], state_window=state_window)
    best_state, history = _train_model(
        model=model,
        train_dataset=FusionDataset(train_x, train_y),
        val_dataset=FusionDataset(model_val_x, model_val_y),
        training_cfg=training_cfg,
        quantiles=quantiles,
        horizons=candidate["horizons"],
        output_style=candidate["output_style"],
        device=device,
    )
    model.load_state_dict(best_state)
    model_val_predictions = _prediction_frames(
        model=model,
        x=model_val_x,
        timestamps=model_val_timestamps,
        horizons=candidate["horizons"],
        quantiles=quantiles,
        output_style=candidate["output_style"],
        target_scaler=target_scaler,
        device=device,
    )
    calibration_predictions = _prediction_frames(
        model=model,
        x=calibration_x,
        timestamps=calibration_timestamps,
        horizons=candidate["horizons"],
        quantiles=quantiles,
        output_style=candidate["output_style"],
        target_scaler=target_scaler,
        device=device,
    )
    model_val_actual = target_train.loc[model_val_timestamps]
    calibration_actual = target_train.loc[calibration_timestamps]
    model_val_metrics = summarize_quantile_metrics(model_val_actual, model_val_predictions, quantiles)

    calibration_context = _calibration_context_score(feature_train.loc[calibration_timestamps]) if calibration_timestamps else pd.Series(dtype=float)
    split_mid = max(1, len(calibration_timestamps) // 2)
    fit_index = pd.Index(calibration_timestamps[:split_mid], name="timestamp")
    eval_index = pd.Index(calibration_timestamps[split_mid:], name="timestamp")
    if len(eval_index) == 0:
        eval_index = fit_index
    fit_actual = calibration_actual.loc[fit_index]
    eval_actual = calibration_actual.loc[eval_index]
    fit_predictions = {h: frame.loc[fit_index] for h, frame in calibration_predictions.items()}
    eval_predictions = {h: frame.loc[eval_index] for h, frame in calibration_predictions.items()}
    fit_context = calibration_context.loc[fit_index]
    eval_context = calibration_context.loc[eval_index]

    calibration_candidates = []
    for method in config["fusion"]["calibration_methods"]:
        calibrator = fit_calibrator(
            method=method,
            actual=fit_actual,
            predictions=fit_predictions,
            quantiles=quantiles,
            context_score=fit_context,
        )
        calibrated_eval = apply_calibrator(
            predictions=eval_predictions,
            calibrator=calibrator,
            quantiles=quantiles,
            context_score=eval_context,
        )
        metrics = summarize_quantile_metrics(eval_actual, calibrated_eval, quantiles)
        calibration_candidates.append({"method": method, "calibrator": calibrator, "metrics": metrics})
    calibration_candidates.sort(key=lambda item: item["metrics"]["selection_score"])
    best_calibration = calibration_candidates[0]

    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "calibrator": best_calibration["calibrator"],
        "calibration_method": best_calibration["method"],
        "calibration_selection": {item["method"]: item["metrics"] for item in calibration_candidates},
        "calibration_calibrators": {item["method"]: item["calibrator"] for item in calibration_candidates},
        "model_val_metrics": model_val_metrics,
        "selection_metrics": best_calibration["metrics"],
        "history": history,
        "state_window": state_window,
        "quantiles": quantiles,
        "horizons": candidate["horizons"],
        "output_style": candidate["output_style"],
    }


def evaluate_candidate_model(
    candidate: dict[str, Any],
    fitted: dict[str, Any],
    feature_eval: pd.DataFrame,
    target_eval: pd.DataFrame,
    device: str,
    include_ablation: bool = False,
) -> dict[str, Any]:
    scaled_eval = fitted["scaler"].transform(feature_eval)
    eval_target_scaled = fitted["target_scaler"].transform(target_eval)
    x_eval, _eval_y, eval_timestamps = _build_sequence_arrays(scaled_eval, eval_target_scaled, fitted["state_window"])
    predictions = _prediction_frames(
        model=fitted["model"],
        x=x_eval,
        timestamps=eval_timestamps,
        horizons=fitted["horizons"],
        quantiles=fitted["quantiles"],
        output_style=fitted["output_style"],
        target_scaler=fitted["target_scaler"],
        device=device,
    )
    actual = target_eval.loc[eval_timestamps]
    context = _calibration_context_score(feature_eval.loc[eval_timestamps]) if eval_timestamps else pd.Series(dtype=float)
    calibrated_predictions = apply_calibrator(
        predictions=predictions,
        calibrator=fitted["calibrator"],
        quantiles=fitted["quantiles"],
        context_score=context,
    )
    test_metrics = summarize_quantile_metrics(actual, calibrated_predictions, fitted["quantiles"])
    role_summary = _role_summary(actual, calibrated_predictions)
    out = {
        "predictions": calibrated_predictions,
        "actual": actual,
        "test_metrics": test_metrics,
        "role_summary": role_summary,
        "deployment_score": _deployment_score(test_metrics, role_summary),
    }
    if include_ablation:
        out["ablation"] = _ablation_summary(candidate, fitted, feature_eval, target_eval, device=device)
    return out


def predict_live_candidate_model(
    candidate: dict[str, Any],
    fitted: dict[str, Any],
    feature_live: pd.DataFrame,
    device: str,
) -> dict[int, pd.DataFrame]:
    state_window = fitted["state_window"]
    if len(feature_live) < state_window:
        raise ValueError("Insufficient live feature history for prediction.")
    scaled = fitted["scaler"].transform(feature_live)
    x = scaled.iloc[-state_window:].to_numpy(dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    timestamps = [pd.Timestamp(feature_live.index[-1])]
    predictions = _prediction_frames(
        model=fitted["model"],
        x=x,
        timestamps=timestamps,
        horizons=fitted["horizons"],
        quantiles=fitted["quantiles"],
        output_style=fitted["output_style"],
        target_scaler=fitted["target_scaler"],
        device=device,
    )
    context = _calibration_context_score(feature_live.iloc[[-1]])
    return apply_calibrator(
        predictions=predictions,
        calibrator=fitted["calibrator"],
        quantiles=fitted["quantiles"],
        context_score=context,
    )


def _feature_payload_for_candidate(
    candidate: dict[str, Any],
    states_by_bucket: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidate["family"] == "baseline":
        frame = build_fusion_feature_frame(states_by_bucket, variant=candidate["variant"], communication_channels=[])
        return frame, {}
    payload = build_role_routed_feature_bundle(
        states_by_bucket=states_by_bucket,
        role_map_name=candidate["role_map_name"],
        interface_name=candidate["interface_name"],
    )
    return payload["feature_frame"], payload


def _fold_dirs(config: dict[str, Any], fold_id: str) -> dict[str, Path]:
    experiment_name = config["experiment_name"]
    base = ensure_dir(Path(config["paths"]["results_dir"]) / experiment_name / fold_id)
    reports = ensure_dir(Path(config["paths"]["reports_dir"]) / experiment_name / fold_id)
    models = ensure_dir(Path(config["paths"]["models_dir"]) / experiment_name / fold_id)
    return {"results": base, "reports": reports, "models": models}


def run_fold(
    config: dict[str, Any],
    data_bundle: Any,
    fold: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    inner_segments = build_inner_segments(data_bundle.close.index, train_end_pos=int(fold["train_end_pos"]), config=config)
    semantic_targets = build_semantic_targets(
        close=data_bundle.close,
        buckets=data_bundle.buckets,
        freshness_scores=data_bundle.freshness_scores,
        missingness=data_bundle.missingness,
    )
    specialist_summary: dict[str, Any] = {}
    train_states_by_bucket: dict[str, pd.DataFrame] = {}
    test_states_by_bucket: dict[str, pd.DataFrame] = {}

    for bucket_name in data_bundle.buckets:
        selection = select_best_specialist(
            bucket_name=bucket_name,
            features=data_bundle.buckets[bucket_name],
            targets=semantic_targets[bucket_name],
            train_end_pos=int(fold["train_end_pos"]),
            config=config,
            artifact_root=_fold_dirs(config, fold["fold_id"])["models"],
            device=device,
        )
        specialist_summary[bucket_name] = selection
        states = build_walkforward_states(
            bucket_name=bucket_name,
            features=data_bundle.buckets[bucket_name],
            targets=semantic_targets[bucket_name],
            freshness=data_bundle.freshness_scores[bucket_name],
            missingness=data_bundle.missingness[bucket_name],
            inner_segments=inner_segments,
            outer_fold=fold,
            selection=selection,
            config=config,
            feature_root=Path(config["paths"]["features_dir"]) / config["experiment_name"] / fold["fold_id"],
            device=device,
        )
        train_states_by_bucket[bucket_name] = states["train"]
        test_states_by_bucket[bucket_name] = states["test"]

    candidate_results: list[dict[str, Any]] = []
    for candidate in enumerate_candidate_specs(config):
        train_frame, train_payload = _feature_payload_for_candidate(candidate, train_states_by_bucket)
        test_frame, test_payload = _feature_payload_for_candidate(candidate, test_states_by_bucket)
        candidate = copy.deepcopy(candidate)
        candidate["feature_metadata"] = train_payload or test_payload
        shared_train_index = sorted(set(train_frame.index).intersection(set.intersection(*(set(frame.index) for frame in train_states_by_bucket.values()))))
        shared_test_index = sorted(set(test_frame.index).intersection(set.intersection(*(set(frame.index) for frame in test_states_by_bucket.values()))))
        train_target = data_bundle.targets.loc[shared_train_index, [f"target_{h}d" for h in candidate["horizons"]]].dropna()
        test_target = data_bundle.targets.loc[shared_test_index, [f"target_{h}d" for h in candidate["horizons"]]].dropna()
        common_train_index = train_target.index.intersection(train_frame.index).sort_values()
        common_test_index = test_target.index.intersection(test_frame.index).sort_values()
        if len(common_train_index) == 0 or len(common_test_index) == 0:
            continue
        feature_train = train_frame.loc[common_train_index]
        feature_test = test_frame.loc[common_test_index]
        target_train = train_target.loc[common_train_index]
        target_test = test_target.loc[common_test_index]
        fitted = fit_candidate_model(candidate, feature_train, target_train, config=config, device=device)
        evaluated = evaluate_candidate_model(
            candidate=candidate,
            fitted=fitted,
            feature_eval=feature_test,
            target_eval=target_test,
            device=device,
            include_ablation=True,
        )
        candidate_results.append(
            {
                "candidate": candidate,
                "selection_metrics": fitted["selection_metrics"],
                "model_val_metrics": fitted["model_val_metrics"],
                "calibration_method": fitted["calibration_method"],
                "calibration_selection": fitted["calibration_selection"],
                "test_metrics": evaluated["test_metrics"],
                "role_summary": evaluated["role_summary"],
                "deployment_score": evaluated["deployment_score"],
                "ablation": evaluated.get("ablation", {}),
            }
        )

    candidate_results.sort(key=lambda item: item["deployment_score"])
    fold_dirs = _fold_dirs(config, fold["fold_id"])
    write_json(
        fold_dirs["results"] / "final_selection_candidates.json",
        [
            {
                "name": item["candidate"]["name"],
                "family": item["candidate"]["family"],
                "horizon_set_name": item["candidate"]["horizon_set_name"],
                "calibration_method": item["calibration_method"],
                "deployment_score": item["deployment_score"],
                "selection_metrics": item["selection_metrics"],
                "test_metrics": item["test_metrics"],
                "role_summary": item["role_summary"],
            }
            for item in candidate_results
        ],
    )
    return {
        "fold": fold,
        "specialists": specialist_summary,
        "candidates": candidate_results,
        "best": candidate_results[0],
    }


def aggregate_specialists(fold_results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket_name in config["specialists"]["buckets"]:
        anchor_stats: dict[str, list[float]] = {}
        candidate_meta: dict[str, dict[str, Any]] = {}
        retained_stats: dict[str, dict[str, list[float]]] = {}
        retained_meta: dict[str, dict[str, dict[str, Any]]] = {}
        for fold_result in fold_results:
            selection = fold_result["specialists"][bucket_name]
            for row in selection["leaderboard"]:
                name = row["name"]
                anchor_stats.setdefault(name, []).append(float(row["selection_score"]))
                candidate_meta[name] = {
                    "architecture": row["architecture"],
                    "lookback": int(row["lookback"]),
                }
            for retained in selection["retained_candidates"]:
                group_name = retained["group_name"]
                candidate = retained["candidate"]
                name = candidate["name"]
                retained_stats.setdefault(group_name, {}).setdefault(name, []).append(float(retained["metrics"]["selection_score"]))
                retained_meta.setdefault(group_name, {})[name] = candidate

        anchor_leaderboard = [
            {
                "name": name,
                "architecture": candidate_meta[name]["architecture"],
                "lookback": candidate_meta[name]["lookback"],
                "mean_selection_score": float(np.mean(scores)),
                "fold_wins": int(sum(name == fold_result["specialists"][bucket_name]["best_candidate"]["name"] for fold_result in fold_results)),
            }
            for name, scores in anchor_stats.items()
        ]
        anchor_leaderboard.sort(key=lambda item: (item["mean_selection_score"], -item["fold_wins"]))
        final_anchor = anchor_leaderboard[0]

        retained = []
        for group_name, group_stats in retained_stats.items():
            rows = [
                {
                    "group_name": group_name,
                    "name": name,
                    "mean_selection_score": float(np.mean(scores)),
                    "retained_count": int(len(scores)),
                    "candidate": retained_meta[group_name][name],
                }
                for name, scores in group_stats.items()
            ]
            rows.sort(key=lambda item: (-item["retained_count"], item["mean_selection_score"]))
            retained.append(rows[0])

        final_selection = {
            "best_candidate": retained_meta.get(retained[0]["group_name"], {}).get(final_anchor["name"], None),
            "retained_candidates": [],
            "post_smooth_alpha": float(config["specialists"]["buckets"][bucket_name]["post_smooth_alpha"]),
        }
        for row in retained:
            final_selection["retained_candidates"].append({"group_name": row["group_name"], "candidate": row["candidate"]})
        best_candidate = next(
            (
                row["candidate"]
                for row in retained
                if row["candidate"]["name"] == final_anchor["name"]
            ),
            None,
        )
        if best_candidate is None:
            best_candidate = copy.deepcopy(config["specialists"]["buckets"][bucket_name]["candidates"][0])
            for candidate in config["specialists"]["buckets"][bucket_name]["candidates"]:
                if candidate["name"] == final_anchor["name"]:
                    best_candidate = copy.deepcopy(candidate)
                    break
        final_selection["best_candidate"] = best_candidate

        out[bucket_name] = {
            "anchor_leaderboard": anchor_leaderboard,
            "retained_selection": retained,
            "final_selection": final_selection,
        }
    return out


def aggregate_candidates(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fold_result in fold_results:
        for candidate in fold_result["candidates"]:
            grouped.setdefault(candidate["candidate"]["name"], []).append(candidate)

    leaderboard = []
    for name, rows in grouped.items():
        first = rows[0]
        avg_wis = float(np.mean([row["test_metrics"]["average_wis"] for row in rows]))
        avg_cal_gap = float(np.mean([row["test_metrics"]["average_calibration_gap"] for row in rows]))
        deployment = float(np.mean([row["deployment_score"] for row in rows]))
        std_wis = float(np.std([row["test_metrics"]["average_wis"] for row in rows]))
        std_cal_gap = float(np.std([row["test_metrics"]["average_calibration_gap"] for row in rows]))
        width_gap = float(np.mean([row["role_summary"]["average_width_gap"] for row in rows]))
        tail_gap = float(
            np.mean(
                [
                    0.5
                    * (
                        abs(row["role_summary"]["average_lower_tail_miss_rate"] - 0.05)
                        + abs(row["role_summary"]["average_upper_tail_miss_rate"] - 0.05)
                    )
                    for row in rows
                ]
            )
        )
        stability_penalty = 0.30 * std_wis + 0.30 * std_cal_gap
        final_score = deployment + stability_penalty
        leaderboard.append(
            {
                "name": name,
                "family": first["candidate"]["family"],
                "horizon_set_name": first["candidate"]["horizon_set_name"],
                "output_style": first["candidate"]["output_style"],
                "role_map_name": first["candidate"].get("role_map_name"),
                "interface_name": first["candidate"].get("interface_name"),
                "fold_count": len(rows),
                "mean_average_wis": avg_wis,
                "mean_average_calibration_gap": avg_cal_gap,
                "mean_deployment_score": deployment,
                "stability_penalty": stability_penalty,
                "final_score": final_score,
                "mean_width_gap": width_gap,
                "mean_tail_gap": tail_gap,
                "mean_center_pinball": float(np.mean([row["role_summary"]["average_center_pinball"] for row in rows])),
                "mean_tail_pinball": float(np.mean([row["role_summary"]["average_tail_pinball"] for row in rows])),
                "mean_uncertainty_error_corr": float(np.mean([row["role_summary"]["average_uncertainty_error_corr"] for row in rows])),
                "calibration_methods": sorted({row["calibration_method"] for row in rows}),
            }
        )

    leaderboard.sort(key=lambda item: item["final_score"])
    return leaderboard


def aggregate_role_matrix(fold_results: list[dict[str, Any]], best_candidate_name: str) -> dict[str, Any]:
    rows = []
    for fold_result in fold_results:
        for candidate in fold_result["candidates"]:
            if candidate["candidate"]["name"] != best_candidate_name:
                continue
            for key, values in candidate.get("ablation", {}).items():
                bucket_name, head_name = key.split("__")
                rows.append(
                    {
                        "bucket": bucket_name,
                        "head": head_name,
                        "delta_center_pinball": float(values["delta_center_pinball"]),
                        "delta_width_gap": float(values["delta_width_gap"]),
                        "delta_tail_pinball": float(values["delta_tail_pinball"]),
                        "delta_uncertainty_error_corr": float(values["delta_uncertainty_error_corr"]),
                    }
                )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    grouped = []
    for (bucket_name, head_name), subset in frame.groupby(["bucket", "head"]):
        grouped.append(
            {
                "bucket": bucket_name,
                "head": head_name,
                "mean_delta_center_pinball": float(subset["delta_center_pinball"].mean()),
                "mean_delta_width_gap": float(subset["delta_width_gap"].mean()),
                "mean_delta_tail_pinball": float(subset["delta_tail_pinball"].mean()),
                "mean_delta_uncertainty_error_corr": float(subset["delta_uncertainty_error_corr"].mean()),
            }
        )
    return {"rows": grouped}


def _fit_live_candidate(
    config: dict[str, Any],
    data_bundle: Any,
    candidate_row: dict[str, Any],
    specialist_selection: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    candidate = {
        "name": candidate_row["name"],
        "family": candidate_row["family"],
        "horizon_set_name": candidate_row["horizon_set_name"],
        "horizons": list(config["fusion"]["horizon_sets"][candidate_row["horizon_set_name"]]),
        "output_style": candidate_row["output_style"],
        "role_map_name": candidate_row.get("role_map_name"),
        "interface_name": candidate_row.get("interface_name"),
        "architecture": (
            _architecture_spec(config, "quantile_mlp")
            if candidate_row["name"].endswith("quantile_mlp")
            else _architecture_spec(config, "horizon_gated_mlp")
            if candidate_row["name"].endswith("horizon_gated_mlp")
            else {"name": "role_routed_structured", "hidden_dim": 128, "dropout": 0.10, "horizon_embedding_dim": 24}
        ),
    }
    if candidate["family"] == "baseline":
        candidate["variant"] = "D" if "__legacy_D__" in candidate["name"] else "A"

    semantic_targets = build_semantic_targets(
        close=data_bundle.close,
        buckets=data_bundle.buckets,
        freshness_scores=data_bundle.freshness_scores,
        missingness=data_bundle.missingness,
    )
    max_horizon = max(candidate["horizons"])
    train_end_pos = len(data_bundle.close.index) - max_horizon
    inner_segments = build_inner_segments(data_bundle.close.index, train_end_pos=train_end_pos, config=config)

    train_states_by_bucket: dict[str, pd.DataFrame] = {}
    live_history_by_bucket: dict[str, pd.DataFrame] = {}
    live_start = max(0, len(data_bundle.close.index) - int(config["fusion"]["state_window"]) - 5)
    for bucket_name in data_bundle.buckets:
        train_parts = []
        for segment in inner_segments:
            train_parts.append(
                emit_specialist_states(
                    bucket_name=bucket_name,
                    features=data_bundle.buckets[bucket_name],
                    targets=semantic_targets[bucket_name],
                    freshness=data_bundle.freshness_scores[bucket_name],
                    missingness=data_bundle.missingness[bucket_name],
                    train_end_pos=int(segment["fit_end_pos"]),
                    predict_start_pos=int(segment["predict_start_pos"]),
                    predict_end_pos=int(segment["predict_end_pos"]),
                    selection=specialist_selection[bucket_name]["final_selection"],
                    config=config,
                    device=device,
                )
            )
        train_states_by_bucket[bucket_name] = pd.concat(train_parts).sort_index()
        live_history_by_bucket[bucket_name] = emit_specialist_states(
            bucket_name=bucket_name,
            features=data_bundle.buckets[bucket_name],
            targets=semantic_targets[bucket_name],
            freshness=data_bundle.freshness_scores[bucket_name],
            missingness=data_bundle.missingness[bucket_name],
            train_end_pos=len(data_bundle.close.index),
            predict_start_pos=live_start,
            predict_end_pos=len(data_bundle.close.index),
            selection=specialist_selection[bucket_name]["final_selection"],
            config=config,
            device=device,
        )

    train_frame, payload = _feature_payload_for_candidate(candidate, train_states_by_bucket)
    candidate["feature_metadata"] = payload
    live_frame, _live_payload = _feature_payload_for_candidate(candidate, live_history_by_bucket)
    train_target = data_bundle.targets.loc[train_frame.index, [f"target_{h}d" for h in candidate["horizons"]]].dropna()
    common_train_index = train_frame.index.intersection(train_target.index).sort_values()
    fitted = fit_candidate_model(
        candidate=candidate,
        feature_train=train_frame.loc[common_train_index],
        target_train=train_target.loc[common_train_index],
        config=config,
        device=device,
    )
    predictions = predict_live_candidate_model(
        candidate=candidate,
        fitted=fitted,
        feature_live=live_frame,
        device=device,
    )
    return {
        "candidate": candidate,
        "fitted": fitted,
        "predictions": {
            horizon: frame.reset_index().to_dict(orient="records")
            for horizon, frame in predictions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final walk-forward modular system selection.")
    parser.add_argument("--config", required=True, help="Path to the JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    device = str(config["device"])
    data_bundle = load_forecast_data(config)
    folds = build_outer_folds(data_bundle.close.index, config)

    fold_results = [run_fold(config=config, data_bundle=data_bundle, fold=fold, device=device) for fold in folds]
    specialist_selection = aggregate_specialists(fold_results, config=config)
    candidate_leaderboard = aggregate_candidates(fold_results)
    best_candidate = candidate_leaderboard[0]
    role_matrix = aggregate_role_matrix(fold_results, best_candidate_name=best_candidate["name"])
    live_refit = _fit_live_candidate(
        config=config,
        data_bundle=data_bundle,
        candidate_row=best_candidate,
        specialist_selection=specialist_selection,
        device=device,
    )

    output_root = ensure_dir(Path(config["paths"]["results_dir"]) / config["experiment_name"])
    report_root = ensure_dir(Path(config["paths"]["reports_dir"]) / config["experiment_name"])
    summary = {
        "candidate_leaderboard": candidate_leaderboard,
        "best_candidate": best_candidate,
        "specialist_selection": specialist_selection,
        "role_matrix": role_matrix,
        "live_refit": live_refit["predictions"],
        "fold_count": len(fold_results),
    }
    write_json(output_root / "final_selection_summary.json", summary)

    lines = [
        "# Final System Selection",
        "",
        f"- Fold count: {len(fold_results)}",
        f"- Selected system: {best_candidate['name']}",
        f"- Family: {best_candidate['family']}",
        f"- Horizon set: {best_candidate['horizon_set_name']}",
        f"- Output style: {best_candidate['output_style']}",
        f"- Mean WIS: {best_candidate['mean_average_wis']:.6f}",
        f"- Mean calibration gap: {best_candidate['mean_average_calibration_gap']:.6f}",
        f"- Mean uncertainty/error corr: {best_candidate['mean_uncertainty_error_corr']:.6f}",
        "",
        "## Specialists",
    ]
    for bucket_name, bucket_summary in specialist_selection.items():
        anchor = bucket_summary["anchor_leaderboard"][0]
        retained = ", ".join(
            f"{row['group_name']}={row['candidate']['name']}" for row in bucket_summary["retained_selection"]
        )
        lines.append(
            f"- {bucket_name}: anchor={anchor['name']} ({anchor['architecture']}, lb={anchor['lookback']}), retained={retained}"
        )
    lines.append("")
    lines.append("## Candidate Leaderboard")
    for row in candidate_leaderboard[:10]:
        lines.append(
            f"- {row['name']}: final_score={row['final_score']:.6f}, wis={row['mean_average_wis']:.6f}, "
            f"cal_gap={row['mean_average_calibration_gap']:.6f}, tail_gap={row['mean_tail_gap']:.6f}"
        )
    if role_matrix:
        lines.append("")
        lines.append("## Role Matrix")
        for row in role_matrix["rows"]:
            lines.append(
                f"- {row['bucket']} -> {row['head']}: "
                f"d_center={row['mean_delta_center_pinball']:.6f}, "
                f"d_width={row['mean_delta_width_gap']:.6f}, "
                f"d_tail={row['mean_delta_tail_pinball']:.6f}"
            )
    lines.append("")
    lines.append("## Live Refit")
    for horizon, rows in live_refit["predictions"].items():
        row = rows[0]
        lines.append(
            f"- {horizon}d: q05={row['q05']:.6f}, q25={row['q25']:.6f}, q50={row['q50']:.6f}, q75={row['q75']:.6f}, q95={row['q95']:.6f}"
        )
    (report_root / "final_selection_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
