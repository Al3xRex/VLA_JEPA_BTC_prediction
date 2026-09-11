from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dual_model_forecaster.metrics import summarize_quantile_metrics
from dual_model_forecaster.specialists import COMMUNICATION_COLUMNS, FeatureScaler, SUMMARY_COLUMNS
from dual_model_forecaster.utils import ensure_dir, quantile_name, write_json


class FusionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class TargetScaler:
    def __init__(self, median: pd.Series, iqr: pd.Series) -> None:
        self.median = median
        self.iqr = iqr.replace(0, 1.0).fillna(1.0)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TargetScaler":
        median = frame.median(axis=0)
        q25 = frame.quantile(0.25, axis=0)
        q75 = frame.quantile(0.75, axis=0)
        iqr = (q75 - q25).replace(0, 1.0).fillna(1.0)
        return cls(median=median.fillna(0.0), iqr=iqr)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return ((frame - self.median) / self.iqr).replace([np.inf, -np.inf], np.nan)

    def inverse_transform_array(self, values: np.ndarray, columns: list[str]) -> np.ndarray:
        out = values.copy()
        for idx, column in enumerate(columns):
            out[:, idx, :] = out[:, idx, :] * float(self.iqr[column]) + float(self.median[column])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"median": self.median.to_dict(), "iqr": self.iqr.to_dict()}


def _semantic_state_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(COMMUNICATION_COLUMNS) | set(SUMMARY_COLUMNS)
    return [column for column in frame.columns if not column.startswith("scale_") and column not in excluded]


def _summary_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column in SUMMARY_COLUMNS]


def _token_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("scale_")]


def _communication_columns(frame: pd.DataFrame, selected: list[str]) -> list[str]:
    return [column for column in selected if column in frame.columns]


def _bucket_direction(bucket_name: str, frame: pd.DataFrame) -> pd.Series:
    if "summary_direction" in frame.columns:
        return frame["summary_direction"].astype(float)
    if bucket_name == "structure":
        return frame["structural_reversion_pressure"].astype(float)
    if bucket_name == "environment":
        return (
            frame["liquidity_tailwind"].astype(float)
            - frame["liquidity_headwind"].astype(float)
            + frame["macro_risk_on"].astype(float)
            - frame["macro_risk_off"].astype(float)
        )
    if bucket_name == "edges":
        return frame["upside_stretch"].astype(float) - frame["downside_stretch"].astype(float)
    if bucket_name == "liquidation":
        return (
            frame["liquidation_directional_pressure"].astype(float)
            + 0.50 * frame["cascade_asymmetry"].astype(float)
        )
    return frame["trend_pressure_up"].astype(float) - frame["trend_pressure_down"].astype(float)


def _bucket_confidence(bucket_name: str, frame: pd.DataFrame) -> pd.Series:
    if "summary_confidence" in frame.columns:
        return frame["summary_confidence"].astype(float)
    if bucket_name == "structure":
        return frame["structural_confidence"].astype(float)
    if bucket_name == "environment":
        return frame["environment_confidence"].astype(float)
    if bucket_name == "edges":
        return frame["edge_confidence"].astype(float)
    if bucket_name == "liquidation":
        return frame["liquidation_confidence"].astype(float)
    return frame["movement_confidence"].astype(float)


def _state_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(default)


def _causal_exposure(series: pd.Series, span: int = 14) -> pd.Series:
    return (
        series.clip(0.0, 1.0)
        .ewm(span=int(span), adjust=False, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )


def _build_fast_regime_dynamics(states_by_bucket: dict[str, pd.DataFrame]) -> pd.DataFrame:
    edges = states_by_bucket["edges"]
    movement = states_by_bucket["movement"]
    index = movement.index
    edges_aligned = edges.reindex(index)

    edge_direction = _bucket_direction("edges", edges_aligned).fillna(0.0).clip(-1.0, 1.0)
    movement_direction = _bucket_direction("movement", movement).fillna(0.0).clip(-1.0, 1.0)
    edge_confidence = _bucket_confidence("edges", edges_aligned).fillna(0.0).clip(0.0, 1.0)
    movement_confidence = _bucket_confidence("movement", movement).fillna(0.0).clip(0.0, 1.0)

    edge_mean_reversion = _state_series(edges_aligned, "mean_reversion_pressure", 0.0).clip(0.0, 1.0)
    edge_stretch = edge_direction.abs().clip(0.0, 1.0)
    trend_persistence = _state_series(movement, "trend_persistence", 0.0).clip(0.0, 1.0)
    chop_risk = _state_series(movement, "chop_risk", 0.5).clip(0.0, 1.0)
    trend_influence = _state_series(movement, "trend_strategy_influence", 0.5).clip(0.0, 1.0)
    momentum_influence = _state_series(movement, "momentum_strategy_influence", 0.0).clip(0.0, 1.0)
    mean_reversion_influence = _state_series(movement, "mean_reversion_strategy_influence", 0.0).clip(0.0, 1.0)

    low_chop_trend = (1.0 - chop_risk).clip(0.0, 1.0)
    trend_strength = pd.concat(
        [trend_influence, trend_persistence, movement_direction.abs().clip(0.0, 1.0)],
        axis=1,
    ).mean(axis=1)
    trending_state = (trend_strength * (0.50 + 0.50 * low_chop_trend)).clip(0.0, 1.0)
    trend_exhaustion = (
        low_chop_trend
        * trend_persistence
        * pd.concat([trend_influence, movement_direction.abs().clip(0.0, 1.0)], axis=1).max(axis=1)
    ).clip(0.0, 1.0)

    stationary_setup = (
        0.50 * mean_reversion_influence
        + 0.30 * edge_mean_reversion
        + 0.20 * chop_risk
    ).clip(0.0, 1.0)
    stationary_state = (
        stationary_setup
        * (0.50 + 0.50 * (chop_risk + trend_exhaustion).clip(0.0, 1.0))
        * (0.75 + 0.25 * edge_stretch)
    ).clip(0.0, 1.0)

    edges_chop_relevance = (
        (0.55 * trend_exhaustion + 0.85 * stationary_state).clip(0.0, 1.0)
        * (0.50 + 0.50 * edge_confidence)
    ).clip(0.0, 1.0)
    movement_chop_relevance = (
        (0.70 * trending_state + 0.30 * momentum_influence)
        * (0.50 + 0.50 * movement_confidence)
    ).clip(0.0, 1.0)

    edges_exposure = _causal_exposure(edges_chop_relevance)
    movement_exposure = _causal_exposure(movement_chop_relevance)
    movement_to_edges = (movement_exposure * (0.70 + 0.30 * trend_exhaustion)).clip(0.0, 1.0)
    edges_to_movement = (edges_exposure * (0.70 + 0.30 * trending_state)).clip(0.0, 1.0)

    edges_raw = (edges_chop_relevance + 0.45 * movement_to_edges).clip(lower=0.0)
    movement_raw = (movement_chop_relevance + 0.45 * edges_to_movement).clip(lower=0.0)
    relevance_total = (edges_raw + movement_raw).replace(0.0, np.nan)
    edges_dynamic = (edges_raw / relevance_total).fillna(0.5).clip(0.0, 1.0)
    movement_dynamic = (movement_raw / relevance_total).fillna(0.5).clip(0.0, 1.0)

    stationary_trending_balance = ((stationary_state + trend_exhaustion).clip(0.0, 1.0) - trending_state).clip(-1.0, 1.0)
    weighted_edge_direction = edge_direction * edge_confidence * edges_dynamic
    weighted_movement_direction = movement_direction * movement_confidence * movement_dynamic
    confidence_mean = pd.concat([edge_confidence, movement_confidence], axis=1).mean(axis=1).clip(0.0, 1.0)

    dynamics = pd.DataFrame(index=index)
    dynamics["fast_trending_state"] = trending_state
    dynamics["fast_stationary_state"] = stationary_state
    dynamics["fast_trend_exhaustion_pressure"] = trend_exhaustion
    dynamics["stationary_trending_balance"] = stationary_trending_balance
    dynamics["edges_chop_relevance"] = edges_chop_relevance
    dynamics["movement_chop_relevance"] = movement_chop_relevance
    dynamics["edges_recurrent_exposure"] = edges_exposure
    dynamics["movement_recurrent_exposure"] = movement_exposure
    dynamics["movement_to_edges_rotation_pressure"] = movement_to_edges
    dynamics["edges_to_movement_rotation_pressure"] = edges_to_movement
    dynamics["edges_dynamic_relevance"] = edges_dynamic
    dynamics["movement_dynamic_relevance"] = movement_dynamic
    dynamics["edge_movement_relevance_spread"] = edges_dynamic - movement_dynamic
    dynamics["fast_regime_weighted_direction"] = weighted_edge_direction + weighted_movement_direction
    dynamics["fast_regime_relevance_disagreement"] = (edge_direction - movement_direction).abs() * confidence_mean
    dynamics["fast_regime_rotation_intensity"] = (movement_to_edges + edges_to_movement).clip(0.0, 1.0)
    dynamics["stationary_trending_transition_risk"] = (
        stationary_trending_balance.diff().abs().ewm(span=7, adjust=False, min_periods=1).mean().fillna(0.0)
    ).clip(0.0, 1.0)
    return dynamics.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_interactions(states_by_bucket: dict[str, pd.DataFrame]) -> pd.DataFrame:
    structure = states_by_bucket["structure"]
    environment = states_by_bucket["environment"]
    edges = states_by_bucket["edges"]
    movement = states_by_bucket["movement"]
    liquidation = states_by_bucket.get("liquidation")

    slow_direction = pd.concat(
        [
            _bucket_direction("structure", structure),
            _bucket_direction("environment", environment),
        ],
        axis=1,
    ).mean(axis=1)
    fast_direction = pd.concat(
        [
            _bucket_direction("edges", edges),
            _bucket_direction("movement", movement),
        ],
        axis=1,
    ).mean(axis=1)
    slow_confidence = pd.concat(
        [
            _bucket_confidence("structure", structure),
            _bucket_confidence("environment", environment),
        ],
        axis=1,
    ).mean(axis=1)
    fast_confidence = pd.concat(
        [
            _bucket_confidence("edges", edges),
            _bucket_confidence("movement", movement),
        ],
        axis=1,
    ).mean(axis=1)

    directional_frame = pd.DataFrame(
        {
            "structure": _bucket_direction("structure", structure),
            "environment": _bucket_direction("environment", environment),
            "edges": _bucket_direction("edges", edges),
            "movement": _bucket_direction("movement", movement),
        }
    )
    if liquidation is not None:
        directional_frame["liquidation"] = _bucket_direction("liquidation", liquidation)
    confidence_frame = pd.DataFrame(
        {
            "structure": _bucket_confidence("structure", structure),
            "environment": _bucket_confidence("environment", environment),
            "edges": _bucket_confidence("edges", edges),
            "movement": _bucket_confidence("movement", movement),
        }
    )
    if liquidation is not None:
        confidence_frame["liquidation"] = _bucket_confidence("liquidation", liquidation)

    interactions = pd.DataFrame(index=structure.index)
    interactions["movement_trend_vs_edges_stretch"] = (
        movement["trend_pressure_up"].astype(float) - movement["trend_pressure_down"].astype(float)
    ) * (edges["upside_stretch"].astype(float) - edges["downside_stretch"].astype(float))
    interactions["movement_persistence_vs_edges_asymmetry"] = (
        movement["trend_persistence"].astype(float) * edges["edge_asymmetry"].astype(float)
    )
    interactions["structure_overvaluation_vs_environment_tailwind"] = (
        structure["structural_overvaluation"].astype(float) * environment["liquidity_tailwind"].astype(float)
    )
    interactions["conviction_environment_alignment"] = (
        structure["holder_conviction"].astype(float) * _bucket_direction("environment", environment)
    )
    interactions["structure_fragility_vs_movement_continuation"] = (
        structure["holder_distribution_fragility"].astype(float) * movement["trend_persistence"].astype(float)
    )
    interactions["macro_risk_on_vs_fragility"] = (
        environment["macro_risk_on"].astype(float) * structure["holder_distribution_fragility"].astype(float)
    )
    interactions["edges_downside_vs_movement_trend"] = (
        edges["downside_stretch"].astype(float) * _bucket_direction("movement", movement)
    )
    interactions["slow_fast_alignment"] = slow_direction * fast_direction
    interactions["slow_fast_conflict"] = (slow_direction - fast_direction).abs()
    interactions["slow_specialist_agreement"] = directional_frame[["structure", "environment"]].std(axis=1).fillna(0.0)
    interactions["fast_specialist_disagreement"] = directional_frame[["edges", "movement"]].std(axis=1).fillna(0.0)
    interactions["confidence_weighted_agreement"] = directional_frame.mean(axis=1) * confidence_frame.mean(axis=1)
    interactions["confidence_weighted_disagreement"] = directional_frame.std(axis=1).fillna(0.0) * (
        1.0 - confidence_frame.mean(axis=1)
    )
    interactions["slow_direction"] = slow_direction
    interactions["fast_direction"] = fast_direction
    interactions["slow_confidence"] = slow_confidence
    interactions["fast_confidence"] = fast_confidence
    interactions["slow_fast_direction_gap"] = slow_direction - fast_direction
    interactions = pd.concat([interactions, _build_fast_regime_dynamics(states_by_bucket)], axis=1)
    if liquidation is not None:
        liquidation_direction = _bucket_direction("liquidation", liquidation)
        liquidation_confidence = _bucket_confidence("liquidation", liquidation)
        liquidation_cascade = liquidation["liquidation_cascade_pressure"].astype(float)
        liquidation_fuel = liquidation["leverage_fuel"].astype(float)
        liquidation_instability = liquidation["liquidation_volatility_instability"].astype(float)
        liquidation_width_pressure = (
            liquidation_cascade
            * (0.50 + 0.50 * liquidation_confidence.clip(0.0, 1.0))
            * (0.50 + 0.25 * liquidation_fuel.clip(0.0, 1.0) + 0.25 * liquidation_instability.clip(0.0, 1.0))
        )
        interactions["liquidation_direction"] = liquidation_direction
        interactions["liquidation_confidence"] = liquidation_confidence
        interactions["liquidation_quartile_width_pressure"] = liquidation_width_pressure.clip(0.0, 1.5)
        interactions["liquidation_edges_jump_alignment"] = liquidation_direction * _bucket_direction("edges", edges)
        interactions["liquidation_movement_jump_alignment"] = liquidation_direction * _bucket_direction("movement", movement)
        interactions["liquidation_ta_directional_alignment"] = liquidation_direction * fast_direction
        interactions["liquidation_ta_confidence_weighted_alignment"] = (
            liquidation_direction * fast_direction * liquidation_confidence * fast_confidence
        )
        interactions["liquidation_ta_conflict"] = (liquidation_direction - fast_direction).abs()
        interactions["liquidation_short_quartile_bias"] = (
            liquidation_direction * liquidation_width_pressure.clip(0.0, 1.5)
        )
    return interactions.fillna(0.0)


def build_fusion_feature_frame(
    states_by_bucket: dict[str, pd.DataFrame],
    variant: str,
    communication_channels: list[str],
) -> pd.DataFrame:
    prefixed_frames = []
    for bucket_name, frame in states_by_bucket.items():
        semantic_columns = _semantic_state_columns(frame)
        summary_columns = _summary_columns(frame)
        token_columns = _token_columns(frame)
        base_columns = semantic_columns.copy()
        if variant in {"E", "F", "G"}:
            base_columns.extend(summary_columns)
            base_columns.extend(token_columns)
        if variant == "G":
            base_columns.extend(_communication_columns(frame, communication_channels))
        selected = frame.loc[:, list(dict.fromkeys(base_columns))].copy()
        prefixed_frames.append(selected.add_prefix(f"{bucket_name}_"))

    features = pd.concat(prefixed_frames, axis=1).sort_index()
    if variant in {"B", "C", "D"}:
        features = pd.concat([features, features.diff().add_prefix("delta_")], axis=1)

    if variant in {"C", "D", "E", "G"}:
        rolling_mean = features.rolling(7, min_periods=3).mean().add_prefix("roll7_mean_")
        rolling_std = features.rolling(7, min_periods=3).std().fillna(0.0).add_prefix("roll7_std_")
        features = pd.concat([features, rolling_mean, rolling_std], axis=1)

    if variant in {"D", "E", "F", "G"}:
        features = pd.concat([features, _build_interactions(states_by_bucket)], axis=1)

    if variant in {"F", "G"}:
        interactions = _build_interactions(states_by_bucket)
        slow_fast = interactions[
            [
                "slow_direction",
                "fast_direction",
                "slow_confidence",
                "fast_confidence",
                "slow_fast_alignment",
                "slow_fast_conflict",
                "slow_fast_direction_gap",
                "confidence_weighted_disagreement",
            ]
        ].copy()
        slow_fast.columns = [f"slow_fast_{column}" for column in slow_fast.columns]
        features = pd.concat([features, slow_fast], axis=1)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_sequence_arrays(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    state_window: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []
    feature_values = features.to_numpy(dtype=np.float32)
    target_values = targets.to_numpy(dtype=np.float32)
    for pos in range(state_window - 1, len(features)):
        target_row = target_values[pos]
        if np.isnan(target_row).any():
            continue
        sequence = feature_values[pos - state_window + 1 : pos + 1]
        if np.isnan(sequence).all():
            continue
        x_rows.append(sequence)
        y_rows.append(target_row)
        timestamps.append(pd.Timestamp(features.index[pos]))
    if not x_rows:
        return (
            np.zeros((0, state_window, features.shape[1]), dtype=np.float32),
            np.zeros((0, targets.shape[1]), dtype=np.float32),
            [],
        )
    return np.stack(x_rows), np.stack(y_rows), timestamps


class QuantileMLP(nn.Module):
    def __init__(self, input_dim: int, state_window: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * state_window, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class HorizonGatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_window: int,
        horizon_count: int,
        output_channels: int,
        hidden_dim: int,
        dropout: float,
        horizon_embedding_dim: int,
    ) -> None:
        super().__init__()
        flat_dim = input_dim * state_window
        self.horizon_count = horizon_count
        self.output_channels = output_channels
        self.context = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.horizon_embeddings = nn.Parameter(torch.randn(horizon_count, horizon_embedding_dim) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        context = self.context(flat)
        batch = x.shape[0]
        horizon_embeddings = self.horizon_embeddings.unsqueeze(0).expand(batch, -1, -1)
        context_expanded = context.unsqueeze(1).expand(-1, self.horizon_count, -1)
        joint = torch.cat([context_expanded, horizon_embeddings], dim=-1)
        gate = self.gate(joint)
        gated_context = context_expanded * gate
        outputs = self.out(torch.cat([gated_context, horizon_embeddings], dim=-1))
        return outputs.reshape(batch, self.horizon_count * self.output_channels)


class DLinearLike(nn.Module):
    def __init__(self, input_dim: int, state_window: int, output_dim: int, hidden_dim: int, moving_average: int) -> None:
        super().__init__()
        self.state_window = state_window
        self.avg_pool = nn.AvgPool1d(kernel_size=moving_average, stride=1, padding=moving_average // 2)
        self.seasonal = nn.Linear(input_dim * state_window, hidden_dim)
        self.trend = nn.Linear(input_dim * state_window, hidden_dim)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, : x.shape[1], :]
        seasonal = x - trend
        trend_features = self.trend(trend.flatten(start_dim=1))
        seasonal_features = self.seasonal(seasonal.flatten(start_dim=1))
        return self.head(torch.cat([trend_features, seasonal_features], dim=1))


class TiDELike(nn.Module):
    def __init__(self, input_dim: int, state_window: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        flat_dim = input_dim * state_window
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(flat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        encoded = self.encoder(flat)
        return self.decoder(encoded) + self.skip(flat)


class NHITSLike(nn.Module):
    def __init__(self, input_dim: int, state_window: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pools = [1, 2, 4]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim * max(1, state_window // pool), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for pool in self.pools
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for pool, branch in zip(self.pools, self.branches):
            if pool == 1:
                pooled = x
            else:
                usable = (x.shape[1] // pool) * pool
                pooled = x[:, x.shape[1] - usable :, :]
                pooled = pooled.reshape(x.shape[0], usable // pool, pool, x.shape[2]).mean(dim=2)
            outputs.append(branch(pooled.flatten(start_dim=1)))
        return torch.stack(outputs, dim=0).sum(dim=0)


def build_fusion_model(
    spec: dict[str, Any],
    input_dim: int,
    state_window: int,
    horizon_count: int,
    output_channels: int,
) -> nn.Module:
    architecture = spec["architecture"]
    hidden_dim = int(spec.get("hidden_dim", 128))
    dropout = float(spec.get("dropout", 0.0))
    output_dim = horizon_count * output_channels
    if architecture == "quantile_mlp":
        return QuantileMLP(input_dim=input_dim, state_window=state_window, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    if architecture == "horizon_gated_mlp":
        return HorizonGatedMLP(
            input_dim=input_dim,
            state_window=state_window,
            horizon_count=horizon_count,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon_embedding_dim=int(spec.get("horizon_embedding_dim", 24)),
        )
    if architecture == "dlinear":
        return DLinearLike(
            input_dim=input_dim,
            state_window=state_window,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            moving_average=int(spec.get("moving_average", 7)),
        )
    if architecture == "tide_like":
        return TiDELike(input_dim=input_dim, state_window=state_window, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    if architecture == "nhits_like":
        return NHITSLike(input_dim=input_dim, state_window=state_window, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported fusion architecture: {architecture}")


def _output_channels(output_style: str, quantiles: list[float]) -> int:
    if output_style == "direct_quantiles":
        return len(quantiles)
    if output_style == "structured_cwt":
        if [round(q, 2) for q in quantiles] != [0.05, 0.25, 0.50, 0.75, 0.95]:
            raise ValueError("structured_cwt currently supports quantiles [0.05, 0.25, 0.50, 0.75, 0.95].")
        return 4
    raise ValueError(f"Unsupported output style: {output_style}")


def _reshape_params(raw: torch.Tensor, horizon_count: int, output_channels: int) -> torch.Tensor:
    return raw.reshape(raw.shape[0], horizon_count, output_channels)


def _softplus_np(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _params_to_quantiles_torch(
    raw: torch.Tensor,
    horizon_count: int,
    quantiles: list[float],
    output_style: str,
) -> torch.Tensor:
    params = _reshape_params(raw, horizon_count=horizon_count, output_channels=_output_channels(output_style, quantiles))
    if output_style == "direct_quantiles":
        return torch.sort(params, dim=-1).values
    center = params[:, :, 0]
    central_half_width = F.softplus(params[:, :, 1])
    lower_extension = F.softplus(params[:, :, 2])
    upper_extension = F.softplus(params[:, :, 3])
    q25 = center - central_half_width
    q75 = center + central_half_width
    q05 = q25 - lower_extension
    q95 = q75 + upper_extension
    return torch.stack([q05, q25, center, q75, q95], dim=-1)


def _params_to_quantiles_np(
    raw: np.ndarray,
    horizon_count: int,
    quantiles: list[float],
    output_style: str,
) -> np.ndarray:
    params = raw.reshape(raw.shape[0], horizon_count, _output_channels(output_style, quantiles))
    if output_style == "direct_quantiles":
        return np.sort(params, axis=-1)
    center = params[:, :, 0]
    central_half_width = _softplus_np(params[:, :, 1])
    lower_extension = _softplus_np(params[:, :, 2])
    upper_extension = _softplus_np(params[:, :, 3])
    q25 = center - central_half_width
    q75 = center + central_half_width
    q05 = q25 - lower_extension
    q95 = q75 + upper_extension
    return np.stack([q05, q25, center, q75, q95], axis=-1)


def fusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
    horizons: list[int],
    output_style: str,
    training_cfg: dict[str, Any],
) -> torch.Tensor:
    pred = _params_to_quantiles_torch(
        raw=prediction,
        horizon_count=len(horizons),
        quantiles=quantiles,
        output_style=output_style,
    )
    target_expanded = target.unsqueeze(-1)
    quantile_tensor = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, len(quantiles))
    errors = target_expanded - pred
    pinball = torch.maximum(quantile_tensor * errors, (quantile_tensor - 1.0) * errors)
    horizon_weights = torch.ones(len(horizons), dtype=pred.dtype, device=pred.device)
    horizon_weight_cfg = training_cfg.get("horizon_weights", {})
    if isinstance(horizon_weight_cfg, dict):
        for horizon_idx, horizon in enumerate(horizons):
            raw_weight = horizon_weight_cfg.get(str(horizon), horizon_weight_cfg.get(int(horizon), 1.0))
            horizon_weights[horizon_idx] = max(float(raw_weight), 0.0)
        if float(horizon_weights.sum().detach().cpu()) <= 0.0:
            horizon_weights = torch.ones(len(horizons), dtype=pred.dtype, device=pred.device)
        horizon_weights = horizon_weights / horizon_weights.mean().clamp_min(1e-6)
    quantile_weights = torch.ones(len(quantiles), dtype=pred.dtype, device=pred.device)
    if 0.05 in quantiles:
        quantile_weights[quantiles.index(0.05)] = float(training_cfg.get("lower_tail_weight", 1.0))
    pinball = (pinball * horizon_weights.view(1, -1, 1) * quantile_weights.view(1, 1, -1)).mean()
    monotonicity = torch.relu(pred[:, :, :-1] - pred[:, :, 1:]).mean()
    median_index = quantiles.index(0.50)
    width_90 = pred[:, :, -1] - pred[:, :, 0]
    coverage_slack = torch.relu((target - pred[:, :, median_index]).abs() - 0.5 * width_90).mean()
    width_floor = torch.relu(0.01 - width_90).mean()
    return (
        pinball
        + float(training_cfg.get("monotonicity_penalty", 0.0)) * monotonicity
        + float(training_cfg.get("coverage_slack_penalty", 0.0)) * coverage_slack
        + float(training_cfg.get("width_floor_penalty", 0.0)) * width_floor
    )


def _train_fusion_model(
    model: nn.Module,
    train_dataset: FusionDataset,
    val_dataset: FusionDataset,
    training_cfg: dict[str, Any],
    quantiles: list[float],
    horizons: list[int],
    output_style: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty train or validation dataset for fusion training.")

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
    patience = int(training_cfg["patience"])
    stale_epochs = 0

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
    result: dict[int, pd.DataFrame] = {}
    for horizon_idx, horizon in enumerate(horizons):
        frame = pd.DataFrame(
            pred[:, horizon_idx, :],
            index=pd.Index(timestamps, name="timestamp"),
            columns=[quantile_name(q) for q in quantiles],
        )
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.reindex(sorted(frame.columns), axis=1)
        result[horizon] = frame
    return result


def _fit_scaler(train_features: pd.DataFrame) -> FeatureScaler:
    return FeatureScaler.fit(train_features)


def _subset_predictions(predictions: dict[int, pd.DataFrame], index: pd.Index) -> dict[int, pd.DataFrame]:
    return {horizon: frame.loc[index] for horizon, frame in predictions.items()}


def _calibration_context_score(feature_frame: pd.DataFrame) -> pd.Series:
    candidates = [
        column
        for column in feature_frame.columns
        if "disagreement" in column or "conflict" in column or "hazard" in column
    ]
    if not candidates:
        return pd.Series(0.0, index=feature_frame.index)
    return feature_frame[candidates].abs().mean(axis=1).fillna(0.0)


def _fit_additive_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    quantiles: list[float],
) -> dict[int, dict[str, float]]:
    calibrator: dict[int, dict[str, float]] = {}
    for horizon, frame in predictions.items():
        row_actual = actual[f"target_{horizon}d"].to_numpy(dtype=float)
        adjustments: dict[str, float] = {}
        for quantile in quantiles:
            name = quantile_name(quantile)
            residual = row_actual - frame[name].to_numpy(dtype=float)
            residual = residual[np.isfinite(residual)]
            adjustments[name] = float(np.quantile(residual, quantile)) if residual.size else 0.0
        calibrator[horizon] = adjustments
    return calibrator


def _fit_isotonic_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    quantiles: list[float],
) -> dict[int, dict[str, dict[str, list[float]]]]:
    calibrator: dict[int, dict[str, dict[str, list[float]]]] = {}
    for horizon, frame in predictions.items():
        horizon_models: dict[str, dict[str, list[float]]] = {}
        y = actual[f"target_{horizon}d"].to_numpy(dtype=float)
        for quantile in quantiles:
            name = quantile_name(quantile)
            x = frame[name].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 8:
                horizon_models[name] = {
                    "x_thresholds": [0.0, 1.0],
                    "y_thresholds": [0.0, 1.0],
                }
                continue
            x_fit = x[mask]
            y_fit = y[mask]
            order = np.argsort(x_fit)
            x_fit = x_fit[order]
            y_fit = y_fit[order]
            if len(np.unique(x_fit)) < 2:
                value = float(np.median(y_fit)) if y_fit.size else 0.0
                horizon_models[name] = {
                    "x_thresholds": [float(x_fit[0]) if x_fit.size else 0.0, float(x_fit[-1]) if x_fit.size else 1.0],
                    "y_thresholds": [value, value],
                }
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_fit, y_fit)
            horizon_models[name] = {
                "x_thresholds": [float(value) for value in iso.X_thresholds_],
                "y_thresholds": [float(value) for value in iso.y_thresholds_],
            }
        calibrator[horizon] = horizon_models
    return calibrator


def _fit_conformal_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
) -> dict[int, dict[str, float]]:
    calibrator: dict[int, dict[str, float]] = {}
    for horizon, frame in predictions.items():
        y = actual[f"target_{horizon}d"].to_numpy(dtype=float)
        q05 = frame["q05"].to_numpy(dtype=float)
        q25 = frame["q25"].to_numpy(dtype=float)
        q50 = frame["q50"].to_numpy(dtype=float)
        q75 = frame["q75"].to_numpy(dtype=float)
        q95 = frame["q95"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(q05) & np.isfinite(q25) & np.isfinite(q50) & np.isfinite(q75) & np.isfinite(q95)
        if not np.any(mask):
            calibrator[horizon] = {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
            continue
        y = y[mask]
        q05 = q05[mask]
        q25 = q25[mask]
        q50 = q50[mask]
        q75 = q75[mask]
        q95 = q95[mask]
        calibrator[horizon] = {
            "q05": float(-max(np.quantile(q05 - y, 0.90), 0.0)),
            "q25": float(-max(np.quantile(q25 - y, 0.50), 0.0)),
            "q50": float(np.median(y - q50)),
            "q75": float(max(np.quantile(y - q75, 0.50), 0.0)),
            "q95": float(max(np.quantile(y - q95, 0.90), 0.0)),
        }
    return calibrator


def _fit_asymmetric_lower_tail_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
) -> dict[int, dict[str, float]]:
    calibrator: dict[int, dict[str, float]] = {}
    for horizon, frame in predictions.items():
        y = actual[f"target_{horizon}d"].to_numpy(dtype=float)
        q05 = frame["q05"].to_numpy(dtype=float)
        q25 = frame["q25"].to_numpy(dtype=float)
        q50 = frame["q50"].to_numpy(dtype=float)
        q75 = frame["q75"].to_numpy(dtype=float)
        q95 = frame["q95"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(q05) & np.isfinite(q25) & np.isfinite(q50) & np.isfinite(q75) & np.isfinite(q95)
        if not np.any(mask):
            calibrator[horizon] = {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
            continue
        y = y[mask]
        q05 = q05[mask]
        q25 = q25[mask]
        q50 = q50[mask]
        q75 = q75[mask]
        q95 = q95[mask]
        lower_05 = float(-max(np.quantile(q05 - y, 0.90), 0.0))
        lower_25 = float(-max(np.quantile(q25 - y, 0.75), 0.0))
        median_shift = float(np.median(y - q50))
        upper_75 = float(max(np.quantile(y - q75, 0.35), 0.0))
        upper_95 = float(max(np.quantile(y - q95, 0.50), 0.0))
        calibrator[horizon] = {
            "q05": lower_05,
            "q25": 0.65 * lower_25,
            "q50": 0.25 * median_shift,
            "q75": 0.25 * upper_75,
            "q95": 0.10 * upper_95,
        }
    return calibrator


def _fit_disagreement_width_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    context_score: pd.Series,
) -> dict[str, Any]:
    threshold = float(context_score.median()) if len(context_score) else 0.0
    calibrator: dict[str, Any] = {"threshold": threshold, "low": {}, "high": {}}
    for label, index in {
        "low": context_score.index[context_score <= threshold],
        "high": context_score.index[context_score > threshold],
    }.items():
        for horizon, frame in predictions.items():
            if len(index) == 0:
                calibrator[label][horizon] = {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
                continue
            subset = frame.loc[index]
            y = actual.loc[index, f"target_{horizon}d"].to_numpy(dtype=float)
            q05 = subset["q05"].to_numpy(dtype=float)
            q25 = subset["q25"].to_numpy(dtype=float)
            q50 = subset["q50"].to_numpy(dtype=float)
            q75 = subset["q75"].to_numpy(dtype=float)
            q95 = subset["q95"].to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(q05) & np.isfinite(q25) & np.isfinite(q50) & np.isfinite(q75) & np.isfinite(q95)
            if not np.any(mask):
                calibrator[label][horizon] = {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
                continue
            y = y[mask]
            q05 = q05[mask]
            q25 = q25[mask]
            q50 = q50[mask]
            q75 = q75[mask]
            q95 = q95[mask]
            miss_width = np.maximum.reduce([q05 - y, y - q95, np.zeros_like(y)])
            extra_width = float(np.quantile(miss_width, 0.90))
            calibrator[label][horizon] = {
                "q05": -extra_width,
                "q25": -0.50 * extra_width,
                "q50": float(np.median(y - q50)),
                "q75": 0.50 * extra_width,
                "q95": extra_width,
            }
    return calibrator


def _fit_conditional_conformal_adjustments(
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    context_score: pd.Series,
) -> dict[str, Any]:
    threshold = float(context_score.median()) if len(context_score) else 0.0
    low_index = context_score.index[context_score <= threshold]
    high_index = context_score.index[context_score > threshold]
    if len(low_index) == 0 or len(high_index) == 0:
        return {
            "threshold": threshold,
            "low": _fit_conformal_adjustments(actual, predictions),
            "high": _fit_conformal_adjustments(actual, predictions),
        }
    return {
        "threshold": threshold,
        "low": _fit_conformal_adjustments(actual.loc[low_index], _subset_predictions(predictions, low_index)),
        "high": _fit_conformal_adjustments(actual.loc[high_index], _subset_predictions(predictions, high_index)),
    }


def fit_calibrator(
    method: str,
    actual: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    quantiles: list[float],
    context_score: pd.Series,
) -> dict[str, Any]:
    if method == "raw":
        return {"method": method, "adjustments": {}}
    if method == "additive_quantile":
        return {"method": method, "adjustments": _fit_additive_adjustments(actual, predictions, quantiles)}
    if method == "isotonic":
        return {"method": method, "adjustments": _fit_isotonic_adjustments(actual, predictions, quantiles)}
    if method == "conformal":
        return {"method": method, "adjustments": _fit_conformal_adjustments(actual, predictions)}
    if method == "asymmetric_lower_tail":
        return {"method": method, "adjustments": _fit_asymmetric_lower_tail_adjustments(actual, predictions)}
    if method == "disagreement_width":
        return {
            "method": method,
            "adjustments": _fit_disagreement_width_adjustments(actual, predictions, context_score=context_score),
        }
    if method == "disagreement_conformal":
        return {
            "method": method,
            "adjustments": _fit_conditional_conformal_adjustments(actual, predictions, context_score=context_score),
        }
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(
    predictions: dict[int, pd.DataFrame],
    calibrator: dict[str, Any],
    quantiles: list[float],
    context_score: pd.Series,
) -> dict[int, pd.DataFrame]:
    method = calibrator["method"]
    if method == "raw":
        return {horizon: frame.copy() for horizon, frame in predictions.items()}
    calibrated: dict[int, pd.DataFrame] = {}

    if method == "isotonic":
        adjustments = calibrator["adjustments"]
        for horizon, frame in predictions.items():
            adjusted = frame.copy()
            for quantile in quantiles:
                name = quantile_name(quantile)
                mapping = adjustments.get(horizon, {}).get(name, None)
                if mapping is None:
                    continue
                x_thresholds = np.asarray(mapping["x_thresholds"], dtype=float)
                y_thresholds = np.asarray(mapping["y_thresholds"], dtype=float)
                adjusted[name] = np.interp(
                    adjusted[name].to_numpy(dtype=float),
                    x_thresholds,
                    y_thresholds,
                    left=float(y_thresholds[0]),
                    right=float(y_thresholds[-1]),
                )
            adjusted.loc[:, [quantile_name(q) for q in quantiles]] = np.sort(
                adjusted[[quantile_name(q) for q in quantiles]].to_numpy(dtype=float),
                axis=1,
            )
            calibrated[horizon] = adjusted
        return calibrated

    if method in {"additive_quantile", "conformal", "asymmetric_lower_tail"}:
        adjustments = calibrator["adjustments"]
        for horizon, frame in predictions.items():
            adjusted = frame.copy()
            for quantile in quantiles:
                name = quantile_name(quantile)
                adjusted[name] = adjusted[name] + float(adjustments.get(horizon, {}).get(name, 0.0))
            adjusted.loc[:, [quantile_name(q) for q in quantiles]] = np.sort(
                adjusted[[quantile_name(q) for q in quantiles]].to_numpy(dtype=float),
                axis=1,
            )
            calibrated[horizon] = adjusted
        return calibrated

    if method == "disagreement_width":
        threshold = float(calibrator["adjustments"]["threshold"])
        low_index = context_score.index[context_score <= threshold]
        high_index = context_score.index[context_score > threshold]
        for horizon, frame in predictions.items():
            adjusted = frame.copy()
            for label, index in [("low", low_index), ("high", high_index)]:
                if len(index) == 0:
                    continue
                horizon_adjustment = calibrator["adjustments"][label].get(horizon, {})
                for quantile in quantiles:
                    name = quantile_name(quantile)
                    adjusted.loc[index, name] = adjusted.loc[index, name] + float(horizon_adjustment.get(name, 0.0))
            adjusted.loc[:, [quantile_name(q) for q in quantiles]] = np.sort(
                adjusted[[quantile_name(q) for q in quantiles]].to_numpy(dtype=float),
                axis=1,
            )
            calibrated[horizon] = adjusted
        return calibrated

    threshold = float(calibrator["adjustments"]["threshold"])
    low_index = context_score.index[context_score <= threshold]
    high_index = context_score.index[context_score > threshold]
    for horizon, frame in predictions.items():
        adjusted = frame.copy()
        for label, index in [("low", low_index), ("high", high_index)]:
            if len(index) == 0:
                continue
            horizon_adjustment = calibrator["adjustments"][label].get(horizon, {})
            for quantile in quantiles:
                name = quantile_name(quantile)
                adjusted.loc[index, name] = adjusted.loc[index, name] + float(horizon_adjustment.get(name, 0.0))
        adjusted.loc[:, [quantile_name(q) for q in quantiles]] = np.sort(
            adjusted[[quantile_name(q) for q in quantiles]].to_numpy(dtype=float),
            axis=1,
        )
        calibrated[horizon] = adjusted
    return calibrated


def _partition_training_frame(length: int, state_window: int) -> dict[str, int]:
    minimum_block = max(state_window + 5, 24)
    if length < minimum_block * 3:
        train_end = max(state_window + 10, int(length * 0.75))
        train_end = min(train_end, max(length - 5, state_window + 1))
        return {
            "train_end": train_end,
            "model_val_end": length,
            "calibration_start": train_end,
        }

    train_end = max(minimum_block, int(length * 0.65))
    model_val_end = max(train_end + 10, int(length * 0.82))
    model_val_end = min(model_val_end, length - minimum_block)
    return {
        "train_end": train_end,
        "model_val_end": model_val_end,
        "calibration_start": model_val_end,
    }


def train_fusion_candidate(
    feature_train: pd.DataFrame,
    feature_test: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    horizon_set_name: str,
    horizons: list[int],
    variant: str,
    communication_channels: list[str],
    spec: dict[str, Any],
    output_style: str,
    config: dict[str, Any],
    artifact_dir: Path,
    device: str,
) -> dict[str, Any]:
    state_window = int(config["fusion"]["state_window"])
    quantiles = list(config["fusion"]["quantiles"])
    training_cfg = config["fusion"]["training"]
    training_partitions = _partition_training_frame(len(feature_train), state_window)

    train_end = int(training_partitions["train_end"])
    model_val_end = int(training_partitions["model_val_end"])
    calibration_start = int(training_partitions["calibration_start"])

    scaler = _fit_scaler(feature_train.iloc[:train_end])
    target_scaler = TargetScaler.fit(target_train.iloc[:train_end])
    train_scaled = scaler.transform(feature_train)
    test_scaled = scaler.transform(feature_test)
    train_target_scaled = target_scaler.transform(target_train)
    test_target_scaled = target_scaler.transform(target_test)

    train_x, train_y, _train_timestamps = _build_sequence_arrays(
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
    test_x, _test_y, test_timestamps = _build_sequence_arrays(test_scaled, test_target_scaled, state_window)

    if len(calibration_timestamps) == 0:
        calibration_x = model_val_x
        calibration_y = model_val_y
        calibration_timestamps = model_val_timestamps

    train_dataset = FusionDataset(train_x, train_y)
    val_dataset = FusionDataset(model_val_x, model_val_y)
    model = build_fusion_model(
        spec=spec,
        input_dim=feature_train.shape[1],
        state_window=state_window,
        horizon_count=len(horizons),
        output_channels=_output_channels(output_style, quantiles),
    )
    best_state, history = _train_fusion_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_cfg=training_cfg,
        quantiles=quantiles,
        horizons=horizons,
        output_style=output_style,
        device=device,
    )
    model.load_state_dict(best_state)

    model_val_predictions = _prediction_frames(
        model=model,
        x=model_val_x,
        timestamps=model_val_timestamps,
        horizons=horizons,
        quantiles=quantiles,
        output_style=output_style,
        target_scaler=target_scaler,
        device=device,
    )
    calibration_predictions = _prediction_frames(
        model=model,
        x=calibration_x,
        timestamps=calibration_timestamps,
        horizons=horizons,
        quantiles=quantiles,
        output_style=output_style,
        target_scaler=target_scaler,
        device=device,
    )
    test_predictions = _prediction_frames(
        model=model,
        x=test_x,
        timestamps=test_timestamps,
        horizons=horizons,
        quantiles=quantiles,
        output_style=output_style,
        target_scaler=target_scaler,
        device=device,
    )

    model_val_actual = target_train.loc[model_val_timestamps]
    calibration_actual = target_train.loc[calibration_timestamps]
    test_actual = target_test.loc[test_timestamps]

    calibration_feature_block = feature_train.loc[calibration_timestamps] if calibration_timestamps else feature_train.iloc[0:0]
    calibration_context = _calibration_context_score(calibration_feature_block)
    split_mid = max(1, len(calibration_timestamps) // 2)
    fit_index = pd.Index(calibration_timestamps[:split_mid], name="timestamp")
    eval_index = pd.Index(calibration_timestamps[split_mid:], name="timestamp")
    if len(eval_index) == 0:
        eval_index = fit_index

    calibration_fit_actual = calibration_actual.loc[fit_index]
    calibration_eval_actual = calibration_actual.loc[eval_index]
    calibration_fit_predictions = _subset_predictions(calibration_predictions, fit_index)
    calibration_eval_predictions = _subset_predictions(calibration_predictions, eval_index)
    calibration_fit_context = calibration_context.loc[fit_index]
    calibration_eval_context = calibration_context.loc[eval_index]
    test_context = _calibration_context_score(feature_test.loc[test_timestamps]) if test_timestamps else pd.Series(dtype=float)

    model_val_metrics = summarize_quantile_metrics(model_val_actual, model_val_predictions, quantiles)

    calibration_candidates = []
    for method in config["fusion"]["calibration_methods"]:
        calibrator = fit_calibrator(
            method=method,
            actual=calibration_fit_actual,
            predictions=calibration_fit_predictions,
            quantiles=quantiles,
            context_score=calibration_fit_context,
        )
        calibrated_eval_predictions = apply_calibrator(
            predictions=calibration_eval_predictions,
            calibrator=calibrator,
            quantiles=quantiles,
            context_score=calibration_eval_context,
        )
        calibrated_eval_metrics = summarize_quantile_metrics(calibration_eval_actual, calibrated_eval_predictions, quantiles)
        calibration_candidates.append(
            {
                "method": method,
                "calibrator": calibrator,
                "metrics": calibrated_eval_metrics,
            }
        )
    calibration_candidates.sort(key=lambda item: item["metrics"]["selection_score"])
    best_calibration = calibration_candidates[0]

    calibrated_test_predictions = apply_calibrator(
        predictions=test_predictions,
        calibrator=best_calibration["calibrator"],
        quantiles=quantiles,
        context_score=test_context,
    )
    test_metrics = summarize_quantile_metrics(test_actual, calibrated_test_predictions, quantiles)

    candidate_dir = ensure_dir(
        artifact_dir
        / f"{horizon_set_name}__{variant}__{'-'.join(communication_channels) if communication_channels else 'base'}__{output_style}__{spec['name']}"
    )
    torch.save(best_state, candidate_dir / "checkpoint.pt")
    write_json(candidate_dir / "history.json", history)
    write_json(candidate_dir / "model_val_metrics.json", model_val_metrics)
    write_json(candidate_dir / "calibration_selection.json", {item["method"]: item["metrics"] for item in calibration_candidates})
    write_json(candidate_dir / "selected_calibrator.json", best_calibration["calibrator"])
    write_json(candidate_dir / "test_metrics.json", test_metrics)
    write_json(candidate_dir / "scaler.json", scaler.to_dict())
    write_json(candidate_dir / "target_scaler.json", target_scaler.to_dict())
    for horizon, frame in calibrated_test_predictions.items():
        frame.to_csv(candidate_dir / f"test_predictions_{horizon}d.csv")

    return {
        "horizon_set_name": horizon_set_name,
        "horizons": horizons,
        "variant": variant,
        "communication_channels": communication_channels,
        "spec": spec,
        "output_style": output_style,
        "history": history,
        "model_val_metrics": model_val_metrics,
        "selection_metrics": best_calibration["metrics"],
        "calibration_method": best_calibration["method"],
        "calibration_metrics": {item["method"]: item["metrics"] for item in calibration_candidates},
        "test_metrics": test_metrics,
        "test_predictions": calibrated_test_predictions,
        "test_actual": test_actual,
        "artifact_dir": str(candidate_dir),
    }


def search_fusion_models(
    train_states_by_bucket: dict[str, pd.DataFrame],
    test_states_by_bucket: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    config: dict[str, Any],
    artifact_root: Path,
    device: str,
) -> dict[str, Any]:
    artifact_dir = ensure_dir(artifact_root / "fusion")
    fusion_cfg = config["fusion"]
    shared_train_index = sorted(set.intersection(*(set(frame.index) for frame in train_states_by_bucket.values())))
    shared_test_index = sorted(set.intersection(*(set(frame.index) for frame in test_states_by_bucket.values())))
    train_states = {name: frame.loc[shared_train_index] for name, frame in train_states_by_bucket.items()}
    test_states = {name: frame.loc[shared_test_index] for name, frame in test_states_by_bucket.items()}

    results = []
    for horizon_set_name, horizons in fusion_cfg["horizon_sets"].items():
        train_target = targets.loc[shared_train_index, [f"target_{h}d" for h in horizons]].dropna()
        test_target = targets.loc[shared_test_index, [f"target_{h}d" for h in horizons]].dropna()
        common_train_index = sorted(set(train_target.index).intersection(*(set(frame.index) for frame in train_states.values())))
        common_test_index = sorted(set(test_target.index).intersection(*(set(frame.index) for frame in test_states.values())))
        horizon_train_states = {name: frame.loc[common_train_index] for name, frame in train_states.items()}
        horizon_test_states = {name: frame.loc[common_test_index] for name, frame in test_states.items()}
        horizon_train_target = train_target.loc[common_train_index]
        horizon_test_target = test_target.loc[common_test_index]

        for variant in fusion_cfg["variants"]:
            communication_variants = [[]] if variant != "G" else fusion_cfg["communication_variants"]
            for communication_channels in communication_variants:
                feature_train = build_fusion_feature_frame(
                    horizon_train_states,
                    variant=variant,
                    communication_channels=communication_channels,
                )
                feature_test = build_fusion_feature_frame(
                    horizon_test_states,
                    variant=variant,
                    communication_channels=communication_channels,
                )
                for output_style in fusion_cfg["output_styles"]:
                    for spec in fusion_cfg["architectures"]:
                        result = train_fusion_candidate(
                            feature_train=feature_train,
                            feature_test=feature_test,
                            target_train=horizon_train_target,
                            target_test=horizon_test_target,
                            horizon_set_name=horizon_set_name,
                            horizons=horizons,
                            variant=variant,
                            communication_channels=communication_channels,
                            spec=spec,
                            output_style=output_style,
                            config=config,
                            artifact_dir=artifact_dir,
                            device=device,
                        )
                        results.append(result)

    results.sort(key=lambda item: item["selection_metrics"]["selection_score"])
    leaderboard = [
        {
            "horizon_set": item["horizon_set_name"],
            "variant": item["variant"],
            "communication_channels": item["communication_channels"],
            "output_style": item["output_style"],
            "architecture": item["spec"]["name"],
            "calibration_method": item["calibration_method"],
            "selection_score": item["selection_metrics"]["selection_score"],
            "val_average_wis": item["selection_metrics"]["average_wis"],
            "val_average_calibration_gap": item["selection_metrics"]["average_calibration_gap"],
            "test_average_pinball": item["test_metrics"]["average_pinball"],
            "test_average_wis": item["test_metrics"]["average_wis"],
        }
        for item in results
    ]
    write_json(artifact_dir / "leaderboard.json", leaderboard)
    return {
        "best": results[0],
        "leaderboard": leaderboard,
    }
