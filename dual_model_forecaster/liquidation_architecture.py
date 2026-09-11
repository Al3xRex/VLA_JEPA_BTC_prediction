from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dual_model_forecaster.jepa.losses import total_jepa_loss
from dual_model_forecaster.jepa.models import SpecialistJEPA
from dual_model_forecaster.metrics import specialist_regression_metrics
from dual_model_forecaster.specialists import FeatureScaler, SequenceDataset, build_specialist_model
from dual_model_forecaster.utils import ensure_dir, set_global_seed, sigmoid, write_json


LIQUIDATION_SIGNAL_PATH = Path("liquidations/data/directional_volatility_potential.csv")
DEFAULT_OUTPUT_DIR = Path("reports/liquidation_architecture")

PRICE_JUMP_TARGET_FIELDS = (
    "btc_forward_jump_1d_pct",
    "btc_forward_jump_3d_pct",
    "btc_forward_jump_7d_pct",
    "btc_forward_abs_jump_7d_pct",
    "btc_forward_jump_direction",
)
JUMP_TARGET_COLUMNS = (
    "btc_forward_jump_1d_pct",
    "btc_forward_jump_3d_pct",
    "btc_forward_jump_7d_pct",
)
LIQUIDATION_SEMANTIC_COLUMNS = (
    "liquidation_cascade_pressure",
    "liquidation_directional_pressure",
    "leverage_fuel",
    "liquidation_volatility_instability",
    "cascade_asymmetry",
    "liquidation_confidence",
)

DEFAULT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "linear_lb24", "architecture": "linear", "lookback": 24},
    {"name": "mlp_lb36", "architecture": "mlp", "lookback": 36, "hidden_dim": 64, "dropout": 0.10},
    {"name": "linear_lb45", "architecture": "linear", "lookback": 45},
    {"name": "mlp_lb60", "architecture": "mlp", "lookback": 60, "hidden_dim": 80, "dropout": 0.10},
    {"name": "gru_lb75", "architecture": "gru", "lookback": 75, "hidden_dim": 48, "num_layers": 1, "dropout": 0.10},
    {
        "name": "tcn_lb75",
        "architecture": "tcn",
        "lookback": 75,
        "hidden_dim": 40,
        "kernel_size": 3,
        "num_layers": 3,
        "dropout": 0.10,
    },
    {
        "name": "patchtst_lb96",
        "architecture": "patchtst_like",
        "lookback": 96,
        "patch_length": 8,
        "hidden_dim": 48,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.10,
    },
    {
        "name": "category_attention_lb96",
        "architecture": "category_attention",
        "lookback": 96,
        "hidden_dim": 64,
        "num_heads": 4,
        "dropout": 0.10,
    },
)

DEFAULT_TRAINING_CONFIG = {
    "batch_size": 128,
    "max_epochs": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 2,
}

JUMP_EVENT_THRESHOLDS = {
    "btc_forward_jump_1d_pct": 1.5,
    "btc_forward_jump_3d_pct": 2.5,
    "btc_forward_jump_7d_pct": 4.0,
}

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "funding": ("funding", "predicted_funding"),
    "open_interest": ("open_interest", "futures_oi", "oi_", "leverage", "anchor"),
    "liquidation": ("liquidation", "liquidated", "heatmap"),
    "cascade_space": ("cascade", "wall", "area", "pressure"),
    "volatility": ("volatility", "std", "trigger", "regime"),
    "coverage": ("confidence", "missing", "contract_count", "venue_count"),
}


@dataclass
class ArchitectureSplit:
    train_end_pos: int
    val_end_pos: int
    test_end_pos: int


class RobustTargetScaler:
    def __init__(self, median: pd.Series, iqr: pd.Series) -> None:
        self.median = median.astype(float).fillna(0.0)
        self.iqr = iqr.astype(float).replace(0.0, 1.0).fillna(1.0)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RobustTargetScaler":
        median = frame.median(axis=0)
        q25 = frame.quantile(0.25, axis=0)
        q75 = frame.quantile(0.75, axis=0)
        return cls(median=median, iqr=q75 - q25)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return ((frame - self.median) / self.iqr).replace([np.inf, -np.inf], np.nan)

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame * self.iqr + self.median


class CategoryAttentionJumpModel(nn.Module):
    def __init__(
        self,
        *,
        category_indices: dict[str, list[int]],
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.category_names = [name for name, indices in category_indices.items() if indices]
        self.category_indices = [category_indices[name] for name in self.category_names]
        if not self.category_indices:
            raise ValueError("Category attention model requires at least one non-empty feature category.")
        self.projections = nn.ModuleList([nn.Linear(len(indices), hidden_dim) for indices in self.category_indices])
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in self.category_indices
            ]
        )
        self.queries = nn.Parameter(torch.randn(len(self.category_indices), 1, hidden_dim) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(self.category_indices)),
            nn.Linear(hidden_dim * len(self.category_indices), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0])
        category_contexts: list[torch.Tensor] = []
        for idx, indices in enumerate(self.category_indices):
            tokens = F.gelu(self.projections[idx](x[:, :, indices]))
            query = self.queries[idx].unsqueeze(0).expand(batch_size, -1, -1)
            context, _weights = self.attention[idx](query, tokens, tokens, need_weights=False)
            category_contexts.append(context.squeeze(1))
        return self.head(torch.cat(category_contexts, dim=-1))


class LiquidationJEPADataset(Dataset):
    def __init__(self, features: pd.DataFrame, *, context_length: int, horizons: tuple[int, ...]) -> None:
        self.features = features.astype(float)
        self.context_length = int(context_length)
        self.horizons = tuple(int(h) for h in horizons)
        self.index = pd.DatetimeIndex(features.index)
        self.sample_index: list[tuple[int, int, int, int]] = []
        for pos in range(self.context_length - 1, len(self.features)):
            for horizon in self.horizons:
                target_start = pos + 1
                target_end = min(pos + horizon, len(self.features) - 1)
                if target_start > target_end:
                    continue
                self.sample_index.append((pos, horizon, target_start, target_end))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pos, horizon, target_start, target_end = self.sample_index[item]
        context = self.features.iloc[pos - self.context_length + 1 : pos + 1].to_numpy(dtype=np.float32)
        target = self.features.iloc[target_start : target_end + 1].to_numpy(dtype=np.float32)
        return {
            "context_features": context,
            "target_features": target,
            "horizon": horizon,
            "as_of_timestamp": pd.Timestamp(self.index[pos]),
        }


def _jepa_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    context = torch.tensor(np.stack([row["context_features"] for row in batch]), dtype=torch.float32)
    max_target = max(int(row["target_features"].shape[0]) for row in batch)
    target = torch.zeros((len(batch), max_target, context.shape[-1]), dtype=torch.float32)
    target_mask = torch.zeros((len(batch), max_target), dtype=torch.bool)
    for idx, row in enumerate(batch):
        values = torch.tensor(row["target_features"], dtype=torch.float32)
        target[idx, : values.shape[0], :] = values
        target_mask[idx, : values.shape[0]] = True
    return {
        "context_features": context,
        "target_features": target,
        "target_mask": target_mask,
        "horizon": torch.tensor([int(row["horizon"]) for row in batch], dtype=torch.long),
        "as_of_timestamp": [row["as_of_timestamp"] for row in batch],
    }


def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_liquidation_signal_frame(path: str | Path = LIQUIDATION_SIGNAL_PATH, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"Liquidation signal file must contain a date column: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    frame = frame.sort_values("date").set_index("date")
    frame.index.name = "timestamp"
    if max_rows is not None and int(max_rows) > 0:
        frame = frame.tail(int(max_rows))
    return frame


def _causal_robust_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    median = clean.rolling(window, min_periods=min_periods).median()
    mad = (clean - median).abs().rolling(window, min_periods=min_periods).median()
    scale = (1.4826 * mad).replace(0.0, np.nan)
    return ((clean - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _score_fraction(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(0.0)
    return (values / 100.0).clip(-1.0, 1.0)


def _positive_fraction(frame: pd.DataFrame, column: str) -> pd.Series:
    return _score_fraction(frame, column).clip(0.0, 1.0)


def build_liquidation_feature_frame(signal_frame: pd.DataFrame) -> pd.DataFrame:
    source = signal_frame.copy()
    non_target = source.drop(columns=[column for column in PRICE_JUMP_TARGET_FIELDS if column in source.columns])
    categorical_columns = [
        column
        for column in (
            "directional_bias",
            "volatility_potential",
            "cascade_threshold_state",
            "regime",
            "heatmap_daily_liquidation_source",
            "liquidation_source",
        )
        if column in non_target.columns
    ]
    numeric = _clean_numeric(non_target.drop(columns=categorical_columns, errors="ignore"))
    missing_flags = numeric.isna().astype(float)
    missing_flags = missing_flags.loc[:, missing_flags.mean(axis=0) > 0.0].add_suffix("_missing")
    numeric_filled = numeric.ffill().fillna(0.0)

    features = numeric_filled.copy()
    if categorical_columns:
        categoricals = non_target[categorical_columns].fillna("").astype(str)
        dummies = pd.get_dummies(categoricals, prefix=categorical_columns, dtype=float)
        features = pd.concat([features, dummies], axis=1)
    if not missing_flags.empty:
        features = pd.concat([features, missing_flags], axis=1)

    enrichment_columns = [
        column
        for column in numeric_filled.columns
        if numeric_filled[column].nunique(dropna=True) > 2
    ]
    enrichment_frames: list[pd.DataFrame] = []
    for column in enrichment_columns:
        values = numeric_filled[column].astype(float)
        enrichment_frames.append(
            pd.DataFrame(
                {
                    f"{column}_roll7_std": values.rolling(7, min_periods=3).std().fillna(0.0),
                    f"{column}_roll30_std": values.rolling(30, min_periods=10).std().fillna(0.0),
                    f"{column}_signalboost": np.tanh(
                        _causal_robust_zscore(values, window=180, min_periods=30) / 2.0
                    ),
                },
                index=numeric_filled.index,
            )
        )

    def feature(column: str) -> pd.Series:
        if column in numeric_filled.columns:
            return numeric_filled[column].astype(float)
        return pd.Series(0.0, index=numeric_filled.index)

    def percent_feature(column: str) -> pd.Series:
        return (feature(column) / 100.0).clip(-1.0, 1.0)

    directional_bias = percent_feature("directional_bias_score")
    source_dominance_impulse = (-feature("source_liquidation_dominance_pct")).clip(-100.0, 100.0)
    heatmap_dominance_impulse = (-feature("heatmap_liquidation_dominance_pct")).clip(-100.0, 100.0)
    heatmap_daily_dominance_impulse = (-feature("heatmap_daily_liquidation_dominance_pct")).clip(-100.0, 100.0)
    observed_dominance_impulse = (-feature("heatmap_daily_observed_liquidation_dominance_pct")).clip(-100.0, 100.0)
    estimated_dominance_impulse = (-feature("heatmap_daily_estimated_liquidation_dominance_pct")).clip(-100.0, 100.0)
    interest_long = feature("heatmap_liquidated_interest_long_usd").where(
        feature("heatmap_liquidated_interest_long_usd") > 0.0,
        feature("heatmap_daily_liquidated_interest_long_usd"),
    )
    interest_short = feature("heatmap_liquidated_interest_short_usd").where(
        feature("heatmap_liquidated_interest_short_usd") > 0.0,
        feature("heatmap_daily_liquidated_interest_short_usd"),
    )
    interest_total = (interest_long + interest_short).replace(0.0, np.nan)
    long_short_liquidation_impulse = ((interest_short - interest_long) / interest_total * 100.0).fillna(0.0).clip(
        -100.0,
        100.0,
    )
    source_impulse = (source_dominance_impulse / 100.0).clip(-1.0, 1.0)
    heatmap_impulse = (heatmap_dominance_impulse / 100.0).clip(-1.0, 1.0)
    heatmap_daily_impulse = (heatmap_daily_dominance_impulse / 100.0).clip(-1.0, 1.0)
    observed_impulse = (observed_dominance_impulse / 100.0).clip(-1.0, 1.0)
    estimated_impulse = (estimated_dominance_impulse / 100.0).clip(-1.0, 1.0)
    long_short_impulse = (long_short_liquidation_impulse / 100.0).clip(-1.0, 1.0)
    oi_area_asymmetry = feature("oi_area_upside_amount_score") - feature("oi_area_downside_amount_score")
    oi_area_direction = (oi_area_asymmetry / 100.0).clip(-1.0, 1.0)

    interaction_features = pd.DataFrame(
        {
            "liquidation_direction_x_potential": (
                feature("directional_bias_score") * feature("volatility_potential_score") / 100.0
            ),
            "cascade_amount_x_confidence": (
                feature("cascade_closeness_amount_score") * feature("signal_confidence_score") / 100.0
            ),
            "funding_x_oi_pressure": (
                feature("funding_crowding_score") * feature("new_oi_addition_score") / 100.0
            ),
            "fuel_x_trigger": feature("leverage_fuel_score") * feature("volatility_trigger_score") / 100.0,
            "heatmap_liquidation_imbalance": (
                feature("heatmap_liquidated_interest_short_usd") - feature("heatmap_liquidated_interest_long_usd")
            )
            / 1_000_000_000.0,
            "wall_amount_asymmetry": (
                feature("liquidation_wall_upside_amount_score") - feature("liquidation_wall_downside_amount_score")
            ),
            "oi_area_amount_asymmetry": (
                oi_area_asymmetry
            ),
            "source_liquidation_dominance_impulse_score": source_dominance_impulse,
            "heatmap_liquidation_dominance_impulse_score": heatmap_dominance_impulse,
            "heatmap_long_short_liquidation_impulse_score": long_short_liquidation_impulse,
            "source_heatmap_dominance_agreement": source_impulse * heatmap_impulse,
            "source_heatmap_dominance_gap": (source_dominance_impulse - heatmap_dominance_impulse).abs(),
            "source_bias_confirmation": directional_bias * source_impulse,
            "heatmap_bias_confirmation": directional_bias * heatmap_impulse,
            "long_short_bias_confirmation": directional_bias * long_short_impulse,
            "observed_estimated_dominance_agreement": observed_impulse * estimated_impulse,
            "observed_estimated_dominance_gap": (observed_dominance_impulse - estimated_dominance_impulse).abs(),
            "oi_area_bias_confirmation": directional_bias * oi_area_direction,
            "oi_area_source_dominance_confirmation": oi_area_direction * source_impulse,
            "oi_area_heatmap_dominance_confirmation": oi_area_direction * heatmap_impulse,
            "funding_oi_area_direction_confirmation": percent_feature("funding_crowding_score") * oi_area_direction,
            "heatmap_daily_source_dominance_agreement": heatmap_daily_impulse * source_impulse,
        },
        index=numeric_filled.index,
    )
    features = pd.concat([features, *enrichment_frames, interaction_features], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated()]
    return features.astype(float)


def build_liquidation_semantic_targets(signal_frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _clean_numeric(signal_frame).ffill().fillna(0.0)
    out = pd.DataFrame(index=signal_frame.index)
    cascade = _positive_fraction(numeric, "cascade_closeness_amount_score")
    potential = _positive_fraction(numeric, "volatility_potential_score")
    activation = _score_fraction(numeric, "cascade_activation_score").abs()
    out["liquidation_cascade_pressure"] = (0.45 * cascade + 0.35 * potential + 0.20 * activation).clip(0.0, 1.0)

    directional = (
        0.45 * _score_fraction(numeric, "directional_bias_score")
        + 0.25 * _score_fraction(numeric, "cascade_closeness_amount_direction_score")
        + 0.15 * _score_fraction(numeric, "liquidation_pressure_net_score")
        + 0.15 * _score_fraction(numeric, "liquidation_impulse_score")
    )
    out["liquidation_directional_pressure"] = np.tanh(directional * 1.4).clip(-1.0, 1.0)

    fuel = (
        0.35 * _positive_fraction(numeric, "leverage_fuel_score")
        + 0.25 * _positive_fraction(numeric, "new_oi_addition_score")
        + 0.20 * _positive_fraction(numeric, "oi_zone_reactivation_score")
        + 0.20
        * (
            _positive_fraction(numeric, "oi_area_downside_amount_score")
            + _positive_fraction(numeric, "oi_area_upside_amount_score")
        )
        / 2.0
    )
    out["leverage_fuel"] = fuel.clip(0.0, 1.0)

    instability = (
        0.40 * _positive_fraction(numeric, "realized_volatility_regime_score")
        + 0.30 * _positive_fraction(numeric, "volatility_trigger_score")
        + 0.30 * _positive_fraction(numeric, "volatility_potential_score")
    )
    out["liquidation_volatility_instability"] = instability.clip(0.0, 1.0)

    wall_asymmetry = (
        _score_fraction(numeric, "liquidation_wall_upside_amount_score")
        - _score_fraction(numeric, "liquidation_wall_downside_amount_score")
    )
    oi_area_asymmetry = _score_fraction(numeric, "oi_area_upside_amount_score") - _score_fraction(
        numeric,
        "oi_area_downside_amount_score",
    )
    asymmetry = (
        0.45 * out["liquidation_directional_pressure"]
        + 0.25 * (0.60 * wall_asymmetry + 0.40 * oi_area_asymmetry)
        + 0.15 * _score_fraction(numeric, "funding_crowding_score")
        + 0.15 * _score_fraction(numeric, "oi_anchor_direction_score")
    )
    out["cascade_asymmetry"] = np.tanh(asymmetry * 1.2).clip(-1.0, 1.0)
    out["liquidation_confidence"] = _positive_fraction(numeric, "signal_confidence_score").clip(0.0, 1.0)
    return out.loc[:, list(LIQUIDATION_SEMANTIC_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_jump_targets(signal_frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in JUMP_TARGET_COLUMNS if column not in signal_frame.columns]
    if missing:
        raise ValueError(f"Missing jump target columns: {missing}")
    return _clean_numeric(signal_frame.loc[:, list(JUMP_TARGET_COLUMNS)])


def _build_split(index: pd.Index, *, validation_days: int, test_days: int, min_train_days: int) -> ArchitectureSplit:
    n_rows = len(index)
    if n_rows < min_train_days + validation_days + test_days:
        min_train_days = max(128, n_rows - validation_days - test_days)
    train_end = int(min_train_days)
    val_end = min(n_rows - test_days, train_end + validation_days)
    if val_end <= train_end:
        raise ValueError("Not enough rows for architecture validation split.")
    test_end = n_rows
    if test_end <= val_end:
        raise ValueError("Not enough rows for architecture test split.")
    return ArchitectureSplit(train_end_pos=train_end, val_end_pos=val_end, test_end_pos=test_end)


def _sequence_arrays(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    lookback: int,
    start_pos: int,
    end_pos: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    feature_values = features.to_numpy(dtype=np.float32)
    target_values = targets.to_numpy(dtype=np.float32)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []
    for pos in range(max(int(lookback) - 1, int(start_pos)), int(end_pos)):
        target_row = target_values[pos]
        if np.isnan(target_row).any():
            continue
        sequence = feature_values[pos - int(lookback) + 1 : pos + 1]
        if sequence.shape[0] != int(lookback):
            continue
        x_rows.append(sequence)
        y_rows.append(target_row)
        timestamps.append(pd.Timestamp(features.index[pos]))
    if not x_rows:
        return (
            np.zeros((0, int(lookback), features.shape[1]), dtype=np.float32),
            np.zeros((0, targets.shape[1]), dtype=np.float32),
            [],
        )
    return np.stack(x_rows), np.stack(y_rows), timestamps


def _category_indices(feature_columns: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    lower_columns = [column.lower() for column in feature_columns]
    used: set[int] = set()
    for group_name, patterns in CATEGORY_PATTERNS.items():
        indices = [
            idx
            for idx, column in enumerate(lower_columns)
            if any(pattern in column for pattern in patterns)
        ]
        groups[group_name] = indices
        used.update(indices)
    residual = [idx for idx in range(len(feature_columns)) if idx not in used]
    if residual:
        groups["residual"] = residual
    return groups


def _build_model(
    candidate: dict[str, Any],
    *,
    input_dim: int,
    output_dim: int,
    feature_columns: list[str],
) -> nn.Module:
    if candidate["architecture"] == "category_attention":
        return CategoryAttentionJumpModel(
            category_indices=_category_indices(feature_columns),
            output_dim=output_dim,
            hidden_dim=int(candidate.get("hidden_dim", 64)),
            num_heads=int(candidate.get("num_heads", 4)),
            dropout=float(candidate.get("dropout", 0.10)),
        )
    return build_specialist_model(candidate, input_dim=input_dim, output_dim=output_dim)


def _train_model(
    model: nn.Module,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    training_cfg: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty train or validation dataset.")
    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    model = model.to(device)
    for _epoch in range(int(training_cfg["max_epochs"])):
        model.train()
        train_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(float(criterion(model(x_batch), y_batch).detach().cpu()))
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
            if stale_epochs >= int(training_cfg["patience"]):
                break
    return best_state, history


def _predict_model(
    model: nn.Module,
    features: pd.DataFrame,
    *,
    lookback: int,
    timestamps: list[pd.Timestamp],
    columns: list[str],
    device: str,
) -> pd.DataFrame:
    if not timestamps:
        return pd.DataFrame(columns=columns)
    positions = [features.index.get_loc(timestamp) for timestamp in timestamps]
    x_rows = [
        features.iloc[pos - int(lookback) + 1 : pos + 1].to_numpy(dtype=np.float32)
        for pos in positions
    ]
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(np.stack(x_rows), dtype=torch.float32, device=device)).cpu().numpy()
    return pd.DataFrame(raw, index=pd.Index(timestamps, name="timestamp"), columns=columns)


def _fit_candidate(
    *,
    candidate: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    split: ArchitectureSplit,
    training_cfg: dict[str, Any],
    device: str,
    scale_targets: bool,
) -> dict[str, Any]:
    lookback = int(candidate["lookback"])
    feature_scaler = FeatureScaler.fit(features.iloc[: split.train_end_pos])
    transformed_features = feature_scaler.transform(features)
    target_scaler = RobustTargetScaler.fit(targets.iloc[: split.train_end_pos]) if scale_targets else None
    training_targets = target_scaler.transform(targets) if target_scaler is not None else targets.copy()

    x_train, y_train, _train_ts = _sequence_arrays(
        transformed_features,
        training_targets,
        lookback,
        0,
        split.train_end_pos,
    )
    x_val, y_val, val_timestamps = _sequence_arrays(
        transformed_features,
        training_targets,
        lookback,
        split.train_end_pos,
        split.val_end_pos,
    )
    _x_test, _y_test, test_timestamps = _sequence_arrays(
        transformed_features,
        training_targets,
        lookback,
        split.val_end_pos,
        split.test_end_pos,
    )
    model = _build_model(
        candidate,
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        feature_columns=features.columns.tolist(),
    )
    best_state, history = _train_model(
        model,
        SequenceDataset(x_train, y_train),
        SequenceDataset(x_val, y_val),
        training_cfg,
        device,
    )
    model.load_state_dict(best_state)
    val_predictions = _predict_model(
        model,
        transformed_features,
        lookback=lookback,
        timestamps=val_timestamps,
        columns=targets.columns.tolist(),
        device=device,
    )
    test_predictions = _predict_model(
        model,
        transformed_features,
        lookback=lookback,
        timestamps=test_timestamps,
        columns=targets.columns.tolist(),
        device=device,
    )
    if target_scaler is not None:
        val_predictions = target_scaler.inverse_transform(val_predictions)
        test_predictions = target_scaler.inverse_transform(test_predictions)
    return {
        "candidate": candidate,
        "history": history,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(score)
    if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], score[mask]))


def _safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(score)
    if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(average_precision_score(y_true[mask], score[mask]))


def _jump_probability(predicted_jump_pct: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(sigmoid((np.abs(predicted_jump_pct) - threshold) / max(threshold, 1e-6), scale=1.0), dtype=float)


def jump_prediction_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    true_frame, pred_frame = y_true.align(y_pred, join="inner", axis=0)
    rows: dict[str, float] = {}
    mae_values: list[float] = []
    rmse_values: list[float] = []
    directional_values: list[float] = []
    auc_values: list[float] = []
    event_penalties: list[float] = []
    for column in JUMP_TARGET_COLUMNS:
        true_values = true_frame[column].to_numpy(dtype=float)
        pred_values = pred_frame[column].to_numpy(dtype=float)
        mask = np.isfinite(true_values) & np.isfinite(pred_values)
        if not mask.any():
            continue
        errors = pred_values[mask] - true_values[mask]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(math.sqrt(np.mean(np.square(errors))))
        direction_mask = mask & (np.abs(true_values) >= 1e-9)
        direction_accuracy = (
            float(np.mean(np.sign(pred_values[direction_mask]) == np.sign(true_values[direction_mask])))
            if direction_mask.any()
            else float("nan")
        )
        threshold = float(JUMP_EVENT_THRESHOLDS[column])
        event_true = (np.abs(true_values) >= threshold).astype(float)
        event_probability = _jump_probability(pred_values, threshold)
        auc = _safe_auc(event_true, event_probability)
        average_precision = _safe_average_precision(event_true, event_probability)
        brier = (
            float(brier_score_loss(event_true[mask], event_probability[mask]))
            if len(np.unique(event_true[mask])) > 1
            else float("nan")
        )
        suffix = column.replace("btc_forward_", "").replace("_pct", "")
        rows[f"{suffix}_mae"] = mae
        rows[f"{suffix}_rmse"] = rmse
        rows[f"{suffix}_directional_accuracy"] = direction_accuracy
        rows[f"{suffix}_event_auc"] = auc
        rows[f"{suffix}_event_average_precision"] = average_precision
        rows[f"{suffix}_event_brier"] = brier
        mae_values.append(mae / threshold)
        rmse_values.append(rmse / threshold)
        if math.isfinite(direction_accuracy):
            directional_values.append(direction_accuracy)
        if math.isfinite(auc):
            auc_values.append(auc)
            event_penalties.append(1.0 - auc)
    rows["normalized_mae"] = float(np.mean(mae_values)) if mae_values else float("nan")
    rows["normalized_rmse"] = float(np.mean(rmse_values)) if rmse_values else float("nan")
    rows["average_directional_accuracy"] = float(np.mean(directional_values)) if directional_values else float("nan")
    rows["average_event_auc"] = float(np.mean(auc_values)) if auc_values else float("nan")
    rows["selection_score"] = (
        rows["normalized_mae"]
        + 0.75 * (float(np.mean(event_penalties)) if event_penalties else 0.5)
        + 0.50 * (1.0 - rows["average_directional_accuracy"] if math.isfinite(rows["average_directional_accuracy"]) else 0.5)
    )
    return rows


def _semantic_candidate_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    predictions = y_pred.loc[:, list(LIQUIDATION_SEMANTIC_COLUMNS)].copy()
    for column in predictions.columns:
        predictions[column] = predictions[column].clip(-1.0, 1.0)
        if column != "liquidation_directional_pressure" and column != "cascade_asymmetry":
            predictions[column] = predictions[column].clip(0.0, 1.0)
    return specialist_regression_metrics(
        y_true=y_true.loc[predictions.index, list(LIQUIDATION_SEMANTIC_COLUMNS)],
        y_pred=predictions,
        confidence_column="liquidation_confidence",
    )


def run_candidate_search(
    *,
    features: pd.DataFrame,
    semantic_targets: pd.DataFrame,
    jump_targets: pd.DataFrame,
    split: ArchitectureSplit,
    candidates: tuple[dict[str, Any], ...] = DEFAULT_CANDIDATES,
    training_cfg: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg = {**DEFAULT_TRAINING_CONFIG, **(training_cfg or {})}
    semantic_rows: list[dict[str, Any]] = []
    jump_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        semantic_result = _fit_candidate(
            candidate=candidate,
            features=features,
            targets=semantic_targets,
            split=split,
            training_cfg=cfg,
            device=device,
            scale_targets=False,
        )
        semantic_metrics = _semantic_candidate_metrics(semantic_targets, semantic_result["test_predictions"])
        semantic_rows.append(
            {
                "name": candidate["name"],
                "architecture": candidate["architecture"],
                "lookback": int(candidate["lookback"]),
                **semantic_metrics,
                "epochs": len(semantic_result["history"]["val_loss"]),
                "last_val_loss": float(semantic_result["history"]["val_loss"][-1]),
            }
        )

        jump_result = _fit_candidate(
            candidate=candidate,
            features=features,
            targets=jump_targets,
            split=split,
            training_cfg=cfg,
            device=device,
            scale_targets=True,
        )
        jump_metrics = jump_prediction_metrics(jump_targets, jump_result["test_predictions"])
        jump_rows.append(
            {
                "name": candidate["name"],
                "architecture": candidate["architecture"],
                "lookback": int(candidate["lookback"]),
                **jump_metrics,
                "epochs": len(jump_result["history"]["val_loss"]),
                "last_val_loss": float(jump_result["history"]["val_loss"][-1]),
            }
        )
    semantic_leaderboard = pd.DataFrame(semantic_rows).sort_values("selection_score", ascending=True).reset_index(drop=True)
    jump_leaderboard = pd.DataFrame(jump_rows).sort_values("selection_score", ascending=True).reset_index(drop=True)
    return {
        "semantic_leaderboard": semantic_leaderboard,
        "jump_leaderboard": jump_leaderboard,
    }


def _encode_context_latents(
    model: SpecialistJEPA,
    features: pd.DataFrame,
    *,
    context_length: int,
    horizon: int,
    device: str,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for pos in range(int(context_length) - 1, len(features)):
            context = features.iloc[pos - int(context_length) + 1 : pos + 1].to_numpy(dtype=np.float32)
            batch = torch.tensor(context[None, :, :], dtype=torch.float32, device=device)
            horizon_tensor = torch.tensor([int(horizon)], dtype=torch.long, device=device)
            latent = model(batch, horizon_tensor)["context_latent"].detach().cpu().numpy()[0]
            rows.append(latent)
            timestamps.append(pd.Timestamp(features.index[pos]))
    columns = [f"liquidation_jepa_z_{idx}" for idx in range(model.latent_dim)]
    return pd.DataFrame(rows, index=pd.Index(timestamps, name="timestamp"), columns=columns)


def run_jepa_probe(
    *,
    features: pd.DataFrame,
    jump_targets: pd.DataFrame,
    split: ArchitectureSplit,
    context_length: int,
    epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if int(epochs) <= 0:
        return {"status": "skipped", "reason": "epochs <= 0"}
    set_global_seed(seed)
    feature_scaler = FeatureScaler.fit(features.iloc[: split.train_end_pos])
    scaled_features = feature_scaler.transform(features)
    dataset = LiquidationJEPADataset(
        scaled_features.iloc[: split.val_end_pos],
        context_length=int(context_length),
        horizons=(1, 3, 7),
    )
    if len(dataset) == 0:
        return {"status": "skipped", "reason": "empty_jepa_dataset"}
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=_jepa_collate)
    model = SpecialistJEPA(
        input_dim=scaled_features.shape[1],
        latent_dim=32,
        patch_length=8,
        hidden_dim=96,
        num_layers=2,
        num_heads=4,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[dict[str, float]] = []
        for batch in loader:
            context = batch["context_features"].to(device)
            target = batch["target_features"].to(device)
            mask = batch["target_mask"].to(device)
            horizon = batch["horizon"].to(device)
            optimizer.zero_grad()
            output = model(context, horizon, target_features=target, target_mask=mask)
            loss, metrics = total_jepa_loss(
                output["predicted_target_latent"],
                output["target_latent"],
                context_latent=output["context_latent"],
            )
            loss.backward()
            optimizer.step()
            losses.append(metrics)
        history.append({"epoch": epoch, **{key: float(np.mean([row[key] for row in losses])) for key in losses[0]}})

    latents = _encode_context_latents(
        model,
        scaled_features,
        context_length=int(context_length),
        horizon=7,
        device=device,
    )
    common = latents.index.intersection(jump_targets.dropna().index)
    latents = latents.loc[common]
    targets = jump_targets.loc[common]
    train_idx = common[common < features.index[split.train_end_pos]]
    test_idx = common[common >= features.index[split.val_end_pos]]
    if len(train_idx) < 50 or len(test_idx) < 20:
        return {"status": "skipped", "reason": "insufficient_probe_rows", "history": history}
    probe = Ridge(alpha=1.0)
    probe.fit(latents.loc[train_idx].to_numpy(dtype=float), targets.loc[train_idx].to_numpy(dtype=float))
    predictions = pd.DataFrame(
        probe.predict(latents.loc[test_idx].to_numpy(dtype=float)),
        index=test_idx,
        columns=targets.columns,
    )
    metrics = jump_prediction_metrics(targets.loc[test_idx], predictions)
    latent_std = latents.loc[test_idx].std(axis=0).to_numpy(dtype=float)
    return {
        "status": "trained",
        "history": history,
        "probe_metrics": metrics,
        "latent_std_mean": float(np.nanmean(latent_std)),
        "latent_std_min": float(np.nanmin(latent_std)),
        "samples": int(len(dataset)),
        "probe_train_rows": int(len(train_idx)),
        "probe_test_rows": int(len(test_idx)),
    }


def _manifest(features: pd.DataFrame, signal_frame: pd.DataFrame, split: ArchitectureSplit) -> dict[str, Any]:
    forbidden = [column for column in features.columns if column in PRICE_JUMP_TARGET_FIELDS or column.startswith("btc_forward_")]
    return {
        "source_rows": int(len(signal_frame)),
        "feature_rows": int(len(features)),
        "feature_columns": int(features.shape[1]),
        "forbidden_target_feature_columns": forbidden,
        "enrichment_feature_counts": {
            "rolling_std": int(sum("_roll7_std" in column or "_roll30_std" in column for column in features.columns)),
            "signalboost": int(sum(column.endswith("_signalboost") for column in features.columns)),
            "missingness": int(sum(column.endswith("_missing") for column in features.columns)),
        },
        "split": {
            "train_end": str(features.index[split.train_end_pos - 1].date()),
            "validation_end": str(features.index[split.val_end_pos - 1].date()),
            "test_end": str(features.index[split.test_end_pos - 1].date()),
            "train_rows": int(split.train_end_pos),
            "validation_rows": int(split.val_end_pos - split.train_end_pos),
            "test_rows": int(split.test_end_pos - split.val_end_pos),
        },
    }


def _markdown_report(summary: dict[str, Any], semantic: pd.DataFrame, jump: pd.DataFrame, jepa: dict[str, Any]) -> str:
    best_semantic = semantic.iloc[0].to_dict()
    best_jump = jump.iloc[0].to_dict()
    lines = [
        "# Liquidation Specialist Architecture Tests",
        "",
        "## Scope",
        "",
        "Pre-integration research pass for a fifth liquidation specialist. The test keeps production routing unchanged and evaluates whether the new data supports specialist-style semantic regression, direct future jump prediction, and a small JEPA representation probe.",
        "",
        "Documentation read: `SPECIALIST_LAYER_REFERENCE.md`, `PRODUCTION_SPEC.md`, `docs/jepa_integration.md`, `liquidations/README.md`, and Liquidation Heatmap theory notes.",
        "",
        "## Feature Contract",
        "",
        f"- Source rows: {summary['manifest']['source_rows']}",
        f"- Feature columns: {summary['manifest']['feature_columns']}",
        f"- Rolling standard deviation enrichments: {summary['manifest']['enrichment_feature_counts']['rolling_std']}",
        f"- Signalboost enrichments: {summary['manifest']['enrichment_feature_counts']['signalboost']}",
        f"- Forward target columns in features: {len(summary['manifest']['forbidden_target_feature_columns'])}",
        "",
        "The feature builder excludes all `btc_forward_*` fields. Those columns are used only as labels for the jump tests.",
        "",
        "## Best Candidates",
        "",
        f"- Semantic-state regression: `{best_semantic['name']}` ({best_semantic['architecture']}, lookback={int(best_semantic['lookback'])}), selection_score={best_semantic['selection_score']:.6f}",
        f"- Future jump objective: `{best_jump['name']}` ({best_jump['architecture']}, lookback={int(best_jump['lookback'])}), selection_score={best_jump['selection_score']:.6f}, avg_event_auc={best_jump.get('average_event_auc', float('nan')):.4f}",
        "",
        "## Recommendation",
        "",
    ]
    if best_jump["architecture"] == "category_attention":
        lines.append("Use the categorical attention jump head as the first research candidate for liquidation jump prediction, then retain the best specialist-style semantic model as the state emitter.")
    else:
        lines.append("Start with the best specialist-style backbone for the liquidation state emitter. Keep categorical attention as a challenger only if later folds show better jump-event calibration.")
    lines.extend(
        [
            "",
            "For production integration, add the liquidation bucket only after a walk-forward fold run confirms that these state outputs improve fusion calibration or JEPA-gated meta-synthesis without destabilizing the current four-specialist baseline.",
            "",
            "## JEPA Probe",
            "",
        ]
    )
    if jepa.get("status") == "trained":
        probe = jepa["probe_metrics"]
        lines.extend(
            [
                f"- Status: trained, samples={jepa['samples']}, latent_std_mean={jepa['latent_std_mean']:.6f}",
                f"- Probe selection_score={probe['selection_score']:.6f}, avg_event_auc={probe.get('average_event_auc', float('nan')):.4f}",
            ]
        )
    else:
        lines.append(f"- Status: {jepa.get('status')} ({jepa.get('reason', 'not available')})")
    lines.extend(
        [
            "",
            "## Leaderboards",
            "",
            "Semantic leaderboard: `semantic_leaderboard.csv`",
            "",
            "Jump leaderboard: `jump_leaderboard.csv`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_architecture_tests(
    *,
    input_path: str | Path = LIQUIDATION_SIGNAL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_rows: int | None = None,
    validation_days: int = 240,
    test_days: int = 365,
    min_train_days: int = 1_600,
    training_cfg: dict[str, Any] | None = None,
    jepa_epochs: int = 2,
    jepa_context_length: int = 96,
    device: str = "cpu",
    seed: int = 7,
    candidates: tuple[dict[str, Any], ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    set_global_seed(seed)
    output_root = ensure_dir(output_dir)
    signal_frame = load_liquidation_signal_frame(input_path, max_rows=max_rows)
    features = build_liquidation_feature_frame(signal_frame)
    semantic_targets = build_liquidation_semantic_targets(signal_frame)
    jump_targets = build_jump_targets(signal_frame)
    common_index = features.index.intersection(semantic_targets.index).intersection(jump_targets.dropna().index)
    features = features.loc[common_index]
    semantic_targets = semantic_targets.loc[common_index]
    jump_targets = jump_targets.loc[common_index]
    split = _build_split(features.index, validation_days=validation_days, test_days=test_days, min_train_days=min_train_days)

    search = run_candidate_search(
        features=features,
        semantic_targets=semantic_targets,
        jump_targets=jump_targets,
        split=split,
        candidates=candidates,
        training_cfg=training_cfg,
        device=device,
    )
    jepa = run_jepa_probe(
        features=features,
        jump_targets=jump_targets,
        split=split,
        context_length=jepa_context_length,
        epochs=jepa_epochs,
        device=device,
        seed=seed,
    )
    manifest = _manifest(features, signal_frame, split)
    semantic_leaderboard = search["semantic_leaderboard"]
    jump_leaderboard = search["jump_leaderboard"]
    summary = {
        "manifest": manifest,
        "best_semantic": semantic_leaderboard.iloc[0].to_dict(),
        "best_jump": jump_leaderboard.iloc[0].to_dict(),
        "jepa_probe": jepa,
        "candidate_count": int(len(candidates)),
    }
    semantic_leaderboard.to_csv(output_root / "semantic_leaderboard.csv", index=False)
    jump_leaderboard.to_csv(output_root / "jump_leaderboard.csv", index=False)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "feature_manifest.json", manifest)
    (output_root / "model_selection_report.md").write_text(
        _markdown_report(summary, semantic_leaderboard, jump_leaderboard, jepa),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_root),
        "summary": summary,
        "semantic_leaderboard": semantic_leaderboard,
        "jump_leaderboard": jump_leaderboard,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-integration liquidation specialist architecture tests.")
    parser.add_argument("--input", default=str(LIQUIDATION_SIGNAL_PATH), help="Liquidation signal CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for report artifacts.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional tail-row limit for smoke runs.")
    parser.add_argument("--validation-days", type=int, default=240)
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--min-train-days", type=int, default=1600)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--jepa-epochs", type=int, default=2)
    parser.add_argument("--jepa-context-length", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_cfg = {
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
    }
    result = run_architecture_tests(
        input_path=args.input,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        validation_days=int(args.validation_days),
        test_days=int(args.test_days),
        min_train_days=int(args.min_train_days),
        training_cfg=training_cfg,
        jepa_epochs=int(args.jepa_epochs),
        jepa_context_length=int(args.jepa_context_length),
        device=str(args.device),
        seed=int(args.seed),
    )
    summary = result["summary"]
    print(
        {
            "output_dir": result["output_dir"],
            "best_semantic": summary["best_semantic"]["name"],
            "best_jump": summary["best_jump"]["name"],
            "jepa_status": summary["jepa_probe"]["status"],
        }
    )


if __name__ == "__main__":
    main()
