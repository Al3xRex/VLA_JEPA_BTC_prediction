from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dual_model_forecaster.metrics import specialist_regression_metrics
from dual_model_forecaster.utils import (
    ewm_smooth_frame,
    ensure_dir,
    sigmoid,
    write_json,
)


NON_NEGATIVE_HINTS = (
    "overvaluation",
    "undervaluation",
    "conviction",
    "fragility",
    "confidence",
    "tailwind",
    "headwind",
    "risk_on",
    "risk_off",
    "transition_risk",
    "stretch",
    "mean_reversion_pressure",
    "instability",
    "trend_pressure_up",
    "trend_pressure_down",
    "trend_persistence",
    "momentum_quality",
    "chop_risk",
    "influence",
    "cascade_pressure",
    "leverage_fuel",
    "volatility_instability",
    "liquidation_confidence",
)

TOKEN_METRICS = ("direction", "amplitude", "confidence", "dispersion")
SUMMARY_COLUMNS = (
    "summary_direction",
    "summary_amplitude",
    "summary_confidence",
    "summary_dispersion",
)
COMMUNICATION_COLUMNS = (
    "internal_disagreement",
    "scale_entropy",
    "state_velocity",
    "state_acceleration",
    "hazard",
    "freshness",
    "validity",
)

CATEGORY_ATTENTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "funding": ("funding", "predicted_funding"),
    "open_interest": ("open_interest", "futures_oi", "oi_", "leverage", "anchor"),
    "liquidation": ("liquidation", "liquidated", "heatmap"),
    "cascade_space": ("cascade", "wall", "area", "pressure"),
    "volatility": ("volatility", "std", "trigger", "regime"),
    "coverage": ("confidence", "missing", "contract_count", "venue_count"),
}


@dataclass
class FeatureScaler:
    median: pd.Series
    iqr: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureScaler":
        median = frame.median(axis=0)
        q25 = frame.quantile(0.25, axis=0)
        q75 = frame.quantile(0.75, axis=0)
        iqr = (q75 - q25).replace(0, 1.0).fillna(1.0)
        return cls(median=median.fillna(0.0), iqr=iqr)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = (frame - self.median) / self.iqr
        return transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.to_dict(),
            "iqr": self.iqr.to_dict(),
        }


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _semantic_columns(bucket_name: str) -> list[str]:
    if bucket_name == "structure":
        return [
            "structural_overvaluation",
            "structural_undervaluation",
            "holder_conviction",
            "holder_distribution_fragility",
            "structural_reversion_pressure",
            "structural_confidence",
        ]
    if bucket_name == "environment":
        return [
            "liquidity_tailwind",
            "liquidity_headwind",
            "macro_risk_on",
            "macro_risk_off",
            "macro_transition_risk",
            "environment_confidence",
        ]
    if bucket_name == "edges":
        return [
            "upside_stretch",
            "downside_stretch",
            "mean_reversion_pressure",
            "local_volatility_instability",
            "edge_asymmetry",
            "edge_confidence",
        ]
    if bucket_name == "liquidation":
        return [
            "liquidation_cascade_pressure",
            "liquidation_directional_pressure",
            "leverage_fuel",
            "liquidation_volatility_instability",
            "cascade_asymmetry",
            "liquidation_confidence",
        ]
    return [
        "trend_pressure_up",
        "trend_pressure_down",
        "trend_persistence",
        "momentum_quality",
        "chop_risk",
        "trend_strategy_influence",
        "momentum_strategy_influence",
        "mean_reversion_strategy_influence",
        "movement_confidence",
    ]


def _clamp_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].clip(lower=-1.0, upper=1.0)
        if any(hint in column for hint in NON_NEGATIVE_HINTS):
            out[column] = out[column].clip(lower=0.0, upper=1.0)
    return out


def _directional_signal(bucket_name: str, frame: pd.DataFrame) -> pd.Series:
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


def _confidence_column(bucket_name: str) -> str:
    if bucket_name == "structure":
        return "structural_confidence"
    if bucket_name == "environment":
        return "environment_confidence"
    if bucket_name == "edges":
        return "edge_confidence"
    if bucket_name == "liquidation":
        return "liquidation_confidence"
    return "movement_confidence"


def _confidence_series(bucket_name: str, frame: pd.DataFrame) -> pd.Series:
    return frame[_confidence_column(bucket_name)].astype(float)


def _sequence_indices(
    index: pd.Index,
    lookback: int,
    start_pos: int,
    end_pos: int,
) -> list[int]:
    lower = max(start_pos, lookback - 1)
    upper = max(lower, end_pos)
    return list(range(lower, upper))


def _build_arrays(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    lookback: int,
    start_pos: int,
    end_pos: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    feature_values = features.to_numpy(dtype=np.float32)
    target_values = targets.to_numpy(dtype=np.float32)
    positions = _sequence_indices(features.index, lookback, start_pos, end_pos)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []
    for pos in positions:
        target_row = target_values[pos]
        if np.isnan(target_row).any():
            continue
        sequence = feature_values[pos - lookback + 1 : pos + 1]
        if sequence.shape[0] != lookback:
            continue
        x_rows.append(sequence)
        y_rows.append(target_row)
        timestamps.append(pd.Timestamp(features.index[pos]))

    if not x_rows:
        return (
            np.zeros((0, lookback, features.shape[1]), dtype=np.float32),
            np.zeros((0, targets.shape[1]), dtype=np.float32),
            [],
        )
    return np.stack(x_rows), np.stack(y_rows), timestamps


class LinearSpecialist(nn.Module):
    def __init__(self, input_dim: int, lookback: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim * lookback, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class MLPSpecialist(nn.Module):
    def __init__(self, input_dim: int, lookback: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * lookback, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class GRUSpecialist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.head(hidden[-1])


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.net(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out + residual
        return F.gelu(out)


class TCNSpecialist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.input_proj(out)
        for block in self.blocks:
            out = block(out)
        return self.head(out[:, :, -1])


class PatchTSTLikeSpecialist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        output_dim: int,
        patch_length: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.num_patches = max(1, lookback // patch_length)
        self.proj = nn.Linear(patch_length * input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, features = x.shape
        usable = self.num_patches * self.patch_length
        if usable > length:
            raise RuntimeError("Patch length exceeds sequence length.")
        trimmed = x[:, length - usable :, :]
        patches = trimmed.reshape(batch, self.num_patches, self.patch_length * features)
        tokens = self.proj(patches) + self.positional
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def _category_indices(feature_columns: list[str], input_dim: int) -> dict[str, list[int]]:
    if not feature_columns:
        return {"all": list(range(input_dim))}
    lower_columns = [column.lower() for column in feature_columns]
    groups: dict[str, list[int]] = {}
    used: set[int] = set()
    for group_name, patterns in CATEGORY_ATTENTION_PATTERNS.items():
        indices = [
            idx
            for idx, column in enumerate(lower_columns)
            if any(pattern in column for pattern in patterns)
        ]
        if indices:
            groups[group_name] = indices
            used.update(indices)
    residual = [idx for idx in range(input_dim) if idx not in used]
    if residual:
        groups["residual"] = residual
    return groups or {"all": list(range(input_dim))}


class CategoryAttentionSpecialist(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        feature_columns: list[str],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        category_map = _category_indices(feature_columns, input_dim)
        self.category_names = list(category_map)
        self.category_indices = [category_map[name] for name in self.category_names]
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


def build_specialist_model(
    candidate: dict[str, Any],
    input_dim: int,
    output_dim: int,
    feature_columns: list[str] | None = None,
) -> nn.Module:
    architecture = candidate["architecture"]
    lookback = int(candidate["lookback"])
    hidden_dim = int(candidate.get("hidden_dim", 64))
    dropout = float(candidate.get("dropout", 0.0))
    if architecture == "linear":
        return LinearSpecialist(input_dim=input_dim, lookback=lookback, output_dim=output_dim)
    if architecture == "mlp":
        return MLPSpecialist(
            input_dim=input_dim,
            lookback=lookback,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if architecture == "gru":
        return GRUSpecialist(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=int(candidate.get("num_layers", 1)),
            dropout=dropout,
        )
    if architecture == "tcn":
        return TCNSpecialist(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            kernel_size=int(candidate.get("kernel_size", 3)),
            num_layers=int(candidate.get("num_layers", 3)),
            dropout=dropout,
        )
    if architecture == "patchtst_like":
        return PatchTSTLikeSpecialist(
            input_dim=input_dim,
            lookback=lookback,
            output_dim=output_dim,
            patch_length=int(candidate.get("patch_length", 8)),
            hidden_dim=hidden_dim,
            num_layers=int(candidate.get("num_layers", 2)),
            num_heads=int(candidate.get("num_heads", 4)),
            dropout=dropout,
        )
    if architecture == "category_attention":
        return CategoryAttentionSpecialist(
            input_dim=input_dim,
            output_dim=output_dim,
            feature_columns=feature_columns or [f"feature_{idx}" for idx in range(input_dim)],
            hidden_dim=hidden_dim,
            num_heads=int(candidate.get("num_heads", 4)),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported specialist architecture: {architecture}")


def _run_training(
    model: nn.Module,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    training_cfg: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty train or validation dataset for specialist training.")

    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.SmoothL1Loss()
    patience = int(training_cfg["patience"])

    history = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0

    for _epoch in range(int(training_cfg["max_epochs"])):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
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
                loss = criterion(prediction, y_batch)
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


def _predict_frame(
    model: nn.Module,
    features: pd.DataFrame,
    timestamps: list[pd.Timestamp],
    lookback: int,
    columns: list[str],
    device: str,
) -> pd.DataFrame:
    if not timestamps:
        return pd.DataFrame(columns=columns)

    positions = [features.index.get_loc(timestamp) for timestamp in timestamps]
    x_rows = []
    for pos in positions:
        sequence = features.iloc[pos - lookback + 1 : pos + 1].to_numpy(dtype=np.float32)
        x_rows.append(sequence)
    array = np.stack(x_rows)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(array, dtype=torch.float32, device=device)).cpu().numpy()
    frame = pd.DataFrame(predictions, index=pd.Index(timestamps, name="timestamp"), columns=columns)
    return _clamp_outputs(frame)


def fit_specialist_candidate(
    bucket_name: str,
    candidate: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    train_end_pos: int,
    output_dir: Path,
    training_cfg: dict[str, Any],
    validation_days: int,
    device: str,
) -> dict[str, Any]:
    lookback = int(candidate["lookback"])
    val_start_pos = max(lookback, train_end_pos - validation_days)
    scaler = FeatureScaler.fit(features.iloc[:val_start_pos])
    transformed = scaler.transform(features)

    x_train, y_train, _ = _build_arrays(transformed, targets, lookback, 0, val_start_pos)
    x_val, y_val, val_timestamps = _build_arrays(transformed, targets, lookback, val_start_pos, train_end_pos)
    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)

    model = build_specialist_model(
        candidate,
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        feature_columns=features.columns.tolist(),
    )
    best_state, history = _run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_cfg=training_cfg,
        device=device,
    )
    model.load_state_dict(best_state)
    predictions = _predict_frame(
        model=model,
        features=transformed,
        timestamps=val_timestamps,
        lookback=lookback,
        columns=targets.columns.tolist(),
        device=device,
    )
    metrics = specialist_regression_metrics(
        y_true=targets.loc[predictions.index],
        y_pred=predictions,
        confidence_column=_confidence_column(bucket_name),
    )

    candidate_dir = ensure_dir(output_dir / candidate["name"])
    torch.save(best_state, candidate_dir / "checkpoint.pt")
    write_json(candidate_dir / "scaler.json", scaler.to_dict())
    write_json(candidate_dir / "metrics.json", metrics)
    write_json(candidate_dir / "history.json", history)

    return {
        "candidate": candidate,
        "scaler": scaler,
        "state_dict": best_state,
        "metrics": metrics,
        "history": history,
        "validation_predictions": predictions,
        "artifact_dir": str(candidate_dir),
    }


def _candidate_group_name(bucket_name: str, lookback: int, config: dict[str, Any]) -> str:
    grouping_cfg = (
        config.get("specialists", {})
        .get("grouping", {})
        .get("buckets", {})
        .get(bucket_name, {})
    )
    for group in grouping_cfg.get("scale_groups", []):
        if int(group["min_lookback"]) <= lookback <= int(group["max_lookback"]):
            return str(group["name"])
    return f"lb_{lookback}"


def _scale_group_order(bucket_name: str, config: dict[str, Any]) -> list[str]:
    grouping_cfg = (
        config.get("specialists", {})
        .get("grouping", {})
        .get("buckets", {})
        .get(bucket_name, {})
    )
    return [str(group["name"]) for group in grouping_cfg.get("scale_groups", [])]


def _select_retained_candidates(
    bucket_name: str,
    candidate_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in candidate_results:
        group_name = _candidate_group_name(bucket_name, int(result["candidate"]["lookback"]), config)
        grouped.setdefault(group_name, []).append(result)

    retained: list[dict[str, Any]] = []
    ordered_group_names = _scale_group_order(bucket_name, config)
    seen = set()
    for group_name in ordered_group_names:
        group_results = grouped.get(group_name, [])
        if not group_results:
            continue
        best = min(group_results, key=lambda item: item["metrics"]["selection_score"])
        retained.append({"group_name": group_name, **best})
        seen.add(group_name)

    for group_name, group_results in grouped.items():
        if group_name in seen:
            continue
        best = min(group_results, key=lambda item: item["metrics"]["selection_score"])
        retained.append({"group_name": group_name, **best})

    minimum_groups = int(config.get("specialists", {}).get("grouping", {}).get("minimum_groups", 1))
    if len(retained) < minimum_groups:
        selected_names = {item["candidate"]["name"] for item in retained}
        for result in sorted(candidate_results, key=lambda item: item["metrics"]["selection_score"]):
            if result["candidate"]["name"] in selected_names:
                continue
            retained.append(
                {
                    "group_name": _candidate_group_name(bucket_name, int(result["candidate"]["lookback"]), config),
                    **result,
                }
            )
            if len(retained) >= minimum_groups:
                break

    retained.sort(
        key=lambda item: (
            ordered_group_names.index(item["group_name"]) if item["group_name"] in ordered_group_names else 999,
            int(item["candidate"]["lookback"]),
        )
    )
    return retained


def _state_schema(bucket_name: str, retained_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    token_columns = []
    for retained in retained_candidates:
        group_name = retained["group_name"]
        token_columns.extend([f"scale_{group_name}_{metric}" for metric in TOKEN_METRICS])
    return {
        "semantic_columns": _semantic_columns(bucket_name),
        "summary_columns": list(SUMMARY_COLUMNS),
        "scale_token_columns": token_columns,
        "communication_columns": list(COMMUNICATION_COLUMNS),
    }


def select_best_specialist(
    bucket_name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    train_end_pos: int,
    config: dict[str, Any],
    artifact_root: Path,
    device: str,
) -> dict[str, Any]:
    specialist_cfg = config["specialists"]
    bucket_cfg = specialist_cfg["buckets"][bucket_name]
    training_cfg = specialist_cfg["training"]
    validation_days = int(config["splits"]["specialist_validation_days"])
    bucket_dir = ensure_dir(artifact_root / "specialists" / bucket_name)

    candidate_results = []
    for candidate in bucket_cfg["candidates"]:
        result = fit_specialist_candidate(
            bucket_name=bucket_name,
            candidate=candidate,
            features=features,
            targets=targets,
            train_end_pos=train_end_pos,
            output_dir=bucket_dir,
            training_cfg=training_cfg,
            validation_days=validation_days,
            device=device,
        )
        candidate_results.append(result)

    candidate_results.sort(key=lambda item: item["metrics"]["selection_score"])
    retained = _select_retained_candidates(bucket_name=bucket_name, candidate_results=candidate_results, config=config)
    anchor = candidate_results[0]
    leaderboard = [
        {
            "name": item["candidate"]["name"],
            "group_name": _candidate_group_name(bucket_name, int(item["candidate"]["lookback"]), config),
            "lookback": int(item["candidate"]["lookback"]),
            "architecture": item["candidate"]["architecture"],
            **item["metrics"],
        }
        for item in candidate_results
    ]
    selection_payload = {
        "bucket_name": bucket_name,
        "best_candidate": anchor["candidate"],
        "retained_candidates": [
            {
                "group_name": item["group_name"],
                "candidate": item["candidate"],
                "metrics": item["metrics"],
                "artifact_dir": item["artifact_dir"],
            }
            for item in retained
        ],
        "scale_groups": _scale_group_order(bucket_name, config),
        "state_schema": _state_schema(bucket_name, retained),
        "leaderboard": leaderboard,
    }
    write_json(bucket_dir / "leaderboard.json", leaderboard)
    write_json(bucket_dir / "selection.json", selection_payload)
    return {
        **selection_payload,
        "post_smooth_alpha": float(bucket_cfg["post_smooth_alpha"]),
    }


def _fit_final_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    train_end_pos: int,
    candidate: dict[str, Any],
    training_cfg: dict[str, Any],
    device: str,
) -> tuple[nn.Module, FeatureScaler]:
    lookback = int(candidate["lookback"])
    validation_days = min(90, max(lookback + 1, train_end_pos // 5))
    val_start_pos = max(lookback, train_end_pos - validation_days)
    scaler = FeatureScaler.fit(features.iloc[:val_start_pos])
    transformed = scaler.transform(features)
    x_train, y_train, _ = _build_arrays(transformed, targets, lookback, 0, val_start_pos)
    x_val, y_val, _ = _build_arrays(transformed, targets, lookback, val_start_pos, train_end_pos)
    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)
    model = build_specialist_model(
        candidate,
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        feature_columns=features.columns.tolist(),
    )
    best_state, _history = _run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_cfg=training_cfg,
        device=device,
    )
    model.load_state_dict(best_state)
    return model, scaler


def _build_scale_token_frame(
    bucket_name: str,
    scale_name: str,
    states: pd.DataFrame,
) -> pd.DataFrame:
    semantic = states[_semantic_columns(bucket_name)].astype(float)
    directional = _directional_signal(bucket_name, semantic)
    confidence = _confidence_series(bucket_name, semantic)
    amplitude = semantic.abs().mean(axis=1)
    dispersion = semantic.std(axis=1).fillna(0.0)
    token = pd.DataFrame(index=semantic.index)
    token[f"scale_{scale_name}_direction"] = directional
    token[f"scale_{scale_name}_amplitude"] = amplitude
    token[f"scale_{scale_name}_confidence"] = confidence
    token[f"scale_{scale_name}_dispersion"] = dispersion
    return token


def _normalized_entropy(values: pd.DataFrame) -> pd.Series:
    if values.empty:
        return pd.Series(0.0, index=values.index)
    arr = values.to_numpy(dtype=float)
    arr = np.abs(arr)
    arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-8, None)
    entropy = -np.sum(arr * np.log(np.clip(arr, 1e-8, None)), axis=1)
    normalizer = np.log(max(arr.shape[1], 2))
    return pd.Series(entropy / normalizer, index=values.index).fillna(0.0).clip(0.0, 1.0)


def _compose_multiscale_states(
    bucket_name: str,
    scale_predictions: dict[str, pd.DataFrame],
    anchor_scale: str,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    if not scale_predictions:
        return pd.DataFrame(columns=_semantic_columns(bucket_name))

    common_index = None
    for frame in scale_predictions.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) == 0:
        return pd.DataFrame(columns=_semantic_columns(bucket_name))
    common_index = common_index.sort_values()

    aligned_scales = {name: frame.loc[common_index, _semantic_columns(bucket_name)] for name, frame in scale_predictions.items()}
    anchor_frame = aligned_scales[anchor_scale].copy()

    token_frames = [
        _build_scale_token_frame(bucket_name=bucket_name, scale_name=scale_name, states=frame)
        for scale_name, frame in aligned_scales.items()
    ]
    tokens = pd.concat(token_frames, axis=1) if token_frames else pd.DataFrame(index=common_index)

    direction_columns = [column for column in tokens.columns if column.endswith("_direction")]
    amplitude_columns = [column for column in tokens.columns if column.endswith("_amplitude")]
    confidence_columns = [column for column in tokens.columns if column.endswith("_confidence")]
    dispersion_columns = [column for column in tokens.columns if column.endswith("_dispersion")]

    directions = tokens[direction_columns] if direction_columns else pd.DataFrame(index=common_index)
    out = anchor_frame.copy()
    out["summary_direction"] = directions.mean(axis=1) if not directions.empty else _directional_signal(bucket_name, anchor_frame)
    out["summary_amplitude"] = tokens[amplitude_columns].mean(axis=1) if amplitude_columns else anchor_frame.abs().mean(axis=1)
    out["summary_confidence"] = tokens[confidence_columns].mean(axis=1) if confidence_columns else _confidence_series(bucket_name, anchor_frame)
    out["summary_dispersion"] = tokens[dispersion_columns].mean(axis=1) if dispersion_columns else anchor_frame.std(axis=1).fillna(0.0)
    out["internal_disagreement"] = directions.std(axis=1).fillna(0.0) if not directions.empty else 0.0
    out["scale_entropy"] = _normalized_entropy(directions.abs()) if not directions.empty else 0.0
    out["state_velocity"] = out["summary_direction"].diff().abs().ewm(span=5, adjust=False).mean().fillna(0.0)
    out["state_acceleration"] = out["state_velocity"].diff().abs().ewm(span=5, adjust=False).mean().fillna(0.0)

    anchor_confidence = _confidence_series(bucket_name, anchor_frame)
    hazard = sigmoid(
        2.2 * out["state_velocity"].astype(float)
        + 1.4 * out["internal_disagreement"].astype(float)
        + 0.8 * out["scale_entropy"].astype(float)
        + (1.0 - anchor_confidence.astype(float)),
        scale=1.0,
    )
    validity = (
        anchor_confidence.astype(float)
        * freshness.loc[common_index].astype(float).clip(0.0, 1.0)
        * (1.0 - missingness.loc[common_index].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["hazard"] = pd.Series(hazard, index=common_index).clip(0.0, 1.0)
    out["freshness"] = freshness.loc[common_index].astype(float).clip(0.0, 1.0)
    out["validity"] = validity

    combined = pd.concat([out, tokens.loc[common_index]], axis=1)
    return combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def emit_specialist_states(
    bucket_name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
    train_end_pos: int,
    predict_start_pos: int,
    predict_end_pos: int,
    selection: dict[str, Any],
    config: dict[str, Any],
    device: str,
) -> pd.DataFrame:
    training_cfg = config["specialists"]["training"]
    scale_predictions: dict[str, pd.DataFrame] = {}

    for retained in selection["retained_candidates"]:
        candidate = retained["candidate"]
        model, scaler = _fit_final_model(
            features=features.iloc[:train_end_pos],
            targets=targets.iloc[:train_end_pos],
            train_end_pos=train_end_pos,
            candidate=candidate,
            training_cfg=training_cfg,
            device=device,
        )
        transformed = scaler.transform(features)
        lookback = int(candidate["lookback"])
        timestamps = [
            pd.Timestamp(features.index[pos])
            for pos in _sequence_indices(features.index, lookback, predict_start_pos, predict_end_pos)
        ]
        predictions = _predict_frame(
            model=model,
            features=transformed,
            timestamps=timestamps,
            lookback=lookback,
            columns=targets.columns.tolist(),
            device=device,
        )
        predictions = ewm_smooth_frame(predictions, alpha=float(selection["post_smooth_alpha"]))
        scale_predictions[retained["group_name"]] = predictions

    anchor_scale = selection["retained_candidates"][0]["group_name"]
    for retained in selection["retained_candidates"]:
        if retained["candidate"]["name"] == selection["best_candidate"]["name"]:
            anchor_scale = retained["group_name"]
            break

    return _compose_multiscale_states(
        bucket_name=bucket_name,
        scale_predictions=scale_predictions,
        anchor_scale=anchor_scale,
        freshness=freshness,
        missingness=missingness,
    )


def build_walkforward_states(
    bucket_name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
    inner_segments: list[dict[str, Any]],
    outer_fold: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
    feature_root: Path,
    device: str,
) -> dict[str, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    for segment in inner_segments:
        state_part = emit_specialist_states(
            bucket_name=bucket_name,
            features=features,
            targets=targets,
            freshness=freshness,
            missingness=missingness,
            train_end_pos=int(segment["fit_end_pos"]),
            predict_start_pos=int(segment["predict_start_pos"]),
            predict_end_pos=int(segment["predict_end_pos"]),
            selection=selection,
            config=config,
            device=device,
        )
        train_parts.append(state_part)

    train_states = pd.concat(train_parts).sort_index() if train_parts else pd.DataFrame()
    test_states = emit_specialist_states(
        bucket_name=bucket_name,
        features=features,
        targets=targets,
        freshness=freshness,
        missingness=missingness,
        train_end_pos=int(outer_fold["train_end_pos"]),
        predict_start_pos=int(outer_fold["test_start_pos"]),
        predict_end_pos=int(outer_fold["test_end_pos"]),
        selection=selection,
        config=config,
        device=device,
    )

    out_dir = ensure_dir(feature_root / "specialist_states")
    train_states.to_csv(out_dir / f"{bucket_name}_train_states.csv")
    test_states.to_csv(out_dir / f"{bucket_name}_test_states.csv")
    write_json(
        out_dir / f"{bucket_name}_selection.json",
        {
            "bucket_name": bucket_name,
            "best_candidate": selection["best_candidate"],
            "retained_candidates": selection["retained_candidates"],
            "scale_groups": selection["scale_groups"],
            "state_schema": selection["state_schema"],
            "leaderboard": selection["leaderboard"],
        },
    )
    return {
        "train": train_states,
        "test": test_states,
    }
