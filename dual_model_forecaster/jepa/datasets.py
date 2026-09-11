from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dual_model_forecaster.jepa.feature_contract import DEFAULT_HORIZONS, numeric_feature_columns
from dual_model_forecaster.jepa.targets import target_bounds, target_window_for_horizon
from dual_model_forecaster.jepa.temporal_safety import validate_context_columns, validate_sample_window


@dataclass(frozen=True)
class JEPASample:
    context_features: np.ndarray
    target_features: np.ndarray
    specialist_name: str
    horizon: int
    as_of_timestamp: pd.Timestamp
    target_start_timestamp: pd.Timestamp
    target_end_timestamp: pd.Timestamp


def _prepare_numeric_panel(panel: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_columns or numeric_feature_columns(panel)
    validate_context_columns(columns)
    numeric = panel.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return numeric.astype(float), columns


def build_sample_index(
    index: pd.Index,
    *,
    context_length: int,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> list[tuple[int, int, pd.Timestamp, pd.Timestamp]]:
    timestamps = pd.DatetimeIndex(index).sort_values()
    rows: list[tuple[int, int, pd.Timestamp, pd.Timestamp]] = []
    if len(timestamps) < int(context_length) + 2:
        return rows
    timestamp_set = set(timestamps)
    for pos in range(int(context_length) - 1, len(timestamps)):
        as_of = pd.Timestamp(timestamps[pos])
        context_start = pd.Timestamp(timestamps[pos - int(context_length) + 1])
        for horizon in horizons:
            start_ts, end_ts = target_bounds(as_of, int(horizon))
            target_index = timestamps[(timestamps >= start_ts) & (timestamps <= end_ts)]
            if len(target_index) == 0:
                continue
            if target_index[-1] > timestamps[-1]:
                continue
            validate_sample_window(
                context_start=context_start,
                context_end=as_of,
                target_start=pd.Timestamp(target_index[0]),
                target_end=pd.Timestamp(target_index[-1]),
                as_of=as_of,
            )
            rows.append((pos, int(horizon), pd.Timestamp(target_index[0]), pd.Timestamp(target_index[-1])))
    return rows


class SpecialistJEPADataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        specialist_name: str,
        *,
        horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
        context_length: int = 128,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.specialist_name = specialist_name
        self.context_length = int(context_length)
        self.panel, self.feature_columns = _prepare_numeric_panel(panel, feature_columns=feature_columns)
        self.index = pd.DatetimeIndex(self.panel.index)
        self.sample_index = build_sample_index(
            self.index,
            context_length=self.context_length,
            horizons=horizons,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pos, horizon, target_start, target_end = self.sample_index[item]
        context = self.panel.iloc[pos - self.context_length + 1 : pos + 1].to_numpy(dtype=np.float32)
        target = self.panel.loc[target_start:target_end].to_numpy(dtype=np.float32)
        as_of = pd.Timestamp(self.index[pos])
        return {
            "context_features": context,
            "target_features": target,
            "specialist_name": self.specialist_name,
            "horizon": horizon,
            "as_of_timestamp": as_of,
            "target_start_timestamp": target_start,
            "target_end_timestamp": target_end,
        }


def jepa_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty JEPA batch.")
    context = torch.tensor(np.stack([row["context_features"] for row in batch]), dtype=torch.float32)
    max_target = max(int(row["target_features"].shape[0]) for row in batch)
    feature_dim = context.shape[-1]
    target = torch.zeros((len(batch), max_target, feature_dim), dtype=torch.float32)
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
        "specialist_name": [row["specialist_name"] for row in batch],
        "as_of_timestamp": [row["as_of_timestamp"] for row in batch],
        "target_start_timestamp": [row["target_start_timestamp"] for row in batch],
        "target_end_timestamp": [row["target_end_timestamp"] for row in batch],
    }


def live_context_array(
    panel: pd.DataFrame,
    *,
    context_length: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, pd.Timestamp]:
    numeric, _ = _prepare_numeric_panel(panel, feature_columns=feature_columns)
    if len(numeric) < int(context_length):
        raise ValueError(
            f"Need at least {context_length} rows for JEPA live encoding, got {len(numeric)}."
        )
    as_of = pd.Timestamp(numeric.index[-1])
    context = numeric.iloc[-int(context_length) :].to_numpy(dtype=np.float32)
    return np.expand_dims(context, axis=0), as_of
