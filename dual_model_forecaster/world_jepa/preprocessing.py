from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RobustWorldScaler:
    columns: list[str]
    median: np.ndarray
    scale: np.ndarray
    lower_clip: float = -12.0
    upper_clip: float = 12.0

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RobustWorldScaler":
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        median = numeric.median(axis=0).to_numpy(dtype=float)
        q25 = numeric.quantile(0.25, axis=0).to_numpy(dtype=float)
        q75 = numeric.quantile(0.75, axis=0).to_numpy(dtype=float)
        scale = q75 - q25
        fallback = numeric.std(axis=0, ddof=0).to_numpy(dtype=float)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        median = np.where(np.isfinite(median), median, 0.0)
        return cls(columns=numeric.columns.tolist(), median=median, scale=scale)

    def transform(self, frame: pd.DataFrame, *, fill_value: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing scaler columns: {missing[:10]}")
        numeric = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce")
        observed = numeric.notna() & np.isfinite(numeric)
        values = (numeric.to_numpy(dtype=float) - self.median) / self.scale
        values = np.clip(values, self.lower_clip, self.upper_clip)
        values[~observed.to_numpy()] = float(fill_value)
        scaled = pd.DataFrame(values, index=numeric.index, columns=self.columns)
        return scaled, observed.astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "lower_clip": float(self.lower_clip),
            "upper_clip": float(self.upper_clip),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustWorldScaler":
        return cls(
            columns=[str(value) for value in payload["columns"]],
            median=np.asarray(payload["median"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            lower_clip=float(payload.get("lower_clip", -12.0)),
            upper_clip=float(payload.get("upper_clip", 12.0)),
        )


def days_since_feature_change(frame: pd.DataFrame, *, atol: float = 1e-12) -> pd.DataFrame:
    """Return causal per-feature staleness in rows without back-looking from the future."""

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    result = np.zeros_like(values, dtype=float)
    if len(numeric) == 0:
        return pd.DataFrame(result, index=numeric.index, columns=numeric.columns)
    for column_index in range(values.shape[1]):
        age = 0.0
        previous = values[0, column_index]
        result[0, column_index] = 0.0
        for row_index in range(1, values.shape[0]):
            current = values[row_index, column_index]
            changed = (
                np.isfinite(current) != np.isfinite(previous)
                or (np.isfinite(current) and np.isfinite(previous) and not np.isclose(current, previous, atol=atol))
            )
            age = 0.0 if changed else age + 1.0
            result[row_index, column_index] = age
            previous = current
    return pd.DataFrame(result, index=numeric.index, columns=numeric.columns)


def days_since_observation(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the causal age of each feature's last finite observation.

    A current finite value has age zero. Consecutive missing values increment
    the age without peeking at the next observation. This is deliberately
    separate from :func:`days_since_feature_change`: a slowly changing macro
    series can be observed today while still having an old value.
    """

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    result = np.zeros_like(values, dtype=float)
    for column_index in range(values.shape[1]):
        age = 0.0
        for row_index in range(values.shape[0]):
            if np.isfinite(values[row_index, column_index]):
                age = 0.0
            elif row_index > 0:
                age += 1.0
            else:
                age = 1.0
            result[row_index, column_index] = age
    return pd.DataFrame(result, index=numeric.index, columns=numeric.columns)


def build_world_observation_channels(
    frame: pd.DataFrame,
    scaler: RobustWorldScaler,
    *,
    staleness_cap: float = 365.0,
) -> dict[str, pd.DataFrame]:
    scaled, observed = scaler.transform(frame)
    selected = frame.loc[:, scaler.columns]
    change_age = days_since_feature_change(selected).clip(0.0, float(staleness_cap))
    observation_age = days_since_observation(selected).clip(0.0, float(staleness_cap))
    denominator = np.log1p(float(staleness_cap))
    change_age = np.log1p(change_age) / denominator
    observation_age = np.log1p(observation_age) / denominator
    validity = observed * (1.0 - observation_age)
    velocity = scaled.diff()
    consecutive_observations = observed.astype(bool) & observed.astype(bool).shift(1, fill_value=False)
    velocity = velocity.where(consecutive_observations)
    return {
        "values": scaled,
        "observed": observed,
        "observation_age": observation_age,
        "change_age": change_age,
        # Backwards-compatible name for callers that used value-change age.
        "staleness": change_age,
        "velocity": velocity,
        "validity": validity,
    }


def build_world_model_frame(
    frame: pd.DataFrame,
    scaler: RobustWorldScaler,
    *,
    staleness_cap: float = 365.0,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build an enriched, mask-preserving model panel for one world.

    The raw feature budget is applied before this expansion. Each retained raw
    feature contributes a robustly scaled value, a causal first difference,
    time since observation, time since value change, and an explicit validity
    channel. Missing value/delta entries remain NaN so the encoder's feature
    mask cannot mistake an imputed zero for an observation.
    """

    channels = build_world_observation_channels(
        frame,
        scaler,
        staleness_cap=staleness_cap,
    )
    observed = channels["observed"].astype(bool)
    values = channels["values"].where(observed)
    velocity = channels["velocity"]
    parts = {
        "value": values,
        "velocity": velocity,
        "observation_age": channels["observation_age"],
        "change_age": channels["change_age"],
        "validity": channels["validity"],
    }
    expanded: list[pd.DataFrame] = []
    source_columns: dict[str, list[str]] = {}
    for channel_name, channel_frame in parts.items():
        renamed = channel_frame.copy()
        renamed.columns = [f"{channel_name}__{column}" for column in scaler.columns]
        expanded.append(renamed)
        source_columns[channel_name] = renamed.columns.tolist()
    model_frame = pd.concat(expanded, axis=1)
    model_frame.index = frame.index
    return model_frame, source_columns
