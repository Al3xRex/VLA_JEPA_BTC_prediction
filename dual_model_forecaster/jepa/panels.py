from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from dual_model_forecaster.brutal_baselines import build_kalman_state_history
from dual_model_forecaster.data import ForecastDataBundle
from dual_model_forecaster.jepa.feature_contract import (
    DEFAULT_HORIZONS,
    KALMAN_CONTEXT_FIELDS,
    SPECIALISTS,
    namespace_frame,
    namespaced,
)
from dual_model_forecaster.jepa.temporal_safety import validate_context_columns


def _causal_rolling_zscore(series: pd.Series, window: int = 365, min_periods: int = 60) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    mean = clean.rolling(window, min_periods=min_periods).mean().shift(1)
    std = clean.rolling(window, min_periods=min_periods).std().shift(1)
    return ((clean - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _time_since_condition(index: pd.Index, condition: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(index)
    condition = condition.reindex(index).fillna(False).astype(bool)
    out: list[float] = []
    last_seen: pd.Timestamp | None = None
    for timestamp, is_active in condition.items():
        timestamp = pd.Timestamp(timestamp)
        if bool(is_active):
            last_seen = timestamp
            out.append(0.0)
        elif last_seen is None:
            out.append(float("nan"))
        else:
            out.append(float((timestamp - last_seen).days))
    return pd.Series(out, index=index, dtype=float)


def build_kalman_context_panel(data_bundle: ForecastDataBundle) -> pd.DataFrame:
    state = build_kalman_state_history(data_bundle)
    if state.empty:
        return pd.DataFrame(index=data_bundle.close.index)
    state = state.reindex(data_bundle.close.index).sort_index()
    selected = pd.DataFrame(index=state.index)
    for column in KALMAN_CONTEXT_FIELDS:
        if column in state.columns:
            selected[column] = pd.to_numeric(state[column], errors="coerce")
        else:
            selected[column] = np.nan

    selected["gap_abs"] = selected["gap"].abs()
    selected["gap_sign"] = np.sign(selected["gap"].fillna(0.0))
    selected["gap_z_abs"] = selected["gap_z"].abs()
    selected["innovation_z_abs"] = selected["innovation_z"].abs()
    selected["gap_x_residual_sigma"] = selected["gap"].fillna(0.0) * selected["residual_sigma"].fillna(0.0)
    selected["gap_z_x_tail_flare"] = selected["gap_z"].fillna(0.0) * selected["tail_flare_score"].fillna(0.0)
    selected["gap_z_x_high_vol"] = selected["gap_z"].fillna(0.0) * selected["kalman_high_vol_signal"].fillna(0.0)
    selected["fair_value_drift_z"] = _causal_rolling_zscore(selected["fair_value_drift"])
    selected["projection_pressure_z"] = _causal_rolling_zscore(selected["kalman_projection_pressure"])
    selected["time_since_gap_z_gt_2"] = _time_since_condition(selected.index, selected["gap_z"] > 2.0)
    selected["time_since_gap_z_lt_minus_2"] = _time_since_condition(selected.index, selected["gap_z"] < -2.0)
    return namespace_frame(selected, "kalman")


def _specialist_id(name: str) -> int:
    return SPECIALISTS.index(name) if name in SPECIALISTS else -1


def build_specialist_panel(
    data_bundle: ForecastDataBundle,
    specialist_name: str,
    *,
    kalman_panel: pd.DataFrame | None = None,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    if specialist_name not in SPECIALISTS:
        raise ValueError(f"Unsupported specialist {specialist_name!r}; expected one of {SPECIALISTS}.")
    if specialist_name not in data_bundle.buckets:
        raise KeyError(f"Data bundle does not contain specialist bucket {specialist_name!r}.")

    index = data_bundle.close.index
    raw = data_bundle.buckets[specialist_name].reindex(index).sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw_panel = namespace_frame(raw, specialist_name)
    if kalman_panel is None:
        kalman_panel = build_kalman_context_panel(data_bundle)
    kalman_panel = kalman_panel.reindex(index)

    meta = pd.DataFrame(index=index)
    timestamp_index = pd.DatetimeIndex(index)
    meta[namespaced("jepa", "timestamp_ordinal")] = timestamp_index.map(pd.Timestamp.toordinal).astype(float)
    meta[namespaced("jepa", "specialist_name")] = specialist_name
    meta[namespaced("jepa", "specialist_id")] = float(_specialist_id(specialist_name))
    meta[namespaced("jepa", "horizon_min")] = float(min(int(h) for h in horizons))
    meta[namespaced("jepa", "horizon_max")] = float(max(int(h) for h in horizons))
    for horizon in horizons:
        meta[namespaced("jepa", f"horizon_{int(horizon)}d_available")] = 1.0

    panel = pd.concat([raw_panel, kalman_panel, meta], axis=1).sort_index()
    panel.index.name = "timestamp"
    validate_context_columns(panel.columns)
    return panel.replace([np.inf, -np.inf], np.nan)


def build_all_specialist_panels(
    data_bundle: ForecastDataBundle,
    *,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> dict[str, pd.DataFrame]:
    kalman_panel = build_kalman_context_panel(data_bundle)
    return {
        specialist: build_specialist_panel(
            data_bundle,
            specialist,
            kalman_panel=kalman_panel,
            horizons=horizons,
        )
        for specialist in SPECIALISTS
    }


def latest_specialist_panel_rows(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for specialist, frame in panels.items():
        if frame.empty:
            continue
        row = frame.tail(1).copy()
        row.insert(0, "specialist", specialist)
        row.insert(1, "timestamp", row.index.astype(str))
        rows.append(row.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
