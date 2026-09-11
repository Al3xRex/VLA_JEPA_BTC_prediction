from __future__ import annotations

from collections.abc import Iterable
import re

import pandas as pd


SPECIALISTS = ("structure", "environment", "edges", "movement", "liquidation")
DEFAULT_HORIZONS = (1, 3, 7, 15)

KALMAN_CONTEXT_FIELDS = (
    "fair_value_log",
    "kalman_fair_value_price",
    "gap",
    "fair_value_drift",
    "residual_sigma",
    "gap_z",
    "innovation_z",
    "kalman_high_vol_signal",
    "tail_flare_score",
    "kalman_scope_days",
    "kalman_projection_pressure",
)

KALMAN_DERIVED_FIELDS = (
    "gap_abs",
    "gap_sign",
    "gap_z_abs",
    "innovation_z_abs",
    "gap_x_residual_sigma",
    "gap_z_x_tail_flare",
    "gap_z_x_high_vol",
    "fair_value_drift_z",
    "projection_pressure_z",
    "time_since_gap_z_gt_2",
    "time_since_gap_z_lt_minus_2",
)

PREFIXES = ("structure__", "environment__", "edges__", "movement__", "liquidation__", "kalman__", "jepa__")


def namespaced(name: str, column: str) -> str:
    if name not in (*SPECIALISTS, "kalman", "jepa"):
        raise ValueError(f"Unsupported namespace: {name}")
    clean = str(column).strip()
    return clean if clean.startswith(f"{name}__") else f"{name}__{clean}"


def namespace_frame(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    return frame.rename(columns={column: namespaced(namespace, column) for column in frame.columns})


def is_namespaced(column: str) -> bool:
    return str(column).startswith(PREFIXES)


def validate_namespaced_columns(columns: Iterable[str]) -> None:
    invalid = [column for column in columns if not is_namespaced(str(column))]
    if invalid:
        preview = ", ".join(map(str, invalid[:8]))
        raise ValueError(f"JEPA feature columns must be namespaced; invalid columns: {preview}")


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if is_namespaced(str(column)) and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return list(dict.fromkeys(columns))


FUTURE_COLUMN_PATTERN = re.compile(r"(^|__|_)(future|target|label|actual)(_|\b)", re.IGNORECASE)
CENTERED_COLUMN_PATTERN = re.compile(
    r"(centered|rolling_centered|full_sample|fullsample|global_z|globalz)",
    re.IGNORECASE,
)


def looks_future_aware(column: str) -> bool:
    return bool(FUTURE_COLUMN_PATTERN.search(str(column)))


def looks_centered_or_full_sample(column: str) -> bool:
    return bool(CENTERED_COLUMN_PATTERN.search(str(column)))
