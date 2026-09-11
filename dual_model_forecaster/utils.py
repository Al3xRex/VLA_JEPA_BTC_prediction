from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EPS = 1e-8


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Index):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    Path(path).write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def write_text(path: str | Path, payload: str) -> None:
    Path(path).write_text(payload)


def sigmoid(x: pd.Series | np.ndarray | float, scale: float = 1.0) -> pd.Series | np.ndarray | float:
    values = np.asarray(x) / max(scale, EPS)
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def positive_strength(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = np.tanh(series.astype(float) / max(scale, EPS))
    return pd.Series(np.clip(values, 0.0, 1.0), index=series.index, name=series.name)


def signed_strength(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = np.tanh(series.astype(float) / max(scale, EPS))
    return pd.Series(values, index=series.index, name=series.name)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator.astype(float) / denom.astype(float)


def safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.astype(float).where(series.astype(float) > 0))


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.astype(float).pct_change(periods=periods)


def robust_rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    if min_periods is None:
        min_periods = max(window // 4, 20)
    median = clean.rolling(window=window, min_periods=min_periods).median()
    mad = (clean - median).abs().rolling(window=window, min_periods=min_periods).median()
    scale = (1.4826 * mad).replace(0, np.nan)
    return ((clean - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 4, 5)
    return series.astype(float).rolling(window=window, min_periods=min_periods).std().fillna(0.0)


def rolling_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 4, 5)
    return series.astype(float).rolling(window=window, min_periods=min_periods).mean().fillna(0.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False).mean()


def consensus_score(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    dispersion = numeric.std(axis=1).fillna(0.0)
    return (1.0 / (1.0 + dispersion)).clip(0.0, 1.0)


def missing_fraction(frame: pd.DataFrame) -> pd.Series:
    return frame.isna().mean(axis=1).astype(float)


def compute_days_since_change(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    shifted = numeric.shift(1)
    equal_mask = numeric.eq(shifted) | (numeric.isna() & shifted.isna())
    changed = ~equal_mask.all(axis=1)
    changed.iloc[0] = True

    last_change = frame.index[0]
    out: list[float] = []
    for timestamp, did_change in zip(frame.index, changed):
        if bool(did_change):
            last_change = timestamp
        delta = pd.Timestamp(timestamp) - pd.Timestamp(last_change)
        out.append(float(delta.days))
    return pd.Series(out, index=frame.index, name="days_since_change")


def freshness_score(days_since_change: pd.Series, half_life_days: float) -> pd.Series:
    half_life = max(float(half_life_days), EPS)
    score = np.exp(-np.log(2.0) * days_since_change.astype(float) / half_life)
    return pd.Series(score, index=days_since_change.index, name="freshness")


def ewm_smooth_frame(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if alpha <= 0:
        return frame.copy()
    smoothed = frame.copy()
    for column in smoothed.columns:
        smoothed[column] = smoothed[column].ewm(alpha=alpha, adjust=False).mean()
    return smoothed


def ensure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(ensure_dir("/tmp/matplotlib-codex"))


def format_date(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def bounded_correlation(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.isna(corr):
        return 0.0
    return float(max(min(corr, 1.0), -1.0))


def quantile_name(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def root_mean_square(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.square(arr))))
