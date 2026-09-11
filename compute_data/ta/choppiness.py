from __future__ import annotations

import numpy as np
import pandas as pd

CHOP_TREND_LEVEL = 23.6
CHOP_CONSOLIDATION_LEVEL = 61.8
CHOP_EPSILON = 0.05
CHOP_TIMEFRAME_COLUMNS = ("chop_1d_14d", "chop_1w_10w", "chop_1m_10m")
CHOP_SHORT_HORIZON_WEIGHTS = {
    "chop_1d_14d": 0.55,
    "chop_1w_10w": 0.30,
    "chop_1m_10m": 0.15,
}


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(index, errors="coerce")
    if getattr(timestamps, "tz", None) is not None:
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(timestamps, name=getattr(index, "name", None) or "timestamp")


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _weighted_row_average(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=frame.index)
    weight_sum = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        valid = values.notna()
        weighted_sum = weighted_sum.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        weight_sum = weight_sum.add(valid.astype(float) * float(weight), fill_value=0.0)
    neutral = (CHOP_TREND_LEVEL + CHOP_CONSOLIDATION_LEVEL) / 2.0
    return (weighted_sum / weight_sum.replace(0.0, np.nan)).fillna(neutral)


def _trend_weight(chop_value: pd.Series) -> pd.Series:
    span = CHOP_CONSOLIDATION_LEVEL - CHOP_TREND_LEVEL
    return ((CHOP_CONSOLIDATION_LEVEL - chop_value) / span).clip(0.0, 1.0)


def build_choppiness_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build threshold-based trend vs mean-reversion features from Chop values."""
    out = frame.copy()
    if out.empty:
        out.index = _as_datetime_index(out.index)
        return out
    out.index = _as_datetime_index(out.index)
    out = out.loc[out.index.notna()].sort_index()

    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in CHOP_TIMEFRAME_COLUMNS:
        if column not in out.columns:
            continue
        trend = _trend_weight(out[column])
        mean_reversion = 1.0 - trend
        suffix = column.removeprefix("chop_")
        out[f"chop_{suffix}_trend_weight"] = trend
        out[f"chop_{suffix}_mean_reversion_weight"] = mean_reversion
        out[f"chop_{suffix}_regime_balance"] = mean_reversion - trend

    blend = _weighted_row_average(out, CHOP_SHORT_HORIZON_WEIGHTS)
    trend = _trend_weight(blend)
    mean_reversion = 1.0 - trend
    ratio = (mean_reversion + CHOP_EPSILON) / (trend + CHOP_EPSILON)

    out["chop_blend_1d_3d"] = blend
    out["chop_trend_weight"] = trend
    out["chop_mean_reversion_weight"] = mean_reversion
    out["chop_mr_trend_ratio"] = ratio
    out["chop_log_mr_trend_ratio"] = np.log(ratio)
    out["chop_regime_balance"] = mean_reversion - trend
    out["chop_regime_intensity"] = out["chop_regime_balance"].abs()
    out["chop_above_consolidation"] = (blend >= CHOP_CONSOLIDATION_LEVEL).astype(float)
    out["chop_below_trend"] = (blend <= CHOP_TREND_LEVEL).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def neutral_choppiness_features(index: pd.Index) -> pd.DataFrame:
    neutral = (CHOP_TREND_LEVEL + CHOP_CONSOLIDATION_LEVEL) / 2.0
    frame = pd.DataFrame(index=_as_datetime_index(index))
    for column in CHOP_TIMEFRAME_COLUMNS:
        frame[column] = neutral
    return build_choppiness_features(frame).fillna(
        {
            "chop_trend_weight": 0.5,
            "chop_mean_reversion_weight": 0.5,
            "chop_mr_trend_ratio": 1.0,
            "chop_log_mr_trend_ratio": 0.0,
            "chop_regime_balance": 0.0,
            "chop_regime_intensity": 0.0,
        }
    )


def _causal_hit_rate(signal: pd.Series, returns: pd.Series, horizon: int, window: int = 90) -> pd.Series:
    lagged_signal = signal.shift(int(horizon))
    aligned_returns = returns.reindex(lagged_signal.index)
    valid = lagged_signal.notna() & aligned_returns.notna() & (lagged_signal.abs() > 1e-12)
    hit = pd.Series(np.nan, index=lagged_signal.index, dtype=float)
    hit.loc[valid] = (
        np.sign(lagged_signal.loc[valid]) * np.sign(aligned_returns.loc[valid]) > 0.0
    ).astype(float)
    return hit.rolling(window, min_periods=max(20, window // 3)).mean().fillna(0.5)


def _causal_information_coefficient(
    signal: pd.Series,
    returns: pd.Series,
    horizon: int,
    window: int = 90,
) -> pd.Series:
    lagged_signal = signal.shift(int(horizon))
    aligned_returns = returns.reindex(lagged_signal.index)
    return (
        lagged_signal.rolling(window, min_periods=max(20, window // 3))
        .corr(aligned_returns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )


def build_strategy_influence_features(
    *,
    close: pd.Series,
    trend_signal: pd.Series,
    momentum_signal: pd.Series,
    mreg_signal: pd.Series,
    choppiness: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3),
    window: int = 90,
) -> pd.DataFrame:
    """Causal strategy pressure and trailing efficacy for short-horizon routing."""
    index = pd.DatetimeIndex(trend_signal.index, name="timestamp")
    close = pd.to_numeric(close.reindex(index), errors="coerce").astype(float)
    log_close = np.log(close.where(close > 0.0))

    trend_signal = pd.to_numeric(trend_signal.reindex(index), errors="coerce").fillna(0.0)
    momentum_signal = pd.to_numeric(momentum_signal.reindex(index), errors="coerce").fillna(0.0)
    mreg_signal = pd.to_numeric(mreg_signal.reindex(index), errors="coerce").fillna(0.0)
    choppiness = build_choppiness_features(choppiness.reindex(index)).ffill()
    trend_weight = _numeric_series(choppiness, "chop_trend_weight", 0.5).fillna(0.5).clip(0.0, 1.0)
    mean_reversion_weight = (
        _numeric_series(choppiness, "chop_mean_reversion_weight", 0.5).fillna(0.5).clip(0.0, 1.0)
    )

    trend_strategy_signal = trend_signal * trend_weight
    momentum_strategy_signal = momentum_signal * (0.50 + 0.50 * trend_weight)
    mean_reversion_strategy_signal = -trend_signal * mean_reversion_weight * (0.50 + 0.50 * mreg_signal.abs())

    out = pd.DataFrame(index=index)
    for horizon in horizons:
        horizon_i = int(horizon)
        realized = log_close.diff(horizon_i)
        out[f"strategy_trend_efficiency_{horizon_i}d"] = _causal_hit_rate(
            trend_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )
        out[f"strategy_momentum_efficiency_{horizon_i}d"] = _causal_hit_rate(
            momentum_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )
        out[f"strategy_mean_reversion_efficiency_{horizon_i}d"] = _causal_hit_rate(
            mean_reversion_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )
        out[f"strategy_trend_ic_{horizon_i}d"] = _causal_information_coefficient(
            trend_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )
        out[f"strategy_momentum_ic_{horizon_i}d"] = _causal_information_coefficient(
            momentum_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )
        out[f"strategy_mean_reversion_ic_{horizon_i}d"] = _causal_information_coefficient(
            mean_reversion_strategy_signal,
            realized,
            horizon_i,
            window=window,
        )

    trend_efficiency = out[[f"strategy_trend_efficiency_{int(h)}d" for h in horizons]].mean(axis=1)
    momentum_efficiency = out[[f"strategy_momentum_efficiency_{int(h)}d" for h in horizons]].mean(axis=1)
    mean_reversion_efficiency = out[
        [f"strategy_mean_reversion_efficiency_{int(h)}d" for h in horizons]
    ].mean(axis=1)

    pressure = pd.DataFrame(index=index)
    pressure["strategy_trend_pressure"] = trend_signal.abs() * trend_weight * (0.50 + trend_efficiency)
    pressure["strategy_momentum_pressure"] = (
        momentum_signal.abs() * (0.50 + 0.50 * trend_weight) * (0.50 + momentum_efficiency)
    )
    pressure["strategy_mean_reversion_pressure"] = (
        (0.65 * trend_signal.abs() + 0.35 * mreg_signal.abs())
        * mean_reversion_weight
        * (0.50 + mean_reversion_efficiency)
    )

    total_pressure = pressure.sum(axis=1).replace(0.0, np.nan)
    out = pd.concat([out, pressure], axis=1)
    out["strategy_trend_influence"] = (pressure["strategy_trend_pressure"] / total_pressure).fillna(1.0 / 3.0)
    out["strategy_momentum_influence"] = (
        pressure["strategy_momentum_pressure"] / total_pressure
    ).fillna(1.0 / 3.0)
    out["strategy_mean_reversion_influence"] = (
        pressure["strategy_mean_reversion_pressure"] / total_pressure
    ).fillna(1.0 / 3.0)

    influence = out[
        [
            "strategy_trend_influence",
            "strategy_momentum_influence",
            "strategy_mean_reversion_influence",
        ]
    ]
    ranked = np.sort(influence.to_numpy(dtype=float), axis=1)
    dominant = influence.idxmax(axis=1)
    out["strategy_dominant_trend"] = (dominant == "strategy_trend_influence").astype(float)
    out["strategy_dominant_momentum"] = (dominant == "strategy_momentum_influence").astype(float)
    out["strategy_dominant_mean_reversion"] = (
        dominant == "strategy_mean_reversion_influence"
    ).astype(float)
    out["strategy_dominance_margin"] = ranked[:, -1] - ranked[:, -2]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
