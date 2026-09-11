from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any
import warnings

import numpy as np
import pandas as pd

from database_interraction import duckdb_connection, quote_identifier
from dual_model_forecaster.data import ForecastDataBundle
from dual_model_forecaster.metrics import pinball_loss, weighted_interval_score
from dual_model_forecaster.utils import ensure_dir, write_json


QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_COLUMNS = ["q05", "q25", "q50", "q75", "q95"]
MODEL_FORECASTER = "model"
LOWER_TAIL_FORECASTERS = {
    "model_lower_tail_residual",
    "model_regime_lower_tail_residual",
}
BASELINE_FORECASTERS = {
    "random_walk",
    "historical_vol_cone",
    "ewma_vol_cone",
    "garch_vol_cone",
    "egarch_vol_cone",
    "regime_conditioned_historical",
}
KALMAN_SYNTH_PREFIX = "kalman_synth"
KALMAN_MIN_SCOPE_DAYS = 7.0
KALMAN_MAX_SCOPE_DAYS = 240.0


@dataclass(frozen=True)
class EvaluationConfig:
    history_min: int = 180
    regime_history_min: int = 60
    ewma_span: int = 30
    garch_min_obs: int = 500
    garch_refit_interval: int = 30
    bootstrap_samples: int = 500
    bootstrap_seed: int = 7
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    include_garch: bool = True
    include_kalman_synthesis: bool = True
    kalman_min_history: int = 240


@dataclass(frozen=True)
class KalmanStateSpec:
    name: str
    phi: float
    q_fv_ratio: float
    q_dev_ratio: float
    r_ratio: float
    drift_span: int
    drift_weight: float
    mean_reversion_weight: float
    anchor_weight: float = 0.0


@dataclass(frozen=True)
class KalmanBlendSpec:
    name: str
    center_weight: float
    disagreement_width_weight: float
    uncertainty_width_weight: float
    tail_weight: float
    center_clip_widths: float
    width_mode: str = "scale"
    sigma_scale: float = 1.0
    high_vol_center_dampen: float = 0.0
    high_vol_sigma_boost: float = 0.0
    tail_flare_tail_boost: float = 0.0
    empirical_tail_weight: float = 0.0
    safety_width_scale: float = 0.0
    safety_high_vol_scale: float = 0.0


KALMAN_STATE_SPECS = [
    KalmanStateSpec(
        name="local_sticky",
        phi=0.85,
        q_fv_ratio=0.002,
        q_dev_ratio=0.35,
        r_ratio=0.20,
        drift_span=90,
        drift_weight=0.60,
        mean_reversion_weight=0.35,
    ),
    KalmanStateSpec(
        name="local_balanced",
        phi=0.70,
        q_fv_ratio=0.008,
        q_dev_ratio=0.55,
        r_ratio=0.15,
        drift_span=60,
        drift_weight=0.75,
        mean_reversion_weight=0.55,
    ),
    KalmanStateSpec(
        name="local_responsive",
        phi=0.55,
        q_fv_ratio=0.025,
        q_dev_ratio=0.80,
        r_ratio=0.12,
        drift_span=30,
        drift_weight=0.90,
        mean_reversion_weight=0.75,
    ),
    KalmanStateSpec(
        name="liquidity_anchor",
        phi=0.78,
        q_fv_ratio=0.004,
        q_dev_ratio=0.45,
        r_ratio=0.18,
        drift_span=90,
        drift_weight=0.65,
        mean_reversion_weight=0.50,
        anchor_weight=0.35,
    ),
    KalmanStateSpec(
        name="fundamental_slow",
        phi=0.98,
        q_fv_ratio=0.002,
        q_dev_ratio=0.25,
        r_ratio=5.50,
        drift_span=90,
        drift_weight=0.45,
        mean_reversion_weight=0.80,
    ),
]


KALMAN_BLEND_SPECS = [
    KalmanBlendSpec(
        name="light",
        center_weight=0.15,
        disagreement_width_weight=0.10,
        uncertainty_width_weight=0.05,
        tail_weight=0.10,
        center_clip_widths=0.75,
    ),
    KalmanBlendSpec(
        name="balanced",
        center_weight=0.30,
        disagreement_width_weight=0.20,
        uncertainty_width_weight=0.10,
        tail_weight=0.20,
        center_clip_widths=1.00,
    ),
    KalmanBlendSpec(
        name="assertive",
        center_weight=0.45,
        disagreement_width_weight=0.35,
        uncertainty_width_weight=0.18,
        tail_weight=0.35,
        center_clip_widths=1.25,
    ),
    KalmanBlendSpec(
        name="sigma_cap",
        center_weight=0.20,
        disagreement_width_weight=0.05,
        uncertainty_width_weight=0.05,
        tail_weight=0.15,
        center_clip_widths=0.85,
        width_mode="cap",
        sigma_scale=1.10,
    ),
    KalmanBlendSpec(
        name="sigma_cap_wide",
        center_weight=0.25,
        disagreement_width_weight=0.05,
        uncertainty_width_weight=0.08,
        tail_weight=0.25,
        center_clip_widths=1.00,
        width_mode="cap",
        sigma_scale=1.35,
    ),
    KalmanBlendSpec(
        name="sigma_replace",
        center_weight=0.35,
        disagreement_width_weight=0.00,
        uncertainty_width_weight=0.00,
        tail_weight=0.20,
        center_clip_widths=1.00,
        width_mode="replace",
        sigma_scale=1.20,
    ),
    KalmanBlendSpec(
        name="directional_anchor",
        center_weight=0.55,
        disagreement_width_weight=0.08,
        uncertainty_width_weight=0.05,
        tail_weight=0.22,
        center_clip_widths=1.15,
        width_mode="cap",
        sigma_scale=1.20,
        high_vol_center_dampen=0.10,
        high_vol_sigma_boost=0.30,
        tail_flare_tail_boost=0.45,
        empirical_tail_weight=0.30,
        safety_width_scale=1.60,
        safety_high_vol_scale=0.35,
    ),
    KalmanBlendSpec(
        name="directional_high_vol",
        center_weight=0.50,
        disagreement_width_weight=0.10,
        uncertainty_width_weight=0.08,
        tail_weight=0.35,
        center_clip_widths=1.10,
        width_mode="cap",
        sigma_scale=1.35,
        high_vol_center_dampen=0.18,
        high_vol_sigma_boost=0.55,
        tail_flare_tail_boost=0.85,
        empirical_tail_weight=0.55,
        safety_width_scale=1.85,
        safety_high_vol_scale=0.55,
    ),
    KalmanBlendSpec(
        name="directional_guarded",
        center_weight=0.42,
        disagreement_width_weight=0.06,
        uncertainty_width_weight=0.05,
        tail_weight=0.30,
        center_clip_widths=0.95,
        width_mode="cap",
        sigma_scale=1.25,
        high_vol_center_dampen=0.22,
        high_vol_sigma_boost=0.45,
        tail_flare_tail_boost=0.70,
        empirical_tail_weight=0.50,
        safety_width_scale=1.70,
        safety_high_vol_scale=0.50,
    ),
]


def load_walk_forward_predictions(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as connection:
        frame = connection.execute(
            f"SELECT * FROM {quote_identifier(table_name)} ORDER BY as_of, horizon;"
        ).df()
    for column in ("as_of", "target_timestamp"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["horizon", "base_close", "actual_return", *QUANTILE_COLUMNS]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _finite_quantile(values: pd.Series | np.ndarray, quantile: float, default: float = 0.0) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float(default)
    return float(np.quantile(clean, quantile))


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    aligned = pd.concat([x.astype(float), y.astype(float)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 5:
        return float("nan")
    if aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return float("nan")
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method)
    return float(value) if pd.notna(value) else float("nan")


def _return_series(close: pd.Series, return_type: str) -> pd.Series:
    close = close.astype(float).sort_index()
    if return_type == "log":
        returns = np.log(close / close.shift(1))
    else:
        returns = close.pct_change()
    returns.name = "daily_return"
    return returns.replace([np.inf, -np.inf], np.nan)


def _target_returns(close: pd.Series, horizon: int, return_type: str) -> pd.Series:
    close = close.astype(float).sort_index()
    shifted = close.shift(-int(horizon))
    if return_type == "log":
        values = np.log(shifted / close)
    else:
        values = shifted / close - 1.0
    values.name = f"target_{int(horizon)}d"
    return values.replace([np.inf, -np.inf], np.nan)


def _known_history(target_returns: pd.Series, as_of: pd.Timestamp, horizon: int) -> pd.Series:
    known_through = pd.Timestamp(as_of) - pd.Timedelta(days=int(horizon))
    return target_returns.loc[target_returns.index <= known_through].dropna()


def _rolling_iqr_zscore(series: pd.Series, window: int = 365, min_periods: int = 90) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    median = clean.rolling(window, min_periods=min_periods).median()
    q25 = clean.rolling(window, min_periods=min_periods).quantile(0.25)
    q75 = clean.rolling(window, min_periods=min_periods).quantile(0.75)
    scale = (q75 - q25).replace(0.0, np.nan)
    return ((clean - median) / scale).replace([np.inf, -np.inf], np.nan)


def _liquidity_score(environment: pd.DataFrame) -> pd.Series:
    positive_columns = [
        "fed_liquidity",
        "m2_yoy_usd",
        "m2_yoy_fixed_fx",
        "manual_m2_supply_of_four_major_central_banks_fixed_exchange_rate_yoy_r",
        "manual_m2_supply_of_four_major_central_banks_usd_yoy_r",
        "stocks",
        "liquidity_fair_value",
    ]
    tight_conditions_columns = ["manual_nfci", "manual_anfci"]
    parts: list[pd.Series] = []
    for column in positive_columns:
        if column in environment.columns:
            parts.append(_rolling_iqr_zscore(environment[column]))
    for column in tight_conditions_columns:
        if column in environment.columns:
            parts.append(-_rolling_iqr_zscore(environment[column]))
    if not parts:
        return pd.Series(0.0, index=environment.index, name="liquidity_score")
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0).rename("liquidity_score")


def build_btc_regimes(
    close: pd.Series,
    environment: pd.DataFrame | None,
    return_type: str,
) -> pd.DataFrame:
    close = close.astype(float).sort_index()
    daily_returns = _return_series(close, return_type)
    realized_vol = daily_returns.rolling(30, min_periods=20).std()
    vol_threshold = realized_vol.expanding(min_periods=180).median().shift(1)
    abs_returns = daily_returns.abs()
    tail_threshold = abs_returns.rolling(365, min_periods=90).quantile(0.95).shift(1)
    tail_ratio = (abs_returns / tail_threshold.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    tail_flare = tail_ratio >= 1.0
    tail_flare_pressure = (tail_ratio - 1.0).clip(lower=0.0, upper=4.0).ewm(
        span=10,
        min_periods=1,
        adjust=False,
    ).mean()
    trend_90d = close / close.rolling(90, min_periods=45).mean() - 1.0

    if environment is not None and not environment.empty:
        liquidity = _liquidity_score(environment.reindex(close.index))
    else:
        liquidity = pd.Series(0.0, index=close.index, name="liquidity_score")
    liquidity_threshold = liquidity.expanding(min_periods=180).median().shift(1)

    regimes = pd.DataFrame(index=close.index)
    regimes["realized_vol_30d"] = realized_vol
    regimes["tail_flare_ratio"] = tail_ratio
    regimes["tail_flare_pressure"] = tail_flare_pressure
    regimes["trend_90d"] = trend_90d
    regimes["liquidity_score"] = liquidity
    regimes["vol_regime"] = np.where((realized_vol > vol_threshold) | tail_flare.fillna(False), "high_vol", "low_vol")
    regimes.loc[vol_threshold.isna() | realized_vol.isna(), "vol_regime"] = "unknown_vol"
    regimes.loc[tail_flare.fillna(False) & realized_vol.notna(), "vol_regime"] = "high_vol"
    regimes["trend_regime"] = np.where(trend_90d >= 0.0, "uptrend", "downtrend")
    regimes.loc[trend_90d.isna(), "trend_regime"] = "unknown_trend"
    regimes["liquidity_regime"] = np.where(liquidity >= liquidity_threshold, "loose_liquidity", "tight_liquidity")
    regimes.loc[liquidity_threshold.isna() | liquidity.isna(), "liquidity_regime"] = "unknown_liquidity"
    regimes["regime_label"] = (
        regimes["vol_regime"].astype(str)
        + "|"
        + regimes["trend_regime"].astype(str)
        + "|"
        + regimes["liquidity_regime"].astype(str)
    )
    return regimes


def _causal_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    mean = clean.rolling(window, min_periods=min_periods).mean().shift(1)
    std = clean.rolling(window, min_periods=min_periods).std().shift(1)
    return ((clean - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _liquidity_anchor_log(close: pd.Series, environment: pd.DataFrame | None) -> pd.Series | None:
    if environment is None or "liquidity_fair_value" not in environment.columns:
        return None
    anchor = pd.to_numeric(environment["liquidity_fair_value"], errors="coerce").reindex(close.index).ffill()
    anchor = anchor.where(anchor > 0.0)
    if anchor.notna().sum() < 30:
        return None
    return np.log(anchor).ewm(span=30, min_periods=5, adjust=False).mean()


def _positive_log_series(frame: pd.DataFrame | None, column: str, index: pd.Index, span: int) -> pd.Series:
    if frame is None or column not in frame.columns:
        return pd.Series(np.nan, index=index, dtype=float)
    series = pd.to_numeric(frame[column], errors="coerce").reindex(index).ffill()
    series = series.where(series > 0.0)
    return np.log(series).ewm(span=span, min_periods=max(span // 5, 5), adjust=False).mean()


def _weighted_nanmean(parts: list[tuple[pd.Series, float]], index: pd.Index) -> pd.Series:
    numerator = pd.Series(0.0, index=index, dtype=float)
    denominator = pd.Series(0.0, index=index, dtype=float)
    for series, weight in parts:
        aligned = pd.to_numeric(series, errors="coerce").reindex(index)
        mask = aligned.notna()
        numerator.loc[mask] += aligned.loc[mask] * float(weight)
        denominator.loc[mask] += float(weight)
    return (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _weighted_series_average(
    parts: list[tuple[pd.Series, pd.Series | float]],
    index: pd.Index,
    default: pd.Series | float | None = None,
) -> pd.Series:
    numerator = pd.Series(0.0, index=index, dtype=float)
    denominator = pd.Series(0.0, index=index, dtype=float)
    for values, weights in parts:
        aligned_values = pd.to_numeric(values, errors="coerce").reindex(index)
        if isinstance(weights, pd.Series):
            aligned_weights = pd.to_numeric(weights, errors="coerce").reindex(index)
        else:
            aligned_weights = pd.Series(float(weights), index=index, dtype=float)
        aligned_weights = aligned_weights.clip(lower=0.0)
        mask = aligned_values.notna() & aligned_weights.notna() & aligned_weights.gt(0.0)
        numerator.loc[mask] += aligned_values.loc[mask] * aligned_weights.loc[mask]
        denominator.loc[mask] += aligned_weights.loc[mask]
    result = (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if default is None:
        return result
    if isinstance(default, pd.Series):
        return result.combine_first(pd.to_numeric(default, errors="coerce").reindex(index))
    return result.fillna(float(default))


def _tanh_score(series: pd.Series, scale: float) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    divisor = max(float(scale), 1e-9)
    return pd.Series(np.tanh(clean.to_numpy(dtype=float) / divisor), index=clean.index)


def _signalboost_average_attention(
    average: pd.Series,
    signalboost: pd.Series,
    *,
    average_window: int,
    average_min_periods: int,
    boost_window: int,
    boost_min_periods: int,
    impact_span: int,
) -> pd.DataFrame:
    index = average.index
    average = pd.to_numeric(average, errors="coerce").reindex(index).ffill()
    signalboost = pd.to_numeric(signalboost, errors="coerce").reindex(index).ffill()
    average_valid = average.notna()

    average_percentile = _causal_percentile_extreme(
        average,
        window=int(average_window),
        min_periods=int(average_min_periods),
    )
    average_z = _causal_robust_zscore(
        average,
        window=int(average_window),
        min_periods=int(average_min_periods),
    ).clip(-4.0, 4.0)
    average_position = _weighted_nanmean(
        [
            (average_percentile, 0.70),
            (_tanh_score(average_z, 2.0), 0.30),
        ],
        index=index,
    ).clip(-1.0, 1.0)
    average_level_fallback = _tanh_score(average, 1.0).clip(-1.0, 1.0).where(average_valid)
    average_position = average_position.combine_first(average_level_fallback)

    boost_abs = signalboost.abs().where(average_valid & signalboost.notna())
    boost_event = boost_abs.gt(1e-9)
    boost_abs_z = _causal_robust_zscore(
        boost_abs,
        window=int(boost_window),
        min_periods=int(boost_min_periods),
    ).clip(lower=0.0, upper=4.0)
    boost_abs_extreme = _causal_percentile_extreme(
        boost_abs,
        window=int(boost_window),
        min_periods=int(boost_min_periods),
    ).clip(lower=0.0).where(boost_event, 0.0)
    attention_gate_raw = _weighted_nanmean(
        [
            (boost_abs_extreme, 0.65),
            (_tanh_score(boost_abs_z, 2.0).clip(lower=0.0), 0.35),
        ],
        index=index,
    ).clip(0.0, 1.0).where(boost_event, 0.0)
    boost_strength_fallback = _tanh_score(boost_abs, 0.05).clip(0.0, 1.0).where(boost_event)
    attention_gate_raw = attention_gate_raw.combine_first(boost_strength_fallback).fillna(0.0)
    attention_gate = attention_gate_raw.ewm(
        span=max(int(impact_span), 2),
        min_periods=min(max(int(impact_span) // 4, 3), max(int(impact_span), 2)),
        adjust=False,
    ).mean().clip(0.0, 1.0)
    attention_score = (average_position * attention_gate).ewm(
        span=max(int(impact_span) // 3, 5),
        min_periods=3,
        adjust=False,
    ).mean().clip(-1.5, 1.5)
    return pd.DataFrame(
        {
            "average_position": average_position,
            "attention_gate": attention_gate,
            "attention_score": attention_score,
            "boost_abs_extreme": boost_abs_extreme,
            "average_available": average_valid.astype(float),
            "boost_event": boost_event.astype(float),
        },
        index=index,
    )


def _causal_lead_correlations(
    signal: pd.Series,
    log_close: pd.Series,
    candidate_days: tuple[int, ...],
    *,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    index = log_close.index
    aligned_signal = pd.to_numeric(signal, errors="coerce").reindex(index).ffill()
    aligned_close = pd.to_numeric(log_close, errors="coerce").reindex(index)
    correlations: dict[int, pd.Series] = {}
    for days in candidate_days:
        horizon = max(int(days), 1)
        future_return = (aligned_close.shift(-horizon) - aligned_close).replace([np.inf, -np.inf], np.nan)
        correlations[horizon] = (
            aligned_signal.rolling(int(window), min_periods=int(min_periods))
            .corr(future_return)
            .shift(horizon + 1)
        )
    return pd.DataFrame(correlations, index=index).replace([np.inf, -np.inf], np.nan)


def _causal_lead_impact_frame(
    signal: pd.Series,
    log_close: pd.Series,
    *,
    candidate_days: tuple[int, ...],
    default_days: int,
    window: int = 1095,
    min_periods: int = 365,
) -> pd.DataFrame:
    index = log_close.index
    candidates = tuple(max(int(day), 1) for day in candidate_days)
    default_days = max(int(default_days), 1)
    aligned_signal = pd.to_numeric(signal, errors="coerce").reindex(index).ffill()
    correlations = _causal_lead_correlations(
        aligned_signal,
        log_close,
        candidates,
        window=int(window),
        min_periods=int(min_periods),
    )

    positive_strength = correlations.clip(lower=0.0).pow(2)
    absolute_strength = correlations.abs().pow(2)
    strength = positive_strength.copy()
    use_absolute = positive_strength.sum(axis=1).le(0.0)
    if use_absolute.any():
        strength.loc[use_absolute] = absolute_strength.loc[use_absolute]
    strength_sum = strength.sum(axis=1).replace(0.0, np.nan)

    shifted = pd.DataFrame(
        {days: aligned_signal.shift(days) for days in candidates},
        index=index,
    )
    impact_score = ((shifted * strength).sum(axis=1) / strength_sum).replace([np.inf, -np.inf], np.nan)
    impact_score = impact_score.combine_first(aligned_signal.shift(default_days))

    days_vector = pd.Series({days: float(days) for days in candidates}, dtype=float)
    lead_days = ((strength * days_vector).sum(axis=1) / strength_sum).replace([np.inf, -np.inf], np.nan)

    filled_abs = correlations.abs().fillna(-np.inf)
    best_strength = filled_abs.max(axis=1).replace(-np.inf, np.nan)
    best_days = pd.to_numeric(filled_abs.idxmax(axis=1), errors="coerce").where(
        best_strength.notna(),
        float(default_days),
    )
    lead_days = lead_days.combine_first(best_days).fillna(float(default_days))

    best_corr = pd.Series(np.nan, index=index, dtype=float)
    for days in candidates:
        mask = best_days.eq(float(days))
        best_corr.loc[mask] = pd.to_numeric(correlations[days], errors="coerce").loc[mask]

    return pd.DataFrame(
        {
            "impact_score": impact_score,
            "lead_days": lead_days.clip(KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS),
            "best_lead_days": best_days.astype(float),
            "best_lead_corr": best_corr,
            "lead_strength": best_strength.fillna(0.0),
        },
        index=index,
    )


def _capped_balanced_log_level_ensemble(
    sources: dict[str, pd.Series],
    index: pd.Index,
    *,
    capped_source: str = "liquidity",
    cap_weight: float = 0.22,
    max_source_weight: float = 0.32,
    outlier_clip_log: float = 0.50,
    reference_log_close: pd.Series | None = None,
    value_weight: float = 0.40,
    value_horizons: tuple[int, ...] = (60, 120),
    value_window: int = 730,
    value_min_periods: int = 120,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {name: pd.to_numeric(series, errors="coerce").reindex(index) for name, series in sources.items()},
        index=index,
    )
    source_count = frame.notna().sum(axis=1)
    median = frame.median(axis=1, skipna=True)
    clipped = frame.clip(lower=median - outlier_clip_log, upper=median + outlier_clip_log, axis=0)

    value_scores = pd.DataFrame(0.0, index=index, columns=frame.columns, dtype=float)
    if reference_log_close is not None and value_weight > 0.0:
        reference = pd.to_numeric(reference_log_close, errors="coerce").reindex(index)
        for column in frame.columns:
            gap_signal = (pd.to_numeric(frame[column], errors="coerce") - reference).replace([np.inf, -np.inf], np.nan)
            horizon_scores: list[pd.Series] = []
            for horizon in value_horizons:
                horizon = max(int(horizon), 1)
                future_return = (reference.shift(-horizon) - reference).replace([np.inf, -np.inf], np.nan)
                corr = gap_signal.rolling(value_window, min_periods=value_min_periods).corr(future_return)
                # Shift by the forecast horizon so every correlation estimate only uses outcomes
                # that would have been known at the current timestamp.
                horizon_scores.append(corr.shift(horizon + 1).clip(lower=0.0, upper=0.75))
            if horizon_scores:
                value_scores[column] = pd.concat(horizon_scores, axis=1).mean(axis=1).fillna(0.0)

    weights = pd.DataFrame(0.0, index=index, columns=frame.columns, dtype=float)

    def _apply_caps(row_weights: dict[str, float]) -> dict[str, float]:
        if not row_weights:
            return row_weights
        columns = list(row_weights.keys())
        raw = np.asarray([max(float(row_weights[column]), 0.0) for column in columns], dtype=float)
        if raw.sum() <= 0.0:
            raw = np.ones_like(raw)
        raw = raw / raw.sum()
        caps = np.asarray(
            [
                float(cap_weight) if column == capped_source else float(max_source_weight)
                for column in columns
            ],
            dtype=float,
        )
        caps = np.clip(caps, 0.0, 1.0)
        if caps.sum() < 1.0:
            caps = caps / max(caps.sum(), 1e-12)
        values = raw.copy()
        for _ in range(8):
            over = values > caps
            if not over.any():
                break
            excess = float((values[over] - caps[over]).sum())
            values[over] = caps[over]
            under = ~over
            capacity = caps[under] - values[under]
            capacity = np.clip(capacity, 0.0, None)
            if excess <= 0.0 or capacity.sum() <= 0.0:
                break
            values[under] += excess * capacity / capacity.sum()
        values = values / max(values.sum(), 1e-12)
        return {column: float(weight) for column, weight in zip(columns, values)}

    for timestamp, count in source_count.items():
        if count <= 0:
            continue
        available = [column for column in frame.columns if pd.notna(frame.at[timestamp, column])]
        if not available:
            continue
        if count == 1:
            weights.loc[timestamp, available[0]] = 1.0
            continue
        equal_weight = 1.0 / float(count)
        equal_weights = {column: equal_weight for column in available}
        row_scores = value_scores.loc[timestamp, available].astype(float).clip(lower=0.0)
        if row_scores.sum() > 0.0:
            score_weights = (row_scores / row_scores.sum()).to_dict()
            blended_value_weight = float(np.clip(value_weight, 0.0, 0.85))
            row_weights = {
                column: (1.0 - blended_value_weight) * equal_weights[column]
                + blended_value_weight * float(score_weights.get(column, 0.0))
                for column in available
            }
        else:
            row_weights = equal_weights
        row_weights = _apply_caps(row_weights)
        total_weight = sum(row_weights.values())
        if total_weight > 0.0:
            for column, weight in row_weights.items():
                weights.loc[timestamp, column] = weight / total_weight

    ensemble = (clipped * weights).sum(axis=1).where(source_count > 0)
    robust_median = median.where(source_count > 1)
    ensemble = (0.75 * ensemble + 0.25 * robust_median).where(source_count > 1, ensemble)
    out = pd.DataFrame(
        {
            "level_anchor_log": ensemble.replace([np.inf, -np.inf], np.nan),
            "level_anchor_source_count": source_count.astype(float),
        },
        index=index,
    )
    for column in weights.columns:
        out[f"level_weight_{column}"] = weights[column]
        out[f"level_value_score_{column}"] = value_scores[column]
    return out


def _causal_robust_zscore(series: pd.Series, window: int = 730, min_periods: int = 180) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    median = clean.rolling(window, min_periods=min_periods).median().shift(1)
    q25 = clean.rolling(window, min_periods=min_periods).quantile(0.25).shift(1)
    q75 = clean.rolling(window, min_periods=min_periods).quantile(0.75).shift(1)
    scale = (q75 - q25).replace(0.0, np.nan)
    return ((clean - median) / scale).replace([np.inf, -np.inf], np.nan)


def _causal_percentile_extreme(series: pd.Series, window: int = 730, min_periods: int = 180) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    values = clean.to_numpy(dtype=float)
    out = np.full(len(clean), np.nan, dtype=float)
    for pos, value in enumerate(values):
        if not math.isfinite(value):
            continue
        start = max(0, pos - int(window))
        history = values[start:pos]
        history = history[np.isfinite(history)]
        if history.size < int(min_periods):
            continue
        percentile = float(np.mean(history <= value))
        if percentile >= 0.75:
            out[pos] = (percentile - 0.75) / 0.25
        elif percentile <= 0.25:
            out[pos] = -(0.25 - percentile) / 0.25
        else:
            out[pos] = 0.0
    return pd.Series(out, index=clean.index, name=clean.name).clip(-1.0, 1.0)


def _mean_available_zscores(frame: pd.DataFrame | None, columns: list[str], index: pd.Index, sign: float = 1.0) -> pd.Series:
    if frame is None:
        return pd.Series(np.nan, index=index, dtype=float)
    parts = []
    for column in columns:
        if column in frame.columns:
            z = _causal_robust_zscore(pd.to_numeric(frame[column], errors="coerce").reindex(index).ffill())
            parts.append(float(sign) * z.clip(-4.0, 4.0))
    if not parts:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1)


def _fundamental_anchor_frame(
    close: pd.Series,
    environment: pd.DataFrame | None,
    structure: pd.DataFrame | None,
) -> pd.DataFrame:
    close = close.astype(float).sort_index()
    index = close.index
    log_close = np.log(close.where(close > 0.0))

    liquidity_log = _positive_log_series(environment, "liquidity_fair_value", index, span=180)
    energy_log = _positive_log_series(structure, "energy_value", index, span=220)
    metcalfe_log = _positive_log_series(structure, "metcalfe_price", index, span=260)
    slow_price_log = log_close.rolling(730, min_periods=180).median().ewm(
        span=180,
        min_periods=30,
        adjust=False,
    ).mean()
    liquidity_level_log = liquidity_log
    energy_level_log = energy_log
    metcalfe_level_log = metcalfe_log

    liquidity_score = _mean_available_zscores(
        environment,
        [
            "fed_liquidity",
            "m2_yoy_usd",
            "m2_yoy_fixed_fx",
            "manual_m2_supply_of_four_major_central_banks_fixed_exchange_rate_yoy_r",
            "manual_m2_supply_of_four_major_central_banks_usd_yoy_r",
        ],
        index,
        sign=1.0,
    )
    risk_score = _mean_available_zscores(
        environment,
        [
            "stocks",
            "dxy_inverse",
            "gold",
            "emerging_markets",
        ],
        index,
        sign=1.0,
    )
    cycle_score = _mean_available_zscores(
        environment,
        [
            "business_activity",
            "manual_us_ism_manufacturing_pmi",
            "manual_us_ism_service_index",
            "oil",
        ],
        index,
        sign=1.0,
    )
    leverage_headwind = _mean_available_zscores(
        environment,
        ["manual_leverage", "manual_nfci", "manual_anfci", "manual_credit"],
        index,
        sign=-1.0,
    )
    financial_conditions_score = _mean_available_zscores(
        environment,
        ["manual_nfci", "manual_anfci"],
        index,
        sign=-1.0,
    )
    liquidity_lead = _causal_lead_impact_frame(
        liquidity_score,
        log_close,
        candidate_days=(14, 21, 28, 35, 42, 56, 70, 84),
        default_days=35,
        window=1095,
        min_periods=365,
    )
    cycle_lead = _causal_lead_impact_frame(
        cycle_score,
        log_close,
        candidate_days=(14, 21, 28, 35, 42, 56, 70, 84, 112),
        default_days=42,
        window=1095,
        min_periods=365,
    )
    fci_lead = _causal_lead_impact_frame(
        financial_conditions_score,
        log_close,
        candidate_days=(7, 14, 21, 28, 35, 42, 56, 70),
        default_days=28,
        window=1095,
        min_periods=365,
    )
    macro_cycle_score = _weighted_nanmean(
        [
            (liquidity_score, 0.28),
            (cycle_score, 0.26),
            (risk_score, 0.10),
            (financial_conditions_score.combine_first(leverage_headwind), 0.26),
            (leverage_headwind, 0.10),
        ],
        index=index,
    )
    macro_cycle_score = macro_cycle_score.ewm(span=90, min_periods=20, adjust=False).mean().clip(-4.0, 4.0)
    macro_impact_score = _weighted_nanmean(
        [
            (liquidity_lead["impact_score"], 0.36),
            (cycle_lead["impact_score"], 0.30),
            (fci_lead["impact_score"], 0.34),
        ],
        index=index,
    ).ewm(span=45, min_periods=10, adjust=False).mean().clip(-4.0, 4.0)
    macro_forward_score = _weighted_nanmean(
        [
            (liquidity_score, 0.36),
            (cycle_score, 0.30),
            (financial_conditions_score, 0.34),
        ],
        index=index,
    ).ewm(span=30, min_periods=10, adjust=False).mean().clip(-4.0, 4.0)
    macro_adjust = (0.12 * np.tanh(macro_impact_score / 1.75)).fillna(0.0)
    macro_forward_adjust = (0.10 * np.tanh(macro_forward_score / 1.75)).fillna(0.0)
    liquidity_activity = _tanh_score(liquidity_score.abs(), 1.50).clip(0.0, 1.0).fillna(0.0) + 0.10
    cycle_activity = _tanh_score(cycle_score.abs(), 1.50).clip(0.0, 1.0).fillna(0.0) + 0.10
    fci_activity = _tanh_score(financial_conditions_score.abs(), 1.50).clip(0.0, 1.0).fillna(0.0) + 0.10
    macro_scope_days = _weighted_series_average(
        [
            (liquidity_lead["lead_days"], liquidity_activity),
            (cycle_lead["lead_days"], cycle_activity),
            (fci_lead["lead_days"], fci_activity),
        ],
        index=index,
        default=35.0,
    ).clip(KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS)
    macro_activity = _weighted_series_average(
        [
            (liquidity_activity, 1.0),
            (cycle_activity, 1.0),
            (fci_activity, 1.0),
        ],
        index=index,
        default=0.10,
    ).clip(0.10, 1.10)

    ism_services = (
        pd.to_numeric(environment.get("manual_us_ism_service_index"), errors="coerce").reindex(index).ffill()
        if environment is not None and "manual_us_ism_service_index" in environment.columns
        else pd.Series(np.nan, index=index)
    )
    inverted_nfci = (
        -pd.to_numeric(environment.get("manual_nfci"), errors="coerce").reindex(index).ffill()
        if environment is not None and "manual_nfci" in environment.columns
        else pd.Series(np.nan, index=index)
    )
    ism_services_stationary_z = _causal_robust_zscore(ism_services, window=1095, min_periods=240).clip(-4.0, 4.0)
    inverted_nfci_stationary_z = _causal_robust_zscore(inverted_nfci, window=1095, min_periods=240).clip(-4.0, 4.0)
    ism_nfci_stationary_score = _weighted_nanmean(
        [
            (ism_services_stationary_z, 0.55),
            (inverted_nfci_stationary_z, 0.45),
        ],
        index=index,
    ).ewm(span=45, min_periods=10, adjust=False).mean().clip(-4.0, 4.0)
    btc_log_cycle_z = _causal_robust_zscore(log_close, window=1095, min_periods=365).clip(-4.0, 4.0)
    macro_stationary_gap_score = (ism_nfci_stationary_score - btc_log_cycle_z).ewm(
        span=45,
        min_periods=10,
        adjust=False,
    ).mean().clip(-4.0, 4.0)
    stationary_macro_adjust = (-0.16 * np.tanh(macro_stationary_gap_score / 1.75)).fillna(0.0)

    lth_average = (
        pd.to_numeric(structure.get("lth_average"), errors="coerce").reindex(index).ffill()
        if structure is not None and "lth_average" in structure.columns
        else pd.Series(np.nan, index=index)
    )
    sth_average = (
        pd.to_numeric(structure.get("sth_average"), errors="coerce").reindex(index).ffill()
        if structure is not None and "sth_average" in structure.columns
        else pd.Series(np.nan, index=index)
    )
    lth_boost = (
        pd.to_numeric(structure.get("lth_signalboost"), errors="coerce").reindex(index).ffill()
        if structure is not None and "lth_signalboost" in structure.columns
        else pd.Series(np.nan, index=index)
    )
    sth_boost = (
        pd.to_numeric(structure.get("sth_signalboost"), errors="coerce").reindex(index).ffill()
        if structure is not None and "sth_signalboost" in structure.columns
        else pd.Series(np.nan, index=index)
    )

    lth_attention = _signalboost_average_attention(
        lth_average,
        lth_boost,
        average_window=365,
        average_min_periods=20,
        boost_window=180,
        boost_min_periods=20,
        impact_span=56,
    )
    sth_attention = _signalboost_average_attention(
        sth_average,
        sth_boost,
        average_window=365,
        average_min_periods=20,
        boost_window=180,
        boost_min_periods=20,
        impact_span=21,
    )
    lth_average_behavior_z = lth_attention["average_position"]
    sth_average_behavior_z = sth_attention["average_position"]
    lth_signalboost_peak_score = lth_attention["attention_gate"]
    sth_signalboost_peak_score = sth_attention["attention_gate"]
    signalboost_peak_score = _weighted_nanmean(
        [
            (lth_signalboost_peak_score, 0.65),
            (sth_signalboost_peak_score, 0.35),
        ],
        index=index,
    ).ewm(span=21, min_periods=3, adjust=False).mean().clip(0.0, 1.0)
    lth_attention_score = lth_attention["attention_score"]
    sth_attention_score = sth_attention["attention_score"]
    lth_average_available = lth_attention["average_available"]
    sth_average_available = sth_attention["average_available"]
    lth_signalboost_event = lth_attention["boost_event"]
    sth_signalboost_event = sth_attention["boost_event"]
    onchain_average_behavior_score = _weighted_nanmean(
        [
            (lth_attention_score, 0.65),
            (sth_attention_score, 0.35),
        ],
        index=index,
    ).ewm(span=21, min_periods=3, adjust=False).mean().clip(-4.0, 4.0)
    onchain_behavior_score = onchain_average_behavior_score
    onchain_reversion_score = _weighted_nanmean(
        [
            (lth_attention_score, 0.70),
            (sth_attention_score, 0.30),
        ],
        index=index,
    ).ewm(span=45, min_periods=10, adjust=False).mean().clip(-4.0, 4.0)
    onchain_adjust = (-0.45 * np.tanh(onchain_reversion_score / 1.60)).fillna(0.0)
    lth_extreme = lth_attention["average_position"]
    sth_extreme = sth_attention["average_position"]
    lth_boost_extreme = lth_attention["boost_abs_extreme"]
    sth_boost_extreme = sth_attention["boost_abs_extreme"]

    lth_signalboost_lead = _causal_lead_impact_frame(
        lth_attention_score,
        log_close,
        candidate_days=(45, 60, 90, 120, 150, 180),
        default_days=120,
        window=365,
        min_periods=90,
    )
    sth_signalboost_lead = _causal_lead_impact_frame(
        sth_attention_score,
        log_close,
        candidate_days=(7, 14, 21, 28, 35, 45, 60, 90),
        default_days=35,
        window=730,
        min_periods=180,
    )
    signalboost_scope_days = _weighted_series_average(
        [
            (lth_signalboost_lead["lead_days"], lth_signalboost_peak_score),
            (sth_signalboost_lead["lead_days"], sth_signalboost_peak_score),
        ],
        index=index,
        default=45.0,
    ).clip(KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS)
    signalboost_activity = signalboost_peak_score.fillna(0.0).clip(0.0, 1.0)
    kalman_scope_days = _weighted_series_average(
        [
            (macro_scope_days, macro_activity),
            (signalboost_scope_days, signalboost_activity),
        ],
        index=index,
        default=macro_scope_days,
    ).clip(KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS)
    kalman_projection_pressure = (macro_forward_adjust + onchain_adjust).clip(-0.45, 0.45)

    macro_cycle_anchor_log = (slow_price_log + macro_adjust).replace([np.inf, -np.inf], np.nan)
    stationary_macro_anchor_log = (slow_price_log + stationary_macro_adjust).replace([np.inf, -np.inf], np.nan)
    onchain_reversion_anchor_log = (slow_price_log + onchain_adjust).replace([np.inf, -np.inf], np.nan)
    level_ensemble = _capped_balanced_log_level_ensemble(
        {
            "liquidity": liquidity_level_log,
            "energy": energy_level_log,
            "metcalfe": metcalfe_level_log,
        },
        index=index,
        capped_source="liquidity",
        cap_weight=0.20,
        max_source_weight=0.30,
        reference_log_close=log_close,
        value_weight=0.45,
        value_window=730,
        value_min_periods=120,
    )
    for signal_source in ("macro_cycle", "stationary_macro", "onchain"):
        for prefix in ("level_weight_", "level_value_score_"):
            column = f"{prefix}{signal_source}"
            if column not in level_ensemble.columns:
                level_ensemble[column] = 0.0
    level_anchor_log = level_ensemble["level_anchor_log"].combine_first(slow_price_log).ffill()

    anchor_log = level_anchor_log.ewm(
        span=85,
        min_periods=15,
        adjust=False,
    ).mean()
    fallback = slow_price_log.combine_first(log_close.ewm(span=365, min_periods=60, adjust=False).mean())
    anchor_log = anchor_log.combine_first(fallback).ffill()

    return pd.DataFrame(
        {
            "fundamental_anchor_log": anchor_log,
            "fundamental_level_anchor_log": level_anchor_log,
            "liquidity_anchor_log": liquidity_log,
            "energy_anchor_log": energy_log,
            "metcalfe_anchor_log": metcalfe_log,
            "balanced_liquidity_anchor_log": liquidity_level_log,
            "balanced_energy_anchor_log": energy_level_log,
            "balanced_metcalfe_anchor_log": metcalfe_level_log,
            "macro_cycle_anchor_log": macro_cycle_anchor_log,
            "stationary_macro_anchor_log": stationary_macro_anchor_log,
            "onchain_reversion_anchor_log": onchain_reversion_anchor_log,
            "level_anchor_source_count": level_ensemble["level_anchor_source_count"],
            "level_weight_liquidity": level_ensemble["level_weight_liquidity"],
            "level_weight_energy": level_ensemble["level_weight_energy"],
            "level_weight_metcalfe": level_ensemble["level_weight_metcalfe"],
            "level_weight_macro_cycle": level_ensemble["level_weight_macro_cycle"],
            "level_weight_stationary_macro": level_ensemble["level_weight_stationary_macro"],
            "level_weight_onchain": level_ensemble["level_weight_onchain"],
            "level_value_score_liquidity": level_ensemble["level_value_score_liquidity"],
            "level_value_score_energy": level_ensemble["level_value_score_energy"],
            "level_value_score_metcalfe": level_ensemble["level_value_score_metcalfe"],
            "level_value_score_macro_cycle": level_ensemble["level_value_score_macro_cycle"],
            "level_value_score_stationary_macro": level_ensemble["level_value_score_stationary_macro"],
            "level_value_score_onchain": level_ensemble["level_value_score_onchain"],
            "liquidity_macro_score": liquidity_score,
            "risk_macro_score": risk_score,
            "cycle_macro_score": cycle_score,
            "leverage_macro_score": leverage_headwind,
            "financial_conditions_macro_score": financial_conditions_score,
            "macro_cycle_score": macro_cycle_score,
            "macro_impact_score": macro_impact_score,
            "macro_forward_score": macro_forward_score,
            "macro_adjust": macro_adjust,
            "macro_forward_adjust": macro_forward_adjust,
            "liquidity_lead_days": liquidity_lead["lead_days"],
            "liquidity_lead_corr": liquidity_lead["best_lead_corr"],
            "cycle_lead_days": cycle_lead["lead_days"],
            "cycle_lead_corr": cycle_lead["best_lead_corr"],
            "financial_conditions_lead_days": fci_lead["lead_days"],
            "financial_conditions_lead_corr": fci_lead["best_lead_corr"],
            "macro_scope_days": macro_scope_days,
            "ism_services_stationary_z": ism_services_stationary_z,
            "inverted_nfci_stationary_z": inverted_nfci_stationary_z,
            "ism_nfci_stationary_score": ism_nfci_stationary_score,
            "btc_log_cycle_z": btc_log_cycle_z,
            "macro_stationary_gap_score": macro_stationary_gap_score,
            "stationary_macro_adjust": stationary_macro_adjust,
            "onchain_reversion_score": onchain_reversion_score,
            "onchain_average_behavior_score": onchain_average_behavior_score,
            "lth_average": lth_average,
            "lth_signalboost": lth_boost,
            "sth_average": sth_average,
            "sth_signalboost": sth_boost,
            "signalboost_peak_score": signalboost_peak_score,
            "onchain_behavior_score": onchain_behavior_score,
            "lth_average_behavior_z": lth_average_behavior_z,
            "sth_average_behavior_z": sth_average_behavior_z,
            "lth_signalboost_peak_score": lth_signalboost_peak_score,
            "sth_signalboost_peak_score": sth_signalboost_peak_score,
            "lth_signalboost_attention_score": lth_attention_score,
            "sth_signalboost_attention_score": sth_attention_score,
            "lth_average_available": lth_average_available,
            "sth_average_available": sth_average_available,
            "lth_signalboost_event": lth_signalboost_event,
            "sth_signalboost_event": sth_signalboost_event,
            "lth_signalboost_lead_days": lth_signalboost_lead["lead_days"],
            "lth_signalboost_lead_corr": lth_signalboost_lead["best_lead_corr"],
            "sth_signalboost_lead_days": sth_signalboost_lead["lead_days"],
            "sth_signalboost_lead_corr": sth_signalboost_lead["best_lead_corr"],
            "signalboost_scope_days": signalboost_scope_days,
            "onchain_adjust": onchain_adjust,
            "kalman_scope_days": kalman_scope_days,
            "kalman_projection_pressure": kalman_projection_pressure,
            "lth_extreme_score": lth_extreme,
            "sth_extreme_score": sth_extreme,
            "lth_boost_extreme_score": lth_boost_extreme,
            "sth_boost_extreme_score": sth_boost_extreme,
        },
        index=index,
    )


def _fundamental_slow_kalman_level_state(
    close: pd.Series,
    environment: pd.DataFrame | None,
    structure: pd.DataFrame | None,
    spec: KalmanStateSpec,
) -> pd.DataFrame:
    close = close.astype(float).sort_index()
    close = close.loc[close > 0.0].dropna()
    if close.empty:
        return pd.DataFrame()

    log_close = np.log(close)
    returns = log_close.diff()
    rolling_var = returns.rolling(365, min_periods=120).var().shift(1)
    expanding_var = returns.expanding(min_periods=120).var().shift(1)
    variance = rolling_var.combine_first(expanding_var).fillna(1e-4).clip(lower=1e-8)
    anchor = _fundamental_anchor_frame(close, environment, structure).reindex(log_close.index)

    first_anchor = anchor["fundamental_anchor_log"].dropna()
    m_prev = float(first_anchor.iloc[0]) if not first_anchor.empty else float(log_close.iloc[0])
    p_prev = 0.05
    fair_values: list[float] = []
    state_vars: list[float] = []
    innovations: list[float] = []
    innovation_vars: list[float] = []

    for timestamp, y_value in log_close.items():
        step_var = float(variance.loc[timestamp])
        anchor_value = _finite_float(anchor.at[timestamp, "fundamental_anchor_log"], m_prev)
        onchain_score = abs(_finite_float(anchor.at[timestamp, "onchain_reversion_score"], 0.0))
        gap_pressure = min(abs(_finite_float(anchor_value - m_prev, 0.0)), 0.50)
        anchor_pull = float(
            np.clip(
                0.012 + 0.012 * min(onchain_score, 4.0) + 0.020 * gap_pressure,
                0.008,
                0.075,
            )
        )
        q = max(step_var * float(spec.q_fv_ratio), 1e-10)
        r = max(step_var * float(spec.r_ratio), 1e-5)

        m_pred = (1.0 - anchor_pull) * m_prev + anchor_pull * anchor_value
        p_pred = p_prev + q
        innovation = float(y_value - m_pred)
        innovation_var = float(p_pred + r)
        if not math.isfinite(innovation_var) or innovation_var <= 0.0:
            innovation_var = 1e-5
        gain = min(max(p_pred / innovation_var, 0.0), 0.040)
        m_prev = m_pred + gain * innovation
        p_prev = max((1.0 - gain) * p_pred, 1e-8)

        fair_values.append(float(m_prev))
        state_vars.append(float(p_prev))
        innovations.append(innovation)
        innovation_vars.append(innovation_var)

    out = pd.DataFrame(
        {
            "log_close": log_close,
            "fair_value_log_raw": fair_values,
            "fair_value_log": fair_values,
            "deviation_raw": log_close.to_numpy(dtype=float) - np.asarray(fair_values, dtype=float),
            "state_var_fv": state_vars,
            "state_var_dev": np.zeros(len(fair_values), dtype=float),
            "innovation": innovations,
            "innovation_var": innovation_vars,
            "return_var": variance.reindex(log_close.index),
        },
        index=log_close.index,
    )
    out = out.join(anchor, how="left")
    out["gap"] = out["log_close"] - out["fair_value_log"]
    out["fair_value_drift"] = out["fair_value_log"].diff().ewm(
        span=max(int(spec.drift_span), 2),
        min_periods=max(min(int(spec.drift_span) // 4, 30), 2),
        adjust=False,
    ).mean()
    residual_sigma = out["gap"].rolling(365, min_periods=120).std().shift(1)
    return_sigma = returns.rolling(180, min_periods=60).std().shift(1)
    out["residual_sigma"] = residual_sigma.combine_first(return_sigma).fillna(np.sqrt(out["return_var"]))
    out["residual_sigma"] = out["residual_sigma"].clip(lower=1e-6)
    out["gap_z"] = _causal_zscore(out["gap"], window=730, min_periods=180).fillna(0.0)
    out["innovation_z"] = (
        out["innovation"] / np.sqrt(out["innovation_var"].clip(lower=1e-8))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["state_uncertainty"] = np.sqrt(out["state_var_fv"].clip(lower=0.0))

    abs_returns = returns.abs().reindex(out.index)
    tail_threshold = abs_returns.rolling(365, min_periods=90).quantile(0.95).shift(1)
    tail_ratio = (abs_returns / tail_threshold.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return_tail_z = _causal_zscore(abs_returns, window=365, min_periods=90).fillna(0.0)
    realized_vol = returns.rolling(30, min_periods=20).std().reindex(out.index)
    vol_threshold = realized_vol.expanding(min_periods=180).median().shift(1)
    vol_ratio = (realized_vol / vol_threshold.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    tail_flare_raw = np.maximum(tail_ratio.fillna(0.0) - 1.0, 0.0)
    tail_flare_raw += 0.25 * np.maximum(return_tail_z - 2.0, 0.0)
    out["tail_flare_score"] = tail_flare_raw.clip(lower=0.0, upper=5.0)
    high_vol_raw = (
        (vol_ratio.fillna(0.0) > 1.0)
        | (tail_ratio.fillna(0.0) >= 1.0)
        | (return_tail_z >= 2.5)
    ).astype(float)
    out["kalman_high_vol_signal"] = high_vol_raw.ewm(span=10, min_periods=1, adjust=False).mean().clip(0.0, 1.0)
    out["return_tail_z"] = return_tail_z
    out["vol_ratio"] = vol_ratio
    out["history_count"] = np.arange(1, len(out) + 1)
    return out


def _two_state_kalman_level_state(
    close: pd.Series,
    environment: pd.DataFrame | None,
    spec: KalmanStateSpec,
    structure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if spec.name == "fundamental_slow":
        return _fundamental_slow_kalman_level_state(close, environment, structure, spec)

    close = close.astype(float).sort_index()
    close = close.loc[close > 0.0].dropna()
    if close.empty:
        return pd.DataFrame()

    log_close = np.log(close)
    returns = log_close.diff()
    rolling_var = returns.rolling(180, min_periods=60).var().shift(1)
    expanding_var = returns.expanding(min_periods=60).var().shift(1)
    variance = rolling_var.combine_first(expanding_var).fillna(1e-4).clip(lower=1e-8)

    m_prev = np.array([float(log_close.iloc[0]), 0.0], dtype=float)
    p_prev = np.diag([0.02, 0.02]).astype(float)
    h_obs = np.array([[1.0, 1.0]], dtype=float)
    identity = np.eye(2)

    fair_values: list[float] = []
    deviations: list[float] = []
    state_var_fv: list[float] = []
    state_var_dev: list[float] = []
    innovations: list[float] = []
    innovation_vars: list[float] = []

    for timestamp, y_value in log_close.items():
        step_var = float(variance.loc[timestamp])
        q_fv = max(step_var * float(spec.q_fv_ratio), 1e-9)
        q_dev = max(step_var * float(spec.q_dev_ratio), 1e-9)
        r_obs = max(step_var * float(spec.r_ratio), 1e-8)

        transition = np.array([[1.0, 0.0], [0.0, float(spec.phi)]], dtype=float)
        process = np.diag([q_fv, q_dev]).astype(float)

        m_pred = transition @ m_prev
        p_pred = transition @ p_prev @ transition.T + process
        y_pred = float((h_obs @ m_pred).item())
        innovation = float(y_value - y_pred)
        innovation_var = float((h_obs @ p_pred @ h_obs.T).item() + r_obs)
        if not math.isfinite(innovation_var) or innovation_var <= 0.0:
            innovation_var = 1e-8

        gain = (p_pred @ h_obs.T) / innovation_var
        m_prev = m_pred + gain.flatten() * innovation
        p_prev = (identity - gain @ h_obs) @ p_pred

        fair_values.append(float(m_prev[0]))
        deviations.append(float(m_prev[1]))
        state_var_fv.append(float(max(p_prev[0, 0], 0.0)))
        state_var_dev.append(float(max(p_prev[1, 1], 0.0)))
        innovations.append(innovation)
        innovation_vars.append(innovation_var)

    out = pd.DataFrame(
        {
            "log_close": log_close,
            "fair_value_log_raw": fair_values,
            "deviation_raw": deviations,
            "state_var_fv": state_var_fv,
            "state_var_dev": state_var_dev,
            "innovation": innovations,
            "innovation_var": innovation_vars,
            "return_var": variance.reindex(log_close.index),
        },
        index=log_close.index,
    )
    out["fair_value_log"] = out["fair_value_log_raw"]

    anchor_log = _liquidity_anchor_log(close, environment)
    if anchor_log is not None and spec.anchor_weight > 0.0:
        anchor = anchor_log.reindex(out.index)
        has_anchor = anchor.notna()
        weight = float(np.clip(spec.anchor_weight, 0.0, 0.95))
        out.loc[has_anchor, "fair_value_log"] = (
            (1.0 - weight) * out.loc[has_anchor, "fair_value_log_raw"]
            + weight * anchor.loc[has_anchor]
        )

    out["gap"] = out["log_close"] - out["fair_value_log"]
    out["fair_value_drift"] = out["fair_value_log"].diff().ewm(
        span=max(int(spec.drift_span), 2),
        min_periods=max(min(int(spec.drift_span) // 4, 20), 2),
        adjust=False,
    ).mean()
    residual_sigma = out["gap"].rolling(180, min_periods=60).std().shift(1)
    return_sigma = returns.rolling(180, min_periods=60).std().shift(1)
    out["residual_sigma"] = residual_sigma.combine_first(return_sigma).fillna(np.sqrt(out["return_var"]))
    out["residual_sigma"] = out["residual_sigma"].clip(lower=1e-6)
    out["gap_z"] = _causal_zscore(out["gap"], window=365, min_periods=90).fillna(0.0)
    out["innovation_z"] = (
        out["innovation"] / np.sqrt(out["innovation_var"].clip(lower=1e-8))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["state_uncertainty"] = np.sqrt((out["state_var_fv"] + out["state_var_dev"]).clip(lower=0.0))
    abs_returns = returns.abs().reindex(out.index)
    tail_threshold = abs_returns.rolling(365, min_periods=90).quantile(0.95).shift(1)
    tail_ratio = (abs_returns / tail_threshold.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return_tail_z = _causal_zscore(abs_returns, window=365, min_periods=90).fillna(0.0)
    realized_vol = returns.rolling(30, min_periods=20).std().reindex(out.index)
    vol_threshold = realized_vol.expanding(min_periods=180).median().shift(1)
    vol_ratio = (realized_vol / vol_threshold.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    tail_flare_raw = np.maximum(tail_ratio.fillna(0.0) - 1.0, 0.0)
    tail_flare_raw += 0.35 * np.maximum(np.abs(out["innovation_z"]) - 2.0, 0.0)
    tail_flare_raw += 0.25 * np.maximum(return_tail_z - 2.0, 0.0)
    out["tail_flare_score"] = tail_flare_raw.clip(lower=0.0, upper=5.0)
    high_vol_raw = (
        (vol_ratio.fillna(0.0) > 1.0)
        | (tail_ratio.fillna(0.0) >= 1.0)
        | (np.abs(out["innovation_z"]) >= 2.0)
    ).astype(float)
    out["kalman_high_vol_signal"] = high_vol_raw.ewm(span=10, min_periods=1, adjust=False).mean().clip(0.0, 1.0)
    out["return_tail_z"] = return_tail_z
    out["vol_ratio"] = vol_ratio
    out["history_count"] = np.arange(1, len(out) + 1)
    return out


def _horizon_weight(base_weight: float, horizon: int) -> float:
    scale = math.sqrt(max(int(horizon), 1) / 7.0)
    return float(np.clip(base_weight * np.clip(scale, 0.35, 1.35), 0.0, 0.90))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _kalman_return_projection(state_row: pd.Series, horizon: int, spec: KalmanStateSpec) -> float:
    drift = _finite_float(state_row.get("fair_value_drift"), 0.0)
    gap = _finite_float(state_row.get("gap"), 0.0)
    scope_days = _finite_float(state_row.get("kalman_scope_days"), float(horizon))
    scope_days = float(np.clip(scope_days, KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS))
    projection_pressure = _finite_float(state_row.get("kalman_projection_pressure"), 0.0)
    horizon_phi = float(spec.phi) ** max(scope_days, 1.0)
    return (
        float(spec.drift_weight) * drift * scope_days
        - float(spec.mean_reversion_weight) * (1.0 - horizon_phi) * gap
        + projection_pressure
    )


def _empirical_calibration_reference(
    model_rows: pd.DataFrame,
    close: pd.Series,
    return_type: str,
    regimes: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    if model_rows.empty:
        return pd.DataFrame(index=model_rows.index)

    horizons = sorted(int(value) for value in model_rows["horizon"].dropna().unique())
    targets_by_horizon = {
        horizon: _target_returns(close, horizon, return_type)
        for horizon in horizons
    }
    regime_labels = regimes["vol_regime"] if "vol_regime" in regimes.columns else pd.Series(dtype=object)
    rows: list[dict[str, Any]] = []

    for index, row in model_rows.iterrows():
        horizon = int(row["horizon"])
        as_of = pd.Timestamp(row["as_of"])
        history = _known_history(targets_by_horizon[horizon], as_of, horizon)
        selected = history
        reference_regime = "all"
        current_vol_regime = str(row.get("vol_regime", "unknown_vol"))

        if len(history) >= int(config.history_min) and current_vol_regime in {"high_vol", "low_vol"}:
            history_regimes = regime_labels.reindex(history.index)
            same_regime = history.loc[history_regimes == current_vol_regime]
            if len(same_regime) >= int(config.regime_history_min):
                selected = same_regime
                reference_regime = current_vol_regime

        if len(selected) < int(config.regime_history_min):
            values = {f"emp_{column}": np.nan for column in QUANTILE_COLUMNS}
        else:
            values = {
                f"emp_{column}": _finite_quantile(selected, quantile)
                for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES)
            }
        values["emp_reference_regime"] = reference_regime
        values["emp_reference_count"] = int(len(selected))
        values["emp_reference_index"] = index
        rows.append(values)

    reference = pd.DataFrame(rows).set_index("emp_reference_index")
    for column in [f"emp_{quantile_column}" for quantile_column in QUANTILE_COLUMNS]:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
    reference["emp_width_50"] = reference["emp_q75"] - reference["emp_q25"]
    reference["emp_width_90"] = reference["emp_q95"] - reference["emp_q05"]
    return reference.reindex(model_rows.index)


def _synthesize_kalman_variant(
    model_rows: pd.DataFrame,
    state_frame: pd.DataFrame,
    state_spec: KalmanStateSpec,
    blend_spec: KalmanBlendSpec,
    min_history: int,
    empirical_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if model_rows.empty or state_frame.empty:
        return pd.DataFrame()

    forecaster = f"{KALMAN_SYNTH_PREFIX}_{state_spec.name}_{blend_spec.name}"
    out = model_rows.copy()
    out["forecaster"] = forecaster
    out["kalman_state_spec"] = state_spec.name
    out["kalman_blend_spec"] = blend_spec.name
    out["kalman_variant"] = forecaster
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    for large_column in [
        "width_diagnostics_json",
        "lower_tail_diagnostics_json",
        "upper_tail_diagnostics_json",
        "calibration_score_comparison_json",
    ]:
        if large_column in out.columns:
            out[large_column] = None

    state_columns = [
        "fair_value_drift",
        "gap",
        "residual_sigma",
        "innovation_z",
        "state_uncertainty",
        "gap_z",
        "tail_flare_score",
        "kalman_high_vol_signal",
        "return_tail_z",
        "vol_ratio",
        "kalman_scope_days",
        "kalman_projection_pressure",
        "history_count",
    ]
    available_state_columns = [column for column in state_columns if column in state_frame.columns]
    work = out.join(state_frame[available_state_columns], on="as_of")
    if empirical_reference is not None and not empirical_reference.empty:
        empirical_columns = [
            "emp_q05",
            "emp_q25",
            "emp_q50",
            "emp_q75",
            "emp_q95",
            "emp_width_50",
            "emp_width_90",
            "emp_reference_regime",
            "emp_reference_count",
        ]
        available_empirical_columns = [
            column for column in empirical_columns if column in empirical_reference.columns
        ]
        work = work.join(empirical_reference[available_empirical_columns], how="left")
    numeric_quantiles = work[QUANTILE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    horizons = pd.to_numeric(work["horizon"], errors="coerce").fillna(1).astype(int).clip(lower=1)

    valid = (
        numeric_quantiles.notna().all(axis=1)
        & pd.to_numeric(work["history_count"], errors="coerce").fillna(0).ge(int(min_history))
        & work["fair_value_drift"].notna()
        & work["gap"].notna()
    )
    if not valid.any():
        return out

    q05 = numeric_quantiles["q05"].to_numpy(dtype=float)
    q25 = numeric_quantiles["q25"].to_numpy(dtype=float)
    q50 = numeric_quantiles["q50"].to_numpy(dtype=float)
    q75 = numeric_quantiles["q75"].to_numpy(dtype=float)
    q95 = numeric_quantiles["q95"].to_numpy(dtype=float)

    half_width = np.maximum((q75 - q25) / 2.0, 1e-6)
    lower_extension = np.maximum(q25 - q05, 1e-6)
    upper_extension = np.maximum(q95 - q75, 1e-6)
    width_90 = np.maximum.reduce([q95 - q05, 2.0 * half_width, np.full_like(half_width, 1e-6)])
    emp_q05 = pd.to_numeric(work.get("emp_q05"), errors="coerce").to_numpy(dtype=float) if "emp_q05" in work else np.full_like(q50, np.nan)
    emp_q25 = pd.to_numeric(work.get("emp_q25"), errors="coerce").to_numpy(dtype=float) if "emp_q25" in work else np.full_like(q50, np.nan)
    emp_q75 = pd.to_numeric(work.get("emp_q75"), errors="coerce").to_numpy(dtype=float) if "emp_q75" in work else np.full_like(q50, np.nan)
    emp_q95 = pd.to_numeric(work.get("emp_q95"), errors="coerce").to_numpy(dtype=float) if "emp_q95" in work else np.full_like(q50, np.nan)
    emp_half_width = np.maximum((emp_q75 - emp_q25) / 2.0, 1e-6)
    emp_lower_extension = np.maximum(emp_q25 - emp_q05, 1e-6)
    emp_upper_extension = np.maximum(emp_q95 - emp_q75, 1e-6)
    has_empirical = (
        np.isfinite(emp_half_width)
        & np.isfinite(emp_lower_extension)
        & np.isfinite(emp_upper_extension)
    )

    drift = pd.to_numeric(work["fair_value_drift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gap = pd.to_numeric(work["gap"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scope_source = work["kalman_scope_days"] if "kalman_scope_days" in work.columns else pd.Series(np.nan, index=work.index)
    pressure_source = (
        work["kalman_projection_pressure"]
        if "kalman_projection_pressure" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    scope_values = (
        pd.to_numeric(scope_source, errors="coerce")
        .fillna(horizons.astype(float))
        .clip(KALMAN_MIN_SCOPE_DAYS, KALMAN_MAX_SCOPE_DAYS)
        .to_numpy(dtype=float)
    )
    projection_pressure = (
        pd.to_numeric(pressure_source, errors="coerce")
        .fillna(0.0)
        .clip(-0.45, 0.45)
        .to_numpy(dtype=float)
    )
    horizon_phi = float(state_spec.phi) ** scope_values
    kalman_return = (
        float(state_spec.drift_weight) * drift * scope_values
        - float(state_spec.mean_reversion_weight) * (1.0 - horizon_phi) * gap
        + projection_pressure
    )
    residual_sigma = pd.to_numeric(work["residual_sigma"], errors="coerce").fillna(1e-6).clip(lower=1e-6).to_numpy(dtype=float)
    tail_flare_score = pd.to_numeric(work.get("tail_flare_score"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if "tail_flare_score" in work else np.zeros_like(q50)
    high_vol_signal = pd.to_numeric(work.get("kalman_high_vol_signal"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if "kalman_high_vol_signal" in work else np.zeros_like(q50)
    tail_flare_unit = np.clip(tail_flare_score / 2.0, 0.0, 1.50)
    high_vol_unit = np.clip(high_vol_signal, 0.0, 1.0)
    sigma_scale_adjustment = (
        1.0
        + float(blend_spec.high_vol_sigma_boost) * high_vol_unit
        + 0.50 * float(blend_spec.high_vol_sigma_boost) * tail_flare_unit
    )
    kalman_sigma = residual_sigma * np.sqrt(scope_values)
    adjusted_kalman_sigma = kalman_sigma * np.clip(sigma_scale_adjustment, 0.50, 3.50)
    disagreement = kalman_return - q50
    clip_width = float(blend_spec.center_clip_widths) * np.maximum.reduce(
        [width_90, adjusted_kalman_sigma, np.full_like(width_90, 1e-6)]
    )
    clipped_disagreement = np.clip(disagreement, -clip_width, clip_width)

    innovation_z = pd.to_numeric(work["innovation_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    state_uncertainty = pd.to_numeric(work["state_uncertainty"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    uncertainty_ratio = state_uncertainty / residual_sigma
    reliability = np.exp(-0.20 * np.abs(innovation_z)) / (1.0 + 0.08 * np.maximum(uncertainty_ratio, 0.0))
    reliability = np.clip(reliability, 0.20, 1.0)

    horizon_scales = np.clip(np.sqrt(scope_values / 7.0), 0.35, 1.35)
    center_weight = np.clip(float(blend_spec.center_weight) * horizon_scales, 0.0, 0.90)
    center_weight = center_weight * (1.0 - float(blend_spec.high_vol_center_dampen) * high_vol_unit)
    center_weight = np.clip(center_weight, 0.0, 0.90)
    center = q50 + center_weight * reliability * clipped_disagreement

    denominator = np.maximum.reduce([width_90, adjusted_kalman_sigma, np.full_like(width_90, 1e-6)])
    disagreement_ratio = np.minimum(np.abs(disagreement) / denominator, 3.0)
    sigma_ratio = np.minimum(adjusted_kalman_sigma / np.maximum(width_90, 1e-6), 3.0)
    width_multiplier = (
        1.0
        + float(blend_spec.disagreement_width_weight) * reliability * disagreement_ratio
        + float(blend_spec.uncertainty_width_weight) * sigma_ratio
    )
    width_multiplier = np.clip(width_multiplier, 0.75, 2.50)

    empirical_weight = np.clip(
        float(blend_spec.empirical_tail_weight) * (0.60 + 0.40 * high_vol_unit),
        0.0,
        0.95,
    )
    empirical_weight = np.where(has_empirical, empirical_weight, 0.0)
    base_half_width = (1.0 - empirical_weight) * half_width + empirical_weight * np.where(
        has_empirical,
        emp_half_width,
        half_width,
    )
    base_lower_extension = (1.0 - empirical_weight) * lower_extension + empirical_weight * np.where(
        has_empirical,
        emp_lower_extension,
        lower_extension,
    )
    base_upper_extension = (1.0 - empirical_weight) * upper_extension + empirical_weight * np.where(
        has_empirical,
        emp_upper_extension,
        upper_extension,
    )

    gap_z = pd.to_numeric(work["gap_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    downside_pressure = np.maximum(gap_z, 0.0) + 0.35 * np.maximum(-innovation_z, 0.0)
    upside_pressure = np.maximum(-gap_z, 0.0) + 0.35 * np.maximum(innovation_z, 0.0)
    high_vol_tail_boost = 1.0 + float(blend_spec.tail_flare_tail_boost) * np.clip(
        0.50 * high_vol_unit + tail_flare_unit,
        0.0,
        2.0,
    )
    lower_multiplier = (
        1.0 + float(blend_spec.tail_weight) * np.minimum(downside_pressure, 3.0) / 3.0
    ) * high_vol_tail_boost
    upper_multiplier = (
        1.0 + float(blend_spec.tail_weight) * np.minimum(upside_pressure, 3.0) / 3.0
    ) * high_vol_tail_boost
    lower_multiplier = np.clip(lower_multiplier, 0.75, 4.0)
    upper_multiplier = np.clip(upper_multiplier, 0.75, 4.0)
    tail_width_multiplier = 1.0 + 0.50 * (width_multiplier - 1.0)

    sigma_half_width = 0.6744897501960817 * adjusted_kalman_sigma * float(blend_spec.sigma_scale)
    sigma_tail_extension = (
        1.6448536269514722 - 0.6744897501960817
    ) * adjusted_kalman_sigma * float(blend_spec.sigma_scale)
    empirical_cap_half = np.where(has_empirical, emp_half_width, 0.0)
    empirical_cap_lower = np.where(has_empirical, emp_lower_extension, 0.0)
    empirical_cap_upper = np.where(has_empirical, emp_upper_extension, 0.0)
    dispersion_cap_half = np.maximum(sigma_half_width, empirical_cap_half)
    dispersion_cap_lower = np.maximum(sigma_tail_extension, empirical_cap_lower)
    dispersion_cap_upper = np.maximum(sigma_tail_extension, empirical_cap_upper)

    if blend_spec.width_mode == "replace":
        new_half_width = np.maximum(dispersion_cap_half, 1e-6)
        new_lower_extension = np.maximum(dispersion_cap_lower, 1e-6)
        new_upper_extension = np.maximum(dispersion_cap_upper, 1e-6)
        tail_width_multiplier = np.ones_like(width_multiplier)
    elif blend_spec.width_mode == "cap":
        new_half_width = np.minimum(base_half_width * width_multiplier, np.maximum(dispersion_cap_half, 1e-6))
        new_lower_extension = np.minimum(
            base_lower_extension * tail_width_multiplier,
            np.maximum(dispersion_cap_lower, 1e-6),
        )
        new_upper_extension = np.minimum(
            base_upper_extension * tail_width_multiplier,
            np.maximum(dispersion_cap_upper, 1e-6),
        )
        tail_width_multiplier = np.ones_like(width_multiplier)
    else:
        new_half_width = base_half_width * width_multiplier
        new_lower_extension = base_lower_extension * tail_width_multiplier
        new_upper_extension = base_upper_extension * tail_width_multiplier

    new_q25 = center - new_half_width
    new_q75 = center + new_half_width
    new_q05 = new_q25 - new_lower_extension * lower_multiplier
    new_q95 = new_q75 + new_upper_extension * upper_multiplier

    if float(blend_spec.safety_width_scale) > 0.0:
        candidate_width_90 = np.maximum(new_q95 - new_q05, 1e-9)
        sigma_width_90 = 2.0 * 1.6448536269514722 * adjusted_kalman_sigma * float(blend_spec.sigma_scale)
        empirical_width_90 = np.where(
            np.isfinite(emp_q95 - emp_q05),
            np.maximum(emp_q95 - emp_q05, 1e-6),
            0.0,
        )
        safety_width = np.maximum(sigma_width_90, empirical_width_90)
        safety_width *= float(blend_spec.safety_width_scale) * (
            1.0 + float(blend_spec.safety_high_vol_scale) * high_vol_unit
        )
        too_wide = np.isfinite(safety_width) & (safety_width > 0.0) & (candidate_width_90 > safety_width)
        shrink = np.ones_like(candidate_width_90)
        shrink[too_wide] = safety_width[too_wide] / candidate_width_90[too_wide]
        new_q05 = center - (center - new_q05) * shrink
        new_q25 = center - (center - new_q25) * shrink
        new_q75 = center + (new_q75 - center) * shrink
        new_q95 = center + (new_q95 - center) * shrink

    sorted_quantiles = np.sort(np.vstack([new_q05, new_q25, center, new_q75, new_q95]).T, axis=1)

    valid_array = valid.to_numpy()
    for idx, column in enumerate(QUANTILE_COLUMNS):
        values = out[column].to_numpy(dtype=float, copy=True)
        values[valid_array] = sorted_quantiles[valid_array, idx]
        out[column] = values
    for column, values in [
        ("kalman_projected_return", kalman_return),
        ("kalman_projected_sigma", adjusted_kalman_sigma),
        ("kalman_scope_days", scope_values),
        ("kalman_projection_pressure", projection_pressure),
        ("kalman_reliability", reliability),
        ("kalman_gap_z", gap_z),
        ("kalman_innovation_z", innovation_z),
        ("kalman_high_vol_signal", high_vol_signal),
        ("kalman_tail_flare_score", tail_flare_score),
    ]:
        output_values = np.full(len(out), np.nan)
        output_values[valid_array] = values[valid_array]
        out[column] = output_values
    return out


def _select_tuned_kalman_variant(
    candidate_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()

    candidates = candidate_rows.copy()
    candidates["as_of"] = pd.to_datetime(candidates["as_of"], errors="coerce")
    candidates["target_timestamp"] = pd.to_datetime(candidates["target_timestamp"], errors="coerce")
    candidates = candidates.dropna(subset=["as_of", "horizon", "forecaster"])
    current_lookup = candidates.set_index(["forecaster", "as_of", "horizon"], drop=False)
    available = set(candidates["forecaster"].astype(str))

    def default_forecaster_for_horizon(horizon_value: int) -> str:
        preferred_candidates = [
            f"{KALMAN_SYNTH_PREFIX}_fundamental_slow_directional_anchor",
            f"{KALMAN_SYNTH_PREFIX}_fundamental_slow_directional_guarded",
            f"{KALMAN_SYNTH_PREFIX}_fundamental_slow_directional_high_vol",
            f"{KALMAN_SYNTH_PREFIX}_local_responsive_directional_anchor",
            f"{KALMAN_SYNTH_PREFIX}_local_responsive_directional_guarded",
        ]
        preferred_candidates.extend(
            [
                f"{KALMAN_SYNTH_PREFIX}_local_responsive_directional_high_vol",
                f"{KALMAN_SYNTH_PREFIX}_local_responsive_sigma_cap",
            ]
        )
        for preferred in preferred_candidates:
            if preferred in available:
                return preferred
        return str(candidates["forecaster"].iloc[0])

    selected_rows: list[dict[str, Any]] = []
    model_order = model_rows.copy()
    model_order["as_of"] = pd.to_datetime(model_order["as_of"], errors="coerce")
    model_order = model_order.sort_values(["horizon", "as_of"])

    for horizon, horizon_rows in model_order.groupby("horizon", sort=True):
        horizon = int(horizon)
        horizon_candidates = candidates.loc[candidates["horizon"].astype(int) == horizon]
        if horizon_candidates.empty:
            continue
        for _, model_row in horizon_rows.iterrows():
            as_of = pd.Timestamp(model_row["as_of"])
            best_forecaster = default_forecaster_for_horizon(horizon)
            best_score = float("nan")

            key = (best_forecaster, as_of, horizon)
            if key not in current_lookup.index:
                continue
            selected = current_lookup.loc[key]
            if isinstance(selected, pd.DataFrame):
                selected = selected.iloc[0]
            out = selected.to_dict()
            out["forecaster"] = f"{KALMAN_SYNTH_PREFIX}_selected"
            out["selected_kalman_variant"] = best_forecaster
            out["kalman_selection_score"] = best_score
            selected_rows.append(out)

    return pd.DataFrame(selected_rows)


def _kalman_synthesis_rows(
    model_rows: pd.DataFrame,
    data_bundle: ForecastDataBundle,
    regimes: pd.DataFrame,
    return_type: str,
    config: EvaluationConfig,
) -> pd.DataFrame:
    if not bool(config.include_kalman_synthesis) or model_rows.empty:
        return pd.DataFrame()

    state_frames = {
        spec.name: _two_state_kalman_level_state(
            close=data_bundle.close,
            environment=data_bundle.buckets.get("environment"),
            spec=spec,
            structure=data_bundle.buckets.get("structure"),
        )
        for spec in KALMAN_STATE_SPECS
    }
    empirical_reference = _empirical_calibration_reference(
        model_rows=model_rows,
        close=data_bundle.close,
        return_type=return_type,
        regimes=regimes,
        config=config,
    )
    candidate_parts: list[pd.DataFrame] = []
    for state_spec in KALMAN_STATE_SPECS:
        state_frame = state_frames.get(state_spec.name, pd.DataFrame())
        for blend_spec in KALMAN_BLEND_SPECS:
            candidate = _synthesize_kalman_variant(
                model_rows=model_rows,
                state_frame=state_frame,
                state_spec=state_spec,
                blend_spec=blend_spec,
                min_history=int(config.kalman_min_history),
                empirical_reference=empirical_reference,
            )
            if not candidate.empty:
                candidate_parts.append(candidate)

    if not candidate_parts:
        return pd.DataFrame()

    candidates = pd.concat(candidate_parts, ignore_index=True, sort=False)
    selected = _select_tuned_kalman_variant(
        candidate_rows=candidates,
        model_rows=model_rows,
    )
    if selected.empty:
        return candidates
    return pd.concat([candidates, selected], ignore_index=True, sort=False)


def build_kalman_projection_overlay(
    predictions: pd.DataFrame,
    data_bundle: ForecastDataBundle,
    horizons: list[int],
    return_type: str,
    state_name: str = "fundamental_slow",
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    state_spec = next((spec for spec in KALMAN_STATE_SPECS if spec.name == state_name), KALMAN_STATE_SPECS[0])
    state_frame = _two_state_kalman_level_state(
        close=data_bundle.close,
        environment=data_bundle.buckets.get("environment"),
        spec=state_spec,
        structure=data_bundle.buckets.get("structure"),
    )
    if state_frame.empty:
        return pd.DataFrame()

    work = predictions.loc[predictions["horizon"].isin(horizons)].copy()
    if work.empty:
        return pd.DataFrame()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce")
    work["target_timestamp"] = pd.to_datetime(work["target_timestamp"], errors="coerce")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype("Int64")
    work["base_close"] = pd.to_numeric(work["base_close"], errors="coerce")
    work = work.dropna(subset=["as_of", "target_timestamp", "horizon", "base_close"])
    if work.empty:
        return pd.DataFrame()

    state_columns = [
        "fair_value_log",
        "fair_value_drift",
        "gap",
        "residual_sigma",
        "gap_z",
        "innovation_z",
        "kalman_high_vol_signal",
        "tail_flare_score",
        "kalman_scope_days",
        "kalman_projection_pressure",
        "macro_forward_adjust",
        "onchain_adjust",
        "history_count",
    ]
    available_state_columns = [column for column in state_columns if column in state_frame.columns]
    work = work.join(state_frame[available_state_columns], on="as_of")
    work = work.dropna(subset=["fair_value_log", "fair_value_drift", "gap"])
    if work.empty:
        return pd.DataFrame()

    drift = pd.to_numeric(work["fair_value_drift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gap = pd.to_numeric(work["gap"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scope_values = (
        pd.to_numeric(work["kalman_scope_days"], errors="coerce")
        if "kalman_scope_days" in work.columns
        else pd.Series(np.nan, index=work.index)
    )
    scope_values = scope_values.fillna(work["horizon"].astype(float)).clip(
        KALMAN_MIN_SCOPE_DAYS,
        KALMAN_MAX_SCOPE_DAYS,
    ).to_numpy(dtype=float)
    projection_pressure = (
        pd.to_numeric(work["kalman_projection_pressure"], errors="coerce")
        if "kalman_projection_pressure" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    projection_pressure = projection_pressure.fillna(0.0).clip(-0.45, 0.45).to_numpy(dtype=float)
    macro_projection = (
        pd.to_numeric(work["macro_forward_adjust"], errors="coerce")
        if "macro_forward_adjust" in work.columns
        else pd.Series(0.0, index=work.index)
    ).fillna(0.0).to_numpy(dtype=float)
    onchain_projection = (
        pd.to_numeric(work["onchain_adjust"], errors="coerce")
        if "onchain_adjust" in work.columns
        else pd.Series(0.0, index=work.index)
    ).fillna(0.0).to_numpy(dtype=float)
    pressure_sum = macro_projection + onchain_projection
    pressure_scale = np.ones_like(pressure_sum)
    pressure_mask = np.isfinite(pressure_sum) & (np.abs(pressure_sum) > 1e-12)
    pressure_scale[pressure_mask] = projection_pressure[pressure_mask] / pressure_sum[pressure_mask]
    macro_projection = macro_projection * pressure_scale
    onchain_projection = onchain_projection * pressure_scale
    horizon_phi = float(state_spec.phi) ** scope_values
    drift_component = float(state_spec.drift_weight) * drift * scope_values
    mean_reversion_component = -float(state_spec.mean_reversion_weight) * (1.0 - horizon_phi) * gap
    projected_return = drift_component + mean_reversion_component + macro_projection + onchain_projection
    base_close = work["base_close"].to_numpy(dtype=float)
    projected_price = np.full(len(work), np.nan)
    sane = np.isfinite(base_close) & np.isfinite(projected_return) & (np.abs(projected_return) <= 2.0)
    if return_type == "log":
        projected_price[sane] = base_close[sane] * np.exp(projected_return[sane])
    else:
        simple_sane = sane & (projected_return > -0.99)
        projected_price[simple_sane] = base_close[simple_sane] * (1.0 + projected_return[simple_sane])

    out = work[
        [
            "as_of",
            "target_timestamp",
            "horizon",
            "base_close",
            "fair_value_log",
            "gap",
            "residual_sigma",
            "gap_z",
            "innovation_z",
            "kalman_high_vol_signal",
            "tail_flare_score",
            "kalman_scope_days",
            "kalman_projection_pressure",
            "macro_forward_adjust",
            "onchain_adjust",
            "history_count",
        ]
    ].copy()
    out["kalman_state_spec"] = state_spec.name
    out["kalman_fair_value_price"] = np.exp(pd.to_numeric(out["fair_value_log"], errors="coerce"))
    out["kalman_scope_days"] = scope_values
    out["kalman_target_timestamp"] = out["as_of"] + pd.to_timedelta(np.rint(scope_values).astype(int), unit="D")
    out["kalman_projected_return"] = projected_return
    out["kalman_drift_component_return"] = drift_component
    out["kalman_mean_reversion_component_return"] = mean_reversion_component
    out["kalman_macro_component_return"] = macro_projection
    out["kalman_onchain_component_return"] = onchain_projection
    out["kalman_projected_price"] = projected_price
    return out.replace([np.inf, -np.inf], np.nan)


def build_kalman_state_history(
    data_bundle: ForecastDataBundle,
    state_name: str = "fundamental_slow",
) -> pd.DataFrame:
    state_spec = next((spec for spec in KALMAN_STATE_SPECS if spec.name == state_name), KALMAN_STATE_SPECS[0])
    state_frame = _two_state_kalman_level_state(
        close=data_bundle.close,
        environment=data_bundle.buckets.get("environment"),
        spec=state_spec,
        structure=data_bundle.buckets.get("structure"),
    )
    if state_frame.empty:
        return pd.DataFrame()
    out = state_frame[
        [
            column
            for column in [
                "fair_value_log",
                "fair_value_drift",
                "gap",
                "residual_sigma",
                "gap_z",
                "innovation_z",
                "kalman_high_vol_signal",
                "tail_flare_score",
                "kalman_scope_days",
                "kalman_projection_pressure",
                "history_count",
                "fundamental_anchor_log",
                "fundamental_level_anchor_log",
                "liquidity_anchor_log",
                "energy_anchor_log",
                "metcalfe_anchor_log",
                "balanced_liquidity_anchor_log",
                "balanced_energy_anchor_log",
                "balanced_metcalfe_anchor_log",
                "macro_cycle_anchor_log",
                "stationary_macro_anchor_log",
                "onchain_reversion_anchor_log",
                "level_anchor_source_count",
                "level_weight_liquidity",
                "level_weight_energy",
                "level_weight_metcalfe",
                "level_weight_macro_cycle",
                "level_weight_stationary_macro",
                "level_weight_onchain",
                "level_value_score_liquidity",
                "level_value_score_energy",
                "level_value_score_metcalfe",
                "level_value_score_macro_cycle",
                "level_value_score_stationary_macro",
                "level_value_score_onchain",
                "liquidity_macro_score",
                "risk_macro_score",
                "cycle_macro_score",
                "leverage_macro_score",
                "financial_conditions_macro_score",
                "macro_cycle_score",
                "macro_impact_score",
                "macro_forward_score",
                "macro_adjust",
                "macro_forward_adjust",
                "liquidity_lead_days",
                "liquidity_lead_corr",
                "cycle_lead_days",
                "cycle_lead_corr",
                "financial_conditions_lead_days",
                "financial_conditions_lead_corr",
                "macro_scope_days",
                "ism_services_stationary_z",
                "inverted_nfci_stationary_z",
                "ism_nfci_stationary_score",
                "btc_log_cycle_z",
                "macro_stationary_gap_score",
                "stationary_macro_adjust",
                "onchain_reversion_score",
                "onchain_average_behavior_score",
                "lth_average",
                "lth_signalboost",
                "sth_average",
                "sth_signalboost",
                "signalboost_peak_score",
                "onchain_behavior_score",
                "lth_average_behavior_z",
                "sth_average_behavior_z",
                "lth_signalboost_peak_score",
                "sth_signalboost_peak_score",
                "lth_signalboost_attention_score",
                "sth_signalboost_attention_score",
                "lth_average_available",
                "sth_average_available",
                "lth_signalboost_event",
                "sth_signalboost_event",
                "lth_signalboost_lead_days",
                "lth_signalboost_lead_corr",
                "sth_signalboost_lead_days",
                "sth_signalboost_lead_corr",
                "signalboost_scope_days",
                "onchain_adjust",
                "lth_extreme_score",
                "sth_extreme_score",
                "lth_boost_extreme_score",
                "sth_boost_extreme_score",
            ]
            if column in state_frame.columns
        ]
    ].copy()
    out["kalman_state_spec"] = state_spec.name
    out["kalman_fair_value_price"] = np.exp(pd.to_numeric(out["fair_value_log"], errors="coerce"))
    out.index.name = "timestamp"
    return out.replace([np.inf, -np.inf], np.nan)


def _base_row(row: pd.Series, forecaster: str) -> dict[str, Any]:
    return {
        "forecaster": forecaster,
        "as_of": pd.Timestamp(row["as_of"]),
        "horizon": int(row["horizon"]),
        "target_timestamp": pd.Timestamp(row["target_timestamp"]) if pd.notna(row.get("target_timestamp")) else pd.NaT,
        "base_close": float(row["base_close"]) if pd.notna(row.get("base_close")) else np.nan,
        "actual_close": float(row["actual_close"]) if pd.notna(row.get("actual_close")) else np.nan,
        "actual_return": float(row["actual_return"]) if pd.notna(row.get("actual_return")) else np.nan,
        "regime_label": row.get("regime_label", "unknown"),
        "vol_regime": row.get("vol_regime", "unknown_vol"),
        "trend_regime": row.get("trend_regime", "unknown_trend"),
        "liquidity_regime": row.get("liquidity_regime", "unknown_liquidity"),
    }


def _random_walk_baseline(model_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in model_rows.iterrows():
        out = _base_row(row, "random_walk")
        out.update({column: 0.0 for column in QUANTILE_COLUMNS})
        rows.append(out)
    return pd.DataFrame(rows)


def _historical_vol_cone_baseline(
    model_rows: pd.DataFrame,
    targets_by_horizon: dict[int, pd.Series],
    config: EvaluationConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in model_rows.iterrows():
        horizon = int(row["horizon"])
        history = _known_history(targets_by_horizon[horizon], pd.Timestamp(row["as_of"]), horizon)
        out = _base_row(row, "historical_vol_cone")
        if len(history) < config.history_min:
            out.update({column: 0.0 for column in QUANTILE_COLUMNS})
        else:
            out.update({column: _finite_quantile(history, q) for column, q in zip(QUANTILE_COLUMNS, QUANTILES)})
        rows.append(out)
    return pd.DataFrame(rows)


def _ewma_vol_cone_baseline(
    model_rows: pd.DataFrame,
    close: pd.Series,
    return_type: str,
    config: EvaluationConfig,
) -> pd.DataFrame:
    daily_returns = _return_series(close, return_type).dropna()
    ewma_sigma = daily_returns.ewm(span=config.ewma_span, adjust=False).std()
    standardized = (daily_returns / ewma_sigma.shift(1)).replace([np.inf, -np.inf], np.nan)
    normal = NormalDist()
    rows = []
    for _, row in model_rows.iterrows():
        as_of = pd.Timestamp(row["as_of"])
        horizon = int(row["horizon"])
        z_history = standardized.loc[standardized.index < as_of].dropna()
        sigma = ewma_sigma.loc[ewma_sigma.index <= as_of].dropna()
        current_sigma = float(sigma.iloc[-1]) if len(sigma) else float(daily_returns.std())
        out = _base_row(row, "ewma_vol_cone")
        if len(z_history) < config.history_min or not math.isfinite(current_sigma):
            z_quantiles = [normal.inv_cdf(q) for q in QUANTILES]
        else:
            z_quantiles = [_finite_quantile(z_history, q, normal.inv_cdf(q)) for q in QUANTILES]
        horizon_sigma = current_sigma * math.sqrt(horizon)
        out.update({column: float(z * horizon_sigma) for column, z in zip(QUANTILE_COLUMNS, z_quantiles)})
        rows.append(out)
    return pd.DataFrame(rows)


def _arch_forecast_quantiles(
    history: pd.Series,
    max_horizon: int,
    vol_model: str,
    min_obs: int,
) -> dict[int, dict[str, float]] | None:
    clean = history.dropna().astype(float)
    if len(clean) < min_obs:
        return None
    try:
        from arch import arch_model
    except Exception:
        return None

    returns_pct = clean.iloc[-2500:] * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if vol_model == "EGARCH":
                model = arch_model(returns_pct, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
            else:
                model = arch_model(returns_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
            result = model.fit(disp="off", show_warning=False)
    except Exception:
        return None

    conditional_vol = pd.Series(result.conditional_volatility).replace(0.0, np.nan)
    standardized = (pd.Series(result.resid) / conditional_vol).replace([np.inf, -np.inf], np.nan).dropna()
    normal = NormalDist()
    if len(standardized) < 50:
        z_quantiles = [normal.inv_cdf(q) for q in QUANTILES]
    else:
        z_quantiles = [_finite_quantile(standardized, q, normal.inv_cdf(q)) for q in QUANTILES]

    one_step_sigma = float(conditional_vol.dropna().iloc[-1] / 100.0) if conditional_vol.notna().any() else float(clean.std())
    variance_path: np.ndarray | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = result.forecast(horizon=max_horizon, reindex=False)
            variance_path = np.asarray(forecast.variance.iloc[-1], dtype=float) / 10000.0
    except Exception:
        variance_path = None

    out: dict[int, dict[str, float]] = {}
    for horizon in range(1, max_horizon + 1):
        if variance_path is not None and len(variance_path) >= horizon and np.isfinite(variance_path[:horizon]).all():
            sigma = math.sqrt(max(float(np.sum(variance_path[:horizon])), 0.0))
        else:
            sigma = one_step_sigma * math.sqrt(horizon)
        out[horizon] = {column: float(z * sigma) for column, z in zip(QUANTILE_COLUMNS, z_quantiles)}
    return out


def _garch_baselines(
    model_rows: pd.DataFrame,
    close: pd.Series,
    return_type: str,
    config: EvaluationConfig,
) -> list[pd.DataFrame]:
    if not config.include_garch:
        return []
    daily_returns = _return_series(close, return_type)
    horizons = sorted(int(value) for value in model_rows["horizon"].dropna().unique())
    if not horizons:
        return []
    max_horizon = max(horizons)
    rows_by_model = {"garch_vol_cone": [], "egarch_vol_cone": []}
    cached: dict[str, dict[int, dict[str, float]] | None] = {"GARCH": None, "EGARCH": None}

    unique_as_ofs = sorted(pd.Timestamp(value) for value in model_rows["as_of"].dropna().unique())
    refit_dates = {
        as_of
        for pos, as_of in enumerate(unique_as_ofs)
        if pos % max(int(config.garch_refit_interval), 1) == 0
    }
    if unique_as_ofs:
        refit_dates.add(unique_as_ofs[0])

    grouped = model_rows.groupby("as_of", sort=True)
    for as_of in unique_as_ofs:
        history = daily_returns.loc[daily_returns.index <= as_of].dropna()
        if as_of in refit_dates or cached["GARCH"] is None:
            cached["GARCH"] = _arch_forecast_quantiles(history, max_horizon, "GARCH", config.garch_min_obs)
            cached["EGARCH"] = _arch_forecast_quantiles(history, max_horizon, "EGARCH", config.garch_min_obs)
        for _, row in grouped.get_group(as_of).iterrows():
            horizon = int(row["horizon"])
            for vol_model, forecaster in [("GARCH", "garch_vol_cone"), ("EGARCH", "egarch_vol_cone")]:
                out = _base_row(row, forecaster)
                forecast = cached[vol_model]
                if forecast is None or horizon not in forecast:
                    sigma = float(history.std()) * math.sqrt(horizon) if len(history) else 0.0
                    normal = NormalDist()
                    out.update({column: normal.inv_cdf(q) * sigma for column, q in zip(QUANTILE_COLUMNS, QUANTILES)})
                else:
                    out.update(forecast[horizon])
                rows_by_model[forecaster].append(out)

    return [pd.DataFrame(rows) for rows in rows_by_model.values() if rows]


def _regime_conditioned_baseline(
    model_rows: pd.DataFrame,
    targets_by_horizon: dict[int, pd.Series],
    regimes: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    rows = []
    regime_labels = regimes["regime_label"]
    for _, row in model_rows.iterrows():
        as_of = pd.Timestamp(row["as_of"])
        horizon = int(row["horizon"])
        history = _known_history(targets_by_horizon[horizon], as_of, horizon)
        out = _base_row(row, "regime_conditioned_historical")
        if len(history) < config.history_min:
            out.update({column: 0.0 for column in QUANTILE_COLUMNS})
            rows.append(out)
            continue
        current_regime = row.get("regime_label", regime_labels.reindex([as_of]).iloc[0] if as_of in regime_labels.index else "unknown")
        history_regimes = regime_labels.reindex(history.index)
        regime_history = history.loc[history_regimes == current_regime]
        selected = regime_history if len(regime_history) >= config.regime_history_min else history
        out.update({column: _finite_quantile(selected, q) for column, q in zip(QUANTILE_COLUMNS, QUANTILES)})
        rows.append(out)
    return pd.DataFrame(rows)


def _lower_tail_residual_variant(
    model_rows: pd.DataFrame,
    forecaster: str,
    config: EvaluationConfig,
    by_regime: bool,
) -> pd.DataFrame:
    adjusted_parts = []
    for horizon, group in model_rows.groupby("horizon", sort=True):
        group = group.sort_values("as_of").copy()
        source = group.copy()
        for index, row in group.iterrows():
            as_of = pd.Timestamp(row["as_of"])
            known = source.loc[
                (pd.to_datetime(source["target_timestamp"], errors="coerce") <= as_of)
                & source["actual_return"].notna()
            ]
            selected = known
            if by_regime and len(known) >= config.regime_history_min:
                same_regime = known.loc[known["regime_label"] == row.get("regime_label")]
                if len(same_regime) >= config.regime_history_min:
                    selected = same_regime
            if len(selected) >= config.regime_history_min:
                q05_adjust = _finite_quantile(selected["actual_return"] - selected["q05"], 0.05)
                q25_adjust = _finite_quantile(selected["actual_return"] - selected["q25"], 0.25)
            else:
                q05_adjust = 0.0
                q25_adjust = 0.0
            new_q25 = min(float(row["q25"]) + q25_adjust, float(row["q50"]) - 1e-9)
            new_q05 = min(float(row["q05"]) + q05_adjust, new_q25 - 1e-9)
            group.at[index, "q25"] = new_q25
            group.at[index, "q05"] = new_q05
        adjusted_parts.append(group)
    adjusted = pd.concat(adjusted_parts).sort_values(["as_of", "horizon"])
    adjusted["forecaster"] = forecaster
    return adjusted


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: format(float(value), floatfmt) if pd.notna(value) else ""
            )
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    widths = {
        column: max(len(str(column)), int(display[column].astype(str).map(len).max()))
        for column in display.columns
    }
    header = "| " + " | ".join(str(column).ljust(widths[column]) for column in display.columns) + " |"
    divider = "| " + " | ".join("-" * widths[column] for column in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).ljust(widths[column]) for column in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def build_forecaster_panel(
    predictions: pd.DataFrame,
    data_bundle: ForecastDataBundle,
    horizons: list[int],
    return_type: str,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = build_btc_regimes(
        close=data_bundle.close,
        environment=data_bundle.buckets.get("environment"),
        return_type=return_type,
    )
    model_rows = predictions.loc[predictions["horizon"].isin(horizons)].copy()
    model_rows["as_of"] = pd.to_datetime(model_rows["as_of"], errors="coerce")
    model_rows = model_rows.dropna(subset=["as_of", "horizon"]).sort_values(["as_of", "horizon"])
    regime_columns = [
        "regime_label",
        "vol_regime",
        "trend_regime",
        "liquidity_regime",
        "realized_vol_30d",
        "tail_flare_ratio",
        "tail_flare_pressure",
        "trend_90d",
        "liquidity_score",
    ]
    model_rows = model_rows.join(regimes[regime_columns], on="as_of")
    model_rows["forecaster"] = MODEL_FORECASTER

    targets_by_horizon = {
        int(horizon): _target_returns(data_bundle.close, int(horizon), return_type)
        for horizon in horizons
    }
    panels = [
        model_rows,
        _lower_tail_residual_variant(model_rows, "model_lower_tail_residual", config, by_regime=False),
        _lower_tail_residual_variant(model_rows, "model_regime_lower_tail_residual", config, by_regime=True),
        _random_walk_baseline(model_rows),
        _historical_vol_cone_baseline(model_rows, targets_by_horizon, config),
        _ewma_vol_cone_baseline(model_rows, data_bundle.close, return_type, config),
        _regime_conditioned_baseline(model_rows, targets_by_horizon, regimes, config),
    ]
    kalman_rows = _kalman_synthesis_rows(model_rows, data_bundle, regimes, return_type, config)
    if not kalman_rows.empty:
        panels.append(kalman_rows)
    panels.extend(_garch_baselines(model_rows, data_bundle.close, return_type, config))
    panel = pd.concat(panels, ignore_index=True, sort=False)
    for large_column in [
        "width_diagnostics_json",
        "lower_tail_diagnostics_json",
        "upper_tail_diagnostics_json",
        "calibration_score_comparison_json",
    ]:
        if large_column in panel.columns:
            panel[large_column] = None
    panel[QUANTILE_COLUMNS] = panel[QUANTILE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    panel = panel.sort_values(["forecaster", "horizon", "as_of"]).reset_index(drop=True)
    return panel, regimes


def _pinball_vector(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> np.ndarray:
    errors = actual - predicted
    return np.maximum(quantile * errors, (quantile - 1.0) * errors)


def _wis_vector(frame: pd.DataFrame) -> np.ndarray:
    y = frame["actual_return"].to_numpy(dtype=float)
    q05 = frame["q05"].to_numpy(dtype=float)
    q25 = frame["q25"].to_numpy(dtype=float)
    q50 = frame["q50"].to_numpy(dtype=float)
    q75 = frame["q75"].to_numpy(dtype=float)
    q95 = frame["q95"].to_numpy(dtype=float)
    score_90 = (q95 - q05) + 20.0 * np.maximum(q05 - y, 0.0) + 20.0 * np.maximum(y - q95, 0.0)
    score_50 = (q75 - q25) + 4.0 * np.maximum(q25 - y, 0.0) + 4.0 * np.maximum(y - q75, 0.0)
    median_loss = np.abs(y - q50)
    return (0.5 * median_loss) + (0.25 * score_50) + (0.25 * score_90)


def add_row_losses(panel: pd.DataFrame, return_type: str, config: EvaluationConfig) -> pd.DataFrame:
    out = panel.copy()
    actual = out["actual_return"].to_numpy(dtype=float)
    row_pinballs = []
    for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES):
        row_pinballs.append(_pinball_vector(actual, out[column].to_numpy(dtype=float), quantile))
        out[f"{column}_pinball_loss"] = row_pinballs[-1]
    pinball_matrix = np.vstack(row_pinballs)
    finite_counts = np.isfinite(pinball_matrix).sum(axis=0)
    pinball_sums = np.nansum(pinball_matrix, axis=0)
    out["avg_pinball_loss"] = np.divide(
        pinball_sums,
        finite_counts,
        out=np.full(pinball_sums.shape, np.nan),
        where=finite_counts > 0,
    )
    out["wis_loss"] = _wis_vector(out)
    if return_type == "log":
        out["actual_simple_return"] = np.exp(out["actual_return"].astype(float)) - 1.0
    else:
        out["actual_simple_return"] = out["actual_return"].astype(float)
    position = np.sign(out["q50"].astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    position = position.where(out["q50"].astype(float).abs() > 1e-12, 0.0)
    cost_rate = (float(config.fee_bps) + float(config.slippage_bps)) / 10000.0
    out["strategy_position"] = position
    out["strategy_net_return"] = position * out["actual_simple_return"] - position.abs() * 2.0 * cost_rate
    out["width_50"] = out["q75"] - out["q25"]
    out["width_90"] = out["q95"] - out["q05"]
    out["q05_miss"] = out["actual_return"] < out["q05"]
    out["q95_miss"] = out["actual_return"] > out["q95"]
    return out.replace([np.inf, -np.inf], np.nan)


def _non_overlapping_sample(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
    selected = []
    last_as_of: pd.Timestamp | None = None
    for index, row in group.sort_values("as_of").iterrows():
        as_of = pd.Timestamp(row["as_of"])
        if last_as_of is None or as_of >= last_as_of + pd.Timedelta(days=int(horizon)):
            selected.append(index)
            last_as_of = as_of
    return group.loc[selected]


def _summary_metrics(group: pd.DataFrame, sample_name: str) -> dict[str, Any]:
    clean = group.dropna(subset=["actual_return", *QUANTILE_COLUMNS]).copy()
    if clean.empty:
        return {
            "forecaster": group["forecaster"].iloc[0] if len(group) else None,
            "horizon": int(group["horizon"].iloc[0]) if len(group) else None,
            "sample": sample_name,
            "observations": 0,
        }
    y = clean["actual_return"].to_numpy(dtype=float)
    q_map = {q: clean[column].to_numpy(dtype=float) for column, q in zip(QUANTILE_COLUMNS, QUANTILES)}
    pinballs = {f"{column}_pinball": pinball_loss(y, clean[column].to_numpy(dtype=float), q) for column, q in zip(QUANTILE_COLUMNS, QUANTILES)}
    sign_mask = clean["actual_return"].ne(0.0) & clean["q50"].ne(0.0)
    sign_accuracy = float(
        (np.sign(clean.loc[sign_mask, "actual_return"]) == np.sign(clean.loc[sign_mask, "q50"])).mean()
    ) if sign_mask.any() else float("nan")
    return {
        "forecaster": clean["forecaster"].iloc[0],
        "horizon": int(clean["horizon"].iloc[0]),
        "sample": sample_name,
        "observations": int(len(clean)),
        "avg_pinball": float(np.mean(list(pinballs.values()))),
        "wis": weighted_interval_score(y, q_map),
        "crps_quantile_approx": float(2.0 * np.mean(list(pinballs.values()))),
        "q05_miss_rate": float(clean["q05_miss"].mean()),
        "q95_miss_rate": float(clean["q95_miss"].mean()),
        "coverage_50": float(((clean["actual_return"] >= clean["q25"]) & (clean["actual_return"] <= clean["q75"])).mean()),
        "coverage_90": float(((clean["actual_return"] >= clean["q05"]) & (clean["actual_return"] <= clean["q95"])).mean()),
        "width_50": float(clean["width_50"].mean()),
        "width_90": float(clean["width_90"].mean()),
        "directional_hit_rate": sign_accuracy,
        "q50_actual_corr": _safe_corr(clean["q50"], clean["actual_return"]),
        "q50_actual_spearman": _safe_corr(clean["q50"], clean["actual_return"], method="spearman"),
        "mean_strategy_net_return": float(clean["strategy_net_return"].mean()),
        "strategy_hit_rate": float((clean.loc[clean["strategy_position"].ne(0.0), "strategy_net_return"] > 0.0).mean())
        if clean["strategy_position"].ne(0.0).any()
        else float("nan"),
        "width_abs_return_spearman": _safe_corr(clean["width_90"], clean["actual_return"].abs(), method="spearman"),
        **pinballs,
    }


def summarize_forecasters(panel_with_losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (forecaster, horizon), group in panel_with_losses.groupby(["forecaster", "horizon"], sort=True):
        rows.append(_summary_metrics(group, "all"))
        rows.append(_summary_metrics(_non_overlapping_sample(group, int(horizon)), "non_overlapping"))
    return pd.DataFrame(rows).sort_values(["sample", "horizon", "forecaster"]).reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, samples: int, block_size: int, rng: np.random.Generator) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 3 or samples <= 0:
        mean = float(np.mean(clean)) if clean.size else float("nan")
        return mean, mean
    block_size = max(1, min(int(block_size), clean.size))
    boot = []
    for _ in range(samples):
        indices: list[int] = []
        while len(indices) < clean.size:
            start = int(rng.integers(0, clean.size))
            indices.extend((np.arange(start, start + block_size) % clean.size).tolist())
        boot.append(float(np.mean(clean[indices[: clean.size]])))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def bootstrap_model_comparisons(panel_with_losses: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.bootstrap_seed))
    rows = []
    available_forecasters = set(panel_with_losses["forecaster"].dropna().astype(str))
    # Keep the bootstrap focused. The full Kalman grid is ranked in the metrics table;
    # bootstrapping every grid member makes evaluation unnecessarily slow.
    kalman_forecasters = {
        f"{KALMAN_SYNTH_PREFIX}_selected"
    }.intersection(available_forecasters)
    compare_forecasters = sorted(
        (BASELINE_FORECASTERS | LOWER_TAIL_FORECASTERS | kalman_forecasters).intersection(available_forecasters)
    )
    for horizon, horizon_frame in panel_with_losses.dropna(subset=["actual_return"]).groupby("horizon", sort=True):
        model = horizon_frame.loc[horizon_frame["forecaster"] == MODEL_FORECASTER]
        if model.empty:
            continue
        for forecaster in compare_forecasters:
            other = horizon_frame.loc[horizon_frame["forecaster"] == forecaster]
            if other.empty:
                continue
            common = model.merge(
                other,
                on=["as_of", "horizon"],
                suffixes=("_model", "_other"),
            ).sort_values("as_of")
            if common.empty:
                continue
            block_size = max(int(horizon), 7)
            pinball_diff = common["avg_pinball_loss_model"].to_numpy(dtype=float) - common["avg_pinball_loss_other"].to_numpy(dtype=float)
            wis_diff = common["wis_loss_model"].to_numpy(dtype=float) - common["wis_loss_other"].to_numpy(dtype=float)
            pnl_diff = common["strategy_net_return_model"].to_numpy(dtype=float) - common["strategy_net_return_other"].to_numpy(dtype=float)
            pinball_ci = _bootstrap_ci(pinball_diff, config.bootstrap_samples, block_size, rng)
            wis_ci = _bootstrap_ci(wis_diff, config.bootstrap_samples, block_size, rng)
            pnl_ci = _bootstrap_ci(pnl_diff, config.bootstrap_samples, block_size, rng)
            rows.append(
                {
                    "horizon": int(horizon),
                    "comparison": f"{MODEL_FORECASTER}_minus_{forecaster}",
                    "observations": int(len(common)),
                    "pinball_diff_mean": float(np.nanmean(pinball_diff)),
                    "pinball_diff_ci_low": pinball_ci[0],
                    "pinball_diff_ci_high": pinball_ci[1],
                    "wis_diff_mean": float(np.nanmean(wis_diff)),
                    "wis_diff_ci_low": wis_ci[0],
                    "wis_diff_ci_high": wis_ci[1],
                    "strategy_net_return_diff_mean": float(np.nanmean(pnl_diff)),
                    "strategy_net_return_diff_ci_low": pnl_ci[0],
                    "strategy_net_return_diff_ci_high": pnl_ci[1],
                }
            )
    return pd.DataFrame(rows)


def grouped_calibration_table(panel_with_losses: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    clean = panel_with_losses.dropna(subset=["actual_return", *QUANTILE_COLUMNS]).copy()
    for (forecaster, horizon, group_value), group in clean.groupby(["forecaster", "horizon", group_column], sort=True):
        if len(group) < 5:
            continue
        summary = _summary_metrics(group, "all")
        rows.append(
            {
                "forecaster": forecaster,
                "horizon": int(horizon),
                group_column: group_value,
                "observations": int(len(group)),
                "avg_pinball": summary["avg_pinball"],
                "wis": summary["wis"],
                "q05_miss_rate": summary["q05_miss_rate"],
                "q95_miss_rate": summary["q95_miss_rate"],
                "coverage_50": summary["coverage_50"],
                "coverage_90": summary["coverage_90"],
                "width_90": summary["width_90"],
            }
        )
    return pd.DataFrame(rows)


def confidence_utility_table(panel_with_losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = panel_with_losses.dropna(subset=["actual_return", "width_90", "q50", "strategy_net_return"]).copy()
    for (forecaster, horizon), group in clean.groupby(["forecaster", "horizon"], sort=True):
        if len(group) < 30 or group["width_90"].nunique() < 3:
            continue
        group = group.copy()
        group["width_bucket"] = pd.qcut(
            group["width_90"].rank(method="first"),
            3,
            labels=["narrow", "middle", "wide"],
        )
        for bucket, bucket_frame in group.groupby("width_bucket", observed=True):
            summary = _summary_metrics(bucket_frame, "all")
            rows.append(
                {
                    "forecaster": forecaster,
                    "horizon": int(horizon),
                    "width_bucket": str(bucket),
                    "observations": int(len(bucket_frame)),
                    "directional_hit_rate": summary["directional_hit_rate"],
                    "q50_actual_corr": summary["q50_actual_corr"],
                    "mean_strategy_net_return": summary["mean_strategy_net_return"],
                    "avg_abs_return": float(bucket_frame["actual_return"].abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def tail_signal_table(panel_with_losses: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    rows = []
    clean = panel_with_losses.dropna(subset=["actual_return", "actual_simple_return", "q05", "q95"]).copy()
    cost_rate = (float(config.fee_bps) + float(config.slippage_bps)) / 10000.0
    for (forecaster, horizon), group in clean.groupby(["forecaster", "horizon"], sort=True):
        if len(group) < 50:
            continue
        high_q95 = group["q95"] >= group["q95"].quantile(0.90)
        low_q05 = group["q05"] <= group["q05"].quantile(0.10)
        for signal, mask, position in [
            ("top_decile_q95_long", high_q95, 1.0),
            ("bottom_decile_q05_short", low_q05, -1.0),
        ]:
            subset = group.loc[mask]
            if subset.empty:
                continue
            net = position * subset["actual_simple_return"].astype(float) - 2.0 * cost_rate
            rows.append(
                {
                    "forecaster": forecaster,
                    "horizon": int(horizon),
                    "signal": signal,
                    "observations": int(len(subset)),
                    "mean_net_return": float(net.mean()),
                    "hit_rate": float((net > 0.0).mean()),
                    "mean_actual_return": float(subset["actual_return"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _forward_drawdown(close: pd.Series, as_of: pd.Timestamp, window_days: int = 30) -> float:
    if as_of not in close.index:
        return float("nan")
    base = float(close.loc[as_of])
    if not math.isfinite(base) or base <= 0.0:
        return float("nan")
    future = close.loc[(close.index > as_of) & (close.index <= as_of + pd.Timedelta(days=window_days))]
    if future.empty:
        return float("nan")
    return float(future.min() / base - 1.0)


def q05_failure_drawdown_table(panel_with_losses: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    rows = []
    close = close.astype(float).sort_index()
    clean = panel_with_losses.dropna(subset=["actual_return", "q05"]).copy()
    drawdown_cache = {
        pd.Timestamp(as_of): _forward_drawdown(close, pd.Timestamp(as_of))
        for as_of in clean["as_of"].dropna().unique()
    }
    clean["forward_30d_drawdown"] = clean["as_of"].map(drawdown_cache)
    for (forecaster, horizon), group in clean.groupby(["forecaster", "horizon"], sort=True):
        group = group.sort_values("as_of")
        fail = group["q05_miss"].astype(bool)
        if len(group) < 20:
            continue
        run_lengths: list[int] = []
        current = 0
        for value in fail:
            if value:
                current += 1
            elif current:
                run_lengths.append(current)
                current = 0
        if current:
            run_lengths.append(current)
        rows.append(
            {
                "forecaster": forecaster,
                "horizon": int(horizon),
                "observations": int(len(group)),
                "q05_miss_rate": float(fail.mean()),
                "q05_miss_lag1_autocorr": _safe_corr(fail.astype(float), fail.shift(1).astype(float)),
                "q05_miss_clusters": int(len(run_lengths)),
                "max_q05_miss_cluster": int(max(run_lengths) if run_lengths else 0),
                "mean_forward_30d_drawdown_after_q05_miss": float(group.loc[fail, "forward_30d_drawdown"].mean()),
                "mean_forward_30d_drawdown_after_no_q05_miss": float(group.loc[~fail, "forward_30d_drawdown"].mean()),
            }
        )
    return pd.DataFrame(rows)


def kalman_synthesis_selection_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    all_metrics = metrics.loc[metrics["sample"] == "all"].copy()
    kalman = all_metrics.loc[all_metrics["forecaster"].astype(str).str.startswith(KALMAN_SYNTH_PREFIX)].copy()
    model = all_metrics.loc[all_metrics["forecaster"] == MODEL_FORECASTER].copy()
    if kalman.empty or model.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for horizon, horizon_kalman in kalman.groupby("horizon", sort=True):
        horizon = int(horizon)
        model_row = model.loc[model["horizon"].astype(int) == horizon]
        if model_row.empty:
            continue
        model_row = model_row.iloc[0]
        best = horizon_kalman.sort_values(["wis", "avg_pinball"]).iloc[0]
        selected = horizon_kalman.loc[horizon_kalman["forecaster"] == f"{KALMAN_SYNTH_PREFIX}_selected"]
        selected_row = selected.iloc[0] if not selected.empty else None
        rows.append(
            {
                "horizon": horizon,
                "model_wis": float(model_row["wis"]),
                "model_avg_pinball": float(model_row["avg_pinball"]),
                "best_kalman_forecaster": best["forecaster"],
                "best_kalman_wis": float(best["wis"]),
                "best_kalman_avg_pinball": float(best["avg_pinball"]),
                "best_kalman_wis_minus_model": float(best["wis"] - model_row["wis"]),
                "best_kalman_pinball_minus_model": float(best["avg_pinball"] - model_row["avg_pinball"]),
                "selected_wis": float(selected_row["wis"]) if selected_row is not None else float("nan"),
                "selected_avg_pinball": float(selected_row["avg_pinball"]) if selected_row is not None else float("nan"),
                "selected_wis_minus_model": float(selected_row["wis"] - model_row["wis"])
                if selected_row is not None
                else float("nan"),
                "selected_pinball_minus_model": float(selected_row["avg_pinball"] - model_row["avg_pinball"])
                if selected_row is not None
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def kalman_hyperparameter_grid_payload() -> dict[str, Any]:
    return {
        "state_specs": [spec.__dict__ for spec in KALMAN_STATE_SPECS],
        "blend_specs": [spec.__dict__ for spec in KALMAN_BLEND_SPECS],
        "selection": {
            "forecaster": f"{KALMAN_SYNTH_PREFIX}_selected",
            "method": (
                "Deterministic tuned profile selected from the fixed walk-forward grid: "
                "local_responsive_directional_anchor for short horizons below 15d where TA/noise can matter, "
                "and fundamental_slow_directional_anchor for 15d+ horizons where the slow fundamental anchor should dominate. "
                "Guarded, high-vol, and sigma-cap profiles are kept as fallbacks. The full fixed grid is still emitted and ranked "
                "in brutal_baseline_metrics.csv and kalman_synthesis_selection.csv."
            ),
        },
    }


def _write_markdown_summary(
    path: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    confidence: pd.DataFrame,
    tail_signals: pd.DataFrame,
    q05_drawdowns: pd.DataFrame,
    kalman_selection: pd.DataFrame | None = None,
) -> None:
    lines = ["# Brutal Baseline Evaluation", ""]
    all_metrics = metrics.loc[metrics["sample"] == "all"].copy()
    model = all_metrics.loc[all_metrics["forecaster"] == MODEL_FORECASTER]
    baselines = all_metrics.loc[all_metrics["forecaster"].isin(BASELINE_FORECASTERS)]
    wins = []
    for _, model_row in model.iterrows():
        horizon = int(model_row["horizon"])
        peers = baselines.loc[baselines["horizon"] == horizon]
        for _, peer in peers.iterrows():
            wins.append(
                {
                    "horizon": horizon,
                    "baseline": peer["forecaster"],
                    "pinball_better": bool(model_row["avg_pinball"] < peer["avg_pinball"]),
                    "wis_better": bool(model_row["wis"] < peer["wis"]),
                    "pnl_better": bool(model_row["mean_strategy_net_return"] > peer["mean_strategy_net_return"]),
                }
            )
    win_frame = pd.DataFrame(wins)
    if not win_frame.empty:
        lines.append("## Model vs Baselines")
        lines.append("")
        lines.append(_markdown_table(win_frame, floatfmt=".6f"))
        lines.append("")

    display_columns = [
        "forecaster",
        "horizon",
        "sample",
        "observations",
        "avg_pinball",
        "wis",
        "q05_miss_rate",
        "directional_hit_rate",
        "mean_strategy_net_return",
    ]
    lines.append("## Core Metrics")
    lines.append("")
    lines.append(_markdown_table(metrics[display_columns], floatfmt=".6f"))
    lines.append("")

    if kalman_selection is not None and not kalman_selection.empty:
        lines.append("## Kalman Synthesis Selection")
        lines.append("")
        lines.append("Negative differences mean the Kalman synthesis variant is better than the current model on that metric.")
        lines.append("")
        lines.append(_markdown_table(kalman_selection, floatfmt=".6f"))
        lines.append("")

    if not bootstrap.empty:
        lines.append("## Block Bootstrap")
        lines.append("")
        lines.append("Negative loss differences mean the model is better; positive PnL differences mean the model is better.")
        lines.append("")
        lines.append(_markdown_table(bootstrap, floatfmt=".6f"))
        lines.append("")

    if not confidence.empty:
        lines.append("## Confidence Utility")
        lines.append("")
        lines.append(_markdown_table(confidence, floatfmt=".6f"))
        lines.append("")

    if not tail_signals.empty:
        lines.append("## Tail Signal Utility")
        lines.append("")
        lines.append("The production forecast schema has q05/q95 rather than q10/q90, so tail utility is tested with top-q95 long and bottom-q05 short deciles.")
        lines.append("")
        lines.append(_markdown_table(tail_signals, floatfmt=".6f"))
        lines.append("")

    if not q05_drawdowns.empty:
        lines.append("## Q05 Failure And Drawdowns")
        lines.append("")
        lines.append(_markdown_table(q05_drawdowns, floatfmt=".6f"))
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def build_brutal_baseline_evaluation(
    *,
    predictions: pd.DataFrame,
    data_bundle: ForecastDataBundle,
    horizons: list[int],
    return_type: str,
    output_dir: str | Path,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    config = config or EvaluationConfig()
    output = ensure_dir(output_dir)
    panel, regimes = build_forecaster_panel(
        predictions=predictions,
        data_bundle=data_bundle,
        horizons=horizons,
        return_type=return_type,
        config=config,
    )
    panel_with_losses = add_row_losses(panel, return_type=return_type, config=config)
    panel_with_losses["year"] = pd.to_datetime(panel_with_losses["as_of"]).dt.year.astype("Int64")

    metrics = summarize_forecasters(panel_with_losses)
    bootstrap = bootstrap_model_comparisons(panel_with_losses, config)
    yearly = grouped_calibration_table(panel_with_losses, "year")
    regime = grouped_calibration_table(panel_with_losses, "regime_label")
    confidence = confidence_utility_table(panel_with_losses)
    tail_signals = tail_signal_table(panel_with_losses, config)
    q05_drawdowns = q05_failure_drawdown_table(panel_with_losses, data_bundle.close)
    kalman_selection = kalman_synthesis_selection_table(metrics)

    artifacts = {
        "panel": output / "brutal_forecaster_panel.csv",
        "metrics": output / "brutal_baseline_metrics.csv",
        "bootstrap": output / "brutal_baseline_block_bootstrap.csv",
        "yearly_calibration": output / "brutal_baseline_yearly_calibration.csv",
        "regime_calibration": output / "brutal_baseline_regime_calibration.csv",
        "confidence_utility": output / "brutal_baseline_confidence_utility.csv",
        "tail_signal_utility": output / "brutal_baseline_tail_signal_utility.csv",
        "q05_failure_drawdowns": output / "brutal_baseline_q05_failure_drawdowns.csv",
        "kalman_synthesis_selection": output / "kalman_synthesis_selection.csv",
        "kalman_synthesis_hyperparameters": output / "kalman_synthesis_hyperparameters.json",
        "regimes": output / "btc_regime_table.csv",
        "markdown": output / "brutal_baseline_evaluation.md",
        "json": output / "brutal_baseline_evaluation.json",
    }
    panel_with_losses.to_csv(artifacts["panel"], index=False)
    metrics.to_csv(artifacts["metrics"], index=False)
    bootstrap.to_csv(artifacts["bootstrap"], index=False)
    yearly.to_csv(artifacts["yearly_calibration"], index=False)
    regime.to_csv(artifacts["regime_calibration"], index=False)
    confidence.to_csv(artifacts["confidence_utility"], index=False)
    tail_signals.to_csv(artifacts["tail_signal_utility"], index=False)
    q05_drawdowns.to_csv(artifacts["q05_failure_drawdowns"], index=False)
    kalman_selection.to_csv(artifacts["kalman_synthesis_selection"], index=False)
    write_json(artifacts["kalman_synthesis_hyperparameters"], kalman_hyperparameter_grid_payload())
    regimes.reset_index(names="timestamp").to_csv(artifacts["regimes"], index=False)
    _write_markdown_summary(
        artifacts["markdown"],
        metrics=metrics,
        bootstrap=bootstrap,
        confidence=confidence,
        tail_signals=tail_signals,
        q05_drawdowns=q05_drawdowns,
        kalman_selection=kalman_selection,
    )
    payload = {
        "config": config.__dict__,
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "kalman_selection": kalman_selection.to_dict(orient="records"),
    }
    write_json(artifacts["json"], payload)
    return payload
