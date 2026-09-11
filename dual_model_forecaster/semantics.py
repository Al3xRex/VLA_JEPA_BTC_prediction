from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dual_model_forecaster.utils import (
    consensus_score,
    ema,
    positive_strength,
    robust_rolling_zscore,
    rolling_mean,
    rolling_std,
    safe_divide,
    safe_log,
    sigmoid,
    signed_strength,
)
from dual_model_forecaster.liquidation_architecture import build_liquidation_semantic_targets


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").astype(float)
    return pd.Series(0.0, index=frame.index, name=column)


def _series_mean(frame: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    if not cols:
        return pd.Series(0.0, index=frame.index, name=prefix)
    return frame[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _confidence(
    components: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
    extra_support: pd.Series | None = None,
) -> pd.Series:
    agreement = consensus_score(components)
    evidence = 1.0 - missingness.astype(float)
    score = 0.45 * agreement + 0.35 * freshness.astype(float) + 0.20 * evidence
    if extra_support is not None:
        score = 0.70 * score + 0.30 * extra_support.astype(float).clip(0.0, 1.0)
    return score.clip(0.0, 1.0)


def _to_probability(series: pd.Series, scale: float = 1.0) -> pd.Series:
    return pd.Series(sigmoid(series.fillna(0.0), scale=scale), index=series.index, name=series.name)


def build_structure_targets(
    frame: pd.DataFrame,
    close: pd.Series,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    anchors = pd.DataFrame(
        {
            "lth_average": _series(frame, "lth_average"),
            "sth_average": _series(frame, "sth_average"),
            "energy_value": _series(frame, "energy_value"),
            "metcalfe_price": _series(frame, "metcalfe_price"),
        },
        index=frame.index,
    ).replace(0, np.nan)
    close_log = safe_log(close)
    valuation_gaps = pd.DataFrame(
        {
            name: robust_rolling_zscore(close_log - safe_log(anchors[name]), window=252)
            for name in anchors.columns
        },
        index=frame.index,
    ).fillna(0.0)
    composite_gap = valuation_gaps.mean(axis=1)

    conviction_components = pd.DataFrame(
        {
            "lth_signalboost": robust_rolling_zscore(_series(frame, "lth_signalboost"), window=180),
            "lth_mvrv": robust_rolling_zscore(_series(frame, "lth_mvrv"), window=180),
            "lth_sopr": robust_rolling_zscore(_series(frame, "lth_sopr"), window=180),
            "lth_aviv": robust_rolling_zscore(_series(frame, "lth_aviv"), window=180),
            "lth_amvrv": robust_rolling_zscore(_series(frame, "lth_amvrv"), window=180),
        },
        index=frame.index,
    ).fillna(0.0)
    fragility_components = pd.DataFrame(
        {
            "sth_signalboost": robust_rolling_zscore(_series(frame, "sth_signalboost"), window=120),
            "sth_mvrv": robust_rolling_zscore(_series(frame, "sth_mvrv"), window=120),
            "sth_utxipp": robust_rolling_zscore(_series(frame, "sth_utxipp"), window=120),
            "sth_sipp": robust_rolling_zscore(_series(frame, "sth_sipp"), window=120),
            "sth_upl": robust_rolling_zscore(_series(frame, "sth_upl"), window=120),
            "sth_ssr": robust_rolling_zscore(_series(frame, "sth_ssr"), window=120),
        },
        index=frame.index,
    ).fillna(0.0)

    conviction_raw = conviction_components.mean(axis=1) - 0.25 * fragility_components.mean(axis=1)
    fragility_raw = fragility_components.mean(axis=1)
    coherence = (1.0 / (1.0 + valuation_gaps.std(axis=1))).clip(0.0, 1.0)

    out = pd.DataFrame(index=frame.index)
    out["structural_overvaluation"] = positive_strength(composite_gap, scale=1.2)
    out["structural_undervaluation"] = positive_strength(-composite_gap, scale=1.2)
    out["holder_conviction"] = _to_probability(conviction_raw, scale=1.2)
    out["holder_distribution_fragility"] = _to_probability(fragility_raw, scale=1.2)
    out["structural_reversion_pressure"] = signed_strength(
        (-composite_gap) + 0.35 * (out["holder_conviction"] - out["holder_distribution_fragility"]),
        scale=1.1,
    )
    out["structural_confidence"] = _confidence(
        pd.concat([valuation_gaps, conviction_components, fragility_components], axis=1),
        freshness=freshness,
        missingness=missingness,
        extra_support=coherence,
    )
    return out


def build_environment_targets(
    frame: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    liquidity_components = pd.DataFrame(
        {
            "fed_liquidity": robust_rolling_zscore(_series(frame, "fed_liquidity").pct_change(30), window=365),
            "m2_yoy_usd": robust_rolling_zscore(_series(frame, "m2_yoy_usd"), window=365),
            "m2_yoy_fixed_fx": robust_rolling_zscore(_series(frame, "m2_yoy_fixed_fx"), window=365),
            "dxy_inverse": robust_rolling_zscore(_series(frame, "dxy_inverse").pct_change(30), window=365),
            "liq_fair_value": robust_rolling_zscore(_series(frame, "liquidity_fair_value").pct_change(30), window=365),
        },
        index=frame.index,
    ).fillna(0.0)

    risk_components = pd.DataFrame(
        {
            "stocks": robust_rolling_zscore(_series(frame, "stocks").pct_change(30), window=365),
            "emerging_markets": robust_rolling_zscore(_series(frame, "emerging_markets").pct_change(30), window=365),
            "business_activity": robust_rolling_zscore(_series(frame, "business_activity"), window=365),
            "ism_mfg": robust_rolling_zscore(_series(frame, "manual_us_ism_manufacturing_pmi"), window=365),
            "ism_services": robust_rolling_zscore(_series(frame, "manual_us_ism_service_index"), window=365),
            "nfci": -robust_rolling_zscore(_series(frame, "manual_nfci"), window=365),
            "anfci": -robust_rolling_zscore(_series(frame, "manual_anfci"), window=365),
            "credit": -robust_rolling_zscore(_series(frame, "manual_credit"), window=365),
        },
        index=frame.index,
    ).fillna(0.0)

    liquidity_signal = liquidity_components.mean(axis=1)
    risk_signal = risk_components.mean(axis=1) + 0.30 * liquidity_signal
    transition_raw = (
        liquidity_components.diff().abs().mean(axis=1).fillna(0.0)
        + risk_components.diff().abs().mean(axis=1).fillna(0.0)
        + rolling_std(liquidity_signal, 30)
        + rolling_std(risk_signal, 30)
    )
    support = (1.0 / (1.0 + pd.concat([liquidity_components, risk_components], axis=1).std(axis=1))).clip(0.0, 1.0)

    out = pd.DataFrame(index=frame.index)
    out["liquidity_tailwind"] = positive_strength(liquidity_signal, scale=1.0)
    out["liquidity_headwind"] = positive_strength(-liquidity_signal, scale=1.0)
    out["macro_risk_on"] = positive_strength(risk_signal, scale=1.0)
    out["macro_risk_off"] = positive_strength(-risk_signal + 0.25 * robust_rolling_zscore(_series(frame, "gold").pct_change(30), window=365), scale=1.0)
    out["macro_transition_risk"] = _to_probability(transition_raw, scale=1.5)
    out["environment_confidence"] = _confidence(
        pd.concat([liquidity_components, risk_components], axis=1),
        freshness=freshness,
        missingness=missingness,
        extra_support=support,
    )
    return out


def build_edges_targets(
    frame: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    mr_components = pd.DataFrame(
        {
            "edges_pca": robust_rolling_zscore(_series(frame, "edges_pca"), window=90),
            "mr_pca": robust_rolling_zscore(_series(frame, "mr_pca"), window=90),
            "mr_mean": robust_rolling_zscore(_series_mean(frame, "mr_"), window=90),
        },
        index=frame.index,
    ).fillna(0.0)
    vol_components = pd.DataFrame(
        {
            "volatility_pca": robust_rolling_zscore(_series(frame, "volatility_pca"), window=90),
            "vol_mean": robust_rolling_zscore(_series_mean(frame, "vol_"), window=90),
        },
        index=frame.index,
    ).fillna(0.0)
    mr_signal = mr_components.mean(axis=1)
    vol_signal = vol_components.mean(axis=1)
    support = (1.0 / (1.0 + mr_components.std(axis=1) + vol_components.std(axis=1))).clip(0.0, 1.0)

    out = pd.DataFrame(index=frame.index)
    out["upside_stretch"] = positive_strength(mr_signal, scale=1.0)
    out["downside_stretch"] = positive_strength(-mr_signal, scale=1.0)
    out["mean_reversion_pressure"] = _to_probability(mr_signal.abs() + 0.30 * _series(frame, "edges_pca").abs(), scale=1.4)
    out["local_volatility_instability"] = _to_probability(vol_signal, scale=1.0)
    out["edge_asymmetry"] = signed_strength(out["upside_stretch"] - out["downside_stretch"], scale=0.5)
    out["edge_confidence"] = _confidence(
        pd.concat([mr_components, vol_components], axis=1),
        freshness=freshness,
        missingness=missingness,
        extra_support=support,
    )
    return out


def build_movement_targets(
    frame: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    trend_components = pd.DataFrame(
        {
            "movement_pca": robust_rolling_zscore(_series(frame, "movement_pca"), window=90),
            "trend_pca": robust_rolling_zscore(_series(frame, "trend_pca"), window=90),
            "trend_mean": robust_rolling_zscore(_series_mean(frame, "trend_"), window=90),
        },
        index=frame.index,
    ).fillna(0.0)
    momentum_components = pd.DataFrame(
        {
            "momentum_pca": robust_rolling_zscore(_series(frame, "momentum_pca"), window=60),
            "momentum_mean": robust_rolling_zscore(_series_mean(frame, "momentum_"), window=60),
        },
        index=frame.index,
    ).fillna(0.0)
    mreg_components = pd.DataFrame(
        {
            "mreg_pca": robust_rolling_zscore(_series(frame, "mreg_pca"), window=60),
            "mreg_mean": robust_rolling_zscore(_series_mean(frame, "mreg_"), window=60),
        },
        index=frame.index,
    ).fillna(0.0)

    trend_signal = trend_components.mean(axis=1)
    momentum_signal = momentum_components.mean(axis=1)
    mreg_signal = mreg_components.mean(axis=1)
    chop_trend_weight = _series(frame, "chop_trend_weight").fillna(0.5).clip(0.0, 1.0)
    chop_mean_reversion_weight = _series(frame, "chop_mean_reversion_weight").fillna(0.5).clip(0.0, 1.0)
    chop_balance = _series(frame, "chop_regime_balance").fillna(0.0).clip(-1.0, 1.0)
    directional = (
        trend_signal * (0.75 + 0.50 * chop_trend_weight)
        + 0.5 * momentum_signal * (0.50 + 0.50 * chop_trend_weight)
        - 0.25 * mreg_signal.abs() * (0.50 + chop_mean_reversion_weight)
    )
    persistence_raw = (
        rolling_mean(trend_signal.abs(), 10)
        - rolling_std(trend_signal, 10)
        - 0.25 * mreg_signal.abs()
        + 0.75 * (chop_trend_weight - 0.5)
    )
    chop_raw = -trend_signal.abs() + mreg_signal.abs() + rolling_std(directional, 10) + 1.15 * chop_balance
    support = (1.0 / (1.0 + pd.concat([trend_components, momentum_components, mreg_components], axis=1).std(axis=1))).clip(0.0, 1.0)

    out = pd.DataFrame(index=frame.index)
    out["trend_pressure_up"] = positive_strength(directional, scale=1.0)
    out["trend_pressure_down"] = positive_strength(-directional, scale=1.0)
    out["trend_persistence"] = _to_probability(persistence_raw, scale=1.1)
    out["momentum_quality"] = _to_probability(
        momentum_signal * (0.75 + 0.25 * chop_trend_weight) - 0.35 * mreg_signal.abs(),
        scale=1.0,
    )
    out["chop_risk"] = _to_probability(chop_raw, scale=1.1)
    out["trend_strategy_influence"] = _series(frame, "strategy_trend_influence").fillna(
        chop_trend_weight / 2.0
    ).clip(0.0, 1.0)
    out["momentum_strategy_influence"] = _series(frame, "strategy_momentum_influence").fillna(0.0).clip(0.0, 1.0)
    out["mean_reversion_strategy_influence"] = _series(
        frame,
        "strategy_mean_reversion_influence",
    ).fillna(chop_mean_reversion_weight / 2.0).clip(0.0, 1.0)
    out["movement_confidence"] = _confidence(
        pd.concat([trend_components, momentum_components, mreg_components], axis=1),
        freshness=freshness,
        missingness=missingness,
        extra_support=support,
    )
    return out


def build_semantic_targets(
    close: pd.Series,
    buckets: dict[str, pd.DataFrame],
    freshness_scores: dict[str, pd.Series],
    missingness: dict[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    targets = {
        "structure": build_structure_targets(
            frame=buckets["structure"],
            close=close,
            freshness=freshness_scores["structure"],
            missingness=missingness["structure"],
        ),
        "environment": build_environment_targets(
            frame=buckets["environment"],
            freshness=freshness_scores["environment"],
            missingness=missingness["environment"],
        ),
        "edges": build_edges_targets(
            frame=buckets["edges"],
            freshness=freshness_scores["edges"],
            missingness=missingness["edges"],
        ),
        "movement": build_movement_targets(
            frame=buckets["movement"],
            freshness=freshness_scores["movement"],
            missingness=missingness["movement"],
        ),
    }
    if "liquidation" in buckets:
        targets["liquidation"] = build_liquidation_semantic_targets(buckets["liquidation"])
    return targets


def append_meta_channels(
    bucket_name: str,
    states: pd.DataFrame,
    freshness: pd.Series,
    missingness: pd.Series,
) -> pd.DataFrame:
    out = states.copy()

    if bucket_name == "structure":
        directional = out["structural_reversion_pressure"]
        confidence = out["structural_confidence"]
    elif bucket_name == "environment":
        directional = (out["liquidity_tailwind"] - out["liquidity_headwind"]) + (out["macro_risk_on"] - out["macro_risk_off"])
        confidence = out["environment_confidence"]
    elif bucket_name == "edges":
        directional = out["upside_stretch"] - out["downside_stretch"]
        confidence = out["edge_confidence"]
    elif bucket_name == "liquidation":
        directional = out["liquidation_directional_pressure"] + 0.50 * out["cascade_asymmetry"]
        confidence = out["liquidation_confidence"]
    else:
        directional = out["trend_pressure_up"] - out["trend_pressure_down"]
        confidence = out["movement_confidence"]

    velocity = ema(directional.diff().abs().fillna(0.0), span=5)
    state_dispersion = out.apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)
    hazard = pd.Series(sigmoid(2.0 * velocity + 0.75 * state_dispersion + (1.0 - confidence), scale=1.0), index=out.index)
    validity = (confidence * freshness * (1.0 - missingness)).clip(0.0, 1.0)

    out["hazard"] = hazard.clip(0.0, 1.0)
    out["freshness"] = freshness.clip(0.0, 1.0)
    out["validity"] = validity.clip(0.0, 1.0)
    return out
