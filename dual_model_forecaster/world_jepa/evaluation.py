from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_COLUMNS = tuple(f"q{int(value * 100):02d}" for value in QUANTILES)


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float)
    return np.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def interval_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def weighted_interval_score(frame: pd.DataFrame) -> np.ndarray:
    """Canonical WIS for median plus central 50% and 90% intervals."""

    actual = frame["actual_return"].to_numpy(dtype=float)
    median = frame["q50"].to_numpy(dtype=float)
    terms = 0.5 * np.abs(actual - median)
    terms += 0.05 * interval_score(actual, frame["q05"], frame["q95"], 0.10)
    terms += 0.25 * interval_score(actual, frame["q25"], frame["q75"], 0.50)
    return terms / 2.5


def add_forecast_losses(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES):
        out[f"{column}_pinball"] = pinball_loss(out["actual_return"], out[column], quantile)
    out["avg_pinball"] = out[[f"{column}_pinball" for column in QUANTILE_COLUMNS]].mean(axis=1)
    out["wis"] = weighted_interval_score(out)
    out["absolute_median_error"] = (out["actual_return"] - out["q50"]).abs()
    out["coverage_50"] = ((out["actual_return"] >= out["q25"]) & (out["actual_return"] <= out["q75"])).astype(float)
    out["coverage_90"] = ((out["actual_return"] >= out["q05"]) & (out["actual_return"] <= out["q95"])).astype(float)
    out["lower_tail_miss"] = (out["actual_return"] < out["q05"]).astype(float)
    out["upper_tail_miss"] = (out["actual_return"] > out["q95"]).astype(float)
    sign_mask = out["actual_return"].ne(0.0) & out["q50"].ne(0.0)
    out["direction_hit"] = np.nan
    out.loc[sign_mask, "direction_hit"] = (
        np.sign(out.loc[sign_mask, "actual_return"]) == np.sign(out.loc[sign_mask, "q50"])
    ).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def summarize_forecasts(frame: pd.DataFrame, *, forecaster: str) -> pd.DataFrame:
    scored = add_forecast_losses(frame).dropna(subset=["actual_return", "wis"])
    rows: list[dict[str, float | int | str]] = []
    for horizon, group in scored.groupby("horizon", sort=True):
        rows.append(
            {
                "forecaster": forecaster,
                "horizon": float(horizon),
                "observations": int(len(group)),
                "wis": float(group["wis"].mean()),
                "avg_pinball": float(group["avg_pinball"].mean()),
                "median_mae": float(group["absolute_median_error"].mean()),
                "coverage_50": float(group["coverage_50"].mean()),
                "coverage_90": float(group["coverage_90"].mean()),
                "q05_miss_rate": float(group["lower_tail_miss"].mean()),
                "q95_miss_rate": float(group["upper_tail_miss"].mean()),
                "directional_hit_rate": float(group["direction_hit"].mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class QuantileCalibrator:
    adjustments: dict[float, dict[str, float]]

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        fitted_horizons = np.asarray(sorted(self.adjustments), dtype=float)
        query_horizons = pd.to_numeric(out["horizon"], errors="coerce").to_numpy(dtype=float)
        for column in QUANTILE_COLUMNS:
            fitted_adjustments = np.asarray(
                [self.adjustments[float(horizon)][column] for horizon in fitted_horizons],
                dtype=float,
            )
            interpolated = np.interp(query_horizons, fitted_horizons, fitted_adjustments)
            out[column] = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float) + interpolated
        values = np.sort(out.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float), axis=1)
        out.loc[:, QUANTILE_COLUMNS] = values
        return out


def fit_quantile_calibrator(selection: pd.DataFrame) -> QuantileCalibrator:
    clean = selection.dropna(subset=["actual_return", *QUANTILE_COLUMNS]).copy()
    adjustments: dict[float, dict[str, float]] = {}
    for horizon, group in clean.groupby("horizon", sort=True):
        adjustments[float(horizon)] = {
            column: float(np.quantile(group["actual_return"].to_numpy() - group[column].to_numpy(), quantile))
            for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES)
        }
    if not adjustments:
        raise ValueError("No realized selection rows are available to fit the quantile calibrator.")
    return QuantileCalibrator(adjustments=adjustments)


def weighted_quantile(values: np.ndarray, quantiles: Iterable[float], weights: np.ndarray) -> np.ndarray:
    requested = tuple(float(value) for value in quantiles)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return np.full(len(requested), np.nan)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(requested, dtype=float), cumulative, values)


def causal_ewma_baseline(
    close: pd.Series,
    as_ofs: Iterable[pd.Timestamp],
    horizons: Iterable[float],
    *,
    span: int = 180,
    minimum_history: int = 180,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    log_close = np.log(close.astype(float))
    rows: list[dict[str, float | pd.Timestamp]] = []
    for as_of in (pd.Timestamp(value) for value in as_ofs):
        if as_of not in log_close.index:
            continue
        for horizon_value in horizons:
            horizon = int(round(float(horizon_value)))
            returns = (log_close.shift(-horizon) - log_close).loc[: as_of - pd.Timedelta(days=horizon)].dropna()
            if len(returns) < minimum_history:
                continue
            positions = np.arange(len(returns), dtype=float)
            decay = math.log(2.0) / max(float(span), 1.0)
            weights = np.exp(-decay * (positions[-1] - positions))
            quantile_values = weighted_quantile(returns.to_numpy(dtype=float), QUANTILES, weights)
            target = as_of + pd.Timedelta(days=horizon)
            actual = float(log_close.loc[target] - log_close.loc[as_of]) if target in log_close.index else np.nan
            row: dict[str, float | pd.Timestamp] = {
                "as_of": as_of,
                "horizon": float(horizon_value),
                "actual_return": actual,
            }
            row.update({column: float(value) for column, value in zip(QUANTILE_COLUMNS, quantile_values)})
            rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap_difference(
    candidate_loss: np.ndarray,
    baseline_loss: np.ndarray,
    *,
    block_length: int,
    samples: int = 1000,
    seed: int = 7,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate_loss, dtype=float)
    baseline = np.asarray(baseline_loss, dtype=float)
    finite = np.isfinite(candidate) & np.isfinite(baseline)
    difference = candidate[finite] - baseline[finite]
    n = len(difference)
    if n == 0:
        raise ValueError("No paired finite losses are available for the bootstrap.")
    block = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block + 1)
    means = np.empty(int(samples), dtype=float)
    blocks_needed = int(math.ceil(n / block))
    for sample_index in range(int(samples)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate([difference[start : start + block] for start in chosen])[:n]
        means[sample_index] = float(draw.mean())
    return {
        "observations": n,
        "block_length": block,
        "samples": int(samples),
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "probability_candidate_better": float(np.mean(means < 0.0)),
    }


def evaluate_promotion_gates(
    candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    attention: pd.DataFrame | None,
    gates: dict[str, object],
    *,
    bootstrap: dict[str, float | int] | None = None,
    temporal_safety_passed: bool = True,
    tests_passed: bool = True,
) -> dict[str, object]:
    merged = candidate_metrics.merge(
        baseline_metrics[["horizon", "wis"]].rename(columns={"wis": "baseline_wis"}),
        on="horizon",
        how="left",
    )
    wis_pass = bool((merged["wis"] < merged["baseline_wis"]).all()) if len(merged) else False
    tail_limit = float(gates.get("maximum_q05_miss_rate", 1.0))
    tail_pass = bool((candidate_metrics["q05_miss_rate"] <= tail_limit).all()) if len(candidate_metrics) else False
    max_attention = 0.0
    min_attention = 0.0
    if attention is not None and not attention.empty:
        weight_columns = [column for column in attention if column.startswith("attention_")]
        if weight_columns:
            mean_attention = attention[weight_columns].mean(axis=0)
            max_attention = float(mean_attention.max())
            min_attention = float(mean_attention.min())
    attention_limit = float(gates.get("maximum_single_world_attention_share", 1.0))
    attention_pass = max_attention <= attention_limit
    minimum_attention_limit = float(gates.get("minimum_world_attention_share", 0.0))
    world_usage_pass = min_attention >= minimum_attention_limit
    bootstrap_required = bool(gates.get("require_block_bootstrap_confidence", False))
    bootstrap_pass = (
        bootstrap is not None and float(bootstrap.get("ci_high", float("inf"))) < 0.0
        if bootstrap_required
        else True
    )
    temporal_pass = bool(temporal_safety_passed) if bool(
        gates.get("require_no_temporal_leakage", False)
    ) else True
    test_gate_pass = bool(tests_passed) if bool(gates.get("require_all_tests", False)) else True
    return {
        "wis_better_than_baseline": wis_pass,
        "q05_miss_rate_within_limit": tail_pass,
        "attention_not_collapsed": attention_pass,
        "all_worlds_receive_attention": world_usage_pass,
        "block_bootstrap_confidence": bootstrap_pass,
        "temporal_safety": temporal_pass,
        "required_tests": test_gate_pass,
        "maximum_mean_world_attention": max_attention,
        "minimum_mean_world_attention": min_attention,
        "passed": bool(
            wis_pass
            and tail_pass
            and attention_pass
            and world_usage_pass
            and bootstrap_pass
            and temporal_pass
            and test_gate_pass
        ),
    }
