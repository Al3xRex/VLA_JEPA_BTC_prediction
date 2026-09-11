from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dual_model_forecaster.utils import bounded_correlation, quantile_name


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1.0) * errors)
    return float(np.mean(loss))


def weighted_interval_score(
    y_true: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
) -> float:
    mask = np.isfinite(y_true)
    for values in quantile_predictions.values():
        mask = mask & np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    quantile_predictions = {q: values[mask] for q, values in quantile_predictions.items()}
    q05 = quantile_predictions[0.05]
    q25 = quantile_predictions[0.25]
    q50 = quantile_predictions[0.50]
    q75 = quantile_predictions[0.75]
    q95 = quantile_predictions[0.95]

    alpha_90 = 0.10
    alpha_50 = 0.50

    score_90 = (q95 - q05) + (2.0 / alpha_90) * np.maximum(q05 - y_true, 0.0) + (2.0 / alpha_90) * np.maximum(y_true - q95, 0.0)
    score_50 = (q75 - q25) + (2.0 / alpha_50) * np.maximum(q25 - y_true, 0.0) + (2.0 / alpha_50) * np.maximum(y_true - q75, 0.0)
    median_loss = np.abs(y_true - q50)
    return float(np.mean((0.5 * median_loss) + (0.25 * score_50) + (0.25 * score_90)))


def summarize_quantile_metrics(
    actuals: pd.DataFrame,
    predictions: dict[int, pd.DataFrame],
    quantiles: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for horizon, frame in predictions.items():
        y_true = actuals[f"target_{horizon}d"].to_numpy(dtype=float)
        q_map = {
            q: frame[quantile_name(q)].to_numpy(dtype=float)
            for q in quantiles
        }
        horizon_pinballs = []
        for q in quantiles:
            loss = pinball_loss(y_true, q_map[q], q)
            horizon_pinballs.append(loss)
            calibration_rows.append(
                {
                    "horizon": horizon,
                    "quantile": q,
                    "nominal": q,
                    "empirical": float(np.mean(y_true <= q_map[q])),
                }
            )

        row = {
            "horizon": horizon,
            "avg_pinball": float(np.mean(horizon_pinballs)),
            "coverage_50": float(np.nanmean((y_true >= q_map[0.25]) & (y_true <= q_map[0.75]))),
            "coverage_90": float(np.nanmean((y_true >= q_map[0.05]) & (y_true <= q_map[0.95]))),
            "coverage_gap_50": float(abs(np.nanmean((y_true >= q_map[0.25]) & (y_true <= q_map[0.75])) - 0.50)),
            "coverage_gap_90": float(abs(np.nanmean((y_true >= q_map[0.05]) & (y_true <= q_map[0.95])) - 0.90)),
            "interval_width_50": float(np.nanmean(q_map[0.75] - q_map[0.25])),
            "interval_width_90": float(np.nanmean(q_map[0.95] - q_map[0.05])),
            "wis": weighted_interval_score(y_true, q_map),
            "crps_quantile_approx": float(2.0 * np.mean(horizon_pinballs)),
        }
        for q, loss in zip(quantiles, horizon_pinballs):
            row[f"{quantile_name(q)}_pinball"] = float(loss)
        rows.append(row)

    metrics_frame = pd.DataFrame(rows)
    average_calibration_gap = float(
        0.5 * metrics_frame["coverage_gap_50"].mean() + 0.5 * metrics_frame["coverage_gap_90"].mean()
    )
    summary = {
        "by_horizon": metrics_frame.to_dict(orient="records"),
        "average_pinball": float(metrics_frame["avg_pinball"].mean()),
        "average_wis": float(metrics_frame["wis"].mean()),
        "average_calibration_gap": average_calibration_gap,
        "selection_score": float(metrics_frame["wis"].mean() + average_calibration_gap),
        "calibration": calibration_rows,
    }
    return summary


def specialist_regression_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    confidence_column: str,
) -> dict[str, float]:
    aligned = y_true.align(y_pred, join="inner", axis=0)
    true_frame, pred_frame = aligned
    errors = (pred_frame - true_frame).astype(float)
    mae = float(errors.abs().mean().mean())
    mse = float(np.square(errors.to_numpy(dtype=float)).mean())
    smoothness = float(pred_frame.diff().abs().mean().mean())

    utility = 0.0
    if confidence_column in pred_frame.columns:
        point_error = errors.abs().mean(axis=1)
        utility = -bounded_correlation(point_error, pred_frame[confidence_column])

    return {
        "mae": mae,
        "mse": mse,
        "smoothness": smoothness,
        "confidence_utility": utility,
        "selection_score": mse + 0.20 * smoothness - 0.10 * utility,
    }
