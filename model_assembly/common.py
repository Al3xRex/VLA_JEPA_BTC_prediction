from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dual_model_forecaster.brutal_baselines import (
    build_kalman_projection_overlay,
    build_kalman_state_history,
)
from dual_model_forecaster.config import load_config
from dual_model_forecaster.data import load_forecast_data
from dual_model_forecaster.final_selection import (
    _feature_payload_for_candidate,
    fit_candidate_model,
    predict_live_candidate_model,
)
from dual_model_forecaster.semantics import build_semantic_targets
from dual_model_forecaster.specialists import _fit_final_model, emit_specialist_states
from dual_model_forecaster.splits import build_inner_segments
from dual_model_forecaster.utils import ensure_dir, set_global_seed, write_json


FINAL_CONFIG_DIR = Path("configs/final")


def load_doc(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_doc(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    write_json(path, payload)


def _format_horizon_label(horizons: list[int]) -> str:
    return " / ".join(f"{int(horizon)}d" for horizon in sorted(horizons))


def load_runtime(
    config_dir: str | Path = FINAL_CONFIG_DIR,
    as_of_timestamp: Any | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir)
    specialists_cfg = load_doc(config_dir / "specialists.yaml")
    roles_cfg = load_doc(config_dir / "roles.yaml")
    fusion_cfg = load_doc(config_dir / "fusion.yaml")
    calibration_cfg = load_doc(config_dir / "calibration.yaml")
    refit_cfg = load_doc(config_dir / "refit.yaml")
    base_config = load_config(refit_cfg["base_config"])
    set_global_seed(int(base_config["seed"]))
    data_bundle = load_forecast_data(base_config, end_timestamp=as_of_timestamp)
    semantic_targets = build_semantic_targets(
        close=data_bundle.close,
        buckets=data_bundle.buckets,
        freshness_scores=data_bundle.freshness_scores,
        missingness=data_bundle.missingness,
    )
    artifact_root = ensure_dir(Path(refit_cfg["artifact_root"]))
    return {
        "config_dir": str(config_dir),
        "base_config": base_config,
        "specialists_cfg": specialists_cfg,
        "roles_cfg": roles_cfg,
        "fusion_cfg": fusion_cfg,
        "calibration_cfg": calibration_cfg,
        "refit_cfg": refit_cfg,
        "data_bundle": data_bundle,
        "semantic_targets": semantic_targets,
        "artifact_root": artifact_root,
        "device": str(base_config["device"]),
        "as_of_timestamp": str(pd.Timestamp(as_of_timestamp)) if as_of_timestamp is not None else None,
    }


def _candidate_lookup(base_config: dict[str, Any], bucket_name: str) -> dict[str, dict[str, Any]]:
    return {
        candidate["name"]: copy.deepcopy(candidate)
        for candidate in base_config["specialists"]["buckets"][bucket_name]["candidates"]
    }


def build_frozen_specialist_selection(
    base_config: dict[str, Any],
    specialists_cfg: dict[str, Any],
) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    for bucket_name, bucket_cfg in specialists_cfg["buckets"].items():
        lookup = _candidate_lookup(base_config, bucket_name)
        retained = []
        for retained_cfg in bucket_cfg["retained"]:
            retained.append(
                {
                    "group_name": retained_cfg["group_name"],
                    "candidate": lookup[retained_cfg["name"]],
                }
            )
        selection[bucket_name] = {
            "best_candidate": lookup[bucket_cfg["anchor"]],
            "retained_candidates": retained,
            "post_smooth_alpha": float(base_config["specialists"]["buckets"][bucket_name]["post_smooth_alpha"]),
        }
    return selection


def build_selected_candidate(
    base_config: dict[str, Any],
    roles_cfg: dict[str, Any],
    fusion_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": fusion_cfg["name"],
        "family": fusion_cfg["family"],
        "horizon_set_name": fusion_cfg["horizon_set_name"],
        "horizons": list(fusion_cfg["horizons"]),
        "output_style": fusion_cfg["output_style"],
        "role_map_name": roles_cfg["role_map_name"],
        "interface_name": roles_cfg["interface_name"],
        "architecture": copy.deepcopy(fusion_cfg["architecture"]),
    }


def _train_end_pos(data_bundle: Any, candidate: dict[str, Any]) -> int:
    return len(data_bundle.close.index) - max(candidate["horizons"])


def _live_start(base_config: dict[str, Any], data_bundle: Any) -> int:
    return max(0, len(data_bundle.close.index) - int(base_config["fusion"]["state_window"]) - 5)


def materialize_specialists(
    runtime: dict[str, Any],
    save_artifacts: bool = True,
) -> dict[str, Any]:
    base_config = runtime["base_config"]
    data_bundle = runtime["data_bundle"]
    semantic_targets = runtime["semantic_targets"]
    specialists_cfg = runtime["specialists_cfg"]
    candidate = build_selected_candidate(base_config, runtime["roles_cfg"], runtime["fusion_cfg"])
    specialist_selection = build_frozen_specialist_selection(base_config, specialists_cfg)
    train_end_pos = _train_end_pos(data_bundle, candidate)
    inner_segments = build_inner_segments(data_bundle.close.index, train_end_pos=train_end_pos, config=base_config)
    live_start = _live_start(base_config, data_bundle)

    train_states_by_bucket: dict[str, pd.DataFrame] = {}
    live_history_by_bucket: dict[str, pd.DataFrame] = {}
    saved_selection: dict[str, Any] = {}
    artifact_root = Path(runtime["artifact_root"])
    device = runtime["device"]

    for bucket_name in specialists_cfg["buckets"]:
        bucket_artifact_root = ensure_dir(artifact_root / bucket_name)
        selection = specialist_selection[bucket_name]
        saved_selection[bucket_name] = selection

        for retained in selection["retained_candidates"]:
            candidate_cfg = retained["candidate"]
            model, scaler = _fit_final_model(
                features=data_bundle.buckets[bucket_name].iloc[:train_end_pos],
                targets=semantic_targets[bucket_name].iloc[:train_end_pos],
                train_end_pos=train_end_pos,
                candidate=candidate_cfg,
                training_cfg=base_config["specialists"]["training"],
                device=device,
            )
            if save_artifacts:
                candidate_dir = ensure_dir(bucket_artifact_root / candidate_cfg["name"])
                torch.save(model.state_dict(), candidate_dir / "checkpoint.pt")
                save_doc(candidate_dir / "scaler.json", scaler.to_dict())
                save_doc(
                    candidate_dir / "manifest.json",
                    {
                        "bucket_name": bucket_name,
                        "group_name": retained["group_name"],
                        "candidate": candidate_cfg,
                    },
                )

        train_parts = []
        for segment in inner_segments:
            train_parts.append(
                emit_specialist_states(
                    bucket_name=bucket_name,
                    features=data_bundle.buckets[bucket_name],
                    targets=semantic_targets[bucket_name],
                    freshness=data_bundle.freshness_scores[bucket_name],
                    missingness=data_bundle.missingness[bucket_name],
                    train_end_pos=int(segment["fit_end_pos"]),
                    predict_start_pos=int(segment["predict_start_pos"]),
                    predict_end_pos=int(segment["predict_end_pos"]),
                    selection=selection,
                    config=base_config,
                    device=device,
                )
            )
        train_states = pd.concat(train_parts).sort_index()
        live_history = emit_specialist_states(
            bucket_name=bucket_name,
            features=data_bundle.buckets[bucket_name],
            targets=semantic_targets[bucket_name],
            freshness=data_bundle.freshness_scores[bucket_name],
            missingness=data_bundle.missingness[bucket_name],
            train_end_pos=len(data_bundle.close.index),
            predict_start_pos=live_start,
            predict_end_pos=len(data_bundle.close.index),
            selection=selection,
            config=base_config,
            device=device,
        )
        train_states_by_bucket[bucket_name] = train_states
        live_history_by_bucket[bucket_name] = live_history

        if save_artifacts:
            train_states.to_csv(bucket_artifact_root / "train_states.csv")
            live_history.to_csv(bucket_artifact_root / "live_history_states.csv")
            save_doc(
                bucket_artifact_root / "selection.json",
                {
                    "bucket_name": bucket_name,
                    "anchor": selection["best_candidate"]["name"],
                    "retained": [
                        {"group_name": item["group_name"], "name": item["candidate"]["name"]}
                        for item in selection["retained_candidates"]
                    ],
                    "selection": selection,
                },
            )

    return {
        "specialist_selection": specialist_selection,
        "train_states_by_bucket": train_states_by_bucket,
        "live_history_by_bucket": live_history_by_bucket,
        "train_end_pos": train_end_pos,
    }


def materialize_role_router(
    runtime: dict[str, Any],
    train_states_by_bucket: dict[str, pd.DataFrame],
    live_history_by_bucket: dict[str, pd.DataFrame],
    save_artifacts: bool = True,
) -> dict[str, Any]:
    base_config = runtime["base_config"]
    candidate = build_selected_candidate(base_config, runtime["roles_cfg"], runtime["fusion_cfg"])
    feature_train, payload = _feature_payload_for_candidate(candidate, train_states_by_bucket)
    feature_metadata = {key: value for key, value in payload.items() if key != "feature_frame"}
    candidate["feature_metadata"] = feature_metadata
    feature_live, _ = _feature_payload_for_candidate(candidate, live_history_by_bucket)

    if save_artifacts:
        fusion_root = ensure_dir(Path(runtime["artifact_root"]) / "fusion")
        feature_train.to_csv(fusion_root / "train_features.csv")
        feature_live.to_csv(fusion_root / "live_features.csv")
        save_doc(fusion_root / "router_metadata.json", feature_metadata)
        save_doc(fusion_root / "candidate.json", candidate)

    return {
        "candidate": candidate,
        "feature_train": feature_train,
        "feature_live": feature_live,
    }


def _training_targets(
    data_bundle: Any,
    candidate: dict[str, Any],
    feature_train: pd.DataFrame,
) -> pd.DataFrame:
    target_train = data_bundle.targets.loc[feature_train.index, [f"target_{h}d" for h in candidate["horizons"]]].dropna()
    common_train_index = feature_train.index.intersection(target_train.index).sort_values()
    return target_train.loc[common_train_index]


def _unique_methods(methods: list[str]) -> list[str]:
    return list(dict.fromkeys(methods))


def _calibration_deployment_report(
    calibration_cfg: dict[str, Any],
    fitted: dict[str, Any],
) -> dict[str, Any]:
    metrics = fitted["calibration_selection"]
    default_method = calibration_cfg["default"]
    if default_method not in metrics:
        raise ValueError(f"Default calibrator {default_method!r} was not evaluated.")
    best_method = min(metrics, key=lambda name: metrics[name]["selection_score"])
    default_score = float(metrics[default_method]["selection_score"])
    best_score = float(metrics[best_method]["selection_score"])
    threshold = float(calibration_cfg.get("replacement_threshold_abs", 0.02))
    replacement_recommended = best_method != default_method and (default_score - best_score) > threshold
    deployed_method = best_method if replacement_recommended else default_method
    return {
        "default_method": default_method,
        "best_method": best_method,
        "deployed_method": deployed_method,
        "replacement_recommended": replacement_recommended,
        "replacement_threshold_abs": threshold,
        "metrics": metrics,
    }


def _deployed_calibration_metrics(calibration_report: dict[str, Any]) -> dict[str, Any]:
    deployed_method = calibration_report["deployed_method"]
    metrics = calibration_report["metrics"][deployed_method]
    calibration_map = {
        (int(row["horizon"]), float(row["quantile"])): float(row["empirical"])
        for row in metrics.get("calibration", [])
    }

    width_diagnostics = []
    lower_tail_diagnostics = []
    upper_tail_diagnostics = []
    for row in metrics.get("by_horizon", []):
        horizon = int(row["horizon"])
        q05_empirical = calibration_map.get((horizon, 0.05), float("nan"))
        q95_empirical = calibration_map.get((horizon, 0.95), float("nan"))
        width_diagnostics.append(
            {
                "horizon": horizon,
                "interval_width_50": float(row["interval_width_50"]),
                "interval_width_90": float(row["interval_width_90"]),
                "coverage_gap_50": float(row["coverage_gap_50"]),
                "coverage_gap_90": float(row["coverage_gap_90"]),
                "wis": float(row["wis"]),
            }
        )
        lower_tail_diagnostics.append(
            {
                "horizon": horizon,
                "q05_empirical": q05_empirical,
                "q05_miss_rate": q05_empirical,
                "q05_gap": abs(q05_empirical - 0.05),
                "q05_pinball": float(row["q05_pinball"]),
            }
        )
        upper_tail_diagnostics.append(
            {
                "horizon": horizon,
                "q95_empirical": q95_empirical,
                "q95_miss_rate": max(0.0, 1.0 - q95_empirical),
                "q95_gap": abs(q95_empirical - 0.95),
                "q95_pinball": float(row["q95_pinball"]),
            }
        )

    return {
        "selected_calibrator": deployed_method,
        "calibration_score_comparison": calibration_report,
        "width_diagnostics": width_diagnostics,
        "lower_tail_diagnostics": lower_tail_diagnostics,
        "upper_tail_diagnostics": upper_tail_diagnostics,
    }


def _project_price(base_price: float, forecast_return: float, return_type: str) -> float:
    if return_type == "log":
        return float(base_price * math.exp(float(forecast_return)))
    return float(base_price * (1.0 + float(forecast_return)))


def _save_live_forecast_plot(
    runtime: dict[str, Any],
    live_forecast: dict[str, list[dict[str, Any]]],
    diagnostics: dict[str, Any],
    save_dir: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    close = runtime["data_bundle"].close.dropna()
    if close.empty:
        raise ValueError("Cannot build forecast plot without close history.")

    history = close.iloc[-min(180, len(close)) :]
    last_timestamp = pd.Timestamp(history.index[-1])
    last_close = float(history.iloc[-1])
    return_type = str(runtime["base_config"]["data"]["return_type"])

    points: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for horizon in sorted(int(key) for key in live_forecast):
        row = live_forecast[str(horizon)][0]
        as_of = pd.Timestamp(row.get("timestamp", last_timestamp))
        date = last_timestamp + pd.Timedelta(days=horizon)
        points.append(
            {
                "horizon": horizon,
                "date": date,
                "q05_price": _project_price(last_close, row["q05"], return_type),
                "q25_price": _project_price(last_close, row["q25"], return_type),
                "q50_price": _project_price(last_close, row["q50"], return_type),
                "q75_price": _project_price(last_close, row["q75"], return_type),
                "q95_price": _project_price(last_close, row["q95"], return_type),
                "q50_return": float(row["q50"]),
            }
        )
        projection_rows.append(
            {
                "as_of": as_of,
                "target_timestamp": date,
                "horizon": horizon,
                "base_close": last_close,
            }
        )

    future_dates = [last_timestamp] + [point["date"] for point in points]
    q05_prices = [last_close] + [point["q05_price"] for point in points]
    q25_prices = [last_close] + [point["q25_price"] for point in points]
    q50_prices = [last_close] + [point["q50_price"] for point in points]
    q75_prices = [last_close] + [point["q75_price"] for point in points]
    q95_prices = [last_close] + [point["q95_price"] for point in points]
    horizons = [point["horizon"] for point in points]
    kalman_history = build_kalman_state_history(runtime["data_bundle"])
    kalman_projection = pd.DataFrame()
    if projection_rows:
        kalman_projection = build_kalman_projection_overlay(
            predictions=pd.DataFrame(projection_rows),
            data_bundle=runtime["data_bundle"],
            horizons=horizons,
            return_type=return_type,
        )

    width_df = pd.DataFrame(diagnostics["width_diagnostics"]).set_index("horizon").sort_index()
    lower_tail_df = pd.DataFrame(diagnostics["lower_tail_diagnostics"]).set_index("horizon").sort_index()
    diagnostic_horizons = width_df.index.to_list()
    horizon_label = _format_horizon_label(diagnostic_horizons)

    fig, (ax_price, ax_diag) = plt.subplots(
        2,
        1,
        figsize=(13, 7.8),
        gridspec_kw={"height_ratios": [3.4, 1.25]},
        constrained_layout=True,
    )

    ax_price.plot(history.index, history.to_numpy(dtype=float), color="#223a5e", linewidth=2.0, label="Close")
    if not kalman_history.empty and "kalman_fair_value_price" in kalman_history.columns:
        recent_kalman = kalman_history.loc[kalman_history.index >= history.index[0]].dropna(
            subset=["kalman_fair_value_price"]
        )
        if not recent_kalman.empty:
            ax_price.plot(
                recent_kalman.index,
                recent_kalman["kalman_fair_value_price"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.55,
                alpha=0.9,
                label="Kalman fair value",
            )
    ax_price.fill_between(future_dates, q05_prices, q95_prices, color="#7aa6d1", alpha=0.18, label="q05-q95")
    ax_price.fill_between(future_dates, q25_prices, q75_prices, color="#2f6da3", alpha=0.30, label="q25-q75")
    ax_price.plot(future_dates, q50_prices, color="#b23a48", linewidth=2.2, marker="o", label="q50")

    latest_high_vol = float("nan")
    latest_tail_flare = float("nan")
    if not kalman_history.empty:
        latest_signal = kalman_history.loc[kalman_history.index <= last_timestamp].tail(1)
        if not latest_signal.empty:
            row = latest_signal.iloc[0]
            if "kalman_high_vol_signal" in row and pd.notna(row["kalman_high_vol_signal"]):
                latest_high_vol = float(row["kalman_high_vol_signal"])
            if "tail_flare_score" in row and pd.notna(row["tail_flare_score"]):
                latest_tail_flare = float(row["tail_flare_score"])
    high_vol_fires = math.isfinite(latest_high_vol) and latest_high_vol >= 0.50
    tail_flare_fires = math.isfinite(latest_tail_flare) and latest_tail_flare >= 1.00
    if high_vol_fires or tail_flare_fires:
        overlay_parts = []
        if high_vol_fires:
            overlay_parts.append(f"high-vol {latest_high_vol:.2f}")
        if tail_flare_fires:
            overlay_parts.append(f"tail flare {latest_tail_flare:.2f}")
        overlay_label = " / ".join(overlay_parts)
        ax_price.axvspan(
            last_timestamp,
            max(future_dates),
            color="#f97316" if high_vol_fires else "#7c3aed",
            alpha=0.09,
            label="Tail/high-vol overlay",
            zorder=0,
        )
        ax_price.annotate(
            f"Risk overlay active\n{overlay_label}",
            xy=(last_timestamp, last_close),
            xytext=(12, -38),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="#7c2d12" if high_vol_fires else "#4c1d95",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#9a3412" if high_vol_fires else "#6d28d9"},
        )
    if not kalman_projection.empty:
        kalman_projected = kalman_projection.dropna(
            subset=["kalman_projected_price", "kalman_target_timestamp"]
        ).sort_values("horizon")
        if not kalman_projected.empty:
            kalman_row = kalman_projected.iloc[0]
            kalman_target = pd.Timestamp(kalman_row["kalman_target_timestamp"])
            kalman_price = float(kalman_row["kalman_projected_price"])
            if pd.notna(kalman_target) and math.isfinite(kalman_price) and kalman_target > last_timestamp:
                ax_price.plot(
                    [last_timestamp, kalman_target],
                    [last_close, kalman_price],
                    color="#16a34a",
                    linestyle="--",
                    linewidth=1.85,
                    label="Kalman projected scope",
                )
                ax_price.scatter([kalman_target], [kalman_price], color="#16a34a", s=28, zorder=5)
                scope_days = float(kalman_row.get("kalman_scope_days", float("nan")))
                scope_label = (
                    f"Kalman {scope_days:.0f}d\n{kalman_price:,.0f}"
                    if math.isfinite(scope_days)
                    else f"Kalman\n{kalman_price:,.0f}"
                )
                ax_price.annotate(
                    scope_label,
                    xy=(kalman_target, kalman_price),
                    xytext=(8, 8),
                    textcoords="offset points",
                    ha="left",
                    fontsize=8,
                    color="#166534",
                )
    ax_price.scatter([last_timestamp], [last_close], color="#111111", s=28, zorder=5)

    for point in points:
        ax_price.axvline(point["date"], color="#d6dde7", linestyle="--", linewidth=0.8, zorder=0)
        ax_price.annotate(
            f"{point['horizon']}d\n{point['q50_return'] * 100:.2f}%",
            xy=(point["date"], point["q50_price"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#5a1f28",
        )

    ax_price.set_title(
        f"BTC Production Forecast: Recent Price with {horizon_label} Forecast Cone",
        fontsize=14,
    )
    ax_price.set_ylabel("BTC Price")
    ax_price.grid(alpha=0.20)
    ax_price.legend(loc="upper left", ncol=3, frameon=False)
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_price.xaxis.get_major_locator()))

    lower_tail_horizon = diagnostic_horizons[0]
    width_horizon = diagnostic_horizons[-1]
    summary_lines = [
        f"Last close: {last_close:,.2f}",
        f"Forecast date: {last_timestamp.date()}",
        f"Calibrator: {diagnostics['selected_calibrator']}",
        f"{lower_tail_horizon}d q05 miss: {lower_tail_df.loc[lower_tail_horizon, 'q05_miss_rate']:.3f}",
        f"{width_horizon}d width90: {width_df.loc[width_horizon, 'interval_width_90']:.3f}",
    ]
    if high_vol_fires or tail_flare_fires:
        summary_lines.append(
            "Tail/high-vol overlay: "
            + ", ".join(
                part
                for part in [
                    f"high-vol {latest_high_vol:.2f}" if high_vol_fires else "",
                    f"tail flare {latest_tail_flare:.2f}" if tail_flare_fires else "",
                ]
                if part
            )
        )
    summary_text = "\n".join(summary_lines)
    ax_price.text(
        0.985,
        0.03,
        summary_text,
        transform=ax_price.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#c7d1dd"},
    )

    horizons = diagnostic_horizons
    x = list(range(len(horizons)))
    width_values = width_df["interval_width_90"].to_list()
    lower_tail_values = lower_tail_df["q05_miss_rate"].to_list()

    bars = ax_diag.bar(x, width_values, width=0.55, color="#8fb9dd", alpha=0.85, label="90% interval width")
    ax_diag.set_xticks(x, [f"{h}d" for h in horizons])
    ax_diag.set_ylabel("Width / Miss Rate")
    ax_diag.grid(axis="y", alpha=0.20)
    ax_diag.set_title("Diagnostics: Width Stability and Lower-Tail Miss Rate", fontsize=11)
    ax_diag.plot(x, lower_tail_values, color="#b23a48", marker="o", linewidth=2.0, label="q05 miss rate")
    ax_diag.axhline(0.05, color="#5b6777", linestyle="--", linewidth=1.0, label="Target q05 miss")

    for bar, width_value in zip(bars, width_values):
        ax_diag.annotate(
            f"{width_value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, width_value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for xpos, miss_rate in zip(x, lower_tail_values):
        ax_diag.annotate(
            f"{miss_rate:.3f}",
            xy=(xpos, miss_rate),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#5a1f28",
        )

    ax_diag.legend(loc="upper left", ncol=3, frameon=False)

    plot_path = (save_dir / "latest_plot.png").resolve()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def fit_production_fusion(
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    feature_train: pd.DataFrame,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    base_config = copy.deepcopy(runtime["base_config"])
    calibration_cfg = runtime["calibration_cfg"]
    base_config["fusion"]["calibration_methods"] = _unique_methods(
        [calibration_cfg["default"]]
        + list(calibration_cfg.get("fallbacks", []))
        + list(calibration_cfg.get("allowed", []))
    )
    target_train = _training_targets(runtime["data_bundle"], candidate, feature_train)
    common_train_index = feature_train.index.intersection(target_train.index).sort_values()
    fitted = fit_candidate_model(
        candidate=candidate,
        feature_train=feature_train.loc[common_train_index],
        target_train=target_train.loc[common_train_index],
        config=base_config,
        device=runtime["device"],
    )
    calibration_report = _calibration_deployment_report(calibration_cfg, fitted)
    deployed_method = calibration_report["deployed_method"]
    fitted["calibrator"] = fitted["calibration_calibrators"][deployed_method]
    fitted["calibration_method"] = deployed_method
    fitted["selection_metrics"] = fitted["calibration_selection"][deployed_method]
    fitted["calibration_report"] = calibration_report
    if save_artifacts:
        fusion_root = ensure_dir(Path(runtime["artifact_root"]) / "fusion")
        torch.save(fitted["model"].state_dict(), fusion_root / "checkpoint.pt")
        save_doc(fusion_root / "scaler.json", fitted["scaler"].to_dict())
        save_doc(fusion_root / "target_scaler.json", fitted["target_scaler"].to_dict())
        save_doc(fusion_root / "history.json", fitted["history"])
        save_doc(fusion_root / "model_val_metrics.json", fitted["model_val_metrics"])
        save_doc(fusion_root / "selection_metrics.json", fitted["selection_metrics"])
        save_doc(fusion_root / "calibration_selection.json", fitted["calibration_selection"])
        save_doc(fusion_root / "selected_calibrator.json", fitted["calibrator"])
        calibration_root = ensure_dir(Path(runtime["artifact_root"]) / "calibration")
        save_doc(calibration_root / "comparison.json", calibration_report)
        save_doc(
            fusion_root / "production_manifest.json",
            {
                "candidate": candidate,
                "horizons": fitted["horizons"],
                "quantiles": fitted["quantiles"],
                "output_style": fitted["output_style"],
                "state_window": fitted["state_window"],
                "calibration_method": fitted["calibration_method"],
            },
        )
    return fitted


def compare_calibrators(
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    feature_train: pd.DataFrame,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    base_config = copy.deepcopy(runtime["base_config"])
    calibration_cfg = runtime["calibration_cfg"]
    base_config["fusion"]["calibration_methods"] = list(calibration_cfg["allowed"])
    target_train = _training_targets(runtime["data_bundle"], candidate, feature_train)
    common_train_index = feature_train.index.intersection(target_train.index).sort_values()
    fitted = fit_candidate_model(
        candidate=candidate,
        feature_train=feature_train.loc[common_train_index],
        target_train=target_train.loc[common_train_index],
        config=base_config,
        device=runtime["device"],
    )
    payload = _calibration_deployment_report(calibration_cfg, fitted)
    if save_artifacts:
        calibration_root = ensure_dir(Path(runtime["artifact_root"]) / "calibration")
        save_doc(calibration_root / "comparison.json", payload)
    return payload


def save_live_forecast(
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    fitted: dict[str, Any],
    feature_live: pd.DataFrame,
    diagnostics: dict[str, Any],
    save_artifacts: bool = True,
) -> dict[str, Any]:
    predictions = predict_live_candidate_model(
        candidate=candidate,
        fitted=fitted,
        feature_live=feature_live,
        device=runtime["device"],
    )
    payload = {
        str(horizon): frame.reset_index().to_dict(orient="records")
        for horizon, frame in predictions.items()
    }
    plot_path = None
    if save_artifacts:
        live_root = ensure_dir(Path(runtime["artifact_root"]) / "live_forecast")
        save_doc(live_root / "latest.json", payload)
        plot_path = _save_live_forecast_plot(runtime, payload, diagnostics, live_root)
        lines = ["# Live Forecast", ""]
        for horizon, rows in payload.items():
            row = rows[0]
            lines.append(
                f"- {horizon}d: q05={row['q05']:.6f}, q25={row['q25']:.6f}, q50={row['q50']:.6f}, "
                f"q75={row['q75']:.6f}, q95={row['q95']:.6f}"
            )
        if plot_path:
            lines.extend(["", f"- Plot: {plot_path}"])
        (live_root / "latest.md").write_text("\n".join(lines) + "\n")
    return {"forecast": payload, "plot_path": plot_path}


def run_refit_full_history(
    config_dir: str | Path = FINAL_CONFIG_DIR,
    as_of_timestamp: Any | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    runtime = load_runtime(config_dir=config_dir, as_of_timestamp=as_of_timestamp)
    specialist_stage = materialize_specialists(runtime, save_artifacts=save_artifacts)
    router_stage = materialize_role_router(
        runtime,
        train_states_by_bucket=specialist_stage["train_states_by_bucket"],
        live_history_by_bucket=specialist_stage["live_history_by_bucket"],
        save_artifacts=save_artifacts,
    )
    fitted = fit_production_fusion(
        runtime,
        candidate=router_stage["candidate"],
        feature_train=router_stage["feature_train"],
        save_artifacts=save_artifacts,
    )
    calibration_report = fitted["calibration_report"]
    diagnostics = _deployed_calibration_metrics(calibration_report)
    live_forecast = save_live_forecast(
        runtime,
        candidate=router_stage["candidate"],
        fitted=fitted,
        feature_live=router_stage["feature_live"],
        diagnostics=diagnostics,
        save_artifacts=save_artifacts,
    )
    forecast_timestamp = None
    if not runtime["data_bundle"].close.empty:
        forecast_timestamp = pd.Timestamp(runtime["data_bundle"].close.index[-1]).isoformat()
    summary = {
        "refit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_timestamp": forecast_timestamp,
        "candidate": router_stage["candidate"],
        "specialist_selection": specialist_stage["specialist_selection"],
        "fusion_calibration_method": fitted["calibration_method"],
        "calibration_report": calibration_report,
        "selected_calibrator": diagnostics["selected_calibrator"],
        "calibration_score_comparison": diagnostics["calibration_score_comparison"],
        "width_diagnostics": diagnostics["width_diagnostics"],
        "lower_tail_diagnostics": diagnostics["lower_tail_diagnostics"],
        "upper_tail_diagnostics": diagnostics["upper_tail_diagnostics"],
        "live_forecast": live_forecast["forecast"],
        "live_forecast_plot_path": live_forecast["plot_path"],
        "artifact_root": str(runtime["artifact_root"]),
    }
    if save_artifacts:
        save_doc(Path(runtime["artifact_root"]) / "refit_summary.json", summary)
    return summary
