from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import traceback
from typing import Any

import numpy as np
import pandas as pd

from database_interraction import (
    duckdb_connection,
    load_table_dataframe,
    quote_identifier,
    replace_table_from_dataframe,
    upsert_dataframe,
)
from dual_model_forecaster.brutal_baselines import (
    EvaluationConfig,
    build_brutal_baseline_evaluation,
    build_kalman_projection_overlay,
    build_kalman_state_history,
)
from dual_model_forecaster.config import load_config
from dual_model_forecaster.data import load_forecast_data
from dual_model_forecaster.jepa.config import JEPAConfig, load_jepa_config
from dual_model_forecaster.jepa.encode import encode_panel_history
from dual_model_forecaster.jepa.feature_contract import SPECIALISTS
from dual_model_forecaster.jepa.meta_synthesis import (
    JEPA_META_CANDIDATES,
    JEPA_META_METRICS,
    JEPAMetaParams,
    apply_jepa_meta_candidate,
    load_jepa_meta_params,
)
from dual_model_forecaster.jepa.panels import build_all_specialist_panels
from main import run_daily_production_cycle
from model_assembly.common import FINAL_CONFIG_DIR, load_doc


DEFAULT_DB_PATH = "database/walk_forward_predictions.duckdb"
DEFAULT_TABLE_NAME = "walk_forward_predictions"
DEFAULT_CHART_DIR = "reports/walk_forward"
ROLLING_ZSCORE_WINDOW = 90
ROLLING_ZSCORE_MIN_PERIODS = 20
MAX_ABS_PROJECTED_RETURN = 2.0
INNER_ASYMMETRY_ZSCORE_WINDOW = 180
INNER_ASYMMETRY_ZSCORE_MIN_PERIODS = 45
INNER_ASYMMETRY_COMPONENT_CLIP = 5.0
INNER_ASYMMETRY_SIGNAL_SPAN = 30
EDGE_FORWARD_WINDOWS = (30, 60)
QUANTILE_COLUMNS = ["q05", "q25", "q50", "q75", "q95"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
META_SYNTH_PREFIX = "meta_synth"
META_SELECTED_FORECASTER = f"{META_SYNTH_PREFIX}_selected"
JEPA_FEATURE_TABLE = "walk_forward_jepa_features"
JEPA_FEATURE_WIDE_TABLE = "walk_forward_jepa_features_wide"
META_DEFAULT_MIN_HISTORY = 120
META_DEFAULT_SELECTION_WINDOW = 365
META_SELECTION_MIN_WIS_IMPROVEMENT = 0.01
META_CANDIDATE_SPECS = [
    {
        "name": "model_passthrough",
        "center_weight": 0.0,
        "clip_widths": 0.0,
        "width_disagreement_weight": 0.0,
        "width_high_vol_weight": 0.0,
        "tail_flare_weight": 0.0,
        "edge_weight": 0.0,
        "high_vol_dampen": 0.0,
        "disagreement_dampen": 0.0,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "light_anchor",
        "center_weight": 0.12,
        "clip_widths": 0.65,
        "width_disagreement_weight": 0.04,
        "width_high_vol_weight": 0.06,
        "tail_flare_weight": 0.04,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.20,
        "disagreement_dampen": 0.35,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "balanced_anchor",
        "center_weight": 0.24,
        "clip_widths": 0.80,
        "width_disagreement_weight": 0.08,
        "width_high_vol_weight": 0.10,
        "tail_flare_weight": 0.08,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.30,
        "disagreement_dampen": 0.45,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "guarded_anchor",
        "center_weight": 0.34,
        "clip_widths": 0.95,
        "width_disagreement_weight": 0.12,
        "width_high_vol_weight": 0.16,
        "tail_flare_weight": 0.12,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.45,
        "disagreement_dampen": 0.60,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "edge_guarded_anchor",
        "center_weight": 0.26,
        "clip_widths": 0.85,
        "width_disagreement_weight": 0.10,
        "width_high_vol_weight": 0.12,
        "tail_flare_weight": 0.10,
        "edge_weight": 0.025,
        "high_vol_dampen": 0.35,
        "disagreement_dampen": 0.50,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "long_anchor",
        "center_weight": 0.42,
        "clip_widths": 1.10,
        "width_disagreement_weight": 0.14,
        "width_high_vol_weight": 0.18,
        "tail_flare_weight": 0.16,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.40,
        "disagreement_dampen": 0.55,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "long_edge_anchor",
        "center_weight": 0.36,
        "clip_widths": 1.00,
        "width_disagreement_weight": 0.12,
        "width_high_vol_weight": 0.16,
        "tail_flare_weight": 0.14,
        "edge_weight": 0.035,
        "high_vol_dampen": 0.38,
        "disagreement_dampen": 0.55,
        "width_mode": "scale",
        "sigma_cap_scale": 0.0,
    },
    {
        "name": "sigma_cap_light",
        "center_weight": 0.12,
        "clip_widths": 0.70,
        "width_disagreement_weight": 0.04,
        "width_high_vol_weight": 0.10,
        "tail_flare_weight": 0.08,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.25,
        "disagreement_dampen": 0.35,
        "width_mode": "cap",
        "sigma_cap_scale": 1.35,
    },
    {
        "name": "sigma_cap_balanced",
        "center_weight": 0.24,
        "clip_widths": 0.85,
        "width_disagreement_weight": 0.08,
        "width_high_vol_weight": 0.14,
        "tail_flare_weight": 0.10,
        "edge_weight": 0.00,
        "high_vol_dampen": 0.35,
        "disagreement_dampen": 0.45,
        "width_mode": "cap",
        "sigma_cap_scale": 1.65,
    },
    {
        "name": "sigma_cap_edge",
        "center_weight": 0.20,
        "clip_widths": 0.80,
        "width_disagreement_weight": 0.08,
        "width_high_vol_weight": 0.14,
        "tail_flare_weight": 0.10,
        "edge_weight": 0.025,
        "high_vol_dampen": 0.32,
        "disagreement_dampen": 0.45,
        "width_mode": "cap",
        "sigma_cap_scale": 1.65,
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _format_horizon_label(horizons: list[int]) -> str:
    return " / ".join(f"{int(horizon)}d" for horizon in sorted(horizons))


def _load_refit_base_config(config_dir: str | Path) -> dict[str, Any]:
    config_dir = Path(config_dir)
    refit_cfg = load_doc(config_dir / "refit.yaml")
    return load_config(refit_cfg["base_config"])


def _load_production_horizons(config_dir: str | Path) -> list[int]:
    fusion_cfg = load_doc(Path(config_dir) / "fusion.yaml")
    return [int(horizon) for horizon in fusion_cfg["horizons"]]


def _live_forecast_dir(config_dir: str | Path) -> Path:
    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    return Path(refit_cfg["artifact_root"]) / "live_forecast"


def _materialize_as_of_config(
    *,
    source_config_dir: str | Path,
    base_config: dict[str, Any],
    work_dir: Path,
    category_db_path: Path,
    artifact_root: Path,
) -> Path:
    source_config_dir = Path(source_config_dir)
    config_dir = work_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    for name in ("specialists.yaml", "roles.yaml", "fusion.yaml", "calibration.yaml"):
        shutil.copyfile(source_config_dir / name, config_dir / name)

    as_of_base_config = json.loads(json.dumps(base_config, default=str))
    as_of_base_config["paths"]["category_db_path"] = str(category_db_path)
    base_config_path = work_dir / "base_config.json"
    base_config_path.write_text(json.dumps(as_of_base_config, indent=2) + "\n")
    (config_dir / "refit.yaml").write_text(
        json.dumps(
            {
                "base_config": str(base_config_path),
                "artifact_root": str(artifact_root),
            },
            indent=2,
        )
        + "\n"
    )
    return config_dir


def _copy_cached_category_tables_for_as_of(
    *,
    source_category_db_path: str | Path,
    target_category_db_path: str | Path,
    bucket_tables: dict[str, str],
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    cutoff = _timestamp(as_of)
    copied: dict[str, dict[str, Any]] = {}
    for bucket_name, table_name in bucket_tables.items():
        frame = load_table_dataframe(
            str(source_category_db_path),
            str(table_name),
            read_only=True,
            order_by=["timestamp"],
        )
        if "timestamp" not in frame.columns:
            raise ValueError(f"Cached feature table {table_name} has no timestamp column.")
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frame = frame.loc[frame["timestamp"] <= cutoff].sort_values("timestamp")
        if frame.empty:
            raise ValueError(f"Cached feature table {table_name} has no rows on or before {cutoff}.")
        with duckdb_connection(str(target_category_db_path)) as connection:
            replace_table_from_dataframe(connection, str(table_name), frame)
        copied[bucket_name] = {
            "table": str(table_name),
            "rows": int(len(frame)),
            "start": str(pd.Timestamp(frame["timestamp"].min())),
            "end": str(pd.Timestamp(frame["timestamp"].max())),
        }
    return {
        "mode": "cached_category_table_slice",
        "source_category_db_path": str(source_category_db_path),
        "target_category_db_path": str(target_category_db_path),
        "as_of": str(cutoff),
        "tables": copied,
    }


def _default_required_rows(base_config: dict[str, Any], horizons: list[int]) -> int:
    split_cfg = base_config["splits"]
    state_window = int(base_config["fusion"]["state_window"])
    max_horizon = max(horizons)
    # The production refit does not use the outer-fold min_train_days value.
    # It needs enough rows to build inner specialist states, drop tail targets,
    # and still leave at least one fusion state window for the live head.
    return (
        int(split_cfg.get("inner_min_train_days", 0))
        + state_window
        + max_horizon
        + 10
    )


def _discover_as_of_dates(
    config_dir: str | Path,
    start: str | None,
    end: str | None,
    limit: int | None,
    min_history_rows: int | None,
) -> tuple[list[pd.Timestamp], dict[str, Any], Any, list[int], int]:
    base_config = _load_refit_base_config(config_dir)
    full_data = load_forecast_data(base_config)
    horizons = _load_production_horizons(config_dir)
    required_rows = int(min_history_rows or _default_required_rows(base_config, horizons))
    if required_rows < 1:
        required_rows = 1
    if len(full_data.close.index) < required_rows:
        raise ValueError(
            f"Only {len(full_data.close.index)} aligned rows are available, "
            f"but {required_rows} rows are required."
        )

    dates = list(pd.DatetimeIndex(full_data.close.index[required_rows - 1 :]))
    if start:
        start_ts = _timestamp(start)
        dates = [date for date in dates if date >= start_ts]
    if end:
        end_ts = _timestamp(end)
        dates = [date for date in dates if date <= end_ts]
    if limit is not None:
        dates = dates[: max(int(limit), 0)]
    return dates, base_config, full_data, horizons, required_rows


def _ensure_predictions_table(connection: Any, table_name: str) -> None:
    safe_table = quote_identifier(table_name)
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
            as_of TIMESTAMP NOT NULL,
            horizon INTEGER NOT NULL,
            target_timestamp TIMESTAMP,
            base_close DOUBLE,
            actual_close DOUBLE,
            actual_return DOUBLE,
            q05 DOUBLE,
            q25 DOUBLE,
            q50 DOUBLE,
            q75 DOUBLE,
            q95 DOUBLE,
            selected_calibrator TEXT,
            fusion_calibration_method TEXT,
            candidate_name TEXT,
            refit_timestamp_utc TEXT,
            as_of_row_count BIGINT,
            config_dir TEXT,
            created_at_utc TEXT,
            width_diagnostics_json TEXT,
            lower_tail_diagnostics_json TEXT,
            upper_tail_diagnostics_json TEXT,
            calibration_score_comparison_json TEXT,
            PRIMARY KEY(as_of, horizon)
        );
        """
    )


def _ensure_failure_table(connection: Any, table_name: str) -> str:
    failure_table = f"{table_name}_failures"
    safe_table = quote_identifier(failure_table)
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
            as_of TIMESTAMP NOT NULL,
            started_at_utc TEXT,
            completed_at_utc TEXT,
            config_dir TEXT,
            error TEXT,
            traceback TEXT
        );
        """
    )
    return failure_table


def _record_failure(
    db_path: str,
    table_name: str,
    as_of: pd.Timestamp,
    started_at_utc: str,
    config_dir: str,
    error: BaseException,
) -> None:
    with duckdb_connection(db_path) as connection:
        failure_table = _ensure_failure_table(connection, table_name)
        connection.execute(
            f"INSERT INTO {quote_identifier(failure_table)} VALUES (?, ?, ?, ?, ?, ?);",
            [
                as_of.to_pydatetime(),
                started_at_utc,
                _utc_now(),
                config_dir,
                str(error),
                traceback.format_exc(),
            ],
        )


def _completed_as_ofs(
    db_path: str,
    table_name: str,
    horizons: list[int],
) -> set[pd.Timestamp]:
    with duckdb_connection(db_path) as connection:
        _ensure_predictions_table(connection, table_name)
        rows = connection.execute(
            f"""
            SELECT as_of, COUNT(DISTINCT horizon) AS horizon_count
            FROM {quote_identifier(table_name)}
            GROUP BY as_of
            HAVING COUNT(DISTINCT horizon) >= ?;
            """,
            [len(horizons)],
        ).fetchall()
    return {pd.Timestamp(row[0]) for row in rows}


def _actual_return(
    close: pd.Series,
    as_of: pd.Timestamp,
    horizon: int,
    return_type: str,
) -> tuple[pd.Timestamp, float | None, float | None]:
    target_timestamp = as_of + pd.Timedelta(days=horizon)
    if as_of not in close.index or target_timestamp not in close.index:
        return target_timestamp, None, None
    base_close = float(close.loc[as_of])
    actual_close = float(close.loc[target_timestamp])
    if return_type == "log":
        actual = math.log(actual_close) - math.log(base_close)
    else:
        actual = actual_close / base_close - 1.0
    return target_timestamp, actual_close, float(actual)


def _rows_from_cycle_summary(
    cycle_summary: dict[str, Any],
    base_config: dict[str, Any],
    full_close: pd.Series,
    config_dir: str,
) -> pd.DataFrame:
    production = cycle_summary["production"]
    live_forecast = production["live_forecast"]
    return_type = str(base_config["data"]["return_type"])
    created_at = _utc_now()
    rows: list[dict[str, Any]] = []

    for horizon_key, forecast_rows in sorted(live_forecast.items(), key=lambda item: int(item[0])):
        horizon = int(horizon_key)
        forecast_row = forecast_rows[0]
        as_of = _timestamp(forecast_row["timestamp"])
        base_close = float(full_close.loc[as_of]) if as_of in full_close.index else None
        target_timestamp, actual_close, actual = _actual_return(
            close=full_close,
            as_of=as_of,
            horizon=horizon,
            return_type=return_type,
        )
        rows.append(
            {
                "as_of": as_of.to_pydatetime(),
                "horizon": horizon,
                "target_timestamp": target_timestamp.to_pydatetime(),
                "base_close": base_close,
                "actual_close": actual_close,
                "actual_return": actual,
                "q05": float(forecast_row["q05"]),
                "q25": float(forecast_row["q25"]),
                "q50": float(forecast_row["q50"]),
                "q75": float(forecast_row["q75"]),
                "q95": float(forecast_row["q95"]),
                "selected_calibrator": production["selected_calibrator"],
                "fusion_calibration_method": production["fusion_calibration_method"],
                "candidate_name": production["candidate"],
                "refit_timestamp_utc": production["refit_timestamp_utc"],
                "as_of_row_count": int(cycle_summary["as_of_row_count"]),
                "config_dir": config_dir,
                "created_at_utc": created_at,
                "width_diagnostics_json": json.dumps(production["width_diagnostics"], default=str),
                "lower_tail_diagnostics_json": json.dumps(production["lower_tail_diagnostics"], default=str),
                "upper_tail_diagnostics_json": json.dumps(production["upper_tail_diagnostics"], default=str),
                "calibration_score_comparison_json": json.dumps(
                    production["calibration_score_comparison"],
                    default=str,
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_main_cycle_for_as_of(
    config_dir: str,
    as_of: pd.Timestamp,
    save_artifacts: bool,
    rebuild_feature_submodels: bool,
    reuse_cached_feature_submodels: bool,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    if rebuild_feature_submodels:
        with tempfile.TemporaryDirectory(prefix="wf_asof_features_") as tmp:
            work_dir = Path(tmp)
            category_db_path = work_dir / "data_classes.duckdb"
            ta_db_path = work_dir / "ta_1d.duckdb"
            artifact_root = (
                Path(load_doc(Path(config_dir) / "refit.yaml")["artifact_root"])
                if save_artifacts
                else work_dir / "artifacts"
            )
            as_of_config_dir = _materialize_as_of_config(
                source_config_dir=config_dir,
                base_config=base_config,
                work_dir=work_dir,
                category_db_path=category_db_path,
                artifact_root=artifact_root,
            )
            cached_feature_summary = None
            if reuse_cached_feature_submodels:
                cached_feature_summary = _copy_cached_category_tables_for_as_of(
                    source_category_db_path=base_config["paths"]["category_db_path"],
                    target_category_db_path=category_db_path,
                    bucket_tables=base_config["data"]["bucket_tables"],
                    as_of=as_of,
                )
            summary = run_daily_production_cycle(
                config_dir=str(as_of_config_dir),
                refresh_data=False,
                rebuild_categories=not reuse_cached_feature_submodels,
                run_production_forecast=True,
                as_of_timestamp=as_of.isoformat(),
                save_artifacts=save_artifacts,
                price_db_path=str(base_config["paths"]["price_db_path"]),
                category_db_path=str(category_db_path),
                ta_output_db_path=str(ta_db_path),
            )
            if cached_feature_summary is not None:
                summary["cached_feature_submodels"] = cached_feature_summary
            summary["as_of_row_count"] = None
            return summary

    summary = run_daily_production_cycle(
        config_dir=config_dir,
        refresh_data=False,
        rebuild_categories=False,
        run_production_forecast=True,
        as_of_timestamp=as_of.isoformat(),
        save_artifacts=save_artifacts,
    )
    summary["as_of_row_count"] = None
    return summary


def _save_prediction_rows(
    db_path: str,
    table_name: str,
    rows: pd.DataFrame,
) -> int:
    if rows.empty:
        return 0
    with duckdb_connection(db_path) as connection:
        _ensure_predictions_table(connection, table_name)
        return upsert_dataframe(
            connection=connection,
            table_name=table_name,
            dataframe=rows,
            key_columns=["as_of", "horizon"],
        )


def _load_predictions(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as connection:
        return connection.execute(
            f"SELECT * FROM {quote_identifier(table_name)} ORDER BY as_of, horizon;"
        ).df()


def _project_price(base_close: pd.Series, forecast_return: pd.Series, return_type: str) -> pd.Series:
    base = base_close.astype(float)
    returns = forecast_return.astype(float)
    sane = np.isfinite(base) & np.isfinite(returns) & returns.abs().le(MAX_ABS_PROJECTED_RETURN)
    projected = pd.Series(np.nan, index=forecast_return.index, dtype=float)
    if return_type == "log":
        projected.loc[sane] = base.loc[sane] * np.exp(returns.loc[sane])
    else:
        simple_sane = sane & returns.gt(-0.99)
        projected.loc[simple_sane] = base.loc[simple_sane] * (1.0 + returns.loc[simple_sane])
    return projected


def _rolling_zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(
        ROLLING_ZSCORE_WINDOW,
        min_periods=ROLLING_ZSCORE_MIN_PERIODS,
    ).mean()
    std = series.rolling(
        ROLLING_ZSCORE_WINDOW,
        min_periods=ROLLING_ZSCORE_MIN_PERIODS,
    ).std()
    return ((series - mean) / std.replace(0.0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA)


def _rolling_zscore_with_window(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0.0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA)


def _plot_value_envelope_chart(
    frame: pd.DataFrame,
    horizon: int,
    chart_dir: Path,
    return_type: str,
    kalman_overlay: pd.DataFrame | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    horizon_frame = frame.loc[frame["horizon"] == horizon].copy()
    horizon_frame = horizon_frame.sort_values("target_timestamp")
    horizon_frame["as_of"] = pd.to_datetime(horizon_frame["as_of"])
    horizon_frame["target_timestamp"] = pd.to_datetime(horizon_frame["target_timestamp"])
    horizon_frame = horizon_frame.dropna(subset=["base_close", "target_timestamp"])
    if horizon_frame.empty:
        raise ValueError(f"No prediction rows with base prices for {horizon}d.")

    for column in ("q05", "q25", "q50", "q75", "q95"):
        horizon_frame[f"{column}_price"] = _project_price(
            horizon_frame["base_close"],
            horizon_frame[column],
            return_type,
        )
    realized = horizon_frame.dropna(subset=["actual_return"])
    if not realized.empty:
        residual = realized["actual_return"].astype(float) - realized["q50"].astype(float)
        horizon_frame.loc[realized.index, "median_error_z"] = _rolling_zscore(residual)
    kalman_horizon = pd.DataFrame()
    if kalman_overlay is not None and not kalman_overlay.empty:
        kalman_horizon = kalman_overlay.loc[kalman_overlay["horizon"].astype(int) == int(horizon)].copy()
        kalman_horizon["as_of"] = pd.to_datetime(kalman_horizon["as_of"], errors="coerce")
        kalman_horizon["target_timestamp"] = pd.to_datetime(kalman_horizon["target_timestamp"], errors="coerce")
        if "kalman_target_timestamp" in kalman_horizon.columns:
            kalman_horizon["kalman_target_timestamp"] = pd.to_datetime(
                kalman_horizon["kalman_target_timestamp"],
                errors="coerce",
            )
        kalman_horizon = kalman_horizon.sort_values("target_timestamp")

    fig, (ax_price, ax_z) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    x = horizon_frame["target_timestamp"].dt.to_pydatetime()
    ax_price.fill_between(
        x,
        horizon_frame["q05_price"].to_numpy(dtype=float),
        horizon_frame["q95_price"].to_numpy(dtype=float),
        color="#7aa6d1",
        alpha=0.20,
        label="model q05-q95",
    )
    ax_price.fill_between(
        x,
        horizon_frame["q25_price"].to_numpy(dtype=float),
        horizon_frame["q75_price"].to_numpy(dtype=float),
        color="#2f6da3",
        alpha=0.30,
        label="model q25-q75",
    )
    ax_price.plot(
        x,
        horizon_frame["q50_price"].to_numpy(dtype=float),
        color="#9b2f3d",
        linewidth=1.8,
        label="model q50 target price",
    )
    if not kalman_horizon.empty:
        fair_value = kalman_horizon.dropna(subset=["kalman_fair_value_price", "as_of"])
        projected_time_column = (
            "kalman_target_timestamp"
            if "kalman_target_timestamp" in kalman_horizon.columns
            and kalman_horizon["kalman_target_timestamp"].notna().any()
            else "target_timestamp"
        )
        projected = kalman_horizon.dropna(subset=["kalman_projected_price", projected_time_column])
        projected = projected.drop_duplicates(subset=["as_of", projected_time_column], keep="last")
        if not fair_value.empty:
            ax_price.plot(
                fair_value["as_of"].dt.to_pydatetime(),
                fair_value["kalman_fair_value_price"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.35,
                alpha=0.88,
                label="Kalman fair value",
            )
        if not projected.empty:
            ax_price.plot(
                projected[projected_time_column].dt.to_pydatetime(),
                projected["kalman_projected_price"].to_numpy(dtype=float),
                color="#16a34a",
                linestyle="--",
                linewidth=1.55,
                alpha=0.90,
                label="Kalman projected scope",
            )
    if not realized.empty:
        realized_x = realized["target_timestamp"].dt.to_pydatetime()
        ax_price.plot(
            realized_x,
            realized["actual_close"].to_numpy(dtype=float),
            color="#222222",
            linewidth=1.35,
            alpha=0.88,
            label="BTC close at target",
        )

    ax_price.set_title(f"{horizon}d Walk-Forward BTC Value Envelope")
    ax_price.set_ylabel("BTC Price")
    ax_price.grid(alpha=0.20)
    ax_price.legend(loc="upper left", ncol=4, frameon=False)
    if realized["actual_close"].dropna().gt(0).all() and horizon_frame["q05_price"].dropna().gt(0).all():
        ax_price.set_yscale("log")

    if not realized.empty:
        coverage_90 = (
            (realized["actual_close"] >= realized["q05_price"])
            & (realized["actual_close"] <= realized["q95_price"])
        ).mean()
        coverage_50 = (
            (realized["actual_close"] >= realized["q25_price"])
            & (realized["actual_close"] <= realized["q75_price"])
        ).mean()
        median_abs_error = (realized["actual_close"] - realized["q50_price"]).abs().mean()
        summary_text = "\n".join(
            [
                f"Forecasts: {len(horizon_frame):,}",
                f"Realized: {len(realized):,}",
                f"q05-q95 coverage: {coverage_90:.3f}",
                f"q25-q75 coverage: {coverage_50:.3f}",
                f"median abs error: ${median_abs_error:,.0f}",
            ]
        )
        ax_price.text(
            0.985,
            0.03,
            summary_text,
            transform=ax_price.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#c7d1dd",
            },
        )

    z_frame = horizon_frame.dropna(subset=["median_error_z"])
    if not z_frame.empty:
        z_x = z_frame["target_timestamp"].dt.to_pydatetime()
        ax_z.fill_between(z_x, 0.0, z_frame["median_error_z"].to_numpy(dtype=float), color="#8c96a3", alpha=0.24)
        ax_z.plot(
            z_x,
            z_frame["median_error_z"].to_numpy(dtype=float),
            color="#334155",
            linewidth=1.25,
            label="model q50 error z",
        )
    ax_z_high_vol = None
    if not kalman_horizon.empty and "kalman_high_vol_signal" in kalman_horizon.columns:
        high_vol = kalman_horizon.dropna(subset=["kalman_high_vol_signal", "target_timestamp"])
        if not high_vol.empty:
            ax_z_high_vol = ax_z.twinx()
            ax_z_high_vol.plot(
                high_vol["target_timestamp"].dt.to_pydatetime(),
                high_vol["kalman_high_vol_signal"].clip(0.0, 1.0).to_numpy(dtype=float),
                color="#d97706",
                linewidth=1.05,
                alpha=0.85,
                label="Kalman high-vol signal",
            )
            ax_z_high_vol.set_ylim(-0.05, 1.05)
            ax_z_high_vol.set_ylabel("High-vol")
            ax_z_high_vol.tick_params(axis="y", colors="#9a5b00")
    ax_z.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.7)
    ax_z.axhline(2.0, color="#9b2f3d", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_z.axhline(-2.0, color="#2f6da3", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_z.set_ylabel("Error z")
    ax_z.set_title(
        f"q50 vs realized {horizon}d return z-score "
        f"({ROLLING_ZSCORE_WINDOW}-forecast rolling window)"
    )
    ax_z.grid(alpha=0.20)
    if ax_z_high_vol is not None:
        lines, labels = ax_z.get_legend_handles_labels()
        high_vol_lines, high_vol_labels = ax_z_high_vol.get_legend_handles_labels()
        ax_z.legend(lines + high_vol_lines, labels + high_vol_labels, loc="upper left", frameon=False)
    ax_z.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_z.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_z.xaxis.get_major_locator()))

    chart_dir.mkdir(parents=True, exist_ok=True)
    path = chart_dir / f"walk_forward_value_envelope_{horizon}d.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def build_envelope_charts(
    db_path: str,
    table_name: str,
    chart_dir: str | Path,
    horizons: list[int],
    return_type: str,
    data_bundle: Any | None = None,
) -> list[str]:
    predictions = _load_predictions(db_path, table_name)
    if predictions.empty:
        return []
    kalman_overlay = pd.DataFrame()
    if data_bundle is not None:
        kalman_overlay = build_kalman_projection_overlay(
            predictions=predictions,
            data_bundle=data_bundle,
            horizons=horizons,
            return_type=return_type,
        )
    out: list[str] = []
    for horizon in horizons:
        if int(horizon) not in set(predictions["horizon"].astype(int)):
            continue
        out.append(
            _plot_value_envelope_chart(
                predictions,
                int(horizon),
                Path(chart_dir),
                return_type=return_type,
                kalman_overlay=kalman_overlay,
            )
        )
    return out


def _build_inner_asymmetry_indicator_frame(
    predictions: pd.DataFrame,
    horizons: list[int],
    data_bundle: Any | None = None,
    return_type: str | None = None,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    close = (
        frame.loc[frame["horizon"] == min(horizons)]
        .sort_values("as_of")
        .drop_duplicates(subset=["as_of"], keep="last")
        .set_index("as_of")["base_close"]
        .astype(float)
        .rename("close")
    )
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    kalman_overlay = pd.DataFrame()
    kalman_state = pd.DataFrame()
    if data_bundle is not None and return_type is not None:
        kalman_overlay = build_kalman_projection_overlay(
            predictions=frame,
            data_bundle=data_bundle,
            horizons=horizons,
            return_type=return_type,
        )
        kalman_state = build_kalman_state_history(data_bundle).reindex(out.index)
        if not kalman_state.empty:
            for column in [
                "kalman_fair_value_price",
                "gap",
                "gap_z",
                "innovation_z",
                "kalman_high_vol_signal",
                "tail_flare_score",
            ]:
                if column in kalman_state.columns:
                    out[column] = pd.to_numeric(kalman_state[column], errors="coerce")

    components = []
    kalman_direction_components = []
    kalman_disagreement_components = []
    for horizon in horizons:
        horizon_frame = (
            frame.loc[frame["horizon"] == int(horizon)]
            .sort_values("as_of")
            .drop_duplicates(subset=["as_of"], keep="last")
            .set_index("as_of")
            .reindex(out.index)
        )
        upside = (horizon_frame["q75"].astype(float) - horizon_frame["q50"].astype(float)).clip(lower=1e-6)
        downside = (horizon_frame["q50"].astype(float) - horizon_frame["q25"].astype(float)).clip(lower=1e-6)
        component = np.log(upside / downside).clip(
            lower=-INNER_ASYMMETRY_COMPONENT_CLIP,
            upper=INNER_ASYMMETRY_COMPONENT_CLIP,
        )
        component_name = f"inner_asymmetry_{horizon}d"
        out[component_name] = component
        components.append(component_name)
        if not kalman_overlay.empty:
            kalman_horizon = (
                kalman_overlay.loc[kalman_overlay["horizon"].astype(int) == int(horizon)]
                .sort_values("as_of")
                .drop_duplicates(subset=["as_of"], keep="last")
                .set_index("as_of")
                .reindex(out.index)
            )
            projected_return = pd.to_numeric(kalman_horizon["kalman_projected_return"], errors="coerce")
            out[f"kalman_projected_return_{horizon}d"] = projected_return
            scope_days = (
                pd.to_numeric(kalman_horizon["kalman_scope_days"], errors="coerce")
                if "kalman_scope_days" in kalman_horizon.columns
                else pd.Series(float(horizon), index=kalman_horizon.index)
            ).fillna(float(horizon)).clip(lower=1.0)
            direction_component = (projected_return / np.sqrt(scope_days)).clip(
                lower=-INNER_ASYMMETRY_COMPONENT_CLIP,
                upper=INNER_ASYMMETRY_COMPONENT_CLIP,
            )
            disagreement_scale = (horizon_frame["q75"].astype(float) - horizon_frame["q25"].astype(float)).abs().clip(lower=1e-6)
            disagreement_component = ((projected_return - horizon_frame["q50"].astype(float)) / disagreement_scale).clip(
                lower=-INNER_ASYMMETRY_COMPONENT_CLIP,
                upper=INNER_ASYMMETRY_COMPONENT_CLIP,
            )
            direction_name = f"kalman_direction_{horizon}d"
            disagreement_name = f"kalman_model_disagreement_{horizon}d"
            out[direction_name] = direction_component
            out[disagreement_name] = disagreement_component
            kalman_direction_components.append(direction_name)
            kalman_disagreement_components.append(disagreement_name)

    out["inner_asymmetry_raw"] = out[components].mean(axis=1)
    out["inner_asymmetry"] = out["inner_asymmetry_raw"].ewm(
        span=INNER_ASYMMETRY_SIGNAL_SPAN,
        adjust=False,
    ).mean()
    out["inner_asymmetry_z"] = _rolling_zscore_with_window(
        out["inner_asymmetry"],
        window=INNER_ASYMMETRY_ZSCORE_WINDOW,
        min_periods=INNER_ASYMMETRY_ZSCORE_MIN_PERIODS,
    )
    if kalman_direction_components:
        out["kalman_direction_raw"] = out[kalman_direction_components].mean(axis=1)
        out["kalman_direction"] = out["kalman_direction_raw"].ewm(
            span=INNER_ASYMMETRY_SIGNAL_SPAN,
            adjust=False,
        ).mean()
        out["kalman_direction_z"] = _rolling_zscore_with_window(
            out["kalman_direction"],
            window=INNER_ASYMMETRY_ZSCORE_WINDOW,
            min_periods=INNER_ASYMMETRY_ZSCORE_MIN_PERIODS,
        )
    if kalman_disagreement_components:
        out["kalman_model_disagreement_raw"] = out[kalman_disagreement_components].mean(axis=1)

    if "kalman_direction_z" in out.columns:
        high_vol = out.get("kalman_high_vol_signal", pd.Series(0.0, index=out.index)).fillna(0.0).clip(0.0, 1.0)
        combined = (
            out["inner_asymmetry_z"].fillna(0.0)
            + 0.15 * out["kalman_direction_z"].fillna(0.0)
        ) * (1.0 - 0.25 * high_vol)
        combined.loc[out["inner_asymmetry_z"].isna() & out["kalman_direction_z"].isna()] = np.nan
        out["kalman_adjusted_edge_z"] = combined
        out["kalman_adjusted_edge"] = combined
    for forward_window in EDGE_FORWARD_WINDOWS:
        out[f"forward_return_{forward_window}d"] = np.log(out["close"].shift(-forward_window) / out["close"])
    return out


def _inner_asymmetry_metrics(indicator: pd.DataFrame) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    signal_columns = [
        ("inner_asymmetry", "inner_asymmetry"),
        ("kalman_direction", "kalman_direction"),
        ("kalman_adjusted_edge", "kalman_adjusted_edge"),
    ]
    for forward_window in EDGE_FORWARD_WINDOWS:
        column = f"forward_return_{forward_window}d"
        for signal_name, signal_column in signal_columns:
            if signal_column not in indicator.columns:
                continue
            common = indicator[[signal_column, column]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(common) < 200:
                continue
            quantiles = pd.qcut(common[signal_column].rank(method="first"), 5, labels=False)
            bottom = common.loc[quantiles == 0, column]
            top = common.loc[quantiles == 4, column]
            metrics.append(
                {
                    "signal": signal_name,
                    "forward_window": forward_window,
                    "observations": int(len(common)),
                    "spearman": float(common[signal_column].corr(common[column], method="spearman")),
                    "top_quintile_avg_return": float(top.mean()),
                    "bottom_quintile_avg_return": float(bottom.mean()),
                    "top_minus_bottom": float(top.mean() - bottom.mean()),
                    "top_quintile_hit_rate": float((top > 0.0).mean()),
                    "bottom_quintile_hit_rate": float((bottom > 0.0).mean()),
                }
            )
    return metrics


def _plot_inner_asymmetry_indicator(
    indicator: pd.DataFrame,
    metrics: list[dict[str, Any]],
    chart_dir: Path,
    horizons: list[int],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plot_frame = indicator.dropna(subset=["close"]).copy()
    plot_frame = plot_frame.sort_index()
    primary_signal_column = "inner_asymmetry_z"
    z_frame = plot_frame.dropna(subset=[primary_signal_column])
    if plot_frame.empty:
        raise ValueError("No rows available for inner asymmetry indicator chart.")

    fig, (ax_price, ax_signal) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.2]},
        constrained_layout=True,
    )

    ax_price.plot(plot_frame.index, plot_frame["close"], color="#20252d", linewidth=1.45, label="BTC close")
    if "kalman_fair_value_price" in plot_frame.columns:
        fair_value = plot_frame.dropna(subset=["kalman_fair_value_price"])
        if not fair_value.empty:
            ax_price.plot(
                fair_value.index,
                fair_value["kalman_fair_value_price"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.15,
                alpha=0.85,
                label="Kalman fair value",
            )
    high_signal = z_frame.loc[z_frame[primary_signal_column] >= 1.0]
    low_signal = z_frame.loc[z_frame[primary_signal_column] <= -1.0]
    if not high_signal.empty:
        ax_price.scatter(
            high_signal.index,
            plot_frame.loc[high_signal.index, "close"],
            s=12,
            color="#0f8b4c",
            alpha=0.55,
            label="edge >= +1z",
        )
    if not low_signal.empty:
        ax_price.scatter(
            low_signal.index,
            plot_frame.loc[low_signal.index, "close"],
            s=12,
            color="#b23a48",
            alpha=0.45,
            label="edge <= -1z",
        )
    if plot_frame["close"].dropna().gt(0).all():
        ax_price.set_yscale("log")
    ax_price.set_title("BTC Inner-Asymmetry Edge Indicator")
    ax_price.set_ylabel("BTC Price")
    ax_price.grid(alpha=0.20)
    ax_price.legend(loc="upper left", ncol=3, frameon=False)

    if not z_frame.empty:
        x = z_frame.index.to_pydatetime()
        z = z_frame[primary_signal_column].to_numpy(dtype=float)
        ax_signal.fill_between(x, 0.0, z, where=z >= 0, color="#0f8b4c", alpha=0.22, interpolate=True)
        ax_signal.fill_between(x, 0.0, z, where=z < 0, color="#b23a48", alpha=0.20, interpolate=True)
        ax_signal.plot(x, z, color="#334155", linewidth=1.25, label="Kalman-adjusted edge z" if primary_signal_column == "kalman_adjusted_edge_z" else "inner asymmetry z")
    if "inner_asymmetry_z" in plot_frame.columns:
        inner_z = plot_frame.dropna(subset=["inner_asymmetry_z"])
        if not inner_z.empty and primary_signal_column != "inner_asymmetry_z":
            ax_signal.plot(
                inner_z.index.to_pydatetime(),
                inner_z["inner_asymmetry_z"].to_numpy(dtype=float),
                color="#64748b",
                linewidth=0.9,
                alpha=0.65,
                label="inner asymmetry z",
            )
    if "kalman_adjusted_edge_z" in plot_frame.columns:
        adjusted_z = plot_frame.dropna(subset=["kalman_adjusted_edge_z"])
        if not adjusted_z.empty:
            ax_signal.plot(
                adjusted_z.index.to_pydatetime(),
                adjusted_z["kalman_adjusted_edge_z"].to_numpy(dtype=float),
                color="#334155",
                linewidth=0.95,
                alpha=0.70,
                label="Kalman-adjusted edge z",
            )
    if "kalman_direction_z" in plot_frame.columns:
        kalman_z = plot_frame.dropna(subset=["kalman_direction_z"])
        if not kalman_z.empty:
            ax_signal.plot(
                kalman_z.index.to_pydatetime(),
                kalman_z["kalman_direction_z"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=0.95,
                alpha=0.80,
                label="Kalman direction z",
            )
    ax_signal_high_vol = None
    if "kalman_high_vol_signal" in plot_frame.columns:
        high_vol = plot_frame.dropna(subset=["kalman_high_vol_signal"])
        if not high_vol.empty:
            ax_signal_high_vol = ax_signal.twinx()
            ax_signal_high_vol.plot(
                high_vol.index.to_pydatetime(),
                high_vol["kalman_high_vol_signal"].clip(0.0, 1.0).to_numpy(dtype=float),
                color="#d97706",
                linewidth=0.95,
                alpha=0.85,
                label="Kalman high-vol",
            )
            ax_signal_high_vol.set_ylim(-0.05, 1.05)
            ax_signal_high_vol.set_ylabel("High-vol")
            ax_signal_high_vol.tick_params(axis="y", colors="#9a5b00")
    ax_signal.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.75)
    ax_signal.axhline(1.0, color="#0f8b4c", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_signal.axhline(-1.0, color="#b23a48", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_signal.set_ylabel("Asymmetry z")
    ax_signal.set_title(
        f"{INNER_ASYMMETRY_SIGNAL_SPAN}d EWM of horizon inner-asymmetry plus Kalman direction "
        f"across {_format_horizon_label(horizons)}, "
        f"{INNER_ASYMMETRY_ZSCORE_WINDOW}-day rolling z-score"
    )
    ax_signal.grid(alpha=0.20)
    lines, labels = ax_signal.get_legend_handles_labels()
    if ax_signal_high_vol is not None:
        high_vol_lines, high_vol_labels = ax_signal_high_vol.get_legend_handles_labels()
        lines += high_vol_lines
        labels += high_vol_labels
    if lines:
        ax_signal.legend(lines, labels, loc="upper left", ncol=4, frameon=False)
    ax_signal.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_signal.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_signal.xaxis.get_major_locator()))

    if metrics:
        metric_lines = []
        for row in metrics[:6]:
            signal = row.get("signal", "inner_asymmetry")
            metric_lines.append(
                f"{signal} {row['forward_window']}d: rho={row['spearman']:.3f}, "
                f"Q5-Q1={row['top_minus_bottom'] * 100:.1f}%"
            )
        ax_price.text(
            0.985,
            0.03,
            "\n".join(metric_lines),
            transform=ax_price.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#c7d1dd",
            },
        )

    chart_dir.mkdir(parents=True, exist_ok=True)
    path = chart_dir / "btc_inner_asymmetry_edge_indicator.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def build_inner_asymmetry_indicator_artifacts(
    db_path: str,
    table_name: str,
    chart_dir: str | Path,
    horizons: list[int],
    data_bundle: Any | None = None,
    return_type: str | None = None,
) -> dict[str, Any]:
    predictions = _load_predictions(db_path, table_name)
    if predictions.empty:
        return {"chart_path": None, "metrics": []}
    indicator = _build_inner_asymmetry_indicator_frame(
        predictions,
        horizons,
        data_bundle=data_bundle,
        return_type=return_type,
    )
    metrics = _inner_asymmetry_metrics(indicator)
    chart_dir = Path(chart_dir)
    chart_path = _plot_inner_asymmetry_indicator(indicator, metrics, chart_dir, horizons)

    export_frame = indicator.reset_index().rename(columns={"as_of": "timestamp"})
    export_frame.to_csv(chart_dir / "btc_inner_asymmetry_edge_indicator.csv", index=False)
    (chart_dir / "btc_inner_asymmetry_edge_indicator_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n"
    )
    return {"chart_path": chart_path, "metrics": metrics}


def _project_price_scalar(base_close: float, forecast_return: float, return_type: str) -> float:
    if not math.isfinite(base_close) or not math.isfinite(forecast_return):
        return float("nan")
    if abs(float(forecast_return)) > MAX_ABS_PROJECTED_RETURN:
        return float("nan")
    if return_type == "log":
        return float(base_close * math.exp(float(forecast_return)))
    if forecast_return <= -0.99:
        return float("nan")
    return float(base_close * (1.0 + float(forecast_return)))


def _meta_pinball_vector(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> np.ndarray:
    errors = actual - predicted
    return np.maximum(quantile * errors, (quantile - 1.0) * errors)


def _meta_wis_vector(frame: pd.DataFrame) -> np.ndarray:
    y = frame["actual_return"].to_numpy(dtype=float)
    q05 = frame["q05"].to_numpy(dtype=float)
    q25 = frame["q25"].to_numpy(dtype=float)
    q50 = frame["q50"].to_numpy(dtype=float)
    q75 = frame["q75"].to_numpy(dtype=float)
    q95 = frame["q95"].to_numpy(dtype=float)
    score_90 = (q95 - q05) + 20.0 * np.maximum(q05 - y, 0.0) + 20.0 * np.maximum(y - q95, 0.0)
    score_50 = (q75 - q25) + 4.0 * np.maximum(q25 - y, 0.0) + 4.0 * np.maximum(y - q75, 0.0)
    median_loss = np.abs(y - q50)
    return 0.5 * median_loss + 0.25 * score_50 + 0.25 * score_90


def _add_meta_losses(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    actual = pd.to_numeric(out["actual_return"], errors="coerce").to_numpy(dtype=float)
    pinballs = []
    for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES):
        predicted = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        losses = _meta_pinball_vector(actual, predicted, float(quantile))
        out[f"{column}_pinball_loss"] = losses
        pinballs.append(losses)
    pinball_matrix = np.vstack(pinballs)
    finite_counts = np.isfinite(pinball_matrix).sum(axis=0)
    pinball_sums = np.nansum(pinball_matrix, axis=0)
    out["avg_pinball_loss"] = np.divide(
        pinball_sums,
        finite_counts,
        out=np.full(pinball_sums.shape, np.nan),
        where=finite_counts > 0,
    )
    out["wis_loss"] = _meta_wis_vector(out)
    out["width_50"] = pd.to_numeric(out["q75"], errors="coerce") - pd.to_numeric(out["q25"], errors="coerce")
    out["width_90"] = pd.to_numeric(out["q95"], errors="coerce") - pd.to_numeric(out["q05"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _build_meta_feature_frame(
    predictions: pd.DataFrame,
    horizons: list[int],
    data_bundle: Any,
    return_type: str,
) -> pd.DataFrame:
    frame = predictions.loc[predictions["horizon"].isin(horizons)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    for column in ["base_close", "actual_close", "actual_return", *QUANTILE_COLUMNS]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["as_of", "horizon", "base_close"]).sort_values(["as_of", "horizon"])

    kalman_overlay = build_kalman_projection_overlay(
        predictions=frame,
        data_bundle=data_bundle,
        horizons=horizons,
        return_type=return_type,
    )
    if not kalman_overlay.empty:
        kalman_overlay = kalman_overlay.copy()
        kalman_overlay["as_of"] = pd.to_datetime(kalman_overlay["as_of"], errors="coerce")
        kalman_overlay["horizon"] = pd.to_numeric(kalman_overlay["horizon"], errors="coerce").astype("Int64")
        overlay_columns = [
            "as_of",
            "horizon",
            "kalman_fair_value_price",
            "kalman_projected_return",
            "kalman_projected_price",
            "kalman_target_timestamp",
            "kalman_scope_days",
            "kalman_projection_pressure",
            "gap",
            "gap_z",
            "innovation_z",
            "residual_sigma",
            "kalman_high_vol_signal",
            "tail_flare_score",
            "history_count",
        ]
        frame = frame.merge(
            kalman_overlay[[column for column in overlay_columns if column in kalman_overlay.columns]],
            on=["as_of", "horizon"],
            how="left",
        )

    state_history = build_kalman_state_history(data_bundle)
    if not state_history.empty:
        state_columns = [
            "onchain_reversion_score",
            "onchain_average_behavior_score",
            "signalboost_peak_score",
            "onchain_behavior_score",
            "ism_services_stationary_z",
            "inverted_nfci_stationary_z",
            "ism_nfci_stationary_score",
            "btc_log_cycle_z",
            "macro_stationary_gap_score",
            "macro_cycle_score",
            "macro_impact_score",
            "macro_forward_score",
            "macro_scope_days",
            "liquidity_lead_days",
            "cycle_lead_days",
            "financial_conditions_lead_days",
            "lth_signalboost_attention_score",
            "sth_signalboost_attention_score",
            "signalboost_scope_days",
            "kalman_scope_days",
            "kalman_projection_pressure",
            "level_weight_liquidity",
            "level_weight_energy",
            "level_weight_metcalfe",
            "level_weight_macro_cycle",
            "level_weight_stationary_macro",
            "level_weight_onchain",
        ]
        available = [
            column
            for column in state_columns
            if column in state_history.columns and column not in frame.columns
        ]
        if available:
            state_subset = state_history[available].copy()
            state_subset.index = pd.to_datetime(state_subset.index)
            frame = frame.join(state_subset, on="as_of")

    indicator = _build_inner_asymmetry_indicator_frame(
        predictions=predictions,
        horizons=horizons,
        data_bundle=data_bundle,
        return_type=return_type,
    )
    if not indicator.empty:
        indicator_columns = [
            "inner_asymmetry_z",
            "kalman_direction_z",
            "kalman_adjusted_edge_z",
        ]
        available = [column for column in indicator_columns if column in indicator.columns]
        if available:
            indicator_subset = indicator[available].copy()
            indicator_subset.index = pd.to_datetime(indicator_subset.index)
            frame = frame.join(indicator_subset, on="as_of")

    return frame.replace([np.inf, -np.inf], np.nan)


def _empty_jepa_feature_frame() -> pd.DataFrame:
    columns = [
        "as_of_date",
        "horizon",
        "specialist",
        *[f"jepa_{metric}" for metric in JEPA_META_METRICS],
        "checkpoint_id",
        "checkpoint_found",
        "created_at",
    ]
    return pd.DataFrame(columns=columns)


def _compact_jepa_history(
    encoded: pd.DataFrame,
    *,
    specialist: str,
    horizon: int,
    checkpoint_id: str,
    checkpoint_found: bool,
    created_at: str,
) -> pd.DataFrame:
    if encoded.empty:
        return _empty_jepa_feature_frame()
    out = pd.DataFrame(
        {
            "as_of_date": pd.DatetimeIndex(encoded.index),
            "horizon": int(horizon),
            "specialist": specialist,
            "checkpoint_id": checkpoint_id,
            "checkpoint_found": bool(checkpoint_found),
            "created_at": created_at,
        },
        index=encoded.index,
    )
    source_map = {
        "kalman_alignment": f"{specialist}_jepa_kalman_alignment_h{int(horizon)}",
        "reversion_pressure": f"{specialist}_jepa_reversion_pressure_h{int(horizon)}",
        "vol_pressure": f"{specialist}_jepa_vol_pressure_h{int(horizon)}",
        "tail_pressure": f"{specialist}_jepa_tail_pressure_h{int(horizon)}",
        "uncertainty_proxy": f"{specialist}_jepa_uncertainty_proxy_h{int(horizon)}",
        "latent_norm": f"{specialist}_jepa_norm_h{int(horizon)}",
        "delta_norm": f"{specialist}_jepa_delta_norm_h{int(horizon)}",
    }
    for metric, source in source_map.items():
        out[f"jepa_{metric}"] = pd.to_numeric(encoded.get(source), errors="coerce")
    return out.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def _build_jepa_wide_features(long_features: pd.DataFrame) -> pd.DataFrame:
    if long_features.empty:
        return pd.DataFrame(columns=["as_of_date", "horizon", "aggregate_jepa_feature_available"])
    work = long_features.copy()
    work["as_of_date"] = pd.to_datetime(work["as_of_date"], errors="coerce")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["as_of_date", "horizon", "specialist"])
    if work.empty:
        return pd.DataFrame(columns=["as_of_date", "horizon", "aggregate_jepa_feature_available"])

    index_columns = ["as_of_date", "horizon"]
    wide_parts: list[pd.DataFrame] = []
    for metric in JEPA_META_METRICS:
        value_column = f"jepa_{metric}"
        if value_column not in work.columns:
            continue
        pivot = work.pivot_table(
            index=index_columns,
            columns="specialist",
            values=value_column,
            aggfunc="last",
        )
        pivot = pivot.rename(columns={specialist: f"{specialist}_jepa_{metric}" for specialist in pivot.columns})
        wide_parts.append(pivot)
    if not wide_parts:
        return pd.DataFrame(columns=["as_of_date", "horizon", "aggregate_jepa_feature_available"])
    wide = pd.concat(wide_parts, axis=1).reset_index()
    for metric in JEPA_META_METRICS:
        specialist_columns = [
            f"{specialist}_jepa_{metric}"
            for specialist in SPECIALISTS
            if f"{specialist}_jepa_{metric}" in wide.columns
        ]
        if specialist_columns:
            wide[f"aggregate_jepa_{metric}"] = wide[specialist_columns].mean(axis=1, skipna=True)
    availability_columns = [
        f"{specialist}_jepa_kalman_alignment"
        for specialist in SPECIALISTS
        if f"{specialist}_jepa_kalman_alignment" in wide.columns
    ]
    if availability_columns:
        wide["aggregate_jepa_feature_count"] = wide[availability_columns].notna().sum(axis=1)
    else:
        wide["aggregate_jepa_feature_count"] = 0
    wide["aggregate_jepa_feature_available"] = wide["aggregate_jepa_feature_count"].astype(int) > 0
    return wide.replace([np.inf, -np.inf], np.nan)


def _materialize_walk_forward_jepa_features(
    *,
    db_path: str,
    predictions: pd.DataFrame,
    horizons: list[int],
    data_bundle: Any,
    config: JEPAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if predictions.empty:
        return _empty_jepa_feature_frame(), pd.DataFrame(), ["No predictions available for JEPA feature generation."], diagnostics
    if not config.is_active:
        message = "JEPA_USE_IN_META requested, but JEPA is inactive. Set JEPA_MODE=passive/frozen and provide checkpoints."
        return _empty_jepa_feature_frame(), pd.DataFrame(), [message], diagnostics

    wanted = predictions.loc[predictions["horizon"].isin(horizons), ["as_of", "horizon"]].copy()
    wanted["as_of_date"] = pd.to_datetime(wanted["as_of"], errors="coerce")
    wanted["horizon"] = pd.to_numeric(wanted["horizon"], errors="coerce").astype("Int64")
    wanted = wanted.dropna(subset=["as_of_date", "horizon"]).drop_duplicates(subset=["as_of_date", "horizon"])
    if wanted.empty:
        return _empty_jepa_feature_frame(), pd.DataFrame(), ["No usable as_of/horizon pairs for JEPA feature generation."], diagnostics

    panels = build_all_specialist_panels(data_bundle, horizons=tuple(int(h) for h in horizons))
    context_length = int(config.context_lengths[0] if config.context_lengths else 128)
    created_at = _utc_now()
    frames: list[pd.DataFrame] = []
    for specialist in SPECIALISTS:
        panel = panels.get(specialist, pd.DataFrame())
        if panel.empty:
            warnings.append(f"JEPA panel for {specialist} is empty.")
            continue
        for horizon in horizons:
            encoded, diagnostic = encode_panel_history(
                panel,
                specialist_name=specialist,
                horizon=int(horizon),
                checkpoint_dir=config.checkpoint_dir,
                context_length=context_length,
                lookback_rows=0,
                batch_size=int(config.chart_batch_size),
            )
            diagnostics.append(diagnostic)
            if encoded.empty:
                warnings.append(
                    f"No JEPA history for {specialist} {int(horizon)}d: "
                    f"{diagnostic.get('status') or 'checkpoint missing/empty'}."
                )
                continue
            compact = _compact_jepa_history(
                encoded,
                specialist=specialist,
                horizon=int(horizon),
                checkpoint_id=config.checkpoint_id,
                checkpoint_found=bool(diagnostic.get("checkpoint_found")),
                created_at=created_at,
            )
            if compact.empty:
                continue
            compact["horizon"] = pd.to_numeric(compact["horizon"], errors="coerce").astype("Int64")
            compact = compact.merge(
                wanted[["as_of_date", "horizon"]],
                on=["as_of_date", "horizon"],
                how="inner",
            )
            if not compact.empty:
                frames.append(compact)

    long_features = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_jepa_feature_frame()
    wide_features = _build_jepa_wide_features(long_features)
    expected_pairs = int(len(wanted))
    if not wide_features.empty and "aggregate_jepa_feature_available" in wide_features.columns:
        available_mask = wide_features["aggregate_jepa_feature_available"].fillna(False).astype(bool)
        covered_pairs = int(
            wide_features.loc[available_mask, ["as_of_date", "horizon"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        covered_pairs = 0
    if covered_pairs < expected_pairs:
        warnings.append(f"JEPA feature coverage is partial: {covered_pairs}/{expected_pairs} as_of/horizon pairs.")

    with duckdb_connection(db_path) as connection:
        replace_table_from_dataframe(connection, JEPA_FEATURE_TABLE, long_features)
        replace_table_from_dataframe(connection, JEPA_FEATURE_WIDE_TABLE, wide_features)
    return long_features, wide_features, warnings, diagnostics


def _merge_jepa_features_for_meta(
    *,
    feature_frame: pd.DataFrame,
    db_path: str,
    predictions: pd.DataFrame,
    horizons: list[int],
    data_bundle: Any,
    config: JEPAConfig,
    params: JEPAMetaParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    long_features, wide_features, warnings, diagnostics = _materialize_walk_forward_jepa_features(
        db_path=db_path,
        predictions=predictions,
        horizons=horizons,
        data_bundle=data_bundle,
        config=config,
    )
    if params.strict_mode and warnings:
        raise RuntimeError("Strict JEPA meta mode failed: " + "; ".join(warnings))
    if wide_features.empty:
        return feature_frame, long_features, wide_features, warnings, diagnostics

    merge_frame = wide_features.copy()
    merge_frame["as_of"] = pd.to_datetime(merge_frame["as_of_date"], errors="coerce")
    merge_frame["horizon"] = pd.to_numeric(merge_frame["horizon"], errors="coerce").astype("Int64")
    merge_frame = merge_frame.drop(columns=["as_of_date"])
    merged = feature_frame.merge(merge_frame, on=["as_of", "horizon"], how="left")
    if "aggregate_jepa_feature_available" not in merged.columns:
        merged["aggregate_jepa_feature_available"] = False
    merged["aggregate_jepa_feature_available"] = merged["aggregate_jepa_feature_available"].fillna(False).astype(bool)
    missing_rows = int((~merged["aggregate_jepa_feature_available"]).sum())
    if missing_rows:
        warnings.append(f"JEPA features missing for {missing_rows} meta feature rows; JEPA candidates will skip those rows.")
        if params.strict_mode:
            raise RuntimeError("Strict JEPA meta mode failed: missing merged JEPA feature rows.")
    return merged.replace([np.inf, -np.inf], np.nan), long_features, wide_features, warnings, diagnostics


def _meta_horizon_scale(horizon: pd.Series) -> np.ndarray:
    values = pd.to_numeric(horizon, errors="coerce").fillna(1).astype(float).to_numpy(dtype=float)
    return np.clip(np.sqrt(np.maximum(values, 1.0) / 15.0), 0.35, 1.35)


def _synthesize_meta_candidate(feature_frame: pd.DataFrame, spec: dict[str, float | str]) -> pd.DataFrame:
    out = feature_frame.copy()
    candidate = str(spec["name"])
    out["forecaster"] = f"{META_SYNTH_PREFIX}_{candidate}"
    out["meta_candidate"] = candidate
    for large_column in [
        "width_diagnostics_json",
        "lower_tail_diagnostics_json",
        "upper_tail_diagnostics_json",
        "calibration_score_comparison_json",
    ]:
        if large_column in out.columns:
            out[large_column] = None

    numeric_quantiles = out[QUANTILE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if candidate == "model_passthrough":
        out["meta_effective_kalman_weight"] = 0.0
        out["meta_center_shift"] = 0.0
        out["meta_width_multiplier"] = 1.0
        out["meta_selected_candidate"] = candidate
        return out

    q05 = numeric_quantiles["q05"].to_numpy(dtype=float)
    q25 = numeric_quantiles["q25"].to_numpy(dtype=float)
    q50 = numeric_quantiles["q50"].to_numpy(dtype=float)
    q75 = numeric_quantiles["q75"].to_numpy(dtype=float)
    q95 = numeric_quantiles["q95"].to_numpy(dtype=float)
    kalman = pd.to_numeric(out.get("kalman_projected_return"), errors="coerce").to_numpy(dtype=float)

    horizon_values = pd.to_numeric(out["horizon"], errors="coerce").fillna(1).astype(float).to_numpy(dtype=float)
    scope_values = (
        pd.to_numeric(out.get("kalman_scope_days"), errors="coerce")
        if "kalman_scope_days" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    scope_values = scope_values.fillna(pd.Series(horizon_values, index=out.index)).clip(lower=1.0).to_numpy(dtype=float)
    scope_sqrt = np.sqrt(np.maximum(scope_values, 1.0))
    half_width = np.maximum((q75 - q25) / 2.0, 1e-6)
    lower_extension = np.maximum(q25 - q05, 1e-6)
    upper_extension = np.maximum(q95 - q75, 1e-6)
    width_90 = np.maximum.reduce([q95 - q05, 2.0 * half_width, np.full_like(q50, 1e-6)])
    disagreement = kalman - q50

    out_numeric = lambda column, default=0.0: pd.to_numeric(
        out[column] if column in out.columns else pd.Series(default, index=out.index),
        errors="coerce",
    ).fillna(default)
    innovation_z = out_numeric("innovation_z", 0.0).to_numpy(dtype=float)
    gap_z = out_numeric("gap_z", 0.0).to_numpy(dtype=float)
    high_vol = out_numeric("kalman_high_vol_signal", 0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    tail_flare = out_numeric("tail_flare_score", 0.0).clip(0.0, 5.0).to_numpy(dtype=float)
    residual_sigma = out_numeric("residual_sigma", 1e-6).clip(lower=1e-6).to_numpy(dtype=float)
    edge_z = out_numeric("kalman_adjusted_edge_z", 0.0).clip(-3.0, 3.0).to_numpy(dtype=float)

    reliability = np.exp(-0.18 * np.abs(innovation_z)) / (1.0 + 0.10 * np.maximum(np.abs(gap_z) - 1.0, 0.0))
    reliability = np.clip(reliability, 0.20, 1.0)
    sign_agreement = np.sign(q50) == np.sign(kalman)
    disagreement_multiplier = np.where(
        sign_agreement | (np.abs(q50) < 1e-9) | (np.abs(kalman) < 1e-9),
        1.0,
        1.0 - float(spec["disagreement_dampen"]),
    )
    effective_weight = (
        float(spec["center_weight"])
        * _meta_horizon_scale(pd.Series(scope_values, index=out.index))
        * reliability
        * disagreement_multiplier
        * (1.0 - float(spec["high_vol_dampen"]) * high_vol)
    )
    effective_weight = np.clip(effective_weight, 0.0, 0.75)

    clip_width = float(spec["clip_widths"]) * np.maximum.reduce(
        [width_90, residual_sigma * scope_sqrt, np.full_like(q50, 1e-6)]
    )
    center_shift = effective_weight * np.clip(disagreement, -clip_width, clip_width)
    edge_shift = float(spec["edge_weight"]) * edge_z * width_90
    center = q50 + center_shift + edge_shift

    disagreement_ratio = np.minimum(np.abs(disagreement) / np.maximum(width_90, 1e-6), 3.0)
    width_multiplier = (
        1.0
        + float(spec["width_disagreement_weight"]) * disagreement_ratio * reliability
        + float(spec["width_high_vol_weight"]) * high_vol
        + float(spec["tail_flare_weight"]) * np.minimum(tail_flare / 2.0, 1.5)
    )
    width_multiplier = np.clip(width_multiplier, 0.75, 2.25)
    new_half_width = half_width * width_multiplier
    tail_multiplier = np.clip(1.0 + 0.50 * (width_multiplier - 1.0), 0.75, 2.25)
    new_lower_extension = lower_extension * tail_multiplier
    new_upper_extension = upper_extension * tail_multiplier
    if str(spec.get("width_mode", "scale")) == "cap":
        cap_scale = float(spec.get("sigma_cap_scale", 1.50))
        cap_high_vol = 1.0 + 0.75 * high_vol + 0.25 * np.minimum(tail_flare / 2.0, 1.5)
        sigma_half_cap = 0.6744897501960817 * residual_sigma * scope_sqrt * cap_scale * cap_high_vol
        sigma_tail_cap = (1.6448536269514722 - 0.6744897501960817) * residual_sigma * scope_sqrt * cap_scale * cap_high_vol
        new_half_width = np.minimum(new_half_width, np.maximum(sigma_half_cap, 1e-6))
        new_lower_extension = np.minimum(new_lower_extension, np.maximum(sigma_tail_cap, 1e-6))
        new_upper_extension = np.minimum(new_upper_extension, np.maximum(sigma_tail_cap, 1e-6))
    new_q25 = center - new_half_width
    new_q75 = center + new_half_width
    new_q05 = new_q25 - new_lower_extension
    new_q95 = new_q75 + new_upper_extension
    sorted_quantiles = np.sort(np.vstack([new_q05, new_q25, center, new_q75, new_q95]).T, axis=1)

    valid = (
        numeric_quantiles.notna().all(axis=1).to_numpy()
        & np.isfinite(kalman)
        & np.isfinite(center)
        & np.isfinite(sorted_quantiles).all(axis=1)
    )
    for idx, column in enumerate(QUANTILE_COLUMNS):
        values = out[column].to_numpy(dtype=float, copy=True)
        values[valid] = sorted_quantiles[valid, idx]
        out[column] = values
    out["meta_effective_kalman_weight"] = np.where(valid, effective_weight, np.nan)
    out["meta_center_shift"] = np.where(valid, center_shift + edge_shift, np.nan)
    out["meta_width_multiplier"] = np.where(valid, width_multiplier, np.nan)
    out["meta_selected_candidate"] = candidate
    return out.replace([np.inf, -np.inf], np.nan)


def _build_meta_candidate_panel(
    feature_frame: pd.DataFrame,
    *,
    include_jepa: bool = False,
    jepa_params: JEPAMetaParams | None = None,
) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame()
    candidates = [
        _synthesize_meta_candidate(feature_frame, spec)
        for spec in META_CANDIDATE_SPECS
    ]
    if include_jepa:
        jepa_source = feature_frame.copy()
        if "aggregate_jepa_feature_available" in jepa_source.columns:
            jepa_source = jepa_source.loc[jepa_source["aggregate_jepa_feature_available"].fillna(False).astype(bool)]
        else:
            jepa_source = jepa_source.iloc[0:0]
        if not jepa_source.empty:
            for candidate_name in JEPA_META_CANDIDATES:
                candidates.append(
                    apply_jepa_meta_candidate(
                        jepa_source,
                        candidate=candidate_name,
                        params=jepa_params,
                    )
                )
    panel = pd.concat(candidates, ignore_index=True, sort=False)
    return _add_meta_losses(panel)


def _default_meta_candidate_for_horizon(horizon: int) -> str:
    return "model_passthrough"


def _has_pathological_quantile_width(row: pd.Series) -> bool:
    try:
        q05 = float(row["q05"])
        q25 = float(row["q25"])
        q50 = float(row["q50"])
        q75 = float(row["q75"])
        q95 = float(row["q95"])
    except Exception:
        return True
    values = np.asarray([q05, q25, q50, q75, q95], dtype=float)
    if not np.isfinite(values).all():
        return True
    if not (q05 <= q25 <= q50 <= q75 <= q95):
        return True
    width_90 = q95 - q05
    if width_90 > 1.50:
        return True
    return bool(np.nanmax(np.abs(values)) > 1.25)


def _select_meta_candidate_walk_forward(
    candidate_panel: pd.DataFrame,
    horizons: list[int],
    *,
    min_history: int,
    selection_window: int,
) -> pd.DataFrame:
    if candidate_panel.empty:
        return pd.DataFrame()
    candidates = candidate_panel.copy()
    candidates["as_of"] = pd.to_datetime(candidates["as_of"], errors="coerce")
    candidates["target_timestamp"] = pd.to_datetime(candidates["target_timestamp"], errors="coerce")
    candidates["horizon"] = pd.to_numeric(candidates["horizon"], errors="coerce").astype("Int64")
    lookup = candidates.set_index(["meta_candidate", "as_of", "horizon"], drop=False)
    selected_rows: list[dict[str, Any]] = []

    for horizon in sorted(int(value) for value in horizons):
        horizon_frame = candidates.loc[candidates["horizon"].astype(int) == horizon].copy()
        if horizon_frame.empty:
            continue
        as_of_values = sorted(pd.Timestamp(value) for value in horizon_frame["as_of"].dropna().unique())
        default_candidate = _default_meta_candidate_for_horizon(horizon)
        for as_of in as_of_values:
            history = horizon_frame.loc[
                horizon_frame["target_timestamp"].notna()
                & (horizon_frame["target_timestamp"] < as_of)
                & horizon_frame["actual_return"].notna()
                & horizon_frame["wis_loss"].notna()
            ].copy()
            if int(selection_window) > 0 and not history.empty:
                eligible_asofs = sorted(pd.Timestamp(value) for value in history["as_of"].dropna().unique())
                keep_asofs = set(eligible_asofs[-int(selection_window) :])
                history = history.loc[history["as_of"].isin(keep_asofs)]

            selected_candidate = default_candidate
            selected_score = float("nan")
            selected_count = 0
            if not history.empty:
                scores = (
                    history.groupby("meta_candidate", sort=True)
                    .agg(
                        observations=("wis_loss", "count"),
                        wis=("wis_loss", "mean"),
                        avg_pinball=("avg_pinball_loss", "mean"),
                    )
                    .reset_index()
                )
                scores = scores.loc[scores["observations"] >= int(min_history)]
                if not scores.empty:
                    best = scores.sort_values(["wis", "avg_pinball", "meta_candidate"]).iloc[0]
                    model_score = scores.loc[scores["meta_candidate"] == "model_passthrough"]
                    if not model_score.empty:
                        model_score = model_score.iloc[0]
                        required_wis = float(model_score["wis"]) * (1.0 - META_SELECTION_MIN_WIS_IMPROVEMENT)
                        if str(best["meta_candidate"]) == "model_passthrough" or float(best["wis"]) <= required_wis:
                            selected_candidate = str(best["meta_candidate"])
                            selected_score = float(best["wis"])
                            selected_count = int(best["observations"])
                        else:
                            selected_candidate = "model_passthrough"
                            selected_score = float(model_score["wis"])
                            selected_count = int(model_score["observations"])
                    else:
                        selected_candidate = str(best["meta_candidate"])
                        selected_score = float(best["wis"])
                        selected_count = int(best["observations"])

            key = (selected_candidate, as_of, horizon)
            if key not in lookup.index:
                fallback_key = ("model_passthrough", as_of, horizon)
                if fallback_key not in lookup.index:
                    continue
                key = fallback_key
                selected_candidate = "model_passthrough"
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if selected_candidate == "model_passthrough" and _has_pathological_quantile_width(row):
                for fallback_candidate in ("sigma_cap_light", "sigma_cap_balanced", "sigma_cap_edge"):
                    fallback_key = (fallback_candidate, as_of, horizon)
                    if fallback_key in lookup.index:
                        fallback = lookup.loc[fallback_key]
                        if isinstance(fallback, pd.DataFrame):
                            fallback = fallback.iloc[0]
                        row = fallback
                        selected_candidate = fallback_candidate
                        break
            out = row.to_dict()
            out["forecaster"] = META_SELECTED_FORECASTER
            out["meta_selected_candidate"] = selected_candidate
            out["meta_selection_wis"] = selected_score
            out["meta_selection_observations"] = selected_count
            selected_rows.append(out)

    if not selected_rows:
        return pd.DataFrame()
    return _add_meta_losses(pd.DataFrame(selected_rows))


def _summarize_meta_panel(panel: pd.DataFrame) -> pd.DataFrame:
    clean = panel.dropna(subset=["actual_return", *QUANTILE_COLUMNS]).copy()
    if clean.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (forecaster, horizon), group in clean.groupby(["forecaster", "horizon"], sort=True):
        y = group["actual_return"].to_numpy(dtype=float)
        pinballs = [
            float(np.nanmean(_meta_pinball_vector(y, group[column].to_numpy(dtype=float), quantile)))
            for column, quantile in zip(QUANTILE_COLUMNS, QUANTILES)
        ]
        sign_mask = group["actual_return"].ne(0.0) & group["q50"].ne(0.0)
        q50_error = group["q50"].astype(float) - group["actual_return"].astype(float)
        row = {
            "forecaster": forecaster,
            "horizon": int(horizon),
            "observations": int(len(group)),
            "avg_pinball": float(np.nanmean(pinballs)),
            "wis": float(np.nanmean(group["wis_loss"])),
            "q50_absolute_error": float(np.nanmean(np.abs(q50_error))),
            "q50_signed_error": float(np.nanmean(q50_error)),
            "coverage_50": float(((group["actual_return"] >= group["q25"]) & (group["actual_return"] <= group["q75"])).mean()),
            "coverage_90": float(((group["actual_return"] >= group["q05"]) & (group["actual_return"] <= group["q95"])).mean()),
            "lower_tail_miss_rate": float((group["actual_return"] < group["q05"]).mean()),
            "upper_tail_miss_rate": float((group["actual_return"] > group["q95"]).mean()),
            "tail_miss_rate": float(((group["actual_return"] < group["q05"]) | (group["actual_return"] > group["q95"])).mean()),
            "width_50": float(group["width_50"].mean()),
            "width_90": float(group["width_90"].mean()),
            "interval_width": float(group["width_90"].mean()),
            "directional_hit_rate": float(
                (np.sign(group.loc[sign_mask, "actual_return"]) == np.sign(group.loc[sign_mask, "q50"])).mean()
            )
            if sign_mask.any()
            else float("nan"),
            "q50_actual_spearman": float(group["q50"].corr(group["actual_return"], method="spearman")),
            "q05_pinball": pinballs[0],
            "q25_pinball": pinballs[1],
            "q50_pinball": pinballs[2],
            "q75_pinball": pinballs[3],
            "q95_pinball": pinballs[4],
        }

        regime_source = lambda column: pd.to_numeric(
            group[column] if column in group.columns else pd.Series(0.0, index=group.index),
            errors="coerce",
        ).fillna(0.0)
        regime_masks = {
            "high_kalman_high_vol": regime_source("kalman_high_vol_signal") >= 0.50,
            "high_tail_flare": regime_source("tail_flare_score") >= 1.00,
            "high_jepa_uncertainty": regime_source("aggregate_jepa_uncertainty_proxy") >= 0.70,
            "gap_z_gt_2": regime_source("gap_z").abs() > 2.00,
        }
        for regime_name, mask in regime_masks.items():
            subset = group.loc[mask]
            row[f"{regime_name}_observations"] = int(len(subset))
            row[f"{regime_name}_wis"] = float(np.nanmean(subset["wis_loss"])) if not subset.empty else float("nan")
            row[f"{regime_name}_avg_pinball"] = float(np.nanmean(subset["avg_pinball_loss"])) if not subset.empty else float("nan")
            row[f"{regime_name}_q50_absolute_error"] = (
                float(np.nanmean(np.abs(subset["q50"].astype(float) - subset["actual_return"].astype(float))))
                if not subset.empty
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon", "forecaster"]).reset_index(drop=True)


def _meta_selection_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    all_metrics = metrics.copy()
    model = all_metrics.loc[all_metrics["forecaster"] == f"{META_SYNTH_PREFIX}_model_passthrough"]
    selected = all_metrics.loc[all_metrics["forecaster"] == META_SELECTED_FORECASTER]
    candidates = all_metrics.loc[all_metrics["forecaster"] != META_SELECTED_FORECASTER]
    rows: list[dict[str, Any]] = []
    for horizon, horizon_candidates in candidates.groupby("horizon", sort=True):
        horizon = int(horizon)
        best = horizon_candidates.sort_values(["wis", "avg_pinball"]).iloc[0]
        model_row = model.loc[model["horizon"].astype(int) == horizon]
        selected_row = selected.loc[selected["horizon"].astype(int) == horizon]
        model_row = model_row.iloc[0] if not model_row.empty else None
        selected_row = selected_row.iloc[0] if not selected_row.empty else None
        rows.append(
            {
                "horizon": horizon,
                "model_wis": float(model_row["wis"]) if model_row is not None else float("nan"),
                "model_avg_pinball": float(model_row["avg_pinball"]) if model_row is not None else float("nan"),
                "best_meta_forecaster": best["forecaster"],
                "best_meta_wis": float(best["wis"]),
                "best_meta_avg_pinball": float(best["avg_pinball"]),
                "best_meta_wis_minus_model": float(best["wis"] - model_row["wis"]) if model_row is not None else float("nan"),
                "selected_meta_wis": float(selected_row["wis"]) if selected_row is not None else float("nan"),
                "selected_meta_avg_pinball": float(selected_row["avg_pinball"]) if selected_row is not None else float("nan"),
                "selected_meta_wis_minus_model": float(selected_row["wis"] - model_row["wis"])
                if selected_row is not None and model_row is not None
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _plot_meta_value_envelope_chart(
    selected: pd.DataFrame,
    model_predictions: pd.DataFrame,
    horizon: int,
    chart_dir: Path,
    return_type: str,
    *,
    filename_prefix: str = "walk_forward_meta_value_envelope",
    title_label: str = "Meta",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    horizon_frame = selected.loc[selected["horizon"].astype(int) == int(horizon)].copy()
    horizon_frame["as_of"] = pd.to_datetime(horizon_frame["as_of"], errors="coerce")
    horizon_frame["target_timestamp"] = pd.to_datetime(horizon_frame["target_timestamp"], errors="coerce")
    horizon_frame = horizon_frame.dropna(subset=["base_close", "target_timestamp"]).sort_values("target_timestamp")
    if horizon_frame.empty:
        raise ValueError(f"No meta prediction rows with base prices for {horizon}d.")

    model_horizon = model_predictions.loc[model_predictions["horizon"].astype(int) == int(horizon)].copy()
    model_horizon["as_of"] = pd.to_datetime(model_horizon["as_of"], errors="coerce")
    model_horizon["target_timestamp"] = pd.to_datetime(model_horizon["target_timestamp"], errors="coerce")
    model_horizon = model_horizon.sort_values("target_timestamp")

    for column in QUANTILE_COLUMNS:
        horizon_frame[f"{column}_price"] = _project_price(
            horizon_frame["base_close"],
            horizon_frame[column],
            return_type,
        )
    if not model_horizon.empty:
        model_horizon["q50_price"] = _project_price(model_horizon["base_close"], model_horizon["q50"], return_type)

    realized = horizon_frame.dropna(subset=["actual_return"])
    if not realized.empty:
        residual = realized["actual_return"].astype(float) - realized["q50"].astype(float)
        horizon_frame.loc[realized.index, "meta_error_z"] = _rolling_zscore(residual)

    fig, (ax_price, ax_meta) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    x = horizon_frame["target_timestamp"].dt.to_pydatetime()
    ax_price.fill_between(
        x,
        horizon_frame["q05_price"].to_numpy(dtype=float),
        horizon_frame["q95_price"].to_numpy(dtype=float),
        color="#7aa6d1",
        alpha=0.18,
        label="meta q05-q95",
    )
    ax_price.fill_between(
        x,
        horizon_frame["q25_price"].to_numpy(dtype=float),
        horizon_frame["q75_price"].to_numpy(dtype=float),
        color="#2f6da3",
        alpha=0.30,
        label="meta q25-q75",
    )
    ax_price.plot(
        x,
        horizon_frame["q50_price"].to_numpy(dtype=float),
        color="#9b2f3d",
        linewidth=1.8,
        label="meta q50 target price",
    )
    if not model_horizon.empty:
        model_plot = model_horizon.dropna(subset=["q50_price", "target_timestamp"])
        if not model_plot.empty:
            ax_price.plot(
                model_plot["target_timestamp"].dt.to_pydatetime(),
                model_plot["q50_price"].to_numpy(dtype=float),
                color="#64748b",
                linestyle=":",
                linewidth=1.35,
                alpha=0.82,
                label="original model q50",
            )
    if "kalman_fair_value_price" in horizon_frame.columns:
        fair_value = horizon_frame.dropna(subset=["kalman_fair_value_price", "as_of"])
        if not fair_value.empty:
            ax_price.plot(
                fair_value["as_of"].dt.to_pydatetime(),
                fair_value["kalman_fair_value_price"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.20,
                alpha=0.88,
                label="Kalman fair value",
            )
    if "kalman_projected_price" in horizon_frame.columns:
        projected = horizon_frame.dropna(subset=["kalman_projected_price", "target_timestamp"])
        if not projected.empty:
            ax_price.plot(
                projected["target_timestamp"].dt.to_pydatetime(),
                projected["kalman_projected_price"].to_numpy(dtype=float),
                color="#16a34a",
                linestyle="--",
                linewidth=1.35,
                alpha=0.88,
                label="Kalman projected target",
            )
    if not realized.empty:
        ax_price.plot(
            realized["target_timestamp"].dt.to_pydatetime(),
            realized["actual_close"].to_numpy(dtype=float),
            color="#222222",
            linewidth=1.25,
            alpha=0.88,
            label="BTC close at target",
        )
    if horizon_frame["base_close"].dropna().gt(0).all() and horizon_frame["q05_price"].dropna().gt(0).all():
        ax_price.set_yscale("log")
    ax_price.set_title(f"{horizon}d Walk-Forward BTC {title_label} Value Envelope")
    ax_price.set_ylabel("BTC Price")
    ax_price.grid(alpha=0.20)
    ax_price.legend(loc="upper left", ncol=4, frameon=False)

    if not realized.empty:
        coverage_90 = ((realized["actual_close"] >= realized["q05_price"]) & (realized["actual_close"] <= realized["q95_price"])).mean()
        coverage_50 = ((realized["actual_close"] >= realized["q25_price"]) & (realized["actual_close"] <= realized["q75_price"])).mean()
        summary_text = "\n".join(
            [
                f"Forecasts: {len(horizon_frame):,}",
                f"Realized: {len(realized):,}",
                f"q05-q95 coverage: {coverage_90:.3f}",
                f"q25-q75 coverage: {coverage_50:.3f}",
                f"latest candidate: {horizon_frame['meta_selected_candidate'].dropna().iloc[-1] if horizon_frame['meta_selected_candidate'].notna().any() else 'n/a'}",
            ]
        )
        ax_price.text(
            0.985,
            0.03,
            summary_text,
            transform=ax_price.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#c7d1dd"},
        )

    z_frame = horizon_frame.dropna(subset=["meta_error_z"])
    if not z_frame.empty:
        ax_meta.plot(
            z_frame["target_timestamp"].dt.to_pydatetime(),
            z_frame["meta_error_z"].to_numpy(dtype=float),
            color="#334155",
            linewidth=1.20,
            label="meta q50 error z",
        )
        ax_meta.fill_between(
            z_frame["target_timestamp"].dt.to_pydatetime(),
            0.0,
            z_frame["meta_error_z"].to_numpy(dtype=float),
            color="#8c96a3",
            alpha=0.22,
        )
    weight_frame = horizon_frame.dropna(subset=["meta_effective_kalman_weight"])
    ax_weight = None
    if not weight_frame.empty:
        ax_weight = ax_meta.twinx()
        ax_weight.plot(
            weight_frame["target_timestamp"].dt.to_pydatetime(),
            weight_frame["meta_effective_kalman_weight"].clip(0.0, 1.0).to_numpy(dtype=float),
            color="#0f766e",
            linewidth=1.0,
            alpha=0.85,
            label="effective Kalman weight",
        )
        ax_weight.set_ylim(-0.05, 1.05)
        ax_weight.set_ylabel("Kalman weight")
        ax_weight.tick_params(axis="y", colors="#0f766e")
    ax_meta.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.7)
    ax_meta.axhline(2.0, color="#9b2f3d", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_meta.axhline(-2.0, color="#2f6da3", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_meta.set_ylabel("Error z")
    ax_meta.grid(alpha=0.20)
    lines, labels = ax_meta.get_legend_handles_labels()
    if ax_weight is not None:
        weight_lines, weight_labels = ax_weight.get_legend_handles_labels()
        lines += weight_lines
        labels += weight_labels
    if lines:
        ax_meta.legend(lines, labels, loc="upper left", frameon=False)
    ax_meta.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_meta.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_meta.xaxis.get_major_locator()))

    chart_dir.mkdir(parents=True, exist_ok=True)
    path = chart_dir / f"{filename_prefix}_{horizon}d.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _plot_latest_meta_forecast(
    selected: pd.DataFrame,
    model_predictions: pd.DataFrame,
    data_bundle: Any,
    horizons: list[int],
    return_type: str,
    chart_dir: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if selected.empty:
        raise ValueError("No meta rows available for latest meta plot.")
    work = selected.copy()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce")
    latest_as_of = pd.Timestamp(work["as_of"].max())
    latest = work.loc[work["as_of"] == latest_as_of].sort_values("horizon")
    if latest.empty:
        raise ValueError("No latest meta rows available.")
    close = data_bundle.close.astype(float).sort_index()
    history = close.loc[close.index <= latest_as_of].iloc[-min(180, len(close.loc[close.index <= latest_as_of])) :]
    if history.empty:
        history = close.iloc[-min(180, len(close)) :]
    last_close = float(latest["base_close"].dropna().iloc[0])

    model_latest = model_predictions.copy()
    model_latest["as_of"] = pd.to_datetime(model_latest["as_of"], errors="coerce")
    model_latest = model_latest.loc[model_latest["as_of"] == latest_as_of].sort_values("horizon")

    future_dates = [latest_as_of]
    q05_prices = [last_close]
    q25_prices = [last_close]
    q50_prices = [last_close]
    q75_prices = [last_close]
    q95_prices = [last_close]
    model_q50_prices = [last_close]
    for _, row in latest.iterrows():
        future_dates.append(pd.Timestamp(row["target_timestamp"]))
        q05_prices.append(_project_price_scalar(last_close, float(row["q05"]), return_type))
        q25_prices.append(_project_price_scalar(last_close, float(row["q25"]), return_type))
        q50_prices.append(_project_price_scalar(last_close, float(row["q50"]), return_type))
        q75_prices.append(_project_price_scalar(last_close, float(row["q75"]), return_type))
        q95_prices.append(_project_price_scalar(last_close, float(row["q95"]), return_type))
        model_row = model_latest.loc[model_latest["horizon"].astype(int) == int(row["horizon"])]
        model_q50 = float(model_row["q50"].iloc[0]) if not model_row.empty else float("nan")
        model_q50_prices.append(_project_price_scalar(last_close, model_q50, return_type))

    state_history = build_kalman_state_history(data_bundle)
    projection_rows = latest[["as_of", "target_timestamp", "horizon", "base_close"]].copy()
    kalman_projection = build_kalman_projection_overlay(
        predictions=projection_rows,
        data_bundle=data_bundle,
        horizons=horizons,
        return_type=return_type,
    )

    fig, (ax_price, ax_meta) = plt.subplots(
        2,
        1,
        figsize=(13, 7.8),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    ax_price.plot(history.index, history.to_numpy(dtype=float), color="#223a5e", linewidth=2.0, label="Close")
    if not state_history.empty and "kalman_fair_value_price" in state_history.columns:
        recent_state = state_history.loc[state_history.index >= history.index[0]].dropna(subset=["kalman_fair_value_price"])
        if not recent_state.empty:
            ax_price.plot(
                recent_state.index,
                recent_state["kalman_fair_value_price"].to_numpy(dtype=float),
                color="#0f766e",
                linewidth=1.45,
                alpha=0.90,
                label="Kalman fair value",
            )
    ax_price.fill_between(future_dates, q05_prices, q95_prices, color="#7aa6d1", alpha=0.18, label="meta q05-q95")
    ax_price.fill_between(future_dates, q25_prices, q75_prices, color="#2f6da3", alpha=0.30, label="meta q25-q75")
    ax_price.plot(future_dates, q50_prices, color="#b23a48", linewidth=2.0, marker="o", label="meta q50")
    ax_price.plot(future_dates, model_q50_prices, color="#64748b", linewidth=1.3, linestyle=":", marker=".", label="original model q50")
    if not kalman_projection.empty:
        projected_time_column = (
            "kalman_target_timestamp"
            if "kalman_target_timestamp" in kalman_projection.columns
            and kalman_projection["kalman_target_timestamp"].notna().any()
            else "target_timestamp"
        )
        projected = kalman_projection.dropna(subset=["kalman_projected_price", projected_time_column]).sort_values("horizon")
        if not projected.empty:
            kalman_row = projected.iloc[0]
            kalman_target = pd.Timestamp(kalman_row[projected_time_column])
            kalman_price = float(kalman_row["kalman_projected_price"])
            if pd.notna(kalman_target) and math.isfinite(kalman_price) and kalman_target > latest_as_of:
                ax_price.plot(
                    [latest_as_of, kalman_target],
                    [last_close, kalman_price],
                    color="#16a34a",
                    linestyle="--",
                    linewidth=1.55,
                    marker="o",
                    markersize=3.0,
                    label="Kalman projected scope",
                )
    for _, row in latest.iterrows():
        price = _project_price_scalar(last_close, float(row["q50"]), return_type)
        ax_price.annotate(
            f"{int(row['horizon'])}d\n{row['q50'] * 100:.2f}%",
            xy=(pd.Timestamp(row["target_timestamp"]), price),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#7a2731",
        )
    ax_price.set_title(f"BTC Meta Forecast: {_format_horizon_label(horizons)} Projection")
    ax_price.set_ylabel("BTC Price")
    ax_price.grid(alpha=0.20)
    ax_price.legend(loc="upper left", ncol=4, frameon=False)

    meta_x = latest["target_timestamp"].to_list()
    weights = pd.to_numeric(latest["meta_effective_kalman_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    shifts = pd.to_numeric(latest["meta_center_shift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ax_meta.bar(meta_x, weights, width=1.4, color="#0f766e", alpha=0.35, label="effective Kalman weight")
    ax_shift = ax_meta.twinx()
    ax_shift.plot(meta_x, shifts * 100.0, color="#9b2f3d", marker="o", linewidth=1.35, label="center shift, pct pts")
    ax_meta.set_ylim(0.0, 1.0)
    ax_meta.set_ylabel("Kalman weight")
    ax_shift.set_ylabel("Center shift")
    ax_meta.grid(alpha=0.20)
    lines, labels = ax_meta.get_legend_handles_labels()
    shift_lines, shift_labels = ax_shift.get_legend_handles_labels()
    ax_meta.legend(lines + shift_lines, labels + shift_labels, loc="upper left", frameon=False)
    latest_candidate = latest["meta_selected_candidate"].astype(str).mode()
    candidate_label = latest_candidate.iloc[0] if not latest_candidate.empty else "n/a"
    ax_price.text(
        0.985,
        0.03,
        "\n".join(
            [
                f"Last close: {last_close:,.2f}",
                f"Forecast date: {latest_as_of.date()}",
                f"Meta candidate: {candidate_label}",
            ]
        ),
        transform=ax_price.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#c7d1dd"},
    )

    chart_dir.mkdir(parents=True, exist_ok=True)
    path = chart_dir / "latest_meta_plot.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _plot_jepa_meta_diagnostics(
    *,
    feature_frame: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    selected: pd.DataFrame,
    chart_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    paths: list[str] = []
    chart_dir.mkdir(parents=True, exist_ok=True)
    features = feature_frame.copy()
    if features.empty or "aggregate_jepa_feature_available" not in features.columns:
        return paths
    features["as_of"] = pd.to_datetime(features["as_of"], errors="coerce")
    features = features.loc[features["aggregate_jepa_feature_available"].fillna(False).astype(bool)].copy()
    if features.empty:
        return paths

    aggregate = features.groupby("as_of", sort=True).agg(
        jepa_kalman_alignment=("aggregate_jepa_kalman_alignment", "mean"),
        jepa_uncertainty_proxy=("aggregate_jepa_uncertainty_proxy", "mean"),
        jepa_reversion_pressure=("aggregate_jepa_reversion_pressure", "mean"),
        jepa_tail_pressure=("aggregate_jepa_tail_pressure", "mean"),
        kalman_projection_pressure=("kalman_projection_pressure", "mean"),
        gap_z=("gap_z", "mean"),
        tail_flare_score=("tail_flare_score", "mean"),
        actual_abs_error=("actual_return", lambda series: float(np.nanmean(np.abs(pd.to_numeric(series, errors="coerce"))))),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
    ax.plot(
        aggregate["as_of"],
        aggregate["jepa_kalman_alignment"],
        color="#345995",
        linewidth=1.6,
        label="JEPA/Kalman alignment",
    )
    ax.plot(
        aggregate["as_of"],
        np.sign(pd.to_numeric(aggregate["kalman_projection_pressure"], errors="coerce")).fillna(0.0),
        color="#0f766e",
        linewidth=1.1,
        alpha=0.75,
        label="Kalman pressure sign",
    )
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax.set_title("JEPA vs Kalman Alignment")
    ax.set_ylabel("Alignment / direction")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    path = chart_dir / "jepa_meta_alignment_over_time.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.resolve()))

    realized = features.dropna(subset=["actual_return"]).copy()
    if not realized.empty:
        selected_errors = selected.copy()
        if not selected_errors.empty:
            selected_errors["as_of"] = pd.to_datetime(selected_errors["as_of"], errors="coerce")
            selected_errors["q50_abs_error"] = (
                pd.to_numeric(selected_errors["q50"], errors="coerce")
                - pd.to_numeric(selected_errors["actual_return"], errors="coerce")
            ).abs()
            error_frame = selected_errors[["as_of", "horizon", "q50_abs_error"]].merge(
                features[["as_of", "horizon", "aggregate_jepa_uncertainty_proxy"]],
                on=["as_of", "horizon"],
                how="left",
            ).dropna(subset=["q50_abs_error", "aggregate_jepa_uncertainty_proxy"])
        else:
            error_frame = pd.DataFrame()
        if not error_frame.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
            ax.scatter(
                error_frame["aggregate_jepa_uncertainty_proxy"],
                error_frame["q50_abs_error"],
                s=13,
                alpha=0.45,
                color="#6d597a",
            )
            ax.set_title("JEPA Uncertainty vs Realized q50 Absolute Error")
            ax.set_xlabel("Aggregate JEPA uncertainty proxy")
            ax.set_ylabel("q50 absolute error")
            ax.grid(alpha=0.22)
            path = chart_dir / "jepa_meta_uncertainty_vs_abs_error.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path.resolve()))

    gap_values = pd.to_numeric(
        features["gap_z"] if "gap_z" in features.columns else pd.Series(0.0, index=features.index),
        errors="coerce",
    ).fillna(0.0)
    gap_extremes = features.loc[gap_values.abs() > 2.0].copy()
    if not gap_extremes.empty and "actual_return" in gap_extremes.columns:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        ax.scatter(
            pd.to_numeric(gap_extremes["aggregate_jepa_reversion_pressure"], errors="coerce"),
            pd.to_numeric(gap_extremes["actual_return"], errors="coerce"),
            s=14,
            alpha=0.45,
            color="#345995",
        )
        ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
        ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
        ax.set_title("JEPA Reversion Pressure vs Future Return after |gap_z| > 2")
        ax.set_xlabel("Aggregate JEPA reversion pressure")
        ax.set_ylabel("Future return")
        ax.grid(alpha=0.22)
        path = chart_dir / "jepa_meta_reversion_vs_gap_return.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path.resolve()))

    if "actual_return" in features.columns:
        tail_frame = features.copy()
        tail_frame["tail_miss"] = (
            (pd.to_numeric(tail_frame["actual_return"], errors="coerce") < pd.to_numeric(tail_frame["q05"], errors="coerce"))
            | (pd.to_numeric(tail_frame["actual_return"], errors="coerce") > pd.to_numeric(tail_frame["q95"], errors="coerce"))
        ).astype(float)
        tail_frame = tail_frame.dropna(subset=["aggregate_jepa_tail_pressure"])
        if not tail_frame.empty:
            tail_frame["pressure_bin"] = pd.qcut(
                pd.to_numeric(tail_frame["aggregate_jepa_tail_pressure"], errors="coerce").rank(method="first"),
                q=min(5, max(1, len(tail_frame) // 20)),
                duplicates="drop",
            )
            binned = tail_frame.groupby("pressure_bin", observed=True).agg(
                tail_pressure=("aggregate_jepa_tail_pressure", "mean"),
                tail_miss_rate=("tail_miss", "mean"),
                observations=("tail_miss", "count"),
            ).dropna().reset_index(drop=True)
            if not binned.empty:
                fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
                ax.plot(
                    binned["tail_pressure"],
                    binned["tail_miss_rate"],
                    color="#d1495b",
                    marker="o",
                    linewidth=1.5,
                )
                ax.set_title("JEPA Tail Pressure vs Tail Miss Rate")
                ax.set_xlabel("Average aggregate JEPA tail pressure")
                ax.set_ylabel("Tail miss rate")
                ax.grid(alpha=0.22)
                path = chart_dir / "jepa_meta_tail_pressure_vs_tail_misses.png"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                paths.append(str(path.resolve()))
    return paths


def build_meta_synthesis_artifacts(
    db_path: str,
    table_name: str,
    chart_dir: str | Path,
    horizons: list[int],
    data_bundle: Any,
    return_type: str,
    *,
    min_history: int = META_DEFAULT_MIN_HISTORY,
    selection_window: int = META_DEFAULT_SELECTION_WINDOW,
) -> dict[str, Any]:
    predictions = _load_predictions(db_path, table_name)
    if predictions.empty:
        return {"artifacts": {}, "chart_paths": [], "metrics": [], "selection": []}

    jepa_config = load_jepa_config()
    jepa_params = load_jepa_meta_params()
    use_jepa_meta = bool(jepa_config.use_in_meta)
    jepa_warnings: list[str] = []
    jepa_diagnostics: list[dict[str, Any]] = []
    jepa_long_features = _empty_jepa_feature_frame()
    jepa_wide_features = pd.DataFrame()

    feature_frame = _build_meta_feature_frame(
        predictions=predictions,
        horizons=horizons,
        data_bundle=data_bundle,
        return_type=return_type,
    )
    if use_jepa_meta:
        feature_frame, jepa_long_features, jepa_wide_features, jepa_warnings, jepa_diagnostics = _merge_jepa_features_for_meta(
            feature_frame=feature_frame,
            db_path=db_path,
            predictions=predictions,
            horizons=horizons,
            data_bundle=data_bundle,
            config=jepa_config,
            params=jepa_params,
        )
    include_jepa_candidates = bool(
        use_jepa_meta
        and "aggregate_jepa_feature_available" in feature_frame.columns
        and feature_frame["aggregate_jepa_feature_available"].fillna(False).astype(bool).any()
    )
    if use_jepa_meta and not include_jepa_candidates:
        jepa_warnings.append("No merged JEPA features were available; JEPA candidates were not added.")
        if jepa_params.strict_mode:
            raise RuntimeError("Strict JEPA meta mode failed: no merged JEPA features available.")

    candidate_panel = _build_meta_candidate_panel(
        feature_frame,
        include_jepa=include_jepa_candidates,
        jepa_params=jepa_params,
    )
    selected = _select_meta_candidate_walk_forward(
        candidate_panel,
        horizons,
        min_history=int(min_history),
        selection_window=int(selection_window),
    )
    combined = pd.concat([candidate_panel, selected], ignore_index=True, sort=False) if not selected.empty else candidate_panel
    metrics = _summarize_meta_panel(combined)
    selection = _meta_selection_table(metrics)

    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "meta_candidates": chart_dir / "meta_synthesis_candidates.csv",
        "meta_selected": chart_dir / "meta_synthesis_selected.csv",
        "meta_metrics": chart_dir / "meta_synthesis_metrics.csv",
        "meta_selection": chart_dir / "meta_synthesis_selection.csv",
        "meta_report": chart_dir / "meta_synthesis_report.json",
    }
    jepa_artifacts = {
        "jepa_candidates": chart_dir / "meta_synthesis_jepa_candidates.csv",
        "jepa_selected": chart_dir / "meta_synthesis_jepa_selected.csv",
        "jepa_metrics": chart_dir / "meta_synthesis_jepa_metrics.csv",
        "jepa_selection": chart_dir / "meta_synthesis_jepa_selection.csv",
        "jepa_report": chart_dir / "meta_synthesis_jepa_report.json",
    }
    candidate_panel.to_csv(artifacts["meta_candidates"], index=False)
    selected.to_csv(artifacts["meta_selected"], index=False)
    metrics.to_csv(artifacts["meta_metrics"], index=False)
    selection.to_csv(artifacts["meta_selection"], index=False)

    chart_paths: list[str] = []
    jepa_chart_paths: list[str] = []
    if not selected.empty:
        model_predictions = predictions.loc[predictions["horizon"].isin(horizons)].copy()
        for horizon in horizons:
            if int(horizon) not in set(selected["horizon"].dropna().astype(int)):
                continue
            chart_paths.append(
                _plot_meta_value_envelope_chart(
                    selected=selected,
                    model_predictions=model_predictions,
                    horizon=int(horizon),
                    chart_dir=chart_dir,
                    return_type=return_type,
                )
            )
        chart_paths.append(
            _plot_latest_meta_forecast(
                selected=selected,
                model_predictions=model_predictions,
                data_bundle=data_bundle,
                horizons=horizons,
                return_type=return_type,
                chart_dir=chart_dir,
            )
        )

    if use_jepa_meta:
        jepa_candidate_panel = candidate_panel.loc[
            candidate_panel["meta_candidate"].astype(str).isin(JEPA_META_CANDIDATES)
        ].copy() if "meta_candidate" in candidate_panel.columns else pd.DataFrame()
        jepa_candidate_panel.to_csv(jepa_artifacts["jepa_candidates"], index=False)
        selected.to_csv(jepa_artifacts["jepa_selected"], index=False)
        metrics.to_csv(jepa_artifacts["jepa_metrics"], index=False)
        selection.to_csv(jepa_artifacts["jepa_selection"], index=False)
        if not selected.empty:
            for horizon in horizons:
                if int(horizon) not in set(selected["horizon"].dropna().astype(int)):
                    continue
                try:
                    jepa_chart_paths.append(
                        _plot_meta_value_envelope_chart(
                            selected=selected,
                            model_predictions=predictions.loc[predictions["horizon"].isin(horizons)].copy(),
                            horizon=int(horizon),
                            chart_dir=chart_dir,
                            return_type=return_type,
                            filename_prefix="walk_forward_jepa_meta_value_envelope",
                            title_label="JEPA-Aware Meta",
                        )
                    )
                except Exception as error:
                    jepa_warnings.append(f"Failed to plot JEPA value envelope for {int(horizon)}d: {error}")
        jepa_chart_paths.extend(
            _plot_jepa_meta_diagnostics(
                feature_frame=feature_frame,
                candidate_panel=candidate_panel,
                selected=selected,
                chart_dir=chart_dir,
            )
        )
        jepa_report = {
            "enabled": True,
            "config": jepa_config.to_dict(),
            "params": jepa_params.to_dict(),
            "warnings": jepa_warnings,
            "diagnostics": jepa_diagnostics,
            "feature_table": JEPA_FEATURE_TABLE,
            "feature_wide_table": JEPA_FEATURE_WIDE_TABLE,
            "long_feature_rows": int(len(jepa_long_features)),
            "wide_feature_rows": int(len(jepa_wide_features)),
            "candidate_names": list(JEPA_META_CANDIDATES),
            "artifacts": {key: str(path.resolve()) for key, path in jepa_artifacts.items()},
            "chart_paths": jepa_chart_paths,
            "selection": selection.to_dict(orient="records"),
            "metrics": metrics.to_dict(orient="records"),
        }
        jepa_artifacts["jepa_report"].write_text(json.dumps(jepa_report, indent=2, default=str) + "\n")
        chart_paths.extend(path for path in jepa_chart_paths if path not in chart_paths)

    report = {
        "horizons": [int(horizon) for horizon in horizons],
        "min_history": int(min_history),
        "selection_window": int(selection_window),
        "candidate_specs": META_CANDIDATE_SPECS,
        "jepa_enabled": bool(use_jepa_meta),
        "jepa_candidates_included": bool(include_jepa_candidates),
        "jepa_warnings": jepa_warnings,
        "jepa_params": jepa_params.to_dict() if use_jepa_meta else {},
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
        "jepa_artifacts": {key: str(path.resolve()) for key, path in jepa_artifacts.items()} if use_jepa_meta else {},
        "chart_paths": chart_paths,
        "selection": selection.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
    }
    artifacts["meta_report"].write_text(json.dumps(report, indent=2, default=str) + "\n")
    return {
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
        "chart_paths": chart_paths,
        "metrics": metrics.to_dict(orient="records"),
        "selection": selection.to_dict(orient="records"),
        "jepa_enabled": bool(use_jepa_meta),
        "jepa_artifacts": {key: str(path.resolve()) for key, path in jepa_artifacts.items()} if use_jepa_meta else {},
        "jepa_warnings": jepa_warnings,
    }


def build_brutal_baseline_artifacts(
    *,
    db_path: str,
    table_name: str,
    output_dir: str | Path,
    horizons: list[int],
    base_config: dict[str, Any],
    full_data: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictions = _load_predictions(db_path, table_name)
    if predictions.empty:
        return {"artifacts": {}, "metrics": [], "bootstrap": []}
    evaluation_config = EvaluationConfig(
        history_min=int(args.baseline_history_min),
        regime_history_min=int(args.baseline_regime_history_min),
        garch_min_obs=int(args.garch_min_obs),
        garch_refit_interval=int(args.garch_refit_interval),
        bootstrap_samples=int(args.bootstrap_samples),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        include_garch=not args.skip_garch_baselines,
        include_kalman_synthesis=not args.skip_kalman_synthesis,
        kalman_min_history=int(args.kalman_min_history),
    )
    return build_brutal_baseline_evaluation(
        predictions=predictions,
        data_bundle=full_data,
        horizons=horizons,
        return_type=str(base_config["data"]["return_type"]),
        output_dir=output_dir,
        config=evaluation_config,
    )


def run_walk_forward(args: argparse.Namespace) -> dict[str, Any]:
    if args.evaluate_only:
        base_config = _load_refit_base_config(args.config_dir)
        full_data = load_forecast_data(base_config)
        horizons = _load_production_horizons(args.config_dir)
        return_type = str(base_config["data"]["return_type"])
        evaluation = build_brutal_baseline_artifacts(
            db_path=args.db_path,
            table_name=args.table,
            output_dir=args.evaluation_dir or args.chart_dir,
            horizons=horizons,
            base_config=base_config,
            full_data=full_data,
            args=args,
        )
        chart_paths: list[str] = []
        indicator_artifacts: list[dict[str, Any]] = []
        meta_artifacts: list[dict[str, Any]] = []
        if not args.skip_charts:
            if not args.skip_meta_synthesis:
                meta_artifacts.append(
                    build_meta_synthesis_artifacts(
                        db_path=args.db_path,
                        table_name=args.table,
                        chart_dir=args.chart_dir,
                        horizons=horizons,
                        data_bundle=full_data,
                        return_type=return_type,
                        min_history=int(args.meta_min_history),
                        selection_window=int(args.meta_selection_window),
                    )
                )
                meta_artifacts.append(
                    build_meta_synthesis_artifacts(
                        db_path=args.db_path,
                        table_name=args.table,
                        chart_dir=_live_forecast_dir(args.config_dir),
                        horizons=horizons,
                        data_bundle=full_data,
                        return_type=return_type,
                        min_history=int(args.meta_min_history),
                        selection_window=int(args.meta_selection_window),
                    )
                )
                for artifact in meta_artifacts:
                    chart_paths.extend(
                        path for path in artifact.get("chart_paths", []) if path not in chart_paths
                    )
            indicator_artifacts.append(
                build_inner_asymmetry_indicator_artifacts(
                    db_path=args.db_path,
                    table_name=args.table,
                    chart_dir=args.chart_dir,
                    horizons=horizons,
                    data_bundle=full_data,
                    return_type=return_type,
                )
            )
            indicator_artifacts.append(
                build_inner_asymmetry_indicator_artifacts(
                    db_path=args.db_path,
                    table_name=args.table,
                    chart_dir=_live_forecast_dir(args.config_dir),
                    horizons=horizons,
                    data_bundle=full_data,
                    return_type=return_type,
                )
            )
            chart_paths.extend(
                artifact["chart_path"]
                for artifact in indicator_artifacts
                if artifact.get("chart_path")
            )
        summary = {
            "config_dir": args.config_dir,
            "db_path": args.db_path,
            "table": args.table,
            "horizons": horizons,
            "evaluation_artifacts": evaluation.get("artifacts", {}),
            "chart_paths": chart_paths,
            "indicator_metrics": indicator_artifacts[-1]["metrics"] if indicator_artifacts else [],
            "meta_artifacts": meta_artifacts[-1].get("artifacts", {}) if meta_artifacts else {},
            "jepa_meta_artifacts": meta_artifacts[-1].get("jepa_artifacts", {}) if meta_artifacts else {},
            "jepa_meta_warnings": meta_artifacts[-1].get("jepa_warnings", []) if meta_artifacts else [],
            "meta_selection": meta_artifacts[-1].get("selection", []) if meta_artifacts else [],
        }
        if getattr(args, "emit_summary", True):
            print(json.dumps(summary, indent=2, default=str))
        return summary

    dates, base_config, full_data, horizons, required_rows = _discover_as_of_dates(
        config_dir=args.config_dir,
        start=args.start,
        end=args.end,
        limit=args.limit,
        min_history_rows=args.min_history_rows,
    )
    if not dates:
        raise ValueError("No as-of dates matched the requested range.")

    reuse_cached_feature_submodels = bool(getattr(args, "reuse_cached_feature_submodels", False))
    if reuse_cached_feature_submodels and not bool(args.rebuild_feature_submodels):
        raise ValueError("--reuse-cached-feature-submodels requires --rebuild-feature-submodels.")

    plan = {
        "config_dir": args.config_dir,
        "db_path": args.db_path,
        "table": args.table,
        "required_rows": required_rows,
        "total_as_of_dates": len(dates),
        "first_as_of": dates[0].isoformat(),
        "last_as_of": dates[-1].isoformat(),
        "horizons": horizons,
        "rebuild_feature_submodels": args.rebuild_feature_submodels,
        "reuse_cached_feature_submodels": reuse_cached_feature_submodels,
        "live_forecast_chart_dir": str(_live_forecast_dir(args.config_dir)),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": plan}, indent=2))
        return {**plan, "dry_run": True}

    completed = _completed_as_ofs(args.db_path, args.table, horizons)
    saved_rows = 0
    failed_days = 0
    skipped_days = 0

    for index, as_of in enumerate(dates, start=1):
        quiet = bool(getattr(args, "quiet", False))
        if not args.force and as_of in completed:
            skipped_days += 1
            if not quiet:
                print(f"[{index}/{len(dates)}] {as_of.date()} already complete; skipping.")
            continue

        started_at = _utc_now()
        if not quiet:
            print(f"[{index}/{len(dates)}] running main.py production cycle as of {as_of.date()}...")
        try:
            cycle_summary = _run_main_cycle_for_as_of(
                config_dir=args.config_dir,
                as_of=as_of,
                save_artifacts=args.save_artifacts,
                rebuild_feature_submodels=args.rebuild_feature_submodels,
                reuse_cached_feature_submodels=reuse_cached_feature_submodels,
                base_config=base_config,
            )
            cycle_summary["as_of_row_count"] = int(
                (full_data.close.index <= as_of).sum()
            )
            rows = _rows_from_cycle_summary(
                cycle_summary=cycle_summary,
                base_config=base_config,
                full_close=full_data.close,
                config_dir=args.config_dir,
            )
            saved_rows += _save_prediction_rows(args.db_path, args.table, rows)
            if not quiet:
                print(f"[{index}/{len(dates)}] saved {len(rows)} horizon predictions.")
        except Exception as error:
            failed_days += 1
            if not quiet:
                print(f"[{index}/{len(dates)}] failed for {as_of.date()}: {error}")
            _record_failure(
                db_path=args.db_path,
                table_name=args.table,
                as_of=as_of,
                started_at_utc=started_at,
                config_dir=args.config_dir,
                error=error,
            )
            if args.stop_on_error:
                raise

    chart_paths: list[str] = []
    indicator_artifacts: list[dict[str, Any]] = []
    meta_artifacts: list[dict[str, Any]] = []
    evaluation_artifacts: dict[str, Any] = {}
    if not args.skip_charts:
        return_type = str(base_config["data"]["return_type"])
        chart_paths = build_envelope_charts(
            db_path=args.db_path,
            table_name=args.table,
            chart_dir=args.chart_dir,
            horizons=horizons,
            return_type=return_type,
            data_bundle=full_data,
        )
        live_chart_paths = build_envelope_charts(
            db_path=args.db_path,
            table_name=args.table,
            chart_dir=_live_forecast_dir(args.config_dir),
            horizons=horizons,
            return_type=return_type,
            data_bundle=full_data,
        )
        chart_paths.extend(path for path in live_chart_paths if path not in chart_paths)
        if not args.skip_meta_synthesis:
            meta_artifacts.append(
                build_meta_synthesis_artifacts(
                    db_path=args.db_path,
                    table_name=args.table,
                    chart_dir=args.chart_dir,
                    horizons=horizons,
                    data_bundle=full_data,
                    return_type=return_type,
                    min_history=int(args.meta_min_history),
                    selection_window=int(args.meta_selection_window),
                )
            )
            meta_artifacts.append(
                build_meta_synthesis_artifacts(
                    db_path=args.db_path,
                    table_name=args.table,
                    chart_dir=_live_forecast_dir(args.config_dir),
                    horizons=horizons,
                    data_bundle=full_data,
                    return_type=return_type,
                    min_history=int(args.meta_min_history),
                    selection_window=int(args.meta_selection_window),
                )
            )
            for artifact in meta_artifacts:
                chart_paths.extend(
                    path for path in artifact.get("chart_paths", []) if path not in chart_paths
                )
        indicator_artifacts.append(
            build_inner_asymmetry_indicator_artifacts(
                db_path=args.db_path,
                table_name=args.table,
                chart_dir=args.chart_dir,
                horizons=horizons,
                data_bundle=full_data,
                return_type=return_type,
            )
        )
        indicator_artifacts.append(
            build_inner_asymmetry_indicator_artifacts(
                db_path=args.db_path,
                table_name=args.table,
                chart_dir=_live_forecast_dir(args.config_dir),
                horizons=horizons,
                data_bundle=full_data,
                return_type=return_type,
            )
        )
        for artifact in indicator_artifacts:
            chart_path = artifact.get("chart_path")
            if chart_path and chart_path not in chart_paths:
                chart_paths.append(chart_path)

    if not args.skip_evaluation:
        evaluation = build_brutal_baseline_artifacts(
            db_path=args.db_path,
            table_name=args.table,
            output_dir=args.evaluation_dir or args.chart_dir,
            horizons=horizons,
            base_config=base_config,
            full_data=full_data,
            args=args,
        )
        evaluation_artifacts = evaluation.get("artifacts", {})

    summary = {
        **plan,
        "saved_rows": saved_rows,
        "skipped_days": skipped_days,
        "failed_days": failed_days,
        "chart_paths": chart_paths,
        "indicator_metrics": indicator_artifacts[-1]["metrics"] if indicator_artifacts else [],
        "meta_artifacts": meta_artifacts[-1].get("artifacts", {}) if meta_artifacts else {},
        "jepa_meta_artifacts": meta_artifacts[-1].get("jepa_artifacts", {}) if meta_artifacts else {},
        "jepa_meta_warnings": meta_artifacts[-1].get("jepa_warnings", []) if meta_artifacts else [],
        "meta_selection": meta_artifacts[-1].get("selection", []) if meta_artifacts else [],
        "evaluation_artifacts": evaluation_artifacts,
    }
    if getattr(args, "emit_summary", True):
        print(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main.py production forecast path in a historical walk-forward loop, "
            "save quantile predictions to DuckDB, and render return-envelope charts."
        )
    )
    parser.add_argument(
        "--config-dir",
        default=str(FINAL_CONFIG_DIR),
        help="Directory containing final refit/fusion/role/specialist config files.",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="DuckDB path where walk-forward predictions will be stored.",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help="Prediction table name.",
    )
    parser.add_argument(
        "--chart-dir",
        default=DEFAULT_CHART_DIR,
        help="Directory for the configured horizon envelope charts.",
    )
    parser.add_argument("--start", default=None, help="Optional first as-of timestamp/date.")
    parser.add_argument("--end", default=None, help="Optional final as-of timestamp/date.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of as-of dates to process, useful for smoke tests.",
    )
    parser.add_argument(
        "--min-history-rows",
        type=int,
        default=None,
        help="Override the minimum aligned history rows required before the walk-forward starts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute dates already present in the prediction table.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Allow each historical run to overwrite the normal production artifacts.",
    )
    feature_table_group = parser.add_mutually_exclusive_group()
    feature_table_group.add_argument(
        "--reuse-feature-tables",
        dest="rebuild_feature_submodels",
        action="store_false",
        help=(
            "Reuse existing category/TA feature tables, including liquidity fair value and PCA latents. "
            "This is the default."
        ),
    )
    feature_table_group.add_argument(
        "--rebuild-feature-submodels",
        dest="rebuild_feature_submodels",
        action="store_true",
        help=(
            "Rebuild category/TA feature tables for each historical as-of date, including "
            "liquidity fair value and PCA latents."
        ),
    )
    parser.set_defaults(rebuild_feature_submodels=False)
    parser.add_argument(
        "--reuse-cached-feature-submodels",
        action="store_true",
        help=(
            "With --rebuild-feature-submodels, create per-as-of temporary category tables by copying "
            "already-stored feature rows through each as-of date instead of recomputing TA, PCA, "
            "and liquidity fair value submodels."
        ),
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Do not render the envelope charts after the database update.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Do not build brutal-baseline, non-overlap, calibration, and trading-utility artifacts.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Build evaluation artifacts from the existing prediction table without running historical forecasts.",
    )
    parser.add_argument(
        "--evaluation-dir",
        default=None,
        help="Directory for brutal-baseline evaluation artifacts. Defaults to --chart-dir.",
    )
    parser.add_argument(
        "--skip-garch-baselines",
        action="store_true",
        help="Skip GARCH/EGARCH volatility-cone baselines.",
    )
    parser.add_argument(
        "--skip-kalman-synthesis",
        action="store_true",
        help="Skip Kalman fair-value synthesis variants during brutal-baseline evaluation.",
    )
    parser.add_argument(
        "--skip-meta-synthesis",
        action="store_true",
        help="Skip the constrained meta-synthesis layer and meta forecast plots.",
    )
    parser.add_argument(
        "--meta-min-history",
        type=int,
        default=META_DEFAULT_MIN_HISTORY,
        help="Minimum prior realized forecasts required before selecting a meta blend candidate.",
    )
    parser.add_argument(
        "--meta-selection-window",
        type=int,
        default=META_DEFAULT_SELECTION_WINDOW,
        help="Number of recent prior as-of rows used to select the best meta blend candidate per horizon.",
    )
    parser.add_argument(
        "--kalman-min-history",
        type=int,
        default=240,
        help="Minimum filtered close observations before Kalman synthesis is allowed to alter model quantiles.",
    )
    parser.add_argument(
        "--baseline-history-min",
        type=int,
        default=180,
        help="Minimum known historical observations required before empirical baseline quantiles are used.",
    )
    parser.add_argument(
        "--baseline-regime-history-min",
        type=int,
        default=60,
        help="Minimum same-regime observations required before regime-conditioned quantiles are used.",
    )
    parser.add_argument(
        "--garch-min-obs",
        type=int,
        default=500,
        help="Minimum observations required to fit GARCH/EGARCH baselines.",
    )
    parser.add_argument(
        "--garch-refit-interval",
        type=int,
        default=30,
        help="Refit GARCH/EGARCH baselines every N as-of dates.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Block bootstrap samples for model-vs-baseline confidence intervals.",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=5.0,
        help="Per-side trading fee in basis points for strategy utility tests.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Per-side slippage in basis points for strategy utility tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned as-of date range without training models.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed historical run instead of logging and continuing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_walk_forward(parse_args())
