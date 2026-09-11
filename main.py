from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from env_loader import load_project_env

load_project_env()

from database_interraction import database_has_tables, duckdb_connection, quote_identifier, table_exists
from compute_data.data_classes import (
    DEFAULT_DATA_CLASSES_DB,
    DEFAULT_LIQUIDATION_SIGNAL_PATH,
    DEFAULT_TA_DB,
    EdgesSet,
    EnvironmentSet,
    LiquidationSet,
    MovementSet,
    StructureSet,
)
from compute_data.ta.dataset_ta import build_1d_ta_db
from get_data.macro.fred import (
    DB_PATH as DEFAULT_MACRO_DB_PATH,
    DEFAULT_OVERLAP_POINTS as DEFAULT_MACRO_OVERLAP_POINTS,
    TABLE_NAME as DEFAULT_MACRO_TABLE,
    update_fred_data,
)
from get_data.macro.manual import (
    MANUAL_DIR as DEFAULT_MANUAL_DIR,
    update_manual_macro_data,
)
from get_data.oc.onchain import (
    DB_PATH as DEFAULT_ONCHAIN_DB_PATH,
    DEFAULT_OVERLAP_POINTS,
    fetch_all as fetch_onchain_history,
    sync_all as sync_onchain_data,
)
from get_data.scraped.checkonchain import sync_choppiness_index
from get_data.scraped.liquidations import sync_liquidation_signals
from get_data.ta.price import sync_binance_ohlcv_to_duckdb
from model_assembly.common import FINAL_CONFIG_DIR, run_refit_full_history

DEFAULT_BTC_SYMBOL = (os.getenv("BTC_SYMBOL") or "BTCUSDT").strip()
DEFAULT_BTC_INTERVAL = (os.getenv("BTC_INTERVAL") or "1d").strip()
DEFAULT_PRICE_DB_PATH = (os.getenv("BTC_PRICE_DB_PATH") or "database/ohlcv.duckdb").strip()
DEFAULT_PRICE_TABLE = (os.getenv("BTC_PRICE_TABLE") or "ohlcv").strip()
DEFAULT_CATEGORY_DB_PATH = (
    os.getenv("CATEGORY_DB_PATH") or DEFAULT_DATA_CLASSES_DB
).strip()
DEFAULT_TA_OUTPUT_DB_PATH = (
    os.getenv("TA_OUTPUT_DB_PATH") or DEFAULT_TA_DB
).strip()
DEFAULT_LIQUIDATION_SIGNAL_PATH_ENV = (
    os.getenv("LIQUIDATION_SIGNAL_PATH") or DEFAULT_LIQUIDATION_SIGNAL_PATH
).strip()
DEFAULT_WALK_FORWARD_DB_PATH = (
    os.getenv("WALK_FORWARD_DB_PATH") or "database/walk_forward_predictions.duckdb"
).strip()
DEFAULT_WALK_FORWARD_TABLE = (
    os.getenv("WALK_FORWARD_TABLE") or "walk_forward_predictions"
).strip()
DEFAULT_CLEAN_HISTORY_ROWS = int(os.getenv("CLEAN_HISTORY_ROWS") or "500")
DEFAULT_AGGREGATE_WINDOWS = (30, 90, 180, 365)
LIVE_OUTPUT_REMOVE_PATTERNS = (
    "walk_forward_value_envelope_*.png",
    "walk_forward_meta_value_envelope_*.png",
    "meta_synthesis_*.csv",
    "meta_synthesis_*.json",
    "btc_inner_asymmetry_edge_indicator*",
    "latest_meta_plot.png",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _utc_day(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def _current_utc_day() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize()


def _latest_price_timestamp(
    *,
    symbol: str = DEFAULT_BTC_SYMBOL,
    interval: str = DEFAULT_BTC_INTERVAL,
    price_db_path: str = DEFAULT_PRICE_DB_PATH,
    price_table_name: str = DEFAULT_PRICE_TABLE,
) -> pd.Timestamp | None:
    db_path = Path(price_db_path)
    if not db_path.exists():
        return None
    try:
        with duckdb_connection(str(db_path), read_only=True) as connection:
            if not table_exists(connection, price_table_name):
                return None
            row = connection.execute(
                f"""
                SELECT MAX(open_time)
                FROM {quote_identifier(price_table_name)}
                WHERE symbol = ? AND interval = ?;
                """,
                [symbol.upper(), interval.lower()],
            ).fetchone()
    except Exception:
        return None
    if row is None or row[0] is None:
        return None
    timestamp = pd.Timestamp(row[0])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _data_refresh_gate(
    *,
    symbol: str = DEFAULT_BTC_SYMBOL,
    interval: str = DEFAULT_BTC_INTERVAL,
    price_db_path: str = DEFAULT_PRICE_DB_PATH,
    price_table_name: str = DEFAULT_PRICE_TABLE,
) -> dict[str, Any]:
    latest = _latest_price_timestamp(
        symbol=symbol,
        interval=interval,
        price_db_path=price_db_path,
        price_table_name=price_table_name,
    )
    required_day = _current_utc_day()
    latest_day = _utc_day(latest) if latest is not None else None
    is_fresh = latest_day is not None and latest_day >= required_day
    return {
        "required_latest_day": required_day.date().isoformat(),
        "latest_price_timestamp": latest.isoformat() if latest is not None else None,
        "latest_price_day": latest_day.date().isoformat() if latest_day is not None else None,
        "fresh_enough": bool(is_fresh),
        "action": "skip_data_pull" if is_fresh else "pull_data",
    }


def _project_price(base_price: float, forecast_return: float, return_type: str) -> float | None:
    if not np.isfinite(base_price) or not np.isfinite(forecast_return):
        return None
    if str(return_type) == "log":
        value = base_price * float(np.exp(forecast_return))
    elif forecast_return > -0.99:
        value = base_price * (1.0 + forecast_return)
    else:
        return None
    return float(value) if np.isfinite(value) else None


def _format_float(value: Any, fmt: str, fallback: str = "n/a") -> str:
    try:
        numeric = float(value)
    except Exception:
        return fallback
    if not np.isfinite(numeric):
        return fallback
    return format(numeric, fmt)


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _log_price_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return np.exp(values).replace([np.inf, -np.inf], np.nan)


def _tail_history(series: pd.Series, *, years: int = 5) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return clean
    cutoff = pd.Timestamp(clean.index[-1]) - pd.DateOffset(years=int(years))
    return clean.loc[clean.index >= cutoff]


def _recent_daily_log_trend(series: pd.Series, *, days: int = 180, cap: float = 0.003) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return 0.0
    cutoff = pd.Timestamp(clean.index[-1]) - pd.DateOffset(days=int(days))
    recent = clean.loc[clean.index >= cutoff]
    if len(recent) < 2:
        recent = clean.tail(2)
    elapsed_days = (pd.Timestamp(recent.index[-1]) - pd.Timestamp(recent.index[0])).total_seconds() / 86400.0
    if elapsed_days <= 0.0:
        return 0.0
    trend = float((recent.iloc[-1] - recent.iloc[0]) / elapsed_days)
    if not np.isfinite(trend):
        return 0.0
    return float(np.clip(trend, -abs(cap), abs(cap)))


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    return plt, mdates, FuncFormatter


def _dollar_formatter(value: float, _: int) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"${value / 1000:.0f}k"
    return f"${value:.0f}"


def _write_kalman_scope_path(
    live_root: Path,
    latest_projection: pd.DataFrame,
    state_history: pd.DataFrame | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    if latest_projection.empty:
        return None, {}
    rows = latest_projection.dropna(
        subset=["as_of", "base_close", "kalman_target_timestamp", "kalman_projected_price"]
    )
    if rows.empty:
        return None, {}
    row = rows.sort_values("horizon").iloc[0]
    as_of = pd.Timestamp(row["as_of"])
    target_timestamp = pd.Timestamp(row["kalman_target_timestamp"])
    base_close = _float_or_none(row.get("base_close"))
    projected_price = _float_or_none(row.get("kalman_projected_price"))
    projected_return = _float_or_none(row.get("kalman_projected_return"))
    scope_days = _float_or_none(row.get("kalman_scope_days"))
    if base_close is None or projected_price is None or target_timestamp <= as_of:
        return None, {}

    future_dates = pd.date_range(as_of, target_timestamp, freq="D")
    if len(future_dates) < 2:
        return None, {}
    progress = np.linspace(0.0, 1.0, len(future_dates))
    if projected_return is None and base_close > 0.0 and projected_price > 0.0:
        projected_return = float(np.log(projected_price / base_close))
    projected_return = projected_return or 0.0
    drift_return = _float_or_none(row.get("kalman_drift_component_return")) or 0.0
    mean_reversion_return = _float_or_none(row.get("kalman_mean_reversion_component_return")) or 0.0
    macro_return = _float_or_none(row.get("kalman_macro_component_return")) or 0.0
    onchain_return = _float_or_none(row.get("kalman_onchain_component_return")) or 0.0
    state_return = drift_return + mean_reversion_return
    pressure_return = macro_return + onchain_return

    def _price_path(total_return: float) -> np.ndarray:
        if base_close <= 0.0 or not np.isfinite(total_return):
            return np.full(len(future_dates), np.nan)
        return base_close * np.exp(total_return * progress)

    frame = pd.DataFrame(
        {
            "timestamp": future_dates,
            "kalman_scope_path_price": _price_path(float(projected_return)),
            "kalman_state_path_price": _price_path(float(state_return)),
            "kalman_drift_path_price": _price_path(float(drift_return)),
            "kalman_mean_reversion_path_price": _price_path(float(mean_reversion_return)),
            "kalman_macro_projection_path_price": _price_path(float(macro_return)),
            "kalman_onchain_projection_path_price": _price_path(float(onchain_return)),
            "kalman_pressure_projection_path_price": _price_path(float(pressure_return)),
            "kalman_total_cumulative_return": float(projected_return) * progress,
            "kalman_state_cumulative_return": float(state_return) * progress,
            "kalman_drift_cumulative_return": float(drift_return) * progress,
            "kalman_mean_reversion_cumulative_return": float(mean_reversion_return) * progress,
            "kalman_macro_cumulative_return": float(macro_return) * progress,
            "kalman_onchain_cumulative_return": float(onchain_return) * progress,
            "kalman_pressure_cumulative_return": float(pressure_return) * progress,
        }
    )

    anchor_projection: dict[str, dict[str, Any]] = {}
    if state_history is not None and not state_history.empty:
        history = state_history.sort_index()
        history = history.loc[history.index <= as_of]
        anchor_columns = {
            "kalman_fair_value_projected_price": ("kalman_fair_value_price", "fair_value_drift", "Kalman fair value"),
            "level_ensemble_projected_price": ("fundamental_level_anchor_log", None, "Level ensemble"),
            "liquidity_projected_price": ("liquidity_anchor_log", None, "Liquidity fair value"),
            "energy_projected_price": ("energy_anchor_log", None, "Energy value"),
            "metcalfe_projected_price": ("metcalfe_anchor_log", None, "Metcalfe price"),
        }
        elapsed = np.array(
            [(pd.Timestamp(timestamp) - as_of).total_seconds() / 86400.0 for timestamp in future_dates],
            dtype=float,
        )
        for output_column, (source_column, drift_column, label) in anchor_columns.items():
            if source_column not in history.columns:
                continue
            if source_column == "kalman_fair_value_price":
                values = pd.to_numeric(history[source_column], errors="coerce")
                values = np.log(values.where(values > 0.0))
            else:
                values = pd.to_numeric(history[source_column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            latest_log = float(values.iloc[-1])
            trend = 0.0
            if drift_column and drift_column in history.columns:
                drift_values = pd.to_numeric(history[drift_column], errors="coerce").dropna()
                if not drift_values.empty:
                    trend = _float_or_none(drift_values.iloc[-1]) or 0.0
                    trend = float(np.clip(trend, -0.003, 0.003))
            else:
                trend = _recent_daily_log_trend(values)
            projected_logs = latest_log + trend * elapsed
            projected_values = pd.Series(np.exp(projected_logs), index=future_dates).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            frame[output_column] = projected_values.to_numpy(dtype=float)
            anchor_projection[output_column] = {
                "label": label,
                "latest_price": float(np.exp(latest_log)),
                "projected_price": _float_or_none(projected_values.iloc[-1]),
                "daily_log_trend": trend,
            }

    path = live_root / "kalman_scope_projection.csv"
    frame.to_csv(path, index=False)
    return path, {
        "as_of": as_of.isoformat(),
        "target_timestamp": target_timestamp.isoformat(),
        "scope_days": scope_days,
        "base_close": base_close,
        "projected_price": projected_price,
        "projected_return": projected_return,
        "state_projected_price": _float_or_none(frame["kalman_state_path_price"].iloc[-1]),
        "state_return": state_return,
        "drift_component_return": drift_return,
        "mean_reversion_component_return": mean_reversion_return,
        "macro_component_return": macro_return,
        "onchain_component_return": onchain_return,
        "projection_pressure_return": pressure_return,
        "anchor_projection": anchor_projection,
        "path_rows": int(len(frame)),
    }


def _write_kalman_projection_chart(
    *,
    live_root: Path,
    close: pd.Series,
    latest_projection: pd.DataFrame,
    state_history: pd.DataFrame,
    scope_path: Path | None,
) -> Path | None:
    if latest_projection.empty or scope_path is None or not scope_path.exists():
        return None
    projection_path = pd.read_csv(scope_path, parse_dates=["timestamp"])
    if projection_path.empty:
        return None

    plt, mdates, FuncFormatter = _setup_matplotlib()
    history_close = _tail_history(close, years=5)
    if history_close.empty:
        return None
    history_state = (
        state_history.loc[state_history.index >= history_close.index[0]].copy()
        if not state_history.empty
        else pd.DataFrame()
    )
    points = latest_projection.dropna(subset=["target_timestamp", "q50_price"]).copy()
    points["target_timestamp"] = pd.to_datetime(points["target_timestamp"], errors="coerce")
    scope_info = latest_projection.dropna(
        subset=["kalman_target_timestamp", "kalman_projected_price"]
    ).sort_values("horizon")
    scope_row = scope_info.iloc[0] if not scope_info.empty else pd.Series(dtype=object)
    scope_days = _float_or_none(scope_row.get("kalman_scope_days"))
    target_timestamp = (
        pd.Timestamp(scope_row.get("kalman_target_timestamp"))
        if pd.notna(scope_row.get("kalman_target_timestamp"))
        else pd.NaT
    )
    projected_price = _float_or_none(scope_row.get("kalman_projected_price"))
    projected_return = _float_or_none(scope_row.get("kalman_projected_return"))
    macro_return = _float_or_none(scope_row.get("kalman_macro_component_return"))
    onchain_return = _float_or_none(scope_row.get("kalman_onchain_component_return"))

    fig, (ax, component_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 8.4),
        dpi=150,
        sharex=False,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.22},
    )
    ax.plot(history_close.index, history_close.values, color="#111827", linewidth=1.7, label="BTC close")

    historical_anchor_columns = [
        ("kalman_fair_value_price", "Kalman fair value", "#0ea5e9"),
        ("fundamental_level_anchor_log", "Level ensemble", "#10b981"),
        ("liquidity_anchor_log", "Liquidity fair value", "#64748b"),
        ("energy_anchor_log", "Energy value", "#f59e0b"),
        ("metcalfe_anchor_log", "Metcalfe price", "#8b5cf6"),
    ]
    for column, label, color in historical_anchor_columns:
        if history_state.empty or column not in history_state.columns:
            continue
        series = (
            pd.to_numeric(history_state[column], errors="coerce")
            if column == "kalman_fair_value_price"
            else _log_price_column(history_state, column)
        )
        series = series.reindex(history_close.index).ffill().dropna()
        if not series.empty:
            ax.plot(series.index, series.values, color=color, linewidth=1.05, alpha=0.82, label=label)

    projected_anchor_columns = [
        ("kalman_fair_value_projected_price", "Fair value projected", "#0ea5e9"),
        ("level_ensemble_projected_price", "Level ensemble projected", "#10b981"),
        ("liquidity_projected_price", "Liquidity projected", "#64748b"),
        ("energy_projected_price", "Energy projected", "#f59e0b"),
        ("metcalfe_projected_price", "Metcalfe projected", "#8b5cf6"),
    ]
    for column, label, color in projected_anchor_columns:
        if column not in projection_path.columns:
            continue
        values = pd.to_numeric(projection_path[column], errors="coerce")
        if values.dropna().empty:
            continue
        ax.plot(
            projection_path["timestamp"],
            values,
            color=color,
            linewidth=1.25,
            linestyle="--",
            alpha=0.88,
            label=label,
        )

    future_price_columns = [
        ("kalman_scope_path_price", "Kalman total projection", "#dc2626", "--", 2.2),
        ("kalman_state_path_price", "State-only projection", "#374151", ":", 1.7),
        ("kalman_macro_projection_path_price", "Macro-only projection", "#2563eb", (0, (1, 1)), 1.25),
        ("kalman_onchain_projection_path_price", "On-chain-only projection", "#7c3aed", (0, (1, 1)), 1.25),
        ("kalman_pressure_projection_path_price", "Macro+on-chain projection", "#0891b2", "-.", 1.25),
    ]
    for column, label, color, linestyle, linewidth in future_price_columns:
        if column not in projection_path.columns:
            continue
        values = pd.to_numeric(projection_path[column], errors="coerce")
        if values.dropna().empty:
            continue
        ax.plot(
            projection_path["timestamp"],
            values,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.92,
            label=label,
        )

    if not points.empty:
        ax.scatter(
            points["target_timestamp"],
            points["q50_price"],
            s=28,
            color="#f59e0b",
            edgecolor="#78350f",
            linewidth=0.5,
            zorder=5,
            label="Model q50 horizons",
        )
    latest_timestamp = pd.Timestamp(history_close.index[-1])
    ax.axvline(latest_timestamp, color="#6b7280", linewidth=1.0, linestyle=":")
    if pd.notna(target_timestamp):
        ax.axvspan(latest_timestamp, target_timestamp, color="#fee2e2", alpha=0.28)
    latest_state_row = (
        state_history.loc[state_history.index <= latest_timestamp].tail(1)
        if not state_history.empty
        else pd.DataFrame()
    )
    latest_high_vol = (
        _float_or_none(latest_state_row.iloc[0].get("kalman_high_vol_signal"))
        if not latest_state_row.empty
        else None
    )
    latest_tail_flare = (
        _float_or_none(latest_state_row.iloc[0].get("tail_flare_score"))
        if not latest_state_row.empty
        else None
    )
    high_vol_fires = latest_high_vol is not None and latest_high_vol >= 0.50
    tail_flare_fires = latest_tail_flare is not None and latest_tail_flare >= 1.00
    if pd.notna(target_timestamp) and (high_vol_fires or tail_flare_fires):
        active_parts = [
            f"high-vol {latest_high_vol:.2f}" if high_vol_fires else "",
            f"tail flare {latest_tail_flare:.2f}" if tail_flare_fires else "",
        ]
        active_parts = [value for value in active_parts if value]
        ax.axvspan(
            latest_timestamp,
            target_timestamp,
            color="#f97316" if high_vol_fires else "#7c3aed",
            alpha=0.10,
            label="Tail/high-vol overlay",
        )
        ax.annotate(
            "Tail/high-vol active\n" + ", ".join(active_parts),
            xy=(latest_timestamp, float(history_close.iloc[-1])),
            xytext=(12, -42),
            textcoords="offset points",
            fontsize=8,
            color="#7c2d12" if high_vol_fires else "#4c1d95",
            arrowprops={"arrowstyle": "->", "color": "#9a3412" if high_vol_fires else "#6d28d9", "linewidth": 0.8},
        )
    if pd.notna(target_timestamp) and projected_price is not None:
        ax.annotate(
            f"{target_timestamp.date()}\n{projected_price:,.0f}",
            xy=(target_timestamp, projected_price),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color="#991b1b",
            arrowprops={"arrowstyle": "->", "color": "#991b1b", "linewidth": 0.8},
        )
    title_scope = f"{scope_days:.1f} days" if scope_days is not None else "dynamic scope"
    component_summary = []
    if projected_return is not None:
        component_summary.append(f"total {projected_return * 100.0:.1f}%")
    if macro_return is not None:
        component_summary.append(f"macro {macro_return * 100.0:.1f}%")
    if onchain_return is not None:
        component_summary.append(f"on-chain {onchain_return * 100.0:.1f}%")
    subtitle = f" ({'; '.join(component_summary)})" if component_summary else ""
    ax.set_title(f"Kalman Projection Scope ({title_scope})", loc="left", fontsize=13, weight="bold")
    if subtitle:
        ax.text(
            1.0,
            1.01,
            subtitle.strip(" ()"),
            transform=ax.transAxes,
            fontsize=8,
            color="#4b5563",
            ha="right",
            va="bottom",
        )
    ax.set_ylabel("Price")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper left", ncols=3, frameon=False, fontsize=7)

    component_columns = [
        ("kalman_total_cumulative_return", "Total", "#dc2626", 1.5),
        ("kalman_state_cumulative_return", "State", "#374151", 1.2),
        ("kalman_drift_cumulative_return", "Drift", "#0ea5e9", 1.0),
        ("kalman_mean_reversion_cumulative_return", "Mean reversion", "#f59e0b", 1.0),
        ("kalman_macro_cumulative_return", "Macro", "#2563eb", 1.2),
        ("kalman_onchain_cumulative_return", "On-chain", "#7c3aed", 1.2),
    ]
    for column, label, color, linewidth in component_columns:
        if column not in projection_path.columns:
            continue
        values = pd.to_numeric(projection_path[column], errors="coerce")
        if values.dropna().empty:
            continue
        component_ax.plot(
            projection_path["timestamp"],
            values,
            color=color,
            linewidth=linewidth,
            label=label,
        )
    component_ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    component_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100.0:.0f}%"))
    component_ax.set_ylabel("Return")
    component_ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
    component_ax.legend(loc="upper left", ncols=6, frameon=False, fontsize=7)
    axis_end = projection_path["timestamp"].max()
    if not points.empty and "target_timestamp" in points.columns:
        point_timestamps = pd.to_datetime(points["target_timestamp"], errors="coerce").dropna()
        if not point_timestamps.empty:
            axis_end = max(pd.Timestamp(axis_end), pd.Timestamp(point_timestamps.max()))
    history_locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(history_locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(history_locator))
    ax.set_xlim(pd.Timestamp(history_close.index[0]), pd.Timestamp(axis_end))
    component_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    component_ax.xaxis.set_major_locator(component_locator)
    component_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(component_locator))
    component_ax.set_xlim(pd.Timestamp(projection_path["timestamp"].min()), pd.Timestamp(axis_end))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.08, hspace=0.24)
    path = live_root / "kalman_projection_scope.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_kalman_history_charts(
    *,
    live_root: Path,
    close: pd.Series,
    state_history: pd.DataFrame,
) -> dict[str, str]:
    if state_history.empty:
        return {}

    plt, mdates, FuncFormatter = _setup_matplotlib()
    charts: dict[str, str] = {}
    history_close = _tail_history(close, years=5)
    history_state = state_history.loc[state_history.index >= history_close.index[0]].copy() if not history_close.empty else state_history.copy()

    if not history_close.empty and (
        "kalman_high_vol_signal" in history_state.columns or "tail_flare_score" in history_state.columns
    ):
        aligned_state = history_state.reindex(history_close.index).ffill()
        high_vol = (
            pd.to_numeric(aligned_state["kalman_high_vol_signal"], errors="coerce").clip(0.0, 1.0)
            if "kalman_high_vol_signal" in aligned_state.columns
            else pd.Series(np.nan, index=history_close.index)
        )
        tail_flare = (
            pd.to_numeric(aligned_state["tail_flare_score"], errors="coerce").clip(0.0, 5.0)
            if "tail_flare_score" in aligned_state.columns
            else pd.Series(np.nan, index=history_close.index)
        )
        high_vol_mask = high_vol.fillna(0.0) >= 0.50
        tail_flare_mask = tail_flare.fillna(0.0) >= 1.00

        def _shade_active_windows(
            axis: Any,
            mask: pd.Series,
            *,
            color: str,
            alpha: float,
            label: str,
        ) -> None:
            if mask.empty or not bool(mask.any()):
                return
            transitions = mask.ne(mask.shift(fill_value=False)).cumsum()
            first_label = True
            for _, active_run in mask.loc[mask].groupby(transitions.loc[mask]):
                start = pd.Timestamp(active_run.index[0])
                end = pd.Timestamp(active_run.index[-1]) + pd.Timedelta(days=1)
                axis.axvspan(
                    start,
                    end,
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                    label=label if first_label else None,
                    zorder=0,
                )
                first_label = False

        fig, ax_price = plt.subplots(figsize=(12, 6.8), dpi=150)
        _shade_active_windows(
            ax_price,
            high_vol_mask,
            color="#f97316",
            alpha=0.13,
            label="High-vol active",
        )
        _shade_active_windows(
            ax_price,
            tail_flare_mask,
            color="#7c3aed",
            alpha=0.10,
            label="Tail-flare active",
        )
        ax_price.plot(history_close.index, history_close.values, color="#111827", linewidth=1.55, label="BTC close")
        ax_price.set_title("Tail-Flare And High-Vol Overlay On BTC Price", loc="left", fontsize=13, weight="bold")
        ax_price.set_ylabel("BTC price")
        ax_price.set_yscale("log")
        ax_price.yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
        ax_price.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
        ax_price.grid(True, which="major", color="#e5e7eb", linewidth=0.8)

        ax_signal = ax_price.twinx()
        if high_vol.notna().any():
            ax_signal.plot(
                high_vol.index,
                high_vol.values,
                color="#d97706",
                linewidth=1.15,
                alpha=0.95,
                label="High-vol signal",
            )
        if tail_flare.notna().any():
            ax_signal.plot(
                tail_flare.index,
                (tail_flare / 3.0).clip(0.0, 1.67).values,
                color="#7c3aed",
                linewidth=1.0,
                alpha=0.82,
                label="Tail flare score / 3",
            )
        ax_signal.axhline(0.5, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax_signal.set_ylim(-0.05, 1.70)
        ax_signal.set_ylabel("Signal")
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_price.xaxis.get_major_locator()))
        price_lines, price_labels = ax_price.get_legend_handles_labels()
        signal_lines, signal_labels = ax_signal.get_legend_handles_labels()
        ax_price.legend(price_lines + signal_lines, price_labels + signal_labels, loc="upper left", ncols=3, frameon=False, fontsize=8)
        fig.tight_layout()
        path = live_root / "tail_flare_highvol_price_history.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        charts["tail_flare_highvol_price_history"] = str(path.resolve())

    if not history_close.empty:
        fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)
        ax.plot(history_close.index, history_close.values, color="#111827", linewidth=1.6, label="BTC close")
        anchor_columns = [
            ("kalman_fair_value_price", "Kalman fair value", "#0ea5e9"),
            ("fundamental_level_anchor_log", "Level ensemble", "#10b981"),
            ("liquidity_anchor_log", "Liquidity fair value", "#64748b"),
            ("energy_anchor_log", "Energy value", "#f59e0b"),
            ("metcalfe_anchor_log", "Metcalfe price", "#8b5cf6"),
        ]
        for column, label, color in anchor_columns:
            series = (
                pd.to_numeric(history_state[column], errors="coerce")
                if column == "kalman_fair_value_price" and column in history_state.columns
                else _log_price_column(history_state, column)
            )
            series = series.reindex(history_close.index).ffill().dropna()
            if not series.empty:
                ax.plot(series.index, series.values, linewidth=1.15, label=label, color=color, alpha=0.90)
        ax.set_title("Kalman Fair Value And Historical Anchors", loc="left", fontsize=13, weight="bold")
        ax.set_ylabel("Price")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
        ax.legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
        fig.tight_layout()
        path = live_root / "kalman_fair_value_history.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        charts["fair_value_history"] = str(path.resolve())

    chart_state = history_state.tail(5 * 365)
    if not chart_state.empty:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=150, sharex=True)
        if not history_close.empty:
            axes[0].plot(history_close.index, history_close.values, color="#111827", linewidth=1.25)
            axes[0].set_yscale("log")
            axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
        axes[0].set_title("Macro Lead And Scope Drivers", loc="left", fontsize=13, weight="bold")
        axes[0].set_ylabel("BTC")

        score_columns = [
            ("kalman_projection_pressure", "Projection pressure", "#111827"),
            ("macro_forward_score", "Macro forward", "#dc2626"),
            ("macro_impact_score", "Lead impact", "#2563eb"),
            ("onchain_adjust", "On-chain adjust", "#8b5cf6"),
            ("liquidity_macro_score", "Liquidity", "#059669"),
            ("cycle_macro_score", "Cycle", "#d97706"),
            ("financial_conditions_macro_score", "FCI", "#7c3aed"),
        ]
        for column, label, color in score_columns:
            if column in chart_state.columns:
                series = pd.to_numeric(chart_state[column], errors="coerce").dropna()
                if not series.empty:
                    axes[1].plot(series.index, series.values, label=label, color=color, linewidth=1.1)
        axes[1].axhline(0.0, color="#9ca3af", linewidth=0.8)
        axes[1].set_ylabel("Score")
        axes[1].legend(loc="upper left", ncols=3, frameon=False, fontsize=7)

        lead_columns = [
            ("kalman_scope_days", "Combined Kalman scope", "#dc2626"),
            ("macro_scope_days", "Macro scope", "#2563eb"),
            ("liquidity_lead_days", "Liquidity lead", "#059669"),
            ("cycle_lead_days", "Business cycle lead", "#d97706"),
            ("financial_conditions_lead_days", "FCI lead", "#7c3aed"),
        ]
        for column, label, color in lead_columns:
            if column in chart_state.columns:
                series = pd.to_numeric(chart_state[column], errors="coerce").dropna()
                if not series.empty:
                    axes[2].plot(series.index, series.values, label=label, color=color, linewidth=1.1)
        axes[2].set_ylabel("Days")
        axes[2].legend(loc="upper left", ncols=2, frameon=False, fontsize=7)
        axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[2].xaxis.get_major_locator()))
        for axis in axes:
            axis.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
        fig.tight_layout()
        path = live_root / "kalman_macro_leads_history.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        charts["macro_leads_history"] = str(path.resolve())

        holder_signal_specs = [
            ("lth", "On-Chain LTH Average And Signalboost", "#2563eb", "#dc2626"),
            ("sth", "On-Chain STH Average And Signalboost", "#059669", "#d97706"),
        ]
        for holder, title, average_color, boost_color in holder_signal_specs:
            average_column = f"{holder}_average"
            boost_column = f"{holder}_signalboost"
            if average_column not in chart_state.columns or boost_column not in chart_state.columns:
                continue
            signal_frame = chart_state[[average_column, boost_column]].apply(pd.to_numeric, errors="coerce")
            signal_frame = signal_frame.dropna(how="all")
            if signal_frame.empty:
                continue

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 7.8),
                dpi=150,
                sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.25]},
            )
            if not history_close.empty:
                axes[0].plot(history_close.index, history_close.values, color="#111827", linewidth=1.25)
                axes[0].set_yscale("log")
                axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
            axes[0].set_title(title, loc="left", fontsize=13, weight="bold")
            axes[0].set_ylabel("BTC")

            axes[1].plot(
                signal_frame.index,
                signal_frame[average_column],
                color=average_color,
                linewidth=1.25,
                label="Average",
            )
            axes[1].plot(
                signal_frame.index,
                signal_frame[boost_column],
                color=boost_color,
                linewidth=1.15,
                alpha=0.9,
                label="Signalboost",
            )
            axes[1].axhline(0.0, color="#9ca3af", linewidth=0.8)
            axes[1].set_ylabel("Signal")
            axes[1].legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
            axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
            axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
            for axis in axes:
                axis.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
            fig.tight_layout()
            path = live_root / f"onchain_{holder}_average_signalboost_history.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            charts[f"onchain_{holder}_average_signalboost_history"] = str(path.resolve())

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=150, sharex=True)
        if not history_close.empty:
            axes[0].plot(history_close.index, history_close.values, color="#111827", linewidth=1.25)
            axes[0].set_yscale("log")
            axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
        axes[0].set_title("Signalboost And On-Chain Attention", loc="left", fontsize=13, weight="bold")
        axes[0].set_ylabel("BTC")
        coverage_columns = [
            column
            for column in ("lth_average_available", "sth_average_available")
            if column in chart_state.columns
        ]
        coverage_start = None
        if coverage_columns:
            coverage = chart_state[coverage_columns].apply(pd.to_numeric, errors="coerce").max(axis=1)
            coverage = coverage.loc[coverage > 0.5]
            if not coverage.empty:
                coverage_start = pd.Timestamp(coverage.index[0])
                for axis in axes:
                    axis.axvline(coverage_start, color="#6b7280", linewidth=0.9, linestyle=":")
                axes[0].annotate(
                    f"LTH/STH averages available\nfrom {coverage_start.date()}",
                    xy=(coverage_start, history_close.reindex([coverage_start], method="nearest").iloc[0] if not history_close.empty else 1.0),
                    xytext=(8, -36),
                    textcoords="offset points",
                    fontsize=8,
                    color="#374151",
                    arrowprops={"arrowstyle": "->", "color": "#6b7280", "linewidth": 0.8},
                )

        gate_columns = [
            ("signalboost_peak_score", "Combined gate", "#dc2626"),
            ("lth_signalboost_peak_score", "LTH gate", "#2563eb"),
            ("sth_signalboost_peak_score", "STH gate", "#059669"),
        ]
        for column, label, color in gate_columns:
            if column in chart_state.columns:
                series = pd.to_numeric(chart_state[column], errors="coerce").dropna()
                if not series.empty:
                    axes[1].plot(series.index, series.values, label=label, color=color, linewidth=1.1)
        axes[1].set_ylabel("Gate")
        axes[1].legend(loc="upper left", ncols=3, frameon=False, fontsize=7)

        attention_columns = [
            ("onchain_reversion_score", "On-chain reversion", "#dc2626"),
            ("lth_signalboost_attention_score", "LTH attention", "#2563eb"),
            ("sth_signalboost_attention_score", "STH attention", "#059669"),
            ("onchain_average_behavior_score", "Average behavior", "#7c3aed"),
        ]
        for column, label, color in attention_columns:
            if column in chart_state.columns:
                series = pd.to_numeric(chart_state[column], errors="coerce").dropna()
                if not series.empty:
                    axes[2].plot(series.index, series.values, label=label, color=color, linewidth=1.1)
        axes[2].axhline(0.0, color="#9ca3af", linewidth=0.8)
        axes[2].set_ylabel("Attention")
        axes[2].legend(loc="upper left", ncols=2, frameon=False, fontsize=7)
        axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[2].xaxis.get_major_locator()))
        for axis in axes:
            axis.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
        fig.tight_layout()
        path = live_root / "kalman_signalboost_attention_history.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        charts["signalboost_attention_history"] = str(path.resolve())

    return charts


def _compact_live_forecast_rows(
    live_forecast: dict[str, list[dict[str, Any]]],
    *,
    base_close: float,
    return_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_key, forecast_rows in sorted(live_forecast.items(), key=lambda item: int(item[0])):
        if not forecast_rows:
            continue
        row = forecast_rows[0]
        as_of = pd.Timestamp(row["timestamp"])
        horizon = int(horizon_key)
        target_timestamp = as_of + pd.Timedelta(days=horizon)
        output = {
            "as_of": as_of,
            "horizon": horizon,
            "target_timestamp": target_timestamp,
            "base_close": float(base_close),
        }
        for column in ("q05", "q25", "q50", "q75", "q95"):
            value = float(row[column])
            output[f"{column}_return"] = value
            output[f"{column}_price"] = _project_price(float(base_close), value, return_type)
        rows.append(output)
    return pd.DataFrame(rows)


def _load_walk_forward_predictions(db_path: str, table_name: str) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        with duckdb_connection(str(path), read_only=True) as connection:
            if not table_exists(connection, table_name):
                return pd.DataFrame()
            frame = connection.execute(
                f"SELECT * FROM {quote_identifier(table_name)} ORDER BY as_of, horizon;"
            ).df()
    except Exception:
        return pd.DataFrame()
    for column in ("as_of", "target_timestamp"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["horizon", "base_close", "actual_close", "actual_return", "q05", "q25", "q50", "q75", "q95"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _aggregate_projection_history(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    required = {"as_of", "horizon", "actual_return", "q05", "q25", "q50", "q75", "q95"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    realized = predictions.dropna(subset=["actual_return", "q05", "q25", "q50", "q75", "q95"]).copy()
    if realized.empty:
        return pd.DataFrame()
    realized = realized.sort_values(["horizon", "as_of"])
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in realized.groupby("horizon", sort=True):
        horizon_frame = horizon_frame.sort_values("as_of")
        windows: list[tuple[str, pd.DataFrame]] = [
            (f"last_{window}", horizon_frame.tail(window))
            for window in DEFAULT_AGGREGATE_WINDOWS
        ]
        windows.append(("all", horizon_frame))
        for window_name, window_frame in windows:
            if window_frame.empty:
                continue
            actual = window_frame["actual_return"].astype(float)
            median = window_frame["q50"].astype(float)
            error = median - actual
            signed = actual.ne(0.0) & median.ne(0.0)
            rows.append(
                {
                    "horizon": int(horizon),
                    "window": window_name,
                    "observations": int(len(window_frame)),
                    "first_as_of": window_frame["as_of"].min(),
                    "last_as_of": window_frame["as_of"].max(),
                    "q50_mae_return": float(error.abs().mean()),
                    "q50_bias_return": float(error.mean()),
                    "q50_rmse_return": float(np.sqrt(np.mean(np.square(error)))),
                    "directional_hit_rate": float(
                        (np.sign(actual.loc[signed]) == np.sign(median.loc[signed])).mean()
                    )
                    if signed.any()
                    else np.nan,
                    "coverage_50": float(((actual >= window_frame["q25"]) & (actual <= window_frame["q75"])).mean()),
                    "coverage_90": float(((actual >= window_frame["q05"]) & (actual <= window_frame["q95"])).mean()),
                    "q05_miss_rate": float((actual < window_frame["q05"]).mean()),
                    "q95_miss_rate": float((actual > window_frame["q95"]).mean()),
                    "avg_width_50": float((window_frame["q75"] - window_frame["q25"]).mean()),
                    "avg_width_90": float((window_frame["q95"] - window_frame["q05"]).mean()),
                }
            )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def _compact_projection_history(predictions: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    columns = [
        "as_of",
        "horizon",
        "target_timestamp",
        "base_close",
        "actual_close",
        "actual_return",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "selected_calibrator",
        "fusion_calibration_method",
        "candidate_name",
        "created_at_utc",
    ]
    available = [column for column in columns if column in predictions.columns]
    if not available:
        return pd.DataFrame()
    recent_asofs = sorted(pd.Timestamp(value) for value in predictions["as_of"].dropna().unique())[-max(int(max_rows), 1):]
    return predictions.loc[predictions["as_of"].isin(recent_asofs), available].sort_values(["as_of", "horizon"])


def _clean_live_forecast_dir(live_root: Path) -> list[str]:
    removed: list[str] = []
    live_root.mkdir(parents=True, exist_ok=True)
    for pattern in LIVE_OUTPUT_REMOVE_PATTERNS:
        for path in live_root.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def _write_clean_live_outputs(
    *,
    production_summary: dict[str, Any] | None,
    config_dir: str,
    walk_forward_db_path: str,
    walk_forward_table: str,
    clean_history_rows: int = DEFAULT_CLEAN_HISTORY_ROWS,
) -> dict[str, Any]:
    from dual_model_forecaster.config import load_config
    from dual_model_forecaster.data import load_forecast_data
    from model_assembly.common import load_doc
    from dual_model_forecaster.brutal_baselines import build_kalman_projection_overlay, build_kalman_state_history

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    live_root = Path(refit_cfg["artifact_root"]) / "live_forecast"
    removed = _clean_live_forecast_dir(live_root)

    base_config = load_config(refit_cfg["base_config"])
    data_bundle = load_forecast_data(base_config)
    close = data_bundle.close.astype(float).sort_index()
    if close.empty:
        raise ValueError("Cannot write clean outputs without close history.")
    latest_as_of = pd.Timestamp(close.index[-1])
    base_close = float(close.iloc[-1])
    return_type = str(base_config["data"]["return_type"])

    live_forecast = (production_summary or {}).get("live_forecast")
    if not live_forecast:
        latest_path = live_root / "latest.json"
        if latest_path.exists():
            live_forecast = json.loads(latest_path.read_text())
    latest_projection = (
        _compact_live_forecast_rows(live_forecast, base_close=base_close, return_type=return_type)
        if live_forecast
        else pd.DataFrame()
    )

    if not latest_projection.empty:
        overlay = build_kalman_projection_overlay(
            predictions=latest_projection[["as_of", "target_timestamp", "horizon", "base_close"]],
            data_bundle=data_bundle,
            horizons=latest_projection["horizon"].astype(int).tolist(),
            return_type=return_type,
        )
        if not overlay.empty:
            overlay_cols = [
                "as_of",
                "horizon",
                "kalman_fair_value_price",
                "kalman_projected_return",
                "kalman_drift_component_return",
                "kalman_mean_reversion_component_return",
                "kalman_macro_component_return",
                "kalman_onchain_component_return",
                "kalman_projected_price",
                "kalman_target_timestamp",
                "kalman_scope_days",
                "kalman_projection_pressure",
                "macro_forward_adjust",
                "onchain_adjust",
                "gap_z",
                "innovation_z",
                "kalman_high_vol_signal",
                "tail_flare_score",
            ]
            latest_projection = latest_projection.merge(
                overlay[[column for column in overlay_cols if column in overlay.columns]],
                on=["as_of", "horizon"],
                how="left",
            )

    state_history = build_kalman_state_history(data_bundle)
    state_columns = [
        "kalman_scope_days",
        "kalman_high_vol_signal",
        "tail_flare_score",
        "macro_scope_days",
        "liquidity_lead_days",
        "cycle_lead_days",
        "financial_conditions_lead_days",
        "signalboost_scope_days",
        "kalman_projection_pressure",
        "onchain_reversion_score",
        "macro_impact_score",
        "macro_forward_score",
    ]
    latest_state: dict[str, Any] = {}
    if not state_history.empty:
        state_row = state_history.loc[state_history.index <= latest_as_of].tail(1)
        if not state_row.empty:
            latest_state = {
                column: _json_default(state_row.iloc[0][column])
                for column in state_columns
                if column in state_row.columns and pd.notna(state_row.iloc[0][column])
            }
    onchain_attention_coverage: dict[str, Any] = {}
    coverage_columns = [
        column
        for column in ("lth_average_available", "sth_average_available")
        if column in state_history.columns
    ]
    if coverage_columns:
        coverage = state_history[coverage_columns].apply(pd.to_numeric, errors="coerce").max(axis=1)
        coverage = coverage.loc[coverage > 0.5]
        if not coverage.empty:
            onchain_attention_coverage = {
                "starts": pd.Timestamp(coverage.index[0]).isoformat(),
                "ends": pd.Timestamp(coverage.index[-1]).isoformat(),
            }

    active_horizons = (
        set(latest_projection["horizon"].dropna().astype(int).tolist())
        if not latest_projection.empty and "horizon" in latest_projection.columns
        else set()
    )
    predictions = _load_walk_forward_predictions(walk_forward_db_path, walk_forward_table)
    if active_horizons and "horizon" in predictions.columns:
        prediction_horizons = pd.to_numeric(predictions["horizon"], errors="coerce").astype("Int64")
        predictions = predictions.loc[prediction_horizons.isin(active_horizons)].copy()
    aggregates = _aggregate_projection_history(predictions)
    history_tail = _compact_projection_history(predictions, max_rows=int(clean_history_rows))
    scope_path, scope_projection = _write_kalman_scope_path(
        live_root,
        latest_projection,
        state_history=state_history,
    )
    projection_chart_path = _write_kalman_projection_chart(
        live_root=live_root,
        close=close,
        latest_projection=latest_projection,
        state_history=state_history,
        scope_path=scope_path,
    )
    historical_charts = _write_kalman_history_charts(
        live_root=live_root,
        close=close,
        state_history=state_history,
    )

    latest_projection_path = live_root / "projection_latest.csv"
    latest_projection_json_path = live_root / "projection_latest.json"
    aggregates_path = live_root / "projection_aggregates.csv"
    history_path = live_root / "projection_history_tail.csv"
    summary_path = live_root / "projection_summary.md"

    if not latest_projection.empty:
        latest_projection.to_csv(latest_projection_path, index=False)
    if not aggregates.empty:
        aggregates.to_csv(aggregates_path, index=False)
    if not history_tail.empty:
        history_tail.to_csv(history_path, index=False)

    latest_records = latest_projection.replace([np.inf, -np.inf], np.nan).where(pd.notna(latest_projection), None)
    payload = {
        "as_of": latest_as_of.isoformat(),
        "base_close": base_close,
        "return_type": return_type,
        "projection": latest_records.to_dict(orient="records") if not latest_records.empty else [],
        "state": latest_state,
        "onchain_attention_coverage": onchain_attention_coverage,
        "kalman_scope_projection": scope_projection,
        "kalman_scope_projection_csv": str(scope_path.resolve()) if scope_path is not None else None,
        "kalman_projection_chart": str(projection_chart_path.resolve()) if projection_chart_path is not None else None,
        "historical_charts": historical_charts,
        "history_aggregate_rows": int(len(aggregates)),
        "history_tail_rows": int(len(history_tail)),
    }
    latest_projection_json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")

    lines = [
        "# BTC Projection",
        "",
        f"- As of: {latest_as_of}",
        f"- Base close: {base_close:,.2f}",
        f"- Return type: {return_type}",
    ]
    if latest_state:
        lines.extend(
            [
                f"- Kalman scope days: {float(latest_state.get('kalman_scope_days', float('nan'))):.1f}"
                if latest_state.get("kalman_scope_days") is not None
                else "- Kalman scope days: n/a",
                f"- Liquidity lead days: {float(latest_state.get('liquidity_lead_days', float('nan'))):.1f}"
                if latest_state.get("liquidity_lead_days") is not None
                else "- Liquidity lead days: n/a",
            ]
        )
        latest_high_vol = _float_or_none(latest_state.get("kalman_high_vol_signal"))
        latest_tail_flare = _float_or_none(latest_state.get("tail_flare_score"))
        active_signals = [
            f"high-vol {latest_high_vol:.2f}"
            if latest_high_vol is not None and latest_high_vol >= 0.50
            else "",
            f"tail flare {latest_tail_flare:.2f}"
            if latest_tail_flare is not None and latest_tail_flare >= 1.00
            else "",
        ]
        active_signals = [value for value in active_signals if value]
        if active_signals:
            lines.extend(["", f"- Tail/high-vol overlay active in latest projection: {', '.join(active_signals)}"])
    if scope_projection:
        projected_price = scope_projection.get("projected_price")
        projected_return = scope_projection.get("projected_return")
        lines.extend(
            [
                f"- Kalman scope target: {pd.Timestamp(scope_projection['target_timestamp']).date()}",
                f"- Kalman scope projected price: {_format_float(projected_price, ',.0f')}",
                f"- Kalman scope projected return: {_format_float(float(projected_return) * 100.0 if projected_return is not None else None, '.2f')}%",
                f"- State component return: {_format_float(float(scope_projection.get('state_return')) * 100.0 if scope_projection.get('state_return') is not None else None, '.2f')}%",
                f"- Macro component return: {_format_float(float(scope_projection.get('macro_component_return')) * 100.0 if scope_projection.get('macro_component_return') is not None else None, '.2f')}%",
                f"- On-chain component return: {_format_float(float(scope_projection.get('onchain_component_return')) * 100.0 if scope_projection.get('onchain_component_return') is not None else None, '.2f')}%",
            ]
        )
        if projection_chart_path is not None:
            lines.extend(["", "![Kalman projection scope](kalman_projection_scope.png)"])
    if onchain_attention_coverage:
        lines.extend(
            [
                "",
                f"- On-chain attention coverage: {pd.Timestamp(onchain_attention_coverage['starts']).date()} to {pd.Timestamp(onchain_attention_coverage['ends']).date()}",
            ]
        )
    if not latest_projection.empty:
        lines.extend(["", "## Latest Projection", ""])
        header = "| Horizon | q50 Price | q05-q95 Price | q50 Return | Kalman Scope | Kalman Price |"
        lines.extend([header, "|---:|---:|---:|---:|---:|---:|"])
        for _, row in latest_projection.sort_values("horizon").iterrows():
            q05_price = row.get("q05_price")
            q50_price = row.get("q50_price")
            q95_price = row.get("q95_price")
            kalman_price = row.get("kalman_projected_price")
            scope = row.get("kalman_scope_days")
            lines.append(
                f"| {int(row['horizon'])}d | "
                f"{_format_float(q50_price, ',.0f')} | "
                f"{_format_float(q05_price, ',.0f')}-{_format_float(q95_price, ',.0f')} | "
                f"{_format_float(float(row['q50_return']) * 100.0, '.2f')}% | "
                f"{_format_float(scope, '.1f')}d | "
                f"{_format_float(kalman_price, ',.0f')} |"
            )
    if historical_charts:
        lines.extend(["", "## Historical PNGs", ""])
        chart_labels = {
            "tail_flare_highvol_price_history": "Tail flare and high-vol overlay",
            "fair_value_history": "Fair value and anchors",
            "macro_leads_history": "Macro lead and scope drivers",
            "onchain_lth_average_signalboost_history": "On-chain LTH average and signalboost",
            "onchain_sth_average_signalboost_history": "On-chain STH average and signalboost",
            "signalboost_attention_history": "Signalboost and on-chain attention",
        }
        chart_filenames = {
            key: Path(value).name
            for key, value in historical_charts.items()
        }
        for key, label in chart_labels.items():
            filename = chart_filenames.get(key)
            if filename:
                lines.extend([f"### {label}", f"![{label}]({filename})", ""])
    if not aggregates.empty:
        lines.extend(["", "## Historical Aggregates", ""])
        for window in ("last_30", "last_90", "last_365", "all"):
            subset = aggregates.loc[aggregates["window"] == window]
            if subset.empty:
                continue
            lines.append(f"### {window}")
            lines.append("| Horizon | Obs | MAE Return | Coverage 50 | Coverage 90 | Direction Hit |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for _, row in subset.sort_values("horizon").iterrows():
                lines.append(
                    f"| {int(row['horizon'])}d | {int(row['observations'])} | "
                    f"{_format_float(float(row['q50_mae_return']) * 100.0, '.2f')}% | "
                    f"{_format_float(row['coverage_50'], '.2f')} | "
                    f"{_format_float(row['coverage_90'], '.2f')} | "
                    f"{_format_float(row['directional_hit_rate'], '.2f')} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            f"- Latest projection CSV: {latest_projection_path}",
            f"- Latest projection JSON: {latest_projection_json_path}",
            f"- Kalman scope path CSV: {scope_path}" if scope_path is not None else "- Kalman scope path CSV: n/a",
            f"- Aggregate history CSV: {aggregates_path}",
            f"- Recent history CSV: {history_path}",
        ]
    )
    summary_path.write_text("\n".join(lines).rstrip() + "\n")

    return {
        "live_output_dir": str(live_root.resolve()),
        "removed_legacy_live_files": removed,
        "latest_projection": str(latest_projection_json_path.resolve()),
        "latest_projection_csv": str(latest_projection_path.resolve()) if latest_projection_path.exists() else None,
        "kalman_scope_projection_csv": str(scope_path.resolve()) if scope_path is not None else None,
        "kalman_projection_chart": str(projection_chart_path.resolve()) if projection_chart_path is not None else None,
        "historical_charts": historical_charts,
        "projection_aggregates": str(aggregates_path.resolve()) if aggregates_path.exists() else None,
        "projection_history_tail": str(history_path.resolve()) if history_path.exists() else None,
        "projection_summary": str(summary_path.resolve()),
        "latest_as_of": latest_as_of.isoformat(),
    }


def _load_latest_jepa_wide_features(latest_csv: Path) -> pd.DataFrame:
    if not latest_csv.exists():
        return pd.DataFrame()
    latest = pd.read_csv(latest_csv)
    if latest.empty or not {"timestamp", "specialist", "horizon"}.issubset(latest.columns):
        return pd.DataFrame()
    metric_sources = {
        "kalman_alignment": "jepa_kalman_alignment",
        "reversion_pressure": "jepa_reversion_pressure",
        "vol_pressure": "jepa_vol_pressure",
        "tail_pressure": "jepa_tail_pressure",
        "uncertainty_proxy": "jepa_uncertainty_proxy",
        "latent_norm": "jepa_norm",
        "delta_norm": "jepa_delta_norm",
    }
    rows: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        try:
            specialist = str(row["specialist"])
            horizon = int(row["horizon"])
            timestamp = pd.Timestamp(row["timestamp"])
        except Exception:
            continue
        output: dict[str, Any] = {"as_of": timestamp, "horizon": horizon}
        for metric, source_token in metric_sources.items():
            source = f"{specialist}_{source_token}_h{horizon}"
            if source in latest.columns and pd.notna(row.get(source)):
                output[f"{specialist}_jepa_{metric}"] = float(row[source])
        rows.append(output)
    if not rows:
        return pd.DataFrame()
    wide = pd.DataFrame(rows).groupby(["as_of", "horizon"], as_index=False).first()
    for metric in metric_sources:
        specialist_columns = [
            column
            for column in wide.columns
            if column.endswith(f"_jepa_{metric}") and not column.startswith("aggregate_")
        ]
        if specialist_columns:
            wide[f"aggregate_jepa_{metric}"] = wide[specialist_columns].mean(axis=1, skipna=True)
    availability_columns = [
        column
        for column in wide.columns
        if column.endswith("_jepa_kalman_alignment") and not column.startswith("aggregate_")
    ]
    wide["aggregate_jepa_feature_count"] = wide[availability_columns].notna().sum(axis=1) if availability_columns else 0
    wide["aggregate_jepa_feature_available"] = wide["aggregate_jepa_feature_count"].astype(int) > 0
    return wide.replace([np.inf, -np.inf], np.nan)


def _write_experimental_jepa_live_meta_outputs(
    *,
    production_summary: dict[str, Any] | None,
    config_dir: str,
    jepa_latest_csv: str | None,
) -> dict[str, Any]:
    from dual_model_forecaster.brutal_baselines import build_kalman_projection_overlay
    from dual_model_forecaster.config import load_config
    from dual_model_forecaster.data import load_forecast_data
    from dual_model_forecaster.jepa.meta_synthesis import (
        JEPA_META_CANDIDATES,
        apply_jepa_meta_candidate,
        load_jepa_meta_params,
    )
    from model_assembly.common import load_doc

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    live_root = Path(refit_cfg["artifact_root"]) / "live_forecast"
    selected_path = live_root / "meta_synthesis_jepa_selected.csv"
    if not selected_path.exists():
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"missing {selected_path.name}; run walk-forward with JEPA_USE_IN_META=true first",
        }

    base_config = load_config(refit_cfg["base_config"])
    data_bundle = load_forecast_data(base_config)
    close = data_bundle.close.astype(float).sort_index()
    if close.empty:
        return {"enabled": True, "status": "skipped", "reason": "empty close history"}
    base_close = float(close.iloc[-1])
    return_type = str(base_config["data"]["return_type"])

    live_forecast = (production_summary or {}).get("live_forecast")
    if not live_forecast:
        latest_path = live_root / "latest.json"
        if latest_path.exists():
            live_forecast = json.loads(latest_path.read_text())
    latest_projection = (
        _compact_live_forecast_rows(live_forecast, base_close=base_close, return_type=return_type)
        if live_forecast
        else pd.DataFrame()
    )
    if latest_projection.empty:
        return {"enabled": True, "status": "skipped", "reason": "empty live projection rows"}

    feature = latest_projection[["as_of", "horizon", "target_timestamp", "base_close"]].copy()
    for column in ("q05", "q25", "q50", "q75", "q95"):
        feature[column] = pd.to_numeric(latest_projection[f"{column}_return"], errors="coerce")

    overlay = build_kalman_projection_overlay(
        predictions=feature[["as_of", "target_timestamp", "horizon", "base_close"]],
        data_bundle=data_bundle,
        horizons=feature["horizon"].astype(int).tolist(),
        return_type=return_type,
    )
    if not overlay.empty:
        overlay["as_of"] = pd.to_datetime(overlay["as_of"], errors="coerce")
        overlay["horizon"] = pd.to_numeric(overlay["horizon"], errors="coerce").astype("Int64")
        overlay_columns = [
            "as_of",
            "horizon",
            "kalman_projected_return",
            "kalman_projected_price",
            "kalman_scope_days",
            "kalman_projection_pressure",
            "gap_z",
            "innovation_z",
            "residual_sigma",
            "kalman_high_vol_signal",
            "tail_flare_score",
        ]
        feature = feature.merge(
            overlay[[column for column in overlay_columns if column in overlay.columns]],
            on=["as_of", "horizon"],
            how="left",
        )

    latest_jepa_path = Path(jepa_latest_csv) if jepa_latest_csv else live_root / "jepa" / "jepa_latest.csv"
    jepa_wide = _load_latest_jepa_wide_features(latest_jepa_path)
    if jepa_wide.empty:
        return {"enabled": True, "status": "skipped", "reason": "missing live JEPA feature rows"}
    jepa_wide["as_of"] = pd.to_datetime(jepa_wide["as_of"], errors="coerce")
    jepa_wide["horizon"] = pd.to_numeric(jepa_wide["horizon"], errors="coerce").astype("Int64")
    feature["horizon"] = pd.to_numeric(feature["horizon"], errors="coerce").astype("Int64")
    feature = feature.merge(jepa_wide, on=["as_of", "horizon"], how="left")
    feature["aggregate_jepa_feature_available"] = feature["aggregate_jepa_feature_available"].fillna(False).astype(bool)

    selected = pd.read_csv(selected_path)
    if selected.empty or "horizon" not in selected.columns:
        return {"enabled": True, "status": "skipped", "reason": "empty JEPA meta selection file"}
    selected["as_of"] = pd.to_datetime(selected["as_of"], errors="coerce")
    selected["horizon"] = pd.to_numeric(selected["horizon"], errors="coerce").astype("Int64")
    selected = selected.sort_values(["horizon", "as_of"])
    selected_lookup: dict[int, str] = {}
    for horizon, group in selected.dropna(subset=["horizon"]).groupby("horizon"):
        candidate = str(group["meta_selected_candidate"].dropna().iloc[-1]) if group["meta_selected_candidate"].notna().any() else ""
        selected_lookup[int(horizon)] = candidate

    params = load_jepa_meta_params()
    output_rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    for _, row in feature.iterrows():
        horizon = int(row["horizon"])
        candidate = selected_lookup.get(horizon, "model_passthrough")
        one = pd.DataFrame([row.to_dict()])
        applied = False
        if candidate in JEPA_META_CANDIDATES and bool(row.get("aggregate_jepa_feature_available")):
            adjusted = apply_jepa_meta_candidate(one, candidate=candidate, params=params)
            applied = True
        else:
            adjusted = one.copy()
            adjusted["forecaster"] = "jepa_live_meta_passthrough"
            adjusted["meta_candidate"] = candidate
            adjusted["meta_selected_candidate"] = candidate
            adjusted["meta_effective_kalman_weight"] = 0.0
            adjusted["meta_center_shift"] = 0.0
            adjusted["meta_width_multiplier"] = 1.0
            if candidate not in JEPA_META_CANDIDATES:
                warnings.append(f"{horizon}d selected non-JEPA candidate {candidate}; left quantiles unchanged.")
        adjusted["jepa_live_meta_applied"] = applied
        output_rows.append(adjusted)
    adjusted = pd.concat(output_rows, ignore_index=True, sort=False) if output_rows else pd.DataFrame()
    if adjusted.empty:
        return {"enabled": True, "status": "skipped", "reason": "no adjusted JEPA live meta rows"}

    export_columns = [
        "as_of",
        "horizon",
        "target_timestamp",
        "base_close",
        "meta_selected_candidate",
        "jepa_live_meta_applied",
        "meta_center_shift",
        "meta_width_multiplier",
        "aggregate_jepa_kalman_alignment",
        "aggregate_jepa_uncertainty_proxy",
        "aggregate_jepa_tail_pressure",
        "aggregate_jepa_reversion_pressure",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    ]
    export = adjusted[[column for column in export_columns if column in adjusted.columns]].copy()
    for column in ("q05", "q25", "q50", "q75", "q95"):
        export[f"{column}_price"] = [
            _project_price(float(base_close), float(value), return_type)
            for value in export[column].to_numpy(dtype=float)
        ]

    csv_path = live_root / "jepa_meta_projection_latest.csv"
    json_path = live_root / "jepa_meta_projection_latest.json"
    summary_path = live_root / "jepa_meta_projection_summary.md"
    export.to_csv(csv_path, index=False)
    payload = {
        "enabled": True,
        "status": "written",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "selection_source": str(selected_path.resolve()),
        "base_close": base_close,
        "return_type": return_type,
        "warnings": sorted(set(warnings)),
        "files": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "summary_md": str(summary_path.resolve()),
        },
        "rows": export.where(pd.notna(export), None).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    lines = [
        "# Experimental JEPA Meta Projection",
        "",
        "This file is written only when `JEPA_USE_LIVE_META=true`.",
        "",
        f"- Base close: {base_close:,.2f}",
        f"- Return type: {return_type}",
        f"- Rows: {len(export)}",
        f"- Source selection: {selected_path.name}",
    ]
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    summary_path.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "enabled": True,
        "status": "written",
        "csv": str(csv_path.resolve()),
        "json": str(json_path.resolve()),
        "summary_md": str(summary_path.resolve()),
        "rows": int(len(export)),
        "warnings": sorted(set(warnings)),
    }


def _temporary_jepa_env_from_config(jepa_config: Any | None) -> dict[str, str | None]:
    env_keys = [
        "JEPA_ENABLED",
        "JEPA_MODE",
        "JEPA_CHECKPOINT_DIR",
        "JEPA_ARTIFACT_DIR",
        "JEPA_HORIZONS",
        "JEPA_CONTEXT_LENGTHS",
        "JEPA_LATENT_DIM",
        "JEPA_MAX_EPOCHS",
        "JEPA_BATCH_SIZE",
        "JEPA_SEED",
    ]
    previous = {key: os.environ.get(key) for key in env_keys}
    if jepa_config is not None and getattr(jepa_config, "is_active", False):
        os.environ["JEPA_ENABLED"] = "true"
        os.environ["JEPA_MODE"] = str(jepa_config.mode)
        os.environ["JEPA_CHECKPOINT_DIR"] = str(jepa_config.checkpoint_dir)
        os.environ["JEPA_ARTIFACT_DIR"] = str(jepa_config.artifact_dir)
        os.environ["JEPA_HORIZONS"] = ",".join(str(int(value)) for value in jepa_config.horizons)
        os.environ["JEPA_CONTEXT_LENGTHS"] = ",".join(str(int(value)) for value in jepa_config.context_lengths)
        os.environ["JEPA_LATENT_DIM"] = str(int(jepa_config.latent_dim))
        os.environ["JEPA_MAX_EPOCHS"] = str(int(jepa_config.max_epochs))
        os.environ["JEPA_BATCH_SIZE"] = str(int(jepa_config.batch_size))
        os.environ["JEPA_SEED"] = str(int(jepa_config.seed))
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_meta_learner_for_live_forecast(
    *,
    config_dir: str,
    db_path: str,
    table_name: str,
    jepa_config: Any | None = None,
    min_history: int | None = None,
    selection_window: int | None = None,
) -> dict[str, Any]:
    from dual_model_forecaster.config import load_config
    from dual_model_forecaster.data import load_forecast_data
    from model_assembly.common import load_doc
    from walk_forward_predictions import (
        META_DEFAULT_MIN_HISTORY,
        META_DEFAULT_SELECTION_WINDOW,
        _load_production_horizons,
        build_meta_synthesis_artifacts,
    )

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    live_root = Path(refit_cfg["artifact_root"]) / "live_forecast"
    base_config = load_config(refit_cfg["base_config"])
    data_bundle = load_forecast_data(base_config)
    previous_env = _temporary_jepa_env_from_config(jepa_config)
    try:
        return build_meta_synthesis_artifacts(
            db_path=db_path,
            table_name=table_name,
            chart_dir=live_root,
            horizons=_load_production_horizons(config_dir),
            data_bundle=data_bundle,
            return_type=str(base_config["data"]["return_type"]),
            min_history=int(min_history if min_history is not None else META_DEFAULT_MIN_HISTORY),
            selection_window=int(selection_window if selection_window is not None else META_DEFAULT_SELECTION_WINDOW),
        )
    finally:
        _restore_env(previous_env)


def _latest_meta_candidate_by_horizon(selected: pd.DataFrame) -> dict[int, str]:
    if selected.empty or "horizon" not in selected.columns:
        return {}
    frame = selected.copy()
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame = frame.dropna(subset=["horizon"]).sort_values(["horizon", "as_of"])
    out: dict[int, str] = {}
    for horizon, group in frame.groupby(frame["horizon"].astype(int), sort=True):
        candidate_source = group.get("meta_selected_candidate", group.get("meta_candidate"))
        if candidate_source is None or candidate_source.dropna().empty:
            continue
        out[int(horizon)] = str(candidate_source.dropna().iloc[-1])
    return out


def _apply_live_meta_candidate(
    feature_row: pd.DataFrame,
    *,
    candidate: str,
) -> pd.DataFrame:
    from dual_model_forecaster.jepa.meta_synthesis import (
        JEPA_META_CANDIDATES,
        apply_jepa_meta_candidate,
        load_jepa_meta_params,
    )
    from walk_forward_predictions import META_CANDIDATE_SPECS, _synthesize_meta_candidate

    if candidate in JEPA_META_CANDIDATES:
        if bool(feature_row.get("aggregate_jepa_feature_available", pd.Series([False])).iloc[0]):
            return apply_jepa_meta_candidate(
                feature_row,
                candidate=candidate,
                params=load_jepa_meta_params(),
            )
        candidate = "model_passthrough"

    spec_by_name = {str(spec["name"]): spec for spec in META_CANDIDATE_SPECS}
    spec = spec_by_name.get(candidate, spec_by_name["model_passthrough"])
    return _synthesize_meta_candidate(feature_row, spec)


def _write_live_meta_projection_outputs(
    *,
    production_summary: dict[str, Any] | None,
    config_dir: str,
    meta_learner_summary: dict[str, Any] | None,
    jepa_latest_csv: str | None = None,
) -> dict[str, Any]:
    from dual_model_forecaster.brutal_baselines import build_kalman_projection_overlay
    from dual_model_forecaster.config import load_config
    from dual_model_forecaster.data import load_forecast_data
    from model_assembly.common import load_doc

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    live_root = Path(refit_cfg["artifact_root"]) / "live_forecast"
    selected_path = Path(
        ((meta_learner_summary or {}).get("artifacts") or {}).get("meta_selected")
        or live_root / "meta_synthesis_selected.csv"
    )
    if not selected_path.exists():
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"missing {selected_path.name}",
        }

    base_config = load_config(refit_cfg["base_config"])
    data_bundle = load_forecast_data(base_config)
    close = data_bundle.close.astype(float).sort_index()
    if close.empty:
        return {"enabled": True, "status": "skipped", "reason": "empty close history"}
    base_close = float(close.iloc[-1])
    return_type = str(base_config["data"]["return_type"])

    live_forecast = (production_summary or {}).get("live_forecast")
    if not live_forecast:
        latest_path = live_root / "latest.json"
        if latest_path.exists():
            live_forecast = json.loads(latest_path.read_text())
    latest_projection = (
        _compact_live_forecast_rows(live_forecast, base_close=base_close, return_type=return_type)
        if live_forecast
        else pd.DataFrame()
    )
    if latest_projection.empty:
        return {"enabled": True, "status": "skipped", "reason": "empty live projection rows"}

    feature = latest_projection[["as_of", "horizon", "target_timestamp", "base_close"]].copy()
    for column in ("q05", "q25", "q50", "q75", "q95"):
        feature[column] = pd.to_numeric(latest_projection[f"{column}_return"], errors="coerce")

    overlay = build_kalman_projection_overlay(
        predictions=feature[["as_of", "target_timestamp", "horizon", "base_close"]],
        data_bundle=data_bundle,
        horizons=feature["horizon"].astype(int).tolist(),
        return_type=return_type,
    )
    if not overlay.empty:
        overlay["as_of"] = pd.to_datetime(overlay["as_of"], errors="coerce")
        overlay["horizon"] = pd.to_numeric(overlay["horizon"], errors="coerce").astype("Int64")
        overlay_columns = [
            "as_of",
            "horizon",
            "kalman_projected_return",
            "kalman_projected_price",
            "kalman_scope_days",
            "kalman_projection_pressure",
            "gap_z",
            "innovation_z",
            "residual_sigma",
            "kalman_high_vol_signal",
            "tail_flare_score",
            "kalman_adjusted_edge_z",
        ]
        feature = feature.merge(
            overlay[[column for column in overlay_columns if column in overlay.columns]],
            on=["as_of", "horizon"],
            how="left",
        )

    latest_jepa_path = Path(jepa_latest_csv) if jepa_latest_csv else live_root / "jepa" / "jepa_latest.csv"
    jepa_wide = _load_latest_jepa_wide_features(latest_jepa_path)
    if not jepa_wide.empty:
        jepa_wide["as_of"] = pd.to_datetime(jepa_wide["as_of"], errors="coerce")
        jepa_wide["horizon"] = pd.to_numeric(jepa_wide["horizon"], errors="coerce").astype("Int64")
        feature["horizon"] = pd.to_numeric(feature["horizon"], errors="coerce").astype("Int64")
        feature = feature.merge(jepa_wide, on=["as_of", "horizon"], how="left")
    if "aggregate_jepa_feature_available" not in feature.columns:
        feature["aggregate_jepa_feature_available"] = False
    feature["aggregate_jepa_feature_available"] = feature["aggregate_jepa_feature_available"].fillna(False).astype(bool)

    selected = pd.read_csv(selected_path)
    selected_lookup = _latest_meta_candidate_by_horizon(selected)
    rows: list[dict[str, Any]] = []
    for _, row in feature.iterrows():
        horizon = int(row["horizon"])
        candidate = selected_lookup.get(horizon, "model_passthrough")
        model_row = row.to_dict()
        applied = _apply_live_meta_candidate(pd.DataFrame([model_row]), candidate=candidate).iloc[0].to_dict()
        output: dict[str, Any] = {
            "as_of": model_row.get("as_of"),
            "horizon": horizon,
            "target_timestamp": model_row.get("target_timestamp"),
            "base_close": base_close,
            "meta_selected_candidate": candidate,
            "meta_effective_kalman_weight": applied.get("meta_effective_kalman_weight"),
            "meta_center_shift": applied.get("meta_center_shift"),
            "meta_width_multiplier": applied.get("meta_width_multiplier"),
            "jepa_feature_available": bool(model_row.get("aggregate_jepa_feature_available", False)),
        }
        for column in ("q05", "q25", "q50", "q75", "q95"):
            model_return = _float_or_none(model_row.get(column))
            meta_return = _float_or_none(applied.get(column))
            output[f"model_{column}_return"] = model_return
            output[f"meta_{column}_return"] = meta_return
            output[f"model_{column}_price"] = _project_price(base_close, model_return, return_type) if model_return is not None else None
            output[f"meta_{column}_price"] = _project_price(base_close, meta_return, return_type) if meta_return is not None else None
            output[f"meta_minus_model_{column}_return"] = (
                float(meta_return - model_return)
                if meta_return is not None and model_return is not None
                else None
            )
        rows.append(output)

    projection = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    csv_path = live_root / "meta_projection_latest.csv"
    json_path = live_root / "meta_projection_latest.json"
    chart_path = live_root / "meta_projection_latest.png"
    summary_path = live_root / "meta_projection_summary.md"
    projection.to_csv(csv_path, index=False)

    _write_live_meta_projection_chart(
        projection=projection,
        close=close,
        chart_path=chart_path,
    )
    payload = {
        "enabled": True,
        "status": "written",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "selection_source": str(selected_path.resolve()),
        "base_close": base_close,
        "return_type": return_type,
        "files": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "chart": str(chart_path.resolve()),
            "summary_md": str(summary_path.resolve()),
        },
        "rows": projection.where(pd.notna(projection), None).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    lines = [
        "# Live Meta Projection",
        "",
        f"- Selection source: {selected_path.name}",
        f"- Base close: {base_close:,.2f}",
        f"- Return type: {return_type}",
        "",
        "This augments the locked production forecast; it does not overwrite `latest.json`.",
        "",
        "![Live meta projection](meta_projection_latest.png)",
        "",
        "| Horizon | Selected Meta | Model q50 | Meta q50 | Center Shift | Width Mult |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in projection.sort_values("horizon").iterrows():
        lines.append(
            f"| {int(row['horizon'])}d | {row.get('meta_selected_candidate', '')} | "
            f"{_format_float(row.get('model_q50_price'), ',.0f')} | "
            f"{_format_float(row.get('meta_q50_price'), ',.0f')} | "
            f"{_format_float(row.get('meta_center_shift'), '.5f')} | "
            f"{_format_float(row.get('meta_width_multiplier'), '.2f')} |"
        )
    summary_path.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "enabled": True,
        "status": "written",
        "csv": str(csv_path.resolve()),
        "json": str(json_path.resolve()),
        "chart": str(chart_path.resolve()),
        "summary_md": str(summary_path.resolve()),
        "rows": int(len(projection)),
    }


def _write_live_meta_projection_chart(
    *,
    projection: pd.DataFrame,
    close: pd.Series,
    chart_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    history = close.dropna().tail(min(180, len(close)))
    if history.empty or projection.empty:
        return
    frame = projection.copy()
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce")
    frame = frame.dropna(subset=["target_timestamp"]).sort_values("horizon")
    if frame.empty:
        return

    fig, (ax_price, ax_shift) = plt.subplots(
        2,
        1,
        figsize=(12, 7.6),
        dpi=150,
        sharex=False,
        gridspec_kw={"height_ratios": [3.0, 1.1]},
        constrained_layout=True,
    )
    ax_price.plot(history.index, history.values, color="#111827", linewidth=1.6, label="BTC close")
    x = frame["target_timestamp"]
    ax_price.fill_between(
        x,
        pd.to_numeric(frame["meta_q05_price"], errors="coerce"),
        pd.to_numeric(frame["meta_q95_price"], errors="coerce"),
        color="#60a5fa",
        alpha=0.18,
        label="meta q05-q95",
    )
    ax_price.fill_between(
        x,
        pd.to_numeric(frame["meta_q25_price"], errors="coerce"),
        pd.to_numeric(frame["meta_q75_price"], errors="coerce"),
        color="#2563eb",
        alpha=0.26,
        label="meta q25-q75",
    )
    ax_price.plot(x, frame["meta_q50_price"], color="#dc2626", marker="o", linewidth=1.9, label="meta q50")
    ax_price.plot(x, frame["model_q50_price"], color="#64748b", marker="o", linestyle=":", linewidth=1.35, label="model q50")
    ax_price.scatter([history.index[-1]], [float(history.iloc[-1])], color="#111827", s=24, zorder=4)
    for _, row in frame.iterrows():
        ax_price.annotate(
            f"{int(row['horizon'])}d\n{row.get('meta_selected_candidate', '')}",
            xy=(row["target_timestamp"], row["meta_q50_price"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#7f1d1d",
        )
    ax_price.set_title("Live Forecast With Meta-Learner Overlay", loc="left", fontsize=13, weight="bold")
    ax_price.set_ylabel("BTC price")
    ax_price.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
    ax_price.legend(loc="upper left", ncols=3, frameon=False, fontsize=8)
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_price.xaxis.get_major_locator()))

    labels = [f"{int(value)}d" for value in frame["horizon"]]
    shift_values = pd.to_numeric(frame["meta_minus_model_q50_return"], errors="coerce").fillna(0.0)
    colors = ["#16a34a" if value >= 0.0 else "#dc2626" for value in shift_values]
    ax_shift.bar(labels, shift_values * 100.0, color=colors, alpha=0.82)
    ax_shift.axhline(0.0, color="#111827", linewidth=0.8)
    ax_shift.set_ylabel("q50 shift, pp")
    ax_shift.set_title("Meta Center Shift Versus Locked Model", loc="left", fontsize=10)
    ax_shift.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)


def _specialist_semantic_columns(bucket_name: str) -> list[str]:
    columns = {
        "structure": [
            "structural_overvaluation",
            "structural_undervaluation",
            "holder_conviction",
            "holder_distribution_fragility",
            "structural_reversion_pressure",
            "structural_confidence",
        ],
        "environment": [
            "liquidity_tailwind",
            "liquidity_headwind",
            "macro_risk_on",
            "macro_risk_off",
            "macro_transition_risk",
            "environment_confidence",
        ],
        "edges": [
            "upside_stretch",
            "downside_stretch",
            "mean_reversion_pressure",
            "local_volatility_instability",
            "edge_asymmetry",
            "edge_confidence",
        ],
        "movement": [
            "trend_pressure_up",
            "trend_pressure_down",
            "trend_persistence",
            "momentum_quality",
            "chop_risk",
            "trend_strategy_influence",
            "momentum_strategy_influence",
            "mean_reversion_strategy_influence",
            "movement_confidence",
        ],
        "liquidation": [
            "liquidation_cascade_pressure",
            "liquidation_directional_pressure",
            "leverage_fuel",
            "liquidation_volatility_instability",
            "cascade_asymmetry",
            "liquidation_confidence",
        ],
    }
    return columns.get(bucket_name, [])


def _load_specialist_live_state(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    first_column = frame.columns[0]
    if first_column == "timestamp":
        frame[first_column] = pd.to_datetime(frame[first_column], errors="coerce")
        frame = frame.set_index(first_column)
    else:
        parsed = pd.to_datetime(frame[first_column], errors="coerce")
        if parsed.notna().any():
            frame = frame.drop(columns=[first_column])
            frame.index = parsed
    return frame.sort_index()


def _row_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = _float_or_none(row.get(column))
    return default if value is None else float(value)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _specialist_indicator_rows(
    *,
    specialist: str,
    timestamp: pd.Timestamp,
    row: pd.Series,
    roles: list[str],
    direction: float,
    confidence: float,
) -> list[dict[str, Any]]:
    role_text = ",".join(roles)
    confidence = _clip01(confidence)

    def record(
        indicator: str,
        upside: float,
        downside: float,
        *,
        intensity: float | None = None,
        interpretation: str | None = None,
        components: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        upside = _clip01(upside)
        downside = _clip01(downside)
        net = float(np.clip(upside - downside, -1.0, 1.0))
        if interpretation is None:
            interpretation = "upside" if net > 0.05 else "downside" if net < -0.05 else "balanced"
        return {
            "timestamp": timestamp,
            "specialist": specialist,
            "indicator": indicator,
            "meta_roles": role_text,
            "upside_score": upside,
            "downside_score": downside,
            "net_score": net,
            "intensity": _clip01(max(upside, downside) if intensity is None else intensity),
            "confidence": confidence,
            "interpretation": interpretation,
            "components_json": json.dumps(components or {}, sort_keys=True),
        }

    if specialist == "liquidation":
        cascade = _clip01(_row_value(row, "liquidation_cascade_pressure"))
        fuel = _clip01(_row_value(row, "leverage_fuel"))
        volatility = _clip01(_row_value(row, "liquidation_volatility_instability"))
        asymmetry = float(np.clip(_row_value(row, "cascade_asymmetry"), -1.0, 1.0))
        directional = float(np.clip(_row_value(row, "liquidation_directional_pressure", direction), -1.0, 1.0))
        fuel_mix = _clip01((cascade + fuel + volatility) / 3.0)
        upside = _clip01((0.65 * max(directional, 0.0) + 0.35 * max(asymmetry, 0.0)) * (0.45 + 0.55 * fuel_mix) * (0.50 + 0.50 * confidence))
        downside = _clip01((0.65 * max(-directional, 0.0) + 0.35 * max(-asymmetry, 0.0)) * (0.45 + 0.55 * fuel_mix) * (0.50 + 0.50 * confidence))
        return [
            record(
                "liquidation_upside_downside_potential",
                upside,
                downside,
                intensity=fuel_mix,
                components={
                    "liquidation_directional_pressure": directional,
                    "cascade_asymmetry": asymmetry,
                    "liquidation_cascade_pressure": cascade,
                    "leverage_fuel": fuel,
                    "liquidation_volatility_instability": volatility,
                },
            )
        ]

    if specialist == "edges":
        upside_stretch = _clip01(_row_value(row, "upside_stretch"))
        downside_stretch = _clip01(_row_value(row, "downside_stretch"))
        mean_reversion = _clip01(_row_value(row, "mean_reversion_pressure"))
        instability = _clip01(_row_value(row, "local_volatility_instability"))
        asymmetry = float(np.clip(_row_value(row, "edge_asymmetry"), -1.0, 1.0))
        upside_reversion = _clip01(downside_stretch * mean_reversion * (0.50 + 0.50 * confidence))
        downside_reversion = _clip01(upside_stretch * mean_reversion * (0.50 + 0.50 * confidence))
        return [
            record(
                "edges_flow_reversion",
                upside_reversion,
                downside_reversion,
                intensity=_clip01(mean_reversion * (0.65 + 0.35 * instability)),
                components={
                    "upside_stretch": upside_stretch,
                    "downside_stretch": downside_stretch,
                    "mean_reversion_pressure": mean_reversion,
                    "local_volatility_instability": instability,
                    "edge_asymmetry": asymmetry,
                },
            )
        ]

    if specialist == "structure":
        overvaluation = _clip01(_row_value(row, "structural_overvaluation"))
        undervaluation = _clip01(_row_value(row, "structural_undervaluation"))
        reversion = _clip01(_row_value(row, "structural_reversion_pressure"))
        fragility = _clip01(_row_value(row, "holder_distribution_fragility"))
        return [
            record(
                "structure_valuation_reversion",
                undervaluation * reversion * (0.50 + 0.50 * confidence),
                overvaluation * max(reversion, fragility) * (0.50 + 0.50 * confidence),
                intensity=max(reversion, overvaluation, undervaluation),
                components={
                    "structural_overvaluation": overvaluation,
                    "structural_undervaluation": undervaluation,
                    "structural_reversion_pressure": reversion,
                    "holder_distribution_fragility": fragility,
                },
            )
        ]

    if specialist == "environment":
        tailwind = _clip01(_row_value(row, "liquidity_tailwind"))
        headwind = _clip01(_row_value(row, "liquidity_headwind"))
        risk_on = _clip01(_row_value(row, "macro_risk_on"))
        risk_off = _clip01(_row_value(row, "macro_risk_off"))
        transition = _clip01(_row_value(row, "macro_transition_risk"))
        return [
            record(
                "environment_liquidity_macro_pressure",
                ((tailwind + risk_on) / 2.0) * (0.50 + 0.50 * confidence),
                ((headwind + risk_off) / 2.0) * (0.50 + 0.50 * confidence),
                intensity=max(tailwind, headwind, risk_on, risk_off, transition),
                components={
                    "liquidity_tailwind": tailwind,
                    "liquidity_headwind": headwind,
                    "macro_risk_on": risk_on,
                    "macro_risk_off": risk_off,
                    "macro_transition_risk": transition,
                },
            )
        ]

    trend_up = _clip01(_row_value(row, "trend_pressure_up"))
    trend_down = _clip01(_row_value(row, "trend_pressure_down"))
    trend_influence = _clip01(_row_value(row, "trend_strategy_influence", 0.5))
    momentum_influence = _clip01(_row_value(row, "momentum_strategy_influence"))
    mean_reversion_influence = _clip01(_row_value(row, "mean_reversion_strategy_influence"))
    chop = _clip01(_row_value(row, "chop_risk", 0.5))
    trend_weight = _clip01((trend_influence + momentum_influence) / 2.0)
    reversion_weight = _clip01((mean_reversion_influence + chop) / 2.0)
    return [
        record(
            "movement_trend_vs_chop",
            trend_up * trend_weight * (0.50 + 0.50 * confidence),
            trend_down * trend_weight * (0.50 + 0.50 * confidence),
            intensity=max(trend_weight, reversion_weight, chop),
            components={
                "trend_pressure_up": trend_up,
                "trend_pressure_down": trend_down,
                "trend_strategy_influence": trend_influence,
                "momentum_strategy_influence": momentum_influence,
                "mean_reversion_strategy_influence": mean_reversion_influence,
                "chop_risk": chop,
            },
        )
    ]


def _write_specialist_live_meta_outputs(
    *,
    production_summary: dict[str, Any] | None,
    config_dir: str,
    meta_projection_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from dual_model_forecaster.fusion import _bucket_confidence, _bucket_direction
    from model_assembly.common import load_doc
    from walk_forward_predictions import _load_production_horizons

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    roles_cfg = load_doc(Path(config_dir) / "roles.yaml")
    artifact_root = Path(refit_cfg["artifact_root"])
    live_root = artifact_root / "live_forecast"
    live_root.mkdir(parents=True, exist_ok=True)
    horizons = _load_production_horizons(config_dir)

    live_forecast = (production_summary or {}).get("live_forecast")
    if not live_forecast:
        latest_path = live_root / "latest.json"
        if latest_path.exists():
            live_forecast = json.loads(latest_path.read_text())
    forecast_rows = (
        _compact_live_forecast_rows(live_forecast, base_close=1.0, return_type="log")
        if live_forecast
        else pd.DataFrame()
    )
    forecast_q50 = {
        int(row["horizon"]): _float_or_none(row.get("q50_return")) or 0.0
        for _, row in forecast_rows.iterrows()
    }

    heads = roles_cfg.get("heads", {})
    specialist_order = ["structure", "environment", "edges", "movement", "liquidation"]
    state_records: list[dict[str, Any]] = []
    semantic_records: list[dict[str, Any]] = []
    indicator_records: list[dict[str, Any]] = []
    horizon_records: list[dict[str, Any]] = []
    states_by_bucket: dict[str, pd.DataFrame] = {}

    for specialist in specialist_order:
        frame = _load_specialist_live_state(artifact_root / specialist / "live_history_states.csv")
        if frame.empty:
            continue
        states_by_bucket[specialist] = frame
        latest = frame.tail(1)
        row = latest.iloc[0]
        timestamp = pd.Timestamp(latest.index[-1])
        direction = _float_or_none(_bucket_direction(specialist, latest).iloc[-1]) or 0.0
        confidence = _float_or_none(_bucket_confidence(specialist, latest).iloc[-1]) or 0.0
        amplitude = _float_or_none(row.get("summary_amplitude")) or abs(direction)
        dispersion = _float_or_none(row.get("summary_dispersion")) or 0.0
        roles = [role for role, buckets in heads.items() if specialist in set(buckets)]
        semantic_columns = [column for column in _specialist_semantic_columns(specialist) if column in frame.columns]
        semantic_values = {
            column: _float_or_none(row.get(column))
            for column in semantic_columns
        }
        state_record = {
            "timestamp": timestamp,
            "specialist": specialist,
            "meta_roles": ",".join(roles),
            "role_center": "center" in roles,
            "role_width": "width" in roles,
            "role_tail": "tail" in roles,
            "direction": direction,
            "confidence": confidence,
            "amplitude": amplitude,
            "dispersion": dispersion,
            **semantic_values,
        }
        state_records.append(state_record)
        indicator_records.extend(
            _specialist_indicator_rows(
                specialist=specialist,
                timestamp=timestamp,
                row=row,
                roles=roles,
                direction=direction,
                confidence=confidence,
            )
        )
        for semantic_name, value in semantic_values.items():
            semantic_records.append(
                {
                    "timestamp": timestamp,
                    "specialist": specialist,
                    "meta_roles": ",".join(roles),
                    "semantic_prediction": semantic_name,
                    "value": value,
                }
            )
        for horizon in horizons:
            horizon_scale = float(np.clip(np.sqrt(max(int(horizon), 1) / 15.0), 0.35, 1.35))
            return_lean_proxy = direction * confidence * horizon_scale
            model_q50 = forecast_q50.get(int(horizon), 0.0)
            horizon_records.append(
                {
                    "timestamp": timestamp,
                    "specialist": specialist,
                    "horizon": int(horizon),
                    "meta_roles": ",".join(roles),
                    "role_center": "center" in roles,
                    "role_width": "width" in roles,
                    "role_tail": "tail" in roles,
                    "direction": direction,
                    "confidence": confidence,
                    "return_lean_proxy": return_lean_proxy,
                    "model_q50_return": model_q50,
                    "lean_vs_model_q50": return_lean_proxy - model_q50,
                }
            )

    if not state_records:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "missing specialist live_history_states.csv files",
        }

    state_frame = pd.DataFrame(state_records).replace([np.inf, -np.inf], np.nan)
    semantic_frame = pd.DataFrame(semantic_records).replace([np.inf, -np.inf], np.nan)
    indicator_frame = pd.DataFrame(indicator_records).replace([np.inf, -np.inf], np.nan)
    horizon_frame = pd.DataFrame(horizon_records).replace([np.inf, -np.inf], np.nan)

    state_csv = live_root / "specialist_live_predictions.csv"
    semantic_csv = live_root / "specialist_semantic_predictions_long.csv"
    indicator_csv = live_root / "specialist_meta_indicators.csv"
    horizon_csv = live_root / "specialist_meta_inputs_by_horizon.csv"
    json_path = live_root / "specialist_live_predictions.json"
    chart_path = live_root / "specialist_live_predictions.png"
    indicator_chart_path = live_root / "specialist_meta_indicators.png"
    indicator_summary_path = live_root / "specialist_meta_indicators.md"
    summary_path = live_root / "specialist_live_predictions.md"

    state_frame.to_csv(state_csv, index=False)
    semantic_frame.to_csv(semantic_csv, index=False)
    indicator_frame.to_csv(indicator_csv, index=False)
    horizon_frame.to_csv(horizon_csv, index=False)
    _write_specialist_live_chart(
        state_frame=state_frame,
        horizon_frame=horizon_frame,
        chart_path=chart_path,
    )
    _write_specialist_indicator_chart(
        indicator_frame=indicator_frame,
        chart_path=indicator_chart_path,
    )
    payload = {
        "enabled": True,
        "status": "written",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "role_map": roles_cfg.get("role_map_name"),
        "interface": roles_cfg.get("interface_name"),
        "meta_projection": meta_projection_summary or {},
        "files": {
            "state_csv": str(state_csv.resolve()),
            "semantic_csv": str(semantic_csv.resolve()),
            "indicator_csv": str(indicator_csv.resolve()),
            "horizon_csv": str(horizon_csv.resolve()),
            "json": str(json_path.resolve()),
            "chart": str(chart_path.resolve()),
            "indicator_chart": str(indicator_chart_path.resolve()),
            "indicator_summary_md": str(indicator_summary_path.resolve()),
            "summary_md": str(summary_path.resolve()),
        },
        "specialists": state_frame.where(pd.notna(state_frame), None).to_dict(orient="records"),
        "indicators": indicator_frame.where(pd.notna(indicator_frame), None).to_dict(orient="records"),
        "horizon_inputs": horizon_frame.where(pd.notna(horizon_frame), None).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")

    lines = [
        "# Specialist Live Predictions",
        "",
        f"- Role map: {roles_cfg.get('role_map_name')}",
        f"- Interface: {roles_cfg.get('interface_name')}",
        "",
        "These are the specialist semantic predictions and role inputs used to interpret the live meta/fusion forecast.",
        "The `return_lean_proxy` is an interpretation layer, not a replacement for the trained quantile forecast.",
        "",
        "![Specialist live predictions](specialist_live_predictions.png)",
        "",
        "| Specialist | Meta Roles | Direction | Confidence | Primary Lean |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in state_frame.sort_values("specialist").iterrows():
        direction = _float_or_none(row.get("direction")) or 0.0
        primary = "upside" if direction > 0.05 else "downside" if direction < -0.05 else "neutral"
        lines.append(
            f"| {row['specialist']} | {row.get('meta_roles', '')} | "
            f"{_format_float(row.get('direction'), '.3f')} | "
            f"{_format_float(row.get('confidence'), '.3f')} | {primary} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Specialist state CSV: {state_csv}",
            f"- Semantic long CSV: {semantic_csv}",
            f"- Separate specialist indicators CSV: {indicator_csv}",
            f"- Horizon meta-input CSV: {horizon_csv}",
        ]
    )
    summary_path.write_text("\n".join(lines).rstrip() + "\n")

    indicator_lines = [
        "# Specialist Meta Indicators",
        "",
        "Separate indicator outputs derived from each specialist's latest live state.",
        "",
        "![Specialist meta indicators](specialist_meta_indicators.png)",
        "",
        "| Specialist | Indicator | Upside | Downside | Net | Confidence | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in indicator_frame.sort_values(["specialist", "indicator"]).iterrows():
        indicator_lines.append(
            f"| {row['specialist']} | {row['indicator']} | "
            f"{_format_float(row.get('upside_score'), '.3f')} | "
            f"{_format_float(row.get('downside_score'), '.3f')} | "
            f"{_format_float(row.get('net_score'), '.3f')} | "
            f"{_format_float(row.get('confidence'), '.3f')} | "
            f"{row.get('interpretation', '')} |"
        )
    indicator_lines.extend(
        [
            "",
            "Key examples:",
            "",
            "- `liquidation_upside_downside_potential`: potential liquidation pressure toward upside versus downside.",
            "- `edges_flow_reversion`: stretch-driven flow reversion pressure, where downside stretch implies upside reversion and upside stretch implies downside reversion.",
        ]
    )
    indicator_summary_path.write_text("\n".join(indicator_lines).rstrip() + "\n")
    return {
        "enabled": True,
        "status": "written",
        "state_csv": str(state_csv.resolve()),
        "semantic_csv": str(semantic_csv.resolve()),
        "indicator_csv": str(indicator_csv.resolve()),
        "horizon_csv": str(horizon_csv.resolve()),
        "json": str(json_path.resolve()),
        "chart": str(chart_path.resolve()),
        "indicator_chart": str(indicator_chart_path.resolve()),
        "indicator_summary_md": str(indicator_summary_path.resolve()),
        "summary_md": str(summary_path.resolve()),
        "specialist_count": int(len(state_frame)),
        "indicator_count": int(len(indicator_frame)),
        "horizon_rows": int(len(horizon_frame)),
    }


def _write_specialist_live_chart(
    *,
    state_frame: pd.DataFrame,
    horizon_frame: pd.DataFrame,
    chart_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    specialists = state_frame["specialist"].astype(str).tolist()
    directions = pd.to_numeric(state_frame["direction"], errors="coerce").fillna(0.0)
    confidences = pd.to_numeric(state_frame["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    colors = ["#16a34a" if value >= 0.0 else "#dc2626" for value in directions]

    fig, (ax_dir, ax_horizon) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        dpi=150,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    ypos = np.arange(len(specialists))
    ax_dir.barh(ypos, directions, color=colors, alpha=0.82)
    ax_dir.axvline(0.0, color="#111827", linewidth=0.8)
    ax_dir.set_yticks(ypos, specialists)
    ax_dir.set_xlabel("Direction score")
    ax_dir.set_title("Latest Specialist Direction", loc="left", fontsize=12, weight="bold")
    ax_dir.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    for pos, confidence in zip(ypos, confidences):
        ax_dir.text(0.02, pos, f"conf {confidence:.2f}", va="center", fontsize=8, color="#374151")

    if not horizon_frame.empty:
        for specialist in specialists:
            subset = horizon_frame.loc[horizon_frame["specialist"].astype(str) == specialist].sort_values("horizon")
            if subset.empty:
                continue
            ax_horizon.plot(
                subset["horizon"].astype(int),
                pd.to_numeric(subset["return_lean_proxy"], errors="coerce") * 100.0,
                marker="o",
                linewidth=1.35,
                label=specialist,
            )
    ax_horizon.axhline(0.0, color="#111827", linewidth=0.8)
    ax_horizon.set_xlabel("Horizon, days")
    ax_horizon.set_ylabel("Return lean proxy, pp")
    ax_horizon.set_title("Specialist Lean By Forecast Horizon", loc="left", fontsize=12, weight="bold")
    ax_horizon.grid(True, color="#e5e7eb", linewidth=0.8)
    ax_horizon.legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)


def _write_specialist_indicator_chart(
    *,
    indicator_frame: pd.DataFrame,
    chart_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    if indicator_frame.empty:
        return
    frame = indicator_frame.copy().sort_values(["specialist", "indicator"])
    labels = [f"{row.specialist}\n{row.indicator}" for row in frame.itertuples()]
    upside = pd.to_numeric(frame["upside_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    downside = pd.to_numeric(frame["downside_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    net = pd.to_numeric(frame["net_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

    fig, (ax_balance, ax_net) = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.6),
        dpi=150,
        gridspec_kw={"height_ratios": [2.3, 1.3]},
        constrained_layout=True,
    )
    y = np.arange(len(frame))
    ax_balance.barh(y, upside, color="#16a34a", alpha=0.72, label="Upside")
    ax_balance.barh(y, -downside, color="#dc2626", alpha=0.72, label="Downside")
    ax_balance.axvline(0.0, color="#111827", linewidth=0.8)
    ax_balance.set_yticks(y, labels, fontsize=8)
    ax_balance.set_xlim(-1.0, 1.0)
    ax_balance.set_xlabel("Directional indicator score")
    ax_balance.set_title("Specialist Meta Indicators: Upside vs Downside", loc="left", fontsize=12, weight="bold")
    ax_balance.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax_balance.legend(loc="upper right", ncols=2, frameon=False, fontsize=8)

    colors = ["#16a34a" if value >= 0.0 else "#dc2626" for value in net]
    ax_net.bar(range(len(frame)), net, color=colors, alpha=0.80)
    ax_net.axhline(0.0, color="#111827", linewidth=0.8)
    ax_net.set_xticks(range(len(frame)), frame["specialist"].astype(str), rotation=0)
    ax_net.set_ylabel("Net")
    ax_net.set_title("Net Indicator Lean", loc="left", fontsize=10)
    ax_net.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)


def _record_current_walk_forward_rows(
    *,
    production_cycle_summary: dict[str, Any],
    config_dir: str,
    db_path: str,
    table_name: str,
) -> dict[str, Any]:
    from dual_model_forecaster.data import load_forecast_data
    from walk_forward_predictions import (
        _load_refit_base_config,
        _rows_from_cycle_summary,
        _save_prediction_rows,
    )

    if "production" not in production_cycle_summary:
        return {"rows": 0, "status": "skipped_no_production"}

    base_config = _load_refit_base_config(config_dir)
    full_data = load_forecast_data(base_config)
    cycle_summary = dict(production_cycle_summary)
    cycle_summary["as_of_row_count"] = int(len(full_data.close))
    rows = _rows_from_cycle_summary(
        cycle_summary=cycle_summary,
        base_config=base_config,
        full_close=full_data.close,
        config_dir=config_dir,
    )
    saved = _save_prediction_rows(db_path, table_name, rows)
    return {
        "rows": int(len(rows)),
        "saved_rows": int(saved),
        "as_of": str(pd.Timestamp(rows["as_of"].iloc[0])) if not rows.empty else None,
        "status": "saved",
    }


def _run_walk_forward_to_latest(
    *,
    config_dir: str,
    db_path: str,
    table_name: str,
    force: bool = False,
    limit: int | None = None,
    rebuild_feature_submodels: bool = False,
    reuse_cached_feature_submodels: bool = False,
    quiet: bool = False,
    skip_charts: bool = True,
    skip_evaluation: bool = True,
    skip_meta_synthesis: bool = True,
) -> dict[str, Any]:
    from argparse import Namespace
    from walk_forward_predictions import (
        DEFAULT_CHART_DIR,
        DEFAULT_DB_PATH,
        DEFAULT_TABLE_NAME,
        META_DEFAULT_MIN_HISTORY,
        META_DEFAULT_SELECTION_WINDOW,
        run_walk_forward,
    )

    args = Namespace(
        config_dir=config_dir,
        db_path=db_path or DEFAULT_DB_PATH,
        table=table_name or DEFAULT_TABLE_NAME,
        chart_dir=DEFAULT_CHART_DIR,
        start=None,
        end=None,
        limit=limit,
        min_history_rows=None,
        force=force,
        save_artifacts=False,
        rebuild_feature_submodels=bool(rebuild_feature_submodels),
        reuse_cached_feature_submodels=bool(reuse_cached_feature_submodels),
        skip_charts=bool(skip_charts),
        skip_evaluation=bool(skip_evaluation),
        evaluate_only=False,
        evaluation_dir=None,
        skip_garch_baselines=True,
        skip_kalman_synthesis=False,
        skip_meta_synthesis=bool(skip_meta_synthesis),
        meta_min_history=META_DEFAULT_MIN_HISTORY,
        meta_selection_window=META_DEFAULT_SELECTION_WINDOW,
        kalman_min_history=240,
        baseline_history_min=180,
        baseline_regime_history_min=60,
        garch_min_obs=500,
        garch_refit_interval=30,
        bootstrap_samples=0,
        fee_bps=5.0,
        slippage_bps=5.0,
        dry_run=False,
        stop_on_error=False,
        emit_summary=False,
        quiet=bool(quiet),
    )
    return run_walk_forward(args)


def _train_jepa_for_live_data(
    *,
    config_dir: str,
    jepa_specialists: list[str] | tuple[str, ...] | None = None,
    jepa_horizons: list[int] | tuple[int, ...] | None = None,
    jepa_context_length: int | None = None,
    jepa_latent_dim: int | None = None,
    jepa_max_epochs: int | None = None,
    jepa_batch_size: int | None = None,
    jepa_seed: int | None = None,
    jepa_device: str = "cpu",
) -> dict[str, Any]:
    from dual_model_forecaster.jepa.config import load_jepa_config
    from dual_model_forecaster.jepa.pretrain import (
        DEFAULT_JEPA_SPECIALISTS,
        run_jepa_pretraining,
    )
    from model_assembly.common import load_doc

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    env_config = load_jepa_config()
    context_lengths = tuple(int(value) for value in env_config.context_lengths)
    context_length = int(
        jepa_context_length
        if jepa_context_length is not None
        else (context_lengths[0] if context_lengths else 128)
    )
    horizons = tuple(int(value) for value in (jepa_horizons or env_config.horizons))
    specialists = tuple(str(value) for value in (jepa_specialists or DEFAULT_JEPA_SPECIALISTS))

    return run_jepa_pretraining(
        base_config_path=refit_cfg["base_config"],
        specialists=specialists,
        horizons=horizons,
        context_length=context_length,
        latent_dim=int(jepa_latent_dim if jepa_latent_dim is not None else env_config.latent_dim),
        checkpoint_dir=env_config.checkpoint_dir,
        artifact_dir=env_config.artifact_dir,
        max_epochs=jepa_max_epochs,
        batch_size=jepa_batch_size,
        seed=jepa_seed,
        device=str(jepa_device),
    )


def _train_world_jepa_for_live_data(
    *,
    config_path: str = "configs/world_jepa.json",
    world_epochs: int | None = None,
    router_epochs: int | None = None,
    batch_size: int | None = None,
    world_stride: int = 1,
    device: str = "cpu",
    smoke: bool = False,
    refresh_feature_audit: bool = True,
) -> dict[str, Any]:
    """Run the new five-world attention JEPA without touching legacy outputs."""

    from dual_model_forecaster.world_jepa.pipeline import (
        RuntimeOverrides,
        run_world_jepa_pipeline,
    )

    runtime = RuntimeOverrides(
        world_epochs=world_epochs or (1 if smoke else None),
        router_epochs=router_epochs or (2 if smoke else None),
        batch_size=batch_size,
        world_stride=max(int(world_stride), 4 if smoke else 1),
        maximum_world_batches_per_epoch=8 if smoke else None,
        maximum_router_batches_per_epoch=12 if smoke else None,
        device=str(device),
        validated_test_suite=False,
    )
    return run_world_jepa_pipeline(
        config_path,
        runtime=runtime,
        refresh_feature_audit=refresh_feature_audit,
    )


def _jepa_config_from_training_summary(training_summary: dict[str, Any] | None):
    if not training_summary or training_summary.get("status") != "ok":
        return None
    payload = dict(training_summary.get("config") or {})
    if not payload:
        return None

    from dual_model_forecaster.jepa.config import JEPAConfig

    def _int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
        if value is None:
            return default
        if isinstance(value, str):
            parts = value.replace(",", " ").split()
        else:
            try:
                parts = list(value)
            except TypeError:
                parts = [value]
        parsed: list[int] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except Exception:
                continue
        return tuple(parsed) if parsed else default

    return JEPAConfig(
        enabled=True,
        mode=str(payload.get("mode") or "passive"),
        use_in_neural=bool(payload.get("use_in_neural", False)),
        use_in_meta=bool(payload.get("use_in_meta", False)),
        checkpoint_dir=Path(payload.get("checkpoint_dir") or training_summary.get("checkpoint_dir") or "artifacts/models/jepa"),
        artifact_dir=Path(payload.get("artifact_dir") or training_summary.get("artifact_dir") or "artifacts/final/live_forecast/jepa"),
        horizons=_int_tuple(payload.get("horizons"), (1, 3, 7, 15)),
        context_lengths=_int_tuple(payload.get("context_lengths"), (128,)),
        latent_dim=int(payload.get("latent_dim", 64)),
        max_epochs=int(payload.get("max_epochs", 5)),
        batch_size=int(payload.get("batch_size", 128)),
        chart_lookback_days=int(payload.get("chart_lookback_days", 365)),
        chart_batch_size=int(payload.get("chart_batch_size", 128)),
        seed=int(payload.get("seed", 7)),
        lambda_var=float(payload.get("lambda_var", 1.0)),
        lambda_cov=float(payload.get("lambda_cov", 0.04)),
        lambda_smooth=float(payload.get("lambda_smooth", 0.02)),
    )


def _jepa_live_meta_selection_path(config_dir: str) -> Path:
    from model_assembly.common import load_doc

    refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
    return Path(refit_cfg["artifact_root"]) / "live_forecast" / "meta_synthesis_jepa_selected.csv"


def _ensure_jepa_live_meta_selection(
    *,
    config_dir: str,
    db_path: str,
    table_name: str,
    force: bool = False,
    limit: int | None = None,
    jepa_config: Any | None = None,
) -> dict[str, Any]:
    selected_path = _jepa_live_meta_selection_path(config_dir)
    if selected_path.exists() and not force:
        return {
            "enabled": True,
            "status": "present",
            "selection_path": str(selected_path.resolve()),
        }

    env_keys = [
        "JEPA_USE_IN_META",
        "JEPA_ENABLED",
        "JEPA_MODE",
        "JEPA_CHECKPOINT_DIR",
        "JEPA_ARTIFACT_DIR",
        "JEPA_HORIZONS",
        "JEPA_CONTEXT_LENGTHS",
        "JEPA_LATENT_DIM",
        "JEPA_MAX_EPOCHS",
        "JEPA_BATCH_SIZE",
        "JEPA_SEED",
    ]
    previous_env = {key: os.environ.get(key) for key in env_keys}
    os.environ["JEPA_USE_IN_META"] = "true"
    if jepa_config is not None and getattr(jepa_config, "is_active", False):
        os.environ["JEPA_ENABLED"] = "true"
        os.environ["JEPA_MODE"] = str(jepa_config.mode)
        os.environ["JEPA_CHECKPOINT_DIR"] = str(jepa_config.checkpoint_dir)
        os.environ["JEPA_ARTIFACT_DIR"] = str(jepa_config.artifact_dir)
        os.environ["JEPA_HORIZONS"] = ",".join(str(int(value)) for value in jepa_config.horizons)
        os.environ["JEPA_CONTEXT_LENGTHS"] = ",".join(str(int(value)) for value in jepa_config.context_lengths)
        os.environ["JEPA_LATENT_DIM"] = str(int(jepa_config.latent_dim))
        os.environ["JEPA_MAX_EPOCHS"] = str(int(jepa_config.max_epochs))
        os.environ["JEPA_BATCH_SIZE"] = str(int(jepa_config.batch_size))
        os.environ["JEPA_SEED"] = str(int(jepa_config.seed))
    try:
        from dual_model_forecaster.data import load_forecast_data
        from walk_forward_predictions import (
            META_DEFAULT_MIN_HISTORY,
            META_DEFAULT_SELECTION_WINDOW,
            _live_forecast_dir,
            _load_production_horizons,
            _load_refit_base_config,
            build_meta_synthesis_artifacts,
        )

        base_config = _load_refit_base_config(config_dir)
        data_bundle = load_forecast_data(base_config)
        horizons = _load_production_horizons(config_dir)
        return_type = str(base_config["data"]["return_type"])
        live_artifacts = build_meta_synthesis_artifacts(
            db_path=db_path,
            table_name=table_name,
            chart_dir=_live_forecast_dir(config_dir),
            horizons=horizons,
            data_bundle=data_bundle,
            return_type=return_type,
            min_history=META_DEFAULT_MIN_HISTORY,
            selection_window=META_DEFAULT_SELECTION_WINDOW,
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return {
        "enabled": True,
        "status": "written" if selected_path.exists() else "missing_after_meta_synthesis",
        "selection_path": str(selected_path.resolve()),
        "live_jepa_meta_artifacts": live_artifacts.get("jepa_artifacts", {}),
        "jepa_meta_warnings": live_artifacts.get("jepa_warnings", []),
    }


def _compact_walk_forward_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    return {
        "db_path": summary.get("db_path"),
        "table": summary.get("table"),
        "required_rows": summary.get("required_rows"),
        "total_as_of_dates": summary.get("total_as_of_dates"),
        "first_as_of": summary.get("first_as_of"),
        "last_as_of": summary.get("last_as_of"),
        "saved_rows": summary.get("saved_rows"),
        "skipped_days": summary.get("skipped_days"),
        "failed_days": summary.get("failed_days"),
    }


def _compact_summary_for_stdout(summary: dict[str, Any]) -> dict[str, Any]:
    production = summary.get("production", {})
    clean_outputs = summary.get("clean_outputs", {})
    jepa_training = summary.get("jepa_training", {})
    world_jepa_training = summary.get("world_jepa_training", {})
    return {
        "manual_macro_update": summary.get("manual_macro_update"),
        "data_refresh_gate": summary.get("data_refresh_gate"),
        "data_update": summary.get("data_update"),
        "checkonchain_choppiness": summary.get("checkonchain_choppiness"),
        "liquidation_signals": summary.get("liquidation_signals"),
        "category_update": summary.get("category_update"),
        "production": {
            "as_of_timestamp": production.get("as_of_timestamp"),
            "candidate": production.get("candidate"),
            "fusion_calibration_method": production.get("fusion_calibration_method"),
            "selected_calibrator": production.get("selected_calibrator"),
            "live_forecast_plot_path": production.get("live_forecast_plot_path"),
        }
        if production
        else None,
        "walk_forward_current": summary.get("walk_forward_current"),
        "walk_forward": _compact_walk_forward_summary(summary.get("walk_forward")),
        "jepa_training": {
            "status": jepa_training.get("status"),
            "artifact_dir": jepa_training.get("artifact_dir"),
            "checkpoint_dir": jepa_training.get("checkpoint_dir"),
            "trained_specialists": jepa_training.get("trained_specialists"),
            "skipped_specialists": jepa_training.get("skipped_specialists"),
            "metric_rows": jepa_training.get("metric_rows"),
        }
        if jepa_training
        else None,
        "world_jepa_training": {
            "status": world_jepa_training.get("status"),
            "mode": world_jepa_training.get("mode"),
            "promoted": world_jepa_training.get("promoted"),
            "source_last_timestamp": world_jepa_training.get("source_last_timestamp"),
            "promotion_gates": world_jepa_training.get("promotion_gates"),
            "live_shadow_forecast": world_jepa_training.get("live_shadow_forecast"),
            "report_root": world_jepa_training.get("report_root"),
        }
        if world_jepa_training
        else None,
        "meta_learner": {
            "artifacts": (summary.get("meta_learner") or {}).get("artifacts"),
            "chart_paths": (summary.get("meta_learner") or {}).get("chart_paths"),
            "jepa_enabled": (summary.get("meta_learner") or {}).get("jepa_enabled"),
            "jepa_warnings": (summary.get("meta_learner") or {}).get("jepa_warnings"),
            "selection": (summary.get("meta_learner") or {}).get("selection"),
        }
        if summary.get("meta_learner")
        else None,
        "live_meta_projection": summary.get("live_meta_projection"),
        "specialist_live_outputs": summary.get("specialist_live_outputs"),
        "clean_outputs": {
            "latest_projection": clean_outputs.get("latest_projection"),
            "kalman_scope_projection_csv": clean_outputs.get("kalman_scope_projection_csv"),
            "kalman_projection_chart": clean_outputs.get("kalman_projection_chart"),
            "historical_charts": clean_outputs.get("historical_charts"),
            "projection_aggregates": clean_outputs.get("projection_aggregates"),
            "projection_history_tail": clean_outputs.get("projection_history_tail"),
            "projection_summary": clean_outputs.get("projection_summary"),
            "latest_as_of": clean_outputs.get("latest_as_of"),
            "removed_legacy_live_files": len(clean_outputs.get("removed_legacy_live_files", [])),
        }
        if clean_outputs
        else None,
        "jepa": summary.get("jepa"),
        "jepa_live_meta_prepass": summary.get("jepa_live_meta_prepass"),
        "jepa_live_meta": summary.get("jepa_live_meta"),
    }


def run_daily_production_cycle(
    config_dir: str = str(FINAL_CONFIG_DIR),
    refresh_data: bool = True,
    gate_data_refresh: bool = True,
    rebuild_categories: bool = True,
    run_production_forecast: bool = True,
    run_walk_forward_update: bool = True,
    clean_live_outputs: bool = True,
    as_of_timestamp: str | None = None,
    save_artifacts: bool = True,
    price_db_path: str = DEFAULT_PRICE_DB_PATH,
    price_table_name: str = DEFAULT_PRICE_TABLE,
    category_db_path: str = DEFAULT_CATEGORY_DB_PATH,
    ta_output_db_path: str = DEFAULT_TA_OUTPUT_DB_PATH,
    onchain_db_path: str = DEFAULT_ONCHAIN_DB_PATH,
    macro_db_path: str = DEFAULT_MACRO_DB_PATH,
    macro_table_name: str = DEFAULT_MACRO_TABLE,
    manual_dir: str = str(DEFAULT_MANUAL_DIR),
    walk_forward_db_path: str = DEFAULT_WALK_FORWARD_DB_PATH,
    walk_forward_table: str = DEFAULT_WALK_FORWARD_TABLE,
    force_walk_forward: bool = False,
    walk_forward_limit: int | None = None,
    walk_forward_rebuild_feature_submodels: bool = False,
    walk_forward_reuse_cached_feature_submodels: bool = False,
    quiet_walk_forward: bool = False,
    train_jepa: bool = False,
    train_world_jepa: bool = False,
    world_jepa_config_path: str = "configs/world_jepa.json",
    world_jepa_world_epochs: int | None = None,
    world_jepa_router_epochs: int | None = None,
    world_jepa_batch_size: int | None = None,
    world_jepa_stride: int = 1,
    world_jepa_device: str = "cpu",
    world_jepa_smoke: bool = False,
    refresh_world_jepa_feature_audit: bool = True,
    jepa_specialists: list[str] | tuple[str, ...] | None = None,
    jepa_horizons: list[int] | tuple[int, ...] | None = None,
    jepa_context_length: int | None = None,
    jepa_latent_dim: int | None = None,
    jepa_max_epochs: int | None = None,
    jepa_batch_size: int | None = None,
    jepa_seed: int | None = None,
    jepa_device: str = "cpu",
    run_meta_learner: bool = True,
    meta_min_history: int | None = None,
    meta_selection_window: int | None = None,
    write_specialist_live_outputs: bool = True,
    refresh_manual_macro: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    manual_macro_refreshed = False
    if refresh_manual_macro and as_of_timestamp is None:
        summary["manual_macro_update"] = update_manual_macro_data(
            manual_dir=manual_dir,
            db_path=macro_db_path,
            table_name=macro_table_name,
        )
        manual_macro_refreshed = True

    if refresh_data:
        gate = _data_refresh_gate(
            price_db_path=price_db_path,
            price_table_name=price_table_name,
        )
        summary["data_refresh_gate"] = gate
        should_pull_data = not gate_data_refresh or not gate["fresh_enough"]
        if should_pull_data:
            data_update = get_btc_data(
                price_db_path=price_db_path,
                price_table_name=price_table_name,
                onchain_db_path=onchain_db_path,
                macro_db_path=macro_db_path,
                macro_table_name=macro_table_name,
                manual_dir=manual_dir,
                include_manual_macro=not manual_macro_refreshed,
            )
            summary["data_update"] = data_update
            summary["checkonchain_choppiness"] = data_update.get("checkonchain_choppiness")
            summary["liquidation_signals"] = data_update.get("liquidation_signals")
        else:
            summary["data_update"] = {
                "skipped": True,
                "reason": "price_data_fresh_enough",
                "latest_price_day": gate["latest_price_day"],
                "required_latest_day": gate["required_latest_day"],
            }
            summary["checkonchain_choppiness"] = sync_choppiness_index(db_path=onchain_db_path)
            summary["liquidation_signals"] = sync_liquidation_signals()
    else:
        summary["data_refresh_gate"] = {"action": "skip_data_pull", "reason": "disabled_by_cli"}
    if rebuild_categories:
        summary["category_update"] = compute_categories(
            price_db_path=price_db_path,
            category_db_path=category_db_path,
            ta_output_db_path=ta_output_db_path,
            onchain_db_path=onchain_db_path,
            macro_db_path=macro_db_path,
            as_of_timestamp=as_of_timestamp,
        )

    if run_walk_forward_update and as_of_timestamp is None:
        summary["walk_forward"] = _run_walk_forward_to_latest(
            config_dir=config_dir,
            db_path=walk_forward_db_path,
            table_name=walk_forward_table,
            force=force_walk_forward,
            limit=walk_forward_limit,
            rebuild_feature_submodels=walk_forward_rebuild_feature_submodels,
            reuse_cached_feature_submodels=walk_forward_reuse_cached_feature_submodels,
            quiet=quiet_walk_forward,
        )

    if train_jepa and as_of_timestamp is None:
        summary["jepa_training"] = _train_jepa_for_live_data(
            config_dir=config_dir,
            jepa_specialists=jepa_specialists,
            jepa_horizons=jepa_horizons,
            jepa_context_length=jepa_context_length,
            jepa_latent_dim=jepa_latent_dim,
            jepa_max_epochs=jepa_max_epochs,
            jepa_batch_size=jepa_batch_size,
            jepa_seed=jepa_seed,
            jepa_device=jepa_device,
        )
    elif train_jepa:
        summary["jepa_training"] = {
            "status": "skipped",
            "reason": "disabled_for_historical_as_of_run",
        }

    if train_world_jepa and as_of_timestamp is None:
        summary["world_jepa_training"] = _train_world_jepa_for_live_data(
            config_path=world_jepa_config_path,
            world_epochs=world_jepa_world_epochs,
            router_epochs=world_jepa_router_epochs,
            batch_size=world_jepa_batch_size,
            world_stride=world_jepa_stride,
            device=world_jepa_device,
            smoke=world_jepa_smoke,
            refresh_feature_audit=refresh_world_jepa_feature_audit,
        )
    elif train_world_jepa:
        summary["world_jepa_training"] = {
            "status": "skipped",
            "reason": "disabled_for_historical_as_of_run",
        }

    if run_production_forecast:
        production = run_refit_full_history(
            config_dir=config_dir,
            as_of_timestamp=as_of_timestamp,
            save_artifacts=save_artifacts,
        )
        summary["production"] = {
            "refit_timestamp_utc": production["refit_timestamp_utc"],
            "as_of_timestamp": production["as_of_timestamp"],
            "candidate": production["candidate"]["name"],
            "fusion_calibration_method": production["fusion_calibration_method"],
            "selected_calibrator": production["selected_calibrator"],
            "calibration_score_comparison": production["calibration_score_comparison"],
            "live_forecast": production["live_forecast"],
            "live_forecast_plot_path": production["live_forecast_plot_path"],
            "width_diagnostics": production["width_diagnostics"],
            "lower_tail_diagnostics": production["lower_tail_diagnostics"],
            "upper_tail_diagnostics": production["upper_tail_diagnostics"],
            "artifact_root": production["artifact_root"],
        }

    if run_walk_forward_update and as_of_timestamp is None and "production" in summary:
        summary["walk_forward_current"] = _record_current_walk_forward_rows(
            production_cycle_summary=summary,
            config_dir=config_dir,
            db_path=walk_forward_db_path,
            table_name=walk_forward_table,
        )

    if clean_live_outputs and save_artifacts and as_of_timestamp is None:
        summary["clean_outputs"] = _write_clean_live_outputs(
            production_summary=summary.get("production"),
            config_dir=config_dir,
            walk_forward_db_path=walk_forward_db_path,
            walk_forward_table=walk_forward_table,
        )
    if save_artifacts and as_of_timestamp is None:
        from dual_model_forecaster.jepa.config import load_jepa_config
        from dual_model_forecaster.jepa.meta_synthesis import use_live_jepa_meta

        jepa_config = load_jepa_config()
        live_artifact_config = _jepa_config_from_training_summary(summary.get("jepa_training")) or jepa_config
        if live_artifact_config.is_active:
            from dual_model_forecaster.config import load_config
            from dual_model_forecaster.data import load_forecast_data
            from dual_model_forecaster.jepa.artifacts import write_live_jepa_artifacts
            from model_assembly.common import load_doc

            refit_cfg = load_doc(Path(config_dir) / "refit.yaml")
            base_config = load_config(refit_cfg["base_config"])
            data_bundle = load_forecast_data(base_config)
            summary["jepa"] = write_live_jepa_artifacts(
                data_bundle=data_bundle,
                config=live_artifact_config,
            )
        if use_live_jepa_meta():
            if not live_artifact_config.is_active:
                summary["jepa_live_meta"] = {
                    "enabled": True,
                    "status": "skipped",
                    "reason": "JEPA_USE_LIVE_META=true but JEPA is inactive; set JEPA_MODE=passive/frozen",
                }
            else:
                if run_walk_forward_update:
                    summary["jepa_live_meta_prepass"] = _ensure_jepa_live_meta_selection(
                        config_dir=config_dir,
                        db_path=walk_forward_db_path,
                        table_name=walk_forward_table,
                        force=force_walk_forward,
                        limit=walk_forward_limit,
                        jepa_config=live_artifact_config,
                    )
                else:
                    summary["jepa_live_meta_prepass"] = {
                        "enabled": True,
                        "status": "skipped",
                        "reason": "walk-forward update disabled",
                    }
                summary["jepa_live_meta"] = _write_experimental_jepa_live_meta_outputs(
                    production_summary=summary.get("production"),
                    config_dir=config_dir,
                    jepa_latest_csv=(summary.get("jepa") or {}).get("latest_csv"),
                )
        if run_meta_learner:
            summary["meta_learner"] = _run_meta_learner_for_live_forecast(
                config_dir=config_dir,
                db_path=walk_forward_db_path,
                table_name=walk_forward_table,
                jepa_config=live_artifact_config,
                min_history=meta_min_history,
                selection_window=meta_selection_window,
            )
            summary["live_meta_projection"] = _write_live_meta_projection_outputs(
                production_summary=summary.get("production"),
                config_dir=config_dir,
                meta_learner_summary=summary.get("meta_learner"),
                jepa_latest_csv=(summary.get("jepa") or {}).get("latest_csv"),
            )
        if write_specialist_live_outputs and "production" in summary:
            summary["specialist_live_outputs"] = _write_specialist_live_meta_outputs(
                production_summary=summary.get("production"),
                config_dir=config_dir,
                meta_projection_summary=summary.get("live_meta_projection"),
            )
    return summary


def get_btc_data(
    symbol: str = DEFAULT_BTC_SYMBOL,
    interval: str = DEFAULT_BTC_INTERVAL,
    price_db_path: str = DEFAULT_PRICE_DB_PATH,
    price_table_name: str = DEFAULT_PRICE_TABLE,
    onchain_db_path: str = DEFAULT_ONCHAIN_DB_PATH,
    onchain_mode: str = "auto",
    onchain_overlap_points: int = DEFAULT_OVERLAP_POINTS,
    macro_db_path: str = DEFAULT_MACRO_DB_PATH,
    macro_table_name: str = DEFAULT_MACRO_TABLE,
    macro_overlap_points: int = DEFAULT_MACRO_OVERLAP_POINTS,
    manual_dir: str = str(DEFAULT_MANUAL_DIR),
    include_price: bool = True,
    include_macro: bool = True,
    include_fred_macro: bool = True,
    include_manual_macro: bool = True,
    include_checkonchain_chop: bool = True,
    include_liquidation_signals: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if include_price:
        summary["price"] = sync_binance_ohlcv_to_duckdb(
            symbol=symbol,
            interval=interval,
            db_path=price_db_path,
            table_name=price_table_name,
        )

    resolved_onchain_mode = onchain_mode
    if onchain_mode == "auto":
        resolved_onchain_mode = "sync" if database_has_tables(onchain_db_path) else "full"

    if resolved_onchain_mode == "full":
        summary["onchain"] = fetch_onchain_history(db_path=onchain_db_path)
    elif resolved_onchain_mode == "sync":
        summary["onchain"] = sync_onchain_data(
            overlap_points=onchain_overlap_points,
            db_path=onchain_db_path,
        )
    if include_checkonchain_chop:
        summary["checkonchain_choppiness"] = sync_choppiness_index(db_path=onchain_db_path)
    if include_liquidation_signals:
        summary["liquidation_signals"] = sync_liquidation_signals()

    if include_macro:
        macro_summary: dict[str, Any] = {}
        if include_fred_macro:
            macro_summary["fred"] = update_fred_data(
                db_path=macro_db_path,
                table_name=macro_table_name,
                overlap_points=macro_overlap_points,
            )
        if include_manual_macro:
            macro_summary["manual"] = update_manual_macro_data(
                manual_dir=manual_dir,
                db_path=macro_db_path,
                table_name=macro_table_name,
            )
        summary["macro"] = macro_summary

    return summary


def compute_categories(
    symbol: str = DEFAULT_BTC_SYMBOL,
    price_db_path: str = DEFAULT_PRICE_DB_PATH,
    category_db_path: str = DEFAULT_CATEGORY_DB_PATH,
    ta_output_db_path: str = DEFAULT_TA_OUTPUT_DB_PATH,
    onchain_db_path: str = DEFAULT_ONCHAIN_DB_PATH,
    macro_db_path: str = DEFAULT_MACRO_DB_PATH,
    liquidation_signal_path: str = DEFAULT_LIQUIDATION_SIGNAL_PATH_ENV,
    interval: str = DEFAULT_BTC_INTERVAL,
    as_of_timestamp: str | None = None,
):
    build_1d_ta_db(
        symbol,
        ohlcv_db_path=price_db_path,
        output_db_path=ta_output_db_path,
        end_timestamp=as_of_timestamp,
    )

    common_kwargs = {
        "symbol": symbol,
        "db_path": category_db_path,
        "price_db_path": price_db_path,
        "onchain_db_path": onchain_db_path,
        "macro_db_path": macro_db_path,
        "ta_db_path": ta_output_db_path,
        "interval": interval,
        "end_timestamp": as_of_timestamp,
    }
    datasets = {
        "structure": StructureSet(**common_kwargs),
        "environment": EnvironmentSet(**common_kwargs),
        "edges": EdgesSet(**common_kwargs),
        "movement": MovementSet(**common_kwargs),
        "liquidation": LiquidationSet(**common_kwargs, signal_path=liquidation_signal_path),
    }
    summary = {
        name: {
            "table": result.table_name,
            "rows": result.rows,
            "start": result.start,
            "end": result.end,
        }
        for name, result in (
            (name, dataset.store()) for name, dataset in datasets.items()
        )
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh BTC data, rebuild feature buckets, refit the locked production model, and emit the live forecast."
    )
    parser.add_argument(
        "--config-dir",
        default=str(FINAL_CONFIG_DIR),
        help="Directory containing the frozen production config files.",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Skip price/onchain/macro refresh and reuse existing local data.",
    )
    parser.add_argument(
        "--force-data-refresh",
        action="store_true",
        help="Pull data even if local price data is already fresh through the previous UTC day.",
    )
    parser.add_argument(
        "--skip-category-refresh",
        action="store_true",
        help="Skip rebuilding specialist category tables and reuse existing local tables.",
    )
    parser.add_argument(
        "--skip-production-forecast",
        action="store_true",
        help="Skip the frozen production refit/forecast step.",
    )
    parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip updating the walk-forward prediction database before JEPA training and the live forecast.",
    )
    parser.add_argument(
        "--force-walk-forward",
        action="store_true",
        help="Recompute walk-forward dates even when all horizons are already present.",
    )
    parser.add_argument(
        "--walk-forward-limit",
        type=int,
        default=None,
        help="Optional maximum number of missing/as-of dates to process in the walk-forward update.",
    )
    parser.add_argument(
        "--walk-forward-rebuild-feature-submodels",
        action="store_true",
        help=(
            "Rebuild category/TA feature tables for each walk-forward as-of date. "
            "Use this for leakage-safe liquidation walk-forward backfills."
        ),
    )
    parser.add_argument(
        "--walk-forward-reuse-cached-feature-submodels",
        action="store_true",
        help=(
            "With --walk-forward-rebuild-feature-submodels, create per-as-of temporary category tables "
            "by copying stored feature rows through each as-of date instead of recomputing heavy "
            "TA/PCA/liquidity-fair-value submodels."
        ),
    )
    parser.add_argument(
        "--quiet-walk-forward",
        action="store_true",
        help="Suppress per-date walk-forward progress lines when using the main.py bridge.",
    )
    parser.add_argument(
        "--skip-jepa-training",
        action="store_true",
        help="Skip daily five-world JEPA retraining before the final live forecast.",
    )
    parser.add_argument(
        "--legacy-jepa-training",
        action="store_true",
        help="Also run the superseded pooled specialist JEPA for compatibility diagnostics.",
    )
    parser.add_argument(
        "--world-jepa-config",
        default="configs/world_jepa.json",
        help="Configuration for the five-world JEPA attention pipeline.",
    )
    parser.add_argument(
        "--world-jepa-router-epochs",
        type=int,
        default=None,
        help="Override five-world router training epochs.",
    )
    parser.add_argument(
        "--world-jepa-stride",
        type=int,
        default=1,
        help="Causal sample stride for per-world JEPA pretraining.",
    )
    parser.add_argument(
        "--world-jepa-smoke",
        action="store_true",
        help="Run a bounded five-world integration smoke instead of full training.",
    )
    parser.add_argument(
        "--reuse-world-jepa-feature-manifest",
        action="store_true",
        help="Do not refresh the causal feature audit from the newest category tables.",
    )
    parser.add_argument(
        "--jepa-specialists",
        nargs="+",
        default=None,
        help="Optional specialist list for daily JEPA training. Defaults to all specialists, including liquidation.",
    )
    parser.add_argument(
        "--jepa-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Optional horizon list for daily JEPA training. Defaults to JEPA_HORIZONS or the production default.",
    )
    parser.add_argument(
        "--jepa-context-length",
        type=int,
        default=None,
        help="Override the JEPA context length for daily training.",
    )
    parser.add_argument(
        "--jepa-latent-dim",
        type=int,
        default=None,
        help="Override the JEPA latent dimension for daily training.",
    )
    parser.add_argument(
        "--jepa-max-epochs",
        type=int,
        default=None,
        help="Override JEPA_MAX_EPOCHS for daily training.",
    )
    parser.add_argument(
        "--jepa-batch-size",
        type=int,
        default=None,
        help="Override JEPA_BATCH_SIZE for daily training.",
    )
    parser.add_argument(
        "--jepa-seed",
        type=int,
        default=None,
        help="Override JEPA_SEED for daily training.",
    )
    parser.add_argument(
        "--jepa-device",
        default="cpu",
        help="Torch device for daily JEPA training, for example cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--skip-meta-learner",
        action="store_true",
        help="Skip live meta-synthesis learner artifacts and the meta projection overlay.",
    )
    parser.add_argument(
        "--meta-min-history",
        type=int,
        default=None,
        help="Minimum realized observations required before a meta candidate can be selected.",
    )
    parser.add_argument(
        "--meta-selection-window",
        type=int,
        default=None,
        help="Rolling realized-observation window used by the meta learner for candidate selection.",
    )
    parser.add_argument(
        "--skip-specialist-live-visuals",
        action="store_true",
        help="Skip specialist live prediction CSV/JSON/PNG outputs in the live forecast folder.",
    )
    parser.add_argument(
        "--keep-legacy-live-files",
        action="store_true",
        help="Do not remove legacy walk-forward/meta diagnostic files from the live forecast folder.",
    )
    parser.add_argument(
        "--verbose-summary",
        action="store_true",
        help="Print the full internal run summary instead of the compact operational summary.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Run the production forecast using only aligned model rows on or before this timestamp.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Run the production forecast without overwriting production artifacts.",
    )
    args = parser.parse_args()

    summary = run_daily_production_cycle(
        config_dir=args.config_dir,
        refresh_data=not args.skip_data_refresh,
        gate_data_refresh=not args.force_data_refresh,
        rebuild_categories=not args.skip_category_refresh,
        run_production_forecast=not args.skip_production_forecast,
        run_walk_forward_update=not args.skip_walk_forward,
        clean_live_outputs=not args.keep_legacy_live_files,
        as_of_timestamp=args.as_of,
        save_artifacts=not args.no_artifacts,
        force_walk_forward=args.force_walk_forward,
        walk_forward_limit=args.walk_forward_limit,
        walk_forward_rebuild_feature_submodels=args.walk_forward_rebuild_feature_submodels,
        walk_forward_reuse_cached_feature_submodels=args.walk_forward_reuse_cached_feature_submodels,
        quiet_walk_forward=args.quiet_walk_forward,
        train_jepa=(
            args.legacy_jepa_training
            and not args.skip_jepa_training
            and not args.no_artifacts
            and args.as_of is None
        ),
        train_world_jepa=not args.skip_jepa_training and not args.no_artifacts and args.as_of is None,
        world_jepa_config_path=args.world_jepa_config,
        world_jepa_world_epochs=args.jepa_max_epochs,
        world_jepa_router_epochs=args.world_jepa_router_epochs,
        world_jepa_batch_size=args.jepa_batch_size,
        world_jepa_stride=args.world_jepa_stride,
        world_jepa_device=args.jepa_device,
        world_jepa_smoke=args.world_jepa_smoke,
        refresh_world_jepa_feature_audit=not args.reuse_world_jepa_feature_manifest,
        jepa_specialists=args.jepa_specialists,
        jepa_horizons=args.jepa_horizons,
        jepa_context_length=args.jepa_context_length,
        jepa_latent_dim=args.jepa_latent_dim,
        jepa_max_epochs=args.jepa_max_epochs,
        jepa_batch_size=args.jepa_batch_size,
        jepa_seed=args.jepa_seed,
        jepa_device=args.jepa_device,
        run_meta_learner=not args.skip_meta_learner,
        meta_min_history=args.meta_min_history,
        meta_selection_window=args.meta_selection_window,
        write_specialist_live_outputs=not args.skip_specialist_live_visuals,
    )
    output = summary if args.verbose_summary else _compact_summary_for_stdout(summary)
    print(json.dumps(output, indent=2, default=_json_default))
