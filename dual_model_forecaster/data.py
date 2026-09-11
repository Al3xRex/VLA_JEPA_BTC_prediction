from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from database_interraction import load_table_dataframe
from dual_model_forecaster.utils import (
    compute_days_since_change,
    format_date,
    freshness_score,
    missing_fraction,
    safe_log,
)


@dataclass
class ForecastDataBundle:
    close: pd.Series
    buckets: dict[str, pd.DataFrame]
    targets: pd.DataFrame
    freshness_days: dict[str, pd.Series]
    freshness_scores: dict[str, pd.Series]
    missingness: dict[str, pd.Series]
    validation_report: dict[str, Any]
    price_available_at: pd.Series | None = None


def _load_bucket(db_path: str, table_name: str) -> pd.DataFrame:
    frame = load_table_dataframe(
        db_path,
        table_name,
        read_only=True,
        order_by=["timestamp"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.sort_values("timestamp").set_index("timestamp")
    frame.index.name = "timestamp"
    return frame


def _binance_interval_close_times(open_times: pd.Series, interval: str) -> pd.Series:
    """Infer Binance's inclusive close time for legacy OHLCV schemas."""

    match = re.fullmatch(r"([1-9][0-9]*)([smhdwM])", str(interval))
    if match is None:
        return pd.Series(pd.NaT, index=open_times.index, dtype="datetime64[ns]")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "M":
        values = [
            pd.Timestamp(value) + pd.DateOffset(months=amount) - pd.Timedelta(milliseconds=1)
            if pd.notna(value)
            else pd.NaT
            for value in open_times
        ]
        return pd.Series(values, index=open_times.index, dtype="datetime64[ns]")
    unit_map = {"s": "s", "m": "min", "h": "h", "d": "D", "w": "W"}
    return open_times + pd.to_timedelta(amount, unit=unit_map[unit]) - pd.Timedelta(milliseconds=1)


def load_closed_ohlcv_frame(
    db_path: str,
    table_name: str,
    symbol: str,
    interval: str,
    *,
    as_of_timestamp: Any | None = None,
) -> pd.DataFrame:
    """Load only OHLCV bars whose final values were available by ``as_of``.

    New stores carry explicit ``available_at`` metadata.  Legacy stores are
    handled by inferring the close boundary from the Binance interval string.
    Unknown interval formats are excluded rather than assumed causal.
    """

    table = load_table_dataframe(
        db_path,
        table_name,
        read_only=True,
        filters={"symbol": symbol.upper(), "interval": interval},
        order_by=["open_time"],
    )
    table["open_time"] = pd.to_datetime(table["open_time"], errors="coerce")
    table = table.dropna(subset=["open_time"]).sort_values("open_time")
    inferred_close = _binance_interval_close_times(table["open_time"], interval)
    if "close_time" in table.columns:
        close_time = pd.to_datetime(table["close_time"], errors="coerce").fillna(inferred_close)
    else:
        close_time = inferred_close
    if "available_at" in table.columns:
        available_at = pd.to_datetime(table["available_at"], errors="coerce").fillna(close_time)
    else:
        available_at = close_time
    cutoff = _normalize_cutoff_timestamp(as_of_timestamp)
    if cutoff is None:
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None)
    table["close_time"] = close_time
    table["available_at"] = available_at
    table["is_closed"] = available_at.notna() & available_at.le(cutoff)
    table = table.loc[table["is_closed"]].drop_duplicates(subset=["open_time"], keep="last")
    return table.reset_index(drop=True)


def _load_close_series(
    db_path: str,
    table_name: str,
    symbol: str,
    interval: str,
    *,
    as_of_timestamp: Any | None = None,
) -> pd.Series:
    table = load_closed_ohlcv_frame(
        db_path,
        table_name,
        symbol,
        interval,
        as_of_timestamp=as_of_timestamp,
    )
    close = table.set_index("open_time")["close"].astype(float)
    close.index.name = "timestamp"
    return close


def _build_targets(close: pd.Series, horizons: list[int], return_type: str) -> pd.DataFrame:
    target_df = pd.DataFrame(index=close.index)
    for horizon in horizons:
        shifted = close.shift(-horizon)
        if return_type == "log":
            values = safe_log(shifted) - safe_log(close)
        else:
            values = shifted / close - 1.0
        target_df[f"target_{horizon}d"] = values
    return target_df


def _table_validation(frame: pd.DataFrame, table_name: str) -> dict[str, Any]:
    diffs = pd.Series(frame.index).diff().dropna()
    diff_days = diffs.dt.days.value_counts().sort_index().to_dict()
    return {
        "table_name": table_name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "first_timestamp": format_date(frame.index.min()),
        "last_timestamp": format_date(frame.index.max()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "missing_value_ratio": float(frame.isna().mean().mean()),
        "day_gap_histogram": {str(int(k)): int(v) for k, v in diff_days.items()},
        "gaps_larger_than_one_day": int(sum(v for k, v in diff_days.items() if int(k) > 1)),
    }


def _infer_canonical_frequency(index: pd.Index) -> dict[str, Any]:
    diffs = pd.Series(index).diff().dropna()
    if diffs.empty:
        return {"mode_days": None, "mode_count": 0, "total_diffs": 0}
    day_counts = diffs.dt.days.value_counts().sort_values(ascending=False)
    mode_days = int(day_counts.index[0])
    mode_count = int(day_counts.iloc[0])
    return {
        "mode_days": mode_days,
        "mode_count": mode_count,
        "total_diffs": int(len(diffs)),
        "share": float(mode_count / max(len(diffs), 1)),
    }


def build_validation_report(
    close: pd.Series,
    buckets: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    freshness_days: dict[str, pd.Series],
) -> dict[str, Any]:
    bucket_reports = {name: _table_validation(frame, f"{name}_set") for name, frame in buckets.items()}
    common_index = close.index
    common_diff = pd.Series(common_index).diff().dropna()
    common_gap_hist = common_diff.dt.days.value_counts().sort_index().to_dict()
    leakage_notes = [
        "Common index is built as the timestamp intersection of price and all configured specialist buckets.",
        "OHLCV bars are admitted only after their close/available_at timestamp; incomplete bars are excluded.",
        "Targets are forward returns aligned as return(t, t+h] and are never merged back into features.",
        "No full-sample scalers are applied in the loader; model-side scalers must be fit fold-by-fold.",
    ]
    environment_columns = buckets.get("environment", pd.DataFrame()).columns.tolist()
    if "m2_shift_fwd_6w" in environment_columns:
        leakage_notes.append(
            "environment_set contains m2_shift_fwd_6w; this series is shifted forward in calendar time during feature engineering, which makes it a lagged-release proxy at the timestamp it appears and not a future leak."
        )

    alignment = {
        name: {
            "raw_rows": int(len(frame)),
            "aligned_rows": int(len(common_index)),
            "aligned_share": float(len(common_index) / max(len(frame), 1)),
            "aligned_missing_ratio": float(frame.reindex(common_index).isna().mean().mean()),
        }
        for name, frame in buckets.items()
    }
    target_alignment = {
        column: {
            "null_rows": int(targets[column].isna().sum()),
            "first_valid_timestamp": format_date(targets[column].dropna().index.min()) if targets[column].notna().any() else None,
            "last_valid_timestamp": format_date(targets[column].dropna().index.max()) if targets[column].notna().any() else None,
        }
        for column in targets.columns
    }
    freshness_summary = {
        name: {
            "median_days_since_change": float(series.median()),
            "p90_days_since_change": float(series.quantile(0.90)),
            "max_days_since_change": float(series.max()),
        }
        for name, series in freshness_days.items()
    }

    coverage = {
        "rows": int(len(common_index)),
        "first_timestamp": format_date(common_index.min()),
        "last_timestamp": format_date(common_index.max()),
        "common_gap_histogram": {str(int(k)): int(v) for k, v in common_gap_hist.items()},
        "common_gaps_larger_than_one_day": int(sum(v for k, v in common_gap_hist.items() if int(k) > 1)),
        "target_null_rows": int(targets.isna().all(axis=1).sum()),
        "canonical_frequency": _infer_canonical_frequency(common_index),
    }

    return {
        "price": {
            "rows": int(len(close)),
            "first_timestamp": format_date(close.index.min()),
            "last_timestamp": format_date(close.index.max()),
            "canonical_frequency": _infer_canonical_frequency(close.index),
        },
        "buckets": bucket_reports,
        "alignment": alignment,
        "target_alignment": target_alignment,
        "freshness_summary": freshness_summary,
        "coverage": coverage,
        "leakage_audit": leakage_notes,
        "limitations": [
            "Release-vintage macro histories are not available in the current source tables, so exact publication-lag reconstruction is approximated with observed timestamps and staleness metadata.",
        ],
    }


def _normalize_cutoff_timestamp(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    cutoff = pd.Timestamp(value)
    if cutoff.tz is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    return cutoff


def load_forecast_data(
    config: dict[str, Any],
    end_timestamp: Any | None = None,
) -> ForecastDataBundle:
    paths = config["paths"]
    data_cfg = config["data"]
    bucket_tables = data_cfg["bucket_tables"]
    cutoff = _normalize_cutoff_timestamp(end_timestamp)

    closed_ohlcv = load_closed_ohlcv_frame(
        db_path=paths["price_db_path"],
        table_name=paths["price_table_name"],
        symbol=data_cfg["symbol"],
        interval=data_cfg["interval"],
        as_of_timestamp=cutoff,
    )
    close = closed_ohlcv.set_index("open_time")["close"].astype(float)
    close.index.name = "timestamp"
    price_available_at = closed_ohlcv.set_index("open_time")["available_at"]
    price_available_at.index.name = "timestamp"
    buckets = {
        name: _load_bucket(paths["category_db_path"], table_name)
        for name, table_name in bucket_tables.items()
    }

    common_index = close.index
    for frame in buckets.values():
        common_index = common_index.intersection(frame.index)
    common_index = common_index.sort_values()
    if cutoff is not None:
        common_index = common_index[common_index <= cutoff]
        if len(common_index) == 0:
            raise ValueError(f"No aligned forecast rows are available on or before {cutoff}.")

    close = close.reindex(common_index)
    buckets = {name: frame.reindex(common_index) for name, frame in buckets.items()}
    targets = _build_targets(
        close=close,
        horizons=list(data_cfg["target_horizons"]),
        return_type=str(data_cfg["return_type"]),
    )

    freshness_days = {
        name: compute_days_since_change(frame)
        for name, frame in buckets.items()
    }
    freshness_scores = {
        name: freshness_score(
            freshness_days[name],
            float(data_cfg["freshness_half_life_days"][name]),
        )
        for name in buckets
    }
    missingness = {
        name: missing_fraction(frame)
        for name, frame in buckets.items()
    }
    validation_report = build_validation_report(
        close=close,
        buckets=buckets,
        targets=targets,
        freshness_days=freshness_days,
    )

    return ForecastDataBundle(
        close=close,
        buckets=buckets,
        targets=targets,
        freshness_days=freshness_days,
        freshness_scores=freshness_scores,
        missingness=missingness,
        validation_report=validation_report,
        price_available_at=price_available_at.reindex(common_index),
    )
