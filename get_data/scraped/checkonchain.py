from __future__ import annotations

from typing import Any

import pandas as pd

from compute_data.ta.choppiness import build_choppiness_features
from database_interraction import (
    duckdb_connection,
    fetch_scalar,
    get_max_value,
    quote_identifier,
    replace_table_from_dataframe,
    table_exists,
)
from liquidations.main import (
    ScrapeError,
    extract_plotly_traces,
    fetch_text,
    trace_points_by_date,
)

# The public charts host serves a JavaScript iframe wrapper; urllib needs the
# direct CDN page that contains the Plotly.newPlot payload.
CHECKONCHAIN_CHOPPINESS_INDEX_URL = (
    "https://charts-cdn.checkonchain.com/btconchain/technical/"
    "technical_choppinessindex/technical_choppinessindex_light.html"
)
CHOPPINESS_TABLE_NAME = "checkonchain_choppiness_index"

CHOPPINESS_TRACE_COLUMNS: dict[str, str] = {
    "Price": "checkonchain_price",
    "Chop Index (1D, 14-d)": "chop_1d_14d",
    "Chop Index (1W, 10-w)": "chop_1w_10w",
    "Chop Index (1M, 10-m)": "chop_1m_10m",
    "CI = 61.8": "chop_consolidation_level",
    "CI = 23.6": "chop_trend_level",
}


def _trace_series(traces: list[dict[str, Any]], trace_name: str, column_name: str) -> pd.Series:
    points = trace_points_by_date(traces, trace_name)
    series = pd.Series(points, name=column_name, dtype=float)
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[series.index.notna()]
    series.index.name = "timestamp"
    return series.sort_index()


def parse_choppiness_index_html(html: str) -> pd.DataFrame:
    traces = extract_plotly_traces(html)
    columns: dict[str, pd.Series] = {}
    missing: list[str] = []
    for trace_name, column_name in CHOPPINESS_TRACE_COLUMNS.items():
        try:
            columns[column_name] = _trace_series(traces, trace_name, column_name)
        except ScrapeError:
            if column_name.startswith("chop_") and column_name not in {
                "chop_consolidation_level",
                "chop_trend_level",
            }:
                missing.append(trace_name)
            continue

    if missing:
        raise ScrapeError(f"Missing required CheckOnChain Chop traces: {', '.join(missing)}")
    if not columns:
        raise ScrapeError("No CheckOnChain Chop traces were parsed.")

    frame = pd.concat(columns.values(), axis=1).sort_index()
    frame.index.name = "timestamp"
    return build_choppiness_features(frame)


def fetch_choppiness_index(*, timeout: float = 30.0) -> pd.DataFrame:
    html = fetch_text(CHECKONCHAIN_CHOPPINESS_INDEX_URL, timeout=timeout)
    return parse_choppiness_index_html(html)


def store_choppiness_index(
    frame: pd.DataFrame,
    *,
    db_path: str = "database/onchain.duckdb",
    table_name: str = CHOPPINESS_TABLE_NAME,
) -> dict[str, Any]:
    if frame.empty:
        return {"table": table_name, "rows": 0, "start": None, "end": None}
    data = build_choppiness_features(frame).copy()
    data.index.name = "timestamp"
    data = data.reset_index().drop_duplicates(subset=["timestamp"], keep="last")
    data = data.sort_values("timestamp")
    with duckdb_connection(db_path) as connection:
        replace_table_from_dataframe(connection, table_name, data)
        rows = fetch_scalar(connection, f"SELECT COUNT(*) FROM {quote_identifier(table_name)};")
    return {
        "table": table_name,
        "rows": int(rows or 0),
        "start": data["timestamp"].min(),
        "end": data["timestamp"].max(),
    }


def sync_choppiness_index(
    *,
    db_path: str = "database/onchain.duckdb",
    table_name: str = CHOPPINESS_TABLE_NAME,
    timeout: float = 30.0,
    force: bool = False,
) -> dict[str, Any]:
    frame = fetch_choppiness_index(timeout=timeout)
    latest_source = pd.Timestamp(frame.index.max()) if not frame.empty else None
    if latest_source is not None and not force:
        with duckdb_connection(db_path) as connection:
            latest_stored = (
                get_max_value(connection, table_name, "timestamp")
                if table_exists(connection, table_name)
                else None
            )
        if latest_stored is not None and pd.Timestamp(latest_stored) >= latest_source:
            return {
                "table": table_name,
                "rows": int(len(frame)),
                "start": pd.Timestamp(frame.index.min()) if not frame.empty else None,
                "end": latest_source,
                "source_url": CHECKONCHAIN_CHOPPINESS_INDEX_URL,
                "status": "skipped",
                "reason": "no_new_data",
                "latest_stored": pd.Timestamp(latest_stored),
            }
    result = store_choppiness_index(frame, db_path=db_path, table_name=table_name)
    result["source_url"] = CHECKONCHAIN_CHOPPINESS_INDEX_URL
    result["status"] = "updated"
    return result
