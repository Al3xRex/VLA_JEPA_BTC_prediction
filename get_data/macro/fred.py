from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import requests

from env_loader import load_project_env
from database_interraction import (
    duckdb_connection,
    ensure_macro_series_table,
    get_distinct_value_at_offset,
    get_min_value,
    load_table_as_dataframe,
    upsert_dataframe,
)

load_project_env()

# ================================
# CONFIGURATION
# ================================
API_KEY = (os.getenv("FRED_API_KEY") or "").strip()
START_DATE = "2013-01-01"
DB_PATH = "database/macro.duckdb"
TABLE_NAME = "macro_series"
SOURCE_NAME = "fred"
DEFAULT_OVERLAP_POINTS = max(1, int((os.getenv("FRED_OVERLAP_POINTS") or "3").strip()))

# FRED series -> column names
SERIES_MAP: Dict[str, str] = {
    "WALCL": "Fed_Assets",
    "WTREGEN": "TGA",
    "RRPONTSYD": "Reverse_Repo_Billions",
    "H41RESPPALDKNWW": "BTFP",
    "WLCFLPCL": "Primary_Credit",
    "BACTSAMFRBDAL": "Buisness_activity",
    "DTWEXEMEGS": "emerging_markets",
    "SP500": "spx",
    "DCOILWTICO": "oil",
    "IQ12260": "gold",
    "DTWEXBGS": "dxy",
}


def fetch_series(
    series_id: str,
    series_name: str,
    api_key: str,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """
    Fetch a single FRED series and return a normalized DataFrame
    with columns: date, source, series_id, series_name, value.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        observations = payload.get("observations", [])
        if not observations:
            print(f"[WARN] No data returned for {series_id} ({series_name}).")
            return pd.DataFrame(columns=["date", "source", "series_id", "series_name", "value"])

        dataframe = pd.DataFrame(observations)
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True).dt.date
        dataframe["value"] = pd.to_numeric(dataframe["value"], errors="coerce")

        dataframe = dataframe[["date", "value"]].copy()
        dataframe["source"] = SOURCE_NAME
        dataframe["series_id"] = series_id
        dataframe["series_name"] = series_name
        dataframe = dataframe[["date", "source", "series_id", "series_name", "value"]]
        dataframe = dataframe.dropna(subset=["date", "value"]).drop_duplicates(
            subset=["date", "source", "series_id"],
            keep="last",
        )

        print(f"[OK] Loaded {series_name} ({len(dataframe)} rows)")
        return dataframe
    except Exception as exc:
        print(f"[ERR] Fetching {series_id}: {exc}")
        return pd.DataFrame(columns=["date", "source", "series_id", "series_name", "value"])


def _resolve_series_start_date(
    connection,
    *,
    table_name: str,
    series_id: str,
    default_start_date: str,
    overlap_points: int,
) -> str:
    filters = {"source": SOURCE_NAME, "series_id": series_id}
    latest_overlap_date = get_distinct_value_at_offset(
        connection,
        table_name,
        "date",
        offset=max(int(overlap_points) - 1, 0),
        filters=filters,
    )
    if latest_overlap_date is not None:
        return pd.to_datetime(latest_overlap_date).strftime("%Y-%m-%d")

    earliest_available_date = get_min_value(
        connection,
        table_name,
        "date",
        filters=filters,
    )
    if earliest_available_date is not None:
        return pd.to_datetime(earliest_available_date).strftime("%Y-%m-%d")

    return default_start_date


def _upsert_data(
    dataframe: pd.DataFrame,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> int:
    if dataframe.empty:
        print("[INFO] Nothing to upsert.")
        return 0

    with duckdb_connection(db_path) as connection:
        ensure_macro_series_table(connection, table_name)
        rows_upserted = upsert_dataframe(
            connection,
            table_name,
            dataframe,
            key_columns=("date", "source", "series_id"),
        )

    print(f"[INFO] Upsert complete. Rows written to {table_name}: {rows_upserted}")
    return rows_upserted


def update_fred_data(
    api_key: Optional[str] = None,
    start_date: str = START_DATE,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    series_map: Optional[Dict[str, str]] = None,
    overlap_points: int = DEFAULT_OVERLAP_POINTS,
) -> dict[str, Any]:
    """
    Fetch all configured FRED series and upsert them into the shared macro table.
    """
    resolved_api_key = (api_key if api_key is not None else API_KEY).strip()
    if not resolved_api_key:
        print("[WARN] FRED_API_KEY is not set; skipping FRED update.")
        return {
            "source": SOURCE_NAME,
            "db_path": db_path,
            "table": table_name,
            "series_count": 0,
            "rows_fetched": 0,
            "rows_upserted": 0,
            "skipped": True,
        }

    resolved_series_map = series_map or SERIES_MAP
    frames: list[pd.DataFrame] = []

    with duckdb_connection(db_path) as connection:
        ensure_macro_series_table(connection, table_name)
        for series_id, series_name in resolved_series_map.items():
            series_start_date = _resolve_series_start_date(
                connection,
                table_name=table_name,
                series_id=series_id,
                default_start_date=start_date,
                overlap_points=overlap_points,
            )
            print(
                f"[INFO] Fetching {series_id} from {series_start_date} "
                f"(overlap_points={overlap_points})"
            )
            dataframe = fetch_series(
                series_id,
                series_name,
                api_key=resolved_api_key,
                start_date=series_start_date,
            )
            if not dataframe.empty:
                frames.append(dataframe)

    if not frames:
        print("[WARN] No data fetched for any FRED series.")
        return {
            "source": SOURCE_NAME,
            "db_path": db_path,
            "table": table_name,
            "series_count": len(resolved_series_map),
            "rows_fetched": 0,
            "rows_upserted": 0,
        }

    combined = pd.concat(frames, axis=0, ignore_index=True)
    rows_upserted = _upsert_data(combined, db_path=db_path, table_name=table_name)
    return {
        "source": SOURCE_NAME,
        "db_path": db_path,
        "table": table_name,
        "series_count": len(resolved_series_map),
        "rows_fetched": int(len(combined)),
        "rows_upserted": int(rows_upserted),
    }


def load_macro_from_db(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Load stored macro series and pivot to a wide DataFrame indexed by date.
    """
    with duckdb_connection(db_path) as connection:
        ensure_macro_series_table(connection, table_name)
        dataframe = load_table_as_dataframe(
            connection,
            table_name,
            columns=["date", "source", "series_name", "value"],
            filters={"source": source} if source else None,
            order_by=["date", "source", "series_name"],
        )

    if dataframe.empty:
        return pd.DataFrame()

    dataframe["date"] = pd.to_datetime(dataframe["date"])
    pivoted = dataframe.pivot_table(
        index="date",
        columns="series_name",
        values="value",
        aggfunc="last",
    ).sort_index()
    return pivoted


def load_fred_from_db(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> pd.DataFrame:
    return load_macro_from_db(
        db_path=db_path,
        table_name=table_name,
        source=SOURCE_NAME,
    )
