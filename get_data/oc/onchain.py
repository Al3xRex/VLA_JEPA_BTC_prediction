from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from env_loader import load_project_env
from database_interraction import (
    append_dataframe,
    delete_rows_from_value,
    duckdb_connection,
    get_distinct_value_at_offset,
    get_preferred_time_column,
    list_tables,
    replace_table_from_dataframe,
    table_exists,
)

load_project_env()

# ================================
# CONFIGURATION
# ================================
PUBLIC_SCHEME = (os.getenv("BITLAB_PUBLIC_SCHEME") or "https").strip().strip('"').strip("'")
PUBLIC_API_DOMAIN = (os.getenv("BITLAB_PUBLIC_API_DOMAIN") or "api.researchbitcoin.net").strip().strip('"').strip("'")
BITLAB_API_BASE = f"{PUBLIC_SCHEME}://{PUBLIC_API_DOMAIN}"
BITLAB_API_TOKEN = (os.getenv("BITLAB_API_TOKEN") or "").strip().strip('"').strip("'")
DB_PATH = "database/onchain.duckdb"

ENDPOINT_PATHS = [
     # cohort_avgBTC ≈ cohort_sumBTC / cohort_N 
    "v2/address_statistics/addresses_by_btc_n",
    "v2/address_statistics/addresses_by_btc_sumbtc",
    "v2/network_statistics/hashrate",

    #LTH
    "/v2/cointime_statistics/active_value_to_investor_value",
    "/v2/cointime_statistics/active_mvrv",
    "/v2/market_value_to_realized_value/mvrv",
    "/v2/net_unrealized_profit_loss/net_unrealized_profit_loss",
    "/v2/spent_output_profit_ratio/sopr",
    "/v2/supply_in_profitloss/supply_in_profit_percent",
    "/v2/supply_in_profitloss/utxo_n_in_profit_percent",
    "/v2/unrealizedcap/unrealized_cap_relative",
    "/v2/market_value_to_realized_value/mvrv_lth",
    "/v2/net_unrealized_profit_loss/net_unrealized_profit_loss_lth",
    "/v2/spent_output_profit_ratio/sopr_lth",
    "/v2/realizedprofit/realized_profit",
    "/v2/realizedloss/realized_loss",
    "/v2/realizedcap/realized_cap",
    "/v2/realizedprofit/realized_profit_lth",
    "/v2/realizedloss/realized_loss_lth",
    "/v2/realizedcap/realized_cap_lth",


    #STH
    "/v2/market_value_to_realized_value/mvrv_sth",
    "/v2/net_unrealized_profit_loss/net_unrealized_profit_loss_sth",
    "/v2/spent_output_profit_ratio/sopr_sth",
    "/v2/supply_in_profitloss/utxo_n_in_profit_sth_percent",
    "/v2/supply_in_profitloss/supply_in_profit_sth_percent",
    "/v2/unrealizedcap/unrealized_cap_sth_relative",
    "/v2/realizedprofit/realized_profit_sth",
    "/v2/realizedloss/realized_loss_sth",
    "/v2/realizedcap/realized_cap_sth",
]


# We set from_time to the start date.
# The API should return data from this date -> Present.
DEFAULT_FROM_TIME = (os.getenv("BITLAB_FROM_TIME") or "2013-01-01 00:00").strip().strip('"').strip("'")
_raw_to_time = os.getenv("BITLAB_TO_TIME")
DEFAULT_TO_TIME = _raw_to_time.strip().strip('"').strip("'") if _raw_to_time else None
DEFAULT_RESOLUTION = (os.getenv("BITLAB_RESOLUTION") or "d1").strip().strip('"').strip("'")
DEFAULT_OVERLAP_POINTS = max(1, int((os.getenv("BITLAB_OVERLAP_POINTS") or "3").strip()))
CONNECT_TIMEOUT_S = max(1.0, float((os.getenv("BITLAB_CONNECT_TIMEOUT_S") or "8").strip()))
READ_TIMEOUT_S = max(1.0, float((os.getenv("BITLAB_READ_TIMEOUT_S") or "25").strip()))
HTTP_RETRIES = max(0, int((os.getenv("BITLAB_HTTP_RETRIES") or "1").strip()))

def _default_to_time(resolution: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    if resolution:
        delta = _resolution_to_timedelta(resolution)
        if delta >= timedelta(days=1):
            # Daily and slower endpoints behave best with UTC day boundaries,
            # but the API rejects future end timestamps.
            now = now.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
    return now.strftime("%Y-%m-%d %H:%M")

COMMON_PARAMS = {
    "output_format": "json",
    "from_time": DEFAULT_FROM_TIME,
    "resolution": DEFAULT_RESOLUTION,
}

# ================================
# FUNCTIONS
# ================================

def _build_session(api_token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    if api_token:
        s.headers.update({"X-API-Token": api_token})
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=HTTP_RETRIES,
        connect=HTTP_RETRIES,
        read=0,
        status=HTTP_RETRIES,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )))
    return s

def _url(path: str) -> str:
    return f"{BITLAB_API_BASE.rstrip('/')}/{path.lstrip('/')}"

def _get_params() -> dict:
    params = dict(COMMON_PARAMS)
    if not DEFAULT_TO_TIME:
        params["to_time"] = _default_to_time(params.get("resolution"))
    else:
        params["to_time"] = DEFAULT_TO_TIME
    return params

def _parse_time(value: str) -> datetime:
    return pd.to_datetime(value, utc=True).to_pydatetime()

def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")

def _resolution_to_timedelta(resolution: str) -> timedelta:
    if not resolution:
        return timedelta(days=1)
    unit = resolution[0].lower()
    try:
        qty = int(resolution[1:]) if len(resolution) > 1 else 1
    except ValueError:
        qty = 1
    if unit == "h":
        return timedelta(hours=qty)
    if unit == "m":
        return timedelta(minutes=qty)
    if unit == "s":
        return timedelta(seconds=qty)
    # default days
    return timedelta(days=qty)


def _align_window_to_resolution(
    start: datetime,
    end: datetime,
    resolution: str,
) -> tuple[datetime, datetime]:
    # For daily windows we align to day boundaries to avoid odd query windows.
    if resolution and resolution[0].lower() == "d":
        start_aligned = start.astimezone(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        end_aligned = end.astimezone(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        if end_aligned <= start_aligned:
            end_aligned = start_aligned + timedelta(days=1)
        return start_aligned, end_aligned
    return start, end

def _is_payload_too_large(data) -> bool:
    if not isinstance(data, dict):
        return False
    msg = data.get("message") or data.get("error") or ""
    return "payload must be <=" in str(msg).lower()


def _extract_allowed_historical_days(data) -> int | None:
    if not isinstance(data, dict):
        return None

    details = data.get("details")
    candidates = []
    if isinstance(details, dict):
        candidates.append(details.get("allowed_historical_days"))
    candidates.append(data.get("allowed_historical_days"))

    for raw in candidates:
        if raw is None:
            continue
        try:
            days = int(raw)
        except (TypeError, ValueError):
            continue
        if days > 0:
            return days
    return None


def _extract_quota_details(data) -> tuple[int | None, int | None]:
    if not isinstance(data, dict):
        return None, None
    details = data.get("details")
    if not isinstance(details, dict):
        return None, None
    payload_cost = details.get("payload_cost")
    quota_remaining = details.get("quota_remaining")
    try:
        payload_cost_i = int(payload_cost) if payload_cost is not None else None
    except (TypeError, ValueError):
        payload_cost_i = None
    try:
        quota_remaining_i = int(quota_remaining) if quota_remaining is not None else None
    except (TypeError, ValueError):
        quota_remaining_i = None
    return payload_cost_i, quota_remaining_i


def _is_insufficient_payload_quota(data) -> bool:
    if not isinstance(data, dict):
        return False
    details = data.get("details")
    if not isinstance(details, dict):
        return False
    return str(details.get("reason") or "").upper() == "INSUFFICIENT_FOR_PAYLOAD"


def _normalize_records(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    return []


def _dataframe_time_column(df: pd.DataFrame) -> str | None:
    for candidate in ("time", "date", "Date", "datetime", "timestamp"):
        if candidate in df.columns:
            return candidate
    return None


def _min_time_value(df: pd.DataFrame, time_col: str):
    parsed = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None
    return parsed.min().to_pydatetime()


def _merge_endpoint_dataframe(conn, table_name: str, df: pd.DataFrame) -> int:
    df = df.drop_duplicates().copy()
    if df.empty:
        return 0

    table_present = table_exists(conn, table_name)
    incoming_time_col = _dataframe_time_column(df)
    existing_time_col = get_preferred_time_column(conn, table_name) if table_present else None

    if table_present and existing_time_col and incoming_time_col == existing_time_col:
        delete_from = _min_time_value(df, incoming_time_col)
        if delete_from is not None:
            df = df.drop_duplicates(subset=[incoming_time_col], keep="last")
            delete_rows_from_value(conn, table_name, existing_time_col, delete_from)
            return append_dataframe(conn, table_name, df)

    replace_table_from_dataframe(conn, table_name, df)
    return int(len(df))


def make_api_request(
    session: requests.Session,
    endpoint: str,
    params: dict,
    timeout: tuple[float, float] = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
):
    """
    Makes a single GET request with explicit connect/read timeouts.
    """
    try:
        resp = session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Network error: {e}", None

    ct = resp.headers.get("content-type", "")
    payload = resp.json() if "application/json" in ct else resp.text

    if resp.status_code >= 400:
        return payload, f"{resp.status_code} {resp.reason}", resp.status_code

    return payload, None, resp.status_code

def _fetch_range_records(
    session: requests.Session,
    endpoint: str,
    base_params: dict,
    start: datetime,
    end: datetime,
    resolution: str,
    depth: int = 0,
    max_depth: int = 20,
) -> list[dict]:
    if end <= start:
        return []
    if depth > max_depth:
        print("  -> Max pagination depth reached; skipping remaining range.")
        return []

    params = dict(base_params)
    params["from_time"] = _format_time(start)
    params["to_time"] = _format_time(end)

    data, err, status = make_api_request(session, endpoint, params=params)

    if status == 422 and _is_payload_too_large(data):
        print(f"  -> Payload too large; splitting range {params['from_time']} -> {params['to_time']}")
        min_span = _resolution_to_timedelta(resolution)
        if end - start <= min_span:
            print("  -> Range too small to split further; skipping.")
            return []
        mid = start + (end - start) / 2
        left = _fetch_range_records(session, endpoint, base_params, start, mid, resolution, depth + 1, max_depth)
        right = _fetch_range_records(session, endpoint, base_params, mid, end, resolution, depth + 1, max_depth)
        return left + right

    if status == 403:
        allowed_days = _extract_allowed_historical_days(data)
        if allowed_days is not None:
            # Keep the requested window safely inside the provider cap.
            safe_days = max(1, allowed_days - 1)
            allowed_start = end - timedelta(days=safe_days)
            adjusted_start = max(start, allowed_start)
            if adjusted_start > start:
                print(
                    "  -> Tier historical limit detected; retrying window "
                    f"from {_format_time(adjusted_start)} to {params['to_time']}"
                )
                return _fetch_range_records(
                    session=session,
                    endpoint=endpoint,
                    base_params=base_params,
                    start=adjusted_start,
                    end=end,
                    resolution=resolution,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

    if status == 402 and _is_insufficient_payload_quota(data):
        payload_cost, quota_remaining = _extract_quota_details(data)
        min_span = _resolution_to_timedelta(resolution)
        if quota_remaining is not None and quota_remaining <= 0:
            print("  -> Quota remaining is 0; skipping this endpoint window.")
            return []
        # Keep requests within remaining quota by shrinking to recent data only.
        if quota_remaining is not None:
            affordable_points = max(1, quota_remaining - 1)
            adjusted_start = max(start, end - (min_span * affordable_points))
        else:
            adjusted_start = start + ((end - start) / 2)
        if adjusted_start <= start:
            if payload_cost is not None and quota_remaining is not None:
                print(
                    "  -> Payload exceeds remaining quota "
                    f"(payload_cost={payload_cost}, quota_remaining={quota_remaining}); skipping."
                )
            return []
        print(
            "  -> Quota-limited payload; retrying smaller window "
            f"from {_format_time(adjusted_start)} to {params['to_time']}"
        )
        return _fetch_range_records(
            session=session,
            endpoint=endpoint,
            base_params=base_params,
            start=adjusted_start,
            end=end,
            resolution=resolution,
            depth=depth + 1,
            max_depth=max_depth,
        )

    if err:
        print(f"  -> Request failed: {err}")
        if isinstance(data, dict):
            details = data.get("details") or data.get("message") or data.get("error")
            if details:
                print(f"  -> Error details: {details}")
        return []

    return _normalize_records(data)


def fetch_full_endpoint(path: str, conn, session: requests.Session) -> None:
    endpoint = _url(path)
    
    # Create a clean table name (e.g., v2_network_statistics_hashrate)
    table_name = path.strip("/").replace("/", "_")

    params = _get_params()

    print(f"\n--- Fetching: {path} ---")
    print(f"Endpoint: {endpoint}")
    print(f"Params  : {params}")

    # 1. Fetch (with pagination if needed)
    try:
        start = _parse_time(params["from_time"])
        end = _parse_time(params["to_time"])
    except Exception as e:
        print(f"  -> Invalid time range: {e}")
        return

    base_params = dict(params)
    base_params.pop("from_time", None)
    base_params.pop("to_time", None)
    resolution = str(params.get("resolution") or "")
    start, end = _align_window_to_resolution(start, end, resolution)

    records = _fetch_range_records(
        session=session,
        endpoint=endpoint,
        base_params=base_params,
        start=start,
        end=end,
        resolution=resolution,
    )

    if not records:
        print("  -> No data returned.")
        return

    # 2. Convert to Pandas
    df = pd.DataFrame(records)
    
    # Basic cleanup
    df = df.drop_duplicates()

    if df.empty:
        print("  -> DataFrame is empty after processing.")
        return

    print(f"  -> Successfully retrieved {len(df)} rows.")
    print(f"  -> Columns: {list(df.columns)}")

    # 3. Save to DuckDB without truncating older rows when the API returns a
    # quota-limited or otherwise partial historical window.
    try:
        rows = _merge_endpoint_dataframe(conn, table_name, df)
        print(f"  -> Merged {rows} rows into DuckDB table: {table_name}")
    except Exception as e:
        print(f"  -> SQL Error: {e}")

def sync_endpoint(
    path: str,
    conn,
    session: requests.Session,
    overlap_points: int = DEFAULT_OVERLAP_POINTS,
) -> None:
    endpoint = _url(path)
    table_name = path.strip("/").replace("/", "_")

    params = _get_params()
    resolution = str(params.get("resolution") or "")
    resolution_delta = _resolution_to_timedelta(resolution)
    overlap_count = max(int(overlap_points), 1)
    try:
        end = _parse_time(params["to_time"])
        configured_start = _parse_time(params["from_time"])
        overlap_window_start = end - (resolution_delta * overlap_count)
    except Exception as e:
        print(f"  -> Invalid time range: {e}")
        return

    table_present = table_exists(conn, table_name)
    time_col = get_preferred_time_column(conn, table_name) if table_present else None
    overlap_start = (
        get_distinct_value_at_offset(
            conn,
            table_name,
            time_col,
            offset=max(int(overlap_points) - 1, 0),
        )
        if table_present and time_col
        else None
    )

    mode = "bootstrap"
    start = configured_start
    delete_from = start
    if table_present and overlap_start is not None:
        try:
            overlap_dt = pd.to_datetime(overlap_start, utc=True).to_pydatetime()
            # Start at the table overlap point so stale local tables backfill
            # their gap. If the API trims the request for quota reasons, the
            # write path below deletes only from the first returned timestamp.
            start = max(overlap_dt, configured_start)
            delete_from = start
            mode = "incremental"
        except Exception:
            mode = "bootstrap"
    elif table_present and not time_col:
        print(f"  -> No time column found in {table_name}; replacing table.")
    elif table_present:
        print(f"  -> No overlap point found in {table_name}; rebuilding full table.")

    start, end = _align_window_to_resolution(start, end, resolution)
    delete_from = start

    print(f"\n--- Syncing: {path} ({mode}) ---")
    print(f"Endpoint: {endpoint}")
    print(f"Window: {_format_time(start)} -> {_format_time(end)}")
    if overlap_start is not None:
        print(f"Overlap: {overlap_points} (from {overlap_start})")

    base_params = dict(params)
    base_params.pop("from_time", None)
    base_params.pop("to_time", None)

    records = _fetch_range_records(
        session=session,
        endpoint=endpoint,
        base_params=base_params,
        start=start,
        end=end,
        resolution=resolution,
    )

    if not records:
        print("  -> No data returned.")
        return

    df = pd.DataFrame(records).drop_duplicates()
    if df.empty:
        print("  -> DataFrame is empty after processing.")
        return

    try:
        if table_present and time_col:
            incoming_time_col = _dataframe_time_column(df)
            fetched_min = (
                _min_time_value(df, incoming_time_col)
                if incoming_time_col == time_col
                else None
            )
            delete_rows_from_value(conn, table_name, time_col, fetched_min or delete_from)
            append_dataframe(conn, table_name, df)
            print(f"  -> Appended {len(df)} rows into {table_name}.")
        else:
            replace_table_from_dataframe(conn, table_name, df)
            print(f"  -> Saved to DuckDB table: {table_name}")
    except Exception as e:
        print(f"  -> SQL Error: {e}")

def fetch_all(db_path: str = DB_PATH) -> dict:
    if not BITLAB_API_TOKEN or BITLAB_API_TOKEN.startswith(("xxxx", "PASTE_")):
        print("WARNING: Set BITLAB_API_TOKEN to your API token (env var is preferred).")

    with _build_session(BITLAB_API_TOKEN) as session, duckdb_connection(db_path) as conn:
        for path in dict.fromkeys(ENDPOINT_PATHS):
            fetch_full_endpoint(path, conn, session)

        print("\n=== Verification ===")
        tables = list_tables(conn)
        print(f"Tables in {db_path}: {tables}")
        return {
            "mode": "full",
            "db_path": db_path,
            "endpoint_count": len(dict.fromkeys(ENDPOINT_PATHS)),
            "tables": tables,
        }

def sync_all(
    overlap_points: int = DEFAULT_OVERLAP_POINTS,
    db_path: str = DB_PATH,
) -> dict:
    if not BITLAB_API_TOKEN or BITLAB_API_TOKEN.startswith(("xxxx", "PASTE_")):
        print("WARNING: Set BITLAB_API_TOKEN to your API token (env var is preferred).")
    print(
        "Onchain sync config: "
        f"overlap_points={overlap_points}, "
        f"connect_timeout_s={CONNECT_TIMEOUT_S}, "
        f"read_timeout_s={READ_TIMEOUT_S}, "
        f"http_retries={HTTP_RETRIES}"
    )

    with _build_session(BITLAB_API_TOKEN) as session, duckdb_connection(db_path) as conn:
        for path in dict.fromkeys(ENDPOINT_PATHS):
            sync_endpoint(path, conn, session, overlap_points=overlap_points)

        print("\n=== Verification ===")
        tables = list_tables(conn)
        print(f"Tables in {db_path}: {tables}")
        return {
            "mode": "sync",
            "db_path": db_path,
            "endpoint_count": len(dict.fromkeys(ENDPOINT_PATHS)),
            "overlap_points": int(overlap_points),
            "tables": tables,
        }
