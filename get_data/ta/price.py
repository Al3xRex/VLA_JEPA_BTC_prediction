from __future__ import annotations

import time
import os
from typing import Optional, Dict, Any, List

import requests
import pandas as pd

from database_interraction import (
    delete_rows_from_value,
    duckdb_connection,
    ensure_ohlcv_table,
    get_distinct_value_at_offset,
    get_max_value,
    get_min_value,
    upsert_dataframe,
)

BINANCE_BASE_URLS = tuple(
    base.strip().rstrip("/")
    for base in (
        os.getenv("BINANCE_BASE_URLS")
        or "https://data-api.binance.vision,https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com"
    ).split(",")
    if base.strip()
)
MAX_LIMIT = 1000  # Binance max for klines

OHLCV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "close_time",
    "available_at",
    "is_closed",
]


def binance_klines_to_frame(
    klines: List[List[Any]],
    *,
    observed_at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Convert Binance klines while preserving their availability boundary.

    Binance timestamps OHLCV rows by candle open, but high/low/close/volume are
    final only at ``closeTime``.  We therefore retain both clocks and mark an
    in-progress candle without discarding it; causal readers can exclude it,
    while the next overlap refresh can still revise it in place.
    """

    if not klines:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    cutoff = pd.Timestamp.now(tz="UTC") if observed_at is None else pd.Timestamp(observed_at)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    rows: list[tuple[Any, ...]] = []
    for kline in klines:
        if len(kline) < 11:
            raise ValueError(f"Binance kline has {len(kline)} fields; expected at least 11.")
        open_time = pd.to_datetime(int(kline[0]), unit="ms", utc=True)
        close_time = pd.to_datetime(int(kline[6]), unit="ms", utc=True)
        rows.append(
            (
                open_time,
                float(kline[1]),
                float(kline[2]),
                float(kline[3]),
                float(kline[4]),
                float(kline[5]),
                int(kline[8]),
                float(kline[7]),
                float(kline[9]),
                float(kline[10]),
                close_time,
                close_time,
                bool(close_time <= cutoff),
            )
        )
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    return frame.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def sync_binance_ohlcv_to_duckdb(
    symbol: str,
    interval: str,
    db_path: str = "database/ohlcv.duckdb",
    table_name: str = "ohlcv",
    overlap_candles: int = 3,
    polite_sleep_s: float = 0.15,
    timeout_s: int = 20,
) -> Dict[str, Any]:
    """
    Fetch the *entire available* OHLCV history for (symbol, interval) from Binance Spot,
    store in DuckDB, and keep it updated via overlap delete+upsert.

    Requirements satisfied:
      - call parameters: interval, symbol
      - store results in database/ohlcv.duckdb (default db_path)
      - if already stored: upsert new data, deleting past candles & updating them (overlap window)

    Storage model:
      - primary key: (symbol, interval, open_time)
      - open_time is UTC timestamp of the candle OPEN (kline[0]) for consistent semantics
    """

    symbol_u = symbol.upper()
    interval_u = interval.lower()

    def _binance_get_klines(
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = MAX_LIMIT,
    ) -> List[List[Any]]:
        params: Dict[str, Any] = {
            "symbol": symbol_u,
            "interval": interval_u,
            "limit": int(limit),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        errors: List[str] = []
        for base_url in BINANCE_BASE_URLS:
            url = f"{base_url}/api/v3/klines"
            for attempt in range(8):
                try:
                    r = requests.get(url, params=params, timeout=timeout_s)
                except requests.RequestException as exc:
                    errors.append(f"{base_url} network error: {exc}")
                    time.sleep(0.5 + attempt * 0.5)
                    continue

                if r.status_code == 451:
                    errors.append(f"{base_url} returned 451 (geo-restricted)")
                    break
                if r.status_code in (418, 429):
                    time.sleep(1.0 + attempt * 1.5)
                    continue
                if 500 <= r.status_code < 600:
                    errors.append(f"{base_url} returned {r.status_code}")
                    time.sleep(0.5 + attempt * 0.5)
                    continue

                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"Unexpected Binance response: {data}")
                return data

        details = "; ".join(errors[-5:]) if errors else "no endpoint details"
        raise RuntimeError(
            f"Unable to fetch Binance klines from configured endpoints ({details})"
        )

    def _klines_to_df(klines: List[List[Any]]) -> pd.DataFrame:
        return binance_klines_to_frame(klines)

    def _prepare_dataframe_for_storage(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        dfw = df.copy()
        dfw["symbol"] = symbol_u
        dfw["interval"] = interval_u
        dfw["open_time"] = pd.to_datetime(dfw["open_time"], utc=True).dt.tz_localize(None)
        for column in ("close_time", "available_at"):
            dfw[column] = pd.to_datetime(dfw[column], utc=True).dt.tz_localize(None)
        dfw["is_closed"] = dfw["is_closed"].astype(bool)
        return dfw.drop_duplicates(subset=["symbol", "interval", "open_time"], keep="last")

    def _fetch_forward_from_start(start_ms: Optional[int]) -> pd.DataFrame:
        """
        Paginate forward using startTime. If start_ms is None, we *must* discover earliest.
        Binance won't give earliest for None, it gives latest 1000. So we use start_ms=0 for full history.
        """
        all_dfs: List[pd.DataFrame] = []
        next_start = 0 if start_ms is None else int(start_ms)

        while True:
            klines = _binance_get_klines(start_time_ms=next_start, limit=MAX_LIMIT)
            if not klines:
                break

            df_page = _klines_to_df(klines)
            if not df_page.empty:
                all_dfs.append(df_page)

            if len(klines) < MAX_LIMIT:
                break

            last_open_ms = int(klines[-1][0])
            # step one ms forward to avoid repeating last candle if interval is weird
            next_start = last_open_ms + 1

            if polite_sleep_s:
                time.sleep(polite_sleep_s)

            # paranoia break
            if next_start <= last_open_ms:
                break

        if not all_dfs:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return df

    # ---------------- actual workflow ----------------
    table_filters = {"symbol": symbol_u, "interval": interval_u}

    with duckdb_connection(db_path) as con:
        ensure_ohlcv_table(con, table_name)

        latest_value = get_max_value(
            con,
            table_name,
            "open_time",
            filters=table_filters,
        )
        latest = pd.to_datetime(latest_value, utc=True) if latest_value is not None else None
        if latest is None:
            # Not stored yet: full history from epoch
            df_new = _fetch_forward_from_start(start_ms=0)
            inserted = upsert_dataframe(
                con,
                table_name,
                _prepare_dataframe_for_storage(df_new),
                key_columns=("symbol", "interval", "open_time"),
            )
            deleted = 0
            mode = "full"
        else:
            # Stored: overlap delete to fix revised candles / mid-day bugs / general human suffering
            overlap_value = get_distinct_value_at_offset(
                con,
                table_name,
                "open_time",
                offset=max(int(overlap_candles), 0),
                filters=table_filters,
            )
            if overlap_value is not None:
                overlap_start = pd.to_datetime(overlap_value, utc=True)
            else:
                overlap_min_value = get_min_value(
                    con,
                    table_name,
                    "open_time",
                    filters=table_filters,
                )
                overlap_start = (
                    pd.to_datetime(overlap_min_value, utc=True)
                    if overlap_min_value is not None
                    else None
                )
            if overlap_start is None:
                overlap_start = latest

            deleted = delete_rows_from_value(
                con,
                table_name,
                "open_time",
                overlap_start,
                filters=table_filters,
            )

            start_ms = int(pd.to_datetime(overlap_start, utc=True).timestamp() * 1000)
            df_new = _fetch_forward_from_start(start_ms=start_ms)

            inserted = upsert_dataframe(
                con,
                table_name,
                _prepare_dataframe_for_storage(df_new),
                key_columns=("symbol", "interval", "open_time"),
            )
            mode = "incremental"
    
    print(symbol + " up to date")

    return {
        "symbol": symbol_u,
        "interval": interval_u,
        "db_path": db_path,
        "table": table_name,
        "mode": mode,
        "latest_before": None if latest is None else str(latest),
        "overlap_candles": int(overlap_candles),
        "rows_fetched": int(len(df_new)),
        "rows_upserted": int(inserted),
        "rows_deleted_from_overlap": int(deleted),
    }
    
