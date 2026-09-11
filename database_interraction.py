from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import duckdb
import pandas as pd


def ensure_database_directory(db_path: str) -> None:
    path = Path(db_path).expanduser()
    if str(path) == ":memory:":
        return
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def duckdb_connection(
    db_path: str,
    *,
    read_only: bool = False,
) -> Iterator[duckdb.DuckDBPyConnection]:
    ensure_database_directory(db_path)
    connection = duckdb.connect(db_path, read_only=read_only)
    try:
        yield connection
    finally:
        connection.close()


def quote_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("DuckDB identifier cannot be empty.")
    return '"' + identifier.replace('"', '""') + '"'


def _table_name_for_pragma(table_name: str) -> str:
    return table_name.replace("'", "''")


def _build_where_clause(
    filters: Mapping[str, Any] | None,
) -> tuple[str, list[Any]]:
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []
    for column_name, value in filters.items():
        clauses.append(f"{quote_identifier(column_name)} = ?")
        params.append(value)
    return " WHERE " + " AND ".join(clauses), params


def _register_dataframe(
    connection: duckdb.DuckDBPyConnection,
    dataframe: pd.DataFrame,
) -> str:
    temp_name = f"__codex_dfw_{abs(id(dataframe))}"
    connection.register(temp_name, dataframe)
    return temp_name


def _unregister_dataframe(
    connection: duckdb.DuckDBPyConnection,
    temp_name: str,
) -> None:
    try:
        connection.unregister(temp_name)
    except Exception:
        pass


def fetch_scalar(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any] | None = None,
) -> Any:
    row = connection.execute(query, params or []).fetchone()
    if row is None:
        return None
    return row[0]


def table_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> bool:
    row_count = fetch_scalar(
        connection,
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;",
        [table_name],
    )
    return bool(row_count)


def get_table_info(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> list[tuple[Any, ...]]:
    safe_table_name = _table_name_for_pragma(table_name)
    return connection.execute(f"PRAGMA table_info('{safe_table_name}')").fetchall()


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> list[str]:
    return [row[1] for row in get_table_info(connection, table_name)]


def get_column_type(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
) -> str | None:
    for row in get_table_info(connection, table_name):
        if len(row) >= 3 and str(row[1]) == column_name:
            return str(row[2] or "")
    return None


def get_preferred_time_column(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> str | None:
    columns = get_table_columns(connection, table_name)
    for candidate in ("time", "date", "timestamp", "datetime"):
        if candidate in columns:
            return candidate
    return None


def get_max_value(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    filters: Mapping[str, Any] | None = None,
) -> Any:
    safe_table = quote_identifier(table_name)
    safe_column = quote_identifier(column_name)
    where_clause, params = _build_where_clause(filters)
    return fetch_scalar(
        connection,
        f"SELECT MAX({safe_column}) FROM {safe_table}{where_clause};",
        params,
    )


def get_min_value(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    filters: Mapping[str, Any] | None = None,
) -> Any:
    safe_table = quote_identifier(table_name)
    safe_column = quote_identifier(column_name)
    where_clause, params = _build_where_clause(filters)
    return fetch_scalar(
        connection,
        f"SELECT MIN({safe_column}) FROM {safe_table}{where_clause};",
        params,
    )


def get_distinct_value_at_offset(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    offset: int = 0,
    filters: Mapping[str, Any] | None = None,
    descending: bool = True,
) -> Any:
    safe_table = quote_identifier(table_name)
    safe_column = quote_identifier(column_name)
    where_clause, params = _build_where_clause(filters)
    order_direction = "DESC" if descending else "ASC"

    query = f"""
        SELECT value
        FROM (
            SELECT DISTINCT {safe_column} AS value
            FROM {safe_table}{where_clause}
        )
        ORDER BY value {order_direction}
        OFFSET ?
        LIMIT 1;
    """
    return fetch_scalar(connection, query, [*params, max(int(offset), 0)])


def _coerce_temporal_value_for_column(
    column_type: str | None,
    value: Any,
) -> Any:
    if not column_type:
        return value

    normalized_type = column_type.upper()
    timestamp = pd.to_datetime(value, utc=True)

    if any(token in normalized_type for token in ("VARCHAR", "CHAR", "STRING", "TEXT")):
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "DATE" in normalized_type and "TIME" not in normalized_type:
        return timestamp.date()
    if "WITH TIME ZONE" in normalized_type or "TIMESTAMPTZ" in normalized_type:
        return timestamp.to_pydatetime()
    return timestamp.tz_localize(None).to_pydatetime()


def delete_rows_from_value(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    start_value: Any,
    filters: Mapping[str, Any] | None = None,
) -> int:
    safe_table = quote_identifier(table_name)
    safe_column = quote_identifier(column_name)
    where_clause, params = _build_where_clause(filters)
    cutoff = _coerce_temporal_value_for_column(
        get_column_type(connection, table_name, column_name),
        start_value,
    )

    if where_clause:
        full_where_clause = (
            f" WHERE {safe_column} >= ? AND {where_clause.removeprefix(' WHERE ')}"
        )
    else:
        full_where_clause = f" WHERE {safe_column} >= ?"

    result = connection.execute(
        f"DELETE FROM {safe_table}{full_where_clause};",
        [cutoff, *params],
    )
    try:
        return int(result.rowcount)
    except Exception:
        return -1


def replace_table_from_dataframe(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:
    safe_table = quote_identifier(table_name)
    temp_name = _register_dataframe(connection, dataframe)
    try:
        connection.execute(
            f"CREATE OR REPLACE TABLE {safe_table} AS SELECT * FROM {quote_identifier(temp_name)};"
        )
    finally:
        _unregister_dataframe(connection, temp_name)


def append_dataframe(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    dataframe: pd.DataFrame,
) -> int:
    table_columns = get_table_columns(connection, table_name)
    if not table_columns:
        raise RuntimeError(f"Table {table_name} not found for append.")

    missing_columns = [column for column in table_columns if column not in dataframe.columns]
    if missing_columns:
        raise RuntimeError(f"Missing columns in append: {missing_columns}")

    dataframe_to_write = dataframe[table_columns].copy()
    safe_table = quote_identifier(table_name)
    safe_columns = ", ".join(quote_identifier(column) for column in table_columns)
    temp_name = _register_dataframe(connection, dataframe_to_write)
    try:
        connection.execute(
            f"INSERT INTO {safe_table} ({safe_columns}) "
            f"SELECT {safe_columns} FROM {quote_identifier(temp_name)};"
        )
    finally:
        _unregister_dataframe(connection, temp_name)
    return int(len(dataframe_to_write))


def upsert_dataframe(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    dataframe: pd.DataFrame,
    key_columns: list[str] | tuple[str, ...],
) -> int:
    if dataframe.empty:
        return 0

    table_columns = get_table_columns(connection, table_name)
    if not table_columns:
        raise RuntimeError(f"Table {table_name} not found for upsert.")

    missing_columns = [column for column in table_columns if column not in dataframe.columns]
    if missing_columns:
        raise RuntimeError(f"Missing columns in upsert: {missing_columns}")

    missing_keys = [column for column in key_columns if column not in table_columns]
    if missing_keys:
        raise RuntimeError(f"Missing key columns in table {table_name}: {missing_keys}")

    dataframe_to_write = dataframe[table_columns].copy()
    key_list = list(key_columns)
    dataframe_to_write = dataframe_to_write.drop_duplicates(subset=key_list, keep="last")

    safe_table = quote_identifier(table_name)
    safe_columns = ", ".join(quote_identifier(column) for column in table_columns)
    safe_keys = ", ".join(quote_identifier(column) for column in key_list)
    temp_name = _register_dataframe(connection, dataframe_to_write)
    try:
        connection.execute(
            f"DELETE FROM {safe_table} "
            f"WHERE ({safe_keys}) IN (SELECT {safe_keys} FROM {quote_identifier(temp_name)});"
        )
        connection.execute(
            f"INSERT INTO {safe_table} ({safe_columns}) "
            f"SELECT {safe_columns} FROM {quote_identifier(temp_name)};"
        )
    finally:
        _unregister_dataframe(connection, temp_name)
    return int(len(dataframe_to_write))


def list_tables(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in connection.execute("SHOW TABLES").fetchall()]


def database_has_tables(db_path: str) -> bool:
    with duckdb_connection(db_path) as connection:
        return bool(list_tables(connection))


def load_table_as_dataframe(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: list[str] | tuple[str, ...] | None = None,
    filters: Mapping[str, Any] | None = None,
    order_by: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    safe_table = quote_identifier(table_name)
    selected_columns = (
        "*"
        if not columns
        else ", ".join(quote_identifier(column) for column in columns)
    )
    where_clause, params = _build_where_clause(filters)
    order_clause = ""
    if order_by:
        order_clause = " ORDER BY " + ", ".join(
            quote_identifier(column) for column in order_by
        )
    return connection.execute(
        f"SELECT {selected_columns} FROM {safe_table}{where_clause}{order_clause};",
        params,
    ).df()


def load_table_dataframe(
    db_path: str,
    table_name: str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    filters: Mapping[str, Any] | None = None,
    order_by: list[str] | tuple[str, ...] | None = None,
    read_only: bool = False,
) -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=read_only) as connection:
        return load_table_as_dataframe(
            connection,
            table_name,
            columns=columns,
            filters=filters,
            order_by=order_by,
        )


def save_dataframe_to_table(
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    db_path: str,
    preserve_index: bool = False,
) -> int:
    if dataframe.empty:
        return 0

    dataframe_to_write = dataframe.copy()
    if preserve_index:
        index_name = dataframe_to_write.index.name or "index"
        if index_name in dataframe_to_write.columns:
            dataframe_to_write = dataframe_to_write.drop(columns=[index_name])
        dataframe_to_write = dataframe_to_write.reset_index()

    with duckdb_connection(db_path) as connection:
        replace_table_from_dataframe(connection, table_name, dataframe_to_write)
        row_count = fetch_scalar(
            connection,
            f"SELECT COUNT(*) FROM {quote_identifier(table_name)};",
        )
    return int(row_count or 0)


def drop_table(
    db_path: str,
    table_name: str,
) -> None:
    with duckdb_connection(db_path) as connection:
        connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)};")


def ensure_ohlcv_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    safe_table = quote_identifier(table_name)
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            trades BIGINT,
            quote_volume DOUBLE,
            taker_buy_base_volume DOUBLE,
            taker_buy_quote_volume DOUBLE,
            close_time TIMESTAMP,
            available_at TIMESTAMP,
            is_closed BOOLEAN,
            PRIMARY KEY(symbol, interval, open_time)
        );
        """
    )

    # Older databases predate explicit bar-availability metadata.  Keep the
    # migration additive so existing callers and primary keys remain intact.
    columns = set(get_table_columns(connection, table_name))
    availability_columns = {
        "close_time": "TIMESTAMP",
        "available_at": "TIMESTAMP",
        "is_closed": "BOOLEAN",
    }
    for column, column_type in availability_columns.items():
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {safe_table} ADD COLUMN {quote_identifier(column)} {column_type};"
            )

    # Backfill common Binance intervals for legacy rows.  The one-millisecond
    # subtraction matches Binance's inclusive kline close-time convention.
    # Unknown/custom interval strings remain NULL and are handled conservatively
    # by readers rather than being assigned a fabricated availability time.
    amount = (
        "TRY_CAST(regexp_extract(interval, '^([0-9]+)', 1) AS BIGINT)"
    )
    inferred_close = f"""
        CASE
            WHEN regexp_matches(interval, '^[0-9]+s$') THEN open_time + to_seconds({amount})
            WHEN regexp_matches(interval, '^[0-9]+m$') THEN open_time + to_minutes({amount})
            WHEN regexp_matches(interval, '^[0-9]+h$') THEN open_time + to_hours({amount})
            WHEN regexp_matches(interval, '^[0-9]+d$') THEN open_time + to_days({amount})
            WHEN regexp_matches(interval, '^[0-9]+w$') THEN open_time + to_days(7 * {amount})
            WHEN regexp_matches(interval, '^[0-9]+M$') THEN open_time + to_months({amount})
            ELSE NULL
        END - INTERVAL 1 MILLISECOND
    """
    connection.execute(
        f"UPDATE {safe_table} SET close_time = {inferred_close} WHERE close_time IS NULL;"
    )
    connection.execute(
        f"UPDATE {safe_table} SET available_at = close_time "
        "WHERE available_at IS NULL AND close_time IS NOT NULL;"
    )
    connection.execute(
        f"UPDATE {safe_table} SET is_closed = "
        "available_at <= CAST(CURRENT_TIMESTAMP AS TIMESTAMP) "
        "WHERE available_at IS NOT NULL;"
    )


def ensure_macro_series_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    safe_table = quote_identifier(table_name)
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
            date DATE NOT NULL,
            source TEXT NOT NULL,
            series_id TEXT NOT NULL,
            series_name TEXT,
            value DOUBLE,
            PRIMARY KEY(date, source, series_id)
        );
        """
    )
