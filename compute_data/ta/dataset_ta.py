import pandas as pd
import duckdb
from compute_data.ta import algorithms as ta
from database_interraction import duckdb_connection


def _align_series_to_index(series: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(series, index=index)
    if series.index.equals(index):
        return series
    if len(series) == len(index):
        return pd.Series(series.to_numpy(), index=index, name=series.name)
    return series.reindex(index)

def _as_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(index=value.index, dtype=float)
        return value.iloc[:, 0]
    return pd.Series(value)

def _ensure_datetime_index(
    df: pd.DataFrame,
    *,
    column: str = "timestamp",
    drop: bool = True,
) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.name != column:
            df = df.copy()
            df.index.name = column
        if drop and column in df.columns:
            df = df.drop(columns=[column])
        return df

    if column in df.columns:
        out = df.copy()
        out[column] = pd.to_datetime(out[column], errors="coerce")
        out = out.set_index(column, drop=drop)
        out.index.name = column
        return out

    if "Date" in df.columns:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.set_index("Date", drop=drop)
        out.index.name = column
        return out

    return df

def _df_with_datetime_column(df: pd.DataFrame, *, column: str = "timestamp") -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        idx_name = out.index.name or column
        if idx_name != column:
            out.index.name = column
        if column in out.columns:
            out = out.drop(columns=[column])
        out = out.reset_index()
        out[column] = pd.to_datetime(out[column], errors="coerce")
        return out

    if column in df.columns:
        out = df.copy()
        out[column] = pd.to_datetime(out[column], errors="coerce")
        return out

    if "Date" in df.columns:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out

    return df

def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?)
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return row is not None


def _normalize_timestamp_for_duckdb(timestamp: pd.Timestamp | str | None) -> object:
    if timestamp is None:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _normalize_cutoff_timestamp(timestamp: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if timestamp is None:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _table_max_timestamp(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    *,
    column: str = "timestamp",
) -> pd.Timestamp | None:
    if not _table_exists(con, table_name):
        return None
    table_ident = _quote_identifier(table_name)
    row = con.execute(
        f"SELECT MAX({column}) FROM {table_ident}"
    ).fetchone()
    if not row or row[0] is None:
        return None
    ts = pd.Timestamp(row[0])
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _latest_common_timestamp(
    con: duckdb.DuckDBPyConnection,
    table_names: list[str],
) -> pd.Timestamp | None:
    latest_values: list[pd.Timestamp] = []
    for table_name in table_names:
        ts = _table_max_timestamp(con, table_name)
        if ts is None:
            return None
        latest_values.append(ts)
    if not latest_values:
        return None
    return min(latest_values)


def _index_position_at_or_before(index: pd.Index, timestamp: pd.Timestamp) -> int:
    pos = int(index.searchsorted(timestamp, side="right")) - 1
    if pos < 0:
        return 0
    if pos >= len(index):
        return len(index) - 1
    return pos


def _slice_bounds_from_anchor(
    index: pd.Index,
    anchor_timestamp: pd.Timestamp | None,
    *,
    context_rows: int,
    overlap_rows: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if index.empty:
        raise ValueError("Cannot compute slice bounds for empty index")

    if anchor_timestamp is None:
        first = pd.Timestamp(index[0])
        return first, first

    anchor_pos = _index_position_at_or_before(index, anchor_timestamp)
    write_pos = max(0, anchor_pos - max(0, int(overlap_rows)))
    compute_pos = max(0, write_pos - max(0, int(context_rows)))
    return pd.Timestamp(index[compute_pos]), pd.Timestamp(index[write_pos])

def load_1d_ohlcv(
    symbol: str,
    *,
    db_path: str = "database/ohlcv.duckdb",
    table_name: str = "ohlcv",
    interval: str = "1d",
) -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as con:
        return _load_1d_ohlcv(con, symbol, table_name=table_name, interval=interval)

def _load_1d_ohlcv(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    *,
    table_name: str = "ohlcv",
    interval: str = "1d",
) -> pd.DataFrame:
    table_ident = _quote_identifier(table_name)
    df = con.execute(
        f"""
        SELECT open_time AS timestamp, open, high, low, close, volume
        FROM {table_ident}
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time
        """,
        [symbol, interval],
    ).fetchdf()
    if df.empty:
        raise ValueError(f"No {table_name} rows found for symbol {symbol} interval {interval}")
    return _ensure_datetime_index(df, drop=True)

def save_df_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    df_to_save = _df_with_datetime_column(df)
    table_ident = _quote_identifier(table_name)
    con.register("tmp_df", df_to_save)
    con.execute(f"CREATE OR REPLACE TABLE {table_ident} AS SELECT * FROM tmp_df;")
    con.unregister("tmp_df")


def upsert_df_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
    *,
    from_timestamp: pd.Timestamp | None = None,
) -> int:
    if df.empty:
        return 0

    df_to_save = _df_with_datetime_column(df)
    table_ident = _quote_identifier(table_name)
    con.register("tmp_df", df_to_save)
    try:
        if not _table_exists(con, table_name):
            con.execute(f"CREATE TABLE {table_ident} AS SELECT * FROM tmp_df ORDER BY timestamp;")
        else:
            if from_timestamp is None:
                con.execute(f"DELETE FROM {table_ident};")
            else:
                con.execute(
                    f"DELETE FROM {table_ident} WHERE timestamp >= ?;",
                    [_normalize_timestamp_for_duckdb(from_timestamp)],
                )
            con.execute(f"INSERT INTO {table_ident} SELECT * FROM tmp_df ORDER BY timestamp;")
    finally:
        try:
            con.unregister("tmp_df")
        except Exception:
            pass
    return int(len(df_to_save))


def categorical_ta_parts(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_datetime_index(df, drop=True)
    date_index = df.index

    vzo, pzo = ta.VolumeZonePriceOscillator(df)
    MR = pd.DataFrame({
        'MR1' : _align_series_to_index(ta.standerdise(ta.Full_VZScore(df)), date_index),
        'MR2' : _align_series_to_index(ta.standerdise(ta.UCS_MurreyMath(df)), date_index),
        'MR3' : _align_series_to_index(ta.standerdise(ta.CommodityChannelIndex(df)), date_index),
        'MR4' : _align_series_to_index(ta.standerdise(ta.RelativeStrengthIndex(df)), date_index),
        'MR5' : _align_series_to_index(ta.standerdise(vzo), date_index),
        'MR6' : _align_series_to_index(ta.standerdise(pzo), date_index),
    })

    Trend = pd.DataFrame({
        'Trend1': _align_series_to_index(ta.normalize(ta.OptimalLeverage(df)), date_index),
        'Trend2': _align_series_to_index(ta.normalize(ta.DetrendPrice(df)), date_index),
        'Trend3': _align_series_to_index(ta.normalize(ta.SchaffTrendCycle(df)), date_index),
        'Trend4': _align_series_to_index(ta.normalize(ta.GunzoTrendSniper(df)), date_index),
        'Trend5': _align_series_to_index(ta.normalize(ta.RMITrendSniper(df)), date_index),
        'Trend6': _align_series_to_index(ta.normalize(ta.TASC2024_09PrecisionTrendAnalysis(df)), date_index),
        'Trend7': _align_series_to_index(ta.normalize(ta.DynamicVolumeRSI(df)), date_index),
    })

    Mreg = pd.DataFrame({
        'MReg1': _align_series_to_index(ta.normalize(ta.rolling_adf(df, 90)), date_index),
        'MReg2': _align_series_to_index(ta.normalize(ta.rolling_kpss(df, 40, 'c')), date_index),
        'MReg3': _align_series_to_index(ta.normalize(ta.rolling_zivot(df)), date_index),
        'MReg4': _align_series_to_index(ta.normalize(ta.rolling_pp(df)) + 0.25, date_index),
        'MReg5': _align_series_to_index(ta.normalize(ta.ehlers_daily_regime(df['close'], snr_threshold=6.0)), date_index,),
    })

    momentum = pd.DataFrame({
        "mom1": _align_series_to_index(ta.normalize(_as_series(ta.sqzmom_hma_momentum(df))),date_index,),
        "mom2": _align_series_to_index(ta.normalize(ta.intraday_momentum_index(df)),date_index,),
        "mom3": _align_series_to_index(ta.normalize(_as_series(ta.ccmi_lazybear(df))),date_index,),
        "mom4": _align_series_to_index(ta.normalize(ta.stoch_momentum_index_slow(df)),date_index,),
        "mom5": _align_series_to_index(ta.normalize(ta.ehlers_sami_lazybear(df)),date_index,),
        "mom6": _align_series_to_index(ta.normalize(ta.ma_mtf_momentum_histogram(df)),date_index,),
    })

    volatility = pd.DataFrame({
        "vol1": _align_series_to_index(ta.standerdise(ta.ls_normalized(df)),date_index,),
        "vol2": _align_series_to_index(ta.standerdise(ta.ls_volatility(df)),date_index,),
        "vol3": _align_series_to_index(ta.standerdise(ta.calc_volatility_this(df)),date_index,),
    })

    return MR, Trend, Mreg, momentum, volatility

def categorical_ta_df(df: pd.DataFrame) -> pd.DataFrame:
    MR, Trend, Mreg, momentum, volatility = categorical_ta_parts(df)
    out = pd.concat([MR, Trend, Mreg, momentum, volatility], axis=1).fillna(0)
    out.index.name = "timestamp"
    return out

def build_1d_ta_db(
    symbol: str,
    *,
    ohlcv_db_path: str = "database/ohlcv.duckdb",
    output_db_path: str = "database/1d_ta.duckdb",
    context_rows: int = 1500,
    overlap_rows: int = 8,
    end_timestamp: pd.Timestamp | str | None = None,
) -> dict[str, pd.DataFrame]:
    symbol_u = symbol.upper()
    with duckdb_connection(ohlcv_db_path, read_only=True) as con:
        price_history = _load_1d_ohlcv(con, symbol_u, table_name="ohlcv", interval="1d")
    cutoff = _normalize_cutoff_timestamp(end_timestamp)
    if cutoff is not None:
        price_history = price_history.loc[price_history.index <= cutoff]
        if price_history.empty:
            raise ValueError(f"No OHLCV rows are available on or before {cutoff}.")

    mr_table = f"{symbol_u}_1d_MR"
    trend_table = f"{symbol_u}_1d_Trend"
    mreg_table = f"{symbol_u}_1d_MReg"
    momentum_table = f"{symbol_u}_1d_Momentum"
    volatility_table = f"{symbol_u}_1d_Volatility"
    with duckdb_connection(output_db_path) as con:
        latest_saved = _latest_common_timestamp(
            con,
            [mr_table, trend_table, mreg_table, momentum_table, volatility_table],
        )

    if cutoff is not None:
        compute_start = pd.Timestamp(price_history.index[0])
        write_start = compute_start
        mode = "as_of_full"
    elif latest_saved is None:
        compute_start = pd.Timestamp(price_history.index[0])
        write_start = compute_start
        mode = "full"
    else:
        compute_start, write_start = _slice_bounds_from_anchor(
            price_history.index,
            latest_saved,
            context_rows=context_rows,
            overlap_rows=overlap_rows,
        )
        mode = "incremental"

    price_slice = price_history.loc[compute_start:]
    mr_df, trend_df, mreg_df, momentum_df, volatility_df = categorical_ta_parts(price_slice)

    mr_write = mr_df.loc[mr_df.index >= write_start]
    trend_write = trend_df.loc[trend_df.index >= write_start]
    mreg_write = mreg_df.loc[mreg_df.index >= write_start]
    momentum_write = momentum_df.loc[momentum_df.index >= write_start]
    volatility_write = volatility_df.loc[volatility_df.index >= write_start]

    with duckdb_connection(output_db_path) as con:
        written_mr = upsert_df_to_duckdb(
            con,
            mr_write,
            mr_table,
            from_timestamp=write_start if mode == "incremental" else None,
        )
        written_trend = upsert_df_to_duckdb(
            con,
            trend_write,
            trend_table,
            from_timestamp=write_start if mode == "incremental" else None,
        )
        written_mreg = upsert_df_to_duckdb(
            con,
            mreg_write,
            mreg_table,
            from_timestamp=write_start if mode == "incremental" else None,
        )
        written_momentum = upsert_df_to_duckdb(
            con,
            momentum_write,
            momentum_table,
            from_timestamp=write_start if mode == "incremental" else None,
        )
        written_volatility = upsert_df_to_duckdb(
            con,
            volatility_write,
            volatility_table,
            from_timestamp=write_start if mode == "incremental" else None,
        )
        print(
            f"stored {mr_table} mode={mode} rows={written_mr} write_start={write_start} db={output_db_path}"
        )
        print(
            f"stored {trend_table} mode={mode} rows={written_trend} write_start={write_start} db={output_db_path}"
        )
        print(
            f"stored {mreg_table} mode={mode} rows={written_mreg} write_start={write_start} db={output_db_path}"
        )
        print(
            f"stored {momentum_table} mode={mode} rows={written_momentum} write_start={write_start} db={output_db_path}"
        )
        print(
            f"stored {volatility_table} mode={mode} rows={written_volatility} write_start={write_start} db={output_db_path}"
        )

    return {
        "MR": mr_write,
        "Trend": trend_write,
        "MReg": mreg_write,
        "Momentum": momentum_write,
        "Volatility": volatility_write,
    }
