import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow running as a script (python value_ind/fred.py) by adding project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database_interraction import duckdb_connection, save_dataframe_to_table
from get_data.macro.fred import DB_PATH, TABLE_NAME, load_macro_from_db


# ==========================================
# 1. MAIN
# ==========================================
def construct_eco_data(
    end_date: pd.Timestamp | None = None,
    *,
    fred_db_path: str = DB_PATH,
    fred_table: str = TABLE_NAME,
):
    """
    Load and build macro/liquidity features from FRED.
    If end_date is provided, extend (ffill) the series to that date.
    """
    print(f"Loading macro data from DuckDB ({fred_db_path})...")

    df = load_macro_from_db(db_path=fred_db_path, table_name=fred_table)
    if df.empty:
        raise RuntimeError(
            f"No macro data found in {fred_db_path}. "
            "Run the macro ingestion step to fetch/store the configured FRED and manual series."
        )

    df = df.sort_index()
    end_ts = None
    if end_date is not None:
        end_ts = pd.to_datetime(end_date).normalize()
        df = df.loc[df.index <= end_ts]
        if df.empty:
            raise RuntimeError(f"No macro data found on or before {end_ts}.")

    # Build a DAILY index from first available to desired end date
    full_end = df.index.max()
    if end_ts is not None and end_ts > full_end:
        full_end = end_ts
    full_index = pd.date_range(df.index.min(), full_end, freq="D")

    # Reindex to daily and forward-fill:
    df = df.reindex(full_index).ffill()

    # master index for everything
    index = df.index

    # helper: ensure we always have a numeric Series aligned to index
    def safe_series(s, index):
        if isinstance(s, pd.Series):
            return s.reindex(index).astype(float).fillna(0.0)
        else:
            return pd.Series(0.0, index=index)

    def resolve_series(*column_candidates: str) -> pd.Series:
        for column_name in column_candidates:
            if column_name in df.columns:
                return safe_series(df.get(column_name), index)
        return safe_series(None, index)

    # helper: drop leading zeros (placeholders) and keep the rest
    def trim_leading_zeros(series: pd.Series) -> pd.Series:
        s = series.replace(0, np.nan)
        start_idx = s.first_valid_index()
        if start_idx is not None:
            s = s.loc[start_idx:]
        return s

    # pull series safely
    walcl = resolve_series("Fed_Assets")             # millions
    tga   = resolve_series("TGA")                    # millions
    rrp_b = resolve_series("Reverse_Repo_Billions")  # billions
    btfp  = resolve_series("BTFP")                   # millions
    pc    = resolve_series("Primary_Credit")         # millions
    
    # ==========================================
    # 2. UNIT CONVERSION & FORMULA
    # ==========================================
    # convert millions to billions
    walcl_b = walcl / 1000.0
    tga_b   = tga   / 1000.0
    btfp_b  = btfp  / 1000.0
    pc_b    = pc    / 1000.0
    # rrp_b already in billions

    # Your formula (in billions):
    # WALCL - WTREGEN - RRPONTSYD + H41RESPPALDKNWW + WLCFLPCL
    liq_bil_daily = walcl_b - tga_b - rrp_b + btfp_b + pc_b

    # Drop NaNs
    liq_bil_daily = liq_bil_daily.dropna()
    
    # *** KEY FIX: remove fake leading zeros ***
    # Treat 0 as "no real value yet" and start where it becomes non-zero
    liq_bil_daily = trim_leading_zeros(liq_bil_daily).rename("liq_bil_daily")
    liq_bil_daily.index.name = "Date"


    def prep_series(*column_candidates: str) -> pd.Series:
        """Align to master index and drop placeholder zeros at the front."""
        return trim_leading_zeros(resolve_series(*column_candidates))

    # ---------------------------
    # DataFrame 1: liquidity set
    # ---------------------------
    m2 = prep_series(
        "M2",
        "M2 Supply of Four Major Central Banks (USD, L)",
        "M2 Supply of Four Major Central Banks (USD, YoY, R)",
        "M2 Supply of Four Major Central Banks (Fixed Exchange Rate, YoY, R)",
    ).rename("m2")
    # shift the series forward by 6 weeks (daily data -> 42 days)
    # i.e., the value at date t is stored at t + 42 days (future dates preserved)
    m2_shift_fwd_6w = m2.shift(6 * 7, freq="D").rename("m2_shift_fwd_6w")

    dxy = prep_series("dxy")
    inv_dxy = (1.0 / dxy.replace(0, np.nan)).rename("inv_dxy")

    liq_m2_dxy = pd.concat(
        [liq_bil_daily, m2_shift_fwd_6w, inv_dxy],
        axis=1,
    )
    liq_m2_dxy.index.name = "Date"
    liq_m2_dxy = liq_m2_dxy.dropna(how="all").dropna(axis=1, how="all")

    # ---------------------------
    # DataFrame 2: other macro
    # ---------------------------
    gold = prep_series("gold").rename("gold")
    copper = prep_series("copper").rename("copper")
    gold_over_copper = (gold / copper.replace(0, np.nan)).rename("gold_over_copper")
    
    ba = prep_series("Buisness_activity").rename("Buisness_activity")
    spx = prep_series("spx").rename("spx")
    oil = prep_series("oil").rename("oil")
    em = prep_series("emerging_markets").rename("emerging_markets")

    macro_rest = pd.concat(
        [gold, copper, gold_over_copper, ba, spx, oil, em],
        axis=1,
    )
    macro_rest.index.name = "Date"
    macro_rest = macro_rest.dropna(how="all").dropna(axis=1, how="all")

    return liq_m2_dxy, macro_rest


def load_btc_close(
    db_path: str = "database/ohlcv.duckdb",
    table_name: str = "ohlcv",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
) -> pd.Series:
    """Load BTC close from DuckDB and return as Date-indexed Series."""
    with duckdb_connection(db_path, read_only=True) as con:
        table_ident = '"' + table_name.replace('"', '""') + '"'
        try:
            df = con.execute(
                f"""
                SELECT
                    CAST(open_time AS DATE) AS Date,
                    close AS close
                FROM {table_ident}
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time
                """,
                [symbol.upper(), interval.lower()],
            ).fetchdf()
        except Exception:
            df = con.execute(
                """
                SELECT
                    CAST(Date AS DATE) AS Date,
                    Close AS close
                FROM btc_history
                ORDER BY Date
                """
            ).fetchdf()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["close"]


def load_manual_macro_data(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> pd.DataFrame:
    return load_macro_from_db(
        db_path=db_path,
        table_name=table_name,
        source="manual",
    )


def store_macro_features(
    *,
    orvian_db_path: str = "orvian.duckdb",
    liq_table: str = "macro_liq_features",
    macro_table: str = "macro_factors",
    fred_db_path: str = DB_PATH,
    fred_table: str = TABLE_NAME,
    align_to_price: bool = True,
    price_db_path: str = "database/ohlcv.duckdb",
    price_table: str = "ohlcv",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
) -> dict[str, int]:
    end_date = None
    if align_to_price:
        btc_close = load_btc_close(
            db_path=price_db_path,
            table_name=price_table,
            symbol=symbol,
            interval=interval,
        )
        if not btc_close.empty:
            end_date = btc_close.index.max()

    liq_m2_dxy, macro_rest = construct_eco_data(
        end_date=end_date,
        fred_db_path=fred_db_path,
        fred_table=fred_table,
    )

    rows: dict[str, int] = {liq_table: 0, macro_table: 0}
    if not liq_m2_dxy.empty:
        rows[liq_table] = save_dataframe_to_table(
            liq_m2_dxy,
            liq_table,
            db_path=orvian_db_path,
            preserve_index=True,
        )
    if not macro_rest.empty:
        rows[macro_table] = save_dataframe_to_table(
            macro_rest,
            macro_table,
            db_path=orvian_db_path,
            preserve_index=True,
        )

    return rows
