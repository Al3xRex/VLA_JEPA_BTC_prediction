import numpy as np
import pandas as pd

from database_interraction import duckdb_connection, save_dataframe_to_table



def load_price_history(
    db_path: str = "database/ohlcv.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
) -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as con:
        df = con.execute("""
            SELECT
                CAST(open_time AS DATE) AS Date,
                close AS Close
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time;
        """, [symbol.upper(), interval.lower()]).fetchdf()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_address_counts(db_path: str = "database/onchain.duckdb") -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as con:
        df = con.execute("""
            WITH n AS (
                SELECT
                    CAST(time AS DATE) AS Date,
                    "group" AS cohort_group,
                    CAST(addresses_by_btc_n AS DOUBLE) AS cohort_n
                FROM v2_address_statistics_addresses_by_btc_n
            ),
            s AS (
                SELECT
                    CAST(time AS DATE) AS Date,
                    "group" AS cohort_group,
                    CAST(addresses_by_btc_sumbtc AS DOUBLE) AS cohort_sumbtc
                FROM v2_address_statistics_addresses_by_btc_sumbtc
            ),
            cohorts AS (
                SELECT
                    n.Date,
                    n.cohort_group,
                    n.cohort_n,
                    s.cohort_sumbtc,
                    s.cohort_sumbtc / NULLIF(n.cohort_n, 0) AS cohort_avgbtc
                FROM n
                JOIN s
                  ON n.Date = s.Date
                 AND n.cohort_group = s.cohort_group
            )
            SELECT
                Date,
                SUM(cohort_n) AS all_addresses_count,
                SUM(cohort_sumbtc) AS total_btc,
                SUM(cohort_sumbtc) / NULLIF(SUM(cohort_n), 0) AS avg_btc_per_address
            FROM cohorts
            GROUP BY Date
            ORDER BY Date;
        """).fetchdf()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_metcalfe_series(
    price_db_path: str = "database/ohlcv.duckdb",
    onchain_db_path: str = "database/onchain.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    end_timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    price = load_price_history(price_db_path, symbol=symbol, interval=interval)
    addrs = load_address_counts(onchain_db_path)
    if end_timestamp is not None:
        cutoff = pd.Timestamp(end_timestamp)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        price = price.loc[price["Date"] <= cutoff]
        addrs = addrs.loc[addrs["Date"] <= cutoff]

    # basic cleaning before merge
    price = price.replace([np.inf, -np.inf], np.nan)
    price = price.dropna(subset=["Close"])
    price = price[price["Close"] > 0]

    addrs = addrs.replace([np.inf, -np.inf], np.nan)
    addrs = addrs.dropna(subset=["all_addresses_count"])
    addrs = addrs[addrs["all_addresses_count"] > 0]

    # inner join
    df = pd.merge(price, addrs, on="Date", how="inner")

    print("Merged date range:", df["Date"].min(), "→", df["Date"].max())
    print("Merged rows:", len(df))

    if df.empty:
        print("[DEBUG] merged df is empty")
        return pd.DataFrame(
            columns=[
                "Date",
                "Close",
                "all_addresses_count",
                "total_btc",
                "avg_btc_per_address",
                "metcalfe_raw",
                "metcalfe_price",
            ]
        )

    df["metcalfe_raw"] = df["all_addresses_count"].astype(float) ** 2

    # simple k * n^2 scaling
    ratio = df["Close"] / df["metcalfe_raw"]
    k = np.median(ratio[np.isfinite(ratio)])
    k = float(k)      # force normal python float
    df["metcalfe_price"] = df["metcalfe_raw"].astype(float) * k

    return df

def save_series():
    metcalfe_series = build_metcalfe_series()

    if metcalfe_series.empty:
        print("No Metcalfe data to save.")
        return

    # enforce datetime and keep order
    metcalfe_series["Date"] = pd.to_datetime(metcalfe_series["Date"])
    metcalfe_series = metcalfe_series.sort_values("Date").set_index("Date")
    metcalfe_series.index.name = "Date"

    # keep only the estimated Metcalfe value (user-based)
    df = metcalfe_series[["metcalfe_price"]]

    # store with explicit Date index in DuckDB
    rows = save_dataframe_to_table(
        df,
        "value_oc_indicator",
        db_path="orvian.duckdb",
        preserve_index=True,
    )
    print(df)
    print("Rows stored:", rows)
