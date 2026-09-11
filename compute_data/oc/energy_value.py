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

def load_onchain_core(db_path: str = "database/onchain.duckdb") -> pd.DataFrame:
    with duckdb_connection(db_path, read_only=True) as con:
        df = con.execute("""
            WITH hashrate AS (
                SELECT
                    CAST(time AS DATE) AS Date,
                    hashrate
                FROM v2_network_statistics_hashrate
            ),
            supply AS (
                SELECT
                    CAST(time AS DATE) AS Date,
                    SUM(CAST(addresses_by_btc_sumbtc AS DOUBLE)) AS supply_total
                FROM v2_address_statistics_addresses_by_btc_sumbtc
                GROUP BY 1
            )
            SELECT
                h.Date,
                h.hashrate,
                s.supply_total
            FROM hashrate h
            JOIN supply s
              ON h.Date = s.Date
            ORDER BY h.Date;
        """).fetchdf()

    df["Date"] = pd.to_datetime(df["Date"])
    return df

def get_asic_generations() -> pd.DataFrame:
    data = {
        "date": [
            "2013-01-01",
            "2015-01-01",
            "2016-07-01",
            "2018-07-01",
            "2020-05-01",
            "2023-01-01",
            "2025-01-01",
        ],
        "j_th": [
            500.0,
            300.0,
            90.0,
            60.0,
            35.0,
            20.0,
            15.0,
        ],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def estimate_joules_per_th_step(dates, asic_points: pd.DataFrame) -> np.ndarray:
    """
    Returns stepwise J/TH aligned to `dates`.
    Uses backward asof (latest ASIC point at or before date),
    then bfill for dates earlier than first ASIC point.
    """
    dates = pd.to_datetime(pd.Series(dates, name="date")).dt.normalize()
    dates = dates.astype("datetime64[ns]")

    df_dates = pd.DataFrame({"date": dates}).sort_values("date")

    ap = asic_points.copy()
    ap["date"] = pd.to_datetime(ap["date"]).dt.normalize().astype("datetime64[ns]")
    ap = ap.sort_values("date")

    merged = pd.merge_asof(
        df_dates,
        ap,
        on="date",
        direction="backward"
    )

    # IMPORTANT: never fill with 0; fill early history with earliest known ASIC point
    merged["j_th"] = merged["j_th"].bfill()

    return merged["j_th"].to_numpy(dtype=float)


def smooth_joules_per_th(dates, j_th_step: np.ndarray, window_days: int = 365) -> pd.Series:
    """
    Rolling mean smoothing to mimic gradual fleet turnover.
    """
    dates = pd.to_datetime(pd.Series(dates, name="date")).dt.normalize()
    dates = dates.astype("datetime64[ns]")

    df = pd.DataFrame({"date": dates, "j_th_step": j_th_step}).sort_values("date")

    j = (
        df["j_th_step"]
        .rolling(window=window_days, min_periods=1)
        .mean()
        .clip(lower=10.0, upper=800.0)
    )

    return pd.Series(j.to_numpy(dtype=float), index=df["date"].to_numpy(), name="j_th_smooth")


def electricity_price_estimate(year: int) -> float:
    t = year - 2014
    return 0.00457 * t**2 + 0.1097 * t + 6.47  # cents/kWh (looks like it)


def electricity_price_estimate_vec(years):
    years = np.asarray(years, dtype=float)
    t = years - 2014.0
    price_cents = 0.00457 * t**2 + 0.1097 * t + 6.47
    # Convert cents/kWh -> USD/kWh
    return price_cents #/ 100.0


def build_energy_value_series(
    price_db_path: str = "database/ohlcv.duckdb",
    onchain_db_path: str = "database/onchain.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    end_timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    # price history timeline
    price_df = load_price_history(price_db_path, symbol=symbol, interval=interval)
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()
    price_df = price_df.sort_values("Date")
    if end_timestamp is not None:
        cutoff = pd.Timestamp(end_timestamp)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        price_df = price_df.loc[price_df["Date"] <= cutoff]
        if price_df.empty:
            raise ValueError(f"No price rows are available on or before {cutoff}.")

    # on-chain metrics
    oc_df = load_onchain_core(onchain_db_path)  # must return Date, hashrate, supply_total
    oc_df["Date"] = pd.to_datetime(oc_df["Date"]).dt.normalize()
    oc_df = oc_df.sort_values("Date").set_index("Date")

    # align to price timeline; carry forward latest values
    oc_df = oc_df.reindex(price_df["Date"]).ffill()

    # if you have leading NaNs (price earlier than onchain coverage), keep them NaN
    # (ffill won't fill the very beginning). That's fine.

    # hardware J/TH series aligned to full timeline
    asic_points = get_asic_generations()
    j_th_step = estimate_joules_per_th_step(oc_df.index, asic_points)
    j_th_smooth = smooth_joules_per_th(oc_df.index, j_th_step, window_days=365)

    # align j_th_smooth to oc_df index (just in case)
    j_th_smooth = j_th_smooth.reindex(oc_df.index)

    # J/hash
    JH = j_th_smooth.to_numpy(dtype=float) / 1e12

    # hashrate is already stored in H/s in the DB; do not up-scale further
    H = oc_df["hashrate"].to_numpy(dtype=float)

    # supply in BTC
    S = oc_df["supply_total"].to_numpy(dtype=float)
    S = np.where(S <= 0, np.nan, S)

    # electricity price USD/kWh
    years = oc_df.index.to_series().dt.year.to_numpy() # type: ignore
    P = electricity_price_estimate_vec(years)

    # if on-chain isn’t available for early history, oc_df will have NaNs there.
    # propagate NaNs through EV cleanly:
    H = np.where(np.isfinite(H), H, np.nan)
    JH = np.where(np.isfinite(JH), JH, np.nan)

    daily_kwh = H * JH * 24.0 #/ 1000.0
    EV = (daily_kwh * P) / S

    merged = price_df[["Date"]].copy()
    merged["energy_value_price"] = EV
    return merged


def ta_ema(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()

def save_energy():
    ev = build_energy_value_series()
    print(ev)

    energy_value = (
        ev[["Date", "energy_value_price"]]
        .rename(columns={"energy_value_price": "energy_value"})
    )
    energy_value["energy_value"] = ta_ema(energy_value["energy_value"], 12)

    energy_value["Date"] = pd.to_datetime(energy_value["Date"])
    energy_value = energy_value.sort_values("Date").set_index("Date")
    energy_value.index.name = "Date"
    #print(energy_value)
    # store with explicit Date index in DuckDB
    rows = save_dataframe_to_table(
        energy_value,
        "value_oc_indicatorE",
        db_path="orvian.duckdb",
        preserve_index=True,
    )
    print("Rows stored:", rows)

#save_energy()
