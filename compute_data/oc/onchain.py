import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from database_interraction import duckdb_connection, save_dataframe_to_table

# --- 1. Configuration & Paths ---
DB_PATH = "database/onchain.duckdb"
# Construct
# Sell_side_risk = (Realized profit + realised loss)/realised cap

lth_paths = [
    "/v2/realizedprofit/realized_profit",
    "/v2/realizedloss/realized_loss",
    "/v2/realizedcap/realized_cap",
    "/v2/realizedprofit/realized_profit_lth",
    "/v2/realizedloss/realized_loss_lth",
    "/v2/realizedcap/realized_cap_lth",


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
]

sth_paths = [
    "/v2/realizedprofit/realized_profit_sth",
    "/v2/realizedloss/realized_loss_sth",
    "/v2/realizedcap/realized_cap_sth",

    "/v2/market_value_to_realized_value/mvrv_sth",
    "/v2/net_unrealized_profit_loss/net_unrealized_profit_loss_sth",
    "/v2/spent_output_profit_ratio/sopr_sth",
    "/v2/supply_in_profitloss/utxo_n_in_profit_sth_percent",
    "/v2/supply_in_profitloss/supply_in_profit_sth_percent",
    "/v2/unrealizedcap/unrealized_cap_sth_relative",
]
# Construct
#/v2/realizedprofit/sell_side_risk_sth = (/v2/realizedprofit/realized_profit_sth + /v2/realizedloss/realized_loss_sth) / /v2/realizedcap/realized_cap_sth


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting helpers.")

def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    dt = dt.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    return dt

def ta_ema(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_onchain_data_duckdb(
    db_path,
    paths_lth,
    paths_sth,
    *,
    align_to_price: bool = False,
    price_db_path: str = "database/ohlcv.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    end_timestamp: pd.Timestamp | str | None = None,
):
    """Connects to DuckDB and merges metrics into LTH and STH dataframes."""
    with duckdb_connection(db_path, read_only=True) as con:
        def fetch_and_merge(paths):
            dataframes = []
            for path in paths:
                table_name = path.strip("/").replace("/", "_")
                try:
                    df = con.execute(f"SELECT * FROM {table_name}").df()
                    time_col = None
                    for candidate in ("time", "date", "Date", "datetime", "timestamp"):
                        if candidate in df.columns:
                            time_col = candidate
                            break
                    if time_col:
                        df[time_col] = _normalize_datetime_series(df[time_col])
                        df = df.sort_values(time_col).set_index(time_col)

                    metric_name = path.split('/')[-1]
                    if len(df.columns) == 1 and df.columns[0] != metric_name:
                        df = df.rename(columns={df.columns[0]: metric_name})
                    dataframes.append(df)
                except Exception as e:
                    print(f"Skipping {table_name}: {e}")

            return pd.concat(dataframes, axis=1) if dataframes else pd.DataFrame()

        df_lth = fetch_and_merge(paths_lth)
        df_sth = fetch_and_merge(paths_sth)

    if align_to_price:
        price_df = load_btc_price(
            db_path=price_db_path,
            symbol=symbol,
            interval=interval,
        )
        if end_timestamp is not None:
            cutoff = pd.Timestamp(end_timestamp)
            if cutoff.tz is not None:
                cutoff = cutoff.tz_convert("UTC").tz_localize(None)
            price_df = price_df.loc[price_df.index <= cutoff]
        if not price_df.empty:
            target_index = pd.to_datetime(price_df.index).normalize()
            if getattr(target_index, "tz", None) is not None:
                target_index = target_index.tz_convert("UTC").tz_localize(None)
            target_index = pd.Index(target_index, name=price_df.index.name or "Date")

            def _align(df: pd.DataFrame) -> pd.DataFrame:
                if df.empty:
                    return df
                idx = df.index
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                out = df.copy()
                out.index = idx
                return out.reindex(target_index).ffill()

            df_lth = _align(df_lth)
            df_sth = _align(df_sth)
    elif end_timestamp is not None:
        cutoff = pd.Timestamp(end_timestamp)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        df_lth = df_lth.loc[df_lth.index <= cutoff]
        df_sth = df_sth.loc[df_sth.index <= cutoff]

    return df_lth, df_sth

def load_btc_price(
    db_path: str = "database/ohlcv.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
):
    """Loads BTC OHLCV history from the shared OHLCV store."""
    with duckdb_connection(db_path, read_only=True) as con:
        try:
            query = """
                SELECT CAST(open_time AS DATE) AS Date, close AS Close
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time;
            """
            df = con.execute(query, [symbol.upper(), interval.lower()]).df()
            if df.empty:
                return df
            df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")
        except Exception as e:
            print(f"Error loading price: {e}")
            return pd.DataFrame()


def standerdise(series, window=None):
    """
    If window=None: normal z-score over whole series.
    If window=int: rolling z-score over that window.
    """
    if window is None:
        return (series - series.mean()) / series.std()

    # Rolling mean and std
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()

    # Avoid division by zero because pandas loves ruining your day
    return (series - roll_mean) / roll_std.replace(0, float('nan'))


def amplitude_equalizer(series,
                        window=365,
                        alpha=0.5,
                        min_scale=0.2,
                        max_scale=5.0):
    """
    alpha controls how strong the equalization is:
      alpha = 0   -> no effect
      alpha = 1   -> full range equalization
    """
    s = pd.Series(series).astype(float)

    roll_min = s.rolling(window, min_periods=window//2).min()
    roll_max = s.rolling(window, min_periods=window//2).max()
    roll_range = (roll_max - roll_min).replace(0, np.nan)

    # global "target" range (e.g. central 90% span)
    global_range = s.quantile(0.95) - s.quantile(0.05)

    # scale factor: how much to stretch/compress local range toward global_range
    scale = (global_range / roll_range).pow(alpha)

    # avoid insane blowups
    scale = scale.clip(lower=min_scale, upper=max_scale)

    # multiply, no re-centering
    hybrid = s * scale

    return hybrid


def causal_amplitude_equalizer(
    series: pd.Series,
    *,
    window: int = 365,
    min_periods: int | None = None,
    alpha: float = 0.5,
    min_scale: float = 0.2,
    max_scale: float = 5.0,
) -> pd.Series:
    """Past-only alternative to :func:`amplitude_equalizer`.

    The legacy function compares a rolling range with a full-history target
    range.  Here both the local range and the expanding target range are known
    at the emitted timestamp, so later observations cannot revise history.
    """

    values = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)
    minimum = int(min_periods or max(int(window) // 2, 20))
    rolling_min = values.rolling(int(window), min_periods=minimum).min()
    rolling_max = values.rolling(int(window), min_periods=minimum).max()
    local_range = (rolling_max - rolling_min).replace(0.0, np.nan)
    expanding_high = values.expanding(min_periods=minimum).quantile(0.95)
    expanding_low = values.expanding(min_periods=minimum).quantile(0.05)
    target_range = (expanding_high - expanding_low).replace(0.0, np.nan)
    scale = (target_range / local_range).pow(float(alpha))
    scale = scale.clip(lower=float(min_scale), upper=float(max_scale)).fillna(1.0)
    return (values * scale).rename(getattr(series, "name", None))


def base_no_rolling(series,
                    smooth_span=1,
                    clip_low=-3.0,
                    clip_high=3.0):
    s = pd.Series(series).astype(float)

    # shift if needed so we can log
    if (s <= 0).any():
        shift = 1 - s.min()
        s = s + shift

    # log
    s = np.log(s)

    # global z-score (one mean/std for entire history)
    s = (s - s.mean()) / s.std()

    # clip extremes
    s = s.clip(clip_low, clip_high)

    # smooth a bit (EMA, no time shift in index)
    s = s.ewm(span=smooth_span, adjust=False).mean()

    return s


def causal_base_transform(
    series: pd.Series,
    *,
    min_periods: int = 90,
    window: int | None = None,
    smooth_span: int = 1,
    clip_low: float = -3.0,
    clip_high: float = 3.0,
) -> pd.Series:
    """Past-only log/standardization alternative to ``base_no_rolling``."""

    values = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)
    # Signed log avoids the legacy full-history minimum used to choose a shift.
    logged = np.sign(values) * np.log1p(values.abs())
    minimum = max(int(min_periods), 2)
    if window is None:
        mean = logged.expanding(min_periods=minimum).mean()
        std = logged.expanding(min_periods=minimum).std(ddof=0)
    else:
        width = int(window)
        if width < minimum:
            raise ValueError("window must be at least min_periods")
        mean = logged.rolling(width, min_periods=minimum).mean()
        std = logged.rolling(width, min_periods=minimum).std(ddof=0)
    standardized = ((logged - mean) / std.replace(0.0, np.nan)).clip(
        float(clip_low),
        float(clip_high),
    )
    return standardized.fillna(0.0).ewm(span=int(smooth_span), adjust=False).mean().rename(
        getattr(series, "name", None)
    )


def _calculate_boost_signal(df):
    """
    Calculates the 'Boost' signal:
    1. Clip inputs to [-1.5, 1.5]
    2. Multiply magnitudes (Product)
    3. Sum signs (Vote)
    """
    # 1. CRITICAL: Clip the input values first!
    # This matches: 'clipped = element.clip(-1.5, 1.5)' from your snippet
    clipped = df.clip(-1.5, 1.5)
    
    # 2. Separate Magnitudes and Signs from the CLIPPED data
    mags = clipped.abs()
    signs = np.sign(clipped)
    valid_count = clipped.notna().sum(axis=1)
    required_count = clipped.shape[1]
    
    # 3. Combined Magnitude: Product across the row (axis=1)
    combined_mag = mags.prod(axis=1, min_count=required_count)
    
    # 4. Combined Sign: Sign of the sum of signs across the row
    combined_sign = np.sign(signs.sum(axis=1, min_count=required_count))
    
    return (combined_mag * combined_sign).where(valid_count >= required_count)


def calculate_boost_signal(df):
    """
    Calculates the 'Boost' signal:
    1. Clip inputs to [-1.5, 1.5]
    2. Multiply magnitudes (Product)
    3. Sum signs (Vote)
    """
    # 1. CRITICAL: Clip the input values first!
    # This matches: 'clipped = element.clip(-1.5, 1.5)' from your snippet
    clipped = df.clip(-1.8, 1.8)
    
    # 2. Separate Magnitudes and Signs from the CLIPPED data
    mags = clipped.abs()
    signs = np.sign(clipped)
    valid_count = clipped.notna().sum(axis=1)
    required_count = clipped.shape[1]
    
    # 3. Combined Magnitude: Product across the row (axis=1)
    combined_mag = mags.prod(axis=1, min_count=required_count)
    
    # 4. Combined Sign: Sign of the sum of signs across the row
    combined_sign = np.sign(signs.sum(axis=1, min_count=required_count))
    
    return (combined_mag * combined_sign).where(valid_count >= required_count)

def create_dual_axis_plot(df_metrics, df_price, title, metric_cols):
    _require_matplotlib()
    # Align data (Inner join ensures dates match)
    df_combined = df_price.join(df_metrics, how='inner')
    
    if df_combined.empty:
        print(f"No data for plot: {title}")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    plt.title(title, fontsize=14)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)

    # Plot Price (Left Axis)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('BTC Price (USD)', color='black')
    ax1.plot(df_combined.index, df_combined['Close'], color='black', label='BTC Price', alpha=0.5, linewidth=1)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_yscale('log')

    # Plot Metrics (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Signal Strength', color='tab:blue')
    
    colors = ['tab:orange', 'tab:blue']
    for i, col in enumerate(metric_cols):
        if col in df_combined.columns:
            ax2.plot(df_combined.index, df_combined[col], label=col, color=colors[i % 2], linewidth=1.5)
            # Add a zero line for reference
            ax2.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()


def create_price_over_indicator_plot(df_metrics, df_price, title, metric_cols):
    _require_matplotlib()
    # Align data (Inner join ensures dates match)
    df_metrics = df_metrics.copy()
    df_price = df_price.copy()
    df_metrics.index = pd.to_datetime(df_metrics.index).normalize()
    df_price.index = pd.to_datetime(df_price.index).normalize()
    if getattr(df_metrics.index, "tz", None) is not None:
        df_metrics.index = df_metrics.index.tz_convert("UTC").tz_localize(None)
    if getattr(df_price.index, "tz", None) is not None:
        df_price.index = df_price.index.tz_convert("UTC").tz_localize(None)
    df_combined = df_price.join(df_metrics, how="inner")

    if df_combined.empty:
        print(f"No data for plot: {title}")
        return

    fig, (ax_price, ax_ind) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(title, fontsize=14)

    # Price panel (top)
    ax_price.plot(df_combined.index, df_combined["Close"], color="black", label="BTC Price", linewidth=1)
    ax_price.set_ylabel("BTC Price (USD)")
    ax_price.grid(True, which="major", linestyle="--", alpha=0.5)
    if (df_combined["Close"] > 0).all():
        ax_price.set_yscale("log")
    ax_price.legend(loc="upper left")

    # Indicator panel (bottom)
    colors = ["tab:orange", "tab:blue", "tab:green"]
    for i, col in enumerate(metric_cols):
        if col in df_combined.columns:
            ax_ind.plot(
                df_combined.index,
                df_combined[col],
                label=col,
                color=colors[i % len(colors)],
                linewidth=1.5,
            )
    ax_ind.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax_ind.set_ylabel("Signal Strength")
    ax_ind.grid(True, which="major", linestyle="--", alpha=0.5)
    ax_ind.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def _add_sell_side_risk(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    profit_col = f"realized_profit{suffix}"
    loss_col = f"realized_loss{suffix}"
    cap_col = f"realized_cap{suffix}"
    required = [profit_col, loss_col, cap_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns for sell_side_risk{suffix}: {missing}")
        return df
    df[f"sell_side_risk{suffix}"] = (df[profit_col] + df[loss_col]) / df[cap_col]
    return df


def add_sell_side_risk_sth(df: pd.DataFrame) -> pd.DataFrame:
    return _add_sell_side_risk(df, "_sth")


def add_sell_side_risk_lth(df: pd.DataFrame) -> pd.DataFrame:
    return _add_sell_side_risk(df, "_lth")


def add_sell_side_risk(df: pd.DataFrame) -> pd.DataFrame:
    return _add_sell_side_risk(df, "")


def store_value_system(
    average: pd.Series,
    boost: pd.Series,
    index: pd.Index,
    table_name: str,
    boost_scale: float,
) -> pd.DataFrame:
    data = pd.concat([average, boost / boost_scale], axis=1).reindex(index)
    data.columns = ["Average", "Boost"]
    data.index = index
    data.index.name = "Date"
    rows = save_dataframe_to_table(
        data,
        table_name,
        db_path="orvian.duckdb",
        preserve_index=True,
    )
    print(f"Stored {table_name}: {data.columns}")
    return data


def build_onchain_signal_components(
    *,
    db_path: str = DB_PATH,
    align_to_price: bool = True,
    price_db_path: str = "database/ohlcv.duckdb",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    end_timestamp: pd.Timestamp | str | None = None,
) -> dict[str, pd.DataFrame]:
    df_lth, df_sth = load_onchain_data_duckdb(
        db_path,
        lth_paths,
        sth_paths,
        align_to_price=align_to_price,
        price_db_path=price_db_path,
        symbol=symbol,
        interval=interval,
        end_timestamp=end_timestamp,
    )
    df_lth = add_sell_side_risk(add_sell_side_risk_lth(df_lth))
    df_sth = add_sell_side_risk_sth(df_sth)

    sth = pd.DataFrame(index=df_sth.index)
    sth["mvrv"] = amplitude_equalizer(base_no_rolling(pd.Series(df_sth["mvrv_sth"])))
    sth["sopr"] = amplitude_equalizer(base_no_rolling(ta_ema(pd.Series(df_sth["sopr_sth"]), 12)))
    sth["utxipp"] = amplitude_equalizer(
        base_no_rolling(pd.Series(df_sth["utxo_n_in_profit_sth_percent"]))
    ) * 1.3 + 0.4
    sth["sipp"] = amplitude_equalizer(
        base_no_rolling(pd.Series(df_sth["supply_in_profit_sth_percent"]))
    ) * 1.3 + 0.4
    sth["ucr"] = amplitude_equalizer(
        base_no_rolling(pd.Series(df_sth["unrealized_cap_sth_relative"]))
    ) + 0.4
    sth["upl"] = amplitude_equalizer(
        base_no_rolling(pd.Series(df_sth["net_unrealized_profit_loss_sth"]))
    ) + 0.2
    sth["ssr"] = amplitude_equalizer(
        base_no_rolling(ta_ema(pd.Series(df_sth["sell_side_risk_sth"]), 10))
    )

    lth = pd.DataFrame(index=df_lth.index)
    lth["aviv"] = amplitude_equalizer(
        base_no_rolling(pd.Series(df_lth["active_value_to_investor_value"]))
    )
    lth["amvrv"] = amplitude_equalizer(base_no_rolling(pd.Series(df_lth["active_mvrv"])))
    lth["mvrv"] = amplitude_equalizer(base_no_rolling(pd.Series(df_lth["mvrv"])))
    lth["sopr"] = amplitude_equalizer(base_no_rolling(ta_ema(pd.Series(df_lth["sopr"]), 12)))

    lth_summary = pd.DataFrame(index=lth.index)
    lth_summary["average"] = lth.mean(axis=1)
    lth_summary["signalboost"] = calculate_boost_signal(lth)

    sth_summary = pd.DataFrame(index=sth.index)
    sth_summary["average"] = sth.mean(axis=1)
    sth_summary["signalboost"] = _calculate_boost_signal(sth)

    return {
        "lth_raw": df_lth,
        "sth_raw": df_sth,
        "lth_components": lth,
        "sth_components": sth,
        "lth_summary": lth_summary,
        "sth_summary": sth_summary,
    }


def save_onchain_value_sys():
    bundles = build_onchain_signal_components()

    # Value System (OC)
    LTHs = store_value_system(
        bundles["lth_summary"]["average"],
        bundles["lth_summary"]["signalboost"],
        bundles["lth_summary"].index,
        "lth_value_sys_oc",
        3,
    )

    # Reversion System (OC)
    STHs = store_value_system(
        bundles["sth_summary"]["average"],
        bundles["sth_summary"]["signalboost"],
        bundles["sth_summary"].index,
        "sth_value_sys_oc",
        4,
    )
    return {"lth": LTHs, "sth": STHs}
    
    '''
    df_btc = load_btc_price()

    lth_data = pd.concat([df_average_lth[["lth"]], df_boost_lth[["lth"]]/3], axis=1)
    lth_data.columns = ["Average", "Boost"]
    create_price_over_indicator_plot(
        lth_data,
        df_btc,
        "Long-Term Holder TOP (LTH) Signals",
        ["Average", "Boost"],
    )

    sth_data = pd.concat([df_average_sth[["sth"]], df_boost_sth[["sth"]]/4], axis=1)
    sth_data.columns = ["Average", "Boost"]
    create_price_over_indicator_plot(
        sth_data,
        df_btc,
        "Short-Term Holder TOP (STH) Signals",
        ["Average", "Boost"],
    )
    '''
