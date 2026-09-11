from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss
from warnings import catch_warnings, simplefilter
from arch.unitroot import PhillipsPerron
from arch.unitroot import ZivotAndrews


# TA functions
def ta_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ta_stdev(series: pd.Series, length: int, ddof: int = 0) -> pd.Series:
    return series.rolling(length, min_periods=length).std(ddof=ddof)

def ta_ema(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()

def ta_dev(series: pd.Series, length: int) -> pd.Series:
    sma = ta_sma(series, length)
    abs_dev = (series - sma).abs()
    return abs_dev.rolling(length, min_periods=length).mean()

def ta_rma(series: pd.Series, length: int):
    if length <= 0:
        raise ValueError("length must be positive")
    alpha = 1.0 / length
    if len(series) < length:
        return pd.Series(np.nan, index=series.index, dtype=float)
    rma = pd.Series(np.nan, index=series.index, dtype=float)
    rma.iloc[length-1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        rma.iloc[i] = rma.iloc[i-1] + alpha * (series.iloc[i] - rma.iloc[i-1])
    return rma

def ta_lowest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).min()

def ta_highest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).max()

def ta_stoch(source: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    lowest = ta_lowest(low, length)
    highest = ta_highest(high, length)
    denom = highest - lowest
    denom = denom.replace(0, np.nan)
    stoch = 100 * (source - lowest) / denom
    return stoch

def ta_change(series, length=1):
    shifted = series.shift(length)
    if series.dtype == bool:
        return series.ne(shifted)
    out = series - shifted
    return out  # pandas handles NaN propagation fine

def standerdise(df):
    return (df - df.mean()) / df.std()

def normalize(series):
    minv = series.min()
    maxv = series.max()
    if maxv == minv:
        return pd.Series(0, index=series.index)  # collapsed range
    norm01 = (series - minv) / (maxv - minv)
    return norm01 * 2 - 1

def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"df must contain: {missing_text}")


# ------------------------
# Mean Reversion
# ------------------------

def Full_VZScore(df: pd.DataFrame, length: int = 20, lenf: int = 5, ddof: int = 0):
    Pric = (df['high'] + df['low']) / 2.0
    a1 = (Pric - Pric.shift(lenf)).abs()
    num1   = (a1 * Pric).rolling(lenf, min_periods=lenf).sum()
    denom1 = a1.rolling(lenf, min_periods=lenf).sum().replace(0, np.nan)
    filt1  = num1 / denom1
    mu  = ta_sma(filt1, length)
    sd  = ta_stdev(filt1, length, ddof=ddof).replace(0, np.nan)
    z1  = (filt1 - mu) / sd
    sz = ta_sma(z1, 3)
    return sz

#Helper function 
def h_zone(src: pd.Series, typ: pd.Series, length: int):
    vp = pd.Series(0.0, index=src.index)
    up_mask = src > src.shift(1)
    vp[up_mask] = typ[up_mask]
    down_mask = src < src.shift(1)
    vp[down_mask] = -typ[down_mask]
    z = 100 * (ta_ema(vp, length) / ta_ema(typ, length))
    return z

def VolumeZonePriceOscillator(df: pd.DataFrame, zlen: int = 21):
    src = df['close']
    vol = df['volume']
    vzo = h_zone(src, vol, zlen)
    pzo = h_zone(src, src, zlen)
    return vzo, pzo

def UCS_MurreyMath(df: pd.DataFrame, length: int = 100):
    high = df['high'].rolling(window=length, min_periods=1).max()
    low = df['low'].rolling(window=length, min_periods=1).min()
    range = high - low
    range = range.replace(0, np.nan)
    multiplier = range * 0.125
    midline = low + multiplier * 4
    oscillator = (df['close'] - midline) / (range / 2)
    return oscillator

def CommodityChannelIndex(df, length=20):
    src = (df['high'] + df['low'] + df['close']) / 3.0
    ma = ta_sma(src, length)
    cci = (src - ma) / (0.015 * ta_dev(src, length))
    return  cci

def RelativeStrengthIndex(df, length=14):
    src = df['close']
    chg = ta_change(src)
    up_raw = chg.clip(lower=0)
    down_raw = (-chg).clip(lower=0)
    up = ta_rma(up_raw, length)
    down = ta_rma(down_raw, length)
    rsi = 100 - 100 / (1 + up / down)
    rsi[down == 0] = 100
    rsi[up == 0] = 0
    return rsi


# ------------------------
# Trend
# ------------------------

def OptimalLeverage(df: pd.DataFrame, length: int = 55, smaLangth: int = 1000):
    src = df['close']
    hi = df['high'].rolling(window=length, min_periods=1).max()
    dd = src / hi * 100 - 100
    avg = ta_sma(dd, smaLangth)
    omisig = dd - avg
    return omisig

def DetrendPrice(df: pd.DataFrame, length: int = 19):
    barsBack = int(length / 2) + 1
    hl2 = (df['high'] + df['low']) / 2.0
    xSMA = ta_sma(hl2, length)
    nRes = hl2 - xSMA.shift(barsBack)
    sma = ta_sma(nRes, length)
    return sma

def SchaffTrendCycle(df: pd.DataFrame, fastLength: int = 23, slowLength: int = 50, cycleLength: int = 10, d1Length: int = 3, d2Length: int = 3):
    src1 = df['close']
    macd = ta_ema(src1, fastLength) - ta_ema(src1, slowLength)
    k = ta_stoch(macd, macd, macd, cycleLength)
    d = ta_ema(k, d1Length)
    kd = ta_stoch(d, d, d, cycleLength)
    stc = ta_ema(kd, d2Length)
    stc = stc.clip(lower=0, upper=100) - 50
    return stc

def fn_calculate_wma_with_coefficient(source: pd.Series, length: int, coefficient: float) -> pd.Series:
    raw_weights = np.arange(length, -1, -1)
    adj_weights = raw_weights - coefficient
    adj_weights = np.where(adj_weights < 0, 0, adj_weights)
    windows = source.rolling(length + 1, min_periods=length + 1).apply(
        lambda x: np.sum(x[::-1] * adj_weights) / np.sum(adj_weights),
        raw=True)
    return windows

def GunzoTrendSniper(df: pd.DataFrame, ma_length: int = 14, smoothing_length: int = 3):
    ma_source = df['close']
    coefficient = ma_length / 3.0
    weighted_line = fn_calculate_wma_with_coefficient(ma_source, ma_length, coefficient)
    weighted_line_smooth = ta_ema(weighted_line, smoothing_length)
    trend_up = weighted_line_smooth - weighted_line_smooth.shift(1)
    return trend_up

def ta_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int
) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    tp_prev = tp.shift(1)
    pos_mf = mf.where(tp > tp_prev, 0.0)
    neg_mf = mf.where(tp < tp_prev, 0.0)
    pos_sum = pos_mf.rolling(length, min_periods=length).sum()
    neg_sum = neg_mf.rolling(length, min_periods=length).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - 100 / (1 + ratio)
    return mfi


def RMITrendSniper(df: pd.DataFrame, length6: int = 14, pmom: int = 66, nmom: int = 30):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    # === RMI-ish RSI part ===
    chg = ta_change(close)                         # ta.change(close)
    up6 = ta_rma(np.maximum(chg, 0), length6)      # ta.rma(max(change, 0), Length6)
    down = ta_rma(np.maximum(-chg, 0), length6)    # ta.rma(-min(change, 0), Length6)
    rsi = pd.Series(index=df.index, dtype=float)
    rsi[down == 0] = 100
    rsi[(down != 0) & (up6 == 0)] = 0
    mask = (down != 0) & (up6 != 0)
    rsi[mask] = 100 - 100 / (1 + up6[mask] / down[mask])
    # === MFI + blend ===
    mf = ta_mfi(high, low, close, vol, length6)
    rsi_mfi = (rsi + mf) / 2.0    # math.avg(rsi, mf)
    # Continuous trend component instead of ternary state persistence.
    rsi_mfi_centered = (rsi_mfi - 50.0) / 50.0
    band = max(float(pmom - nmom), 1.0)
    momentum_bias = ((rsi_mfi - 50.0) / (band / 2.0)).clip(lower=-2.0, upper=2.0) / 2.0
    ema_fast = ta_ema(close, 5)
    ema_slow = ta_ema(close, 21)
    ema_scale = close.rolling(length6, min_periods=length6).std().replace(0, np.nan)
    ema_bias = ((ema_fast - ema_slow) / ema_scale).clip(lower=-3.0, upper=3.0) / 3.0
    trend_component = 0.6 * rsi_mfi_centered + 0.25 * momentum_bias + 0.15 * ema_bias
    return ta_ema(trend_component, 3)

    
def highPass(price: pd.Series, period: float) -> pd.Series:
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1 + c2 - c3) / 4
    # output series
    hp = np.zeros(len(price))
    for i in range(len(price)):
        if i >= 2:
            p0 = price.iloc[i]
            p1 = price.iloc[i - 1]
            p2 = price.iloc[i - 2]

            hp_1 = hp[i - 1] if i >= 1 else 0.0
            hp_2 = hp[i - 2] if i >= 2 else 0.0
            hp[i] = (
                c1 * (p0 - 2 * p1 + p2)
                + c2 * hp_1
                + c3 * hp_2)
        else:
            hp[i] = 0.0
    return pd.Series(hp, index=price.index)

def TASC2024_09PrecisionTrendAnalysis(df: pd.DataFrame, length1: int = 250, length2: int = 40):
    HP1 = highPass(df['close'], length1)
    HP2 = highPass(df['close'], length2)
    Trend = HP1 - HP2
    return Trend

def DynamicEMA(source: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return source.astype(float)
    fast_end = 2.0 / (2.0 + 1.0)
    slow_end = 2.0 / (30.0 + 1.0)
    change = (source - source.shift(length)).abs()
    volatility = (source - source.shift(1)).abs().rolling(
        length, min_periods=length
    ).sum()
    # avoid division problems
    efficiency_ratio = change / volatility.replace(0, np.nan)
    efficiency_ratio = efficiency_ratio.clip(lower=0, upper=1).fillna(0)
    smooth_factor = (efficiency_ratio * (fast_end - slow_end) + slow_end) ** 2
    base = ta_rma(source, length)
    dyn = base + smooth_factor * (source - base)
    return dyn

def NoiseReducer(source: pd.Series, length: int):

    ema = ta_ema(source, length)
    smooth = 2.0 / (length + 1.0) if length > 0 else 1.0
    return ema + smooth * (source - ema)

def dynamic_volume_rsi_series(source: pd.Series,  volume: pd.Series,  length: int, vol_smooth_len: int,):
    smoothed_volume = DynamicEMA(volume, vol_smooth_len)
    price_change = ta_change(source)
    up_raw = price_change.clip(lower=0) * smoothed_volume
    down_raw = (-price_change).clip(lower=0) * smoothed_volume
    up = ta_rma(up_raw, length)
    down = ta_rma(down_raw, length)
    index = pd.Series(index=source.index, dtype=float)
    index[down == 0] = 100
    index[(down != 0) & (up == 0)] = 0
    mask = (down != 0) & (up != 0)
    index[mask] = 100 - 100 / (1 + up[mask] / down[mask])
    index = NoiseReducer(index, length)
    return index

def DynamicVolumeRSI(df: pd.DataFrame, rsi_length: int = 14, smoothing_length5: int = 14):
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    dvrsi = dynamic_volume_rsi_series(
        source=hlc3,
        volume=df["volume"],
        length=rsi_length,
        vol_smooth_len=smoothing_length5,)
    return dvrsi - 50.0


# ------------------------
# Momentum
# ------------------------

def ta_wma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.nan if np.any(np.isnan(x)) else float(np.dot(x, weights) / weight_sum),
        raw=True,
    )

def ta_hma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    half = max(length // 2, 1)
    sqrt_len = max(int(round(np.sqrt(length))), 1)
    return ta_wma(2.0 * ta_wma(series, half) - ta_wma(series, length), sqrt_len)

def ta_true_range(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("high", "low", "close"))
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def ta_ema_strict(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def ta_linreg_last(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 1:
        return pd.Series(np.nan, index=series.index, dtype=float)

    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    def _calc(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        return float(intercept + slope * (length - 1))

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)

def sqzmom_hma_momentum(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ("open", "high", "low", "close"))
    source = df["open"].astype(float)
    length_kc = 20

    hh = ta_highest(df["high"].astype(float), length_kc)
    ll = ta_lowest(df["low"].astype(float), length_kc)
    mid_hl = (hh + ll) / 2.0
    hma_close = ta_hma(df["close"].astype(float), length_kc)
    center = (mid_hl + hma_close) / 2.0

    val = ta_linreg_last(source - center, length_kc)
    return pd.DataFrame({"val": val}, index=df.index)

def intraday_momentum_index(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if length < 1:
        raise ValueError("length must be >= 1")
    _require_columns(df, ("open", "close"))

    open_price = df["open"].astype(float)
    close_price = df["close"].astype(float)

    gain = (close_price - open_price).where(close_price > open_price, 0.0)
    loss = (open_price - close_price).where(close_price <= open_price, 0.0)

    up_sum = gain.rolling(length, min_periods=length).sum()
    down_sum = loss.rolling(length, min_periods=length).sum()
    denom = up_sum + down_sum

    imi = (100.0 * up_sum / denom).where(denom != 0, np.nan)
    return imi.rename("imi")

def ta_dema(series: pd.Series, length: int) -> pd.Series:
    ema_1 = ta_ema_strict(series, length)
    ema_2 = ta_ema_strict(ema_1, length)
    return 2.0 * ema_1 - ema_2

def _cmo_like(src: pd.Series, window: int) -> pd.Series:
    prev = src.shift(1)
    up_move = (src - prev).where(src > prev, 0.0)
    down_move = (prev - src).where(src < prev, 0.0)

    up = up_move.rolling(window, min_periods=window).sum()
    down = down_move.rolling(window, min_periods=window).sum()
    denom = up + down

    raw = (up - down) / denom
    return (100.0 * raw.replace([np.inf, -np.inf], np.nan).fillna(0.0))

def ccmi_lazybear(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ("close",))
    src = df["close"].astype(float)

    cmo5 = ta_dema(_cmo_like(src, 5), 3)
    cmo10 = ta_dema(_cmo_like(src, 10), 3)
    cmo20 = ta_dema(_cmo_like(src, 20), 3)

    sd5 = ta_stdev(src, 5, ddof=0)
    sd10 = ta_stdev(src, 10, ddof=0)
    sd20 = ta_stdev(src, 20, ddof=0)

    numerator = (sd5 * cmo5) + (sd10 * cmo10) + (sd20 * cmo20)
    denominator = sd5 + sd10 + sd20
    dmi = (numerator / denominator).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({"dmi": dmi}, index=df.index)

def stoch_momentum_index_slow(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("high", "low", "close"))
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    ll = ta_lowest(low, 10)
    hh = ta_highest(high, 10)

    diff = hh - ll
    rdiff = close - (hh + ll) / 2.0
    avgrel = ta_ema_strict(rdiff, 3)
    avgdiff = ta_ema_strict(diff, 3)

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2.0)) * 100.0, 0.0)
    smi_smoothed = ta_sma(pd.Series(smi, index=df.index), 5)
    return ta_ema_strict(smi_smoothed, 10).rename("smi_slow")

def ehlers_sami_lazybear(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("high", "low"))
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    src = (high + low) / 2.0

    n = len(src)
    pi = np.pi
    dtr = pi / 180.0

    s = np.full(n, np.nan)
    c_arr = np.full(n, np.nan)
    q1 = np.full(n, np.nan)
    i1 = np.full(n, np.nan)
    dp = np.full(n, np.nan)
    md = np.full(n, np.nan)
    dc = np.full(n, np.nan)
    ip = np.full(n, np.nan)
    p = np.full(n, np.nan)
    f3 = np.full(n, np.nan)

    def nz(x: float, fallback: float = 0.0) -> float:
        return fallback if (x is None or np.isnan(x)) else x

    def med3(x: float, y: float, z: float) -> float:
        return (x + y + z) - min(x, y, z) - max(x, y, z)

    a = 0.07
    co = 8.0
    a1 = np.exp(-pi / co)
    b1 = 2.0 * a1 * np.cos((1.738 * 180.0 / co) * dtr)
    c1 = a1 * a1
    coef2 = b1 + c1
    coef3 = -(c1 + b1 * c1)
    coef4 = c1 * c1
    coef1 = 1.0 - coef2 - coef3 - coef4

    for i in range(n):
        s_i = (
            nz(src[i], 0.0)
            + 2.0 * nz(src[i - 1] if i - 1 >= 0 else np.nan, 0.0)
            + 2.0 * nz(src[i - 2] if i - 2 >= 0 else np.nan, 0.0)
            + nz(src[i - 3] if i - 3 >= 0 else np.nan, 0.0)
        ) / 6.0
        s[i] = s_i

        term_fallback = (
            nz(src[i], 0.0)
            - 2.0 * nz(src[i - 1] if i - 1 >= 0 else np.nan, 0.0)
            + nz(src[i - 2] if i - 2 >= 0 else np.nan, 0.0)
        ) / 4.0

        c_candidate = (
            (1 - 0.5 * a) * (1 - 0.5 * a) * (
                s_i
                - 2.0 * nz(s[i - 1] if i - 1 >= 0 else np.nan, 0.0)
                + nz(s[i - 2] if i - 2 >= 0 else np.nan, 0.0)
            )
            + 2.0 * (1 - a) * nz(c_arr[i - 1] if i - 1 >= 0 else np.nan, 0.0)
            - (1 - a) * (1 - a) * nz(c_arr[i - 2] if i - 2 >= 0 else np.nan, 0.0)
        )
        c_arr[i] = c_candidate if not np.isnan(c_candidate) else term_fallback

        ip_prev = nz(ip[i - 1] if i - 1 >= 0 else np.nan, 0.0)
        q1_i = (
            (
                0.0962 * nz(c_arr[i], 0.0)
                + 0.5769 * nz(c_arr[i - 2] if i - 2 >= 0 else np.nan, 0.0)
                - 0.5769 * nz(c_arr[i - 4] if i - 4 >= 0 else np.nan, 0.0)
                - 0.0962 * nz(c_arr[i - 6] if i - 6 >= 0 else np.nan, 0.0)
            )
            * (0.5 + 0.08 * ip_prev)
        )
        q1[i] = q1_i

        i1_i = nz(c_arr[i - 3] if i - 3 >= 0 else np.nan, 0.0)
        i1[i] = i1_i

        q1_prev = nz(q1[i - 1] if i - 1 >= 0 else np.nan, 0.0)
        i1_prev = nz(i1[i - 1] if i - 1 >= 0 else np.nan, 0.0)

        if q1_i != 0.0 and q1_prev != 0.0:
            numerator = (i1_i / q1_i) - (i1_prev / q1_prev)
            denominator = 1.0 + (i1_i * i1_prev) / (q1_i * q1_prev)
            dp_raw = numerator / denominator if denominator != 0.0 else 0.0
        else:
            dp_raw = 0.0

        dp_i = 0.1 if dp_raw < 0.1 else (1.1 if dp_raw > 1.1 else dp_raw)
        dp[i] = dp_i

        dp1 = nz(dp[i - 1] if i - 1 >= 0 else np.nan, 0.0)
        dp2 = nz(dp[i - 2] if i - 2 >= 0 else np.nan, 0.0)
        dp3 = nz(dp[i - 3] if i - 3 >= 0 else np.nan, 0.0)
        dp4 = nz(dp[i - 4] if i - 4 >= 0 else np.nan, 0.0)
        md_i = med3(dp_i, dp1, med3(dp2, dp3, dp4))
        md[i] = md_i

        dc_i = 15.0 if md_i == 0.0 else (2.0 * pi / md_i + 0.5)
        dc[i] = dc_i

        ip_i = 0.33 * dc_i + 0.67 * ip_prev
        ip[i] = ip_i

        p_prev = nz(p[i - 1] if i - 1 >= 0 else np.nan, 0.0)
        p_i = 0.15 * ip_i + 0.85 * p_prev
        p[i] = p_i

        pr_i = max(0, min(75, int(round(abs(p_i - 1.0)))))
        v1_i = 0.0 if pr_i == 0 or i - pr_i < 0 else src[i] - src[i - pr_i]

        f3_1 = nz(f3[i - 1] if i - 1 >= 0 else np.nan, 0.0)
        f3_2 = nz(f3[i - 2] if i - 2 >= 0 else np.nan, 0.0)
        f3_3 = nz(f3[i - 3] if i - 3 >= 0 else np.nan, 0.0)
        f3_candidate = coef1 * v1_i + coef2 * f3_1 + coef3 * f3_2 + coef4 * f3_3
        f3[i] = f3_candidate if not np.isnan(f3_candidate) else v1_i

    return pd.Series(f3, index=df.index, name="esam_lb_f3")

def smma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    smaval = ta_sma(series, length)
    out = np.full(len(series), np.nan)
    prev = np.nan

    for i in range(len(series)):
        x = series.iat[i]
        if np.isnan(x):
            continue

        if np.isnan(prev):
            out[i] = smaval.iat[i]
        else:
            out[i] = (prev * (length - 1) + x) / length
        prev = out[i]

    return pd.Series(out, index=series.index)

def ma_mtf_momentum_histogram(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("close",))
    src = df["close"].astype(float)
    fast = smma(src, 4)
    slow = smma(src, 12)
    return (fast - slow).rename("ma_mtf_momentum")


# ------------------------
# Volatility
# ------------------------

def ls_normalized(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("high", "low", "close"))
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    ma = ta_ema_strict(close, 55)

    anchor = high.where(high >= ma, low)
    distance = (anchor - ma).abs() / ma.replace(0, np.nan)
    distance = distance * 100.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    historical_vol = tr.ewm(alpha=1.0 / 89.0, adjust=False, min_periods=89).mean()

    ls_vol = (distance / historical_vol.replace(0, np.nan)) * 100.0

    lo = ta_lowest(ls_vol, 89)
    hi = ta_highest(ls_vol, 89)
    denom = (hi - lo).replace(0, np.nan)

    ls_norm = (ls_vol - lo) * 100.0 / denom
    return ls_norm.rename("lsNormalized")

def ls_volatility(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("close",))
    close = df["close"].astype(float)

    length = 21
    days = 252
    logret = np.log(close / close.shift(1))

    stdev = ta_stdev(logret, length, ddof=0)
    hvi = stdev * np.sqrt(days) * 100.0

    avg_hvi = ta_sma(hvi, length)
    sma = ta_sma(close, length)
    price_dev = (close - sma).abs() / sma.replace(0, np.nan) * 100.0

    ls_vol = (price_dev / avg_hvi.replace(0, np.nan)) * 100.0
    return ls_vol.rename("lsVolatility")

def calc_volatility_this(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ("close",))
    close = df["close"].astype(float)

    logret = np.log(close / close.shift(1))
    vol = 100.0 * ta_stdev(logret, 30, ddof=0)
    return vol.rename("calc_volatility_this")


# ------------------------
# Market Rregime
# ------------------------

def rolling_adf(df: pd.DataFrame, window: int = 100, output: str = 'pvalue') -> pd.Series:
    close = df['close']
    results = []
    index = close.index
    for i in range(len(close)):
        if i < window - 1:
            results.append(np.nan)
            continue
        window_data = close.iloc[i - window + 1:i + 1].dropna()
        if len(window_data) < window:
            results.append(np.nan)
            continue
        try:
            adf_result = adfuller(window_data, autolag='AIC')
            results.append(adf_result[1] if output == 'pvalue' else adf_result[0])
        except Exception:
            results.append(np.nan)
    return pd.Series(results, index=index)

def adx_strength(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    tr_smooth = tr.rolling(length).sum()
    plus_di = 100 * plus_dm.rolling(length).sum() / tr_smooth
    minus_di = 100 * minus_dm.rolling(length).sum() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(length).mean()
    return adx


def calculate_kpss_stat(window_data: pd.Series, regression: str) -> float:
    """Helper function to run KPSS on a rolling window."""
    # Ensure there are enough non-NaN observations for the test to run
    if window_data.empty or len(window_data.dropna()) < 3: # KPSS requires at least 3 points
        return np.nan
        
    try:
        with catch_warnings():
            # The KPSS test often throws a warning about the chosen number of lags
            simplefilter("ignore")
            # We only need the test statistic (the first element of the tuple)
            stat = kpss(window_data.dropna(), regression=regression, nlags="auto")[0] # type: ignore
            return stat
    except Exception:
        # Catch any other errors (e.g., singular matrix due to lack of variation)
        return np.nan

def rolling_kpss(df: pd.DataFrame, window: int = 100, regression: str = 'ct') -> pd.Series:
    """
    Calculates the rolling KPSS test statistic using the pandas.Series.rolling.apply() method.
    """
    close = df['close']
    
    # Use a lambda to pass the required 'regression' argument to the helper function
    kpss_series = close.rolling(window=window, min_periods=window) \
                       .apply(lambda x: calculate_kpss_stat(x, regression), raw=False)
                       
    return kpss_series

def rolling_zivot(df: pd.DataFrame, window: int = 100) -> pd.Series:
    close = df['close']
    out = []
    for i in range(len(close)):
        if i < window:
            out.append(np.nan)
            continue
        x = close.iloc[i - window + 1:i + 1].dropna()
        if len(x) < window:
            out.append(np.nan)
            continue
        try:
            za = ZivotAndrews(x)
            out.append(za.stat)
        except:
            out.append(np.nan)
    return pd.Series(out, index=close.index)

def rolling_pp(df: pd.DataFrame, window: int = 100) -> pd.Series:
    close = df['close']
    out = []
    for i in range(len(close)):
        if i < window:
            out.append(np.nan)
            continue
        x = close.iloc[i - window + 1:i + 1].dropna()
        if len(x) < window:
            out.append(np.nan)
            continue
        try:
            pp = PhillipsPerron(x)
            out.append(pp.stat)
        except:
            out.append(np.nan)
    return pd.Series(out, index=close.index)



def ehlers_daily_regime(
    close: pd.Series,
    snr_threshold: float = 6.0,
    # strength shaping
    snr_scale: float = 6.0,        # dB above threshold that maps to ~1.0 strength
    ema_scale: float = 0.01,       # 1% EMA separation maps to ~1.0 strength
    clip: float = 2.0,             # cap strength to avoid stupid spikes
    return_components: bool = False
):
    # --- your original ehlers_daily_regime code, up to snr_series and EMA trend ---
    smooth = (4 * close + 3 * close.shift(1) + 2 * close.shift(2) + close.shift(3)) / 10

    i1 = pd.Series(0.0, index=close.index)
    q1 = pd.Series(0.0, index=close.index)
    i2 = pd.Series(0.0, index=close.index)
    q2 = pd.Series(0.0, index=close.index)
    re = pd.Series(0.0, index=close.index)
    im = pd.Series(0.0, index=close.index)
    period = pd.Series(0.0, index=close.index)
    smooth_period = pd.Series(0.0, index=close.index)

    smooth = smooth.fillna(0.0)

    def nz_iloc(series, t_idx, n):
        return series.iloc[t_idx - n] if (t_idx - n) >= 0 else 0.0

    for t in range(len(close)):
        if t < 6:
            continue

        i1_current = (
            0.0962 * smooth.iloc[t] +
            0.5769 * smooth.iloc[t-2] -
            0.5769 * smooth.iloc[t-4] -
            0.0962 * smooth.iloc[t-6]
        )
        q1_current = (
            0.0962 * (smooth.iloc[t] - smooth.iloc[t-6]) +
            0.5769 * (smooth.iloc[t-2] - smooth.iloc[t-4])
        )

        jI = (
            0.0962 * i1.iloc[t] +
            0.5769 * nz_iloc(i1, t, 2) -
            0.5769 * nz_iloc(i1, t, 4) -
            0.0962 * nz_iloc(i1, t, 6)
        )
        jQ = (
            0.0962 * q1.iloc[t] +
            0.5769 * nz_iloc(q1, t, 2) -
            0.5769 * nz_iloc(q1, t, 4) -
            0.0962 * nz_iloc(q1, t, 6)
        )

        i1.iloc[t] = i1_current
        q1.iloc[t] = q1_current

        i2_raw = i1.iloc[t] - jQ
        q2_raw = q1.iloc[t] + jI
        i2.iloc[t] = 0.2 * i2_raw + 0.8 * i2.iloc[t-1]
        q2.iloc[t] = 0.2 * q2_raw + 0.8 * q2.iloc[t-1]

        re_raw = i2.iloc[t] * i2.iloc[t-1] + q2.iloc[t] * q2.iloc[t-1]
        im_raw = i2.iloc[t] * q2.iloc[t-1] - q2.iloc[t] * i2.iloc[t-1]
        re.iloc[t] = 0.2 * re_raw + 0.8 * re.iloc[t-1]
        im.iloc[t] = 0.2 * im_raw + 0.8 * im.iloc[t-1]

        current_period = period.iloc[t-1]
        if im.iloc[t] != 0.0 and re.iloc[t] != 0.0:
            if re.iloc[t] != 0:
                current_period = 2 * np.pi / np.arctan(im.iloc[t] / re.iloc[t])

        if current_period > 1.5 * period.iloc[t-1]:
            current_period = 1.5 * period.iloc[t-1]
        if current_period < 0.67 * period.iloc[t-1]:
            current_period = 0.67 * period.iloc[t-1]

        current_period = min(max(current_period, 6), 50)
        period.iloc[t] = current_period
        smooth_period.iloc[t] = 0.2 * period.iloc[t] + 0.8 * smooth_period.iloc[t-1]

    cycle = smooth_period.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    cycle = cycle.round().astype(int)

    snr_series = pd.Series(0.0, index=close.index)
    for t in range(len(close)):
        p = cycle.iloc[t]
        if p <= 0 or t < p:
            continue

        signal_power = (close.iloc[t] - close.iloc[t - p]) ** 2
        sum_noise = 0.0
        for i in range(p):
            if (t - i - 1) >= 0:
                sum_noise += (close.iloc[t - i] - close.iloc[t - i - 1]) ** 2
            else:
                sum_noise = 0.0
                break

        noise_power = (sum_noise / p) if p > 0 else 0.0
        if noise_power > 0:
            ratio = signal_power / noise_power
            snr = 10 * np.log10(ratio) if ratio > 0 else 0.0
        else:
            snr = 0.0

        snr_series.iloc[t] = snr

    ema_fast = ta_ema(close, 10)
    ema_slow = ta_ema(close, 20)
    valid_trend = ema_fast.notna() & ema_slow.notna()

    trend_up = (ema_fast > ema_slow) & (close > ema_slow) & valid_trend
    trend_down = (ema_fast < ema_slow) & (close < ema_slow) & valid_trend
    snr_high = snr_series > snr_threshold

    # --- discrete regime (unchanged semantics) ---
    regime = pd.Series(0, index=close.index, dtype=int)
    regime[snr_high & trend_up] = 1
    regime[snr_high & trend_down] = -1

    # --- continuous strength (Option A vibe) ---
    # 1) SNR margin strength: only positive above threshold
    snr_margin = (snr_series - snr_threshold).clip(lower=0.0)
    snr_strength = (snr_margin / snr_scale).clip(0.0, clip)

    # 2) Trend strength: EMA separation as % of price (scale-free)
    denom = close.replace(0, np.nan).abs()
    ema_sep_pct = ((ema_fast - ema_slow).abs() / denom).fillna(0.0)
    trend_strength = (ema_sep_pct / ema_scale).clip(0.0, clip)

    # 3) Combine and apply sign
    sign = regime.astype(float)  # -1/0/+1
    strength = (sign * snr_strength * trend_strength).rename("ehlers_regime_strength")

    if return_components:
        out = pd.DataFrame({
            "regime": regime,
            "snr": snr_series,
            "snr_strength": snr_strength,
            "ema_sep_pct": ema_sep_pct,
            "trend_strength": trend_strength,
            "strength": strength
        }, index=close.index)
        return out

    return strength
