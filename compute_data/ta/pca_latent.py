import duckdb
import pandas as pd
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass
from database_interraction import duckdb_connection

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

MR_CLIP_BOUNDS: dict[str, tuple[float, float] | None] = {
    "MR1": (-1.6, 1.6),
    "MR2": None,
    "MR3": (-2.2, 2.2),
    "MR4": (-2.0, 2.0),
    "MR5": (-2.0, 2.0),
    "MR6": (-2.0, 2.0),
}
DEFAULT_MR_CLIP_BOUNDS = (-2.0, 2.0)
MR_BOUND = 5.0


def causal_robust_standardize(
    series: pd.Series,
    *,
    min_periods: int = 20,
    window: int | None = None,
) -> pd.Series:
    """Causally robust-standardize a series using present and past rows only.

    ``window=None`` uses an expanding history.  A finite window provides an
    adaptive rolling alternative.  Appending future rows cannot revise any
    previously emitted value.
    """

    clean = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = max(int(min_periods), 2)
    if window is None:
        median = clean.expanding(min_periods=minimum).median()
        deviation = (clean - median).abs()
        mad = deviation.expanding(min_periods=minimum).median()
        fallback = clean.expanding(min_periods=minimum).std(ddof=0)
    else:
        width = int(window)
        if width < minimum:
            raise ValueError("window must be at least min_periods")
        median = clean.rolling(width, min_periods=minimum).median()
        deviation = (clean - median).abs()
        mad = deviation.rolling(width, min_periods=minimum).median()
        fallback = clean.rolling(width, min_periods=minimum).std(ddof=0)
    scale = 1.4826 * mad
    scale = scale.where(np.isfinite(scale) & (scale > 1e-8), fallback)
    standardized = (clean - median) / scale.where(scale > 1e-8)
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename(series.name)


@dataclass(frozen=True)
class FoldLocalRobustPCA:
    """A train-fitted robust PCA transform safe for validation/test use.

    Fit this object on an outer/inner training frame, persist it with the model,
    and call :meth:`transform` on later rows.  Unlike :func:`pca_stage3`, no
    statistic or loading is recomputed from the transform frame.
    """

    columns: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    components: np.ndarray
    score_mean: np.ndarray
    score_scale: np.ndarray
    standardize_output: bool
    fitted_rows: int

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        *,
        n_components: int = 1,
        standardize_output: bool = True,
    ) -> "FoldLocalRobustPCA":
        numeric = train_frame.apply(pd.to_numeric, errors="coerce").astype(float)
        if numeric.shape[0] < 2 or numeric.shape[1] < 2:
            raise ValueError("train_frame must contain at least two rows and two features")
        components_requested = int(n_components)
        maximum = min(numeric.shape)
        if components_requested < 1 or components_requested > maximum:
            raise ValueError(f"n_components must be in [1, {maximum}]")

        values = numeric.to_numpy(dtype=float)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        absolute_deviation = np.abs(values - median)
        mad = np.nanmedian(absolute_deviation, axis=0) * 1.4826
        fallback = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        filled = np.where(np.isfinite(values), values, median)
        standardized = (filled - median) / scale
        center = standardized.mean(axis=0)
        centered = standardized - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = np.asarray(vt[:components_requested], dtype=float)

        # Resolve the arbitrary SVD sign deterministically so serialized models
        # and repeated fits on identical training data have stable coordinates.
        for component_index in range(len(components)):
            pivot = int(np.argmax(np.abs(components[component_index])))
            if components[component_index, pivot] < 0.0:
                components[component_index] *= -1.0

        train_scores = centered @ components.T
        score_mean = train_scores.mean(axis=0)
        score_scale = train_scores.std(axis=0, ddof=0)
        score_scale = np.where(
            np.isfinite(score_scale) & (score_scale > 1e-8),
            score_scale,
            1.0,
        )
        return cls(
            columns=tuple(str(column) for column in numeric.columns),
            median=median,
            scale=scale,
            center=center,
            components=components,
            score_mean=score_mean,
            score_scale=score_scale,
            standardize_output=bool(standardize_output),
            fitted_rows=int(len(numeric)),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing fold-local PCA columns: {missing[:8]}")
        numeric = frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        filled = np.where(np.isfinite(values), values, self.median)
        standardized = (filled - self.median) / self.scale
        scores = (standardized - self.center) @ self.components.T
        if self.standardize_output:
            scores = (scores - self.score_mean) / self.score_scale
        columns = [f"pc{index + 1}" for index in range(scores.shape[1])]
        return pd.DataFrame(scores, index=frame.index, columns=columns)

    def transform_component(self, frame: pd.DataFrame, component: int = 0) -> pd.Series:
        transformed = self.transform(frame)
        component_index = int(component)
        if component_index < 0 or component_index >= transformed.shape[1]:
            raise ValueError(
                f"component must be in [0, {transformed.shape[1] - 1}]"
            )
        return transformed.iloc[:, component_index].rename("latent_signal")


def _single_thread_context(enabled: bool):
    if enabled and threadpool_limits is not None:
        return threadpool_limits(limits=1)
    return nullcontext()

def mad_standardize(x: pd.Series) -> pd.Series:
    median = x.median()
    mad = (x - median).abs().median()
    scale = mad * 1.4826  # consistency constant

    if scale == 0:
        return pd.Series(0.0, index=x.index)

    return (x - median) / scale

def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'

def load_ta_table(
    db_path: str,
    table_name: str,
) -> pd.DataFrame:
    table_ident = _quote_identifier(table_name)
    with duckdb_connection(db_path, read_only=True) as con:
        df = con.execute(
            f"SELECT * FROM {table_ident} ORDER BY timestamp"
        ).fetchdf()

    if "timestamp" not in df.columns:
        raise ValueError(f"{table_name} missing timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    df.index.name = "timestamp"
    return df

def _split_ta_categories(
    ta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mr_cols = [c for c in ta_df.columns if c.startswith("MR")]
    trend_cols = [c for c in ta_df.columns if c.startswith("Trend")]
    mreg_cols = [c for c in ta_df.columns if c.startswith("MReg")]
    momentum_cols = [c for c in ta_df.columns if c.startswith("mom")]
    volatility_cols = [c for c in ta_df.columns if c.startswith("vol")]
    if not mr_cols:
        raise ValueError("No MR columns found.")
    if not trend_cols:
        raise ValueError("No Trend columns found.")
    if not mreg_cols:
        raise ValueError("No MReg columns found.")
    if not momentum_cols:
        raise ValueError("No momentum columns found.")
    if not volatility_cols:
        raise ValueError("No volatility columns found.")
    return (
        ta_df[mr_cols],
        ta_df[trend_cols],
        ta_df[mreg_cols],
        ta_df[momentum_cols],
        ta_df[volatility_cols],
    )

def _mad_standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = pd.to_numeric(out[col], errors="coerce")
        out[col] = mad_standardize(series)
    return out.fillna(0)


def _initial_static_pair_drop_count(series: pd.Series) -> int:
    values = series.to_numpy()
    drop_count = 0
    while drop_count + 1 < len(values) and values[drop_count] == values[drop_count + 1]:
        drop_count += 2
    return drop_count


def trim_initial_static_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return df
    drop_count = 0
    for col in df.columns:
        drop_count = max(drop_count, _initial_static_pair_drop_count(df[col]))
    if drop_count == 0:
        return df
    return df.iloc[drop_count:].copy()


def _clip_and_bound(
    series: pd.Series,
    *,
    clip_bounds: tuple[float, float] | None,
    bound: float,
) -> pd.Series:
    out = series
    if clip_bounds is not None:
        out = out.clip(lower=clip_bounds[0], upper=clip_bounds[1])
    return out.clip(lower=-bound, upper=bound)


def _stabilize_mr_df(df: pd.DataFrame) -> pd.DataFrame:
    trimmed = trim_initial_static_pairs(df)
    out = trimmed.copy()
    for col in out.columns:
        raw = pd.to_numeric(out[col], errors="coerce")
        robust = mad_standardize(raw)
        clip_bounds = MR_CLIP_BOUNDS.get(col, DEFAULT_MR_CLIP_BOUNDS)
        out[col] = _clip_and_bound(robust, clip_bounds=clip_bounds, bound=MR_BOUND)
    return out.fillna(0)

def _load_close_series(
    db_path: str,
    table_name: str,
    symbol: str,
    *,
    interval: str = "1d",
) -> pd.Series:
    table_ident = _quote_identifier(table_name)
    with duckdb_connection(db_path, read_only=True) as con:
        df = con.execute(
            f"""
            SELECT open_time AS timestamp, close
            FROM {table_ident}
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time
            """,
            [symbol, interval],
        ).fetchdf()

    if df.empty:
        empty_index = pd.DatetimeIndex([], name="timestamp")
        return pd.Series(dtype=float, index=empty_index, name="close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    out = df["close"].astype(float)
    out.index.name = "timestamp"
    return out

def _aligned_log_returns(series: pd.Series, index: pd.Index) -> pd.Series:
    if series.empty:
        empty_index = pd.DatetimeIndex([], name=getattr(index, "name", None) or "timestamp")
        return pd.Series(dtype=float, index=empty_index)
    aligned = series.reindex(index).dropna()
    if len(aligned) < 3:
        return aligned.iloc[:0].astype(float)
    return np.log(aligned).diff().dropna()


def _normalize_cutoff_timestamp(timestamp: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if timestamp is None:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _filter_to_end_timestamp(
    frame: pd.DataFrame | pd.Series,
    end_timestamp: pd.Timestamp | str | None,
) -> pd.DataFrame | pd.Series:
    cutoff = _normalize_cutoff_timestamp(end_timestamp)
    if cutoff is None:
        return frame
    return frame.loc[frame.index <= cutoff]

def _maybe_flip_series(series: pd.Series, returns: pd.Series) -> pd.Series:
    if series.empty or returns.empty:
        return series
    aligned_series = series.reindex(returns.index).dropna()
    aligned_returns = returns.reindex(aligned_series.index).dropna()
    if len(aligned_returns) < 10:
        return series
    corr = aligned_returns.corr(aligned_series)
    if pd.isna(corr):
        return series
    if corr < 0:
        return series * -1.0
    return series


def pca_stage3(
    Z: pd.DataFrame,
    n_components: int = 1,
    use_component: int = 0,
    standardize_output: bool = True,
    plot: bool = True,
    single_thread: bool = True,
):
    """
    Stage 3 only: PCA denoising on already-standardized indicators.

    Z: pd.DataFrame (rows=time, cols=indicators), already standardized/robust-zscored.
    Returns:
      latent_signal: pd.Series
      explained_ratio: np.ndarray (per PC)
    """
    if not isinstance(Z, pd.DataFrame):
        Z = pd.DataFrame(Z)

    X = Z.dropna().astype(float)
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 indicator columns for PCA.")

    if use_component < 0 or use_component >= n_components:
        raise ValueError("use_component must be in [0, n_components-1].")

    centered = X.values - X.values.mean(axis=0, keepdims=True)
    max_components = min(centered.shape[0], centered.shape[1])
    if n_components > max_components:
        raise ValueError(f"n_components must be <= {max_components}.")

    with _single_thread_context(single_thread):
        u, s, vt = np.linalg.svd(centered, full_matrices=False)

    scores = u[:, :n_components] * s[:n_components]
    explained = s**2
    explained_ratio = explained / explained.sum()
    latent_signal = pd.Series(scores[:, use_component], index=X.index, name="latent_signal")

    if standardize_output:
        s = latent_signal.std()
        latent_signal = (latent_signal - latent_signal.mean()) / (s if s != 0 else 1.0)

    return latent_signal


def _load_standardized_categories(
    ta_db_path: str,
    table_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ta_df = load_ta_table(ta_db_path, table_name)
    mr_df, trend_df, mreg_df, momentum_df, volatility_df = _split_ta_categories(ta_df)
    trend_df = _mad_standardize_df(trend_df)
    mreg_df = _mad_standardize_df(mreg_df)
    mr_df = _stabilize_mr_df(mr_df)
    momentum_df = _mad_standardize_df(momentum_df)
    volatility_df = _mad_standardize_df(volatility_df)
    return trend_df, mreg_df, mr_df, momentum_df, volatility_df

def load_symbol_ta_tables(
    ta_db_path: str,
    symbol: str,
    *,
    table_suffix: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_u = symbol.upper()
    mr_table = f"{symbol_u}_{table_suffix}_MR"
    trend_table = f"{symbol_u}_{table_suffix}_Trend"
    mreg_table = f"{symbol_u}_{table_suffix}_MReg"
    momentum_table = f"{symbol_u}_{table_suffix}_Momentum"
    volatility_table = f"{symbol_u}_{table_suffix}_Volatility"
    try:
        mr_df = load_ta_table(ta_db_path, mr_table)
        trend_df = load_ta_table(ta_db_path, trend_table)
        mreg_df = load_ta_table(ta_db_path, mreg_table)
        momentum_df = load_ta_table(ta_db_path, momentum_table)
        volatility_df = load_ta_table(ta_db_path, volatility_table)
        return trend_df, mreg_df, mr_df, momentum_df, volatility_df
    except duckdb.Error:
        ta_table = f"{symbol_u}_{table_suffix}_ta"
        ta_df = load_ta_table(ta_db_path, ta_table)
        mr_df, trend_df, mreg_df, momentum_df, volatility_df = _split_ta_categories(ta_df)
        return trend_df, mreg_df, mr_df, momentum_df, volatility_df


def load_edge_components(
    ta_db_path: str,
    symbol: str,
    *,
    table_suffix: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, mr_df, _, volatility_df = load_symbol_ta_tables(
        ta_db_path,
        symbol,
        table_suffix=table_suffix,
    )
    return _stabilize_mr_df(mr_df), _mad_standardize_df(volatility_df)


def load_movement_components(
    ta_db_path: str,
    symbol: str,
    *,
    table_suffix: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trend_df, mreg_df, _, momentum_df, _ = load_symbol_ta_tables(
        ta_db_path,
        symbol,
        table_suffix=table_suffix,
    )
    return trend_df, momentum_df, mreg_df


def load_standardized_categories_for_symbol(
    ta_db_path: str,
    symbol: str,
    *,
    table_suffix: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trend_df, mreg_df, mr_df, momentum_df, volatility_df = load_symbol_ta_tables(
        ta_db_path,
        symbol,
        table_suffix=table_suffix,
    )
    trend_df = _mad_standardize_df(trend_df)
    mreg_df = _mad_standardize_df(mreg_df)
    mr_df = _stabilize_mr_df(mr_df)
    momentum_df = _mad_standardize_df(momentum_df)
    volatility_df = _mad_standardize_df(volatility_df)
    return trend_df, mreg_df, mr_df, momentum_df, volatility_df

def ta_ema(series: pd.Series, length: int) -> pd.Series:
    alpha = 2 / (length + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


# Modular --> Apply to BTC, ETH, SOL
class PCALatentSystem:
    def __init__(
        self,
        *,
        ta_db_path: str = "database/1d_ta.duckdb",
        ohlcv_db_path: str = "database/ohlcv.duckdb",
        ohlcv_table: str = "ohlcv",
        interval: str = "1d",
        table_suffix: str = "1d",
        n_components: int = 2,
        use_component: int = 0,
        standardize_output: bool = True,
        flip_with_returns: bool = True,
        single_thread: bool = True,
        end_timestamp: pd.Timestamp | str | None = None,
    ) -> None:
        self.ta_db_path = ta_db_path
        self.ohlcv_db_path = ohlcv_db_path
        self.ohlcv_table = ohlcv_table
        self.interval = interval
        self.table_suffix = table_suffix
        self.n_components = n_components
        self.use_component = use_component
        self.standardize_output = standardize_output
        self.flip_with_returns = flip_with_returns
        self.single_thread = single_thread
        self.end_timestamp = end_timestamp

    def _load_standardized(
        self, symbol: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trend_df, mreg_df, mr_df, momentum_df, volatility_df = load_symbol_ta_tables(
            self.ta_db_path,
            symbol,
            table_suffix=self.table_suffix,
        )
        trend_df = _mad_standardize_df(trend_df)
        mreg_df = _mad_standardize_df(mreg_df)
        mr_df = _stabilize_mr_df(mr_df)
        momentum_df = _mad_standardize_df(momentum_df)
        volatility_df = _mad_standardize_df(volatility_df)
        trend_df = _filter_to_end_timestamp(trend_df, self.end_timestamp)
        mreg_df = _filter_to_end_timestamp(mreg_df, self.end_timestamp)
        mr_df = _filter_to_end_timestamp(mr_df, self.end_timestamp)
        momentum_df = _filter_to_end_timestamp(momentum_df, self.end_timestamp)
        volatility_df = _filter_to_end_timestamp(volatility_df, self.end_timestamp)
        return trend_df, mreg_df, mr_df, momentum_df, volatility_df

    def run(self, symbol: str) -> dict[str, pd.Series]:
        symbol_u = symbol.upper()
        trend_df, mreg_df, mr_df, momentum_df, volatility_df = self._load_standardized(symbol_u)
        returns = _aligned_log_returns(
            _filter_to_end_timestamp(
                _load_close_series(
                    self.ohlcv_db_path,
                    self.ohlcv_table,
                    symbol_u,
                    interval=self.interval,
                ),
                self.end_timestamp,
            ),
            trend_df.index,
        )

        trend_latent = pca_stage3(
            trend_df,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=self.standardize_output,
            plot=False,
            single_thread=self.single_thread,
        )
        mreg_latent = pca_stage3(
            mreg_df,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=self.standardize_output,
            plot=False,
            single_thread=self.single_thread,
        )
        mr_latent = pca_stage3(
            mr_df,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=self.standardize_output,
            plot=False,
            single_thread=self.single_thread,
        )
        momentum_latent = pca_stage3(
            momentum_df,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=self.standardize_output,
            plot=False,
            single_thread=self.single_thread,
        )
        volatility_latent = pca_stage3(
            volatility_df,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=self.standardize_output,
            plot=False,
            single_thread=self.single_thread,
        )

        if self.flip_with_returns:
            trend_latent = _maybe_flip_series(trend_latent, returns)
            mr_latent = _maybe_flip_series(mr_latent, returns)

        return {
            "trend": trend_latent,
            "mreg": ta_ema(mreg_latent, 5),
            "mr": mr_latent,
            "momentum": momentum_latent,
            "volatility": volatility_latent,
        }


def main_series() -> dict[str, dict[str, pd.Series]]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    system = PCALatentSystem()
    return {symbol: system.run(symbol) for symbol in symbols}
