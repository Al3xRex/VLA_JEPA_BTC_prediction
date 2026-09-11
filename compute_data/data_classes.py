from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from compute_data.macro.fred import construct_eco_data, load_btc_close, load_manual_macro_data
from compute_data.macro.liquidity_fair_value import store_liquidity_fair_value
from compute_data.oc.energy_value import build_energy_value_series, ta_ema as energy_ema
from compute_data.oc.metcalfe_series import build_metcalfe_series
from compute_data.oc.onchain import build_onchain_signal_components
from compute_data.ta import algorithms as ta_algorithms
from compute_data.ta.choppiness import (
    build_choppiness_features,
    build_strategy_influence_features,
    neutral_choppiness_features,
)
from compute_data.ta.pca_latent import (
    PCALatentSystem,
    load_edge_components,
    load_movement_components,
    load_standardized_categories_for_symbol,
    pca_stage3,
)
from database_interraction import (
    drop_table,
    duckdb_connection,
    load_table_dataframe,
    save_dataframe_to_table,
    table_exists,
)
from dual_model_forecaster.liquidation_architecture import build_liquidation_feature_frame

DEFAULT_DATA_CLASSES_DB = "database/data_classes.duckdb"
DEFAULT_TA_DB = "database/ta_1d.duckdb"
DEFAULT_TMP_LIQUIDITY_TABLE = "__tmp_liquidity_fair_value"
DEFAULT_LIQUIDATION_SIGNAL_PATH = "liquidations/data/directional_volatility_potential.csv"
DEFAULT_LIQUIDATION_DAILY_PATH = "liquidations/data/liquidations_daily.csv"
DEFAULT_LIQUIDATION_HEATMAP_DAILY_PATH = "liquidations/Liquedation heatmap/data/daily/btc_oi_liquidations_daily.csv"
DEFAULT_CHOPPINESS_TABLE = "checkonchain_choppiness_index"


def _snake_case(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        return "column"
    if normalized[0].isdigit():
        return f"series_{normalized}"
    return normalized


def _with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(columns=lambda column: f"{prefix}{_snake_case(str(column))}")


def _normalize_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    min_value = clean.min()
    max_value = clean.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=clean.index, name=series.name)
    return ((clean - min_value) / (max_value - min_value) * 2.0 - 1.0).rename(series.name)


def _normalize_cutoff_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _align_to_index(df: pd.DataFrame, index: pd.Index, *, forward_fill: bool = True) -> pd.DataFrame:
    aligned = df.copy()
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index().reindex(index)
    if forward_fill:
        aligned = aligned.ffill()
    return aligned


def _load_table(db_path: str, table_name: str) -> pd.DataFrame:
    df = load_table_dataframe(
        db_path,
        table_name,
        read_only=True,
    )
    first_column = df.columns[0]
    df[first_column] = pd.to_datetime(df[first_column], errors="coerce")
    df = df.dropna(subset=[first_column]).sort_values(first_column).set_index(first_column)
    df.index.name = "timestamp"
    return df


def _load_choppiness_features(
    db_path: str,
    index: pd.Index,
    *,
    table_name: str = DEFAULT_CHOPPINESS_TABLE,
) -> pd.DataFrame:
    try:
        with duckdb_connection(db_path, read_only=True) as connection:
            if not table_exists(connection, table_name):
                return neutral_choppiness_features(index)
        raw = _load_table(db_path, table_name)
    except Exception:
        return neutral_choppiness_features(index)
    if raw.empty:
        return neutral_choppiness_features(index)
    chop = build_choppiness_features(raw)
    neutral = neutral_choppiness_features(index)
    return _align_to_index(chop, index, forward_fill=True).combine_first(neutral)


def _numeric_feature(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(default)


def _load_optional_daily_feature_source(
    path: str,
    *,
    prefix: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    try:
        source = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    if "date" not in source.columns:
        return pd.DataFrame()
    keep = ["date", *[column for column in columns if column in source.columns]]
    source = source.loc[:, keep].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    source = source.sort_values("date").set_index("date")
    source.index.name = "timestamp"
    rename = {column: f"{prefix}{column}" for column in source.columns}
    return source.rename(columns=rename)

@dataclass
class StoredSetResult:
    table_name: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


@dataclass
class BaseStoredSet:
    symbol: str = "BTCUSDT"
    db_path: str = DEFAULT_DATA_CLASSES_DB
    price_db_path: str = "database/ohlcv.duckdb"
    onchain_db_path: str = "database/onchain.duckdb"
    macro_db_path: str = "database/macro.duckdb"
    ta_db_path: str = DEFAULT_TA_DB
    interval: str = "1d"
    table_suffix: str = "1d"
    end_timestamp: pd.Timestamp | str | None = None

    table_name: str = ""

    def build(self) -> pd.DataFrame:
        raise NotImplementedError

    def _base_price_close(self) -> pd.Series:
        price = load_btc_close(
            db_path=self.price_db_path,
            symbol=self.symbol,
            interval=self.interval,
        )
        close = pd.to_numeric(price, errors="coerce").astype(float).sort_index()
        close.index = pd.DatetimeIndex(close.index, name="timestamp")
        cutoff = _normalize_cutoff_timestamp(self.end_timestamp)
        if cutoff is not None:
            close = close.loc[close.index <= cutoff]
        return close

    def _base_price_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._base_price_close().index, name="timestamp")

    def store(self) -> StoredSetResult:
        frame = self.build().sort_index()
        frame.index.name = "timestamp"
        rows = save_dataframe_to_table(
            frame,
            self.table_name,
            db_path=self.db_path,
            preserve_index=True,
        )
        start = pd.Timestamp(frame.index.min()) if not frame.empty else None
        end = pd.Timestamp(frame.index.max()) if not frame.empty else None
        return StoredSetResult(
            table_name=self.table_name,
            rows=rows,
            start=start,
            end=end,
        )


@dataclass
class StructureSet(BaseStoredSet):
    table_name: str = "structure_set"

    def build(self) -> pd.DataFrame:
        base_index = self._base_price_index()
        bundles = build_onchain_signal_components(
            db_path=self.onchain_db_path,
            align_to_price=True,
            price_db_path=self.price_db_path,
            symbol=self.symbol,
            interval=self.interval,
            end_timestamp=self.end_timestamp,
        )

        lth_summary = _align_to_index(bundles["lth_summary"], base_index, forward_fill=True)
        sth_summary = _align_to_index(bundles["sth_summary"], base_index, forward_fill=True)
        lth_components = _align_to_index(
            _with_prefix(bundles["lth_components"], "lth_"),
            base_index,
            forward_fill=True,
        )
        sth_components = _align_to_index(
            _with_prefix(bundles["sth_components"], "sth_"),
            base_index,
            forward_fill=True,
        )

        summary = pd.DataFrame(index=base_index)
        summary["lth_average"] = lth_summary["average"]
        summary["lth_signalboost_raw"] = lth_summary["signalboost"]
        summary["lth_signalboost"] = lth_summary["signalboost"] / 3.0
        summary["sth_average"] = sth_summary["average"]
        summary["sth_signalboost_raw"] = sth_summary["signalboost"]
        summary["sth_signalboost"] = sth_summary["signalboost"] / 4.0

        energy = build_energy_value_series(
            price_db_path=self.price_db_path,
            onchain_db_path=self.onchain_db_path,
            symbol=self.symbol,
            interval=self.interval,
            end_timestamp=self.end_timestamp,
        )
        energy["Date"] = pd.to_datetime(energy["Date"], errors="coerce")
        energy = energy.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        energy.index.name = "timestamp"
        energy = energy.rename(columns={"energy_value_price": "energy_value_raw"})
        energy["energy_value"] = energy_ema(energy["energy_value_raw"], 12)
        energy = _align_to_index(energy[["energy_value_raw", "energy_value"]], base_index, forward_fill=True)

        metcalfe = build_metcalfe_series(
            price_db_path=self.price_db_path,
            onchain_db_path=self.onchain_db_path,
            symbol=self.symbol,
            interval=self.interval,
            end_timestamp=self.end_timestamp,
        )
        metcalfe["Date"] = pd.to_datetime(metcalfe["Date"], errors="coerce")
        metcalfe = metcalfe.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        metcalfe.index.name = "timestamp"
        metcalfe = _align_to_index(
            metcalfe[["metcalfe_price", "all_addresses_count", "avg_btc_per_address"]],
            base_index,
            forward_fill=True,
        ).rename(
            columns={
                "all_addresses_count": "metcalfe_all_addresses_count",
                "avg_btc_per_address": "metcalfe_avg_btc_per_address",
            }
        )

        structure = pd.concat([summary, lth_components, sth_components, energy, metcalfe], axis=1)
        structure.index.name = "timestamp"
        return structure


@dataclass
class EnvironmentSet(BaseStoredSet):
    table_name: str = "environment_set"

    def build(self) -> pd.DataFrame:
        base_index = self._base_price_index()
        end_date = pd.Timestamp(base_index.max()) if len(base_index) else None

        liq_m2_dxy, macro_rest = construct_eco_data(
            end_date=end_date,
            fred_db_path=self.macro_db_path,
            fred_table="macro_series",
        )
        manual = load_manual_macro_data(
            db_path=self.macro_db_path,
            table_name="macro_series",
        )

        liq_m2_dxy = _align_to_index(liq_m2_dxy, base_index, forward_fill=True)
        macro_rest = _align_to_index(macro_rest, base_index, forward_fill=True)
        manual = _align_to_index(manual, base_index, forward_fill=True).dropna(axis=1, how="all")

        environment = pd.DataFrame(index=base_index)
        environment["fed_liquidity"] = liq_m2_dxy.get("liq_bil_daily")
        environment["m2"] = manual.get("M2 Supply of Four Major Central Banks (USD, L)")
        environment["m2_yoy_usd"] = manual.get("M2 Supply of Four Major Central Banks (USD, YoY, R)")
        environment["m2_yoy_fixed_fx"] = manual.get(
            "M2 Supply of Four Major Central Banks (Fixed Exchange Rate, YoY, R)"
        )
        environment["m2_shift_fwd_6w"] = liq_m2_dxy.get("m2_shift_fwd_6w")
        environment["dxy_inverse"] = liq_m2_dxy.get("inv_dxy")
        environment["stocks"] = macro_rest.get("spx")
        environment["gold"] = macro_rest.get("gold")
        environment["business_activity"] = macro_rest.get("Buisness_activity")
        environment["oil"] = macro_rest.get("oil")
        environment["emerging_markets"] = macro_rest.get("emerging_markets")

        manual_prefixed = _with_prefix(manual, "manual_")
        manual_prefixed = manual_prefixed.drop(
            columns=["manual_nonfinancial_leverage", "manual_risk"],
            errors="ignore",
        )
        environment = pd.concat([environment, manual_prefixed], axis=1)

        tmp_table = DEFAULT_TMP_LIQUIDITY_TABLE
        drop_table(self.db_path, tmp_table)
        store_liquidity_fair_value(
            orvian_db_path=self.db_path,
            table_name=tmp_table,
            fred_db_path=self.macro_db_path,
            fred_table="macro_series",
            price_db_path=self.price_db_path,
            symbol=self.symbol,
            interval=self.interval,
            end_date=end_date,
        )
        liquidity = _load_table(self.db_path, tmp_table)
        drop_table(self.db_path, tmp_table)
        liquidity = _align_to_index(liquidity, base_index, forward_fill=True).rename(
            columns={
                "fair_value": "liquidity_fair_value",
                "dev_osc_z": "liquidity_dev_osc_z",
                "band_plus_1": "liquidity_band_plus_1",
                "band_minus_1": "liquidity_band_minus_1",
                "band_plus_2": "liquidity_band_plus_2",
                "band_minus_2": "liquidity_band_minus_2",
            }
        )
        liquidity = liquidity.drop(
            columns=[
                "liquidity_dev_osc_z",
                "liquidity_band_plus_1",
                "liquidity_band_minus_1",
                "liquidity_band_plus_2",
                "liquidity_band_minus_2",
            ],
            errors="ignore",
        )
        environment = pd.concat([environment, liquidity], axis=1)
        environment.index.name = "timestamp"
        return environment


@dataclass
class EdgesSet(BaseStoredSet):
    table_name: str = "edges_set"
    n_components: int = 2
    use_component: int = 0

    def build(self) -> pd.DataFrame:
        base_index = self._base_price_index()
        end_timestamp = pd.Timestamp(base_index.max()) if len(base_index) else None
        mr_df, volatility_df = load_edge_components(
            self.ta_db_path,
            self.symbol,
            table_suffix=self.table_suffix,
        )
        if end_timestamp is not None:
            mr_df = mr_df.loc[mr_df.index <= end_timestamp]
            volatility_df = volatility_df.loc[volatility_df.index <= end_timestamp]
        combined = pd.concat([mr_df, volatility_df], axis=1).dropna(how="all")
        combined_latent = pca_stage3(
            combined,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=True,
            plot=False,
        ).rename("edges_pca")
        system = PCALatentSystem(
            ta_db_path=self.ta_db_path,
            ohlcv_db_path=self.price_db_path,
            interval=self.interval,
            table_suffix=self.table_suffix,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=True,
            end_timestamp=end_timestamp,
        )
        latent = system.run(self.symbol)

        edges = pd.DataFrame(index=combined.index)
        edges["edges_pca"] = combined_latent.reindex(edges.index)
        edges["mr_pca"] = latent["mr"].reindex(edges.index)
        edges["volatility_pca"] = latent["volatility"].reindex(edges.index)
        edges = pd.concat(
            [
                edges,
                _with_prefix(mr_df, "mr_"),
                _with_prefix(volatility_df, "vol_"),
            ],
            axis=1,
        )
        edges.index.name = "timestamp"
        return edges


@dataclass
class MovementSet(BaseStoredSet):
    table_name: str = "movement_set"
    n_components: int = 2
    use_component: int = 0

    def build(self) -> pd.DataFrame:
        base_index = self._base_price_index()
        end_timestamp = pd.Timestamp(base_index.max()) if len(base_index) else None
        trend_df, momentum_df, mreg_df = load_movement_components(
            self.ta_db_path,
            self.symbol,
            table_suffix=self.table_suffix,
        )
        trend_std, mreg_std, _, momentum_std, _ = load_standardized_categories_for_symbol(
            self.ta_db_path,
            self.symbol,
            table_suffix=self.table_suffix,
        )
        if end_timestamp is not None:
            trend_df = trend_df.loc[trend_df.index <= end_timestamp]
            momentum_df = momentum_df.loc[momentum_df.index <= end_timestamp]
            mreg_df = mreg_df.loc[mreg_df.index <= end_timestamp]
            trend_std = trend_std.loc[trend_std.index <= end_timestamp]
            mreg_std = mreg_std.loc[mreg_std.index <= end_timestamp]
            momentum_std = momentum_std.loc[momentum_std.index <= end_timestamp]
        combined_std = pd.concat([trend_std, momentum_std, mreg_std], axis=1).dropna(how="all")
        combined_latent = pca_stage3(
            combined_std,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=False,
            plot=False,
        ).rename("movement_pca")
        system = PCALatentSystem(
            ta_db_path=self.ta_db_path,
            ohlcv_db_path=self.price_db_path,
            interval=self.interval,
            table_suffix=self.table_suffix,
            n_components=self.n_components,
            use_component=self.use_component,
            standardize_output=False,
            end_timestamp=end_timestamp,
        )
        latent = system.run(self.symbol)

        movement = pd.DataFrame(index=combined_std.index)
        movement["movement_pca"] = _normalize_series(combined_latent).reindex(movement.index)
        movement["trend_pca"] = _normalize_series(latent["trend"]).reindex(movement.index)
        movement["momentum_pca"] = _normalize_series(latent["momentum"]).reindex(movement.index)
        movement["mreg_pca"] = _normalize_series(latent["mreg"]).reindex(movement.index)
        chop = _load_choppiness_features(self.onchain_db_path, movement.index)
        strategy = build_strategy_influence_features(
            close=self._base_price_close(),
            trend_signal=movement["trend_pca"],
            momentum_signal=movement["momentum_pca"],
            mreg_signal=movement["mreg_pca"],
            choppiness=chop,
            horizons=(1, 3),
        )
        movement = pd.concat(
            [
                movement,
                chop,
                strategy,
                _with_prefix(trend_df.apply(ta_algorithms.normalize), "trend_"),
                _with_prefix(momentum_df.apply(ta_algorithms.normalize), "momentum_"),
                _with_prefix(mreg_df.apply(ta_algorithms.normalize), "mreg_"),
            ],
            axis=1,
        )
        movement.index.name = "timestamp"
        return movement


@dataclass
class LiquidationSet(BaseStoredSet):
    table_name: str = "liquidation_set"
    signal_path: str = DEFAULT_LIQUIDATION_SIGNAL_PATH
    daily_path: str = DEFAULT_LIQUIDATION_DAILY_PATH
    heatmap_daily_path: str = DEFAULT_LIQUIDATION_HEATMAP_DAILY_PATH

    def build(self) -> pd.DataFrame:
        base_index = self._base_price_index()
        source = pd.read_csv(self.signal_path)
        if "date" not in source.columns:
            raise ValueError(f"Liquidation signal file must contain a date column: {self.signal_path}")
        source["date"] = pd.to_datetime(source["date"], errors="coerce")
        source = source.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        source = source.sort_values("date").set_index("date")
        source.index.name = "timestamp"
        cutoff = _normalize_cutoff_timestamp(self.end_timestamp)
        if cutoff is not None:
            source = source.loc[source.index <= cutoff]
        source_daily = _load_optional_daily_feature_source(
            self.daily_path,
            prefix="source_",
            columns=(
                "aggregated_funding_rate_avg_pct",
                "aggregated_predicted_funding_rate_avg_pct",
                "futures_oi_7day_change_pct",
                "liquidation_dominance_pct",
                "funding_contract_count",
                "predicted_funding_contract_count",
            ),
        )
        heatmap_daily = _load_optional_daily_feature_source(
            self.heatmap_daily_path,
            prefix="heatmap_daily_",
            columns=(
                "oi_binance_usd",
                "oi_bybit_usd",
                "oi_kraken_usd",
                "oi_okx_usd",
                "open_interest_usd",
                "oi_venue_count",
                "estimated_long_liquidated_usd",
                "estimated_short_liquidated_usd",
                "observed_liquidated_long_usd",
                "observed_liquidated_short_usd",
                "observed_long_event_count",
                "observed_short_event_count",
                "observed_liquidation_dominance_pct",
                "estimated_liquidation_dominance_pct",
                "liquidated_interest_long_usd",
                "liquidated_interest_short_usd",
                "liquidation_dominance_pct",
                "liquidation_source",
                "liquidation_dominance_14d_pct",
            ),
        )
        source = pd.concat(
            [
                source,
                source_daily.reindex(source.index),
                heatmap_daily.reindex(source.index),
            ],
            axis=1,
        )
        features = build_liquidation_feature_frame(source)
        features = _align_to_index(features, base_index, forward_fill=True).fillna(0.0)
        features.index.name = "timestamp"
        return features.loc[:, ~features.columns.duplicated()].replace([float("inf"), float("-inf")], 0.0)
