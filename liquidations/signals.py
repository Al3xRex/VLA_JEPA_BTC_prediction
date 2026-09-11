from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

try:
    from .main import (
        DEFAULT_BTC_PRICE_PATH,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_SIGNAL_OUTPUT_PATH,
        PROJECT_ROOT,
        ScrapeError,
        _format_number,
        write_csv_rows,
    )
except ImportError:
    from main import (
        DEFAULT_BTC_PRICE_PATH,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_SIGNAL_OUTPUT_PATH,
        PROJECT_ROOT,
        ScrapeError,
        _format_number,
        write_csv_rows,
    )

DEFAULT_HEATMAP_DAILY_PATH = (
    PROJECT_ROOT
    / "Liquedation heatmap"
    / "data"
    / "daily"
    / "btc_oi_liquidations_daily.csv"
)
DEFAULT_HEATMAP_PRESSURE_PATH = (
    PROJECT_ROOT
    / "Liquedation heatmap"
    / "outputs"
    / "btc_all_available_history_pressure.csv"
)
DEFAULT_HEATMAP_LEVELS_PATH = (
    PROJECT_ROOT
    / "Liquedation heatmap"
    / "outputs"
    / "btc_all_available_history_levels.csv"
)
DEFAULT_OI_AREA_LEVELS_PATH = (
    PROJECT_ROOT
    / "Liquedation heatmap"
    / "outputs"
    / "btc_oi_interest_areas_levels.csv"
)

LIQUIDATION_SPACE_FIELDS = [
    "heatmap_open_interest_usd",
    "heatmap_oi_venue_count",
    "heatmap_liquidation_dominance_pct",
    "heatmap_liquidated_interest_long_usd",
    "heatmap_liquidated_interest_short_usd",
    "liquidation_pressure_downside_score",
    "liquidation_pressure_upside_score",
    "liquidation_pressure_net_score",
    "liquidation_wall_downside_distance_bps",
    "liquidation_wall_downside_amount_score",
    "liquidation_wall_upside_distance_bps",
    "liquidation_wall_upside_amount_score",
    "oi_area_downside_distance_bps",
    "oi_area_downside_amount_score",
    "oi_area_upside_distance_bps",
    "oi_area_upside_amount_score",
    "cascade_closeness_amount_score",
    "cascade_closeness_amount_direction_score",
    "cascade_threshold_state",
]

PRICE_JUMP_TARGET_FIELDS = [
    "btc_forward_jump_1d_pct",
    "btc_forward_jump_3d_pct",
    "btc_forward_jump_7d_pct",
    "btc_forward_abs_jump_7d_pct",
    "btc_forward_jump_direction",
]

SIGNAL_FIELDS = [
    "date",
    "directional_bias",
    "directional_bias_score",
    "volatility_potential",
    "volatility_potential_score",
    "directional_volatility_score",
    "leverage_fuel_score",
    "new_oi_addition_score",
    "oi_zone_reactivation_score",
    "oi_anchor_direction_score",
    "estimated_oi_anchor_price",
    "volatility_trigger_score",
    "realized_volatility_regime_score",
    "cascade_activation_score",
    "liquidation_impulse_score",
    "funding_crowding_score",
    "signal_confidence_score",
    *LIQUIDATION_SPACE_FIELDS,
    *PRICE_JUMP_TARGET_FIELDS,
    "regime",
]

RAW_FIELDS = {
    "date",
    "aggregated_funding_rate_avg_pct",
    "aggregated_predicted_funding_rate_avg_pct",
    "futures_oi_7day_change_pct",
    "liquidation_dominance_pct",
    "funding_contract_count",
    "predicted_funding_contract_count",
}

HISTORY_WINDOW = 365
MIN_HISTORY = 60
RETURN_WINDOW = 30
MIN_RETURN_HISTORY = 15
OI_DISTRIBUTION_DAYS = 7
OI_ANCHOR_HALF_LIFE_DAYS = 60
OI_DEPLETION_RATE = 0.35
MIN_PRICE_BAND = 0.04
MAX_PRICE_BAND = 0.18
PRICE_BAND_VOL_MULTIPLIER = 2.0
MIN_BASELINE_VOLATILITY = 0.005
DIRECTION_ACTIVATION_THRESHOLD = 12.0
LIQUIDATION_SATURATION_PCT = 20.0
MAX_FUNDING_CONTRACTS = 13
MAX_PREDICTED_CONTRACTS = 12
MIN_CASCADE_BAND_BPS = 150.0
MAX_CASCADE_BAND_BPS = 1200.0
CASCADE_BAND_VOL_MULTIPLIER = 2.0
JEPA_JUMP_DIRECTION_THRESHOLD_PCT = 1.0


@dataclass
class _OITranche:
    anchor_log_price: float
    vulnerable_direction: float
    weight: float


@dataclass(frozen=True)
class _WallCandidate:
    price: float
    distance_bps: float
    notional_usd: float


@dataclass
class _HeatmapDay:
    open_interest_usd: float | None = None
    oi_venue_count: float | None = None
    liquidation_dominance_pct: float | None = None
    liquidated_interest_long_usd: float | None = None
    liquidated_interest_short_usd: float | None = None
    downside_pressure: float | None = None
    upside_pressure: float | None = None
    net_pressure: float | None = None
    active_cluster_notional_usd: float | None = None
    observed_liquidation_notional_usd: float | None = None
    liquidation_downside: list[_WallCandidate] = field(default_factory=list)
    liquidation_upside: list[_WallCandidate] = field(default_factory=list)
    oi_area_downside: list[_WallCandidate] = field(default_factory=list)
    oi_area_upside: list[_WallCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class _SpaceScores:
    pressure_downside_score: float | None
    pressure_upside_score: float | None
    pressure_net_score: float | None
    liq_downside_candidate: _WallCandidate | None
    liq_downside_amount_score: float | None
    liq_downside_wall_score: float | None
    liq_upside_candidate: _WallCandidate | None
    liq_upside_amount_score: float | None
    liq_upside_wall_score: float | None
    oi_downside_candidate: _WallCandidate | None
    oi_downside_amount_score: float | None
    oi_downside_wall_score: float | None
    oi_upside_candidate: _WallCandidate | None
    oi_upside_amount_score: float | None
    oi_upside_wall_score: float | None
    cascade_score: float | None
    cascade_direction_score: float | None
    threshold_state: str


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_optional(value: float | None) -> str:
    return _format_number(value) if value is not None else ""


def _weighted_optional_score(
    first: float | None,
    second: float | None,
    *,
    first_weight: float,
    second_weight: float,
) -> float | None:
    components: list[tuple[float, float]] = []
    if first is not None:
        components.append((first, first_weight))
    if second is not None:
        components.append((second, second_weight))
    if not components:
        return None
    return sum(value * weight for value, weight in components) / sum(
        weight for _, weight in components
    )


def _max_optional(*values: float | None) -> float | None:
    active = [value for value in values if value is not None]
    return max(active) if active else None


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _robust_score(value: float | None, history: deque[float]) -> float | None:
    if value is None or len(history) < MIN_HISTORY:
        return None

    baseline = list(history)
    median = statistics.median(baseline)
    deviations = [abs(item - median) for item in baseline]
    mad = statistics.median(deviations)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = statistics.pstdev(baseline)
    if scale <= 1e-12:
        return 0.0

    z_score = _clamp((value - median) / scale, -4.0, 4.0)
    return 100.0 * math.tanh(z_score / 2.0)


def _funding_crowding_score(
    funding: float | None,
    predicted: float | None,
    funding_history: deque[float],
    predicted_history: deque[float],
) -> float | None:
    components: list[tuple[float, float]] = []
    funding_score = _robust_score(funding, funding_history)
    predicted_score = _robust_score(predicted, predicted_history)
    if funding_score is not None:
        components.append((funding_score, 0.6))
    if predicted_score is not None:
        components.append((predicted_score, 0.4))
    if not components:
        return None

    weighted_average = sum(value * weight for value, weight in components) / sum(
        weight for _, weight in components
    )
    # Above-normal positive funding means crowded longs and downside squeeze
    # potential. Below-normal/negative funding means crowded shorts and upside.
    return -weighted_average


def _liquidation_impulse_score(dominance: float | None) -> float | None:
    if dominance is None:
        return None
    # Positive source values are long-liquidation dominance (downside impulse);
    # negative source values are short-liquidation dominance (upside impulse).
    return _clamp(
        -dominance / LIQUIDATION_SATURATION_PCT * 100.0,
        -100.0,
        100.0,
    )


def _amount_score(value: float | None, history: deque[float]) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 0.0
    if len(history) < MIN_HISTORY:
        return None
    high_reference = _percentile(history, 0.90)
    if high_reference is None or high_reference <= 0:
        return 0.0
    return _clamp(value / high_reference * 100.0, 0.0, 100.0)


def _leverage_fuel_score(
    oi_change: float | None, positive_oi_history: deque[float]
) -> float | None:
    if oi_change is None:
        return None
    if oi_change <= 0:
        return 0.0
    if len(positive_oi_history) < MIN_HISTORY:
        return None

    high_reference = _percentile(positive_oi_history, 0.90)
    if high_reference is None or high_reference <= 0:
        return 0.0
    return _clamp(oi_change / high_reference * 100.0, 0.0, 100.0)


def _price_zone_proximity(
    anchor_price: float, current_price: float, band: float
) -> float:
    if anchor_price <= 0 or current_price <= 0 or band <= 0:
        return 0.0
    log_distance = math.log(current_price / anchor_price)
    return math.exp(-0.5 * (log_distance / band) ** 2)


def _volatility_trigger_score(
    log_return: float | None, baseline_volatility: float | None
) -> float | None:
    if log_return is None or baseline_volatility is None:
        return None
    baseline = max(baseline_volatility, MIN_BASELINE_VOLATILITY)
    standardized_move = abs(log_return) / baseline
    # Ignore routine moves below 0.75 sigma and saturate at 2.5 sigma.
    return _clamp((standardized_move - 0.75) / 1.75 * 100.0, 0.0, 100.0)


def _cascade_band_bps(
    baseline_volatility: float | None, realized_volatility: float | None
) -> float:
    volatility = baseline_volatility or realized_volatility or MIN_BASELINE_VOLATILITY
    return _clamp(
        CASCADE_BAND_VOL_MULTIPLIER
        * volatility
        * math.sqrt(OI_DISTRIBUTION_DAYS)
        * 10_000.0,
        MIN_CASCADE_BAND_BPS,
        MAX_CASCADE_BAND_BPS,
    )


def _closeness_score(distance_bps: float | None, band_bps: float) -> float | None:
    if distance_bps is None or not math.isfinite(distance_bps):
        return None
    return 100.0 * math.exp(-0.5 * (abs(distance_bps) / band_bps) ** 2)


def _candidate_selection_score(
    candidate: _WallCandidate, band_bps: float
) -> float:
    closeness = _closeness_score(candidate.distance_bps, band_bps) or 0.0
    return candidate.notional_usd * closeness


def _select_candidate(
    candidates: list[_WallCandidate], band_bps: float
) -> _WallCandidate | None:
    usable = [
        candidate
        for candidate in candidates
        if candidate.notional_usd > 0
        and candidate.price > 0
        and math.isfinite(candidate.notional_usd)
        and math.isfinite(candidate.price)
        and math.isfinite(candidate.distance_bps)
    ]
    if not usable:
        return None
    return max(
        usable,
        key=lambda candidate: (
            _candidate_selection_score(candidate, band_bps),
            -abs(candidate.distance_bps),
        ),
    )


def _wall_score(
    candidate: _WallCandidate | None,
    amount_history: deque[float],
    band_bps: float,
) -> tuple[float | None, float | None]:
    if candidate is None:
        return None, None
    amount = _amount_score(candidate.notional_usd, amount_history)
    closeness = _closeness_score(candidate.distance_bps, band_bps)
    if amount is None or closeness is None:
        return amount, None
    return amount, math.sqrt(amount * closeness)


def _threshold_state(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 65.0:
        return "cascade-ready"
    if score >= 35.0:
        return "watch"
    if score > 0:
        return "building"
    return "none"


def _realized_volatility_regime_score(
    realized_volatility: float | None, history: deque[float]
) -> float | None:
    if realized_volatility is None or len(history) < MIN_HISTORY:
        return None
    high_reference = _percentile(history, 0.90)
    if high_reference is None or high_reference <= 0:
        return 0.0
    return _clamp(realized_volatility / high_reference * 100.0, 0.0, 100.0)


def _direction_label(score: float) -> str:
    if score >= DIRECTION_ACTIVATION_THRESHOLD:
        return "upside"
    if score <= -DIRECTION_ACTIVATION_THRESHOLD:
        return "downside"
    return "neutral"


def _potential_label(score: float) -> str:
    if score >= 65:
        return "high"
    if score >= 35:
        return "elevated"
    return "low"


def _regime(direction: str, potential: str, fuel: float) -> str:
    if direction == "neutral":
        return "high-fuel neutral" if fuel >= 65 else "neutral"
    if potential == "high":
        return f"high {direction} volatility potential"
    if potential == "elevated":
        return f"elevated {direction} volatility potential"
    return f"low {direction} pressure"


def _agreement_score(
    liquidation_score: float | None, funding_score: float | None
) -> float:
    active = [
        score
        for score in (liquidation_score, funding_score)
        if score is not None and abs(score) >= 10
    ]
    if len(active) == 2:
        return 1.0 if active[0] * active[1] > 0 else 0.0
    if len(active) == 1:
        return 0.5
    return 0.25


def _date_from_timestamp_ms(value: str | None) -> str | None:
    timestamp = _parse_float(value)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp / 1_000.0, UTC).date().isoformat()


def _get_heatmap_day(
    space: dict[str, _HeatmapDay], date_text: str
) -> _HeatmapDay:
    if date_text not in space:
        space[date_text] = _HeatmapDay()
    return space[date_text]


def _read_optional_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _append_candidate(
    target: list[_WallCandidate],
    *,
    price: float | None,
    mark_price: float | None,
    distance_bps: float | None,
    notional_usd: float | None,
) -> None:
    if price is None or price <= 0 or not math.isfinite(price):
        return
    if distance_bps is None:
        if mark_price is None or mark_price <= 0:
            return
        distance_bps = (price / mark_price - 1.0) * 10_000.0
    if notional_usd is None or notional_usd <= 0:
        return
    if not math.isfinite(distance_bps) or not math.isfinite(notional_usd):
        return
    target.append(_WallCandidate(price, distance_bps, notional_usd))


def read_heatmap_space(
    *,
    daily_path: Path | None = DEFAULT_HEATMAP_DAILY_PATH,
    pressure_path: Path | None = DEFAULT_HEATMAP_PRESSURE_PATH,
    levels_path: Path | None = DEFAULT_HEATMAP_LEVELS_PATH,
    oi_area_levels_path: Path | None = DEFAULT_OI_AREA_LEVELS_PATH,
) -> dict[str, _HeatmapDay]:
    space: dict[str, _HeatmapDay] = {}

    for row in _read_optional_rows(daily_path):
        date_text = row.get("date")
        if not date_text:
            continue
        day = _get_heatmap_day(space, date_text)
        day.open_interest_usd = _parse_float(row.get("open_interest_usd"))
        day.oi_venue_count = _parse_float(row.get("oi_venue_count"))
        day.liquidation_dominance_pct = _parse_float(
            row.get("liquidation_dominance_pct")
        )
        day.liquidated_interest_long_usd = _parse_float(
            row.get("liquidated_interest_long_usd")
        )
        day.liquidated_interest_short_usd = _parse_float(
            row.get("liquidated_interest_short_usd")
        )

    for row in _read_optional_rows(pressure_path):
        date_text = _date_from_timestamp_ms(row.get("ts_ms"))
        if date_text is None:
            continue
        day = _get_heatmap_day(space, date_text)
        mark_price = _parse_float(row.get("mark_price"))
        day.downside_pressure = _parse_float(row.get("downside_pressure"))
        day.upside_pressure = _parse_float(row.get("upside_pressure"))
        day.net_pressure = _parse_float(row.get("net_pressure"))
        day.active_cluster_notional_usd = _parse_float(
            row.get("active_cluster_notional_usd")
        )
        day.observed_liquidation_notional_usd = _parse_float(
            row.get("observed_liquidation_notional_usd")
        )
        _append_candidate(
            day.liquidation_downside,
            price=_parse_float(row.get("nearest_below_price")),
            mark_price=mark_price,
            distance_bps=None,
            notional_usd=_parse_float(row.get("nearest_below_notional_usd")),
        )
        _append_candidate(
            day.liquidation_upside,
            price=_parse_float(row.get("nearest_above_price")),
            mark_price=mark_price,
            distance_bps=None,
            notional_usd=_parse_float(row.get("nearest_above_notional_usd")),
        )

    for row in _read_optional_rows(levels_path):
        date_text = _date_from_timestamp_ms(row.get("ts_ms"))
        if date_text is None:
            continue
        day = _get_heatmap_day(space, date_text)
        location = row.get("location")
        target = (
            day.liquidation_upside
            if location == "above"
            else day.liquidation_downside
            if location == "below"
            else None
        )
        if target is None:
            continue
        _append_candidate(
            target,
            price=_parse_float(row.get("liquidation_price")),
            mark_price=_parse_float(row.get("mark_price")),
            distance_bps=_parse_float(row.get("distance_bps")),
            notional_usd=_parse_float(row.get("estimated_notional_usd")),
        )

    for row in _read_optional_rows(oi_area_levels_path):
        date_text = _date_from_timestamp_ms(row.get("ts_ms"))
        if date_text is None:
            continue
        day = _get_heatmap_day(space, date_text)
        location = row.get("location")
        target = (
            day.oi_area_upside
            if location == "above"
            else day.oi_area_downside
            if location == "below"
            else None
        )
        if target is None:
            continue
        _append_candidate(
            target,
            price=_parse_float(row.get("interest_price")),
            mark_price=_parse_float(row.get("mark_price")),
            distance_bps=_parse_float(row.get("distance_bps")),
            notional_usd=_parse_float(row.get("active_open_interest_usd")),
        )

    return space


def _score_liquidation_space(
    day: _HeatmapDay | None,
    *,
    band_bps: float,
    downside_pressure_history: deque[float],
    upside_pressure_history: deque[float],
    liquidation_wall_amount_history: deque[float],
    oi_area_amount_history: deque[float],
) -> _SpaceScores:
    if day is None:
        return _SpaceScores(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "",
        )

    pressure_downside = _amount_score(
        day.downside_pressure, downside_pressure_history
    )
    pressure_upside = _amount_score(day.upside_pressure, upside_pressure_history)
    pressure_net = (
        _clamp((pressure_upside or 0.0) - (pressure_downside or 0.0), -100.0, 100.0)
        if pressure_downside is not None or pressure_upside is not None
        else None
    )

    liq_downside = _select_candidate(day.liquidation_downside, band_bps)
    liq_upside = _select_candidate(day.liquidation_upside, band_bps)
    oi_downside = _select_candidate(day.oi_area_downside, band_bps)
    oi_upside = _select_candidate(day.oi_area_upside, band_bps)

    liq_downside_amount, liq_downside_score = _wall_score(
        liq_downside, liquidation_wall_amount_history, band_bps
    )
    liq_upside_amount, liq_upside_score = _wall_score(
        liq_upside, liquidation_wall_amount_history, band_bps
    )
    oi_downside_amount, oi_downside_score = _wall_score(
        oi_downside, oi_area_amount_history, band_bps
    )
    oi_upside_amount, oi_upside_score = _wall_score(
        oi_upside, oi_area_amount_history, band_bps
    )

    downside_score = _max_optional(
        liq_downside_score,
        oi_downside_score,
        pressure_downside,
    )
    upside_score = _max_optional(
        liq_upside_score,
        oi_upside_score,
        pressure_upside,
    )
    cascade_score = (
        max(upside_score or 0.0, downside_score or 0.0)
        if upside_score is not None or downside_score is not None
        else None
    )
    cascade_direction = (
        _clamp((upside_score or 0.0) - (downside_score or 0.0), -100.0, 100.0)
        if cascade_score is not None
        else None
    )
    if cascade_direction is not None and pressure_net is not None:
        cascade_direction = _clamp(
            0.75 * cascade_direction + 0.25 * pressure_net,
            -100.0,
            100.0,
        )

    return _SpaceScores(
        pressure_downside,
        pressure_upside,
        pressure_net,
        liq_downside,
        liq_downside_amount,
        liq_downside_score,
        liq_upside,
        liq_upside_amount,
        liq_upside_score,
        oi_downside,
        oi_downside_amount,
        oi_downside_score,
        oi_upside,
        oi_upside_amount,
        oi_upside_score,
        cascade_score,
        cascade_direction,
        _threshold_state(cascade_score),
    )


def _remember_liquidation_space(
    day: _HeatmapDay | None,
    *,
    downside_pressure_history: deque[float],
    upside_pressure_history: deque[float],
    liquidation_wall_amount_history: deque[float],
    oi_area_amount_history: deque[float],
) -> None:
    if day is None:
        return
    if day.downside_pressure is not None and day.downside_pressure > 0:
        downside_pressure_history.append(day.downside_pressure)
    if day.upside_pressure is not None and day.upside_pressure > 0:
        upside_pressure_history.append(day.upside_pressure)
    liquidation_amounts = [
        candidate.notional_usd
        for candidate in day.liquidation_downside + day.liquidation_upside
        if candidate.notional_usd > 0
    ]
    if liquidation_amounts:
        liquidation_wall_amount_history.append(max(liquidation_amounts))
    oi_area_amounts = [
        candidate.notional_usd
        for candidate in day.oi_area_downside + day.oi_area_upside
        if candidate.notional_usd > 0
    ]
    if oi_area_amounts:
        oi_area_amount_history.append(max(oi_area_amounts))


def _signed_forward_jump(
    current_price: float,
    future_prices: list[float],
) -> float | None:
    if not future_prices or current_price <= 0:
        return None
    upside = max(future_prices) / current_price - 1.0
    downside = min(future_prices) / current_price - 1.0
    return (upside if abs(upside) >= abs(downside) else downside) * 100.0


def _forward_jump_targets(
    prices: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    ordinal_prices: dict[int, float] = {}
    for date_text, price in prices.items():
        if price <= 0 or not math.isfinite(price):
            continue
        try:
            ordinal_prices[date.fromisoformat(date_text).toordinal()] = price
        except ValueError:
            continue

    targets: dict[str, dict[str, float | str]] = {}
    for ordinal, current_price in sorted(ordinal_prices.items()):
        date_text = date.fromordinal(ordinal).isoformat()
        row_targets: dict[str, float | str] = {}
        for horizon in (1, 3, 7):
            future_prices = [
                ordinal_prices.get(ordinal + offset)
                for offset in range(1, horizon + 1)
            ]
            if any(value is None for value in future_prices):
                continue
            jump = _signed_forward_jump(
                current_price,
                [value for value in future_prices if value is not None],
            )
            if jump is not None:
                row_targets[f"btc_forward_jump_{horizon}d_pct"] = jump
        seven_day_jump = row_targets.get("btc_forward_jump_7d_pct")
        if isinstance(seven_day_jump, float):
            row_targets["btc_forward_abs_jump_7d_pct"] = abs(seven_day_jump)
            row_targets["btc_forward_jump_direction"] = (
                "upside"
                if seven_day_jump >= JEPA_JUMP_DIRECTION_THRESHOLD_PCT
                else "downside"
                if seven_day_jump <= -JEPA_JUMP_DIRECTION_THRESHOLD_PCT
                else "neutral"
            )
        targets[date_text] = row_targets
    return targets


def score_rows(
    rows: list[dict[str, str]],
    prices: dict[str, float] | None = None,
    heatmap_space: dict[str, _HeatmapDay] | None = None,
) -> list[dict[str, str]]:
    prices = prices or {}
    heatmap_space = heatmap_space or {}
    forward_targets = _forward_jump_targets(prices)
    funding_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    predicted_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    positive_oi_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    negative_oi_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    active_inventory_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    return_history: deque[float] = deque(maxlen=RETURN_WINDOW)
    realized_volatility_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    downside_pressure_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    upside_pressure_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    liquidation_wall_amount_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    oi_area_amount_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
    tranches: list[_OITranche] = []
    previous_price: float | None = None
    daily_anchor_decay = 0.5 ** (1.0 / OI_ANCHOR_HALF_LIFE_DAYS)
    output: list[dict[str, str]] = []

    for row in sorted(rows, key=lambda item: item["date"]):
        heatmap_day = heatmap_space.get(row["date"])
        funding = _parse_float(row.get("aggregated_funding_rate_avg_pct"))
        predicted = _parse_float(
            row.get("aggregated_predicted_funding_rate_avg_pct")
        )
        oi_change = _parse_float(row.get("futures_oi_7day_change_pct"))
        dominance = _parse_float(row.get("liquidation_dominance_pct"))

        source_liquidation_score = _liquidation_impulse_score(dominance)
        heatmap_liquidation_score = _liquidation_impulse_score(
            heatmap_day.liquidation_dominance_pct if heatmap_day else None
        )
        liquidation_score = _weighted_optional_score(
            source_liquidation_score,
            heatmap_liquidation_score,
            first_weight=0.65,
            second_weight=0.35,
        )
        funding_score = _funding_crowding_score(
            funding,
            predicted,
            funding_history,
            predicted_history,
        )
        new_oi_score = _leverage_fuel_score(
            oi_change, positive_oi_history
        )
        price = prices.get(row["date"])
        if price is not None and (price <= 0 or not math.isfinite(price)):
            price = None

        log_return = (
            math.log(price / previous_price)
            if price is not None and previous_price is not None
            else None
        )
        baseline_volatility = (
            statistics.pstdev(return_history)
            if len(return_history) >= MIN_RETURN_HISTORY
            else None
        )
        recent_returns = list(return_history)[-(OI_DISTRIBUTION_DAYS - 1) :]
        if log_return is not None:
            recent_returns.append(log_return)
        realized_volatility = (
            math.sqrt(
                sum(value * value for value in recent_returns)
                / len(recent_returns)
            )
            if recent_returns
            else None
        )
        volatility_regime_score = _realized_volatility_regime_score(
            realized_volatility,
            realized_volatility_history,
        )
        volatility_trigger_score = _volatility_trigger_score(
            log_return, baseline_volatility
        )

        depletion_score = 0.0
        if (
            oi_change is not None
            and oi_change < 0
            and len(negative_oi_history) >= MIN_HISTORY
        ):
            negative_reference = _percentile(negative_oi_history, 0.90)
            if negative_reference and negative_reference > 0:
                depletion_score = _clamp(
                    abs(oi_change) / negative_reference, 0.0, 1.0
                )
        depletion_multiplier = (
            1.0 - OI_DEPLETION_RATE * depletion_score / OI_DISTRIBUTION_DAYS
        )
        for tranche in tranches:
            tranche.weight *= daily_anchor_decay * depletion_multiplier
        tranches = [tranche for tranche in tranches if tranche.weight >= 0.001]

        if price is not None and new_oi_score is not None and new_oi_score > 0:
            tranches.append(
                _OITranche(
                    anchor_log_price=math.log(price),
                    vulnerable_direction=_clamp(
                        funding_score or 0.0, -100.0, 100.0
                    )
                    / 100.0,
                    weight=(new_oi_score / 100.0) / OI_DISTRIBUTION_DAYS,
                )
            )

        price_band = (
            _clamp(
                PRICE_BAND_VOL_MULTIPLIER
                * baseline_volatility
                * math.sqrt(OI_DISTRIBUTION_DAYS),
                MIN_PRICE_BAND,
                MAX_PRICE_BAND,
            )
            if baseline_volatility is not None
            else MIN_PRICE_BAND
        )
        active_inventory = 0.0
        total_inventory = 0.0
        directional_inventory = 0.0
        anchor_log_sum = 0.0
        if price is not None:
            for tranche in tranches:
                total_inventory += tranche.weight
                anchor_price = math.exp(tranche.anchor_log_price)
                active_weight = tranche.weight * _price_zone_proximity(
                    anchor_price, price, price_band
                )
                active_inventory += active_weight
                directional_inventory += (
                    active_weight * tranche.vulnerable_direction
                )
                anchor_log_sum += active_weight * tranche.anchor_log_price

        fuel_score: float | None = None
        if len(active_inventory_history) >= MIN_HISTORY:
            inventory_reference = _percentile(
                active_inventory_history, 0.90
            )
            fuel_score = (
                _clamp(
                    active_inventory / inventory_reference * 100.0,
                    0.0,
                    100.0,
                )
                if inventory_reference and inventory_reference > 0
                else new_oi_score or 0.0
            )
        reactivation_score = (
            _clamp(
                active_inventory / total_inventory * 100.0,
                0.0,
                100.0,
            )
            if total_inventory > 0
            else 0.0
        )
        anchor_direction_score = (
            _clamp(
                directional_inventory / active_inventory * 100.0,
                -100.0,
                100.0,
            )
            if active_inventory > 0
            else 0.0
        )
        estimated_anchor_price = (
            math.exp(anchor_log_sum / active_inventory)
            if active_inventory > 0
            else None
        )
        anchor_pressure = (
            anchor_direction_score * reactivation_score / 100.0
        )

        jump_direction = (
            1.0
            if log_return is not None and log_return > 0
            else -1.0
            if log_return is not None and log_return < 0
            else 0.0
        )
        cascade_alignment = max(
            0.0, jump_direction * anchor_direction_score / 100.0
        )
        cascade_score = (
            jump_direction
            * (volatility_trigger_score or 0.0)
            * ((fuel_score or 0.0) / 100.0)
            * cascade_alignment
        )
        cascade_band_bps = _cascade_band_bps(
            baseline_volatility,
            realized_volatility,
        )
        space_scores = _score_liquidation_space(
            heatmap_day,
            band_bps=cascade_band_bps,
            downside_pressure_history=downside_pressure_history,
            upside_pressure_history=upside_pressure_history,
            liquidation_wall_amount_history=liquidation_wall_amount_history,
            oi_area_amount_history=oi_area_amount_history,
        )

        base_direction_score = _clamp(
            0.45 * cascade_score
            + 0.20 * (liquidation_score or 0.0)
            + 0.20 * anchor_pressure
            + 0.15 * (funding_score or 0.0),
            -100.0,
            100.0,
        )
        direction_score = base_direction_score
        if space_scores.cascade_direction_score is not None:
            direction_score = _clamp(
                0.72 * base_direction_score
                + 0.28 * space_scores.cascade_direction_score,
                -100.0,
                100.0,
            )
        base_potential_score = _clamp(
            0.55 * (volatility_regime_score or 0.0)
            + 0.20 * (volatility_trigger_score or 0.0)
            + 0.15 * (fuel_score or 0.0)
            + 0.10 * abs(direction_score),
            0.0,
            100.0,
        )
        potential_score = base_potential_score
        if space_scores.cascade_score is not None:
            potential_score = _clamp(
                0.78 * base_potential_score
                + 0.22 * space_scores.cascade_score,
                0.0,
                100.0,
            )
        signed_potential = (
            math.copysign(potential_score, direction_score)
            if abs(direction_score) >= DIRECTION_ACTIVATION_THRESHOLD
            else 0.0
        )

        base_available_metrics = sum(
            value is not None
            for value in (funding, predicted, oi_change, dominance, price)
        )
        heatmap_available = heatmap_day is not None and any(
            value is not None
            for value in (
                heatmap_day.open_interest_usd if heatmap_day else None,
                heatmap_day.downside_pressure if heatmap_day else None,
                heatmap_day.upside_pressure if heatmap_day else None,
                heatmap_day.liquidation_dominance_pct if heatmap_day else None,
            )
        )
        completeness = (
            (base_available_metrics + int(heatmap_available)) / 6.0
            if heatmap_space
            else base_available_metrics / 5.0
        )
        funding_count = _parse_float(row.get("funding_contract_count")) or 0.0
        predicted_count = (
            _parse_float(row.get("predicted_funding_contract_count")) or 0.0
        )
        contract_coverage = statistics.fmean(
            [
                _clamp(funding_count / MAX_FUNDING_CONTRACTS, 0.0, 1.0),
                _clamp(
                    predicted_count / MAX_PREDICTED_CONTRACTS,
                    0.0,
                    1.0,
                ),
            ]
        )
        agreement_peer = (
            space_scores.cascade_direction_score
            if space_scores.cascade_direction_score is not None
            else anchor_pressure
            if price is not None
            else funding_score
        )
        agreement = _agreement_score(
            liquidation_score,
            agreement_peer,
        )
        history_ready = statistics.fmean(
            [
                min(len(funding_history) / MIN_HISTORY, 1.0),
                min(len(positive_oi_history) / MIN_HISTORY, 1.0),
                min(len(return_history) / MIN_RETURN_HISTORY, 1.0),
                min(len(active_inventory_history) / MIN_HISTORY, 1.0),
                min(len(liquidation_wall_amount_history) / MIN_HISTORY, 1.0)
                if heatmap_space
                else 1.0,
            ]
        )
        confidence_score = 100.0 * (
            0.30 * completeness
            + 0.20 * contract_coverage
            + 0.25 * agreement
            + 0.25 * history_ready
        )

        direction = _direction_label(direction_score)
        potential = _potential_label(potential_score)
        jump_targets = forward_targets.get(row["date"], {})
        output.append(
            {
                "date": row["date"],
                "directional_bias": direction,
                "directional_bias_score": _format_number(direction_score),
                "volatility_potential": potential,
                "volatility_potential_score": _format_number(potential_score),
                "directional_volatility_score": _format_number(
                    signed_potential
                ),
                "leverage_fuel_score": (
                    _format_number(fuel_score) if fuel_score is not None else ""
                ),
                "new_oi_addition_score": (
                    _format_number(new_oi_score)
                    if new_oi_score is not None
                    else ""
                ),
                "oi_zone_reactivation_score": _format_number(
                    reactivation_score
                ),
                "oi_anchor_direction_score": _format_number(
                    anchor_direction_score
                ),
                "estimated_oi_anchor_price": (
                    _format_number(estimated_anchor_price)
                    if estimated_anchor_price is not None
                    else ""
                ),
                "volatility_trigger_score": (
                    _format_number(volatility_trigger_score)
                    if volatility_trigger_score is not None
                    else ""
                ),
                "realized_volatility_regime_score": (
                    _format_number(volatility_regime_score)
                    if volatility_regime_score is not None
                    else ""
                ),
                "cascade_activation_score": _format_number(cascade_score),
                "liquidation_impulse_score": (
                    _format_number(liquidation_score)
                    if liquidation_score is not None
                    else ""
                ),
                "funding_crowding_score": (
                    _format_number(funding_score)
                    if funding_score is not None
                    else ""
                ),
                "signal_confidence_score": _format_number(confidence_score),
                "heatmap_open_interest_usd": _format_optional(
                    heatmap_day.open_interest_usd if heatmap_day else None
                ),
                "heatmap_oi_venue_count": _format_optional(
                    heatmap_day.oi_venue_count if heatmap_day else None
                ),
                "heatmap_liquidation_dominance_pct": _format_optional(
                    heatmap_day.liquidation_dominance_pct
                    if heatmap_day
                    else None
                ),
                "heatmap_liquidated_interest_long_usd": _format_optional(
                    heatmap_day.liquidated_interest_long_usd
                    if heatmap_day
                    else None
                ),
                "heatmap_liquidated_interest_short_usd": _format_optional(
                    heatmap_day.liquidated_interest_short_usd
                    if heatmap_day
                    else None
                ),
                "liquidation_pressure_downside_score": _format_optional(
                    space_scores.pressure_downside_score
                ),
                "liquidation_pressure_upside_score": _format_optional(
                    space_scores.pressure_upside_score
                ),
                "liquidation_pressure_net_score": _format_optional(
                    space_scores.pressure_net_score
                ),
                "liquidation_wall_downside_distance_bps": _format_optional(
                    space_scores.liq_downside_candidate.distance_bps
                    if space_scores.liq_downside_candidate
                    else None
                ),
                "liquidation_wall_downside_amount_score": _format_optional(
                    space_scores.liq_downside_amount_score
                ),
                "liquidation_wall_upside_distance_bps": _format_optional(
                    space_scores.liq_upside_candidate.distance_bps
                    if space_scores.liq_upside_candidate
                    else None
                ),
                "liquidation_wall_upside_amount_score": _format_optional(
                    space_scores.liq_upside_amount_score
                ),
                "oi_area_downside_distance_bps": _format_optional(
                    space_scores.oi_downside_candidate.distance_bps
                    if space_scores.oi_downside_candidate
                    else None
                ),
                "oi_area_downside_amount_score": _format_optional(
                    space_scores.oi_downside_amount_score
                ),
                "oi_area_upside_distance_bps": _format_optional(
                    space_scores.oi_upside_candidate.distance_bps
                    if space_scores.oi_upside_candidate
                    else None
                ),
                "oi_area_upside_amount_score": _format_optional(
                    space_scores.oi_upside_amount_score
                ),
                "cascade_closeness_amount_score": _format_optional(
                    space_scores.cascade_score
                ),
                "cascade_closeness_amount_direction_score": _format_optional(
                    space_scores.cascade_direction_score
                ),
                "cascade_threshold_state": space_scores.threshold_state,
                "btc_forward_jump_1d_pct": _format_optional(
                    jump_targets.get("btc_forward_jump_1d_pct")
                    if isinstance(
                        jump_targets.get("btc_forward_jump_1d_pct"), float
                    )
                    else None
                ),
                "btc_forward_jump_3d_pct": _format_optional(
                    jump_targets.get("btc_forward_jump_3d_pct")
                    if isinstance(
                        jump_targets.get("btc_forward_jump_3d_pct"), float
                    )
                    else None
                ),
                "btc_forward_jump_7d_pct": _format_optional(
                    jump_targets.get("btc_forward_jump_7d_pct")
                    if isinstance(
                        jump_targets.get("btc_forward_jump_7d_pct"), float
                    )
                    else None
                ),
                "btc_forward_abs_jump_7d_pct": _format_optional(
                    jump_targets.get("btc_forward_abs_jump_7d_pct")
                    if isinstance(
                        jump_targets.get("btc_forward_abs_jump_7d_pct"), float
                    )
                    else None
                ),
                "btc_forward_jump_direction": str(
                    jump_targets.get("btc_forward_jump_direction", "")
                ),
                "regime": _regime(direction, potential, fuel_score or 0.0),
            }
        )

        if funding is not None:
            funding_history.append(funding)
        if predicted is not None:
            predicted_history.append(predicted)
        if oi_change is not None and oi_change > 0:
            positive_oi_history.append(oi_change)
        if oi_change is not None and oi_change < 0:
            negative_oi_history.append(abs(oi_change))
        if price is not None:
            active_inventory_history.append(active_inventory)
        if log_return is not None:
            return_history.append(log_return)
        if realized_volatility is not None and log_return is not None:
            realized_volatility_history.append(realized_volatility)
        if price is not None:
            previous_price = price
        _remember_liquidation_space(
            heatmap_day,
            downside_pressure_history=downside_pressure_history,
            upside_pressure_history=upside_pressure_history,
            liquidation_wall_amount_history=liquidation_wall_amount_history,
            oi_area_amount_history=oi_area_amount_history,
        )

    return output


def read_raw_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise ScrapeError(f"Raw data file does not exist: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = RAW_FIELDS - fieldnames
        if missing:
            raise ScrapeError(
                f"{input_path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        return list(reader)


def write_signal_rows(output_path: Path, rows: list[dict[str, str]]) -> int:
    normalized_rows = [
        {field: row.get(field, "") for field in SIGNAL_FIELDS} for row in rows
    ]
    return write_csv_rows(
        output_path,
        normalized_rows,
        fieldnames=SIGNAL_FIELDS,
    )


def generate_signal_file(
    input_path: Path,
    output_path: Path,
    price_path: Path = DEFAULT_BTC_PRICE_PATH,
    heatmap_daily_path: Path | None = DEFAULT_HEATMAP_DAILY_PATH,
    heatmap_pressure_path: Path | None = DEFAULT_HEATMAP_PRESSURE_PATH,
    heatmap_levels_path: Path | None = DEFAULT_HEATMAP_LEVELS_PATH,
    oi_area_levels_path: Path | None = DEFAULT_OI_AREA_LEVELS_PATH,
) -> int:
    try:
        from .indicator import read_price_rows
    except ImportError:
        from indicator import read_price_rows

    prices = read_price_rows(price_path)
    if not prices:
        raise ScrapeError(f"BTC price file has no rows: {price_path}")
    heatmap_space = read_heatmap_space(
        daily_path=heatmap_daily_path,
        pressure_path=heatmap_pressure_path,
        levels_path=heatmap_levels_path,
        oi_area_levels_path=oi_area_levels_path,
    )
    rows = score_rows(read_raw_rows(input_path), prices, heatmap_space)
    return write_signal_rows(output_path, rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate directional volatility potential signals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Raw derivatives CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SIGNAL_OUTPUT_PATH,
        help="Signal CSV to replace atomically (default: %(default)s)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_BTC_PRICE_PATH,
        help="BTC daily-close CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--heatmap-daily",
        type=Path,
        default=DEFAULT_HEATMAP_DAILY_PATH,
        help="Optional heatmap daily OI/liquidation CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--heatmap-pressure",
        type=Path,
        default=DEFAULT_HEATMAP_PRESSURE_PATH,
        help="Optional heatmap pressure CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--heatmap-levels",
        type=Path,
        default=DEFAULT_HEATMAP_LEVELS_PATH,
        help="Optional heatmap liquidation level CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--oi-area-levels",
        type=Path,
        default=DEFAULT_OI_AREA_LEVELS_PATH,
        help="Optional active OI area level CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--no-heatmap-enrichment",
        action="store_true",
        help="Generate only the root-source signal fields and jump targets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = generate_signal_file(
            args.input.resolve(),
            args.output.resolve(),
            args.prices.resolve(),
            None if args.no_heatmap_enrichment else args.heatmap_daily.resolve(),
            None if args.no_heatmap_enrichment else args.heatmap_pressure.resolve(),
            None if args.no_heatmap_enrichment else args.heatmap_levels.resolve(),
            None if args.no_heatmap_enrichment else args.oi_area_levels.resolve(),
        )
        with args.output.resolve().open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        latest = rows[-1] if rows else None
        print(f"Generated {count} signal rows at {args.output.resolve()}")
        if latest:
            print(
                f"Latest {latest['date']}: {latest['regime']} "
                f"(direction {latest['directional_bias_score']}, "
                f"potential {latest['volatility_potential_score']}, "
                f"confidence {latest['signal_confidence_score']})"
            )
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
