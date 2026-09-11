from __future__ import annotations

import argparse
import csv
import html
import math
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

try:
    from .main import (
        COINALYZE_API_BASE,
        COINALYZE_API_DOCS,
        DEFAULT_BTC_PRICE_PATH,
        DEFAULT_INDICATOR_CHART_PATH,
        DEFAULT_SIGNAL_OUTPUT_PATH,
        ConfigurationError,
        ScrapeError,
        _format_number,
        fetch_json,
        write_csv_rows,
    )
except ImportError:
    from main import (
        COINALYZE_API_BASE,
        COINALYZE_API_DOCS,
        DEFAULT_BTC_PRICE_PATH,
        DEFAULT_INDICATOR_CHART_PATH,
        DEFAULT_SIGNAL_OUTPUT_PATH,
        ConfigurationError,
        ScrapeError,
        _format_number,
        fetch_json,
        write_csv_rows,
    )


BTC_PRICE_SYMBOL = "BTCUSDT_PERP.A"
BTC_PRICE_FIELDS = ["date", "btc_close"]
DEFAULT_PRICE_START = date(2019, 1, 1)

SVG_WIDTH = 1800
SVG_HEIGHT = 900
PLOT_LEFT = 105
PLOT_RIGHT = 1680
PLOT_TOP = 92
PLOT_BOTTOM = 795
ZERO_Y = (PLOT_TOP + PLOT_BOTTOM) / 2


def _date_timestamp(value: date, *, end_of_day: bool = False) -> int:
    boundary = datetime.max.time() if end_of_day else datetime.min.time()
    return int(datetime.combine(value, boundary, tzinfo=UTC).timestamp())


def parse_price_history(response: object) -> dict[str, float]:
    if not isinstance(response, list):
        raise ScrapeError("Coinalyze BTC price response is not a list")

    prices: dict[str, float] = {}
    for item in response:
        if not isinstance(item, dict) or item.get("symbol") != BTC_PRICE_SYMBOL:
            continue
        history = item.get("history")
        if not isinstance(history, list):
            continue
        for candle in history:
            if not isinstance(candle, dict):
                continue
            try:
                timestamp = int(candle["t"])
                close = float(candle["c"])
            except (KeyError, TypeError, ValueError):
                continue
            if close <= 0 or not math.isfinite(close):
                continue
            date_text = datetime.fromtimestamp(timestamp, UTC).date().isoformat()
            prices[date_text] = close
    return prices


def read_price_rows(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != BTC_PRICE_FIELDS:
            raise ScrapeError(
                f"{path} has unexpected columns. Expected: "
                + ", ".join(BTC_PRICE_FIELDS)
            )
        return {
            row["date"]: float(row["btc_close"])
            for row in reader
            if row.get("date") and row.get("btc_close")
        }


def update_btc_prices(
    api_key: str,
    output_path: Path = DEFAULT_BTC_PRICE_PATH,
    *,
    full_history: bool = False,
    end_date: date | None = None,
    timeout: float = 30.0,
) -> int:
    if not api_key:
        raise ConfigurationError(
            "COINALYZE_API_KEY is required. Generate a free key at "
            f"{COINALYZE_API_DOCS}"
        )

    end_date = end_date or (datetime.now(UTC).date() - timedelta(days=1))
    existing = {} if full_history else read_price_rows(output_path)
    end_text = end_date.isoformat()
    existing = {
        observed_date: close
        for observed_date, close in existing.items()
        if observed_date <= end_text
    }
    if existing:
        latest = date.fromisoformat(max(existing))
        start_date = max(DEFAULT_PRICE_START, latest - timedelta(days=7))
    else:
        start_date = DEFAULT_PRICE_START

    query = urlencode(
        {
            "symbols": BTC_PRICE_SYMBOL,
            "interval": "daily",
            "from": _date_timestamp(start_date),
            "to": _date_timestamp(end_date, end_of_day=True),
        }
    )
    response = fetch_json(
        f"{COINALYZE_API_BASE}/ohlcv-history?{query}",
        headers={"api_key": api_key},
        timeout=timeout,
    )
    existing.update(parse_price_history(response))
    rows = [
        {"date": date_text, "btc_close": _format_number(existing[date_text])}
        for date_text in sorted(existing)
    ]
    return write_csv_rows(output_path, rows, fieldnames=BTC_PRICE_FIELDS)


def read_indicator_points(
    signal_path: Path, price_path: Path
) -> list[tuple[date, float, float, float]]:
    if not signal_path.exists():
        raise ScrapeError(f"Signal file does not exist: {signal_path}")
    prices = read_price_rows(price_path)
    if not prices:
        raise ScrapeError(f"BTC price file has no rows: {price_path}")

    points: list[tuple[date, float, float, float]] = []
    with signal_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "date",
            "directional_volatility_score",
            "signal_confidence_score",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ScrapeError(
                f"{signal_path} is missing columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            price = prices.get(row["date"])
            if price is None or not row.get("directional_volatility_score"):
                continue
            try:
                score = float(row["directional_volatility_score"])
                confidence = float(row["signal_confidence_score"])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (price, score, confidence)):
                continue
            points.append(
                (date.fromisoformat(row["date"]), score, confidence, price)
            )
    if len(points) < 2:
        raise ScrapeError("Not enough overlapping signal and BTC price data")
    return sorted(points)


def _x_mapper(points: list[tuple[date, float, float, float]]):
    start = points[0][0].toordinal()
    span = max(points[-1][0].toordinal() - start, 1)

    def map_x(value: date) -> float:
        return PLOT_LEFT + (
            (value.toordinal() - start) / span * (PLOT_RIGHT - PLOT_LEFT)
        )

    return map_x


def _indicator_y(value: float) -> float:
    value = max(-100.0, min(100.0, value))
    return PLOT_TOP + (100.0 - value) / 200.0 * (PLOT_BOTTOM - PLOT_TOP)


def _price_mapper(prices: list[float]):
    minimum = min(prices)
    maximum = max(prices)
    log_min = math.log10(minimum)
    log_max = math.log10(maximum)
    span = max(log_max - log_min, 1e-9)

    def map_price(value: float) -> float:
        return PLOT_BOTTOM - (
            (math.log10(value) - log_min) / span * (PLOT_BOTTOM - PLOT_TOP)
        )

    return map_price


def _path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def _area_path(
    points: list[tuple[date, float, float, float]],
    map_x,
    *,
    positive: bool,
) -> str:
    upper = [
        (
            map_x(point_date),
            _indicator_y(max(score, 0.0) if positive else min(score, 0.0)),
        )
        for point_date, score, _, _ in points
    ]
    lower = [(x, ZERO_Y) for x, _ in reversed(upper)]
    return _path(upper + lower) + " Z"


def _price_ticks(minimum: float, maximum: float) -> list[float]:
    ticks: list[float] = []
    start_power = math.floor(math.log10(minimum))
    end_power = math.ceil(math.log10(maximum))
    for power in range(start_power, end_power + 1):
        base = 10**power
        for multiplier in (1, 2, 5):
            value = multiplier * base
            if minimum <= value <= maximum:
                ticks.append(float(value))
    return ticks


def _format_usd(value: float) -> str:
    if value >= 1000:
        return f"${value / 1000:g}k"
    return f"${value:g}"


def render_svg(
    points: list[tuple[date, float, float, float]], output_path: Path
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_x = _x_mapper(points)
    prices = [price for _, _, _, price in points]
    map_price = _price_mapper(prices)
    scores = [score for _, score, _, _ in points]

    price_line = _path(
        [(map_x(point_date), map_price(price)) for point_date, _, _, price in points]
    )
    indicator_line = _path(
        [
            (map_x(point_date), _indicator_y(score))
            for (point_date, _, _, _), score in zip(points, scores, strict=True)
        ]
    )
    positive_area = _area_path(points, map_x, positive=True)
    negative_area = _area_path(points, map_x, positive=False)

    years = range(points[0][0].year, points[-1][0].year + 1)
    year_grid = []
    for year in years:
        year_date = date(year, 1, 1)
        if year_date < points[0][0] or year_date > points[-1][0]:
            continue
        x = map_x(year_date)
        year_grid.append(
            f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" '
            f'y2="{PLOT_BOTTOM}" class="grid"/>'
            f'<text x="{x:.2f}" y="830" class="axis center">{year}</text>'
        )

    indicator_grid = []
    for value in (-100, -65, -35, 0, 35, 65, 100):
        y = _indicator_y(value)
        grid_class = "zero" if value == 0 else "grid"
        indicator_grid.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{PLOT_RIGHT}" '
            f'y2="{y:.2f}" class="{grid_class}"/>'
            f'<text x="{PLOT_LEFT - 14}" y="{y + 5:.2f}" '
            f'class="axis end">{value:+d}</text>'
        )

    price_grid = []
    for value in _price_ticks(min(prices), max(prices)):
        y = map_price(value)
        price_grid.append(
            f'<text x="{PLOT_RIGHT + 16}" y="{y + 5:.2f}" '
            f'class="price-axis">{html.escape(_format_usd(value))}</text>'
        )

    latest_date, latest_score, latest_confidence, latest_price = points[-1]
    subtitle = (
        f"{points[0][0].isoformat()} to {latest_date.isoformat()} | "
        f"Latest score {latest_score:+.1f} | BTC {_format_usd(latest_price)} | "
        f"Confidence {latest_confidence:.1f}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
<defs>
  <linearGradient id="upFill" x1="0" y1="1" x2="0" y2="0">
    <stop offset="0%" stop-color="#19c37d" stop-opacity="0.04"/>
    <stop offset="100%" stop-color="#19c37d" stop-opacity="0.62"/>
  </linearGradient>
  <linearGradient id="downFill" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#ff4d6d" stop-opacity="0.04"/>
    <stop offset="100%" stop-color="#ff4d6d" stop-opacity="0.62"/>
  </linearGradient>
  <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<style>
  .bg {{ fill:#0b1020; }}
  .plot {{ fill:#10182b; stroke:#24324e; stroke-width:1; }}
  .grid {{ stroke:#263552; stroke-width:1; opacity:.65; }}
  .zero {{ stroke:#8390aa; stroke-width:1.5; opacity:.9; }}
  .axis {{ fill:#9eabc4; font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .price-axis {{ fill:#f6c453; font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .title {{ fill:#f4f7ff; font:700 28px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .subtitle {{ fill:#aab6ce; font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .legend {{ fill:#d9e1f2; font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .center {{ text-anchor:middle; }}
  .end {{ text-anchor:end; }}
</style>
<rect width="100%" height="100%" class="bg"/>
<text x="{PLOT_LEFT}" y="38" class="title">BTC Directional Volatility Potential</text>
<text x="{PLOT_LEFT}" y="65" class="subtitle">{html.escape(subtitle)}</text>
<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT - PLOT_LEFT}" height="{PLOT_BOTTOM - PLOT_TOP}" class="plot"/>
<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT - PLOT_LEFT}" height="{_indicator_y(65) - PLOT_TOP}" fill="#19c37d" opacity=".035"/>
<rect x="{PLOT_LEFT}" y="{_indicator_y(-65)}" width="{PLOT_RIGHT - PLOT_LEFT}" height="{PLOT_BOTTOM - _indicator_y(-65)}" fill="#ff4d6d" opacity=".035"/>
{''.join(year_grid)}
{''.join(indicator_grid)}
<path d="{positive_area}" fill="url(#upFill)" stroke="none"/>
<path d="{negative_area}" fill="url(#downFill)" stroke="none"/>
<path d="{indicator_line}" fill="none" stroke="#edf3ff" stroke-width="1.8" opacity=".9"/>
<path d="{price_line}" fill="none" stroke="#f6c453" stroke-width="2.2" opacity=".95" filter="url(#glow)"/>
{''.join(price_grid)}
<text x="{PLOT_LEFT}" y="866" class="legend">Signed indicator: + upside potential / - downside potential</text>
<line x1="680" y1="861" x2="720" y2="861" stroke="#edf3ff" stroke-width="2"/>
<text x="730" y="866" class="legend">Daily causal score (unsmoothed)</text>
<line x1="865" y1="861" x2="905" y2="861" stroke="#f6c453" stroke-width="3"/>
<text x="915" y="866" class="legend">BTC close (log scale, right axis)</text>
<text x="{PLOT_LEFT}" y="88" class="legend" fill="#19c37d">UPSIDE</text>
<text x="{PLOT_LEFT}" y="{PLOT_BOTTOM - 10}" class="legend" fill="#ff4d6d">DOWNSIDE</text>
</svg>
"""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(svg, encoding="utf-8")
    os.replace(temporary, output_path)
    return len(points)


def refresh_indicator(
    api_key: str,
    *,
    full_history: bool = False,
    signal_path: Path = DEFAULT_SIGNAL_OUTPUT_PATH,
    price_path: Path = DEFAULT_BTC_PRICE_PATH,
    output_path: Path = DEFAULT_INDICATOR_CHART_PATH,
    timeout: float = 30.0,
) -> tuple[int, int]:
    price_count = update_btc_prices(
        api_key,
        price_path,
        full_history=full_history,
        timeout=timeout,
    )
    chart_points = render_svg(
        read_indicator_points(signal_path, price_path),
        output_path,
    )
    return price_count, chart_points


def render_indicator(
    signal_path: Path = DEFAULT_SIGNAL_OUTPUT_PATH,
    price_path: Path = DEFAULT_BTC_PRICE_PATH,
    output_path: Path = DEFAULT_INDICATOR_CHART_PATH,
) -> int:
    return render_svg(
        read_indicator_points(signal_path, price_path),
        output_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the historical indicator with BTC log-price overlay."
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=DEFAULT_SIGNAL_OUTPUT_PATH,
        help="Directional-volatility signal CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_BTC_PRICE_PATH,
        help="BTC daily-close cache (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDICATOR_CHART_PATH,
        help="SVG chart path (default: %(default)s)",
    )
    parser.add_argument(
        "--coinalyze-api-key",
        default=os.environ.get("COINALYZE_API_KEY", ""),
        help="Coinalyze API key (prefer COINALYZE_API_KEY)",
    )
    parser.add_argument(
        "--full-price-history",
        action="store_true",
        help="Replace the BTC price cache instead of updating recent days",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        price_count, chart_points = refresh_indicator(
            args.coinalyze_api_key,
            full_history=args.full_price_history,
            signal_path=args.signals.resolve(),
            price_path=args.prices.resolve(),
            output_path=args.output.resolve(),
            timeout=args.timeout,
        )
        print(
            f"Rendered {chart_points} indicator points to {args.output.resolve()} "
            f"using {price_count} BTC price rows"
        )
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
