from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import statistics
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "liquidations_daily.csv"
DEFAULT_SIGNAL_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "directional_volatility_potential.csv"
)
DEFAULT_BTC_PRICE_PATH = PROJECT_ROOT / "data" / "btc_price_daily.csv"
DEFAULT_INDICATOR_CHART_PATH = (
    PROJECT_ROOT / "data" / "directional_volatility_indicator.svg"
)

COINALYZE_API_BASE = "https://api.coinalyze.net/v1"
COINALYZE_API_DOCS = "https://api.coinalyze.net/v1/doc/"
# The public charts host serves JavaScript iframe wrappers rather than the
# Plotly payloads consumed by this scraper.
CHECKONCHAIN_OI_URL = (
    "https://charts-cdn.checkonchain.com/btconchain/derivatives/"
    "derivatives_futuresoi_1daychange/"
    "derivatives_futuresoi_1daychange_light.html"
)
CHECKONCHAIN_LIQUIDATION_URL = (
    "https://charts-cdn.checkonchain.com/btconchain/derivatives/"
    "derivatives_btc_longliqdominance/"
    "derivatives_btc_longliqdominance_light.html"
)

# These contracts are the defaults in Coinalyze's BTC aggregated funding chart.
# Hyperliquid and Kraken publish hourly rates, so Coinalyze multiplies them by
# eight when displaying the chart's normalized 8-hour rate.
COINALYZE_FUNDING_CONTRACTS = {
    "BTCUSD_PERP.A": 1.0,
    "BTCUSDT_PERP.A": 1.0,
    "BTCUSD_PERP.0": 1.0,
    "BTCUSDT_PERP.0": 1.0,
    "BTCUSD.6": 1.0,
    "BTCUSDT.6": 1.0,
    "BTC-PERPETUAL.2": 1.0,
    "BTCUSD_PERP.4": 1.0,
    "BTCUSDT_PERP.4": 1.0,
    "BTC.H": 8.0,
    "pf_xbtusd.K": 8.0,
    "BTCUSD_PERP.3": 1.0,
    "BTCUSDT_PERP.3": 1.0,
}
COINALYZE_PREDICTED_CONTRACTS = {
    symbol: multiplier
    for symbol, multiplier in COINALYZE_FUNDING_CONTRACTS.items()
    if symbol != "BTC-PERPETUAL.2"
}

CSV_FIELDS = [
    "date",
    "aggregated_funding_rate_avg_pct",
    "aggregated_predicted_funding_rate_avg_pct",
    "futures_oi_7day_change_pct",
    "liquidation_dominance_pct",
    "funding_contract_count",
    "predicted_funding_contract_count",
]

RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
USER_AGENT = "liquidations-daily-updater/0.1 (+local research script)"


class ScrapeError(RuntimeError):
    pass


class ConfigurationError(ScrapeError):
    pass


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _iso_utc(timestamp: int | float) -> str:
    # Coinalyze's current-metric endpoints return update timestamps in
    # milliseconds, while some historical endpoints use Unix seconds.
    if abs(timestamp) >= 100_000_000_000:
        timestamp /= 1000
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _read_response(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    attempts: int = 3,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    for attempt in range(attempts):
        request = Request(url)
        # Request(..., headers=...) title-cases names. Coinalyze incorrectly
        # requires its documented api_key header to remain exactly lowercase.
        for name, value in request_headers.items():
            request.headers[name] = value
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code not in RETRYABLE_HTTP_STATUSES or attempt == attempts - 1:
                raise ScrapeError(
                    f"HTTP {exc.code} while requesting {url}: {body}"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
        except URLError as exc:
            if attempt == attempts - 1:
                raise ScrapeError(f"Could not request {url}: {exc.reason}") from exc
            delay = 2**attempt

        time.sleep(min(delay, 30.0))

    raise AssertionError("unreachable")


def fetch_text(url: str, *, timeout: float = 30.0) -> str:
    return _read_response(url, timeout=timeout).decode("utf-8")


def fetch_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    raw = _read_response(
        url,
        headers={"Accept": "application/json", **(headers or {})},
        timeout=timeout,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw.decode("utf-8", errors="replace")[:500]
        raise ScrapeError(f"Invalid JSON from {url}: {preview}") from exc


def extract_plotly_traces(html: str) -> list[dict[str, Any]]:
    call_start = html.find("Plotly.newPlot")
    if call_start < 0:
        raise ScrapeError("Plotly.newPlot call was not found in CheckOnChain HTML")

    data_start = html.find("[", call_start)
    if data_start < 0:
        raise ScrapeError("Plotly trace array was not found in CheckOnChain HTML")

    try:
        traces, _ = json.JSONDecoder().raw_decode(html[data_start:])
    except json.JSONDecodeError as exc:
        raise ScrapeError("Could not decode CheckOnChain Plotly traces") from exc

    if not isinstance(traces, list):
        raise ScrapeError("CheckOnChain Plotly data is not a trace list")
    return traces


def decode_plotly_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or "bdata" not in value or "dtype" not in value:
        raise ScrapeError("Unsupported Plotly array encoding")

    dtype = str(value["dtype"])
    endian = ">" if dtype.startswith(">") else "<"
    dtype = dtype.lstrip("<>=|")
    format_codes = {
        "f8": "d",
        "f4": "f",
        "i8": "q",
        "u8": "Q",
        "i4": "i",
        "u4": "I",
        "i2": "h",
        "u2": "H",
        "i1": "b",
        "u1": "B",
    }
    format_code = format_codes.get(dtype)
    if format_code is None:
        raise ScrapeError(f"Unsupported Plotly dtype: {value['dtype']}")

    try:
        raw = base64.b64decode(value["bdata"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ScrapeError("Invalid Plotly base64 data") from exc

    item_size = struct.calcsize(format_code)
    if len(raw) % item_size:
        raise ScrapeError(
            f"Plotly {dtype} array has {len(raw)} bytes, not a multiple of {item_size}"
        )

    unpacker = struct.iter_unpack(endian + format_code, raw)
    return [item[0] for item in unpacker]


def latest_trace_point(
    traces: list[dict[str, Any]], trace_name: str
) -> tuple[str, float]:
    points = trace_points_by_date(traces, trace_name)
    latest_date = max(points)
    return latest_date, points[latest_date]


def trace_points_by_date(
    traces: list[dict[str, Any]], trace_name: str
) -> dict[str, float]:
    trace = next((item for item in traces if item.get("name") == trace_name), None)
    if trace is None:
        available = ", ".join(
            str(item.get("name")) for item in traces if item.get("name")
        )
        raise ScrapeError(
            f'CheckOnChain trace "{trace_name}" was not found; available: {available}'
        )

    x_values = decode_plotly_array(trace.get("x"))
    y_values = decode_plotly_array(trace.get("y"))
    if len(x_values) != len(y_values):
        raise ScrapeError(
            f'CheckOnChain trace "{trace_name}" has mismatched x/y lengths'
        )

    points: dict[str, float] = {}
    for observed_at, value in zip(x_values, y_values, strict=True):
        if (
            isinstance(observed_at, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            try:
                observed_date = date.fromisoformat(observed_at[:10]).isoformat()
            except ValueError as exc:
                raise ScrapeError(
                    f'Invalid date in CheckOnChain trace "{trace_name}": {observed_at}'
                ) from exc
            # CheckOnChain can include the current date twice. The final point
            # is the most recently generated value, so later duplicates win.
            points[observed_date] = float(value)

    if not points:
        raise ScrapeError(
            f'CheckOnChain trace "{trace_name}" has no finite data points'
        )
    return points


def parse_checkonchain(
    oi_html: str,
    liquidation_html: str,
    *,
    through_date: date | None = None,
) -> dict[str, str | float]:
    oi_points = trace_points_by_date(
        extract_plotly_traces(oi_html), "Futures OI 1-day Change (%)"
    )
    liquidation_traces = extract_plotly_traces(liquidation_html)
    short_points = trace_points_by_date(
        liquidation_traces, "Short Liquidation Dominance"
    )
    long_points = trace_points_by_date(
        liquidation_traces, "Long Liquidation Dominance"
    )
    common_dates = set(oi_points) & set(short_points) & set(long_points)
    if through_date is not None:
        through_text = through_date.isoformat()
        common_dates = {
            observed_date
            for observed_date in common_dates
            if observed_date <= through_text
        }
    if not common_dates:
        raise ScrapeError(
            "CheckOnChain has no common OI and liquidation date"
        )
    observed_date = max(common_dates)

    return {
        "oi_date": observed_date,
        "futures_oi_7day_change_pct": oi_points[observed_date] * 100.0,
        "liquidation_date": observed_date,
        "liquidation_dominance_pct": (
            short_points[observed_date] + long_points[observed_date]
        )
        * 100.0,
    }


def fetch_checkonchain(
    *,
    timeout: float = 30.0,
    through_date: date | None = None,
) -> dict[str, str | float]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        oi_future = executor.submit(fetch_text, CHECKONCHAIN_OI_URL, timeout=timeout)
        liquidation_future = executor.submit(
            fetch_text, CHECKONCHAIN_LIQUIDATION_URL, timeout=timeout
        )
        return parse_checkonchain(
            oi_future.result(),
            liquidation_future.result(),
            through_date=through_date,
        )


def coinalyze_average(
    response: Any, contracts: dict[str, float]
) -> tuple[float, int, int | None]:
    if not isinstance(response, list):
        raise ScrapeError("Coinalyze response is not a list")

    by_symbol: dict[str, dict[str, Any]] = {}
    for item in response:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            by_symbol[item["symbol"]] = item

    values: list[float] = []
    updates: list[int] = []
    for symbol, multiplier in contracts.items():
        item = by_symbol.get(symbol)
        if item is None or item.get("value") is None:
            continue
        try:
            value = float(item["value"])
        except (TypeError, ValueError) as exc:
            raise ScrapeError(
                f"Invalid Coinalyze value for {symbol}: {item.get('value')}"
            ) from exc
        if not math.isfinite(value):
            continue
        values.append(value * multiplier)
        if isinstance(item.get("update"), (int, float)):
            updates.append(int(item["update"]))

    if not values:
        raise ScrapeError("Coinalyze returned no usable values for the chart contracts")

    return statistics.fmean(values), len(values), max(updates, default=None)


def fetch_coinalyze(
    api_key: str, *, timeout: float = 30.0
) -> dict[str, str | float | int]:
    if not api_key:
        raise ConfigurationError(
            "COINALYZE_API_KEY is required. Generate a free key at "
            f"{COINALYZE_API_DOCS}"
        )

    headers = {"api_key": api_key}

    def request_metric(path: str, contracts: dict[str, float]) -> Any:
        query = urlencode({"symbols": ",".join(contracts)})
        return fetch_json(
            f"{COINALYZE_API_BASE}/{path}?{query}",
            headers=headers,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        funding_future = executor.submit(
            request_metric, "funding-rate", COINALYZE_FUNDING_CONTRACTS
        )
        predicted_future = executor.submit(
            request_metric,
            "predicted-funding-rate",
            COINALYZE_PREDICTED_CONTRACTS,
        )
        funding_response = funding_future.result()
        predicted_response = predicted_future.result()

    funding_avg, funding_count, funding_update = coinalyze_average(
        funding_response, COINALYZE_FUNDING_CONTRACTS
    )
    predicted_avg, predicted_count, predicted_update = coinalyze_average(
        predicted_response, COINALYZE_PREDICTED_CONTRACTS
    )
    updates = [
        timestamp
        for timestamp in (funding_update, predicted_update)
        if timestamp is not None
    ]

    return {
        "aggregated_funding_rate_avg_pct": funding_avg,
        "aggregated_predicted_funding_rate_avg_pct": predicted_avg,
        "funding_contract_count": funding_count,
        "predicted_funding_contract_count": predicted_count,
        "observed_at_utc": _iso_utc(max(updates)) if updates else "",
    }


def coinalyze_daily_history_average(
    response: Any,
    contracts: dict[str, float],
    observed_date: date,
) -> tuple[float, int]:
    if not isinstance(response, list):
        raise ScrapeError("Coinalyze historical response is not a list")

    values_by_symbol: dict[str, float] = {}
    for item in response:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        if symbol not in contracts or not isinstance(item.get("history"), list):
            continue
        for candle in item["history"]:
            if not isinstance(candle, dict):
                continue
            try:
                candle_date = datetime.fromtimestamp(
                    int(candle["t"]), UTC
                ).date()
                close = float(candle["c"])
            except (KeyError, TypeError, ValueError):
                continue
            if candle_date == observed_date and math.isfinite(close):
                values_by_symbol[symbol] = close * contracts[symbol]

    if not values_by_symbol:
        raise ScrapeError(
            f"Coinalyze returned no daily values for {observed_date}"
        )
    return statistics.fmean(values_by_symbol.values()), len(values_by_symbol)


def fetch_coinalyze_daily(
    api_key: str,
    observed_date: date,
    *,
    timeout: float = 30.0,
) -> dict[str, float | int]:
    if not api_key:
        raise ConfigurationError(
            "COINALYZE_API_KEY is required. Generate a free key at "
            f"{COINALYZE_API_DOCS}"
        )

    start = int(
        datetime.combine(observed_date, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    end = int(
        datetime.combine(observed_date, datetime.max.time(), tzinfo=UTC).timestamp()
    )
    headers = {"api_key": api_key}

    def request_history(path: str, contracts: dict[str, float]) -> Any:
        query = urlencode(
            {
                "symbols": ",".join(contracts),
                "interval": "daily",
                "from": start,
                "to": end,
            }
        )
        return fetch_json(
            f"{COINALYZE_API_BASE}/{path}?{query}",
            headers=headers,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        funding_future = executor.submit(
            request_history,
            "funding-rate-history",
            COINALYZE_FUNDING_CONTRACTS,
        )
        predicted_future = executor.submit(
            request_history,
            "predicted-funding-rate-history",
            COINALYZE_PREDICTED_CONTRACTS,
        )
        funding_response = funding_future.result()
        predicted_response = predicted_future.result()

    funding_avg, funding_count = coinalyze_daily_history_average(
        funding_response,
        COINALYZE_FUNDING_CONTRACTS,
        observed_date,
    )
    predicted_avg, predicted_count = coinalyze_daily_history_average(
        predicted_response,
        COINALYZE_PREDICTED_CONTRACTS,
        observed_date,
    )
    return {
        "aggregated_funding_rate_avg_pct": funding_avg,
        "aggregated_predicted_funding_rate_avg_pct": predicted_avg,
        "funding_contract_count": funding_count,
        "predicted_funding_contract_count": predicted_count,
    }


def _validate_source_freshness(
    source_name: str,
    source_date_text: str,
    latest_completed_date: date,
    max_source_age_days: int,
) -> None:
    source_date = date.fromisoformat(source_date_text)
    age_days = (latest_completed_date - source_date).days
    if age_days < 0:
        raise ScrapeError(
            f"{source_name} returned future date {source_date}; latest completed "
            f"date is {latest_completed_date}"
        )
    if age_days > max_source_age_days:
        raise ScrapeError(
            f"{source_name} data is stale: latest date is {source_date}, "
            f"{age_days} days behind {latest_completed_date}"
        )


def build_daily_row(
    api_key: str,
    *,
    timeout: float = 30.0,
    max_source_age_days: int = 1,
    now: datetime | None = None,
) -> dict[str, str]:
    if not api_key:
        raise ConfigurationError(
            "COINALYZE_API_KEY is required. Generate a free key at "
            f"{COINALYZE_API_DOCS}"
        )

    scraped_at = now or datetime.now(UTC)
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=UTC)
    else:
        scraped_at = scraped_at.astimezone(UTC)

    run_date = scraped_at.date()
    completed_date = run_date - timedelta(days=1)
    checkonchain = fetch_checkonchain(
        timeout=timeout,
        through_date=completed_date,
    )
    oi_date = str(checkonchain["oi_date"])
    liquidation_date = str(checkonchain["liquidation_date"])
    _validate_source_freshness(
        "CheckOnChain OI", oi_date, completed_date, max_source_age_days
    )
    _validate_source_freshness(
        "CheckOnChain liquidation",
        liquidation_date,
        completed_date,
        max_source_age_days,
    )
    observed_date = date.fromisoformat(oi_date)
    coinalyze = fetch_coinalyze_daily(
        api_key,
        observed_date,
        timeout=timeout,
    )

    return {
        "date": observed_date.isoformat(),
        "aggregated_funding_rate_avg_pct": _format_number(
            float(coinalyze["aggregated_funding_rate_avg_pct"])
        ),
        "aggregated_predicted_funding_rate_avg_pct": _format_number(
            float(coinalyze["aggregated_predicted_funding_rate_avg_pct"])
        ),
        "futures_oi_7day_change_pct": _format_number(
            float(checkonchain["futures_oi_7day_change_pct"])
        ),
        "liquidation_dominance_pct": _format_number(
            float(checkonchain["liquidation_dominance_pct"])
        ),
        "funding_contract_count": str(coinalyze["funding_contract_count"]),
        "predicted_funding_contract_count": str(
            coinalyze["predicted_funding_contract_count"]
        ),
    }


def write_csv_rows(
    output_path: Path,
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str] | None = None,
) -> int:
    fieldnames = fieldnames or CSV_FIELDS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_date = {row["date"]: row for row in rows if row.get("date")}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row_date in sorted(rows_by_date):
                writer.writerow(rows_by_date[row_date])
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return len(rows_by_date)


def upsert_csv(
    output_path: Path,
    row: dict[str, str],
    *,
    drop_later_rows: bool = False,
) -> int:
    rows: list[dict[str, str]] = []
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != CSV_FIELDS:
                raise ScrapeError(
                    f"{output_path} has unexpected columns. Expected: "
                    + ", ".join(CSV_FIELDS)
                )
            rows.extend(reader)
    if drop_later_rows:
        rows = [
            existing
            for existing in rows
            if existing.get("date", "") <= row["date"]
        ]
    rows.append(row)
    return write_csv_rows(output_path, rows)


def _latest_csv_date(output_path: Path) -> date | None:
    if not output_path.exists():
        return None
    with output_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        latest: date | None = None
        for row in reader:
            value = row.get("date")
            if not value:
                continue
            try:
                row_date = date.fromisoformat(value)
            except ValueError:
                continue
            latest = row_date if latest is None or row_date > latest else latest
    return latest


def _derived_outputs_current(
    *,
    row_date: date,
    signal_output_path: Path,
    btc_price_path: Path,
    indicator_chart_path: Path,
) -> bool:
    latest_signal = _latest_csv_date(signal_output_path)
    return (
        latest_signal is not None
        and latest_signal >= row_date
        and btc_price_path.exists()
        and indicator_chart_path.exists()
    )


def sync_daily_outputs(
    api_key: str,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    signal_output_path: Path = DEFAULT_SIGNAL_OUTPUT_PATH,
    btc_price_path: Path = DEFAULT_BTC_PRICE_PATH,
    indicator_chart_path: Path = DEFAULT_INDICATOR_CHART_PATH,
    timeout: float = 30.0,
    max_source_age_days: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    row = build_daily_row(
        api_key,
        timeout=timeout,
        max_source_age_days=max_source_age_days,
    )
    row_date = date.fromisoformat(str(row["date"]))
    latest_stored = _latest_csv_date(output_path)
    if (
        latest_stored is not None
        and latest_stored == row_date
        and _derived_outputs_current(
            row_date=row_date,
            signal_output_path=signal_output_path,
            btc_price_path=btc_price_path,
            indicator_chart_path=indicator_chart_path,
        )
        and not force
    ):
        return {
            "status": "skipped",
            "reason": "no_new_data",
            "date": row_date.isoformat(),
            "latest_stored": latest_stored.isoformat(),
            "output_path": str(output_path),
            "signal_output_path": str(signal_output_path),
        }

    row_count = upsert_csv(
        output_path,
        row,
        drop_later_rows=True,
    )
    try:
        from .indicator import render_indicator, update_btc_prices
        from .signals import generate_signal_file
    except ImportError:
        from indicator import render_indicator, update_btc_prices
        from signals import generate_signal_file

    price_count = update_btc_prices(
        api_key,
        btc_price_path,
        full_history=False,
        timeout=timeout,
    )
    signal_count = generate_signal_file(
        output_path,
        signal_output_path,
        btc_price_path,
    )
    chart_points = render_indicator(
        signal_output_path,
        btc_price_path,
        indicator_chart_path,
    )
    return {
        "status": "updated",
        "date": row_date.isoformat(),
        "raw_rows": int(row_count),
        "signal_rows": int(signal_count),
        "btc_price_rows": int(price_count),
        "chart_points": int(chart_points),
        "output_path": str(output_path),
        "signal_output_path": str(signal_output_path),
        "btc_price_path": str(btc_price_path),
        "indicator_chart_path": str(indicator_chart_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the daily BTC derivatives and liquidation CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path to create or update (default: %(default)s)",
    )
    parser.add_argument(
        "--coinalyze-api-key",
        default=os.environ.get("COINALYZE_API_KEY", ""),
        help="Coinalyze API key (prefer the COINALYZE_API_KEY environment variable)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-source-age-days",
        type=int,
        default=1,
        help="Reject CheckOnChain data older than this many days (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print the row without changing the CSV",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.max_source_age_days < 0:
        parser.error("--max-source-age-days cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.dry_run:
            row = build_daily_row(
                args.coinalyze_api_key,
                timeout=args.timeout,
                max_source_age_days=args.max_source_age_days,
            )
            print(json.dumps(row, indent=2, sort_keys=True))
        else:
            output_path = args.output.resolve()
            result = sync_daily_outputs(
                args.coinalyze_api_key,
                output_path=output_path,
                timeout=args.timeout,
                max_source_age_days=args.max_source_age_days,
                force=True,
            )
            print(
                f"Updated {output_path} for {result['date']} "
                f"({result['raw_rows']} data row{'s' if result['raw_rows'] != 1 else ''}); "
                f"refreshed {result['signal_rows']} signal rows at "
                f"{result['signal_output_path']}; refreshed {result['btc_price_rows']} BTC "
                f"price rows and rendered {result['chart_points']} chart points at "
                f"{result['indicator_chart_path']}"
            )
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
