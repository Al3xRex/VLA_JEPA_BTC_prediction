from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from env_loader import load_project_env
from liquidations.main import (
    DEFAULT_BTC_PRICE_PATH,
    DEFAULT_INDICATOR_CHART_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SIGNAL_OUTPUT_PATH,
    sync_daily_outputs,
)

load_project_env()


def sync_liquidation_signals(
    *,
    api_key: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    signal_output_path: str | Path = DEFAULT_SIGNAL_OUTPUT_PATH,
    btc_price_path: str | Path = DEFAULT_BTC_PRICE_PATH,
    indicator_chart_path: str | Path = DEFAULT_INDICATOR_CHART_PATH,
    timeout: float = 30.0,
    max_source_age_days: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    resolved_key = (api_key if api_key is not None else os.getenv("COINALYZE_API_KEY", "")).strip()
    if not resolved_key:
        return {
            "status": "skipped",
            "reason": "missing_coinalyze_api_key",
            "output_path": str(output_path),
            "signal_output_path": str(signal_output_path),
        }

    return sync_daily_outputs(
        resolved_key,
        output_path=Path(output_path),
        signal_output_path=Path(signal_output_path),
        btc_price_path=Path(btc_price_path),
        indicator_chart_path=Path(indicator_chart_path),
        timeout=timeout,
        max_source_age_days=max_source_age_days,
        force=force,
    )
