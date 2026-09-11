from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_WINDOWS: dict[int, tuple[int, int]] = {
    1: (1, 2),
    3: (2, 4),
    7: (5, 9),
    15: (10, 20),
}


@dataclass(frozen=True)
class TargetWindow:
    horizon: int
    start_offset_days: int
    end_offset_days: int

    @property
    def length_days(self) -> int:
        return int(self.end_offset_days - self.start_offset_days + 1)


def target_window_for_horizon(horizon: int) -> TargetWindow:
    if int(horizon) not in TARGET_WINDOWS:
        raise ValueError(f"Unsupported JEPA horizon {horizon}; expected one of {sorted(TARGET_WINDOWS)}.")
    start, end = TARGET_WINDOWS[int(horizon)]
    return TargetWindow(horizon=int(horizon), start_offset_days=start, end_offset_days=end)


def target_bounds(as_of: pd.Timestamp, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    window = target_window_for_horizon(int(horizon))
    timestamp = pd.Timestamp(as_of)
    return (
        timestamp + pd.Timedelta(days=window.start_offset_days),
        timestamp + pd.Timedelta(days=window.end_offset_days),
    )
