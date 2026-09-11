from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from dual_model_forecaster.jepa.feature_contract import (
    looks_centered_or_full_sample,
    looks_future_aware,
)


class TemporalSafetyError(ValueError):
    """Raised when a JEPA sample or feature panel violates causal ordering."""


@dataclass(frozen=True)
class SampleWindow:
    context_start: pd.Timestamp
    context_end: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    as_of: pd.Timestamp


def assert_context_before_target(window: SampleWindow) -> None:
    if window.context_end > window.as_of:
        raise TemporalSafetyError(
            f"Context end {window.context_end} is after as-of timestamp {window.as_of}."
        )
    if window.target_start <= window.context_end:
        raise TemporalSafetyError(
            f"Target start {window.target_start} must be strictly after context end {window.context_end}."
        )
    if window.target_end < window.target_start:
        raise TemporalSafetyError(
            f"Target end {window.target_end} is before target start {window.target_start}."
        )


def assert_no_future_columns(columns: Iterable[str]) -> None:
    bad = [column for column in columns if looks_future_aware(str(column))]
    if bad:
        raise TemporalSafetyError(f"Future-aware columns are not allowed in JEPA context: {bad[:8]}")


def assert_no_centered_or_full_sample_columns(columns: Iterable[str]) -> None:
    bad = [column for column in columns if looks_centered_or_full_sample(str(column))]
    if bad:
        raise TemporalSafetyError(
            f"Centered or full-sample normalized columns are not allowed in JEPA context: {bad[:8]}"
        )


def assert_kalman_fields_live_compatible(columns: Iterable[str]) -> None:
    bad_tokens = ("smoothed_future", "two_sided", "centered", "lookahead")
    bad = [
        column
        for column in columns
        if str(column).startswith("kalman__") and any(token in str(column).lower() for token in bad_tokens)
    ]
    if bad:
        raise TemporalSafetyError(f"Kalman context fields must be live-compatible; rejected: {bad[:8]}")


def validate_context_columns(columns: Iterable[str]) -> None:
    assert_no_future_columns(columns)
    assert_no_centered_or_full_sample_columns(columns)
    assert_kalman_fields_live_compatible(columns)


def validate_sample_window(
    *,
    context_start: pd.Timestamp,
    context_end: pd.Timestamp,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> SampleWindow:
    window = SampleWindow(
        context_start=pd.Timestamp(context_start),
        context_end=pd.Timestamp(context_end),
        target_start=pd.Timestamp(target_start),
        target_end=pd.Timestamp(target_end),
        as_of=pd.Timestamp(as_of),
    )
    assert_context_before_target(window)
    return window
