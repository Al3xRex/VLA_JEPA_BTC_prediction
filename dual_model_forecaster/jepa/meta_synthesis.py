from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import numpy as np
import pandas as pd


JEPA_META_CANDIDATES = (
    "jepa_kalman_agreement_shift",
    "jepa_kalman_disagreement_damp",
    "jepa_reversion_gap_shift",
    "jepa_width_uncertainty_scale",
    "jepa_tail_flare_asymmetry",
    "jepa_combo_conservative",
)

JEPA_META_METRICS = (
    "kalman_alignment",
    "reversion_pressure",
    "vol_pressure",
    "tail_pressure",
    "uncertainty_proxy",
    "latent_norm",
    "delta_norm",
)

QUANTILE_COLUMNS = ("q05", "q25", "q50", "q75", "q95")


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class JEPAMetaParams:
    base_shift: float = 0.14
    max_center_shift_frac: float = 0.18
    width_lambda: float = 0.18
    tail_lambda: float = 0.12
    reversion_lambda: float = 0.10
    uncertainty_threshold: float = 0.70
    alignment_threshold: float = 0.12
    strict_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jepa_meta_params(env: dict[str, str] | None = None) -> JEPAMetaParams:
    source = os.environ if env is None else env
    return JEPAMetaParams(
        base_shift=float(source.get("JEPA_META_BASE_SHIFT", "0.14")),
        max_center_shift_frac=float(source.get("JEPA_META_MAX_CENTER_SHIFT_FRAC", "0.18")),
        width_lambda=float(source.get("JEPA_META_WIDTH_LAMBDA", "0.18")),
        tail_lambda=float(source.get("JEPA_META_TAIL_LAMBDA", "0.12")),
        reversion_lambda=float(source.get("JEPA_META_REVERSION_LAMBDA", "0.10")),
        uncertainty_threshold=float(source.get("JEPA_META_UNCERTAINTY_THRESHOLD", "0.70")),
        alignment_threshold=float(source.get("JEPA_META_ALIGNMENT_THRESHOLD", "0.12")),
        strict_mode=_parse_bool(source.get("JEPA_META_STRICT_MODE"), default=False),
    )


def use_live_jepa_meta(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return _parse_bool(source.get("JEPA_USE_LIVE_META"), default=False)


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _quantile_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quantiles = frame.loc[:, list(QUANTILE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return tuple(quantiles[column].to_numpy(dtype=float) for column in QUANTILE_COLUMNS)  # type: ignore[return-value]


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _ordered(q05: np.ndarray, q25: np.ndarray, q50: np.ndarray, q75: np.ndarray, q95: np.ndarray) -> np.ndarray:
    return np.sort(np.vstack([q05, q25, q50, q75, q95]).T, axis=1)


def _base_parts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    q05, q25, q50, q75, q95 = _quantile_arrays(frame)
    width_90 = np.maximum.reduce([q95 - q05, q75 - q25, np.full_like(q50, 1e-6)])
    half_50 = np.maximum((q75 - q25) / 2.0, 1e-6)
    lower_tail = np.maximum(q25 - q05, 1e-6)
    upper_tail = np.maximum(q95 - q75, 1e-6)
    return {
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "width_90": width_90,
        "half_50": half_50,
        "lower_tail": lower_tail,
        "upper_tail": upper_tail,
        "kalman": _series(frame, "kalman_projected_return", default=np.nan).to_numpy(dtype=float),
        "gap_z": _series(frame, "gap_z", default=0.0).to_numpy(dtype=float),
        "projection_pressure": _series(frame, "kalman_projection_pressure", default=0.0).to_numpy(dtype=float),
        "tail_flare": _series(frame, "tail_flare_score", default=0.0).clip(0.0, 5.0).to_numpy(dtype=float),
        "alignment": _series(frame, "aggregate_jepa_kalman_alignment", default=0.0).clip(-1.0, 1.0).to_numpy(dtype=float),
        "reversion": _series(frame, "aggregate_jepa_reversion_pressure", default=0.0).to_numpy(dtype=float),
        "uncertainty": _series(frame, "aggregate_jepa_uncertainty_proxy", default=0.0).clip(lower=0.0).to_numpy(dtype=float),
        "tail_pressure": _series(frame, "aggregate_jepa_tail_pressure", default=0.0).clip(lower=0.0).to_numpy(dtype=float),
    }


def _centered_quantiles(
    *,
    center: np.ndarray,
    parts: dict[str, np.ndarray],
    width_multiplier: np.ndarray,
    lower_tail_multiplier: np.ndarray,
    upper_tail_multiplier: np.ndarray,
) -> np.ndarray:
    half = parts["half_50"] * width_multiplier
    q25 = center - half
    q75 = center + half
    q05 = q25 - parts["lower_tail"] * lower_tail_multiplier
    q95 = q75 + parts["upper_tail"] * upper_tail_multiplier
    return _ordered(q05, q25, center, q75, q95)


def _cap_shift(shift: np.ndarray, width_90: np.ndarray, params: JEPAMetaParams) -> np.ndarray:
    cap = np.maximum(width_90 * float(params.max_center_shift_frac), 1e-6)
    return np.clip(shift, -cap, cap)


def _positive_gate(values: np.ndarray, threshold: float) -> np.ndarray:
    scale = max(1.0 - float(threshold), 1e-6)
    return np.clip((values - float(threshold)) / scale, 0.0, 1.0)


def _uncertainty_gate(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(values - float(threshold), 0.0, 3.0)


def apply_jepa_meta_candidate(
    feature_frame: pd.DataFrame,
    *,
    candidate: str,
    params: JEPAMetaParams | None = None,
) -> pd.DataFrame:
    if candidate not in JEPA_META_CANDIDATES:
        raise ValueError(f"Unsupported JEPA meta candidate {candidate!r}.")
    params = params or load_jepa_meta_params()
    out = feature_frame.copy()
    out["forecaster"] = f"meta_synth_{candidate}"
    out["meta_candidate"] = candidate

    if out.empty:
        return out

    parts = _base_parts(out)
    q50 = parts["q50"]
    center = q50.copy()
    width_multiplier = np.ones_like(q50, dtype=float)
    lower_tail_multiplier = np.ones_like(q50, dtype=float)
    upper_tail_multiplier = np.ones_like(q50, dtype=float)
    center_shift = np.zeros_like(q50, dtype=float)
    tail_extension = np.zeros_like(q50, dtype=float)

    agreement = _positive_gate(parts["alignment"], params.alignment_threshold)
    disagreement = np.clip((-parts["alignment"] - float(params.alignment_threshold)) / max(1.0 - float(params.alignment_threshold), 1e-6), 0.0, 1.0)
    uncertainty = _uncertainty_gate(parts["uncertainty"], params.uncertainty_threshold)
    kalman_delta = parts["kalman"] - q50

    if candidate == "jepa_kalman_agreement_shift":
        center_shift = _cap_shift(float(params.base_shift) * agreement * kalman_delta, parts["width_90"], params)
    elif candidate == "jepa_kalman_disagreement_damp":
        damp = np.clip(1.0 - disagreement, 0.0, 1.0)
        center_shift = _cap_shift(float(params.base_shift) * damp * kalman_delta, parts["width_90"], params)
    elif candidate == "jepa_reversion_gap_shift":
        gap_gate = np.clip((np.abs(parts["gap_z"]) - 2.0) / 2.0, 0.0, 1.0)
        reversion_strength = np.clip(np.abs(parts["reversion"]), 0.0, 1.5)
        direction = -np.sign(parts["gap_z"])
        center_shift = _cap_shift(
            float(params.reversion_lambda) * gap_gate * reversion_strength * direction * parts["width_90"],
            parts["width_90"],
            params,
        )
    elif candidate == "jepa_width_uncertainty_scale":
        width_multiplier = np.clip(1.0 + float(params.width_lambda) * uncertainty, 1.0, 2.0)
        lower_tail_multiplier = np.clip(1.0 + 0.50 * float(params.width_lambda) * uncertainty, 1.0, 2.0)
        upper_tail_multiplier = lower_tail_multiplier.copy()
    elif candidate == "jepa_tail_flare_asymmetry":
        tail_gate = np.clip(parts["tail_pressure"], 0.0, 2.0) * np.clip(parts["tail_flare"] / 2.0, 0.0, 2.0)
        tail_extension = float(params.tail_lambda) * tail_gate
        positive_pressure = parts["projection_pressure"] > 0.0
        negative_pressure = parts["projection_pressure"] < 0.0
        upper_tail_multiplier = np.where(positive_pressure, np.clip(1.0 + tail_extension, 1.0, 2.5), 1.0)
        lower_tail_multiplier = np.where(negative_pressure, np.clip(1.0 + tail_extension, 1.0, 2.5), 1.0)
    elif candidate == "jepa_combo_conservative":
        center_shift = _cap_shift(0.75 * float(params.base_shift) * agreement * kalman_delta, parts["width_90"], params)
        width_multiplier = np.clip(1.0 + 0.75 * float(params.width_lambda) * uncertainty, 1.0, 1.75)
        tail_gate = np.clip(parts["tail_pressure"], 0.0, 2.0) * np.clip(parts["tail_flare"] / 2.0, 0.0, 2.0) * agreement
        tail_extension = 0.75 * float(params.tail_lambda) * tail_gate
        positive_pressure = parts["projection_pressure"] > 0.0
        negative_pressure = parts["projection_pressure"] < 0.0
        upper_tail_multiplier = np.where(positive_pressure, np.clip(1.0 + tail_extension, 1.0, 2.0), 1.0)
        lower_tail_multiplier = np.where(negative_pressure, np.clip(1.0 + tail_extension, 1.0, 2.0), 1.0)

    center = q50 + center_shift
    ordered = _centered_quantiles(
        center=center,
        parts=parts,
        width_multiplier=width_multiplier,
        lower_tail_multiplier=lower_tail_multiplier,
        upper_tail_multiplier=upper_tail_multiplier,
    )
    valid = _finite_mask(
        parts["q05"],
        parts["q25"],
        parts["q50"],
        parts["q75"],
        parts["q95"],
        parts["kalman"],
        center,
    )
    if "aggregate_jepa_feature_available" in out.columns:
        valid &= out["aggregate_jepa_feature_available"].fillna(False).astype(bool).to_numpy()
    for idx, column in enumerate(QUANTILE_COLUMNS):
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float, copy=True)
        values[valid] = ordered[valid, idx]
        out[column] = values

    out["meta_effective_kalman_weight"] = np.where(
        valid,
        np.clip(np.abs(center_shift) / np.maximum(np.abs(kalman_delta), 1e-6), 0.0, 1.0),
        np.nan,
    )
    out["meta_center_shift"] = np.where(valid, center_shift, np.nan)
    out["meta_width_multiplier"] = np.where(valid, width_multiplier, np.nan)
    out["jepa_meta_tail_extension"] = np.where(valid, tail_extension, np.nan)
    out["meta_selected_candidate"] = candidate
    return out.replace([np.inf, -np.inf], np.nan)


def validate_jepa_quantile_order(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    values = frame.loc[:, list(QUANTILE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return bool(
        (
            values["q05"].le(values["q25"])
            & values["q25"].le(values["q50"])
            & values["q50"].le(values["q75"])
            & values["q75"].le(values["q95"])
        ).all()
    )
