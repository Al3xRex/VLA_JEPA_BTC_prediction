from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(np.tanh(values.to_numpy(dtype=float) / max(float(scale), 1e-9)), index=values.index)


def semantic_fields_from_embedding(
    frame: pd.DataFrame,
    *,
    specialist_name: str,
    horizon: int,
    latent_columns: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    horizon = int(horizon)
    if latent_columns:
        latent = frame[latent_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        norm = np.sqrt(np.square(latent.to_numpy(dtype=float)).sum(axis=1))
        out[f"{specialist_name}_jepa_norm_h{horizon}"] = norm
        out[f"{specialist_name}_jepa_delta_norm_h{horizon}"] = pd.Series(norm, index=frame.index).diff().abs().fillna(0.0)
        latent_direction = pd.Series(latent.iloc[:, 0].to_numpy(dtype=float), index=frame.index)
        latent_tail = pd.Series(latent.abs().mean(axis=1).to_numpy(dtype=float), index=frame.index)
    else:
        out[f"{specialist_name}_jepa_norm_h{horizon}"] = 0.0
        out[f"{specialist_name}_jepa_delta_norm_h{horizon}"] = 0.0
        latent_direction = pd.Series(0.0, index=frame.index)
        latent_tail = pd.Series(0.0, index=frame.index)

    gap_z = pd.to_numeric(frame.get("kalman__gap_z"), errors="coerce").fillna(0.0)
    projection_pressure = pd.to_numeric(frame.get("kalman__kalman_projection_pressure"), errors="coerce").fillna(0.0)
    tail_flare = pd.to_numeric(frame.get("kalman__tail_flare_score"), errors="coerce").fillna(0.0)
    residual_sigma = pd.to_numeric(frame.get("kalman__residual_sigma"), errors="coerce").fillna(0.0)
    high_vol = pd.to_numeric(frame.get("kalman__kalman_high_vol_signal"), errors="coerce").fillna(0.0)

    implied_pressure = _safe_tanh(latent_direction, scale=1.0)
    kalman_direction = np.sign(projection_pressure.fillna(0.0))
    out[f"{specialist_name}_jepa_kalman_alignment_h{horizon}"] = (
        implied_pressure * pd.Series(kalman_direction, index=frame.index)
    ).clip(-1.0, 1.0)
    # Conservative placeholder: pressure toward fair-value reversion when the filtered gap is extreme.
    out[f"{specialist_name}_jepa_reversion_pressure_h{horizon}"] = (
        -_safe_tanh(gap_z, scale=2.0) * (1.0 + 0.25 * implied_pressure.abs())
    ).clip(-1.5, 1.5)
    out[f"{specialist_name}_jepa_vol_pressure_h{horizon}"] = (
        _safe_tanh(residual_sigma, scale=max(float(residual_sigma.quantile(0.90) or 1.0), 1e-6))
        + high_vol.clip(0.0, 1.0)
    ).clip(0.0, 2.0)
    out[f"{specialist_name}_jepa_tail_pressure_h{horizon}"] = (
        _safe_tanh(latent_tail, scale=1.0).clip(0.0, 1.0)
        * (1.0 + tail_flare.clip(0.0, 5.0) / 5.0)
    ).clip(0.0, 2.0)
    out[f"{specialist_name}_jepa_uncertainty_proxy_h{horizon}"] = (
        out[f"{specialist_name}_jepa_delta_norm_h{horizon}"]
        + high_vol.clip(0.0, 1.0)
        + tail_flare.clip(0.0, 5.0) / 5.0
    ).clip(0.0, 3.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
