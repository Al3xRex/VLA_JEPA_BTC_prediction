from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_model_forecaster.data import ForecastDataBundle
from dual_model_forecaster.jepa.config import JEPAConfig
from dual_model_forecaster.jepa.encode import encode_live_panels, encode_panel_history
from dual_model_forecaster.jepa.feature_contract import DEFAULT_HORIZONS
from dual_model_forecaster.jepa.panels import build_all_specialist_panels, latest_specialist_panel_rows
from dual_model_forecaster.jepa.semantic_export import semantic_fields_from_embedding
from dual_model_forecaster.jepa.visuals import write_jepa_visuals


SEMANTIC_COLUMNS = {
    "jepa_norm": "jepa_norm",
    "jepa_delta_norm": "jepa_delta_norm",
    "jepa_kalman_alignment": "kalman_alignment",
    "jepa_reversion_pressure": "reversion_pressure",
    "jepa_vol_pressure": "vol_pressure",
    "jepa_tail_pressure": "tail_pressure",
    "jepa_uncertainty_proxy": "uncertainty_proxy",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    return str(value)


def _placeholder_semantics(
    panels: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for specialist, panel in panels.items():
        if panel.empty:
            continue
        latest = panel.tail(1)
        for horizon in horizons:
            semantic = semantic_fields_from_embedding(
                latest,
                specialist_name=specialist,
                horizon=int(horizon),
                latent_columns=[],
            )
            semantic = semantic.copy()
            semantic.insert(0, "specialist", specialist)
            semantic.insert(1, "horizon", int(horizon))
            semantic.insert(2, "checkpoint_found", False)
            rows.append(semantic.reset_index())
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _placeholder_semantic_history(
    panel: pd.DataFrame,
    *,
    specialist: str,
    horizon: int,
    lookback_rows: int,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    recent = panel.tail(max(int(lookback_rows), 1))
    return semantic_fields_from_embedding(
        recent,
        specialist_name=specialist,
        horizon=int(horizon),
        latent_columns=[],
    )


def _compact_semantic_history(
    encoded: pd.DataFrame,
    *,
    specialist: str,
    horizon: int,
    checkpoint_found: bool,
) -> pd.DataFrame:
    if encoded.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(encoded.index).astype(str),
            "specialist": specialist,
            "horizon": int(horizon),
            "checkpoint_found": bool(checkpoint_found),
        },
        index=encoded.index,
    )
    for semantic_token, output_name in SEMANTIC_COLUMNS.items():
        source = f"{specialist}_{semantic_token}_h{int(horizon)}"
        out[output_name] = pd.to_numeric(encoded.get(source), errors="coerce")
    return out.reset_index(drop=True)


def _build_jepa_timeseries(
    panels: dict[str, pd.DataFrame],
    *,
    config: JEPAConfig,
    horizons: tuple[int, ...],
    context_length: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    lookback_rows = max(int(config.chart_lookback_days), 1)
    batch_size = max(int(config.chart_batch_size), 1)
    for specialist, panel in panels.items():
        for horizon in horizons:
            encoded, diagnostic = encode_panel_history(
                panel,
                specialist_name=specialist,
                horizon=int(horizon),
                checkpoint_dir=config.checkpoint_dir,
                context_length=context_length,
                lookback_rows=lookback_rows,
                batch_size=batch_size,
            )
            used_checkpoint = bool(diagnostic.get("checkpoint_found")) and not encoded.empty
            if encoded.empty:
                encoded = _placeholder_semantic_history(
                    panel,
                    specialist=specialist,
                    horizon=int(horizon),
                    lookback_rows=lookback_rows,
                )
                diagnostic = {
                    **diagnostic,
                    "placeholder_history": True,
                    "history_rows": int(len(encoded)),
                }
            else:
                diagnostic = {**diagnostic, "placeholder_history": False}
            diagnostics.append(diagnostic)
            compact = _compact_semantic_history(
                encoded,
                specialist=specialist,
                horizon=int(horizon),
                checkpoint_found=used_checkpoint,
            )
            if not compact.empty:
                frames.append(compact)
    timeseries = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return timeseries.replace([np.inf, -np.inf], np.nan), diagnostics


def _write_markdown_summary(
    path: Path,
    *,
    config: JEPAConfig,
    latest: pd.DataFrame,
    diagnostics: list[dict[str, Any]],
    timeseries: pd.DataFrame,
    visuals: dict[str, str],
) -> None:
    found = sum(1 for row in diagnostics if row.get("checkpoint_found"))
    total = len(diagnostics)
    lines = [
        "# JEPA Diagnostics",
        "",
        f"- Enabled: {config.enabled}",
        f"- Mode: {config.mode}",
        f"- Use in neural: {config.use_in_neural}",
        f"- Use in meta: {config.use_in_meta}",
        f"- Checkpoint id: {config.checkpoint_id}",
        f"- Checkpoints found: {found}/{total}",
        f"- Latest rows: {len(latest)}",
        f"- Timeseries rows: {len(timeseries)}",
        "",
        "JEPA is passive unless `JEPA_USE_IN_NEURAL=true` or `JEPA_USE_IN_META=true` is explicitly enabled.",
    ]
    if visuals:
        lines.extend(
            [
                "",
                "## Visuals",
                "",
            ]
        )
        for name, visual_path in visuals.items():
            lines.append(f"- {name.replace('_', ' ').title()}: {Path(visual_path).name}")
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_live_jepa_artifacts(
    *,
    data_bundle: ForecastDataBundle,
    config: JEPAConfig,
) -> dict[str, Any]:
    if not config.is_active:
        return {"enabled": False, "mode": config.mode, "status": "disabled"}

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in (config.horizons or DEFAULT_HORIZONS))
    context_length = int(config.context_lengths[0] if config.context_lengths else 128)

    panels = build_all_specialist_panels(data_bundle, horizons=horizons)
    specialist_states = latest_specialist_panel_rows(panels)
    encoded, diagnostics = encode_live_panels(
        panels,
        horizons=horizons,
        checkpoint_dir=config.checkpoint_dir,
        context_length=context_length,
    )
    latest = encoded if not encoded.empty else _placeholder_semantics(panels, horizons)
    latest = latest.replace([np.inf, -np.inf], np.nan)

    latest_csv = artifact_dir / "jepa_latest.csv"
    latest_json = artifact_dir / "jepa_latest.json"
    summary_md = artifact_dir / "jepa_summary.md"
    states_csv = artifact_dir / "jepa_specialist_states.csv"
    alignment_csv = artifact_dir / "jepa_kalman_alignment.csv"
    timeseries_csv = artifact_dir / "jepa_timeseries.csv"

    latest.to_csv(latest_csv, index=False)
    specialist_states.to_csv(states_csv, index=False)
    alignment_columns = [
        column
        for column in latest.columns
        if "jepa_kalman_alignment" in str(column)
        or str(column) in {"timestamp", "specialist", "horizon", "checkpoint_found"}
    ]
    latest.loc[:, alignment_columns].to_csv(alignment_csv, index=False)
    timeseries, history_diagnostics = _build_jepa_timeseries(
        panels,
        config=config,
        horizons=horizons,
        context_length=context_length,
    )
    timeseries.to_csv(timeseries_csv, index=False)
    visuals = write_jepa_visuals(
        artifact_dir=artifact_dir,
        timeseries=timeseries,
        panels=panels,
        close=data_bundle.close,
    )

    payload = {
        "config": config.to_dict(),
        "diagnostics": diagnostics,
        "history_diagnostics": history_diagnostics,
        "latest": latest.where(pd.notna(latest), None).to_dict(orient="records"),
        "files": {
            "latest_csv": str(latest_csv.resolve()),
            "latest_json": str(latest_json.resolve()),
            "summary_md": str(summary_md.resolve()),
            "specialist_states_csv": str(states_csv.resolve()),
            "kalman_alignment_csv": str(alignment_csv.resolve()),
            "timeseries_csv": str(timeseries_csv.resolve()),
            "visuals": visuals,
        },
    }
    latest_json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    _write_markdown_summary(
        summary_md,
        config=config,
        latest=latest,
        diagnostics=diagnostics,
        timeseries=timeseries,
        visuals=visuals,
    )

    return {
        "enabled": True,
        "mode": config.mode,
        "artifact_dir": str(artifact_dir.resolve()),
        "latest_csv": str(latest_csv.resolve()),
        "latest_json": str(latest_json.resolve()),
        "summary_md": str(summary_md.resolve()),
        "specialist_states_csv": str(states_csv.resolve()),
        "kalman_alignment_csv": str(alignment_csv.resolve()),
        "timeseries_csv": str(timeseries_csv.resolve()),
        "visuals": visuals,
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_found_count": sum(1 for row in diagnostics if row.get("checkpoint_found")),
        "diagnostic_count": len(diagnostics),
        "history_diagnostic_count": len(history_diagnostics),
        "visual_count": len(visuals),
    }
