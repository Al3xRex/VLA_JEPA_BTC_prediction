from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dual_model_forecaster.jepa.checkpoints import checkpoint_path, load_model_from_checkpoint
from dual_model_forecaster.jepa.datasets import _prepare_numeric_panel, live_context_array
from dual_model_forecaster.jepa.semantic_export import semantic_fields_from_embedding


def encode_live_panel(
    panel: pd.DataFrame,
    *,
    specialist_name: str,
    horizon: int,
    checkpoint_dir: str | Path,
    context_length: int,
    map_location: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = checkpoint_path(checkpoint_dir, specialist_name)
    if not path.exists():
        return pd.DataFrame(), {
            "specialist": specialist_name,
            "horizon": int(horizon),
            "checkpoint": str(path),
            "checkpoint_found": False,
        }
    model, payload = load_model_from_checkpoint(path, map_location=map_location)
    feature_columns = list(payload.get("feature_columns", []))
    context, as_of = live_context_array(panel, context_length=context_length, feature_columns=feature_columns)
    with torch.no_grad():
        output = model(
            torch.tensor(context, dtype=torch.float32, device=map_location),
            torch.tensor([int(horizon)], dtype=torch.long, device=map_location),
        )
    latent = output["predicted_target_latent"].detach().cpu().numpy()[0]
    row = pd.DataFrame(index=pd.Index([as_of], name="timestamp"))
    latent_columns = [f"{specialist_name}_jepa_z_h{int(horizon)}_{idx}" for idx in range(len(latent))]
    for column, value in zip(latent_columns, latent):
        row[column] = float(value)
    semantic = semantic_fields_from_embedding(
        pd.concat([panel.tail(1), row], axis=1),
        specialist_name=specialist_name,
        horizon=int(horizon),
        latent_columns=latent_columns,
    )
    encoded = pd.concat([row, semantic], axis=1)
    return encoded, {
        "specialist": specialist_name,
        "horizon": int(horizon),
        "checkpoint": str(path),
        "checkpoint_found": True,
        "feature_count": len(feature_columns),
        "latent_dim": int(len(latent)),
        "as_of": as_of.isoformat(),
    }


def encode_live_panels(
    panels: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] | list[int],
    checkpoint_dir: str | Path,
    context_length: int,
    map_location: str = "cpu",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for specialist, panel in panels.items():
        for horizon in horizons:
            encoded, diagnostic = encode_live_panel(
                panel,
                specialist_name=specialist,
                horizon=int(horizon),
                checkpoint_dir=checkpoint_dir,
                context_length=context_length,
                map_location=map_location,
            )
            diagnostics.append(diagnostic)
            if not encoded.empty:
                encoded = encoded.copy()
                encoded.insert(0, "specialist", specialist)
                encoded.insert(1, "horizon", int(horizon))
                frames.append(encoded.reset_index())
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined.replace([np.inf, -np.inf], np.nan), diagnostics


def encode_panel_history(
    panel: pd.DataFrame,
    *,
    specialist_name: str,
    horizon: int,
    checkpoint_dir: str | Path,
    context_length: int,
    lookback_rows: int = 365,
    batch_size: int = 128,
    map_location: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = checkpoint_path(checkpoint_dir, specialist_name)
    if not path.exists():
        return pd.DataFrame(), {
            "specialist": specialist_name,
            "horizon": int(horizon),
            "checkpoint": str(path),
            "checkpoint_found": False,
            "history_rows": 0,
        }

    model, payload = load_model_from_checkpoint(path, map_location=map_location)
    feature_columns = list(payload.get("feature_columns", []))
    numeric, _ = _prepare_numeric_panel(panel, feature_columns=feature_columns)
    context_length = int(context_length)
    if len(numeric) < context_length:
        return pd.DataFrame(), {
            "specialist": specialist_name,
            "horizon": int(horizon),
            "checkpoint": str(path),
            "checkpoint_found": True,
            "feature_count": len(feature_columns),
            "latent_dim": int(getattr(model, "latent_dim", 0)),
            "history_rows": 0,
            "status": f"insufficient_rows:{len(numeric)}<{context_length}",
        }

    positions = np.arange(context_length - 1, len(numeric), dtype=int)
    if int(lookback_rows) > 0:
        positions = positions[-int(lookback_rows) :]
    if len(positions) == 0:
        return pd.DataFrame(), {
            "specialist": specialist_name,
            "horizon": int(horizon),
            "checkpoint": str(path),
            "checkpoint_found": True,
            "feature_count": len(feature_columns),
            "latent_dim": int(getattr(model, "latent_dim", 0)),
            "history_rows": 0,
        }

    predictions: list[np.ndarray] = []
    batch_size = max(int(batch_size), 1)
    values = numeric.to_numpy(dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            context = np.stack(
                [values[pos - context_length + 1 : pos + 1] for pos in batch_positions],
                axis=0,
            )
            output = model(
                torch.tensor(context, dtype=torch.float32, device=map_location),
                torch.full((len(batch_positions),), int(horizon), dtype=torch.long, device=map_location),
            )
            predictions.append(output["predicted_target_latent"].detach().cpu().numpy())

    latent = np.concatenate(predictions, axis=0)
    index = pd.DatetimeIndex(numeric.index[positions], name="timestamp")
    latent_columns = [f"{specialist_name}_jepa_z_h{int(horizon)}_{idx}" for idx in range(latent.shape[1])]
    row = pd.DataFrame(latent, index=index, columns=latent_columns)
    semantic = semantic_fields_from_embedding(
        pd.concat([panel.reindex(index), row], axis=1),
        specialist_name=specialist_name,
        horizon=int(horizon),
        latent_columns=latent_columns,
    )
    encoded = pd.concat([row, semantic], axis=1)
    return encoded.replace([np.inf, -np.inf], np.nan), {
        "specialist": specialist_name,
        "horizon": int(horizon),
        "checkpoint": str(path),
        "checkpoint_found": True,
        "feature_count": len(feature_columns),
        "latent_dim": int(latent.shape[1]),
        "history_rows": int(len(encoded)),
        "history_start": index[0].isoformat(),
        "history_end": index[-1].isoformat(),
    }
