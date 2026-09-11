from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dual_model_forecaster.jepa.models import SpecialistJEPA


def checkpoint_path(checkpoint_dir: str | Path, specialist_name: str) -> Path:
    return Path(checkpoint_dir) / f"{specialist_name}_jepa.pt"


def save_checkpoint(
    path: str | Path,
    model: SpecialistJEPA,
    *,
    config: dict[str, Any],
    feature_columns: list[str],
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
        },
        "config": config,
        "feature_columns": feature_columns,
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def load_model_from_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[SpecialistJEPA, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    model_cfg = dict(payload.get("model_config", {}))
    if "input_dim" not in model_cfg:
        feature_columns = payload.get("feature_columns", [])
        model_cfg["input_dim"] = len(feature_columns)
    model = SpecialistJEPA(
        input_dim=int(model_cfg["input_dim"]),
        latent_dim=int(model_cfg.get("latent_dim", payload.get("config", {}).get("latent_dim", 64))),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
