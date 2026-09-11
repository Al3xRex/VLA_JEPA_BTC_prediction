from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dual_model_forecaster.config import load_config
from dual_model_forecaster.data import load_forecast_data
from dual_model_forecaster.jepa.checkpoints import checkpoint_path, save_checkpoint
from dual_model_forecaster.jepa.config import JEPAConfig, load_jepa_config
from dual_model_forecaster.jepa.datasets import SpecialistJEPADataset, jepa_collate
from dual_model_forecaster.jepa.losses import total_jepa_loss
from dual_model_forecaster.jepa.models import SpecialistJEPA
from dual_model_forecaster.jepa.panels import build_all_specialist_panels
from dual_model_forecaster.utils import set_global_seed

DEFAULT_JEPA_SPECIALISTS = ("structure", "environment", "edges", "movement", "liquidation")
DEFAULT_JEPA_HORIZONS = (1, 3, 7, 15)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _collapse_diagnostics(latents: np.ndarray) -> dict[str, float]:
    if latents.size == 0:
        return {"latent_std_mean": float("nan"), "latent_std_min": float("nan"), "latent_abs_mean": float("nan")}
    std = np.std(latents, axis=0)
    return {
        "latent_std_mean": float(np.nanmean(std)),
        "latent_std_min": float(np.nanmin(std)),
        "latent_abs_mean": float(np.nanmean(np.abs(latents))),
    }


def train_specialist(
    *,
    specialist_name: str,
    panel: pd.DataFrame,
    config: JEPAConfig,
    context_length: int,
    horizons: list[int],
    device: str,
) -> dict[str, Any]:
    dataset = SpecialistJEPADataset(
        panel,
        specialist_name=specialist_name,
        horizons=tuple(horizons),
        context_length=int(context_length),
    )
    feature_columns = dataset.feature_columns
    if len(dataset) == 0:
        return {
            "specialist": specialist_name,
            "status": "skipped_empty_dataset",
            "samples": 0,
            "feature_columns": feature_columns,
        }

    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        collate_fn=jepa_collate,
    )
    model = SpecialistJEPA(input_dim=len(feature_columns), latent_dim=int(config.latent_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []

    for epoch in range(int(config.max_epochs)):
        model.train()
        losses: list[dict[str, float]] = []
        epoch_latents: list[np.ndarray] = []
        for batch in loader:
            context = batch["context_features"].to(device)
            target = batch["target_features"].to(device)
            target_mask = batch["target_mask"].to(device)
            horizon = batch["horizon"].to(device)
            optimizer.zero_grad()
            output = model(context, horizon, target_features=target, target_mask=target_mask)
            loss, metrics = total_jepa_loss(
                output["predicted_target_latent"],
                output["target_latent"],
                context_latent=output["context_latent"],
                lambda_var=float(config.lambda_var),
                lambda_cov=float(config.lambda_cov),
                lambda_smooth=float(config.lambda_smooth),
            )
            loss.backward()
            optimizer.step()
            losses.append(metrics)
            epoch_latents.append(output["predicted_target_latent"].detach().cpu().numpy())
        averaged = {
            key: float(np.mean([row[key] for row in losses]))
            for key in losses[0]
        }
        collapse = _collapse_diagnostics(np.vstack(epoch_latents) if epoch_latents else np.empty((0, config.latent_dim)))
        rows.append({"epoch": epoch, "specialist": specialist_name, **averaged, **collapse})
        embedding_rows.append({"epoch": epoch, "specialist": specialist_name, **collapse})

    checkpoint = save_checkpoint(
        checkpoint_path(config.checkpoint_dir, specialist_name),
        model,
        config=config.to_dict(),
        feature_columns=feature_columns,
        metrics=rows[-1] if rows else {},
    )
    return {
        "specialist": specialist_name,
        "status": "trained",
        "samples": len(dataset),
        "feature_columns": feature_columns,
        "checkpoint": str(checkpoint),
        "metrics": rows,
        "embedding_diagnostics": embedding_rows,
    }


def run_jepa_pretraining(
    *,
    base_config_path: str | Path = "configs/final_role_selection.json",
    specialists: list[str] | tuple[str, ...] | None = None,
    horizons: list[int] | tuple[int, ...] | None = None,
    context_length: int = 128,
    latent_dim: int = 64,
    checkpoint_dir: str | Path = "artifacts/models/jepa",
    artifact_dir: str | Path = "artifacts/final/live_forecast/jepa",
    max_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    env_config = load_jepa_config()
    selected_horizons = tuple(int(h) for h in (horizons or DEFAULT_JEPA_HORIZONS))
    selected_specialists = [str(name) for name in (specialists or DEFAULT_JEPA_SPECIALISTS)]
    config = JEPAConfig(
        enabled=True,
        mode="passive",
        use_in_neural=env_config.use_in_neural,
        use_in_meta=env_config.use_in_meta,
        checkpoint_dir=Path(checkpoint_dir),
        artifact_dir=Path(artifact_dir),
        horizons=selected_horizons,
        context_lengths=(int(context_length),),
        latent_dim=int(latent_dim),
        max_epochs=int(max_epochs if max_epochs is not None else env_config.max_epochs),
        batch_size=int(batch_size if batch_size is not None else env_config.batch_size),
        seed=int(seed if seed is not None else env_config.seed),
        lambda_var=env_config.lambda_var,
        lambda_cov=env_config.lambda_cov,
        lambda_smooth=env_config.lambda_smooth,
    )
    set_global_seed(int(config.seed))
    base_config = load_config(str(base_config_path))
    data_bundle = load_forecast_data(base_config)
    panels = build_all_specialist_panels(data_bundle, horizons=config.horizons)

    all_metrics: list[dict[str, Any]] = []
    embedding_diagnostics: list[dict[str, Any]] = []
    collapse_diagnostics: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"config": config.to_dict(), "specialists": {}}
    for specialist in selected_specialists:
        panel = panels.get(specialist, pd.DataFrame())
        result = train_specialist(
            specialist_name=specialist,
            panel=panel,
            config=config,
            context_length=int(context_length),
            horizons=[int(h) for h in config.horizons],
            device=str(device),
        )
        manifest["specialists"][specialist] = {
            "status": result["status"],
            "samples": result.get("samples", 0),
            "checkpoint": result.get("checkpoint"),
            "feature_columns": result.get("feature_columns", []),
        }
        all_metrics.extend(result.get("metrics", []))
        embedding_diagnostics.extend(result.get("embedding_diagnostics", []))
        if result.get("metrics"):
            collapse_diagnostics.append(result["metrics"][-1])

    artifact_dir_path = Path(artifact_dir)
    artifact_dir_path.mkdir(parents=True, exist_ok=True)
    config_path = artifact_dir_path / "jepa_pretrain_config.json"
    manifest_path = artifact_dir_path / "jepa_feature_column_manifest.json"
    metrics_path = artifact_dir_path / "jepa_training_metrics.csv"
    embedding_path = artifact_dir_path / "jepa_embedding_diagnostics.csv"
    collapse_path = artifact_dir_path / "jepa_collapse_diagnostics.csv"

    _write_json(config_path, config.to_dict())
    _write_json(manifest_path, manifest)
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(embedding_diagnostics).to_csv(embedding_path, index=False)
    pd.DataFrame(collapse_diagnostics).to_csv(collapse_path, index=False)

    trained = [
        name
        for name, payload in manifest["specialists"].items()
        if payload.get("status") == "trained"
    ]
    skipped = [
        name
        for name, payload in manifest["specialists"].items()
        if payload.get("status") != "trained"
    ]
    return {
        "status": "ok",
        "base_config": str(base_config_path),
        "artifact_dir": str(artifact_dir_path.resolve()),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "config": config.to_dict(),
        "manifest": manifest,
        "trained_specialists": trained,
        "skipped_specialists": skipped,
        "metric_rows": int(len(all_metrics)),
        "files": {
            "config": str(config_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "training_metrics": str(metrics_path.resolve()),
            "embedding_diagnostics": str(embedding_path.resolve()),
            "collapse_diagnostics": str(collapse_path.resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain optional specialist JEPA encoders.")
    parser.add_argument("--base-config", default="configs/final_role_selection.json")
    parser.add_argument("--specialists", nargs="+", default=list(DEFAULT_JEPA_SPECIALISTS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 7, 15])
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default="artifacts/models/jepa")
    parser.add_argument("--artifact-dir", default="artifacts/final/live_forecast/jepa")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_jepa_pretraining(
        base_config_path=args.base_config,
        specialists=args.specialists,
        horizons=args.horizons,
        context_length=int(args.context_length),
        latent_dim=int(args.latent_dim),
        checkpoint_dir=args.checkpoint_dir,
        artifact_dir=args.artifact_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
