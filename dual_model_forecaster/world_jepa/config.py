from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


WORLD_NAMES = ("structure", "environment", "edges", "movement", "liquidation")


@dataclass(frozen=True)
class WorldSpec:
    context_length: int
    target_block_length: int
    feature_budget: int
    regime_count: int


@dataclass(frozen=True)
class WorldJEPAConfig:
    path: Path
    version: int
    enabled: bool
    mode: str
    seed: int
    base_config: Path
    artifact_root: Path
    report_root: Path
    feature_manifest: Path
    worlds: dict[str, WorldSpec]
    horizons: dict[str, Any]
    encoder: dict[str, Any]
    router: dict[str, Any]
    objective: dict[str, Any]
    splits: dict[str, Any]
    training: dict[str, Any]
    promotion_gates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("path", "base_config", "artifact_root", "report_root", "feature_manifest"):
            payload[key] = str(payload[key])
        return payload


def _positive_int(payload: dict[str, Any], key: str, *, section: str) -> int:
    value = int(payload[key])
    if value <= 0:
        raise ValueError(f"{section}.{key} must be positive, got {value}.")
    return value


def _validate_horizons(horizons: dict[str, Any]) -> dict[str, Any]:
    out = dict(horizons)
    for key in ("world_target_days", "training_days", "live_days", "evaluation_days"):
        values = [float(value) for value in out.get(key, [])]
        if not values or any(value <= 0.0 for value in values):
            raise ValueError(f"horizons.{key} must contain positive values.")
        if values != sorted(set(values)):
            raise ValueError(f"horizons.{key} must be sorted and unique.")
        out[key] = values
    maximum = float(out["maximum_days"])
    if maximum < max(
        out["world_target_days"]
        + out["training_days"]
        + out["live_days"]
        + out["evaluation_days"]
    ):
        raise ValueError("horizons.maximum_days must cover every configured query horizon.")
    out["maximum_days"] = maximum
    out["fourier_dimension"] = _positive_int(out, "fourier_dimension", section="horizons")
    return out


def load_world_jepa_config(path: str | Path = "configs/world_jepa.json") -> WorldJEPAConfig:
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text())
    mode = str(payload.get("mode", "shadow")).lower()
    if mode not in {"off", "shadow", "primary"}:
        raise ValueError(f"Unsupported world JEPA mode: {mode!r}.")

    raw_worlds = dict(payload.get("worlds") or {})
    missing = [world for world in WORLD_NAMES if world not in raw_worlds]
    extra = [world for world in raw_worlds if world not in WORLD_NAMES]
    if missing or extra:
        raise ValueError(f"World config mismatch; missing={missing}, extra={extra}.")
    worlds = {
        name: WorldSpec(
            context_length=_positive_int(raw_worlds[name], "context_length", section=f"worlds.{name}"),
            target_block_length=_positive_int(
                raw_worlds[name], "target_block_length", section=f"worlds.{name}"
            ),
            feature_budget=_positive_int(raw_worlds[name], "feature_budget", section=f"worlds.{name}"),
            regime_count=_positive_int(raw_worlds[name], "regime_count", section=f"worlds.{name}"),
        )
        for name in WORLD_NAMES
    }

    splits = dict(payload["splits"])
    for key in (
        "minimum_train_rows",
        "selection_rows",
        "router_validation_rows",
        "calibration_rows",
        "test_rows",
        "purge_days",
    ):
        splits[key] = _positive_int(splits, key, section="splits")
    if int(splits["purge_days"]) < max(float(v) for v in payload["horizons"]["training_days"]):
        raise ValueError("splits.purge_days must be at least the maximum trained target horizon.")
    required_selection = (
        int(splits["router_validation_rows"])
        + int(splits["purge_days"])
        + int(splits["calibration_rows"])
    )
    if required_selection > int(splits["selection_rows"]):
        raise ValueError(
            "splits.router_validation_rows + purge_days + calibration_rows "
            "must fit inside splits.selection_rows."
        )

    router = dict(payload["router"])
    top_k = int(router["top_k_worlds"])
    if top_k < 1 or top_k > len(WORLD_NAMES):
        raise ValueError(f"router.top_k_worlds must be in [1, {len(WORLD_NAMES)}].")
    router["top_k_worlds"] = top_k
    world_dropout = float(router["world_dropout"])
    if not 0.0 <= world_dropout < 1.0:
        raise ValueError("router.world_dropout must be in [0, 1).")
    router["world_dropout"] = world_dropout
    semantic_role_routing = router.get("semantic_role_routing", False)
    if not isinstance(semantic_role_routing, bool):
        raise ValueError("router.semantic_role_routing must be a boolean.")
    router["semantic_role_routing"] = semantic_role_routing

    return WorldJEPAConfig(
        path=resolved_path,
        version=int(payload.get("version", 1)),
        enabled=bool(payload.get("enabled", True)),
        mode=mode,
        seed=int(payload.get("seed", 7)),
        base_config=Path(payload["base_config"]),
        artifact_root=Path(payload["artifact_root"]),
        report_root=Path(payload["report_root"]),
        feature_manifest=Path(payload["feature_manifest"]),
        worlds=worlds,
        horizons=_validate_horizons(dict(payload["horizons"])),
        encoder=dict(payload["encoder"]),
        router=router,
        objective=dict(payload["objective"]),
        splits=splits,
        training=dict(payload["training"]),
        promotion_gates=dict(payload.get("promotion_gates") or {}),
    )
