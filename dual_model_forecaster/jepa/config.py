from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


JEPA_MODES = {"off", "passive", "frozen", "finetune"}


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_int_list(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return default
    parts = str(value).replace(",", " ").split()
    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError:
            continue
    return tuple(parsed) if parsed else default


@dataclass(frozen=True)
class JEPAConfig:
    enabled: bool = False
    mode: str = "off"
    use_in_neural: bool = False
    use_in_meta: bool = False
    checkpoint_dir: Path = Path("artifacts/models/jepa")
    artifact_dir: Path = Path("artifacts/final/live_forecast/jepa")
    horizons: tuple[int, ...] = (1, 3, 7, 15)
    context_lengths: tuple[int, ...] = (64, 128, 256)
    latent_dim: int = 64
    max_epochs: int = 5
    batch_size: int = 128
    chart_lookback_days: int = 365
    chart_batch_size: int = 128
    seed: int = 7
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    lambda_smooth: float = 0.02

    @property
    def is_active(self) -> bool:
        return bool(self.enabled and self.mode != "off")

    @property
    def checkpoint_id(self) -> str:
        return f"mode={self.mode};latent={self.latent_dim};h={','.join(map(str, self.horizons))}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        payload["artifact_dir"] = str(self.artifact_dir)
        payload["horizons"] = list(self.horizons)
        payload["context_lengths"] = list(self.context_lengths)
        return payload


def load_jepa_config(env: dict[str, str] | None = None) -> JEPAConfig:
    source = os.environ if env is None else env
    mode = str(source.get("JEPA_MODE", "off")).strip().lower() or "off"
    if mode not in JEPA_MODES:
        raise ValueError(f"Unsupported JEPA_MODE={mode!r}; expected one of {sorted(JEPA_MODES)}.")
    enabled_default = mode != "off"
    enabled = _parse_bool(source.get("JEPA_ENABLED"), default=enabled_default)
    if not enabled:
        mode = "off"
    return JEPAConfig(
        enabled=enabled,
        mode=mode,
        use_in_neural=_parse_bool(source.get("JEPA_USE_IN_NEURAL"), default=False),
        use_in_meta=_parse_bool(source.get("JEPA_USE_IN_META"), default=False),
        checkpoint_dir=Path(source.get("JEPA_CHECKPOINT_DIR", "artifacts/models/jepa")),
        artifact_dir=Path(source.get("JEPA_ARTIFACT_DIR", "artifacts/final/live_forecast/jepa")),
        horizons=_parse_int_list(source.get("JEPA_HORIZONS"), (1, 3, 7, 15)),
        context_lengths=_parse_int_list(source.get("JEPA_CONTEXT_LENGTHS"), (64, 128, 256)),
        latent_dim=int(source.get("JEPA_LATENT_DIM", "64")),
        max_epochs=int(source.get("JEPA_MAX_EPOCHS", "5")),
        batch_size=int(source.get("JEPA_BATCH_SIZE", "128")),
        chart_lookback_days=int(source.get("JEPA_CHART_LOOKBACK_DAYS", "365")),
        chart_batch_size=int(source.get("JEPA_CHART_BATCH_SIZE", "128")),
        seed=int(source.get("JEPA_SEED", "7")),
        lambda_var=float(source.get("JEPA_LAMBDA_VAR", "1.0")),
        lambda_cov=float(source.get("JEPA_LAMBDA_COV", "0.04")),
        lambda_smooth=float(source.get("JEPA_LAMBDA_SMOOTH", "0.02")),
    )
