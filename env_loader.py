from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def load_project_env(*, override: bool = False) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    loaded: dict[str, str] = {}
    for path in (
        root / ".env",
        root / "liquidations" / ".env",
        root / "liquidations" / "Liquedation heatmap" / ".env",
    ):
        loaded.update(load_env_file(path, override=override))
    return loaded
