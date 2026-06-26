"""Configuration loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.paths import DEFAULT_DATA_ROOT


@dataclass(frozen=True)
class AppConfig:
    """Small project configuration container."""

    project_name: str = "medgazear_final"
    data_root: Path | None = DEFAULT_DATA_ROOT
    parameters: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load JSON configuration if provided, otherwise return defaults."""

    if config_path is None:
        return AppConfig()

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    data_root = raw.get("data_root")
    return AppConfig(
        project_name=raw.get("project_name", "medgazear_final"),
        data_root=Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT,
        parameters=raw.get("parameters", {}),
    )
