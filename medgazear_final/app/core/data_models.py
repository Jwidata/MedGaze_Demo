"""Lightweight data models used by placeholder pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineResult:
    """Result metadata emitted by a placeholder pipeline stage."""

    step_name: str
    output_dir: Path
    message: str
