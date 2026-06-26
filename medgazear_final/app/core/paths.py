"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def resolve_output_root(output_root: str | Path | None = None) -> Path:
    """Resolve and create the output root directory."""

    root = Path(output_root).expanduser() if output_root else DEFAULT_OUTPUT_ROOT
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_step_output_dir(output_root: str | Path | None, step_name: str) -> Path:
    """Create a stable output directory for a pipeline step."""

    step_dir = resolve_output_root(output_root) / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir
