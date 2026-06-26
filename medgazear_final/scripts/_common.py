"""Shared CLI helpers for placeholder pipeline scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import load_config
from app.core.data_models import PipelineResult
from app.core.logging_utils import configure_logging
from app.core.paths import create_step_output_dir


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory where step output folders are created. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level. Defaults to INFO.",
    )
    return parser


def run_placeholder_step(step_name: str, description: str, argv: list[str] | None = None) -> PipelineResult:
    parser = build_parser(description)
    args = parser.parse_args(argv)
    logger = configure_logging(args.log_level)
    config = load_config(args.config)
    output_dir = create_step_output_dir(args.output_root, step_name)

    manifest = {
        "step_name": step_name,
        "project_name": config.project_name,
        "data_root": str(config.data_root) if config.data_root else None,
        "status": "placeholder_complete",
        "note": "Foundation placeholder only; no clinical processing performed.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logger.info("Created placeholder output: %s", manifest_path)
    return PipelineResult(step_name=step_name, output_dir=output_dir, message="placeholder_complete")
