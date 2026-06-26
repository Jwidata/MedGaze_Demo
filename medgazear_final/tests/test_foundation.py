from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_import_app() -> None:
    assert importlib.import_module("app") is not None


def test_scan_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/01_scan_dicom_dataset.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_review_workstation_smoke_test(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/20_launch_review_workstation.py",
            "--smoke-test",
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (tmp_path / "20_launch_review_workstation" / "manifest.json").exists()
