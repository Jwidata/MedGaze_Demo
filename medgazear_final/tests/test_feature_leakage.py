from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from app.features.feature_schema import FORBIDDEN_LABEL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_behavior_feature_table_excludes_rule_labels(tmp_path: Path) -> None:
    gaze = tmp_path / "gaze.csv"
    roi = tmp_path / "roi.csv"
    output_root = tmp_path / "outputs"
    _write_csv(gaze, [_gaze_row(0), _gaze_row(16.667)])
    _write_csv(roi, [_roi_row()])

    result = subprocess.run(
        [sys.executable, "scripts/04_extract_roi_scanpath_features.py", "--gaze", str(gaze), "--roi-geometry", str(roi), "--output-root", str(output_root)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0
    table = output_root / "features" / "behavior_feature_table.csv"
    assert table.exists()
    with table.open("r", newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    assert not FORBIDDEN_LABEL_COLUMNS.intersection(columns)
    assert "hidden_behavior_label" in columns


def _gaze_row(timestamp: float) -> dict[str, str]:
    return {
        "session_id": "S1",
        "reader_id": "R1",
        "reader_profile": "expert_systematic",
        "case_id": "C1",
        "roi_id": "ROI1",
        "slice_index": "1",
        "hidden_behavior_label": "focused_roi_confirmation",
        "timestamp_ms": str(timestamp),
        "sample_index": str(int(timestamp)),
        "gaze_x_norm": "0.5",
        "gaze_y_norm": "0.5",
        "screen_x": "960",
        "screen_y": "540",
        "image_x": "15",
        "image_y": "15",
        "is_valid": "True",
        "is_dropout": "False",
        "is_blink": "False",
        "is_invalid_burst": "False",
        "is_outside_ct": "False",
        "is_ui_glance": "False",
        "calibration_offset_x": "0",
        "calibration_offset_y": "0",
        "drift_x": "0",
        "drift_y": "0",
        "jitter_x": "1",
        "jitter_y": "1",
    }


def _roi_row() -> dict[str, str]:
    return {
        "roi_id": "ROI1",
        "seg_sop_instance_uid": "SEG1",
        "ct_sop_instance_uid": "CT1",
        "patient_id": "C1",
        "study_instance_uid": "STUDY1",
        "ct_series_instance_uid": "SERIES1",
        "seg_series_instance_uid": "SEGSERIES1",
        "slice_index": "1",
        "rows": "512",
        "columns": "512",
        "mask_area_px": "100",
        "bbox_x_min": "10",
        "bbox_y_min": "10",
        "bbox_x_max": "20",
        "bbox_y_max": "20",
        "bbox_width": "11",
        "bbox_height": "11",
        "centroid_x": "15",
        "centroid_y": "15",
        "is_empty": "false",
        "rejection_reason": "",
        "mask_npz_path": "mask.npz",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
