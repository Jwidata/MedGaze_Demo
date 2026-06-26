from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.gaze.gaze_degradation_model import DegradationConfig, apply_gaze_degradation
from app.synthetic.behavior_label_schema import HIDDEN_BEHAVIOR_LABELS, RULE_STATUS_LABELS, is_valid_hidden_behavior_label
from app.synthetic.behavior_policy import BEHAVIOR_DURATION_RANGES_MS, choose_behavior_duration_ms
from app.synthetic.synthetic_gaze_generator import generate_synthetic_gaze


def test_behavior_label_schema_validity() -> None:
    assert "expert_like_systematic_review" in HIDDEN_BEHAVIOR_LABELS
    assert all(is_valid_hidden_behavior_label(label) for label in HIDDEN_BEHAVIOR_LABELS)
    assert set(HIDDEN_BEHAVIOR_LABELS).isdisjoint(RULE_STATUS_LABELS)


def test_synthetic_generation_reproducible_with_seed(tmp_path: Path) -> None:
    roi_csv, ct_csv = _write_inputs(tmp_path)
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"

    result_a = generate_synthetic_gaze(roi_csv, ct_csv, output_a, num_sessions=4, seed=7, duration_range_ms=(200, 200))
    result_b = generate_synthetic_gaze(roi_csv, ct_csv, output_b, num_sessions=4, seed=7, duration_range_ms=(200, 200))

    assert result_a.raw_gaze_csv.read_text(encoding="utf-8") == result_b.raw_gaze_csv.read_text(encoding="utf-8")
    assert result_a.hidden_labels_csv.exists()
    assert result_a.session_table_csv.exists()
    assert result_a.quality_report_md.exists()
    assert result_a.roi_sampling_report_md.exists()


def test_gaze_degradation_clipping_and_flags() -> None:
    degraded = apply_gaze_degradation(
        screen_x=9999,
        screen_y=-9999,
        sample_index=5,
        rng=np.random.default_rng(1),
        screen_width=1920,
        screen_height=1080,
        ct_bounds=(450, 177, 1470, 903),
        calibration_offset=(0, 0),
        config=DegradationConfig(
            dropout_probability=1.0,
            blink_probability=1.0,
            invalid_burst_probability=1.0,
            outside_ct_probability=1.0,
            ui_glance_probability=1.0,
        ),
    )
    assert 0 <= degraded["screen_x"] <= 1919
    assert 0 <= degraded["screen_y"] <= 1079
    assert degraded["is_dropout"] is True
    assert degraded["is_blink"] is True
    assert degraded["is_invalid_burst"] is True
    assert degraded["is_valid"] is False


def test_hidden_labels_not_rule_statuses(tmp_path: Path) -> None:
    roi_csv, ct_csv = _write_inputs(tmp_path)
    result = generate_synthetic_gaze(roi_csv, ct_csv, tmp_path / "out", num_sessions=10, seed=9, duration_range_ms=(100, 100))
    rows = _read_csv(result.hidden_labels_csv)

    assert rows
    assert all(row["hidden_behavior_label"] in HIDDEN_BEHAVIOR_LABELS for row in rows)
    assert all(row["hidden_behavior_label"] not in RULE_STATUS_LABELS for row in rows)


def test_behavior_duration_ranges_are_overlapping_but_valid() -> None:
    rng = np.random.default_rng(3)
    labels = [
        "skipped_slice",
        "focused_roi_confirmation",
        "missed_roi_search",
        "partial_near_miss_review",
        "expert_like_systematic_review",
        "high_load_fragmented_review",
    ]
    means = {}
    for label in labels:
        durations = [choose_behavior_duration_ms(label, "partial_reviewer", rng) for _ in range(200)]
        low, high = BEHAVIOR_DURATION_RANGES_MS[label]
        assert min(durations) >= low
        assert max(durations) <= high
        means[label] = sum(durations) / len(durations)

    assert BEHAVIOR_DURATION_RANGES_MS["focused_roi_confirmation"][1] > BEHAVIOR_DURATION_RANGES_MS["missed_roi_search"][0]
    assert BEHAVIOR_DURATION_RANGES_MS["partial_near_miss_review"][1] > BEHAVIOR_DURATION_RANGES_MS["expert_like_systematic_review"][0]
    assert BEHAVIOR_DURATION_RANGES_MS["skipped_slice"][1] > BEHAVIOR_DURATION_RANGES_MS["missed_roi_search"][0]
    assert means["skipped_slice"] < means["high_load_fragmented_review"]


def test_quality_report_includes_session_label_duration_stats(tmp_path: Path) -> None:
    roi_csv, ct_csv = _write_inputs(tmp_path)
    result = generate_synthetic_gaze(roi_csv, ct_csv, tmp_path / "out", num_sessions=25, seed=11)
    report = result.quality_report_md.read_text(encoding="utf-8")

    assert "label distribution by session" in report
    assert "duration ms by hidden behavior label" in report
    assert "samples per session by hidden behavior label" in report


def test_roi_sampling_report_includes_patient_coverage(tmp_path: Path) -> None:
    roi_csv, ct_csv = _write_inputs(tmp_path, patient_count=4)
    result = generate_synthetic_gaze(roi_csv, ct_csv, tmp_path / "out", num_sessions=8, seed=13, duration_range_ms=(100, 100))
    report = result.roi_sampling_report_md.read_text(encoding="utf-8")

    assert "sampling mode: patient_balanced" in report
    assert "sampled patients: 4" in report
    assert "patient coverage percentage" in report
    assert "ROI reuse distribution" in report


def _write_inputs(tmp_path: Path, patient_count: int = 1) -> tuple[Path, Path]:
    roi_csv = tmp_path / "seg_roi_geometry.csv"
    ct_csv = tmp_path / "ct_series_summary.csv"
    roi_rows = []
    for idx in range(patient_count):
        roi_rows.append(
            {
                "roi_id": f"ROI_{idx + 1}",
                "seg_sop_instance_uid": f"SEG_{idx + 1}",
                "ct_sop_instance_uid": f"CT_{idx + 1}",
                "patient_id": f"CASE_{idx + 1}",
                "study_instance_uid": f"STUDY_{idx + 1}",
                "ct_series_instance_uid": f"CT_SERIES_{idx + 1}",
                "seg_series_instance_uid": f"SEG_SERIES_{idx + 1}",
                "slice_index": "3",
                "rows": "512",
                "columns": "512",
                "mask_area_px": "25",
                "bbox_x_min": "100",
                "bbox_y_min": "120",
                "bbox_x_max": "110",
                "bbox_y_max": "130",
                "bbox_width": "11",
                "bbox_height": "11",
                "centroid_x": "105",
                "centroid_y": "125",
                "is_empty": "false",
                "rejection_reason": "",
                "mask_npz_path": "mask.npz",
            }
        )
    _write_csv(
        roi_csv,
        roi_rows,
    )
    _write_csv(
        ct_csv,
        [
            {
                "study_instance_uid": "STUDY_1",
                "series_instance_uid": "CT_SERIES_1",
                "patient_id": "CASE_1",
                "slice_count": "10",
                "first_file_path": "a.dcm",
                "last_file_path": "b.dcm",
                "sort_method": "image_position_patient_z",
            }
        ],
    )
    return roi_csv, ct_csv


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
