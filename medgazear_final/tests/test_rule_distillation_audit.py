from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml_rule_audit.deleaked_feature_sets import DIRECT_RULE_FEATURES, FORBIDDEN_TRAINING_FEATURES, RANDOM_NOISE_CONTROL_FEATURES, ULTRA_DELEAKED_GEOMETRY_CONTEXT_ONLY, build_feature_sets, numeric_training_columns
from app.ml_rule_audit.rule_audit_report import write_rule_recovery_report
from app.ml_rule_audit.rule_distillation_dataset import build_rule_distillation_dataset
from app.ml_rule_audit.rule_distillation_models import evaluate_models_for_feature_set
from app.ml_rule_audit.rule_recovery_audit import run_rule_recovery_audit


def test_target_merge_correctness(tmp_path: Path) -> None:
    features, attention = _write_inputs(tmp_path)
    dataset = build_rule_distillation_dataset(features, attention)

    assert len(dataset) == 4
    assert dataset.loc[dataset["roi_id"] == "ROI_1", "rule_attention_status"].iloc[0] == "reviewed"


def test_forbidden_feature_removal_and_feature_sets(tmp_path: Path) -> None:
    features, attention = _write_inputs(tmp_path)
    dataset = build_rule_distillation_dataset(features, attention)
    train_columns = numeric_training_columns(dataset)
    feature_sets = build_feature_sets(dataset)

    assert FORBIDDEN_TRAINING_FEATURES.isdisjoint(train_columns)
    assert DIRECT_RULE_FEATURES.intersection(feature_sets["full_feature_set"])
    assert not DIRECT_RULE_FEATURES.intersection(feature_sets["no_direct_rule_features"])
    assert set(feature_sets["geometry_only_negative_control"]) == ULTRA_DELEAKED_GEOMETRY_CONTEXT_ONLY
    assert set(feature_sets["ultra_deleaked_geometry_context_only"]).issubset(ULTRA_DELEAKED_GEOMETRY_CONTEXT_ONLY)
    assert not DIRECT_RULE_FEATURES.intersection(feature_sets["ultra_deleaked_geometry_context_only"])
    assert feature_sets["ultra_deleaked_random_noise_control"] == RANDOM_NOISE_CONTROL_FEATURES
    assert "gaze_validity_ratio" in feature_sets["gaze_quality_only_negative_control"]


def test_shuffled_label_control_runs(tmp_path: Path) -> None:
    features, attention = _write_inputs(tmp_path)
    dataset = build_rule_distillation_dataset(features, attention)
    columns = build_feature_sets(dataset)["shuffled_label_control"]
    results = evaluate_models_for_feature_set(dataset, columns, "shuffled_label_control", shuffled_labels=True, random_state=1)

    assert results
    assert all(row["feature_set"] == "shuffled_label_control" for row in results)


def test_report_generation(tmp_path: Path) -> None:
    features, attention = _write_inputs(tmp_path)
    dataset = build_rule_distillation_dataset(features, attention)
    results = run_rule_recovery_audit(dataset, random_state=2)
    report_path = tmp_path / "report.md"

    write_rule_recovery_report(report_path, results)

    assert report_path.exists()
    assert "Rule Recovery Audit Report" in report_path.read_text(encoding="utf-8")


def test_ultra_deleaked_random_noise_control_runs(tmp_path: Path) -> None:
    features, attention = _write_inputs(tmp_path)
    dataset = build_rule_distillation_dataset(features, attention)
    results = run_rule_recovery_audit(dataset, random_state=4)

    assert "ultra_deleaked_random_noise_control" in set(results["feature_set"])
    assert "ultra_deleaked_geometry_context_only" in set(results["feature_set"])


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    features = pd.DataFrame(
        [
            _feature("S1", "ROI_1", 1000, 10, 1, 900, 0.95, 100, 20),
            _feature("S2", "ROI_2", 600, 5, 0, 800, 0.90, 200, 30),
            _feature("S3", "ROI_3", 0, 0, 0, 100, 0.95, 300, 40),
            _feature("S4", "ROI_4", 50, 1, 0, 900, 0.95, 400, 50),
        ]
    )
    attention = pd.DataFrame(
        [
            _attention("S1", "ROI_1", "reviewed"),
            _attention("S2", "ROI_2", "weakly_reviewed"),
            _attention("S3", "ROI_3", "not_evaluated"),
            _attention("S4", "ROI_4", "not_reviewed"),
        ]
    )
    features_path = tmp_path / "features.csv"
    attention_path = tmp_path / "attention.csv"
    features.to_csv(features_path, index=False)
    attention.to_csv(attention_path, index=False)
    return features_path, attention_path


def _feature(session_id: str, roi_id: str, inside_dwell: float, inside_hits: float, inside_fix: float, valid_time: float, validity: float, area: float, width: float) -> dict[str, object]:
    return {
        "session_id": session_id,
        "reader_id": "R",
        "reader_profile": "expert_systematic",
        "case_id": "C",
        "roi_id": roi_id,
        "slice_index": 1,
        "hidden_behavior_label": "focused_roi_confirmation",
        "total_gaze_time_inside_roi_ms": inside_dwell,
        "total_gaze_time_near_roi_ms": 800,
        "gaze_hit_count_inside_roi": inside_hits,
        "gaze_hit_count_near_roi": 10,
        "fixation_count_inside_roi": inside_fix,
        "fixation_count_near_roi": 1,
        "valid_gaze_time_on_roi_slice_ms": valid_time,
        "time_on_roi_slice_ms": valid_time + 100,
        "gaze_validity_ratio": validity,
        "dropout_ratio": 0.01,
        "blink_ratio": 0.01,
        "invalid_burst_ratio": 0.0,
        "outside_ct_ratio": 0.0,
        "jitter_px": 5.0,
        "scanpath_length_px": 1000,
        "gaze_entropy": 1.5,
        "background_gaze_ratio": 0.3,
        "mean_fixation_duration_ms": 100,
        "saccade_like_ratio": 0.4,
        "roi_area_px": area,
        "roi_bbox_width": width,
        "roi_bbox_height": width + 1,
        "roi_center_x": area / 10,
        "roi_center_y": area / 11,
        "normalized_roi_position_x": 0.2,
        "normalized_roi_position_y": 0.3,
        "number_of_rois_on_slice": 2,
        "roi_density_context": 0.7,
    }


def _attention(session_id: str, roi_id: str, status: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "roi_id": roi_id,
        "hidden_behavior_label": "focused_roi_confirmation",
        "rule_attention_status": status,
        "attention_reason": "reason",
        "rule_confidence_proxy": 1.0,
        "key_evidence_summary": "evidence",
    }
