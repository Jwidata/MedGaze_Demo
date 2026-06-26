from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from app.ml_behavior.behavior_feature_schema import allowed_feature_columns
from app.ml_behavior.behavior_robustness_audit import HARD_ABLATION_REMOVE, run_behavior_robustness_audit, write_robustness_report


def test_behavior_robustness_audit_outputs_required_rows(tmp_path: Path) -> None:
    dataset = _dataset()
    features = allowed_feature_columns(dataset)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(dataset[features], dataset["hidden_behavior_label"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    results = run_behavior_robustness_audit(dataset, model_path)

    audit_names = set(results["audit_name"])
    assert "case_id_group_split" in audit_names
    assert "reader_held_out_split" in audit_names
    assert "hard_feature_ablation" in audit_names
    assert "low_noise_stress" in audit_names
    assert "medium_noise_stress" in audit_names
    assert "high_noise_stress" in audit_names
    assert "new_seed_external_test_proxy" in audit_names


def test_hard_ablation_remove_list_contains_required_features() -> None:
    assert "valid_gaze_time_on_roi_slice_ms" in HARD_ABLATION_REMOVE
    assert "scanpath_length_px" in HARD_ABLATION_REMOVE
    assert "outside_ct_ratio" in HARD_ABLATION_REMOVE


def test_robustness_report_wording(tmp_path: Path) -> None:
    results = pd.DataFrame([{"audit_name": "case_id_group_split", "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0, "note": "ok"}])
    path = tmp_path / "report.md"
    write_robustness_report(path, results)
    text = path.read_text(encoding="utf-8")
    assert "Perfect score on a random split" in text
    assert "not clinical validation" in text
    assert "before using the model in the UI" in text


def _dataset() -> pd.DataFrame:
    labels = [
        "expert_like_systematic_review",
        "focused_roi_confirmation",
        "partial_near_miss_review",
        "missed_roi_search",
        "skipped_slice",
        "high_load_fragmented_review",
    ] * 4
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "session_id": f"S{idx}",
                "reader_id": f"R{idx % 4}",
                "reader_profile": "expert_systematic",
                "case_id": f"C{idx % 6}",
                "roi_id": f"ROI{idx}",
                "slice_index": idx,
                "hidden_behavior_label": label,
                "total_gaze_time_inside_roi_ms": idx + 1,
                "time_on_roi_slice_ms": idx + 2,
                "valid_gaze_time_on_roi_slice_ms": idx + 3,
                "scanpath_length_px": idx + 4,
                "scanpath_length_on_roi_slice_px": idx + 5,
                "outside_ct_ratio": 0.1,
                "dropout_ratio": 0.1,
                "blink_ratio": 0.1,
                "invalid_burst_ratio": 0.1,
                "roi_area_px": idx + 6,
            }
        )
    return pd.DataFrame(rows)
