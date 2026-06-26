from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

from app.ml_behavior.behavior_dataset_builder import build_behavior_dataset
from app.ml_behavior.behavior_feature_schema import BEHAVIOR_TO_REVIEW_STATUS, FORBIDDEN_TRAINING_FEATURES, allowed_feature_columns
from app.ml_behavior.behavior_model_card import write_behavior_model_card, write_behavior_summary
from app.ml_behavior.behavior_model_export import export_behavior_model
from app.ml_behavior.behavior_model_registry import behavior_models
from app.ml_behavior.behavior_negative_controls import run_behavior_negative_controls
from app.ml_behavior.behavior_prediction_service import BehaviorPredictionService


def test_forbidden_feature_removal_and_mapping(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path)
    dataset = build_behavior_dataset(path)
    columns = allowed_feature_columns(dataset)

    assert FORBIDDEN_TRAINING_FEATURES.isdisjoint(columns)
    assert not any(column.startswith("mapped_review_status") for column in columns)
    assert dataset.loc[0, "mapped_review_status"] == BEHAVIOR_TO_REVIEW_STATUS[dataset.loc[0, "hidden_behavior_label"]]


def test_model_export_load_and_prediction_schema(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path))
    feature_columns = allowed_feature_columns(dataset)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(dataset[feature_columns], dataset["hidden_behavior_label"])
    export_behavior_model(model, tmp_path, feature_columns, {"best_model": "dummy"})

    service = BehaviorPredictionService(tmp_path / "best_behavior_model.joblib", tmp_path / "behavior_feature_schema.json", tmp_path / "behavior_label_mapping.json")
    prediction = service.predict_behavior(dataset.iloc[0].to_dict())

    assert set(prediction) == {"predicted_behavior_label", "confidence", "class_probabilities", "mapped_review_status"}
    assert prediction["predicted_behavior_label"] in BEHAVIOR_TO_REVIEW_STATUS


def test_shuffled_label_negative_control_runs(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path, repeats=3))
    results = run_behavior_negative_controls(dataset, seed=3)

    assert "shuffled_label" in set(results["feature_set"])


def test_behavior_report_generation(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path))
    comparison = pd.DataFrame([{"model": "DummyClassifier", "macro_f1": 0.1, "balanced_accuracy": 0.2, "evaluation_split": "validation"}])
    test_results = pd.DataFrame([{"model": "DummyClassifier", "macro_f1": 0.2, "balanced_accuracy": 0.3, "evaluation_split": "test"}])
    card = tmp_path / "card.md"
    summary = tmp_path / "summary.md"

    write_behavior_model_card(card, dataset, comparison, test_results, {"train_rows": 4, "validation_rows": 1, "test_rows": 1, "train_validation_rows": 5})
    write_behavior_summary(summary, comparison, test_results)

    card_text = card.read_text(encoding="utf-8")
    summary_text = summary.read_text(encoding="utf-8")
    assert "synthetic hidden behavior labels" in card_text
    assert "test set was not used during model selection" in card_text
    assert "Final test macro F1" in summary_text
    assert "does not make clinical diagnosis" in summary_text


def test_mlp_classifier_registered() -> None:
    assert "MLPClassifier" in behavior_models()


def _write_dataset(tmp_path: Path, repeats: int = 1) -> Path:
    labels = [
        "expert_like_systematic_review",
        "focused_roi_confirmation",
        "partial_near_miss_review",
        "missed_roi_search",
        "skipped_slice",
        "high_load_fragmented_review",
    ] * repeats
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "session_id": f"S{idx}",
                "reader_id": f"R{idx % 2}",
                "reader_profile": "expert_systematic",
                "case_id": f"C{idx % 3}",
                "roi_id": f"ROI{idx}",
                "slice_index": idx,
                "hidden_behavior_label": label,
                "rule_attention_status": "reviewed",
                "total_gaze_time_inside_roi_ms": idx + 1,
                "scanpath_length_px": idx + 2,
                "mean_fixation_duration_ms": idx + 3,
                "gaze_validity_ratio": 0.9,
                "roi_area_px": idx + 4,
            }
        )
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
