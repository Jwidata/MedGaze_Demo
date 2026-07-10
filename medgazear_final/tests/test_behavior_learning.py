from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

from app.ml_behavior.behavior_dataset_builder import build_behavior_dataset
from app.ml_behavior.evaluation_integrity import apply_split_manifest, build_split_manifest, feature_columns_without_slice_index, prepare_behavior_dataset
from app.ml_behavior.behavior_feature_schema import BEHAVIOR_TO_REVIEW_STATUS, FORBIDDEN_TRAINING_FEATURES, allowed_feature_columns
from app.features.behavior_feature_builder import FrameRoiFeatureAccumulator, build_behavior_feature_row, feature_parity_matrix, validate_feature_schema
from app.features.feature_schema import ROI_FEATURES, SCANPATH_FEATURES, TEMPORAL_FEATURES, QUALITY_FEATURES, GEOMETRY_FEATURES
from app.ml_behavior.inference_readiness import assess_prediction_readiness
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


def test_case_grouped_split_has_zero_case_overlap(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path, repeats=7, grouped=True))
    manifest = build_split_manifest(dataset, "case_grouped_primary", seed=17)
    train, valid, test = apply_split_manifest(dataset, manifest)
    assert set(train["case_id"].astype(str)).isdisjoint(set(valid["case_id"].astype(str)))
    assert set(train["case_id"].astype(str)).isdisjoint(set(test["case_id"].astype(str)))
    assert set(valid["case_id"].astype(str)).isdisjoint(set(test["case_id"].astype(str)))


def test_reader_grouped_split_has_zero_reader_overlap(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path, repeats=7, grouped=True))
    manifest = build_split_manifest(dataset, "reader_grouped_robustness", seed=17)
    train, valid, test = apply_split_manifest(dataset, manifest)
    assert set(train["reader_id"].astype(str)).isdisjoint(set(valid["reader_id"].astype(str)))
    assert set(train["reader_id"].astype(str)).isdisjoint(set(test["reader_id"].astype(str)))
    assert set(valid["reader_id"].astype(str)).isdisjoint(set(test["reader_id"].astype(str)))


def test_split_manifest_is_reproducible(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path, repeats=7, grouped=True))
    first = build_split_manifest(dataset, "case_grouped_primary", seed=99)
    second = build_split_manifest(dataset, "case_grouped_primary", seed=99)
    assert first.train_row_ids == second.train_row_ids
    assert first.validation_row_ids == second.validation_row_ids
    assert first.test_row_ids == second.test_row_ids


def test_slice_index_ablation_uses_same_manifest_and_only_removes_slice_index(tmp_path: Path) -> None:
    dataset = build_behavior_dataset(_write_dataset(tmp_path, repeats=7, grouped=True))
    manifest = build_split_manifest(dataset, "case_grouped_primary", seed=42)
    manifest_again = build_split_manifest(dataset, "case_grouped_primary", seed=42)
    assert manifest.train_row_ids == manifest_again.train_row_ids
    all_features = allowed_feature_columns(dataset)
    ablated = feature_columns_without_slice_index(dataset)
    assert sorted(set(all_features) - set(ablated)) == ["slice_index"]


def test_dataset_preservation_before_split(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, repeats=7, grouped=True)
    dataset = build_behavior_dataset(path)
    prepared = prepare_behavior_dataset(dataset)
    assert len(prepared) == len(dataset)
    assert prepared["hidden_behavior_label"].tolist() == dataset["hidden_behavior_label"].tolist()


def test_offline_live_feature_parity_and_incremental_accumulator() -> None:
    roi = {
        "roi_id": "ROI1__frame_0000",
        "slice_index": 10,
        "ct_stack_index": 10,
        "rows": 64,
        "columns": 64,
        "bbox_x_min": 8,
        "bbox_y_min": 8,
        "bbox_x_max": 20,
        "bbox_y_max": 20,
        "bbox_width": 12,
        "bbox_height": 12,
        "centroid_x": 14,
        "centroid_y": 14,
        "mask_area_px": 144,
        "ct_series_instance_uid": "SERIES1",
    }
    samples = pd.DataFrame(
        [
            {"session_id": "S1", "reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "roi_id": "ROI1__frame_0000", "slice_index": 10, "sample_index": 0, "timestamp_ms": 0.0, "image_x": 10.0, "image_y": 10.0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0, "hidden_behavior_label": "focused_roi_confirmation"},
            {"session_id": "S1", "reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "roi_id": "ROI1__frame_0000", "slice_index": 10, "sample_index": 1, "timestamp_ms": 100.0, "image_x": 11.0, "image_y": 11.0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0, "hidden_behavior_label": "focused_roi_confirmation"},
            {"session_id": "S1", "reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "roi_id": "ROI1__frame_0000", "slice_index": 10, "sample_index": 2, "timestamp_ms": 200.0, "image_x": 12.0, "image_y": 12.0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0, "hidden_behavior_label": "focused_roi_confirmation"},
        ]
    )
    metadata = {"reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "hidden_behavior_label": "focused_roi_confirmation"}
    offline = build_behavior_feature_row(samples, roi, metadata)
    accumulator = FrameRoiFeatureAccumulator(roi, metadata)
    for sample in samples.to_dict("records"):
        accumulator.add_sample(sample)
    live = accumulator.build()
    feature_columns = ["slice_index", *ROI_FEATURES, *SCANPATH_FEATURES, *TEMPORAL_FEATURES, *QUALITY_FEATURES, *GEOMETRY_FEATURES]
    matrix = feature_parity_matrix(offline.row, live.row, feature_columns)
    assert all(row["parity_pass"] for row in matrix)


def test_feature_schema_ordering_and_readiness() -> None:
    feature_row = {"slice_index": 10, "total_gaze_time_inside_roi_ms": 100.0, "total_gaze_time_near_roi_ms": 0.0, "gaze_hit_count_inside_roi": 1, "gaze_hit_count_near_roi": 0, "fixation_count_inside_roi": 0, "fixation_count_near_roi": 0, "mean_fixation_duration_inside_roi_ms": 0.0, "max_fixation_duration_inside_roi_ms": 0.0, "time_to_first_roi_fixation_ms": -1.0, "valid_gaze_time_on_roi_slice_ms": 100.0, "time_on_roi_slice_ms": 100.0, "scanpath_length_px": 10.0, "scanpath_length_on_roi_slice_px": 10.0, "gaze_dispersion_px": 5.0, "gaze_entropy": 0.1, "number_of_gaze_clusters": 1, "background_gaze_ratio": 0.0, "roi_revisit_count": 0, "near_roi_revisit_count": 0, "slice_transition_count": 0, "adjacent_slice_toggle_count": 0, "scroll_event_count": 0, "search_to_confirmation_ratio": 0.0, "late_roi_discovery_flag": 0, "mean_fixation_duration_ms": 0.0, "max_fixation_duration_ms": 0.0, "fixation_duration_variance": 0.0, "saccade_like_ratio": 0.0, "fixation_like_ratio": 0.0, "first_half_roi_attention_ratio": 1.0, "second_half_roi_attention_ratio": 1.0, "delayed_attention_score": 0.0, "gaze_validity_ratio": 1.0, "dropout_ratio": 0.0, "blink_ratio": 0.0, "invalid_burst_ratio": 0.0, "outside_ct_ratio": 0.0, "jitter_px": 0.0, "roi_area_px": 144.0, "roi_bbox_width": 12.0, "roi_bbox_height": 12.0, "roi_center_x": 14.0, "roi_center_y": 14.0, "normalized_roi_position_x": 0.2, "normalized_roi_position_y": 0.2, "number_of_rois_on_slice": 1, "roi_density_context": 1.0, "_sample_count": 3, "_fixation_ready": True}
    columns = ["slice_index", *ROI_FEATURES, *SCANPATH_FEATURES, *TEMPORAL_FEATURES, *QUALITY_FEATURES, *GEOMETRY_FEATURES]
    validation = validate_feature_schema(feature_row, columns)
    assert validation["duplicates"] == []
    assert validation["missing"] == []
    readiness = assess_prediction_readiness(feature_row, columns)
    assert readiness.status == "READY"
    missing_row = dict(feature_row)
    missing_row.pop("fixation_count_inside_roi")
    readiness = assess_prediction_readiness(missing_row, columns)
    assert readiness.status == "MISSING_REQUIRED_FEATURES"
    collecting = dict(feature_row)
    collecting["_sample_count"] = 1
    readiness = assess_prediction_readiness(collecting, columns)
    assert readiness.status == "COLLECTING_EVIDENCE"


def _write_dataset(tmp_path: Path, repeats: int = 1, grouped: bool = False) -> Path:
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
        if grouped:
            case_id = f"C{idx}"
            reader_id = f"R{idx}"
        else:
            case_id = f"C{idx % 3}"
            reader_id = f"R{idx % 2}"
        rows.append(
            {
                "session_id": f"S{idx}",
                "reader_id": reader_id,
                "reader_profile": "expert_systematic",
                "case_id": case_id,
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
