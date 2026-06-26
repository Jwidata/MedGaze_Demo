"""Feature schema and label mapping for behavior learning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BEHAVIOR_LABELS = [
    "expert_like_systematic_review",
    "focused_roi_confirmation",
    "partial_near_miss_review",
    "missed_roi_search",
    "skipped_slice",
    "high_load_fragmented_review",
]

BEHAVIOR_TO_REVIEW_STATUS = {
    "expert_like_systematic_review": "reviewed",
    "focused_roi_confirmation": "reviewed",
    "partial_near_miss_review": "weakly_reviewed",
    "missed_roi_search": "not_reviewed",
    "skipped_slice": "not_evaluated",
    "high_load_fragmented_review": "uncertain_review",
}

FORBIDDEN_TRAINING_FEATURES = {
    "hidden_behavior_label",
    "rule_attention_status",
    "attention_reason",
    "key_evidence_summary",
    "rule_confidence_proxy",
    "reader_profile",
    "reader_id",
    "session_id",
    "case_id",
    "roi_id",
}

ROI_COVERAGE = ["total_gaze_time_inside_roi_ms", "total_gaze_time_near_roi_ms", "gaze_hit_count_inside_roi", "gaze_hit_count_near_roi", "fixation_count_inside_roi", "fixation_count_near_roi", "mean_fixation_duration_inside_roi_ms", "max_fixation_duration_inside_roi_ms", "time_to_first_roi_fixation_ms", "valid_gaze_time_on_roi_slice_ms", "time_on_roi_slice_ms"]
SCANPATH_SEARCH = ["scanpath_length_px", "scanpath_length_on_roi_slice_px", "gaze_dispersion_px", "gaze_entropy", "number_of_gaze_clusters", "background_gaze_ratio", "roi_revisit_count", "near_roi_revisit_count", "slice_transition_count", "adjacent_slice_toggle_count", "scroll_event_count", "search_to_confirmation_ratio", "late_roi_discovery_flag"]
TEMPORAL = ["mean_fixation_duration_ms", "max_fixation_duration_ms", "fixation_duration_variance", "saccade_like_ratio", "fixation_like_ratio", "first_half_roi_attention_ratio", "second_half_roi_attention_ratio", "delayed_attention_score"]
QUALITY = ["gaze_validity_ratio", "dropout_ratio", "blink_ratio", "invalid_burst_ratio", "outside_ct_ratio", "jitter_px"]
GEOMETRY = ["roi_area_px", "roi_bbox_width", "roi_bbox_height", "roi_center_x", "roi_center_y", "normalized_roi_position_x", "normalized_roi_position_y", "number_of_rois_on_slice", "roi_density_context"]


def allowed_feature_columns(dataset: pd.DataFrame) -> list[str]:
    forbidden = set(FORBIDDEN_TRAINING_FEATURES) | {c for c in dataset.columns if c.startswith("mapped_review_status")}
    return [c for c in dataset.columns if c not in forbidden and pd.api.types.is_numeric_dtype(dataset[c])]


def ablation_feature_sets(dataset: pd.DataFrame) -> dict[str, list[str]]:
    allowed = set(allowed_feature_columns(dataset))
    return {
        "roi_coverage_only": [c for c in ROI_COVERAGE if c in allowed],
        "scanpath_search_only": [c for c in SCANPATH_SEARCH if c in allowed],
        "temporal_only": [c for c in TEMPORAL if c in allowed],
        "quality_only": [c for c in QUALITY if c in allowed],
        "geometry_context_only": [c for c in GEOMETRY if c in allowed],
        "all_behavior_features": [c for c in allowed_feature_columns(dataset)],
    }


def negative_control_sets(dataset: pd.DataFrame) -> dict[str, list[str]]:
    sets = ablation_feature_sets(dataset)
    return {
        "geometry_only": sets["geometry_context_only"],
        "gaze_quality_only": sets["quality_only"],
        "shuffled_label": sets["all_behavior_features"],
        "case_id_reader_leakage_check": [],
    }


def write_schema(path: Path, feature_columns: list[str]) -> None:
    path.write_text(json.dumps({"feature_columns": feature_columns}, indent=2) + "\n", encoding="utf-8")


def write_label_mapping(path: Path) -> None:
    path.write_text(json.dumps(BEHAVIOR_TO_REVIEW_STATUS, indent=2) + "\n", encoding="utf-8")
