"""Feature-set definitions with leakage protection."""

from __future__ import annotations

import pandas as pd


FORBIDDEN_TRAINING_FEATURES = {
    "hidden_behavior_label",
    "rule_attention_status",
    "attention_reason",
    "key_evidence_summary",
    "rule_confidence_proxy",
    "reader_profile",
}

METADATA_COLUMNS = {"session_id", "reader_id", "case_id", "roi_id", "slice_index"}

DIRECT_RULE_FEATURES = {
    "total_gaze_time_inside_roi_ms",
    "total_gaze_time_near_roi_ms",
    "gaze_hit_count_inside_roi",
    "gaze_hit_count_near_roi",
    "fixation_count_inside_roi",
    "fixation_count_near_roi",
    "valid_gaze_time_on_roi_slice_ms",
    "time_on_roi_slice_ms",
    "gaze_validity_ratio",
}

GEOMETRY_FEATURES = {
    "roi_area_px",
    "roi_bbox_width",
    "roi_bbox_height",
    "roi_center_x",
    "roi_center_y",
    "normalized_roi_position_x",
    "normalized_roi_position_y",
    "number_of_rois_on_slice",
    "roi_density_context",
}

ULTRA_DELEAKED_GEOMETRY_CONTEXT_ONLY = {
    "roi_area_px",
    "roi_bbox_width",
    "roi_bbox_height",
    "roi_center_x",
    "roi_center_y",
    "normalized_roi_position_x",
    "normalized_roi_position_y",
    "number_of_rois_on_slice",
    "roi_density_context",
}

RANDOM_NOISE_CONTROL_FEATURES = ["random_noise_0", "random_noise_1", "random_noise_2", "random_noise_3", "random_noise_4"]

GAZE_QUALITY_FEATURES = {"gaze_validity_ratio", "dropout_ratio", "blink_ratio", "invalid_burst_ratio", "outside_ct_ratio", "jitter_px"}

TEMPORAL_SEARCH_FEATURES = {
    "scanpath_length_px",
    "scanpath_length_on_roi_slice_px",
    "gaze_dispersion_px",
    "gaze_entropy",
    "number_of_gaze_clusters",
    "background_gaze_ratio",
    "roi_revisit_count",
    "near_roi_revisit_count",
    "slice_transition_count",
    "adjacent_slice_toggle_count",
    "scroll_event_count",
    "search_to_confirmation_ratio",
    "late_roi_discovery_flag",
    "mean_fixation_duration_ms",
    "max_fixation_duration_ms",
    "fixation_duration_variance",
    "saccade_like_ratio",
    "fixation_like_ratio",
    "first_half_roi_attention_ratio",
    "second_half_roi_attention_ratio",
    "delayed_attention_score",
}


def numeric_training_columns(dataset: pd.DataFrame) -> list[str]:
    blocked = FORBIDDEN_TRAINING_FEATURES | METADATA_COLUMNS
    return [column for column in dataset.columns if column not in blocked and pd.api.types.is_numeric_dtype(dataset[column])]


def build_feature_sets(dataset: pd.DataFrame) -> dict[str, list[str]]:
    full = numeric_training_columns(dataset)
    return {
        "full_feature_set": full,
        "no_direct_rule_features": [column for column in full if column not in DIRECT_RULE_FEATURES],
        "geometry_only_negative_control": [column for column in full if column in GEOMETRY_FEATURES],
        "gaze_quality_only_negative_control": [column for column in full if column in GAZE_QUALITY_FEATURES],
        "temporal_search_only": [column for column in full if column in TEMPORAL_SEARCH_FEATURES],
        "ultra_deleaked_geometry_context_only": [column for column in full if column in ULTRA_DELEAKED_GEOMETRY_CONTEXT_ONLY],
        "ultra_deleaked_random_noise_control": RANDOM_NOISE_CONTROL_FEATURES,
        "shuffled_label_control": full,
    }


def assert_no_forbidden_features(columns: list[str]) -> None:
    leaked = FORBIDDEN_TRAINING_FEATURES.intersection(columns)
    if leaked:
        raise ValueError(f"Forbidden training features present: {sorted(leaked)}")
