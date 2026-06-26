"""Unified ROI-level feature schema."""

from __future__ import annotations

from pathlib import Path


METADATA_COLUMNS = ["session_id", "reader_id", "reader_profile", "case_id", "roi_id", "slice_index", "hidden_behavior_label"]
ROI_FEATURES = ["total_gaze_time_inside_roi_ms", "total_gaze_time_near_roi_ms", "gaze_hit_count_inside_roi", "gaze_hit_count_near_roi", "fixation_count_inside_roi", "fixation_count_near_roi", "mean_fixation_duration_inside_roi_ms", "max_fixation_duration_inside_roi_ms", "time_to_first_roi_fixation_ms", "valid_gaze_time_on_roi_slice_ms", "time_on_roi_slice_ms"]
SCANPATH_FEATURES = ["scanpath_length_px", "scanpath_length_on_roi_slice_px", "gaze_dispersion_px", "gaze_entropy", "number_of_gaze_clusters", "background_gaze_ratio", "roi_revisit_count", "near_roi_revisit_count", "slice_transition_count", "adjacent_slice_toggle_count", "scroll_event_count", "search_to_confirmation_ratio", "late_roi_discovery_flag"]
TEMPORAL_FEATURES = ["mean_fixation_duration_ms", "max_fixation_duration_ms", "fixation_duration_variance", "saccade_like_ratio", "fixation_like_ratio", "first_half_roi_attention_ratio", "second_half_roi_attention_ratio", "delayed_attention_score"]
QUALITY_FEATURES = ["gaze_validity_ratio", "dropout_ratio", "blink_ratio", "invalid_burst_ratio", "outside_ct_ratio", "jitter_px"]
GEOMETRY_FEATURES = ["roi_area_px", "roi_bbox_width", "roi_bbox_height", "roi_center_x", "roi_center_y", "normalized_roi_position_x", "normalized_roi_position_y", "number_of_rois_on_slice", "roi_density_context"]
FORBIDDEN_LABEL_COLUMNS = {"rule_attention_status", "reviewed", "weakly_reviewed", "not_reviewed", "not_evaluated"}


FEATURE_GROUPS = {
    "METADATA": METADATA_COLUMNS,
    "ROI": ROI_FEATURES,
    "SCANPATH": SCANPATH_FEATURES,
    "TEMPORAL": TEMPORAL_FEATURES,
    "QUALITY": QUALITY_FEATURES,
    "GEOMETRY": GEOMETRY_FEATURES,
}


def behavior_feature_columns() -> list[str]:
    columns: list[str] = []
    for group in ("METADATA", "ROI", "SCANPATH", "TEMPORAL", "QUALITY", "GEOMETRY"):
        columns.extend(FEATURE_GROUPS[group])
    return columns


def validate_no_leakage(columns: list[str]) -> None:
    leaked = FORBIDDEN_LABEL_COLUMNS.intersection(columns)
    if leaked:
        raise ValueError(f"Forbidden rule-label columns present: {sorted(leaked)}")


def write_feature_schema(path: Path) -> None:
    lines = ["# Feature Schema", ""]
    for group, columns in FEATURE_GROUPS.items():
        lines.append(f"## {group}")
        lines.extend(f"- `{column}`" for column in columns)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
