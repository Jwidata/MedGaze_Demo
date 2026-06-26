"""Hidden behavior label schema for synthetic gaze generation."""

from __future__ import annotations

HIDDEN_BEHAVIOR_LABELS = (
    "expert_like_systematic_review",
    "focused_roi_confirmation",
    "partial_near_miss_review",
    "missed_roi_search",
    "skipped_slice",
    "high_load_fragmented_review",
)

RULE_STATUS_LABELS = {"matched_strict", "matched_partial", "unmatched_missing_ct", "invalid_no_references"}


def is_valid_hidden_behavior_label(label: str) -> bool:
    return label in HIDDEN_BEHAVIOR_LABELS


def validate_hidden_label(label: str) -> None:
    if not is_valid_hidden_behavior_label(label):
        raise ValueError(f"Invalid hidden behavior label: {label}")
    if label in RULE_STATUS_LABELS:
        raise ValueError(f"Hidden behavior label must not be a rule status: {label}")
