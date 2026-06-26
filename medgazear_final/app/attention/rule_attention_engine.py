"""Explainable rule-based ROI attention engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.attention.attention_status_schema import REVIEW_QUEUE_SEVERITY, validate_attention_status
from app.attention.attention_thresholds import AttentionThresholds


@dataclass(frozen=True)
class AttentionEngineResult:
    status_rows: list[dict[str, str]]
    review_queue_rows: list[dict[str, str]]
    distribution_rows: list[dict[str, str]]


def classify_feature_row(row: dict[str, str], thresholds: AttentionThresholds) -> dict[str, str]:
    valid_time = _float(row, "valid_gaze_time_on_roi_slice_ms")
    validity = _float(row, "gaze_validity_ratio")
    inside_dwell = _float(row, "total_gaze_time_inside_roi_ms")
    inside_hits = _float(row, "gaze_hit_count_inside_roi")
    inside_fix = _float(row, "fixation_count_inside_roi")
    near_dwell = _float(row, "total_gaze_time_near_roi_ms")
    near_hits = _float(row, "gaze_hit_count_near_roi")

    if valid_time < thresholds.minimum_valid_time_on_roi_slice_ms or validity < thresholds.minimum_gaze_validity_ratio:
        status = "not_evaluated"
        reason = "insufficient valid exposure to ROI slice"
        confidence = min(valid_time / max(1.0, thresholds.minimum_valid_time_on_roi_slice_ms), validity / max(0.001, thresholds.minimum_gaze_validity_ratio))
    elif inside_dwell >= thresholds.reviewed_min_inside_roi_dwell_ms and inside_hits >= thresholds.reviewed_min_inside_hit_count and inside_fix >= thresholds.reviewed_min_fixation_count_inside_roi:
        status = "reviewed"
        reason = "direct ROI dwell, hit count, and fixation evidence meet reviewed thresholds"
        confidence = min(1.0, (inside_dwell / thresholds.reviewed_min_inside_roi_dwell_ms + inside_hits / thresholds.reviewed_min_inside_hit_count + inside_fix / thresholds.reviewed_min_fixation_count_inside_roi) / 3)
    elif inside_dwell >= thresholds.weak_min_inside_roi_dwell_ms or (near_dwell >= thresholds.weak_min_near_roi_dwell_ms and near_hits >= thresholds.weak_min_near_hit_count):
        status = "weakly_reviewed"
        reason = "partial direct ROI evidence or strong near-ROI evidence"
        confidence = min(1.0, max(inside_dwell / thresholds.weak_min_inside_roi_dwell_ms, near_dwell / thresholds.weak_min_near_roi_dwell_ms, near_hits / thresholds.weak_min_near_hit_count))
    else:
        status = "not_reviewed"
        reason = "ROI slice viewed but ROI was not sufficiently inspected"
        confidence = min(1.0, valid_time / max(1.0, thresholds.minimum_valid_time_on_roi_slice_ms))
    validate_attention_status(status)
    return {
        "session_id": row["session_id"],
        "roi_id": row["roi_id"],
        "hidden_behavior_label": row["hidden_behavior_label"],
        "rule_attention_status": status,
        "attention_reason": reason,
        "rule_confidence_proxy": f"{confidence:.4f}",
        "key_evidence_summary": f"valid_time_ms={valid_time:.1f}; validity={validity:.3f}; inside_dwell_ms={inside_dwell:.1f}; inside_hits={inside_hits:.0f}; inside_fix={inside_fix:.0f}; near_dwell_ms={near_dwell:.1f}; near_hits={near_hits:.0f}",
    }


def run_attention_engine(feature_rows: list[dict[str, str]], thresholds: AttentionThresholds) -> AttentionEngineResult:
    status_rows = [classify_feature_row(row, thresholds) for row in feature_rows]
    review_queue_rows = sorted(
        [row for row in status_rows if row["rule_attention_status"] != "reviewed"],
        key=lambda row: (REVIEW_QUEUE_SEVERITY[row["rule_attention_status"]], row["session_id"], row["roi_id"]),
    )
    distribution: dict[str, int] = {}
    for row in status_rows:
        distribution[row["rule_attention_status"]] = distribution.get(row["rule_attention_status"], 0) + 1
    distribution_rows = [{"rule_attention_status": status, "count": str(distribution.get(status, 0))} for status in ("reviewed", "weakly_reviewed", "not_reviewed", "not_evaluated")]
    return AttentionEngineResult(status_rows, review_queue_rows, distribution_rows)


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0)
