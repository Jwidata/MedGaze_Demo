from __future__ import annotations

from app.attention.attention_sensitivity_audit import run_threshold_sensitivity
from app.attention.attention_thresholds import AttentionThresholds
from app.attention.rule_attention_engine import classify_feature_row, run_attention_engine


def test_reviewed_status() -> None:
    row = _feature_row(inside_dwell=900, inside_hits=10, inside_fix=1, valid_time=1000, validity=0.95)
    assert classify_feature_row(row, AttentionThresholds())["rule_attention_status"] == "reviewed"


def test_weakly_reviewed_status() -> None:
    row = _feature_row(inside_dwell=100, inside_hits=2, inside_fix=0, near_dwell=900, near_hits=12, valid_time=1000, validity=0.95)
    assert classify_feature_row(row, AttentionThresholds())["rule_attention_status"] == "weakly_reviewed"


def test_not_reviewed_status() -> None:
    row = _feature_row(inside_dwell=50, inside_hits=1, inside_fix=0, near_dwell=100, near_hits=2, valid_time=1000, validity=0.95)
    assert classify_feature_row(row, AttentionThresholds())["rule_attention_status"] == "not_reviewed"


def test_not_evaluated_status() -> None:
    row = _feature_row(inside_dwell=900, inside_hits=10, inside_fix=1, valid_time=100, validity=0.95)
    assert classify_feature_row(row, AttentionThresholds())["rule_attention_status"] == "not_evaluated"


def test_review_queue_severity_ordering() -> None:
    rows = [
        _feature_row("weak", near_dwell=900, near_hits=12, valid_time=1000, validity=0.95),
        _feature_row("none", valid_time=100, validity=0.95),
        _feature_row("not", valid_time=1000, validity=0.95),
    ]
    queue = run_attention_engine(rows, AttentionThresholds()).review_queue_rows
    assert [row["rule_attention_status"] for row in queue] == ["not_evaluated", "not_reviewed", "weakly_reviewed"]


def test_threshold_sensitivity_output() -> None:
    rows = [_feature_row(inside_dwell=700, inside_hits=8, inside_fix=1, valid_time=1000, validity=0.95)]
    sensitivity = run_threshold_sensitivity(rows, AttentionThresholds())

    assert [row["variation"] for row in sensitivity] == ["lower_25_percent", "baseline", "higher_25_percent"]
    assert all(row["roi_count"] == "1" for row in sensitivity)


def _feature_row(
    suffix: str = "x",
    *,
    inside_dwell: float = 0,
    inside_hits: float = 0,
    inside_fix: float = 0,
    near_dwell: float = 0,
    near_hits: float = 0,
    valid_time: float = 1000,
    validity: float = 1.0,
) -> dict[str, str]:
    return {
        "session_id": f"S_{suffix}",
        "roi_id": f"ROI_{suffix}",
        "hidden_behavior_label": "focused_roi_confirmation",
        "valid_gaze_time_on_roi_slice_ms": str(valid_time),
        "gaze_validity_ratio": str(validity),
        "total_gaze_time_inside_roi_ms": str(inside_dwell),
        "gaze_hit_count_inside_roi": str(inside_hits),
        "fixation_count_inside_roi": str(inside_fix),
        "total_gaze_time_near_roi_ms": str(near_dwell),
        "gaze_hit_count_near_roi": str(near_hits),
    }
