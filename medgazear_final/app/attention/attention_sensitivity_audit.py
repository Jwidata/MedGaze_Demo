"""Threshold sensitivity audit for rule-based attention statuses."""

from __future__ import annotations

from app.attention.attention_thresholds import AttentionThresholds
from app.attention.rule_attention_engine import run_attention_engine


def run_threshold_sensitivity(feature_rows: list[dict[str, str]], thresholds: AttentionThresholds) -> list[dict[str, str]]:
    baseline = run_attention_engine(feature_rows, thresholds).status_rows
    baseline_by_key = {(row["session_id"], row["roi_id"]): row["rule_attention_status"] for row in baseline}
    rows: list[dict[str, str]] = []
    for label, factor in (("lower_25_percent", 0.75), ("baseline", 1.0), ("higher_25_percent", 1.25)):
        result = run_attention_engine(feature_rows, thresholds.scaled(factor)).status_rows
        changed = sum(1 for row in result if row["rule_attention_status"] != baseline_by_key[(row["session_id"], row["roi_id"])])
        rows.append({"variation": label, "threshold_factor": str(factor), "roi_count": str(len(result)), "changed_status_count": str(changed)})
    return rows
