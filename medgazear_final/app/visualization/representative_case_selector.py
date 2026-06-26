"""Select representative examples for visualization."""

from __future__ import annotations

import pandas as pd


BEHAVIOR_LABELS = [
    "expert_like_systematic_review",
    "focused_roi_confirmation",
    "partial_near_miss_review",
    "missed_roi_search",
    "skipped_slice",
    "high_load_fragmented_review",
]


def select_representative_cases(
    features: pd.DataFrame,
    attention: pd.DataFrame | None = None,
    cognitive: pd.DataFrame | None = None,
    examples_per_behavior: int = 3,
) -> pd.DataFrame:
    rows = features.copy()
    if attention is not None and not attention.empty:
        rows = rows.merge(attention[["session_id", "roi_id", "rule_attention_status"]], on=["session_id", "roi_id"], how="left")
    if cognitive is not None and not cognitive.empty:
        keep = [column for column in ("session_id", "roi_id", "cognitive_load_proxy", "cognitive_load_proxy_score") if column in cognitive.columns]
        rows = rows.merge(cognitive[keep], on=["session_id", "roi_id"], how="left")
    selected = []
    for label in BEHAVIOR_LABELS:
        subset = rows[rows["hidden_behavior_label"] == label].copy()
        if subset.empty:
            continue
        subset["_score"] = _representative_score(subset)
        selected.append(subset.sort_values(["_score", "session_id", "roi_id"], ascending=[False, True, True]).head(examples_per_behavior))
    if not selected:
        return pd.DataFrame(columns=list(rows.columns))
    result = pd.concat(selected, ignore_index=True).drop(columns=["_score"], errors="ignore")
    result.insert(0, "visualization_rank", result.groupby("hidden_behavior_label").cumcount() + 1)
    return result


def _representative_score(rows: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=rows.index)
    for column in ("gaze_validity_ratio", "valid_gaze_time_on_roi_slice_ms"):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce").fillna(0)
            max_value = max(float(values.max()), 1.0)
            score += values / max_value
    if "cognitive_load_proxy_score" in rows.columns:
        score += pd.to_numeric(rows["cognitive_load_proxy_score"], errors="coerce").fillna(0) * 0.1
    return score
