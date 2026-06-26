"""Guided narration for representative gaze visualizations."""

from __future__ import annotations

import pandas as pd


HEATMAP_TEXT = {
    "focused_roi_confirmation": "The gaze rapidly concentrates near the ROI, suggesting direct confirmation behavior.",
    "missed_roi_search": "The scanpath shows search activity, but the ROI receives little direct attention.",
    "partial_near_miss_review": "The gaze approaches the ROI region but remains partly outside the mask, suggesting weak or incomplete inspection.",
    "skipped_slice": "The valid gaze exposure on this ROI slice is low, so the region is treated as not sufficiently evaluated.",
    "high_load_fragmented_review": "The gaze path is scattered with higher dispersion and revisits, consistent with fragmented review behavior.",
    "expert_like_systematic_review": "The gaze shows broader but structured inspection with ROI coverage, consistent with systematic review.",
}


def build_case_narration(row: pd.Series | dict[str, object]) -> dict[str, object]:
    label = str(_get(row, "hidden_behavior_label", "unknown"))
    attention = str(_get(row, "rule_attention_status", "unknown"))
    cognitive = str(_get(row, "cognitive_load_proxy", "unknown"))
    return {
        "session_id": str(_get(row, "session_id", "")),
        "roi_id": str(_get(row, "roi_id", "")),
        "behavior_label": label,
        "rule_attention_status": attention,
        "cognitive_load_proxy": cognitive,
        "heatmap_suggestion": HEATMAP_TEXT.get(label, "The heatmap summarizes gaze density in image coordinates."),
        "behavior_rationale": _behavior_rationale(label, attention, cognitive),
        "limitation_note": "This is a source-agnostic visualization. Synthetic gaze and future Tobii gaze use the same schema, but exact real Tobii validation remains future work.",
    }


def _behavior_rationale(label: str, attention: str, cognitive: str) -> str:
    return (
        f"The predicted behavior is consistent with the rendered gaze pattern, the rule attention status `{attention}`, "
        f"and the cognitive-load proxy `{cognitive}`. Synthetic and real gaze should be compared by behavior-level "
        "features and heatmap similarity, not exact point matching."
    )


def _get(row: pd.Series | dict[str, object], key: str, default: object) -> object:
    return row.get(key, default) if isinstance(row, dict) else row.get(key, default)
