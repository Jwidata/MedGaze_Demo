"""ROI review queue helpers."""

from __future__ import annotations

import pandas as pd


STATUS_ORDER = {"not_evaluated": 0, "not_reviewed": 1, "weakly_reviewed": 2, "reviewed": 3}


def build_review_queue(features: pd.DataFrame, attention: pd.DataFrame) -> pd.DataFrame:
    rows = features.merge(attention[["session_id", "roi_id", "rule_attention_status"]], on=["session_id", "roi_id"], how="left")
    rows["_priority"] = rows["rule_attention_status"].map(STATUS_ORDER).fillna(9)
    slice_column = "ct_stack_index" if "ct_stack_index" in rows.columns else "slice_index"
    return rows.sort_values(["case_id", "_priority", slice_column, "session_id"]).drop(columns=["_priority"])
