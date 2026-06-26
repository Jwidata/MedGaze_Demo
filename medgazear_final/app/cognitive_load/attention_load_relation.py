"""Cross-tab analyses between attention, behavior, and cognitive-load proxy labels."""

from __future__ import annotations

import pandas as pd


def attention_vs_cognitive_proxy(labels: pd.DataFrame, attention: pd.DataFrame) -> pd.DataFrame:
    merged = labels[["session_id", "roi_id", "cognitive_load_proxy"]].merge(
        attention[["session_id", "roi_id", "rule_attention_status"]], on=["session_id", "roi_id"], how="inner"
    )
    return _crosstab(merged, "rule_attention_status")


def behavior_vs_cognitive_proxy(labels: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    if "hidden_behavior_label" in labels.columns:
        source = labels
    else:
        source = labels[["session_id", "roi_id", "cognitive_load_proxy"]].merge(
            behavior[["session_id", "roi_id", "hidden_behavior_label"]], on=["session_id", "roi_id"], how="inner"
        )
    return _crosstab(source, "hidden_behavior_label")


def _crosstab(data: pd.DataFrame, row_column: str) -> pd.DataFrame:
    table = pd.crosstab(data[row_column], data["cognitive_load_proxy"])
    for column in ["low_load_proxy", "medium_load_proxy", "high_load_proxy"]:
        if column not in table.columns:
            table[column] = 0
    table = table[["low_load_proxy", "medium_load_proxy", "high_load_proxy"]]
    table["total"] = table.sum(axis=1)
    return table.reset_index()
