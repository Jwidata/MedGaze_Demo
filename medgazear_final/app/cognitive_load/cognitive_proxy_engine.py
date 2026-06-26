"""Cognitive-load proxy scoring and label assignment."""

from __future__ import annotations

import pandas as pd


PROXY_LABELS = ["low_load_proxy", "medium_load_proxy", "high_load_proxy"]


def assign_cognitive_proxy_labels(proxy_features: pd.DataFrame) -> pd.DataFrame:
    if "cognitive_load_proxy_score" not in proxy_features.columns:
        raise ValueError("cognitive_load_proxy_score column is required.")
    rows = proxy_features.copy()
    low_threshold = float(rows["cognitive_load_proxy_score"].quantile(1 / 3))
    high_threshold = float(rows["cognitive_load_proxy_score"].quantile(2 / 3))
    rows["cognitive_load_proxy"] = rows["cognitive_load_proxy_score"].apply(lambda score: _label_score(float(score), low_threshold, high_threshold))
    rows["low_threshold"] = low_threshold
    rows["high_threshold"] = high_threshold
    return rows


def distribution(labels: pd.DataFrame) -> pd.DataFrame:
    counts = labels["cognitive_load_proxy"].value_counts().reindex(PROXY_LABELS, fill_value=0)
    total = max(1, int(counts.sum()))
    return pd.DataFrame({"cognitive_load_proxy": counts.index, "count": counts.values, "percentage": [round(count / total * 100, 2) for count in counts.values]})


def _label_score(score: float, low_threshold: float, high_threshold: float) -> str:
    if score <= low_threshold:
        return "low_load_proxy"
    if score <= high_threshold:
        return "medium_load_proxy"
    return "high_load_proxy"
