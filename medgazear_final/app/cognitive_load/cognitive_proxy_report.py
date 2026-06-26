"""Reports for gaze-derived cognitive-load proxy analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.cognitive_load.cognitive_proxy_features import PROXY_FEATURE_WEIGHTS


LIMITATION_TEXT = "Cognitive-load proxy is inferred from gaze behavior and signal-quality patterns. It is not a validated cognitive-load measurement. Real validation would require additional ground truth such as pupil diameter, NASA-TLX, task difficulty ratings, or expert workload annotation."


def write_cognitive_proxy_report(
    path: Path,
    distribution: pd.DataFrame,
    attention_relation: pd.DataFrame,
    behavior_relation: pd.DataFrame,
    feature_count: int,
) -> None:
    lines = [
        "# Cognitive-Load Proxy Analysis Report",
        "",
        LIMITATION_TEXT,
        "",
        "## Proxy Formula",
        "Selected gaze behavior and signal-quality features were percentile-ranked, then combined as a weighted average into cognitive_load_proxy_score.",
        f"Available proxy features used: {feature_count}",
        "",
        "Feature weights:",
    ]
    for feature, weight in PROXY_FEATURE_WEIGHTS.items():
        lines.append(f"- {feature}: {weight}")
    lines.extend(["", "## Distribution", _markdown_table(distribution), "", "## Rule Attention Status Vs Cognitive Proxy", _markdown_table(attention_relation), "", "## Hidden Behavior Label Vs Cognitive Proxy", _markdown_table(behavior_relation)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cognitive_limitations(path: Path) -> None:
    lines = [
        "# Cognitive-Load Proxy Limitations",
        "",
        LIMITATION_TEXT,
        "",
        "This secondary analysis does not use pupil diameter, NASA-TLX, task difficulty ratings, real workload labels, or expert workload annotation.",
        "The low/medium/high labels are proxy labels derived from the distribution of engineered gaze features in this synthetic dataset.",
        "These outputs should be described as gaze-derived workload-like patterns, not true cognitive load.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "No rows."
    columns = list(data.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)
