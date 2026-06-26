"""Report writer for rule-based attention outputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from app.attention.attention_thresholds import AttentionThresholds


DISCLAIMER = "These are explainable simulation/reference thresholds, not clinical truth."


def write_attention_report(path: Path, distribution_rows: list[dict[str, str]], sensitivity_rows: list[dict[str, str]], thresholds: AttentionThresholds, status_rows: list[dict[str, str]] | None = None) -> None:
    lines = ["# Rule Attention Report", "", DISCLAIMER, "", "## Thresholds"]
    lines.extend(f"- {key}: {value}" for key, value in asdict(thresholds).items())
    lines.extend(["", "## Distribution"])
    lines.extend(f"- {row['rule_attention_status']}: {row['count']}" for row in distribution_rows)
    lines.extend(["", "## Sensitivity"])
    lines.extend(f"- {row['variation']}: changed {row['changed_status_count']} of {row['roi_count']} ROIs" for row in sensitivity_rows)
    if status_rows is not None:
        lines.extend(["", "## rule_attention_status by hidden_behavior_label"])
        for label, counts in _crosstab(status_rows).items():
            lines.append(f"- {label}: {counts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _crosstab(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row["hidden_behavior_label"]
        status = row["rule_attention_status"]
        table.setdefault(label, {})[status] = table.setdefault(label, {}).get(status, 0) + 1
    return dict(sorted(table.items()))
