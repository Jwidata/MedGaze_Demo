"""Right-side insight panel."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


KEY_FEATURES = [
    "total_gaze_time_inside_roi_ms",
    "valid_gaze_time_on_roi_slice_ms",
    "gaze_hit_count_inside_roi",
    "gaze_dispersion_px",
    "roi_revisit_count",
    "background_gaze_ratio",
    "gaze_validity_ratio",
]


def format_insight_text(row: pd.Series | dict[str, object]) -> str:
    confidence = float(row.get("prediction_confidence", 0) or 0)
    lines = [
        f"Hidden behavior label: {row.get('hidden_behavior_label', 'unknown')}",
        f"Predicted behavior label: {row.get('predicted_behavior_label', 'unavailable')}",
        f"Prediction confidence: {confidence:.3f}",
        f"Rule attention status: {row.get('rule_attention_status', 'unknown')}",
        f"Cognitive-load proxy: {row.get('cognitive_load_proxy', 'unknown')}",
        "",
        "Key feature summary:",
    ]
    for feature in KEY_FEATURES:
        if feature in row:
            value = row.get(feature)
            lines.append(f"- {feature}: {_format_value(value)}")
    lines.extend(["", "Guided narration:", str(row.get("guided_narration", "No narration available."))])
    return "\n".join(lines)


class InsightPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.title = QLabel("Insights")
        self.case_summary = QLabel("Case summary: -")
        self.case_summary.setWordWrap(True)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.case_summary)
        layout.addWidget(self.text, stretch=1)

    def set_case(self, row: pd.Series | dict[str, object]) -> None:
        self.text.setPlainText(format_insight_text(row))

    def set_case_summary(self, text: str) -> None:
        self.case_summary.setText(text)


def _format_value(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
