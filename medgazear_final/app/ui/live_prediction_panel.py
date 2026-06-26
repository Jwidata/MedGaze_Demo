"""Right-side structured prediction and case insight panel."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class LivePredictionPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.roi_summary = QLabel("Selected ROI: none")
        self.roi_summary.setWordWrap(True)
        self.live_status = QLabel("Live gaze: synthetic replay/reference mode")
        self.live_status.setWordWrap(True)
        self.prediction = QLabel("Prediction: hidden in Silent mode")
        self.prediction.setWordWrap(True)
        self.features = QLabel("Attention features: none")
        self.features.setWordWrap(True)
        self.explanation = QLabel("Guided explanation: select an ROI episode.")
        self.explanation.setWordWrap(True)
        for title, widget in (
            ("Selected ROI", self.roi_summary),
            ("Live Gaze Status", self.live_status),
            ("Live/Model Prediction", self.prediction),
            ("Attention Features", self.features),
            ("Guided Explanation", self.explanation),
        ):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(widget)
            layout.addWidget(box)
        layout.addStretch(1)

    def set_case_summary(self, summary: dict[str, object]) -> None:
        del summary

    def set_live_status(self, text: str) -> None:
        self.live_status.setText(text)

    def set_roi(self, row: pd.Series | dict[str, object] | None, show_prediction: bool = True) -> None:
        if row is None:
            self.roi_summary.setText("Selected ROI: none")
            self.prediction.setText("Behaviour: -\nAttention: -\nCognitive load: -\nConfidence: -")
            self.features.setText("Attention features: none")
            self.explanation.setText("Guided explanation: select an ROI episode.")
            return
        ct_slice = _fmt_int(row.get("ct_stack_index"), one_based=True)
        roi_slice = _fmt_int(row.get("slice_index"), one_based=False)
        self.roi_summary.setText(
            f"ROI ID: {row.get('roi_id', '-')}\n"
            f"CT slice number: {ct_slice}\n"
            f"ROI geometry slice index: {roi_slice}\n"
            f"bbox: ({row.get('bbox_x_min', '-')}, {row.get('bbox_y_min', '-')}) to ({row.get('bbox_x_max', '-')}, {row.get('bbox_y_max', '-')})\n"
            f"centroid: ({row.get('centroid_x', '-')}, {row.get('centroid_y', '-')})"
        )
        if show_prediction:
            confidence = _confidence_bar(row.get("prediction_confidence"))
            self.prediction.setText(
                f"Behaviour: {_display_label(row.get('predicted_behavior_label', 'unavailable'))}\n"
                f"Confidence: {confidence}\n"
                f"Attention: {_display_label(row.get('rule_attention_status', 'unknown'))}\n"
                f"Cognitive load: {_display_label(row.get('cognitive_load_proxy', 'unknown'))}"
            )
        else:
            self.prediction.setText("Prediction feedback hidden in Silent mode.")
        self.features.setText(
            f"dwell inside ROI: {_fmt_float(row.get('total_gaze_time_inside_roi_ms'))} ms\n"
            f"hit count: {_fmt_float(row.get('gaze_hit_count_inside_roi'))}\n"
            f"validity ratio: {_fmt_float(row.get('gaze_validity_ratio'))}\n"
            f"time to first fixation: {_fmt_float(row.get('time_to_first_fixation_inside_roi_ms'))} ms\n"
            f"revisit count: {_fmt_float(row.get('roi_revisit_count'))}"
        )
        self.explanation.setText(_evidence_text(row))


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: object, one_based: bool = False) -> str:
    try:
        number = int(float(value))
        if one_based:
            number += 1
        return str(number)
    except (TypeError, ValueError):
        return "-"


def _display_label(value: object) -> str:
    return str(value).replace("_", " ")


def _confidence_bar(value: object) -> str:
    try:
        confidence = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        confidence = 0.0
    filled = int(round(confidence * 10))
    return f"{'█' * filled}{'░' * (10 - filled)} {confidence * 100:.0f}%"


def _evidence_text(row: pd.Series | dict[str, object]) -> str:
    evidence = []
    try:
        if float(row.get("total_gaze_time_inside_roi_ms", 0) or 0) > 0:
            evidence.append("✓ Dwell inside ROI")
        if float(row.get("gaze_hit_count_inside_roi", 0) or 0) > 0:
            evidence.append("✓ Fixation/hit evidence")
        if float(row.get("background_gaze_ratio", 1) or 1) < 0.5:
            evidence.append("✓ Limited background gaze")
        if float(row.get("gaze_validity_ratio", 0) or 0) >= 0.8:
            evidence.append("✓ High validity")
        if float(row.get("roi_revisit_count", 0) or 0) > 0:
            evidence.append("✓ ROI revisits")
    except (TypeError, ValueError):
        pass
    if not evidence:
        evidence.append("No strong ROI-level gaze evidence available.")
    return "Why?\n" + "\n".join(evidence[:5])
