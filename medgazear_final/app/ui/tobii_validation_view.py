"""Live Tobii Validation view."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from app.gaze_sources.tobii_status import CALIBRATION_REMINDER


class TobiiValidationView(QWidget):
    def __init__(self, detect_callback: Callable[[], None], start_callback: Callable[[], None], stop_callback: Callable[[], None]) -> None:
        super().__init__()
        self.sdk_status = QLabel("SDK status: unknown")
        self.device_status = QLabel("Device: not checked")
        self.capture_status = QLabel("Capture: stopped")
        for label in (self.sdk_status, self.device_status, self.capture_status):
            label.setWordWrap(True)
        layout = QVBoxLayout(self)
        title = QLabel("Live Tobii Validation")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        subtitle = QLabel("Real gaze validation path. Tobii samples feed the same canonical schema, feature buffer, behavior model, attention rules, and cognitive proxy as synthetic replay.")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        tracker = QGroupBox("Tracker")
        tracker_layout = QVBoxLayout(tracker)
        tracker_layout.addWidget(self.sdk_status)
        tracker_layout.addWidget(self.device_status)
        tracker_layout.addWidget(QLabel(f"Calibration: {CALIBRATION_REMINDER}"))
        detect = QPushButton("Detect Tobii")
        start = QPushButton("Start live capture")
        stop = QPushButton("Stop live capture")
        detect.clicked.connect(detect_callback)
        start.clicked.connect(start_callback)
        stop.clicked.connect(stop_callback)
        tracker_layout.addWidget(detect)
        tracker_layout.addWidget(start)
        tracker_layout.addWidget(stop)
        tracker_layout.addWidget(self.capture_status)
        layout.addWidget(tracker)

        metrics = QGroupBox("Planned Comparison Metrics")
        metrics_layout = QVBoxLayout(metrics)
        for metric in (
            "heatmap similarity",
            "ROI dwell difference",
            "time-to-first-fixation difference",
            "behavior agreement",
            "attention agreement",
            "cognitive proxy difference",
        ):
            metrics_layout.addWidget(QLabel(f"- {metric}"))
        layout.addWidget(metrics)
        layout.addStretch(1)

    def update_status(self, sdk: str, device: str, capture: str) -> None:
        self.sdk_status.setText(f"SDK status: {sdk}")
        self.device_status.setText(f"Device: {device}")
        self.capture_status.setText(f"Capture: {capture}")
