"""Bottom selected-source gaze timeline."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class GazeTimelineWidget(QWidget):
    sample_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.samples = pd.DataFrame(columns=["timestamp_ms", "is_valid"])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.label = QLabel("Replay: select an ROI/session or behaviour example")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._emit_current)
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        self.live_status = QLabel("Live Tobii mode: use Start/Stop live gaze controls.")
        self.live_status.setVisible(False)
        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed.setCurrentText("1x")
        self.follow_slice = QCheckBox("Follow gaze slice during replay")
        self.follow_slice.setChecked(True)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)
        controls = QHBoxLayout()
        self.speed_label = QLabel("Speed")
        self.replay_controls = [self.play_button, self.pause_button, self.reset_button, self.speed_label, self.speed, self.follow_slice]
        for widget in self.replay_controls:
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.live_status)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addLayout(controls)

    def set_samples(self, samples: pd.DataFrame) -> None:
        if samples.empty:
            self.samples = pd.DataFrame(columns=["timestamp_ms", "is_valid"])
        else:
            self.samples = samples.sort_values("timestamp_ms").reset_index(drop=True) if "timestamp_ms" in samples.columns else samples.reset_index(drop=True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.samples) - 1))
        self.reset()
        self._update_label(0)

    def play(self) -> None:
        if not self.samples.empty:
            self._emit_current(self.slider.value())
            self.timer.start(max(25, int(120 / self._speed_multiplier())))

    def pause(self) -> None:
        self.timer.stop()

    def reset(self) -> None:
        self.timer.stop()
        self.slider.setValue(0)
        if not self.samples.empty:
            self._emit_current(0)

    def follow_enabled(self) -> bool:
        return self.follow_slice.isChecked()

    def set_replay_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.speed.setEnabled(enabled)
        self.follow_slice.setEnabled(enabled)
        for widget in self.replay_controls:
            widget.setVisible(enabled)
        self.live_status.setVisible(not enabled)
        if not enabled:
            self.timer.stop()
            self.label.setText("Live Tobii mode: use Start/Stop live gaze controls.")

    def _advance(self) -> None:
        value = self.slider.value() + 1
        if value > self.slider.maximum():
            self.timer.stop()
            return
        self.slider.setValue(value)

    def _speed_multiplier(self) -> float:
        return float(self.speed.currentText().replace("x", ""))

    def _emit_current(self, index: int) -> None:
        if self.samples.empty or index >= len(self.samples):
            self._update_label(index)
            return
        row = self.samples.iloc[index].to_dict()
        self.sample_changed.emit(row)
        self._update_label(index)

    def _update_label(self, index: int) -> None:
        if self.samples.empty or index >= len(self.samples):
            self.label.setText("Replay: select an ROI/session or behaviour example")
            return
        row = self.samples.iloc[index]
        valid = "valid" if bool(row.get("is_valid", False)) else "invalid/dropout"
        flags = []
        if bool(row.get("is_outside_ct", False)):
            flags.append("outside CT")
        if bool(row.get("is_ui_glance", False)):
            flags.append("UI glance")
        suffix = f" | {', '.join(flags)}" if flags else ""
        seconds = float(row.get("timestamp_ms", 0)) / 1000.0
        slice_text = int(float(row.get("ct_stack_index", -1))) + 1 if "ct_stack_index" in row else "-"
        self.label.setText(f"Replay: |●{'─' * 14}| {seconds:.1f} sec | Current Slice {slice_text} | Speed {self.speed.currentText()} | {valid}{suffix}")
