"""Bottom gaze timeline widget with simple playback controls."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt


class TimelineWidget(QWidget):
    sample_changed = pyqtSignal(object)
    def __init__(self) -> None:
        super().__init__()
        self.samples = pd.DataFrame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.label = QLabel("Timeline: no samples")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._update_position_label)
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed.setCurrentText("1x")
        self.playback_mode = QComboBox()
        self.playback_mode.addItems(["CT slice cine (all slices)", "Whole case evidence replay", "Selected ROI evidence replay"])
        self.playback_mode.setCurrentText("CT slice cine (all slices)")
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(QLabel("Playback"))
        controls.addWidget(self.playback_mode)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addLayout(controls)

    def set_samples(self, samples: pd.DataFrame) -> None:
        if samples.empty:
            self.samples = pd.DataFrame(columns=["timestamp_ms", "is_valid"])
        elif "timestamp_ms" in samples.columns:
            self.samples = samples.sort_values("timestamp_ms").reset_index(drop=True)
        else:
            self.samples = samples.reset_index(drop=True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.samples) - 1))
        self.reset()
        valid = int(self.samples["is_valid"].sum()) if "is_valid" in self.samples and not self.samples.empty else 0
        invalid = len(self.samples) - valid
        self.label.setText(f"Timeline: {len(self.samples)} samples | valid={valid} | invalid/dropout={invalid}")
        self._update_position_label(self.slider.value())

    def playback_mode_text(self) -> str:
        return self.playback_mode.currentText()

    def play(self) -> None:
        if not self.samples.empty:
            self._update_position_label(self.slider.value())
        self.timer.start(max(25, int(120 / self._speed_multiplier())))

    def pause(self) -> None:
        self.timer.stop()

    def reset(self) -> None:
        self.timer.stop()
        self.slider.setValue(0)

    def _advance(self) -> None:
        value = self.slider.value() + 1
        if value > self.slider.maximum():
            self.timer.stop()
            return
        self.slider.setValue(value)

    def _speed_multiplier(self) -> float:
        return float(self.speed.currentText().replace("x", ""))

    def _update_position_label(self, index: int) -> None:
        if self.samples.empty or index >= len(self.samples):
            return
        row = self.samples.iloc[index]
        self.sample_changed.emit(row.to_dict())
        if "mode" in row and row.get("mode") == "ct_cine":
            self.label.setText(f"CT cine: slice {int(row.get('ct_stack_index', 0)) + 1} / {len(self.samples)}")
            return
        valid = "valid" if bool(row.get("is_valid", False)) else "invalid/dropout"
        self.label.setText(
            f"Timeline: {len(self.samples)} samples | t={float(row.get('timestamp_ms', 0)):.1f} ms | "
            f"gaze=({float(row.get('image_x', 0)):.1f}, {float(row.get('image_y', 0)):.1f}) | {valid}"
        )
