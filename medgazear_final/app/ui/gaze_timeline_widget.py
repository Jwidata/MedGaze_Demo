"""Bottom selected-source gaze timeline."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSizePolicy, QWidget


class GazeTimelineWidget(QWidget):
    sample_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TimelineShell")
        self.setMinimumHeight(58)
        self.setMaximumHeight(68)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.samples = pd.DataFrame(columns=["timestamp_ms", "is_valid"])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)

        self.source_label = QLabel("Synthetic Replay")
        self.source_label.setObjectName("CompactMeta")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.valueChanged.connect(self._emit_current)
        self.elapsed_label = QLabel("0.0s")
        self.elapsed_label.setObjectName("CompactMeta")

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        for button in (self.play_button, self.pause_button, self.reset_button):
            button.setMaximumWidth(68)

        self.live_status = QLabel("Tobii Live")
        self.live_status.setObjectName("CompactMeta")
        self.live_status.setVisible(False)

        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed.setCurrentText("1x")
        self.speed.setMaximumWidth(84)
        self.follow_slice = QCheckBox("Follow gaze slice")
        self.follow_slice.setChecked(True)

        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        layout.addWidget(self.source_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.elapsed_label)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.speed)
        layout.addWidget(self.follow_slice)
        layout.addWidget(self.live_status)

    def set_samples(self, samples: pd.DataFrame) -> None:
        if samples.empty:
            self.samples = pd.DataFrame(columns=["timestamp_ms", "is_valid"])
        else:
            self.samples = samples.sort_values("timestamp_ms").reset_index(drop=True) if "timestamp_ms" in samples.columns else samples.reset_index(drop=True)
        self.timer.stop()
        with QSignalBlocker(self.slider):
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, len(self.samples) - 1))
            self.slider.setValue(0)
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
        else:
            self.elapsed_label.setText("0.0s")

    def follow_enabled(self) -> bool:
        return self.follow_slice.isChecked()

    def set_replay_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.speed.setEnabled(enabled)
        self.follow_slice.setEnabled(enabled)
        for widget in (self.play_button, self.pause_button, self.reset_button, self.speed, self.follow_slice, self.elapsed_label, self.slider):
            widget.setVisible(enabled)
        self.source_label.setText("Synthetic Replay" if enabled else "Tobii Live")
        self.live_status.setVisible(not enabled)
        if not enabled:
            self.timer.stop()

    def set_live_summary(self, sample_count: int, signal_quality: str) -> None:
        self.live_status.setText(f"Ready | Samples: {sample_count} | Signal: {signal_quality}")

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
            self.elapsed_label.setText("0.0s")
            return
        row = self.samples.iloc[index]
        seconds = float(row.get("timestamp_ms", 0)) / 1000.0
        self.elapsed_label.setText(f"{seconds:.1f}s")
