"""Behaviour Library view for representative synthetic-trained classes."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget


BEHAVIOR_CARDS = {
    "focused_roi_confirmation": {
        "title": "Focused ROI Confirmation",
        "meaning": "Gaze quickly concentrates near the ROI, suggesting direct confirmation.",
        "pattern": "Compact heatmap on/near the ROI with early direct hits.",
        "features": "High ROI dwell, high validity, short time to first fixation.",
    },
    "expert_like_systematic_review": {
        "title": "Expert Systematic Review",
        "meaning": "Gaze shows structured inspection with ROI coverage.",
        "pattern": "Orderly sweep with sustained ROI coverage.",
        "features": "Balanced dwell, ROI revisits, low dropout.",
    },
    "partial_near_miss_review": {
        "title": "Partial Near Miss Review",
        "meaning": "Gaze approaches ROI but remains partly outside the mask.",
        "pattern": "Heatmap adjacent to ROI with limited direct overlap.",
        "features": "Moderate background gaze, partial ROI hits, delayed fixation.",
    },
    "missed_roi_search": {
        "title": "Missed ROI Search",
        "meaning": "Search activity exists, but the ROI receives little direct attention.",
        "pattern": "Gaze scans the slice while missing the ROI region.",
        "features": "Low ROI dwell, low hit count, high background gaze.",
    },
    "skipped_slice": {
        "title": "Skipped Slice",
        "meaning": "The slice/ROI has too little valid exposure.",
        "pattern": "Few or no valid samples on the ROI slice.",
        "features": "Low valid time, high dropout or minimal exposure.",
    },
    "high_load_fragmented_review": {
        "title": "High Load Fragmented Review",
        "meaning": "Gaze is scattered with revisits and higher dispersion.",
        "pattern": "Fragmented gaze with multiple returns and broad heatmap spread.",
        "features": "High dispersion, revisits, elevated background gaze.",
    },
}


class BehaviorLibraryView(QWidget):
    def __init__(self, open_callback: Callable[[str], None]) -> None:
        super().__init__()
        self.status = QLabel("Open a representative replay to inspect a class in CT Review.")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        title = QLabel("Reference Behaviour Library")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        subtitle = QLabel("Educational reference for behavior classes learned from synthetic gaze. No raw tables are shown here.")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        grid = QGridLayout()
        for index, (key, card_data) in enumerate(BEHAVIOR_CARDS.items()):
            card = QGroupBox(card_data["title"])
            card_layout = QVBoxLayout(card)
            for label, value in (
                ("Meaning", card_data["meaning"]),
                ("Expected gaze", card_data["pattern"]),
                ("Key features", card_data["features"]),
            ):
                text = QLabel(f"{label}: {value}")
                text.setWordWrap(True)
                card_layout.addWidget(text)
            button = QPushButton("Open representative replay")
            button.clicked.connect(lambda _checked=False, behavior=key: open_callback(behavior))
            card_layout.addWidget(button)
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
