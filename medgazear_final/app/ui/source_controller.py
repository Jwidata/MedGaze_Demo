"""Gaze source state for the review workstation."""

from __future__ import annotations

from dataclasses import dataclass

from app.ui.ui_theme import TOBII_PLACEHOLDER_MESSAGE


SYNTHETIC_LABEL = "Synthetic replay"
TOBII_LABEL = "Tobii live"
SYNTHETIC_DESCRIPTION = "synthetic ROI-level replay/reference"


@dataclass
class SourceController:
    source: str = SYNTHETIC_LABEL

    @property
    def is_tobii(self) -> bool:
        return self.source == TOBII_LABEL

    @property
    def status_text(self) -> str:
        if self.is_tobii:
            return "Synthetic replay is reference only. Real Tobii gaze will later enter the same feature extractor and behavior model. Tobii SDK integration is planned in Step 13. Please calibrate in Tobii Manager before live capture."
        return f"Current source: {SYNTHETIC_DESCRIPTION}. Synthetic replay is reference only."


def source_from_cli(source: str) -> str:
    if source == "future_tobii_placeholder":
        raise ValueError(TOBII_PLACEHOLDER_MESSAGE)
    return SYNTHETIC_LABEL
