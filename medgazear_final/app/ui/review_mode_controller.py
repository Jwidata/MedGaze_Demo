"""Review mode to overlay/feedback mapping."""

from __future__ import annotations

from dataclasses import dataclass

from app.ui.review_mode import ReviewMode


@dataclass
class ReviewModeState:
    mode: str = ReviewMode.SILENT.value

    def layers(self) -> dict[str, bool]:
        if self.mode == ReviewMode.SILENT.value:
            return {"roi": True, "heatmap": False, "gaze_points": False, "scanpath": False}
        if self.mode == ReviewMode.AMBIENT.value:
            return {"roi": True, "heatmap": True, "gaze_points": False, "scanpath": False}
        if self.mode == ReviewMode.FEEDBACK.value:
            return {"roi": True, "heatmap": False, "gaze_points": False, "scanpath": False}
        return {"roi": True, "heatmap": True, "gaze_points": True, "scanpath": False}

    @property
    def show_prediction_feedback(self) -> bool:
        return self.mode != ReviewMode.SILENT.value

    @property
    def feedback_colors(self) -> bool:
        return self.mode == ReviewMode.FEEDBACK.value
