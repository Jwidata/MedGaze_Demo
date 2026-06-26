"""Review mode definitions for the workstation."""

from __future__ import annotations

from enum import Enum


class ReviewMode(str, Enum):
    SILENT = "Silent"
    TRAINING = "Training"
    AMBIENT = "Ambient"
    FEEDBACK = "Feedback"


def default_layer_visibility(mode: str) -> dict[str, bool]:
    if mode == ReviewMode.SILENT.value:
        return {"roi": True, "heatmap": False, "gaze_points": False, "scanpath": False}
    if mode == ReviewMode.AMBIENT.value:
        return {"roi": True, "heatmap": True, "gaze_points": False, "scanpath": False}
    if mode == ReviewMode.FEEDBACK.value:
        return {"roi": True, "heatmap": True, "gaze_points": False, "scanpath": False}
    return {"roi": True, "heatmap": True, "gaze_points": True, "scanpath": True}
