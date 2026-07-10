"""Prediction readiness and schema validation for behavior-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from app.features.behavior_feature_builder import validate_feature_schema


@dataclass(frozen=True)
class PredictionReadiness:
    status: str
    message: str
    missing_features: list[str]
    schema_duplicates: list[str]
    unexpected_features: list[str]


def assess_prediction_readiness(feature_row: Mapping[str, object], feature_columns: list[str]) -> PredictionReadiness:
    schema = validate_feature_schema(feature_row, feature_columns)
    if schema["duplicates"]:
        return PredictionReadiness("MISSING_REQUIRED_FEATURES", "Prediction unavailable: schema mismatch", schema["missing"], schema["duplicates"], schema["unexpected"])
    if schema["missing"]:
        return PredictionReadiness("MISSING_REQUIRED_FEATURES", "Prediction unavailable: feature mismatch", schema["missing"], schema["duplicates"], schema["unexpected"])

    valid_ratio = _float(feature_row.get("gaze_validity_ratio"))
    valid_time = _float(feature_row.get("valid_gaze_time_on_roi_slice_ms"))
    sample_count = int(_float(feature_row.get("_sample_count", 2)))
    fixation_ready = feature_row.get("_fixation_ready", True)

    if sample_count < 2 or valid_time <= 0:
        return PredictionReadiness("COLLECTING_EVIDENCE", "Collecting evidence...", [], [], schema["unexpected"])
    if valid_ratio <= 0:
        return PredictionReadiness("INVALID_GAZE", "Insufficient valid gaze", [], [], schema["unexpected"])
    if fixation_ready is False:
        return PredictionReadiness("COLLECTING_EVIDENCE", "Waiting for fixation evidence", [], [], schema["unexpected"])
    return PredictionReadiness("READY", "Ready", [], [], schema["unexpected"])


def _float(value: object) -> float:
    try:
        number = float(value)
        return 0.0 if pd.isna(number) else number
    except (TypeError, ValueError):
        return 0.0
