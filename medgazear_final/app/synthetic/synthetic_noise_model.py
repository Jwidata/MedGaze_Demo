"""Synthetic gaze noise configuration helpers."""

from __future__ import annotations

from app.gaze.gaze_degradation_model import DegradationConfig
from app.synthetic.synthetic_reader_profiles import READER_PROFILES


def degradation_for_profile(profile_name: str) -> DegradationConfig:
    profile = READER_PROFILES[profile_name]
    return DegradationConfig(
        jitter_std_px=7.0 * profile.jitter_multiplier,
        dropout_probability=0.015 * profile.dropout_multiplier,
        blink_probability=0.006 * profile.dropout_multiplier,
        invalid_burst_probability=0.004 * profile.dropout_multiplier,
    )
