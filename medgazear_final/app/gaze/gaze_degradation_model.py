"""Shared Tobii-like gaze degradation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationConfig:
    jitter_std_px: float = 7.0
    calibration_std_px: float = 18.0
    drift_std_px_per_sample: float = 0.06
    dropout_probability: float = 0.015
    blink_probability: float = 0.006
    invalid_burst_probability: float = 0.004
    outside_ct_probability: float = 0.025
    ui_glance_probability: float = 0.035


def apply_gaze_degradation(
    screen_x: float,
    screen_y: float,
    sample_index: int,
    rng: np.random.Generator,
    screen_width: int,
    screen_height: int,
    ct_bounds: tuple[float, float, float, float],
    calibration_offset: tuple[float, float],
    config: DegradationConfig | None = None,
) -> dict[str, float | bool]:
    """Apply shared degradation and clip screen coordinates to display bounds."""

    cfg = config or DegradationConfig()
    drift_x = float(rng.normal(0.0, cfg.drift_std_px_per_sample * sample_index))
    drift_y = float(rng.normal(0.0, cfg.drift_std_px_per_sample * sample_index))
    jitter_x = float(rng.normal(0.0, cfg.jitter_std_px))
    jitter_y = float(rng.normal(0.0, cfg.jitter_std_px))
    is_dropout = bool(rng.random() < cfg.dropout_probability)
    is_blink = bool(rng.random() < cfg.blink_probability)
    is_invalid_burst = bool(rng.random() < cfg.invalid_burst_probability)
    is_outside_ct = bool(rng.random() < cfg.outside_ct_probability)
    is_ui_glance = bool(rng.random() < cfg.ui_glance_probability)

    x = screen_x + calibration_offset[0] + drift_x + jitter_x
    y = screen_y + calibration_offset[1] + drift_y + jitter_y
    ct_x_min, ct_y_min, ct_x_max, ct_y_max = ct_bounds
    if is_outside_ct:
        x = float(rng.choice([rng.uniform(0, ct_x_min), rng.uniform(ct_x_max, screen_width)]))
        y = float(rng.uniform(0, screen_height))
    if is_ui_glance:
        x = float(rng.uniform(0, screen_width))
        y = float(rng.uniform(0, max(1.0, ct_y_min - 5)))

    x = float(np.clip(x, 0, screen_width - 1))
    y = float(np.clip(y, 0, screen_height - 1))
    return {
        "screen_x": x,
        "screen_y": y,
        "gaze_x_norm": x / screen_width,
        "gaze_y_norm": y / screen_height,
        "is_valid": not (is_dropout or is_blink or is_invalid_burst),
        "is_dropout": is_dropout,
        "is_blink": is_blink,
        "is_invalid_burst": is_invalid_burst,
        "is_outside_ct": is_outside_ct,
        "is_ui_glance": is_ui_glance,
        "calibration_offset_x": float(calibration_offset[0]),
        "calibration_offset_y": float(calibration_offset[1]),
        "drift_x": drift_x,
        "drift_y": drift_y,
        "jitter_x": jitter_x,
        "jitter_y": jitter_y,
    }
