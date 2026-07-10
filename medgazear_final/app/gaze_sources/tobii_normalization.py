"""Tobii sample normalization into the shared gaze schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable


CoordinateMapper = Callable[[float, float], tuple[float, float, bool] | None]


@dataclass(frozen=True)
class EyeCombinationResult:
    x_norm: float | None
    y_norm: float | None
    valid: bool
    policy: str
    left_valid: bool
    right_valid: bool
    rejection_reason: str | None


def combine_eye_points(gaze_data: dict[str, Any]) -> EyeCombinationResult:
    left = gaze_data.get("left_gaze_point_on_display_area") or (None, None)
    right = gaze_data.get("right_gaze_point_on_display_area") or (None, None)
    left_valid = _usable_eye_coordinate(left, gaze_data.get("left_gaze_point_validity", 0))
    right_valid = _usable_eye_coordinate(right, gaze_data.get("right_gaze_point_validity", 0))
    if left_valid and right_valid:
        return EyeCombinationResult(
            x_norm=(float(left[0]) + float(right[0])) / 2.0,
            y_norm=(float(left[1]) + float(right[1])) / 2.0,
            valid=True,
            policy="both_valid_mean",
            left_valid=True,
            right_valid=True,
            rejection_reason=None,
        )
    if left_valid:
        return EyeCombinationResult(float(left[0]), float(left[1]), True, "left_only", True, False, None)
    if right_valid:
        return EyeCombinationResult(float(right[0]), float(right[1]), True, "right_only", False, True, None)
    return EyeCombinationResult(None, None, False, "neither_valid", False, False, "LEFT_INVALID_RIGHT_INVALID")


def canonicalize_tobii_sample(
    gaze_data: dict[str, Any],
    coordinate_mapper: CoordinateMapper | None = None,
    screen_geometry: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    eye = combine_eye_points(gaze_data)
    timestamp_ms = _tobii_timestamp_ms(gaze_data)
    invalid_reason = eye.rejection_reason
    image_x = None
    image_y = None
    is_outside_ct = False
    is_ui_glance = False
    screen_x = None
    screen_y = None
    normalized_in_range = False
    if eye.valid and eye.x_norm is not None and eye.y_norm is not None:
        normalized_in_range = 0.0 <= float(eye.x_norm) <= 1.0 and 0.0 <= float(eye.y_norm) <= 1.0
        if not normalized_in_range:
            invalid_reason = "OUTSIDE_NORMALIZED_RANGE"
    if eye.valid and eye.x_norm is not None and eye.y_norm is not None and screen_geometry is not None and normalized_in_range:
        sx, sy, sw, sh = screen_geometry
        screen_x = sx + float(eye.x_norm) * sw
        screen_y = sy + float(eye.y_norm) * sh
    elif eye.valid and normalized_in_range:
        screen_x = float(eye.x_norm)
        screen_y = float(eye.y_norm)
    elif eye.valid and screen_geometry is None:
        invalid_reason = invalid_reason or "SCREEN_MAPPING_FAILURE"
    if coordinate_mapper is not None and eye.valid and eye.x_norm is not None and eye.y_norm is not None and normalized_in_range:
        mapped = coordinate_mapper(float(eye.x_norm), float(eye.y_norm))
        if mapped is not None:
            image_x, image_y, is_outside_ct = mapped
            is_ui_glance = bool(is_outside_ct)
        else:
            invalid_reason = invalid_reason or "VIEWER_MAPPING_UNAVAILABLE"
    elif eye.valid and normalized_in_range and coordinate_mapper is None:
        invalid_reason = invalid_reason or "VIEWER_MAPPING_UNAVAILABLE"
    tracking_valid = bool(eye.valid and normalized_in_range)
    return {
        "source_type": "live_tobii",
        "timestamp_ms": timestamp_ms,
        "screen_x": screen_x,
        "screen_y": screen_y,
        "gaze_x_norm": eye.x_norm,
        "gaze_y_norm": eye.y_norm,
        "image_x": image_x,
        "image_y": image_y,
        "is_valid": tracking_valid,
        "is_dropout": not tracking_valid,
        "is_blink": not tracking_valid,
        "is_invalid_burst": False,
        "is_outside_ct": bool(is_outside_ct),
        "is_ui_glance": bool(is_ui_glance),
        "eye_policy": eye.policy,
        "left_eye_valid": eye.left_valid,
        "right_eye_valid": eye.right_valid,
        "invalid_reason": invalid_reason,
        "left_gaze_point_raw": gaze_data.get("left_gaze_point_on_display_area"),
        "right_gaze_point_raw": gaze_data.get("right_gaze_point_on_display_area"),
        "device_timestamp_us": gaze_data.get("device_time_stamp"),
        "system_timestamp_us": gaze_data.get("system_time_stamp"),
        "left_gaze_point_validity_raw": gaze_data.get("left_gaze_point_validity"),
        "right_gaze_point_validity_raw": gaze_data.get("right_gaze_point_validity"),
        "left_gaze_origin_validity_raw": gaze_data.get("left_gaze_origin_validity"),
        "right_gaze_origin_validity_raw": gaze_data.get("right_gaze_origin_validity"),
        "left_pupil_validity_raw": gaze_data.get("left_pupil_validity"),
        "right_pupil_validity_raw": gaze_data.get("right_pupil_validity"),
    }


def live_timing_diagnostics(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "effective_sampling_rate_hz": 0.0,
            "median_interval_ms": 0.0,
            "mean_interval_ms": 0.0,
            "interval_std_ms": 0.0,
            "large_gap_count": 0,
            "invalid_sample_ratio": 0.0,
        }
    timestamps = [float(sample.get("timestamp_ms", 0) or 0) for sample in samples]
    intervals = [max(0.0, b - a) for a, b in zip(timestamps, timestamps[1:])]
    invalid_ratio = sum(not bool(sample.get("is_valid", False)) for sample in samples) / len(samples)
    if not intervals:
        return {
            "sample_count": len(samples),
            "effective_sampling_rate_hz": 0.0,
            "median_interval_ms": 0.0,
            "mean_interval_ms": 0.0,
            "interval_std_ms": 0.0,
            "large_gap_count": 0,
            "invalid_sample_ratio": invalid_ratio,
        }
    mean_interval = sum(intervals) / len(intervals)
    variance = sum((interval - mean_interval) ** 2 for interval in intervals) / len(intervals)
    sorted_intervals = sorted(intervals)
    median_interval = sorted_intervals[len(sorted_intervals) // 2]
    effective_rate = 1000.0 / mean_interval if mean_interval > 0 else 0.0
    large_gap_count = sum(interval > max(100.0, mean_interval * 3.0) for interval in intervals)
    return {
        "sample_count": len(samples),
        "effective_sampling_rate_hz": effective_rate,
        "median_interval_ms": median_interval,
        "mean_interval_ms": mean_interval,
        "interval_std_ms": variance**0.5,
        "large_gap_count": large_gap_count,
        "invalid_sample_ratio": invalid_ratio,
    }


def _tobii_timestamp_ms(gaze_data: dict[str, Any]) -> float:
    device_time = gaze_data.get("device_time_stamp")
    if device_time is not None:
        return float(device_time) / 1000.0
    system_time = gaze_data.get("system_time_stamp")
    if system_time is not None:
        return float(system_time) / 1000.0
    return time.time_ns() / 1_000_000.0


def _usable_eye_coordinate(point: Any, validity: Any) -> bool:
    try:
        valid_flag = int(validity or 0) == 1
        x = point[0] if point is not None else None
        y = point[1] if point is not None else None
        return bool(valid_flag and x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)))
    except Exception:
        return False
