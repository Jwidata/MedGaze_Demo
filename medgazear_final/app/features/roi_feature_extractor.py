"""ROI coverage feature extraction from degraded gaze samples."""

from __future__ import annotations

import math


def roi_masks_for_samples(samples: list[dict[str, str]], roi: dict[str, str]) -> tuple[list[bool], list[bool], list[bool]]:
    x_min = float(roi["bbox_x_min"])
    y_min = float(roi["bbox_y_min"])
    x_max = float(roi["bbox_x_max"])
    y_max = float(roi["bbox_y_max"])
    margin = max(12.0, max(float(roi["bbox_width"]), float(roi["bbox_height"])) * 0.75)
    same_slice = [str(sample["slice_index"]) == str(roi["slice_index"]) for sample in samples]
    inside = []
    near = []
    for sample, on_slice in zip(samples, same_slice):
        x = float(sample["image_x"])
        y = float(sample["image_y"])
        in_roi = on_slice and x_min <= x <= x_max and y_min <= y <= y_max
        near_roi = on_slice and x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin and not in_roi
        inside.append(in_roi)
        near.append(near_roi)
    return inside, near, same_slice


def extract_roi_features(samples: list[dict[str, str]], roi: dict[str, str], fixations: list[dict[str, float | int]]) -> dict[str, float | int]:
    sample_ms = _sample_interval_ms(samples)
    inside, near, same_slice = roi_masks_for_samples(samples, roi)
    valid = [_bool(sample["is_valid"]) for sample in samples]
    valid_ct = [
        is_valid and not _bool(sample.get("is_outside_ct", "False")) and not _bool(sample.get("is_ui_glance", "False"))
        for sample, is_valid in zip(samples, valid)
    ]
    inside_valid = [hit and is_valid for hit, is_valid in zip(inside, valid_ct)]
    near_valid = [hit and is_valid for hit, is_valid in zip(near, valid_ct)]
    inside_fix = [fix for fix in fixations if _point_in_roi(float(fix["x"]), float(fix["y"]), roi, 0.0)]
    near_fix = [fix for fix in fixations if _point_in_roi(float(fix["x"]), float(fix["y"]), roi, max(12.0, max(float(roi["bbox_width"]), float(roi["bbox_height"])) * 0.75))]
    first_idx = next((idx for idx, hit in enumerate(inside_valid) if hit), None)
    return {
        "total_gaze_time_inside_roi_ms": sum(inside_valid) * sample_ms,
        "total_gaze_time_near_roi_ms": sum(near_valid) * sample_ms,
        "gaze_hit_count_inside_roi": int(sum(inside_valid)),
        "gaze_hit_count_near_roi": int(sum(near_valid)),
        "fixation_count_inside_roi": len(inside_fix),
        "fixation_count_near_roi": len(near_fix),
        "mean_fixation_duration_inside_roi_ms": _mean([float(fix["duration_ms"]) for fix in inside_fix]),
        "max_fixation_duration_inside_roi_ms": max([float(fix["duration_ms"]) for fix in inside_fix], default=0.0),
        "time_to_first_roi_fixation_ms": float(samples[first_idx]["timestamp_ms"]) if first_idx is not None else -1.0,
        "valid_gaze_time_on_roi_slice_ms": sum(hit and is_valid for hit, is_valid in zip(same_slice, valid_ct)) * sample_ms,
        "time_on_roi_slice_ms": sum(same_slice) * sample_ms,
    }


def _point_in_roi(x: float, y: float, roi: dict[str, str], margin: float) -> bool:
    return float(roi["bbox_x_min"]) - margin <= x <= float(roi["bbox_x_max"]) + margin and float(roi["bbox_y_min"]) - margin <= y <= float(roi["bbox_y_max"]) + margin


def _sample_interval_ms(samples: list[dict[str, str]]) -> float:
    if len(samples) < 2:
        return 0.0
    intervals = []
    for previous, current in zip(samples, samples[1:]):
        delta = float(current["timestamp_ms"]) - float(previous["timestamp_ms"])
        if delta > 0:
            intervals.append(delta)
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    return float(intervals[len(intervals) // 2])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"
