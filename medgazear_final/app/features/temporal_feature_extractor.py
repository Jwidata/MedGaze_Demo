"""Temporal and fixation-like feature extraction."""

from __future__ import annotations

import math


def detect_fixations(samples: list[dict[str, str]], velocity_threshold_px: float = 35.0, min_duration_ms: float = 50.0) -> list[dict[str, float | int]]:
    if not samples:
        return []
    fixations: list[dict[str, float | int]] = []
    start = 0
    for idx in range(1, len(samples)):
        distance = math.dist((float(samples[idx - 1]["image_x"]), float(samples[idx - 1]["image_y"])), (float(samples[idx]["image_x"]), float(samples[idx]["image_y"])))
        if distance > velocity_threshold_px or not _bool(samples[idx]["is_valid"]):
            _append_fixation(samples, start, idx - 1, min_duration_ms, fixations)
            start = idx
    _append_fixation(samples, start, len(samples) - 1, min_duration_ms, fixations)
    return fixations


def extract_temporal_features(samples: list[dict[str, str]], inside_roi: list[bool], fixations: list[dict[str, float | int]]) -> dict[str, float | int]:
    durations = [float(fix["duration_ms"]) for fix in fixations]
    movements = [math.dist((float(a["image_x"]), float(a["image_y"])), (float(b["image_x"]), float(b["image_y"]))) for a, b in zip(samples, samples[1:])]
    half = max(1, len(inside_roi) // 2)
    first_roi = sum(inside_roi[:half]) / half if inside_roi else 0.0
    second_roi = sum(inside_roi[half:]) / max(1, len(inside_roi) - half) if inside_roi else 0.0
    saccade_count = sum(move > 35.0 for move in movements)
    return {
        "mean_fixation_duration_ms": _mean(durations),
        "max_fixation_duration_ms": max(durations, default=0.0),
        "fixation_duration_variance": _variance(durations),
        "saccade_like_ratio": saccade_count / len(movements) if movements else 0.0,
        "fixation_like_ratio": len(fixations) / max(1, len(samples)),
        "first_half_roi_attention_ratio": first_roi,
        "second_half_roi_attention_ratio": second_roi,
        "delayed_attention_score": max(0.0, second_roi - first_roi),
    }


def _append_fixation(samples: list[dict[str, str]], start: int, end: int, min_duration_ms: float, fixations: list[dict[str, float | int]]) -> None:
    if end < start:
        return
    duration = float(samples[end]["timestamp_ms"]) - float(samples[start]["timestamp_ms"])
    if duration >= min_duration_ms:
        points = samples[start : end + 1]
        fixations.append({"start_index": start, "end_index": end, "duration_ms": duration, "x": _mean([float(p["image_x"]) for p in points]), "y": _mean([float(p["image_y"]) for p in points])})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"
