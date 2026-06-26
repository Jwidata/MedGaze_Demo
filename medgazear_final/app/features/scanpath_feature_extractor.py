"""Scanpath and search feature extraction."""

from __future__ import annotations

import math


def extract_scanpath_features(samples: list[dict[str, str]], inside_roi: list[bool], near_roi: list[bool], same_slice: list[bool]) -> dict[str, float | int]:
    valid_points = [(float(s["image_x"]), float(s["image_y"])) for s in samples if _bool(s["is_valid"])]
    on_slice_points = [(float(s["image_x"]), float(s["image_y"])) for s, on_slice in zip(samples, same_slice) if on_slice and _bool(s["is_valid"])]
    transitions = sum(1 for a, b in zip(samples, samples[1:]) if str(a["slice_index"]) != str(b["slice_index"]))
    adjacent = sum(1 for a, b in zip(samples, samples[1:]) if abs(_int(a["slice_index"]) - _int(b["slice_index"])) == 1)
    background = sum(1 for in_roi, near in zip(inside_roi, near_roi) if not in_roi and not near)
    return {
        "scanpath_length_px": _path_length(valid_points),
        "scanpath_length_on_roi_slice_px": _path_length(on_slice_points),
        "gaze_dispersion_px": _dispersion(valid_points),
        "gaze_entropy": _entropy(valid_points),
        "number_of_gaze_clusters": _cluster_count(valid_points),
        "background_gaze_ratio": background / len(samples) if samples else 0.0,
        "roi_revisit_count": _revisit_count(inside_roi),
        "near_roi_revisit_count": _revisit_count(near_roi),
        "slice_transition_count": transitions,
        "adjacent_slice_toggle_count": adjacent,
        "scroll_event_count": transitions,
        "search_to_confirmation_ratio": background / max(1, sum(inside_roi)),
        "late_roi_discovery_flag": int(_first_true_fraction(inside_roi) > 0.5),
    }


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _dispersion(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    xs, ys = zip(*points)
    return (max(xs) - min(xs) + max(ys) - min(ys)) / 2


def _entropy(points: list[tuple[float, float]], bins: int = 4) -> float:
    if not points:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for x, y in points:
        key = (min(bins - 1, max(0, int(x / 512 * bins))), min(bins - 1, max(0, int(y / 512 * bins))))
        counts[key] = counts.get(key, 0) + 1
    total = len(points)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _cluster_count(points: list[tuple[float, float]], cell_px: int = 64) -> int:
    return len({(int(x // cell_px), int(y // cell_px)) for x, y in points})


def _revisit_count(mask: list[bool]) -> int:
    visits = 0
    previous = False
    for value in mask:
        if value and not previous:
            visits += 1
        previous = value
    return max(0, visits - 1)


def _first_true_fraction(mask: list[bool]) -> float:
    idx = next((index for index, value in enumerate(mask) if value), None)
    return 1.0 if idx is None or not mask else idx / len(mask)


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def _int(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0
