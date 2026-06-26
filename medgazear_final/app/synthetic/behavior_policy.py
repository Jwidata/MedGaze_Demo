"""Policies mapping reader profiles to hidden behavior labels."""

from __future__ import annotations

import numpy as np

from app.synthetic.behavior_label_schema import HIDDEN_BEHAVIOR_LABELS, validate_hidden_label


PROFILE_LABEL_WEIGHTS = {
    "expert_systematic": [0.62, 0.22, 0.06, 0.03, 0.02, 0.05],
    "fast_confirmer": [0.12, 0.58, 0.12, 0.05, 0.05, 0.08],
    "partial_reviewer": [0.12, 0.18, 0.44, 0.12, 0.07, 0.07],
    "noisy_searcher": [0.08, 0.12, 0.18, 0.38, 0.10, 0.14],
    "high_load_reader": [0.08, 0.12, 0.14, 0.14, 0.12, 0.40],
    "distracted_reader": [0.05, 0.10, 0.16, 0.22, 0.34, 0.13],
}


BEHAVIOR_DURATION_RANGES_MS = {
    "focused_roi_confirmation": (2000, 7000),
    "expert_like_systematic_review": (6000, 18000),
    "partial_near_miss_review": (4000, 14000),
    "missed_roi_search": (3000, 12000),
    "skipped_slice": (500, 4000),
    "high_load_fragmented_review": (7000, 22000),
}

BEHAVIOR_NEIGHBORS = {
    "focused_roi_confirmation": ("expert_like_systematic_review", "partial_near_miss_review"),
    "expert_like_systematic_review": ("focused_roi_confirmation", "high_load_fragmented_review"),
    "partial_near_miss_review": ("missed_roi_search", "focused_roi_confirmation"),
    "missed_roi_search": ("partial_near_miss_review", "skipped_slice"),
    "skipped_slice": ("missed_roi_search",),
    "high_load_fragmented_review": ("partial_near_miss_review", "expert_like_systematic_review"),
}


def choose_hidden_behavior_label(reader_profile: str, rng: np.random.Generator) -> str:
    weights = PROFILE_LABEL_WEIGHTS[reader_profile]
    label = str(rng.choice(HIDDEN_BEHAVIOR_LABELS, p=np.asarray(weights, dtype=float)))
    validate_hidden_label(label)
    return label


def choose_behavior_duration_ms(label: str, reader_profile: str, rng: np.random.Generator) -> int:
    """Choose duration primarily from hidden behavior, lightly adjusted by reader profile."""

    low, high = BEHAVIOR_DURATION_RANGES_MS[label]
    base_duration = float(rng.integers(low, high + 1))
    profile_multiplier = {
        "expert_systematic": 1.08,
        "fast_confirmer": 0.88,
        "partial_reviewer": 1.0,
        "noisy_searcher": 1.04,
        "high_load_reader": 1.12,
        "distracted_reader": 0.92,
    }[reader_profile]
    return int(round(np.clip(base_duration * profile_multiplier, low, high)))


def choose_behavior_blend(label: str, rng: np.random.Generator, probability: float = 0.32) -> tuple[str, float]:
    """Return an effective behavior template and blend weight while preserving the hidden label."""

    if rng.random() >= probability:
        return label, 0.0
    neighbor = str(rng.choice(BEHAVIOR_NEIGHBORS[label]))
    return neighbor, float(rng.uniform(0.35, 0.85))


def choose_blended_behavior_duration_ms(label: str, blend_label: str, blend_weight: float, reader_profile: str, rng: np.random.Generator) -> int:
    base = choose_behavior_duration_ms(label, reader_profile, rng)
    if blend_label == label or blend_weight <= 0:
        return base
    neighbor = choose_behavior_duration_ms(blend_label, reader_profile, rng)
    low, high = BEHAVIOR_DURATION_RANGES_MS[label]
    # Allow ambiguity beyond the nominal label range without making duration a perfect classifier.
    expanded_low = min(low, BEHAVIOR_DURATION_RANGES_MS[blend_label][0])
    expanded_high = max(high, BEHAVIOR_DURATION_RANGES_MS[blend_label][1])
    return int(round(np.clip(base * (1 - blend_weight) + neighbor * blend_weight, expanded_low, expanded_high)))


def label_focus_strength(label: str) -> float:
    return {
        "expert_like_systematic_review": 0.60,
        "focused_roi_confirmation": 0.66,
        "partial_near_miss_review": 0.43,
        "missed_roi_search": 0.33,
        "skipped_slice": 0.20,
        "high_load_fragmented_review": 0.41,
    }[label]
