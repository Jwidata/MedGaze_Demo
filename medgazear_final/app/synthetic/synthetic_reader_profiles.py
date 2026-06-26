"""Synthetic reader profile definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderProfile:
    name: str
    roi_focus_probability: float
    jitter_multiplier: float
    dropout_multiplier: float


READER_PROFILES = {
    "expert_systematic": ReaderProfile("expert_systematic", 0.88, 0.75, 0.7),
    "fast_confirmer": ReaderProfile("fast_confirmer", 0.82, 0.9, 0.9),
    "partial_reviewer": ReaderProfile("partial_reviewer", 0.58, 1.1, 1.1),
    "noisy_searcher": ReaderProfile("noisy_searcher", 0.40, 1.5, 1.3),
    "high_load_reader": ReaderProfile("high_load_reader", 0.52, 1.35, 1.5),
    "distracted_reader": ReaderProfile("distracted_reader", 0.35, 1.7, 1.8),
}


def profile_names() -> tuple[str, ...]:
    return tuple(READER_PROFILES.keys())
