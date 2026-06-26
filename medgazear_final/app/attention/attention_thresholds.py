"""Configurable thresholds for the rule-based attention engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttentionThresholds:
    minimum_valid_time_on_roi_slice_ms: float = 500
    minimum_gaze_validity_ratio: float = 0.60
    reviewed_min_inside_roi_dwell_ms: float = 800
    reviewed_min_inside_hit_count: float = 8
    reviewed_min_fixation_count_inside_roi: float = 1
    weak_min_inside_roi_dwell_ms: float = 250
    weak_min_near_roi_dwell_ms: float = 700
    weak_min_near_hit_count: float = 8

    def scaled(self, factor: float) -> "AttentionThresholds":
        return AttentionThresholds(**{key: value * factor for key, value in asdict(self).items()})


def load_attention_thresholds(path: Path | None = None) -> AttentionThresholds:
    if path is None or not path.exists():
        return AttentionThresholds()
    values = asdict(AttentionThresholds())
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if key in values:
            values[key] = float(raw_value.strip())
    return AttentionThresholds(**values)
