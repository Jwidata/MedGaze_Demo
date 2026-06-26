from __future__ import annotations

from app.features.temporal_feature_extractor import detect_fixations, extract_temporal_features


def test_temporal_feature_extractor_detects_fixation_and_ratios() -> None:
    samples = [_sample(0, 10, 10), _sample(50, 11, 10), _sample(100, 12, 11), _sample(150, 100, 100)]
    fixations = detect_fixations(samples, velocity_threshold_px=10, min_duration_ms=50)
    features = extract_temporal_features(samples, [False, True, True, False], fixations)

    assert len(fixations) == 1
    assert features["mean_fixation_duration_ms"] == 100
    assert features["saccade_like_ratio"] > 0
    assert features["first_half_roi_attention_ratio"] == 0.5
    assert features["second_half_roi_attention_ratio"] == 0.5


def _sample(timestamp: float, x: float, y: float) -> dict[str, str]:
    return {"timestamp_ms": str(timestamp), "image_x": str(x), "image_y": str(y), "is_valid": "True"}
