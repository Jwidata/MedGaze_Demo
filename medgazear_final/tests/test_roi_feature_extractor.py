from __future__ import annotations

from app.features.roi_feature_extractor import extract_roi_features, roi_masks_for_samples
from app.features.temporal_feature_extractor import detect_fixations


def test_roi_feature_extractor_counts_inside_and_near_roi() -> None:
    roi = _roi()
    samples = [_sample(0, 12, 12), _sample(16.667, 15, 15), _sample(33.334, 28, 15), _sample(50.001, 80, 80)]
    inside, near, same_slice = roi_masks_for_samples(samples, roi)
    features = extract_roi_features(samples, roi, detect_fixations(samples, velocity_threshold_px=100, min_duration_ms=10))

    assert inside == [True, True, False, False]
    assert near == [False, False, True, False]
    assert all(same_slice)
    assert features["gaze_hit_count_inside_roi"] == 2
    assert features["gaze_hit_count_near_roi"] == 1
    assert features["total_gaze_time_inside_roi_ms"] > 0


def test_valid_roi_slice_time_excludes_outside_ct_and_ui_glances() -> None:
    roi = _roi()
    samples = [
        _sample(0, 12, 12),
        {**_sample(16.667, 12, 12), "is_outside_ct": "True"},
        {**_sample(33.334, 12, 12), "is_ui_glance": "True"},
    ]
    features = extract_roi_features(samples, roi, [])

    assert features["valid_gaze_time_on_roi_slice_ms"] == 16.667
    assert features["gaze_hit_count_inside_roi"] == 1


def _sample(timestamp: float, x: float, y: float) -> dict[str, str]:
    return {"timestamp_ms": str(timestamp), "image_x": str(x), "image_y": str(y), "slice_index": "1", "is_valid": "True", "is_outside_ct": "False", "is_ui_glance": "False"}


def _roi() -> dict[str, str]:
    return {"bbox_x_min": "10", "bbox_y_min": "10", "bbox_x_max": "20", "bbox_y_max": "20", "bbox_width": "11", "bbox_height": "11", "slice_index": "1"}
