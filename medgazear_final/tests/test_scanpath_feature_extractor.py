from __future__ import annotations

from app.features.scanpath_feature_extractor import extract_scanpath_features


def test_scanpath_feature_extractor_computes_path_and_revisits() -> None:
    samples = [
        _sample(0, 0, 0, 1),
        _sample(1, 3, 4, 1),
        _sample(2, 300, 300, 2),
        _sample(3, 303, 304, 1),
    ]
    features = extract_scanpath_features(samples, [True, True, False, True], [False, False, True, False], [True, True, False, True])

    assert features["scanpath_length_px"] > 15
    assert features["roi_revisit_count"] == 1
    assert features["slice_transition_count"] == 2
    assert features["adjacent_slice_toggle_count"] == 2
    assert features["gaze_entropy"] > 0


def _sample(index: int, x: float, y: float, slice_index: int) -> dict[str, str]:
    return {"sample_index": str(index), "image_x": str(x), "image_y": str(y), "slice_index": str(slice_index), "is_valid": "True"}
