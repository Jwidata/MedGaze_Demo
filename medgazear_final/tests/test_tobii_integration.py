from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.gaze_sources import tobii_status
from app.gaze_sources.live_validation import tracking_preflight_summary, write_live_validation_bundle
from app.gaze_sources.tobii_normalization import canonicalize_tobii_sample, combine_eye_points, live_timing_diagnostics
from app.gaze_sources.tobii_live_source import TobiiLiveSource


def test_tobii_status_payload_schema_is_stable() -> None:
    payload = tobii_status.get_status_payload()

    assert set(payload) == {
        "sdk_available",
        "device_connected",
        "device_count",
        "device_model",
        "device_address",
        "device_serial",
        "status_label",
        "device_state",
        "calibration_reminder",
        "error",
    }


def test_eye_combination_policy() -> None:
    both = combine_eye_points(
        {
            "left_gaze_point_on_display_area": (0.2, 0.4),
            "right_gaze_point_on_display_area": (0.4, 0.6),
            "left_gaze_point_validity": 1,
            "right_gaze_point_validity": 1,
        }
    )
    assert both.valid is True
    assert both.policy == "both_valid_mean"
    assert both.x_norm == 0.30000000000000004
    left_only = combine_eye_points(
        {
            "left_gaze_point_on_display_area": (0.2, 0.4),
            "right_gaze_point_on_display_area": (0.4, 0.6),
            "left_gaze_point_validity": 1,
            "right_gaze_point_validity": 0,
        }
    )
    assert left_only.policy == "left_only"
    neither = combine_eye_points({"left_gaze_point_validity": 0, "right_gaze_point_validity": 0})
    assert neither.valid is False
    assert neither.policy == "neither_valid"


def test_tobii_canonical_sample_normalization_and_mapping() -> None:
    sample = canonicalize_tobii_sample(
        {
            "device_time_stamp": 123456,
            "left_gaze_point_on_display_area": (0.25, 0.5),
            "right_gaze_point_on_display_area": (0.35, 0.5),
            "left_gaze_point_validity": 1,
            "right_gaze_point_validity": 1,
        },
        coordinate_mapper=lambda x, y: (100.0, 120.0, False),
        screen_geometry=(10, 20, 1000, 800),
    )
    assert sample["timestamp_ms"] == 123.456
    assert sample["screen_x"] == 310.0
    assert sample["screen_y"] == 420.0
    assert sample["image_x"] == 100.0
    assert sample["image_y"] == 120.0
    assert sample["is_valid"] is True


def test_invalid_tobii_sample_and_outside_ct_handling() -> None:
    invalid = canonicalize_tobii_sample({"left_gaze_point_validity": 0, "right_gaze_point_validity": 0})
    assert invalid["is_valid"] is False
    assert invalid["is_dropout"] is True
    outside = canonicalize_tobii_sample(
        {
            "left_gaze_point_on_display_area": (0.5, 0.5),
            "left_gaze_point_validity": 1,
        },
        coordinate_mapper=lambda x, y: (50.0, 50.0, True),
        screen_geometry=(0, 0, 100, 100),
    )
    assert outside["is_outside_ct"] is True
    assert outside["is_ui_glance"] is True
    assert outside["is_valid"] is True


def test_preflight_summary_distinguishes_ready_and_mapping_unavailable() -> None:
    ready = tracking_preflight_summary(pd.DataFrame([
        {"is_valid": True, "image_x": 10.0, "image_y": 12.0, "is_outside_ct": False},
        {"is_valid": True, "image_x": 11.0, "image_y": 12.5, "is_outside_ct": False},
    ]))
    mapping_fail = tracking_preflight_summary(pd.DataFrame([
        {"is_valid": True, "image_x": 10.0, "image_y": 12.0, "is_outside_ct": True},
        {"is_valid": True, "image_x": 11.0, "image_y": 12.5, "is_outside_ct": True},
    ]))
    assert ready["message"] == "Ready to start"
    assert ready["failure_kind"] == ""
    assert mapping_fail["message"] == "CT mapping unavailable"
    assert mapping_fail["failure_kind"] == "mapping_unavailable"


def test_live_timing_diagnostics() -> None:
    diagnostics = live_timing_diagnostics(
        [
            {"timestamp_ms": 0.0, "is_valid": True},
            {"timestamp_ms": 16.0, "is_valid": True},
            {"timestamp_ms": 32.0, "is_valid": False},
            {"timestamp_ms": 100.0, "is_valid": True},
        ]
    )
    assert diagnostics["sample_count"] == 4
    assert diagnostics["mean_interval_ms"] > 0
    assert diagnostics["invalid_sample_ratio"] == 0.25


def test_sdk_missing_path_does_not_crash() -> None:
    source = TobiiLiveSource()

    status = source.get_status()
    payload = source.get_status_payload()

    assert status.source_type == "live_tobii"
    assert isinstance(payload["sdk_available"], bool)


def test_live_source_reports_unavailable_if_sdk_missing() -> None:
    source = TobiiLiveSource()

    if source.is_sdk_available():
        return

    source.start_stream(lambda _sample: None)
    status = source.get_status()
    payload = source.get_status_payload()

    assert source.streaming is False
    assert payload["sdk_available"] is False
    assert payload["status_label"] == tobii_status.SDK_MISSING_MESSAGE
    assert status.connected is False


def test_no_fake_samples_generated_when_sdk_missing() -> None:
    source = TobiiLiveSource()
    received: list[dict[str, object]] = []

    if source.is_sdk_available():
        return

    source.start_stream(received.append)

    assert received == []


def test_live_validation_bundle_writes_machine_readable_outputs(tmp_path: Path) -> None:
    summary = write_live_validation_bundle(
        tmp_path,
        session_id="session_1",
        device_payload={"device_state": "NO_DEVICE", "status_label": "SDK missing"},
        samples=[
            {"session_id": "session_1", "roi_id": "ROI1", "timestamp_ms": 0.0, "image_x": 10.0, "image_y": 10.0, "slice_index": 0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0},
            {"session_id": "session_1", "roi_id": "ROI1", "timestamp_ms": 100.0, "image_x": 11.0, "image_y": 11.0, "slice_index": 0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0},
        ],
        roi_rows=[{"roi_id": "ROI1", "slice_index": 0, "ct_stack_index": 0, "rows": 64, "columns": 64, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "bbox_width": 12, "bbox_height": 12, "centroid_x": 14, "centroid_y": 14, "mask_area_px": 144, "ct_series_instance_uid": "SERIES1"}],
        feature_columns=["slice_index", "total_gaze_time_inside_roi_ms", "total_gaze_time_near_roi_ms", "gaze_hit_count_inside_roi", "gaze_hit_count_near_roi", "fixation_count_inside_roi", "fixation_count_near_roi", "mean_fixation_duration_inside_roi_ms", "max_fixation_duration_inside_roi_ms", "time_to_first_roi_fixation_ms", "valid_gaze_time_on_roi_slice_ms", "time_on_roi_slice_ms", "scanpath_length_px", "scanpath_length_on_roi_slice_px", "gaze_dispersion_px", "gaze_entropy", "number_of_gaze_clusters", "background_gaze_ratio", "roi_revisit_count", "near_roi_revisit_count", "slice_transition_count", "adjacent_slice_toggle_count", "scroll_event_count", "search_to_confirmation_ratio", "late_roi_discovery_flag", "mean_fixation_duration_ms", "max_fixation_duration_ms", "fixation_duration_variance", "saccade_like_ratio", "fixation_like_ratio", "first_half_roi_attention_ratio", "second_half_roi_attention_ratio", "delayed_attention_score", "gaze_validity_ratio", "dropout_ratio", "blink_ratio", "invalid_burst_ratio", "outside_ct_ratio", "jitter_px", "roi_area_px", "roi_bbox_width", "roi_bbox_height", "roi_center_x", "roi_center_y", "normalized_roi_position_x", "normalized_roi_position_y", "number_of_rois_on_slice", "roi_density_context"],
        current_state_store={"ROI1": {"total_gaze_time_inside_roi_ms": 100.0, "total_gaze_time_near_roi_ms": 0.0, "gaze_hit_count_inside_roi": 2, "gaze_hit_count_near_roi": 0, "fixation_count_inside_roi": 0, "fixation_count_near_roi": 0, "mean_fixation_duration_inside_roi_ms": 0.0, "max_fixation_duration_inside_roi_ms": 0.0, "time_to_first_roi_fixation_ms": 0.0, "valid_gaze_time_on_roi_slice_ms": 100.0, "time_on_roi_slice_ms": 100.0, "scanpath_length_px": 1.41421356237, "scanpath_length_on_roi_slice_px": 1.41421356237, "gaze_dispersion_px": 1.0, "gaze_entropy": 1.0, "number_of_gaze_clusters": 1, "background_gaze_ratio": 0.0, "roi_revisit_count": 0, "near_roi_revisit_count": 0, "slice_transition_count": 0, "adjacent_slice_toggle_count": 0, "scroll_event_count": 0, "search_to_confirmation_ratio": 0.0, "late_roi_discovery_flag": 0, "mean_fixation_duration_ms": 0.0, "max_fixation_duration_ms": 0.0, "fixation_duration_variance": 0.0, "saccade_like_ratio": 0.0, "fixation_like_ratio": 0.0, "first_half_roi_attention_ratio": 1.0, "second_half_roi_attention_ratio": 1.0, "delayed_attention_score": 0.0, "gaze_validity_ratio": 1.0, "dropout_ratio": 0.0, "blink_ratio": 0.0, "invalid_burst_ratio": 0.0, "outside_ct_ratio": 0.0, "jitter_px": 0.0, "roi_area_px": 144.0, "roi_bbox_width": 12.0, "roi_bbox_height": 12.0, "roi_center_x": 14.0, "roi_center_y": 14.0, "normalized_roi_position_x": 0.2, "normalized_roi_position_y": 0.2, "number_of_rois_on_slice": 1, "roi_density_context": 1.0}},
    )
    assert summary["roi_count"] == 1
    assert (tmp_path / "tobii_device_audit.json").exists()
    assert (tmp_path / "live_sample_quality_summary.json").exists()
    assert (tmp_path / "live_timing_diagnostics.csv").exists()
    assert (tmp_path / "live_feature_readiness_summary.csv").exists()
    assert (tmp_path / "live_vs_replay_feature_parity.csv").exists()
    assert (tmp_path / "session_metadata.json").exists()


def test_live_validation_bundle_writes_mapping_failure_diagnostics_when_provided(tmp_path: Path) -> None:
    write_live_validation_bundle(
        tmp_path,
        session_id="session_2",
        device_payload={"device_state": "CONNECTED", "status_label": "Connected"},
        samples=[],
        roi_rows=[],
        feature_columns=[],
        current_state_store={},
        mapping_diagnostics=[{"gaze_x_norm": 0.5, "gaze_y_norm": 0.5, "mapping_failure_reason": "OUTSIDE_IMAGE_RECT"}],
    )
    assert (tmp_path / "mapping_failure_diagnostics.csv").exists()
