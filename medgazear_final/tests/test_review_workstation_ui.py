from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from PyQt6.QtCore import Qt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.gaze_sources.synthetic_replay_source import canonicalize_synthetic_samples
from app.ui.overlay_layer_manager import OverlayLayerManager
from app.ui.session_browser import SessionSelectionModel
from app.ui.case_review_model import build_review_targets
from app.ui.ui_data_loader import load_workstation_data
from app.ui.ui_theme import TOBII_PLACEHOLDER_MESSAGE
from app.visualization.coordinate_mapper import ImageSpace
from app.visualization.heatmap_renderer import render_heatmap_layer
from app.visualization.scanpath_renderer import render_scanpath_layer
from app.ui.case_review_model import base_roi_id, build_case_review_model, build_review_targets
from app.ui_training import build_class_reference_baselines, build_reference_payload, smoke_test_workstation
from app.ui_training.main_window import MedGazeReviewWorkstation, SESSION_ACTIVE, SESSION_IDLE, SESSION_STOPPED, _count_roi_slices_viewed, _coverage_counts


@pytest.fixture()
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _settle_ui(qapp, ms: int = 70) -> None:
    from PyQt6.QtTest import QTest

    QTest.qWait(ms)
    qapp.processEvents()


def test_ui_data_loading(tmp_path: Path) -> None:
    _write_minimal_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    assert len(data.representative_cases) == 1
    assert len(data.gaze) == 8
    assert data.source == "synthetic"


def test_session_selection_next_previous_and_filter() -> None:
    sessions = pd.DataFrame(
        [
            {"session_id": "S1", "hidden_behavior_label": "focused_roi_confirmation"},
            {"session_id": "S2", "hidden_behavior_label": "missed_roi_search"},
        ]
    )
    model = SessionSelectionModel(sessions)
    assert model.current()["session_id"] == "S1"
    assert model.next()["session_id"] == "S2"
    assert model.previous()["session_id"] == "S1"
    assert model.current("missed_roi_search")["session_id"] == "S2"


def test_overlay_layer_state() -> None:
    manager = OverlayLayerManager()
    assert manager.visible["heatmap"] is True
    assert manager.toggle_layer("heatmap") is False
    with pytest.raises(ValueError):
        manager.set_layer("unknown", True)


def test_smoke_test_launch_without_data(tmp_path: Path) -> None:
    result = smoke_test_workstation(tmp_path, source="synthetic")
    assert result["status"] == "ok"
    assert result["sample_row_loaded"] is False


def test_tobii_placeholder_message() -> None:
    result = smoke_test_workstation(source="future_tobii_placeholder")
    assert result["status"] == "placeholder"
    assert TOBII_PLACEHOLDER_MESSAGE in result["message"]
    with pytest.raises(ValueError, match="Tobii live mode requires SDK integration"):
        load_workstation_data(source="future_tobii_placeholder")


def test_reference_payload_uses_same_roi_first(tmp_path: Path) -> None:
    _write_minimal_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    row = data.features.iloc[0].copy()
    row["predicted_behavior_label"] = "missed_roi_search"
    row["prediction_confidence"] = 0.25
    payload = build_reference_payload(row, data.features, build_class_reference_baselines(data), data)
    assert payload is not None
    assert payload["source"] == "same_roi"
    assert payload["behavior_label"] == "focused_roi_confirmation"


def test_reference_payload_falls_back_to_class_baseline(tmp_path: Path) -> None:
    _write_minimal_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    row = pd.Series({"roi_id": "missing", "session_id": "", "predicted_behavior_label": "focused_roi_confirmation"})
    payload = build_reference_payload(row, data.features, build_class_reference_baselines(data), data)
    assert payload is not None
    assert payload["source"] == "class_baseline"
    assert payload["behavior_label"] == "focused_roi_confirmation"


def test_canonicalize_synthetic_samples_handles_missing_norm_columns() -> None:
    samples = pd.DataFrame([
        {"image_x": 10.0, "image_y": 15.0, "screen_x": 0.1, "screen_y": 0.2, "is_valid": True}
    ])
    result = canonicalize_synthetic_samples(samples, ct_stack_index=2)
    assert float(result.iloc[0]["gaze_x_norm"]) == 0.0
    assert float(result.iloc[0]["gaze_y_norm"]) == 0.0
    assert int(result.iloc[0]["ct_stack_index"]) == 2


def test_case_load_starts_at_first_slice_with_no_selection_and_overlays_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    assert window.current_slice_index == 0
    assert window.viewer.current_slice_index == 0
    assert window.viewer.slice_label.text() == "1 / 3"
    assert window.selected_roi is None
    assert window.slice_worklist.currentRow() == -1
    assert window.session_state == SESSION_IDLE
    assert window.overlay_effective == {"roi": False, "gaze_points": False, "heatmap": False, "scanpath": False}
    window.close()


def test_silent_mode_conceals_targets_and_results_until_session_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    assert window.slice_worklist.isEnabled()
    first_label = window.slice_worklist.item(0).text()
    assert "Slice" in first_label
    window._synthetic_session_started()
    qapp.processEvents()
    assert window.session_state == SESSION_ACTIVE
    assert window.prediction_state_value.text() == "Prediction hidden during silent review"
    assert not any(window.overlay_effective.values())
    window._synthetic_session_stopped()
    qapp.processEvents()
    assert window.session_state == SESSION_STOPPED
    assert window.slice_worklist.isEnabled()
    window.close()


def test_navigation_sources_share_canonical_slice_setter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    qapp.processEvents()
    window.viewer.next_button.click()
    qapp.processEvents()
    assert window.current_slice_index == 1
    assert window.viewer.current_slice_index == 1
    assert window.viewer.slice_label.text() == "2 / 3"
    window.viewer.previous_button.click()
    qapp.processEvents()
    assert window.current_slice_index == 0
    assert window.viewer.slice_label.text() == "1 / 3"
    window.viewer.slice_slider.setValue(2)
    qapp.processEvents()
    assert window.current_slice_index == 2
    assert window.viewer.slice_label.text() == "3 / 3"
    window.viewer.previous_button.click()
    qapp.processEvents()
    assert window.current_slice_index == 1
    window.close()


def test_overlay_policy_restores_preferences_and_keeps_assisted_controls_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    qapp.processEvents()
    window.slice_worklist.setCurrentRow(0)
    qapp.processEvents()
    window.overlay_roi.setChecked(True)
    window.overlay_gaze.setChecked(True)
    qapp.processEvents()
    assert window.overlay_preferences["roi"] is True
    assert window.overlay_preferences["gaze_points"] is True
    assert window.overlay_effective["roi"] is True
    assert window.overlay_effective["gaze_points"] is True
    assert not window.viewer._selected_gaze_on_current_slice().empty
    window._synthetic_session_started()
    qapp.processEvents()
    assert window.overlay_roi.isEnabled() is True
    assert window.overlay_gaze.isEnabled() is True
    assert window.overlay_effective["roi"] is True
    assert window.overlay_effective["gaze_points"] is True
    window.mode_selector.setCurrentText("Silent")
    qapp.processEvents()
    assert window.overlay_roi.isEnabled() is False
    assert window.overlay_gaze.isEnabled() is False
    assert window.overlay_effective["roi"] is False
    assert window.overlay_effective["gaze_points"] is False
    window._synthetic_session_stopped()
    qapp.processEvents()
    assert window.overlay_roi.isEnabled() is True
    assert window.overlay_gaze.isEnabled() is True
    assert window.overlay_preferences["roi"] is True
    window.mode_selector.setCurrentText("Assisted")
    qapp.processEvents()
    window.slice_worklist.setCurrentRow(0)
    qapp.processEvents()
    assert window.overlay_effective["roi"] is True
    assert window.overlay_effective["gaze_points"] is True
    window.close()


def test_replay_follow_slice_does_not_fight_manual_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window.slice_worklist.setCurrentRow(0)
    qapp.processEvents()
    sample = {"ct_stack_index": 2, "timestamp_ms": 1000.0, "source_type": "synthetic"}
    window.timeline.follow_slice.setChecked(False)
    window.set_current_slice(0)
    window._timeline_sample_changed(sample)
    qapp.processEvents()
    assert window.current_slice_index == 0
    window.timeline.follow_slice.setChecked(True)
    window._timeline_sample_changed(sample)
    qapp.processEvents()
    assert window.current_slice_index == 2
    window.set_current_slice(0)
    window._begin_user_scrub()
    window._timeline_sample_changed(sample)
    qapp.processEvents()
    assert window.current_slice_index == 0
    window._end_user_scrub()
    window.close()


def test_forbidden_panels_remain_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    visible_text = "\n".join(_visible_label_texts(window))
    for forbidden in (
        "Data Loader",
        "Annotation Counts",
        "Viewer Diagnostics",
        "Viewer Notes",
        "Study Metadata",
        "Gaze Debug Panel",
        "Synthetic Reference Comparison",
        "Feedback Status",
        "Legend",
    ):
        assert forbidden not in visible_text
    window.close()


def test_left_panel_contains_four_required_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    _settle_ui(qapp)
    visible_text = "\n".join(_visible_label_texts(window))
    for required in ("Case Summary", "ROI Level Coverage", "Case Level Coverage", "Review Queue"):
        assert required in visible_text
    assert "ROI-bearing slices" not in visible_text
    window.close()


def test_primary_controls_are_not_clipped_across_window_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    for size in ((1366, 768), (1536, 864), (1920, 1080)):
        window.resize(*size)
        qapp.processEvents()
        _assert_widgets_inside_central_widget(
            window,
            [
                window.timeline.play_button,
                window.timeline.pause_button,
                window.timeline.reset_button,
                window.timeline.speed,
                window.timeline.follow_slice,
                window.viewer.previous_button,
                window.viewer.next_button,
                window.viewer.slice_slider,
                window.viewer.zoom_out_button,
                window.viewer.reset_view_button,
                window.viewer.zoom_in_button,
            ],
        )
    window.close()


def test_ct_image_viewport_and_navigation_do_not_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    _settle_ui(qapp)
    viewport_bottom = window.viewer.viewport_frame.mapToGlobal(window.viewer.viewport_frame.rect().bottomLeft()).y()
    navigation_top = window.viewer.findChild(type(window.viewer.previous_button), None).parentWidget().mapToGlobal(window.viewer.findChild(type(window.viewer.previous_button), None).parentWidget().rect().topLeft()).y()
    assert navigation_top >= viewport_bottom
    window.close()


def test_coverage_math_and_roi_slices_viewed_invariants() -> None:
    rows = pd.DataFrame(
        [
            {"finding_id": "R1", "ct_stack_index": 10, "display_review_state": "Sufficient"},
            {"finding_id": "R2", "ct_stack_index": 10, "display_review_state": "Weak"},
            {"finding_id": "R3", "ct_stack_index": 20, "display_review_state": "Pending"},
            {"finding_id": "R4", "ct_stack_index": 30, "display_review_state": "Insufficient"},
        ]
    )
    case_counts = _coverage_counts(rows)
    assert case_counts == {
        "ROI instances": 4,
        "Reviewed": 1,
        "Weakly reviewed": 1,
        "Missed": 1,
        "Not evaluated": 1,
    }
    current_slice_counts = _coverage_counts(rows[rows["ct_stack_index"] == 10].reset_index(drop=True))
    assert current_slice_counts == {
        "ROI instances": 2,
        "Reviewed": 1,
        "Weakly reviewed": 1,
        "Missed": 0,
        "Not evaluated": 0,
    }
    roi_slices_viewed = _count_roi_slices_viewed({10, 20, 30}, {10: {"status": "Reviewed"}, 20: {"status": "Unvisited"}, 30: {"status": "Briefly viewed"}})
    assert roi_slices_viewed == 2
    assert roi_slices_viewed <= 3


def test_raw_geometry_rows_vs_unique_targets_vs_target_bearing_slices() -> None:
    geometry = pd.DataFrame(
        [
            {"roi_id": "A__frame_0000", "ct_stack_index": 10},
            {"roi_id": "A__frame_0001", "ct_stack_index": 11},
            {"roi_id": "A__frame_0002", "ct_stack_index": 12},
            {"roi_id": "B__frame_0000", "ct_stack_index": 12},
            {"roi_id": "B__frame_0001", "ct_stack_index": 13},
            {"roi_id": "C__frame_0000", "ct_stack_index": 20},
        ]
    )
    geometry["target_id"] = geometry["roi_id"].astype(str).map(base_roi_id)
    assert len(geometry) == 6
    assert geometry["target_id"].nunique() == 3
    assert geometry["ct_stack_index"].nunique() == 5
    current_slice_targets = geometry[geometry["ct_stack_index"] == 12]["target_id"].nunique()
    assert current_slice_targets == 2


def test_overlapping_annotations_group_into_one_review_target() -> None:
    geometry = pd.DataFrame(
        [
            {"roi_id": "A1", "ct_stack_index": 90, "slice_index": 4, "bbox_x_min": 300, "bbox_y_min": 347, "bbox_x_max": 336, "bbox_y_max": 388, "centroid_x": 318.0, "centroid_y": 368.0},
            {"roi_id": "A2", "ct_stack_index": 90, "slice_index": 4, "bbox_x_min": 302, "bbox_y_min": 349, "bbox_x_max": 333, "bbox_y_max": 384, "centroid_x": 317.5, "centroid_y": 366.5},
            {"roi_id": "A3", "ct_stack_index": 90, "slice_index": 4, "bbox_x_min": 301, "bbox_y_min": 348, "bbox_x_max": 334, "bbox_y_max": 386, "centroid_x": 317.2, "centroid_y": 367.0},
            {"roi_id": "A4", "ct_stack_index": 90, "slice_index": 4, "bbox_x_min": 302, "bbox_y_min": 348, "bbox_x_max": 335, "bbox_y_max": 387, "centroid_x": 318.0, "centroid_y": 367.0},
        ]
    )
    targets = build_review_targets(geometry)
    assert len(targets) == 1
    assert int(targets.iloc[0]["annotation_instance_count"]) == 4


def test_duplicate_rows_do_not_inflate_target_count() -> None:
    geometry = pd.DataFrame(
        [
            {"roi_id": "D1", "ct_stack_index": 10, "slice_index": 1, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14.0, "centroid_y": 14.0},
            {"roi_id": "D2", "ct_stack_index": 10, "slice_index": 1, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14.0, "centroid_y": 14.0},
        ]
    )
    targets = build_review_targets(geometry)
    assert len(targets) == 1


def test_frame_roi_case_and_slice_coverage_math() -> None:
    rows = pd.DataFrame(
        [
            {"roi_id": "A__frame_0000", "base_roi_id": "A", "ct_stack_index": 10, "display_review_state": "Sufficient"},
            {"roi_id": "A__frame_0001", "base_roi_id": "A", "ct_stack_index": 11, "display_review_state": "Weak"},
            {"roi_id": "A__frame_0002", "base_roi_id": "A", "ct_stack_index": 12, "display_review_state": "Pending"},
            {"roi_id": "B__frame_0000", "base_roi_id": "B", "ct_stack_index": 11, "display_review_state": "Insufficient"},
        ]
    )
    case_counts = _coverage_counts(rows)
    assert case_counts == {"ROI instances": 4, "Reviewed": 1, "Weakly reviewed": 1, "Missed": 1, "Not evaluated": 1}
    current_slice_counts = _coverage_counts(rows[rows["ct_stack_index"] == 11].reset_index(drop=True))
    assert current_slice_counts == {"ROI instances": 2, "Reviewed": 0, "Weakly reviewed": 1, "Missed": 1, "Not evaluated": 0}


def test_frame_state_isolation_and_deterministic_display_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.show()
    qapp.processEvents()
    rows = window._queue_rows()
    assert rows["roi_id"].tolist() == ["A__frame_0000", "A__frame_0001", "B__frame_0000", "A__frame_0002"]
    assert rows["display_index"].tolist() == ["ROI 01", "ROI 02", "ROI 03", "ROI 04"]
    window.roi_state_store["A__frame_0000"]["mapped_review_status"] = "reviewed"
    window.roi_state_store["A__frame_0001"]["mapped_review_status"] = "weakly_reviewed"
    window.roi_state_store["A__frame_0002"]["mapped_review_status"] = "not_evaluated"
    window.roi_state_store["B__frame_0000"]["mapped_review_status"] = "not_reviewed"
    assert window.roi_state_store["A__frame_0000"]["mapped_review_status"] != window.roi_state_store["A__frame_0001"]["mapped_review_status"]
    assert window.roi_state_store["A__frame_0001"]["mapped_review_status"] != window.roi_state_store["A__frame_0002"]["mapped_review_status"]
    window._sync_case_model_roi_states()
    window._refresh_case_progress(window._queue_rows())
    assert window.progress_total.text() == "4"
    window.set_current_slice(11, record_visit=False)
    _settle_ui(qapp)
    assert window.current_slice_target_count.text() == "2"
    window.close()


def test_every_roi_bearing_slice_is_displayed_and_count_matches_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    backend_count = int(data.roi_geometry["ct_stack_index"].astype(int).nunique())
    displayed_slices = []
    for index in range(window.slice_worklist.count()):
        item = window.slice_worklist.item(index)
        displayed_slices.append(int(item.data(Qt.ItemDataRole.UserRole)))
    assert len(displayed_slices) == backend_count
    assert sorted(displayed_slices) == sorted(data.roi_geometry["ct_stack_index"].astype(int).unique().tolist())
    window.close()


def test_unvisited_roi_bearing_slices_are_not_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    labels = [window.slice_worklist.item(index).text() for index in range(window.slice_worklist.count())]
    assert any("Not visited" in label for label in labels)
    assert window.slice_worklist.count() == 3
    window.close()


def test_review_queue_is_one_row_per_roi_bearing_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    assert window.slice_worklist.count() == 3
    labels = [window.slice_worklist.item(index).text() for index in range(window.slice_worklist.count())]
    assert all("Slice " in label for label in labels)
    assert all("ROI" in label for label in labels)
    window.close()


def test_current_slice_roi_selector_lists_each_roi_once_on_current_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    window.set_current_slice(11, record_visit=False)
    _settle_ui(qapp)
    roi_ids = [str(window.current_slice_roi_selector.itemData(i)) for i in range(window.current_slice_roi_selector.count()) if window.current_slice_roi_selector.itemData(i)]
    assert sorted(roi_ids) == ["A__frame_0001", "B__frame_0000"]
    window.close()


def test_slice_rows_navigate_and_selecting_current_slice_roi_updates_contour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    target_row = -1
    for index in range(window.slice_worklist.count()):
        item = window.slice_worklist.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == 11:
            target_row = index
            break
    assert target_row >= 0
    window.slice_worklist.setCurrentRow(target_row)
    _settle_ui(qapp)
    assert window.current_slice_index == 11
    roi_index = window.current_slice_roi_selector.findData("B__frame_0000")
    assert roi_index >= 0
    window.current_slice_roi_selector.setCurrentIndex(roi_index)
    _settle_ui(qapp)
    assert str(window.selected_roi.get("roi_id", "")) == "B__frame_0000"
    assert str(window.viewer.selected_row.get("roi_id", "")) == "B__frame_0000"
    window.close()


def test_slice_and_case_coverage_sums_are_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_frame_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_frame_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    total = int(window.progress_total.text())
    case_sum = sum(int(label.text()) for label in (window.progress_sufficient, window.progress_weak, window.progress_insufficient, window.progress_pending))
    assert total == case_sum
    window.set_current_slice(11, record_visit=False)
    _settle_ui(qapp)
    current_total = int(window.current_slice_target_count.text())
    current_sum = sum(int(label.text()) for label in (window.current_slice_reviewed_count, window.current_slice_weak_count, window.current_slice_missed_count, window.current_slice_pending_count))
    assert current_total == current_sum
    window.close()


def test_changing_case_resets_coverage_selection_and_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_two_case_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_two_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.mode_selector.setCurrentText("Assisted")
    window.show()
    _settle_ui(qapp)
    window.case_selector.setCurrentText("C2")
    _settle_ui(qapp)
    assert window.selected_roi is None
    assert window.current_slice_index == 0
    assert window.progress_total.text() == "1"
    assert window.slice_worklist.count() == 1
    roi_ids = [window.current_slice_roi_selector.itemData(i) for i in range(window.current_slice_roi_selector.count()) if window.current_slice_roi_selector.itemData(i)]
    assert roi_ids in ([], ["R2"])
    window.close()


def test_real_case_count_audit_if_outputs_present() -> None:
    output_root = Path("C:/Users/conta/MedGaze_demo/medgazear_final/outputs")
    if not (output_root / "roi_geometry" / "seg_roi_geometry.csv").exists():
        pytest.skip("real outputs not available")
    data = load_workstation_data(output_root, source="synthetic")
    model = _build_real_case_model(data, "LIDC-IDRI-0001")
    geometry = model.roi_geometry.copy()
    geometry["base_roi_id"] = geometry["roi_id"].astype(str).map(base_roi_id)
    assert len(geometry) == 32
    assert geometry["ct_stack_index"].astype(int).nunique() == 9
    current_slice = geometry[(geometry["ct_stack_index"].astype(int) + 1) == 91]
    assert len(current_slice) == 4


def test_end_to_end_roi_evidence_updates_state_overlays_and_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    qapp.processEvents()
    window.slice_worklist.setCurrentRow(0)
    qapp.processEvents()
    window.overlay_roi.setChecked(True)
    window.overlay_gaze.setChecked(True)
    window.overlay_heatmap.setChecked(True)
    window.overlay_scanpath.setChecked(True)
    qapp.processEvents()

    session_samples = window.case_model.session_gaze_samples("S1")
    for sample in session_samples.to_dict("records"):
        window._timeline_sample_changed(sample)
    qapp.processEvents()

    assert not window.viewer.overlay_samples.empty
    assert int(window.viewer.overlay_samples.iloc[-1]["image_x"]) == 16
    assert int(window.viewer.overlay_samples.iloc[-1]["image_y"]) == 16
    assert int(window.roi_state_store["R1"]["gaze_hit_count_inside_roi"]) >= 8
    assert float(window.roi_state_store["R1"]["total_gaze_time_inside_roi_ms"]) >= 800.0
    assert int(window.roi_state_store["R1"]["fixation_count_inside_roi"]) >= 1
    assert window.roi_state_store["R1"]["mapped_review_status"] in {"reviewed", "weakly_reviewed", "not_reviewed", "not_evaluated"}
    assert window.roi_state_store["R1"]["mapped_review_status"] == "reviewed"
    assert window.current_slice_target_count.text() == "1"
    assert window.progress_sufficient.text() == "1"
    report = window.coverage_debug_report()
    assert "Current Slice Coverage:" in report
    assert "Case Coverage:" in report
    assert window.viewer.slice_label.text() == "2 / 3"

    heatmap = render_heatmap_layer(window.viewer.overlay_samples, ImageSpace(64, 64))
    assert any(pixel[3] > 0 for pixel in heatmap.getdata())
    scanpath = render_scanpath_layer(window.viewer.overlay_samples, ImageSpace(64, 64))
    assert any(pixel[3] > 0 for pixel in scanpath.getdata())
    window.close()


def _build_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MedGazeReviewWorkstation:
    _write_minimal_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    return window


def test_synthetic_session_replay_slice_history_without_roi_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_multislice_session_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_multislice_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window.overlay_gaze.setChecked(True)
    window.overlay_scanpath.setChecked(True)
    qapp.processEvents()
    assert window.selected_roi is None
    assert window.current_session_id == "S_MULTI"
    assert set(window.slice_gaze_history) == {0, 1, 2}
    first_slice = window._current_overlay_samples()
    assert set(first_slice["ct_stack_index"].astype(int)) == {0}
    window.set_current_slice(2, record_visit=False)
    qapp.processEvents()
    third_slice = window._current_overlay_samples()
    assert set(third_slice["ct_stack_index"].astype(int)) == {2}
    window.set_current_slice(0, record_visit=False)
    qapp.processEvents()
    restored = window._current_overlay_samples()
    assert set(restored["ct_stack_index"].astype(int)) == {0}
    assert len(restored) == len(first_slice)
    window.close()


def test_current_slice_without_roi_shows_not_applicable_roi_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.set_current_slice(0, record_visit=False)
    qapp.processEvents()
    assert window.selected_roi_id_value.text() in {"No ROI selected", "Selected ROI is off current slice"}
    assert window.selected_roi_slice_value.text() == "No ROI on current slice"
    assert window.attention_evidence_value.text() == "Not applicable on this slice"
    assert window.prediction_state_value.text() == "Not applicable on this slice"
    window.close()


def test_selected_roi_off_current_slice_is_explicitly_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window.slice_worklist.setCurrentRow(0)
    _settle_ui(qapp)
    window.set_current_slice(0, record_visit=False)
    _settle_ui(qapp)
    assert window.selected_roi_id_value.text() == "No ROI selected"
    assert window.attention_evidence_value.text() == "Not applicable on this slice"
    window.close()


def test_scanpath_history_is_slice_local_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_multislice_session_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_multislice_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window.overlay_scanpath.setChecked(True)
    qapp.processEvents()
    for slice_index in (0, 1, 2):
        window.set_current_slice(slice_index, record_visit=False)
        qapp.processEvents()
        overlay = window._current_overlay_samples()
        assert overlay["ct_stack_index"].astype(int).nunique() == 1
        assert int(overlay["ct_stack_index"].astype(int).iloc[0]) == slice_index
    window.close()


def test_outside_ct_gaze_is_not_rendered_on_ct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.overlay_gaze.setChecked(True)
    outside = pd.DataFrame([
        {"session_id": "L1", "ct_stack_index": 1, "timestamp_ms": 0.0, "image_x": 10.0, "image_y": 10.0, "screen_x": 10.0, "screen_y": 10.0, "is_valid": True, "is_outside_ct": True, "is_ui_glance": True}
    ])
    window.viewer.set_overlay_samples(outside)
    rendered = window.viewer._selected_gaze_on_current_slice()
    assert rendered.empty
    window.close()


def test_pending_and_collecting_evidence_do_not_show_active_assisted_cue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_minimal_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window._synthetic_session_started()
    sample = window.case_model.session_gaze_samples("S1").iloc[0].to_dict()
    window._timeline_sample_changed(sample)
    qapp.processEvents()
    state = window.roi_state_store["R1"]
    assert state["prediction_readiness"] in {"COLLECTING_EVIDENCE", "MISSING_REQUIRED_FEATURES"}
    assert state["roi_overlay_visible"] is False
    assert state["roi_cue_state"] == "none"
    window.close()


def test_missed_cue_requires_review_opportunity_maturity_and_assisted_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _write_missed_cue_outputs(tmp_path)
    data = load_workstation_data(tmp_path, source="synthetic")
    monkeypatch.setattr("app.ui_training.main_window.build_case_review_model", _fake_case_review_model(data))
    window = MedGazeReviewWorkstation(data, output_root=tmp_path)
    window.source_selector.setCurrentText("Synthetic Replay")
    window.show()
    qapp.processEvents()
    window.mode_selector.setCurrentText("Assisted")
    window._synthetic_session_started()
    session_samples = window.case_model.session_gaze_samples("S1")
    for sample in session_samples.to_dict("records"):
        window._timeline_sample_changed(sample)
    qapp.processEvents()
    state = window.roi_state_store["R1"]
    assert state["mapped_review_status"] == "not_reviewed"
    assert state["roi_overlay_visible"] is False
    window.set_current_slice(2)
    qapp.processEvents()
    window.set_current_slice(1, record_visit=False)
    replay_samples = window._session_samples(include_playback_limit=True)
    window._update_roi_states_from_samples(replay_samples, source_label="replay")
    state = window.roi_state_store["R1"]
    assert state["roi_overlay_visible"] is True
    assert state["roi_cue_state"] == "missed"
    window.mode_selector.setCurrentText("Silent")
    qapp.processEvents()
    replay_samples = window._session_samples(include_playback_limit=True)
    window._update_roi_states_from_samples(replay_samples, source_label="replay")
    state = window.roi_state_store["R1"]
    assert state["roi_overlay_visible"] is False
    window.close()


def test_connected_idle_valid_gaze_reports_ready_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.current_source = "Tobii Live"
    window.tobii_preflight_result = {"status": "READY_FOR_SESSION", "message": "Ready to start", "failure_kind": ""}
    monkeypatch.setattr(window.tobii_source, "get_status_payload", lambda: {"device_connected": True, "device_state": "CONNECTED", "calibration_reminder": ""})
    assert window._tobii_status_text() == "Connected"
    window.close()


def test_streaming_tracking_good_and_waiting_and_mapping_unavailable_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.current_source = "Tobii Live"
    monkeypatch.setattr(window.tobii_source, "get_status_payload", lambda: {"device_connected": True, "device_state": "STREAMING", "calibration_reminder": ""})
    window.tobii_source.streaming = True
    window.live_samples = []
    assert window._tobii_status_text() == "Streaming · Waiting for gaze"
    window.live_samples = [{"is_valid": True, "is_outside_ct": False, "timestamp_ms": 0.0}]
    assert window._tobii_status_text() == "Streaming · Tracking good"
    window.live_samples = [{"is_valid": True, "is_outside_ct": True, "timestamp_ms": 0.0}]
    assert window._tobii_status_text() == "Streaming · CT mapping unavailable"
    window.close()


def test_mapping_failure_only_reported_when_preflight_really_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.current_source = "Tobii Live"
    monkeypatch.setattr(window.tobii_source, "get_status_payload", lambda: {"device_connected": True, "device_state": "CONNECTED", "calibration_reminder": ""})
    window.tobii_preflight_result = {"status": "READY_FOR_SESSION", "message": "Ready to start", "failure_kind": ""}
    assert window._tobii_status_text() == "Connected"
    window.tobii_preflight_result = {"status": "TRACKING_NOT_READY", "message": "CT mapping unavailable", "failure_kind": "mapping_unavailable"}
    assert window._tobii_status_text() == "Mapping unavailable"
    window.close()


def test_preflight_mapping_does_not_update_scientific_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    sample = {
        "timestamp_ms": 0.0,
        "screen_x": 10.0,
        "screen_y": 12.0,
        "image_x": 12.0,
        "image_y": 12.0,
        "gaze_x_norm": 0.5,
        "gaze_y_norm": 0.5,
        "is_valid": True,
        "is_outside_ct": False,
    }
    monkeypatch.setattr(window.tobii_source, "start_stream", lambda callback: (setattr(window.tobii_source, "streaming", True), callback(sample))[1])
    monkeypatch.setattr(window.tobii_source, "stop_stream", lambda: setattr(window.tobii_source, "streaming", False))
    before = {roi_id: dict(state) for roi_id, state in window.roi_state_store.items()}
    before_history = {key: list(value) for key, value in window.slice_gaze_history.items()}
    window._run_tobii_preflight(duration_s=0.0)
    qapp.processEvents()
    assert window.live_samples == []
    assert window.slice_gaze_history == before_history
    assert all(window.roi_state_store[roi_id]["mapped_review_status"] == before[roi_id]["mapped_review_status"] for roi_id in before)
    window.close()


def test_slice_scroll_defers_expensive_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    window = _build_window(tmp_path, monkeypatch)
    window.show()
    qapp.processEvents()
    window.set_current_slice(2, record_visit=False)
    window.set_current_slice(1, record_visit=False)
    window.set_current_slice(0, record_visit=False)
    assert window.current_slice_index == 0
    assert window.slice_update_timer.isActive() is True
    _settle_ui(qapp)
    assert window.slice_update_timer.isActive() is False
    assert window.current_slice_target_count.text() == "0"
    window.close()


def _write_minimal_outputs(root: Path) -> None:
    (root / "visualizations").mkdir(parents=True)
    (root / "features").mkdir()
    (root / "attention").mkdir()
    (root / "behavior_learning").mkdir()
    (root / "cognitive_load").mkdir()
    (root / "synthetic_gaze").mkdir()
    (root / "roi_geometry").mkdir()
    (root / "dicom_audit").mkdir()
    feature = {
        "session_id": "S1",
        "reader_id": "Reader1",
        "reader_profile": "fast_confirmer",
        "case_id": "C1",
        "roi_id": "R1",
        "slice_index": 1,
        "hidden_behavior_label": "focused_roi_confirmation",
        "total_gaze_time_inside_roi_ms": 100.0,
        "valid_gaze_time_on_roi_slice_ms": 600.0,
        "time_on_roi_slice_ms": 900.0,
        "gaze_hit_count_inside_roi": 6,
        "fixation_count_inside_roi": 2,
        "gaze_dispersion_px": 20.0,
        "roi_revisit_count": 1,
        "background_gaze_ratio": 0.2,
        "gaze_validity_ratio": 0.95,
    }
    pd.DataFrame([feature]).to_csv(root / "visualizations" / "representative_cases.csv", index=False)
    pd.DataFrame([feature]).to_csv(root / "features" / "behavior_feature_table.csv", index=False)
    pd.DataFrame([feature]).to_csv(root / "behavior_learning" / "behavior_learning_dataset.csv", index=False)
    pd.DataFrame([{"session_id": "S1", "roi_id": "R1", "rule_attention_status": "reviewed"}]).to_csv(root / "attention" / "rule_attention_status.csv", index=False)
    pd.DataFrame([{"session_id": "S1", "roi_id": "R1", "cognitive_load_proxy": "low_load_proxy", "cognitive_load_proxy_score": 0.1}]).to_csv(root / "cognitive_load" / "cognitive_proxy_labels.csv", index=False)
    pd.DataFrame([
        _gaze_row(10, 10, 0, 1),
        _gaze_row(11, 11, 100, 1),
        _gaze_row(12, 12, 200, 1),
        _gaze_row(13, 13, 300, 1),
        _gaze_row(14, 14, 400, 1),
        _gaze_row(15, 15, 500, 1),
        _gaze_row(16, 16, 600, 1),
        _gaze_row(16, 16, 700, 1),
    ]).to_csv(root / "synthetic_gaze" / "raw_behavior_labeled_synthetic_gaze.csv", index=False)
    pd.DataFrame([
        {
            "patient_id": "C1",
            "ct_series_instance_uid": "SERIES1",
            "ct_sop_instance_uid": "SOP1",
            "roi_id": "R1",
            "slice_index": 1,
            "ct_stack_index": 1,
            "rows": 64,
            "columns": 64,
            "bbox_x_min": 8,
            "bbox_y_min": 8,
            "bbox_x_max": 20,
            "bbox_y_max": 20,
            "centroid_x": 14,
            "centroid_y": 14,
        }
    ]).to_csv(root / "roi_geometry" / "seg_roi_geometry.csv", index=False)
    pd.DataFrame([
        {
            "classification": "CT",
            "patient_id": "C1",
            "series_instance_uid": "SERIES1",
            "study_instance_uid": "STUDY1",
            "sop_instance_uid": "SOP1",
            "file_path": str(root / "fake_slice.dcm"),
            "instance_number": 1,
            "image_position_patient": "0|0|0",
        }
    ]).to_csv(root / "dicom_audit" / "dicom_inventory.csv", index=False)
    (root / "behavior_learning" / "behavior_feature_schema.json").write_text(json.dumps({"feature_columns": []}), encoding="utf-8")
    (root / "behavior_learning" / "behavior_label_mapping.json").write_text(json.dumps({}), encoding="utf-8")


def _write_frame_case_outputs(root: Path) -> None:
    _write_minimal_outputs(root)
    feature_rows = [
        {"session_id": "S1", "reader_id": "Reader1", "reader_profile": "fast_confirmer", "case_id": "C1", "roi_id": "A__frame_0000", "slice_index": 10, "hidden_behavior_label": "focused_roi_confirmation", "total_gaze_time_inside_roi_ms": 10.0, "valid_gaze_time_on_roi_slice_ms": 20.0, "time_on_roi_slice_ms": 25.0, "gaze_hit_count_inside_roi": 1, "fixation_count_inside_roi": 0, "gaze_dispersion_px": 2.0, "roi_revisit_count": 0, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
        {"session_id": "S2", "reader_id": "Reader1", "reader_profile": "fast_confirmer", "case_id": "C1", "roi_id": "A__frame_0001", "slice_index": 11, "hidden_behavior_label": "partial_near_miss_review", "total_gaze_time_inside_roi_ms": 10.0, "valid_gaze_time_on_roi_slice_ms": 20.0, "time_on_roi_slice_ms": 25.0, "gaze_hit_count_inside_roi": 1, "fixation_count_inside_roi": 0, "gaze_dispersion_px": 2.0, "roi_revisit_count": 0, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
        {"session_id": "S3", "reader_id": "Reader1", "reader_profile": "fast_confirmer", "case_id": "C1", "roi_id": "A__frame_0002", "slice_index": 12, "hidden_behavior_label": "skipped_slice", "total_gaze_time_inside_roi_ms": 10.0, "valid_gaze_time_on_roi_slice_ms": 20.0, "time_on_roi_slice_ms": 25.0, "gaze_hit_count_inside_roi": 1, "fixation_count_inside_roi": 0, "gaze_dispersion_px": 2.0, "roi_revisit_count": 0, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
        {"session_id": "S4", "reader_id": "Reader1", "reader_profile": "fast_confirmer", "case_id": "C1", "roi_id": "B__frame_0000", "slice_index": 11, "hidden_behavior_label": "missed_roi_search", "total_gaze_time_inside_roi_ms": 10.0, "valid_gaze_time_on_roi_slice_ms": 20.0, "time_on_roi_slice_ms": 25.0, "gaze_hit_count_inside_roi": 1, "fixation_count_inside_roi": 0, "gaze_dispersion_px": 2.0, "roi_revisit_count": 0, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
    ]
    pd.DataFrame(feature_rows).to_csv(root / "features" / "behavior_feature_table.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(root / "behavior_learning" / "behavior_learning_dataset.csv", index=False)
    pd.DataFrame([
        {"session_id": "S1", "roi_id": "A__frame_0000", "rule_attention_status": "reviewed"},
        {"session_id": "S2", "roi_id": "A__frame_0001", "rule_attention_status": "weakly_reviewed"},
        {"session_id": "S3", "roi_id": "A__frame_0002", "rule_attention_status": "not_evaluated"},
        {"session_id": "S4", "roi_id": "B__frame_0000", "rule_attention_status": "not_reviewed"},
    ]).to_csv(root / "attention" / "rule_attention_status.csv", index=False)
    pd.DataFrame([
        {"patient_id": "C1", "ct_series_instance_uid": "SERIES1", "ct_sop_instance_uid": "SOPA0", "roi_id": "A__frame_0000", "slice_index": 10, "ct_stack_index": 10, "rows": 64, "columns": 64, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14, "centroid_y": 14},
        {"patient_id": "C1", "ct_series_instance_uid": "SERIES1", "ct_sop_instance_uid": "SOPA1", "roi_id": "A__frame_0001", "slice_index": 11, "ct_stack_index": 11, "rows": 64, "columns": 64, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14, "centroid_y": 14},
        {"patient_id": "C1", "ct_series_instance_uid": "SERIES1", "ct_sop_instance_uid": "SOPB0", "roi_id": "B__frame_0000", "slice_index": 11, "ct_stack_index": 11, "rows": 64, "columns": 64, "bbox_x_min": 24, "bbox_y_min": 24, "bbox_x_max": 30, "bbox_y_max": 30, "centroid_x": 27, "centroid_y": 27},
        {"patient_id": "C1", "ct_series_instance_uid": "SERIES1", "ct_sop_instance_uid": "SOPA2", "roi_id": "A__frame_0002", "slice_index": 12, "ct_stack_index": 12, "rows": 64, "columns": 64, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14, "centroid_y": 14},
    ]).to_csv(root / "roi_geometry" / "seg_roi_geometry.csv", index=False)


def _write_multislice_session_outputs(root: Path) -> None:
    _write_minimal_outputs(root)
    pd.DataFrame([
        _gaze_row(6, 6, 0, 0, session_id="S_MULTI", roi_id="R1"),
        _gaze_row(10, 10, 100, 0, session_id="S_MULTI", roi_id="R1"),
        _gaze_row(14, 14, 200, 1, session_id="S_MULTI", roi_id="R1"),
        _gaze_row(16, 16, 300, 1, session_id="S_MULTI", roi_id="R1"),
        _gaze_row(22, 22, 400, 2, session_id="S_MULTI", roi_id="R1"),
        _gaze_row(24, 24, 500, 2, session_id="S_MULTI", roi_id="R1"),
    ]).to_csv(root / "synthetic_gaze" / "raw_behavior_labeled_synthetic_gaze.csv", index=False)
    feature = pd.read_csv(root / "features" / "behavior_feature_table.csv")
    feature["session_id"] = "S_MULTI"
    feature.to_csv(root / "features" / "behavior_feature_table.csv", index=False)
    feature.to_csv(root / "behavior_learning" / "behavior_learning_dataset.csv", index=False)
    pd.DataFrame([{"session_id": "S_MULTI", "roi_id": "R1", "rule_attention_status": "reviewed"}]).to_csv(root / "attention" / "rule_attention_status.csv", index=False)


def _write_two_case_outputs(root: Path) -> None:
    _write_minimal_outputs(root)
    features = pd.DataFrame([
        {"session_id": "S1", "reader_id": "Reader1", "reader_profile": "fast_confirmer", "case_id": "C1", "roi_id": "R1", "slice_index": 1, "hidden_behavior_label": "focused_roi_confirmation", "total_gaze_time_inside_roi_ms": 100.0, "valid_gaze_time_on_roi_slice_ms": 600.0, "time_on_roi_slice_ms": 900.0, "gaze_hit_count_inside_roi": 6, "fixation_count_inside_roi": 2, "gaze_dispersion_px": 20.0, "roi_revisit_count": 1, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
        {"session_id": "S2", "reader_id": "Reader2", "reader_profile": "fast_confirmer", "case_id": "C2", "roi_id": "R2", "slice_index": 2, "hidden_behavior_label": "missed_roi_search", "total_gaze_time_inside_roi_ms": 10.0, "valid_gaze_time_on_roi_slice_ms": 20.0, "time_on_roi_slice_ms": 25.0, "gaze_hit_count_inside_roi": 1, "fixation_count_inside_roi": 0, "gaze_dispersion_px": 2.0, "roi_revisit_count": 0, "background_gaze_ratio": 0.2, "gaze_validity_ratio": 0.95},
    ])
    features.to_csv(root / "features" / "behavior_feature_table.csv", index=False)
    features.to_csv(root / "behavior_learning" / "behavior_learning_dataset.csv", index=False)
    pd.DataFrame([
        {"session_id": "S1", "roi_id": "R1", "rule_attention_status": "reviewed"},
        {"session_id": "S2", "roi_id": "R2", "rule_attention_status": "not_evaluated"},
    ]).to_csv(root / "attention" / "rule_attention_status.csv", index=False)
    pd.DataFrame([
        {"patient_id": "C1", "ct_series_instance_uid": "SERIES1", "ct_sop_instance_uid": "SOP1", "roi_id": "R1", "slice_index": 1, "ct_stack_index": 1, "rows": 64, "columns": 64, "bbox_x_min": 8, "bbox_y_min": 8, "bbox_x_max": 20, "bbox_y_max": 20, "centroid_x": 14, "centroid_y": 14},
        {"patient_id": "C2", "ct_series_instance_uid": "SERIES2", "ct_sop_instance_uid": "SOP2", "roi_id": "R2", "slice_index": 2, "ct_stack_index": 2, "rows": 64, "columns": 64, "bbox_x_min": 12, "bbox_y_min": 12, "bbox_x_max": 18, "bbox_y_max": 18, "centroid_x": 15, "centroid_y": 15},
    ]).to_csv(root / "roi_geometry" / "seg_roi_geometry.csv", index=False)


def _write_missed_cue_outputs(root: Path) -> None:
    _write_minimal_outputs(root)
    rows = []
    for index in range(60):
        rows.append(_gaze_row(40, 40, float(index * 100), 1))
    pd.DataFrame(rows).to_csv(root / "synthetic_gaze" / "raw_behavior_labeled_synthetic_gaze.csv", index=False)


def _gaze_row(x: float, y: float, timestamp: float, ct_stack_index: int, session_id: str = "S1", roi_id: str = "R1") -> dict[str, object]:
    return {
        "session_id": session_id,
        "case_id": "C1",
        "roi_id": roi_id,
        "slice_index": ct_stack_index,
        "ct_stack_index": ct_stack_index,
        "timestamp_ms": timestamp,
        "image_x": x,
        "image_y": y,
        "screen_x": x,
        "screen_y": y,
        "is_valid": True,
        "is_outside_ct": False,
        "is_ui_glance": False,
    }


def _fake_case_review_model(data):
    class FakeCaseReviewModel:
        def __init__(self, case_id: str, series_uid: str | None = None) -> None:
            self.patient_id = case_id
            self.series_uid = series_uid or "SERIES1"
            self.roi_geometry = data.roi_geometry.copy()
            self.roi_geometry["patient_id"] = case_id
            self.roi_geometry["ct_series_instance_uid"] = self.series_uid
            self.roi_geometry["ct_stack_index"] = 1
            self.roi_geometry["rule_attention_status"] = "reviewed"
            self.review_targets = build_review_targets(self.roi_geometry)
            self.features = data.features.copy()
            self.features["case_id"] = case_id
            self.features["ct_stack_index"] = 1
            self.features["bbox_x_min"] = 8
            self.features["bbox_y_min"] = 8
            self.features["bbox_x_max"] = 20
            self.features["bbox_y_max"] = 20
            self.features["centroid_x"] = 14
            self.features["centroid_y"] = 14
            self.attention = data.attention.copy()
            self.gaze = data.gaze.copy()
            self.total_slices = 3

        def image_for_slice(self, slice_index: int, window_center=None, window_width=None):
            del slice_index, window_center, window_width
            return Image.new("L", (64, 64), color=32)

        def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
            return self.roi_geometry[self.roi_geometry["ct_stack_index"] == int(slice_index)]

        def gaze_on_slice(self, session_id: str, roi_id: str, slice_index: int) -> pd.DataFrame:
            del session_id, roi_id, slice_index
            return self.gaze.copy()

        def session_gaze_samples(self, session_id: str) -> pd.DataFrame:
            del session_id
            return self.gaze.copy()

    def _builder(case_id: str, _data, series_uid: str | None = None):
        return FakeCaseReviewModel(case_id, series_uid)

    return _builder


def _fake_multislice_case_review_model(data):
    class FakeMultisliceCaseReviewModel:
        def __init__(self, case_id: str, series_uid: str | None = None) -> None:
            self.patient_id = case_id
            self.series_uid = series_uid or "SERIES1"
            self.roi_geometry = data.roi_geometry.copy()
            self.review_targets = build_review_targets(self.roi_geometry)
            self.features = data.features.copy()
            self.attention = data.attention.copy()
            self.gaze = data.gaze.copy()
            self.total_slices = 4

        def image_for_slice(self, slice_index: int, window_center=None, window_width=None):
            del slice_index, window_center, window_width
            return Image.new("L", (64, 64), color=32)

        def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
            return self.roi_geometry[self.roi_geometry["ct_stack_index"].astype(int) == int(slice_index)]

        def gaze_on_slice(self, session_id: str, roi_id: str, slice_index: int) -> pd.DataFrame:
            rows = self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy()
            rows = rows[rows["ct_stack_index"].astype(int) == int(slice_index)].copy()
            if roi_id is not None:
                rows = rows[rows["roi_id"].astype(str) == str(roi_id)].copy()
            return rows

        def session_gaze_samples(self, session_id: str) -> pd.DataFrame:
            return self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy().sort_values("timestamp_ms").reset_index(drop=True)

    def _builder(case_id: str, _data, series_uid: str | None = None):
        return FakeMultisliceCaseReviewModel(case_id, series_uid)

    return _builder


def _fake_frame_case_review_model(data):
    class FakeFrameCaseReviewModel:
        def __init__(self, case_id: str, series_uid: str | None = None) -> None:
            self.patient_id = case_id
            self.series_uid = series_uid or "SERIES1"
            self.roi_geometry = data.roi_geometry.copy().sort_values(["ct_stack_index", "roi_id"]).reset_index(drop=True)
            self.review_targets = build_review_targets(self.roi_geometry)
            self.features = data.features.copy()
            self.attention = data.attention.copy()
            self.gaze = data.gaze.copy()
            self.total_slices = 20

        def image_for_slice(self, slice_index: int, window_center=None, window_width=None):
            del slice_index, window_center, window_width
            return Image.new("L", (64, 64), color=32)

        def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
            return self.roi_geometry[self.roi_geometry["ct_stack_index"] == int(slice_index)]

        def gaze_on_slice(self, session_id: str, roi_id: str, slice_index: int) -> pd.DataFrame:
            del session_id, roi_id, slice_index
            return self.gaze.copy()

        def session_gaze_samples(self, session_id: str) -> pd.DataFrame:
            return self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy()

    def _builder(case_id: str, _data, series_uid: str | None = None):
        return FakeFrameCaseReviewModel(case_id, series_uid)

    return _builder


def _fake_two_case_review_model(data):
    class FakeTwoCaseReviewModel:
        def __init__(self, case_id: str, series_uid: str | None = None) -> None:
            self.patient_id = case_id
            self.series_uid = series_uid or ("SERIES1" if case_id == "C1" else "SERIES2")
            self.roi_geometry = data.roi_geometry[data.roi_geometry["patient_id"].astype(str) == str(case_id)].copy().sort_values(["ct_stack_index", "roi_id"]).reset_index(drop=True)
            self.review_targets = build_review_targets(self.roi_geometry)
            self.features = data.features[data.features["case_id"].astype(str) == str(case_id)].copy()
            self.attention = data.attention[data.attention["roi_id"].astype(str).isin(self.roi_geometry["roi_id"].astype(str))].copy()
            self.gaze = data.gaze[data.gaze["case_id"].astype(str) == str(case_id)].copy()
            self.total_slices = 4

        def image_for_slice(self, slice_index: int, window_center=None, window_width=None):
            del slice_index, window_center, window_width
            return Image.new("L", (64, 64), color=32)

        def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
            return self.roi_geometry[self.roi_geometry["ct_stack_index"].astype(int) == int(slice_index)]

        def gaze_on_slice(self, session_id: str, roi_id: str, slice_index: int) -> pd.DataFrame:
            rows = self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy()
            rows = rows[rows["ct_stack_index"].astype(int) == int(slice_index)].copy()
            if roi_id:
                rows = rows[rows["roi_id"].astype(str) == str(roi_id)].copy()
            return rows

        def session_gaze_samples(self, session_id: str) -> pd.DataFrame:
            return self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy()

    def _builder(case_id: str, _data, series_uid: str | None = None):
        return FakeTwoCaseReviewModel(case_id, series_uid)

    return _builder


def _build_real_case_model(data, case_id: str):
    return build_case_review_model(case_id, data)


def _visible_label_texts(window) -> list[str]:
    from PyQt6.QtWidgets import QLabel

    texts = []
    for label in window.findChildren(QLabel):
        if label.isVisible() and label.text().strip():
            texts.append(label.text())
    return texts


def _assert_widgets_inside_central_widget(window, widgets: list[object]) -> None:
    central_rect = window.centralWidget().rect()
    for widget in widgets:
        top_left = widget.mapTo(window.centralWidget(), widget.rect().topLeft())
        bottom_right = widget.mapTo(window.centralWidget(), widget.rect().bottomRight())
        assert central_rect.contains(top_left)
        assert central_rect.contains(bottom_right)
