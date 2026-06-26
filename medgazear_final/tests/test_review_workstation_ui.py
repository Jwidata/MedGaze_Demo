from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.ui.insight_panel import format_insight_text
from app.ui.overlay_layer_manager import OverlayLayerManager
from app.ui.review_workstation import smoke_test_workstation
from app.ui.session_browser import SessionSelectionModel
from app.ui.ui_data_loader import load_workstation_data
from app.ui.ui_theme import TOBII_PLACEHOLDER_MESSAGE


def test_ui_data_loading(tmp_path: Path) -> None:
    _write_minimal_outputs(tmp_path)

    data = load_workstation_data(tmp_path, source="synthetic")

    assert len(data.representative_cases) == 1
    assert len(data.gaze) == 2
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


def test_insight_panel_formatting() -> None:
    text = format_insight_text(
        {
            "hidden_behavior_label": "focused_roi_confirmation",
            "predicted_behavior_label": "focused_roi_confirmation",
            "prediction_confidence": 0.91,
            "rule_attention_status": "reviewed",
            "cognitive_load_proxy": "low_load_proxy",
            "total_gaze_time_inside_roi_ms": 1234.5,
            "guided_narration": "The gaze rapidly concentrates near the ROI.",
        }
    )

    assert "Hidden behavior label: focused_roi_confirmation" in text
    assert "Prediction confidence: 0.910" in text
    assert "Guided narration" in text


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


def _write_minimal_outputs(root: Path) -> None:
    (root / "visualizations").mkdir(parents=True)
    (root / "features").mkdir()
    (root / "attention").mkdir()
    (root / "behavior_learning").mkdir()
    (root / "cognitive_load").mkdir()
    (root / "synthetic_gaze").mkdir()
    (root / "roi_geometry").mkdir()
    feature = {
        "session_id": "S1",
        "reader_id": "Reader1",
        "reader_profile": "fast_confirmer",
        "case_id": "C1",
        "roi_id": "R1",
        "slice_index": 1,
        "hidden_behavior_label": "focused_roi_confirmation",
        "total_gaze_time_inside_roi_ms": 100.0,
        "valid_gaze_time_on_roi_slice_ms": 200.0,
        "gaze_hit_count_inside_roi": 6,
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
    pd.DataFrame(
        [
            _gaze_row(10, 10, 0),
            _gaze_row(12, 12, 16.667),
        ]
    ).to_csv(root / "synthetic_gaze" / "raw_behavior_labeled_synthetic_gaze.csv", index=False)
    pd.DataFrame(
        [
            {
                "roi_id": "R1",
                "slice_index": 1,
                "rows": 64,
                "columns": 64,
                "bbox_x_min": 8,
                "bbox_y_min": 8,
                "bbox_x_max": 20,
                "bbox_y_max": 20,
                "centroid_x": 14,
                "centroid_y": 14,
            }
        ]
    ).to_csv(root / "roi_geometry" / "seg_roi_geometry.csv", index=False)
    (root / "behavior_learning" / "behavior_feature_schema.json").write_text(json.dumps({"feature_columns": []}), encoding="utf-8")
    (root / "behavior_learning" / "behavior_label_mapping.json").write_text(json.dumps({}), encoding="utf-8")


def _gaze_row(x: float, y: float, timestamp: float) -> dict[str, object]:
    return {
        "session_id": "S1",
        "case_id": "C1",
        "roi_id": "R1",
        "slice_index": 1,
        "timestamp_ms": timestamp,
        "image_x": x,
        "image_y": y,
        "screen_x": x,
        "screen_y": y,
        "is_valid": True,
        "is_outside_ct": False,
        "is_ui_glance": False,
    }
