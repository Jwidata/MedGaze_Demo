from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from app.visualization.coordinate_mapper import ImageSpace, clip_point
from app.visualization.gaze_schema import normalize_gaze_samples
from app.visualization.guided_narration import build_case_narration
from app.visualization.heatmap_renderer import generate_heatmap_array, save_heatmap_overlay
from app.visualization.interactive_report_builder import write_guided_narrations, write_interactive_report
from app.visualization.representative_case_selector import select_representative_cases
from app.visualization.roi_overlay_renderer import save_roi_overlay
from app.visualization.scanpath_renderer import save_gaze_points_overlay, save_scanpath_overlay
from app.visualization.visual_report_builder import write_visualization_report


def test_gaze_schema_validation_adds_source_type() -> None:
    normalized = normalize_gaze_samples(_gaze_rows().drop(columns=["source_type"]), source_type="synthetic")

    assert normalized.loc[0, "source_type"] == "synthetic"
    assert normalized["is_valid"].dtype == bool


def test_gaze_schema_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required schema"):
        normalize_gaze_samples(_gaze_rows().drop(columns=["image_x"]))


def test_coordinate_clipping() -> None:
    assert clip_point(-10, 99, ImageSpace(width=64, height=32)) == (0.0, 31)
    assert clip_point(100, -3, ImageSpace(width=64, height=32)) == (63, 0.0)


def test_heatmap_array_generation() -> None:
    heatmap = generate_heatmap_array(_gaze_rows(), ImageSpace(width=64, height=64), sigma=2)

    assert heatmap.shape == (64, 64)
    assert heatmap.max() <= 1.0
    assert heatmap.sum() > 0


def test_representative_case_selection() -> None:
    selected = select_representative_cases(_feature_rows(), _attention_rows(), _cognitive_rows(), examples_per_behavior=1)

    assert set(selected["hidden_behavior_label"]) == {"focused_roi_confirmation", "missed_roi_search"}
    assert "rule_attention_status" in selected.columns
    assert "cognitive_load_proxy" in selected.columns


def test_output_image_generation(tmp_path: Path) -> None:
    roi = _roi_row()
    samples = _gaze_rows()
    image_space = ImageSpace(width=64, height=64)
    paths = [
        save_roi_overlay(tmp_path / "roi.png", roi),
        save_gaze_points_overlay(tmp_path / "gaze.png", samples, roi, image_space),
        save_heatmap_overlay(tmp_path / "heatmap.png", samples, roi, image_space),
        save_scanpath_overlay(tmp_path / "scanpath.png", samples, roi, image_space),
    ]

    for path in paths:
        assert path.exists()
        assert Image.open(path).size == (64, 64)


def test_report_wording_and_interactive_outputs(tmp_path: Path) -> None:
    selected = select_representative_cases(_feature_rows(), _attention_rows(), _cognitive_rows(), examples_per_behavior=1)
    selected = selected.head(1).copy()
    label = selected.iloc[0]["hidden_behavior_label"]
    session_id = selected.iloc[0]["session_id"]
    case_dir = tmp_path / label
    case_dir.mkdir()
    for suffix in ("canvas", "roi_overlay", "gaze_points", "heatmap_overlay", "scanpath_overlay", "combined_overlay"):
        Image.new("RGB", (8, 8), "black").save(case_dir / f"{session_id}_{suffix}.png")
    narrations = [build_case_narration(selected.iloc[0])]
    report = tmp_path / "visualization_report.md"
    html = tmp_path / "interactive_gaze_review.html"
    json_path = tmp_path / "guided_case_narrations.json"

    write_visualization_report(report, selected, interactive_enabled=True)
    write_guided_narrations(json_path, narrations)
    write_interactive_report(html, selected, narrations, tmp_path)

    report_text = report.read_text(encoding="utf-8")
    html_text = html.read_text(encoding="utf-8")
    assert "source-agnostic gaze overlays" in report_text
    assert "Heatmaps are generated in image/CT coordinates" in report_text
    assert "Synthetic gaze and future Tobii gaze use the same" in html_text
    assert "Show heatmap" in html_text
    assert json_path.exists()


def _gaze_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _gaze_row(10, 10, True),
            _gaze_row(12, 12, True),
            _gaze_row(90, -5, False),
        ]
    )


def _gaze_row(x: float, y: float, valid: bool) -> dict[str, object]:
    return {
        "source_type": "synthetic",
        "session_id": "S1",
        "case_id": "C1",
        "roi_id": "R1",
        "slice_index": 1,
        "timestamp_ms": x,
        "image_x": x,
        "image_y": y,
        "screen_x": x * 2,
        "screen_y": y * 2,
        "is_valid": valid,
        "is_outside_ct": False,
        "is_ui_glance": False,
    }


def _roi_row() -> dict[str, object]:
    return {
        "roi_id": "R1",
        "rows": 64,
        "columns": 64,
        "bbox_x_min": 8,
        "bbox_y_min": 8,
        "bbox_x_max": 20,
        "bbox_y_max": 20,
        "centroid_x": 14,
        "centroid_y": 14,
    }


def _feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session_id": "S1",
                "case_id": "C1",
                "roi_id": "R1",
                "hidden_behavior_label": "focused_roi_confirmation",
                "gaze_validity_ratio": 0.9,
                "valid_gaze_time_on_roi_slice_ms": 1000,
            },
            {
                "session_id": "S2",
                "case_id": "C2",
                "roi_id": "R2",
                "hidden_behavior_label": "missed_roi_search",
                "gaze_validity_ratio": 0.8,
                "valid_gaze_time_on_roi_slice_ms": 800,
            },
        ]
    )


def _attention_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_id": "S1", "roi_id": "R1", "rule_attention_status": "reviewed"},
            {"session_id": "S2", "roi_id": "R2", "rule_attention_status": "not_reviewed"},
        ]
    )


def _cognitive_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_id": "S1", "roi_id": "R1", "cognitive_load_proxy": "low_load_proxy", "cognitive_load_proxy_score": 0.1},
            {"session_id": "S2", "roi_id": "R2", "cognitive_load_proxy": "medium_load_proxy", "cognitive_load_proxy_score": 0.5},
        ]
    )
