"""Generate source-agnostic gaze visualizations and guided reports."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.visualization.coordinate_mapper import image_space_from_roi
from app.visualization.gaze_schema import load_gaze_samples
from app.visualization.guided_narration import build_case_narration
from app.visualization.heatmap_renderer import render_heatmap_layer, save_heatmap_overlay
from app.visualization.interactive_report_builder import write_guided_narrations, write_interactive_report
from app.visualization.representative_case_selector import select_representative_cases
from app.visualization.roi_overlay_renderer import create_placeholder_canvas, render_roi_layer, save_roi_overlay
from app.visualization.scanpath_renderer import render_gaze_points_layer, render_scanpath_layer, save_gaze_points_overlay, save_scanpath_overlay
from app.visualization.visual_report_builder import write_visualization_report
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Generate source-agnostic gaze visualizations and guided report.")
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--roi-geometry", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--behavior-learning", default=None)
    parser.add_argument("--cognitive", required=True)
    parser.add_argument("--examples-per-behavior", type=int, default=3)
    parser.add_argument("--source-type", default="synthetic")
    parser.add_argument("--interactive-report", action="store_true")
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    gaze = load_gaze_samples(args.gaze, source_type=args.source_type)
    roi_geometry = pd.read_csv(args.roi_geometry)
    features = pd.read_csv(args.features)
    attention = pd.read_csv(args.attention)
    cognitive = pd.read_csv(args.cognitive)
    if args.behavior_learning:
        _ = pd.read_csv(args.behavior_learning)

    selected = select_representative_cases(features, attention, cognitive, args.examples_per_behavior)
    selected = _render_cases(selected, gaze, roi_geometry, output_dir)
    representative_path = output_dir / "representative_cases.csv"
    selected.to_csv(representative_path, index=False)

    narrations = [build_case_narration(row) for _, row in selected.iterrows()]
    write_guided_narrations(output_dir / "guided_case_narrations.json", narrations)
    if args.interactive_report:
        write_interactive_report(output_dir / "interactive_gaze_review.html", selected, narrations, output_dir)
    write_visualization_report(output_dir / "visualization_report.md", selected, interactive_enabled=args.interactive_report)

    logger.info("Gaze visualization outputs written to: %s", output_dir)
    return 0


def _render_cases(selected: pd.DataFrame, gaze: pd.DataFrame, roi_geometry: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    roi_lookup = roi_geometry.set_index("roi_id", drop=False)
    rows: list[dict[str, object]] = []
    for _, case in selected.iterrows():
        roi_id = str(case["roi_id"])
        session_id = str(case["session_id"])
        behavior_label = str(case["hidden_behavior_label"])
        if roi_id not in roi_lookup.index:
            continue
        roi = roi_lookup.loc[roi_id]
        if isinstance(roi, pd.DataFrame):
            roi = roi.iloc[0]
        samples = gaze[(gaze["session_id"] == session_id) & (gaze["roi_id"] == roi_id)].copy()
        if samples.empty:
            continue
        case_dir = output_dir / behavior_label
        case_dir.mkdir(parents=True, exist_ok=True)
        image_space = image_space_from_roi(roi)
        paths = {
            "canvas_path": case_dir / f"{session_id}_canvas.png",
            "roi_overlay_path": case_dir / f"{session_id}_roi_overlay.png",
            "gaze_points_path": case_dir / f"{session_id}_gaze_points.png",
            "heatmap_overlay_path": case_dir / f"{session_id}_heatmap_overlay.png",
            "scanpath_overlay_path": case_dir / f"{session_id}_scanpath_overlay.png",
            "combined_overlay_path": case_dir / f"{session_id}_combined_overlay.png",
        }
        create_placeholder_canvas(image_space).convert("RGB").save(paths["canvas_path"])
        save_roi_overlay(paths["roi_overlay_path"], roi)
        save_gaze_points_overlay(paths["gaze_points_path"], samples, roi, image_space)
        save_heatmap_overlay(paths["heatmap_overlay_path"], samples, roi, image_space)
        save_scanpath_overlay(paths["scanpath_overlay_path"], samples, roi, image_space)
        _save_combined_overlay(paths["combined_overlay_path"], samples, roi, image_space)
        rendered = case.to_dict()
        rendered.update({key: str(value.relative_to(output_dir)) for key, value in paths.items()})
        rows.append(rendered)
    return pd.DataFrame(rows)


def _save_combined_overlay(path: Path, samples: pd.DataFrame, roi: pd.Series, image_space) -> Path:
    image = create_placeholder_canvas(image_space)
    for layer in (
        render_heatmap_layer(samples, image_space),
        render_roi_layer(roi, image_space),
        render_gaze_points_layer(samples, image_space),
        render_scanpath_layer(samples, image_space),
    ):
        image = Image.alpha_composite(image, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
