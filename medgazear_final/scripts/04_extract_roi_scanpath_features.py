"""Extract ROI, scanpath, temporal, quality, and geometry features."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.features.feature_quality_report import write_feature_quality_report
from app.features.feature_schema import (
    GEOMETRY_FEATURES,
    METADATA_COLUMNS,
    QUALITY_FEATURES,
    SCANPATH_FEATURES,
    TEMPORAL_FEATURES,
    behavior_feature_columns,
    validate_no_leakage,
    write_feature_schema,
)
from app.features.roi_feature_extractor import extract_roi_features, roi_masks_for_samples
from app.features.scanpath_feature_extractor import extract_scanpath_features
from app.features.temporal_feature_extractor import detect_fixations, extract_temporal_features
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Extract ROI-level gaze features from degraded synthetic gaze samples.")
    parser.add_argument("--gaze", required=True, help="Raw behavior-labeled synthetic gaze CSV.")
    parser.add_argument("--roi-geometry", required=True, help="SEG ROI geometry CSV.")
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = extract_feature_rows(Path(args.gaze), Path(args.roi_geometry))
    validate_no_leakage(behavior_feature_columns())
    _write_csv(output_dir / "roi_level_features.csv", rows, behavior_feature_columns())
    _write_csv(output_dir / "behavior_feature_table.csv", rows, behavior_feature_columns())
    _write_csv(output_dir / "scanpath_features.csv", rows, METADATA_COLUMNS + SCANPATH_FEATURES)
    _write_csv(output_dir / "slice_level_features.csv", _slice_level_rows(rows), METADATA_COLUMNS + ["roi_count_on_slice", "mean_gaze_validity_ratio", "mean_total_gaze_time_inside_roi_ms"])
    write_feature_schema(output_dir / "feature_schema.md")
    write_feature_quality_report(output_dir / "feature_quality_report.md", rows)
    logger.info("Feature rows written: %s", len(rows))
    logger.info("Feature output directory: %s", output_dir)
    return 0


def extract_feature_rows(gaze_csv: Path, roi_geometry_csv: Path) -> list[dict[str, object]]:
    roi_index = {row["roi_id"]: row for row in _read_csv(roi_geometry_csv) if row.get("rejection_reason", "") == "" and row.get("is_empty") == "false"}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for sample in _read_csv(gaze_csv):
        if sample.get("roi_id") in roi_index:
            grouped[(sample["session_id"], sample["roi_id"])].append(sample)

    slice_context = _slice_context(roi_index.values())
    feature_rows: list[dict[str, object]] = []
    for (_session_id, roi_id), samples in sorted(grouped.items()):
        samples.sort(key=lambda row: int(float(row["sample_index"])))
        roi = roi_index[roi_id]
        inside, near, same_slice = roi_masks_for_samples(samples, roi)
        fixations = detect_fixations(samples)
        row: dict[str, object] = {column: samples[0][column] for column in METADATA_COLUMNS}
        row.update(extract_roi_features(samples, roi, fixations))
        row.update(extract_scanpath_features(samples, inside, near, same_slice))
        row.update(extract_temporal_features(samples, inside, fixations))
        row.update(_quality_features(samples))
        row.update(_geometry_features(roi, slice_context))
        feature_rows.append(row)
    return feature_rows


def _quality_features(samples: list[dict[str, str]]) -> dict[str, float]:
    return {
        "gaze_validity_ratio": _ratio(samples, "is_valid"),
        "dropout_ratio": _ratio(samples, "is_dropout"),
        "blink_ratio": _ratio(samples, "is_blink"),
        "invalid_burst_ratio": _ratio(samples, "is_invalid_burst"),
        "outside_ct_ratio": _ratio(samples, "is_outside_ct"),
        "jitter_px": _mean([(float(row["jitter_x"]) ** 2 + float(row["jitter_y"]) ** 2) ** 0.5 for row in samples]),
    }


def _geometry_features(roi: dict[str, str], slice_context: dict[tuple[str, str], int]) -> dict[str, float | int]:
    rows = float(roi["rows"])
    columns = float(roi["columns"])
    key = (roi["ct_series_instance_uid"], roi["slice_index"])
    rois_on_slice = slice_context.get(key, 1)
    return {
        "roi_area_px": float(roi["mask_area_px"]),
        "roi_bbox_width": float(roi["bbox_width"]),
        "roi_bbox_height": float(roi["bbox_height"]),
        "roi_center_x": float(roi["centroid_x"]),
        "roi_center_y": float(roi["centroid_y"]),
        "normalized_roi_position_x": float(roi["centroid_x"]) / columns if columns else 0.0,
        "normalized_roi_position_y": float(roi["centroid_y"]) / rows if rows else 0.0,
        "number_of_rois_on_slice": rois_on_slice,
        "roi_density_context": rois_on_slice / max(1.0, rows * columns / 100000.0),
    }


def _slice_context(rois: object) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for roi in rois:
        counts[(roi["ct_series_instance_uid"], roi["slice_index"])] += 1
    return counts


def _slice_level_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["session_id"]), str(row["case_id"]), str(row["slice_index"]))].append(row)
    output: list[dict[str, object]] = []
    for (_session_id, _case_id, _slice), group in sorted(grouped.items()):
        base = {column: group[0][column] for column in METADATA_COLUMNS}
        base["roi_count_on_slice"] = len(group)
        base["mean_gaze_validity_ratio"] = _mean([float(row["gaze_validity_ratio"]) for row in group])
        base["mean_total_gaze_time_inside_roi_ms"] = _mean([float(row["total_gaze_time_inside_roi_ms"]) for row in group])
        output.append(base)
    return output


def _ratio(rows: list[dict[str, str]], key: str) -> float:
    return sum(_bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
