"""Extract ROI, scanpath, temporal, quality, and geometry features."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.features.behavior_feature_builder import build_behavior_feature_row
from app.features.roi_spatial_modes import RoiMaskLibrary
from app.features.feature_quality_report import write_feature_quality_report
from app.features.feature_schema import (
    METADATA_COLUMNS,
    SCANPATH_FEATURES,
    behavior_feature_columns,
    validate_no_leakage,
    write_feature_schema,
)
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Extract ROI-level gaze features from degraded synthetic gaze samples.")
    parser.add_argument("--gaze", required=True, help="Raw behavior-labeled synthetic gaze CSV.")
    parser.add_argument("--roi-geometry", required=True, help="SEG ROI geometry CSV.")
    parser.add_argument("--geometry-mode", choices=["bbox", "mask"], default="bbox")
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = extract_feature_rows(Path(args.gaze), Path(args.roi_geometry), geometry_mode=args.geometry_mode)
    validate_no_leakage(behavior_feature_columns())
    full_columns = behavior_feature_columns() + ["geometry_mode"]
    _write_csv(output_dir / "roi_level_features.csv", rows, full_columns)
    _write_csv(output_dir / "behavior_feature_table.csv", rows, full_columns)
    _write_csv(output_dir / "scanpath_features.csv", rows, METADATA_COLUMNS + SCANPATH_FEATURES + ["geometry_mode"])
    _write_csv(output_dir / "slice_level_features.csv", _slice_level_rows(rows), METADATA_COLUMNS + ["roi_count_on_slice", "mean_gaze_validity_ratio", "mean_total_gaze_time_inside_roi_ms", "geometry_mode"])
    write_feature_schema(output_dir / "feature_schema.md")
    write_feature_quality_report(output_dir / "feature_quality_report.md", rows)
    logger.info("Feature rows written: %s", len(rows))
    logger.info("Feature output directory: %s", output_dir)
    return 0


def extract_feature_rows(gaze_csv: Path, roi_geometry_csv: Path, geometry_mode: str = "bbox") -> list[dict[str, object]]:
    roi_index = {row["roi_id"]: row for row in _read_csv(roi_geometry_csv) if row.get("rejection_reason", "") == "" and row.get("is_empty") == "false"}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    mask_library = RoiMaskLibrary() if geometry_mode == "mask" else None
    for sample in _read_csv(gaze_csv):
        if sample.get("roi_id") in roi_index:
            grouped.setdefault((sample["session_id"], sample["roi_id"]), []).append(sample)

    feature_rows: list[dict[str, object]] = []
    for (_session_id, roi_id), samples in sorted(grouped.items()):
        roi = roi_index[roi_id]
        result = build_behavior_feature_row(pd.DataFrame(samples), roi, geometry_mode=geometry_mode, mask_library=mask_library)
        feature_rows.append(result.row)
    return feature_rows


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
        base["geometry_mode"] = group[0].get("geometry_mode", "bbox")
        output.append(base)
    return output


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
