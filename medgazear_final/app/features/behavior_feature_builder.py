"""Shared behavior feature building for offline extraction and live inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from app.features.feature_schema import METADATA_COLUMNS, SCANPATH_FEATURES, TEMPORAL_FEATURES, QUALITY_FEATURES, GEOMETRY_FEATURES, behavior_feature_columns
from app.features.roi_feature_extractor import _sample_interval_ms
from app.features.roi_spatial_modes import RoiMaskLibrary, classify_samples_for_roi, point_in_roi, sample_disagreement
from app.features.scanpath_feature_extractor import extract_scanpath_features
from app.features.temporal_feature_extractor import detect_fixations, extract_temporal_features


@dataclass(frozen=True)
class FeatureBuildResult:
    row: dict[str, object]
    sample_count: int
    valid_sample_count: int
    fixation_count: int
    ready: bool


class FrameRoiFeatureAccumulator:
    def __init__(self, roi_row: Mapping[str, object], metadata: Mapping[str, object] | None = None, geometry_mode: str = "bbox", mask_library: RoiMaskLibrary | None = None) -> None:
        self.roi_row = dict(roi_row)
        self.metadata = dict(metadata or {})
        self.geometry_mode = geometry_mode
        self.mask_library = mask_library
        self.samples: list[dict[str, object]] = []

    def add_sample(self, sample: Mapping[str, object]) -> None:
        self.samples.append(dict(sample))

    def add_samples(self, samples: pd.DataFrame | list[dict[str, object]]) -> None:
        if isinstance(samples, pd.DataFrame):
            self.samples.extend(samples.to_dict("records"))
        else:
            self.samples.extend(dict(sample) for sample in samples)

    def build(self) -> FeatureBuildResult:
        frame = pd.DataFrame(self.samples)
        return build_behavior_feature_row(frame, self.roi_row, self.metadata, geometry_mode=self.geometry_mode, mask_library=self.mask_library)


def build_behavior_feature_row(
    samples: pd.DataFrame,
    roi_row: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    geometry_mode: str = "bbox",
    mask_library: RoiMaskLibrary | None = None,
) -> FeatureBuildResult:
    roi = _normalized_roi_row(roi_row)
    rows = _normalized_samples(samples, int(roi["slice_index"]))
    if rows.empty:
        row = _empty_row(roi, metadata or {})
        row["geometry_mode"] = geometry_mode
        return FeatureBuildResult(row=row, sample_count=0, valid_sample_count=0, fixation_count=0, ready=False)

    sample_dicts = rows.to_dict("records")
    fixations = detect_fixations(sample_dicts)
    inside, near, same_slice = classify_samples_for_roi(sample_dicts, roi, geometry_mode=geometry_mode, mask_library=mask_library)
    row = _metadata_row(sample_dicts[0], roi, metadata or {})
    row.update(_roi_features(sample_dicts, roi, fixations, inside, near, same_slice, geometry_mode, mask_library))
    row.update(extract_scanpath_features(sample_dicts, inside, near, same_slice))
    row.update(extract_temporal_features(sample_dicts, inside, fixations))
    row.update(_quality_features(sample_dicts))
    row.update(_geometry_features(roi, _slice_context([roi])))
    valid_samples = [sample for sample in sample_dicts if _bool(sample.get("is_valid", False)) and not _bool(sample.get("is_outside_ct", False)) and not _bool(sample.get("is_ui_glance", False))]
    ready = len(sample_dicts) >= 2 and len(valid_samples) > 0
    row["geometry_mode"] = geometry_mode
    row["_sample_count"] = len(sample_dicts)
    row["_valid_sample_count"] = len(valid_samples)
    row["_fixation_ready"] = len(sample_dicts) >= 2
    row["_fixation_count_runtime"] = len(fixations)
    return FeatureBuildResult(row=row, sample_count=len(sample_dicts), valid_sample_count=len(valid_samples), fixation_count=len(fixations), ready=ready)


def feature_parity_matrix(offline_row: Mapping[str, object], live_row: Mapping[str, object], feature_columns: list[str], tolerance: float = 1e-6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        offline_value = offline_row.get(feature)
        live_value = live_row.get(feature)
        missing_offline = _is_missing(offline_value)
        missing_live = _is_missing(live_value)
        if missing_offline and missing_live:
            status = "NOT APPLICABLE"
            diff = None
            rel = None
            passed = True
        elif missing_offline:
            status = "MISSING TRAINING"
            diff = None
            rel = None
            passed = False
        elif missing_live:
            status = "MISSING LIVE"
            diff = None
            rel = None
            passed = False
        else:
            diff = abs(float(offline_value) - float(live_value))
            rel = diff / max(1e-9, abs(float(offline_value))) if float(offline_value) != 0 else diff
            passed = diff <= tolerance or rel <= 1e-6
            status = "EXACT MATCH" if passed else "SEMANTIC MISMATCH"
        rows.append(
            {
                "feature_name": feature,
                "offline_value": offline_value,
                "live_value": live_value,
                "absolute_difference": diff,
                "relative_difference": rel,
                "semantic_status": status,
                "availability_status": "present" if not (missing_offline or missing_live) else "missing",
                "parity_pass": passed,
            }
        )
    return rows


def validate_feature_schema(feature_row: Mapping[str, object], feature_columns: list[str]) -> dict[str, object]:
    duplicates = [column for index, column in enumerate(feature_columns) if column in feature_columns[:index]]
    missing = [column for column in feature_columns if column not in feature_row or _is_missing(feature_row.get(column))]
    unexpected = [column for column in feature_row if column not in set(feature_columns) and column in behavior_feature_columns()]
    return {
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "valid": not duplicates and not missing,
    }


def _normalized_samples(samples: pd.DataFrame, default_slice_index: int) -> pd.DataFrame:
    rows = samples.copy()
    if rows.empty:
        return rows
    if "slice_index" not in rows.columns:
        rows["slice_index"] = default_slice_index
    rows["slice_index"] = pd.to_numeric(rows["slice_index"], errors="coerce").fillna(default_slice_index).astype(int)
    sort_columns = [column for column in ("sample_index", "timestamp_ms") if column in rows.columns]
    rows = rows.sort_values(sort_columns or [rows.index.name or rows.columns[0]]).reset_index(drop=True)
    if "sample_index" not in rows.columns:
        rows["sample_index"] = list(range(len(rows)))
    defaults = {
        "session_id": "session",
        "reader_id": "reader",
        "reader_profile": "unknown",
        "case_id": "case",
        "roi_id": str(roi_row_id_placeholder()),
        "hidden_behavior_label": "",
        "timestamp_ms": 0.0,
        "image_x": 0.0,
        "image_y": 0.0,
        "is_valid": False,
        "is_dropout": False,
        "is_blink": False,
        "is_invalid_burst": False,
        "is_outside_ct": False,
        "is_ui_glance": False,
        "jitter_x": 0.0,
        "jitter_y": 0.0,
    }
    for key, value in defaults.items():
        if key not in rows.columns:
            rows[key] = value
    rows["timestamp_ms"] = pd.to_numeric(rows["timestamp_ms"], errors="coerce").fillna(0.0)
    rows["image_x"] = pd.to_numeric(rows["image_x"], errors="coerce").fillna(0.0)
    rows["image_y"] = pd.to_numeric(rows["image_y"], errors="coerce").fillna(0.0)
    for key in ("jitter_x", "jitter_y"):
        rows[key] = pd.to_numeric(rows[key], errors="coerce").fillna(0.0)
    for key in ("is_valid", "is_dropout", "is_blink", "is_invalid_burst", "is_outside_ct", "is_ui_glance"):
        rows[key] = rows[key].map(_bool)
    return rows


def _metadata_row(sample: Mapping[str, object], roi: Mapping[str, object], metadata: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}
    defaults = {
        "session_id": sample.get("session_id", "session"),
        "reader_id": sample.get("reader_id", metadata.get("reader_id", "reader")),
        "reader_profile": sample.get("reader_profile", metadata.get("reader_profile", "unknown")),
        "case_id": sample.get("case_id", metadata.get("case_id", roi.get("patient_id", "case"))),
        "roi_id": roi.get("roi_id", sample.get("roi_id", "")),
        "slice_index": int(roi.get("slice_index", sample.get("slice_index", 0))),
        "hidden_behavior_label": metadata.get("hidden_behavior_label", sample.get("hidden_behavior_label", "")),
    }
    for column in METADATA_COLUMNS:
        row[column] = defaults[column]
    return row


def _normalized_roi_row(roi_row: Mapping[str, object]) -> dict[str, object]:
    roi = dict(roi_row)
    roi["bbox_width"] = float(roi.get("bbox_width") or (float(roi.get("bbox_x_max", 0) or 0) - float(roi.get("bbox_x_min", 0) or 0)))
    roi["bbox_height"] = float(roi.get("bbox_height") or (float(roi.get("bbox_y_max", 0) or 0) - float(roi.get("bbox_y_min", 0) or 0)))
    roi["slice_index"] = int(float(roi.get("slice_index", roi.get("ct_stack_index", 0)) or 0))
    roi["ct_stack_index"] = int(float(roi.get("ct_stack_index", roi.get("slice_index", 0)) or 0))
    roi["rows"] = float(roi.get("rows", 512) or 512)
    roi["columns"] = float(roi.get("columns", 512) or 512)
    roi.setdefault("mask_area_px", 0.0)
    roi.setdefault("centroid_x", 0.0)
    roi.setdefault("centroid_y", 0.0)
    return roi


def _empty_row(roi: Mapping[str, object], metadata: Mapping[str, object]) -> dict[str, object]:
    row = _metadata_row({}, roi, metadata)
    for feature in behavior_feature_columns():
        if feature not in row:
            row[feature] = 0.0
    row["time_to_first_roi_fixation_ms"] = -1.0
    row["slice_index"] = int(roi.get("slice_index", 0))
    return row


def _roi_features(
    samples: list[dict[str, object]],
    roi: Mapping[str, object],
    fixations: list[dict[str, float | int]],
    inside: list[bool],
    near: list[bool],
    same_slice: list[bool],
    geometry_mode: str,
    mask_library: RoiMaskLibrary | None,
) -> dict[str, float | int]:
    sample_ms = _sample_interval_ms(samples)
    valid = [_bool(sample["is_valid"]) for sample in samples]
    valid_ct = [
        is_valid and not _bool(sample.get("is_outside_ct", False)) and not _bool(sample.get("is_ui_glance", False))
        for sample, is_valid in zip(samples, valid)
    ]
    inside_valid = [hit and is_valid for hit, is_valid in zip(inside, valid_ct)]
    near_valid = [hit and is_valid for hit, is_valid in zip(near, valid_ct)]
    margin = max(12.0, max(float(roi["bbox_width"]), float(roi["bbox_height"])) * 0.75)
    inside_fix = [fix for fix in fixations if point_in_roi(float(fix["x"]), float(fix["y"]), roi, geometry_mode=geometry_mode, margin=0.0, mask_library=mask_library)]
    near_fix = [fix for fix in fixations if point_in_roi(float(fix["x"]), float(fix["y"]), roi, geometry_mode=geometry_mode, margin=margin, mask_library=mask_library)]
    first_idx = next((idx for idx, hit in enumerate(inside_valid) if hit), None)
    min_distance = _minimum_distance_to_roi(samples, roi, geometry_mode, mask_library)
    return {
        "total_gaze_time_inside_roi_ms": sum(inside_valid) * sample_ms,
        "total_gaze_time_near_roi_ms": sum(near_valid) * sample_ms,
        "gaze_hit_count_inside_roi": int(sum(inside_valid)),
        "gaze_hit_count_near_roi": int(sum(near_valid)),
        "fixation_count_inside_roi": len(inside_fix),
        "fixation_count_near_roi": len(near_fix) - len(inside_fix),
        "mean_fixation_duration_inside_roi_ms": _mean([float(fix["duration_ms"]) for fix in inside_fix]),
        "max_fixation_duration_inside_roi_ms": max([float(fix["duration_ms"]) for fix in inside_fix], default=0.0),
        "time_to_first_roi_fixation_ms": float(samples[first_idx]["timestamp_ms"]) if first_idx is not None else -1.0,
        "min_distance_to_roi_px": min_distance,
        "valid_gaze_time_on_roi_slice_ms": sum(hit and is_valid for hit, is_valid in zip(same_slice, valid_ct)) * sample_ms,
        "time_on_roi_slice_ms": sum(same_slice) * sample_ms,
    }


def _minimum_distance_to_roi(
    samples: list[dict[str, object]],
    roi: Mapping[str, object],
    geometry_mode: str,
    mask_library: RoiMaskLibrary | None,
) -> float:
    valid_samples = [sample for sample in samples if _bool(sample.get("is_valid", False)) and not _bool(sample.get("is_outside_ct", False)) and not _bool(sample.get("is_ui_glance", False))]
    if not valid_samples:
        return -1.0
    if geometry_mode == "mask" and mask_library is not None:
        distances = []
        for sample in valid_samples:
            disagreement = sample_disagreement(sample, roi, mask_library)
            if disagreement["mask_inside"]:
                return 0.0
            x = float(sample.get("image_x", 0.0) or 0.0)
            y = float(sample.get("image_y", 0.0) or 0.0)
            distances.append(((x - float(roi.get("centroid_x", 0.0) or 0.0)) ** 2 + (y - float(roi.get("centroid_y", 0.0) or 0.0)) ** 2) ** 0.5)
        return min(distances, default=-1.0)
    distances = []
    centroid_x = float(roi.get("centroid_x", 0.0) or 0.0)
    centroid_y = float(roi.get("centroid_y", 0.0) or 0.0)
    for sample in valid_samples:
        if point_in_roi(float(sample.get("image_x", 0.0) or 0.0), float(sample.get("image_y", 0.0) or 0.0), roi, geometry_mode=geometry_mode, margin=0.0, mask_library=mask_library):
            return 0.0
        x = float(sample.get("image_x", 0.0) or 0.0)
        y = float(sample.get("image_y", 0.0) or 0.0)
        distances.append(((x - centroid_x) ** 2 + (y - centroid_y) ** 2) ** 0.5)
    return min(distances, default=-1.0)


def _quality_features(samples: list[dict[str, object]]) -> dict[str, float]:
    return {
        "gaze_validity_ratio": _ratio(samples, "is_valid"),
        "dropout_ratio": _ratio(samples, "is_dropout"),
        "blink_ratio": _ratio(samples, "is_blink"),
        "invalid_burst_ratio": _ratio(samples, "is_invalid_burst"),
        "outside_ct_ratio": _ratio(samples, "is_outside_ct"),
        "jitter_px": _mean([(float(row.get("jitter_x", 0.0)) ** 2 + float(row.get("jitter_y", 0.0)) ** 2) ** 0.5 for row in samples]),
    }


def _geometry_features(roi: Mapping[str, object], slice_context: dict[tuple[str, str], int]) -> dict[str, float | int]:
    rows = float(roi["rows"])
    columns = float(roi["columns"])
    key = (str(roi.get("ct_series_instance_uid", "")), str(roi.get("slice_index", 0)))
    rois_on_slice = slice_context.get(key, 1)
    return {
        "roi_area_px": float(roi.get("mask_area_px", 0.0) or 0.0),
        "roi_bbox_width": float(roi.get("bbox_width", 0.0) or 0.0),
        "roi_bbox_height": float(roi.get("bbox_height", 0.0) or 0.0),
        "roi_center_x": float(roi.get("centroid_x", 0.0) or 0.0),
        "roi_center_y": float(roi.get("centroid_y", 0.0) or 0.0),
        "normalized_roi_position_x": float(roi.get("centroid_x", 0.0) or 0.0) / columns if columns else 0.0,
        "normalized_roi_position_y": float(roi.get("centroid_y", 0.0) or 0.0) / rows if rows else 0.0,
        "number_of_rois_on_slice": rois_on_slice,
        "roi_density_context": rois_on_slice / max(1.0, rows * columns / 100000.0),
    }


def _slice_context(rois: list[Mapping[str, object]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for roi in rois:
        counts[(str(roi.get("ct_series_instance_uid", "")), str(roi.get("slice_index", 0)))] += 1
    return counts


def _ratio(rows: list[dict[str, object]], key: str) -> float:
    return sum(_bool(row.get(key, False)) for row in rows) / len(rows) if rows else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def roi_row_id_placeholder() -> str:
    return "roi"
