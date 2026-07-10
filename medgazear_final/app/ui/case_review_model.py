"""Case-level CT review model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.ui.dicom_ct_loader import CTSeries, load_ct_series_for_case


MASK_IOU_THRESHOLD = 0.50
BBOX_IOU_THRESHOLD = 0.50
CENTROID_DISTANCE_RATIO = 0.50


@dataclass
class CaseReviewModel:
    patient_id: str
    series_uid: str
    ct_series: CTSeries
    roi_geometry: pd.DataFrame
    review_targets: pd.DataFrame
    features: pd.DataFrame
    attention: pd.DataFrame
    gaze: pd.DataFrame

    @property
    def total_slices(self) -> int:
        return self.ct_series.total_slices

    def image_for_slice(self, slice_index: int, window_center: float | None = None, window_width: float | None = None) -> Image.Image | None:
        return self.ct_series.image_for_index(slice_index, center=window_center, width=window_width)

    def rois_on_slice(self, slice_index: int) -> pd.DataFrame:
        return self.review_targets[pd.to_numeric(self.review_targets["ct_stack_index"], errors="coerce").fillna(-1).astype(int) == int(slice_index)]

    def raw_rois_on_slice(self, slice_index: int) -> pd.DataFrame:
        return self.roi_geometry[pd.to_numeric(self.roi_geometry["ct_stack_index"], errors="coerce").fillna(-1).astype(int) == int(slice_index)]

    def gaze_on_slice(self, session_id: str, roi_id: str | int | None, slice_index: int | None = None) -> pd.DataFrame:
        if slice_index is None:
            slice_index = int(roi_id) if roi_id is not None else -1
            roi_filter = None
        else:
            roi_filter = None if roi_id is None else str(roi_id)
        rows = self.session_gaze_samples(session_id)
        rows = rows[pd.to_numeric(rows.get("ct_stack_index"), errors="coerce").fillna(-1).astype(int) == int(slice_index)].copy()
        if roi_filter is not None and "roi_id" in rows.columns:
            rows = rows[rows["roi_id"].astype(str) == roi_filter].copy()
        return rows

    def case_gaze_samples(self) -> pd.DataFrame:
        mapping = self.roi_geometry[["roi_id", "ct_stack_index"]].drop_duplicates()
        rows = self.gaze.merge(mapping, on="roi_id", how="inner")
        return rows.sort_values(["timestamp_ms", "session_id"]).reset_index(drop=True)

    def session_gaze_samples(self, session_id: str) -> pd.DataFrame:
        mapping = self.roi_geometry[["roi_id", "ct_stack_index", "slice_index"]].drop_duplicates()
        rows = self.gaze[self.gaze["session_id"].astype(str) == str(session_id)].copy()
        if "ct_stack_index" not in rows.columns:
            rows = rows.merge(mapping, on="roi_id", how="left")
        else:
            rows = rows.merge(mapping, on="roi_id", how="left", suffixes=("", "_roi"))
            rows["ct_stack_index"] = pd.to_numeric(rows.get("ct_stack_index"), errors="coerce").fillna(pd.to_numeric(rows.get("ct_stack_index_roi"), errors="coerce"))
            rows["slice_index"] = pd.to_numeric(rows.get("slice_index"), errors="coerce").fillna(pd.to_numeric(rows.get("slice_index_roi"), errors="coerce"))
            rows = rows.drop(columns=[column for column in ("ct_stack_index_roi", "slice_index_roi") if column in rows.columns])
        if "ct_stack_index" not in rows.columns and "slice_index" in rows.columns:
            rows["ct_stack_index"] = pd.to_numeric(rows["slice_index"], errors="coerce")
        return rows.sort_values("timestamp_ms").reset_index(drop=True)

    def roi_slice_indices(self) -> list[int]:
        return sorted(pd.to_numeric(self.roi_geometry["ct_stack_index"], errors="coerce").dropna().astype(int).unique().tolist())

    def summary(self) -> dict[str, object]:
        statuses = self.attention["rule_attention_status"].value_counts().to_dict() if "rule_attention_status" in self.attention else {}
        distinct_rois = distinct_roi_count(self.roi_geometry)
        return {
            "patient_id": self.patient_id,
            "series_uid": self.series_uid,
            "total_slices": self.total_slices,
            "roi_slices": int(self.roi_geometry["ct_stack_index"].nunique()),
            "roi_count": int(distinct_rois),
            "roi_frame_count": int(len(self.roi_geometry)),
            "evaluated_roi_episodes": int(len(self.features)),
            "uncovered_seg_rois": int(max(0, distinct_rois - distinct_roi_count(self.features))) if "roi_id" in self.features else int(distinct_rois),
            "reviewed": int(statuses.get("reviewed", 0)),
            "weakly_reviewed": int(statuses.get("weakly_reviewed", 0)),
            "not_reviewed": int(statuses.get("not_reviewed", 0)),
            "not_evaluated": int(statuses.get("not_evaluated", 0)),
        }


def build_case_review_model(case_id: str, data, series_uid: str | None = None) -> CaseReviewModel:
    geometry = data.roi_geometry[data.roi_geometry["patient_id"].astype(str) == str(case_id)].copy()
    if geometry.empty:
        raise ValueError(f"No ROI geometry found for case {case_id}")
    if series_uid is not None:
        geometry = geometry[geometry["ct_series_instance_uid"].astype(str) == str(series_uid)].copy()
        if geometry.empty:
            raise ValueError(f"No ROI geometry found for case {case_id}, series {series_uid}")
    series_uid = str(series_uid or geometry.iloc[0]["ct_series_instance_uid"])
    ct_series = load_ct_series_for_case(case_id, series_uid, data.dicom_inventory)
    sop_to_stack_index = {slice_.sop_instance_uid: slice_.slice_index for slice_ in ct_series.slices}
    geometry["ct_stack_index"] = geometry["ct_sop_instance_uid"].astype(str).map(sop_to_stack_index).fillna(-1).astype(int)
    geometry = geometry[geometry["ct_stack_index"] >= 0].copy()
    geometry["source_row_index"] = geometry.index.astype(int)
    case_roi_ids = set(geometry["roi_id"].astype(str))
    features = data.features[data.features["roi_id"].astype(str).isin(case_roi_ids)].copy()
    if "case_id" in features.columns:
        features = features[features["case_id"].astype(str) == str(case_id)].copy()
    if not features.empty:
        features = features.merge(geometry[["roi_id", "ct_stack_index", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "centroid_x", "centroid_y"]], on="roi_id", how="left")
    attention = data.attention[data.attention["roi_id"].astype(str).isin(case_roi_ids)].copy()
    if not attention.empty:
        status = attention.sort_values("roi_id").drop_duplicates("roi_id")[["roi_id", "rule_attention_status"]]
        geometry = geometry.merge(status, on="roi_id", how="left")
        geometry["rule_attention_status"] = geometry["rule_attention_status"].fillna("not_evaluated")
    else:
        geometry["rule_attention_status"] = "not_evaluated"
    review_targets = build_review_targets(geometry)
    return CaseReviewModel(case_id, series_uid, ct_series, geometry, review_targets, features, attention, data.gaze)


def build_review_targets(geometry: pd.DataFrame) -> pd.DataFrame:
    if geometry.empty:
        return geometry.copy()
    rows = geometry.copy().reset_index(drop=True)
    if "source_row_index" not in rows.columns:
        rows["source_row_index"] = rows.index.astype(int)
    targets: list[dict[str, object]] = []
    for slice_index, group in rows.groupby("ct_stack_index", sort=True):
        explicit_groups = _group_slice_annotations(group)
        for target_index, target_group in enumerate(explicit_groups, start=1):
            representative = target_group.iloc[0].copy()
            review_target_id = f"{str(representative.get('patient_id', 'case'))}_{str(representative.get('ct_sop_instance_uid', 'sop'))}_target_{target_index:02d}"
            bbox_x_min = pd.to_numeric(target_group["bbox_x_min"], errors="coerce").min()
            bbox_y_min = pd.to_numeric(target_group["bbox_y_min"], errors="coerce").min()
            bbox_x_max = pd.to_numeric(target_group["bbox_x_max"], errors="coerce").max()
            bbox_y_max = pd.to_numeric(target_group["bbox_y_max"], errors="coerce").max()
            has_masks = bool("mask_npz_path" in target_group.columns and target_group["mask_npz_path"].astype(str).str.strip().any())
            target_row = representative.to_dict()
            target_row.update(
                {
                    "roi_id": review_target_id,
                    "review_target_id": review_target_id,
                    "seg_frame_index": int(representative.get("slice_index", 0) or 0),
                    "slice_index": int(slice_index),
                    "ct_stack_index": int(slice_index),
                    "representative_raw_roi_id": str(representative.get("roi_id", "")),
                    "raw_roi_ids": ",".join(target_group["roi_id"].astype(str).tolist()),
                    "annotation_instance_count": int(len(target_group)),
                    "renderable_contour_count": 1,
                    "consolidation_method": "union_mask" if has_masks else "bounding_box_fallback",
                    "bbox_x_min": bbox_x_min,
                    "bbox_y_min": bbox_y_min,
                    "bbox_x_max": bbox_x_max,
                    "bbox_y_max": bbox_y_max,
                    "bbox_width": float(bbox_x_max - bbox_x_min + 1),
                    "bbox_height": float(bbox_y_max - bbox_y_min + 1),
                    "centroid_x": float(pd.to_numeric(target_group["centroid_x"], errors="coerce").mean()),
                    "centroid_y": float(pd.to_numeric(target_group["centroid_y"], errors="coerce").mean()),
                    "source_row_indices": ",".join(target_group["source_row_index"].astype(int).astype(str).tolist()),
                    "source_annotation_ids": ",".join(target_group["roi_id"].astype(str).tolist()),
                    "source_annotator_ids": ",".join(sorted({str(value) for value in target_group.get("annotation_id", pd.Series(dtype=object)).dropna().astype(str).tolist() if value})) if "annotation_id" in target_group.columns else "",
                    "source_mask_paths": ",".join(target_group["mask_npz_path"].astype(str).tolist()) if "mask_npz_path" in target_group.columns else "",
                    "review_identity_kind": _review_identity_kind(target_group),
                }
            )
            targets.append(target_row)
    return pd.DataFrame(targets).sort_values(["ct_stack_index", "roi_id"]).reset_index(drop=True)


def _group_slice_annotations(group: pd.DataFrame) -> list[pd.DataFrame]:
    if group.empty:
        return []
    clusters: list[list[int]] = []
    used: set[int] = set()
    group = group.reset_index(drop=True)
    for index, row in group.iterrows():
        if index in used:
            continue
        cluster = [index]
        used.add(index)
        for other_index, other in group.iloc[index + 1 :].iterrows():
            if other_index in used:
                continue
            if _same_review_target(row, other):
                cluster.append(other_index)
                used.add(other_index)
        clusters.append(cluster)
    return [group.iloc[indices].copy().reset_index(drop=True) for indices in clusters]


def _same_review_target(left: pd.Series, right: pd.Series) -> bool:
    for column in ("lesion_id", "nodule_id", "nodule_uid"):
        if column in left.index and column in right.index:
            left_value = str(left.get(column, "") or "").strip()
            right_value = str(right.get(column, "") or "").strip()
            if left_value and right_value:
                return left_value == right_value
    left_mask = load_roi_mask(left)
    right_mask = load_roi_mask(right)
    if left_mask is not None and right_mask is not None and left_mask.shape == right_mask.shape:
        if _mask_iou(left_mask, right_mask) >= MASK_IOU_THRESHOLD and _compatible_centroid_distance(left, right):
            return True
        return False
    left_box = _bbox(left)
    right_box = _bbox(right)
    if _bbox_iou(left_box, right_box) >= BBOX_IOU_THRESHOLD and _compatible_centroid_distance(left, right):
        return True
    return False


def _review_identity_kind(group: pd.DataFrame) -> str:
    for column in ("lesion_id", "nodule_id", "nodule_uid"):
        if column in group.columns and group[column].notna().any():
            return f"grouped_by_{column}"
    return "grouped_by_spatial_overlap"


def _bbox(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row.get("bbox_x_min", 0) or 0),
        float(row.get("bbox_y_min", 0) or 0),
        float(row.get("bbox_x_max", 0) or 0),
        float(row.get("bbox_y_max", 0) or 0),
    )


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 < x0 or y1 < y0:
        return 0.0
    intersection = (x1 - x0 + 1) * (y1 - y0 + 1)
    left_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    right_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    union = max(1.0, left_area + right_area - intersection)
    return float(intersection / union)


def _centroid_distance(left: pd.Series, right: pd.Series) -> float:
    dx = float(left.get("centroid_x", 0) or 0) - float(right.get("centroid_x", 0) or 0)
    dy = float(left.get("centroid_y", 0) or 0) - float(right.get("centroid_y", 0) or 0)
    return float((dx * dx + dy * dy) ** 0.5)


def _compatible_centroid_distance(left: pd.Series, right: pd.Series) -> bool:
    distance = _centroid_distance(left, right)
    size = max(
        float(left.get("bbox_width", 0) or (float(left.get("bbox_x_max", 0) or 0) - float(left.get("bbox_x_min", 0) or 0) + 1)),
        float(left.get("bbox_height", 0) or (float(left.get("bbox_y_max", 0) or 0) - float(left.get("bbox_y_min", 0) or 0) + 1)),
        float(right.get("bbox_width", 0) or (float(right.get("bbox_x_max", 0) or 0) - float(right.get("bbox_x_min", 0) or 0) + 1)),
        float(right.get("bbox_height", 0) or (float(right.get("bbox_y_max", 0) or 0) - float(right.get("bbox_y_min", 0) or 0) + 1)),
        1.0,
    )
    return distance <= size * CENTROID_DISTANCE_RATIO


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left).astype(bool)
    right_mask = np.asarray(right).astype(bool)
    intersection = float(np.logical_and(left_mask, right_mask).sum())
    union = float(np.logical_or(left_mask, right_mask).sum())
    return 0.0 if union <= 0 else intersection / union


def load_roi_mask(row: pd.Series) -> np.ndarray | None:
    raw_roi_ids = [value for value in str(row.get("raw_roi_ids", "")).split(",") if value]
    source_paths = [value for value in str(row.get("source_mask_paths", row.get("mask_npz_path", ""))).split(",") if value]
    if raw_roi_ids and source_paths:
        masks: list[np.ndarray] = []
        for roi_id, path_value in zip(raw_roi_ids, source_paths):
            path = Path(str(path_value))
            if not path.exists():
                continue
            try:
                data = np.load(path, allow_pickle=True)
                roi_ids = [str(item) for item in data["roi_ids"]]
                if roi_id not in roi_ids:
                    continue
                masks.append(data["masks"][roi_ids.index(roi_id)].astype(bool))
            except Exception:
                continue
        if masks:
            merged = np.asarray(masks[0]).astype(bool)
            for mask in masks[1:]:
                if mask.shape == merged.shape:
                    merged = np.logical_or(merged, mask)
            return merged
    path = Path(str(row.get("mask_npz_path", "")))
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        roi_ids = [str(item) for item in data["roi_ids"]]
        index = roi_ids.index(str(row["roi_id"]))
        return data["masks"][index].astype(bool)
    except Exception:
        return None


def distinct_roi_count(rows: pd.DataFrame) -> int:
    if rows.empty or "roi_id" not in rows.columns:
        return 0
    return int(rows["roi_id"].astype(str).map(base_roi_id).nunique())


def base_roi_id(roi_id: object) -> str:
    value = str(roi_id)
    marker = "__frame_"
    return value.split(marker, 1)[0] if marker in value else value
