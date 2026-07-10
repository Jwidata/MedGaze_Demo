"""ROI spatial interaction helpers for bbox and true-mask modes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from app.roi.roi_mask_store import load_roi_masks


@dataclass(frozen=True)
class MaskResolution:
    roi_id: str
    mask_path: str
    mask_shape: tuple[int, int] | None
    ct_shape: tuple[int, int]
    ct_stack_index: int | None
    ct_sop_instance_uid: str
    resolved: bool
    reason: str


class RoiMaskLibrary:
    def __init__(self) -> None:
        self._mask_cache: dict[str, np.ndarray] = {}
        self._distance_cache: dict[str, np.ndarray] = {}
        self.mask_load_count = 0
        self.distance_build_count = 0

    def audit_row(self, roi_row: Mapping[str, object]) -> MaskResolution:
        roi_id = str(roi_row.get("roi_id", ""))
        path = str(roi_row.get("mask_npz_path", "") or "")
        ct_shape = (int(float(roi_row.get("rows", 0) or 0)), int(float(roi_row.get("columns", 0) or 0)))
        ct_stack_index = _int_or_none(roi_row.get("ct_stack_index"))
        ct_sop = str(roi_row.get("ct_sop_instance_uid", "") or "")
        mask = self.get_mask(roi_row)
        if mask is None:
            return MaskResolution(roi_id, path, None, ct_shape, ct_stack_index, ct_sop, False, "mask_not_found")
        if mask.shape != ct_shape:
            return MaskResolution(roi_id, path, mask.shape, ct_shape, ct_stack_index, ct_sop, False, "shape_mismatch")
        return MaskResolution(roi_id, path, mask.shape, ct_shape, ct_stack_index, ct_sop, True, "ok")

    def get_mask(self, roi_row: Mapping[str, object]) -> np.ndarray | None:
        roi_id = str(roi_row.get("roi_id", ""))
        if roi_id in self._mask_cache:
            return self._mask_cache[roi_id]
        path_value = str(roi_row.get("mask_npz_path", "") or "")
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        try:
            data = load_roi_masks(path)
            roi_ids = [str(item) for item in data["roi_ids"]]
            index = roi_ids.index(roi_id)
            mask = data["masks"][index].astype(bool)
            self._mask_cache[roi_id] = mask
            self.mask_load_count += 1
            return mask
        except Exception:
            return None

    def get_distance_field(self, roi_row: Mapping[str, object]) -> np.ndarray | None:
        roi_id = str(roi_row.get("roi_id", ""))
        if roi_id in self._distance_cache:
            return self._distance_cache[roi_id]
        mask = self.get_mask(roi_row)
        if mask is None:
            return None
        distance = distance_transform_edt(~mask)
        self._distance_cache[roi_id] = distance
        self.distance_build_count += 1
        return distance


def classify_samples_for_roi(
    samples: list[dict[str, object]],
    roi_row: Mapping[str, object],
    geometry_mode: str = "bbox",
    mask_library: RoiMaskLibrary | None = None,
) -> tuple[list[bool], list[bool], list[bool]]:
    same_slice = [str(sample.get("slice_index")) == str(_slice_index_for_row(roi_row)) for sample in samples]
    if geometry_mode == "bbox":
        return _classify_bbox(samples, roi_row, same_slice)
    if geometry_mode == "mask":
        return _classify_mask(samples, roi_row, same_slice, mask_library or RoiMaskLibrary())
    raise ValueError(f"Unknown geometry mode: {geometry_mode}")


def point_in_roi(
    x: float,
    y: float,
    roi_row: Mapping[str, object],
    geometry_mode: str = "bbox",
    margin: float = 0.0,
    mask_library: RoiMaskLibrary | None = None,
) -> bool:
    if geometry_mode == "bbox":
        return _point_in_bbox(x, y, roi_row, margin)
    if geometry_mode == "mask":
        return _point_in_mask(x, y, roi_row, margin, mask_library or RoiMaskLibrary())
    raise ValueError(f"Unknown geometry mode: {geometry_mode}")


def sample_disagreement(sample: Mapping[str, object], roi_row: Mapping[str, object], mask_library: RoiMaskLibrary) -> dict[str, object]:
    x = float(sample.get("image_x", 0) or 0)
    y = float(sample.get("image_y", 0) or 0)
    bbox_inside = _point_in_bbox(x, y, roi_row, 0.0)
    mask_inside = _point_in_mask(x, y, roi_row, 0.0, mask_library)
    margin = _near_margin(roi_row)
    bbox_near = _point_in_bbox(x, y, roi_row, margin) and not bbox_inside
    mask_near = _point_near_mask(x, y, roi_row, margin, mask_library) and not mask_inside
    return {
        "bbox_inside": bbox_inside,
        "mask_inside": mask_inside,
        "bbox_near": bbox_near,
        "mask_near": mask_near,
    }


def mask_fill_ratio(roi_row: Mapping[str, object], mask_library: RoiMaskLibrary) -> float | None:
    mask = mask_library.get_mask(roi_row)
    if mask is None:
        return None
    bbox_area = max(1.0, float(roi_row.get("bbox_width", 0) or 0) * float(roi_row.get("bbox_height", 0) or 0))
    return float(mask.sum()) / bbox_area


def _classify_bbox(samples: list[dict[str, object]], roi_row: Mapping[str, object], same_slice: list[bool]) -> tuple[list[bool], list[bool], list[bool]]:
    x_min = float(roi_row["bbox_x_min"])
    y_min = float(roi_row["bbox_y_min"])
    x_max = float(roi_row["bbox_x_max"])
    y_max = float(roi_row["bbox_y_max"])
    margin = _near_margin(roi_row)
    inside: list[bool] = []
    near: list[bool] = []
    for sample, on_slice in zip(samples, same_slice):
        x = float(sample["image_x"])
        y = float(sample["image_y"])
        in_roi = on_slice and x_min <= x <= x_max and y_min <= y <= y_max
        near_roi = on_slice and x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin and not in_roi
        inside.append(in_roi)
        near.append(near_roi)
    return inside, near, same_slice


def _classify_mask(samples: list[dict[str, object]], roi_row: Mapping[str, object], same_slice: list[bool], mask_library: RoiMaskLibrary) -> tuple[list[bool], list[bool], list[bool]]:
    margin = _near_margin(roi_row)
    inside: list[bool] = []
    near: list[bool] = []
    for sample, on_slice in zip(samples, same_slice):
        x = float(sample["image_x"])
        y = float(sample["image_y"])
        in_roi = on_slice and _point_in_mask(x, y, roi_row, 0.0, mask_library)
        near_roi = on_slice and not in_roi and _point_near_mask(x, y, roi_row, margin, mask_library)
        inside.append(in_roi)
        near.append(near_roi)
    return inside, near, same_slice


def _point_in_bbox(x: float, y: float, roi_row: Mapping[str, object], margin: float) -> bool:
    return float(roi_row["bbox_x_min"]) - margin <= x <= float(roi_row["bbox_x_max"]) + margin and float(roi_row["bbox_y_min"]) - margin <= y <= float(roi_row["bbox_y_max"]) + margin


def _point_in_mask(x: float, y: float, roi_row: Mapping[str, object], margin: float, mask_library: RoiMaskLibrary) -> bool:
    if margin > 0:
        return _point_near_mask(x, y, roi_row, margin, mask_library)
    mask = mask_library.get_mask(roi_row)
    if mask is None:
        return False
    pixel = _point_to_pixel(x, y, mask.shape)
    if pixel is None:
        return False
    py, px = pixel
    return bool(mask[py, px])


def _point_near_mask(x: float, y: float, roi_row: Mapping[str, object], threshold: float, mask_library: RoiMaskLibrary) -> bool:
    mask = mask_library.get_mask(roi_row)
    distance = mask_library.get_distance_field(roi_row)
    if mask is None or distance is None:
        return False
    pixel = _point_to_pixel(x, y, mask.shape)
    if pixel is None:
        return False
    py, px = pixel
    return float(distance[py, px]) <= threshold


def _point_to_pixel(x: float, y: float, shape: tuple[int, int]) -> tuple[int, int] | None:
    height, width = shape
    px = int(round(x))
    py = int(round(y))
    if px < 0 or py < 0 or px >= width or py >= height:
        return None
    return py, px


def _near_margin(roi_row: Mapping[str, object]) -> float:
    return max(12.0, max(float(roi_row.get("bbox_width", 0) or 0), float(roi_row.get("bbox_height", 0) or 0)) * 0.75)


def _slice_index_for_row(roi_row: Mapping[str, object]) -> int:
    try:
        return int(float(roi_row.get("slice_index", roi_row.get("ct_stack_index", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: object) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
