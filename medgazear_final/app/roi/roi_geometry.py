"""DICOM SEG ROI geometry extraction and reporting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.roi.roi_mask_store import save_roi_masks
from app.roi.seg_loader import SegFrame, SegObject, load_seg_object


@dataclass(frozen=True)
class ROIGeometryResult:
    """Output paths and summary for SEG ROI extraction."""

    output_dir: Path
    geometry_csv: Path
    report_md: Path
    mask_dir: Path
    summary: dict[str, int]


def extract_roi_geometry_from_matches(
    output_root: Path,
    max_seg: int | None = None,
    include_partial: bool = False,
) -> ROIGeometryResult:
    """Extract binary ROI masks from matched SEG objects and write geometry outputs."""

    output_root = output_root.expanduser().resolve()
    roi_dir = output_root / "roi_geometry"
    audit_dir = output_root / "dicom_audit"
    match_table_csv = roi_dir / "ct_seg_match_table.csv"
    inventory_csv = audit_dir / "dicom_inventory.csv"
    seg_inventory_csv = audit_dir / "seg_inventory.csv"
    _require_file(match_table_csv)
    _require_file(inventory_csv)
    _require_file(seg_inventory_csv)

    ct_index = _build_ct_index(inventory_csv)
    seg_inventory = {row.get("sop_instance_uid", ""): row for row in _read_csv(seg_inventory_csv)}
    selected_matches = _selected_match_rows(_read_csv(match_table_csv), include_partial=include_partial, max_seg=max_seg)

    mask_dir = roi_dir / "roi_masks"
    geometry_csv = roi_dir / "seg_roi_geometry.csv"
    report_md = roi_dir / "roi_geometry_report.md"
    geometry_rows: list[dict[str, str]] = []
    summary = {
        "seg_objects_inspected": 0,
        "seg_frames_inspected": 0,
        "accepted_roi_masks": 0,
        "rejected_empty_masks": 0,
        "missing_ct_references": 0,
        "resolution_mismatches": 0,
    }

    for match_row in selected_matches:
        seg_sop_uid = match_row.get("seg_sop_instance_uid", "")
        seg_path = Path(match_row.get("seg_file_path", ""))
        if not seg_path.exists() and seg_sop_uid in seg_inventory:
            seg_path = Path(seg_inventory[seg_sop_uid].get("file_path", ""))
        seg_object = load_seg_object(seg_path)
        summary["seg_objects_inspected"] += 1

        accepted_masks: list[np.ndarray] = []
        accepted_roi_ids: list[str] = []
        accepted_frame_indices: list[int] = []
        accepted_ct_uids: list[str] = []
        seg_rows: list[dict[str, str]] = []

        for frame in seg_object.frames:
            summary["seg_frames_inspected"] += 1
            row = _geometry_row(seg_object, frame, ct_index)
            reason = row["rejection_reason"]
            if reason == "empty_mask":
                summary["rejected_empty_masks"] += 1
            elif reason == "missing_ct_reference":
                summary["missing_ct_references"] += 1
            elif reason == "resolution_mismatch":
                summary["resolution_mismatches"] += 1
            else:
                summary["accepted_roi_masks"] += 1
                accepted_masks.append(frame.mask)
                accepted_roi_ids.append(row["roi_id"])
                accepted_frame_indices.append(frame.frame_index)
                accepted_ct_uids.append(row["ct_sop_instance_uid"])
            seg_rows.append(row)

        mask_path = save_roi_masks(mask_dir, seg_sop_uid or "unknown_seg", accepted_masks, accepted_roi_ids, accepted_frame_indices, accepted_ct_uids)
        if mask_path is not None:
            for row in seg_rows:
                if not row["rejection_reason"]:
                    row["mask_npz_path"] = str(mask_path)
        geometry_rows.extend(seg_rows)

    _write_geometry_csv(geometry_csv, geometry_rows)
    _write_report(report_md, summary)
    return ROIGeometryResult(output_dir=roi_dir, geometry_csv=geometry_csv, report_md=report_md, mask_dir=mask_dir, summary=summary)


def compute_mask_geometry(mask: np.ndarray) -> dict[str, object]:
    """Compute bbox, centroid, and area for a binary ROI mask."""

    binary = np.asarray(mask).astype(bool)
    area = int(binary.sum())
    if area == 0:
        return {
            "mask_area_px": 0,
            "bbox_x_min": "",
            "bbox_y_min": "",
            "bbox_x_max": "",
            "bbox_y_max": "",
            "bbox_width": "",
            "bbox_height": "",
            "centroid_x": "",
            "centroid_y": "",
            "is_empty": "true",
        }
    y_indices, x_indices = np.where(binary)
    x_min = int(x_indices.min())
    x_max = int(x_indices.max())
    y_min = int(y_indices.min())
    y_max = int(y_indices.max())
    return {
        "mask_area_px": area,
        "bbox_x_min": x_min,
        "bbox_y_min": y_min,
        "bbox_x_max": x_max,
        "bbox_y_max": y_max,
        "bbox_width": x_max - x_min + 1,
        "bbox_height": y_max - y_min + 1,
        "centroid_x": float(x_indices.mean()),
        "centroid_y": float(y_indices.mean()),
        "is_empty": "false",
    }


def _geometry_row(seg_object: SegObject, frame: SegFrame, ct_index: dict[str, dict[str, str]]) -> dict[str, str]:
    roi_id = f"{seg_object.sop_instance_uid or 'unknown_seg'}__frame_{frame.frame_index:04d}"
    geometry = compute_mask_geometry(frame.mask)
    ct_uid = frame.referenced_sop_instance_uid or ""
    ct_row = ct_index.get(ct_uid)
    rejection_reason = ""
    if geometry["is_empty"] == "true":
        rejection_reason = "empty_mask"
    elif ct_row is None:
        rejection_reason = "missing_ct_reference"
    elif _int_or_none(ct_row.get("rows")) != frame.mask.shape[0] or _int_or_none(ct_row.get("columns")) != frame.mask.shape[1]:
        rejection_reason = "resolution_mismatch"

    return {
        "roi_id": roi_id,
        "seg_sop_instance_uid": seg_object.sop_instance_uid or "",
        "ct_sop_instance_uid": ct_uid,
        "patient_id": (ct_row or {}).get("patient_id", seg_object.patient_id or ""),
        "study_instance_uid": (ct_row or {}).get("study_instance_uid", seg_object.study_instance_uid or ""),
        "ct_series_instance_uid": (ct_row or {}).get("series_instance_uid", ""),
        "seg_series_instance_uid": seg_object.series_instance_uid or "",
        "slice_index": str(frame.frame_index),
        "rows": str(frame.mask.shape[0]),
        "columns": str(frame.mask.shape[1]),
        "mask_area_px": str(geometry["mask_area_px"]),
        "bbox_x_min": str(geometry["bbox_x_min"]),
        "bbox_y_min": str(geometry["bbox_y_min"]),
        "bbox_x_max": str(geometry["bbox_x_max"]),
        "bbox_y_max": str(geometry["bbox_y_max"]),
        "bbox_width": str(geometry["bbox_width"]),
        "bbox_height": str(geometry["bbox_height"]),
        "centroid_x": str(geometry["centroid_x"]),
        "centroid_y": str(geometry["centroid_y"]),
        "is_empty": str(geometry["is_empty"]),
        "rejection_reason": rejection_reason,
        "mask_npz_path": "",
    }


def _selected_match_rows(rows: list[dict[str, str]], include_partial: bool, max_seg: int | None) -> list[dict[str, str]]:
    allowed = {"matched_strict"}
    if include_partial:
        allowed.add("matched_partial")
    selected = [row for row in rows if row.get("match_status") in allowed]
    return selected[:max_seg] if max_seg is not None else selected


def _build_ct_index(inventory_csv: Path) -> dict[str, dict[str, str]]:
    return {row.get("sop_instance_uid", ""): row for row in _read_csv(inventory_csv) if row.get("classification") == "CT" and row.get("sop_instance_uid")}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_geometry_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "roi_id",
        "seg_sop_instance_uid",
        "ct_sop_instance_uid",
        "patient_id",
        "study_instance_uid",
        "ct_series_instance_uid",
        "seg_series_instance_uid",
        "slice_index",
        "rows",
        "columns",
        "mask_area_px",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
        "bbox_width",
        "bbox_height",
        "centroid_x",
        "centroid_y",
        "is_empty",
        "rejection_reason",
        "mask_npz_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, int]) -> None:
    lines = [
        "# ROI Geometry Report",
        "",
        "DICOM SEG masks are the ROI source. Bounding boxes are derived visualization aids only.",
        "",
        f"- number of SEG objects inspected: {summary['seg_objects_inspected']}",
        f"- number of SEG frames inspected: {summary['seg_frames_inspected']}",
        f"- number of accepted ROI masks: {summary['accepted_roi_masks']}",
        f"- number of rejected empty masks: {summary['rejected_empty_masks']}",
        f"- number of missing CT references: {summary['missing_ct_references']}",
        f"- number of resolution mismatches: {summary['resolution_mismatches']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required ROI extraction input not found: {path}")
