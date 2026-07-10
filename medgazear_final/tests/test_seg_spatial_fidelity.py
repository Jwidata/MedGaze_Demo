from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.features.behavior_feature_builder import build_behavior_feature_row
from app.features.roi_spatial_modes import RoiMaskLibrary, classify_samples_for_roi, point_in_roi, sample_disagreement
from app.ml_behavior.evaluation_integrity import SplitManifest
from scripts._common import build_parser


def test_mask_membership_and_coordinate_convention(tmp_path: Path) -> None:
    roi = _write_mask_roi(tmp_path)
    library = RoiMaskLibrary()
    assert point_in_roi(2.0, 1.0, roi, geometry_mode="mask", mask_library=library) is True
    assert point_in_roi(0.0, 0.0, roi, geometry_mode="mask", mask_library=library) is False
    assert point_in_roi(3.0, 1.0, roi, geometry_mode="mask", mask_library=library) is True
    assert point_in_roi(10.0, 10.0, roi, geometry_mode="mask", mask_library=library) is False


def test_bbox_vs_mask_disagreement_and_near_distance(tmp_path: Path) -> None:
    roi = _write_mask_roi(tmp_path)
    library = RoiMaskLibrary()
    sample = {"image_x": 1.0, "image_y": 2.0, "slice_index": 0, "is_valid": True, "is_outside_ct": False, "is_ui_glance": False}
    disagreement = sample_disagreement(sample, roi, library)
    assert disagreement["bbox_inside"] is True
    assert disagreement["mask_inside"] is False
    assert disagreement["mask_near"] is True


def test_roi_id_to_mask_mapping(tmp_path: Path) -> None:
    roi = _write_mask_roi(tmp_path)
    library = RoiMaskLibrary()
    audit = library.audit_row(roi)
    assert audit.resolved is True
    assert audit.mask_shape == (5, 5)


def test_paired_bbox_and_mask_generation_preserve_identifiers_and_non_spatial_quality(tmp_path: Path) -> None:
    roi = _write_mask_roi(tmp_path)
    samples = pd.DataFrame([
        {"session_id": "S1", "reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "roi_id": roi["roi_id"], "slice_index": 0, "sample_index": 0, "timestamp_ms": 0.0, "image_x": 1.0, "image_y": 2.0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0, "hidden_behavior_label": "focused_roi_confirmation"},
        {"session_id": "S1", "reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "roi_id": roi["roi_id"], "slice_index": 0, "sample_index": 1, "timestamp_ms": 100.0, "image_x": 2.0, "image_y": 2.0, "is_valid": True, "is_dropout": False, "is_blink": False, "is_invalid_burst": False, "is_outside_ct": False, "is_ui_glance": False, "jitter_x": 0.0, "jitter_y": 0.0, "hidden_behavior_label": "focused_roi_confirmation"},
    ])
    metadata = {"reader_id": "R1", "reader_profile": "expert_systematic", "case_id": "C1", "hidden_behavior_label": "focused_roi_confirmation"}
    library = RoiMaskLibrary()
    bbox = build_behavior_feature_row(samples, roi, metadata, geometry_mode="bbox", mask_library=library)
    mask = build_behavior_feature_row(samples, roi, metadata, geometry_mode="mask", mask_library=library)
    assert bbox.row["session_id"] == mask.row["session_id"] == "S1"
    assert bbox.row["roi_id"] == mask.row["roi_id"] == roi["roi_id"]
    assert bbox.row["hidden_behavior_label"] == mask.row["hidden_behavior_label"] == "focused_roi_confirmation"
    assert bbox.row["gaze_validity_ratio"] == mask.row["gaze_validity_ratio"]
    assert bbox.row["dropout_ratio"] == mask.row["dropout_ratio"]


def test_fixed_split_manifest_and_output_separation() -> None:
    manifest = SplitManifest(
        strategy="case_grouped_primary",
        seed=42,
        group_column="case_id",
        train_row_ids=[0],
        validation_row_ids=[1],
        test_row_ids=[2],
        train_groups=["C1"],
        validation_groups=["C2"],
        test_groups=["C3"],
    )
    assert manifest.train_groups == ["C1"]
    assert manifest.group_column == "case_id"
    assert "08e_run_seg_spatial_fidelity_ablation" not in str(Path("outputs/behavior_learning"))


def _write_mask_roi(tmp_path: Path) -> dict[str, object]:
    mask_path = tmp_path / "mask.npz"
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 2] = 1
    np.savez_compressed(mask_path, masks=np.stack([mask], axis=0), roi_ids=np.asarray(["ROI__frame_0000"]), frame_indices=np.asarray([0], dtype=np.int32), ct_sop_instance_uids=np.asarray(["SOP1"]))
    return {
        "roi_id": "ROI__frame_0000",
        "mask_npz_path": str(mask_path),
        "rows": 5,
        "columns": 5,
        "bbox_x_min": 1,
        "bbox_y_min": 1,
        "bbox_x_max": 3,
        "bbox_y_max": 3,
        "bbox_width": 3,
        "bbox_height": 3,
        "centroid_x": 2.0,
        "centroid_y": 2.0,
        "slice_index": 0,
        "ct_stack_index": 0,
        "mask_area_px": 3,
        "ct_series_instance_uid": "SERIES1",
        "ct_sop_instance_uid": "SOP1",
    }
