from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.roi.roi_geometry import compute_mask_geometry, extract_roi_geometry_from_matches
from app.roi.roi_mask_store import load_roi_masks


def test_binary_mask_geometry_bbox_centroid_and_npz_roundtrip(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    audit_dir = output_root / "dicom_audit"
    roi_dir = output_root / "roi_geometry"
    audit_dir.mkdir(parents=True)
    roi_dir.mkdir(parents=True)

    ct_sop_uid = "CT_ACCEPTED"
    seg_sop_uid = "SEG_GEOMETRY"
    seg_path = tmp_path / "seg_geometry.dcm"
    masks = np.zeros((2, 5, 6), dtype=np.uint8)
    masks[0, 1:4, 2:5] = 1
    _write_seg_dicom(seg_path, seg_sop_uid=seg_sop_uid, referenced_sop_uids=[ct_sop_uid, ct_sop_uid], masks=masks)
    _write_audit_inputs(output_root, seg_path, seg_sop_uid, ct_sop_uid)

    result = extract_roi_geometry_from_matches(output_root)

    assert result.geometry_csv.exists()
    assert result.report_md.exists()
    assert result.summary == {
        "seg_objects_inspected": 1,
        "seg_frames_inspected": 2,
        "accepted_roi_masks": 1,
        "rejected_empty_masks": 1,
        "missing_ct_references": 0,
        "resolution_mismatches": 0,
    }

    rows = _read_csv(result.geometry_csv)
    accepted = rows[0]
    rejected = rows[1]
    assert accepted["rejection_reason"] == ""
    assert accepted["mask_area_px"] == "9"
    assert accepted["bbox_x_min"] == "2"
    assert accepted["bbox_y_min"] == "1"
    assert accepted["bbox_x_max"] == "4"
    assert accepted["bbox_y_max"] == "3"
    assert accepted["bbox_width"] == "3"
    assert accepted["bbox_height"] == "3"
    assert accepted["centroid_x"] == "3.0"
    assert accepted["centroid_y"] == "2.0"
    assert accepted["is_empty"] == "false"
    assert rejected["rejection_reason"] == "empty_mask"
    assert rejected["is_empty"] == "true"

    mask_path = Path(accepted["mask_npz_path"])
    assert mask_path.exists()
    loaded = load_roi_masks(mask_path)
    assert loaded["masks"].shape == (1, 5, 6)
    assert int(loaded["masks"].sum()) == 9
    assert loaded["roi_ids"][0] == accepted["roi_id"]
    assert loaded["ct_sop_instance_uids"][0] == ct_sop_uid


def test_compute_mask_geometry_directly() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True
    mask[1, 3] = True
    mask[3, 1] = True
    geometry = compute_mask_geometry(mask)

    assert geometry["mask_area_px"] == 3
    assert geometry["bbox_x_min"] == 1
    assert geometry["bbox_y_min"] == 1
    assert geometry["bbox_x_max"] == 3
    assert geometry["bbox_y_max"] == 3
    assert geometry["bbox_width"] == 3
    assert geometry["bbox_height"] == 3
    assert geometry["centroid_x"] == 5 / 3
    assert geometry["centroid_y"] == 5 / 3
    assert geometry["is_empty"] == "false"


def _write_audit_inputs(output_root: Path, seg_path: Path, seg_sop_uid: str, ct_sop_uid: str) -> None:
    audit_dir = output_root / "dicom_audit"
    roi_dir = output_root / "roi_geometry"
    _write_csv(
        audit_dir / "dicom_inventory.csv",
        [
            {
                "classification": "CT",
                "file_path": "ct.dcm",
                "patient_id": "PATIENT_1",
                "study_instance_uid": "STUDY_1",
                "series_instance_uid": "CT_SERIES_1",
                "sop_instance_uid": ct_sop_uid,
                "modality": "CT",
                "instance_number": "1",
                "image_position_patient": "0|0|0",
                "rows": "5",
                "columns": "6",
                "referenced_sop_instance_uids": "",
                "error": "",
            }
        ],
    )
    _write_csv(
        audit_dir / "seg_inventory.csv",
        [
            {
                "file_path": str(seg_path),
                "patient_id": "PATIENT_1",
                "study_instance_uid": "STUDY_1",
                "series_instance_uid": "SEG_SERIES_1",
                "sop_instance_uid": seg_sop_uid,
                "modality": "SEG",
                "referenced_sop_instance_uids": ct_sop_uid,
                "rows": "5",
                "columns": "6",
            }
        ],
    )
    _write_csv(
        roi_dir / "ct_seg_match_table.csv",
        [
            {
                "seg_file_path": str(seg_path),
                "seg_sop_instance_uid": seg_sop_uid,
                "seg_patient_id": "PATIENT_1",
                "seg_study_instance_uid": "STUDY_1",
                "seg_series_instance_uid": "SEG_SERIES_1",
                "referenced_sop_instance_uids": ct_sop_uid,
                "referenced_sop_count": "1",
                "matched_sop_count": "1",
                "missing_sop_count": "0",
                "match_status": "matched_strict",
                "matched_ct_study_instance_uids": "STUDY_1",
                "matched_ct_series_instance_uids": "CT_SERIES_1",
                "multiple_ct_series_reference": "false",
            }
        ],
    )


def _write_seg_dicom(path: Path, seg_sop_uid: str, referenced_sop_uids: list[str], masks: np.ndarray) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = seg_sop_uid
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = "PATIENT_1"
    dataset.StudyInstanceUID = "STUDY_1"
    dataset.SeriesInstanceUID = "SEG_SERIES_1"
    dataset.SOPInstanceUID = seg_sop_uid
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.Modality = "SEG"
    dataset.Rows = masks.shape[1]
    dataset.Columns = masks.shape[2]
    dataset.NumberOfFrames = masks.shape[0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = masks.tobytes()
    dataset.PerFrameFunctionalGroupsSequence = Sequence()
    for referenced_sop_uid in referenced_sop_uids:
        frame_group = Dataset()
        derivation = Dataset()
        source = Dataset()
        source.ReferencedSOPInstanceUID = referenced_sop_uid
        derivation.SourceImageSequence = Sequence([source])
        frame_group.DerivationImageSequence = Sequence([derivation])
        dataset.PerFrameFunctionalGroupsSequence.append(frame_group)
    pydicom.dcmwrite(path, dataset, write_like_original=False)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
