from __future__ import annotations

import csv
from pathlib import Path

from app.roi.roi_matcher import match_ct_seg_from_audit


def test_roi_matcher_classifies_required_cases(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    audit_dir = output_root / "dicom_audit"
    audit_dir.mkdir(parents=True)

    _write_csv(
        audit_dir / "dicom_inventory.csv",
        [
            {
                "classification": "CT",
                "file_path": "ct_a_1.dcm",
                "patient_id": "P1",
                "study_instance_uid": "STUDY_A",
                "series_instance_uid": "SERIES_A",
                "sop_instance_uid": "CT_A_1",
                "modality": "CT",
                "instance_number": "1",
                "image_position_patient": "0|0|0",
                "rows": "16",
                "columns": "16",
                "referenced_sop_instance_uids": "",
                "error": "",
            },
            {
                "classification": "CT",
                "file_path": "ct_a_2.dcm",
                "patient_id": "P1",
                "study_instance_uid": "STUDY_A",
                "series_instance_uid": "SERIES_A",
                "sop_instance_uid": "CT_A_2",
                "modality": "CT",
                "instance_number": "2",
                "image_position_patient": "0|0|1",
                "rows": "16",
                "columns": "16",
                "referenced_sop_instance_uids": "",
                "error": "",
            },
            {
                "classification": "CT",
                "file_path": "ct_b_1.dcm",
                "patient_id": "P2",
                "study_instance_uid": "STUDY_B",
                "series_instance_uid": "SERIES_B",
                "sop_instance_uid": "CT_B_1",
                "modality": "CT",
                "instance_number": "1",
                "image_position_patient": "0|0|0",
                "rows": "16",
                "columns": "16",
                "referenced_sop_instance_uids": "",
                "error": "",
            },
        ],
    )
    _write_csv(
        audit_dir / "ct_series_summary.csv",
        [
            {
                "study_instance_uid": "STUDY_A",
                "series_instance_uid": "SERIES_A",
                "patient_id": "P1",
                "slice_count": "2",
                "first_file_path": "ct_a_1.dcm",
                "last_file_path": "ct_a_2.dcm",
                "sort_method": "image_position_patient_z",
            },
        ],
    )
    _write_csv(
        audit_dir / "seg_inventory.csv",
        [
            _seg_row("SEG_STRICT", "CT_A_1|CT_A_2"),
            _seg_row("SEG_PARTIAL", "CT_A_1|MISSING_CT"),
            _seg_row("SEG_UNMATCHED", "MISSING_A|MISSING_B"),
            _seg_row("SEG_NO_REFERENCE", ""),
            _seg_row("SEG_MULTI_SERIES", "CT_A_1|CT_B_1"),
        ],
    )

    result = match_ct_seg_from_audit(output_root)

    assert result.match_table_csv.exists()
    assert result.summary_md.exists()
    assert result.summary == {
        "seg_objects": 5,
        "strict_matches": 2,
        "partial_matches": 1,
        "unmatched_missing_ct": 1,
        "invalid_no_references": 1,
        "multiple_ct_series_reference_count": 1,
    }

    rows = {row["seg_sop_instance_uid"]: row for row in _read_csv(result.match_table_csv)}
    assert rows["SEG_STRICT"]["match_status"] == "matched_strict"
    assert rows["SEG_STRICT"]["matched_ct_series_instance_uids"] == "SERIES_A"
    assert rows["SEG_PARTIAL"]["match_status"] == "matched_partial"
    assert rows["SEG_PARTIAL"]["missing_sop_count"] == "1"
    assert rows["SEG_UNMATCHED"]["match_status"] == "unmatched_missing_ct"
    assert rows["SEG_NO_REFERENCE"]["match_status"] == "invalid_no_references"
    assert rows["SEG_MULTI_SERIES"]["match_status"] == "matched_strict"
    assert rows["SEG_MULTI_SERIES"]["multiple_ct_series_reference"] == "true"
    assert rows["SEG_MULTI_SERIES"]["matched_ct_series_instance_uids"] == "SERIES_A|SERIES_B"


def _seg_row(sop_uid: str, references: str) -> dict[str, str]:
    return {
        "file_path": f"{sop_uid}.dcm",
        "patient_id": "P_SEG",
        "study_instance_uid": "SEG_STUDY",
        "series_instance_uid": "SEG_SERIES",
        "sop_instance_uid": sop_uid,
        "modality": "SEG",
        "referenced_sop_instance_uids": references,
        "rows": "16",
        "columns": "16",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
