from __future__ import annotations

import csv
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.dicom.dicom_scanner import scan_dicom_dataset


def test_scan_dicom_dataset_writes_inventory_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    data_root.mkdir()

    study_uid = generate_uid()
    ct_series_uid = generate_uid()
    ct_sop_1 = generate_uid()
    ct_sop_2 = generate_uid()
    seg_sop = generate_uid()

    _write_dicom(
        data_root / "ct_z10.dcm",
        modality="CT",
        patient_id="PATIENT_A",
        study_uid=study_uid,
        series_uid=ct_series_uid,
        sop_uid=ct_sop_1,
        instance_number=2,
        image_position_patient=[0, 0, 10],
    )
    _write_dicom(
        data_root / "nested" / "ct_z0.dcm",
        modality="CT",
        patient_id="PATIENT_A",
        study_uid=study_uid,
        series_uid=ct_series_uid,
        sop_uid=ct_sop_2,
        instance_number=1,
        image_position_patient=[0, 0, 0],
    )
    _write_dicom(
        data_root / "seg.dcm",
        modality="SEG",
        patient_id="PATIENT_A",
        study_uid=study_uid,
        series_uid=generate_uid(),
        sop_uid=seg_sop,
        referenced_sop_uids=[ct_sop_1, ct_sop_2],
    )
    _write_dicom(
        data_root / "sr.dcm",
        modality="SR",
        patient_id="PATIENT_A",
        study_uid=study_uid,
        series_uid=generate_uid(),
        sop_uid=generate_uid(),
    )
    (data_root / "not_dicom.dcm").write_text("not a dicom file", encoding="utf-8")

    result = scan_dicom_dataset(data_root=data_root, output_root=output_root)

    assert result.inventory_csv.exists()
    assert result.ct_series_summary_csv.exists()
    assert result.seg_inventory_csv.exists()
    assert result.summary_md.exists()
    assert result.summary == {
        "files_scanned": 5,
        "readable_dicom_files": 4,
        "invalid_files": 1,
        "ct_slices": 2,
        "ct_series": 1,
        "seg_objects": 1,
        "other_dicom_files": 1,
    }

    inventory_rows = _read_csv(result.inventory_csv)
    assert [row["classification"] for row in inventory_rows].count("CT") == 2
    assert [row["classification"] for row in inventory_rows].count("SEG") == 1
    assert [row["classification"] for row in inventory_rows].count("OTHER") == 1
    assert [row["classification"] for row in inventory_rows].count("INVALID") == 1

    summary_rows = _read_csv(result.ct_series_summary_csv)
    assert len(summary_rows) == 1
    assert summary_rows[0]["slice_count"] == "2"
    assert summary_rows[0]["sort_method"] == "image_position_patient_z"
    assert summary_rows[0]["first_file_path"].endswith("ct_z0.dcm")
    assert summary_rows[0]["last_file_path"].endswith("ct_z10.dcm")

    seg_rows = _read_csv(result.seg_inventory_csv)
    assert len(seg_rows) == 1
    assert ct_sop_1 in seg_rows[0]["referenced_sop_instance_uids"]
    assert ct_sop_2 in seg_rows[0]["referenced_sop_instance_uids"]


def test_ct_sort_falls_back_to_instance_number(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    data_root.mkdir()

    study_uid = generate_uid()
    series_uid = generate_uid()
    _write_dicom(
        data_root / "ct_instance_2.dcm",
        modality="CT",
        patient_id="PATIENT_B",
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=generate_uid(),
        instance_number=2,
    )
    _write_dicom(
        data_root / "ct_instance_1.dcm",
        modality="CT",
        patient_id="PATIENT_B",
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=generate_uid(),
        instance_number=1,
    )

    result = scan_dicom_dataset(data_root=data_root, output_root=output_root)
    summary_rows = _read_csv(result.ct_series_summary_csv)

    assert summary_rows[0]["sort_method"] == "instance_number"
    assert summary_rows[0]["first_file_path"].endswith("ct_instance_1.dcm")
    assert summary_rows[0]["last_file_path"].endswith("ct_instance_2.dcm")


def _write_dicom(
    path: Path,
    *,
    modality: str,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    instance_number: int = 1,
    image_position_patient: list[float] | None = None,
    referenced_sop_uids: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPInstanceUID = sop_uid
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.Modality = modality
    dataset.InstanceNumber = instance_number
    dataset.Rows = 16
    dataset.Columns = 32
    if image_position_patient is not None:
        dataset.ImagePositionPatient = image_position_patient
    if referenced_sop_uids:
        dataset.ReferencedSeriesSequence = [Dataset()]
        dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence = []
        for referenced_uid in referenced_sop_uids:
            reference = Dataset()
            reference.ReferencedSOPInstanceUID = referenced_uid
            dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence.append(reference)

    pydicom.dcmwrite(path, dataset, write_like_original=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
