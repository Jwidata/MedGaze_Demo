"""CT series grouping and slice ordering helpers."""

from __future__ import annotations

from dataclasses import dataclass


from app.dicom.dicom_metadata import DicomMetadata


@dataclass(frozen=True)
class CTSeriesSummary:
    """Summary of one CT series discovered during metadata scanning."""

    study_instance_uid: str
    series_instance_uid: str
    patient_id: str | None
    slice_count: int
    first_file_path: str
    last_file_path: str
    sort_method: str


def group_ct_series(ct_files: list[DicomMetadata]) -> dict[tuple[str, str], list[DicomMetadata]]:
    """Group CT files by StudyInstanceUID and SeriesInstanceUID."""

    grouped: dict[tuple[str, str], list[DicomMetadata]] = {}
    for item in ct_files:
        if not item.study_instance_uid or not item.series_instance_uid:
            continue
        key = (item.study_instance_uid, item.series_instance_uid)
        grouped.setdefault(key, []).append(item)
    return grouped


def sort_ct_slices(slices: list[DicomMetadata]) -> tuple[list[DicomMetadata], str]:
    """Sort CT slices by ImagePositionPatient z value, then InstanceNumber."""

    if all(item.image_position_patient and len(item.image_position_patient) >= 3 for item in slices):
        return sorted(slices, key=lambda item: (item.image_position_patient[2], str(item.file_path))), "image_position_patient_z"
    return sorted(slices, key=lambda item: (item.instance_number if item.instance_number is not None else 10**9, str(item.file_path))), "instance_number"


def summarize_ct_series(ct_files: list[DicomMetadata]) -> list[CTSeriesSummary]:
    """Build sorted CT series summaries."""

    summaries: list[CTSeriesSummary] = []
    for (study_uid, series_uid), slices in sorted(group_ct_series(ct_files).items()):
        sorted_slices, sort_method = sort_ct_slices(slices)
        summaries.append(
            CTSeriesSummary(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                patient_id=sorted_slices[0].patient_id,
                slice_count=len(sorted_slices),
                first_file_path=str(sorted_slices[0].file_path),
                last_file_path=str(sorted_slices[-1].file_path),
                sort_method=sort_method,
            )
        )
    return summaries
