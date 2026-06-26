"""Safe DICOM metadata extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydicom
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.filereader import read_partial


REQUIRED_METADATA_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "InstanceNumber",
    "ImagePositionPatient",
    "Rows",
    "Columns",
    "ReferencedSeriesSequence",
    "ReferencedInstanceSequence",
    "ReferencedSOPInstanceUID",
]
REQUIRED_METADATA_TAG_VALUES = [tag for tag in (tag_for_keyword(keyword) for keyword in REQUIRED_METADATA_TAGS) if tag]


@dataclass(frozen=True)
class DicomMetadata:
    """Metadata captured without reading pixel data."""

    file_path: Path
    classification: str
    patient_id: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    modality: str | None = None
    instance_number: int | None = None
    image_position_patient: tuple[float, ...] | None = None
    rows: int | None = None
    columns: int | None = None
    referenced_sop_instance_uids: list[str] = field(default_factory=list)
    error: str | None = None


def read_dicom_metadata(file_path: Path) -> DicomMetadata:
    """Read DICOM metadata safely, never loading pixel data."""

    try:
        with file_path.open("rb") as handle:
            dataset = read_partial(
                handle,
                stop_when=_after_required_metadata,
                specific_tags=REQUIRED_METADATA_TAG_VALUES,
            )
    except Exception as exc:  # pydicom raises several exception types for invalid files.
        return DicomMetadata(file_path=file_path, classification="INVALID", error=str(exc))

    modality = _string_or_none(getattr(dataset, "Modality", None))
    classification = _classify_modality(modality)
    return DicomMetadata(
        file_path=file_path,
        classification=classification,
        patient_id=_string_or_none(getattr(dataset, "PatientID", None)),
        study_instance_uid=_string_or_none(getattr(dataset, "StudyInstanceUID", None)),
        series_instance_uid=_string_or_none(getattr(dataset, "SeriesInstanceUID", None)),
        sop_instance_uid=_string_or_none(getattr(dataset, "SOPInstanceUID", None)),
        modality=modality,
        instance_number=_int_or_none(getattr(dataset, "InstanceNumber", None)),
        image_position_patient=_float_tuple_or_none(getattr(dataset, "ImagePositionPatient", None)),
        rows=_int_or_none(getattr(dataset, "Rows", None)),
        columns=_int_or_none(getattr(dataset, "Columns", None)),
        referenced_sop_instance_uids=_referenced_sop_instance_uids(dataset),
    )


def _classify_modality(modality: str | None) -> str:
    if modality == "CT":
        return "CT"
    if modality == "SEG":
        return "SEG"
    return "OTHER"


def _after_required_metadata(tag: pydicom.tag.BaseTag, vr: str | None, length: int) -> bool:
    del vr, length
    return tag.group > 0x0028


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_tuple_or_none(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _referenced_sop_instance_uids(dataset: Dataset) -> list[str]:
    uids: set[str] = set()
    for element in dataset.iterall():
        if element.keyword == "ReferencedSOPInstanceUID":
            value = _string_or_none(element.value)
            if value:
                uids.add(value)
    return sorted(uids)


def format_sequence(value: tuple[float, ...] | list[str] | None) -> str:
    """Serialize short sequence metadata for reproducible CSV output."""

    if value is None:
        return ""
    return "|".join(str(item) for item in value)
