"""DICOM SEG pixel and frame-reference loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset


@dataclass(frozen=True)
class SegFrame:
    """One binary frame extracted from a DICOM SEG object."""

    frame_index: int
    mask: np.ndarray
    referenced_sop_instance_uid: str | None


@dataclass(frozen=True)
class SegObject:
    """Loaded DICOM SEG object with binary frame masks."""

    file_path: Path
    sop_instance_uid: str | None
    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str | None
    frames: list[SegFrame]


def load_seg_object(file_path: Path) -> SegObject:
    """Read SEG pixel data and return one binary mask per frame."""

    dataset = pydicom.dcmread(file_path)
    pixel_array = _as_frame_array(dataset.pixel_array)
    frame_groups = list(getattr(dataset, "PerFrameFunctionalGroupsSequence", []) or [])

    frames: list[SegFrame] = []
    for frame_index, frame_pixels in enumerate(pixel_array):
        frame_group = frame_groups[frame_index] if frame_index < len(frame_groups) else None
        frames.append(
            SegFrame(
                frame_index=frame_index,
                mask=np.asarray(frame_pixels > 0, dtype=np.bool_),
                referenced_sop_instance_uid=_referenced_sop_for_frame(frame_group),
            )
        )

    return SegObject(
        file_path=file_path,
        sop_instance_uid=_string_or_none(getattr(dataset, "SOPInstanceUID", None)),
        patient_id=_string_or_none(getattr(dataset, "PatientID", None)),
        study_instance_uid=_string_or_none(getattr(dataset, "StudyInstanceUID", None)),
        series_instance_uid=_string_or_none(getattr(dataset, "SeriesInstanceUID", None)),
        frames=frames,
    )


def _as_frame_array(pixel_array: np.ndarray) -> np.ndarray:
    array = np.asarray(pixel_array)
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"Unsupported SEG pixel array shape: {array.shape}")


def _referenced_sop_for_frame(frame_group: Dataset | None) -> str | None:
    if frame_group is None:
        return None
    for keyword in ("DerivationImageSequence", "ReferencedImageSequence"):
        uid = _find_referenced_sop(getattr(frame_group, keyword, None))
        if uid:
            return uid
    return _find_referenced_sop(frame_group)


def _find_referenced_sop(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Dataset):
        direct = _string_or_none(getattr(value, "ReferencedSOPInstanceUID", None))
        if direct:
            return direct
        for element in value:
            nested = _find_referenced_sop(element.value)
            if nested:
                return nested
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "Sequence":
        for item in value:
            nested = _find_referenced_sop(item)
            if nested:
                return nested
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
