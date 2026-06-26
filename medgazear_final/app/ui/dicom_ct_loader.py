"""Real CT DICOM series loading for the review workstation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image

from app.ui.ct_windowing import dicom_window_value, window_ct_pixels


@dataclass
class CTSlice:
    slice_index: int
    sop_instance_uid: str
    file_path: Path
    instance_number: int | None = None
    image_position_z: float | None = None
    image: Image.Image | None = None


@dataclass
class CTSeries:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    slices: list[CTSlice]

    @property
    def total_slices(self) -> int:
        return len(self.slices)

    def image_for_index(self, index: int) -> Image.Image | None:
        if 0 <= index < len(self.slices):
            slice_ = self.slices[index]
            if slice_.image is None:
                slice_.image = _load_ct_image(slice_.file_path)
            return slice_.image
        return None


def load_ct_series_for_case(patient_id: str, series_uid: str, inventory: pd.DataFrame) -> CTSeries:
    rows = inventory[
        (inventory["classification"].astype(str) == "CT")
        & (inventory["patient_id"].astype(str) == str(patient_id))
        & (inventory["series_instance_uid"].astype(str) == str(series_uid))
    ].copy()
    if rows.empty:
        raise FileNotFoundError(f"No CT DICOM slices found for patient={patient_id}, series={series_uid}")
    rows["_instance"] = pd.to_numeric(rows.get("instance_number"), errors="coerce")
    rows["_ipp_z"] = rows.get("image_position_patient", pd.Series(index=rows.index, dtype=object)).map(_image_position_z)
    if rows["_ipp_z"].notna().all():
        rows = rows.sort_values(["_ipp_z", "_instance", "sop_instance_uid"], na_position="last")
    else:
        rows = rows.sort_values(["_instance", "sop_instance_uid"], na_position="last")
    slices: list[CTSlice] = []
    for slice_index, (_, row) in enumerate(rows.iterrows()):
        path = Path(str(row["file_path"]))
        instance_number = None if pd.isna(row.get("_instance")) else int(row.get("_instance"))
        image_position_z = None if pd.isna(row.get("_ipp_z")) else float(row.get("_ipp_z"))
        slices.append(CTSlice(slice_index=slice_index, sop_instance_uid=str(row["sop_instance_uid"]), file_path=path, instance_number=instance_number, image_position_z=image_position_z))
    first = rows.iloc[0]
    return CTSeries(str(patient_id), str(first.get("study_instance_uid", "")), str(series_uid), slices)


def _image_position_z(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        parts = str(value).replace("\\", "|").split("|")
        if len(parts) < 3:
            return None
        return float(parts[2])
    except (TypeError, ValueError):
        return None


def _load_ct_image(path: Path) -> Image.Image:
    try:
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array.astype("float32")
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0) or 1.0) + float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        center = dicom_window_value(getattr(dataset, "WindowCenter", None), -600.0)
        width = dicom_window_value(getattr(dataset, "WindowWidth", None), 1500.0)
        return window_ct_pixels(pixels, center=center, width=width)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CT pixels from {path}: {exc}") from exc
