"""Lazy CT pixel loading for the review workstation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


def load_dicom_inventory(path: Path) -> pd.DataFrame:
    columns = ["classification", "patient_id", "study_instance_uid", "series_instance_uid", "sop_instance_uid", "file_path", "instance_number", "image_position_patient"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    available = pd.read_csv(path, nrows=0).columns.tolist()
    return pd.read_csv(path, usecols=[column for column in columns if column in available])


def build_ct_path_lookup(inventory: pd.DataFrame) -> dict[str, Path]:
    if inventory.empty:
        return {}
    rows = inventory[inventory["classification"].astype(str) == "CT"]
    return {str(uid): Path(str(path)) for uid, path in zip(rows["sop_instance_uid"].astype(str), rows["file_path"].astype(str)) if uid}


def load_ct_image_for_sop(sop_instance_uid: str, ct_path_lookup: dict[str, Path]) -> Image.Image | None:
    path = ct_path_lookup.get(str(sop_instance_uid))
    if path is None or not path.exists():
        return None
    try:
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        pixels = pixels * slope + intercept
        return _window_ct(pixels, center=float(getattr(dataset, "WindowCenter", 40) or 40), width=float(getattr(dataset, "WindowWidth", 400) or 400))
    except Exception:
        return None


def build_case_slice_images(case_id: str, roi_geometry: pd.DataFrame, ct_path_lookup: dict[str, Path], max_slices: int = 80) -> dict[int, Image.Image]:
    if roi_geometry.empty or not ct_path_lookup:
        return {}
    rows = roi_geometry.copy()
    if "patient_id" in rows.columns:
        rows = rows[rows["patient_id"].astype(str) == str(case_id)]
    if rows.empty or "ct_sop_instance_uid" not in rows.columns:
        return {}
    images: dict[int, Image.Image] = {}
    for _, row in rows.drop_duplicates("slice_index").sort_values("slice_index").head(max_slices).iterrows():
        image = load_ct_image_for_sop(str(row["ct_sop_instance_uid"]), ct_path_lookup)
        if image is not None:
            images[int(float(row["slice_index"]))] = image.convert("RGBA")
    return images


def _window_ct(pixels: np.ndarray, center: float, width: float) -> Image.Image:
    if width <= 1:
        width = 400.0
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(pixels, low, high)
    normalized = ((clipped - low) / max(1e-6, high - low) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGBA")
