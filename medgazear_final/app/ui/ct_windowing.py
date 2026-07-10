"""CT pixel windowing helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


WINDOW_PRESETS = {
    "Lung": (-600.0, 1500.0),
    "Mediastinum": (40.0, 400.0),
    "Bone": (300.0, 1500.0),
    "Soft tissue": (50.0, 350.0),
}


def window_ct_pixels(pixels: np.ndarray, center: float = -600.0, width: float = 1500.0) -> Image.Image:
    data = pixels.astype(np.float32)
    if width <= 1:
        width = float(np.nanpercentile(data, 99) - np.nanpercentile(data, 1)) or 400.0
        center = float((np.nanpercentile(data, 99) + np.nanpercentile(data, 1)) / 2)
    low = center - width / 2
    high = center + width / 2
    normalized = np.clip((data - low) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L").convert("RGBA")


def dicom_window_value(value: object, default: float) -> float:
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except Exception:
        return default


def window_preset(name: str) -> tuple[float, float]:
    return WINDOW_PRESETS.get(name, WINDOW_PRESETS["Lung"])
