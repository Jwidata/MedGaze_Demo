"""Coordinate mapping helpers for image-coordinate visualization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ImageSpace:
    width: int
    height: int


def image_space_from_roi(roi: pd.Series | dict[str, object]) -> ImageSpace:
    width = int(float(_get(roi, "columns", 512) or 512))
    height = int(float(_get(roi, "rows", 512) or 512))
    return ImageSpace(width=max(1, width), height=max(1, height))


def clip_point(x: float, y: float, image_space: ImageSpace) -> tuple[float, float]:
    return (
        min(max(float(x), 0.0), image_space.width - 1),
        min(max(float(y), 0.0), image_space.height - 1),
    )


def clip_samples_to_image(samples: pd.DataFrame, image_space: ImageSpace) -> pd.DataFrame:
    rows = samples.copy()
    rows["image_x"] = rows["image_x"].astype(float).clip(0, image_space.width - 1)
    rows["image_y"] = rows["image_y"].astype(float).clip(0, image_space.height - 1)
    return rows


def _get(row: pd.Series | dict[str, object], key: str, default: object) -> object:
    return row.get(key, default) if isinstance(row, dict) else row.get(key, default)
