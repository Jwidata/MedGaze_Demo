"""ROI and placeholder CT canvas rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app.visualization.coordinate_mapper import ImageSpace, image_space_from_roi


def create_placeholder_canvas(image_space: ImageSpace) -> Image.Image:
    image = Image.new("RGBA", (image_space.width, image_space.height), (42, 42, 42, 255))
    draw = ImageDraw.Draw(image)
    step = max(32, min(image_space.width, image_space.height) // 8)
    for x in range(0, image_space.width, step):
        draw.line((x, 0, x, image_space.height), fill=(56, 56, 56, 255))
    for y in range(0, image_space.height, step):
        draw.line((0, y, image_space.width, y), fill=(56, 56, 56, 255))
    draw.text((12, 12), "CT placeholder canvas", fill=(170, 170, 170, 255), font=ImageFont.load_default())
    return image


def render_roi_layer(roi: pd.Series | dict[str, object], image_space: ImageSpace | None = None) -> Image.Image:
    space = image_space or image_space_from_roi(roi)
    layer = Image.new("RGBA", (space.width, space.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x0 = float(_get(roi, "bbox_x_min", 0))
    y0 = float(_get(roi, "bbox_y_min", 0))
    x1 = float(_get(roi, "bbox_x_max", 0))
    y1 = float(_get(roi, "bbox_y_max", 0))
    cx = float(_get(roi, "centroid_x", (x0 + x1) / 2))
    cy = float(_get(roi, "centroid_y", (y0 + y1) / 2))
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 80, 255), width=3, fill=(0, 255, 80, 40))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 0, 255))
    return layer


def save_roi_overlay(path: Path, roi: pd.Series | dict[str, object]) -> Path:
    space = image_space_from_roi(roi)
    image = Image.alpha_composite(create_placeholder_canvas(space), render_roi_layer(roi, space))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path


def _get(row: pd.Series | dict[str, object], key: str, default: object) -> object:
    return row.get(key, default) if isinstance(row, dict) else row.get(key, default)
