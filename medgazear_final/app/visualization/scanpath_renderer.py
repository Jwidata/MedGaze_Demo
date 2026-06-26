"""Gaze point and scanpath rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from app.visualization.coordinate_mapper import ImageSpace, clip_samples_to_image
from app.visualization.roi_overlay_renderer import create_placeholder_canvas, render_roi_layer


def render_gaze_points_layer(samples: pd.DataFrame, image_space: ImageSpace) -> Image.Image:
    layer = Image.new("RGBA", (image_space.width, image_space.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    valid = _valid_samples(samples, image_space)
    if valid.empty:
        return layer
    stride = max(1, len(valid) // 400)
    for _, row in valid.iloc[::stride].iterrows():
        x = float(row["image_x"])
        y = float(row["image_y"])
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(0, 180, 255, 135))
    return layer


def render_scanpath_layer(samples: pd.DataFrame, image_space: ImageSpace) -> Image.Image:
    layer = Image.new("RGBA", (image_space.width, image_space.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    segments = _valid_segments(samples, image_space)
    if not segments:
        return layer
    total_points = sum(len(segment) for segment in segments)
    point_index = 0
    for segment in segments:
        if len(segment) < 2:
            continue
        stride = max(1, len(segment) // 120)
        points = [(float(row["image_x"]), float(row["image_y"])) for _, row in segment.iloc[::stride].iterrows()]
        for start, end in zip(points, points[1:]):
            intensity = int(80 + 175 * point_index / max(1, total_points - 1))
            draw.line((*start, *end), fill=(255, intensity, 0, 185), width=2)
            point_index += 1
    return layer


def save_gaze_points_overlay(path: Path, samples: pd.DataFrame, roi: pd.Series | dict[str, object], image_space: ImageSpace) -> Path:
    image = Image.alpha_composite(create_placeholder_canvas(image_space), render_roi_layer(roi, image_space))
    image = Image.alpha_composite(image, render_gaze_points_layer(samples, image_space))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path


def save_scanpath_overlay(path: Path, samples: pd.DataFrame, roi: pd.Series | dict[str, object], image_space: ImageSpace) -> Path:
    image = Image.alpha_composite(create_placeholder_canvas(image_space), render_roi_layer(roi, image_space))
    image = Image.alpha_composite(image, render_scanpath_layer(samples, image_space))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path


def _valid_samples(samples: pd.DataFrame, image_space: ImageSpace) -> pd.DataFrame:
    rows = samples[(samples["is_valid"] == True) & (samples["is_outside_ct"] == False) & (samples["is_ui_glance"] == False)].copy()  # noqa: E712
    rows = rows.sort_values("timestamp_ms")
    return clip_samples_to_image(rows, image_space)


def _valid_segments(samples: pd.DataFrame, image_space: ImageSpace) -> list[pd.DataFrame]:
    if samples.empty:
        return []
    rows = samples.sort_values("timestamp_ms").copy()
    segments: list[pd.DataFrame] = []
    current: list[pd.Series] = []
    for _, row in rows.iterrows():
        valid = bool(row.get("is_valid", False)) and not bool(row.get("is_outside_ct", False)) and not bool(row.get("is_ui_glance", False))
        if valid:
            current.append(row)
        elif current:
            segments.append(clip_samples_to_image(pd.DataFrame(current), image_space))
            current = []
    if current:
        segments.append(clip_samples_to_image(pd.DataFrame(current), image_space))
    return [segment for segment in segments if not segment.empty]
