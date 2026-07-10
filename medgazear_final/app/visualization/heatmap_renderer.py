"""Image-coordinate gaze heatmap rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from app.visualization.coordinate_mapper import ImageSpace, clip_samples_to_image
from app.visualization.roi_overlay_renderer import create_placeholder_canvas, render_roi_layer


def generate_heatmap_array(samples: pd.DataFrame, image_space: ImageSpace, sigma: float = 10.0) -> np.ndarray:
    valid = samples[(samples["is_valid"] == True) & (samples["is_outside_ct"] == False) & (samples["is_ui_glance"] == False)].copy()  # noqa: E712
    heatmap = np.zeros((image_space.height, image_space.width), dtype=np.float32)
    if valid.empty:
        return heatmap
    clipped = clip_samples_to_image(valid, image_space)
    xs = clipped["image_x"].round().astype(int).clip(0, image_space.width - 1).to_numpy()
    ys = clipped["image_y"].round().astype(int).clip(0, image_space.height - 1).to_numpy()
    np.add.at(heatmap, (ys, xs), 1.0)
    max_count = float(heatmap.max())
    if max_count <= 0:
        return heatmap
    grayscale = np.clip(heatmap / max_count * 255, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(grayscale, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    heatmap = np.asarray(blurred, dtype=np.float32)
    max_value = float(heatmap.max())
    return heatmap / max_value if max_value > 0 else heatmap


def render_heatmap_layer(samples: pd.DataFrame, image_space: ImageSpace, alpha: int = 170) -> Image.Image:
    heatmap = generate_heatmap_array(samples, image_space)
    rgba = np.zeros((image_space.height, image_space.width, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(np.maximum(0.0, (heatmap - 0.45) / 0.55) * 255, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(np.maximum(0.0, 1.0 - np.abs(heatmap - 0.30) / 0.30) * 255, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(np.maximum(0.0, 1.0 - heatmap / 0.25) * 40, 0, 40).astype(np.uint8)
    rgba[..., 3] = np.clip(np.maximum(0.0, (heatmap - 0.08) / 0.92) * alpha, 0, alpha).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def save_heatmap_overlay(path: Path, samples: pd.DataFrame, roi: pd.Series | dict[str, object], image_space: ImageSpace) -> Path:
    image = Image.alpha_composite(create_placeholder_canvas(image_space), render_heatmap_layer(samples, image_space))
    image = Image.alpha_composite(image, render_roi_layer(roi, image_space))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path
