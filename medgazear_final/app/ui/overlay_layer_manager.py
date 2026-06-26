"""Overlay layer state and cache management."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


LAYER_NAMES = ("roi", "heatmap", "gaze_points", "scanpath")


@dataclass
class OverlayLayerManager:
    visible: dict[str, bool] = field(default_factory=lambda: {name: True for name in LAYER_NAMES})
    cache: dict[str, Image.Image] = field(default_factory=dict)

    def set_layer(self, layer_name: str, enabled: bool) -> None:
        if layer_name not in self.visible:
            raise ValueError(f"Unknown overlay layer: {layer_name}")
        self.visible[layer_name] = bool(enabled)

    def toggle_layer(self, layer_name: str) -> bool:
        self.set_layer(layer_name, not self.visible[layer_name])
        return self.visible[layer_name]

    def set_cache(self, layer_name: str, image: Image.Image) -> None:
        if layer_name not in LAYER_NAMES and layer_name != "canvas":
            raise ValueError(f"Unknown overlay layer: {layer_name}")
        self.cache[layer_name] = image

    def composite(self) -> Image.Image | None:
        canvas = self.cache.get("canvas")
        if canvas is None:
            return None
        image = canvas.copy().convert("RGBA")
        for layer_name in LAYER_NAMES:
            layer = self.cache.get(layer_name)
            if layer is not None and self.visible[layer_name]:
                image = Image.alpha_composite(image, layer.convert("RGBA"))
        return image
