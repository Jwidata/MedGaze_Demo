"""Real CT slice viewer with slice-aware ROI and gaze overlays."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from app.ui.case_review_model import CaseReviewModel, load_roi_mask
from app.visualization.coordinate_mapper import ImageSpace
from app.visualization.heatmap_renderer import render_heatmap_layer
from app.visualization.scanpath_renderer import render_gaze_points_layer, render_scanpath_layer


class CTViewerWidget(QWidget):
    slice_changed = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.case_model: CaseReviewModel | None = None
        self.selected_row: pd.Series | None = None
        self.current_slice_index = 0
        self.playback_sample: dict[str, object] | None = None
        self.layer_visible = {"roi": True, "heatmap": False, "gaze_points": False, "scanpath": False}
        self.feedback_colors = False
        self.roi_only_navigation = False
        self.canvas = _CTCanvasWidget(self)
        self.slice_label = QLabel("Slice: -")
        self.roi_label = QLabel("Slice - / - | ROI - | Replay -")
        self.legend_label = QLabel(
            "<span style='background:#00cc66;color:#00cc66'>■</span> ROI &nbsp; "
            "<span style='border:2px solid #ffdd00;color:#ffdd00'>□</span> selected ROI &nbsp; "
            "<span style='color:#00aaff'>●</span> gaze point &nbsp; "
            "<span style='color:#ff3333'>■</span><span style='color:#ffdd00'>■</span> heatmap &nbsp; "
            "<span style='color:#ff9900'>━</span> scanpath &nbsp; "
            "<span style='background:#00cc66;color:#001b10'> reviewed </span> &nbsp; "
            "<span style='background:#ffdd00;color:#2b2500'> weak </span> &nbsp; "
            "<span style='background:#ff9900;color:#2b1600'> not reviewed </span> &nbsp; "
            "<span style='background:#ff3333;color:#2b0000'> not evaluated </span>"
        )
        self.legend_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ffb0a0;")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self.set_slice)

        previous_slice = QPushButton("Previous slice")
        next_slice = QPushButton("Next slice")
        fit_button = QPushButton("Fit")
        zoom_in = QPushButton("Zoom +")
        zoom_out = QPushButton("Zoom -")
        previous_slice.clicked.connect(lambda: self.step_slice(-1))
        next_slice.clicked.connect(lambda: self.step_slice(1))
        fit_button.clicked.connect(self.canvas.fit_to_view)
        zoom_in.clicked.connect(lambda: self.canvas.adjust_zoom(1.2))
        zoom_out.clicked.connect(lambda: self.canvas.adjust_zoom(1 / 1.2))

        controls = QHBoxLayout()
        controls.addStretch(1)
        for button in (previous_slice, next_slice, fit_button, zoom_out, zoom_in):
            controls.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.slice_slider)
        layout.addWidget(self.slice_label)
        layout.addWidget(self.roi_label)
        layout.addWidget(self.legend_label)
        layout.addWidget(self.error_label)

    def load_case(self, case_model: CaseReviewModel, selected_row: pd.Series | None = None) -> None:
        self.case_model = case_model
        self.selected_row = selected_row
        self.playback_sample = None
        self.current_slice_index = int(float(selected_row.get("ct_stack_index", selected_row.get("slice_index", 0)))) if selected_row is not None else 0
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max(0, len(self._navigable_slices()) - 1))
        self.slice_slider.setValue(self._position_for_slice(self.current_slice_index))
        self.slice_slider.blockSignals(False)
        self.error_label.setText("")
        self.canvas.fit_to_view()
        self.refresh()

    def set_selected_roi(self, selected_row: pd.Series | None) -> None:
        self.selected_row = selected_row
        self.playback_sample = None
        if selected_row is not None:
            self.set_slice(int(float(selected_row.get("ct_stack_index", selected_row.get("slice_index", self.current_slice_index)))))
        else:
            self.refresh()

    def set_layer_visible(self, layer_name: str, enabled: bool) -> None:
        self.layer_visible[layer_name] = bool(enabled)
        self.refresh()

    def set_playback_sample(self, sample: dict[str, object], follow_slice: bool = True) -> None:
        self.playback_sample = sample
        if follow_slice and "ct_stack_index" in sample:
            self.current_slice_index = int(sample["ct_stack_index"])
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(self._position_for_slice(self.current_slice_index))
            self.slice_slider.blockSignals(False)
        self.refresh()

    def set_roi_only_navigation(self, enabled: bool) -> None:
        self.roi_only_navigation = bool(enabled)
        if self.case_model is not None:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setMaximum(max(0, len(self._navigable_slices()) - 1))
            self.slice_slider.setValue(self._position_for_slice(self.current_slice_index))
            self.slice_slider.blockSignals(False)
        self.refresh()

    def set_feedback_colors(self, enabled: bool) -> None:
        self.feedback_colors = bool(enabled)
        self.refresh()

    def set_slice(self, slice_index: int) -> None:
        if self.case_model is None:
            return
        slices = self._navigable_slices()
        if 0 <= int(slice_index) < len(slices) and self.roi_only_navigation:
            self.current_slice_index = slices[int(slice_index)]
        else:
            self.current_slice_index = max(0, min(int(slice_index), self.case_model.total_slices - 1))
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self._position_for_slice(self.current_slice_index))
        self.slice_slider.blockSignals(False)
        self.refresh()
        self.slice_changed.emit(self.current_slice_index)

    def step_slice(self, delta: int) -> None:
        slices = self._navigable_slices()
        if self.current_slice_index in slices:
            pos = slices.index(self.current_slice_index)
        else:
            pos = 0
        self.current_slice_index = slices[(pos + delta) % len(slices)] if slices else self.current_slice_index
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self._position_for_slice(self.current_slice_index))
        self.slice_slider.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        image = self._render_current_slice()
        if image is not None:
            self.canvas.set_image(image)
        if self.case_model is not None:
            label = f"CT slice: {self.current_slice_index + 1} / {self.case_model.total_slices}"
            if self.roi_only_navigation:
                label += f" | ROI slice position: {self._position_for_slice(self.current_slice_index) + 1} / {len(self._navigable_slices())}"
            self.slice_label.setText(label)

    def export_current_view(self, path: str) -> None:
        image = self._render_current_slice()
        if image is not None:
            image.convert("RGB").save(path)

    def _render_current_slice(self) -> Image.Image | None:
        if self.case_model is None:
            return None
        base = self.case_model.image_for_slice(self.current_slice_index)
        if base is None:
            self.error_label.setText("CT pixels could not be loaded for this slice. Check DICOM audit file paths and pixel transfer syntax support.")
            return None
        image = base.copy().convert("RGBA")
        image_space = ImageSpace(image.width, image.height)
        current_rois = self.case_model.rois_on_slice(self.current_slice_index)
        if self.layer_visible.get("roi", True):
            _draw_rois(image, current_rois, self.selected_row, self.feedback_colors)
        selected_gaze = self._selected_gaze_on_current_slice()
        if not selected_gaze.empty:
            if self.layer_visible.get("heatmap", True):
                image = Image.alpha_composite(image, render_heatmap_layer(selected_gaze, image_space, alpha=45))
            if self.layer_visible.get("gaze_points", True):
                points = selected_gaze.tail(1) if self.playback_sample is not None else selected_gaze.tail(20)
                image = Image.alpha_composite(image, render_gaze_points_layer(points, image_space))
            if self.layer_visible.get("scanpath", True):
                image = Image.alpha_composite(image, render_scanpath_layer(selected_gaze.tail(20), image_space))
        self._update_roi_label(current_rois)
        return image

    def _selected_gaze_on_current_slice(self) -> pd.DataFrame:
        if self.case_model is None or self.selected_row is None:
            return pd.DataFrame()
        if self.playback_sample is not None and self.playback_sample.get("mode") == "ct_cine":
            return pd.DataFrame()
        if self.playback_sample is not None and self.playback_sample.get("source_type") == "tobii_live":
            if int(self.playback_sample.get("ct_stack_index", self.current_slice_index)) != int(self.current_slice_index):
                return pd.DataFrame()
            return pd.DataFrame([self.playback_sample])
        session_id = str(self.selected_row.get("session_id", ""))
        roi_id = str(self.selected_row.get("roi_id", ""))
        timestamp = None
        if self.playback_sample is not None:
            session_id = str(self.playback_sample.get("session_id", session_id))
            roi_id = str(self.playback_sample.get("roi_id", roi_id))
            timestamp = float(self.playback_sample.get("timestamp_ms", 0))
        rows = self.case_model.gaze_on_slice(session_id, roi_id, self.current_slice_index)
        if timestamp is not None and not rows.empty:
            rows = rows[pd.to_numeric(rows["timestamp_ms"], errors="coerce").fillna(0) <= timestamp]
        return rows

    def _update_roi_label(self, current_rois: pd.DataFrame) -> None:
        if self.selected_row is None:
            self.roi_label.setText(f"Slice {self.current_slice_index + 1} | ROIs on slice {len(current_rois)} | Replay -")
            return
        self.roi_label.setText(
            f"Slice {self.current_slice_index + 1} | selected ROI {self.selected_row.get('roi_id', '-')} | ROIs on slice {len(current_rois)}"
        )

    def jump_to_ct_slice(self, ct_stack_index: int) -> None:
        if self.case_model is None:
            return
        self.current_slice_index = max(0, min(int(ct_stack_index), self.case_model.total_slices - 1))
        self.refresh()

    def map_normalized_screen_to_image(self, x_norm: float, y_norm: float) -> tuple[float, float, bool]:
        rect = self.canvas.image_rect()
        screen_x = rect.x() + float(x_norm) * rect.width()
        screen_y = rect.y() + float(y_norm) * rect.height()
        outside = not rect.contains(screen_x, screen_y)
        if self.canvas._image_size == (0, 0):
            return 0.0, 0.0, True
        image_w, image_h = self.canvas._image_size
        image_x = (screen_x - rect.x()) / max(1.0, rect.width()) * image_w
        image_y = (screen_y - rect.y()) / max(1.0, rect.height()) * image_h
        return max(0.0, min(float(image_w - 1), image_x)), max(0.0, min(float(image_h - 1), image_y)), outside

    def _navigable_slices(self) -> list[int]:
        if self.case_model is None:
            return [0]
        roi_slices = self.case_model.roi_slice_indices()
        return roi_slices if self.roi_only_navigation and roi_slices else list(range(self.case_model.total_slices))

    def _position_for_slice(self, slice_index: int) -> int:
        slices = self._navigable_slices()
        if int(slice_index) in slices:
            return slices.index(int(slice_index))
        return 0

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.canvas.wheelEvent(event)


def _draw_rois(image: Image.Image, rois: pd.DataFrame, selected_row: pd.Series | None, feedback_colors: bool = False) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    selected_roi_id = None if selected_row is None else str(selected_row.get("roi_id", ""))
    for _, roi in rois.iterrows():
        is_selected = str(roi.get("roi_id", "")) == selected_roi_id
        color = (255, 220, 0, 230) if is_selected else _status_color(str(roi.get("rule_attention_status", "")), feedback_colors)
        fill = (255, 220, 0, 60) if is_selected else (*color[:3], 35)
        mask = load_roi_mask(roi)
        if mask is not None:
            _draw_mask_outline(draw, mask, color, fill)
        x0, y0, x1, y1 = [float(roi.get(key, 0)) for key in ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max")]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3 if is_selected else 2)
        draw.ellipse((float(roi.get("centroid_x", 0)) - 3, float(roi.get("centroid_y", 0)) - 3, float(roi.get("centroid_x", 0)) + 3, float(roi.get("centroid_y", 0)) + 3), fill=color)


def _status_color(status: str, feedback_colors: bool) -> tuple[int, int, int, int]:
    if not feedback_colors:
        return (0, 255, 120, 180)
    return {
        "reviewed": (0, 220, 90, 210),
        "weakly_reviewed": (255, 220, 0, 220),
        "not_reviewed": (255, 145, 0, 220),
        "not_evaluated": (255, 60, 60, 220),
    }.get(status, (150, 160, 170, 180))


def _draw_mask_outline(draw: ImageDraw.ImageDraw, mask, color: tuple[int, int, int, int], fill: tuple[int, int, int, int]) -> None:
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return
    # Lightweight fill by sparse points plus bbox/centroid; avoids heavy contour dependencies.
    stride = max(1, len(xs) // 2500)
    for x, y in zip(xs[::stride], ys[::stride]):
        draw.point((int(x), int(y)), fill=fill)


class _CTCanvasWidget(QWidget):
    def __init__(self, viewer: CTViewerWidget) -> None:
        super().__init__()
        self.viewer = viewer
        self.setMinimumSize(640, 640)
        self._pixmap: QPixmap | None = None
        self._image_size = (1, 1)
        self._zoom_factor = 1.0
        self.setMouseTracking(True)

    def set_image(self, image: Image.Image) -> None:
        self._image_size = image.size
        self._pixmap = _pil_to_pixmap(image)
        self.update()

    def fit_to_view(self) -> None:
        self._zoom_factor = 1.0
        self.update()

    def adjust_zoom(self, factor: float) -> None:
        self._zoom_factor = max(0.25, min(8.0, self._zoom_factor * factor))
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.pixelDelta().y() or event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.adjust_zoom(1.15 if delta > 0 else 1 / 1.15)
        else:
            self.viewer.step_slice(1 if delta > 0 else -1)
        event.accept()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05070b"))
        if self._pixmap is None:
            painter.setPen(QColor("#ffb0a0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No CT pixels loaded")
            return
        target = self._image_rect()
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

    def _image_rect(self) -> QRectF:
        width, height = map(float, self._image_size)
        available = self.rect().adjusted(18, 18, -18, -18)
        scale = min(available.width() / width, available.height() / height) * self._zoom_factor
        scaled_width = width * scale
        scaled_height = height * scale
        return QRectF(available.x() + (available.width() - scaled_width) / 2, available.y() + (available.height() - scaled_height) / 2, scaled_width, scaled_height)

    def image_rect(self) -> QRectF:
        return self._image_rect()


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    buffer = BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap
