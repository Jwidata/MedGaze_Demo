"""Real CT slice viewer with slice-aware ROI and gaze overlays."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw
from PyQt6.QtCore import QPointF, QRectF, QSizeF, Qt, QSignalBlocker, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget

from app.ui.case_review_model import CaseReviewModel, load_roi_mask
from app.visualization.coordinate_mapper import ImageSpace
from app.visualization.heatmap_renderer import render_heatmap_layer
from app.visualization.scanpath_renderer import render_gaze_points_layer, render_scanpath_layer


ROI_COLOR = (152, 212, 164, 190)
SELECTED_COLOR = (224, 238, 168, 230)
GAZE_COLOR = (0, 180, 255, 135)
SCANPATH_COLOR = (255, 160, 0, 185)
HEATMAP_LOW = (255, 0, 0, 170)
HEATMAP_HIGH = (255, 190, 0, 170)


class CTViewerWidget(QWidget):
    slice_requested = pyqtSignal(int)
    slider_scrub_started = pyqtSignal()
    slider_scrub_finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.case_model: CaseReviewModel | None = None
        self.selected_row: pd.Series | None = None
        self.current_slice_index = 0
        self.total_slice_count = 0
        self.playback_sample: dict[str, object] | None = None
        self.overlay_samples = pd.DataFrame()
        self.layer_visible = {"roi": False, "heatmap": False, "gaze_points": False, "scanpath": False}
        self.feedback_colors = False
        self.show_all_rois = True
        self.window_center: float | None = None
        self.window_width: float | None = None

        self.canvas = _CTCanvasWidget(self)
        self.square_host = _SquareViewportHost(self.canvas)

        self.external_controls_host = QWidget()
        self.external_controls_layout = QHBoxLayout(self.external_controls_host)
        self.external_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.external_controls_layout.setSpacing(8)
        self.external_controls_host.setVisible(False)

        self.previous_button = QPushButton("‹")
        self.previous_button.setProperty("variant", "quiet")
        self.previous_button.setMaximumWidth(44)
        self.previous_button.clicked.connect(lambda: self.slice_requested.emit(self.current_slice_index - 1))

        self.next_button = QPushButton("›")
        self.next_button.setProperty("variant", "quiet")
        self.next_button.setMaximumWidth(44)
        self.next_button.clicked.connect(lambda: self.slice_requested.emit(self.current_slice_index + 1))

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.valueChanged.connect(self._slider_requested)
        self.slice_slider.sliderPressed.connect(self.slider_scrub_started.emit)
        self.slice_slider.sliderReleased.connect(self.slider_scrub_finished.emit)

        self.slice_label = QLabel("1 / 1")
        self.slice_label.setObjectName("ContextValue")

        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setProperty("variant", "quiet")
        self.zoom_out_button.setMaximumWidth(44)
        self.zoom_out_button.clicked.connect(lambda: self.canvas.adjust_zoom(1 / 1.2))

        self.reset_view_button = QPushButton("⊙")
        self.reset_view_button.setProperty("variant", "quiet")
        self.reset_view_button.setMaximumWidth(44)
        self.reset_view_button.clicked.connect(self.canvas.fit_to_view)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setProperty("variant", "quiet")
        self.zoom_in_button.setMaximumWidth(44)
        self.zoom_in_button.clicked.connect(lambda: self.canvas.adjust_zoom(1.2))

        toolbar = QWidget()
        toolbar.setMaximumHeight(38)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)
        toolbar_layout.addWidget(self.external_controls_host)
        toolbar_layout.addStretch(1)

        self.viewport_frame = QFrame()
        self.viewport_frame.setObjectName("ViewerViewport")
        self.viewport_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewport_layout = QHBoxLayout(self.viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        viewport_layout.addWidget(self.square_host)

        navigation = QWidget()
        navigation.setObjectName("ViewerFooter")
        navigation.setMaximumHeight(48)
        navigation_layout = QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(8, 6, 8, 6)
        navigation_layout.setSpacing(6)
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.slice_slider, stretch=1)
        navigation_layout.addWidget(self.next_button)
        navigation_layout.addWidget(self.slice_label)
        navigation_layout.addWidget(self.zoom_out_button)
        navigation_layout.addWidget(self.reset_view_button)
        navigation_layout.addWidget(self.zoom_in_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(toolbar)
        layout.addWidget(self.viewport_frame, stretch=1)
        layout.addWidget(navigation)

    def load_case(self, case_model: CaseReviewModel) -> None:
        self.case_model = case_model
        self.total_slice_count = case_model.total_slices
        self.selected_row = None
        self.playback_sample = None
        self.current_slice_index = 0
        with QSignalBlocker(self.slice_slider):
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(max(0, self.total_slice_count - 1))
            self.slice_slider.setValue(0)
        self._update_slice_label()
        self.refresh()

    def clear_selection(self) -> None:
        self.selected_row = None
        self.playback_sample = None
        self.refresh()

    def set_selected_roi(self, selected_row: pd.Series | None) -> None:
        self.selected_row = selected_row
        self.playback_sample = None
        self.refresh()

    def clear_playback_sample(self) -> None:
        self.playback_sample = None
        self.refresh()

    def set_playback_sample(self, sample: dict[str, object] | None) -> None:
        self.playback_sample = sample
        self.refresh()

    def set_overlay_samples(self, samples: pd.DataFrame) -> None:
        self.overlay_samples = samples.copy() if not samples.empty else pd.DataFrame()
        self.refresh()

    def set_current_slice(self, slice_index: int) -> None:
        if self.case_model is None:
            return
        self.current_slice_index = max(0, min(int(slice_index), self.total_slice_count - 1))
        with QSignalBlocker(self.slice_slider):
            self.slice_slider.setValue(self.current_slice_index)
        self._update_slice_label()
        self.refresh()

    def set_layer_visible(self, layer_name: str, enabled: bool) -> None:
        self.layer_visible[layer_name] = bool(enabled)
        self.refresh()

    def set_external_controls(self, widgets: list[QWidget]) -> None:
        while self.external_controls_layout.count():
            item = self.external_controls_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
        for widget in widgets:
            self.external_controls_layout.addWidget(widget)
        self.external_controls_host.setVisible(bool(widgets))

    def set_feedback_colors(self, enabled: bool) -> None:
        self.feedback_colors = bool(enabled)
        self.refresh()

    def set_show_all_rois(self, enabled: bool) -> None:
        self.show_all_rois = bool(enabled)
        self.refresh()

    def set_window_preset(self, center: float, width: float) -> None:
        self.window_center = float(center)
        self.window_width = float(width)
        self.refresh()

    def diagnostics_text(self, screen_mode: str) -> str:
        native_w, native_h = self.canvas.native_image_size()
        viewport_w, viewport_h = self.canvas.current_viewport_size()
        display_scale = self.canvas.display_scale()
        fit_label = "full" if self.canvas.is_fit_mode() else "zoom"
        return f"Native: {native_w}x{native_h} | Viewport: {viewport_w}x{viewport_h} | Display scale: {display_scale:.2f}x | Fit: {fit_label} | Screen: {screen_mode}"

    def refresh(self) -> None:
        image = self._render_current_slice()
        if image is not None:
            self.canvas.set_image(image)

    def export_current_view(self, path: str) -> None:
        image = self._render_current_slice()
        if image is not None:
            image.convert("RGB").save(path)

    def map_normalized_screen_to_image(self, x_norm: float, y_norm: float) -> tuple[float, float, bool]:
        diagnostics = self.map_normalized_screen_to_image_diagnostics(x_norm, y_norm)
        return float(diagnostics["image_x"]), float(diagnostics["image_y"]), bool(diagnostics["is_outside_ct"])

    def map_normalized_screen_to_image_diagnostics(self, x_norm: float, y_norm: float) -> dict[str, object]:
        if self.canvas._image_size == (0, 0):
            return {
                "gaze_x_norm": x_norm,
                "gaze_y_norm": y_norm,
                "screen_x": None,
                "screen_y": None,
                "viewer_local_x": None,
                "viewer_local_y": None,
                "image_rect": None,
                "image_x": 0.0,
                "image_y": 0.0,
                "is_outside_ct": True,
                "failure_reason": "NO_IMAGE_LOADED",
            }
        screen = self.window().windowHandle().screen() if self.window().windowHandle() is not None else None
        if screen is None:
            return {
                "gaze_x_norm": x_norm,
                "gaze_y_norm": y_norm,
                "screen_x": None,
                "screen_y": None,
                "viewer_local_x": None,
                "viewer_local_y": None,
                "image_rect": None,
                "image_x": 0.0,
                "image_y": 0.0,
                "is_outside_ct": True,
                "failure_reason": "SCREEN_UNAVAILABLE",
            }
        screen_geometry = screen.geometry()
        global_x = screen_geometry.x() + float(x_norm) * screen_geometry.width()
        global_y = screen_geometry.y() + float(y_norm) * screen_geometry.height()
        rect = self.canvas.image_rect()
        rect_top_left = self.canvas.mapToGlobal(rect.topLeft().toPoint())
        global_rect = QRectF(QPointF(rect_top_left.x(), rect_top_left.y()), rect.size())
        outside = not global_rect.contains(global_x, global_y)
        image_w, image_h = self.canvas._image_size
        image_x = (global_x - global_rect.x()) / max(1.0, global_rect.width()) * image_w
        image_y = (global_y - global_rect.y()) / max(1.0, global_rect.height()) * image_h
        local_point = self.mapFromGlobal(rect_top_left)
        return {
            "gaze_x_norm": x_norm,
            "gaze_y_norm": y_norm,
            "screen_x": global_x,
            "screen_y": global_y,
            "viewer_local_x": global_x - global_rect.x(),
            "viewer_local_y": global_y - global_rect.y(),
            "viewer_host_x": local_point.x(),
            "viewer_host_y": local_point.y(),
            "image_rect": {
                "x": global_rect.x(),
                "y": global_rect.y(),
                "width": global_rect.width(),
                "height": global_rect.height(),
            },
            "image_x": max(0.0, min(float(image_w - 1), image_x)),
            "image_y": max(0.0, min(float(image_h - 1), image_y)),
            "is_outside_ct": outside,
            "failure_reason": "OUTSIDE_IMAGE_RECT" if outside else "",
        }

    def _slider_requested(self, index: int) -> None:
        self.slice_requested.emit(int(index))

    def _update_slice_label(self) -> None:
        display_total = max(1, self.total_slice_count)
        self.slice_label.setText(f"{self.current_slice_index + 1} / {display_total}")

    def _render_current_slice(self) -> Image.Image | None:
        if self.case_model is None:
            return None
        base = self.case_model.image_for_slice(self.current_slice_index, self.window_center, self.window_width)
        if base is None:
            return None
        image = base.copy().convert("RGBA")
        image_space = ImageSpace(image.width, image.height)
        current_rois = self.case_model.rois_on_slice(self.current_slice_index)
        selected_gaze = self._selected_gaze_on_current_slice()
        recent_gaze = _recent_gaze_samples(selected_gaze, window_ms=2000.0)
        if not recent_gaze.empty and self.layer_visible.get("heatmap", False):
            image = Image.alpha_composite(image, render_heatmap_layer(recent_gaze, image_space, alpha=38))
        if self.layer_visible.get("roi", False):
            rois_to_draw = current_rois
            if not self.show_all_rois and self.selected_row is not None:
                selected_roi_id = str(self.selected_row.get("roi_id", ""))
                rois_to_draw = current_rois[current_rois["roi_id"].astype(str) == selected_roi_id]
            elif not self.show_all_rois:
                rois_to_draw = current_rois.iloc[0:0]
            _draw_rois(image, rois_to_draw, self.selected_row)
        if not recent_gaze.empty:
            is_live_gaze = str(recent_gaze.iloc[-1].get("source_type", "")) == "live_tobii"
            if self.layer_visible.get("scanpath", False):
                image = Image.alpha_composite(image, render_scanpath_layer(recent_gaze, image_space))
            if self.layer_visible.get("gaze_points", False):
                points = recent_gaze.tail(1) if self.playback_sample is not None or is_live_gaze else recent_gaze
                image = Image.alpha_composite(image, render_gaze_points_layer(points, image_space))
        return image

    def _selected_gaze_on_current_slice(self) -> pd.DataFrame:
        if not self.overlay_samples.empty:
            return _renderable_gaze_samples(self.overlay_samples)
        if self.case_model is None:
            return pd.DataFrame()
        if self.playback_sample is not None and self.playback_sample.get("mode") == "ct_cine":
            return pd.DataFrame()
        if self.playback_sample is not None and self.playback_sample.get("source_type") == "live_tobii":
            if int(self.playback_sample.get("ct_stack_index", self.current_slice_index)) != int(self.current_slice_index):
                return pd.DataFrame()
            return _renderable_gaze_samples(pd.DataFrame([self.playback_sample]))
        if self.selected_row is None:
            return pd.DataFrame()
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
        return _renderable_gaze_samples(rows)


def _renderable_gaze_samples(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    valid = rows.copy()
    if "is_valid" in valid.columns:
        valid = valid[valid["is_valid"] == True]  # noqa: E712
    if "is_outside_ct" in valid.columns:
        valid = valid[valid["is_outside_ct"] == False]  # noqa: E712
    return valid.copy()


def _recent_gaze_samples(rows: pd.DataFrame, window_ms: float) -> pd.DataFrame:
    if rows.empty or "timestamp_ms" not in rows.columns:
        return rows
    last_timestamp = pd.to_numeric(rows["timestamp_ms"], errors="coerce").fillna(0.0).max()
    return rows[pd.to_numeric(rows["timestamp_ms"], errors="coerce").fillna(0.0) >= max(0.0, last_timestamp - window_ms)].copy()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.canvas.wheelEvent(event)


def _draw_rois(image: Image.Image, rois: pd.DataFrame, selected_row: pd.Series | None) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    selected_roi_id = None if selected_row is None else str(selected_row.get("roi_id", ""))
    for _, roi in rois.iterrows():
        if not bool(roi.get("roi_overlay_visible", True)):
            continue
        is_selected = str(roi.get("roi_id", "")) == selected_roi_id
        color = SELECTED_COLOR if is_selected else _status_color(str(roi.get("roi_cue_state", "none")))
        mask = load_roi_mask(roi)
        if mask is not None:
            _draw_mask_outline(draw, mask, color, selected=is_selected)
            continue
        x0, y0, x1, y1 = [float(roi.get(key, 0)) for key in ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max")]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2 if is_selected else 1)


def _status_color(status: str) -> tuple[int, int, int, int]:
    return {
        "selected": SELECTED_COLOR,
        "weak": (224, 198, 96, 210),
        "missed": (226, 167, 98, 210),
        "none": ROI_COLOR,
    }.get(status, ROI_COLOR)


def _draw_mask_outline(draw: ImageDraw.ImageDraw, mask, color: tuple[int, int, int, int], selected: bool = False) -> None:
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return
    stride = max(1, len(xs) // 1800)
    for x, y in zip(xs[::stride], ys[::stride]):
        draw.point((int(x), int(y)), fill=(color[0], color[1], color[2], 26))
    boundary = _mask_boundary_points(mask)
    boundary_stride = max(1, len(boundary) // 2000)
    for x, y in boundary[::boundary_stride]:
        draw.point((int(x), int(y)), fill=(color[0], color[1], color[2], 220 if selected else 168))


def _mask_boundary_points(mask) -> list[tuple[int, int]]:
    ys, xs = mask.nonzero()
    points: list[tuple[int, int]] = []
    height, width = mask.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
            points.append((x, y))
            continue
        if not (mask[y - 1, x] and mask[y + 1, x] and mask[y, x - 1] and mask[y, x + 1]):
            points.append((x, y))
    return points


class _SquareViewportHost(QWidget):
    def __init__(self, child: QWidget) -> None:
        super().__init__()
        self.child = child
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)
        row.addWidget(child)
        row.addStretch(1)
        layout.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        side = max(0, min(self.width(), self.height()))
        x = max(0, (self.width() - side) // 2)
        y = max(0, (self.height() - side) // 2)
        self.child.setGeometry(x, y, side, side)
        if self.child.is_fit_mode():
            QTimer.singleShot(0, self.child.fit_to_view)


class _CTCanvasWidget(QGraphicsView):
    def __init__(self, viewer: CTViewerWidget) -> None:
        super().__init__()
        self.viewer = viewer
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#05070b"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._image_size = (1, 1)
        self._fit_mode = True
        self._wheel_accumulator = 0
        self.setMouseTracking(True)

    def set_image(self, image: Image.Image) -> None:
        self._image_size = image.size
        pixmap = _pil_to_pixmap(image)
        first_image = self._pixmap_item.pixmap().isNull()
        self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setOffset(0, 0)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, float(pixmap.width()), float(pixmap.height())))
        if first_image or self._fit_mode:
            QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._fit_mode = True
        self.resetTransform()
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        rect = self._pixmap_item.boundingRect()
        view_w = self.viewport().width() - 14
        view_h = self.viewport().height() - 14
        if view_w <= 0 or view_h <= 0:
            return
        scale_x = view_w / max(rect.width(), 1.0)
        scale_y = view_h / max(rect.height(), 1.0)
        scale_factor = min(scale_x, scale_y) * 0.975
        self.scale(scale_factor, scale_factor)
        self.centerOn(rect.center())

    def adjust_zoom(self, factor: float) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._fit_mode = False
        self.scale(factor, factor)
        self.centerOn(self._pixmap_item)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.adjust_zoom(1.15 if delta > 0 else 1 / 1.15)
        else:
            step_threshold = 120
            self._wheel_accumulator += int(delta)
            while abs(self._wheel_accumulator) >= step_threshold:
                self.viewer.slice_requested.emit(self.viewer.current_slice_index + (1 if self._wheel_accumulator > 0 else -1))
                self._wheel_accumulator += -step_threshold if self._wheel_accumulator > 0 else step_threshold
        event.accept()

    def paintEvent(self, event) -> None:
        if self._pixmap_item.pixmap().isNull():
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#ffb0a0"))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, "No CT pixels loaded")
            return
        super().paintEvent(event)

    def _image_rect(self) -> QRectF:
        if self._pixmap_item.pixmap().isNull():
            return QRectF()
        mapped = self.mapFromScene(self._pixmap_item.sceneBoundingRect()).boundingRect()
        top_left = self.viewport().mapTo(self, mapped.topLeft())
        return QRectF(QPointF(top_left), QSizeF(mapped.size()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            QTimer.singleShot(0, self.fit_to_view)

    def image_rect(self) -> QRectF:
        return self._image_rect()

    def native_image_size(self) -> tuple[int, int]:
        pixmap = self._pixmap_item.pixmap()
        return pixmap.width(), pixmap.height()

    def current_viewport_size(self) -> tuple[int, int]:
        rect = self.viewport().rect()
        return rect.width(), rect.height()

    def display_scale(self) -> float:
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull() or pixmap.width() == 0:
            return 0.0
        rect = self._image_rect()
        return rect.width() / float(pixmap.width())

    def is_fit_mode(self) -> bool:
        return self._fit_mode


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    buffer = BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap
