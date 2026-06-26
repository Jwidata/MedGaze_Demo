"""Single-screen MedGazeAR CT review workstation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.gaze_sources.synthetic_replay_source import canonicalize_synthetic_samples
from app.gaze_sources.tobii_live_source import TobiiLiveSource
from app.ui.behavior_library_view import BEHAVIOR_CARDS, BehaviorLibraryView
from app.ui.case_review_model import CaseReviewModel, build_case_review_model
from app.ui.ct_viewer_widget import CTViewerWidget
from app.ui.gaze_timeline_widget import GazeTimelineWidget
from app.ui.live_prediction_panel import LivePredictionPanel
from app.ui.review_mode import ReviewMode
from app.ui.review_mode_controller import ReviewModeState
from app.ui.source_controller import SYNTHETIC_LABEL, TOBII_LABEL, SourceController
from app.ui.tobii_validation_view import TobiiValidationView
from app.ui.ui_data_loader import WorkstationData, enrich_case_row, load_workstation_data, predict_behavior
from app.ui.ui_theme import LIMITATION_BANNER, TOBII_PLACEHOLDER_MESSAGE, stylesheet


class MedGazeReviewWorkstation(QMainWindow):
    def __init__(self, data: WorkstationData) -> None:
        super().__init__()
        self.data = data
        self.case_model: CaseReviewModel | None = None
        self.current_row: pd.Series | None = None
        self.review_mode_state = ReviewModeState()
        self.source_state = SourceController()
        self.tobii_source = TobiiLiveSource(coordinate_mapper=lambda x, y: self.viewer.map_normalized_screen_to_image(x, y))
        self.live_samples: list[dict[str, object]] = []
        self.overlay_checkboxes: dict[str, QCheckBox] = {}

        self.setWindowTitle("MedGazeAR CT Review Workstation")
        self.resize(1680, 980)

        self.case_selector = QComboBox()
        self.source_selector = QComboBox()
        self.review_mode_selector = QComboBox()
        self.queue = QListWidget()
        self.tobii_status_value = QLabel("Status: not connected")
        self.tobii_status_value.setWordWrap(True)
        self.case_summary = QLabel("Case summary: load a case.")
        self.case_summary.setWordWrap(True)
        self.tobii_status = QLabel("Synthetic replay is reference only. Real Tobii gaze will later enter the same feature extractor and behavior model.")
        self.tobii_status.setWordWrap(True)
        self.source_label = QLabel("Source: Synthetic replay/reference")
        self.source_label.setWordWrap(True)
        self.viewer = CTViewerWidget()
        self.insights = LivePredictionPanel()
        self.timeline = GazeTimelineWidget()
        self.tabs = QTabWidget()
        self.behavior_view: BehaviorLibraryView | None = None
        self.tobii_view: TobiiValidationView | None = None

        self._build_ui()
        self._populate_controls()
        self._connect_signals()
        self.load_case(force=True)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        banner = QLabel(LIMITATION_BANNER)
        banner.setObjectName("Banner")
        layout.addWidget(banner)
        layout.addLayout(self._toolbar())

        self.tabs.addTab(self._ct_review_view(), "CT Review")
        self.behavior_view = BehaviorLibraryView(self.open_behavior_replay)
        self.tabs.addTab(self.behavior_view, "Behaviour Library")
        self.tobii_view = TobiiValidationView(self.detect_tobii, self.start_live_gaze, self.stop_live_gaze)
        self.tabs.addTab(self.tobii_view, "Live Tobii Validation")
        layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(root)

    def _ct_review_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.insights)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 7)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([260, 1080, 300])
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.timeline)
        return page

    def _toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        open_folder = QPushButton("Open CT Folder")
        open_folder.clicked.connect(self.open_ct_folder)
        detect = QPushButton("Detect Tobii")
        start_live = QPushButton("Start Live")
        stop_live = QPushButton("Stop Live")
        record = QPushButton("Record")
        export = QPushButton("Export")
        settings = QPushButton("Settings")
        help_button = QPushButton("Help")
        detect.clicked.connect(self.detect_tobii)
        start_live.clicked.connect(self.start_live_gaze)
        stop_live.clicked.connect(self.stop_live_gaze)
        record.clicked.connect(lambda: QMessageBox.information(self, "Record", "Recording export will be enabled with Tobii session persistence."))
        export.clicked.connect(self.export_current_view)
        settings.clicked.connect(lambda: QMessageBox.information(self, "Settings", "Windowing and Tobii settings are planned for the next UI pass."))
        help_button.clicked.connect(lambda: QMessageBox.information(self, "MedGazeAR", "Workflow: CT + ROI -> live Tobii or synthetic replay -> shared features -> behaviour, attention, cognitive proxy."))
        for widget in (
            open_folder,
            QLabel("Case"),
            self.case_selector,
            QLabel("Source"),
            self.source_selector,
            QLabel("Review Mode"),
            self.review_mode_selector,
            detect,
            start_live,
            stop_live,
            record,
            export,
            settings,
            help_button,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        toolbar.addWidget(self.tobii_status_value)
        return toolbar

    def _left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.case_summary)
        title = QLabel("Review Queue")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        layout.addWidget(title)
        layout.addWidget(self.queue, stretch=1)

        roi_nav = QHBoxLayout()
        previous_roi = QPushButton("Previous ROI")
        next_roi = QPushButton("Next ROI")
        previous_roi.clicked.connect(lambda: self._step_roi(-1))
        next_roi.clicked.connect(lambda: self._step_roi(1))
        roi_nav.addWidget(previous_roi)
        roi_nav.addWidget(next_roi)
        layout.addLayout(roi_nav)

        overlays = QGroupBox("Overlay layers")
        overlay_layout = QVBoxLayout(overlays)
        for text, layer in (("ROI mask/bbox", "roi"), ("Gaze point", "gaze_points"), ("Heatmap", "heatmap"), ("Scanpath", "scanpath")):
            checkbox = QCheckBox(text)
            checkbox.setChecked(layer == "roi")
            checkbox.toggled.connect(lambda enabled, layer_name=layer: self.viewer.set_layer_visible(layer_name, enabled))
            self.overlay_checkboxes[layer] = checkbox
            overlay_layout.addWidget(checkbox)
        layout.addWidget(overlays)
        return panel

    def _populate_controls(self) -> None:
        cases = sorted(self.data.roi_geometry["patient_id"].dropna().astype(str).unique().tolist())
        if not cases and "case_id" in self.data.features:
            cases = sorted(self.data.features["case_id"].dropna().astype(str).unique().tolist())
        self.case_selector.addItems(cases)
        self.source_selector.addItems([SYNTHETIC_LABEL, TOBII_LABEL])
        self.source_selector.setCurrentText(SYNTHETIC_LABEL)
        self.review_mode_selector.addItems([ReviewMode.SILENT.value, ReviewMode.TRAINING.value, ReviewMode.AMBIENT.value, ReviewMode.FEEDBACK.value])
        self.review_mode_selector.setCurrentText(ReviewMode.SILENT.value)

    def _connect_signals(self) -> None:
        self.source_selector.currentTextChanged.connect(self._source_changed)
        self.review_mode_selector.currentTextChanged.connect(self._review_mode_changed)
        self.case_selector.currentTextChanged.connect(lambda _text: self.load_case(force=True))
        self.queue.itemClicked.connect(lambda _item: self.load_selected_roi())
        self.timeline.sample_changed.connect(self._timeline_sample_changed)
        self._review_mode_changed(self.review_mode_selector.currentText())
        self._source_changed(self.source_selector.currentText())

    def load_case(self, force: bool = False) -> None:
        case_id = self.case_selector.currentText()
        if not case_id:
            return
        if force:
            self.timeline.pause()
            self.case_model = None
            self.current_row = None
        if self.case_model is None:
            self.case_model = build_case_review_model(case_id, self.data)
            self.viewer.set_roi_only_navigation(False)
            self.viewer.load_case(self.case_model, None)
            self.viewer.set_slice(0)
            self._set_case_summary(self.case_model.summary())
        self._refresh_queue()
        self.queue.clearSelection()
        self.insights.set_roi(None, show_prediction=self.review_mode_state.show_prediction_feedback)
        self.timeline.set_samples(pd.DataFrame())
        self._review_mode_changed(self.review_mode_selector.currentText())

    def open_ct_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open CT/output folder")
        if not folder:
            return
        path = Path(folder)
        output_root = path.parent if path.name == "dicom_audit" else path
        if not (output_root / "dicom_audit" / "dicom_inventory.csv").exists():
            QMessageBox.information(self, "Open CT folder", "Select a MedGazeAR output folder containing dicom_audit/dicom_inventory.csv.")
            return
        try:
            self.stop_live_gaze()
            self.data = load_workstation_data(output_root, source="synthetic")
            self.case_selector.blockSignals(True)
            self.case_selector.clear()
            cases = sorted(self.data.roi_geometry["patient_id"].dropna().astype(str).unique().tolist())
            self.case_selector.addItems(cases)
            self.case_selector.blockSignals(False)
            self.load_case(force=True)
        except Exception as exc:
            QMessageBox.warning(self, "Open CT folder", f"Could not load workstation data from selected folder:\n{exc}")

    def export_current_view(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export current CT view", "medgazear_current_view.png", "PNG Files (*.png)")
        if path:
            self.viewer.export_current_view(path)

    def load_selected_roi(self) -> None:
        if self.case_model is None:
            self.load_case(force=False)
        if self.case_model is None:
            return
        rows = self._queue_rows()
        item = self.queue.currentItem()
        if item is None or rows.empty:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if not 0 <= index < len(rows):
            return
        row = rows.iloc[index].copy()
        enriched = enrich_case_row(row, self.data) if str(row.get("session_id", "")) else row.copy()
        if not str(enriched.get("session_id", "")):
            enriched["predicted_behavior_label"] = "unavailable_no_replay_evidence"
            enriched["prediction_confidence"] = 0.0
            enriched["guided_narration"] = "This segmentation ROI has no synthetic ROI-level replay/reference episode yet. It remains anatomical context until Tobii or another source provides gaze evidence."
        self.current_row = enriched
        self.viewer.set_selected_roi(enriched)
        self.insights.set_roi(enriched, show_prediction=self.review_mode_state.show_prediction_feedback)
        self.timeline.set_samples(self._gaze_for(enriched))

    def open_behavior_replay(self, behavior: str) -> None:
        row = self._find_behavior_row(behavior)
        if row is None:
            if self.behavior_view is not None:
                self.behavior_view.set_status(f"No representative replay found for {BEHAVIOR_CARDS[behavior]['title']} in the loaded outputs.")
            return
        self.tabs.setCurrentIndex(0)
        self.review_mode_selector.setCurrentText(ReviewMode.TRAINING.value)
        for layer, enabled in {"roi": True, "heatmap": True, "gaze_points": True, "scanpath": False}.items():
            checkbox = self.overlay_checkboxes.get(layer)
            if checkbox is not None:
                checkbox.setChecked(enabled)
            self.viewer.set_layer_visible(layer, enabled)
        self._load_row(row)
        if self.behavior_view is not None:
            self.behavior_view.set_status(f"Opened representative replay for {BEHAVIOR_CARDS[behavior]['title']}.")

    def _find_behavior_row(self, behavior: str) -> pd.Series | None:
        rows = self._queue_rows()
        matched = rows[(rows.get("hidden_behavior_label", pd.Series(index=rows.index, dtype=object)).astype(str) == behavior) & (rows.get("session_id", pd.Series(index=rows.index, dtype=object)).astype(str) != "")]
        if matched.empty:
            global_rows = self.data.features[self.data.features["hidden_behavior_label"].astype(str) == behavior].copy()
            if global_rows.empty or "case_id" not in global_rows.columns:
                return None
            case_id = str(global_rows.iloc[0]["case_id"])
            index = self.case_selector.findText(case_id)
            if index >= 0:
                self.case_selector.setCurrentIndex(index)
                self.load_case(force=True)
                rows = self._queue_rows()
                matched = rows[(rows.get("hidden_behavior_label", pd.Series(index=rows.index, dtype=object)).astype(str) == behavior) & (rows.get("session_id", pd.Series(index=rows.index, dtype=object)).astype(str) != "")]
        if matched.empty:
            return None
        selected = matched.iloc[0].copy()
        for item_index in range(self.queue.count()):
            item = self.queue.item(item_index)
            queue_index = int(item.data(Qt.ItemDataRole.UserRole))
            if queue_index < len(rows) and str(rows.iloc[queue_index].get("session_id", "")) == str(selected.get("session_id", "")) and str(rows.iloc[queue_index].get("roi_id", "")) == str(selected.get("roi_id", "")):
                self.queue.setCurrentRow(item_index)
                break
        return selected

    def _load_row(self, row: pd.Series) -> None:
        enriched = enrich_case_row(row, self.data) if str(row.get("session_id", "")) else row.copy()
        self.current_row = enriched
        self.viewer.set_selected_roi(enriched)
        self.insights.set_roi(enriched, show_prediction=self.review_mode_state.show_prediction_feedback)
        self.timeline.set_samples(self._gaze_for(enriched))

    def _refresh_queue(self) -> None:
        self.queue.clear()
        rows = self._queue_rows()
        for index, row in rows.iterrows():
            ct_slice = _display_slice(row.get("ct_stack_index"), one_based=True)
            status = str(row.get("rule_attention_status", "not_evaluated"))
            badge = _status_badge(status)
            behavior = str(row.get("hidden_behavior_label", "ROI")).replace("_", " ")
            label = f"{badge} {row.get('roi_id', '-')}\n{behavior}\nSlice {ct_slice}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(index))
            self.queue.addItem(item)

    def _queue_rows(self) -> pd.DataFrame:
        if self.case_model is None:
            return pd.DataFrame()
        geometry = self.case_model.roi_geometry.copy()
        if self.case_model.features.empty:
            rows = geometry.copy()
            rows["session_id"] = ""
            rows["hidden_behavior_label"] = "no_replay_evidence"
        else:
            feature_columns = [column for column in self.case_model.features.columns if column not in geometry.columns or column == "roi_id"]
            rows = geometry.merge(self.case_model.features[feature_columns], on="roi_id", how="left")
            rows["session_id"] = rows.get("session_id", pd.Series(index=rows.index, dtype=object)).fillna("")
            rows["hidden_behavior_label"] = rows.get("hidden_behavior_label", pd.Series(index=rows.index, dtype=object)).fillna("no_replay_evidence")
        rows["rule_attention_status"] = rows.get("rule_attention_status", pd.Series(index=rows.index, dtype=object)).fillna("not_evaluated")
        priority = {"not_evaluated": 0, "not_reviewed": 1, "weakly_reviewed": 2, "reviewed": 3}
        rows["_priority"] = rows["rule_attention_status"].map(priority).fillna(9)
        return rows.sort_values(["_priority", "ct_stack_index", "roi_id", "session_id"]).drop(columns=["_priority"]).reset_index(drop=True)

    def _gaze_for(self, row: pd.Series) -> pd.DataFrame:
        session_id = str(row.get("session_id", ""))
        roi_id = str(row.get("roi_id", ""))
        if not session_id or not roi_id:
            return pd.DataFrame()
        samples = self.data.gaze[(self.data.gaze["session_id"].astype(str) == session_id) & (self.data.gaze["roi_id"].astype(str) == roi_id)].copy()
        if not samples.empty:
            samples = canonicalize_synthetic_samples(samples, ct_stack_index=int(float(row.get("ct_stack_index", 0))))
        return samples

    def _step_roi(self, delta: int) -> None:
        if self.queue.count() == 0:
            return
        current = self.queue.currentRow()
        if current < 0:
            current = 0 if delta >= 0 else self.queue.count() - 1
        else:
            current = (current + delta) % self.queue.count()
        self.queue.setCurrentRow(current)
        self.load_selected_roi()

    def start_replay(self) -> None:
        if self.source_selector.currentText() == TOBII_LABEL:
            QMessageBox.information(self, "Live source selected", "Use Start live gaze for Tobii live mode.")
            return
        if self.timeline.samples.empty:
            QMessageBox.information(self, "Select replay source", "Select an ROI/session or behavior example before replay.")
            return
        self.timeline.play()

    def _source_changed(self, source: str) -> None:
        self.source_state.source = source
        self.source_label.setText(self.source_state.status_text)
        is_tobii = self.source_state.is_tobii
        self.timeline.set_replay_enabled(not is_tobii)
        if is_tobii:
            self.timeline.pause()
            self.insights.set_live_status("Live gaze: Tobii selected. Detect tracker, then start live gaze.")
        else:
            if self.tobii_source.streaming:
                self.stop_live_gaze()
            self.insights.set_live_status("Live gaze: synthetic replay/reference mode")

    def detect_tobii(self) -> None:
        devices = self.tobii_source.find_devices()
        status = self.tobii_source.get_status()
        self.tobii_status_value.setText(f"Status: {status.message}")
        self.insights.set_live_status(f"Live gaze: {status.message}")
        self._sync_tobii_validation_view()
        if devices:
            self.tobii_source.connect_first_device()
            status = self.tobii_source.get_status()
            self.tobii_status_value.setText(f"Status: {status.message}")
            self.insights.set_live_status(f"Live gaze: {status.message}")
            self._sync_tobii_validation_view()

    def start_live_gaze(self) -> None:
        self.source_selector.setCurrentText(TOBII_LABEL)
        self.live_samples.clear()
        self.tobii_source.start_stream(self._live_gaze_sample_received)
        status = self.tobii_source.get_status()
        self.tobii_status_value.setText(f"Status: {status.message}")
        self.insights.set_live_status(f"Live gaze: {status.message}")
        self._sync_tobii_validation_view()

    def stop_live_gaze(self) -> None:
        self.tobii_source.stop_stream()
        status = self.tobii_source.get_status()
        self.tobii_status_value.setText(f"Status: {status.message}")
        self.insights.set_live_status(f"Live gaze: {status.message}")
        self._sync_tobii_validation_view()

    def _live_gaze_sample_received(self, sample: dict[str, object]) -> None:
        if self.current_row is not None:
            sample["session_id"] = "tobii_live"
            sample["roi_id"] = str(self.current_row.get("roi_id", ""))
            sample["ct_stack_index"] = self.viewer.current_slice_index
        self.live_samples.append(sample)
        self.live_samples = self.live_samples[-600:]
        self.viewer.set_playback_sample(sample, follow_slice=False)
        self._update_live_prediction()

    def _update_live_prediction(self) -> None:
        if self.current_row is None:
            self.insights.set_live_status(f"Live gaze: {len(self.live_samples)} samples buffered. Select an ROI to score live gaze.")
            return
        row = self.current_row.copy()
        samples = pd.DataFrame(self.live_samples)
        if samples.empty:
            return
        valid = samples[(samples["is_valid"] == True) & (samples["is_outside_ct"] == False)]  # noqa: E712
        x0, y0, x1, y1 = [float(row.get(key, 0) or 0) for key in ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max")]
        inside = valid[(pd.to_numeric(valid["image_x"], errors="coerce") >= x0) & (pd.to_numeric(valid["image_x"], errors="coerce") <= x1) & (pd.to_numeric(valid["image_y"], errors="coerce") >= y0) & (pd.to_numeric(valid["image_y"], errors="coerce") <= y1)]
        row["total_gaze_time_inside_roi_ms"] = float(len(inside) * 16.667)
        row["gaze_hit_count_inside_roi"] = int(len(inside))
        row["gaze_validity_ratio"] = float(len(valid) / max(1, len(samples)))
        row["roi_revisit_count"] = 1 if len(inside) > 3 else 0
        row["background_gaze_ratio"] = float(max(0, len(valid) - len(inside)) / max(1, len(valid)))
        row["rule_attention_status"] = "reviewed" if len(inside) >= 5 else "weakly_reviewed" if len(inside) > 0 else "not_reviewed"
        row["cognitive_load_proxy"] = "high_load_proxy" if row["background_gaze_ratio"] > 0.7 else "medium_load_proxy" if row["background_gaze_ratio"] > 0.35 else "low_load_proxy"
        prediction = predict_behavior(row, self.data)
        row["predicted_behavior_label"] = prediction["label"]
        row["prediction_confidence"] = prediction["confidence"]
        self.insights.set_roi(row, show_prediction=self.review_mode_state.show_prediction_feedback)
        self.insights.set_live_status(f"Live gaze: {len(samples)} samples buffered | ROI hits={len(inside)}")

    def _sync_tobii_validation_view(self) -> None:
        if self.tobii_view is None:
            return
        status = self.tobii_source.get_status()
        sdk = "available" if self.tobii_source.is_sdk_available() else "missing"
        device = status.message
        capture = "streaming" if status.streaming else "stopped"
        self.tobii_view.update_status(sdk, device, capture)

    def _set_case_summary(self, summary: dict[str, object]) -> None:
        self.case_summary.setText(
            f"Case: {summary.get('patient_id', '-')} | CT slices: {summary.get('total_slices', '-')} | "
            f"ROI slices: {summary.get('roi_slices', '-')} | ROIs: {summary.get('roi_count', '-')}\n"
            f"Queue: reviewed={summary.get('reviewed', 0)}, weak={summary.get('weakly_reviewed', 0)}, "
            f"not reviewed={summary.get('not_reviewed', 0)}, not evaluated={summary.get('not_evaluated', 0)}"
        )

    def _review_mode_changed(self, mode: str) -> None:
        self.review_mode_state.mode = mode
        for layer, enabled in self.review_mode_state.layers().items():
            checkbox = self.overlay_checkboxes.get(layer)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(enabled)
                checkbox.blockSignals(False)
            self.viewer.set_layer_visible(layer, enabled)
        self.viewer.set_feedback_colors(self.review_mode_state.feedback_colors)
        self.insights.set_roi(self.current_row, show_prediction=self.review_mode_state.show_prediction_feedback)

    def _timeline_sample_changed(self, sample: object) -> None:
        if isinstance(sample, dict):
            self.viewer.set_playback_sample(sample, follow_slice=self.timeline.follow_enabled())


def _display_slice(value: object, one_based: bool) -> str:
    try:
        number = int(float(value))
        if one_based:
            number += 1
        return str(number)
    except (TypeError, ValueError):
        return "-"


def _status_badge(status: str) -> str:
    return {
        "reviewed": "[green]",
        "weakly_reviewed": "[yellow]",
        "not_reviewed": "[orange]",
        "not_evaluated": "[red]",
    }.get(status, "[gray]")


def launch_review_workstation(output_root: str | Path | None = None, source: str = "synthetic") -> int:
    if source == "future_tobii_placeholder":
        raise ValueError(TOBII_PLACEHOLDER_MESSAGE)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(stylesheet())
    data = load_workstation_data(output_root, source="synthetic")
    window = MedGazeReviewWorkstation(data)
    window.show()
    return app.exec()


def smoke_test_workstation(output_root: str | Path | None = None, source: str = "synthetic") -> dict[str, object]:
    if source == "future_tobii_placeholder":
        return {"status": "placeholder", "message": TOBII_PLACEHOLDER_MESSAGE}
    try:
        data = load_workstation_data(output_root, source="synthetic")
        cases = int(data.roi_geometry["patient_id"].nunique()) if "patient_id" in data.roi_geometry else 0
        return {"status": "ok", "cases": cases, "sample_row_loaded": not data.roi_geometry.empty}
    except FileNotFoundError as exc:
        return {"status": "ok", "warning": str(exc), "sample_row_loaded": False}
