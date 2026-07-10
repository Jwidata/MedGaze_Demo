"""Gaze-aware MedGazeAR review workstation."""

from __future__ import annotations

import csv
import json
import logging
import time
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, QSignalBlocker, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.attention.rule_attention_engine import classify_feature_row
from app.attention.attention_thresholds import load_attention_thresholds
from app.features.behavior_feature_builder import build_behavior_feature_row, feature_parity_matrix
from app.features.roi_feature_extractor import roi_masks_for_samples
from app.gaze_sources.live_validation import tracking_preflight_summary, write_live_validation_bundle
from app.gaze_sources.tobii_live_source import TobiiLiveSource
from app.ui.case_review_model import CaseReviewModel, base_roi_id, build_case_review_model, load_roi_mask
from app.ui.ct_viewer_widget import CTViewerWidget
from app.ui.ct_windowing import window_preset
from app.ui.gaze_timeline_widget import GazeTimelineWidget
from app.ui.ui_data_loader import WorkstationData, enrich_case_row, load_workstation_data, predict_behavior
from app.ui.ui_theme import TOBII_PLACEHOLDER_MESSAGE
from app.ui_training.model_explanations import BehaviorExplanationService
from app.ui_training.theme import stylesheet


SESSION_IDLE = "idle"
SESSION_ACTIVE = "active"
SESSION_STOPPED = "stopped"


def _safe_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


REFERENCE_METRICS = {
    "ROI dwell ms": "total_gaze_time_inside_roi_ms",
    "ROI hit count": "gaze_hit_count_inside_roi",
    "Validity ratio": "gaze_validity_ratio",
    "Background ratio": "background_gaze_ratio",
    "ROI revisit count": "roi_revisit_count",
}


@dataclass
class ReviewModeState:
    mode: str = "Silent"


class MedGazeReviewWorkstation(QMainWindow):
    live_sample_received = pyqtSignal(object)

    def __init__(self, data: WorkstationData, output_root: str | Path | None = None) -> None:
        super().__init__()
        self.data = data
        self.output_root = Path(output_root).resolve() if output_root else Path.cwd() / "outputs"
        self.attention_thresholds = load_attention_thresholds()

        self.case_model: CaseReviewModel | None = None
        self.current_slice_index = 0
        self.total_slice_count = 0
        self.selected_roi: pd.Series | None = None
        self.current_source = "Tobii Live"
        self.review_mode = ReviewModeState(mode="Silent")
        self.session_state = SESSION_IDLE
        self.current_session_id = ""
        self.live_samples: list[dict[str, object]] = []
        self.live_roi_overrides: dict[str, dict[str, object]] = {}
        self.slice_gaze_history: dict[int, list[dict[str, object]]] = {}
        self.slice_visit_counts: dict[int, int] = {}
        self.slice_state_store: dict[int, dict[str, object]] = {}
        self.roi_state_store: dict[str, dict[str, object]] = {}
        self.live_validation_summary: dict[str, object] | None = None
        self.tobii_preflight_result: dict[str, object] | None = None
        self.mapping_failure_diagnostics: list[dict[str, object]] = []
        self._last_mapping_diagnostic: dict[str, object] | None = None
        self.saved_session_path: Path | None = None
        self.screen_mode = "work-area"
        self.class_reference_baselines = build_class_reference_baselines(self.data)
        self.explanation_service = BehaviorExplanationService(self.data.behavior_dataset, self.data.feature_columns, self.data.model, self.data.label_mapping)
        self.explanation_checkpoints: dict[str, list[dict[str, object]]] = {}
        self.overlay_preferences = {"roi": False, "gaze_points": False, "heatmap": False, "scanpath": False}
        self.overlay_effective = dict(self.overlay_preferences)
        self._pending_live_samples: list[dict[str, object]] = []
        self._syncing_worklist = False
        self._user_scrubbing = False
        self._deferred_slice_update_index: int | None = None

        self.viewer = CTViewerWidget()
        self.tobii_source = TobiiLiveSource(
            coordinate_mapper=self._map_live_coordinate,
            screen_geometry_provider=self._current_screen_geometry,
        )

        self.live_update_timer = QTimer(self)
        self.live_update_timer.setInterval(16)
        self.live_update_timer.timeout.connect(self._process_pending_live_sample)
        self.slice_update_timer = QTimer(self)
        self.slice_update_timer.setSingleShot(True)
        self.slice_update_timer.setInterval(45)
        self.slice_update_timer.timeout.connect(self._run_deferred_slice_update)

        self.setWindowTitle("MedGazeAR Review Workstation")
        self.setMinimumSize(1100, 700)

        self.case_selector = QComboBox()
        self.series_selector = QComboBox()
        self.source_selector = QComboBox()
        self.mode_selector = QComboBox()
        self.detect_button = QPushButton("Detect Tobii")
        self.detect_button.setProperty("variant", "quiet")
        self.start_button = QPushButton("Start Session")
        self.pause_button = QPushButton("Pause Session")
        self.end_button = QPushButton("End Session")
        self.end_button.setProperty("liveState", "stop")
        self.start_button.setProperty("liveState", "inactive")
        self.pause_button.setProperty("variant", "quiet")

        self.overlay_roi = QCheckBox("ROI contour")
        self.overlay_roi.setObjectName("OverlayRoiToggle")
        self.overlay_gaze = QCheckBox("Live gaze")
        self.overlay_gaze.setObjectName("OverlayGazeToggle")
        self.overlay_heatmap = QCheckBox("Heatmap")
        self.overlay_heatmap.setObjectName("OverlayHeatmapToggle")
        self.overlay_scanpath = QCheckBox("Scanpath")
        self.overlay_scanpath.setObjectName("OverlayScanpathToggle")
        self._configure_overlay_controls()
        self.viewer.set_external_controls([self.overlay_roi, self.overlay_gaze, self.overlay_heatmap, self.overlay_scanpath])

        self.slice_worklist = QListWidget()
        self.slice_worklist.setObjectName("RoiQueue")
        self.slice_worklist.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        self.case_status_value = QLabel("-")
        self.case_status_value.setObjectName("ContextValue")
        self.tobii_status_value = QLabel("Disconnected")
        self.tobii_status_value.setObjectName("ContextValue")
        self.top_validity_value = QLabel("Gaze tracking inactive")
        self.top_validity_value.setObjectName("CompactMeta")
        self.session_timer_value = QLabel("00:00")
        self.session_timer_value.setObjectName("ContextValue")
        self.slice_meta = QLabel("Slice 1 of 1")
        self.slice_meta.setObjectName("CompactMeta")

        self.progress_total = QLabel("0")
        self.progress_sufficient = QLabel("0")
        self.progress_weak = QLabel("0")
        self.progress_insufficient = QLabel("0")
        self.progress_pending = QLabel("0")
        self.case_roi_slices_viewed = QLabel("0")
        self.case_target_bearing_slices = QLabel("0")
        self.case_queued_count = QLabel("0")
        self.session_progress_value = QLabel("0 / 0")
        self.case_completed_slices_value = QLabel("0")

        self.current_slice_target_count = QLabel("0")
        self.current_slice_reviewed_count = QLabel("0")
        self.current_slice_weak_count = QLabel("0")
        self.current_slice_missed_count = QLabel("0")
        self.current_slice_pending_count = QLabel("0")

        self.selected_roi_id_value = QLabel("No ROI selected")
        self.selected_roi_id_value.setObjectName("ContextValue")
        self.selected_roi_slice_value = QLabel("Slice -")
        self.selected_roi_slice_value.setObjectName("CompactMeta")
        self.selected_roi_type_value = QLabel("Select a segmentation-derived ROI on this slice to inspect its attention evidence.")
        self.selected_roi_type_value.setObjectName("CompactMeta")
        self.current_slice_roi_selector = QComboBox()
        self.current_slice_roi_selector.setObjectName("CurrentSliceRoiSelector")

        self.slice_behavior_status = QLabel("Unknown")
        self.slice_behavior_status.setObjectName("ContextValue")
        self.slice_behavior_meta = QLabel("Dwell: -  |  Visits: 0  |  Revisits: 0")
        self.slice_behavior_meta.setObjectName("CompactMeta")
        self.slice_behavior_quality = QLabel("Validity: -  |  Outside-CT: -")
        self.slice_behavior_quality.setObjectName("CompactMeta")

        self.prediction_state_value = QLabel("Pending / No prediction")
        self.prediction_state_value.setObjectName("ContextValue")
        self.prediction_confidence_value = QLabel("Confidence: -")
        self.prediction_confidence_value.setObjectName("CompactMeta")
        self.prediction_probability_value = QLabel("")
        self.prediction_probability_value.setObjectName("CompactMeta")
        self.shap_summary_value = QLabel("")
        self.shap_summary_value.setWordWrap(True)
        self.shap_timeline_value = QLabel("")
        self.shap_timeline_value.setWordWrap(True)
        self.synthetic_context_value = QLabel("")
        self.synthetic_context_value.setWordWrap(True)
        self.combined_interpretation_value = QLabel("")
        self.combined_interpretation_value.setWordWrap(True)
        self.technical_evidence_toggle = QPushButton("View technical evidence")
        self.technical_evidence_toggle.setProperty("variant", "quiet")
        self.technical_evidence_toggle.setCheckable(True)
        self.technical_evidence_text = QTextEdit()
        self.technical_evidence_text.setReadOnly(True)
        self.technical_evidence_text.setVisible(False)
        self.waterfall_toggle = QPushButton("View full model explanation")
        self.waterfall_toggle.setProperty("variant", "quiet")
        self.waterfall_toggle.setCheckable(True)
        self.waterfall_text = QTextEdit()
        self.waterfall_text.setReadOnly(True)
        self.waterfall_text.setVisible(False)
        self.shap_disclaimer_value = QLabel("SHAP explains how model features influenced this prediction. It does not establish that the prediction is correct or clinically valid.")
        self.shap_disclaimer_value.setObjectName("MicroMeta")
        self.shap_disclaimer_value.setWordWrap(True)
        self.attention_status_value = QLabel("Not yet evaluated")
        self.attention_status_value.setObjectName("ContextValue")
        self.attention_reason_value = QLabel("Evaluation begins after sufficient slice viewing time.")
        self.attention_reason_value.setObjectName("CompactMeta")
        self.attention_reason_value.setWordWrap(True)

        self.attention_evidence_value = QLabel("No ROI selected")
        self.attention_evidence_value.setWordWrap(True)
        self.behavior_support_value = QLabel("No ROI selected")
        self.behavior_support_value.setWordWrap(True)
        self.cognitive_proxy_value = QLabel("Unknown")
        self.cognitive_proxy_value.setObjectName("ContextValue")
        self.cognitive_proxy_meta = QLabel("No cognitive proxy available")
        self.cognitive_proxy_meta.setObjectName("CompactMeta")
        self.cognitive_proxy_meta.setWordWrap(True)
        self.scanpath_disclaimer_value = QLabel("Research model trained on synthetic gaze patterns. Output is not a clinical or cognitive assessment.")
        self.scanpath_disclaimer_value.setObjectName("MicroMeta")
        self.scanpath_disclaimer_value.setWordWrap(True)
        self.research_notice_value = QLabel("MedGazeAR is a non-clinical research prototype for gaze-aware radiology workflow modeling. It does not assess diagnostic correctness or clinical competence.")
        self.research_notice_value.setObjectName("CompactMeta")
        self.research_notice_value.setWordWrap(True)
        self.research_diagnostics_toggle = QPushButton("Research diagnostics")
        self.research_diagnostics_toggle.setProperty("variant", "quiet")
        self.research_diagnostics_toggle.setCheckable(True)
        self.research_diagnostics_text = QTextEdit()
        self.research_diagnostics_text.setReadOnly(True)
        self.research_diagnostics_text.setVisible(False)

        self.timeline = GazeTimelineWidget()
        self.live_session_title = QLabel("Tobii Live")
        self.live_session_title.setObjectName("DisplayValue")
        self.live_session_status = QLabel("Current slice: 1 / 1")
        self.live_session_status.setObjectName("CompactMeta")
        self.live_session_metrics = QLabel("Recent slices: none")
        self.live_session_metrics.setObjectName("CompactMeta")
        self.live_session_quality = QLabel("Valid gaze: -  |  Samples: 0  |  Session: 00:00")
        self.live_session_quality.setObjectName("CompactMeta")

        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

        self._build_ui()
        self._populate_controls()
        self._connect_signals()
        self._window_preset_changed("Lung")
        self._update_live_button_state(False)
        self._source_changed(self.current_source)
        self.load_case(force=True)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        top_bar = self._build_experiment_bar()
        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(10)
        workspace_layout.addWidget(self._wrap_sidebar(self._build_left_sidebar(), 290, 320))
        workspace_layout.addWidget(self._build_center_workspace(), stretch=1)
        workspace_layout.addWidget(self._wrap_sidebar(self._build_right_sidebar(), 360, 400))
        session_bar = self._build_session_bar()

        layout.addWidget(top_bar, 0)
        layout.addWidget(workspace, 1)
        layout.addWidget(session_bar, 0)
        self.setCentralWidget(root)

    def _build_experiment_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("ExperimentBar")
        bar.setMinimumHeight(82)
        bar.setMaximumHeight(96)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(self._top_group("Case", self.case_selector))
        layout.addWidget(self._top_group("Gaze Source", self.source_selector))
        layout.addWidget(self._top_group("Tobii Status", self._tobii_status_group()))
        layout.addWidget(self._top_group("Experiment Mode", self.mode_selector))
        layout.addWidget(self._top_group("Session Timer", self.session_timer_value))
        layout.addWidget(self._top_group("Session Controls", self._session_controls_group()))
        return bar

    def _tobii_status_group(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(self.tobii_status_value, stretch=1)
        row_layout.addWidget(self.detect_button)
        layout.addWidget(row)
        layout.addWidget(self.top_validity_value)
        return wrapper

    def _session_controls_group(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.end_button)
        return wrapper

    def _build_left_sidebar(self) -> QWidget:
        shell = QWidget()
        shell.setObjectName("SidebarShell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._section("Case Level Coverage", self._build_case_progress_section()))
        layout.addWidget(self._section("ROI Level Coverage", self._build_roi_level_coverage_section()))
        layout.addWidget(self._section("Review Queue", self._build_slice_list_section()), stretch=1)
        return shell

    def _build_slice_list_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.slice_worklist)
        return widget

    def _build_current_slice_coverage_section(self) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        rows = [
            ("ROI instances present", self.current_slice_target_count),
            ("Reviewed", self.current_slice_reviewed_count),
            ("Weakly reviewed", self.current_slice_weak_count),
            ("Missed", self.current_slice_missed_count),
            ("Not evaluated", self.current_slice_pending_count),
        ]
        for index, (label, value) in enumerate(rows):
            text = QLabel(label)
            text.setObjectName("CompactMeta")
            value.setObjectName("ContextValue")
            grid.addWidget(text, index, 0)
            grid.addWidget(value, index, 1, alignment=Qt.AlignmentFlag.AlignRight)
        return widget

    def _build_case_progress_section(self) -> QWidget:
        rows = [
            ("ROI slices viewed", self.case_roi_slices_viewed),
            ("ROI slices remaining", self.case_queued_count),
            ("Completed ROI slices", self.case_completed_slices_value),
            ("Total slices", self.case_status_value),
            ("ROI-bearing slices", self.case_target_bearing_slices),
            ("Review completion", self.session_progress_value),
        ]
        return self._metric_card_grid(rows, columns=2)

    def _build_roi_level_coverage_section(self) -> QWidget:
        rows = [
            ("Target ROIs", self.progress_total),
            ("Reviewed", self.progress_sufficient),
            ("Weak evidence", self.progress_weak),
            ("Missed", self.progress_insufficient),
            ("Not evaluated", self.progress_pending),
        ]
        return self._metric_card_grid(rows, columns=2)

    def _metric_card_grid(self, rows: list[tuple[str, QLabel]], columns: int = 2) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for index, (label, value) in enumerate(rows):
            card = QFrame()
            card.setObjectName("MetricCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(2)
            text = QLabel(label)
            text.setObjectName("CompactMeta")
            value.setObjectName("MetricCardValue")
            layout.addWidget(text)
            layout.addWidget(value)
            row_index = index // columns
            column_index = index % columns
            grid.addWidget(card, row_index, column_index)
        return widget

    def _build_center_workspace(self) -> QWidget:
        shell = QWidget()
        shell.setObjectName("CenterShell")
        shell.setMinimumWidth(760)
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        header = QWidget()
        header.setMaximumHeight(38)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addStretch(1)
        header_layout.addWidget(self.slice_meta)
        layout.addWidget(header, 0)
        layout.addWidget(self.viewer, 1)
        return shell

    def _build_right_sidebar(self) -> QWidget:
        shell = QWidget()
        shell.setObjectName("SidebarShell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._section("Selected ROI Review", self._selected_roi_body()))
        layout.addWidget(self._section("Current Slice Coverage", self._build_current_slice_coverage_section()))
        layout.addWidget(self._section("Attention interpretation", self._attention_interpretation_body()))
        layout.addWidget(self._section("Live Behavior Analysis", self._behavior_model_body()))
        layout.addWidget(self._section("Why the model predicted this", self.shap_summary_value))
        layout.addWidget(self._section("Prediction / Explanation Timeline", self.shap_timeline_value))
        layout.addWidget(self._section("Live vs Synthetic Feature Context", self.synthetic_context_value))
        layout.addWidget(self._section("ROI Review Evidence", self.attention_evidence_value))
        layout.addWidget(self._section("Why this status?", self.attention_reason_value))
        layout.addWidget(self._section("Current Slice Summary", self._slice_behavior_body()))
        layout.addWidget(self._section("Cognitive-load proxy - Experimental", self._cognitive_proxy_body()))
        layout.addWidget(self._section("Behavior-Load Relation", self.combined_interpretation_value))
        layout.addWidget(self._section("Technical Evidence", self._technical_evidence_body()))
        layout.addWidget(self._section("Full SHAP Waterfall", self._waterfall_body()))
        layout.addWidget(self._section("Research diagnostics", self._research_diagnostics_body()))
        layout.addWidget(self.research_notice_value)
        layout.addStretch(1)
        return shell

    def _selected_roi_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.current_slice_roi_selector)
        layout.addWidget(self.selected_roi_id_value)
        layout.addWidget(self.selected_roi_slice_value)
        layout.addWidget(self.selected_roi_type_value)
        return widget

    def _attention_interpretation_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.attention_status_value)
        return widget

    def _slice_behavior_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.slice_behavior_status)
        layout.addWidget(self.slice_behavior_meta)
        layout.addWidget(self.slice_behavior_quality)
        return widget

    def _behavior_model_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.prediction_state_value)
        layout.addWidget(self.prediction_confidence_value)
        layout.addWidget(self.prediction_probability_value)
        layout.addWidget(self.behavior_support_value)
        layout.addWidget(self.scanpath_disclaimer_value)
        layout.addWidget(self.shap_disclaimer_value)
        return widget

    def _technical_evidence_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.technical_evidence_toggle)
        layout.addWidget(self.technical_evidence_text)
        return widget

    def _waterfall_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.waterfall_toggle)
        layout.addWidget(self.waterfall_text)
        return widget

    def _cognitive_proxy_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.cognitive_proxy_value)
        layout.addWidget(self.cognitive_proxy_meta)
        return widget

    def _research_diagnostics_body(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.research_diagnostics_toggle)
        layout.addWidget(self.research_diagnostics_text)
        return widget

    def _build_session_bar(self) -> QWidget:
        self.session_shell = QWidget()
        self.session_shell.setObjectName("SessionShell")
        self.session_shell.setMinimumHeight(58)
        self.session_shell.setMaximumHeight(72)
        layout = QVBoxLayout(self.session_shell)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)
        self.session_stack = QStackedLayout()
        self.session_stack.addWidget(self.timeline)
        self.session_stack.addWidget(self._build_live_session_panel())
        layout.addLayout(self.session_stack)
        return self.session_shell

    def _build_live_session_panel(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)
        self.live_session_title.setObjectName("CompactMeta")
        layout.addWidget(self.live_session_title)
        layout.addWidget(self.live_session_status)
        layout.addWidget(self.live_session_metrics, stretch=1)
        layout.addWidget(self.live_session_quality)
        return widget

    def _wrap_sidebar(self, content: QWidget, min_width: int, max_width: int) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setMinimumWidth(min_width)
        scroll.setMaximumWidth(max_width)
        return scroll

    def _top_group(self, title: str, body: QWidget) -> QWidget:
        frame = QFrame()
        frame.setObjectName("StatusGroup")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)
        label = QLabel(title)
        label.setObjectName("TopBarTitle")
        layout.addWidget(label)
        layout.addWidget(body)
        return frame

    def _section(self, title: str, body: QWidget) -> QWidget:
        frame = QFrame()
        frame.setObjectName("PanelSection")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        layout.addWidget(body)
        return frame

    def _populate_controls(self) -> None:
        cases = sorted(self.data.roi_geometry.get("patient_id", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
        if not cases and "case_id" in self.data.features:
            cases = sorted(self.data.features["case_id"].dropna().astype(str).unique().tolist())
        self.case_selector.addItems(cases)
        self.source_selector.addItems(["Tobii Live", "Synthetic Replay"])
        self.mode_selector.addItems(["Silent", "Assisted"])
        self.source_selector.setCurrentText(self.current_source)
        self.mode_selector.setCurrentText(self.review_mode.mode)
        self._populate_series_selector()

    def _populate_series_selector(self) -> None:
        self.series_selector.blockSignals(True)
        self.series_selector.clear()
        case_id = self.case_selector.currentText()
        if case_id:
            rows = self.data.roi_geometry[self.data.roi_geometry["patient_id"].astype(str) == str(case_id)]
            for value in sorted(rows["ct_series_instance_uid"].dropna().astype(str).unique().tolist()):
                self.series_selector.addItem(value)
        self.series_selector.blockSignals(False)

    def _connect_signals(self) -> None:
        self.case_selector.currentTextChanged.connect(self._case_changed)
        self.source_selector.currentTextChanged.connect(self._source_changed)
        self.mode_selector.currentTextChanged.connect(self._mode_changed)
        self.detect_button.clicked.connect(self.detect_tobii)
        self.start_button.clicked.connect(self._start_or_resume_session)
        self.pause_button.clicked.connect(self._pause_session)
        self.end_button.clicked.connect(self._end_session)
        self.overlay_roi.toggled.connect(lambda enabled: self._overlay_preference_changed("roi", enabled))
        self.overlay_gaze.toggled.connect(lambda enabled: self._overlay_preference_changed("gaze_points", enabled))
        self.overlay_heatmap.toggled.connect(lambda enabled: self._overlay_preference_changed("heatmap", enabled))
        self.overlay_scanpath.toggled.connect(lambda enabled: self._overlay_preference_changed("scanpath", enabled))
        self.slice_worklist.currentItemChanged.connect(self._slice_worklist_selection_changed)
        self.current_slice_roi_selector.currentIndexChanged.connect(self._current_slice_roi_changed)
        self.viewer.slice_requested.connect(self.set_current_slice)
        self.viewer.slider_scrub_started.connect(self._begin_user_scrub)
        self.viewer.slider_scrub_finished.connect(self._end_user_scrub)
        self.timeline.sample_changed.connect(self._timeline_sample_changed)
        self.timeline.play_button.clicked.connect(self._synthetic_session_started)
        self.timeline.pause_button.clicked.connect(self._synthetic_session_stopped)
        self.timeline.reset_button.clicked.connect(self._synthetic_session_reset)
        self.live_sample_received.connect(self._handle_live_sample_on_ui_thread)
        self.research_diagnostics_toggle.toggled.connect(self.research_diagnostics_text.setVisible)
        self.technical_evidence_toggle.toggled.connect(self.technical_evidence_text.setVisible)
        self.waterfall_toggle.toggled.connect(self.waterfall_text.setVisible)

    def _case_changed(self, _text: str) -> None:
        self._populate_series_selector()
        self.load_case(force=True)

    def _overlay_preference_changed(self, layer_name: str, enabled: bool) -> None:
        self.overlay_preferences[layer_name] = bool(enabled)
        self._refresh_overlay_policy()

    def _configure_overlay_controls(self) -> None:
        for checkbox in (self.overlay_roi, self.overlay_gaze, self.overlay_heatmap, self.overlay_scanpath):
            checkbox.setIconSize(checkbox.iconSize())

    def _map_live_coordinate(self, x_norm: float, y_norm: float) -> tuple[float, float, bool] | None:
        diagnostics = self.viewer.map_normalized_screen_to_image_diagnostics(x_norm, y_norm)
        self._last_mapping_diagnostic = diagnostics
        failure_reason = str(diagnostics.get("failure_reason", "") or "")
        if failure_reason:
            self.mapping_failure_diagnostics.append(diagnostics)
            self.mapping_failure_diagnostics = self.mapping_failure_diagnostics[-25:]
        return float(diagnostics["image_x"]), float(diagnostics["image_y"]), bool(diagnostics["is_outside_ct"])

    def _begin_user_scrub(self) -> None:
        self._user_scrubbing = True

    def _end_user_scrub(self) -> None:
        self._user_scrubbing = False

    def load_case(self, force: bool = False) -> None:
        case_id = self.case_selector.currentText()
        if not case_id:
            return
        if force:
            self.case_model = None
        if self.case_model is None:
            self.case_model = build_case_review_model(case_id, self.data, series_uid=self.series_selector.currentText() or None)
            self.viewer.load_case(self.case_model)
        self._reset_session_state()
        self.total_slice_count = self.case_model.total_slices if self.case_model is not None else 0
        self.case_status_value.setText(str(self.total_slice_count))
        self.selected_roi = None
        self.viewer.clear_selection()
        self._initialize_case_state_store()
        self.current_session_id = self._default_session_id_for_case()
        self._sync_case_model_roi_states()
        self._reload_synthetic_session_history()
        self._set_all_overlay_preferences(False)
        self._refresh_overlay_policy()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_current_slice_roi_selector()
        self.set_current_slice(0, record_visit=False, clear_playback=True)
        self.slice_worklist.clearSelection()
        self._refresh_case_progress(self._queue_rows())
        self._refresh_source_ui()
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def _reset_session_state(self) -> None:
        self.session_state = SESSION_IDLE
        self.live_samples.clear()
        self.live_roi_overrides.clear()
        self.slice_gaze_history.clear()
        self.slice_visit_counts.clear()
        self.slice_state_store.clear()
        self.mapping_failure_diagnostics.clear()
        self._last_mapping_diagnostic = None
        self.saved_session_path = None
        self._pending_live_samples.clear()
        self.viewer.clear_playback_sample()
        self.viewer.set_overlay_samples(pd.DataFrame())

    def _default_session_id_for_case(self) -> str:
        if self.case_model is None or self.case_model.gaze.empty:
            return ""
        rows = self.case_model.gaze.copy()
        if "case_id" in rows.columns:
            rows = rows[rows["case_id"].astype(str) == str(self.case_selector.currentText() or "")].copy()
        session_ids = rows.get("session_id", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        return sorted(session_ids)[0] if session_ids else ""

    def _session_samples(self, include_playback_limit: bool = True) -> pd.DataFrame:
        if self.case_model is None or not self.current_session_id:
            return pd.DataFrame()
        rows = self.case_model.session_gaze_samples(self.current_session_id)
        if rows.empty:
            return rows
        rows = self._ensure_ct_stack_index(rows)
        if include_playback_limit and self.viewer.playback_sample is not None:
            timestamp = float(self.viewer.playback_sample.get("timestamp_ms", 0) or 0)
            rows = rows[pd.to_numeric(rows.get("timestamp_ms"), errors="coerce").fillna(0.0) <= timestamp].copy()
        return rows.sort_values("timestamp_ms").reset_index(drop=True)

    def _reload_synthetic_session_history(self) -> None:
        if self.current_source != "Synthetic Replay":
            return
        session_rows = self._session_samples(include_playback_limit=True)
        self._rebuild_slice_gaze_history(session_rows)
        self._rebuild_slice_states(session_rows)

    def _ensure_ct_stack_index(self, rows: pd.DataFrame) -> pd.DataFrame:
        result = rows.copy()
        if "ct_stack_index" not in result.columns:
            result["ct_stack_index"] = pd.to_numeric(result.get("slice_index"), errors="coerce")
        if "slice_index" not in result.columns:
            result["slice_index"] = pd.to_numeric(result.get("ct_stack_index"), errors="coerce")
        result["ct_stack_index"] = pd.to_numeric(result["ct_stack_index"], errors="coerce").fillna(-1).astype(int)
        result["slice_index"] = pd.to_numeric(result["slice_index"], errors="coerce").fillna(result["ct_stack_index"]).astype(int)
        return result

    def _rebuild_slice_gaze_history(self, samples: pd.DataFrame) -> None:
        self.slice_gaze_history.clear()
        if samples.empty:
            return
        normalized = self._ensure_ct_stack_index(samples)
        for slice_index, group in normalized.groupby("ct_stack_index", sort=True):
            self.slice_gaze_history[int(slice_index)] = group.sort_values("timestamp_ms").to_dict("records")

    def _slice_history_frame(self, slice_index: int | None = None) -> pd.DataFrame:
        key = self.current_slice_index if slice_index is None else int(slice_index)
        rows = self.slice_gaze_history.get(key, [])
        return pd.DataFrame(rows)

    def _rebuild_slice_states(self, samples: pd.DataFrame) -> None:
        preserved = {
            int(index): {
                "completed_visits": int(state.get("completed_visits", 0) or 0),
                "cue_mature": bool(state.get("cue_mature", False)),
            }
            for index, state in self.slice_state_store.items()
        }
        self.slice_state_store.clear()
        normalized = self._ensure_ct_stack_index(samples) if not samples.empty else pd.DataFrame()
        observed_slices = sorted(set(self.slice_visit_counts) | set(self.slice_gaze_history))
        for slice_index in observed_slices:
            slice_rows = normalized[normalized["ct_stack_index"] == int(slice_index)].copy() if not normalized.empty else pd.DataFrame()
            valid_rows = slice_rows[(slice_rows.get("is_valid") == True) & (slice_rows.get("is_outside_ct") == False)] if not slice_rows.empty else pd.DataFrame()  # noqa: E712
            valid_ms = float(len(valid_rows) * 16.667) if not valid_rows.empty else 0.0
            dwell_ms = float(len(slice_rows) * 16.667)
            outside_ct_ratio = float(slice_rows.get("is_outside_ct", pd.Series(dtype=bool)).mean()) if not slice_rows.empty else 0.0
            visits = self.slice_visit_counts.get(int(slice_index), 0)
            if valid_ms <= 0:
                status = "Unvisited"
            elif valid_ms >= float(self.attention_thresholds.minimum_valid_time_on_roi_slice_ms):
                status = "Reviewed"
            else:
                status = "Briefly viewed"
            previous = preserved.get(int(slice_index), {})
            self.slice_state_store[int(slice_index)] = {
                "status": status,
                "valid_ms": valid_ms,
                "dwell_ms": dwell_ms,
                "outside_ct_ratio": outside_ct_ratio,
                "visits": visits,
                "completed_visits": int(previous.get("completed_visits", 0) or 0),
                "cue_mature": bool(previous.get("cue_mature", False)),
            }

    def _finalize_slice_review_opportunity(self, slice_index: int) -> None:
        state = self.slice_state_store.get(int(slice_index))
        if state is None:
            return
        if str(state.get("status", "")) in {"Briefly viewed", "Reviewed"}:
            state["completed_visits"] = int(state.get("completed_visits", 0) or 0) + 1
            state["cue_mature"] = True
            self.slice_state_store[int(slice_index)] = state
        self._evaluate_slice_rois(int(slice_index), source_label="slice_exit")

    def _evaluate_slice_rois(self, slice_index: int, source_label: str) -> None:
        if self.case_model is None:
            return
        samples = self._slice_history_frame(slice_index)
        if samples.empty:
            return
        roi_rows = self.case_model.rois_on_slice(slice_index).copy()
        if roi_rows.empty:
            return
        current_slice = self.current_slice_index
        self.current_slice_index = int(slice_index)
        try:
            self._update_roi_states_from_samples(samples, source_label=source_label)
        finally:
            self.current_slice_index = current_slice

    def _set_all_overlay_preferences(self, enabled: bool) -> None:
        for key in self.overlay_preferences:
            self.overlay_preferences[key] = bool(enabled)

    def _sync_case_model_roi_states(self) -> None:
        if self.case_model is None or not self.roi_state_store:
            return
        status_lookup = {
            str(roi_id): str(state.get("mapped_review_status", "not_evaluated"))
            for roi_id, state in self.roi_state_store.items()
        }
        cue_state_lookup = {
            str(roi_id): str(state.get("roi_cue_state", "none"))
            for roi_id, state in self.roi_state_store.items()
        }
        visibility_lookup = {
            str(roi_id): bool(state.get("roi_overlay_visible", True))
            for roi_id, state in self.roi_state_store.items()
        }
        self.case_model.review_targets["rule_attention_status"] = self.case_model.review_targets["roi_id"].astype(str).map(status_lookup).fillna("not_evaluated")
        self.case_model.review_targets["roi_cue_state"] = self.case_model.review_targets["roi_id"].astype(str).map(cue_state_lookup).fillna("none")
        self.case_model.review_targets["roi_overlay_visible"] = self.case_model.review_targets["roi_id"].astype(str).map(visibility_lookup).fillna(True)

    def _initialize_case_state_store(self) -> None:
        self.roi_state_store.clear()
        if self.case_model is None:
            return
        geometry = self.case_model.review_targets.copy().sort_values(["ct_stack_index", "roi_id"]).reset_index(drop=True)
        geometry["base_roi_id"] = geometry["roi_id"].astype(str).map(base_roi_id)
        for index, row in enumerate(geometry.itertuples(index=False), start=1):
            row_dict = row._asdict()
            roi_id = str(row_dict["roi_id"])
            feature_match = pd.DataFrame()
            if self.case_model is not None and not self.case_model.features.empty:
                feature_rows = self.case_model.features.copy()
                representative_raw = str(row_dict.get("representative_raw_roi_id", ""))
                feature_match = feature_rows[feature_rows["roi_id"].astype(str) == representative_raw]
            session_id = str(feature_match.iloc[0].get("session_id", "")) if not feature_match.empty else ""
            hidden_behavior_label = str(feature_match.iloc[0].get("hidden_behavior_label", "")) if not feature_match.empty else str(row_dict.get("hidden_behavior_label", ""))
            self.roi_state_store[roi_id] = {
                "roi_id": roi_id,
                "display_index": f"ROI {index:02d}",
                "base_roi_id": str(row_dict["base_roi_id"]),
                "ct_stack_index": int(float(row_dict.get("ct_stack_index", 0) or 0)),
                "slice_index": int(float(row_dict.get("slice_index", row_dict.get("ct_stack_index", 0)) or 0)),
                "case_id": str(self.case_selector.currentText() or ""),
                "session_id": session_id,
                "rule_attention_status": "not_evaluated",
                "mapped_review_status": "not_evaluated",
                "predicted_behavior_label": "awaiting_live_gaze",
                "prediction_confidence": 0.0,
                "review_state": "pending",
                "prediction_readiness": "COLLECTING_EVIDENCE",
                "prediction_readiness_message": "Collecting evidence...",
                "hidden_behavior_label": hidden_behavior_label,
                "time_on_roi_slice_ms": 0.0,
                "valid_gaze_time_on_roi_slice_ms": 0.0,
                "total_gaze_time_inside_roi_ms": 0.0,
                "total_gaze_time_near_roi_ms": 0.0,
                "gaze_hit_count_inside_roi": 0,
                "gaze_hit_count_near_roi": 0,
                "fixation_count_inside_roi": 0,
                "fixation_count_near_roi": 0,
                "roi_revisit_count": 0,
                "gaze_validity_ratio": 0.0,
                "background_gaze_ratio": 0.0,
                "cognitive_load_proxy": "unknown",
                "last_updated": None,
                "representative_row": row_dict,
            }

    def _queue_rows(self) -> pd.DataFrame:
        if self.case_model is None or not self.roi_state_store:
            return pd.DataFrame()
        rows = pd.DataFrame(self.roi_state_store.values())
        rows["display_review_state"] = rows["mapped_review_status"].astype(str).map(_display_review_state).fillna("Pending")
        return rows.sort_values(["ct_stack_index", "roi_id"]).reset_index(drop=True)

    def _slice_navigation_rows(self) -> pd.DataFrame:
        rows = self._queue_rows()
        if rows.empty:
            return rows
        nav_rows: list[dict[str, object]] = []
        for slice_index, group in rows.groupby("ct_stack_index", sort=True):
            statuses = group["mapped_review_status"].astype(str).tolist()
            reviewed_count = sum(1 for status in statuses if status == "reviewed")
            total_count = len(statuses)
            annotation_series = pd.to_numeric(group["annotation_instance_count"], errors="coerce").fillna(1) if "annotation_instance_count" in group.columns else pd.Series(1, index=group.index, dtype="float64")
            annotation_instances = int(annotation_series.sum())
            visited = int(slice_index) in self.slice_visit_counts or str(self.slice_state_store.get(int(slice_index), {}).get("status", "")) != "Unvisited"
            pending_count = sum(1 for status in statuses if status == "not_evaluated")
            weak_count = sum(1 for status in statuses if status == "weakly_reviewed")
            missed_count = sum(1 for status in statuses if status == "not_reviewed")
            state = self._slice_queue_status(int(slice_index), statuses, visited)
            if pending_count + weak_count + missed_count <= 0:
                continue
            nav_rows.append({
                "ct_stack_index": int(slice_index),
                "status": state,
                "roi_instances": total_count,
                "reviewed_count": reviewed_count,
                "annotation_instances": annotation_instances,
                "pending_count": pending_count,
                "weak_count": weak_count,
                "missed_count": missed_count,
            })
        if not nav_rows:
            return pd.DataFrame(columns=["ct_stack_index", "status", "roi_instances", "reviewed_count", "annotation_instances", "pending_count", "weak_count", "missed_count"])
        return pd.DataFrame(nav_rows).sort_values("ct_stack_index").reset_index(drop=True)

    def _slice_queue_status(self, slice_index: int, statuses: list[str], visited: bool) -> str:
        if slice_index == self.current_slice_index:
            return "Current"
        if statuses and all(status == "reviewed" for status in statuses):
            return "Reviewed"
        if any(status == "not_reviewed" for status in statuses):
            return "Missed"
        if any(status == "weakly_reviewed" for status in statuses):
            return "Weak evidence"
        return "Not visited" if not visited else "Not visited"

    def _refresh_roi_worklist(self, select_first: bool = False) -> None:
        rows = self._slice_navigation_rows()
        self._syncing_worklist = True
        try:
            self.slice_worklist.clear()
            target_row = -1
            for index, row in rows.iterrows():
                item = QListWidgetItem(_slice_worklist_label(row, False))
                item.setData(Qt.ItemDataRole.UserRole, int(row.get("ct_stack_index", -1)))
                item.setToolTip("Navigate to ROI-bearing slice")
                self.slice_worklist.addItem(item)
                if int(row.get("ct_stack_index", -1)) == self.current_slice_index:
                    target_row = index
            if target_row >= 0:
                self.slice_worklist.setCurrentRow(target_row)
            elif select_first and rows.shape[0] > 0:
                self.slice_worklist.setCurrentRow(0)
            else:
                self.slice_worklist.clearSelection()
        finally:
            self._syncing_worklist = False
        self.slice_worklist.setEnabled(True)

    def _refresh_roi_queue(self) -> None:
        rows = self._filtered_queue_rows()
        concealed = self._conceal_targets()
        selected_roi_id = "" if self.selected_roi is None else str(self.selected_roi.get("roi_id", ""))
        self._syncing_worklist = True
        try:
            self.roi_queue_list.clear()
            current_slice = None
            selected_row_index = -1
            for _, row in rows.iterrows():
                row_slice = int(row.get("ct_stack_index", -1))
                if current_slice != row_slice:
                    current_slice = row_slice
                    header = QListWidgetItem(f"Slice {_display_slice(current_slice, one_based=True)}")
                    header.setFlags(Qt.ItemFlag.NoItemFlags)
                    self.roi_queue_list.addItem(header)
                item = QListWidgetItem(_roi_queue_item_label(row, concealed, row_slice == self.current_slice_index))
                item.setData(Qt.ItemDataRole.UserRole, str(row.get("roi_id", "")))
                item.setToolTip("Navigate to ROI and inspect evidence" if not concealed else "ROI queue hidden during silent review")
                self.roi_queue_list.addItem(item)
                if selected_roi_id and str(row.get("roi_id", "")) == selected_roi_id:
                    selected_row_index = self.roi_queue_list.count() - 1
            if not concealed and selected_row_index >= 0:
                self.roi_queue_list.setCurrentRow(selected_row_index)
            elif concealed:
                self.roi_queue_list.clearSelection()
        finally:
            self._syncing_worklist = False
        self.roi_queue_list.setEnabled(not concealed)

    def _filtered_queue_rows(self) -> pd.DataFrame:
        rows = self._queue_rows()
        if rows.empty:
            return rows
        filter_text = self.roi_queue_filter.currentText()
        if filter_text == "All":
            return rows
        state_map = {
            "Not evaluated": "Pending",
            "Weak evidence": "Weak",
            "Reviewed": "Sufficient",
            "Missed": "Insufficient",
        }
        target = state_map.get(filter_text, "")
        return rows[rows["display_review_state"].astype(str) == target].reset_index(drop=True)

    def _refresh_case_progress(self, rows: pd.DataFrame) -> None:
        current_slice_rows = self._current_slice_state_rows()
        hidden = self._results_hidden()
        self.case_status_value.setText(str(self.total_slice_count))
        self.progress_total.setText(str(len(rows)))
        self.current_slice_target_count.setText(str(len(current_slice_rows)))
        target_bearing_slices = self._target_bearing_slice_count()
        viewed_slices = self._roi_slices_viewed_count()
        reviewed_slices = self._reviewed_roi_slices_count()
        remaining_slices = max(0, target_bearing_slices - reviewed_slices)
        completed_slices = self._completed_roi_slices_count()
        completion = 0.0 if target_bearing_slices <= 0 else reviewed_slices / target_bearing_slices
        self.session_progress_value.setText(f"{completion * 100:.1f}%")
        self.case_completed_slices_value.setText(str(completed_slices))
        if hidden:
            for label in (
                self.progress_sufficient,
                self.progress_weak,
                self.progress_insufficient,
                self.current_slice_reviewed_count,
                self.current_slice_weak_count,
                self.current_slice_missed_count,
            ):
                label.setText("—")
            self.progress_pending.setText(str(len(rows)))
            self.current_slice_pending_count.setText(str(len(current_slice_rows)))
            self.case_target_bearing_slices.setText(str(target_bearing_slices))
            self.case_roi_slices_viewed.setText(str(reviewed_slices))
            self.case_queued_count.setText(str(remaining_slices))
            return
        counts = _coverage_counts(rows)
        current_counts = _coverage_counts(current_slice_rows)
        self.progress_sufficient.setText(str(counts["Reviewed"]))
        self.progress_weak.setText(str(counts["Weakly reviewed"]))
        self.progress_insufficient.setText(str(counts["Missed"]))
        self.progress_pending.setText(str(counts["Not evaluated"]))
        self.current_slice_reviewed_count.setText(str(current_counts["Reviewed"]))
        self.current_slice_weak_count.setText(str(current_counts["Weakly reviewed"]))
        self.current_slice_missed_count.setText(str(current_counts["Missed"]))
        self.current_slice_pending_count.setText(str(current_counts["Not evaluated"]))
        self.case_target_bearing_slices.setText(str(target_bearing_slices))
        self.case_roi_slices_viewed.setText(str(reviewed_slices))
        self.case_queued_count.setText(str(remaining_slices))

    def _current_slice_state_rows(self) -> pd.DataFrame:
        rows = self._queue_rows()
        if rows.empty:
            return rows
        return rows[pd.to_numeric(rows["ct_stack_index"], errors="coerce").fillna(-1).astype(int) == self.current_slice_index].reset_index(drop=True)

    def _roi_slices_viewed_count(self) -> int:
        if self.case_model is None:
            return 0
        geometry = self.case_model.roi_geometry.copy()
        roi_slices = set(pd.to_numeric(geometry["ct_stack_index"], errors="coerce").dropna().astype(int).tolist())
        return _count_roi_slices_viewed(roi_slices, self.slice_state_store)

    def _target_bearing_slice_count(self) -> int:
        if self.case_model is None:
            return 0
        geometry = self.case_model.roi_geometry.copy()
        return len(set(pd.to_numeric(geometry["ct_stack_index"], errors="coerce").dropna().astype(int).tolist()))

    def _completed_roi_slices_count(self) -> int:
        rows = self._queue_rows()
        if rows.empty:
            return 0
        count = 0
        for _, group in rows.groupby("ct_stack_index", sort=True):
            states = set(group["mapped_review_status"].astype(str))
            if "not_evaluated" not in states:
                count += 1
        return count

    def _reviewed_roi_slices_count(self) -> int:
        rows = self._queue_rows()
        if rows.empty:
            return 0
        count = 0
        for _, group in rows.groupby("ct_stack_index", sort=True):
            states = set(group["mapped_review_status"].astype(str))
            if states and states == {"reviewed"}:
                count += 1
        return count

    def coverage_debug_report(self) -> str:
        rows = self._queue_rows()
        current_rows = self._current_slice_state_rows()
        geometry = self.case_model.roi_geometry.copy() if self.case_model is not None else pd.DataFrame()
        geometry = geometry.copy()
        if not geometry.empty:
            geometry["base_roi_id"] = geometry["roi_id"].astype(str).map(base_roi_id)
        unique_roi_bearing_slices = sorted(set(pd.to_numeric(geometry.get("ct_stack_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()))
        current_geometry = geometry[pd.to_numeric(geometry.get("ct_stack_index", pd.Series(dtype=float)), errors="coerce").fillna(-1).astype(int) == self.current_slice_index] if not geometry.empty else pd.DataFrame()
        lines = [
            f"Case ID: {self.case_selector.currentText() or '-'}",
            f"Raw ROI geometry rows: {len(geometry)}",
            f"Unique roi_id count: {int(geometry['roi_id'].astype(str).nunique()) if not geometry.empty else 0}",
            f"Unique target count: {int(geometry['base_roi_id'].astype(str).nunique()) if not geometry.empty else 0}",
            f"ROI-bearing slices: {len(unique_roi_bearing_slices)}",
            f"Current slice: {self.current_slice_index + 1}",
            f"Raw ROI rows on current slice: {len(current_geometry)}",
            f"Unique roi_id on current slice: {int(current_geometry['roi_id'].astype(str).nunique()) if not current_geometry.empty else 0}",
            f"Unique target IDs on current slice: {int(current_geometry['base_roi_id'].astype(str).nunique()) if not current_geometry.empty else 0}",
        ]
        if not geometry.empty:
            grouped = geometry.sort_values(["base_roi_id", "ct_stack_index"]).groupby("base_roi_id", sort=False)
            for display_index, (base_id, group) in enumerate(grouped, start=1):
                lines.append(
                    f"ROI {display_index:02d} | target_id={base_id} | geometry rows={len(group)} | "
                    f"slices={group['ct_stack_index'].astype(int).nunique()} | range={group['ct_stack_index'].astype(int).min()+1}-{group['ct_stack_index'].astype(int).max()+1}"
                )
        for _, row in rows.iterrows():
            lines.append(
                f"{row.get('display_index')} | base_target_id={row.get('base_roi_id')} | internal={row.get('roi_id')} | slice={_display_slice(row.get('ct_stack_index'), one_based=True)} | "
                f"canonical={row.get('mapped_review_status')} | model={row.get('predicted_behavior_label')} | rule={row.get('rule_attention_status')}"
            )
        lines.append(f"Current Slice Coverage: {_coverage_counts(current_rows)}")
        lines.append(f"Case Coverage: {_coverage_counts(rows)}")
        return "\n".join(lines)

    def _slice_worklist_selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._syncing_worklist or current is None:
            return
        slice_index = current.data(Qt.ItemDataRole.UserRole)
        if slice_index is not None:
            self.set_current_slice(int(slice_index), record_visit=False)

    def _roi_queue_selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._syncing_worklist or current is None or self._conceal_targets():
            return
        roi_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        if not roi_id:
            return
        rows = self._queue_rows()
        match = rows[rows["roi_id"].astype(str) == roi_id]
        if match.empty:
            return
        row = self._hydrate_selected_row(self._row_with_representative(match.iloc[0].copy()))
        self.current_session_id = str(row.get("session_id", "") or self.current_session_id)
        self.selected_roi = row
        self.viewer.set_selected_roi(row)
        self.set_current_slice(int(float(row.get("ct_stack_index", self.current_slice_index) or 0)), record_visit=False)
        self._refresh_current_slice_roi_selector()
        self._refresh_review_evidence()

    def _load_current_selection(self, index: int) -> None:
        rows = self._queue_rows()
        if not 0 <= int(index) < len(rows):
            return
        row = self._hydrate_selected_row(self._row_with_representative(rows.iloc[int(index)].copy()))
        self.current_session_id = str(row.get("session_id", "") or self.current_session_id)
        self.selected_roi = row
        self.viewer.set_selected_roi(row)
        self._reload_synthetic_session_history()
        if not self._conceal_targets():
            self.set_current_slice(int(float(row.get("ct_stack_index", self.current_slice_index) or 0)), record_visit=False)
        self._refresh_overlay_policy()
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def _refresh_current_slice_roi_selector(self) -> None:
        rows = self._current_slice_state_rows()
        current_roi_id = "" if self.selected_roi is None else str(self.selected_roi.get("roi_id", ""))
        if current_roi_id and int(float(self.selected_roi.get("ct_stack_index", -1) or -1)) != self.current_slice_index:
            current_roi_id = ""
            self.selected_roi = None
            self.viewer.clear_selection()
        with QSignalBlocker(self.current_slice_roi_selector):
            self.current_slice_roi_selector.clear()
            self.current_slice_roi_selector.addItem("Select ROI on this slice", "")
            for _, row in rows.iterrows():
                annotation_suffix = ""
                if int(row.get("annotation_instance_count", 1) or 1) > 1:
                    annotation_suffix = f" ({int(row.get('annotation_instance_count', 1))} annotations)"
                label = f"{row.get('display_index', row.get('roi_id'))}{annotation_suffix} - {_display_attention_status(row.get('mapped_review_status', 'not_evaluated'))}"
                self.current_slice_roi_selector.addItem(label, str(row.get("roi_id", "")))
            if current_roi_id:
                index = self.current_slice_roi_selector.findData(current_roi_id)
                self.current_slice_roi_selector.setCurrentIndex(index if index >= 0 else 0)
            else:
                self.current_slice_roi_selector.setCurrentIndex(0)
        self.current_slice_roi_selector.setEnabled(not rows.empty and not self._conceal_targets())

    def _auto_select_current_slice_roi(self) -> None:
        if self._conceal_targets() or self.case_model is None:
            return
        rows = self._current_slice_state_rows()
        if rows.empty:
            self.selected_roi = None
            self.viewer.clear_selection()
            return
        if self.selected_roi is not None and int(float(self.selected_roi.get("ct_stack_index", -1) or -1)) == self.current_slice_index:
            return
        unresolved = rows[rows["mapped_review_status"].astype(str) != "reviewed"].reset_index(drop=True)
        chosen = unresolved.iloc[0] if not unresolved.empty else rows.iloc[0]
        self.selected_roi = self._hydrate_selected_row(self._row_with_representative(chosen.copy()))
        self.viewer.set_selected_roi(self.selected_roi)

    def _current_slice_roi_changed(self, index: int) -> None:
        if self._syncing_worklist or index < 0:
            return
        roi_id = str(self.current_slice_roi_selector.itemData(index) or "")
        if not roi_id:
            self.selected_roi = None
            self.viewer.clear_selection()
            self._refresh_review_evidence()
            return
        rows = self._queue_rows()
        match = rows[rows["roi_id"].astype(str) == roi_id]
        if match.empty:
            return
        self.selected_roi = self._hydrate_selected_row(self._row_with_representative(match.iloc[0].copy()))
        self.viewer.set_selected_roi(self.selected_roi)
        self._refresh_review_evidence()

    def _hydrate_selected_row(self, row: pd.Series) -> pd.Series:
        if self.current_source == "Synthetic Replay":
            enriched = enrich_case_row(row.copy(), self.data)
            if "review_state" not in enriched or not str(enriched.get("review_state", "")).strip():
                enriched["review_state"] = row.get("review_state", "not_evaluated")
            return enriched
        if "mapped_review_status" not in row:
            row["mapped_review_status"] = "not_evaluated"
        return row

    def _row_with_representative(self, row: pd.Series) -> pd.Series:
        representative = row.get("representative_row", {})
        if isinstance(representative, dict):
            for key, value in representative.items():
                if key not in row or pd.isna(row.get(key)):
                    row[key] = value
        return row

    def _related_target_extent_text(self, row: pd.Series) -> str:
        if self.case_model is None:
            return ""
        base_id = str(row.get("base_roi_id", ""))
        geometry = self.case_model.roi_geometry.copy()
        geometry["base_roi_id"] = geometry["roi_id"].astype(str).map(base_roi_id)
        group = geometry[geometry["base_roi_id"].astype(str) == base_id]
        if group.empty:
            return ""
        min_slice = int(pd.to_numeric(group["ct_stack_index"], errors="coerce").dropna().astype(int).min()) + 1
        max_slice = int(pd.to_numeric(group["ct_stack_index"], errors="coerce").dropna().astype(int).max()) + 1
        return f" | Extent S{min_slice}-S{max_slice}" if min_slice != max_slice else f" | Extent S{min_slice}"

    def set_current_slice(self, index: int, record_visit: bool = True, clear_playback: bool = False) -> None:
        if self.case_model is None:
            return
        started = time.perf_counter()
        clamped = max(0, min(int(index), self.case_model.total_slices - 1))
        previous_slice = self.current_slice_index
        changed = clamped != self.current_slice_index
        if changed and self.session_state == SESSION_ACTIVE:
            self._finalize_slice_review_opportunity(previous_slice)
        self.current_slice_index = clamped
        if self.selected_roi is not None and int(float(self.selected_roi.get("ct_stack_index", -1) or -1)) != clamped:
            self.selected_roi = None
            self.viewer.clear_selection()
        if clear_playback:
            self.viewer.clear_playback_sample()
        self.viewer.set_current_slice(clamped)
        self.viewer.set_overlay_samples(self._current_overlay_samples())
        if changed and record_visit and self.session_state == SESSION_ACTIVE:
            self.slice_visit_counts[clamped] = self.slice_visit_counts.get(clamped, 0) + 1
        self._refresh_slice_ui()
        self._schedule_deferred_slice_update()
        if self.current_source == "Tobii Live":
            self._refresh_session_bar()
        logging.getLogger("medgazear_final").debug("Slice change to %s took %.2fms", clamped, (time.perf_counter() - started) * 1000.0)

    def _schedule_deferred_slice_update(self) -> None:
        self._deferred_slice_update_index = self.current_slice_index
        self.slice_update_timer.start()

    def _run_deferred_slice_update(self) -> None:
        if self._deferred_slice_update_index is None or self._deferred_slice_update_index != self.current_slice_index:
            return
        self._refresh_current_slice_roi_selector()
        self._auto_select_current_slice_roi()
        self._refresh_current_slice_roi_selector()
        self._refresh_overlay_policy()
        self._refresh_case_progress(self._queue_rows())
        self._refresh_review_evidence()
        self._refresh_research_diagnostics()

    def _start_or_resume_session(self) -> None:
        if self.current_source == "Synthetic Replay":
            self._synthetic_session_started()
            return
        self.start_live_gaze()

    def _pause_session(self) -> None:
        if self.current_source == "Synthetic Replay":
            self._synthetic_session_stopped()
            return
        self.pause_live_gaze()

    def _end_session(self) -> None:
        if self.current_source == "Synthetic Replay":
            self._synthetic_session_reset()
            return
        self.stop_live_gaze()

    def detect_tobii(self) -> None:
        self.tobii_source.find_devices()
        if self.current_source == "Tobii Live" and self.tobii_source.get_status_payload().get("device_connected"):
            self._run_tobii_preflight()
        self._refresh_source_ui()
        self._refresh_session_bar()

    def _run_tobii_preflight(self, duration_s: float = 2.0) -> None:
        samples: list[dict[str, object]] = []
        self.mapping_failure_diagnostics.clear()
        self.tobii_source.start_stream(samples.append)
        if not self.tobii_source.streaming:
            self.tobii_preflight_result = {"status": "TRACKING_NOT_READY", "message": self.tobii_source.error or "Unable to start preflight"}
            return
        time.sleep(duration_s)
        self.tobii_source.stop_stream()
        self.tobii_preflight_result = tracking_preflight_summary(pd.DataFrame(samples))

    def _source_changed(self, source: str) -> None:
        self.current_source = source
        if source != "Tobii Live":
            self.tobii_source.stop_stream()
        self.session_state = SESSION_IDLE
        self.live_samples.clear()
        self.slice_gaze_history.clear()
        self.viewer.clear_playback_sample()
        if source == "Synthetic Replay":
            self.current_session_id = self._default_session_id_for_case()
            self._reload_synthetic_session_history()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_source_ui()
        self._refresh_overlay_policy()
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def _mode_changed(self, mode: str) -> None:
        self.review_mode.mode = mode
        if self._conceal_targets():
            self.selected_roi = None
            self.viewer.clear_selection()
            self.slice_worklist.clearSelection()
        self._refresh_overlay_policy()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_case_progress(self._queue_rows())
        self._refresh_review_evidence()
        self._refresh_source_ui()

    def _window_preset_changed(self, preset_name: str) -> None:
        center, width = window_preset(preset_name)
        self.viewer.set_window_preset(center, width)

    def _synthetic_session_started(self) -> None:
        if self.current_source != "Synthetic Replay":
            return
        self.session_state = SESSION_ACTIVE
        self.slice_visit_counts.clear()
        self.slice_visit_counts[self.current_slice_index] = 1
        self._reload_synthetic_session_history()
        self._refresh_overlay_policy()
        self._refresh_case_progress(self._queue_rows())
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def _synthetic_session_stopped(self) -> None:
        if self.current_source != "Synthetic Replay":
            return
        if self.session_state == SESSION_ACTIVE:
            self._finalize_slice_review_opportunity(self.current_slice_index)
            self.session_state = SESSION_STOPPED
        self._refresh_overlay_policy()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_case_progress(self._queue_rows())
        self._refresh_review_evidence()

    def _synthetic_session_reset(self) -> None:
        if self.current_source != "Synthetic Replay":
            return
        self.session_state = SESSION_IDLE
        self.slice_gaze_history.clear()
        self.slice_visit_counts.clear()
        self.slice_state_store.clear()
        self.viewer.clear_playback_sample()
        self._reload_synthetic_session_history()
        self._refresh_overlay_policy()
        self._refresh_case_progress(self._queue_rows())
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def start_live_gaze(self) -> None:
        if self.tobii_preflight_result is None or self.tobii_preflight_result.get("status") != "READY_FOR_SESSION":
            self._run_tobii_preflight()
            if self.tobii_preflight_result is None or self.tobii_preflight_result.get("status") != "READY_FOR_SESSION":
                self._refresh_source_ui()
                self._refresh_session_bar()
                return
        self.session_state = SESSION_ACTIVE
        if not self.live_samples:
            self.slice_gaze_history.clear()
            self.slice_visit_counts.clear()
            self.slice_visit_counts[self.current_slice_index] = 1
        elif self.current_slice_index not in self.slice_visit_counts:
            self.slice_visit_counts[self.current_slice_index] = 1
        self._pending_live_samples.clear()
        self.mapping_failure_diagnostics.clear()
        self.tobii_source.find_devices()
        self.tobii_source.start_stream(self._live_gaze_sample_received)
        self._update_live_button_state(self.tobii_source.streaming)
        if not self.tobii_source.streaming:
            self.session_state = SESSION_IDLE
        self._refresh_overlay_policy()
        self._refresh_case_progress(self._queue_rows())
        self._refresh_source_ui()
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def pause_live_gaze(self) -> None:
        if self.current_source != "Tobii Live":
            return
        if self.tobii_source.streaming:
            self.tobii_source.stop_stream()
        self.live_update_timer.stop()
        if self.session_state == SESSION_ACTIVE:
            self._finalize_slice_review_opportunity(self.current_slice_index)
        self.session_state = SESSION_IDLE
        self._update_live_button_state(False)
        self._refresh_source_ui()
        self._refresh_session_bar()
        self._refresh_review_evidence()

    def stop_live_gaze(self) -> None:
        was_streaming = self.tobii_source.streaming
        self.tobii_source.stop_stream()
        self.live_update_timer.stop()
        self._pending_live_samples.clear()
        self._update_live_button_state(False)
        if was_streaming:
            self._finalize_slice_review_opportunity(self.current_slice_index)
        if was_streaming and self.live_samples:
            self.session_state = SESSION_STOPPED
            self.save_session(prompt=False)
        elif self.session_state == SESSION_ACTIVE:
            self.session_state = SESSION_STOPPED
        self._refresh_overlay_policy()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_case_progress(self._queue_rows())
        self._refresh_source_ui()
        self._refresh_review_evidence()
        self._refresh_session_bar()

    def _live_gaze_sample_received(self, sample: dict[str, object]) -> None:
        self.live_sample_received.emit(sample)

    def _handle_live_sample_on_ui_thread(self, sample: object) -> None:
        if isinstance(sample, dict):
            self._pending_live_samples.append(sample)
            if not self.live_update_timer.isActive():
                self.live_update_timer.start()

    def _process_pending_live_sample(self) -> None:
        if not self._pending_live_samples:
            self.live_update_timer.stop()
            return
        pending = list(self._pending_live_samples)
        self._pending_live_samples.clear()
        for sample in pending:
            sample["session_id"] = sample.get("session_id") or "tobii_live"
            sample["case_id"] = sample.get("case_id") or (self.case_selector.currentText() or "")
            sample["slice_index"] = self.current_slice_index
            sample["ct_stack_index"] = self.current_slice_index
            sample["source"] = sample.get("source") or sample.get("source_type") or "live_tobii"
            if self._last_mapping_diagnostic is not None:
                sample.update({
                    "viewer_local_x": self._last_mapping_diagnostic.get("viewer_local_x"),
                    "viewer_local_y": self._last_mapping_diagnostic.get("viewer_local_y"),
                    "image_rect": self._last_mapping_diagnostic.get("image_rect"),
                    "mapping_failure_reason": self._last_mapping_diagnostic.get("failure_reason", ""),
                })
            self.live_samples.append(sample)
        self.live_samples = self.live_samples[-1200:]
        self._rebuild_slice_gaze_history(pd.DataFrame(self.live_samples))
        self._rebuild_slice_states(pd.DataFrame(self.live_samples))
        self._update_roi_states_from_samples(pd.DataFrame(self.live_samples), source_label="live")
        self.viewer.set_overlay_samples(self._current_overlay_samples())
        self._refresh_source_ui()
        self._refresh_session_bar()
        if not self._pending_live_samples:
            self.live_update_timer.stop()

    def _update_roi_states_from_samples(self, samples: pd.DataFrame, source_label: str) -> None:
        if self.case_model is None or samples.empty:
            return
        start = time.perf_counter()
        roi_rows = self.case_model.rois_on_slice(self.current_slice_index).copy()
        roi_rows["base_roi_id"] = roi_rows["roi_id"].astype(str).map(base_roi_id)
        for _, roi_row in roi_rows.sort_values(["ct_stack_index", "roi_id"]).iterrows():
            roi_id = str(roi_row.get("roi_id", ""))
            feature_row = self._feature_row_for_roi(roi_row, samples)
            status_row = classify_feature_row(feature_row, self.attention_thresholds)
            canonical_status = str(status_row["rule_attention_status"])
            prediction = predict_behavior(pd.Series(feature_row), self.data)
            mapped_review_status = _canonical_display_state(prediction.get("mapped_review_status", "not_evaluated"), canonical_status)
            state = self.roi_state_store.get(roi_id, {}).copy()
            cue_state, overlay_visible = self._cue_presentation_for_roi(roi_row, feature_row, prediction, mapped_review_status)
            state.update(feature_row)
            state.update({
                "roi_id": roi_id,
                "base_roi_id": str(roi_row.get("base_roi_id", state.get("base_roi_id", ""))),
                "ct_stack_index": int(float(roi_row.get("ct_stack_index", 0) or 0)),
                "slice_index": int(float(roi_row.get("slice_index", roi_row.get("ct_stack_index", 0)) or 0)),
                "rule_attention_status": canonical_status,
                "mapped_review_status": str(mapped_review_status),
                "predicted_behavior_label": prediction.get("label", "unavailable"),
                "prediction_confidence": prediction.get("confidence", 0.0),
                "class_probabilities": prediction.get("class_probabilities", {}),
                "prediction_readiness": prediction.get("readiness", "READY"),
                "prediction_readiness_message": prediction.get("readiness_message", "Ready"),
                "review_state": canonical_status,
                "roi_cue_state": cue_state,
                "roi_overlay_visible": overlay_visible,
                "background_gaze_ratio": float(feature_row.get("background_gaze_ratio", 0.0)),
                "cognitive_load_proxy": "high_load_proxy" if float(feature_row.get("background_gaze_ratio", 0.0)) > 0.7 else "medium_load_proxy" if float(feature_row.get("background_gaze_ratio", 0.0)) > 0.35 else "low_load_proxy",
                "last_updated": time.time(),
                "representative_row": state.get("representative_row", roi_row.to_dict()),
            })
            explanation = self.explanation_service.explain_live_row(state)
            if explanation is not None:
                state["explanation"] = explanation
                self._record_explanation_checkpoint(roi_id, feature_row, explanation)
            self.roi_state_store[roi_id] = state
            if int(feature_row.get("gaze_hit_count_inside_roi", 0)) > 0:
                logging.getLogger("medgazear_final").info(
                    "%s | %s | S%s | x=%s y=%s | inside=%s | dwell=%sms | hits=%s | fix=%s | revisits=%s | state=%s",
                    state.get("display_index", roi_id),
                    source_label,
                    int(float(roi_row.get("ct_stack_index", 0) or 0)) + 1,
                    _fmt_value(feature_row.get("last_image_x")),
                    _fmt_value(feature_row.get("last_image_y")),
                    True,
                    _fmt_value(feature_row.get("total_gaze_time_inside_roi_ms")),
                    _fmt_value(feature_row.get("gaze_hit_count_inside_roi")),
                    _fmt_value(feature_row.get("fixation_count_inside_roi")),
                    _fmt_value(feature_row.get("roi_revisit_count")),
                    _display_review_state(str(mapped_review_status)),
                )
        if self.selected_roi is not None:
            selected_id = str(self.selected_roi.get("roi_id", ""))
            if selected_id in self.roi_state_store:
                self.selected_roi = self._row_with_representative(pd.Series(self.roi_state_store[selected_id]).copy())
        logging.getLogger("medgazear_final").debug("ROI state update %.2fms", (time.perf_counter() - start) * 1000.0)
        self._sync_case_model_roi_states()
        self._refresh_roi_worklist(select_first=False)
        self._refresh_case_progress(self._queue_rows())
        self._refresh_overlay_policy()
        self._refresh_review_evidence()

    def _record_explanation_checkpoint(self, roi_id: str, feature_row: Mapping[str, object], explanation) -> None:
        timestamp = float(feature_row.get("time_on_roi_slice_ms", 0.0) or 0.0)
        checkpoints = self.explanation_checkpoints.setdefault(roi_id, [])
        if checkpoints and timestamp - float(checkpoints[-1].get("timestamp_ms", 0.0) or 0.0) < 500.0:
            return
        top_feature = explanation.top_features[0]["label"] if explanation.top_features else ""
        checkpoints.append(
            {
                "timestamp_ms": timestamp,
                "predicted_label": explanation.predicted_label,
                "confidence": explanation.predicted_probability,
                "dominant_feature": top_feature,
            }
        )
        self.explanation_checkpoints[roi_id] = checkpoints[-12:]

    def _cue_presentation_for_roi(self, roi_row: pd.Series, feature_row: dict[str, object], prediction: dict[str, object], mapped_review_status: object) -> tuple[str, bool]:
        roi_id = str(roi_row.get("roi_id", ""))
        is_selected = self.selected_roi is not None and str(self.selected_roi.get("roi_id", "")) == roi_id
        if self.review_mode.mode == "Silent" and self.session_state == SESSION_ACTIVE:
            return ("selected", True) if is_selected else ("none", False)
        if self.session_state != SESSION_ACTIVE:
            return ("selected", True) if is_selected else ("none", True)
        if self.current_source != "Synthetic Replay" and self.current_source != "Tobii Live":
            return ("selected", True) if is_selected else ("none", True)
        if not self._feature_evidence_ready(feature_row, prediction):
            return ("selected", True) if is_selected else ("none", True)
        slice_index = int(float(roi_row.get("ct_stack_index", 0) or 0))
        slice_state = self.slice_state_store.get(slice_index, {})
        if not bool(slice_state.get("cue_mature", False)):
            return ("selected", True) if is_selected else ("none", True)
        display_state = _display_review_state(str(mapped_review_status))
        if display_state == "Weak":
            return ("selected", True) if is_selected else ("weak", True)
        if display_state == "Insufficient":
            return ("selected", True) if is_selected else ("missed", True)
        return ("selected", True) if is_selected else ("none", False)

    def _feature_evidence_ready(self, feature_row: dict[str, object], prediction: dict[str, object]) -> bool:
        sample_count = int(_safe_float(feature_row.get("_sample_count", 0)))
        valid_time = _safe_float(feature_row.get("valid_gaze_time_on_roi_slice_ms"))
        validity_ratio = _safe_float(feature_row.get("gaze_validity_ratio"))
        fixation_ready = feature_row.get("_fixation_ready", True)
        readiness = str(prediction.get("readiness", ""))
        if readiness in {"COLLECTING_EVIDENCE", "INVALID_GAZE"}:
            return False
        if sample_count < 2 or valid_time <= 0 or validity_ratio <= 0:
            return False
        if fixation_ready is False:
            return False
        return True

    def _feature_row_for_roi(self, roi_row: pd.Series, samples: pd.DataFrame) -> dict[str, object]:
        roi = self._normalized_roi_row(roi_row)
        slice_samples = samples[pd.to_numeric(samples.get("ct_stack_index"), errors="coerce").fillna(-1).astype(int) == int(roi["ct_stack_index"])].copy()
        metadata = {
            "reader_id": str(self.selected_roi.get("reader_id", "reader") if self.selected_roi is not None else "reader"),
            "reader_profile": str(self.selected_roi.get("reader_profile", "unknown") if self.selected_roi is not None else "unknown"),
            "case_id": str(self.case_selector.currentText() or roi.get("patient_id", "case")),
            "hidden_behavior_label": str(self.roi_state_store.get(str(roi.get("roi_id", "")), {}).get("hidden_behavior_label", "")),
        }
        result = build_behavior_feature_row(slice_samples, roi, metadata)
        feature_row = result.row
        feature_row["roi_revisit_count"] = self.slice_visit_counts.get(self.current_slice_index, 0) - 1 if self.slice_visit_counts.get(self.current_slice_index, 0) > 0 else 0
        if slice_samples.empty:
            return self._empty_feature_row(roi)
        valid_points = slice_samples[(slice_samples.get("is_valid") == True) & (slice_samples.get("is_outside_ct") == False) & (slice_samples.get("is_ui_glance") == False)]  # noqa: E712
        last_valid = valid_points.iloc[-1] if not valid_points.empty else pd.Series({"image_x": 0.0, "image_y": 0.0})
        feature_row["last_image_x"] = float(last_valid.get("image_x", 0.0) or 0.0)
        feature_row["last_image_y"] = float(last_valid.get("image_y", 0.0) or 0.0)
        return feature_row

    def _normalized_roi_row(self, roi_row: pd.Series) -> dict[str, object]:
        roi = roi_row.to_dict()
        roi["base_roi_id"] = base_roi_id(roi.get("roi_id", ""))
        roi["bbox_width"] = float(roi.get("bbox_width") or (float(roi.get("bbox_x_max", 0) or 0) - float(roi.get("bbox_x_min", 0) or 0)))
        roi["bbox_height"] = float(roi.get("bbox_height") or (float(roi.get("bbox_y_max", 0) or 0) - float(roi.get("bbox_y_min", 0) or 0)))
        roi["slice_index"] = int(float(roi.get("slice_index", roi.get("ct_stack_index", 0)) or 0))
        roi["ct_stack_index"] = int(float(roi.get("ct_stack_index", 0) or 0))
        return roi

    def _empty_feature_row(self, roi: dict[str, object]) -> dict[str, object]:
        return {
            "session_id": "session",
            "roi_id": str(roi.get("roi_id", "")),
            "hidden_behavior_label": str(self.roi_state_store.get(str(roi.get("roi_id", "")), {}).get("hidden_behavior_label", "")),
            "gaze_validity_ratio": 0.0,
            "background_gaze_ratio": 0.0,
            "roi_revisit_count": 0,
            "last_image_x": 0.0,
            "last_image_y": 0.0,
            "total_gaze_time_inside_roi_ms": 0.0,
            "total_gaze_time_near_roi_ms": 0.0,
            "gaze_hit_count_inside_roi": 0,
            "gaze_hit_count_near_roi": 0,
            "fixation_count_inside_roi": 0,
            "fixation_count_near_roi": 0,
            "mean_fixation_duration_inside_roi_ms": 0.0,
            "max_fixation_duration_inside_roi_ms": 0.0,
            "time_to_first_roi_fixation_ms": -1.0,
            "valid_gaze_time_on_roi_slice_ms": 0.0,
            "time_on_roi_slice_ms": 0.0,
            "_sample_count": 0,
            "_valid_sample_count": 0,
            "_fixation_ready": False,
        }

    def _derive_review_state(self, valid_time_ms: float, inside_hits: int) -> str:
        if valid_time_ms < 750.0:
            return "pending"
        if inside_hits >= 5:
            return "reviewed"
        if inside_hits >= 1 or valid_time_ms < 2500.0:
            return "in_review"
        return "needs_revisit"

    def _refresh_slice_ui(self) -> None:
        if self.case_model is None:
            self.slice_meta.setText("Slice 1 of 1")
            return
        roi_rows = self.case_model.rois_on_slice(self.current_slice_index)
        visits = self.slice_visit_counts.get(self.current_slice_index, 0)
        self.slice_meta.setText(f"Slice {self.current_slice_index + 1} of {max(1, self.total_slice_count)}  |  ROI instances on slice: {len(roi_rows)}  |  Visits: {visits}")

    def _refresh_review_evidence(self) -> None:
        hidden = self._results_hidden()
        slice_state = self._current_slice_behavior()
        self.slice_behavior_status.setText(slice_state["status"] if not hidden else "Hidden during silent review")
        self.slice_behavior_meta.setText(slice_state["meta"] if not hidden else "Slice behavior hidden during silent review")
        self.slice_behavior_quality.setText(slice_state["quality"] if not hidden else "Slice quality hidden during silent review")
        current_slice_rois = self.case_model.rois_on_slice(self.current_slice_index) if self.case_model is not None else pd.DataFrame()
        if not hidden and not current_slice_rois.empty and self.selected_roi is None:
            self._auto_select_current_slice_roi()
        row = self._selected_roi_on_current_slice()
        if row is None:
            selection_text = "No ROI selected" if not hidden else "Selection hidden during silent review"
            self.selected_roi_id_value.setText(selection_text)
            self.selected_roi_slice_value.setText("No ROI on current slice" if current_slice_rois.empty and not hidden else "Select a segmentation-derived ROI on this slice to inspect its attention evidence.")
            self.selected_roi_type_value.setText("Select a segmentation-derived ROI on this slice to inspect its attention evidence.")
            self.attention_status_value.setText("Not applicable on this slice" if current_slice_rois.empty and not hidden else "Not yet evaluated")
            self.prediction_state_value.setText("Prediction hidden during silent review" if hidden else ("Not applicable on this slice" if current_slice_rois.empty else "Scanpath pattern unavailable"))
            self.prediction_confidence_value.setText("Confidence: -")
            self.attention_evidence_value.setText("" if not hidden else "Evidence hidden during silent review")
            self.attention_reason_value.setText("" if not hidden else "Explanation hidden during silent review")
            self.behavior_support_value.setText("" if not hidden else "Behavioral support hidden during silent review")
            self.cognitive_proxy_value.setVisible(False)
            self.cognitive_proxy_meta.setVisible(False)
            return
        self.selected_roi_id_value.setText(str(row.get("display_index", row.get("roi_id", "-"))))
        self.selected_roi_slice_value.setText(f"Slice {_display_slice(row.get('ct_stack_index'), one_based=True)}{self._related_target_extent_text(row)}")
        self.selected_roi_type_value.setText("ROI type: segmentation-derived region")
        if hidden:
            self.attention_status_value.setText("Hidden during silent review")
            self.prediction_state_value.setText("Prediction hidden during silent review")
            self.prediction_confidence_value.setText("Confidence: -")
            self.attention_evidence_value.setText("Evidence hidden during silent review")
            self.attention_reason_value.setText("Explanation hidden during silent review")
            self.behavior_support_value.setText("Behavioral support hidden during silent review")
            self.cognitive_proxy_value.setVisible(False)
            self.cognitive_proxy_meta.setVisible(False)
            return
        readiness = str(row.get("prediction_readiness", "READY"))
        readiness_message = str(row.get("prediction_readiness_message", "Ready"))
        attention_state = self._selected_roi_attention_state(row, readiness)
        self.attention_status_value.setText(attention_state)
        if attention_state in {"Collecting evidence", "Not yet evaluated"}:
            self.prediction_state_value.setText(_display_behavior_label(row.get("predicted_behavior_label", "awaiting_live_gaze")))
            self.prediction_confidence_value.setText("Confidence: -")
        else:
            review_status = _display_attention_status(str(row.get("mapped_review_status", "not_evaluated")))
            self.prediction_state_value.setText(_display_behavior_label(row.get("predicted_behavior_label", "awaiting_live_gaze")))
            confidence = _fmt_percent(row.get("prediction_confidence"))
            if str(row.get("predicted_behavior_label", "")) in {"", "unavailable", "awaiting_live_gaze"} or review_status == "Not yet evaluated":
                confidence = "-"
            self.prediction_confidence_value.setText(f"Confidence: {confidence}")
        self.attention_evidence_value.setText(_attention_evidence_text(row))
        self.attention_reason_value.setText(_attention_explanation_text(row, attention_state, self.attention_thresholds))
        self.behavior_support_value.setText(_behavior_pattern_text(row, readiness_message))
        score = _fmt_value(row.get("cognitive_load_proxy_score"))
        if str(row.get("cognitive_load_proxy", "unknown")) == "unknown":
            self.cognitive_proxy_value.setVisible(False)
            self.cognitive_proxy_meta.setVisible(False)
        else:
            self.cognitive_proxy_value.setVisible(True)
            self.cognitive_proxy_meta.setVisible(True)
            self.cognitive_proxy_value.setText(_display_cognitive_proxy(row.get("cognitive_load_proxy")))
            self.cognitive_proxy_meta.setText(_cognitive_basis_text(row) if score == "-" else f"{_cognitive_basis_text(row)}\nProxy score: {score}")

    def _selected_roi_attention_state(self, row: pd.Series, readiness: str) -> str:
        if str(row.get("mapped_review_status", "not_evaluated")) != "not_evaluated":
            return _display_attention_status(str(row.get("mapped_review_status", "not_evaluated")))
        if readiness in {"COLLECTING_EVIDENCE", "INVALID_GAZE", "MISSING_REQUIRED_FEATURES"}:
            valid_time = _safe_float(row.get("valid_gaze_time_on_roi_slice_ms"))
            if valid_time <= 0:
                return "Not yet evaluated"
            return "Collecting evidence"
        return "Not yet evaluated"

    def _selected_roi_on_current_slice(self) -> pd.Series | None:
        if self.selected_roi is None:
            return None
        if int(float(self.selected_roi.get("ct_stack_index", -1) or -1)) != self.current_slice_index:
            return None
        return self.selected_roi

    def _current_slice_behavior(self) -> dict[str, str]:
        entries = self.slice_visit_counts.get(self.current_slice_index, 0)
        revisits = max(0, entries - 1)
        slice_rows = self._slice_history_frame(self.current_slice_index)
        valid_rows = slice_rows[(slice_rows.get("is_valid") == True) & (slice_rows.get("is_outside_ct") == False)] if not slice_rows.empty else pd.DataFrame()  # noqa: E712
        dwell_ms = float(len(slice_rows) * 16.667) if not slice_rows.empty else 0.0
        valid_ms = float(len(valid_rows) * 16.667) if not valid_rows.empty else 0.0
        dwell_value = _fmt_seconds(dwell_ms)
        roi_rows = self._current_slice_state_rows()
        if entries <= 0 and _safe_float(valid_ms) <= 0:
            status = "Unvisited"
        elif roi_rows.empty:
            status = "In progress"
        else:
            states = roi_rows["mapped_review_status"].astype(str).tolist()
            if all(state == "reviewed" for state in states):
                status = "Reviewed"
            elif any(state == "not_reviewed" for state in states):
                status = "Missed"
            elif any(state == "weakly_reviewed" for state in states):
                status = "Weak evidence"
            else:
                status = "In progress"
        meta = f"Dwell: {dwell_value}  |  Entries: {entries}  |  Revisits: {revisits}"
        outside_ratio = float(slice_rows.get("is_outside_ct", pd.Series(dtype=bool)).mean()) if not slice_rows.empty else 0.0
        quality = f"Validity: {_fmt_percent(_ratio_from_ms(valid_ms, dwell_ms))}  |  Outside-CT: {_fmt_percent(outside_ratio)}"
        return {"status": status, "meta": meta, "quality": quality}

    def _refresh_source_ui(self) -> None:
        if self.current_source == "Synthetic Replay":
            self.tobii_status_value.setText("Synthetic replay")
            self.top_validity_value.setText("Developer replay mode")
            self.detect_button.setVisible(False)
            self.session_stack.setCurrentIndex(0)
            self.session_shell.setVisible(True)
            return
        payload = self.tobii_source.get_status_payload()
        reminder = str(payload.get("calibration_reminder", "") or "")
        self.detect_button.setToolTip(reminder)
        self.tobii_status_value.setText(self._tobii_status_text())
        self.top_validity_value.setText(self._live_validity_summary_text())
        self.detect_button.setVisible(not self.tobii_source.streaming)
        self.session_stack.setCurrentIndex(1)
        self.session_shell.setVisible(False)

    def _tobii_state_label(self) -> str:
        payload = self.tobii_source.get_status_payload()
        state = str(payload.get("device_state", ""))
        return {
            "NO_DEVICE": "No device",
            "DEVICE_FOUND": "Connected",
            "CONNECTED": "Connected",
            "STREAMING": "Tracking",
            "ERROR": "Disconnected",
        }.get(state, "Disconnected")

    def _tobii_status_text(self) -> str:
        payload = self.tobii_source.get_status_payload()
        connected = bool(payload.get("device_connected"))
        if self.tobii_source.streaming:
            if not self.live_samples:
                return "Streaming · Waiting for gaze"
            recent = pd.DataFrame(self.live_samples[-24:])
            valid_recent = recent[recent.get("is_valid", pd.Series(dtype=bool)) == True] if not recent.empty else pd.DataFrame()  # noqa: E712
            mapped_recent = valid_recent[valid_recent.get("is_outside_ct", pd.Series(dtype=bool)) == False] if not valid_recent.empty else pd.DataFrame()  # noqa: E712
            if valid_recent.empty:
                return "Low gaze validity"
            if mapped_recent.empty:
                return "Streaming · CT mapping unavailable"
            return "Streaming · Tracking good"
        if not connected:
            return self._tobii_state_label()
        if self.tobii_preflight_result is None:
            return "Connected"
        status = str(self.tobii_preflight_result.get("status", ""))
        failure_kind = str(self.tobii_preflight_result.get("failure_kind", ""))
        if status == "READY_FOR_SESSION":
            return "Connected"
        if failure_kind == "mapping_unavailable":
            return "Mapping unavailable"
        if failure_kind == "tracking_not_ready":
            return "Calibrating"
        return "Connected"

    def _live_validity_summary_text(self) -> str:
        if self.current_source != "Tobii Live":
            return "Developer replay mode"
        if not self.live_samples:
            if self.tobii_preflight_result and self.tobii_preflight_result.get("failure_kind") == "mapping_unavailable":
                return "Gaze is being received, but viewer mapping is not calibrated."
            return "Gaze tracking inactive"
        recent = pd.DataFrame(self.live_samples[-120:])
        valid_ratio = float(recent.get("is_valid", pd.Series(dtype=bool)).mean()) if not recent.empty else 0.0
        mapped_ratio = 0.0
        if not recent.empty:
            valid_recent = recent[recent.get("is_valid", pd.Series(dtype=bool)) == True]  # noqa: E712
            if not valid_recent.empty:
                mapped_ratio = float((valid_recent.get("is_outside_ct", pd.Series(dtype=bool)) == False).mean())  # noqa: E712
        if valid_ratio <= 0.5:
            return "Low gaze validity"
        return f"Valid {valid_ratio * 100:.0f}% | Samples {len(self.live_samples)} | Mapped {mapped_ratio * 100:.0f}%"

    def _update_live_button_state(self, streaming: bool) -> None:
        self.start_button.setProperty("liveState", "active" if streaming else "inactive")
        self.pause_button.setEnabled(streaming)
        self.end_button.setEnabled(streaming or bool(self.live_samples))
        for button in (self.start_button, self.pause_button, self.end_button):
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _refresh_session_bar(self) -> None:
        if self.current_source == "Synthetic Replay":
            self.timeline.set_replay_enabled(True)
            self.timeline.set_samples(self._session_samples(include_playback_limit=False))
            self.session_timer_value.setText(_format_elapsed_text(self._session_elapsed_seconds()))
            self._refresh_button_states()
            return
        self.timeline.set_replay_enabled(False)
        elapsed_text = _format_elapsed_text(self._session_elapsed_seconds())
        self.session_timer_value.setText(elapsed_text)
        self.live_session_status.setText(f"Current slice: {self.current_slice_index + 1} / {max(1, self.total_slice_count)}")
        self.live_session_metrics.setText(f"Recent slice navigation: {self._recent_slice_timeline_text()}")
        self.live_session_quality.setText(f"Valid gaze: {_fmt_percent(self._current_valid_gaze_ratio())}  |  Samples: {len(self.live_samples)}  |  Elapsed: {elapsed_text}")
        self._refresh_button_states()

    def _session_elapsed_seconds(self) -> float:
        if self.current_source == "Synthetic Replay" and self.viewer.playback_sample is not None:
            return float(self.viewer.playback_sample.get("timestamp_ms", 0) or 0) / 1000.0
        if len(self.live_samples) >= 2:
            first = float(self.live_samples[0].get("timestamp_ms", 0) or 0)
            last = float(self.live_samples[-1].get("timestamp_ms", 0) or 0)
            return max(0.0, last - first) / 1000.0
        if len(self.live_samples) == 1:
            return float(self.live_samples[0].get("timestamp_ms", 0) or 0) / 1000.0
        return 0.0

    def _recent_slice_timeline_text(self) -> str:
        if not self.slice_visit_counts:
            return "none"
        ordered = sorted(self.slice_visit_counts)
        recent = ordered[-4:]
        return " -> ".join(f"S{index + 1}" for index in recent)

    def _current_valid_gaze_ratio(self) -> float:
        if not self.live_samples:
            return 0.0
        rows = pd.DataFrame(self.live_samples[-120:])
        if rows.empty:
            return 0.0
        return float(rows.get("is_valid", pd.Series(dtype=bool)).mean())

    def _refresh_button_states(self) -> None:
        live_mode = self.current_source == "Tobii Live"
        self.start_button.setEnabled(live_mode and not self.tobii_source.streaming)
        self.pause_button.setEnabled(live_mode and self.tobii_source.streaming)
        self.end_button.setEnabled(live_mode and (self.tobii_source.streaming or bool(self.live_samples)))

    def _refresh_research_diagnostics(self) -> None:
        diagnostics = [self.coverage_debug_report(), self._current_slice_diagnostics_text()]
        diagnostics.append(self.viewer.diagnostics_text(self.screen_mode))
        diagnostics.append(f"Status: {self._tobii_status_text()}")
        diagnostics.append(f"Source: {self.current_source}")
        diagnostics.append(f"Mapping failures buffered: {len(self.mapping_failure_diagnostics)}")
        self.research_diagnostics_text.setPlainText("\n\n".join(diagnostics))

    def _current_slice_diagnostics_text(self) -> str:
        if self.case_model is None:
            return "Current slice: -"
        raw_rows = self.case_model.raw_rois_on_slice(self.current_slice_index).copy()
        targets = self.case_model.rois_on_slice(self.current_slice_index).copy()
        renderable_masks = sum(load_roi_mask(row) is not None for _, row in targets.iterrows()) if not targets.empty else 0
        lines = [
            f"Current slice: {self.current_slice_index + 1}",
            f"Raw annotation rows: {len(raw_rows)}",
            f"Unique lesion IDs: {_nunique_if_present(raw_rows, ['lesion_id', 'nodule_id', 'nodule_uid'])}",
            f"Unique review targets: {len(targets)}",
            f"Renderable masks: {renderable_masks}",
            f"Rendered contours: {len(targets)}",
            f"Selected ROI options: {max(0, self.current_slice_roi_selector.count() - 1)}",
        ]
        for _, row in raw_rows.iterrows():
            lines.append(
                " | ".join(
                    [
                        f"patient_id={row.get('patient_id', '')}",
                        f"series_uid={row.get('ct_series_instance_uid', '')}",
                        f"sop_instance_uid={row.get('ct_sop_instance_uid', '')}",
                        f"slice_index={row.get('ct_stack_index', row.get('slice_index', ''))}",
                        f"roi_id={row.get('roi_id', '')}",
                        f"lesion_id={row.get('lesion_id', row.get('nodule_id', ''))}",
                        f"annotation_id={row.get('annotation_id', row.get('reader_id', ''))}",
                        f"segmentation_object_id={row.get('seg_sop_instance_uid', '')}",
                        f"frame_number={row.get('slice_index', '')}",
                        f"bbox=({row.get('bbox_x_min', '')},{row.get('bbox_y_min', '')})-({row.get('bbox_x_max', '')},{row.get('bbox_y_max', '')})",
                        f"source_row_index={row.get('source_row_index', '')}",
                    ]
                )
            )
        for _, row in targets.iterrows():
            lines.append(
                f"target={row.get('roi_id', '')} | annotations={row.get('annotation_instance_count', 1)} | identity={row.get('review_identity_kind', '')} | representative={row.get('representative_raw_roi_id', '')}"
            )
        return "\n".join(lines)

    def _current_screen_geometry(self) -> tuple[int, int, int, int] | None:
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else None
        if screen is None:
            return None
        geometry = screen.geometry()
        return geometry.x(), geometry.y(), geometry.width(), geometry.height()

    def _current_overlay_samples(self) -> pd.DataFrame:
        if self.case_model is None:
            return pd.DataFrame()
        if self.current_source == "Tobii Live":
            return self._slice_history_frame(self.current_slice_index)
        return self._slice_history_frame(self.current_slice_index)

    def _refresh_overlay_policy(self) -> None:
        overlays_locked = self.review_mode.mode == "Silent" and self.session_state == SESSION_ACTIVE
        controls = {
            "roi": self.overlay_roi,
            "gaze_points": self.overlay_gaze,
            "heatmap": self.overlay_heatmap,
            "scanpath": self.overlay_scanpath,
        }
        availability = self._overlay_availability()
        for layer_name, checkbox in controls.items():
            allowed = bool(availability[layer_name])
            requested = bool(self.overlay_preferences[layer_name])
            effective = False if overlays_locked else requested and allowed
            self.overlay_effective[layer_name] = effective
            self.viewer.set_layer_visible(layer_name, effective)
            with QSignalBlocker(checkbox):
                checkbox.setChecked(requested)
            checkbox.setEnabled((not overlays_locked) and allowed)

    def _overlay_availability(self) -> dict[str, bool]:
        has_case = self.case_model is not None
        has_live_samples = bool(self.live_samples)
        has_synthetic_gaze = bool(self.current_source == "Synthetic Replay" and self.current_session_id)
        has_gaze_source = self.current_source == "Tobii Live" or has_synthetic_gaze or has_live_samples
        return {
            "roi": has_case,
            "gaze_points": has_gaze_source,
            "heatmap": has_gaze_source,
            "scanpath": has_gaze_source,
        }

    def _timeline_sample_changed(self, sample: object) -> None:
        if isinstance(sample, dict):
            if self.timeline.follow_enabled() and not self._user_scrubbing and "ct_stack_index" in sample:
                self.set_current_slice(int(float(sample.get("ct_stack_index", self.current_slice_index))), record_visit=False)
            self.viewer.set_playback_sample(sample)
            self._reload_synthetic_session_history()
            self.viewer.set_overlay_samples(self._current_overlay_samples())
            if self.current_source == "Synthetic Replay":
                replay_samples = self._session_samples(include_playback_limit=True)
                if not replay_samples.empty:
                    self._update_roi_states_from_samples(replay_samples, source_label="replay")

    def _results_hidden(self) -> bool:
        return self.review_mode.mode == "Silent" and self.session_state == SESSION_ACTIVE

    def _conceal_targets(self) -> bool:
        return self.review_mode.mode == "Silent" and self.session_state != SESSION_STOPPED

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        self.screen_mode = "fullscreen"
        self.showFullScreen()
        QTimer.singleShot(0, self.viewer.canvas.fit_to_view)

    def exit_fullscreen(self) -> None:
        self.screen_mode = "work-area"
        self.showMaximized()
        QTimer.singleShot(0, self.viewer.canvas.fit_to_view)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.exit_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.viewer.canvas.is_fit_mode():
            QTimer.singleShot(0, self.viewer.canvas.fit_to_view)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.viewer.canvas.fit_to_view)

    def save_session(self, prompt: bool = False) -> Path | None:
        if not self.live_samples and not prompt:
            return None
        base_dir = self.output_root / "live_sessions"
        base_dir.mkdir(parents=True, exist_ok=True)
        case_id = self.case_selector.currentText() or "unknown_case"
        filename = f"{case_id}_{self.current_source.replace(' ', '_').lower()}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
        csv_path = base_dir / f"{filename}.csv"
        json_path = base_dir / f"{filename}.json"
        if self.live_samples:
            fieldnames = sorted({key for row in self.live_samples for key in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.live_samples)
        payload = {
            "case_id": case_id,
            "series_uid": self.series_selector.currentText(),
            "source": self.current_source,
            "mode": self.review_mode.mode,
            "sample_count": len(self.live_samples),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.saved_session_path = csv_path if csv_path.exists() else json_path
        if self.current_source == "Tobii Live" and self.live_samples and self.case_model is not None:
            validation_dir = self.output_root / "tobii_validation_phase5" / filename
            self.live_validation_summary = write_live_validation_bundle(
                validation_dir,
                session_id=filename,
                device_payload=self.tobii_source.get_status_payload(),
                samples=self.live_samples,
                roi_rows=self.case_model.roi_geometry.to_dict("records"),
                feature_columns=list(self.data.feature_columns),
                current_state_store=self.roi_state_store,
                mapping_diagnostics=self.mapping_failure_diagnostics,
            )
        if prompt:
            QMessageBox.information(self, "Session Saved", f"Session artifacts saved under:\n{base_dir}")
        return self.saved_session_path


def _worklist_label(row: pd.Series, concealed: bool) -> str:
    if concealed:
        return f"{row.get('display_index', 'ROI')}\nPending"
    return f"{row.get('display_index', 'ROI')}  Slice {_display_slice(row.get('ct_stack_index'), one_based=True)}  {row.get('display_review_state', 'Pending')}  ›"


def _slice_worklist_label(row: pd.Series, concealed: bool) -> str:
    if concealed:
        return "ROI-bearing slice\nHidden"
    roi_count = int(row.get("roi_instances", 0))
    annotation_count = int(row.get("annotation_instances", roi_count) or roi_count)
    annotation_suffix = f"     {annotation_count} annotations" if annotation_count > roi_count else ""
    return (
        f"Slice {_display_slice(row.get('ct_stack_index'), one_based=True)}     "
        f"P:{int(row.get('pending_count', 0))}   W:{int(row.get('weak_count', 0))}   M:{int(row.get('missed_count', 0))}     "
        f"{row.get('status', 'Not visited')}{annotation_suffix}"
    )


def _roi_queue_item_label(row: pd.Series, concealed: bool, is_current_slice: bool) -> str:
    if concealed:
        return f"{row.get('display_index', 'ROI')}   Hidden"
    status = _display_attention_status(str(row.get("mapped_review_status", "not_evaluated")))
    if is_current_slice:
        status = "Current" if status == "Not yet evaluated" else status
    return f"{row.get('display_index', row.get('roi_id'))}   {status}"


def _attention_evidence_text(row: pd.Series) -> str:
    lines = [
        f"Dwell time inside ROI: {_render_metric(row.get('total_gaze_time_inside_roi_ms'), ' ms')}",
        f"Fixations inside ROI: {_render_metric(row.get('fixation_count_inside_roi'))}",
        f"Nearest gaze distance: {_render_metric(row.get('min_distance_to_roi_px'), ' px')}",
        f"ROI revisits: {_render_metric(row.get('roi_revisit_count'))}",
        f"Slice viewing time: {_render_metric(row.get('time_on_roi_slice_ms'), ' ms')}",
        f"Valid gaze percentage: {_fmt_percent(row.get('gaze_validity_ratio')) if _fmt_percent(row.get('gaze_validity_ratio')) != '-' else 'Not available'}",
    ]
    return "\n".join(lines)


def _render_metric(value: object, suffix: str = "") -> str:
    rendered = _fmt_value(value)
    if rendered == "-":
        return "Not available"
    return f"{rendered}{suffix}"


def _behavior_support_text(row: pd.Series) -> str:
    bullets: list[str] = []
    dwell = _safe_float(row.get("total_gaze_time_inside_roi_ms"))
    hits = _safe_float(row.get("gaze_hit_count_inside_roi"))
    fixations = _safe_float(row.get("fixation_count_inside_roi"))
    revisits = _safe_float(row.get("roi_revisit_count"))
    validity = _safe_float(row.get("gaze_validity_ratio"))
    background = _safe_float(row.get("background_gaze_ratio"))
    if dwell >= 800:
        bullets.append("- Strong ROI dwell")
    elif dwell > 0:
        bullets.append("- Brief ROI dwell")
    if fixations >= 1:
        bullets.append("- Strong fixation evidence")
    elif "fixation_count_inside_roi" in row:
        bullets.append("- Limited fixation evidence")
    if hits >= 5:
        bullets.append("- Multiple direct ROI hits")
    elif hits > 0:
        bullets.append("- Some direct ROI hits")
    else:
        bullets.append("- Limited direct ROI evidence")
    if revisits >= 2:
        bullets.append("- Multiple ROI revisits")
    elif revisits >= 1:
        bullets.append("- ROI revisit evidence")
    if validity >= 0.85:
        bullets.append("- High gaze validity")
    elif validity > 0:
        bullets.append("- Reduced gaze validity")
    if background <= 0.35 and "background_gaze_ratio" in row:
        bullets.append("- Attention remained near ROI")
    elif background >= 0.7:
        bullets.append("- Attention frequently remained outside ROI")
    if str(row.get("predicted_behavior_label", "")) not in {"", "unavailable", "awaiting_live_gaze"}:
        bullets.append(f"- Behavior class: {_display_behavior_label(row.get('predicted_behavior_label'))}")
    return "\n".join(bullets[:5]) if bullets else "No supporting evidence available"


def _attention_explanation_text(row: pd.Series, attention_state: str, thresholds) -> str:
    status = str(row.get("mapped_review_status", "not_evaluated"))
    if attention_state == "Collecting evidence":
        valid_samples = int(_safe_float(row.get("_valid_sample_count", row.get("_sample_count", 0))))
        required_samples = max(1, int(round(float(thresholds.minimum_valid_time_on_roi_slice_ms) / 16.667)))
        valid_time = _safe_float(row.get("valid_gaze_time_on_roi_slice_ms")) / 1000.0
        required_time = float(thresholds.minimum_valid_time_on_roi_slice_ms) / 1000.0
        if valid_samples > 0:
            return f"Collecting evidence\n{valid_samples} of {required_samples} usable samples collected\n{valid_time:.1f} of {required_time:.1f} seconds observed"
        return f"Collecting evidence\n{valid_time:.1f} of {required_time:.1f} seconds observed"
    if attention_state == "Not yet evaluated":
        return "The reader is still reviewing this slice. Evaluation will begin after sufficient slice viewing time."
    if status == "reviewed":
        return "The ROI was marked as reviewed because gaze entered the ROI, included fixation evidence, and exceeded the configured dwell-time threshold."
    if status == "weakly_reviewed":
        return "Gaze approached or briefly entered the ROI, but the available evidence did not meet the full review criteria."
    if status == "not_reviewed":
        return "No sufficient ROI-directed attention was observed during the completed slice review."
    return "The reader is still reviewing this slice. Evaluation will begin after sufficient slice viewing time."


def _behavior_pattern_text(row: pd.Series, readiness_message: str) -> str:
    label = str(row.get("predicted_behavior_label", "awaiting_live_gaze"))
    if label in {"", "unavailable", "awaiting_live_gaze"}:
        return readiness_message or "The scanpath pattern will be shown after sufficient ROI-specific features are available."
    explanations = []
    if _safe_float(row.get("gaze_hit_count_inside_roi")) >= 2:
        explanations.append("direct transitions toward the ROI")
    if _safe_float(row.get("fixation_count_inside_roi")) >= 1:
        explanations.append("multiple ROI fixations")
    if _safe_float(row.get("roi_revisit_count")) >= 1:
        explanations.append("one or more ROI revisits")
    if not explanations:
        explanations.append("the available scanpath evidence")
    return f"The scanpath pattern was associated with {', '.join(explanations[:3])}."


def _cognitive_basis_text(row: pd.Series) -> str:
    return "Basis: fixation dispersion, gaze transitions, revisits, tracking stability"


def _display_review_state(status: str) -> str:
    return {
        "reviewed": "Sufficient",
        "weakly_reviewed": "Weak",
        "uncertain_review": "Weak",
        "in_review": "Weak",
        "not_reviewed": "Insufficient",
        "needs_revisit": "Insufficient",
        "missed": "Insufficient",
        "not_evaluated": "Pending",
        "pending": "Pending",
    }.get(str(status), "Pending")


def _display_attention_status(status: str) -> str:
    return {
        "reviewed": "Reviewed",
        "weakly_reviewed": "Weak evidence",
        "not_reviewed": "Not reviewed",
        "not_evaluated": "Not yet evaluated",
    }.get(str(status), "Not yet evaluated")


def _canonical_display_state(model_status: object, rule_status: object) -> str:
    return str(rule_status or "not_evaluated")


def _coverage_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts = {"ROI instances": int(len(rows)), "Reviewed": 0, "Weakly reviewed": 0, "Missed": 0, "Not evaluated": 0}
    if rows.empty:
        return counts
    for state in rows.get("display_review_state", pd.Series(dtype=object)).astype(str):
        if state == "Sufficient":
            counts["Reviewed"] += 1
        elif state == "Weak":
            counts["Weakly reviewed"] += 1
        elif state == "Insufficient":
            counts["Missed"] += 1
        else:
            counts["Not evaluated"] += 1
    assert counts["ROI instances"] == counts["Reviewed"] + counts["Weakly reviewed"] + counts["Missed"] + counts["Not evaluated"]
    return counts


def _count_roi_slices_viewed(roi_slices: set[int], slice_state_store: dict[int, dict[str, object]]) -> int:
    return sum(1 for index in roi_slices if str(slice_state_store.get(index, {}).get("status", "")) in {"Briefly viewed", "Reviewed"})


def _display_behavior_label(value: object) -> str:
    return {
        "awaiting_live_gaze": "Awaiting live gaze",
        "focused_roi_confirmation": "Focused ROI confirmation",
        "expert_like_systematic_review": "Systematic review",
        "partial_near_miss_review": "Partial near miss",
        "missed_roi_search": "Missed ROI search",
        "skipped_slice": "Skipped slice",
        "high_load_fragmented_review": "Fragmented review",
        "unavailable": "Unavailable",
    }.get(str(value), str(value).replace("_", " "))


def _display_cognitive_proxy(value: object) -> str:
    return {
        "low_load_proxy": "Low scanpath fragmentation",
        "medium_load_proxy": "Moderate scanpath fragmentation",
        "high_load_proxy": "High scanpath fragmentation",
    }.get(str(value), "Unknown")


def _nunique_if_present(rows: pd.DataFrame, columns: list[str]) -> int:
    for column in columns:
        if column in rows.columns and rows[column].notna().any():
            return int(rows[column].astype(str).nunique())
    return 0


def _fmt_value(value: object) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "-"
        if number.is_integer():
            return str(int(number))
        return f"{number:.1f}"
    except (TypeError, ValueError, AttributeError):
        return "-"


def _fmt_percent(value: object) -> str:
    try:
        number = max(0.0, min(1.0, float(value)))
        return f"{number * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio_from_ms(numerator_ms: object, denominator_ms: object) -> float:
    numerator = _safe_float(numerator_ms)
    denominator = _safe_float(denominator_ms)
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _fmt_seconds(value: object) -> str:
    try:
        number = float(value)
        if pd.isna(number) or number <= 0:
            return "-"
        return f"{number / 1000.0:.1f} s"
    except (TypeError, ValueError):
        return "-"


def _format_elapsed_text(value: float) -> str:
    total_seconds = max(0, int(round(float(value))))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _display_slice(value: object, one_based: bool) -> str:
    try:
        number = int(float(value))
        if one_based:
            number += 1
        return str(number)
    except (TypeError, ValueError):
        return "-"


def build_class_reference_baselines(data: WorkstationData) -> dict[str, dict[str, object]]:
    baselines: dict[str, dict[str, object]] = {}
    features = data.features.copy()
    if features.empty or "hidden_behavior_label" not in features.columns:
        return baselines
    for label, group in features.groupby("hidden_behavior_label", sort=False):
        first_row = enrich_case_row(group.iloc[0].copy(), data)
        metrics = {
            display_name: float(pd.to_numeric(group[column], errors="coerce").mean())
            for display_name, column in REFERENCE_METRICS.items()
            if column in group.columns
        }
        baselines[str(label)] = {
            "source": "class_baseline",
            "title": f"Synthetic reference: class baseline ({len(group)} episodes)",
            "behavior_label": str(label),
            "confidence": first_row.get("prediction_confidence"),
            "review_state": first_row.get("review_state", "pending"),
            "class_probabilities": first_row.get("class_probabilities", {}),
            "metrics": metrics,
        }
    return baselines


def build_reference_payload(
    row: pd.Series | dict[str, object] | None,
    case_features: pd.DataFrame,
    baselines: dict[str, dict[str, object]],
    data: WorkstationData,
) -> dict[str, object] | None:
    if row is None:
        return None
    roi_id = str(row.get("roi_id", ""))
    session_id = str(row.get("session_id", ""))
    same_roi = case_features[(case_features.get("roi_id", pd.Series(index=case_features.index, dtype=object)).astype(str) == roi_id) & (case_features.get("session_id", pd.Series(index=case_features.index, dtype=object)).astype(str) == session_id)]
    if not same_roi.empty:
        reference_row = enrich_case_row(same_roi.iloc[0].copy(), data)
        return {
            "source": "same_roi",
            "title": "Synthetic reference: same ROI replay",
            "behavior_label": reference_row.get("hidden_behavior_label", "unavailable"),
            "confidence": reference_row.get("prediction_confidence"),
            "class_probabilities": reference_row.get("class_probabilities", {}),
            "metrics": {display_name: reference_row.get(column) for display_name, column in REFERENCE_METRICS.items()},
        }
    predicted_label = str(row.get("predicted_behavior_label", ""))
    if predicted_label and predicted_label in baselines:
        return baselines[predicted_label]
    hidden_label = str(row.get("hidden_behavior_label", ""))
    if hidden_label and hidden_label in baselines:
        return baselines[hidden_label]
    return None


def launch_review_workstation(output_root: str | Path | None = None, source: str = "synthetic") -> int:
    if source == "future_tobii_placeholder":
        raise ValueError(TOBII_PLACEHOLDER_MESSAGE)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(stylesheet())
    _install_qt_exception_hook()
    data = load_workstation_data(output_root, source="synthetic")
    window = MedGazeReviewWorkstation(data, output_root=output_root)
    window.showMaximized()
    QTimer.singleShot(0, window.viewer.canvas.fit_to_view)
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


def _install_qt_exception_hook() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.getLogger("medgazear_final").exception("Unhandled workstation exception")
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(None, "MedGazeAR Error", f"The workstation encountered an unexpected error and must stop.\n\n{exc_value}\n\nDetails:\n{message}")
            app.quit()
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception
