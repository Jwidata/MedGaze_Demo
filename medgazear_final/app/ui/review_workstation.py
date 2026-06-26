"""Tabbed MedGazeAR research workstation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.case_review_model import CaseReviewModel, build_case_review_model
from app.ui.ct_viewer_widget import CTViewerWidget
from app.ui.insight_panel import InsightPanel
from app.ui.review_mode import ReviewMode, default_layer_visibility
from app.ui.timeline_widget import TimelineWidget
from app.ui.ui_data_loader import WorkstationData, enrich_case_row, load_workstation_data
from app.ui.ui_theme import LIMITATION_BANNER, TOBII_PLACEHOLDER_MESSAGE, stylesheet


class ReviewWorkstation(QMainWindow):
    def __init__(self, data: WorkstationData) -> None:
        super().__init__()
        self.data = data
        self.current_row: pd.Series | None = None
        self.case_model: CaseReviewModel | None = None
        self.overlay_checkboxes: dict[str, QCheckBox] = {}

        self.setWindowTitle("MedGazeAR Validation Workstation")
        self.resize(1650, 980)

        self.case_selector = QComboBox()
        self.source_selector = QComboBox()
        self.review_mode_selector = QComboBox()
        self.navigation_mode_selector = QComboBox()
        self.behavior_filter = QComboBox()
        self.coverage_banner = QLabel("Evidence coverage: -")
        self.coverage_banner.setWordWrap(True)
        self.coverage_banner.setStyleSheet("background:#1f3552; color:#eaf4ff; padding:8px; border-radius:6px; font-weight:bold;")
        self.roi_selector = QListWidget()
        self.viewer = CTViewerWidget()
        self.insight_panel = InsightPanel()
        self.timeline = TimelineWidget()
        self.timeline.sample_changed.connect(self._timeline_sample_changed)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        self._build_ui()
        self._populate_controls()
        self._connect_signals()
        self._refresh_roi_selector()
        self.load_current_roi()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        banner = QLabel(LIMITATION_BANNER)
        banner.setObjectName("Banner")
        layout.addWidget(banner)
        layout.addWidget(self.progress)

        tabs = QTabWidget()
        tabs.addTab(self._pipeline_dashboard_tab(), "1 Validation Dashboard")
        tabs.addTab(self._case_review_tab(), "2 Case ROI Map")
        tabs.addTab(self._synthetic_reference_tab(), "3 Synthetic Reference")
        tabs.addTab(self._tobii_comparison_tab(), "4 Tobii Comparison")
        tabs.addTab(self._attention_tab(), "5 Attention Analysis")
        tabs.addTab(self._cognitive_tab(), "6 Cognitive Proxy")
        tabs.addTab(self._summary_tab(), "7 Export Summary")
        layout.addWidget(tabs, stretch=1)
        self.setCentralWidget(root)

    def _case_review_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._case_sidebar())
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.insight_panel)
        splitter.setSizes([330, 900, 420])
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.timeline)
        return tab

    def _pipeline_dashboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel("MedGazeAR Validation Workstation")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff; padding: 8px;")
        subtitle = QLabel(
            "Synthetic/model outputs are aggregated backend reference. Future Tobii gaze is the live validation path. "
            "Attention adequacy and cognitive-load proxy stay separate."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; color: #c8d7ea; padding: 0 8px 14px 8px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.addWidget(_dashboard_card("1. Backend Reference", "Synthetic gaze trains and summarizes expected source-agnostic gaze patterns.", [
            f"Feature rows: {len(self.data.features)}",
            f"Behavior classes: {self.data.features['hidden_behavior_label'].nunique()}",
            "Used to create model expectations, not clinical truth.",
        ]))
        cards.addWidget(_dashboard_card("2. Tobii Validation", "Future live/recorded Tobii data should enter the exact same schema and model pipeline.", [
            "Same CT/ROI comparison",
            "New CT/ROI generalization",
            "Heatmap, attention, behavior, and cognitive-proxy agreement.",
        ]))
        cards.addWidget(_dashboard_card("3. Separated Analyses", "Do not mix model behavior, rule attention, and cognitive proxy into one claim.", [
            f"Attention rows: {len(self.data.attention)}",
            f"Cognitive rows: {len(self.data.cognitive)}",
            "Each analysis has its own visual tab and limitation wording.",
        ]))
        layout.addLayout(cards)

        flow = QTextEdit()
        flow.setReadOnly(True)
        flow.setPlainText(
            "Recommended validation workflow:\n\n"
            "1. Case ROI Map: inspect the real CT stack and ROI locations. Default playback scans all CT slices.\n"
            "2. Synthetic Reference: use aggregate synthetic/model outputs as the baseline, not as clinical truth.\n"
            "3. Tobii Comparison: future real Tobii data should be imported or streamed into the same schema.\n"
            "4. Attention Analysis: evaluate ROI review adequacy separately.\n"
            "5. Cognitive Proxy: evaluate workload-like gaze complexity separately.\n\n"
            "Key thesis message: synthetic trained the model; Tobii validates it."
        )
        layout.addWidget(flow, stretch=1)
        return tab

    def _case_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Case"))
        layout.addWidget(self.case_selector)
        layout.addWidget(self.coverage_banner)
        layout.addWidget(QLabel("Validation source"))
        layout.addWidget(self.source_selector)
        layout.addWidget(QLabel("Overlay preset"))
        layout.addWidget(self.review_mode_selector)
        layout.addWidget(QLabel("Navigation"))
        layout.addWidget(self.navigation_mode_selector)
        layout.addWidget(QLabel("ROI filter / evidence view"))
        layout.addWidget(self.behavior_filter)
        layout.addWidget(QLabel("ROI review queue"))
        layout.addWidget(self.roi_selector, stretch=1)

        for label, callback in (
            ("Load selected ROI", self.load_current_roi),
            ("Previous ROI", self.previous_roi),
            ("Next ROI", self.next_roi),
            ("Export current CT view", self.export_current_view),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(button)

        overlays = QGroupBox("Overlays and legend")
        overlay_layout = QVBoxLayout(overlays)
        for label, layer in (("ROI overlay", "roi"), ("Heatmap", "heatmap"), ("Gaze points", "gaze_points"), ("Scanpath", "scanpath")):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda enabled, layer_name=layer: self.viewer.set_layer_visible(layer_name, enabled))
            self.overlay_checkboxes[layer] = checkbox
            overlay_layout.addWidget(checkbox)
        legend = QLabel(
            "Legend\n"
            "Green ROI: segmentation ROI on current CT slice\n"
            "Yellow ROI: selected ROI\n"
            "Blue dots: gaze samples\n"
            "Orange/yellow lines: recent scanpath\n"
            "Red/yellow density: heatmap\n"
            "Gray/no evidence: anatomical ROI exists, but no synthetic/Tobii evidence yet\n\n"
            "No labels are drawn inside the CT image; this panel explains the colors."
        )
        legend.setWordWrap(True)
        overlay_layout.addWidget(legend)
        layout.addWidget(overlays)
        return panel

    def _synthetic_reference_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        table = QTableWidget()
        guide = QLabel(
            "Aggregated synthetic/model baseline. Use this to explain what the backend learned before real Tobii validation; "
            "do not treat individual synthetic episodes as clinical evidence."
        )
        guide.setWordWrap(True)
        left_layout.addWidget(QLabel("Synthetic backend reference baseline"))
        left_layout.addWidget(guide)
        left_layout.addWidget(table, stretch=1)
        details = QTextEdit()
        details.setReadOnly(True)
        details.setPlainText(
            "How to read this tab\n\n"
            "- Rows are behavior-level aggregates from synthetic/source-agnostic gaze features.\n"
            "- The table supports future Tobii validation by defining expected ranges.\n"
            "- Real Tobii sessions should be scored by the same feature extractor and compared against these aggregate baselines.\n"
            "- This is not clinical validation and not a reader-performance claim."
        )
        layout.addWidget(left, stretch=2)
        layout.addWidget(details, stretch=1)
        rows = self.data.features.copy()
        metrics = [
            "gaze_validity_ratio",
            "total_gaze_time_inside_roi_ms",
            "gaze_dispersion_px",
            "background_gaze_ratio",
        ]
        available = [col for col in metrics if col in rows.columns]
        if not rows.empty and "hidden_behavior_label" in rows.columns and available:
            summary = rows.groupby("hidden_behavior_label")[available].mean(numeric_only=True).reset_index()
            summary.insert(1, "episodes", rows.groupby("hidden_behavior_label").size().values)
        else:
            summary = pd.DataFrame(columns=["hidden_behavior_label", "episodes", *available])
        _fill_table(table, summary)
        return tab

    def _tobii_comparison_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        synthetic = QTextEdit()
        synthetic.setReadOnly(True)
        synthetic.setPlainText(
            "Synthetic reference\n\n"
            "- Uses existing synthetic gaze episodes.\n"
            "- Trained model predicts behavior from shared gaze features.\n"
            "- Provides expected behavior-level patterns for ROI review."
        )
        tobii = QTextEdit()
        tobii.setReadOnly(True)
        tobii.setPlainText(
            "Future real Tobii experiment\n\n"
            f"{TOBII_PLACEHOLDER_MESSAGE}\n\n"
            "Planned comparison modes:\n"
            "1. Same CT/ROI: compare Tobii gaze to synthetic reference on the same ROI.\n"
            "2. New CT/ROI: apply the synthetic-trained model to new real gaze.\n\n"
            "Comparison metrics: heatmap similarity, ROI dwell difference, time-to-first-fixation difference, "
            "scanpath length difference, behavior agreement, attention agreement, and cognitive-proxy difference."
        )
        metrics = QTableWidget()
        _fill_table(
            metrics,
            pd.DataFrame(
                [
                    {"metric": "heatmap similarity", "status": "pending real Tobii input"},
                    {"metric": "behavior agreement", "status": "pending real Tobii input"},
                    {"metric": "attention agreement", "status": "pending real Tobii input"},
                    {"metric": "cognitive proxy difference", "status": "pending real Tobii input"},
                ]
            ),
        )
        layout.addWidget(synthetic)
        layout.addWidget(tobii)
        layout.addWidget(metrics)
        return tab

    def _attention_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Rule-based ROI attention evidence. This is separate from behavior prediction."))
        table = QTableWidget()
        rows = self.data.attention.merge(self.data.features, on=["session_id", "roi_id", "hidden_behavior_label"], how="left")
        cols = ["rule_attention_status", "hidden_behavior_label", "case_id", "session_id", "valid_gaze_time_on_roi_slice_ms", "total_gaze_time_inside_roi_ms", "gaze_hit_count_inside_roi", "gaze_validity_ratio"]
        _fill_table(table, rows[[c for c in cols if c in rows.columns]].head(500))
        layout.addWidget(table)
        return tab

    def _cognitive_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Cognitive-load proxy analysis. This is a proxy, not validated cognitive load."))
        table = QTableWidget()
        rows = self.data.cognitive.merge(self.data.features, on=["session_id", "roi_id"], how="left")
        cols = ["cognitive_load_proxy", "cognitive_load_proxy_score", "hidden_behavior_label", "case_id", "gaze_dispersion_px", "scanpath_length_px", "roi_revisit_count", "background_gaze_ratio", "dropout_ratio"]
        _fill_table(table, rows[[c for c in cols if c in rows.columns]].head(500))
        layout.addWidget(table)
        return tab

    def _summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "MedGazeAR visual pipeline\n\n"
            "1. CT DICOM and SEG provide the anatomical ROI map.\n"
            "2. Synthetic gaze provides learned reference behavior patterns.\n"
            "3. The trained model predicts review behavior from source-agnostic gaze features.\n"
            "4. Future Tobii gaze should use the same schema, feature extractor, renderer, and model.\n"
            "5. Attention analysis and cognitive-proxy analysis are separated to avoid overclaiming.\n\n"
            f"Synthetic feature rows: {len(self.data.features)}\n"
            f"Attention rows: {len(self.data.attention)}\n"
            f"Cognitive proxy rows: {len(self.data.cognitive)}\n"
            f"ROI geometry rows: {len(self.data.roi_geometry)}\n\n"
            "Research prototype. Not for clinical diagnosis."
        )
        layout.addWidget(text)
        return tab

    def _populate_controls(self) -> None:
        cases = sorted(self.data.representative_cases["case_id"].astype(str).unique().tolist())
        self.case_selector.addItems(cases)
        self.source_selector.addItems(["Synthetic replay / learned behavior simulation", "Future Tobii live experiment"])
        self.source_selector.setCurrentText("Synthetic replay / learned behavior simulation")
        self.review_mode_selector.addItems([mode.value for mode in ReviewMode])
        self.review_mode_selector.setCurrentText(ReviewMode.SILENT.value)
        self.navigation_mode_selector.addItems(["ROI slices only", "All CT slices"])
        self.navigation_mode_selector.setCurrentText("All CT slices")
        self.behavior_filter.addItems(["All", "ROIs with gaze evidence", "ROIs without gaze evidence", "Weak/not reviewed"])
        self.behavior_filter.addItems(sorted(self.data.features["hidden_behavior_label"].dropna().astype(str).unique().tolist()))

    def _connect_signals(self) -> None:
        self.case_selector.currentTextChanged.connect(lambda _value: self._refresh_roi_selector())
        self.source_selector.currentTextChanged.connect(self._source_changed)
        self.review_mode_selector.currentTextChanged.connect(self._review_mode_changed)
        self.navigation_mode_selector.currentTextChanged.connect(self._navigation_mode_changed)
        self.behavior_filter.currentTextChanged.connect(lambda _value: self._refresh_roi_selector())
        self.roi_selector.currentTextChanged.connect(self._select_queue_item)
        self.roi_selector.itemClicked.connect(lambda _item: self.load_current_roi())
        self.timeline.playback_mode.currentTextChanged.connect(lambda _value: self._set_timeline_samples())
        self._review_mode_changed(self.review_mode_selector.currentText())
        self._navigation_mode_changed(self.navigation_mode_selector.currentText())

    def _source_changed(self, source: str) -> None:
        if "Tobii" in source:
            QMessageBox.information(self, "Future Tobii Source", TOBII_PLACEHOLDER_MESSAGE)
            self.source_selector.setCurrentText("Synthetic replay / learned behavior simulation")

    def _review_mode_changed(self, mode: str) -> None:
        visibility = default_layer_visibility(mode)
        for layer, enabled in visibility.items():
            checkbox = self.overlay_checkboxes.get(layer)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(enabled)
                checkbox.blockSignals(False)
            self.viewer.set_layer_visible(layer, enabled)

    def _navigation_mode_changed(self, mode: str) -> None:
        self.viewer.set_roi_only_navigation(mode == "ROI slices only")

    def _load_case_model(self) -> None:
        case_id = self.case_selector.currentText()
        if not case_id:
            return
        if self.case_model is not None and self.case_model.patient_id == case_id:
            return
        try:
            self.progress.setVisible(True)
            self.case_model = build_case_review_model(case_id, self.data)
            summary = self.case_model.summary()
            self.coverage_banner.setText(
                f"Evidence coverage: {summary['evaluated_roi_episodes']} synthetic replay episodes for "
                f"{summary['roi_count']} segmentation ROIs. "
                f"{summary['uncovered_seg_rois']} ROIs are anatomical-only until Tobii or more synthetic replay exists."
            )
            self.insight_panel.set_case_summary(
                "Case summary:\n"
                f"patient/case: {summary['patient_id']}\n"
                f"total slices: {summary['total_slices']}\n"
                f"ROI slices: {summary['roi_slices']}\n"
                f"segmentation ROIs: {summary['roi_count']}\n"
                f"synthetic replay episodes: {summary['evaluated_roi_episodes']}\n"
                f"ROIs without replay evidence: {summary['uncovered_seg_rois']}\n"
                f"reviewed/weak/missed/not eval episodes: {summary['reviewed']} / {summary['weakly_reviewed']} / {summary['not_reviewed']} / {summary['not_evaluated']}"
            )
        except Exception as exc:
            self.case_model = None
            self.coverage_banner.setText("Evidence coverage: case failed to load")
            self.insight_panel.set_case_summary(f"Case loading error:\n{exc}")
        finally:
            self.progress.setVisible(False)

    def _refresh_roi_selector(self) -> None:
        self.current_row = None
        self.roi_selector.clear()
        self._load_case_model()
        rows = self._queue_rows()
        for index, row in rows.iterrows():
            ct_slice = int(row.get("ct_stack_index", 0)) + 1
            status = str(row.get("rule_attention_status", "no_replay_evidence"))
            label = str(row.get("hidden_behavior_label", "no_replay_evidence")).replace("_", " ")
            item = QListWidgetItem(f"CT {ct_slice} | {status} | {label}")
            item.setData(Qt.ItemDataRole.UserRole, int(index))
            item.setToolTip(f"ROI: {row.get('roi_id', '-')}\nSynthetic evidence: {'yes' if row.get('session_id', '') else 'no'}")
            self.roi_selector.addItem(item)
        if self.roi_selector.count() > 0:
            self.roi_selector.setCurrentRow(0)

    def _queue_rows(self) -> pd.DataFrame:
        self._load_case_model()
        if self.case_model is None:
            return pd.DataFrame()
        geometry = self.case_model.roi_geometry.copy()
        feature_columns = [c for c in self.case_model.features.columns if c not in geometry.columns or c == "roi_id"]
        features = self.case_model.features[feature_columns].copy() if not self.case_model.features.empty else pd.DataFrame(columns=["roi_id"])
        rows = geometry.merge(features, on="roi_id", how="left")
        if not self.case_model.attention.empty and {"session_id", "roi_id"}.issubset(rows.columns):
            rows = rows.merge(self.case_model.attention[["session_id", "roi_id", "rule_attention_status"]], on=["session_id", "roi_id"], how="left")
        rows["session_id"] = rows.get("session_id", pd.Series(index=rows.index, dtype=object)).fillna("")
        rows["hidden_behavior_label"] = rows.get("hidden_behavior_label", pd.Series(index=rows.index, dtype=object)).fillna("no_replay_evidence")
        rows["rule_attention_status"] = rows.get("rule_attention_status", pd.Series(index=rows.index, dtype=object)).fillna("no_replay_evidence")
        filter_text = self.behavior_filter.currentText()
        if filter_text == "ROIs with gaze evidence":
            rows = rows[rows["session_id"].astype(str) != ""]
        elif filter_text == "ROIs without gaze evidence":
            rows = rows[rows["session_id"].astype(str) == ""]
        elif filter_text == "Weak/not reviewed":
            rows = rows[rows["rule_attention_status"].isin(["weakly_reviewed", "not_reviewed", "not_evaluated"])]
        elif filter_text and filter_text != "All":
            rows = rows[rows["hidden_behavior_label"].astype(str) == filter_text]
        priority = {"not_evaluated": 0, "not_reviewed": 1, "weakly_reviewed": 2, "no_replay_evidence": 3, "reviewed": 4}
        rows["_priority"] = rows["rule_attention_status"].map(priority).fillna(9)
        return rows.sort_values(["_priority", "ct_stack_index", "roi_id"]).drop(columns=["_priority"]).reset_index(drop=True)

    def _select_queue_item(self, _value: str) -> None:
        item = self.roi_selector.currentItem()
        rows = self._queue_rows()
        if item is None:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if 0 <= index < len(rows):
            self.current_row = rows.iloc[index]

    def load_current_roi(self) -> None:
        row = self.current_row
        if row is None:
            rows = self._queue_rows()
            if rows.empty:
                return
            row = rows.iloc[0]
        enriched = enrich_case_row(row, self.data) if str(row.get("session_id", "")) else row.copy()
        if not str(enriched.get("session_id", "")):
            enriched["predicted_behavior_label"] = "unavailable_no_replay_evidence"
            enriched["prediction_confidence"] = 0.0
            enriched["guided_narration"] = "This segmentation ROI has no synthetic gaze replay evidence. It is anatomical-only until real Tobii data or additional synthetic sessions are available."
        self._load_case_model()
        if self.case_model is None:
            return
        self.viewer.load_case(self.case_model, enriched)
        self.insight_panel.set_case(enriched)
        self.current_row = enriched
        self._set_timeline_samples()

    def previous_roi(self) -> None:
        self._step_roi(-1)

    def next_roi(self) -> None:
        self._step_roi(1)

    def _step_roi(self, delta: int) -> None:
        if self.roi_selector.count() == 0:
            return
        self.roi_selector.setCurrentRow((self.roi_selector.currentRow() + delta) % self.roi_selector.count())
        self.load_current_roi()

    def export_current_view(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export current CT view", "medgazear_current_view.png", "PNG Files (*.png)")
        if path:
            self.viewer.export_current_view(path)

    def _gaze_for(self, row: pd.Series) -> pd.DataFrame:
        return self.data.gaze[(self.data.gaze["session_id"].astype(str) == str(row.get("session_id", ""))) & (self.data.gaze["roi_id"].astype(str) == str(row.get("roi_id", "")))].copy()

    def _set_timeline_samples(self) -> None:
        if self.case_model is None:
            return
        mode = self.timeline.playback_mode_text()
        if mode.startswith("CT slice cine"):
            self.timeline.set_samples(
                pd.DataFrame(
                    [
                        {
                            "mode": "ct_cine",
                            "timestamp_ms": idx * 100.0,
                            "ct_stack_index": idx,
                            "image_x": 0.0,
                            "image_y": 0.0,
                            "is_valid": True,
                        }
                        for idx in range(self.case_model.total_slices)
                    ]
                )
            )
        elif mode.startswith("Whole case evidence"):
            self.timeline.set_samples(self.case_model.case_gaze_samples())
        elif self.current_row is not None and str(self.current_row.get("session_id", "")):
            selected = self._gaze_for(self.current_row)
            selected["ct_stack_index"] = int(self.current_row.get("ct_stack_index", 0))
            self.timeline.set_samples(selected)
        else:
            self.timeline.set_samples(pd.DataFrame())

    def _timeline_sample_changed(self, sample: object) -> None:
        if isinstance(sample, dict) and "ct_stack_index" in sample:
            self.viewer.set_playback_sample(sample)


def _fill_table(table: QTableWidget, data: pd.DataFrame) -> None:
    table.clear()
    table.setRowCount(len(data))
    table.setColumnCount(len(data.columns))
    table.setHorizontalHeaderLabels([str(c) for c in data.columns])
    for r, (_, row) in enumerate(data.iterrows()):
        for c, col in enumerate(data.columns):
            table.setItem(r, c, QTableWidgetItem(str(row[col])))
    table.resizeColumnsToContents()


def _dashboard_card(title: str, description: str, bullets: list[str]) -> QWidget:
    card = QGroupBox(title)
    card.setStyleSheet("QGroupBox { background:#162235; border:1px solid #3b5a80; border-radius:10px; padding:12px; font-weight:bold; }")
    layout = QVBoxLayout(card)
    desc = QLabel(description)
    desc.setWordWrap(True)
    desc.setStyleSheet("color:#d9e8f8; font-weight:normal;")
    layout.addWidget(desc)
    for bullet in bullets:
        label = QLabel(f"- {bullet}")
        label.setWordWrap(True)
        label.setStyleSheet("color:#b9cce2; font-weight:normal;")
        layout.addWidget(label)
    layout.addStretch(1)
    return card


def launch_review_workstation(output_root: str | Path | None = None, source: str = "synthetic") -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(stylesheet())
    data = load_workstation_data(output_root, source=source)
    window = ReviewWorkstation(data)
    window.show()
    return app.exec()


def smoke_test_workstation(output_root: str | Path | None = None, source: str = "synthetic") -> dict[str, object]:
    if source == "future_tobii_placeholder":
        return {"status": "placeholder", "message": TOBII_PLACEHOLDER_MESSAGE}
    try:
        data = load_workstation_data(output_root, source=source)
        return {"status": "ok", "cases": len(data.representative_cases), "sample_row_loaded": not data.representative_cases.empty}
    except FileNotFoundError as exc:
        return {"status": "ok", "warning": str(exc), "sample_row_loaded": False}
