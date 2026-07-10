"""Theme for the MedGazeAR workstation."""

from __future__ import annotations


def stylesheet() -> str:
    return """
    QMainWindow, QWidget {
        background: #070a0f;
        color: #e7edf5;
        font-family: Segoe UI;
        font-size: 13px;
    }
    QWidget#ExperimentBar, QWidget#SidebarShell, QWidget#CenterShell, QWidget#SessionShell, QFrame#PanelSection, QFrame#StatusGroup {
        background: #0f141b;
        border: 1px solid #1b2532;
        border-radius: 4px;
    }
    QFrame#PanelSection {
        background: transparent;
        border: 0;
        border-top: 1px solid #182331;
        border-radius: 0;
        padding-top: 6px;
    }
    QFrame#MetricCard {
        background: #101823;
        border: 1px solid #1d2d40;
        border-radius: 4px;
    }
    QFrame#StatusGroup {
        background: #10161e;
    }
    QLabel#SectionTitle, QLabel#TopBarTitle {
        color: #94a5b8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    QLabel#TopBarTitle {
        font-size: 10px;
    }
    QLabel#DisplayValue {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 700;
    }
    QLabel#ContextValue {
        color: #f4f7fb;
        font-size: 14px;
        font-weight: 600;
    }
    QLabel#MetricCardValue {
        color: #f8fbff;
        font-size: 22px;
        font-weight: 700;
    }
    QLabel#CompactMeta {
        color: #a2b2c3;
        font-size: 11px;
    }
    QLabel#MicroMeta {
        color: #7f92a8;
        font-size: 10px;
    }
    QPushButton {
        background: #1b3349;
        color: #f8fbff;
        border: 1px solid #274662;
        border-radius: 4px;
        padding: 4px 9px;
        min-height: 24px;
        min-width: 56px;
        font-weight: 500;
    }
    QPushButton:hover { background: #21415b; }
    QPushButton[liveState="active"] { background: #1f6e49; border-color: #31895d; }
    QPushButton[liveState="stop"] { background: #612d37; border-color: #87414f; }
    QPushButton[variant="quiet"] {
        background: #111822;
        border-color: #223143;
        color: #d6e1ee;
    }
    QComboBox, QListWidget, QTextEdit {
        background: #0b1117;
        color: #edf2f7;
        border: 1px solid #1c2a3a;
        border-radius: 4px;
        padding: 3px 6px;
    }
    QComboBox {
        min-width: 104px;
        min-height: 24px;
    }
    QListWidget#RoiQueue {
        background: #0b1118;
        border: 1px solid #1d2938;
        padding: 2px;
    }
    QListWidget::item {
        padding: 7px 9px;
        border: 1px solid #182332;
        margin-bottom: 3px;
    }
    QListWidget::item:hover {
        background: #112234;
        border: 1px solid #2a4561;
    }
    QListWidget::item:selected {
        background: #14304b;
        border: 1px solid #2f5d89;
    }
    QCheckBox {
        spacing: 4px;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border-radius: 2px;
        border: 1px solid #23384f;
        background: #0a121b;
    }
    QCheckBox::indicator:checked {
        background: #1d4b7a;
        border: 1px solid #4477b0;
    }
    QCheckBox#OverlayRoiToggle::indicator:checked {
        background: #63d27b;
        border: 1px solid #79e491;
    }
    QCheckBox#OverlayGazeToggle::indicator:checked {
        background: #18cfff;
        border: 1px solid #53dcff;
    }
    QCheckBox#OverlayHeatmapToggle::indicator:checked {
        background: #ff9c2c;
        border: 1px solid #ffc35d;
    }
    QCheckBox#OverlayScanpathToggle::indicator:checked {
        background: #ffcc53;
        border: 1px solid #ffe08d;
    }
    QTextEdit {
        min-height: 120px;
    }
    QSlider::groove:horizontal {
        height: 5px;
        background: #1b293a;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
        background: #d8e2ec;
    }
    QFrame#ViewerViewport {
        background: #05080d;
        border: 1px solid #131b25;
        border-radius: 3px;
    }
    QWidget#ViewerFooter {
        background: #0a1018;
        border-top: 1px solid #111a25;
        border-radius: 2px;
    }
    GazeTimelineWidget, QWidget#TimelineShell {
        background: #0b1219;
        border: 1px solid #172233;
        border-radius: 3px;
    }
    QScrollArea {
        border: 0;
        background: transparent;
    }
    QScrollBar:vertical {
        background: #0c1118;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #233242;
        min-height: 24px;
    }
    """
