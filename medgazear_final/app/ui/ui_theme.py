"""Shared UI theme constants."""

from __future__ import annotations


LIMITATION_BANNER = "Research prototype. Not for clinical diagnosis."
TOBII_PLACEHOLDER_MESSAGE = "Tobii live mode requires SDK integration. Tobii SDK integration is planned in Step 13. Please calibrate in Tobii Manager before live capture."


def stylesheet() -> str:
    return """
    QMainWindow, QWidget { background: #111722; color: #eef3fb; font-family: Arial; }
    QLabel#Banner { background: #6d321d; color: #fff3e8; padding: 8px; font-weight: bold; }
    QPushButton { background: #28476f; color: white; border: 0; border-radius: 5px; padding: 5px 8px; min-height: 26px; }
    QPushButton:hover { background: #386294; }
    QPushButton[liveState="active"] { background: #1f8f55; }
    QPushButton[liveState="active"]:hover { background: #26a563; }
    QPushButton[liveState="stop"] { background: #6b3040; }
    QPushButton[liveState="stop"]:hover { background: #864055; }
    QComboBox, QListWidget, QTextEdit, QScrollArea { background: #182233; color: #eef3fb; border: 1px solid #33445f; min-height: 24px; }
    QGroupBox { border: 1px solid #33445f; border-radius: 8px; margin-top: 8px; padding: 8px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QListWidget::item { padding: 5px; }
    QListWidget::item:selected { background: #265f9d; }
    QLabel#TileTitle { color: #a7bdd8; font-size: 11px; font-weight: bold; }
    QLabel#TileValue { color: #ffffff; font-size: 20px; font-weight: bold; }
    QGroupBox#MetricTile { background: #162235; border: 1px solid #2c4263; border-radius: 10px; margin-top: 0px; }
    QFrame#ViewerViewport { background: #05070b; border: 1px solid #1a2433; border-radius: 6px; }
    QWidget#ViewerFooter { background: #0b1220; border-top: 1px solid #22324a; border-radius: 6px; }
    """
