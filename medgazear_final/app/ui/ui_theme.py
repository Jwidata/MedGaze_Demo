"""Shared UI theme constants."""

from __future__ import annotations


LIMITATION_BANNER = "Research prototype. Not for clinical diagnosis."
TOBII_PLACEHOLDER_MESSAGE = "Tobii live mode requires SDK integration. Tobii SDK integration is planned in Step 13. Please calibrate in Tobii Manager before live capture."


def stylesheet() -> str:
    return """
    QMainWindow, QWidget { background: #111722; color: #eef3fb; font-family: Arial; }
    QLabel#Banner { background: #6d321d; color: #fff3e8; padding: 8px; font-weight: bold; }
    QPushButton { background: #28476f; color: white; border: 0; border-radius: 5px; padding: 7px; }
    QPushButton:hover { background: #386294; }
    QComboBox, QListWidget, QTextEdit { background: #182233; color: #eef3fb; border: 1px solid #33445f; }
    QGroupBox { border: 1px solid #33445f; border-radius: 8px; margin-top: 8px; padding: 8px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QListWidget::item { padding: 5px; }
    QListWidget::item:selected { background: #265f9d; }
    QTabWidget::pane { border: 1px solid #2e3d55; top: -1px; }
    QTabBar::tab { background: #1b2638; color: #dbe8f8; padding: 10px 18px; border: 1px solid #2e3d55; border-bottom: none; }
    QTabBar::tab:selected { background: #2f5f94; color: #ffffff; font-weight: bold; }
    QTabBar::tab:hover { background: #27466a; }
    """
