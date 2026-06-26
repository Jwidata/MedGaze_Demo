"""Markdown report generation for gaze visualizations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_visualization_report(path: Path, selected_cases: pd.DataFrame, interactive_enabled: bool = False) -> None:
    counts = selected_cases["hidden_behavior_label"].value_counts().to_dict() if not selected_cases.empty else {}
    lines = [
        "# Gaze Visualization Report",
        "",
        "These are source-agnostic gaze overlays. Synthetic gaze and future Tobii gaze use the same visualization schema and rendering pipeline.",
        "",
        "Heatmaps are generated in image/CT coordinates, not screen-only coordinates.",
        "",
        "If CT pixels are not loaded, the renderer uses a grayscale placeholder canvas with the correct image size and ROI overlay.",
        "",
        "Exact real Tobii validation is future work. Synthetic and real gaze should be compared by behavior-level features and heatmap similarity, not exact point matching.",
        "",
        f"Interactive guided report generated: {'yes' if interactive_enabled else 'no'}",
        "",
        "## Representative Cases",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Outputs", ""])
    lines.extend(
        [
            "Each selected example includes ROI overlay, gaze point overlay, heatmap overlay, scanpath overlay, and combined overlay PNGs.",
            "The optional interactive report is standalone HTML with plain HTML/CSS/JavaScript and no server requirement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
