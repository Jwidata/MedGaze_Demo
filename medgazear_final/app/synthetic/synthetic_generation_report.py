"""Report writers for synthetic gaze generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def write_quality_report(
    path: Path,
    rows: list[dict[str, object]],
    session_rows: list[dict[str, object]],
) -> dict[str, float | int | dict[str, int] | dict[str, dict[str, float]]]:
    total = len(rows)
    sample_labels = Counter(str(row["hidden_behavior_label"]) for row in rows)
    session_labels = Counter(str(row["hidden_behavior_label"]) for row in session_rows)
    profiles = Counter(str(row["reader_profile"]) for row in rows)
    summary = {
        "total_samples": total,
        "number_of_sessions": len(session_rows),
        "number_of_unique_rois": len({str(row["roi_id"]) for row in rows}),
        "label_distribution_by_sample": dict(sample_labels),
        "label_distribution_by_session": dict(session_labels),
        "reader_profile_distribution": dict(profiles),
        "duration_ms_by_hidden_behavior_label": _describe_by_label(session_rows, "duration_ms"),
        "samples_per_session_by_hidden_behavior_label": _describe_by_label(session_rows, "sample_count"),
        "mean_validity_ratio": _ratio(rows, "is_valid"),
        "dropout_ratio": _ratio(rows, "is_dropout"),
        "blink_ratio": _ratio(rows, "is_blink"),
        "invalid_burst_ratio": _ratio(rows, "is_invalid_burst"),
        "outside_ct_ratio": _ratio(rows, "is_outside_ct"),
        "ui_glance_ratio": _ratio(rows, "is_ui_glance"),
        "mean_jitter_px": _mean_abs_pair(rows, "jitter_x", "jitter_y"),
        "mean_calibration_offset_px": _mean_abs_pair(rows, "calibration_offset_x", "calibration_offset_y"),
        "mean_drift_px": _mean_abs_pair(rows, "drift_x", "drift_y"),
    }
    lines = ["# Synthetic Gaze Quality Report", ""]
    for key, value in summary.items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_behavior_generation_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Behavior Generation Report",
        "",
        "Hidden behavior labels are synthetic ground truth labels and are stored separately from future rule-based labels.",
        "No aggregate gaze features are generated in this step.",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ratio(rows: list[dict[str, object]], key: str) -> float:
    return 0.0 if not rows else sum(1 for row in rows if bool(row[key])) / len(rows)


def _mean_abs_pair(rows: list[dict[str, object]], x_key: str, y_key: str) -> float:
    if not rows:
        return 0.0
    return sum((float(row[x_key]) ** 2 + float(row[y_key]) ** 2) ** 0.5 for row in rows) / len(rows)


def _describe_by_label(rows: list[dict[str, object]], value_key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["hidden_behavior_label"]), []).append(float(row[value_key]))
    return {
        label: {
            "count": float(len(values)),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
        for label, values in sorted(grouped.items())
    }
