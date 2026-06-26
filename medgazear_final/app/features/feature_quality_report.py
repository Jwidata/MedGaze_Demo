"""Feature extraction report writer."""

from __future__ import annotations

from pathlib import Path


def write_feature_quality_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Feature Quality Report", "", f"- roi/session feature rows: {len(rows)}"]
    if rows:
        lines.append(f"- unique sessions: {len({row['session_id'] for row in rows})}")
        lines.append(f"- unique ROIs: {len({row['roi_id'] for row in rows})}")
        lines.append(f"- mean gaze validity ratio: {_mean([float(row['gaze_validity_ratio']) for row in rows])}")
        lines.extend(["", "## valid_gaze_time_on_roi_slice_ms by hidden_behavior_label"])
        for label, stats in _describe_by_label(rows, "valid_gaze_time_on_roi_slice_ms").items():
            lines.append(f"- {label}: count={stats['count']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _describe_by_label(rows: list[dict[str, object]], value_key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["hidden_behavior_label"]), []).append(float(row[value_key]))
    return {
        label: {
            "count": float(len(values)),
            "mean": _mean(values),
            "min": min(values),
            "max": max(values),
        }
        for label, values in sorted(grouped.items())
    }
