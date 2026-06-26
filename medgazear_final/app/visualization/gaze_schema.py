"""Canonical gaze sample schema shared by synthetic and future Tobii sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


GAZE_SCHEMA_COLUMNS = [
    "source_type",
    "session_id",
    "case_id",
    "roi_id",
    "slice_index",
    "timestamp_ms",
    "image_x",
    "image_y",
    "screen_x",
    "screen_y",
    "is_valid",
    "is_outside_ct",
    "is_ui_glance",
]


def load_gaze_samples(path: str | Path, source_type: str = "synthetic") -> pd.DataFrame:
    samples = pd.read_csv(path)
    return normalize_gaze_samples(samples, source_type=source_type)


def normalize_gaze_samples(samples: pd.DataFrame, source_type: str = "synthetic") -> pd.DataFrame:
    rows = samples.copy()
    if "source_type" not in rows.columns:
        rows["source_type"] = source_type
    missing = [column for column in GAZE_SCHEMA_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"Gaze samples missing required schema columns: {missing}")
    rows = rows[GAZE_SCHEMA_COLUMNS].copy()
    for column in ("session_id", "case_id", "roi_id", "source_type"):
        rows[column] = rows[column].astype(str)
    for column in ("slice_index", "timestamp_ms", "image_x", "image_y", "screen_x", "screen_y"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ("is_valid", "is_outside_ct", "is_ui_glance"):
        rows[column] = rows[column].map(_to_bool)
    if rows[["session_id", "case_id", "roi_id"]].isna().any().any():
        raise ValueError("Gaze samples contain missing identifiers.")
    if rows[["image_x", "image_y", "timestamp_ms"]].isna().any().any():
        raise ValueError("Gaze samples contain missing image coordinates or timestamps.")
    return rows


def _to_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}
