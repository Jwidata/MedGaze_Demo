"""Synthetic replay adapter using the canonical gaze schema."""

from __future__ import annotations

import pandas as pd


def _series_or_default(rows: pd.DataFrame, column: str, default: object = 0.0) -> pd.Series:
    if column in rows:
        return rows[column]
    return pd.Series([default] * len(rows), index=rows.index)


def canonicalize_synthetic_samples(samples: pd.DataFrame, ct_stack_index: int | None = None) -> pd.DataFrame:
    rows = samples.copy()
    if rows.empty:
        return rows
    rows["source_type"] = "synthetic_replay"
    rows["screen_x"] = pd.to_numeric(_series_or_default(rows, "screen_x", _series_or_default(rows, "image_x", 0.0)), errors="coerce").fillna(0.0)
    rows["screen_y"] = pd.to_numeric(_series_or_default(rows, "screen_y", _series_or_default(rows, "image_y", 0.0)), errors="coerce").fillna(0.0)
    rows["gaze_x_norm"] = pd.to_numeric(_series_or_default(rows, "gaze_x_norm", 0.0), errors="coerce").fillna(0.0)
    rows["gaze_y_norm"] = pd.to_numeric(_series_or_default(rows, "gaze_y_norm", 0.0), errors="coerce").fillna(0.0)
    rows["image_x"] = pd.to_numeric(_series_or_default(rows, "image_x", 0.0), errors="coerce").fillna(0.0)
    rows["image_y"] = pd.to_numeric(_series_or_default(rows, "image_y", 0.0), errors="coerce").fillna(0.0)
    rows["is_valid"] = _series_or_default(rows, "is_valid", True).astype(bool)
    rows["is_dropout"] = ~rows["is_valid"]
    rows["is_blink"] = _series_or_default(rows, "is_blink", False).astype(bool)
    rows["is_outside_ct"] = _series_or_default(rows, "is_outside_ct", False).astype(bool)
    rows["is_ui_glance"] = _series_or_default(rows, "is_ui_glance", False).astype(bool)
    if ct_stack_index is not None:
        rows["ct_stack_index"] = int(ct_stack_index)
    elif "ct_stack_index" not in rows.columns:
        rows["ct_stack_index"] = pd.to_numeric(_series_or_default(rows, "slice_index", 0), errors="coerce").fillna(0).astype(int)
    return rows
