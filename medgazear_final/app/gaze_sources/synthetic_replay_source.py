"""Synthetic replay adapter using the canonical gaze schema."""

from __future__ import annotations

import pandas as pd


def canonicalize_synthetic_samples(samples: pd.DataFrame, ct_stack_index: int | None = None) -> pd.DataFrame:
    rows = samples.copy()
    if rows.empty:
        return rows
    rows["source_type"] = "synthetic_replay"
    rows["screen_x"] = pd.to_numeric(rows.get("screen_x", rows.get("image_x", 0)), errors="coerce").fillna(0.0)
    rows["screen_y"] = pd.to_numeric(rows.get("screen_y", rows.get("image_y", 0)), errors="coerce").fillna(0.0)
    rows["gaze_x_norm"] = pd.to_numeric(rows.get("gaze_x_norm", 0), errors="coerce").fillna(0.0)
    rows["gaze_y_norm"] = pd.to_numeric(rows.get("gaze_y_norm", 0), errors="coerce").fillna(0.0)
    rows["image_x"] = pd.to_numeric(rows.get("image_x", 0), errors="coerce").fillna(0.0)
    rows["image_y"] = pd.to_numeric(rows.get("image_y", 0), errors="coerce").fillna(0.0)
    rows["is_valid"] = rows.get("is_valid", True).astype(bool)
    rows["is_dropout"] = ~rows["is_valid"]
    rows["is_blink"] = rows.get("is_blink", False).astype(bool) if "is_blink" in rows else False
    rows["is_outside_ct"] = rows.get("is_outside_ct", False).astype(bool) if "is_outside_ct" in rows else False
    rows["is_ui_glance"] = rows.get("is_ui_glance", False).astype(bool) if "is_ui_glance" in rows else False
    if ct_stack_index is not None:
        rows["ct_stack_index"] = int(ct_stack_index)
    return rows
