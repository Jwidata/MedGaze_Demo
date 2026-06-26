"""Feature preparation for gaze-derived cognitive-load proxy analysis."""

from __future__ import annotations

import pandas as pd


PROXY_FEATURE_WEIGHTS: dict[str, float] = {
    "gaze_dispersion_px": 1.0,
    "scanpath_length_px": 1.0,
    "roi_revisit_count": 0.9,
    "near_roi_revisit_count": 0.8,
    "slice_transition_count": 0.9,
    "adjacent_slice_toggle_count": 0.9,
    "delayed_attention_score": 1.0,
    "fixation_duration_variance": 0.8,
    "saccade_like_ratio": 0.8,
    "dropout_ratio": 0.35,
    "blink_ratio": 0.35,
    "invalid_burst_ratio": 0.35,
    "outside_ct_ratio": 0.45,
    "background_gaze_ratio": 0.8,
}

IDENTIFIER_COLUMNS = ["session_id", "roi_id", "case_id", "reader_id", "hidden_behavior_label"]


def available_proxy_features(features: pd.DataFrame) -> list[str]:
    return [column for column in PROXY_FEATURE_WEIGHTS if column in features.columns and pd.api.types.is_numeric_dtype(features[column])]


def build_cognitive_proxy_features(features: pd.DataFrame) -> pd.DataFrame:
    proxy_columns = available_proxy_features(features)
    if not proxy_columns:
        raise ValueError("No cognitive-load proxy feature columns are available.")

    rows = features[[column for column in IDENTIFIER_COLUMNS if column in features.columns]].copy()
    weighted_sum = pd.Series(0.0, index=features.index)
    total_weight = 0.0
    for column in proxy_columns:
        rank_column = f"{column}_percentile"
        rows[rank_column] = _percentile_rank(features[column])
        weight = PROXY_FEATURE_WEIGHTS[column]
        weighted_sum = weighted_sum + rows[rank_column] * weight
        total_weight += weight
    rows["cognitive_load_proxy_score"] = weighted_sum / max(total_weight, 1e-9)
    rows["proxy_feature_count"] = len(proxy_columns)
    return rows


def _percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(series.median() if series.notna().any() else 0)
    if numeric.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return numeric.rank(method="average", pct=True)
