"""Frozen-model local explanations for workstation predictions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Mapping

import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_LABELS = {
    "total_gaze_time_inside_roi_ms": "Exact ROI dwell",
    "total_gaze_time_near_roi_ms": "Near-ROI dwell",
    "gaze_hit_count_inside_roi": "Exact ROI hits",
    "gaze_hit_count_near_roi": "Near-ROI hits",
    "fixation_count_inside_roi": "ROI fixation count",
    "fixation_count_near_roi": "Near-ROI fixation count",
    "mean_fixation_duration_inside_roi_ms": "Mean ROI fixation",
    "max_fixation_duration_inside_roi_ms": "Maximum fixation duration",
    "time_to_first_roi_fixation_ms": "Time to first ROI fixation",
    "valid_gaze_time_on_roi_slice_ms": "Usable slice dwell",
    "time_on_roi_slice_ms": "Slice viewing time",
    "scanpath_length_px": "Scanpath length",
    "scanpath_length_on_roi_slice_px": "On-slice scanpath length",
    "gaze_dispersion_px": "Scanpath dispersion",
    "gaze_entropy": "Gaze entropy",
    "number_of_gaze_clusters": "Gaze clusters",
    "background_gaze_ratio": "Outside-ROI gaze ratio",
    "roi_revisit_count": "ROI revisits",
    "near_roi_revisit_count": "Near-ROI revisits",
    "slice_transition_count": "Slice transitions",
    "adjacent_slice_toggle_count": "Adjacent slice toggles",
    "scroll_event_count": "Scroll events",
    "search_to_confirmation_ratio": "Search-to-confirm ratio",
    "late_roi_discovery_flag": "Late ROI discovery",
    "mean_fixation_duration_ms": "Mean fixation duration",
    "max_fixation_duration_ms": "Maximum fixation duration",
    "fixation_duration_variance": "Fixation duration variance",
    "saccade_like_ratio": "Saccade-like ratio",
    "fixation_like_ratio": "Fixation-like ratio",
    "first_half_roi_attention_ratio": "Early ROI attention",
    "second_half_roi_attention_ratio": "Late ROI attention",
    "delayed_attention_score": "Delayed attention",
    "gaze_validity_ratio": "Gaze validity",
    "dropout_ratio": "Dropout ratio",
    "blink_ratio": "Blink ratio",
    "invalid_burst_ratio": "Invalid burst ratio",
    "outside_ct_ratio": "Outside-CT ratio",
    "jitter_px": "Gaze instability",
    "roi_area_px": "ROI area",
    "roi_bbox_width": "ROI width",
    "roi_bbox_height": "ROI height",
    "roi_center_x": "ROI center X",
    "roi_center_y": "ROI center Y",
    "normalized_roi_position_x": "Normalized ROI X",
    "normalized_roi_position_y": "Normalized ROI Y",
    "number_of_rois_on_slice": "ROI targets on slice",
    "roi_density_context": "ROI density context",
    "slice_index": "Slice index",
}


@dataclass(frozen=True)
class BackgroundReference:
    rows: pd.DataFrame
    metadata: dict[str, object]
    class_feature_stats: dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class LocalExplanation:
    predicted_label: str
    predicted_probability: float
    class_probabilities: dict[str, float]
    expected_value: float
    top_features: list[dict[str, object]]
    full_features: list[dict[str, object]]
    competing_label: str | None
    competing_probability: float | None
    model_kind: str


class BehaviorExplanationService:
    def __init__(self, behavior_dataset: pd.DataFrame, feature_columns: list[str], model, label_mapping: Mapping[str, str]) -> None:
        self.behavior_dataset = behavior_dataset.copy()
        self.feature_columns = list(feature_columns)
        self.model = model
        self.label_mapping = dict(label_mapping)
        self.background = build_background_reference(self.behavior_dataset, self.feature_columns)

    def explain_live_row(self, row: Mapping[str, object]) -> LocalExplanation | None:
        if self.model is None or not self.feature_columns:
            return None
        values = []
        for column in self.feature_columns:
            value = row.get(column)
            if value is None or pd.isna(value):
                return None
            values.append(float(value))
        X = pd.DataFrame([values], columns=self.feature_columns)
        class_probabilities = _predict_probabilities(self.model, X)
        if not class_probabilities:
            return None
        predicted_label = max(class_probabilities, key=class_probabilities.get)
        predicted_probability = float(class_probabilities[predicted_label])
        sorted_classes = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
        competing_label = sorted_classes[1][0] if len(sorted_classes) > 1 else None
        competing_probability = float(sorted_classes[1][1]) if len(sorted_classes) > 1 else None
        if isinstance(self.model, xgb.XGBClassifier):
            explanation = _xgboost_explanation(self.model, X, self.feature_columns, predicted_label)
            model_kind = "tree_xgboost"
        else:
            explanation = _linear_fallback_explanation(self.model, X, self.feature_columns, predicted_label)
            model_kind = "linear_fallback"
        return LocalExplanation(
            predicted_label=predicted_label,
            predicted_probability=predicted_probability,
            class_probabilities=class_probabilities,
            expected_value=float(explanation["expected_value"]),
            top_features=explanation["top_features"],
            full_features=explanation["full_features"],
            competing_label=competing_label,
            competing_probability=competing_probability,
            model_kind=model_kind,
        )


def build_background_reference(behavior_dataset: pd.DataFrame, feature_columns: list[str], rows_per_class: int = 24) -> BackgroundReference:
    working = behavior_dataset.copy()
    if "hidden_behavior_label" not in working.columns:
        working["hidden_behavior_label"] = "unknown"
    sampled_parts = []
    for label, group in working.groupby("hidden_behavior_label", sort=True):
        sampled_parts.append(group.head(rows_per_class).copy())
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else working.head(0).copy()
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for label, group in working.groupby("hidden_behavior_label", sort=True):
        stats[str(label)] = {}
        for feature in feature_columns:
            numeric = pd.to_numeric(group.get(feature, pd.Series(dtype=float)), errors="coerce").dropna()
            if numeric.empty:
                continue
            stats[str(label)][feature] = {
                "median": float(numeric.median()),
                "iqr_low": float(numeric.quantile(0.25)),
                "iqr_high": float(numeric.quantile(0.75)),
            }
    version_payload = f"{len(sampled)}|{'|'.join(feature_columns)}|{'|'.join(sorted(stats))}"
    metadata = {
        "background_dataset_version": sha1(version_payload.encode("utf-8")).hexdigest()[:12],
        "sampling_method": f"head_{rows_per_class}_per_behavior_class",
        "feature_schema_version": sha1('|'.join(feature_columns).encode('utf-8')).hexdigest()[:12],
        "background_rows": int(len(sampled)),
        "source": "synthetic_training_records",
    }
    return BackgroundReference(rows=sampled[[column for column in feature_columns if column in sampled.columns]].copy(), metadata=metadata, class_feature_stats=stats)


def _predict_probabilities(model, X: pd.DataFrame) -> dict[str, float]:
    predicted = model.predict(X)
    if hasattr(model, "_medgazear_label_encoder"):
        predicted = model._medgazear_label_encoder.inverse_transform(predicted)
        classes = [str(label) for label in model._medgazear_label_encoder.classes_]
    else:
        classes = [str(label) for label in getattr(model, "classes_", [])]
    probabilities = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else np.zeros(len(classes), dtype=float)
    return {label: float(probability) for label, probability in zip(classes, probabilities)}


def _xgboost_explanation(model: xgb.XGBClassifier, X: pd.DataFrame, feature_columns: list[str], predicted_label: str) -> dict[str, object]:
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_columns)
    contributions = booster.predict(dmatrix, pred_contribs=True)[0]
    if hasattr(model, "_medgazear_label_encoder"):
        classes = [str(label) for label in model._medgazear_label_encoder.classes_]
    else:
        classes = [str(label) for label in model.classes_]
    class_index = classes.index(predicted_label) if predicted_label in classes else 0
    class_values = contributions[class_index]
    expected_value = float(class_values[-1])
    feature_values = []
    for feature, shap_value, live_value in zip(feature_columns, class_values[:-1], X.iloc[0].tolist()):
        feature_values.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ")),
                "live_value": float(live_value),
                "shap_value": float(shap_value),
                "direction": "supports" if float(shap_value) >= 0 else "opposes",
            }
        )
    ordered = sorted(feature_values, key=lambda item: abs(float(item["shap_value"])), reverse=True)
    return {"expected_value": expected_value, "top_features": ordered[:5], "full_features": ordered}


def _linear_fallback_explanation(model, X: pd.DataFrame, feature_columns: list[str], predicted_label: str) -> dict[str, object]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    classes = [str(label) for label in getattr(estimator, "classes_", [])]
    class_index = classes.index(predicted_label) if predicted_label in classes else 0
    coefficients = estimator.coef_[class_index] if hasattr(estimator, "coef_") else np.zeros(len(feature_columns), dtype=float)
    feature_values = []
    for feature, coefficient, live_value in zip(feature_columns, coefficients, X.iloc[0].tolist()):
        shap_value = float(coefficient) * float(live_value)
        feature_values.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ")),
                "live_value": float(live_value),
                "shap_value": shap_value,
                "direction": "supports" if shap_value >= 0 else "opposes",
            }
        )
    ordered = sorted(feature_values, key=lambda item: abs(float(item["shap_value"])), reverse=True)
    return {"expected_value": 0.0, "top_features": ordered[:5], "full_features": ordered}
