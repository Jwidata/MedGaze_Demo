"""Evaluate behavior-learning models."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


def metric_row(name: str, model_name: str, y_true: Iterable[str], y_pred: Iterable[str], y_proba=None) -> dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    confidence = []
    if y_proba is not None:
        confidence = [float(max(row)) for row in y_proba]
    return {
        "feature_set": name,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mean_confidence": sum(confidence) / len(confidence) if confidence else 0.0,
        "min_confidence": min(confidence) if confidence else 0.0,
        "max_confidence": max(confidence) if confidence else 0.0,
        "per_class_metrics_json": json.dumps({label: report[label] for label in labels}, sort_keys=True),
        "confusion_matrix_json": json.dumps({"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}, sort_keys=True),
    }


def write_classification_report(path, y_true, y_pred) -> None:
    path.write_text("# Behavior Classification Report\n\n```text\n" + classification_report(y_true, y_pred, zero_division=0) + "```\n", encoding="utf-8")


def confusion_matrix_dataframe(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
