"""Model training and metric helpers for rule distillation."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def available_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    except Exception:
        pass
    return models


def evaluate_models_for_feature_set(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    feature_set_name: str,
    shuffled_labels: bool = False,
    random_state: int = 42,
) -> list[dict[str, object]]:
    if not feature_columns:
        return []
    X = dataset[feature_columns].fillna(0).astype(float)
    y = dataset["rule_attention_status"].astype(str).copy()
    if shuffled_labels:
        y = y.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        X = X.reset_index(drop=True)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=stratify)
    rows: list[dict[str, object]] = []
    for model_name, model in available_models(random_state).items():
        try:
            if model_name == "XGBoostClassifier":
                encoder = LabelEncoder()
                y_train_fit = encoder.fit_transform(y_train)
                model.fit(X_train, y_train_fit)
                predictions = encoder.inverse_transform(model.predict(X_test))
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            rows.append(_metric_row(feature_set_name, model_name, y_test, predictions, feature_columns, skipped=False))
        except Exception as exc:
            rows.append({"feature_set": feature_set_name, "model": model_name, "skipped": True, "skip_reason": str(exc), "feature_count": len(feature_columns)})
    return rows


def _metric_row(feature_set_name: str, model_name: str, y_true: Iterable[str], y_pred: Iterable[str], feature_columns: list[str], skipped: bool) -> dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "skipped": skipped,
        "skip_reason": "",
        "feature_count": len(feature_columns),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_metrics_json": json.dumps({label: report[label] for label in labels}, sort_keys=True),
        "confusion_matrix_json": json.dumps({"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}, sort_keys=True),
    }
