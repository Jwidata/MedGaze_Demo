"""Train behavior-learning models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from app.ml_behavior.behavior_feature_schema import allowed_feature_columns
from app.ml_behavior.behavior_model_registry import behavior_models
from app.ml_behavior.evaluate_behavior_models import metric_row


@dataclass(frozen=True)
class TrainResult:
    results: pd.DataFrame
    test_results: pd.DataFrame
    best_model: object
    best_model_name: str
    feature_columns: list[str]
    test_data: tuple[pd.DataFrame, pd.Series]
    predictions: list[str]
    split_summary: dict[str, object]


def split_dataset(dataset: pd.DataFrame, seed: int = 42, split: str = "stratified"):
    y = dataset["hidden_behavior_label"].astype(str)
    if split == "case_group" and dataset["case_id"].nunique() > 1:
        train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(dataset, y, groups=dataset["case_id"]))
        train = dataset.iloc[train_idx]
        temp = dataset.iloc[test_idx]
    elif split == "reader_held_out" and dataset["reader_id"].nunique() > 1:
        train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=seed).split(dataset, y, groups=dataset["reader_id"]))
        train = dataset.iloc[train_idx]
        temp = dataset.iloc[test_idx]
    else:
        train, temp = train_test_split(dataset, test_size=0.30, random_state=seed, stratify=y if y.value_counts().min() >= 2 else None)
    valid, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp["hidden_behavior_label"] if temp["hidden_behavior_label"].value_counts().min() >= 2 else None)
    return train, valid, test


def train_models(dataset: pd.DataFrame, feature_columns: list[str] | None = None, seed: int = 42, split: str = "stratified") -> TrainResult:
    feature_columns = feature_columns or allowed_feature_columns(dataset)
    train, valid, test = split_dataset(dataset, seed, split)
    X_train = train[feature_columns].fillna(0).astype(float)
    y_train = train["hidden_behavior_label"].astype(str)
    X_valid = valid[feature_columns].fillna(0).astype(float)
    y_valid = valid["hidden_behavior_label"].astype(str)
    X_test = test[feature_columns].fillna(0).astype(float)
    y_test = test["hidden_behavior_label"].astype(str)
    rows = []
    candidates = []
    for model_name, model in behavior_models(seed).items():
        try:
            fitted_model = _fit_model(model, model_name, X_train, y_train)
            pred, proba = _predict_model(fitted_model, X_valid)
            row = metric_row("all_behavior_features", model_name, y_valid, pred, proba)
            row["evaluation_split"] = "validation"
            rows.append(row)
            candidates.append((row["macro_f1"], row["balanced_accuracy"], model_name, model))
        except Exception as exc:
            rows.append({"feature_set": "all_behavior_features", "model": model_name, "evaluation_split": "validation", "accuracy": 0, "balanced_accuracy": 0, "macro_f1": 0, "weighted_f1": 0, "mean_confidence": 0, "min_confidence": 0, "max_confidence": 0, "per_class_metrics_json": "{}", "confusion_matrix_json": "{}", "skip_reason": str(exc)})
    if not candidates:
        raise RuntimeError("No behavior models were successfully fitted.")
    _best_f1, _best_balanced, best_model_name, best_template = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[feature_columns].fillna(0).astype(float)
    y_train_valid = train_valid["hidden_behavior_label"].astype(str)
    best_model = _fit_model(clone(best_template), best_model_name, X_train_valid, y_train_valid)
    best_pred, best_proba = _predict_model(best_model, X_test)
    test_row = metric_row("all_behavior_features", best_model_name, y_test, best_pred, best_proba)
    test_row["evaluation_split"] = "test"
    split_summary = {
        "split": split,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "train_validation_rows": len(train_valid),
        "selection_primary_metric": "validation_macro_f1",
        "selection_secondary_metric": "validation_balanced_accuracy",
    }
    return TrainResult(pd.DataFrame(rows), pd.DataFrame([test_row]), best_model, best_model_name, feature_columns, (X_test, y_test), list(best_pred), split_summary)


def _fit_model(model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name in {"XGBoostClassifier", "LightGBMClassifier"}:
        enc = LabelEncoder()
        model.fit(X_train, enc.fit_transform(y_train))
        model._medgazear_label_encoder = enc
    else:
        model.fit(X_train, y_train)
    return model


def _predict_model(model, X: pd.DataFrame):
    pred = model.predict(X)
    if hasattr(model, "_medgazear_label_encoder"):
        pred = model._medgazear_label_encoder.inverse_transform(pred)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return pred, proba
