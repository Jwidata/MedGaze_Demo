"""Robustness audits for behavior-learning models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml_behavior.behavior_feature_schema import allowed_feature_columns


HARD_ABLATION_REMOVE = {
    "time_to_first_roi_fixation_ms",
    "mean_fixation_duration_inside_roi_ms",
    "max_fixation_duration_inside_roi_ms",
    "mean_fixation_duration_ms",
    "max_fixation_duration_ms",
    "fixation_duration_variance",
    "time_on_roi_slice_ms",
    "valid_gaze_time_on_roi_slice_ms",
    "scanpath_length_px",
    "scanpath_length_on_roi_slice_px",
    "outside_ct_ratio",
    "dropout_ratio",
    "blink_ratio",
    "invalid_burst_ratio",
}


def run_behavior_robustness_audit(
    dataset: pd.DataFrame,
    model_path: Path,
    external_dataset: pd.DataFrame | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    feature_columns = allowed_feature_columns(dataset)
    rows: list[dict[str, object]] = []
    rows.append(_group_split_audit(dataset, feature_columns, "case_id", "case_id_group_split", seed))
    rows.append(_group_split_audit(dataset, feature_columns, "reader_id", "reader_held_out_split", seed))
    rows.append(_hard_ablation_audit(dataset, feature_columns, seed))
    rows.extend(_noise_stress_audit(dataset, model_path, feature_columns, seed))
    rows.append(_external_seed_audit(dataset, external_dataset, feature_columns, seed))
    return pd.DataFrame(rows)


def write_robustness_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# Behavior Robustness Audit Report",
        "",
        "Perfect score on a random split means synthetic classes are highly separable.",
        "This is not clinical validation.",
        "Robustness tests are needed before using the model in the UI.",
        "If group/reader/new-seed scores drop, report generalization limits honestly.",
        "",
        "## Results",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"- {row['audit_name']}: macro_f1={float(row['macro_f1']):.3f}, balanced_accuracy={float(row['balanced_accuracy']):.3f}, note={row.get('note', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_split_audit(dataset: pd.DataFrame, features: list[str], group_column: str, audit_name: str, seed: int) -> dict[str, object]:
    if group_column not in dataset or dataset[group_column].nunique() < 2:
        return _skipped(audit_name, f"not enough {group_column} groups")
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.25, random_state=seed).split(dataset, dataset["hidden_behavior_label"], groups=dataset[group_column]))
    return _fit_eval(audit_name, dataset.iloc[train_idx], dataset.iloc[test_idx], features, "no shared groups", seed)


def _hard_ablation_audit(dataset: pd.DataFrame, features: list[str], seed: int) -> dict[str, object]:
    ablated = [feature for feature in features if feature not in HARD_ABLATION_REMOVE]
    train, test = train_test_split(dataset, test_size=0.25, random_state=seed, stratify=dataset["hidden_behavior_label"])
    return _fit_eval("hard_feature_ablation", train, test, ablated, f"removed {len(features) - len(ablated)} generator-separation features", seed)


def _noise_stress_audit(dataset: pd.DataFrame, model_path: Path, features: list[str], seed: int) -> list[dict[str, object]]:
    model = joblib.load(model_path)
    X = dataset[features].fillna(0).astype(float)
    y = dataset["hidden_behavior_label"].astype(str)
    rows = []
    for name, scale in (("low_noise_stress", 0.03), ("medium_noise_stress", 0.10), ("high_noise_stress", 0.25)):
        noisy = _add_noise(X, scale, seed)
        pred = _predict_labels(model, noisy)
        rows.append(_metrics(name, y, pred, f"gaussian noise scale={scale}"))
    return rows


def _external_seed_audit(dataset: pd.DataFrame, external_dataset: pd.DataFrame | None, features: list[str], seed: int) -> dict[str, object]:
    train, test = train_test_split(dataset, test_size=0.25, random_state=seed, stratify=dataset["hidden_behavior_label"])
    if external_dataset is None:
        proxy_train, proxy_test = train_test_split(dataset, test_size=0.25, random_state=99, stratify=dataset["hidden_behavior_label"])
        return _fit_eval("new_seed_external_test_proxy", proxy_train, proxy_test, features, "no external seed-99 dataset supplied; used seed=99 split proxy", seed)
    common = [feature for feature in features if feature in external_dataset.columns]
    return _fit_eval("new_seed_external_test", train, external_dataset, common, "external dataset supplied", seed)


def _fit_eval(audit_name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str], note: str, seed: int) -> dict[str, object]:
    if not features:
        return _skipped(audit_name, "no features available")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    model.fit(train[features].fillna(0).astype(float), train["hidden_behavior_label"].astype(str))
    pred = model.predict(test[features].fillna(0).astype(float))
    return _metrics(audit_name, test["hidden_behavior_label"].astype(str), pred, note)


def _metrics(audit_name: str, y_true, y_pred, note: str) -> dict[str, object]:
    return {
        "audit_name": audit_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "note": note,
    }


def _predict_labels(model, X: pd.DataFrame) -> list[str]:
    pred = model.predict(X)
    if hasattr(model, "_medgazear_label_encoder"):
        return list(model._medgazear_label_encoder.inverse_transform(pred))
    return [str(item) for item in pred]


def _add_noise(X: pd.DataFrame, scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    std = X.std(axis=0).replace(0, 1).to_numpy()
    noise = rng.normal(0, scale, X.shape) * std
    return pd.DataFrame(X.to_numpy() + noise, columns=X.columns)


def _skipped(audit_name: str, reason: str) -> dict[str, object]:
    return {"audit_name": audit_name, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "note": f"skipped: {reason}"}
