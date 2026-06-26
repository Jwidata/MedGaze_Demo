"""Reports for behavior-learning models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DISCLAIMER = "The behavior model is trained on synthetic hidden behavior labels. It is experimental for live Tobii use until real Tobii/radiologist validation is performed. It does not make clinical diagnosis."


def write_behavior_model_card(path: Path, dataset: pd.DataFrame, comparison: pd.DataFrame, test_results: pd.DataFrame | None = None, split_summary: dict[str, object] | None = None) -> None:
    lines = ["# Behavior Learning Model Card", "", DISCLAIMER, "", f"Rows: {len(dataset)}", f"Class distribution: {dataset['hidden_behavior_label'].value_counts().to_dict()}", f"Models evaluated: {sorted(comparison['model'].dropna().unique().tolist())}"]
    lines.extend(["", "## Split And Model Selection", "Candidate models were trained on the training split and selected using validation macro F1, with validation balanced accuracy as the tie-breaker.", "Final reported performance is from the held-out test set. The test set was not used during model selection."])
    if split_summary:
        lines.append(f"Split rows: train={split_summary.get('train_rows')}, validation={split_summary.get('validation_rows')}, test={split_summary.get('test_rows')}, retrain_train_validation={split_summary.get('train_validation_rows')}.")
    if test_results is not None and not test_results.empty:
        test = test_results.iloc[0]
        lines.extend(["", "## Final Held-Out Test Performance", f"Selected model: {test['model']}", f"Test macro F1: {float(test['macro_f1']):.3f}", f"Test balanced accuracy: {float(test['balanced_accuracy']):.3f}"])
    interpretation = _mlp_interpretation(comparison)
    if interpretation:
        lines.extend(["", "## Lightweight Neural Baseline", interpretation])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_behavior_summary(path: Path, comparison: pd.DataFrame, test_results: pd.DataFrame | None = None) -> None:
    best = comparison.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    lines = ["# Behavior Learning Summary", "", DISCLAIMER, "", "Model selection used validation macro F1, with validation balanced accuracy as the tie-breaker.", "Final reported performance is from the held-out test set. The test set was not used during model selection.", "", f"Best model selected on validation: {best['model']}", f"Validation macro F1: {float(best['macro_f1']):.3f}", f"Validation balanced accuracy: {float(best['balanced_accuracy']):.3f}"]
    if test_results is not None and not test_results.empty:
        test = test_results.iloc[0]
        lines.extend(["", f"Final test model: {test['model']}", f"Final test macro F1: {float(test['macro_f1']):.3f}", f"Final test balanced accuracy: {float(test['balanced_accuracy']):.3f}"])
    interpretation = _mlp_interpretation(comparison)
    if interpretation:
        lines.extend(["", "## Lightweight Neural Baseline", interpretation])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mlp_interpretation(comparison: pd.DataFrame) -> str:
    if "model" not in comparison or comparison.empty:
        return ""
    mlp = _best_for_model(comparison, "MLPClassifier")
    logistic = _best_for_model(comparison, "LogisticRegression")
    if mlp is None:
        return "MLPClassifier was not evaluated."
    if logistic is None:
        return f"MLPClassifier macro F1={float(mlp['macro_f1']):.3f}; compare against classical baselines before using it as the main model."
    mlp_f1 = float(mlp["macro_f1"])
    logistic_f1 = float(logistic["macro_f1"])
    if mlp_f1 <= logistic_f1 + 0.01:
        return f"MLPClassifier macro F1={mlp_f1:.3f}, similar to LogisticRegression macro F1={logistic_f1:.3f}; prefer the simpler, more explainable classical model unless the neural baseline clearly improves performance."
    return f"MLPClassifier macro F1={mlp_f1:.3f}, higher than LogisticRegression macro F1={logistic_f1:.3f}; report it as a lightweight neural baseline while discussing interpretability limitations."


def _best_for_model(comparison: pd.DataFrame, model_name: str):
    rows = comparison[comparison["model"] == model_name]
    if rows.empty:
        return None
    return rows.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
