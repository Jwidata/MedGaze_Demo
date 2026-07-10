"""Evaluation-integrity helpers for grouped split auditing and comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from app.ml_behavior.behavior_feature_schema import allowed_feature_columns
from app.ml_behavior.behavior_model_export import export_behavior_model
from app.ml_behavior.evaluate_behavior_models import confusion_matrix_dataframe, write_classification_report
from app.ml_behavior.train_behavior_models import TrainResult, train_models_from_splits


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    group_column: str | None
    train_row_ids: list[int]
    validation_row_ids: list[int]
    test_row_ids: list[int]
    train_groups: list[str]
    validation_groups: list[str]
    test_groups: list[str]


def prepare_behavior_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    working = dataset.reset_index(drop=False).rename(columns={"index": "source_row_id"}).copy()
    working["base_roi_id"] = working["roi_id"].astype(str).str.replace(r"__frame_.*$", "", regex=True) if "roi_id" in working.columns else ""
    return working


def build_split_manifest(dataset: pd.DataFrame, strategy: str, seed: int = 42) -> SplitManifest:
    working = prepare_behavior_dataset(dataset)
    if strategy == "row_stratified_baseline":
        return _row_stratified_manifest(working, seed)
    if strategy == "case_grouped_primary":
        return _grouped_manifest(working, group_column="case_id", strategy=strategy, seed=seed)
    if strategy == "reader_grouped_robustness":
        return _grouped_manifest(working, group_column="reader_id", strategy=strategy, seed=seed)
    raise ValueError(f"Unknown split strategy: {strategy}")


def apply_split_manifest(dataset: pd.DataFrame, manifest: SplitManifest) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = prepare_behavior_dataset(dataset)
    by_id = working.set_index("source_row_id", drop=False)
    train = by_id.loc[manifest.train_row_ids].reset_index(drop=True)
    valid = by_id.loc[manifest.validation_row_ids].reset_index(drop=True)
    test = by_id.loc[manifest.test_row_ids].reset_index(drop=True)
    return train, valid, test


def overlap_audit(dataset: pd.DataFrame, manifest: SplitManifest) -> dict[str, object]:
    train, valid, test = apply_split_manifest(dataset, manifest)
    audit = {
        "strategy": manifest.strategy,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
    }
    for name, part in (("train", train), ("validation", valid), ("test", test)):
        audit[f"{name}_unique_cases"] = _nunique(part, "case_id")
        audit[f"{name}_unique_readers"] = _nunique(part, "reader_id")
        audit[f"{name}_unique_roi_ids"] = _nunique(part, "roi_id")
        audit[f"{name}_unique_grouped_targets"] = _nunique(part, "base_roi_id")
        audit[f"{name}_unique_sessions"] = _nunique(part, "session_id")
    for key in ("case_id", "reader_id", "roi_id", "base_roi_id", "session_id"):
        train_values = _value_set(train, key)
        valid_values = _value_set(valid, key)
        test_values = _value_set(test, key)
        prefix = key.replace("_id", "") if key != "base_roi_id" else "grouped_target"
        audit[f"{prefix}_overlap_train_validation"] = len(train_values & valid_values)
        audit[f"{prefix}_overlap_train_test"] = len(train_values & test_values)
        audit[f"{prefix}_overlap_validation_test"] = len(valid_values & test_values)
    return audit


def run_strategy_evaluation(
    dataset: pd.DataFrame,
    manifest: SplitManifest,
    feature_columns: list[str],
    seed: int,
    evaluation_name: str,
    output_dir: Path,
) -> dict[str, object]:
    train, valid, test = apply_split_manifest(dataset, manifest)
    split_summary = {
        "split": manifest.strategy,
        "seed": seed,
        "group_column": manifest.group_column,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "train_validation_rows": len(train) + len(valid),
        "selection_primary_metric": "validation_macro_f1",
        "selection_secondary_metric": "validation_balanced_accuracy",
    }
    result = train_models_from_splits(train, valid, test, feature_columns, seed=seed, feature_set_name=evaluation_name, split_summary=split_summary)
    strategy_dir = output_dir / evaluation_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    result.results.to_csv(strategy_dir / "model_comparison.csv", index=False)
    result.test_results.to_csv(strategy_dir / "test_results.csv", index=False)
    confusion_matrix_dataframe(result.test_data[1], result.predictions).to_csv(strategy_dir / "confusion_matrix.csv")
    write_classification_report(strategy_dir / "classification_report.md", result.test_data[1], result.predictions)
    best_validation = result.results.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    final_test = result.test_results.iloc[0].to_dict()
    export_behavior_model(result.best_model, strategy_dir, result.feature_columns, {"best_model": best_validation, "final_test_metrics": final_test, "seed": seed, "split_summary": result.split_summary})
    return {
        "evaluation_name": evaluation_name,
        "manifest": manifest,
        "result": result,
        "best_validation": best_validation,
        "final_test": final_test,
        "output_dir": strategy_dir,
    }


def feature_columns_without_slice_index(dataset: pd.DataFrame) -> list[str]:
    return [column for column in allowed_feature_columns(dataset) if column != "slice_index"]


def write_split_manifest(path: Path, manifest: SplitManifest) -> None:
    payload = {
        "strategy": manifest.strategy,
        "seed": manifest.seed,
        "group_column": manifest.group_column,
        "train_row_ids": manifest.train_row_ids,
        "validation_row_ids": manifest.validation_row_ids,
        "test_row_ids": manifest.test_row_ids,
        "train_groups": manifest.train_groups,
        "validation_groups": manifest.validation_groups,
        "test_groups": manifest.test_groups,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def comparison_summary_row(
    evaluation_name: str,
    feature_set: str,
    summary: dict[str, object],
    audit: dict[str, object],
) -> dict[str, object]:
    return {
        "evaluation_strategy": evaluation_name,
        "feature_set": feature_set,
        "best_model": summary.get("model", ""),
        "accuracy": summary.get("accuracy"),
        "balanced_accuracy": summary.get("balanced_accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "weighted_f1": summary.get("weighted_f1"),
        "dangerous_miss_metric": None,
        "false_alert_metric": None,
        "case_overlap": audit.get("case_overlap_train_test", 0) + audit.get("case_overlap_train_validation", 0) + audit.get("case_overlap_validation_test", 0),
        "reader_overlap": audit.get("reader_overlap_train_test", 0) + audit.get("reader_overlap_train_validation", 0) + audit.get("reader_overlap_validation_test", 0),
        "grouped_target_overlap": audit.get("grouped_target_overlap_train_test", 0) + audit.get("grouped_target_overlap_train_validation", 0) + audit.get("grouped_target_overlap_validation_test", 0),
    }


def _row_stratified_manifest(dataset: pd.DataFrame, seed: int) -> SplitManifest:
    y = dataset["hidden_behavior_label"].astype(str)
    train, temp = train_test_split(dataset, test_size=0.30, random_state=seed, stratify=y if y.value_counts().min() >= 2 else None)
    temp_y = temp["hidden_behavior_label"].astype(str)
    valid, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp_y if temp_y.value_counts().min() >= 2 else None)
    return SplitManifest(
        strategy="row_stratified_baseline",
        seed=seed,
        group_column=None,
        train_row_ids=train["source_row_id"].astype(int).tolist(),
        validation_row_ids=valid["source_row_id"].astype(int).tolist(),
        test_row_ids=test["source_row_id"].astype(int).tolist(),
        train_groups=[],
        validation_groups=[],
        test_groups=[],
    )


def _grouped_manifest(dataset: pd.DataFrame, group_column: str, strategy: str, seed: int) -> SplitManifest:
    groups = dataset[group_column].astype(str)
    y = dataset["hidden_behavior_label"].astype(str)
    outer_splits = max(3, min(7, groups.nunique()))
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    train_valid_idx, test_idx = next(outer.split(dataset, y, groups))
    train_valid = dataset.iloc[train_valid_idx].reset_index(drop=True)
    inner_groups = train_valid[group_column].astype(str)
    inner_y = train_valid["hidden_behavior_label"].astype(str)
    inner_splits = max(3, min(6, inner_groups.nunique()))
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    train_idx, valid_idx = next(inner.split(train_valid, inner_y, inner_groups))
    train = train_valid.iloc[train_idx].reset_index(drop=True)
    valid = train_valid.iloc[valid_idx].reset_index(drop=True)
    test = dataset.iloc[test_idx].reset_index(drop=True)
    return SplitManifest(
        strategy=strategy,
        seed=seed,
        group_column=group_column,
        train_row_ids=train["source_row_id"].astype(int).tolist(),
        validation_row_ids=valid["source_row_id"].astype(int).tolist(),
        test_row_ids=test["source_row_id"].astype(int).tolist(),
        train_groups=sorted(train[group_column].astype(str).unique().tolist()),
        validation_groups=sorted(valid[group_column].astype(str).unique().tolist()),
        test_groups=sorted(test[group_column].astype(str).unique().tolist()),
    )


def _nunique(df: pd.DataFrame, column: str) -> int:
    return int(df[column].astype(str).nunique()) if column in df.columns else 0


def _value_set(df: pd.DataFrame, column: str) -> set[str]:
    return set(df[column].astype(str).tolist()) if column in df.columns else set()
