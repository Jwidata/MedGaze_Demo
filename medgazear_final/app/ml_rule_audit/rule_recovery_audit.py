"""Run feature-set audits for rule recovery."""

from __future__ import annotations

import pandas as pd
import numpy as np

from app.ml_rule_audit.deleaked_feature_sets import RANDOM_NOISE_CONTROL_FEATURES, assert_no_forbidden_features, build_feature_sets
from app.ml_rule_audit.rule_distillation_models import evaluate_models_for_feature_set


def run_rule_recovery_audit(dataset: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rows = []
    working_dataset = dataset.copy()
    rng = np.random.default_rng(random_state)
    for column in RANDOM_NOISE_CONTROL_FEATURES:
        working_dataset[column] = rng.normal(0, 1, len(working_dataset))
    for feature_set_name, columns in build_feature_sets(working_dataset).items():
        assert_no_forbidden_features(columns)
        rows.extend(
            evaluate_models_for_feature_set(
                working_dataset,
                columns,
                feature_set_name,
                shuffled_labels=feature_set_name == "shuffled_label_control",
                random_state=random_state,
            )
        )
    return pd.DataFrame(rows)


def best_non_skipped(results: pd.DataFrame, feature_set: str) -> pd.Series | None:
    subset = results[(results["feature_set"] == feature_set) & (results["skipped"] != True)].copy()
    if subset.empty:
        return None
    return subset.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
