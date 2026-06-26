"""Build behavior-learning dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml_behavior.behavior_feature_schema import BEHAVIOR_LABELS, BEHAVIOR_TO_REVIEW_STATUS


def build_behavior_dataset(features_csv: Path) -> pd.DataFrame:
    dataset = pd.read_csv(features_csv)
    dataset = dataset[dataset["hidden_behavior_label"].isin(BEHAVIOR_LABELS)].copy()
    dataset["mapped_review_status"] = dataset["hidden_behavior_label"].map(BEHAVIOR_TO_REVIEW_STATUS)
    if dataset.empty:
        raise ValueError("No valid hidden_behavior_label rows found")
    return dataset
