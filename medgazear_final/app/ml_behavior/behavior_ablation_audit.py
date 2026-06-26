"""Ablation audit for behavior models."""

from __future__ import annotations

import pandas as pd

from app.ml_behavior.behavior_feature_schema import ablation_feature_sets
from app.ml_behavior.train_behavior_models import train_models


def run_behavior_ablation(dataset: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for name, columns in ablation_feature_sets(dataset).items():
        if columns:
            result = train_models(dataset, columns, seed)
            df = result.results.copy()
            df["feature_set"] = name
            rows.extend(df.to_dict("records"))
    return pd.DataFrame(rows)
