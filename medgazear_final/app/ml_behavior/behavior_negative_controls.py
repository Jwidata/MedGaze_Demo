"""Negative controls for behavior learning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml_behavior.behavior_feature_schema import negative_control_sets
from app.ml_behavior.train_behavior_models import train_models


def run_behavior_negative_controls(dataset: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for name, columns in negative_control_sets(dataset).items():
        working = dataset.copy()
        if name == "shuffled_label":
            working["hidden_behavior_label"] = working["hidden_behavior_label"].sample(frac=1.0, random_state=seed).to_numpy()
        elif name == "case_id_reader_leakage_check":
            rng = np.random.default_rng(seed)
            working["random_noise_0"] = rng.normal(size=len(working))
            columns = ["random_noise_0"]
        if columns:
            result = train_models(working, columns, seed)
            df = result.results.copy()
            df["feature_set"] = name
            rows.extend(df.to_dict("records"))
    return pd.DataFrame(rows)
