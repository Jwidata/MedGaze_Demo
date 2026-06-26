"""Dataset construction for rule-distillation audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_rule_distillation_dataset(features_csv: Path, attention_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    attention = pd.read_csv(attention_csv)
    target_columns = ["session_id", "roi_id", "rule_attention_status", "attention_reason", "rule_confidence_proxy", "key_evidence_summary"]
    merged = features.merge(attention[target_columns], on=["session_id", "roi_id"], how="inner", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError(f"Target merge lost rows: features={len(features)} merged={len(merged)}")
    if merged["rule_attention_status"].isna().any():
        raise ValueError("Missing rule_attention_status after merge")
    return merged


def write_rule_distillation_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
