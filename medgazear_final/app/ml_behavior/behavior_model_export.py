"""Export behavior model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.ml_behavior.behavior_feature_schema import write_label_mapping, write_schema


def export_behavior_model(model, output_dir: Path, feature_columns: list[str], metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "best_behavior_model.joblib")
    write_schema(output_dir / "behavior_feature_schema.json", feature_columns)
    write_label_mapping(output_dir / "behavior_label_mapping.json")
    (output_dir / "behavior_model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
