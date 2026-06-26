"""Load exported behavior model and predict behavior labels."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


class BehaviorPredictionService:
    def __init__(self, model_path: Path, feature_schema_path: Path, label_mapping_path: Path):
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(feature_schema_path.read_text(encoding="utf-8"))["feature_columns"]
        self.label_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))

    def predict_behavior(self, feature_row: dict) -> dict:
        X = pd.DataFrame([{column: feature_row.get(column, 0) for column in self.feature_columns}]).fillna(0).astype(float)
        pred = self.model.predict(X)
        if hasattr(self.model, "_medgazear_label_encoder"):
            pred_label = self.model._medgazear_label_encoder.inverse_transform(pred)[0]
            classes = list(self.model._medgazear_label_encoder.classes_)
        else:
            pred_label = str(pred[0])
            classes = [str(c) for c in getattr(self.model, "classes_", [])]
        proba = self.model.predict_proba(X)[0] if hasattr(self.model, "predict_proba") else []
        class_probs = {label: float(prob) for label, prob in zip(classes, proba)}
        return {"predicted_behavior_label": pred_label, "confidence": max(class_probs.values()) if class_probs else 0.0, "class_probabilities": class_probs, "mapped_review_status": self.label_mapping[pred_label]}


def predict_behavior(feature_row: dict, artifact_dir: Path = Path("outputs/behavior_learning")) -> dict:
    service = BehaviorPredictionService(artifact_dir / "best_behavior_model.joblib", artifact_dir / "behavior_feature_schema.json", artifact_dir / "behavior_label_mapping.json")
    return service.predict_behavior(feature_row)
