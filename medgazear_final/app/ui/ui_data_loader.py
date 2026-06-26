"""Data loading for the review workstation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import pandas as pd

from app.core.paths import resolve_output_root
from app.ui.ct_pixel_loader import build_ct_path_lookup, load_dicom_inventory
from app.ui.ui_theme import TOBII_PLACEHOLDER_MESSAGE
from app.visualization.gaze_schema import load_gaze_samples
from app.visualization.guided_narration import build_case_narration


@dataclass
class WorkstationData:
    representative_cases: pd.DataFrame
    features: pd.DataFrame
    attention: pd.DataFrame
    behavior_dataset: pd.DataFrame
    cognitive: pd.DataFrame
    gaze: pd.DataFrame
    roi_geometry: pd.DataFrame
    dicom_inventory: pd.DataFrame
    ct_path_lookup: dict[str, Path]
    model: object | None
    feature_columns: list[str]
    label_mapping: dict[str, str]
    source: str = "synthetic"


def default_paths(output_root: str | Path | None = None) -> dict[str, Path]:
    root = resolve_output_root(output_root)
    return {
        "representative_cases": root / "visualizations" / "representative_cases.csv",
        "features": root / "features" / "behavior_feature_table.csv",
        "attention": root / "attention" / "rule_attention_status.csv",
        "behavior_dataset": root / "behavior_learning" / "behavior_learning_dataset.csv",
        "cognitive": root / "cognitive_load" / "cognitive_proxy_labels.csv",
        "gaze": root / "synthetic_gaze" / "raw_behavior_labeled_synthetic_gaze.csv",
        "roi_geometry": root / "roi_geometry" / "seg_roi_geometry.csv",
        "dicom_inventory": root / "dicom_audit" / "dicom_inventory.csv",
        "model": root / "behavior_learning" / "best_behavior_model.joblib",
        "feature_schema": root / "behavior_learning" / "behavior_feature_schema.json",
        "label_mapping": root / "behavior_learning" / "behavior_label_mapping.json",
    }


def load_workstation_data(output_root: str | Path | None = None, source: str = "synthetic") -> WorkstationData:
    if source == "future_tobii_placeholder":
        raise ValueError(TOBII_PLACEHOLDER_MESSAGE)
    paths = default_paths(output_root)
    representative = _read_csv(paths["representative_cases"])
    features = _read_csv(paths["features"])
    attention = _read_csv(paths["attention"])
    behavior = _read_csv(paths["behavior_dataset"])
    cognitive = _read_csv(paths["cognitive"])
    gaze = load_gaze_samples(paths["gaze"], source_type=source)
    roi_geometry = _read_csv(paths["roi_geometry"])
    dicom_inventory = load_dicom_inventory(paths["dicom_inventory"])
    ct_path_lookup = build_ct_path_lookup(dicom_inventory)
    model = joblib.load(paths["model"]) if paths["model"].exists() else None
    feature_columns = _read_json(paths["feature_schema"]).get("feature_columns", []) if paths["feature_schema"].exists() else []
    label_mapping = _read_json(paths["label_mapping"]) if paths["label_mapping"].exists() else {}
    if representative.empty:
        representative = features.head(18).copy()
    return WorkstationData(representative, features, attention, behavior, cognitive, gaze, roi_geometry, dicom_inventory, ct_path_lookup, model, feature_columns, label_mapping, source)


def enrich_case_row(row: pd.Series, data: WorkstationData) -> pd.Series:
    result = row.copy()
    for frame, columns in (
        (data.attention, ["rule_attention_status", "attention_reason", "key_evidence_summary"]),
        (data.cognitive, ["cognitive_load_proxy", "cognitive_load_proxy_score"]),
    ):
        match = frame[(frame["session_id"].astype(str) == str(row["session_id"])) & (frame["roi_id"].astype(str) == str(row["roi_id"]))]
        if not match.empty:
            for column in columns:
                if column in match.columns:
                    result[column] = match.iloc[0][column]
    prediction = predict_behavior(result, data)
    result["predicted_behavior_label"] = prediction["label"]
    result["prediction_confidence"] = prediction["confidence"]
    narration = build_case_narration(result)
    result["guided_narration"] = "\n".join(
        [
            narration["heatmap_suggestion"],
            narration["behavior_rationale"],
            narration["limitation_note"],
        ]
    )
    return result


def predict_behavior(row: pd.Series, data: WorkstationData) -> dict[str, object]:
    if data.model is None or not data.feature_columns:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        values = {column: float(row.get(column, 0) or 0) for column in data.feature_columns}
        X = pd.DataFrame([values], columns=data.feature_columns)
        pred = data.model.predict(X)
        if hasattr(data.model, "_medgazear_label_encoder"):
            pred = data.model._medgazear_label_encoder.inverse_transform(pred)
        confidence = 0.0
        if hasattr(data.model, "predict_proba"):
            confidence = float(max(data.model.predict_proba(X)[0]))
        return {"label": str(pred[0]), "confidence": confidence}
    except Exception:
        return {"label": "unavailable", "confidence": 0.0}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required workstation input not found: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
