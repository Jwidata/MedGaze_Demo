"""Evaluate exported behavior-learning model."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_behavior.behavior_prediction_service import BehaviorPredictionService
from app.ml_behavior.evaluate_behavior_models import confusion_matrix_dataframe, metric_row, write_classification_report
from app.ml_behavior.train_behavior_models import split_dataset
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Evaluate exported behavior-learning model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    output_dir = resolve_output_root(args.output_root) / "behavior_learning"
    service = BehaviorPredictionService(Path(args.model), output_dir / "behavior_feature_schema.json", output_dir / "behavior_label_mapping.json")
    dataset = pd.read_csv(args.dataset)
    metadata_path = output_dir / "behavior_model_metadata.json"
    model_name = "exported_best_behavior_model"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_name = str(metadata.get("best_model", {}).get("model", model_name))
        split_summary = metadata.get("split_summary", {})
        _train, _valid, dataset = split_dataset(dataset, seed=int(split_summary.get("seed", 42)), split=str(split_summary.get("split", "stratified")))
    predictions = [service.predict_behavior(row.to_dict()) for _, row in dataset.iterrows()]
    y_true = dataset["hidden_behavior_label"].astype(str).tolist()
    y_pred = [row["predicted_behavior_label"] for row in predictions]
    proba = [[row["class_probabilities"].get(label, 0.0) for label in sorted(set(y_true) | set(y_pred))] for row in predictions]
    row = metric_row("all_behavior_features", model_name, y_true, y_pred, proba)
    row["evaluation_split"] = "test"
    pd.DataFrame([row]).to_csv(output_dir / "behavior_test_results.csv", index=False)
    confusion_matrix_dataframe(y_true, y_pred).to_csv(output_dir / "behavior_confusion_matrix.csv")
    write_classification_report(output_dir / "behavior_classification_report.md", y_true, y_pred)
    logger.info("Behavior model evaluation written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
