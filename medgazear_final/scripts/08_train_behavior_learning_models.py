"""Train reviewer behavior-learning models."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_behavior.behavior_ablation_audit import run_behavior_ablation
from app.ml_behavior.behavior_dataset_builder import build_behavior_dataset
from app.ml_behavior.behavior_model_card import write_behavior_model_card, write_behavior_summary
from app.ml_behavior.behavior_model_export import export_behavior_model
from app.ml_behavior.behavior_negative_controls import run_behavior_negative_controls
from app.ml_behavior.evaluate_behavior_models import confusion_matrix_dataframe, write_classification_report
from app.ml_behavior.train_behavior_models import train_models
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Train behavior-learning models from ROI-level gaze features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    output_dir = resolve_output_root(args.output_root) / "behavior_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_behavior_dataset(Path(args.features))
    dataset.to_csv(output_dir / "behavior_learning_dataset.csv", index=False)
    result = train_models(dataset, seed=args.seed)
    result.results.to_csv(output_dir / "behavior_model_comparison.csv", index=False)
    result.test_results.to_csv(output_dir / "behavior_test_results.csv", index=False)
    run_behavior_ablation(dataset, args.seed).to_csv(output_dir / "behavior_ablation_results.csv", index=False)
    run_behavior_negative_controls(dataset, args.seed).to_csv(output_dir / "behavior_negative_control_results.csv", index=False)
    _write_feature_importance(output_dir / "behavior_feature_importance.csv", result.best_model, result.feature_columns)
    X_test, y_test = result.test_data
    confusion_matrix_dataframe(y_test, result.predictions).to_csv(output_dir / "behavior_confusion_matrix.csv")
    write_classification_report(output_dir / "behavior_classification_report.md", y_test, result.predictions)
    best = result.results.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    final_test = result.test_results.iloc[0].to_dict()
    export_behavior_model(result.best_model, output_dir, result.feature_columns, {"best_model": best, "final_test_metrics": final_test, "seed": args.seed, "split_summary": result.split_summary})
    write_behavior_model_card(output_dir / "behavior_model_card.md", dataset, result.results, result.test_results, result.split_summary)
    write_behavior_summary(output_dir / "behavior_learning_summary.md", result.results, result.test_results)
    logger.info("Behavior learning outputs written to: %s", output_dir)
    return 0


def _write_feature_importance(path: Path, model, feature_columns: list[str]) -> None:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_).mean(axis=0)
    else:
        values = [0.0] * len(feature_columns)
    pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False).to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
