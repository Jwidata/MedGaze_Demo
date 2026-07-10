"""Run Phase 1 evaluation-integrity suite without replacing deployed behavior model artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_behavior.behavior_dataset_builder import build_behavior_dataset
from app.ml_behavior.behavior_feature_schema import allowed_feature_columns
from app.ml_behavior.evaluation_integrity import (
    build_split_manifest,
    comparison_summary_row,
    feature_columns_without_slice_index,
    overlap_audit,
    run_strategy_evaluation,
    write_split_manifest,
)
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run Phase 1 evaluation-integrity and split-leakage repair suite.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "behavior_learning_evaluation_phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_behavior_dataset(Path(args.features))
    dataset.to_csv(output_dir / "behavior_learning_dataset.csv", index=False)

    manifests = {
        "row_stratified_baseline": build_split_manifest(dataset, "row_stratified_baseline", seed=args.seed),
        "case_grouped_primary": build_split_manifest(dataset, "case_grouped_primary", seed=args.seed),
        "reader_grouped_robustness": build_split_manifest(dataset, "reader_grouped_robustness", seed=args.seed),
    }
    for name, manifest in manifests.items():
        write_split_manifest(output_dir / f"split_manifest_{name}.json", manifest)

    audits = {name: overlap_audit(dataset, manifest) for name, manifest in manifests.items()}
    pd.DataFrame(audits.values()).to_csv(output_dir / "split_overlap_audit.csv", index=False)

    all_features = allowed_feature_columns(dataset)
    no_slice_features = feature_columns_without_slice_index(dataset)
    (output_dir / "feature_sets.json").write_text(json.dumps({"all_features": all_features, "without_slice_index": no_slice_features}, indent=2) + "\n", encoding="utf-8")

    evaluations = [
        ("row_stratified_baseline", "row_stratified_baseline", manifests["row_stratified_baseline"], all_features),
        ("case_grouped_primary", "case_grouped_all_features", manifests["case_grouped_primary"], all_features),
        ("case_grouped_primary", "case_grouped_without_slice_index", manifests["case_grouped_primary"], no_slice_features),
        ("reader_grouped_robustness", "reader_grouped_robustness", manifests["reader_grouped_robustness"], all_features),
    ]

    comparison_rows: list[dict[str, object]] = []
    evaluation_summaries: list[dict[str, object]] = []
    for strategy_name, evaluation_name, manifest, feature_columns in evaluations:
        logger.info("Running evaluation: %s", evaluation_name)
        result = run_strategy_evaluation(dataset, manifest, feature_columns, args.seed, evaluation_name, output_dir)
        evaluation_summaries.append({
            "evaluation_name": evaluation_name,
            "strategy": strategy_name,
            "feature_columns": feature_columns,
            "best_validation": result["best_validation"],
            "final_test": result["final_test"],
        })
        comparison_rows.append(comparison_summary_row(evaluation_name, evaluation_name, result["final_test"], audits[strategy_name]))

    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_summary.csv", index=False)
    (output_dir / "evaluation_summaries.json").write_text(json.dumps(evaluation_summaries, indent=2) + "\n", encoding="utf-8")
    logger.info("Phase 1 evaluation outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
