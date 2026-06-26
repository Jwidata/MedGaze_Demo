"""Train ML models to reproduce rule_attention_status."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_rule_audit.deleaked_feature_sets import assert_no_forbidden_features, build_feature_sets
from app.ml_rule_audit.rule_audit_report import write_model_card, write_rule_recovery_report
from app.ml_rule_audit.rule_distillation_dataset import build_rule_distillation_dataset, write_rule_distillation_dataset
from app.ml_rule_audit.rule_distillation_models import evaluate_models_for_feature_set
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Train ML rule-distillation audit models.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_dir = resolve_output_root(args.output_root) / "rule_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_rule_distillation_dataset(Path(args.features), Path(args.attention))
    dataset_path = output_dir / "rule_distillation_dataset.csv"
    write_rule_distillation_dataset(dataset, dataset_path)

    full_features = build_feature_sets(dataset)["full_feature_set"]
    assert_no_forbidden_features(full_features)
    results = evaluate_models_for_feature_set(dataset, full_features, "full_feature_set", random_state=args.seed)
    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    # Create initial report/model card from the full feature-set comparison; script 07 overwrites with full deleaked audit.
    write_rule_recovery_report(output_dir / "rule_recovery_report.md", results_df)
    write_model_card(output_dir / "rule_distillation_model_card.md", dataset, results_df)
    logger.info("Rule distillation dataset written to: %s", dataset_path)
    logger.info("Model comparison written to: %s", output_dir / "model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
