"""Run deleaked and negative-control rule distillation audit."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_rule_audit.rule_audit_report import write_model_card, write_rule_recovery_report
from app.ml_rule_audit.rule_recovery_audit import run_rule_recovery_audit
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run deleaked feature-set rule recovery audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_dir = resolve_output_root(args.output_root) / "rule_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(args.dataset)
    results = run_rule_recovery_audit(dataset, random_state=args.seed)
    results.to_csv(output_dir / "deleaked_feature_set_results.csv", index=False)
    # Also copy the comprehensive results into model_comparison for convenience after full audit.
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    write_rule_recovery_report(output_dir / "rule_recovery_report.md", results)
    write_model_card(output_dir / "rule_distillation_model_card.md", dataset, results)
    logger.info("Deleaked rule audit written to: %s", output_dir / "deleaked_feature_set_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
