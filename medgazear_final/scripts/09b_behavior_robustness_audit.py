"""Run behavior-learning robustness audits."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ml_behavior.behavior_robustness_audit import run_behavior_robustness_audit, write_robustness_report
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run robustness audits for the behavior-learning model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--external-dataset", default=None, help="Optional seed-99 behavior_learning_dataset.csv or behavior_feature_table.csv.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_dir = resolve_output_root(args.output_root) / "behavior_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(args.dataset)
    external = pd.read_csv(args.external_dataset) if args.external_dataset else None
    results = run_behavior_robustness_audit(dataset, Path(args.model), external, args.seed)
    results.to_csv(output_dir / "behavior_robustness_results.csv", index=False)
    write_robustness_report(output_dir / "behavior_robustness_report.md", results)
    logger.info("Behavior robustness audit written to: %s", output_dir / "behavior_robustness_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
