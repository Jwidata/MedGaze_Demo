"""Run explainable rule-based ROI attention engine."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attention.attention_report import write_attention_report
from app.attention.attention_sensitivity_audit import run_threshold_sensitivity
from app.attention.attention_thresholds import load_attention_thresholds
from app.attention.rule_attention_engine import read_feature_rows, run_attention_engine
from app.core.logging_utils import configure_logging
from app.core.paths import PROJECT_ROOT, resolve_output_root
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run explainable rule-based ROI attention engine.")
    parser.add_argument("--features", required=True, help="Behavior feature table CSV.")
    parser.add_argument("--threshold-config", default=str(PROJECT_ROOT / "configs" / "attention_thresholds.yaml"))
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_attention_thresholds(Path(args.threshold_config))
    feature_rows = read_feature_rows(Path(args.features))
    result = run_attention_engine(feature_rows, thresholds)
    sensitivity_rows = run_threshold_sensitivity(feature_rows, thresholds)

    _write_csv(output_dir / "rule_attention_status.csv", result.status_rows, ["session_id", "roi_id", "hidden_behavior_label", "rule_attention_status", "attention_reason", "rule_confidence_proxy", "key_evidence_summary"])
    _write_csv(output_dir / "review_queue.csv", result.review_queue_rows, ["session_id", "roi_id", "hidden_behavior_label", "rule_attention_status", "attention_reason", "rule_confidence_proxy", "key_evidence_summary"])
    _write_csv(output_dir / "attention_distribution.csv", result.distribution_rows, ["rule_attention_status", "count"])
    _write_csv(output_dir / "attention_threshold_sensitivity.csv", sensitivity_rows, ["variation", "threshold_factor", "roi_count", "changed_status_count"])
    write_attention_report(output_dir / "rule_attention_report.md", result.distribution_rows, sensitivity_rows, thresholds, result.status_rows)

    logger.info("Attention status rows written: %s", len(result.status_rows))
    logger.info("Attention output directory: %s", output_dir)
    return 0


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
