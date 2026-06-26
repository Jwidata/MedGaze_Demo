"""Run gaze-derived cognitive-load proxy analysis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.cognitive_load.attention_load_relation import attention_vs_cognitive_proxy, behavior_vs_cognitive_proxy
from app.cognitive_load.cognitive_proxy_engine import assign_cognitive_proxy_labels, distribution
from app.cognitive_load.cognitive_proxy_features import build_cognitive_proxy_features
from app.cognitive_load.cognitive_proxy_report import write_cognitive_limitations, write_cognitive_proxy_report
from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run gaze-derived cognitive-load proxy analysis.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--behavior-dataset", default=None)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "cognitive_load"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    attention = pd.read_csv(args.attention)
    behavior_path = Path(args.behavior_dataset) if args.behavior_dataset else output_root / "behavior_learning" / "behavior_learning_dataset.csv"
    behavior = pd.read_csv(behavior_path) if behavior_path.exists() else features

    proxy_features = build_cognitive_proxy_features(features)
    proxy_labels = assign_cognitive_proxy_labels(proxy_features)
    load_distribution = distribution(proxy_labels)
    attention_relation = attention_vs_cognitive_proxy(proxy_labels, attention)
    behavior_relation = behavior_vs_cognitive_proxy(proxy_labels, behavior)

    proxy_features.to_csv(output_dir / "cognitive_proxy_features.csv", index=False)
    proxy_labels[["session_id", "roi_id", "cognitive_load_proxy_score", "cognitive_load_proxy", "low_threshold", "high_threshold"]].to_csv(output_dir / "cognitive_proxy_labels.csv", index=False)
    load_distribution.to_csv(output_dir / "cognitive_load_distribution.csv", index=False)
    attention_relation.to_csv(output_dir / "attention_vs_cognitive_proxy.csv", index=False)
    behavior_relation.to_csv(output_dir / "behavior_vs_cognitive_proxy.csv", index=False)
    write_cognitive_proxy_report(output_dir / "cognitive_proxy_report.md", load_distribution, attention_relation, behavior_relation, int(proxy_features["proxy_feature_count"].iloc[0]))
    write_cognitive_limitations(output_dir / "cognitive_limitations.md")

    logger.info("Cognitive-load proxy outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
