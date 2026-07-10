"""Run Phase 3 synthetic-to-live feature parity audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.features.behavior_feature_builder import FrameRoiFeatureAccumulator, build_behavior_feature_row, feature_parity_matrix
from app.ml_behavior.inference_readiness import assess_prediction_readiness
from app.ui.ui_data_loader import load_workstation_data
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Run Phase 3 synthetic-to-live feature parity audit.")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--roi-id", default=None)
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "behavior_feature_parity_phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_workstation_data(output_root, source="synthetic")
    feature_rows = data.features.copy()
    if args.session_id:
        feature_rows = feature_rows[feature_rows["session_id"].astype(str) == str(args.session_id)].copy()
    if args.roi_id:
        feature_rows = feature_rows[feature_rows["roi_id"].astype(str) == str(args.roi_id)].copy()
    if feature_rows.empty:
        raise ValueError("No feature rows found for requested session/roi selection")

    offline_row = feature_rows.iloc[0].to_dict()
    roi_id = str(offline_row["roi_id"])
    session_id = str(offline_row["session_id"])
    roi_row = data.roi_geometry[data.roi_geometry["roi_id"].astype(str) == roi_id].iloc[0].to_dict()
    raw_samples = data.gaze[(data.gaze["session_id"].astype(str) == session_id) & (data.gaze["roi_id"].astype(str) == roi_id)].copy()

    metadata = {
        "reader_id": offline_row.get("reader_id", "reader"),
        "reader_profile": offline_row.get("reader_profile", "unknown"),
        "case_id": offline_row.get("case_id", "case"),
        "hidden_behavior_label": offline_row.get("hidden_behavior_label", ""),
    }
    direct_result = build_behavior_feature_row(raw_samples, roi_row, metadata)
    accumulator = FrameRoiFeatureAccumulator(roi_row, metadata)
    for sample in raw_samples.to_dict("records"):
        accumulator.add_sample(sample)
    incremental_result = accumulator.build()

    feature_columns = list(data.feature_columns)
    direct_matrix = feature_parity_matrix(offline_row, direct_result.row, feature_columns)
    incremental_matrix = feature_parity_matrix(offline_row, incremental_result.row, feature_columns)

    pd.DataFrame(direct_matrix).to_csv(output_dir / "offline_vs_live_direct.csv", index=False)
    pd.DataFrame(incremental_matrix).to_csv(output_dir / "offline_vs_live_incremental.csv", index=False)

    direct_readiness = assess_prediction_readiness(direct_result.row, feature_columns)
    incremental_readiness = assess_prediction_readiness(incremental_result.row, feature_columns)
    summary = {
        "session_id": session_id,
        "roi_id": roi_id,
        "model_feature_count": len(feature_columns),
        "direct_exact_matches": sum(1 for row in direct_matrix if row["parity_pass"]),
        "direct_mismatches": sum(1 for row in direct_matrix if not row["parity_pass"]),
        "incremental_exact_matches": sum(1 for row in incremental_matrix if row["parity_pass"]),
        "incremental_mismatches": sum(1 for row in incremental_matrix if not row["parity_pass"]),
        "direct_readiness": direct_readiness.__dict__,
        "incremental_readiness": incremental_readiness.__dict__,
    }
    (output_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Phase 3 parity outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
