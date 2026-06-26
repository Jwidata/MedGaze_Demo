"""CLI for behavior-labeled synthetic Tobii-like gaze generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.synthetic.synthetic_gaze_generator import generate_synthetic_gaze
from scripts._common import build_parser

def main() -> int:
    parser = build_parser("Generate synthetic behavior-labeled Tobii-like gaze from ROI geometry.")
    parser.add_argument("--roi-geometry", default="outputs/roi_geometry/seg_roi_geometry.csv")
    parser.add_argument("--num-sessions", type=int, default=100)
    parser.add_argument("--sampling-rate", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["random", "patient_balanced", "roi_balanced"], default="patient_balanced")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--duration-range-ms", default=None, help="Optional override as inclusive min,max duration range in milliseconds.")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    duration_range = _parse_duration_range(args.duration_range_ms) if args.duration_range_ms else None
    result = generate_synthetic_gaze(
        roi_geometry_csv=Path(args.roi_geometry),
        ct_series_summary_csv=output_root / "dicom_audit" / "ct_series_summary.csv",
        output_root=output_root,
        num_sessions=args.num_sessions,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
        max_rois=args.max_rois,
        max_patients=args.max_patients,
        sampling_mode=args.sampling_mode,
        image_size=args.image_size,
        duration_range_ms=duration_range,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )
    logger.info("Synthetic gaze written to: %s", result.raw_gaze_csv)
    logger.info("Generated samples: %s", result.sample_count)
    return 0


def _parse_duration_range(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2 or parts[0] <= 0 or parts[1] < parts[0]:
        raise ValueError("--duration-range-ms must be formatted as positive min,max")
    return parts[0], parts[1]


if __name__ == "__main__":
    raise SystemExit(main())
