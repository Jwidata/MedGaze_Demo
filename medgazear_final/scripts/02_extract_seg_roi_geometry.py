"""CLI for CT/SEG matching and DICOM SEG ROI mask extraction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.roi.roi_geometry import extract_roi_geometry_from_matches
from app.roi.roi_matcher import match_ct_seg_from_audit
from scripts._common import build_parser

def main() -> int:
    parser = build_parser("Match SEG objects to CT series and optionally extract DICOM SEG ROI masks.")
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="Perform CT/SEG matching and reporting only.",
    )
    parser.add_argument(
        "--extract-masks",
        action="store_true",
        help="Extract binary ROI masks from matched DICOM SEG objects.",
    )
    parser.add_argument(
        "--max-seg",
        type=int,
        default=None,
        help="Maximum number of matched SEG objects to inspect during mask extraction.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partially matched SEG objects during mask extraction.",
    )
    args = parser.parse_args()
    if args.match_only == args.extract_masks:
        parser.error("Choose exactly one of --match-only or --extract-masks.")

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    if args.match_only:
        result = match_ct_seg_from_audit(output_root)
        logger.info("CT/SEG match table written to: %s", result.match_table_csv)
        logger.info("Summary: %s", result.summary)
    else:
        result = extract_roi_geometry_from_matches(
            output_root=output_root,
            max_seg=args.max_seg,
            include_partial=args.include_partial,
        )
        logger.info("ROI geometry written to: %s", result.geometry_csv)
        logger.info("ROI masks written under: %s", result.mask_dir)
        logger.info("Summary: %s", result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
