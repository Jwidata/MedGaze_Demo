"""CLI for scanning a DICOM dataset inventory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_config
from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.dicom.dicom_scanner import scan_dicom_dataset
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Scan a DICOM dataset and write CT/SEG inventory outputs.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing source DICOM/metadata/SEG files. Defaults to config data_root or ../data.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel metadata reader processes. Defaults to a bounded CPU-based value.",
    )
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    config = load_config(args.config)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else config.data_root
    if data_root is None:
        parser.error("No data root configured. Pass --data-root or set data_root in config.")

    output_root = resolve_output_root(args.output_root)
    logger.info("Scanning DICOM-like files under: %s", data_root)
    result = scan_dicom_dataset(data_root=data_root, output_root=output_root, workers=args.workers, use_processes=True)
    logger.info("DICOM audit written to: %s", result.output_dir)
    logger.info("Summary: %s", result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
