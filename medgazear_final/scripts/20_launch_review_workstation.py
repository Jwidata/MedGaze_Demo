"""Launch the MedGazeAR interactive review workstation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.ui_training import launch_review_workstation, smoke_test_workstation
from scripts._common import build_parser


def main() -> int:
    parser = build_parser("Launch MedGazeAR interactive review workstation.")
    parser.add_argument("--smoke-test", action="store_true", help="Run startup checks without launching a UI.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "future_tobii_placeholder"])
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    step_dir = output_root / "20_launch_review_workstation"
    step_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        result = smoke_test_workstation(output_root, source=args.source)
        manifest = {"step_name": "20_launch_review_workstation", "source": args.source, **result}
        (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Smoke test passed: {step_dir}")
        if result.get("message"):
            print(result["message"])
        return 0

    logger.info("Launching review workstation from %s with source=%s", output_root, args.source)
    return launch_review_workstation(output_root, source=args.source)


if __name__ == "__main__":
    raise SystemExit(main())
