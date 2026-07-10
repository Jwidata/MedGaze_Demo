"""Run Phase 4 bbox-vs-mask spatial fidelity ablation."""

from __future__ import annotations

import json
import math
import time
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.logging_utils import configure_logging
from app.core.paths import resolve_output_root
from app.features.behavior_feature_builder import build_behavior_feature_row
from app.features.roi_spatial_modes import RoiMaskLibrary, mask_fill_ratio, sample_disagreement
from app.ml_behavior.behavior_dataset_builder import build_behavior_dataset
from app.ml_behavior.evaluation_integrity import comparison_summary_row, feature_columns_without_slice_index, overlap_audit, run_strategy_evaluation
from scripts._common import build_parser


REQUIRED_CACHE_COLUMNS = {"session_id", "roi_id", "hidden_behavior_label", "geometry_mode"}
CHUNK_SIZE = 200


def main() -> int:
    parser = build_parser("Run Phase 4 SEG spatial fidelity bbox-vs-mask ablation.")
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--roi-geometry", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / "seg_spatial_fidelity_phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    gaze_path = Path(args.gaze)
    roi_geometry_path = Path(args.roi_geometry)
    fingerprint = _source_fingerprint(gaze_path, roi_geometry_path)

    roi_geometry = pd.read_csv(roi_geometry_path)
    gaze = pd.read_csv(gaze_path)
    accepted_roi_geometry = _accepted_roi_geometry(roi_geometry)
    grouped_samples = gaze.groupby([gaze["session_id"].astype(str), gaze["roi_id"].astype(str)], sort=True).groups
    sorted_group_keys = sorted(grouped_samples)

    mask_library = RoiMaskLibrary()
    geometry_audit = pd.DataFrame([resolution.__dict__ for resolution in (mask_library.audit_row(row) for row in accepted_roi_geometry.to_dict("records"))])
    geometry_audit.to_csv(output_dir / "geometry_mapping_audit.csv", index=False)

    roi_shape = accepted_roi_geometry.copy()
    roi_shape["bbox_area_px"] = pd.to_numeric(roi_shape["bbox_width"], errors="coerce").fillna(0) * pd.to_numeric(roi_shape["bbox_height"], errors="coerce").fillna(0)
    roi_shape["mask_fill_ratio"] = [mask_fill_ratio(row, mask_library) for row in roi_shape.to_dict("records")]
    roi_shape.to_csv(output_dir / "roi_shape_audit.csv", index=False)

    bbox_df, bbox_metrics = _build_feature_table_with_checkpoints(
        gaze,
        accepted_roi_geometry,
        grouped_samples,
        sorted_group_keys,
        geometry_mode="bbox",
        output_dir=output_dir,
        fingerprint=fingerprint,
        logger=logger,
    )
    mask_df, mask_metrics = _build_feature_table_with_checkpoints(
        gaze,
        accepted_roi_geometry,
        grouped_samples,
        sorted_group_keys,
        geometry_mode="mask",
        output_dir=output_dir,
        fingerprint=fingerprint,
        logger=logger,
    )

    disagreement = _sample_disagreement_audit(gaze, accepted_roi_geometry, mask_library, logger)
    disagreement.to_csv(output_dir / "bbox_vs_mask_sample_disagreement.csv", index=False)

    deltas = _feature_delta_table(bbox_df, mask_df)
    deltas.to_csv(output_dir / "bbox_vs_mask_feature_deltas.csv", index=False)

    baseline_dataset = build_behavior_dataset(output_root / "behavior_learning_evaluation_phase1" / "behavior_learning_dataset.csv")
    bbox_dataset = build_behavior_dataset(output_dir / "bbox_behavior_feature_table.csv")
    mask_dataset = build_behavior_dataset(output_dir / "mask_behavior_feature_table.csv")
    _validate_dataset_alignment(baseline_dataset, bbox_dataset)
    _validate_dataset_alignment(baseline_dataset, mask_dataset)

    manifest = _load_manifest(output_root / "behavior_learning_evaluation_phase1" / "split_manifest_case_grouped_primary.json")
    feature_columns = feature_columns_without_slice_index(bbox_dataset)
    bbox_eval = run_strategy_evaluation(bbox_dataset, manifest, feature_columns, args.seed, "case_grouped_bbox", output_dir)
    mask_eval = run_strategy_evaluation(mask_dataset, manifest, feature_columns, args.seed, "case_grouped_mask", output_dir)
    audit = overlap_audit(bbox_dataset, manifest)
    comparison = pd.DataFrame(
        [
            comparison_summary_row("case_grouped_bbox", "case_grouped_bbox", bbox_eval["final_test"], audit),
            comparison_summary_row("case_grouped_mask", "case_grouped_mask", mask_eval["final_test"], audit),
        ]
    )
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)

    metadata = {
        "seed": args.seed,
        "feature_set": "without_slice_index",
        "bbox_feature_rows": len(bbox_df),
        "mask_feature_rows": len(mask_df),
        "bbox_metrics": bbox_metrics,
        "mask_metrics": mask_metrics,
        "bbox_false_inside_rate": _overall_rate(disagreement, "bbox_false_inside_rate"),
        "inside_disagreement_rate": _overall_rate(disagreement, "inside_disagreement_rate"),
        "near_disagreement_rate": _overall_rate(disagreement, "near_disagreement_rate"),
        "mask_load_count": mask_library.mask_load_count,
        "distance_build_count": mask_library.distance_build_count,
        "synthetic_generator_dependency": "Generator uses centroid_x, centroid_y, bbox_width, and bbox_height; it does not sample from binary mask pixels directly.",
        "workstation_mask_compatibility": "Mask inference is feasible because roi_id, mask_npz_path, exact frame roi_id, current slice, and image-space gaze coordinates are all available.",
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Phase 4 outputs written to: %s", output_dir)
    return 0


def _accepted_roi_geometry(roi_geometry: pd.DataFrame) -> pd.DataFrame:
    rows = roi_geometry.copy()
    rejection = rows.get("rejection_reason", pd.Series(index=rows.index, dtype=object)).fillna("").astype(str)
    is_empty = rows.get("is_empty", pd.Series(index=rows.index, dtype=object)).fillna(False).astype(str).str.lower()
    return rows[(rejection == "") & (is_empty == "false")].copy()


def _source_fingerprint(gaze_path: Path, roi_path: Path) -> dict[str, object]:
    def stats(path: Path) -> dict[str, object]:
        info = path.stat()
        return {"path": str(path.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}

    payload = {"gaze": stats(gaze_path), "roi_geometry": stats(roi_path)}
    payload["hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def _build_feature_table_with_checkpoints(
    gaze: pd.DataFrame,
    accepted_roi_geometry: pd.DataFrame,
    grouped_samples: dict[tuple[str, str], object],
    sorted_group_keys: list[tuple[str, str]],
    geometry_mode: str,
    output_dir: Path,
    fingerprint: dict[str, object],
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    final_path = output_dir / f"{geometry_mode}_behavior_feature_table.csv"
    tmp_path = output_dir / f"{geometry_mode}_behavior_feature_table.csv.tmp"
    meta_path = output_dir / f"{geometry_mode}_behavior_feature_table.meta.json"
    parts_dir = output_dir / "feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parts_dir / f"{geometry_mode}_progress_manifest.json"

    cached = _validate_feature_cache(final_path, meta_path, fingerprint, geometry_mode)
    if cached is not None:
        logger.info("Using valid %s cache with %s rows", geometry_mode, len(cached))
        return cached, {"cache_reused": True, "elapsed_seconds": None}

    roi_index = {str(row["roi_id"]): row for row in accepted_roi_geometry.to_dict("records")}
    part_manifest = _load_progress_manifest(manifest_path, fingerprint, geometry_mode)
    start = time.perf_counter()
    part_files: list[Path] = []
    shared_mask_library = RoiMaskLibrary() if geometry_mode == "mask" else None
    for chunk_id, start_index in enumerate(range(0, len(sorted_group_keys), CHUNK_SIZE), start=1):
        end_index = min(len(sorted_group_keys), start_index + CHUNK_SIZE)
        part_key = f"{geometry_mode}_part_{chunk_id:04d}"
        part_path = parts_dir / f"{part_key}.csv"
        if part_key in part_manifest.get("completed_parts", []) and _valid_part_file(part_path, geometry_mode):
            part_files.append(part_path)
            continue
        rows: list[dict[str, object]] = []
        for group_key in sorted_group_keys[start_index:end_index]:
            session_id, roi_id = group_key
            if roi_id not in roi_index:
                continue
            sample_rows = gaze.iloc[grouped_samples[group_key]].copy()
            result = build_behavior_feature_row(sample_rows, roi_index[roi_id], geometry_mode=geometry_mode, mask_library=shared_mask_library)
            rows.append(result.row)
        part_tmp = part_path.with_suffix(part_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(part_tmp, index=False)
        if not _valid_part_file(part_tmp, geometry_mode):
            raise RuntimeError(f"Invalid part file generated for {part_key}")
        part_tmp.replace(part_path)
        part_manifest.setdefault("completed_parts", []).append(part_key)
        part_manifest["fingerprint"] = fingerprint
        part_manifest["geometry_mode"] = geometry_mode
        manifest_path.write_text(json.dumps(part_manifest, indent=2) + "\n", encoding="utf-8")
        part_files.append(part_path)
        elapsed = time.perf_counter() - start
        groups_done = end_index
        eta = elapsed / max(1, groups_done) * (len(sorted_group_keys) - groups_done)
        logger.info("%s groups %s/%s, elapsed %.1fs, ETA %.1fs", geometry_mode, groups_done, len(sorted_group_keys), elapsed, eta)

    combined = pd.concat([pd.read_csv(path) for path in sorted(part_files)], ignore_index=True)
    combined.to_csv(tmp_path, index=False)
    if not _valid_part_file(tmp_path, geometry_mode):
        raise RuntimeError(f"Combined {geometry_mode} feature cache failed validation")
    tmp_path.replace(final_path)
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "geometry_mode": geometry_mode, "row_count": len(combined)}, indent=2) + "\n", encoding="utf-8")
    metrics = {"cache_reused": False, "elapsed_seconds": time.perf_counter() - start}
    if shared_mask_library is not None:
        metrics["mask_load_count"] = shared_mask_library.mask_load_count
        metrics["distance_build_count"] = shared_mask_library.distance_build_count
    return combined, metrics


def _validate_feature_cache(path: Path, meta_path: Path, fingerprint: dict[str, object], geometry_mode: str) -> pd.DataFrame | None:
    if not path.exists() or not meta_path.exists() or path.stat().st_size <= 16:
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint", {}).get("hash") != fingerprint.get("hash"):
            return None
        if meta.get("geometry_mode") != geometry_mode:
            return None
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or not REQUIRED_CACHE_COLUMNS.issubset(df.columns):
        return None
    if df["geometry_mode"].astype(str).nunique() != 1 or str(df["geometry_mode"].iloc[0]) != geometry_mode:
        return None
    if df[["session_id", "roi_id"]].isna().any().any():
        return None
    return df


def _valid_part_file(path: Path, geometry_mode: str) -> bool:
    if not path.exists() or path.stat().st_size <= 16:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or not REQUIRED_CACHE_COLUMNS.issubset(df.columns):
        return False
    return df["geometry_mode"].astype(str).eq(geometry_mode).all()


def _load_progress_manifest(path: Path, fingerprint: dict[str, object], geometry_mode: str) -> dict[str, object]:
    if not path.exists():
        return {"completed_parts": [], "fingerprint": fingerprint, "geometry_mode": geometry_mode}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint", {}).get("hash") != fingerprint.get("hash") or payload.get("geometry_mode") != geometry_mode:
            return {"completed_parts": [], "fingerprint": fingerprint, "geometry_mode": geometry_mode}
        return payload
    except Exception:
        return {"completed_parts": [], "fingerprint": fingerprint, "geometry_mode": geometry_mode}


def _load_manifest(path: Path):
    from app.ml_behavior.evaluation_integrity import SplitManifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(**payload)


def _validate_dataset_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    ref = reference[["session_id", "roi_id", "hidden_behavior_label"]].reset_index(drop=True)
    cand = candidate[["session_id", "roi_id", "hidden_behavior_label"]].reset_index(drop=True)
    if not ref.equals(cand):
        raise ValueError("Phase 4 dataset row ordering or labels do not match the fixed baseline dataset")


def _sample_disagreement_audit(gaze: pd.DataFrame, roi_geometry: pd.DataFrame, mask_library: RoiMaskLibrary, logger) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = gaze[(gaze["is_valid"] == True) & (gaze["is_outside_ct"] == False)]  # noqa: E712
    overall = _disagreement_summary(valid, roi_geometry, mask_library)
    overall.update({"scope": "overall", "case_id": "all", "hidden_behavior_label": "all"})
    rows.append(overall)
    grouped = valid.groupby([valid["case_id"].astype(str), valid["hidden_behavior_label"].astype(str)], sort=False)
    total_groups = grouped.ngroups
    for index, ((case_id, label), group) in enumerate(grouped, start=1):
        summary = _disagreement_summary(group.copy(), roi_geometry, mask_library)
        summary.update({"scope": "case_label", "case_id": case_id, "hidden_behavior_label": label})
        rows.append(summary)
        if index % 25 == 0 or index == total_groups:
            logger.info("disagreement groups %s/%s", index, total_groups)
    return pd.DataFrame(rows)


def _disagreement_summary(samples: pd.DataFrame, roi_geometry: pd.DataFrame, mask_library: RoiMaskLibrary) -> dict[str, object]:
    bbox_inside_mask_outside = 0
    bbox_outside_mask_inside = 0
    bbox_inside_mask_inside = 0
    bbox_outside_mask_outside = 0
    bbox_near_mask_disagree = 0
    total = 0
    roi_index = {str(row["roi_id"]): row for row in roi_geometry.to_dict("records")}
    for sample in samples.to_dict("records"):
        roi = roi_index.get(str(sample.get("roi_id", "")))
        if roi is None:
            continue
        result = sample_disagreement(sample, roi, mask_library)
        total += 1
        if result["bbox_inside"] and result["mask_inside"]:
            bbox_inside_mask_inside += 1
        elif result["bbox_inside"] and not result["mask_inside"]:
            bbox_inside_mask_outside += 1
        elif not result["bbox_inside"] and result["mask_inside"]:
            bbox_outside_mask_inside += 1
        else:
            bbox_outside_mask_outside += 1
        if result["bbox_near"] != result["mask_near"]:
            bbox_near_mask_disagree += 1
    return {
        "total_valid_samples": total,
        "bbox_inside_mask_inside": bbox_inside_mask_inside,
        "bbox_inside_mask_outside": bbox_inside_mask_outside,
        "bbox_outside_mask_inside": bbox_outside_mask_inside,
        "bbox_outside_mask_outside": bbox_outside_mask_outside,
        "bbox_false_inside_rate": bbox_inside_mask_outside / max(1, total),
        "inside_disagreement_rate": (bbox_inside_mask_outside + bbox_outside_mask_inside) / max(1, total),
        "near_disagreement_rate": bbox_near_mask_disagree / max(1, total),
    }


def _feature_delta_table(bbox_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    merged = bbox_df.merge(mask_df, on=["session_id", "roi_id", "hidden_behavior_label"], suffixes=("_bbox", "_mask"))
    tracked = [
        "total_gaze_time_inside_roi_ms",
        "total_gaze_time_near_roi_ms",
        "fixation_count_inside_roi",
        "fixation_count_near_roi",
        "max_fixation_duration_inside_roi_ms",
        "mean_fixation_duration_inside_roi_ms",
        "time_to_first_roi_fixation_ms",
        "gaze_hit_count_inside_roi",
        "gaze_hit_count_near_roi",
    ]
    rows = []
    for feature in tracked:
        diff = (pd.to_numeric(merged[f"{feature}_bbox"], errors="coerce") - pd.to_numeric(merged[f"{feature}_mask"], errors="coerce")).abs()
        rows.append(
            {
                "feature_name": feature,
                "mean_absolute_difference": float(diff.mean()),
                "median_absolute_difference": float(diff.median()),
                "percentage_rows_changed": float((diff > 0).mean()),
                "maximum_difference": float(diff.max()),
            }
        )
    return pd.DataFrame(rows)


def _overall_rate(disagreement: pd.DataFrame, column: str) -> float:
    row = disagreement[disagreement["scope"] == "overall"]
    return float(row.iloc[0][column]) if not row.empty else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
