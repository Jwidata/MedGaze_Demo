"""Live Tobii session diagnostics and replay-parity reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from app.features.behavior_feature_builder import build_behavior_feature_row, feature_parity_matrix
from app.gaze_sources.tobii_normalization import live_timing_diagnostics
from app.ml_behavior.inference_readiness import assess_prediction_readiness


def write_live_validation_bundle(
    output_dir: Path,
    session_id: str,
    device_payload: Mapping[str, object],
    samples: list[dict[str, object]],
    roi_rows: list[dict[str, object]],
    feature_columns: list[str],
    current_state_store: Mapping[str, Mapping[str, object]],
    mapping_diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_df = pd.DataFrame(samples)
    (output_dir / "tobii_device_audit.json").write_text(json.dumps(dict(device_payload), indent=2) + "\n", encoding="utf-8")
    timing = live_timing_diagnostics(samples)
    pd.DataFrame([timing]).to_csv(output_dir / "live_timing_diagnostics.csv", index=False)
    raw_audit = _raw_payload_validity_audit(samples_df)
    (output_dir / "raw_payload_validity_audit.json").write_text(json.dumps(raw_audit, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(_invalid_reason_summary(samples_df)).to_csv(output_dir / "invalid_reason_summary.csv", index=False)
    (output_dir / "eye_validity_summary.json").write_text(json.dumps(_eye_validity_summary(samples_df), indent=2) + "\n", encoding="utf-8")
    quality_summary = {
        "sample_count": len(samples),
        "valid_sample_ratio": 1.0 - float(timing.get("invalid_sample_ratio", 0.0)),
        "invalid_sample_ratio": float(timing.get("invalid_sample_ratio", 0.0)),
        "both_eye_valid_ratio": _ratio(samples_df, lambda row: bool(row.get("left_eye_valid")) and bool(row.get("right_eye_valid"))),
        "left_eye_valid_ratio": _ratio(samples_df, lambda row: bool(row.get("left_eye_valid"))),
        "right_eye_valid_ratio": _ratio(samples_df, lambda row: bool(row.get("right_eye_valid"))),
        "coordinate_finite_ratio": _ratio(samples_df, lambda row: pd.notna(row.get("gaze_x_norm")) and pd.notna(row.get("gaze_y_norm"))),
        "coordinate_in_normalized_range_ratio": _ratio(samples_df, lambda row: pd.notna(row.get("gaze_x_norm")) and pd.notna(row.get("gaze_y_norm")) and 0.0 <= float(row.get("gaze_x_norm")) <= 1.0 and 0.0 <= float(row.get("gaze_y_norm")) <= 1.0),
        "mapped_to_screen_ratio": _ratio(samples_df, lambda row: pd.notna(row.get("screen_x")) and pd.notna(row.get("screen_y"))),
        "mapped_to_ct_ratio": _ratio(samples_df, lambda row: pd.notna(row.get("image_x")) and pd.notna(row.get("image_y")) and not bool(row.get("is_outside_ct", False))),
        "outside_ct_ratio": float(samples_df.get("is_outside_ct", pd.Series(dtype=bool)).mean()) if not samples_df.empty else 0.0,
        "ui_glance_ratio": float(samples_df.get("is_ui_glance", pd.Series(dtype=bool)).mean()) if not samples_df.empty else 0.0,
    }
    (output_dir / "live_sample_quality_summary.json").write_text(json.dumps(quality_summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "tracking_preflight_summary.json").write_text(json.dumps(tracking_preflight_summary(samples_df), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(_screen_to_ct_mapping_rows(samples_df)).to_csv(output_dir / "screen_to_ct_mapping_audit.csv", index=False)
    if mapping_diagnostics:
        pd.DataFrame(mapping_diagnostics).to_csv(output_dir / "mapping_failure_diagnostics.csv", index=False)

    readiness_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    roi_index = {str(row.get("roi_id", "")): row for row in roi_rows}
    for roi_id, state in current_state_store.items():
        if roi_id not in roi_index:
            continue
        roi_samples = samples_df[samples_df.get("roi_id", pd.Series(dtype=object)).astype(str) == str(roi_id)].copy() if not samples_df.empty and "roi_id" in samples_df.columns else pd.DataFrame()
        if roi_samples.empty:
            continue
        metadata = {
            "reader_id": state.get("reader_id", "reader"),
            "reader_profile": state.get("reader_profile", "unknown"),
            "case_id": state.get("case_id", "case"),
            "hidden_behavior_label": state.get("hidden_behavior_label", ""),
        }
        result = build_behavior_feature_row(roi_samples, roi_index[roi_id], metadata)
        readiness = assess_prediction_readiness(result.row, feature_columns)
        readiness_rows.append({"roi_id": roi_id, "status": readiness.status, "message": readiness.message, "missing_features": ",".join(readiness.missing_features)})
        parity = feature_parity_matrix(state, result.row, feature_columns)
        for row in parity:
            row.update({"roi_id": roi_id})
            parity_rows.append(row)

    pd.DataFrame(readiness_rows).to_csv(output_dir / "live_feature_readiness_summary.csv", index=False)
    pd.DataFrame(parity_rows).to_csv(output_dir / "live_vs_replay_feature_parity.csv", index=False)
    (output_dir / "real_fixation_summary.json").write_text(json.dumps(_fixation_summary(samples_df), indent=2) + "\n", encoding="utf-8")
    metadata = {
        "session_id": session_id,
        "sample_count": len(samples),
        "roi_count_with_live_evidence": len(readiness_rows),
    }
    (output_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    example = _real_roi_evidence_example(current_state_store)
    (output_dir / "real_roi_evidence_example.json").write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    return {"timing": timing, "quality": quality_summary, "roi_count": len(readiness_rows)}


def _raw_payload_validity_audit(samples_df: pd.DataFrame) -> dict[str, object]:
    records = []
    for _, row in samples_df.head(25).iterrows():
        records.append(
            {
                "timestamp_ms": float(row.get("timestamp_ms", 0) or 0),
                "left_gaze_point_raw": row.get("left_gaze_point_raw"),
                "right_gaze_point_raw": row.get("right_gaze_point_raw"),
                "left_gaze_point_validity_raw": row.get("left_gaze_point_validity_raw"),
                "right_gaze_point_validity_raw": row.get("right_gaze_point_validity_raw"),
                "left_pupil_validity_raw": row.get("left_pupil_validity_raw"),
                "right_pupil_validity_raw": row.get("right_pupil_validity_raw"),
                "left_gaze_origin_validity_raw": row.get("left_gaze_origin_validity_raw"),
                "right_gaze_origin_validity_raw": row.get("right_gaze_origin_validity_raw"),
                "eye_policy": row.get("eye_policy"),
                "gaze_x_norm": row.get("gaze_x_norm"),
                "gaze_y_norm": row.get("gaze_y_norm"),
                "is_valid": bool(row.get("is_valid", False)),
                "invalid_reason": row.get("invalid_reason"),
            }
        )
    return {"sample_count": len(samples_df), "samples": records}


def _invalid_reason_summary(samples_df: pd.DataFrame) -> list[dict[str, object]]:
    if samples_df.empty or "invalid_reason" not in samples_df.columns:
        return []
    counts = samples_df["invalid_reason"].fillna("NONE").astype(str).value_counts().to_dict()
    return [{"invalid_reason": key, "count": int(value)} for key, value in counts.items()]


def _eye_validity_summary(samples_df: pd.DataFrame) -> dict[str, object]:
    if samples_df.empty:
        return {"both_eye_valid": 0, "left_only_valid": 0, "right_only_valid": 0, "neither_valid": 0}
    left = samples_df.get("left_eye_valid", pd.Series(False, index=samples_df.index)).astype(bool)
    right = samples_df.get("right_eye_valid", pd.Series(False, index=samples_df.index)).astype(bool)
    both = int((left & right).sum())
    left_only = int((left & ~right).sum())
    right_only = int((~left & right).sum())
    neither = int((~left & ~right).sum())
    return {"both_eye_valid": both, "left_only_valid": left_only, "right_only_valid": right_only, "neither_valid": neither}


def tracking_preflight_summary(samples_df: pd.DataFrame) -> dict[str, object]:
    valid_ratio = _ratio(samples_df, lambda row: bool(row.get("is_valid", False)))
    mapped_to_ct_ratio = _ratio(samples_df, lambda row: pd.notna(row.get("image_x")) and pd.notna(row.get("image_y")) and not bool(row.get("is_outside_ct", False)))
    if valid_ratio >= 0.2 and mapped_to_ct_ratio >= 0.1:
        status = "READY_FOR_SESSION"
        message = "Ready to start"
        failure_kind = ""
    elif valid_ratio <= 0.0:
        status = "TRACKING_NOT_READY"
        message = "Tracking not ready"
        failure_kind = "tracking_not_ready"
    else:
        status = "TRACKING_NOT_READY"
        message = "CT mapping unavailable"
        failure_kind = "mapping_unavailable"
    return {"status": status, "message": message, "failure_kind": failure_kind, "valid_ratio": valid_ratio, "mapped_to_ct_ratio": mapped_to_ct_ratio}


def _screen_to_ct_mapping_rows(samples_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, row in samples_df.head(25).iterrows():
        rows.append(
            {
                "timestamp_ms": row.get("timestamp_ms"),
                "gaze_x_norm": row.get("gaze_x_norm"),
                "gaze_y_norm": row.get("gaze_y_norm"),
                "screen_x": row.get("screen_x"),
                "screen_y": row.get("screen_y"),
                "image_x": row.get("image_x"),
                "image_y": row.get("image_y"),
                "viewer_local_x": row.get("viewer_local_x"),
                "viewer_local_y": row.get("viewer_local_y"),
                "image_rect": row.get("image_rect"),
                "mapping_failure_reason": row.get("mapping_failure_reason") or row.get("invalid_reason"),
                "is_outside_ct": row.get("is_outside_ct"),
                "is_ui_glance": row.get("is_ui_glance"),
                "invalid_reason": row.get("invalid_reason"),
            }
        )
    return rows


def _fixation_summary(samples_df: pd.DataFrame) -> dict[str, object]:
    if samples_df.empty:
        return {"valid_samples": 0, "detected_fixations": 0, "minimum_duration_ms": 0.0, "median_duration_ms": 0.0, "maximum_duration_ms": 0.0}
    valid = samples_df[(samples_df.get("is_valid") == True) & (samples_df.get("is_outside_ct") == False)].copy()  # noqa: E712
    if valid.empty:
        return {"valid_samples": 0, "detected_fixations": 0, "minimum_duration_ms": 0.0, "median_duration_ms": 0.0, "maximum_duration_ms": 0.0}
    from app.features.temporal_feature_extractor import detect_fixations

    fixations = detect_fixations(valid.to_dict("records"))
    durations = sorted(float(fix["duration_ms"]) for fix in fixations)
    median = durations[len(durations) // 2] if durations else 0.0
    return {
        "valid_samples": len(valid),
        "detected_fixations": len(fixations),
        "minimum_duration_ms": min(durations) if durations else 0.0,
        "median_duration_ms": median,
        "maximum_duration_ms": max(durations) if durations else 0.0,
    }


def _real_roi_evidence_example(current_state_store: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    for roi_id, state in current_state_store.items():
        if float(state.get("total_gaze_time_inside_roi_ms", 0.0) or 0.0) > 0 or int(state.get("gaze_hit_count_inside_roi", 0) or 0) > 0:
            return {
                "roi_id": roi_id,
                "slice_index": state.get("slice_index"),
                "valid_samples": state.get("_valid_sample_count"),
                "inside_hits": state.get("gaze_hit_count_inside_roi"),
                "near_hits": state.get("gaze_hit_count_near_roi"),
                "inside_dwell_ms": state.get("total_gaze_time_inside_roi_ms"),
                "near_dwell_ms": state.get("total_gaze_time_near_roi_ms"),
                "fixation_count": state.get("fixation_count_inside_roi"),
                "time_to_first_roi_fixation_ms": state.get("time_to_first_roi_fixation_ms"),
                "readiness_state": state.get("prediction_readiness"),
                "prediction": state.get("predicted_behavior_label"),
                "confidence": state.get("prediction_confidence"),
            }
    return {}


def _ratio(samples_df: pd.DataFrame, predicate) -> float:
    if samples_df.empty:
        return 0.0
    return sum(1 for _, row in samples_df.iterrows() if predicate(row)) / len(samples_df)
