"""Synthetic behavior-labeled Tobii-like gaze generator."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.gaze.gaze_degradation_model import apply_gaze_degradation
from app.synthetic.behavior_label_schema import validate_hidden_label
from app.synthetic.behavior_policy import choose_behavior_blend, choose_blended_behavior_duration_ms, choose_hidden_behavior_label, label_focus_strength
from app.synthetic.synthetic_generation_report import write_behavior_generation_report, write_quality_report
from app.synthetic.synthetic_noise_model import degradation_for_profile
from app.synthetic.synthetic_reader_profiles import profile_names


RAW_GAZE_COLUMNS = [
    "session_id",
    "reader_id",
    "reader_profile",
    "case_id",
    "roi_id",
    "slice_index",
    "hidden_behavior_label",
    "timestamp_ms",
    "sample_index",
    "gaze_x_norm",
    "gaze_y_norm",
    "screen_x",
    "screen_y",
    "image_x",
    "image_y",
    "is_valid",
    "is_dropout",
    "is_blink",
    "is_invalid_burst",
    "is_outside_ct",
    "is_ui_glance",
    "calibration_offset_x",
    "calibration_offset_y",
    "drift_x",
    "drift_y",
    "jitter_x",
    "jitter_y",
]


@dataclass(frozen=True)
class ViewerLayout:
    screen_width: int = 1920
    screen_height: int = 1080
    app_width: int = 1600
    app_height: int = 950
    ct_canvas_width: int = 1020
    ct_canvas_height: int = 726

    @property
    def ct_bounds(self) -> tuple[float, float, float, float]:
        x_min = (self.screen_width - self.ct_canvas_width) / 2
        y_min = (self.screen_height - self.ct_canvas_height) / 2
        return (x_min, y_min, x_min + self.ct_canvas_width, y_min + self.ct_canvas_height)


@dataclass(frozen=True)
class SyntheticGenerationResult:
    output_dir: Path
    raw_gaze_csv: Path
    session_table_csv: Path
    hidden_labels_csv: Path
    quality_report_md: Path
    behavior_report_md: Path
    roi_sampling_report_md: Path
    sample_count: int


def generate_synthetic_gaze(
    roi_geometry_csv: Path,
    ct_series_summary_csv: Path,
    output_root: Path,
    num_sessions: int,
    sampling_rate: int = 60,
    seed: int = 42,
    max_rois: int | None = None,
    max_patients: int | None = None,
    sampling_mode: str = "patient_balanced",
    image_size: int | None = None,
    duration_range_ms: tuple[int, int] | None = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> SyntheticGenerationResult:
    """Generate raw synthetic gaze samples from accepted ROI geometry rows."""

    _require_file(roi_geometry_csv)
    _require_file(ct_series_summary_csv)
    _read_csv(ct_series_summary_csv)  # Validate input exists and is parseable for reproducible pipeline use.
    all_roi_rows = _read_csv(roi_geometry_csv)
    rois = [row for row in all_roi_rows if row.get("rejection_reason", "") == "" and row.get("is_empty") == "false"]
    if not rois:
        raise ValueError("No accepted ROI geometry rows available for synthetic gaze generation.")

    rng = np.random.default_rng(seed)
    if sampling_mode not in {"random", "patient_balanced", "roi_balanced"}:
        raise ValueError("sampling_mode must be one of: random, patient_balanced, roi_balanced")
    rois = _limit_roi_pool(rois, rng, max_patients=max_patients, max_rois=max_rois)
    roi_sequence = _sample_roi_sequence(rois, num_sessions, rng, sampling_mode)
    layout = ViewerLayout(screen_width=screen_width, screen_height=screen_height)
    output_dir = output_root.expanduser().resolve() / "synthetic_gaze"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    hidden_rows: list[dict[str, object]] = []
    profiles = profile_names()

    for session_idx in range(num_sessions):
        session_id = f"synthetic_session_{session_idx:05d}"
        reader_profile = str(rng.choice(profiles))
        reader_id = f"reader_{reader_profile}_{session_idx % 17:02d}"
        roi = roi_sequence[session_idx]
        label = choose_hidden_behavior_label(reader_profile, rng)
        validate_hidden_label(label)
        blend_label, blend_weight = choose_behavior_blend(label, rng)
        duration_ms = (
            int(rng.integers(duration_range_ms[0], duration_range_ms[1] + 1))
            if duration_range_ms is not None
            else choose_blended_behavior_duration_ms(label, blend_label, blend_weight, reader_profile, rng)
        )
        sample_count = max(1, int(round(duration_ms * sampling_rate / 1000)))
        calibration_offset = (float(rng.normal(0, 18)), float(rng.normal(0, 18)))
        case_id = roi.get("patient_id") or roi.get("study_instance_uid") or "unknown_case"
        session_rows.append(
            {
                "session_id": session_id,
                "reader_id": reader_id,
                "reader_profile": reader_profile,
                "case_id": case_id,
                "roi_id": roi["roi_id"],
                "hidden_behavior_label": label,
                "duration_ms": duration_ms,
                "sample_count": sample_count,
                "sampling_rate": sampling_rate,
            }
        )
        hidden_rows.append({"session_id": session_id, "roi_id": roi["roi_id"], "hidden_behavior_label": label})
        raw_rows.extend(
            _generate_session_samples(
                session_id=session_id,
                reader_id=reader_id,
                reader_profile=reader_profile,
                case_id=str(case_id),
                roi=roi,
                label=label,
                sample_count=sample_count,
                sampling_rate=sampling_rate,
                rng=rng,
                layout=layout,
                image_size=image_size or _detect_image_size(roi),
                calibration_offset=calibration_offset,
                blend_label=blend_label,
                blend_weight=blend_weight,
            )
        )

    raw_gaze_csv = output_dir / "raw_behavior_labeled_synthetic_gaze.csv"
    session_table_csv = output_dir / "synthetic_session_table.csv"
    hidden_labels_csv = output_dir / "hidden_behavior_labels.csv"
    quality_report_md = output_dir / "synthetic_gaze_quality_report.md"
    behavior_report_md = output_dir / "behavior_generation_report.md"
    roi_sampling_report_md = output_dir / "roi_sampling_report.md"
    _write_csv(raw_gaze_csv, raw_rows, RAW_GAZE_COLUMNS)
    _write_csv(session_table_csv, session_rows, list(session_rows[0].keys()))
    _write_csv(hidden_labels_csv, hidden_rows, list(hidden_rows[0].keys()))
    quality_summary = write_quality_report(quality_report_md, raw_rows, session_rows)
    write_behavior_generation_report(behavior_report_md, {"seed": seed, "num_sessions": num_sessions, **quality_summary})
    _write_roi_sampling_report(roi_sampling_report_md, all_roi_rows, rois, session_rows, sampling_mode, seed, max_patients, max_rois)
    return SyntheticGenerationResult(output_dir, raw_gaze_csv, session_table_csv, hidden_labels_csv, quality_report_md, behavior_report_md, roi_sampling_report_md, len(raw_rows))


def _limit_roi_pool(rois: list[dict[str, str]], rng: np.random.Generator, max_patients: int | None, max_rois: int | None) -> list[dict[str, str]]:
    limited = list(rois)
    if max_patients is not None:
        patients = sorted({_patient_id(row) for row in limited})
        if max_patients < len(patients):
            selected = set(rng.choice(patients, size=max_patients, replace=False).tolist())
            limited = [row for row in limited if _patient_id(row) in selected]
    if max_rois is not None and max_rois < len(limited):
        indices = rng.choice(len(limited), size=max_rois, replace=False)
        limited = [limited[int(index)] for index in indices]
    if not limited:
        raise ValueError("ROI filters removed all eligible ROI rows.")
    return limited


def _sample_roi_sequence(rois: list[dict[str, str]], num_sessions: int, rng: np.random.Generator, sampling_mode: str) -> list[dict[str, str]]:
    if sampling_mode == "random":
        return [rois[int(rng.integers(0, len(rois)))] for _ in range(num_sessions)]
    if sampling_mode == "roi_balanced":
        sequence: list[dict[str, str]] = []
        order = np.arange(len(rois))
        while len(sequence) < num_sessions:
            rng.shuffle(order)
            sequence.extend(rois[int(index)] for index in order[: num_sessions - len(sequence)])
        return sequence

    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for roi in rois:
        by_patient[_patient_id(roi)].append(roi)
    patients = sorted(by_patient)
    sequence = []
    while len(sequence) < num_sessions:
        patient_order = np.array(patients, dtype=object)
        rng.shuffle(patient_order)
        for patient in patient_order:
            patient_rois = by_patient[str(patient)]
            sequence.append(patient_rois[int(rng.integers(0, len(patient_rois)))])
            if len(sequence) == num_sessions:
                break
    return sequence


def _write_roi_sampling_report(
    path: Path,
    all_roi_rows: list[dict[str, str]],
    eligible_rois: list[dict[str, str]],
    session_rows: list[dict[str, object]],
    sampling_mode: str,
    seed: int,
    max_patients: int | None,
    max_rois: int | None,
) -> None:
    all_patients = {_patient_id(row) for row in all_roi_rows if _patient_id(row)}
    sampled_roi_counts = Counter(str(row["roi_id"]) for row in session_rows)
    sampled_patient_counts = Counter(str(row["case_id"]) for row in session_rows)
    coverage = len(sampled_patient_counts) / max(1, len(all_patients)) * 100
    lines = [
        "# ROI Sampling Report",
        "",
        f"- sampling mode: {sampling_mode}",
        f"- seed: {seed}",
        f"- max patients: {max_patients if max_patients is not None else 'none'}",
        f"- max rois: {max_rois if max_rois is not None else 'none'}",
        f"- total available ROI rows: {len(all_roi_rows)}",
        f"- total eligible ROI rows after filters/limits: {len(eligible_rois)}",
        f"- total available unique ROI IDs: {len({str(row.get('roi_id', '')) for row in all_roi_rows if row.get('roi_id', '')})}",
        f"- total available patients: {len(all_patients)}",
        f"- sampled sessions: {len(session_rows)}",
        f"- sampled unique ROI IDs: {len(sampled_roi_counts)}",
        f"- sampled patients: {len(sampled_patient_counts)}",
        f"- patient coverage percentage: {coverage:.2f}%",
        f"- ROI reuse distribution: {_describe_counts(list(sampled_roi_counts.values()))}",
        f"- sessions per patient summary: {_describe_counts(list(sampled_patient_counts.values()))}",
        "",
        "## Top Reused ROIs",
    ]
    for roi_id, count in sampled_roi_counts.most_common(10):
        lines.append(f"- {roi_id}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _describe_counts(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0, "max": 0}
    return {"count": len(values), "min": min(values), "mean": round(float(np.mean(values)), 3), "max": max(values)}


def _patient_id(row: dict[str, str]) -> str:
    return str(row.get("patient_id") or row.get("case_id") or row.get("study_instance_uid") or "unknown_case")


def _generate_session_samples(
    session_id: str,
    reader_id: str,
    reader_profile: str,
    case_id: str,
    roi: dict[str, str],
    label: str,
    sample_count: int,
    sampling_rate: int,
    rng: np.random.Generator,
    layout: ViewerLayout,
    image_size: int,
    calibration_offset: tuple[float, float],
    blend_label: str,
    blend_weight: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader_directness = float(rng.lognormal(mean=0.0, sigma=0.18))
    reader_dispersion = float(rng.lognormal(mean=0.0, sigma=0.25))
    focus = np.clip(
        label_focus_strength(label) * (1 - blend_weight) + label_focus_strength(blend_label) * blend_weight,
        0.04,
        0.90,
    )
    focus = float(np.clip(focus * reader_directness, 0.03, 0.92))
    roi_x = float(roi["centroid_x"])
    roi_y = float(roi["centroid_y"])
    cfg = degradation_for_profile(reader_profile)
    cfg = _behavior_degradation_config(cfg, label, blend_label, blend_weight)
    for sample_index in range(sample_count):
        timestamp_ms = round(sample_index * 1000 / sampling_rate, 3)
        image_x, image_y = _behavior_target_point(
            label if rng.random() > blend_weight else blend_label,
            sample_index,
            sample_count,
            roi_x,
            roi_y,
            float(roi.get("bbox_width", 20)),
            float(roi.get("bbox_height", 20)),
            image_size,
            focus,
            reader_dispersion,
            rng,
        )
        image_x = float(np.clip(image_x, 0, image_size - 1))
        image_y = float(np.clip(image_y, 0, image_size - 1))
        screen_x, screen_y = _image_to_screen(image_x, image_y, image_size, layout)
        degraded = apply_gaze_degradation(
            screen_x,
            screen_y,
            sample_index,
            rng,
            layout.screen_width,
            layout.screen_height,
            layout.ct_bounds,
            calibration_offset,
            cfg,
        )
        degraded_image_x, degraded_image_y = _screen_to_image(float(degraded["screen_x"]), float(degraded["screen_y"]), image_size, layout)
        rows.append(
            {
                "session_id": session_id,
                "reader_id": reader_id,
                "reader_profile": reader_profile,
                "case_id": case_id,
                "roi_id": roi["roi_id"],
                "slice_index": roi["slice_index"],
                "hidden_behavior_label": label,
                "timestamp_ms": timestamp_ms,
                "sample_index": sample_index,
                "image_x": degraded_image_x,
                "image_y": degraded_image_y,
                **degraded,
            }
        )
    return rows


def _behavior_target_point(
    label: str,
    sample_index: int,
    sample_count: int,
    roi_x: float,
    roi_y: float,
    bbox_width: float,
    bbox_height: float,
    image_size: int,
    focus: float,
    dispersion_multiplier: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    progress = sample_index / max(1, sample_count - 1)
    roi_scale_x = max(4.0, bbox_width / 2)
    roi_scale_y = max(4.0, bbox_height / 2)
    if rng.random() < 0.22:
        return _generic_ambiguous_point(roi_x, roi_y, bbox_width, bbox_height, image_size, dispersion_multiplier, rng)
    if label == "expert_like_systematic_review":
        if progress < 0.35 and rng.random() < 0.75:
            row = int(progress / 0.45 * 5)
            x = (progress * 5 % 1) * image_size
            y = (row + 0.5) / 5 * image_size
            return float(x + rng.normal(0, 18 * dispersion_multiplier)), float(y + rng.normal(0, 14 * dispersion_multiplier))
        if rng.random() < 0.56:
            return float(rng.normal(roi_x, roi_scale_x * 1.8 * dispersion_multiplier)), float(rng.normal(roi_y, roi_scale_y * 1.8 * dispersion_multiplier))
        revisit_angle = rng.uniform(0, 2 * np.pi)
        return roi_x + np.cos(revisit_angle) * rng.uniform(25, 120) * dispersion_multiplier, roi_y + np.sin(revisit_angle) * rng.uniform(25, 120) * dispersion_multiplier
    if label == "focused_roi_confirmation":
        if progress < 0.2:
            start_x, start_y = image_size * 0.15, image_size * 0.15
            return start_x + (roi_x - start_x) * progress / 0.2, start_y + (roi_y - start_y) * progress / 0.2
        if rng.random() < 0.22:
            return float(rng.uniform(0, image_size)), float(rng.uniform(0, image_size))
        return float(rng.normal(roi_x, roi_scale_x * 1.55 * dispersion_multiplier)), float(rng.normal(roi_y, roi_scale_y * 1.55 * dispersion_multiplier))
    if label == "partial_near_miss_review":
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(max(bbox_width, bbox_height) * 0.65, max(bbox_width, bbox_height) * 2.8) * dispersion_multiplier
        if rng.random() < 0.32:
            radius *= 0.35
        return roi_x + np.cos(angle) * radius, roi_y + np.sin(angle) * radius
    if label == "missed_roi_search":
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(55, 240) * dispersion_multiplier
        if rng.random() < 0.26:
            radius = rng.uniform(25, 60)
        return roi_x + np.cos(angle) * radius, roi_y + np.sin(angle) * radius
    if label == "skipped_slice":
        if rng.random() < 0.72:
            return float(rng.uniform(-180, image_size + 180)), float(rng.choice([rng.uniform(-180, 40), rng.uniform(image_size - 40, image_size + 180)]))
        if rng.random() < 0.28:
            angle = rng.uniform(0, 2 * np.pi)
            return roi_x + np.cos(angle) * rng.uniform(60, 180), roi_y + np.sin(angle) * rng.uniform(60, 180)
        return float(rng.uniform(0, image_size)), float(rng.uniform(0, image_size))
    if label == "high_load_fragmented_review":
        if rng.random() < 0.26:
            return float(rng.normal(roi_x, roi_scale_x * 2.2 * dispersion_multiplier)), float(rng.normal(roi_y, roi_scale_y * 2.2 * dispersion_multiplier))
        if rng.random() < 0.38:
            angle = rng.uniform(0, 2 * np.pi)
            return roi_x + np.cos(angle) * rng.uniform(35, 190) * dispersion_multiplier, roi_y + np.sin(angle) * rng.uniform(35, 190) * dispersion_multiplier
        return float(rng.uniform(0, image_size)), float(rng.uniform(0, image_size))
    if rng.random() < focus:
        return float(rng.normal(roi_x, roi_scale_x)), float(rng.normal(roi_y, roi_scale_y))
    return float(rng.uniform(0, image_size)), float(rng.uniform(0, image_size))


def _generic_ambiguous_point(roi_x: float, roi_y: float, bbox_width: float, bbox_height: float, image_size: int, dispersion_multiplier: float, rng: np.random.Generator) -> tuple[float, float]:
    mode = rng.choice(["roi", "near", "background"], p=[0.34, 0.36, 0.30])
    if mode == "roi":
        return float(rng.normal(roi_x, max(8.0, bbox_width) * dispersion_multiplier)), float(rng.normal(roi_y, max(8.0, bbox_height) * dispersion_multiplier))
    if mode == "near":
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(max(bbox_width, bbox_height), max(55.0, max(bbox_width, bbox_height) * 4.0)) * dispersion_multiplier
        return roi_x + np.cos(angle) * radius, roi_y + np.sin(angle) * radius
    return float(rng.uniform(0, image_size)), float(rng.uniform(0, image_size))


def _behavior_degradation_config(config, label: str, blend_label: str, blend_weight: float):
    primary = _single_behavior_degradation_config(config, label)
    if blend_label == label or blend_weight <= 0:
        return primary
    blended = _single_behavior_degradation_config(config, blend_label)
    fields = primary.__dataclass_fields__.keys()
    return replace(primary, **{field: getattr(primary, field) * (1 - blend_weight) + getattr(blended, field) * blend_weight for field in fields})


def _single_behavior_degradation_config(config, label: str):
    if label == "high_load_fragmented_review":
        return replace(
            config,
            jitter_std_px=config.jitter_std_px * 1.20,
            dropout_probability=min(0.11, config.dropout_probability * 1.35),
            blink_probability=min(0.09, config.blink_probability * 1.35),
            invalid_burst_probability=min(0.065, config.invalid_burst_probability * 1.35),
            ui_glance_probability=min(0.13, config.ui_glance_probability * 1.5),
        )
    if label == "skipped_slice":
        return replace(
            config,
            jitter_std_px=config.jitter_std_px * 1.15,
            dropout_probability=0.22,
            blink_probability=0.055,
            invalid_burst_probability=0.08,
            outside_ct_probability=0.68,
            ui_glance_probability=0.40,
        )
    if label == "focused_roi_confirmation":
        return replace(config, jitter_std_px=config.jitter_std_px * 0.75, ui_glance_probability=config.ui_glance_probability * 0.5)
    if label == "expert_like_systematic_review":
        return replace(config, dropout_probability=config.dropout_probability * 0.75, invalid_burst_probability=config.invalid_burst_probability * 0.6)
    return config


def _image_to_screen(image_x: float, image_y: float, image_size: int, layout: ViewerLayout) -> tuple[float, float]:
    x_min, y_min, _, _ = layout.ct_bounds
    return x_min + image_x / image_size * layout.ct_canvas_width, y_min + image_y / image_size * layout.ct_canvas_height


def _screen_to_image(screen_x: float, screen_y: float, image_size: int, layout: ViewerLayout) -> tuple[float, float]:
    x_min, y_min, _, _ = layout.ct_bounds
    image_x = (screen_x - x_min) / layout.ct_canvas_width * image_size
    image_y = (screen_y - y_min) / layout.ct_canvas_height * image_size
    return float(np.clip(image_x, 0, image_size - 1)), float(np.clip(image_y, 0, image_size - 1))


def _detect_image_size(roi: dict[str, str]) -> int:
    rows = int(float(roi.get("rows") or 512))
    columns = int(float(roi.get("columns") or 512))
    return rows if rows == columns else 512


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required synthetic gaze input not found: {path}")
