"""Report writers for rule distillation audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml_rule_audit.rule_recovery_audit import best_non_skipped


def write_rule_recovery_report(path: Path, results: pd.DataFrame) -> None:
    full = best_non_skipped(results, "full_feature_set")
    no_direct = best_non_skipped(results, "no_direct_rule_features")
    geometry = best_non_skipped(results, "geometry_only_negative_control")
    ultra_geometry = best_non_skipped(results, "ultra_deleaked_geometry_context_only")
    quality = best_non_skipped(results, "gaze_quality_only_negative_control")
    random_noise = best_non_skipped(results, "ultra_deleaked_random_noise_control")
    shuffled = best_non_skipped(results, "shuffled_label_control")
    lines = ["# Rule Recovery Audit Report", ""]
    lines.append("This audit tests whether ML models mainly recover handcrafted rule logic rather than discovering independent clinical truth.")
    lines.append("")
    for label, row in (
        ("full_feature_set", full),
        ("no_direct_rule_features", no_direct),
        ("geometry_only_negative_control", geometry),
        ("ultra_deleaked_geometry_context_only", ultra_geometry),
        ("gaze_quality_only_negative_control", quality),
        ("ultra_deleaked_random_noise_control", random_noise),
        ("shuffled_label_control", shuffled),
    ):
        if row is not None:
            lines.append(f"- best {label}: {row['model']} macro_f1={float(row['macro_f1']):.3f} balanced_accuracy={float(row['balanced_accuracy']):.3f}")
    lines.append("")
    lines.append("## Interpretation")
    if full is not None and no_direct is not None and float(full["macro_f1"]) >= 0.85 and float(no_direct["macro_f1"]) + 0.10 < float(full["macro_f1"]):
        lines.append("- Full feature performance is high and drops after removing direct rule features; the model is primarily recovering handcrafted rule logic.")
    elif full is not None and float(full["macro_f1"]) >= 0.85:
        lines.append("- Full feature performance is high; inspect deleaked sets carefully for synthetic artifacts or indirect rule proxies.")
    if geometry is not None and float(geometry["macro_f1"]) > 0.60:
        lines.append("- Geometry-only control is unexpectedly high; flag potential leakage or synthetic artifact.")
    if ultra_geometry is not None and float(ultra_geometry["macro_f1"]) <= 0.45:
        lines.append("- Ultra-deleaked geometry/context-only performance is low; no strong geometry leakage is evident.")
    if full is not None and no_direct is not None and ultra_geometry is not None and float(no_direct["macro_f1"]) >= 0.80 and float(ultra_geometry["macro_f1"]) <= 0.45:
        lines.append("- no_direct_rule_features remains high while ultra-deleaked geometry/context is low; this suggests indirect temporal/scanpath proxy recovery, not direct geometry leakage.")
    if quality is not None and float(quality["macro_f1"]) > 0.60:
        lines.append("- Gaze-quality control is unexpectedly high; flag potential leakage or synthetic artifact.")
    if random_noise is not None and float(random_noise["balanced_accuracy"]) > 0.40:
        lines.append("- Random-noise control is above chance; flag a possible evaluation bug.")
    if shuffled is not None and float(shuffled["balanced_accuracy"]) > 0.40:
        lines.append("- Shuffled-label control is above chance; flag a possible evaluation bug.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(path: Path, dataset: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Rule Distillation Model Card",
        "",
        "Purpose: audit whether ML can reproduce rule_attention_status from engineered gaze/ROI features.",
        "Target: rule_attention_status.",
        "Clinical limitation: this is a simulation/reference audit, not clinical truth.",
        f"Rows: {len(dataset)}",
        f"Target distribution: {dataset['rule_attention_status'].value_counts().to_dict()}",
        f"Models evaluated: {sorted(results['model'].dropna().unique().tolist())}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
