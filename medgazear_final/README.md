# MedGazeAR Final

MedGazeAR Final is a non-clinical thesis research prototype for modeling radiology review attention behavior from CT segmentation context and gaze-derived features. The project combines DICOM/SEG metadata processing, segmentation-derived regions of interest, synthetic Tobii-like gaze generation, ROI/scanpath feature extraction, explainable attention rules, behavior-learning models, robustness audits, and cognitive-load proxy analysis.

This repository should be interpreted as a research framework and simulation environment. It is not a medical device and must not be used for diagnosis, treatment decisions, patient care, or operational radiology review.

## Thesis Framing

The recommended thesis claim is:

> This thesis develops and evaluates a non-clinical gaze-aware framework for modeling ROI attention behavior in radiology review workflows using CT segmentation context and synthetic Tobii-like gaze data.

The strongest contribution is the full technical pipeline, not a clinical claim. The current model performance demonstrates internal consistency of the synthetic simulation and feature-learning workflow. It does not validate real radiologist cognition or clinical safety.

## Research Scope

Implemented capabilities include:

- DICOM metadata scanning and CT/SEG inventory generation.
- CT series and segmentation object matching.
- Segmentation-derived ROI geometry extraction.
- Behavior-labeled synthetic gaze generation over real ROI geometry.
- Tobii-like gaze degradation, including jitter, drift, dropout, blink, invalid bursts, outside-CT samples, and UI glances.
- ROI coverage, scanpath, temporal, quality, and geometry feature extraction.
- Explainable rule-based ROI attention classification.
- ML behavior classification from engineered gaze features.
- Rule-recovery, leakage, negative-control, ablation, and robustness audits.
- Gaze-derived cognitive-load proxy analysis.
- Source-agnostic static and interactive gaze visualizations.
- PyQt6 interactive review workstation for representative ROI/gaze episodes.

Out of scope for the current prototype:

- Clinical diagnosis.
- Validated cognitive-load measurement.
- Real radiologist behavior claims.
- Prospective deployment.
- Medical device functionality.

## Current Dataset And Run Summary

The latest generated outputs report:

- 250,273 readable DICOM files scanned.
- 243,414 CT slices.
- 1,017 CT series.
- 6,859 SEG objects.
- 6,844 strict CT/SEG matches.
- 39,890 ROI/frame rows.
- 6,844 SEG objects represented in ROI geometry.
- 871 LIDC-derived patients represented in ROI geometry.
- 2,000 synthetic review sessions.
- 1,034,799 raw synthetic gaze samples.
- 1,923 sampled unique ROI IDs.

Behavior-learning summary from the current run:

- Best selected model: `XGBoostClassifier`.
- Validation macro F1: `0.819`.
- Validation balanced accuracy: `0.822`.
- Held-out test macro F1: `0.812`.
- Held-out test balanced accuracy: `0.808`.

These results are simulation results. They should be reported as evidence that the pipeline can recover synthetic behavior classes, not as evidence of clinical validity.

## Pipeline

Run order:

1. `scripts/01_scan_dicom_dataset.py`
2. `scripts/02_extract_seg_roi_geometry.py`
3. `scripts/03_generate_behavior_labeled_synthetic_gaze.py`
4. `scripts/04_extract_roi_scanpath_features.py`
5. `scripts/05_run_rule_attention_engine.py`
6. `scripts/06_train_rule_distillation_audit.py`
7. `scripts/07_run_deleaked_rule_audit.py`
8. `scripts/08_train_behavior_learning_models.py`
9. `scripts/09_evaluate_behavior_learning_model.py`
10. `scripts/09b_behavior_robustness_audit.py`
11. `scripts/10_run_cognitive_load_proxy_analysis.py`
12. `scripts/11_generate_gaze_visualizations.py`
13. `scripts/20_launch_review_workstation.py --smoke-test`
14. `scripts/20_launch_review_workstation.py --source synthetic`

Each script supports `--help`. Most scripts support `--output-root`; the DICOM scanner also supports `--data-root`.

## Quick Start

From `medgazear_final`:

```bash
python scripts/01_scan_dicom_dataset.py --help
python scripts/01_scan_dicom_dataset.py --data-root ../data --output-root outputs
python scripts/02_extract_seg_roi_geometry.py --output-root outputs
python scripts/03_generate_behavior_labeled_synthetic_gaze.py --output-root outputs --num-sessions 2000
python scripts/04_extract_roi_scanpath_features.py --output-root outputs
python scripts/05_run_rule_attention_engine.py --output-root outputs
python scripts/08_train_behavior_learning_models.py --output-root outputs
python scripts/09b_behavior_robustness_audit.py --output-root outputs
python scripts/10_run_cognitive_load_proxy_analysis.py --output-root outputs
python scripts/11_generate_gaze_visualizations.py --gaze outputs/synthetic_gaze/raw_behavior_labeled_synthetic_gaze.csv --roi-geometry outputs/roi_geometry/seg_roi_geometry.csv --features outputs/features/behavior_feature_table.csv --attention outputs/attention/rule_attention_status.csv --cognitive outputs/cognitive_load/cognitive_proxy_labels.csv --output-root outputs --examples-per-behavior 3 --interactive-report
python scripts/20_launch_review_workstation.py --smoke-test
python scripts/20_launch_review_workstation.py --source synthetic
pytest
```

## Key Outputs

DICOM audit:

- `outputs/dicom_audit/dicom_inventory.csv`
- `outputs/dicom_audit/ct_series_summary.csv`
- `outputs/dicom_audit/seg_inventory.csv`
- `outputs/dicom_audit/dicom_audit_summary.md`

ROI geometry:

- `outputs/roi_geometry/ct_seg_match_table.csv`
- `outputs/roi_geometry/ct_seg_match_summary.md`
- `outputs/roi_geometry/roi_geometry.csv`

Synthetic gaze:

- `outputs/synthetic_gaze/raw_behavior_labeled_synthetic_gaze.csv`
- `outputs/synthetic_gaze/synthetic_session_table.csv`
- `outputs/synthetic_gaze/hidden_behavior_labels.csv`
- `outputs/synthetic_gaze/synthetic_gaze_quality_report.md`
- `outputs/synthetic_gaze/behavior_generation_report.md`
- `outputs/synthetic_gaze/roi_sampling_report.md`

Features:

- `outputs/features/roi_level_features.csv`
- `outputs/features/scanpath_features.csv`
- `outputs/features/slice_level_features.csv`
- `outputs/features/behavior_feature_table.csv`
- `outputs/features/feature_quality_report.md`
- `outputs/features/feature_schema.md`

Attention rules:

- `outputs/attention/rule_attention_status.csv`
- `outputs/attention/review_queue.csv`
- `outputs/attention/attention_distribution.csv`
- `outputs/attention/attention_threshold_sensitivity.csv`
- `outputs/attention/rule_attention_report.md`

Rule and leakage audits:

- `outputs/rule_audit/rule_distillation_dataset.csv`
- `outputs/rule_audit/model_comparison.csv`
- `outputs/rule_audit/deleaked_feature_set_results.csv`
- `outputs/rule_audit/rule_recovery_report.md`
- `outputs/rule_audit/rule_distillation_model_card.md`

Behavior learning:

- `outputs/behavior_learning/behavior_learning_dataset.csv`
- `outputs/behavior_learning/behavior_model_comparison.csv`
- `outputs/behavior_learning/behavior_test_results.csv`
- `outputs/behavior_learning/behavior_feature_importance.csv`
- `outputs/behavior_learning/behavior_learning_summary.md`
- `outputs/behavior_learning/behavior_model_card.md`
- `outputs/behavior_learning/behavior_robustness_report.md`
- `outputs/behavior_learning/best_behavior_model.joblib`

Cognitive-load proxy:

- `outputs/cognitive_load/cognitive_proxy_features.csv`
- `outputs/cognitive_load/cognitive_proxy_labels.csv`
- `outputs/cognitive_load/cognitive_load_distribution.csv`
- `outputs/cognitive_load/attention_vs_cognitive_proxy.csv`
- `outputs/cognitive_load/behavior_vs_cognitive_proxy.csv`
- `outputs/cognitive_load/cognitive_proxy_report.md`
- `outputs/cognitive_load/cognitive_limitations.md`

Visualizations and interactive guided report:

- `outputs/visualizations/representative_cases.csv`
- `outputs/visualizations/visualization_report.md`
- `outputs/visualizations/interactive_gaze_review.html`
- `outputs/visualizations/guided_case_narrations.json`
- `outputs/visualizations/{behavior_label}/{session_id}_roi_overlay.png`
- `outputs/visualizations/{behavior_label}/{session_id}_gaze_points.png`
- `outputs/visualizations/{behavior_label}/{session_id}_heatmap_overlay.png`
- `outputs/visualizations/{behavior_label}/{session_id}_scanpath_overlay.png`
- `outputs/visualizations/{behavior_label}/{session_id}_combined_overlay.png`

Review workstation:

- `outputs/20_launch_review_workstation/manifest.json`

## Documentation

Thesis-facing documents are in `docs/`:

- `docs/methodology.md`
- `docs/evaluation.md`
- `docs/limitations.md`
- `docs/validation_plan.md`
- `docs/thesis_outline.md`
- `docs/project_journey.md`

## Interpretation Guidance

The project should be evaluated on whether it provides a coherent, reproducible, auditable framework for gaze-aware ROI attention modeling. It should not be evaluated as a clinically validated radiology AI system.

The current synthetic results support these claims:

- The DICOM/SEG processing pipeline can build a large ROI context base.
- Synthetic gaze can be generated over real segmentation-derived ROI geometry.
- Engineered gaze features can recover simulation-defined behavior labels.
- Rule-recovery and negative-control audits can identify direct and indirect leakage risks.
- Cognitive-load-like patterns can be explored as a proxy, but not validated as true workload.

The current synthetic results do not support these claims:

- The system improves diagnostic accuracy.
- The model predicts true radiologist intent.
- The cognitive-load proxy measures actual mental workload.
- The system is ready for clinical deployment.

## Future Validation

The most important next research step is real Tobii validation with radiologist or reader participants. A minimal pilot should collect real gaze recordings, ROI attention evidence, task difficulty ratings, and optional workload labels such as NASA-TLX. The synthetic assumptions should then be compared against measured fixation, dwell-time, scanpath, revisit, and signal-quality patterns.
