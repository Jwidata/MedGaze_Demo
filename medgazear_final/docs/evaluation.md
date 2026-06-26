# Evaluation

## Evaluation Goal

The evaluation goal is to determine whether the MedGazeAR prototype forms a coherent, reproducible, and auditable research framework for ROI attention modeling. The current evaluation is simulation-based and should not be interpreted as clinical validation.

## Evaluation Levels

The project should be evaluated at four levels:

1. Data-processing validity.
2. Synthetic gaze and feature-generation consistency.
3. Rule and behavior model auditability.
4. Readiness for real Tobii validation.

## Data-Processing Validity

The DICOM audit demonstrates that the project can process a large imaging metadata base:

- 250,273 readable DICOM files.
- 243,414 CT slices.
- 1,017 CT series.
- 6,859 SEG objects.
- 0 invalid files in the current run.

The CT/SEG matching stage demonstrates that most segmentation objects can be tied to CT context:

- 6,844 strict matches.
- 10 partial matches.
- 5 unmatched missing CT cases.

This supports the claim that the project has a credible imaging-context foundation.

## Synthetic Gaze Evaluation

The synthetic gaze generator produced:

- 2,000 synthetic review sessions.
- 1,034,799 raw gaze samples.
- 1,923 unique sampled ROI IDs.

The synthetic gaze quality report includes signal-quality statistics such as validity, dropout, blink, invalid-burst, outside-CT, UI-glance, jitter, calibration offset, and drift. These metrics are useful for checking whether the simulation is plausible and internally consistent.

Important interpretation: synthetic gaze quality metrics do not prove real gaze realism. They only document the properties of the generated samples.

## Feature Extraction Evaluation

The feature extraction stage produced 2,000 ROI/session feature rows. The feature schema covers ROI coverage, scanpath/search, temporal behavior, signal quality, and ROI geometry context.

Feature quality output reports:

- 2,000 ROI/session feature rows.
- 2,000 unique sessions.
- 1,923 unique ROIs.
- Mean gaze validity ratio around `0.940` after feature aggregation.

This supports the claim that the raw gaze pipeline can be transformed into structured behavior features suitable for rules and ML.

## Rule Attention Evaluation

The rule attention engine creates explainable ROI attention statuses using thresholded evidence.

Current distribution:

- `reviewed`: 17
- `weakly_reviewed`: 1,347
- `not_reviewed`: 500
- `not_evaluated`: 136

Sensitivity checks report that changing thresholds by 25 percent changes a subset of decisions:

- Lower 25 percent threshold run changed 156 of 2,000 ROIs.
- Higher 25 percent threshold run changed 150 of 2,000 ROIs.

This suggests that the rule system is sensitive enough to threshold choices and should be treated as an explainable reference model, not as ground truth.

## Rule-Recovery Audit

The rule-recovery audit asks whether ML can recover the rule attention status from engineered features. This is useful for identifying direct or indirect rule leakage.

Current results:

- Full feature set macro F1: `0.976`.
- No direct rule features macro F1: `0.970`.
- Geometry-only negative control macro F1: `0.251`.
- Ultra-deleaked geometry/context-only macro F1: `0.251`.
- Gaze-quality-only negative control macro F1: `0.576`.
- Random-noise control macro F1: `0.242`.
- Shuffled-label control macro F1: `0.236`.

Interpretation:

- Low geometry-only and shuffled-label control scores are positive signs.
- High no-direct-rule-feature performance indicates that temporal and scanpath features indirectly encode the rule logic.
- This is expected in a synthetic system and should be reported as rule recoverability, not independent clinical truth.

## Behavior-Learning Evaluation

The behavior-learning model predicts hidden synthetic behavior labels from engineered features. The split strategy uses training, validation, and held-out test sets. Model selection is based on validation macro F1, with validation balanced accuracy as a tie-breaker.

Current model-card summary:

- Rows: 2,000.
- Selected model: `XGBoostClassifier`.
- Validation macro F1: `0.819`.
- Validation balanced accuracy: `0.822`.
- Held-out test macro F1: `0.812`.
- Held-out test balanced accuracy: `0.808`.

Class distribution:

- `focused_roi_confirmation`: 448
- `expert_like_systematic_review`: 366
- `partial_near_miss_review`: 363
- `missed_roi_search`: 322
- `high_load_fragmented_review`: 285
- `skipped_slice`: 216

Interpretation:

- The model can recover the simulation-defined behavior classes.
- The result is useful for demonstrating pipeline feasibility.
- The result does not prove generalization to real radiologists.

## Robustness Evaluation

The behavior robustness audit includes:

- Case ID group split.
- Reader-held-out split.
- Hard feature ablation.
- Low, medium, and high noise stress.
- New-seed external-test proxy.

Current robustness results are very high, with macro F1 values near or above `0.985` in several audits.

Interpretation:

- The synthetic behavior classes remain highly separable.
- The pipeline is stable under current synthetic assumptions.
- The results may reflect generator separability rather than real-world robustness.

## Cognitive-Load Proxy Evaluation

The cognitive-load proxy analysis assigns balanced tertile-like labels:

- `low_load_proxy`: 667
- `medium_load_proxy`: 666
- `high_load_proxy`: 667

The proxy relates gaze dispersion, scanpath length, revisits, slice toggling, delayed attention, fixation/saccade fragmentation, and weak signal-quality indicators.

Interpretation:

- The analysis is useful as an exploratory workload-like signal.
- It is not validated cognitive load.
- True validation would require pupil diameter, NASA-TLX, task difficulty ratings, expert workload labels, or comparable ground truth.

## Recommended Evaluation Claims

Safe claims:

- The system processes a large CT/SEG metadata base.
- The system generates synthetic gaze over real segmentation-derived ROI geometry.
- The system extracts interpretable gaze behavior features.
- The system provides explainable attention labels.
- The ML pipeline can recover simulation-defined behavior classes.
- The audit suite identifies direct and indirect leakage risks.

Unsafe claims:

- The system predicts real radiologist behavior.
- The system measures true cognitive load.
- The system improves diagnostic accuracy.
- The system is clinically validated.
- The system is ready for deployment.

## Recommended Thesis Results Wording

Use wording like:

> In the synthetic experiment, the behavior-learning model achieved a held-out macro F1 of 0.812 and balanced accuracy of 0.808. These results demonstrate that the proposed feature pipeline can recover simulation-defined review behavior classes. Because the labels and gaze patterns are generated synthetically, these metrics should be interpreted as internal pipeline validation rather than evidence of real radiologist behavior prediction.
