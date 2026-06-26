# Thesis Outline

## Recommended Title

MedGazeAR: A Non-Clinical Gaze-Aware Framework for ROI Attention Modeling in Radiology Review

## Recommended Thesis Statement

This thesis develops and evaluates a non-clinical research framework that combines CT segmentation-derived ROI context, synthetic Tobii-like gaze simulation, gaze feature extraction, explainable attention rules, behavior-learning models, and leakage-aware audits to support future validation of gaze-aware radiology review workflows.

## Chapter 1: Introduction

Purpose:

- Introduce the problem of understanding visual attention during radiology review.
- Explain why gaze behavior, ROI context, and review attention are relevant.
- Clarify that the project is non-clinical and research-focused.

Key points:

- Radiology review involves complex visual search.
- Eye tracking can expose attention patterns but real data collection is difficult.
- A synthetic-to-real framework can prepare methods before real Tobii validation.

Recommended research question:

> Can segmentation-derived ROI context and gaze-derived scanpath features be used to build an auditable framework for modeling radiology review attention behavior, and what validation is required before applying the framework to real eye-tracking data?

## Chapter 2: Background

Purpose:

- Review related work and technical foundations.

Suggested sections:

- Radiology visual search and missed findings.
- Eye tracking in medical image review.
- Fixations, saccades, dwell time, scanpaths, and attention metrics.
- DICOM and DICOM SEG context.
- Augmented or assisted review workflows.
- Machine learning risks in medical workflow modeling.
- Data leakage and negative controls.
- Cognitive load and workload measurement.

## Chapter 3: Dataset And Imaging Context

Purpose:

- Explain the DICOM/SEG audit and ROI geometry base.

Include:

- DICOM audit pipeline.
- CT series inventory.
- SEG object inventory.
- CT/SEG matching.
- ROI geometry extraction.
- Dataset statistics.

Current statistics to report:

- 250,273 readable DICOM files.
- 243,414 CT slices.
- 1,017 CT series.
- 6,859 SEG objects.
- 6,844 strict CT/SEG matches.
- 39,890 ROI/frame rows.
- 871 LIDC-derived patients.

## Chapter 4: Synthetic Gaze Generation

Purpose:

- Explain how synthetic gaze sessions are generated over real ROI geometry.

Include:

- Reader profiles.
- Hidden behavior labels.
- Session duration generation.
- ROI target policies.
- Synthetic scanpath generation.
- Tobii-like degradation model.
- Sampling strategy.

Current statistics to report:

- 2,000 synthetic sessions.
- 1,034,799 raw synthetic gaze samples.
- 1,923 sampled unique ROI IDs.

Important caveat:

- Hidden behavior labels are generated, not measured.

## Chapter 5: Feature Extraction And Rule Attention Modeling

Purpose:

- Present the technical core of the thesis.

Include:

- ROI coverage features.
- Scanpath/search features.
- Temporal features.
- Signal-quality features.
- Geometry/context features.
- Rule attention thresholds.
- Review queue construction.
- Attention threshold sensitivity.

Current rule distribution:

- `reviewed`: 17
- `weakly_reviewed`: 1,347
- `not_reviewed`: 500
- `not_evaluated`: 136

## Chapter 6: Behavior Learning And Leakage-Aware Auditing

Purpose:

- Explain ML behavior classification and why leakage-aware evaluation is necessary.

Include:

- Behavior labels.
- Feature inclusion/exclusion rules.
- Model candidates.
- Train/validation/test split.
- Model selection criteria.
- Held-out test evaluation.
- Rule-recovery audit.
- Negative controls.
- Hard feature ablation.
- Group split audits.
- Noise stress audits.

Current behavior-learning result:

- Selected model: `XGBoostClassifier`.
- Validation macro F1: `0.819`.
- Held-out test macro F1: `0.812`.
- Held-out test balanced accuracy: `0.808`.

Important interpretation:

- Performance demonstrates recovery of synthetic behavior classes, not real radiologist behavior prediction.

## Chapter 7: Cognitive-Load Proxy Analysis

Purpose:

- Present the exploratory workload-like proxy analysis.

Include:

- Proxy feature weights.
- Percentile ranking.
- Low, medium, high proxy labels.
- Relationship to behavior labels.
- Relationship to attention rule status.

Important caveat:

- This is not validated cognitive load because there is no pupil diameter, NASA-TLX, task difficulty rating, or expert workload label.

## Chapter 8: Results

Purpose:

- Consolidate quantitative results.

Suggested sections:

- DICOM/SEG processing results.
- ROI geometry results.
- Synthetic gaze quality results.
- Feature extraction results.
- Rule attention results.
- Rule-recovery audit results.
- Behavior-learning results.
- Robustness audit results.
- Cognitive-load proxy results.

Recommended wording:

> The results demonstrate the internal consistency and auditability of the proposed framework under synthetic assumptions. They do not establish clinical validity or real radiologist behavior prediction.

## Chapter 9: Limitations And Validation Plan

Purpose:

- Honestly state current limits and define how the system should be validated next.

Include:

- Synthetic gaze limitation.
- Generated-label limitation.
- Rule-threshold limitation.
- Cognitive-load proxy limitation.
- No real Tobii data.
- No radiologist ground truth.
- No clinical outcome validation.
- Proposed Tobii pilot.
- Synthetic-to-real transfer evaluation.

## Chapter 10: Conclusion

Purpose:

- Summarize what was built and what was learned.

Safe conclusion:

> This thesis presents a reproducible research framework for linking CT segmentation-derived ROI context with gaze-derived behavior features and audit-aware ML analysis. The project demonstrates feasibility under synthetic conditions and establishes a path for future Tobii-based validation.

Avoid concluding:

- The system is clinically validated.
- The system predicts true radiologist cognition.
- The system measures real cognitive load.
- The system should be deployed clinically.

## Core Contribution List

Recommended thesis contributions:

1. A DICOM/SEG-to-ROI geometry processing pipeline for gaze-aware review research.
2. A synthetic Tobii-like gaze generation framework over segmentation-derived ROI context.
3. A feature extraction pipeline for ROI coverage, scanpath, temporal, quality, and geometry features.
4. An explainable rule-based ROI attention engine.
5. A behavior-learning workflow with validation split, held-out testing, negative controls, and robustness audits.
6. A cognitive-load proxy analysis with explicit limitations.
7. A future validation protocol for real Tobii/radiologist studies.
