# Validation Plan

## Purpose

The next major research step is to validate the synthetic assumptions against real eye-tracking data. The goal is not immediate clinical deployment. The goal is to determine whether the synthetic gaze features, rule attention labels, and behavior categories correspond to measurable reader behavior.

## Validation Questions

Recommended validation questions:

1. Do real readers produce ROI dwell-time, fixation-count, revisit, and scanpath patterns similar to the synthetic behavior templates?
2. Can the rule attention engine classify real ROI review behavior in a way that agrees with human annotation?
3. Which synthetic features transfer poorly to real Tobii recordings?
4. Does the behavior-learning model generalize from synthetic data to real gaze data?
5. Which signals are needed to validate the cognitive-load proxy?

## Minimal Tobii Pilot

A minimal pilot should include:

- 2 to 5 readers.
- 5 to 10 CT cases.
- Segmentation-derived ROI targets.
- Tobii gaze recording during review.
- Screen recording or event logs if available.
- Post-case task difficulty rating.
- Optional NASA-TLX workload questionnaire.
- Expert annotation of whether each ROI was reviewed.

This pilot would not prove clinical effectiveness, but it would greatly improve the thesis by testing whether the synthetic assumptions are plausible.

## Preferred Study Design

Participants:

- Radiologists are ideal.
- Radiology residents or trained medical-image readers are acceptable for a feasibility study.
- Non-expert participants can be used only for interface/gaze feasibility, not radiology behavior claims.

Cases:

- Use CT cases with known ROI segmentations.
- Prefer varied ROI sizes, locations, and slice contexts.
- Include cases with multiple ROI densities if possible.

Tasks:

- Ask readers to inspect each case and identify or confirm regions of interest.
- Record whether the reader noticed the ROI.
- Record time spent per case.
- Collect subjective workload after each case or block.

Signals:

- Raw gaze points.
- Fixations.
- Saccades if available.
- Dwell time inside ROI.
- Dwell time near ROI.
- Time to first ROI fixation.
- Scanpath length.
- ROI revisit count.
- Blink/dropout/validity metrics.
- Optional pupil diameter.

Labels:

- Human-reviewed ROI status.
- Missed ROI status.
- Weak or partial review status.
- Workload score if collected.

## Synthetic-To-Real Comparison

Compare real Tobii recordings against synthetic outputs using:

- Distribution plots for dwell time, fixation count, scanpath length, gaze dispersion, and revisit count.
- Per-feature summary statistics by behavior category.
- Rule attention agreement with human annotation.
- Confusion matrix for reviewed, weakly reviewed, not reviewed, and not evaluated labels.
- Calibration error and signal-quality comparison.
- Reader-held-out evaluation.
- Case-held-out evaluation.

## Model Transfer Evaluation

Three transfer experiments are recommended:

1. Train on synthetic, test on real.
2. Train on real, test on held-out real readers or cases.
3. Pretrain on synthetic, fine-tune on a small real set, then test on held-out real data.

Expected outcome:

- Synthetic-only performance will likely drop on real data.
- Fine-tuning with real samples should improve transfer.
- Features with poor transfer should be revised or removed.

## Cognitive-Load Validation

To validate the cognitive-load proxy, collect at least one workload ground truth signal:

- NASA-TLX.
- Simple 1 to 7 task difficulty rating.
- Pupil diameter.
- Expert workload annotation.
- Time pressure condition.
- Error rate or missed-ROI rate.

Evaluation should report correlation or classification agreement between the proxy and the ground truth. Without this, the project must continue calling the output a proxy.

## Ethics And Safety

A real-reader study should use an approved protocol if required by the institution. The study should clearly state:

- The software is non-clinical.
- Cases are for research use only.
- Outputs are not used for patient care.
- Participants are not being clinically evaluated.
- Data should be de-identified.

## Success Criteria

Minimal success criteria:

- Real gaze can be mapped into the same feature schema.
- ROI dwell and fixation features show plausible distributions.
- Rule attention status has measurable agreement with human annotation.
- Synthetic-to-real gaps are quantified honestly.

Strong success criteria:

- Reader-held-out real-data model performance is meaningfully above baseline.
- Rule attention labels agree with expert review labels.
- Cognitive-load proxy correlates with NASA-TLX, pupil diameter, or task difficulty.
- Synthetic pretraining improves real-data performance compared with real-only training on small data.

## Recommended Thesis Wording

Use wording like:

> Future validation should compare synthetic gaze assumptions with real Tobii eye-tracking recordings collected during ROI-based CT review tasks. The validation should measure feature-distribution alignment, rule-label agreement with human annotation, behavior-model transfer, and cognitive-load proxy correlation with workload ground truth.
