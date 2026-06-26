# Limitations

## Non-Clinical Status

MedGazeAR Final is not a medical device. It is not intended for diagnosis, treatment decisions, patient care, clinical workflow triage, or operational radiology use.

No output from this project should be interpreted as clinical truth.

## Synthetic Gaze Limitation

The current gaze data is synthetic. It is generated from predefined behavior policies, reader profiles, ROI geometry, and Tobii-like noise assumptions.

This means:

- Hidden behavior labels are generated, not observed.
- Gaze trajectories are simulated, not measured.
- Reader profiles are approximations, not real reader phenotypes.
- Tobii-like degradation is parameterized, not empirically calibrated against the current dataset.

The synthetic gaze pipeline is useful for testing system design, but it cannot prove real radiologist behavior validity.

## Label Generation Limitation

The hidden behavior label influences several generated properties, including:

- Session duration.
- ROI focus strength.
- Gaze target policy.
- Fragmentation behavior.
- Skipped-slice behavior.
- Dropout, blink, invalid-burst, outside-CT, and UI-glance tendencies.

Because the model is trained to predict these generated labels from features derived from the generated gaze, strong ML performance is expected. The behavior-learning model demonstrates recovery of synthetic patterns, not discovery of independent clinical behavior.

## Rule Attention Limitation

The rule attention engine uses explainable thresholds for dwell time, hit count, fixation count, valid exposure, and gaze quality. These thresholds are reference rules for simulation and auditing.

They are not validated clinical thresholds.

Changing thresholds changes a subset of decisions, so rule outputs should be interpreted as a baseline for research rather than ground truth.

## Leakage And Proxy Limitation

Even when direct rule features are removed, temporal and scanpath features can indirectly encode rule decisions or synthetic behavior labels.

The current audits are useful because they expose this risk. However, they do not eliminate the need for external validation.

High performance in the synthetic environment may indicate:

- Real behavioral structure in the simulation.
- Direct feature leakage.
- Indirect proxy leakage.
- Generator separability.

The thesis should explicitly separate these interpretations.

## Cognitive-Load Proxy Limitation

The cognitive-load analysis is a proxy only. It combines gaze dispersion, scanpath length, revisits, slice toggling, delayed attention, fixation variance, saccade-like behavior, and signal-quality indicators.

It does not include validated cognitive-load ground truth such as:

- Pupil diameter.
- NASA-TLX.
- Expert workload annotation.
- Task difficulty labels.
- Physiological workload signals.
- Time pressure or error-rate measurements.

Therefore, the output should be called `cognitive-load proxy`, not measured cognitive load.

## Dataset Limitation

The imaging context is derived from a LIDC-style dataset and segmentation objects. This provides useful CT/ROI geometry, but it does not represent all clinical imaging conditions, modalities, acquisition protocols, scanner vendors, lesion types, or radiologist workflows.

The project currently emphasizes CT/SEG context and does not validate generalization to other modalities or clinical tasks.

## User Interface Limitation

The review workstation stage is currently a placeholder manifest. The project does not yet implement a complete AR interface, real-time Tobii stream ingestion, or reader-facing clinical review workflow.

Any AR or workstation claim should be framed as future integration.

## Evaluation Limitation

The current evaluation is synthetic and internal. It includes useful audits, but it does not include:

- Real Tobii recordings.
- Radiologist participants.
- Expert annotation of ROI review adequacy.
- Diagnostic accuracy outcomes.
- Prospective validation.
- Multi-site validation.

## Recommended Thesis Limitation Statement

Use wording like:

> The current study validates the internal consistency and auditability of a gaze-aware ROI attention modeling framework under synthetic assumptions. Because gaze trajectories and behavior labels are generated rather than measured, the results should not be interpreted as evidence of real radiologist behavior prediction or clinical effectiveness. Future work requires real Tobii recordings, expert review labels, and workload ground truth before clinical or human-performance claims can be made.
