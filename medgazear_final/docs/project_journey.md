# MedGazeAR Project Journey

## Prototype Dataset

The initial behavior-learning run used 500 synthetic review sessions. This produced 500 ROI/session-level behavior examples after raw gaze samples were aggregated into engineered features.

That run was useful as a feasibility prototype, but it used only a small subset of the available LIDC-derived ROI geometry: 323 unique ROIs and 12 LIDC patients.

## Available ROI Geometry

The full ROI geometry table contains a much broader source pool: 39,890 ROI/frame rows, 6,844 SEG objects, and 871 LIDC patients.

## Validation Fix

Step 9 behavior learning now uses the 70/15/15 split correctly. Candidate models are trained on the training split, selected on validation macro F1 with validation balanced accuracy as the tie-breaker, then the selected model is retrained on train plus validation and evaluated once on the untouched held-out test split.

## Expanded Synthetic Experiment

The synthetic gaze generator now supports patient-balanced ROI sampling across the full ROI geometry. The expanded run targets 2,000 synthetic sessions with default patient-balanced sampling to improve patient and ROI diversity.

This remains a synthetic experiment. It improves experimental stability and coverage, but it is not clinical validation. Real Tobii recordings and radiologist review are still required before clinical claims.

## Expanded Run Outcome

The 2,000-session patient-balanced run generated 1,034,799 raw synthetic gaze samples and 2,000 ROI/session-level behavior rows. It sampled 1,923 unique ROI IDs across all 871 available LIDC patients.

The behavior-learning comparison now uses validation metrics for candidate model selection, while final performance is reported only on the held-out test split. Negative controls remained low after expansion, supporting that the behavior model is not driven by geometry-only or shuffled-label leakage.

## Step 10 Cognitive-Load Proxy

Step 10 adds a secondary gaze-derived cognitive-load proxy analysis. The proxy uses engineered gaze behavior and signal-quality features, including gaze dispersion, scanpath length, revisits, slice transitions, delayed attention, fixation variance, saccade-like behavior, dropout, blink, invalid burst, outside-CT ratio, and background gaze ratio.

The output labels are low_load_proxy, medium_load_proxy, and high_load_proxy. These are not true cognitive-load labels. The project explicitly reports that validation would require ground truth such as pupil diameter, NASA-TLX, task difficulty ratings, or expert workload annotation.

## Step 10 Cognitive-Load Proxy

Step 10 adds a secondary gaze-derived cognitive-load proxy analysis. It combines percentile-ranked gaze dispersion, scanpath length, revisit behavior, slice toggling, delayed attention, fixation/saccade fragmentation, and weak signal-quality indicators into low, medium, and high proxy labels.

This is not true cognitive load. The project does not include pupil diameter, NASA-TLX, task difficulty ratings, or expert workload annotations, so the output is explicitly reported as an unvalidated proxy only.
