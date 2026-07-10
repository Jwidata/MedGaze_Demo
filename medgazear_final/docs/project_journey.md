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

## Phase: Evaluation Integrity and Leakage Repair

This phase was necessary because the earlier behavior-model evaluation used row-level stratified splitting. A later forensic audit confirmed that the default split allowed overlap across cases, grouped targets, and readers between partitions. Those overlaps mean the historical row-level results should be treated as a useful baseline, but not as the primary generalization result for the thesis.

The audit also confirmed the training unit precisely:

- one training example = one synthetic session x one SEG-frame ROI

The model is still trained on gaze-derived numeric features, not on CT pixels and not directly on raw SEG mask arrays. SEG-derived geometry remains the reference frame used to calculate ROI-relative gaze features.

### Changes Implemented

Phase 1 preserved the historical row-level baseline under the explicit name `row_stratified_baseline`, then added:

- `case_grouped_primary`: no `case_id` overlap across train, validation, and test
- `reader_grouped_robustness`: no `reader_id` overlap across train, validation, and test
- persisted split manifests for all evaluated strategies
- overlap auditing for case, reader, `roi_id`, grouped target, and session
- a controlled case-grouped `slice_index` ablation using the exact same split manifest

The current workstation model artifacts were not replaced in this phase. All new outputs were written under `outputs/behavior_learning_evaluation_phase1/`.

### Measured Results

| Evaluation strategy | Feature set | Best model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Case overlap | Reader overlap | Grouped target overlap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_stratified_baseline | row_stratified_baseline | XGBoostClassifier | 0.800 | 0.808 | 0.812 | 0.801 | 512 | 279 | 139 |
| case_grouped_primary | case_grouped_all_features | XGBoostClassifier | 0.804 | 0.809 | 0.810 | 0.804 | 0 | 276 | 0 |
| case_grouped_primary | case_grouped_without_slice_index | XGBoostClassifier | 0.804 | 0.809 | 0.811 | 0.804 | 0 | 276 | 0 |
| reader_grouped_robustness | reader_grouped_robustness | XGBoostClassifier | 0.832 | 0.836 | 0.833 | 0.833 | 499 | 0 | 157 |

The overlap audit shows the intended invariants were achieved:

- case overlap is zero for `case_grouped_primary`
- reader overlap is zero for `reader_grouped_robustness`

### Interpretation

The historical row-level split remains a useful synthetic baseline, but it is not the strongest thesis-facing generalization result because cases and grouped targets overlap across partitions. The case-grouped result is now the primary evaluation because it removes case leakage and grouped-target leakage simultaneously.

In this run, the best model family did not change: `XGBoostClassifier` remained the top model under all evaluated strategies. The measured case-grouped performance is close to the row-level baseline, which is reassuring, but the grouped split is still the more trustworthy result because it enforces case separation.

The `slice_index` ablation under the fixed case-grouped manifest produced almost identical headline metrics and slightly improved macro F1 (`0.8109` without `slice_index` vs `0.8101` with it). That suggests `slice_index` is not providing a major performance advantage under case-grouped evaluation, though it remains part of the historical feature set and was not removed from the deployed workstation in this phase.

The reader-grouped robustness result was also strong in this synthetic setting, but it does not replace the case-grouped result as the primary thesis metric. Reader-held-out evaluation addresses a different question: sensitivity to synthetic reader identity rather than sensitivity to case leakage.

The project does not currently define separate dangerous-miss or false-alert metrics in the behavior-learning evaluation code, so those columns remain intentionally unavailable rather than being backfilled with invented definitions.

### Decision For Next Phase

The next phase will align workstation and UI prediction units with the verified model training unit:

- SEG-frame ROI

That alignment work was not performed in Phase 1. Phase 1 was limited to evaluation integrity, overlap auditing, reproducible manifests, and the `slice_index` ablation.

## Phase: SEG-Frame ROI Prediction and UI Unit Alignment

This phase was necessary because the verified training unit and the workstation prediction unit were not aligned.

- training unit: one synthetic session x one SEG-frame ROI
- previous workstation state: grouped base target state using `base_roi_id(roi_id)`

That mismatch meant the UI could aggregate evidence and review state across several SEG frames even though the saved behavior model had been trained to classify one frame ROI at a time.

### Decision

The primary prediction and coverage unit is now:

- SEG-frame ROI instance

Volumetric grouping is still retained as descriptive context only. It can be shown as related extent metadata for a selected ROI frame, but it no longer replaces frame-level prediction or frame-level coverage counts.

### Changes Implemented

Phase 2 refactored the workstation so that:

- the canonical review-state store is keyed by `roi_id`
- prediction write-back is frame-specific
- Current Slice Coverage counts frame ROI instances on the displayed CT slice
- Case Coverage counts all frame ROI instances in the case
- the evidence panel now reads exact frame ROI state
- deterministic display numbering is based on frame ROI ordering by slice and stable `roi_id`
- `base_roi_id` is retained only as descriptive volume context

### Real Case Validation

For `LIDC-IDRI-0001`, the verified counts are:

- SEG-frame ROI records: 32
- ROI-bearing slices: 9
- frame ROI instances on slice 91: 4

Post-implementation, the canonical case inventory now preserves all 32 frame ROI instances rather than collapsing them to 4 grouped targets for primary coverage counting. The grouped targets still exist as descriptive metadata, but the main coverage counts are now frame-instance counts.

### Tests

Phase 2 added deterministic checks for:

- one state per frame `roi_id`
- frame prediction isolation
- frame-level Current Slice Coverage invariants
- frame-level Case Coverage invariants
- deterministic frame ROI display numbering
- real-case audit for `LIDC-IDRI-0001` (`32 / 9 / 4`)

### Next Phase

The next phase will verify feature parity between synthetic training and live Tobii inference, including missing-feature handling.

That feature-parity work was not part of Phase 2.

## Phase: Synthetic-to-Live Feature Parity and Inference Readiness

This phase was necessary because the behavior model was trained offline on synthetic gaze-derived features, while workstation and live inference had to be checked for exact semantic parity before any further model-promotion or live-validation claims could be made.

### Feature Parity Audit

The currently configured model artifact requires 48 ordered numeric features.

The Phase 3 parity audit first found four semantic mismatches in the shared synthetic replay path:

- `dropout_ratio`
- `blink_ratio`
- `invalid_burst_ratio`
- `jitter_px`

Those mismatches were traced to the canonical gaze-schema normalization path, which was discarding source columns needed for parity. After preserving those fields in the shared gaze schema, the deterministic parity audit reached:

- exact matches: 48 / 48
- mismatches remaining in the tested synthetic parity path: 0

Both the direct shared-builder path and the incremental accumulator path reached full parity on the tested deterministic replay session.

### Fixes Implemented

Phase 3 implemented:

- shared behavior feature building logic for offline extraction and live/replay inference
- fixation, dwell, ROI-hit, scanpath, temporal, quality, and geometry parity through shared formulas
- preservation of dropout, blink, invalid-burst, and jitter fields in the canonical gaze schema
- schema validation before inference
- explicit missing-vs-zero handling through prediction-readiness checks
- prediction readiness states:
  - `READY`
  - `COLLECTING_EVIDENCE`
  - `MISSING_REQUIRED_FEATURES`
  - `INVALID_GAZE`

The workstation now blocks prediction when required features are unavailable instead of silently fabricating zeros for missing required evidence.

### Deterministic Parity Results

The Phase 3 audit wrote machine-readable results under:

- `outputs/behavior_feature_parity_phase3/`

For the audited real synthetic example:

- `session_id = synthetic_session_00000`
- `roi_id = 1.2.276.0.7230010.3.1.4.0.57823.1553343864.578878__frame_0023`

The summary result was:

- model-required feature count: 48
- direct offline-vs-live exact matches: 48
- incremental replay-style exact matches: 48
- readiness: `READY` in both direct and incremental paths

### Remaining Mismatches

No remaining mismatches were observed in the tested deterministic synthetic parity path after the gaze-schema repair.

The project still needs a future live-Tobii-specific audit to determine whether every synthetic training feature has a truly equivalent live counterpart or only a documented approximation under real capture conditions.

### Next Phase

The next phase will evaluate SEG spatial fidelity:

- bounding-box-based hit testing versus true binary SEG-mask membership and near-mask distance

That SEG spatial-fidelity work was not part of Phase 3.

## Phase: SEG Spatial Fidelity - Bounding Box vs True Mask Ablation

This phase was necessary because the current behavior-feature pipeline was spatially SEG-derived but still used bounding-box membership for training-time gaze interaction. The SEG masks themselves were already available, so the key question was whether true binary mask interaction would materially change the evidence or improve behavior-model evaluation.

### Controlled Comparison

The comparison used:

- the same raw synthetic gaze samples
- the same behavior labels
- the same frame-ROI unit
- the same fixed case-grouped split manifest from Phase 1
- the same feature-selection policy (`without_slice_index`)
- the same candidate models, preprocessing, and seed

Only the spatial interaction mode changed:

- `bbox`
- `mask`

### Spatial Disagreement Result

Overall valid-sample disagreement was measurable but modest:

- total valid samples compared: 951,093
- bbox false-inside rate: 0.00443
- inside-classification disagreement rate: 0.00561
- near-classification disagreement rate: 0.03560

ROI shape statistics showed that many masks already fill much of their bounding boxes, but some are substantially sparser:

- mask fill-ratio min: 0.0404
- median: 0.7016
- mean: 0.6926
- max: 1.0

This confirms that bbox interaction is usually close to mask interaction, but not identical, especially for less box-filling or more irregular frame ROIs.

## Phase: Full-Stack Slice Gaze Tracking and Delayed Assisted Cues

This phase corrected workstation behavior without retraining any model, changing behavior labels, or changing the scientific Phase 1 to Phase 5 results. The prediction unit remains one synthetic session x one exact SEG-frame ROI.

### Workstation Corrections

- synthetic replay gaze is now loaded as a session-level raw gaze sequence rather than requiring ROI selection for visualization
- slice gaze history is tracked for every visited CT slice and keyed by `ct_stack_index`
- returning to a slice restores only that slice's stored gaze history and slice-local scanpath
- exact frame ROI evidence and prediction remain keyed by `roi_id`
- slice scanning behavior and ROI review behavior are now handled as separate layers of state

### Cue Timing And Presentation

- Assisted-mode ROI cues are delayed until the review opportunity on that slice is mature
- cue timing now uses existing slice dwell, visit, and review semantics together with feature-evidence readiness, rather than revealing cues immediately on slice entry
- Silent mode remains cue-free during active review
- ROI cue rendering was simplified to a subtle thin-outline presentation so CT anatomy remains visible

### Tobii State Clarity

- idle, preflight-ready, active streaming, and mapping-unavailable states are now distinguished explicitly in the workstation status text
- preflight mapping checks remain separate from scientific evidence accumulation
- bounded mapping diagnostics now capture normalized coordinates, screen coordinates, viewer-local coordinates, image-rect context, and failure reasons for mapping failures only

### Overlay Controls

- the duplicate legend row was removed
- overlay controls now use one compact row with integrated visual markers for ROI, gaze, heatmap, and scanpath

This phase was limited to workstation interaction correctness, replay/history presentation, and state clarity. It did not retrain or promote any behavior model.

## Phase: Live Tobii Workstation Cleanup and Slice-vs-ROI Separation

This phase corrected the workstation so the UI better reflects the real project goal: live Tobii CT review analysis. No model retraining was performed, and no scientific backend phase outputs were changed.

### UI Behavior Corrections

- slice-level gaze tracking now remains active across the full CT stack and is stored by `ct_stack_index`
- slice scan behavior and ROI-specific review evidence are now shown as separate concepts in the UI
- the right panel now treats ROI evidence as exact selected-ROI evidence only, rather than allowing stale off-slice ROI context to appear as if it applied to the current slice
- slices with no ROI now explicitly report that ROI-specific evidence is not applicable on that slice

### Live Tobii State Cleanup

- idle connected state is now separated from active streaming state
- active streaming status now distinguishes waiting-for-gaze, tracking-good, and genuine CT-mapping-unavailable cases
- mapping-unavailable is no longer shown merely because the scientific session is idle or because a previous preflight was incomplete

### Mapping And Rendering Fixes

- gaze points classified as outside the CT image are retained as diagnostics but are no longer rendered over the CT image
- bounded mapping diagnostics now capture normalized gaze, screen coordinates, viewer-local coordinates, native image coordinates, and inside/outside CT results for failures only

### Interaction And Responsiveness

- slice changes now update the CT image immediately while more expensive evidence and panel refresh work is deferred through a small debounce window
- scanpath and gaze overlays remain slice-local and do not connect across slices
- overlay controls remain a single compact row with integrated visual cues

### Assisted Cue Behavior

- Assisted-mode cues remain delayed until review opportunity maturity
- Silent active review remains cue-free
- ROI cue rendering remains subtle and low-obstruction rather than using thick bright boxes

This phase was limited to workstation behavior, UI clarity, live-state handling, and mapping/rendering correctness. It did not alter the trained behavior model, label definitions, or the scientific backend phases.

### Feature Impact

The main changed features were the expected spatial ones. Mean absolute differences were:

- `total_gaze_time_inside_roi_ms`: 34.65 ms
- `total_gaze_time_near_roi_ms`: 200.42 ms
- `fixation_count_inside_roi`: 0.1535
- `fixation_count_near_roi`: 0.769
- `time_to_first_roi_fixation_ms`: 520.06 ms
- `gaze_hit_count_inside_roi`: 2.079
- `gaze_hit_count_near_roi`: 12.025

Non-spatial feature semantics remained shared.

### Model Results

Using the same fixed case-grouped manifest and the same no-`slice_index` feature policy:

| Geometry mode | Best model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| bbox | XGBoostClassifier | 0.8070 | 0.8159 | 0.8161 | 0.8079 |
| mask | RandomForestClassifier | 0.8035 | 0.8139 | 0.8118 | 0.8046 |

The model ranking changed under mask mode (`RandomForestClassifier` became best), but the overall performance difference was small.

### Scientific Conclusion

In this run, mask geometry clearly changes some spatial evidence values, but it does not materially improve case-grouped evaluation performance over the bbox baseline. The safest interpretation is that true mask interaction changes the evidence distribution, especially near-ROI quantities, while bbox approximation remains broadly sufficient for the current synthetic behavioral classification setup.

### Limitation

The synthetic gaze generator itself still depends on SEG-derived centroid and bounding-box geometry (`centroid_x`, `centroid_y`, `bbox_width`, `bbox_height`) rather than direct binary-mask pixel sampling. That means mask-mode training can still be influenced by a generator that was spatially organized around bbox-scale information. This dependency must be documented as a limitation of the current ablation.

### Next Phase

The next phase will focus on workstation functional stabilization and/or real Tobii source validation depending on the practical priority after this spatial-ablation result.

Phase 5 was not implemented here.

## Phase: Real Tobii Source Validation and Live Inference Verification

This phase was necessary because Phase 3 proved synthetic replay and live-style software parity, but it did not prove parity for a physical Tobii source entering the pipeline through the real SDK callback path.

### Tobii Normalization Contract

Phase 5 defined and verified a canonical Tobii normalization contract covering:

- timestamp conversion to milliseconds
- left/right-eye combination policy
- screen-coordinate handling
- image-coordinate mapping through the CT viewport transform
- invalid / dropout handling
- outside-CT and UI-glance flags

The deterministic eye-combination policy is:

- both eyes valid: mean of left and right normalized display coordinates
- one eye valid: use the valid eye
- neither valid: mark the sample invalid

### End-to-End Live Path

The live path is now explicitly:

- Tobii SDK sample
- canonical normalization
- shared gaze schema fields
- shared behavior feature builder
- readiness gate
- exact frame-ROI prediction state

This preserves the same feature semantics used by the offline synthetic training pipeline.

### Hardware Validation Status

In this environment, a real Tobii device was detected and a live SDK stream successfully produced raw samples.

One Phase 5 smoke stream recorded:

- sample count: 101
- effective sampling rate: 60.08 Hz
- median interval: 16.64 ms
- mean interval: 16.64 ms
- interval standard deviation: 0.49 ms

That smoke stream had an invalid-sample ratio of 1.0, so it confirmed device transport and stream timing, but not a fully valid gaze-to-CT experiment trial. A proper operator-facing live validation trial with valid tracked gaze remains necessary for complete end-to-end real-use confirmation.

### Replay Parity

Phase 5 also added replayable live-session recording and a machine-readable validation bundle so recorded Tobii sessions can be compared against replay-derived feature vectors through the same shared builder path.

### Remaining Limitations

- the smoke-stream hardware check verified device detection and transport, but not a full valid CT-mapped experiment episode
- a complete physical-trial validation still requires a user actively tracked on the CT viewport
- no new behavior model was retrained or promoted in this phase

### Next Phase

The next phase will focus on workstation functional stabilization and final experiment-facing UI refinement.

Phase 6 was not implemented here.

## Phase: Real Tobii Valid-Gaze Trial and Replay Parity

Phase 5A had already confirmed that Tobii SDK transport was healthy:

- device detected
- callback stream active
- approximately 60 Hz timing

However, the first smoke capture still produced a canonical valid-gaze ratio of 0.0.

### Verified Root Cause

The root cause was verified from real SDK payload inspection rather than inferred from mocks.

In the bounded smoke captures, the raw payloads reported:

- `left_gaze_point_validity = 0`
- `right_gaze_point_validity = 0`
- left and right gaze-point coordinates as `NaN`

The resulting invalid reason was therefore:

- `LEFT_INVALID_RIGHT_INVALID`

This means the zero valid-gaze ratio in the smoke capture was not a transport failure. It was a real no-tracked-eye condition in that physical capture.

### Fixes Implemented

Phase 5B added:

- explicit invalid-reason reporting for canonical live samples
- raw payload validity diagnostics
- deterministic left/right-eye combination policy reporting
- tracking-quality summaries
- live timing diagnostics
- preflight readiness reporting before scientific recording
- replayable live-session validation bundles

### Real Tracked Trial Status

In this environment, the Tobii Pro Spark was detected and a second smoke validation capture again confirmed healthy transport timing, but still produced no tracked eyes:

- sample count: 103
- effective sampling rate: 60.10 Hz
- median interval: 16.68 ms
- mean interval: 16.64 ms
- both-eye valid ratio: 0.0
- left-eye valid ratio: 0.0
- right-eye valid ratio: 0.0
- canonical valid ratio: 0.0

This is enough to confirm the physical source path and diagnostics, but not a full valid-gaze CT experiment trial. A valid operator-in-front-of-device session remains required for complete live ROI evidence and replay-parity validation.

### Replay Parity Status

Phase 5B implemented the replayability and parity-report path for recorded live sessions, but because the smoke captures contained no valid tracked gaze, no ROI-linked live-vs-replay feature parity result could yet be established from a true valid trial.

### Remaining Limitations

- no valid physical CT-tracked Tobii trial was completed in this environment
- fixation and ROI-evidence validation for real gaze therefore remain pending a proper operator session
- no behavior model was retrained or promoted here

### Next Phase

The next phase will focus on workstation functional stabilization and final experiment-facing UI refinement once a valid live-gaze engineering trial has been completed.
