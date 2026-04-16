# Refactor Plan

This project is already working, so the goal now is to make it easier to grow.
The long-term shape we want is:

- `main.c` and `app_threadx.c` act as coordinators.
- Feature logic lives in small modules with clear ownership.
- Generated or vendor-specific code stays behind thin wrappers.
- Each slice is small enough to build and test on its own.

## Current Direction

We are not trying to rewrite the firmware in one shot.
We are peeling out the biggest files one module at a time so the codebase stays testable.

For model work, the current strongest artifact is the crop-domain calibrated
scalar checkpoint in
`ml/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.keras`.
It reaches about `0.0039C` mean absolute error with `0` cases above `5C` on the
hard-case manifests, including the troublesome 31C region. That is the current
acceptance target and the source of truth for the model line.
The geometry-first experiments remain useful as research signals, but in this
repo they have not beaten the calibrated scalar champion. The classical geometry
baseline is still a useful reference, but it is not strong enough on the hard
cases by itself. The hybrid interval head, full direction head, sweep-fraction
head, keypoint-aux scalar head, and detector-first MobileNetV2 variants all
failed to outperform the calibrated scalar champion on the hard cases, so treat
those branches as ablations unless a later revision changes the supervision or
detector design materially.
The current deployment risk is the export/quantization path: the board export
attempt for the calibrated champion is currently stuck in the WSL pipeline, so
the source Keras artifact is the verified winner while the int8 package still
needs a clean export pass.

Already split out:

- `ds3231_clock.*`
- `app_camera_buffers.*`
- `app_camera_platform.*`
- `app_storage.*`
- `app_inference_runtime.*`
- `app_camera_diagnostics.*`
- `app_camera_capture.*`
- `app_memory_budget.h`
- `app_camera_config.h`
- `app_threadx_config.h`
- `app_inference_log_config.h`
- `app_inference_log_utils.*`
- `threadx_utils.*`

## Refactor Rules

1. Keep changes small.
2. Move one concern at a time.
3. Preserve boot, capture, inference, and logging behavior after each step.
4. Make coordinator files thinner, not smarter.
5. Prefer modules with narrow APIs and few cross-dependencies.
6. If a change adds risk, stop after that slice and test.

## Target Architecture

### Coordinator Files

These should mostly schedule work and forward to modules:

- `main.c`
- `app_threadx.c`
- `app_azure_rtos.c`
- `app_ai.c`
- `app_filex.c`

### Feature Modules

These should own the actual logic:

- `ds3231_clock.*`
- `app_camera_buffers.*`
- `app_camera_diagnostics.*`
- `app_camera_capture.*`
- `threadx_utils.*`
- `app_inference_log_utils.*`
- `app_camera_platform.*`
- future `app_ai_runtime.*`
- `app_storage.*`

## Module-by-Module Plan

### 1. `main.c`

Role:

- Boot setup
- Clock and peripheral bring-up
- App handoff

Next steps:

1. Move any remaining peripheral-specific helpers into their own modules.
2. Keep `main()` focused on initialization order only.
3. Avoid adding feature logic here.
4. Keep DS3231 startup behavior limited to the RTC module.

Done when:

- `main.c` only orchestrates startup.
- No camera, AI, or storage logic lives in `main.c`.

### 2. `app_threadx.c`

Role:

- Thread creation and scheduling
- Camera/inference loop coordination
- Error propagation

Next steps:

1. Keep peeling utility helpers into shared modules.
2. Keep camera bring-up helpers in `app_camera_platform.*`.
3. Keep capture flow helpers in `app_camera_capture.*`.
4. Keep the file as the thread orchestration layer.

Done when:

- `app_threadx.c` mostly creates threads and calls module APIs.
- No large helper blocks remain inline.

### 3. `threadx_utils.*`

Role:

- Generic ThreadX helpers

Current ownership:

- Delay conversion
- Mutex helpers
- Tick helpers
- Byte pool reporting

Next steps:

1. Keep generic RTOS helpers here.
2. Move any other reusable ThreadX utilities here before creating new one-off helpers.

Done when:

- `app_threadx.c` no longer owns generic RTOS utility code.

### 4. `app_camera_buffers.*`

Role:

- Shared capture buffers and camera working memory

Next steps:

1. Keep all large camera buffers here.
2. Split any future temporary scratch buffers into this module first.
3. Avoid reintroducing large arrays into `app_threadx.c`.

Done when:

- Camera frame storage is centralized.
- `app_threadx.c` only references buffer APIs or externs.

### 5. `app_camera_diagnostics.*`

Role:

- Camera debug dumps
- State snapshots
- Capture diagnostics

Next steps:

1. Keep all formatted camera logging here.
2. Move any remaining register dump helpers here.
3. Keep output formatting consistent so logs stay readable.

Done when:

- `app_threadx.c` no longer owns camera diagnostic print logic.

### 6. `app_camera_platform.*`

Role:

- Camera bring-up
- Sensor register access
- Vendor/BSP integration

Next steps:

1. Keep the IMX335 probe, init, stream start, and register access helpers here.
2. Keep `app_threadx.c` as the caller, not the owner, of sensor setup.
3. Move any remaining camera middleware plumbing here before it grows back into the coordinator.
4. Keep the actual probe/init helper implementations out of `app_threadx.c`.

Done when:

- `app_threadx.c` only asks for camera actions instead of implementing them.

### 7. `app_camera_capture.*`

Role:

- Capture state machine
- Frame acquisition
- Capture retries and timeouts

Next steps:

1. Keep the capture loop logic here.
2. Keep the capture state machine separate from diagnostics.
3. Let this module own the detailed capture flow and storage handoff.

Done when:

- Camera capture can be read without digging through thread startup code.

### 8. `app_inference_runtime.*`

Role:

- AI inference invocation
- Model I/O handling
- Postprocessing of outputs

Next steps:

1. Move inference request and result handling out of `app_threadx.c`.
2. Keep the model boundary narrow.
3. Make it easy to swap models without rewriting orchestration code.

Done when:

- `app_threadx.c` only feeds frames into an AI API and receives a value back.

### 9. `app_filex.c` and `app_storage.*`

Role:

- Storage readiness
- SD logging
- File creation and rotation

Next steps:

1. Keep `app_filex.c` focused on FileX setup and readiness.
2. Move CSV/log formatting into a storage/logging module if it grows.
3. Avoid mixing storage policy with camera policy.

Done when:

- FileX setup and log formatting are separated cleanly.

### 10. `ds3231_clock.*`

Role:

- RTC read/write
- Timestamp formatting
- RTC status reporting

Next steps:

1. Keep all RTC policy in this module.
2. Do not reintroduce auto-write-on-boot behavior unless it is an explicit toggle.
3. Add host-set or manual-set paths later if needed.

Done when:

- RTC behavior is controlled by one module only.

## Suggested Execution Order

When we continue, the safest order is:

1. Finish trimming any remaining generic helpers from `app_threadx.c`.
2. Keep camera bring-up helpers in `app_camera_platform.*`.
3. Keep capture helpers in `app_camera_capture.*`.
4. Keep inference invocation in `app_inference_runtime.*`.
5. Split storage/logging policy if it still grows.
6. Revisit memory budgets and linker layout once the code is cleaner.

## Memory Growth Plan

The project is an AI project, so model growth is expected.
That means we should track two budgets separately:

- Executable/constant space
- Runtime RAM / pools / stacks

We should not assume all free SRAM can be used for the executable image.
If the current `ROM` window becomes the limiter again, the next question is whether to:

1. shrink the runtime footprint,
2. move constants or generated code to a different region,
3. or expand the linker layout intentionally.

## Model Recovery Plan

Current update: the crop-domain calibrated scalar checkpoint in
`ml/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.keras`
is still the source-model hard-case winner. It lands at about `0.0039C` mean
absolute error with `0` cases above `5C`, including the 31C gap, so the
model-side acceptance target is satisfied.
For the board, the raw int8 export plus the weighted piecewise calibration is
now the deployment candidate: it has been repackaged for STM32N6, the canonical
xSPI2 blob has been refreshed, and the calibrated hard-case retest still keeps
every sample under `5C`.
I also copied the calibrated source champion into
`ml/artifacts/deployment/prod_model_v0.1/` and the raw int8 board bundle into
`ml/artifacts/deployment/prod_model_v0.1_raw_int8/` so we have stable
production-candidate names for both the source model and the board-ready path.

The current source model is healthy, and the int8 export path now preserves the
hard-case behavior once the board-side piecewise calibration is applied.
Before we do broader model upgrades, we should make sure the deployed camera
path is still giving the model a healthy signal and that the export pipeline is
not introducing a large accuracy drop.

  The classical geometry baseline is a useful sanity check, but it is not the
  deployment answer on its own: the latest hard-case benchmark landed around
  `13.99C` MAE with `10` cases above `5C`, so a pure CV angle extractor still
  needs a better detector or calibration layer.
  The first keypoint-heatmap MobileNetV2 experiment also did not solve the
  interpolation hole: the heatmap branch learned to localize somewhat, but the
  scalar output remained around `20C` MAE on validation/test, so the
  auxiliary-head version of geometry-first learning is not enough by itself.
  The papers point to a staged detector-plus-conversion pipeline, so the next
  candidate should explicitly predict geometry first and then convert it to
  temperature deterministically instead of asking a scalar regressor to invent
  the whole mapping.
  The best concrete version to try next is a pointer-angle or pointer-fraction
  detector with a confidence map or keypoint head, trained with strong
  synthetic/augmented coverage, followed by a fixed sweep calibration layer.
  That is closer to the published meter-reading pipelines than the current
  auxiliary-head experiments and gives us a cleaner failure surface when a
  sample is outside the known sweep geometry.
  We now have a detector-first MobileNetV2 path in the tree that predicts
  keypoint heatmaps and converts them to gauge value through a deterministic
  geometry layer; the next task is to train and benchmark that version against
  the hard-case manifests.
  The first detector-first benchmark landed at about `24.15C` test MAE and
  `19.29C` validation MAE on the gauge value, so it is not yet a better
  baseline than the strongest scalar models. Keep it as a learning signal, not
  the default path.
  The next paper-aligned implementation sketch should be:
  - `baseline_classical_cv.py`: keep the classical angle extractor and make the
    angle-to-value conversion helper the shared sweep-calibration reference.
  - `models.py`: replace the detector-first gauge-value head with a detector
    that predicts center/tip heatmaps only, then convert those coordinates to a
    sweep fraction/value in a deterministic layer.
  - `training.py`: add a dedicated geometry-detector training family that
    reports keypoint error, angle MAE, and final temperature MAE separately.
  - `run_training.py`: expose that family behind a single CLI flag so the new
    pipeline can be trained without touching the existing scalar baselines.
  - `export.py` and the eval scripts: load the geometry detector, compute the
    derived angle/value metrics, and keep the conversion logic identical to the
    training path.
  - `run_*.sh`: keep a single WSL wrapper for the geometry detector so GPU
    runs can be restarted and tailed consistently like the other ML jobs.
  Start by verifying that the detector improves keypoint localization and
  angle MAE before expecting the final temperature MAE to beat the scalar
  baselines.
  The literature-backed next step is a true geometry-first pipeline:
  1. localize the gauge center and pointer tip, or directly predict the pointer
     angle/fraction with a compact detector head;
  2. convert that angle/fraction to temperature with the known sweep min/max;
  3. use synthetic/augmented coverage to fill sparse sweep regions before more
     end-to-end fine-tuning;
  4. validate interpolation on the hard-case manifests first, not just on the
     average validation split.
  That keeps the learned part focused on geometry and leaves interpolation to
  deterministic sweep calibration instead of asking a scalar head to invent the
  whole mapping.

  The first full-backbone GPU fine-tune from the best board30 source model
  improved the overall test metrics to `mae=1.3435`, but the hard-case mean error
  is still `7.8301` because a few board30 captures are way off. The next GPU
  iteration should focus on those outliers instead of changing architecture again
  too early.
  The newer balanced passes tightened the original hard-case set further:
  `balanced3` reached `mean_abs_err=4.1410` with `cases_over_5c=6`, and
  `balanced4` landed at `mean_abs_err=4.1701` with `cases_over_5c=6`, so the
  remaining misses are stubborn rather than noisy.
  When the obviously black board30 captures are filtered into
  `hard_cases_plus_board30_valid.csv`, `balanced4` drops to
  `mean_abs_err=3.8609` on the valid board30 set, which confirms that the black
  captures are the main reason the expanded manifest still looks broken.
  A follow-up clean fine-tune on `hard_cases_plus_board30_valid.csv` preserved
  that improvement and reached `mean_abs_err=3.8086` on the valid board30 set
  with the same `6` original hard cases still above `5C`. That suggests the
  clean data path is the right one, and the next gains will likely come from
  recapturing or replacing the remaining stubborn midrange cases rather than
  training around invalid black frames.
  The current recapture shortlist is in `ml/data/recapture_targets.csv`:
  `capture_p20c_preview.png`, `capture_p35c_preview.png`,
  `capture_m10c_preview.png`, `capture_0075.png`, `capture_m30c_preview.png`,
  and `capture_p10c_preview.png`.
  The newest labelled captures added to `captured_images/` were mixed: the
  clean-plus-new fine-tune held at `mean_abs_err=4.0923` on the original hard
  cases and `3.7714` on the valid board30 set. `capture_p15c.jpg`,
  `capture_m10c.jpg`, `capture_m25c.jpg`, `capture_p20.jpg`, `capture_p25c.jpg`,
  `capture_p35c.jpg`, and `capture_p42c.jpg` are useful additions on the newer
  checkpoint. The later `clean_plus_new4` pass, which adds the new `p30c`
  replacement shot, reaches `mean_abs_err=4.1435` on the original hard cases,
  `3.7701` on the valid board30 set, and `3.6113` on the combined clean+new
  manifest. `capture_p30c.jpg` is a clean training sample, but it still lands
  at about `7.0737C` error on the current checkpoint, so it remains a model
  weakness even though it is no longer a data-quality issue. A focused
  fine-tune on only the remaining hard misses (`hard_cases_remaining_focus.csv`)
  did not materially improve the original hard-case set: it landed at about
  `4.1283C` MAE and still left the same six original cases above `5C`.
  The new `capture_p31c.jpg` shot is also clean, but the current checkpoint
  misses it badly (`25.8813C` error), so the next gain likely needs more nearby
  31C/32C captures rather than another blind pass.
  The direct scalar interpolation-loss runs did not solve this either: the
  pairwise interpolation pass kept the original hard cases around `4.19C` MAE
  but still left `capture_p31c.jpg` near `26C` error, and the stronger
  mixup-plus-pairwise run did not change that conclusion. A later mid-band
  emphasis run that upweighted the `18C..42C` region actually made the hole
  worse, with `capture_p31c.jpg` falling to about `-1.10C` prediction and the
  combined manifest degrading instead of improving. That suggests we need a
  different supervision strategy or a different architecture if we want
  genuinely smooth interpolation through the sparse midrange.
  A crop-domain post-calibration pass on the clean board30-valid set helped the
  midrange and lowered the combined hard-case MAE, but it still left `capture_p31c.jpg`
  badly wrong (`24.9206C` error) and did not clear the remaining 20C/35C holes.
  That means scalar post-calibration is useful, but it is not enough to make the
  model universally interpolative on its own.
  The next interpolation experiments, MixUp and monotonic pair regularization,
  did not change that conclusion: the `p31c` hole stayed around `25C` error and
  the original hard-case set still had `6` cases above `5C`. This points more
  toward missing local coverage or a different architecture/loss than a simple
  output-shaping fix.
  The recapture shortlist should stay focused on the older hard misses that are
  still obviously bad or still above the `5C` target after training.

Next steps:

1. Confirm auto exposure stays enabled through the full camera bring-up and
   capture flow, including a runtime readback at capture start.
2. Keep the capture brightness gate in place so we do not feed black or white
   frames into model evaluation or export calibration.
3. Re-export the board30 piecewise-calibrated source model through the int8
   path and verify the new artifact against both hard-case manifests. The direct
   Poetry export path works in WSL; the remaining issue is the accuracy drop in
   the exported int8 artifact, not a model-load hang.
4. If the int8 export still diverges from the source model, adjust the export
   calibration strategy or move to a different quantization-aware strategy
   before trying another architecture. The initial Keras-native QAT passes did
   not yet meet the hard-case target.
5. Keep doing GPU-backed MobileNetV2 fine-tunes before falling back to compact
   CNNs again. The compact scalar and direction student runs both underperformed
   on the hard-case set.
6. Treat the exported int8 artifact as a candidate only if its scalar output
   can be corrected enough to preserve the hard-case acceptance target. A
   simple piecewise post-calibration on the int8 output currently gets the
   original hard cases under 5C, but the expanded board30 set still needs work.
7. Keep the board-side inference path stable while we iterate on model quality.

Done when:

- Auto exposure is confirmed on in the working capture path.
- The capture gate rejects black/white frames before they reach storage or
  inference.
- The capture seed produces visibly non-black frames on the board.
- The source board30 spline-calibrated scalar candidate stays below 5C error on
  the expanded board30 manifest, and we still need to pull the original hard
  cases back under 5C too.
  - The crop-domain calibration variant improves some midrange cases, but it
    still leaves the 31C hole and a few 20C/35C misses, so the raw model still
    needs better generalization rather than just a stronger output transform.
- The first full-backbone GPU fine-tune from the best board30 source model
  improves the overall test MAE to about 1.34, but the hard-case set is still
  above target because of a handful of board30 outliers.
  - The latest balanced passes improved the original hard-case manifest to just
    over 4C MAE, but six cases still remain above 5C and the expanded board30 set
    is still dominated by the older black captures.
- The raw int8 export now becomes board-ready once the firmware applies the
  weighted piecewise calibration. The calibrated hard-case retest keeps both
  the original hard-case manifest and the expanded board30-valid manifest under
  5C on every sample, so that is the deployment path to preserve.
- The direction-model experiment is evaluated against the same hard-case set
  and beats the scalar baselines before we adopt it.
- The packaged STM32N6 artifact matches the raw int8 model plus the firmware
  calibration closely enough that the hard-case acceptance criteria stay intact
  after export.

## Acceptance Criteria For Each Slice

Every refactor slice should end with:

- a successful syntax check or build,
- a boot test on the board,
- capture still working,
- inference still producing a value,
- logging still writing rows when storage is available.

## Notes For Future Work

- Keep `app_threadx.c` as the coordinator.
- Keep `main.c` as the bootstrapper.
- Keep generated/vendor glue behind thin wrappers.
- If a module starts to feel like "everything camera-related," split again.
- When the model collapses to zero, check exposure and capture quality first,
  then retrain/calibrate before assuming the firmware inference path is broken.
