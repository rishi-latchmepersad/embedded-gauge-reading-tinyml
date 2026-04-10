# AI Memory

This file is the stable memory for future AI agents working in this repo.
It should hold facts that matter across sessions, not the active task list.

## Project Goal

- Build a gauge-reading pipeline for low-power embedded hardware.
- Start with a baseline CV model, then CNNs, then a vision transformer.
- Keep the firmware small enough to fit and run well on STM32 N6 hardware.

## Core Hardware

- Target board: STM32 N6 NPU Nucleo board.
- MCU family: STM32N657.
- Camera sensor: IMX335.
- RTC: DS3231.
- Storage: FileX on SD card.
- Model blob storage: external xSPI flash used by the board boot flow.

## Important Data Locations

- Labeled training data lives in `ml/data/labelled`.
- Captured images live in `captured_images/`.
- Board-generated model artifacts live in `st_ai_output/`.
- The deployment-ready TFLite model has been tested from `ml/artifacts/deployment/`.

## Runtime Layout

- `main.c` should stay a bootstrapper and startup coordinator.
- `app_threadx.c` should stay a thread orchestration layer.
- Feature logic should live in small modules with narrow APIs.
- Generated AI runtime code and vendor glue should stay behind wrappers.

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

## Memory Lessons We Learned

- Do not assume "unused SRAM" can be used for executable image space.
- The linker `ROM` region is separate from general-purpose RAM.
- The current board image is constrained more by executable `.text` and `.rodata` size than by total SRAM availability.
- Large AI runtime tables and generated kernels are the main ROM consumers.
- The capture path uses large frame buffers and snapshots, so memory ownership must stay explicit.
- The board has a secure/noncacheable memory story, so DMA/capture buffers must be placed deliberately.

## Secure Buffer / Capture Buffer Lesson

- The capture pipeline uses a large YUV422 frame buffer.
- We also keep a copied snapshot for dry-run inference.
- Those buffers must stay in the right RAM region for DMA and cache coherency.
- If a buffer is moved casually, the camera path can appear to work while inference or logging silently breaks.

## Current Camera / Inference Facts

- The capture pipeline is built around the IMX335 and the STM32 DCMIPP/CSI path.
- The board can run a live optical path, not just test-pattern input.
- The AI path consumes a 224x224 YUV422 capture path.
- The capture buffer size is 100,352 bytes for the current 224x224 YUV422 frame.
- The inference output is logged as a floating-point reading with one decimal place.
- Storage readiness is coordinated by `app_storage.*`, which owns the event flag group used to wait for FileX media.
- The camera middleware is not safe to enter from the ISP background thread and the probe/capture thread at the same time; pause the ISP background loop and also take the shared camera middleware mutex while probe and snapshot setup touch `CMW_CAMERA_*` / `ISP_*` state.
- A hardfault we saw in `ISP_Algo_Process()` / `_ISP_BackgroundProcess()` went away after serializing those camera middleware entry points with the mutex.
- The FileX thread should not hold the media mutex while draining the debug log queue; the SD log service already serializes its own file access and the outer lock can make capture writes look stalled.
- Intermittent DCMIPP error `0x00008100` decodes to `CSI_SYNC | CSI_DPHY_CTRL`, which points at the camera link/CSI side rather than the AI worker or FileX path.
- When `0x00008100` shows up after a full frame buffer has already been reported, the current capture path retries once because it often behaves like a late CSI/DPHY rearm hiccup rather than a hard failure.
- Auto exposure is enabled during probe and now gets a runtime readback at capture start so we can verify it stays on while the model recovery work is in progress.
- The current capture acceptance check still only requires nonzero bytes, so near-black frames can slip through; today's 2026-04-09 captures were almost flat black even with AEC reported on, which means the next quality gate should look at brightness/variation instead of byte-count alone.
- We intentionally bumped the IMX335 seed exposure/gain higher to help the sensor escape the black-frame startup state faster while we debug the capture quality problem.
- We now have a capture brightness gate that should reject frames that are still effectively black or blown out, then nudge exposure/gain and retry before saving the image or queueing inference.
- The brightness gate should stay conservative enough to avoid false rejections, but it should stop another batch of near-black captures from reaching the model.
- If the first brightness gate still leaves us in black frames, increase the nudge step size and give the sensor a longer settle delay before retrying.
- After increasing the exposure seed and adding the brightness gate, the newer 2026-04-09 captures stopped being flat black and started showing a visible gauge, so the capture recovery path is heading in the right direction even if the frames may still need more tuning.

## RTC Facts

- The DS3231 should not be silently overwritten on every boot.
- We previously removed the auto-write-from-build-time behavior.
- RTC timestamp generation is centralized in `ds3231_clock.*`.
- The old DS3231 implementation block has been removed from `main.c`; it only calls into `ds3231_clock.*` now.

## File / Module Responsibilities

### `main.c`

- Boot and system bring-up only.
- No camera, AI, or storage policy should live here.

### `app_threadx.c`

- Thread creation, startup ordering, and high-level orchestration.
- It should not own generic utility code if it can live elsewhere.
- Legacy camera capture/state helpers were removed from the active build; keep this file as orchestration only.

### `threadx_utils.*`

- Generic ThreadX helpers only.
- Delay conversion, mutex helpers, tick helpers, and byte-pool reporting belong here.

### `app_camera_buffers.*`

- Large persistent camera buffers and snapshot storage.

### `app_camera_platform.*`

- Low-level camera board support
- IMX335 chip-ID and reset helpers
- Camera enable/shutdown pin control
- Active DCMIPP handle selection
- IMX335 probe/init helpers and sensor register access
- DCMIPP arm/start helpers for snapshot capture
- IMX335 stream start sequencing
- Compatibility wrappers for camera-related tick helpers while the refactor is in progress
- This module now owns the low-level sensor probe and middleware init path that used to live in `app_threadx.c`.
- `app_threadx.c` should call this module for camera bring-up instead of implementing probe/init/register access inline.

### `app_storage.*`

- FileX media readiness coordination
- Storage-ready event signaling
- The event flag group and sync state live inside `app_storage.c`
- RTC-backed capture filename generation with FileX fallback

### `app_inference_runtime.*`

- AI worker thread
- Inference request queueing
- Inference log thread
- One-shot dry-run frame handling
- The AI/runtime boundary should stay behind this module

### `app_camera_diagnostics.*`

- Camera debug dumps, state snapshots, and capture diagnostics.

### `app_camera_capture.*`

- High-level camera capture flow.
- Capture-state snapshots and frame-acquisition helpers now live here.
- Frame save / SD handoff.
- Dry-run inference queueing after a successful processed capture.
- This module now owns the capture/save orchestration that used to live in `app_threadx.c`.
- The capture single-frame logic and capture-state logging are the active implementations now, not the old `CameraPlatform_*` copies.

### `ds3231_clock.*`

- RTC read/write, timestamp formatting, and RTC status behavior.

### `app_inference_log_utils.*`

- Small reusable inference log formatting helpers.

## Current Build / Tooling

- Use `poetry` for Python environment and scripts.
- Use `pytest` for Python tests.
- Use Unity for C tests when needed.
- Prefer WSL for ML work.
- Use STM32CubeIDE / CubeMX for board code and BSP work.
- Keep Python code typed.

## Memory / Build Gotchas

- Debug builds can overflow the `ROM` region even when RAM still looks available.
- Some log formatting choices are not cheap on embedded C builds.
- RTC filename logging can hide capture progress if it is too verbose, so keep the capture hot path breadcrumb-only unless we are actively debugging timestamping.
- Avoid holding the FileX media mutex around the SD debug-log drain; that can deadlock against the capture save path when both sides try to log and touch media at the same time.
- When isolating FileX/capture deadlocks, it is useful to disable the SD debug-log drain entirely so capture save can be tested without concurrent media writes from the log service.
- Keep the ISP background loop paused until the full capture/save/inference handoff is complete; resuming CMW/ISP too early can hardfault inside the ISP middleware.
- For hardfault isolation, it is valid to disable `CMW_CAMERA_Run()` entirely and keep the background ISP path off until the capture/save path is proven stable.
- A feature that feels "small" can still consume ROM through generated tables or verbose logging.
- The model package, runtime, and app binary must stay in sync.
- The current source model is the board30 piecewise-calibrated scalar candidate in `ml/artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras`. It is essentially perfect on the expanded board30 manifest, but it still misses a few samples in the original hard-case manifest.
- Keep using the hard-case manifest as the acceptance test for model work; the target is still to keep every hard case under 5C error before packaging anything for the board.
- The raw pretrained MobileNetV2 fine-tune on `hard_cases_plus_board30.csv` did not solve the problem: it converged to a midrange prediction around 16-17C on many samples and left 22 of 26 hard cases above 5C error. The spline-calibrated `v8` artifact was still the best hard-case performer before the board30 source model arrived.
- If MobileNetV2 transfer learning is used again, prefer fine-tuning from the best existing scalar checkpoint with the board30 hard-case manifest rather than repeating a from-scratch raw fit.
- The current active risk is the export/quantization path: the fresh int8 TFLite export of the board30 source model still loses accuracy badly on the hard-case manifests, so the packaged int8 artifact is not yet a deployment target.
- The direction-model experiment did not beat the scalar baseline on either hard-case manifest, so it should be treated as a dead end unless a future revision changes the loss or supervision structure.
- The classical geometry baseline is a useful sanity check, but it is not strong enough on the hard cases by itself: the hard-case benchmark on `ml/data/hard_cases.csv` landed at `mean_abs_err=13.9928`, `rmse=24.2054`, `max_abs_err=66.1517`, and `cases_over_5c=10`.
- The board30 source model is the current reference model, but the int8 artifact still needs export/quantization work before we can package it for the board, and the source model still needs a pass to pull the original hard cases back under 5C.
- The first Keras-native QAT experiments on that board30 source model did not solve the deployment gap. The full-network run regressed badly, and the more conservative head-only run still left too many hard cases above 5C, so QAT needs a different recipe if we revisit it.
- The export pipeline itself is working when run directly through Poetry in WSL: the board30 staged model loads in about 5 seconds, representative-example building completes, and TFLite conversion finishes. The current blocker is accuracy loss in the exported int8 artifact, not a model-loading hang.
- The most recent direct board30 int8 export still lands around MAE 8.05C on the original hard cases and 8.45C on the expanded board30 set, with max error around 32.48C, so we still need a better quantization/export strategy or a new training recipe.
- A direct piecewise calibration fitted on top of the v2 board30 int8 TFLite output brings the original hard-case manifest back under 5C max error, but the expanded board30 manifest still has outliers up to about 8C. That means a scalar post-calibration is useful, but it does not fully solve the board30 generalization gap by itself.
- A crop-domain piecewise calibration fitted using the same crop pipeline as training helps the midrange, but it still leaves the 31C hole badly wrong and does not clear the remaining 20C/35C misses. Post-calibration can smooth the raw scalar output, but it cannot replace missing distribution coverage in the source data.
- MixUp, the monotonic pair regularizer, and the direct scalar pairwise interpolation loss all failed to fix the 31C interpolation hole. On the latest runs, `capture_p31c.jpg` stayed around 25C error and the original hard-case set still had 6 cases above 5C. A later mid-band emphasis run that upweighted the 18C..42C region actually made the 31C example worse instead of better. That means the problem is not just a weak output transform; we likely need a different supervision formulation or a different architecture if we want genuinely smooth interpolation.
- The compact CNN and compact direction experiments on the combined hard-case set did not beat the MobileNetV2 teacher. The compact scalar run landed at about 13.8C test MAE, and the compact direction run landed at about 18.6C value MAE with poor angle loss, so this architecture family should be treated as a dead end unless we add a much stronger distillation or preprocessing recipe.
- The first full-backbone GPU fine-tune from the best board30 source model improved the overall test MAE to about 1.34C, but the hard-case mean error is still about 7.83C because a few board30 outliers are far off. That means the model improved, but we still need another pass focused on those outliers before we can call the source model stable.
- The later balanced fine-tunes tightened the original hard-case set further but did not fully clear it: `balanced3` reached about `4.1410C` MAE on the original hard cases with `6` cases still over `5C`, and `balanced4` stayed in the same band at about `4.1701C` MAE with the same `6` cases over `5C`. The expanded board30 set is still dominated by the older black captures, which look like recapture/data-quality problems more than pure model capacity problems.
- If the board30 manifest is filtered down to the clearly valid captures only (`hard_cases_plus_board30_valid.csv`), the same `balanced4` checkpoint drops to about `3.8609C` MAE. That strongly suggests the remaining >5C errors on the expanded manifest are mostly the black captures, not a lack of model capacity.
- A clean fine-tune on `hard_cases_plus_board30_valid.csv` kept the valid-set improvement and scored about `3.8086C` MAE on the valid board30 manifest. The same six original hard cases are still above `5C`, so the remaining work is now about recapturing or replacing those stubborn midrange examples rather than training around invalid black frames.
- The current recapture shortlist lives in `ml/data/recapture_targets.csv` and is centered on the six remaining >5C cases: `capture_p20c_preview.png`, `capture_p35c_preview.png`, `capture_m10c_preview.png`, `capture_0075.png`, `capture_m30c_preview.png`, and `capture_p10c_preview.png`.
- The newest labelled images added under `captured_images/` are mostly useful. The clean-plus-new fine-tune held at about `4.0923C` MAE on the original hard cases and `3.7714C` on the valid board30 set. `capture_p15c.jpg`, `capture_m10c.jpg`, `capture_m25c.jpg`, `capture_p20.jpg`, `capture_p25c.jpg`, `capture_p35c.jpg`, and `capture_p42c.jpg` are useful additions on the current checkpoint. The newer `clean_plus_new4` pass, which adds the replacement `capture_p30c.jpg` shot, reaches about `4.1435C` MAE on the original hard cases, `3.7701C` on the valid board30 set, and `3.6113C` on the combined clean+new manifest. `capture_p30c.jpg` is a clean training sample, but it still lands around `7.0737C` error on that checkpoint, so it is a model weakness rather than a data-quality issue. A focused fine-tune on only the remaining hard misses (`hard_cases_remaining_focus.csv`) did not materially improve the original hard-case set: it landed at about `4.1283C` MAE and still left the same six original cases above `5C`.
- The new `capture_p31c.jpg` shot is also clean, but the current checkpoint misses it badly (`25.8813C` error). That suggests we need more nearby 31C/32C captures if we want to close the midrange gap; another blind pass on the current set is unlikely to help much.
- The hybrid MobileNetV2 interval-head experiment did not improve the situation. The run finished with about `18.7C` validation MAE on the scalar output, `interval_logits_acc` stayed around `0.02`, and the 31C region still did not behave like a smoothly interpolating thermometer. Treat that coarse-bin/residual idea as a failed ablation unless the supervision or head design changes materially.
- The direct scalar interpolation-loss, MixUp, monotonic-pair, and mid-band-emphasis experiments also failed to solve the 31C hole. The next geometry-first experiment should use the direction model so the network predicts needle angle/direction and the sweep calibration converts that to temperature deterministically.
- The geometry-first MobileNetV2 direction run also failed to generalize well: the direction loss improved somewhat, but the converted temperature stayed around `20.9C` MAE on test data and the validation angle MAE stayed around `63deg`. That means the plain unit-vector head is not enough by itself; if we revisit geometry-first learning, an explicit sweep-fraction head or a classical/ML hybrid is a better next step.
- The explicit sweep-fraction head also failed to close the interpolation hole: the warm-started MobileNetV2 fraction run reached about `21.4C` test MAE on converted temperature and only about `0.267` fraction MAE. That suggests we need a more direct geometry detector, not just a normalized scalar head, if we want the model to interpolate like the gauge sweep.
- The new MobileNetV2 keypoint-heatmap experiment also did not solve the interpolation hole. The heatmap branch learned somewhat, but the scalar output still sat near the mean on validation/test and the keypoint-augmented model was still around `20.0C` MAE on the hard-case-style test split. So the auxiliary-head version of geometry-first learning is not enough by itself.
- The recapture shortlist should stay focused on the older stubborn misses that are still obviously bad or still above the `5C` target after training; `capture_p25c.jpg` is no longer considered a recapture candidate after the latest checkpoint handled it well.
  - The relocatable packaging flow needs a Windows-writable staging build directory for the NPU make step. If it uses a WSL path there, the pack tool tries to write to `//wsl.localhost` and fails.

## Current Refactor Direction

- Keep peeling `app_threadx.c` into smaller modules.
- Turn `main.c` and `app_threadx.c` into coordinators.
- Keep AI request/logging plumbing in `app_inference_runtime.*`.
- Split camera bring-up, capture, AI runtime, and storage into separate modules when they grow.
- Keep each refactor slice small enough to build and board-test on its own.

## Stable Working References

- Roadmap: `PLANS.md`
- Repo working rules: `AGENTS.md`

## Things To Preserve

- Boot still succeeds.
- Camera probe and capture still work.
- AI inference should not collapse to zero on valid captures; the next model should produce meaningful nonzero values again.
- FileX logging still writes inference rows.
- The build must stay small enough to fit the current linker layout until we intentionally expand it.
