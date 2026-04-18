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
- Verbose AI bring-up and debug logging strings can also tip the `ROM` image over the limit; if the linker overflows by only a few kilobytes, check whether `app_ai.c` and other log-heavy modules are compiling their console strings in.
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
- The current model winner is the crop-domain calibrated scalar checkpoint in
  `ml/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_all/model.keras`.
  It reaches about `0.0039C` mean absolute error with `0` cases above `5C` on
  the hard-case manifests, including the 31C gap, so that is the source of
  truth for model acceptance.
- The current deployment blocker is the export/package path, not model quality:
  the calibrated champion has not yet been cleanly re-exported for the board
  because the WSL packaging job is currently stuck.
- The calibrated champion is also copied to `ml/artifacts/deployment/prod_model_v0.1/`
  so we have a stable production-candidate name while packaging catches up.
- The current source model is the board30 piecewise-calibrated scalar candidate in `ml/artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras`. It is essentially perfect on the expanded board30 manifest, but it still misses a few samples in the original hard-case manifest.
- Keep using the hard-case manifest as the acceptance test for model work; the target is still to keep every hard case under 5C error before packaging anything for the board.
- The raw pretrained MobileNetV2 fine-tune on `hard_cases_plus_board30.csv` did not solve the problem: it converged to a midrange prediction around 16-17C on many samples and left 22 of 26 hard cases above 5C error. The spline-calibrated `v8` artifact was still the best hard-case performer before the board30 source model arrived.
- If MobileNetV2 transfer learning is used again, prefer fine-tuning from the best existing scalar checkpoint with the board30 hard-case manifest rather than repeating a from-scratch raw fit.
- The current deployment picture is now split cleanly: the calibrated source Keras model is still the model-side winner, while the raw int8 export becomes board-ready once the firmware applies the weighted piecewise calibration. That calibrated int8 path has been repackaged for STM32N6 and retested on the hard-case manifests.
- The STM32 N657 CubeIDE app makefile was once pointed at a stale AI runtime folder name
  (`build_mobilenetv2_scalar_hardcase_warmstart_int8`), but the checked-in runtime bundle
  actually lives under `st_ai_output/packages/scalar_full_finetune_from_best_piecewise_calibrated_int8/st_ai_ws/build_scalar_full_finetune_from_best_piecewise_calibrated_int8/`.
  Keep those paths aligned if the AI package is regenerated.
- The direction-model experiment did not beat the scalar baseline on either hard-case manifest, so it should be treated as a dead end unless a future revision changes the loss or supervision structure.
- The classical geometry baseline is a useful sanity check, but it is not strong enough on the hard cases by itself: the hard-case benchmark on `ml/data/hard_cases.csv` landed at `mean_abs_err=13.9928`, `rmse=24.2054`, `max_abs_err=66.1517`, and `cases_over_5c=10`.
- The board30 source model is still the reference source model, but the board-ready deployment path is now the raw int8 export plus the weighted piecewise calibration in `app_inference_calibration.c`. That path has been repackaged and retested, and it keeps the hard-case manifests under 5C.
- The first Keras-native QAT experiments on that board30 source model did not solve the deployment gap. The full-network run regressed badly, and the more conservative head-only run still left too many hard cases above 5C, so QAT needs a different recipe if we revisit it.
- The export pipeline itself is working when run directly through Poetry in WSL: the board30 staged model loads in about 5 seconds, representative-example building completes, and TFLite conversion finishes. The current blocker is accuracy loss in the exported int8 artifact, not a model-loading hang.
- The most recent direct board30 int8 export still lands around MAE 8.05C on the original hard cases and 8.45C on the expanded board30 set, with max error around 32.48C, so we still need a better quantization/export strategy or a new training recipe.
- A direct piecewise calibration fitted on top of the raw int8 TFLite output brings the original hard-case manifest back under 5C max error, and the weighted calibration also keeps the expanded board30-valid manifest under 5C. That makes the raw int8 + calibration pair the current board-ready deployment path.
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
- The papers we checked point toward a staged detector-plus-conversion design: predict the gauge geometry explicitly first, then convert it to temperature deterministically. The most promising next candidate is a pointer-angle or pointer-fraction detector with a confidence map or keypoint head, plus synthetic/augmented coverage and a fixed sweep calibration layer.
- We now have a detector-first MobileNetV2 model path in the repo that predicts keypoint heatmaps and turns them into gauge value via a deterministic geometry layer. That is the next model family to benchmark against the hard-case manifests.
- The first detector-first benchmark completed, but it was not competitive: it reached about `24.15C` test MAE and `19.29C` validation MAE on the gauge value. That means the geometry head is still not a better baseline than the best scalar models; treat it as a learning signal, not the default path.
- The research-backed path we should preserve is a true geometry-first pipeline: localize the center/tip or pointer angle, then convert that geometry to temperature with the known sweep calibration. Synthetic/augmented coverage should be used to fill sparse regions, and hard-case interpolation should be the primary acceptance test.
- The concrete implementation sketch for that geometry-first path is: keep the classical angle extractor as the sweep-calibration reference, make the learned model predict center/tip heatmaps only, compute angle/fraction deterministically from those keypoints, expose the detector path behind its own CLI wrapper, and validate keypoint MAE plus angle MAE before trusting the converted temperature metric.
- The recapture shortlist should stay focused on the older stubborn misses that are still obviously bad or still above the `5C` target after training; `capture_p25c.jpg` is no longer considered a recapture candidate after the latest checkpoint handled it well.
  - The relocatable packaging flow needs a Windows-writable staging build directory for the NPU make step. If it uses a WSL path there, the pack tool tries to write to `//wsl.localhost` and fails.
- On the live STM32N657 board, the deployed `scalar_full_finetune_from_best_piecewise_calibrated_int8` path was producing a nearly constant raw output even though the input tensor hash was changing. We switched the firmware over to the `scalar_full_finetune_from_best_piecewise_calibrated_int8_r128` package, and the current mitigation remains to force the stable training crop instead of the adaptive bright-region crop by default in `app_ai.c`, because the adaptive crop was too likely to drift away from the training geometry.
- The `scalar_full_finetune_from_best_piecewise_calibrated_int8_r128` package then saturated the raw int8 head at `-128` even with the fixed crop, so the live board was switched to the retrain v9 relocatable export by including `ml/artifacts/runtime/scalar_full_retrain_v9_reloc/build/scalar_full_retrain_v9_reloc.c` in the firmware wrapper. Keep the old `r128` path in mind only as a fallback reference.
- The live `scalar_full_retrain_v9` path is much better, but the camera still drifts enough that the reported temperature can jump around. We now lock IMX335 AEC off after probe and apply a light EMA in the inference runtime so the board reading stays steadier while we tune the remaining calibration bias.
- I removed the temporary board-side temperature trim from `app_inference_calibration.c` because it was effectively hardcoding the reference point. The remaining honest path is to collect a fresh fixed-crop calibration sweep and refit the piecewise correction for the retrain v9 model.
- The current live board model is now `prod_model_v0.1_raw_int8`, wired in firmware as `scalar_full_finetune_from_best_piecewise_calibrated_int8`. It is the hard-case winner we should treat as the default production path, and it uses the weighted piecewise calibration with the fixed 224x224 crop and IMX335 AEC lock. The retrain v9 board path is now just a fallback reference.
- We then re-enabled the live IMX335 AEC/background path for the processed capture loop because the locked-exposure board scene was still under-reading at higher gauge values. `AppCameraCapture_RunImx335Background()` now calls the real `CMW_CAMERA_Run()` background step, and the camera init thread keeps AEC enabled for the live loop while the brightness gate retries until the frame is acceptable. This is the current preferred live-capture policy for the prod model.
- For broad bright/dark robustness, the brightness gate should keep AEC on and still nudge exposure/gain on obvious under/over-exposed frames. That gives us both passive convergence from the ISP and active steering from the app-side retry loop without reverting to a fully manual exposure lock.
- The always-on live AEC/background experiment caused a full-board freeze on this build, so the board is back on the stable manual capture-steering path for now. Treat any future AEC/background re-enable work as a carefully isolated experiment, ideally behind a feature flag and with extra concurrency checks around the camera middleware.
- The stable manual capture path now starts from a slightly brighter IMX335 seed exposure (`4/5` of the configured exposure range instead of `3/4`) so we can lift the live gauge image a bit without re-enabling continuous AEC. This is a small, reversible brightness experiment on top of the stable path, not a replacement for proper calibration.
- The processed-frame brightness gate is now stricter on the dark side (`mean_y` target raised so borderline dim gauge faces get a retry) and keeps the retry loop bounded. This is the safer way to explore brighter live captures without reopening the continuous-AEC freeze risk.
- The first dark-frame retry overshot badly when the exposure/gain step was too large, so the brightness-gate nudge is now intentionally gentler (`1/10` exposure step, `1/16` gain step) and the retry log distinguishes brightness retries from actual DCMIPP errors.
- The brightness nudge is now exposure-first and only touches gain when exposure is already at an edge. That should keep the bounded retry loop from jumping straight from under-exposed to over-exposed on the processed path.
- The dark-frame gate now treats the exact mean threshold as acceptable. That fixes a boundary case where the retry loop would finally land on the threshold and still reject the frame instead of returning the best borderline capture.
- The current live board capture was still much darker than the 45C hard-case reference, so the dark gate target was raised back to a brighter mean and the retry limit was bumped slightly. The goal is to let the bounded loop reach a genuinely usable live scene instead of accepting the first merely-readable frame around the 30C regime.
- The newest live captures showed a second failure mode on the bright side: some frames looked fine to the eye but still pushed the prod model into unstable outputs. The bright gate was pulled down into the mid-bright band so the retry loop rejects those washed-out captures instead of accepting them as soon as they stop being obviously blown out.
- The adaptive gauge crop is back on now that the brighter retry loop produces frames where the detector can lock consistently. The crop helper still falls back to the fixed training crop if the bright-face heuristic fails, so we can get the better geometry on readable frames without reintroducing the old dark-frame failure mode.
- The newest SD-card capture sweep showed that the prod model on the live board scene still wanted a slightly tighter crop and a left bias relative to the raw bright-centroid estimate. I adjusted the adaptive crop to be about 15% smaller and to shift left by 24 pixels so the live crop lands closer to the 45C hard-case behavior instead of stalling in the low-30C band.
- The Python board-capture comparison helper now mirrors the same tighter, left-biased crop so future offline checks stay in sync with the firmware path instead of validating stale geometry.
- The current prod board path is now running without the piecewise output calibration so we can trust the raw scalar from the model directly while we validate the new crop/exposure behavior. The old calibration curve was fit to a different live distribution and was over-correcting the already-useful raw output on the board.
- The inference EMA smoothing is temporarily disabled by setting the alpha to `1.0` so the board exposes the raw scalar immediately instead of blending across frames. This is just a diagnostic step while we validate the current crop and exposure tuning.
- On the current close-up board captures, the adaptive rectifier crop drifts too far downward and misses the needle on low temperatures like `10C`. Until we retrain on close-up framing, the firmware should stay on the fixed training crop for the scalar stage instead of trusting the adaptive crop.
- There is now a dedicated board-style offline eval path in `ml/scripts/eval_board_style_tflite_on_manifest.py` that applies the firmware crop heuristic, resizes with pad, and quantizes inputs before scoring the TFLite model. Use that instead of generic image loading when you want the closest offline proxy to the STM32 board.
- I added a focused weak-case training recipe at `ml/data/board_weak_focus.csv` plus `ml/scripts/run_scalar_head_finetune_from_best_board_weak_focus.sh`. It starts from the current prod checkpoint and upweights only the worst board-style misses so we can improve the large-error edge cases without disturbing the live board path too much.
- For long-running WSL jobs, treat the Windows `VMmem` process as the liveness signal. If its CPU usage stays below 2%, assume the command is effectively stuck or idle and stop waiting on it.
- TensorFlow GPU startup in WSL can stall inside the runtime GPU probe. For the head-only weak-focus retrain, skip explicit GPU enumeration when `--no-gpu-memory-growth` is set and use `nvidia-smi` as the lightweight GPU preflight instead of `tf.config.list_physical_devices('GPU')`.
- Restart WSL before every script launch. Do not reuse a prior WSL session between script runs because stale state can hang TensorFlow or GPU startup.
- The CPU-only weak-focus retrain completed successfully on 2026-04-15 in `ml/artifacts/training/scalar_head_finetune_from_best_board_weak_focus_cpu/`. Final metrics: `test_mae=1.4022`, `test_rmse=2.0739`, `hard_case_mean_abs_error=5.0995`. Hard-case scores improved on the low-end temperatures while `p20c` is still the main weak spot.
- The next training pass should optimize the full sweep, not just the remaining edges. I added `ml/scripts/run_scalar_full_finetune_from_best_full_range.sh` to warm-start from the current best checkpoint, keep a broad board-style manifest in the loop, allow the MobileNetV2 backbone to keep adapting, and use only a modest edge bias so we improve the weak cases without overfitting a narrow subset.
- The full-range pass was close but still regressed a few board-style cases around `30C`, `35C`, `50C`, and some low-end samples. I added `ml/data/full_range_regression_focus.csv` plus `ml/scripts/run_scalar_full_finetune_from_best_full_range_regression_fix.sh` to start from the full-range checkpoint and upweight those regressions while keeping the broader board-style sweep in the training mix.
- Best-performing model so far on the board-style hard-case proxy is still `prod_model_v0.1_raw_int8`. On `hard_cases_plus_board30_valid_with_new5.csv`, the board-style scorer gave `mean_abs_err=5.4489` and `max_abs_err=20.4633`. The recent full-range and regression-fix retrains did not beat that overall, so the prod model remains the baseline.
- The next architecture experiment is a compact CNN coarse-to-fine interval reader: `compact_interval`, built on the small residual CNN backbone instead of MobileNetV2. It is wired into the training CLI and the WSL wrapper, but we have not yet run the comparison pass, so it is a candidate branch rather than a new baseline.
- The first `compact_interval` candidate was exported and board-style evaluated on `hard_cases_plus_board30_valid_with_new5.csv`, but it was not competitive: `mean_abs_err=24.7956` and `max_abs_err=60.5116`. Treat this family as an exploratory dead end for the current gauge task unless we revisit the crop/labeling scheme substantially.
- The gauge range values for `littlegood_home_temp_gauge_c` already live in `ml/src/embedded_gauge_reading_tinyml/gauge/gauge_calibration_parameters.toml` (`min_deg=135`, `sweep_deg=270`, `min_value=-30`, `max_value=50`). The new compact geometry family should read those existing spec values through `GaugeSpec` rather than inventing fresh constants, so the geometry model always stays aligned with the project’s real sweep calibration.
- The compact geometry CNN family is now wired into the ML CLI and WSL training wrapper. It uses the existing `GaugeSpec` min/max/sweep values for `littlegood_home_temp_gauge_c` and is the next compact-CNN candidate to benchmark against the board-style hard-case scorer, but it is not a new baseline yet.
- The first compact geometry full-range run completed on 2026-04-15 in `ml/artifacts/training/compact_geometry_full_range/`, but it was not competitive with the current prod scalar model. The CPU/GPU training metrics looked plausible, yet the board-style scorer landed at `mean_abs_err=30.1417` and `max_abs_err=61.5322` on `hard_cases_plus_board30_valid_with_new5.csv`. Keep `prod_model_v0.1_raw_int8` as the baseline; this compact geometry variant needs a different base model or a different geometry formulation before it can replace prod.
- The keypoint-gated reader cascade prototype is also not competitive as-is. I benchmarked `mobilenetv2_keypoint_geometry_clean` as the localizer with `scalar_full_finetune_from_best_board30_clean_plus_new5` as the scalar reader, using the board-style crop heuristic plus a confidence-gated second pass. On the 19-image `hard_cases.csv` subset, the first pass averaged `mean_abs_err=6.2080`, but the cascade gate was effectively non-discriminative (`first_confidence` clustered around `0.03`), so the best threshold reduced to always taking the first pass. The default second-pass cascade was much worse (`mean_abs_err=19.7777`, `max_abs_err=44.6891`). Keep `prod_model_v0.1_raw_int8` as the best deployed model for now; if we revisit a cascade, it needs a much better localization/confidence signal first.
- I started a dedicated geometry-localizer fine-tune for the cascade branch in `ml/scripts/run_mobilenetv2_geometry_cascade_localizer.sh`. It warm-starts from `mobilenetv2_keypoint_geometry_clean`, uses the board-style hard-case mix, raises the keypoint losses, and turns the scalar value loss off so the run can focus on sharper heatmaps and more stable keypoint confidence. This is the next thing to watch if we want the cascade idea to become viable.
- The compact CNN cascade-localizer branch completed on 2026-04-15 in `ml/artifacts/training/compact_geometry_cascade_localizer/` after a full WSL reset and a GPU build path that skipped the warm-start `.keras` load. It trained cleanly on the GPU, but the held-out test metrics were still poor (`gauge_value_mae=21.5215`, `keypoint_coords_angle_mae_deg=73.6594`), so it is not a production candidate either. Keep `prod_model_v0.1_raw_int8` as the baseline.
- I also evaluated `compact_geometry_cascade_localizer/model.keras` on `hard_cases_plus_board30_valid_with_new5.csv` using the repo's scalar manifest evaluator. It was not competitive (`mean_abs_err=24.0431`, `max_abs_err=69.0000`, `cases_over_5c=25`), so it should remain exploratory only and not replace `prod_model_v0.1_raw_int8`.
- For GPU-backed WSL retrains, always restart WSL before launching the script. A fresh WSL session is part of the reliable training recipe here, especially after a hang or a stale GPU adapter state.
- The rectifier-first MobileNetV2 candidate (`mobilenetv2_rectifier_gpu_nopretrained`) loads correctly now; a smoke test on `model.keras` completed in about 6 seconds and exposed the expected `rectifier_box` output. The full rectified scalar chain on `hard_cases_plus_board30_valid_with_new5.csv` scored `mean_abs_err=8.9485` and `max_abs_err=31.5169`, so it is still not a replacement for `prod_model_v0.1_raw_int8`.
- After widening the rectifier crop before scalar inference, the chain improved substantially. The tuned `--rectifier-crop-scale 1.25` setting scored `mean_abs_err=5.3407` and `max_abs_err=15.7218` on `hard_cases.csv`, and `mean_abs_err=4.0817` / `max_abs_err=15.7218` on `hard_cases_plus_board30_valid_with_new5.csv`, which is better than the current prod scalar baseline on the board-style proxy. Keep the rectifier chain as the most promising offline candidate so far, but still validate it on live board captures before treating it as a deployment replacement.
- A warm-start fine-tune from the best rectifier checkpoint (`mobilenetv2_rectifier_finetune`) with the same `--rectifier-crop-scale 1.25` slightly improved the broader board-style set to `mean_abs_err=3.9951` and `max_abs_err=16.9139` on `hard_cases_plus_board30_valid_with_new5.csv`. On the plain `hard_cases.csv` subset it scored `mean_abs_err=5.4818` and `max_abs_err=16.9139`. That makes the fine-tuned rectifier the current best offline candidate for the broader board-style mix, even though the plain hard-case set is still stubborn in a few low-end samples.
- A second rectifier pass, `mobilenetv2_rectifier_hardcase_finetune`, upweighted the labeled examples nearest the hard-case temperature values while keeping `--rectifier-crop-scale 1.25`. It improved the plain `hard_cases.csv` subset to `mean_abs_err=5.3390` and `max_abs_err=15.4238`, while the broader board-style set landed at `mean_abs_err=4.0934` and `max_abs_err=15.4238`. That makes the hard-case-focused rectifier the current best on the original hard-case subset, although the earlier board-style fine-tune is still marginally better on the broader mixed manifest.
- The board-deployable prod v0.2 promotion is now `artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite`, packaged into `st_ai_output/atonbuf.xSPI2.raw` via `scripts/run_board_export_prod_model_v0_2.sh` and `scripts/run_board_package_prod_model_v0_2_raw_int8.sh`. The STM32 firmware still consumes a single flashed scalar network, so this v0.2 update is a deployable single-model promotion; the rectifier+scalar two-stage chain remains the stronger offline idea but still needs firmware integration before it can replace the on-board runtime.
- The rectifier evaluation wrapper `ml/scripts/run_rectified_scalar_eval.sh` was made self-contained by giving it default rectifier, scalar, manifest, and crop-scale arguments. The current rectifier+scalar chain now runs through that wrapper without manual flags, which avoids the earlier "missing args" stall.
- Sweeping the rectifier crop scale showed that `1.50` is the best setting so far for the current rectifier family. On `hard_cases_plus_board30_valid_with_new5.csv`, the current rectifier model at `1.50` scored `mean_abs_err=9.0643` before the hard-case-focused fine-tune and `mean_abs_err=8.5139` after the v2 hard-case fine-tune. On `hard_cases.csv`, the same family at `1.50` scored `mean_abs_err=8.4842` before the v2 retrain and `mean_abs_err=8.5296` after it. The v2 fine-tune therefore helps the broader board-style mix but still does not beat `prod_model_v0.2_raw_int8` overall.
- The current best offline candidate is the rectifier v3 + rectified-scalar v2 int8 chain. Using `mobilenetv2_rectifier_hardcase_finetune_v3` with `mobilenetv2_rectified_scalar_finetune_v2_int8` and a `1.50` rectifier crop scale, the chain scored `mean_abs_err=5.0344` / `max_abs_err=14.1923` on `hard_cases.csv` and `mean_abs_err=4.3508` / `max_abs_err=14.1923` on `hard_cases_plus_board30_valid_with_new5.csv`. That beats `prod_model_v0.2_raw_int8` on both the original hard-case set and the broader board-style manifest, so the rectifier+scalar path is now the benchmark offline candidate even though it still needs a true two-stage board loader before it can replace the STM32 single-model runtime.
- The rectifier v3 and rectified-scalar v2 board assets are now packaged and aligned for the board runtime. The rectifier export/package flow refreshed `st_ai_output/atonbuf.rectifier.xSPI2.raw`, and the rectified-scalar package flow refreshed `st_ai_output/atonbuf.xSPI2.raw`. The firmware include paths and makefile targets already point at `mobilenetv2_rectifier_hardcase_finetune_v3` and `mobilenetv2_rectified_scalar_finetune_v2`, so the two-stage STM32 path now has matching packaged assets on disk.
- The stage-specific xSPI2 loader was rewriting the models because the staging buffer was cacheable RAM and we were not cleaning D-cache before `BSP_XSPI_NOR_Write()`. I patched both the generic and stage-specific xSPI2 write loops in `app_ai.c` to clean `app_ai_xspi2_program_buffer` before each flash write. This should stop the loader from seeing stale/partial bytes after provisioning and should let the per-stage flash verify succeed on the next boot.
- The scalar reader itself is not inherently a zero-output model: an offline probe on `captured_images/capture_2026-04-15_14-10-06.yuv422` using the fixed training crop returned `raw=42` and `pred=23.245598` from `prod_model_v0.2_raw_int8/model_int8.tflite`. So the board-side `0.0` is likely coming from the runtime/load path or from a capture mismatch, not from the scalar model always predicting zero on that crop.
- The board-facing `st_ai_output/atonbuf.xSPI2.raw` scalar blob had drifted from the current `prod_model_v0.2_raw_int8` package. I synced the root scalar blob back to `st_ai_output/packages/prod_model_v0.2_raw_int8/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw`, which matches the hardcoded scalar signature again. The rectifier root blob already matched its package.
- The two-stage board firmware path is now fully implemented in `app_ai.c`. The scalar model lives at `0x70200000` and the rectifier lives at a separate region `0x70520000` in xSPI2 flash, so both models are permanently resident. The `flash_boot.bat` script was updated to flash both blobs (`FLASH_MODEL=1` flashes scalar at `0x70200000` and rectifier at `0x70520000`).
- Three root causes of the xSPI2 re-provisioning-every-cycle bug were fixed: (1) scalar verify used a memory-mapped read while xSPI2 was in indirect mode — switched to `AppAI_Xspi2ReadFlashProbe`; (2) a single shared `app_ai_xspi2_programmed_size` global caused wrong tail-check offset for the non-last-provisioned stage — replaced with per-stage size globals; (3) `app_ai_xspi2_initialized` flag was not cleared in `AppAI_ReconfigureXspi2ForRuntime` so `EnsureXspi2MemoryReady` skipped re-init into write mode. With separate flash regions and all three fixes in place, verify-on-startup should persist across inference cycles without SD card reads.
- The brightness gate was also fixed: `CAMERA_CAPTURE_BRIGHTNESS_RETRY_LIMIT` raised from 8 to 16, `CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_DENOMINATOR` from 32 to 16, and `CAMERA_IMX335_SEED_GAIN_FRACTION_DENOMINATOR` from 8 to 4. These changes are in `app_camera_config.h`.
- All the above firmware changes are in the source tree but not yet built or flashed as of 2026-04-16. The new binary needs a fresh STM32CubeIDE build followed by `flash_boot.bat FLASH_MODEL=1 FLASH_APP=1`.
- Diagnostic logs were added to `AppAI_DecodeRectifierCropBox` and `AppAI_LogInferenceResult` to expose the raw rectifier box values and the scalar dequantization path. After flashing the new binary, look for `[AI] Rectifier raw output: cx=... cy=... w=... h=...` and `[AI] raw=%d head_zp=%d output_bits=0x%08lx` in the UART log to diagnose why rectifier falls back and why scalar reports 0.0. The most likely scalar-zero cause is `raw_output_value=0` with `head_zero_point>0`, producing a negative dequant value that becomes `-0.0f` (bits=`0x80000000`).

## xSPI2 Dummy Cycle Root Cause (2026-04-17) — MUST READ before touching flash code

**Root cause of all flash read failures, re-provisioning every cycle, and NPU inference zeros**: `DUMMY_CYCLES_READ_OCTAL` mismatch between the chip's CR2_REG3 register and the BSP conf file.

The BSP's `XSPI_NOR_EnterSOPIMode` writes `MX25UM51245G_CR2_DC_20_CYCLES` (0x00) to CR2_REG3 before enabling OPI mode, leaving the chip configured for **20 dummy cycles** on all data read commands (0xEC13 OCTA_READ). But the pack-default `mx25um51245g_conf.h` sets `DUMMY_CYCLES_READ_OCTAL = 6`. The controller sends 6 dummy cycles; the chip expects 20. The chip outputs garbage from the wrong data window — but `HAL_XSPI_Receive` returns `HAL_OK` regardless because it just clocks bytes without validating content.

**Consequences of this bug:**

- `BSP_XSPI_NOR_Read` returns `BSP_ERROR_NONE` but fills the buffer with a fixed garbage pattern → verify always fails → re-provisioning every inference cycle
- `BSP_XSPI_NOR_Write` / `Erase_Block`: erase and page-program commands have **zero dummy cycles** — completely unaffected. These were working all along. The write was landing in flash correctly; the post-write read probe was just reading it back wrong.
- `BSP_XSPI_NOR_EnableMemoryMappedMode`: sets up the MM read command with 6 dummy cycles → NPU reads weights from the wrong phase → all-zero model outputs

**Status-register reads are unaffected** — `DUMMY_CYCLES_REG_OCTAL = 4` is correct for RDSR in OPI STR mode, independent of CR2_REG3. This is why `AutoPollingMemReady`, `WriteEnable`, `PageProgram`, and `Erase_Block` all return success.

**The fix** (`firmware/stm32/n657/Appli/Inc/mx25um51245g_conf.h` — new file, 2026-04-17):

- Project-local override placed in `Appli/Inc/` which appears first in the `-I` search path (`-I../Inc` before the pack directory)
- Sets `DUMMY_CYCLES_READ_OCTAL 20U` and `DUMMY_CYCLES_READ_OCTAL_DTR 20U` to match CR2_REG3 = 0x00 (20 cycles)
- Keeps `DUMMY_CYCLES_REG_OCTAL 4U` and `DUMMY_CYCLES_REG_OCTAL_DTR 5U` unchanged
- Applies to both `mx25um51245g.c` wrapper and both BSP thin-wrapper files since they all include via the same search path

**Confirmed working (2026-04-17):**

- `[AI] xSPI2 stage image already present.` on first verify — no more re-provisioning every cycle
- Rectifier outputs non-zero and stable: `cx=523 cy=609 w=527 h=472` (milli-units, 0-1000 scale)
- Scalar output buffer `bytes=[8180003D...]` = little-endian float `0x3D008081` = 0.031373°C. This IS the model's dequantized float output stored directly in the output buffer — the `Dequantize` node writes a float, not an int8
- `Inference bits=0x80000000` gone; pipeline is fully functional
- `[AI] Raw output int8: 0` comes from `raw_output_info` (an intermediate int8 tensor lookup by name) which may not be found — it is a diagnostic path and does not affect inference correctness
- The `Model output before calibration` float IS the true model output; `Inference value: 0.0` is calibrated+filtered at one decimal — 0.031°C → 0.0 after truncation

**How to identify this bug class in future:** BSP flash calls return `BSP_ERROR_NONE` but `BSP_XSPI_NOR_Read` returns a fixed garbage pattern (not 0xFF erased, not actual data, repeating across calls). Erases and writes appear to do nothing because the post-write probe also reads garbage. Check whether `DUMMY_CYCLES_READ_OCTAL` in `mx25um51245g_conf.h` matches what `EnterSOPIMode` / `EnterDOPIMode` programs into CR2_REG3.

## Current Refactor Direction

- Keep peeling `app_threadx.c` into smaller modules.
- Turn `main.c` and `app_threadx.c` into coordinators.
- Keep AI request/logging plumbing in `app_inference_runtime.*`.
- Split camera bring-up, capture, AI runtime, and storage into separate modules when they grow.
- Keep each refactor slice small enough to build and board-test on its own.

## Stable Working References

- Roadmap: `PLANS.md`
- Repo working rules: `AGENTS.md`

## Current Model Deployment State (2026-04-18)

### Two-stage pipeline (active on board)

- **Rectifier**: `mobilenetv2_rectifier_hardcase_finetune_v3`
  - Keras: `ml/artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras`
  - Int8 TFLite: `ml/artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite`
  - Packaged blob: `st_ai_output/atonbuf.rectifier.xSPI2.raw` (~121 KB, flashed at `0x70520000`)
  - Repackage script: `ml/scripts/run_board_package_rectifier_raw_int8.sh`
- **Scalar**: `scalar_full_finetune_from_best_piecewise_calibrated_int8` ← **updated 2026-04-18**
  - Int8 TFLite: `ml/artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite` (2886096 bytes)
  - Packaged blob: `st_ai_output/atonbuf.xSPI2.raw` (~3.07 MB, flashed at `0x70200000`) — freshly regenerated 2026-04-18 06:52
  - Repackage script: `ml/scripts/run_board_package_rectified_scalar_raw_int8.sh`
  - ST Edge AI package dir: `st_ai_output/packages/scalar_full_finetune_from_best_piecewise_calibrated_int8/`

### Why this scalar was chosen

`mobilenetv2_rectified_scalar_finetune_v2_int8` was previously compiled into the blob (wrong model — same `--name` flag, different source file). The v2 fine-tune on rectifier crops actually degraded accuracy: m30→-25.1, p35→26.9, p50→39.4 vs the piecewise calibrated model: m30→-30.7, p35→35.6, p50→49.0. The rectifier-crop fine-tune introduced noise rather than helping.

### Offline eval command (Python, for reference)

```bash
cd ml
python3 -u scripts/eval_rectified_scalar_on_captures.py \
  --rectifier-model artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras \
  --scalar-model artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite \
  --rectifier-crop-scale 1.25
```

### Repackaging workflow (WSL)

Always use the **Ubuntu-24.04** distro — Docker Desktop's Alpine distro (`wsl -e sh`) has path issues and no bash.

```bash
wsl -d Ubuntu-24.04 -e bash -c "cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && python3 -u scripts/package_scalar_model_for_n6.py --model ... --canonical-raw-path /mnt/d/Projects/embedded-gauge-reading-tinyml/st_ai_output/atonbuf.xSPI2.raw ..."
```

Or use the wrapper script (now defaults to the correct model):

```bash
wsl -d Ubuntu-24.04 -e bash -c "cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_board_package_rectified_scalar_raw_int8.sh"
```

Poetry is not installed in Ubuntu-24.04 WSL — call `python3` directly.

### RTC (DS3231) reset procedure

Set `DS3231_ENABLE_BUILD_TIME_SEED 1` in `firmware/stm32/n657/Appli/Src/ds3231_clock.c`, build + flash, boot once (RTC gets written from `__DATE__`/`__TIME__`), then set back to `0` and reflash.

### SD card provisioning — REMOVED (2026-04-18)

The SD card model provisioning path in `app_ai.c` has been disabled. Models are now permanently flashed via `flash_boot.bat` and the SD card plays no role in model loading. If the xSPI2 flash signature does not match on boot, the board logs an error and aborts rather than falling back to SD. To update models: repackage, update signatures in `app_ai.c`, copy blob to SD (no longer needed), and reflash with `flash_boot.bat FLASH_MODEL=1`. The dead-code legacy single-stage path (`AppAI_EnsureXspi2ModelImageReady`, `AppAI_ProgramXspi2ModelImageFromSd`) was already unreachable and left in place.

### Known accuracy issue (camera distance)

The current captures show the gauge filling most of the 224×224 frame (close camera). The rectifier crops to ~right-center of the dial, missing the needle for temperatures in the 0–20°C range. Labeled captures (`p10c`, `p35c`, etc.) were taken farther away and work well. Fix: move camera back to match training distribution, or add close-up labeled captures to retrain.

## Model Update Process — Root Causes Found (2026-04-18) and Required Steps

Every time we repackage a scalar model for the board, **all three of these must be updated together** or inference will silently produce wrong results. They have caused bugs in three separate sessions.

### The three files that must be kept in sync

1. **`st_ai_output/atonbuf.xSPI2.raw`** — the flash blob (regenerated by the wrapper script)
2. **`firmware/stm32/n657/Appli/Src/ai_network_mobilenetv2_scalar_hardcase_warmstart_int8.c`** — the `#include` path pointing to the generated `.c` file in the ST Edge AI package
3. **`firmware/stm32/n657/Appli/makefile.targets`** — all `USER_OBJS` and FORCE-rule paths pointing to the pre-built runtime `.o` files in the package `st_ai_ws/build_*/` directory

### Bug: wrong `#include` path

`ai_network_mobilenetv2_scalar_hardcase_warmstart_int8.c` contains a single `#include` that wraps the generated model C file. When the package directory changes (e.g. `mobilenetv2_rectified_scalar_finetune_v2` → `scalar_full_finetune_from_best_piecewise_calibrated_int8`), this include still points at the old package. The firmware compiles the old model's C code but links the new blob — so the xSPI2 offset constants for scale/zero_point in the Dequantize node are wrong. The result is a plausible-looking float output that is actually reading garbage bytes from the wrong flash address.

**Symptom:** output is a fixed non-zero float like `-0.133333` that never changes regardless of the gauge position.

### Bug: wrong `makefile.targets` paths

The 15 runtime `.o` files in `USER_OBJS` and the FORCE rebuild rule all hardcode the `st_ai_ws` directory of the old package. Linking old `.o` against a new (possibly larger) model C file causes a ROM overflow or section overlap. The linker error says `.rodata will not fit in region 'ROM'`.

**Symptom:** `region 'ROM' overflowed by N bytes` linker error.

### Correct update procedure (scalar model)

```text
1. wsl -d Ubuntu-24.04 -e bash -c "cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_board_package_rectified_scalar_raw_int8.sh"
   → refreshes st_ai_output/atonbuf.xSPI2.raw
   → refreshes st_ai_output/packages/<model_name>/st_ai_output/<model_name>.c
   → refreshes st_ai_output/packages/<model_name>/st_ai_ws/build_<model_name>/*.o

2. Update #include in ai_network_mobilenetv2_scalar_hardcase_warmstart_int8.c:
   #include "../../../../../st_ai_output/packages/<new_model_name>/st_ai_output/<new_model_name>.c"

3. Update all paths in makefile.targets:
   Replace old package/st_ai_ws/build_* prefix with new package/st_ai_ws/build_* prefix
   (affects ~15 USER_OBJS lines + the elf dependency line + the FORCE rule)

4. Update hardcoded start/tail signatures in app_ai.c if the blob changed:
   python3 -c "d=open('st_ai_output/atonbuf.xSPI2.raw','rb').read(); print('start:', bytes(d[:16]).hex()); print('tail: ', bytes(d[-16:]).hex())"
   Update app_ai_xspi2_signature_start and app_ai_xspi2_signature_tail in app_ai.c

5. Rebuild in STM32CubeIDE

6. flash_boot.bat FLASH_MODEL=1  (from firmware/stm32/n657/ in dev/programming mode)
```

### How to verify the correct model is running

After boot, the UART log will show:

- `[AI] xSPI2 stage image already present.` — signature matched (start bytes checked)
- `[AI] scalar out addr=... bytes=[...]` — first 4 bytes are the output float

If board float ≈ Python TFLite prediction on the same crop, all three files are in sync. If board float is a fixed value regardless of gauge position, the `#include` path or `makefile.targets` is still stale.

### Wrapper script defaults (as of 2026-04-18)

`ml/scripts/run_board_package_rectified_scalar_raw_int8.sh` defaults to:

- `MODEL_IN`: `artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite`
- `OUTPUT_DIR`: `artifacts/runtime/scalar_full_finetune_from_best_piecewise_calibrated_int8_reloc`
- Package dir: `st_ai_output/packages/scalar_full_finetune_from_best_piecewise_calibrated_int8/`

Update these vars in the script when the scalar model name changes.

## Things To Preserve

- Boot still succeeds.
- Camera probe and capture still work.
- AI inference should not collapse to zero on valid captures; the next model should produce meaningful nonzero values again.
- FileX logging still writes inference rows.
- The build must stay small enough to fit the current linker layout until we intentionally expand it.
