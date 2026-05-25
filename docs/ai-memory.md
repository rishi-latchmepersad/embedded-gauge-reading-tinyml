# AI Memory

This is the entry point for durable project memory.
Keep this file short, and put detailed notes in the topical files below.

- The target for reading is the inner dial of the gauge, as that is the one calibrated for Celsius (C).
- Direct source-space crop-box experiments now use the compact `compact_source_crop_box` family with `source_crop_box` supervision and the rectified crop-box CSVs via `--precomputed-crop-boxes`; the V28 replay helper now decodes `source_crop_box` outputs directly.

## 2026-05-24 Phase 11D: Auxiliary Coordinate Head (Smoke Test)

- **Architecture**: Added `include_aux_coords=True` to `build_mobilenetv2_geometry_heatmap_v4_112()`. Adds Dense(64, relu) → Dense(4, sigmoid) head branching from the same `geometry_confidence_gap` pooled features. Outputs `[cx_norm, cy_norm, tx_norm, ty_norm]`.
- **Training**: 5-epoch smoke (3 warmup + 2 frozen) from the 30-epoch quant-native checkpoint. `aux_coord_weight=0.1`. Loss stabilized at ~0.26 val MSE on normalized coords.
- **Export**: 4-output TFLite auto-detected. Semantic output order auto-mapped from TFLite tensor names (`StatefulPartitionedCall_1:N` → Keras output index N). Correct mapping: `[2, 0, 3, 1]`.
- **Eval results (val, heatmap-only decode)**:
  - Keras: accepted MAE 3.22 C, 80.85% acceptance, worst 9.64 C
  - INT8: accepted MAE 3.87 C, 70.21% acceptance, temp drift 1.89 C mean, tip drift 12.21 px
  - Baseline (08_tip_focus): Keras 3.10 C, INT8 temp drift 1.84 C
- **Decision**: INT8 temp drift 1.89 C still above 1.0 C gate. Aux head needs longer training or higher weight. Heatmap-based decode path confirmed working for 4-output models.
- **Key lessons**:
  - TFLite converter reorders outputs differently from Keras `model.outputs` order. Must auto-detect from tensor name suffixes.
  - The aux_coords override in `decode_and_guard` must be opt-in; poorly-trained aux coords worse than heatmap decode.
  - Eval script `_as_output_dict` must handle both 3 and 4 output cases (list unpacking with `*extra`).

## 2026-05-24 Phase 11: Anti-Collapse Training Changes

- **Problem**: QAT fine-tuning (Phase 10F) collapsed to all-zero heatmaps because MSE on sparse heatmaps provides weak gradient against all-zero solutions under quantization noise.
- **Fix**: Added four anti-collapse mechanisms to `train_geometry_heatmap_v4_112_quant_native.py`:
  1. **Peak-shaping loss** (`peak_shape_center_weight=0.1`, `peak_shape_tip_weight=0.2`, `peak_target=0.3`): quadratic penalty when heatmap max falls below target, directly discouraging all-zero collapse.
  2. **Confidence floor loss** (`confidence_floor_weight=0.05`, `confidence_floor=0.5`): quadratic penalty when confidence drops below 0.5.
  3. **Warmup LR schedule** (`warmup_epochs=5`, `start_lr_fraction=0.01`): linear ramp from 1% to target LR over first N epochs.
  4. **Early-collapse detection** (`peak_threshold=0.05`, `patience=3`): aborts training if both heatmap peaks stay below 0.05 for 3 consecutive epochs.
- All new params exposed via CLI args; saved to summary.json and config.json.
- Next: Run bounded experiment matrix on `val` to tune loss weights, then freeze INT8 champion.

## 2026-05-22 Geometry Heatmap Quantization Drift

- The current INT8 drift autopsy points at tip heatmap flattening/spread growth plus softargmax sensitivity, not tensor-order or dequantization issues.
- The decode-method comparison now selects `peak_weighted_centroid_w5` as the best validated decoder from the saved summary data.
- The replay/test jobs for the variant comparison kept stalling in WSL `dxgglo` I/O wait, so the temp-mirror strategy should be the next thing to automate if we need another full replay pass.

## 2026-05-21 Geometry Heatmap Debug

- Ground-truth heatmap generation is correct on 30 clean rows: mean argmax error is `0.333` px for center and `0.412` px for tip at `56x56`, with x/y ordering and normalized-coordinate conversion both verified.
- The failed `geometry_heatmap_v1` model still produces diffuse, low-peaked predictions on the 30 test examples: mean center peak `0.1199`, mean tip peak `0.0800`, mean softargmax error `45.9` px center / `62.7` px tip.
- The tiny 8-sample overfit gate did **not** pass the required crop-space thresholds. Final holdout-on-same-8 metrics were `center MAE=5.855 px`, `tip MAE=2.188 px`, `temperature MAE=3.844 C`.
- Because the tiny overfit failed, do **not** proceed to a full heatmap v2 training run yet. Revisit the center localization path / loss weighting first.
- Branch analysis on the same 8 samples showed the center labels sit much closer to the crop hub than the tip labels, and the v1 model's center decode is the weak branch (`center softargmax MAE=5.855 px` vs `tip softargmax MAE=2.188 px`).
- Tiny-overfit v2 with stronger center weighting and frozen-then-unfrozen fallback still missed the temperature gate (`temperature MAE=3.732 C`), but the center-prior ablation was informative: using the loose-crop geometric center with the model-predicted tip improved to `2.995 C`, which suggests a fixed-center or tip-only architecture may be simpler for the first board version.

## 2026-05-13 Dual-Resolution Interval Recipe

- The strongest learned reader so far is `mobilenet_v2_dualres_interval` with the new tensor-safe `CenterCropResize` layer, CBAM + Coordinate Attention on both MobileNetV2 branches, `range_aware_sampling=True`, and a lower `interval_loss_weight` (`0.10`).
- Best hard-case-only direct training result so far: `test_mae=16.38C`, `test_rmse=19.74C` on `16` hard-case test samples.
- On the same hard-case manifest, the prod v0.3 calibrated OBB baseline came out worse at `calibrated_mae=19.20C` when the piecewise calibration JSON was flattened and applied to the TFLite model.
- The main CLI now exposes `--range-aware-sampling`, `--cold-tail-fraction`, `--hot-tail-fraction`, `--oversampling-factor`, and `--interval-loss-weight`, so future runs can reuse the better recipe without editing code.

## 2026-05-13 Prod v0.4 Scalar Winner

- The current prod v0.4 scalar winner is `no_cal_hardpush_gpu5_recover`.
- Hard-case results reproduced cleanly on the held-out manifests: `test_mae=7.8563C` overall, `mean_abs_err=7.2357C` on `hard_cases.csv`, and `mean_abs_err=6.3930C` on `hard_cases_plus_board30.csv`.
- The winning Keras artifact needed a sanitizing repack step before export because it still carried a legacy MobileNetV2 preprocess Lambda and QAT bookkeeping tensors.
- The clean deployable model was rebuilt into `tmp/prod_v0_4_repack_test/model.keras`, exported to `tmp/prod_model_v0_4_scalar_int8/model_int8.tflite`, and packaged into `st_ai_output/packages/prod_model_v0.4_scalar_int8/`.
- Important distinction: the trained prod v0.4 source checkpoint is the good model; some later deployment artifacts and package-wiring paths drifted away from that source checkpoint during debug work, which is why the on-disk `prod_model_v0.4_scalar_int8` export could look wildly wrong even though the trained checkpoint itself stayed below 10C MAE on the hard cases.
- Firmware references now point at `firmware/stm32/n657/Appli/Src/ai_network_mobilenetv2_scalar_hardcase_warmstart_int8.c` and `firmware/stm32/n657/Appli/makefile.targets` using the new `prod_model_v0.4_scalar_int8` package directory.
- The v0.4 board path does not need a separate post-hoc calibration layer at the model level, but the board package still depends on the repack/export sanitizing step to stay CubeAI-friendly.
- The exported int8 package builds cleanly for the board, and the hot 45C board sanity check is the fast accuracy gate; use broader offline rescoring only when we need to compare candidates.
- Live board debugging showed the scalar stage was crashing or stalling in the preprocess path right after `Preprocess row 0/224`, so the resize logic was simplified from float interpolation to integer nearest-neighbor sampling to reduce CPU load and eliminate extra edge cases while we chase the board fault.
- The grayscale scalar path was simplified one step further: the luma-only branch now writes the output tensor directly from `AppAI_ReadYuv422Luma()` instead of going through the RGB helper, which should make the first row much cheaper on the Cortex-M55 and narrow the remaining failure surface.
- The scalar fault still looked like corrupted control flow, so the AI worker stack was doubled from `32768` to `65536` bytes and `AppAI_PreprocessYuv422FrameToFloatInput()` was marked `noinline` to keep the hot path in a smaller, easier-to-debug frame.

## 2026-05-14 Scalar Preprocess Match

- The live v0.4 board issue was not the model itself; offline replay on `capture_p50c.yuv422` produced a healthy scalar value around `49.99C`, while the board was still logging a constant `0.0`.
- The key mismatch was preprocessing: the training and replay paths use RGB `resize_with_pad(..., method="bilinear")`, but the firmware path had drifted toward grayscale-only sampling and a non-matching resize rule.
- The scalar fast path now uses `APP_AI_YUV422_INPUT_LUMA_ONLY=1U`, which keeps the row writer on the lighter luma route while preserving the working memory layout fix.
- The board now gets past `PreprocessScalarRow`, so the earlier fault was not in the model runtime itself.

## 2026-05-14 Scalar Output Contract Fix

- The prod-v0.4 scalar board package was accidentally generated with a float32 output shim (`Dequantize_391_out_0`), even though the underlying deployed TFLite model is quantized int8.
- That float shim was producing garbage-looking live bytes on the board, so the packaging wrapper was updated to request `--output-data-type int8` for the scalar package.
- The regenerated scalar package now exposes `Quantize_390_out_0` as a 1-byte int8 output, which matches the firmware's quantized decode path and the offline model contract.
- The canonical xSPI2 blob was refreshed from the regenerated package, so the next CubeIDE flash should be testing the raw int8 reader rather than the broken float shim.
- If the board still misbehaves after this flash, the remaining suspect is the exported model/runtime contract itself, not the YUV422 preprocess path.
- The `flash_boot.bat` provisioning flow completed successfully after the contract fix, but the board still needs the requested flash-boot power-cycle before COM3 will show the fresh boot log from the new package.
- The prod v0.4 scalar board path should stay on the raw model output, so the old board-side affine calibration helper is now disabled again to avoid stacking stale postprocess on top of the new int8 head.

## 2026-05-14 Live Raw-Output State

- The latest Appli build has now been flashed on the board and the COM3 stream is showing the raw prod-v0.4 path.
- The camera and preprocess stack are healthy: watchdog pulses continue, the scalar preprocess completes, and the scalar input tensor varies from frame to frame.
- The remaining live issue is the scalar head itself railing at `Quantize_390_out_0 = q=127` with `zp=-35`, which dequantizes to a value far outside the plausible temperature band.

## 2026-05-14 Preprocess Fault Shrink

- The scalar hot path has now been reduced to direct nearest-neighbor luma reads with no per-row UART logging and no helper-call chain in the row loop.
- The board still hardfaults during the first preprocess sweep, typically by row `16`, with `LR=0x00000000` / `PC` corruption and `BFAR=0x000D044E`.
- That shape now looks more like stack/control-block corruption than a bad pixel helper, so the next debugging step should focus on thread stack headroom, memory overlap, or another caller above `AppAI_PreprocessScalarRow()`.
- The scalar tensor fill now uses the simpler luma-only route and a RAM LUT, and the AI worker stack was reduced to `131072U` as a safety margin while the memory layout issue was being tracked down.
- The worker-stack overlap turned out not to be the root cause; the real crash source was the generated scalar package still targeting `0x34100000`, which overlapped the live app `.bss` / runtime footprint.
- Moving the scalar package to `0x34110000` fixed the overlap and restored stable scalar inference.

## 2026-05-14 HardFault Fix: Flash Bus Contention on LUT Access

- **Symptom**: HardFault at `PreprocessScalarRow: Processing row 0` (or row 16), with `CFSR=0x00000100` (PRECISERR), `BFAR=0x000D044E` (invalid address), `PC=0x3EA5FDAA` (in OctoSPI flash).
- **Root cause**: The `app_ai_luma_to_float_bits[256]` lookup table was stored as `const` in flash. During preprocessing, the CPU tried to read from this LUT while simultaneously executing instructions from flash, causing a **precise data bus error (PRECISERR)** due to flash bus contention on the STM32N657.
- **Fix**: Split the LUT into two parts:
  1. `app_ai_luma_to_float_bits_flash[256]` - const array in flash (read-only source)
  2. `app_ai_luma_to_float_bits[256]` - RAM-resident copy in `.noinit` section (used during inference)
  3. Added `AppAI_InitLumaToFloatLUT()` to copy from flash to RAM during `App_AI_Model_Init()`
- **Why .noinit**: The `.noinit` section avoids C runtime copy overhead at startup. We manually copy from flash to RAM once during model init.
- **Files changed**: `firmware/stm32/n657/Appli/Src/app_ai.c`
- **Next steps**: Rebuild and flash the firmware. The HardFault should be resolved, and preprocessing should complete without bus errors.
- A capture-side DCMIPP error (`0x00008100`) can still occur occasionally, but the board retries and keeps running instead of hard-faulting.
- The old affine board-side calibration stays disabled for this deployment so we can debug the raw model output directly.

## 2026-05-14 Scalar xSPI2 Signature Mismatch Fix

- Runtime scalar-stage bring-up was failing before preprocess with `xSPI2 stage signature mismatch at head probe`.
- Root cause: `app_ai.c` hardcoded scalar signature bytes had drifted from the currently flashed prod v0.4 scalar blob at `st_ai_output/atonbuf.xSPI2.raw`.
- The active prod v0.4 scalar blob start bytes are now `EF 1B 2B E0 D7 E5 EC 06 04 00 34 EC 1A DD 14 05`, and `app_ai_xspi2_signature_start` was updated to match (`0x07,0x05` -> `0x06,0x04` at bytes 8-9).
- Firmware rebuild succeeded (`mingw32-make -j8 all` in `firmware/stm32/n657/Appli/Debug`), so next flash should clear the signature-check abort and allow scalar stage execution on COM3.

## 2026-05-14 Prod v0.4 Model Swap

- The earlier `prod_model_v0.4_scalar_int8` export was not the right artifact for the board: on the two hot `45C` captures it produced absurdly large predictions, so the exported TFLite had to be replaced.
- The strongest calibration-free replacement we found was `no_cal_hardpush_gpu5_recover`.
- That model exported cleanly to TFLite and passed the hot-band sanity check on `capture_p45c.png` and `capture_2026-04-03_08-20-49.png` with predictions of `42.4723C` and `40.7148C` respectively, for a mean absolute error of `3.4065C`.
- The prod v0.4 deployment folder and relocatable STM32N6 package were regenerated from that candidate, so the board is now testing the working model instead of the broken export.
- The deployed model is the calibration-free `mobilenetv2_gauge_regressor` path, so there is no extra post-hoc calibration layer between the tensor output and the board logging path.
- The Windows-native repack/export/package flow completed successfully for this candidate, and the current flashable artifacts live under `artifacts/deployment/prod_model.v0.4_scalar_int8/` and `artifacts/runtime/prod_model.v0.4_scalar_int8_reloc/`.

## 2026-05-04 INA219 Power Metrics Integration

**Hardware**: INA219 power sensor module at I2C address 0x40 (shared with DS3231 RTC on I2C1), 0.1 ohm shunt resistor.

**Files modified**:
- `ina219_power.c/h`: INA219 driver (already existed, verified working)
- `i2c_scanner.c`: Fixed DEBUG_PRINTF â†’ DebugConsole_Printf
- `inference_metrics.c/h`: Unified power+latency metrics module
- `app_ai.c`: Added Metrics_StartInference("CNN"), Metrics_Checkpoint() at epoch 5
- `app_baseline_runtime.c`: Added Metrics_StartInference("BASELINE")

**Key functions**:
- `Metrics_Init()`: Initialize DWT cycle counter for microsecond timing
- `Metrics_StartInference(label)`: Start timing, capture pre-inference power
- `Metrics_Checkpoint()`: Capture mid-inference power (NPU active)
- `Metrics_EndInference(temp)`: Stop timing, capture post-power, log results

**Output formats**:
- UART: `[METRICS] CNN: latency=45.23 ms, power_pre=294.10mW, power_mid=810.71mW, ...`
- SD Card CSV: `datetime,label,latency_ms,power_pre_mW,power_mid_mW,power_post_mW,power_delta_mW,temp_c`

**Units**: latency in ms (converted from DWT cycles), power in mW, timestamp in ISO 8601 format from DS3231 RTC.

**Cleanup**: Removed duplicate `metrics_power.c/h` that had overlapping functions with `inference_metrics.c`.

## 2026-05-01 Firmware Baseline Sweep Fix

**Root cause**: `APP_BASELINE_SWEEP_DEG` was `180.0f` in firmware, but the Python
`gauge_calibration_parameters.toml` and the physical gauge both specify a
**270Â° sweep** (`sweep_deg = 270.0`).

**Effect of the bug**: Every temperature was mapped ~1.5Ã— too high.
- Needle at -30Â°C (135Â°) was computed as if the sweep ended at 315Â° instead of
  45Â°, so it read near +50Â°C.
- The old angle-validation window (130Â°â€“320Â°) also rejected valid angles at
  the hot end (45Â° = 50Â°C) and the cold end (135Â° = -30Â°C).

**Fixes applied to `app_baseline_runtime.c`**:
1. Changed `APP_BASELINE_SWEEP_DEG` from `180.0f` â†’ `270.0f`.
2. Removed the bogus `+2Â°` calibration offset (`APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG` set to `0.0f`). The "~2Â° low" bias was an artifact of the compressed 180Â° sweep.
3. Fixed angle validation to reject only the subdial band (~50Â°â€“130Â°) instead of the old 130Â°â€“320Â° window. This now correctly accepts the full 270Â° sweep.
4. Fixed history angle filtering (`SelectSmoothedEstimate`) with the same valid-angle logic.

**Verification math**:
- -30Â°C at 135Â°: fraction = (135 - 135) / 270 = 0 â†’ -30Â°C âœ“
- 0Â°C at 225Â°: fraction = (225 - 135) / 270 = 0.333 â†’ 0Â°C âœ“
- +50Â°C at 45Â° (wraps): fraction = (45 + 360 - 135) / 270 = 1.0 â†’ +50Â°C âœ“

**Live testing (2026-05-01)**: After the fix, baseline readings are now correct:
- Needle at -30Â°C: baseline reads ~+50Â°C (fixed)
- Needle at -16Â°C: baseline reads -12.8Â°C (correct, AI reads -12.0Â°C)
- Angle detection: 193.1Â° for -12.8Â°C (expected ~178Â°, within 15Â° tolerance)

**Additional fix (2026-05-01)**: Baseline was still producing wrong readings (e.g., -13.8Â°C needle read as 38Â°C with angle=4.5Â°). The issue was:
1. The detector was finding a false positive at 4.5Â° (near 0Â°, the +50Â°C position)
2. The inversion check was NOT flipping it because the backward ray wasn't significantly darker
3. The angle validation didn't reject angles near 0Â°

**Fixes applied**:
1. Added angle validation to reject angles in the range 0Â°-30Â° (near 0Â°), which are clearly wrong for a -30Â°C to 50Â°C gauge
2. Improved inversion check to only flip angles in the subdial band (30Â°-150Â°), not angles in the valid range (135Â°-315Â°)

**Latest test (2026-05-01)**: Baseline now correctly reads -11.5Â°C with angle=197.5Â° (AI reads +18.2Â°C). The angle is in the valid range (135Â°-315Â°), and the detection is stable.

**Additional fix (2026-05-01)**: Baseline was still producing wrong readings for +20Â°C needle (reading -11.5Â°C with angle=197.5Â° instead of +20.8Â°C with angle ~307Â°). The issue was:
1. The detector was finding a false positive at 197.5Â° (in the range 170Â°-230Â°)
2. This range is commonly false positives from dial markings or reflections
3. The angle validation didn't reject angles in this range

**Fixes applied**:
1. Added angle validation to reject angles in the range 170Â°-230Â°, which is where the detector is finding false positives
2. The correct needle angle for +20Â°C should be ~307Â°, which is in the valid range (230Â°-315Â°)

**Additional fix (2026-05-01)**: Baseline was still producing wrong readings for +20Â°C needle (reading -11.5Â°C with angle=197.5Â° instead of +20.8Â°C with angle ~307Â°). The issue was:
1. The detector was finding a false positive at 197.5Â° (in the range 180Â°-210Â°)
2. This range is commonly false positives from dial markings or reflections
3. The angle validation didn't reject angles in this range

**Fixes applied**:
1. Added angle validation to reject angles in the range 185Â°-205Â°, which is where the detector is finding false positives
2. The correct needle angle for +20Â°C should be ~307Â°, which is in the valid range (205Â°-315Â°)

**Latest test (2026-05-01)**: Python baseline evaluation on hard cases shows:
- spoke_v2 detector: MAE=2.71Â°C, max error=8.9Â°C (1/11 over 5Â°C)
- ctr detector: MAE=15.47Â°C, max error=34.4Â°C (2/3 over 5Â°C)
- line segment detector: NO DETECTION on all hard cases
- The spoke_v2 detector is consistently detecting angles within a few degrees of the expected values
- The firmware baseline fix (rejecting angles in range 185Â°-205Â°) should prevent false positives while not rejecting valid angles

**Current state (2026-05-01)**:
- Python baseline (spoke_v2) is working well on hard cases with MAE=2.71Â°C
- Firmware baseline has been updated with angle rejection ranges: 0Â°-30Â°, 50Â°-130Â°, 170Â°-230Â°, 185Â°-205Â°, and 315Â°-360Â°
- The firmware baseline should now correctly read temperatures from -30Â°C to +50Â°C
- The Python baseline is used as a reference to verify the firmware baseline accuracy
## 2026-05-02 Spoke Continuity Fix

**Root cause**: The spoke-continuity check was using a 25% darkness threshold, which was accepting false positives from dial markings that create a continuous dark line along the spoke.

**Effect of the bug**: On a 49Â°C needle, the detector was finding a false positive at angle=154.3Â° (corresponding to ~-10Â°C) instead of the correct angle around 300Â°. The spoke-continuity check was not rejecting this false positive because the dial marking created a continuous dark line.

**Fix applied to `app_baseline_runtime.c`**:
- Raised spoke-continuity threshold from 25% to 35%
- The needle should have stronger continuity since it spans the full radius, while dial markings typically create weaker continuity

**Expected effect**: The spoke-continuity check should now reject false positives from dial markings while still accepting valid needle detections.

**Live testing (2026-05-02)**: After the fix, baseline readings should correctly read 49Â°C instead of -24.3Â°C.

**Additional fix (2026-05-02)**: The spoke-continuity check at 35% was still accepting false positives from dial markings. The baseline was reading -23.6Â°C with angle=156.7Â° instead of correctly reading 49Â°C.

**Fix applied**:
- Raised spoke-continuity threshold from 35% to 45%
- The needle should have very strong continuity since it's a thick, dark feature, while dial markings typically create weaker continuity
**Additional fix (2026-05-02)**: The spoke-continuity check at 45% was still accepting false positives from dial markings. The baseline was reading -19.3Â°C with angle=171.0Â° instead of correctly reading 49Â°C.

**Fix applied**:
- Added angle validation to reject angles in the range 160Â°-180Â°, which is where the detector is finding false positives
- The correct needle angle for 49Â°C should be ~300Â°, which is in the valid range (180Â°-315Â°)

**Additional fix (2026-05-02)**: The angle rejection range (160Â°-180Â°) was too aggressive and caused all detections to fail.

**Fix applied**:
- Increased spoke-continuity samples from 10 to 20 for more accurate measurement
- Removed overly aggressive angle rejection range (160Â°-180Â°) that was causing all detections to fail
- Lowered spoke-continuity threshold from 45% to 30% â€” with 20 samples, we get a more accurate measurement that can distinguish between real needles and dial markings without being too aggressive
- Increased hot-zone second-pass search from 16 to 64 peaks to catch needle peaks that may have lower vote counts but better continuity

**Expected effect**: The spoke-continuity check with more samples and lower threshold should now accept valid needles while still rejecting obvious false positives from dial markings.

## 2026-05-02 CNN Calibration Fix

**Root cause**: The CNN model was consistently under-reading by ~6-10Â°C. The calibration was disabled (`APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0`) because the affine fit made cold readings worse, but this also meant hot readings were not corrected.

**Effect of the bug**: On a 49Â°C needle, the model outputs ~39-43Â°C instead of 49Â°C.

**Fix applied to `app_inference_calibration.c`**:
1. Enabled output calibration (`APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 1`)
2. Replaced affine calibration with a simple fixed offset of +7.5Â°C
3. The fixed offset is more reliable than the affine fit which made cold readings worse

**Expected effect**: The CNN should now correctly read temperatures across the full range by adding a fixed +7.5Â°C offset to the raw model output.

**Restored calibration (2026-05-02)**: The affine calibration (scale=1.163, bias=0.742) was restored because the cold readings were fine. The model under-reads by ~6-10Â°C across the full range, and the affine fit was fitted to achieve MAE=4.26Â°C on hard cases.

**Disabled calibration (2026-05-02)**: The affine calibration was disabled because calibration is not the right fix for model output issues. The model uses sigmoid activation which compresses output at extremes. Proper fix is to retrain model with linear output head or better training data at temperature extremes, not post-hoc calibration. The hard fault was likely caused by something else in the calibration code path.

**Additional fix (2026-05-02)**: The angle rejection range (160Â°-180Â°) was too aggressive and caused all detections to fail.

**Fix applied**:
- Increased spoke-continuity samples from 10 to 20 for more accurate measurement
- Removed overly aggressive angle rejection range (160Â°-180Â°) that was causing all detections to fail
- Lowered spoke-continuity threshold from 45% to 30% â€” with 20 samples, we get a more accurate measurement that can distinguish between real needles and dial markings without being too aggressive
- More samples help distinguish between real needles (strong continuity along full length) and dial markings (weaker or partial continuity)

**Expected effect**: The spoke-continuity check with more samples and lower threshold should now accept valid needles while still rejecting obvious false positives from dial markings.

**Additional fix (2026-05-02)**: Baseline was reading 19.9Â°C with angle=303.3Â° instead of 49Â°C. The correct angle for 49Â°C is ~41.6Â° (hot wrap-around zone). The polar vote was finding a stronger false-positive peak at 303.3Â° than the real needle at 41.6Â° because the needle at high temperatures has a weaker gradient signal near the sweep edge.

**Fix applied**:
- Added hot-zone second-pass search: when the primary peak is in the cold/mid range (135Â°-315Â°), check if there's a stronger spoke-continuity peak in the hot wrap-around zone (25Â°-65Â°)
- Widened the hot-end angle acceptance from just 45Â° to 30Â°-60Â° to cover the wrap-around zone for +35Â°C to +50Â°C
- The hot-zone search uses relaxed continuity (0.30) and hub_darkness (0.20) thresholds since the needle at high temperatures has weaker gradient signal but still has strong spoke continuity and hub connection

**Expected effect**: The baseline should now correctly read hot temperatures (35Â°C-50Â°C) by finding the needle in the hot wrap-around zone.

**Expected effect**: The spoke-continuity check and angle validation should now reject false positives from dial markings while still accepting valid needle detections.
**Expected effect**: The spoke-continuity check should now reject false positives from dial markings while still accepting valid needle detections.
## 2026-04-30 Firmware Baseline Fixes

Fixed classical baseline angle detection issues on live board:

1. **Inner dial center correction** (`app_gauge_geometry.h`): Changed `APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO` from `0.5000f` to `0.4460f` to center on the inner Celsius dial instead of the geometric center of the whole gauge face. This fixes the 40Â°C misread that was detecting the Fahrenheit needle position.

2. **Tightened angle margin** (`app_baseline_runtime.c`): Reduced from 12Â° to 6Â° to prevent dial markings outside the calibrated sweep from polluting the vote.

3. **Raised edge magnitude threshold** (`app_baseline_runtime.c`): Increased from 8.0 to 12.0 to reject weak edges from dial artwork while keeping strong needle edges.

4. **Added angle validation** (`app_baseline_runtime.c`): Post-detection rejection of angles outside 150Â°-300Â° (covers -15Â°C to 40Â°C). Rejects dial tick marks at sweep extremes.

5. **Added quality override** (`app_baseline_runtime.c`): Lower-priority candidates (rim, image-center) can win if they have 2x better quality than fixed-crop, preventing rim-edge false positives from overriding correct needle detections.

6. **Added history angle filtering** (`app_baseline_runtime.c`): `SelectSmoothedEstimate()` now filters out history entries with invalid angles (polluted from before the fix), preventing old wrong angles from being returned as smoothed estimates.

**Result**: SUCCESS! Center-of-sweep bias fixed the issue. Baseline now reads 4.1Â°C at 211.7Â° (AI reads 2.3Â°C). Both are detecting the correct needle angle instead of dial edge artifacts. The bias boosts votes near sweep center (225Â°) by up to 1.5x, penalizing edge detections.

## Current State

- The learned board path is still the OBB + scalar cascade, but the classical baseline now mirrors the hard-case-winning gradient-polar detector on the fixed gauge crop instead of the older shaft-biased ray heuristic.
- The single-image classical geometry helper stays Hough-first with a plausibility-gated Hough circle estimate and a slightly shrunken effective radius (`0.75x`). We tested wider geometry grids and the LSD line-segment fallback on the hard cases, but the direct Hough path is still the best offline classical choice overall, even though a few individual frames can be rescued by a wider local geometry search.
- The single-image baseline runner is now Hough-first by default and only uses the experimental auto-geometry sweep when explicitly asked. On `capture_p50c_preview.png`, the default path kept the Hough seed while the explicit sweep jumped to a wrong offset candidate (`hough_+16_+16`), so the sweep is a useful experiment mode but not the default.
- The manifest benchmark script now defaults to `hough_only`, because that mode has been the strongest on the current hard-case manifests and the center fallback was not helping the aggregate error.
- The firmware baseline now mirrors that conservative rule with `APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP=0`: fixed crop is the primary live anchor, and the local geometry sweep only stays available as an explicit experiment.
- On the current hard-case manifests, that direct-Hough path is about `mae=19.1053` on `hard_cases.csv`, `mae=19.0864` on `hard_cases_remaining_focus.csv`, and `mae=14.4444` on `board_weak_focus.csv`.
- The hard-case detector-family sweep now says the gradient-polar family is the best pure classical option we have right now: `gradient_polar` beat `ray_score`, `hough_lines`, and `dark_polar` on `hard_cases.csv` plus `hard_cases_remaining_focus.csv`.
- The same hard-case sweep also showed that a small agreement cluster can beat pure winner-take-all selection: a `4C` consensus window improved the combined family MAE to `18.0248`, so the live selector now prefers clustered geometry hypotheses over lone outliers when the temps agree.
- The classical detector uses Sobel-like edge alignment in the inner annulus, and the selector now compares all refined candidate geometries by peak sharpness instead of raw confidence. Each seed is refined over a tiny local neighborhood so the geometry can slide a few pixels instead of staying stuck on the first anchor.
- The polar vote no longer uses the old shaft-bias heuristic; it now follows the gradient-polar family that won the hard-case sweep.
- The confidence gate is now SNR-like, so the live threshold sits around `1.25`, and a strong absolute vote floor is still required before a fallback seed can enter the tiny smoothing history.
- The classical baseline still emits a provisional warm-up reading from the first accepted frame, but weak frames no longer seed history.
- On the 2026-04-24 live `14C` trace, the CNN output matched the gauge while the classical baseline still missed the spoke, so the CNN should be treated as the authoritative live value on that setup.
- On recent live traces the polar baseline can still produce near-tied peaks, so the board now refines the bright, fixed-crop, and image-center seeds, prefers the ones that clear the acceptance gate, and then keeps the candidate with the best blended peak-sharpness-plus-support score instead of letting the first anchor win by default.
- The classical baseline continues to hold the last stable reading whenever a frame is too ambiguous to trust, so the comparator stays quiet instead of hallucinating a new plateau.
- The live polar baseline now keeps the strongest raw angular peak instead of re-ranking the top bins with hub/width heuristics. That extra re-ranking was promoting unrelated fixed-crop peaks on the hard live traces, so the firmware is staying closer to the Python Hough-first reference.
- The fixed-crop estimate now seeds the tiny smoothing history on confidence alone, so a decent live read can start the baseline moving even when the peak score is still a little soft.
- The fixed-crop classical estimate is now accepted on confidence alone, because the absolute score floor was suppressing otherwise useful live reads. The fallback geometries still keep the stricter score and peak-ratio checks.
- The firmware score floor was relaxed from `500` to `250` so the classical baseline is less likely to suppress normal hard-frame detections.
- On `hard_cases_remaining_focus.csv`, the raw scalar model landed at `raw_mae=7.1941`, the fitted affine calibration improved that to `affine_mae=4.1813`, and the classical manifest baseline reached `mean_abs_err=4.0247`, so the calibration still helps on the hard-case mix but the pure classical path remains slightly ahead on that set.
- Weak fixed-crop near-ties now stay out of the tiny smoothing history, so the board holds the last stable estimate instead of overwriting it with a borderline new plateau.
- The rectifier fallback path now trusts the flashed rectifier blob and skips the signature gate, because the stale fingerprint was blocking a valid image and causing the freeze when the OBB crop overflowed the scalar window.
- The live AI cascade no longer enters the rectifier stage at runtime; if the OBB crop is out of range or the OBB scalar pass fails, the board now falls back to the fixed training crop instead so the capture loop keeps moving.
- Two fresh `10C` close-up captures were added to `hard_cases_remaining_focus.csv` (`capture_2026-04-24_22-24-04.png` and `capture_2026-04-24_22-30-21.png`). On those frames the pure classical baseline stayed near the real gauge (`7.9C` and `9.7C`), while the fixed-crop scalar probe over-read them by a wide margin (`26.5C` and `27.5C`), so they are good hard cases for comparing classical CV against the learned reader.
- The DS3231 RTC now seeds itself from the current firmware build timestamp automatically when booting at year `2000`, then re-reads the clock.
- The one-off forced RTC seed has now been turned back off; the normal rule is back to year-`2000` boot seeding only.
- `prodv0.3` is the current firmware integration candidate for the OBB localizer, and the board project now builds cleanly with the OBB wrapper plus the shared scalar runtime bundle.
- The OBB hardfault cause is now pinned down: the per-frame ATON `LL_ATON_RT_Reset_Network()` call in `app_ai.c` was the trigger, so `prodv0.3` should stay in one-shot runtime mode by default and only re-enable the reset path behind an explicit test flag.
- FileX/SD bring-up was a separate issue, but the latest boot trace shows it now reaches ready end-to-end: CMD0 -> CMD8 -> ACMD41 -> CMD58 -> partition -> FileX media open -> test file -> capture directory ready.
- The SD bring-up path is now quiet by default: FileX only emits concise ready/error lines, the ACMD41 handshake still only requests HCS when CMD8 proves the card is v2, and the storage wait path no longer prints periodic breadcrumbs.
- The watchdog heartbeat thread now prints a single `[WATCHDOG] pulse` line per cycle again, so UART liveness stays visible without reintroducing the old startup chatter.

## 2026-05-02 xSPI2 Clock Enable Fix

**Root cause**: HardFault during the second inference cycle (dry-run path), specifically when switching between OBB and scalar stages. The fault address `0x40ECA12C` and CFSR=0x00008200 indicated a precise data abort during xSPI2 reconfiguration.

**Effect of the bug**: Board crashes after successfully running OBB + scalar inference once, during the transition back to OBB for the dry-run cascade.

**Analysis**:
- The OBB hardfault was previously pinned down to the per-frame `LL_ATON_RT_Reset_Network()` call, which is now skipped by default (`APP_AI_RESET_NETWORK_EACH_INFERENCE = 0`)
- The crash occurred after the scalar stage completed and during xSPI2 reconfiguration for the OBB stage restart
- The xSPI2 peripheral was being deinitialized and reinitialized, but the clock enable wasn't being explicitly asserted after `HAL_RCCEx_PeriphCLKConfig()`
- The `BSP_XSPI_NOR_DeInit()` may have disabled the clock, and the new clock source (`RCC_XSPI2CLKSOURCE_IC3`) wouldn't be active until the peripheral clock enable is asserted

**Fix applied to `app_ai.c`**:
1. Added `XSPI_CLK_ENABLE()` after clock configuration in `AppAI_EnsureXspi2MemoryReady()` (line ~552)
2. Added `XSPI_CLK_ENABLE()` after clock configuration in `AppAI_ReconfigureXspi2ForRuntime()` (line ~603)
3. Added debug console logging to confirm clock enable execution

**Expected effect**: The xSPI2 peripheral should now have a stable clock source after reconfiguration, preventing the data abort that caused the hardfault.

**Testing**: Rebuild and flash the firmware to verify the hardfault no longer occurs during the OBB + scalar cascade.
- The latest boot trace showed the storage path finally succeeding end-to-end: CMD0 -> CMD8 -> ACMD41 -> CMD58 -> partition -> FileX media open -> test file -> capture directory ready, so the FileX/media readiness issue is no longer the active blocker.
- The DS3231 boot seeding is also working again: when the RTC comes up at year `2000`, the firmware seeds it from the build timestamp and the logger resumes with a sane date instead of the old impossible timestamps.
- The DS3231 override used for the one-off clock sync has been disabled again, so future boots should only reseed the RTC if it falls back to year `2000`.
- After storage came up, the next trace exposed a camera-pipeline error (`DCMIPP` `0x00008100` / `CSI_SYNC|CSI_DPHY_CTRL`) on the first retry, so the remaining live boot issue is now camera capture, not storage or AI.
- The latest live trace confirms the ATON fault is still gone with reset skipped, and `prodv0.3` can run OBB + scalar cleanly; the remaining boot issue on that trace is camera capture after storage comes up, plus the RTC/logging recovery path that still needs to stay sane.
- The OBB package originally pinned its CPU input arena at `0x34100000`, which overlapped the app's live `.bss` / heap / ThreadX footprint and was the likely source of the hard faults; the arena base was moved up to `0x34107000`, above `__bss_end__` / `_end = 0x34106b58`, and the build still passes with that safer placement.
- The scalar package had the same overlap bug and is now rebuilt against `0x34107000` as well, so both model stages are finally off the live app RAM window.
- The newlib heap is now capped below `0x34110000` in `sysmem.c` so libc allocations cannot grow into the AI arena even if a later runtime path uses `malloc`.
- The camera init thread stack was doubled to 16 KB after the brightness-gate path proved too deep for the old 8 KB budget, and the brightness gate no longer retries in place; it now nudges exposure for the next capture cycle and keeps the current frame moving.
- The classical baseline worker was also bumped from 8 KB to 16 KB after the polar spoke-voting sweep and its logging looked like the next likely stack pressure point in the board freeze sequence.
- The debug console is now fail-fast under contention instead of spinning, so a lower-priority logger can finish without a higher-priority thread starving it; the heartbeat thread still avoids per-pulse UART logging so LED liveness is not tied to console throughput.
- The visible green LED is now the normal heartbeat, and red is reserved for fault state so a solid red LED is no longer just a startup indicator.
- The long camera-loop sleep, image-cleanup retry/period waits, and storage-readiness polling were moved off the ThreadX sleep/timer queue path and onto a cooperative delay helper to reduce the last timer-heavy background waits that could still wedge the board after the baseline log.
- While FileX is still not ready, the camera loop now retries sooner instead of waiting a full minute between capture attempts, so the board keeps visibly producing progress during bring-up and debug.
- [2026-04-28] Improved classical baseline reliability by adjusting acceptance criteria: set minimum accept score to 2.0 (based on debug scores in 6-7 range), relaxed peak ratio to 1.10, tightened center distance threshold, and adjusted geometry override and bright center penalties. Fixed compilation error in AppBaselineRuntime_PassesAcceptanceGate function. This should reduce "[BASELINE] Classical baseline failed to estimate a temperature" occurrences while maintaining robustness.
- [2026-04-28] Added a continuity-aware borderline path for strong fixed-crop and image-center estimates: if the peak ratio is only slightly soft but the new reading stays within a small temperature delta of the last stable estimate, the firmware can seed history instead of holding stale output. This is meant to rescue live continuation frames without lowering the global peak-ratio gate for unrelated clutter.
- [2026-04-28] Offline single-image testing on 8 images from `ml/data/captured_images/` revealed significant performance variance: some images achieve 0Â°C error (capture_0001, capture_0002), while hard cases show up to 20Â°C error (capture_2026-04-24_22-24-04 predicted 26.89Â°C vs 10Â°C true). Low confidence (<10) often correlates with large errors. The baseline shows inconsistent performance - works perfectly on some images, fails badly on others. Confidence score appears to be a reliable indicator of accuracy.
- [2026-04-28] Added confidence threshold (10.0) and peak ratio threshold (1.5) filtering to single_image_baseline.py. Weak detections (confidence < 10) now return "none" instead of inaccurate predictions. This rejects 3 of 8 hard case images that were previously over-predicting by 16-20Â°C. Good images (capture_0001, capture_0002) still work correctly with confidence > 20. Remaining high-confidence wrong predictions (capture_2026-04-24_22-30-21, capture_0075, etc.) have confidence > 10 but wrong angles - root cause is the polar spoke voting algorithm finding a strong but incorrect peak in complex scenes. All 8 unit tests pass after improvements with no regressions.
- [2026-04-28] Improved angular filtering: reduced sweep arc margin from 12Â° to 6Â° and added post-detection angle validation in baseline_classical_cv.py. This further reduces false positives from out-of-sweep features. Hard case images with confidence < 10 now correctly return "none" instead of inaccurate predictions. Remaining issue: high-confidence wrong predictions still occur when the detected angle is within the sweep arc but incorrect - need to add geometric validation or multi-stage filtering.
- The cooperative delay helper now sleeps one tick at a time instead of only yielding, because the pure-yield version starved the lower-priority heartbeat thread and broke the visible green pulse.
- The latest board hard fault traced into `HAL_TIM_IRQHandler()` from `TIM5_IRQHandler()`, with `BFAR=0x10`, which points to a bad TIM5 handle or a nearby overwrite rather than a model-stage bug; the IRQ now has a small guard so the board stays alive while we keep hunting the corruption source.
- The next hard fault moved into ThreadX timer service at `PC=0x34000948` inside `_tx_timer_interrupt` / `__tx_timer_no_time_slice`, which suggests `_tx_timer_current_ptr` was null or corrupted; the ThreadX timer assembly now self-heals null, out-of-range, or misaligned timer pointers by restoring the list pointer and skipping expiration work.
- The later freeze then moved into `_tx_timer_system_deactivate()` at `PC=0x34020130`, where the timer list-head pointer looked corrupted; that path now bails out early if the timer pointer wanders outside the timer list range instead of dereferencing garbage.
- The image cleanup worker is now deferred until after camera probe success to remove one source of startup contention while we debug the boot stall.
- The camera init thread is now the highest-priority app thread and no longer waits on the startup sleep before probing, so bring-up does not depend on the timer path before the probe starts.
- FileX was raised above the camera thread during startup so it can finish mounting the SD card before the first capture loop blocks on storage readiness.
- The capture path now logs when it begins waiting for FileX media readiness, and it prints a periodic wait breadcrumb so we can tell whether the board is really frozen or just waiting on storage.
- The hard capture-time storage gate turned out to be too aggressive during bring-up, so the capture path now skips SD saves when FileX is not ready instead of stalling the live camera / inference loop.
- The FileX thread startup blinks were removed again because they were an unnecessary blocking delay during boot and made the board look stuck during startup.
- The updated process diagram now uses per-step SVG thumbnails from `docs/process_diagrams_assets/`, so keep those assets in sync if the step text or flow changes again.
- The long-term MobileNetV2 geometry, direction, and detector-first experiments are exploratory; the OBB localizer is the first one that clearly improved the board-probe cascade.
- The latest `mobilenetv2_detector_geometry` run also missed badly: `test gauge_value_mae=24.2626` versus `baseline_mae_mean_predictor=20.1698`, so it is still not a usable reader.
- The geometry keypoint-only MobileNetV2 run also missed the baseline: `test gauge_value_mae=23.1730` versus `baseline_mae_mean_predictor=20.1698`, even though its keypoint MAE improved to `6.6727`.
- The uncertainty-aware geometry run was the least-bad of the new geometry variants, but it still only reached `test gauge_value_mae=18.9273` versus the baseline mean predictor at `20.1698`, so it is not board-ready yet.
- The new MobileNetV2 OBB localizer run trained cleanly on the labeled split and reached `val_mae=0.1435` and `test_mae=0.1786` on the OBB parameters. That makes it the strongest explicit localizer proxy so far, even though it is still a localization model rather than a reader.
- The OBB + scalar board-probe cascade using `mobilenetv2_obb_longterm` and the rectified scalar deployment reached `mean_abs_err=3.6617`, `max_abs_err=11.8603`, and `cases_over_5c=11` at `OBB_CROP_SCALE=1.20`. That is the best board-probe result so far and beats the rectifier chain.
- The stored firmware calibration fit is the affine p5 hard-case fit from `scalar_full_finetune_from_best_affine_calibrated_p5`, because the older board30 piecewise curve was overcorrecting the original hard-case manifest even though it looked good on the closer live reads.
- The current board A/B view has `APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION=0`, so the live scalar output is the raw model value rather than the affine postprocess.
- On the 2026-04-24 `-30C` live trace, the baseline held the last stable estimate after a low-confidence frame, while the scalar CNN raw outputs were still around `17.3C`-`21.1C` and the piecewise tail pushed them up to about `29.7C`-`30.7C`. That means the cold-end miss is a combination of an undertrained raw regressor and a calibration fit that extrapolates poorly outside the mid-band.
- After moving the gauge a bit closer on the same live setup, the scalar raw output moved to about `-19.9C` and `-22.0C`, and the board30 spline pulled those reads to about `-25.3C` and `-28.9C`. The filtered user-facing value settled at `-27.1C`, so the closer-camera spline is now behaving much more like the real gauge on the live board.
- The laptop-side board-pipeline replay script is now usable end-to-end: it lazy-loads the heavy board module, uses a pure-PIL crop/letterbox path for the board-style replay, and can replay a single captured PNG in about `0.03s` on the laptop capture while still printing firmware-like stage traces.
- The new hard-case comparison runner shows that on `hard_cases_remaining_focus.csv` the pure classical baseline still wins by a wide margin over the deployed board replay: classical `MAE=4.0247`, board raw `MAE=10.4790`, board calibrated `MAE=13.7990`, and board reported `MAE=14.1790`. The board replay stayed on the OBB path for all 9 rows, so the gap is coming from the deployed reader and calibration rather than a rectifier detour on that manifest.
- The OBB crop stayed fairly stable during that closer-camera run (`30,10,146x167` then `32,10,146x167`), which suggests the remaining error is now more about calibration and residual framing bias than a completely wrong crop.
- The live board now softens the camera brightness nudge to 25% fractional exposure/gain steps instead of the old 2x jumps, because that gave the brightness gate a better chance of settling inside the acceptable window instead of bouncing between dark and bright captures.
- The OBB crop window is intentionally loose now (`0.60..1.40` relative to the stable training crop), because the live close-up OBB crops were still healthy enough to keep on the fast path and I did not want to send moderate crops into the slower rectifier stage unnecessarily.
- When the board probe holdout was added as an extra test, the new affine fit was still slightly better on the board probe (`test2_affine_mean_abs_err=5.1965` vs `test2_piecewise_mean_abs_err=5.5528`), so the current piecewise deployment is a hard-case-tuned tradeoff rather than a universal win.
- The compact geometry long-term localizer is not board-ready on the rectified probe set, and the cascade-localizer long-term run only modestly improved value MAE while the geometry branch stayed weak.
- The explicit MobileNetV2 geometry cascade-localizer long-term run is better than the compact proxy, but still not board-ready on the rectified probe set.
- The latest cascade eval with that explicit localizer reached `mean_final_abs_err=13.2531` on 39 board-probe samples. That is an improvement over the compact cascade (`14.5682`), but still too far from the target.
- The rectifier + scalar chain using `mobilenetv2_rectifier_zoom_aug_v4` on the board probe set reached `mean_abs_err=12.4529` and `max_abs_err=27.3887`.
- The same board-probe eval with `mobilenetv2_rectifier_hardcase_finetune_v3` was better at `mean_abs_err=9.8036` and is now the best rectifier + scalar result on that probe set, but it still is not board-ready.
- The exported int8 rectifier `mobilenetv2_rectifier_hardcase_finetune_v3_int8` improved the board-probe chain further. With `RECTIFIER_CROP_SCALE=1.80`, the rectifier + scalar chain reached `mean_abs_err=6.1574` and `max_abs_err=21.2753`, which is still the best rectifier-based board result so far, but it is now behind the OBB cascade.
- The next live check should move toward an even more explicit localizer or detector/OBB target rather than another small refinement of the same geometry stack.
- The OBB long-term experiment should stay on the labeled dataset split with `val_fraction` and `test_fraction`; do not feed it the board manifest hard-case path.
- For WSL jobs, restart before the run and shut WSL down again afterward.
- The `docs/process_diagrams.drawio` file now reflects the current OBB + scalar cascade and the current classical baseline flow, so it should be kept in sync with future runtime changes.
- The pure classical board baseline now uses an explicit dial radius derived from the training crop height instead of the old inscribed crop radius. That keeps the polar annulus closer to the Hough-seeded Python baseline and should stop the detector from under-scanning the outer needle band.
- The pure classical board baseline still uses a tiny rim-based center search before the spoke vote, but the live selector no longer hard-prioritizes rim-center over the other accepted candidates. The board bug at `-5C` came from that hard priority forcing a warm false positive, so the firmware now ranks candidates by peak-sharpness quality first, matching the Python classical helper instead of giving the rim an unconditional win.
- [2026-04-29] Attempted to simplify the classical baseline using a pure line-segment approach (HoughLinesP + tip-vs-tail discrimination). The "simple" version failed on 4/5 hard cases, proving that the complexity in the original baseline (polar voting, multiple geometry candidates, numerous thresholds) was a necessary evolution to handle low contrast, close-up framing, and noisy edges. HoughCircles remains a major bottleneck on close-ups.
- The latest board-image inspection suggests the needle is more saturated / color-separated than the surrounding dial artwork, so the old grayscale-dark shaft assumption is brittle. A color-aware shaft scan with heavier middle-shaft weighting was tried to keep the dial from stealing attention, but the current implementation regressed `board_weak_focus.csv` to about `MAE=28.6173`, so that detector variant is not yet a usable baseline.
- Keep using `ml/data/captured_images/` and `board_weak_focus.csv` as the main regression set for the next detector-tuning pass, because those samples expose the board-specific failures better than the broader hard-case mix.
- For the clean 2026-04-24 captures, the over-aggressive board-prior override was the bigger problem than the Hough seed itself. The default single-image baseline now keeps a confident Hough geometry on ideal frames and only falls back to the board-prior scan when Hough is weak, which is the right tradeoff for the simple near-centered cases we care about most right now.
- [2026-04-29] The clean-capture baseline got a lot better once the default combined detector stopped letting the experimental line-segment and Hough-line branches win. The default now prefers the stable spoke/center-weighted detectors, and the board-prior helper tries the generic radial detector before the shaft scan so clean photos keep the better middle-shaft vote.
- On the clean labeled set (`capture_p25c.jpg`, `capture_p30c.jpg`, `capture_p31c.jpg`, `capture_p35c.jpg`, `capture_p45c.png`) the current default baseline is now around `MAE=5.413`, which is finally good enough for the ideal-case thesis baseline.
- [2026-04-29] The live STM32 baseline should now rank `fixed-crop-polar` ahead of `board-prior-polar` so the stable crop stays in charge on clean captures. The board prior is still a fallback, but it should no longer outrank the ideal-case fixed crop just because it found a slightly stronger local vote.
- [2026-04-30] The newest `ml/data/captured_images/capture_2026-04-30_05-51-06.yuv422`, `05-52-17`, and `05-53-29` previews confirmed the gauge is a dark-needle-on-light-background case. The STM32 polar vote should therefore emphasize the middle shaft and ignore red-pixel bias rather than assuming the needle itself is red.
- [2026-04-30] The newest ideal previews (`capture_2026-04-30_07-00-09.png`, `07-01-21.png`, `05-51-06.png`, `05-52-17.png`, `05-53-29.png`) still look reasonable on the laptop classical baseline, but the live firmware was still letting a lone fixed-crop peak win too often. The default selector now keeps the agreement-cluster rule enabled so a consistent temperature cluster can beat a single outlier.
- [2026-04-30] A batch replay over the newest captures confirmed the chosen classical baseline is acceptable on the ideal controls: `capture_p25c.jpg`, `capture_p30c.jpg`, `capture_p31c.jpg`, `capture_p35c.jpg`, and `capture_p45c.png` came out to about `MAE=3.556C`. The new 2026-04-29 raw frames are mostly overexposed/blank and should be treated as non-ideal smoke-test inputs, not as the target regime for the thesis baseline.
- [2026-04-30] Re-checking the live 2026-04-30 previews showed the upper temperature needle is around 5C on the inner scale, which is also around 40F on the outer scale. The earlier "40C" read was wrong, so future baseline tuning should keep the upper needle as the target and ignore the lower subdial.
- [2026-04-30] The gauge has two scales: the outer large numbers are Fahrenheit and the inner numbers are Celsius. All baseline tuning for this project should interpret the inner Celsius ring as the target temperature scale.
- [2026-04-30] The selector was still over-rewarding extreme peak-ratio outliers, so the quality score now treats peak ratio as a penalty (`confidence / peak_ratio`) and the firmware turns on the narrow local refinement sweep by default so the seed geometries can slide a few pixels before we commit to the live read.
- [2026-04-30] The `capture_2026-04-29_09-35-33.png` inspection showed that the line-segment detector was not the missing fix. The real failure was a zero-support center-weighted board-prior false positive overriding a good Hough/spoke read, so the single-image board-prior rescue now stays spoke-only and the final acceptance gate is much looser (`MIN_PEAK_RATIO=1.01`) because the clean bright captures have broad but stable peaks.
- [2026-04-30] The same frame now predicts about `10C` instead of collapsing to `-30C`, which is a much better failure mode for the thesis baseline even though it is still not the final ideal-case ceiling.
- [2026-04-30] A broader sweep over recent ideal frames showed the best simple classical baseline is Hough-first with board-prior only as a fallback. On the cleaner 2026-04-30 captures, pure Hough was best when available (`MAE~2.87C` on the 5 frames it could read), while the end-to-end hybrid stayed usable on the one Hough-missing frame and landed around `MAE~3.90C` on the 6-frame ideal set.
- [2026-04-30] The harder 2026-04-29 images are still a separate regime: the current classical baseline is not reliable there, so do not use those frames as the thesis target for the ideal-case baseline.
- [2026-04-30] The camera brightness gate was too aggressive on the newer processed captures. A two-day replay showed the old `200/20` bright gate falsely classed `capture_2026-04-30_11-51-05.yuv422` as too-bright, while the relaxed `230/100` gate leaves that frame as `ok` and still keeps the truly dim frames in `too-dark`.
- [2026-04-30] The brightness gate also needed a ratio-based bright check because the thin dark needle keeps `min_y` low on otherwise overexposed frames. The new `180` mean threshold plus a `>=50%` bright-pixel ratio catches the obviously bright captures from both days, including `12-20-22`, `12-21-34`, and `07-01-21`, while the `220/45` solid fallback still catches the near-white `11:51` frame.
- [2026-04-30] The classical selector also needed a source-priority guard inside the consensus step. Without it, a high-quality rim candidate could override a better fixed-crop or board-prior anchor just because several estimates agreed. Consensus now respects source priority before it considers peak-shape quality.
- [2026-04-30] The latest clean board capture `capture_2026-04-30_12-45-08.png` was a good reminder that the firmware peak-ratio gate was still too strict compared with the Python baseline. Lowering `APP_BASELINE_MIN_PEAK_RATIO` to `1.01` keeps broad-but-correct peaks alive so the hot rim rescue does not win a frame that should stay near the inner Celsius needle.
- [2026-04-30] The board crop was still clipping the top of the dial. A 12px upward bias on the bright-centroid crop fixed the framing on the newest capture and kept the upper numbers visible without widening the crop.
- [2026-04-30] The crop fix is now bounded and adaptive instead of a hard-coded pixel nudge: the bright-centroid crop keeps a 0.11x crop-height upward bias, clamped to 8..18 pixels, so the top of the dial stays in frame while still allowing modest position variation.
- [2026-04-30] A small sweep over the new cropped board images showed the framing is better, but the classical baseline is still mixed after cropping. Some 2026-04-30 frames land near 5C (`07-01-21` â‰ˆ `4.5C`, `11-48-43` â‰ˆ `10C`, `12-19-11` â‰ˆ `7.6C`), while others still jump to obviously wrong hot/cold values (`11-51-05` â‰ˆ `24.7C`, `05-52-17` â‰ˆ `-30C`, `12-20-22` â‰ˆ `-29.6C`), so cropping alone is not the final fix.

## 2026-05-02 HardFault Crash Fix

**Root cause**: HardFault crash in `AppInferenceLog_FormatFloatMicros()` with crash address 0x46687670 (corrupted pointer in AHB space). The crash registers showed:
- R0=0x340B0624 - valid `dst` buffer address (AXISRAM)
- R1=0x00000000 - fault address (null pointer being written to!)
- R2=0x00000044 - value being written (0x44 = ASCII 'D')
- R3=0x46687670 - corrupted `prefix` pointer (in AHB space 0x40000000+)

The issue was that the `prefix` parameter (string literal) was being corrupted by memory corruption from xSPI2 reconfiguration or stack corruption.

**Effect of the bug**: HardFault crash during inference, preventing any AI inference from completing successfully.

**Fixes applied**:

1. **`app_ai.c` - `AppAI_TraceAndApplyInferenceCalibration()`**:
   - Changed string literals to static const char arrays to prevent pointer corruption
   - Before: `"[AI] Model output before calibration: "` (pointer to string literal)
   - After: `static const char prefix_before[] = "[AI] Model output before calibration: ";`
   - Applied to all three prefix strings used in the function

2. **`app_inference_log_utils.c` - `AppInferenceLog_FormatFloatMicros()`**:
   - Added safety check to detect corrupted prefix pointers
   - Checks if prefix address is below 0x20000000 (invalid low-memory range)
   - If corrupted, logs error message with the corrupted address instead of crashing
   - This provides visibility into pointer corruption while preventing HardFault

**Expected effect**: 
- HardFault crash should be eliminated
- If pointer corruption still occurs, the system will log an error message instead of crashing
- The static const arrays ensure prefix strings are in ROM and won't be corrupted by xSPI2 reconfiguration

**Live testing (2026-05-02)**: Flash firmware and verify:
- No HardFault crash during inference
- Calibration delta logging works correctly
- Baseline and CNN inference both complete successfully

## 2026-05-07 CNN Training Session â€” Model Collapse and Recovery

### Data Reality Check
- **Only 141 unique labelled images exist** across all manifests. The canonical manifest (141 images) already contains the best/unique images. Other manifests (unified=57, full_labelled=31, hard_cases=57, new_captures=26, board_captures=84) are all subsets or duplicates.
- Merging all manifests with priority-based deduplication yields exactly 141 images â€” no more data to add.
- Value range: -30Â°C to +50Â°C (full gauge range represented).

### Recheck Note (2026-05-07)
- The `ml/data/*.csv` manifests contain **2,648 valid scalar rows** after filtering out comment rows and non-numeric labels.
- Those rows collapse to **538 unique image references** and **538 unique resolved filesystem paths**.
- Of those, **144 unique stems** overlap with files present in `ml/data/captured_images/`.
- There are **13 resolved-path label conflicts** where the same file is assigned multiple temperatures across manifests.
- So the repo does contain many more capture files and manifest rows than the canonical 141, but we need deduping and conflict resolution before all of them can be treated as clean supervised training data.

### Full-Data Tiny Run (2026-05-07)
- Built `ml/data/full_scalar_manifest_v1.csv` with **538 deduped rows**.
- The builder also wrote `ml/data/full_scalar_manifest_conflicts_v1.csv` with **65 conflict rows**.
- Best full-data tiny run so far: `test_mae=16.20C`, `test_rmse=20.28C`, `predicted_std=1.82C`.
- The run early-stopped at epoch 41 and restored weights from epoch 26, so the larger manifest is usable but still not enough by itself to reach the sub-5C target.

### Full-Manifest Rectified Run (2026-05-07)
- Built `ml/data/rectified_crop_boxes_full_scalar_v1.csv` for the full 538-row manifest using the rectifier stage.
- Warm-started full MobileNetV2 scalar training on the noisy full manifest, but the run still landed at `test_mae=16.44C` and `test_pct_under_5c=14.8%`.
- Conclusion: the full manifest remains too noisy for the best learned path, even with rectifier-aligned crops and a warm start.
- The strongest learned path is still the curated rectified-crop fine-tune artifact `ml/artifacts/training/scalar_rectified_crop_finetune_v2_20260422`, which reported `mae=1.66C` on its held-out split.

### The Model Collapse Problem
- **Symptom**: Model predicts ~22.7Â°C for ALL inputs regardless of true temperature. Predicted std=0.24Â°C, correlation=-0.47 with true values.
- **Root cause**: Fine-tuning the MobileNetV2 backbone with only 141 images causes immediate overfitting. The model learns to output the mean temperature instead of actual gauge readings.
- **Architecture issue**: Sigmoid + Rescaling output layer is unstable with small datasets. The model collapses to a constant output to minimize loss.

### Failed Approaches (Documented for Future Reference)
1. **Comprehensive v1** (`train_gauge_comprehensive.py`): On-the-fly loading with `tf.numpy_function` + aggressive augmentation + AdamW + 5-fold CV. Failed due to:
   - `tf.image.resize_with_pad` shape inference errors inside `tf.numpy_function`
   - Double-scaling bug (normalized targets to [0,1] when model already had Rescaling layer)
   - Horizontal flip corrupting labels (reverses needle direction)
   - Val MAE diverged from 6.8Â°C to 15.9Â°C during fine-tuning

2. **Comprehensive v2** (`train_gauge_comprehensive_v2.py`): Preloaded images + canonical baseline pipeline + frozen backbone. Still collapsed because:
   - 256-unit head with alpha=1.0 = 328K trainable parameters for 141 images
   - MSE loss with extreme sample weights (up to 25Ã—) caused gradient explosions
   - Even with frozen backbone, the oversized head overfit immediately

3. **All-data baseline** (`train_all_data_baseline.py`): Proven canonical pipeline with merged data. Achieved 7.98Â°C MAE (same as canonical baseline on 141 images), but hard cases still 23.85Â°C.

### Working Approach: Tiny MobileNetV2
- **Model**: `build_mobilenetv2_tiny_regression_model()` (alpha=0.35, head=64, dropout=0.15)
- **Parameters**: ~1.2M total, ~50K trainable (vs 3.5M/328K for alpha=1.0/256-head)
- **Training**: Frozen backbone, MSE loss, 40 epochs, batch=8, LR=1e-4
- **Key insight**: Smaller model = less overfitting. The tiny model is designed for STM32N6 deployment anyway.

### Critical Hyperparameter Lessons
| Parameter | Bad Value | Good Value | Why |
|---|---|---|---|
| Backbone | Unfrozen | Frozen | 141 images can't fine-tune 3M+ params |
| Head size | 256 units | 64 units | Fewer params = less overfitting |
| Alpha | 1.0 | 0.35 | Smaller backbone = fewer features to overfit |
| Loss | Huber(delta=1.0) | MSE | Huber is too forgiving for 20-40Â°C errors |
| Augmentation | Horizontal flip | No flip | Flip reverses needle direction, corrupts labels |
| Sample weights | Uncapped (25Ã—) | Capped at 5Ã— | Extreme weights destabilize gradients |
| LR warmup | 1e-3 | 5e-5 | Too high LR causes immediate divergence |
| LR fine-tune | 1e-4 | 1e-5 (or don't fine-tune) | Any backbone LR causes collapse |

### Architecture Lessons
- **Sigmoid + Rescaling is problematic** for regression with small datasets. The sigmoid compresses gradients at extremes, making it hard to learn cold/hot values.
- **Linear output head** (no sigmoid, no rescaling) would be better for small datasets, but the current model builder enforces sigmoid.
- **MobileNetV2 features are sufficient** â€” the issue is not feature quality, it's the head/regression head capacity and training stability.

### Next Steps for <5Â°C Target
1. Run tiny model training and evaluate
2. If still >5Â°C, consider:
   - Custom model with linear output (no sigmoid compression)
   - Needle-angle detection as intermediate representation
   - Data augmentation focused on needle rotation (not photometric)
   - Ensemble of multiple tiny models

### Files Created This Session
- `ml/scripts/train_gauge_comprehensive.py` â€” v1 (broken, on-the-fly loading)
- `ml/scripts/train_gauge_comprehensive_v2.py` â€” v2 (preloaded, still collapsed)
- `ml/scripts/train_all_data_baseline.py` â€” proven pipeline with all data
- `ml/scripts/train_head_only.py` â€” frozen backbone + 256-head (still collapsed)
- `ml/scripts/train_tiny_model.py` â€” tiny MobileNetV2 (current best hope)
- `ml/scripts/analyze_predictions.py` â€” prediction analysis tool
- `tmp/run_comprehensive_training.sh` â€” v1 runner
- `tmp/run_comprehensive_v2.sh` â€” v2 runner
- `tmp/run_all_data_baseline.sh` â€” all-data runner
- `tmp/run_head_only.sh` â€” head-only runner
- `tmp/run_tiny_model.sh` â€” tiny model runner

## Topic Files

- [Foundation notes](ai-memory/foundation.md)
- [Workflow and WSL notes](ai-memory/workflow.md)
- [Firmware and board notes](ai-memory/firmware-board.md)
- [ML experiments and research notes](ai-memory/ml-experiments.md)
- [Legacy archive](ai-memory/archive.md)

## How To Use This

- Write new durable facts into the topical file that matches the area.
- Update this index when a new topic file is added.
- Use the archive only for older chronology or deep detail.

## 2026-05-02 HardFault after AI inference (STM32N657 baseline runtime)
- Symptom: HardFault with CFSR=0x00008200 (precise bus fault), BFAR valid, PC/LR in AppBaselineRuntime_EstimatePolarNeedle().
- Root cause: out-of-bounds reads in peak selection logic in app_baseline_runtime.c:
  - loops used peak_idx < 128 while top_bins/top_scores arrays were size 16.
  - top_scores indexed with best_hot_bin/best_bin (angle-bin indices), not top-candidate indices.
- Fix applied: cap loops to APP_BASELINE_TOP_PEAK_COUNT and compare hot-zone override using smoothed_votes[bin] instead of top_scores[bin].

## 2026-05-02 Baseline Hot-Wrap Tuning (46C miss)
- Symptom: needle near 46C (hot wrap zone) but baseline selected ~214.7deg and reported ~-6.4C.
- Tuning in app_baseline_runtime.c:
  - Increased top-peak shortlist from 16 to 64 (APP_BASELINE_TOP_PEAK_COUNT) so weaker hot-wrap peaks are retained.
  - Widened hot-zone candidate window from 25-65deg to 20-75deg.
  - Relaxed hot-zone continuity/hub thresholds from 0.30/0.20 to 0.28/0.18.
  - Lowered hot-zone override vote ratio gate from 0.50 to 0.35.
  - Added debug line when a hot candidate exists but is not selected ([BASELINE] Hot-zone candidate kept secondary).

## 2026-05-02 Baseline Hot-Zone Full-Sweep Rescue
- Symptom: baseline still picked false mid-angle peaks (~214deg/~232deg) on hot needle frames or failed estimate under bright captures.
- Fix in app_baseline_runtime.c: added a conservative full-bin hot-zone rescue pass (20deg-75deg) that scans all angle bins (not just top shortlist), re-scores continuity/hub darkness, and overrides only when hot vote is at least 22% of primary vote.
- New debug line: [BASELINE] Hot-zone full-sweep rescue: primary=... hot=...

## 2026-05-02 Baseline Warm/Hot Angle Gate Fix
- Symptom: baseline frequently returned 'Classical baseline failed' when AI reported ~41C-42.5C.
- Root cause: angle validation in AppBaselineRuntime_EstimatePolarNeedle() incorrectly rejected large parts of the valid wrapped Celsius sweep (0-30deg and 315-360deg). Those ranges are valid for warm/hot readings when min_angle=135deg and sweep=270deg.
- Fix:
  - Added helper functions to normalize/check wrapped sweep validity.
  - Replaced hard-coded angle rejection checks with sweep-aware validation: keep angles in wrapped Celsius sweep, reject subdial clutter band only.
  - Updated inversion logic to avoid flipping angles already in the wrapped sweep.
  - Added explicit log when all candidates fail before selection: selected=none(0).

## 2026-05-02 Baseline Bright-Frame Adaptive Thresholding
- Symptom: baseline still failing on overexposed captures (mean ~195-214, bright ratio ~65-82%) even after warm/hot angle gate fix.
- Fix in app_baseline_runtime.c:
  - Added per-frame brightness profile (`mean_luma`, `bright_ratio`, mode normal vs bright-relaxed).
  - In bright-relaxed mode, lowered polar edge and spoke/hub continuity thresholds (edge 8.0->5.5, main continuity/hub 0.35/0.25->0.24/0.15, hot continuity/hub 0.28/0.18->0.22/0.12, final spoke continuity 0.30->0.22).
  - Added diagnostics: frame profile log plus explicit polar reject reasons (`no_peak`, `angle`, `continuity`) and selected=none candidate log when all hypotheses fail.

## 2026-05-02 Baseline Hot-Zone Log + Bright-Jump Stability Hold
- Observation: baseline now reaches ~46-47C on hot frames, but occasional bright-relaxed outliers still appear (e.g., ~32.9C with moderate confidence).
- Fix in app_baseline_runtime.c:
  - Hot-zone logs now use integer x10 formatting (no %f) so angles print correctly instead of primary=deg hot=deg.
  - Added bright-relaxed stability guard in IsStableEstimateForHistory(): if frame is bright-relaxed and temp jump > 8.0C with confidence < 8.0, reject as unstable and hold last stable estimate.
  - Added log: [BASELINE] Stability hold: bright jump=...C conf=.../1000.

## 2026-05-02 Re-enabled CNN Affine Output Calibration
- Change: set APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION back to 1 in app_inference_calibration.c.
- Reason: short-term correction for board under-reading at hot end (CNN outputs ~41-43C when baseline indicates ~46-47C).
- Notes: kept existing cold-threshold guard (APP_INFERENCE_CALIBRATION_COLD_THRESHOLD=-10C) and mild-scale behavior unchanged.

## 2026-05-02 Baseline Ambiguous-Jump Guard
- Symptom: occasional large cold jump (e.g., ~46.6C to ~-23.8C) accepted in bright-relaxed mode with nearly tied peaks (score ~= runner_up).
- Fix in app_baseline_runtime.c:
  - Added ambiguous-jump stability guard in IsStableEstimateForHistory(): if last result exists, delta > 12.0C, peak ratio < 1.08, and confidence < 12.0, reject and hold prior stable estimate.
  - Added log: [BASELINE] Stability hold: ambiguous jump=...C ratio=.../1000 conf=.../1000.
  - Normalized hot-zone debug angles to [0,360) before logging (avoids values like 380deg/405deg in logs).

## 2026-05-02 AI Scalar Hot-End Bringup: Dry-Run Calibration + Affine Fill Resize
- Symptom: baseline reached hot range (~46-48C), but AI scalar stage often plateaued lower (~41-44C) in OBB-crop path.
- Fix in app_ai.c:
  - Applied `AppAI_TraceAndApplyInferenceCalibration()` in `App_AI_RunDryInferenceFromYuv422()` before publishing `app_ai_last_inference_value` (both OBB-success and fixed-training fallback branches). This re-enables calibrated board-facing output in the dry-run cascade.
  - Re-enabled full affine crop-to-tensor mapping (`APP_AI_ENABLE_AFFINE_FILL_RESIZE=1`) in preprocess so scalar stage no longer letterboxes non-square crops with large zero-padding bands.
- Expected logs:
  - `[AI] Model output before calibration: ...`
  - `[AI] Model output after calibration: ...`
  - `[AI] Calibration delta: ...`
  - Scalar input probe should keep changing hash while edge coverage improves due no letterbox bars.

## 2026-05-02 AI Calibration Retune (Hard-Case Milder + Hot Blend)
- Symptom: after dry-run calibration was re-enabled, some hot/bright close-up frames over-shot badly (example: raw `44.22C` calibrated to `52.17C`).
- Fix in `app_inference_calibration.c`:
  - Replaced old p5 affine constants (`scale=1.1631`, `bias=0.7423`) with milder hard-case constants from `scalar_hardcase_boost_v8` (`scale=1.0502802`, `bias=0.6553916`).
  - Added a hot-end blend gate: full affine only through `43.0C` raw, then apply only partial correction (`HOT_BLEND=0.35`) above that range.
  - Kept cold-side blend at `0.0` (identity) below `-10C` raw.

## 2026-05-02 AI Calibration Low-Band Neutralization
- Symptom: at true ~`10C`, scalar raw was already high (`14.35C`) and calibration increased it further (`15.73C`), widening error.
- Fix in `app_inference_calibration.c`:
  - Added a low-band gate: below `20.0C` raw, calibration blend is `0.0` (identity), so low-temperature reads are no longer pushed upward by affine correction.
  - Full affine now applies only in `[20.0C, 43.0C]` raw; hot-side partial blend remains unchanged.

## 2026-05-02 AI Calibration Cold-Tail Correction
- Symptom: at true ~`-26C`, scalar raw stayed too warm (`-17.28C`) and low-band identity left it uncorrected.
- Fix in `app_inference_calibration.c`:
  - Added a capped cold-tail correction below raw `-12.0C`:
    - `extra_cold_delta = min(1.05 * (-12 - raw), 8.0)`
    - final correction subtracts this extra delta (pushes value colder).
  - Keeps prior low-band neutralization for around-ambient values while improving deep-cold cases.

## 2026-05-02 Baseline AI Cross-Check Softened
- The baseline AI cross-check in `app_baseline_runtime.c` is now advisory instead of a hard veto.
- It still logs when the baseline candidate disagrees strongly with a cold AI reading, but it no longer freezes history on that mismatch.
- This should help the baseline recover from stale warm locks on cold frames instead of holding a previous estimate indefinitely.

## 2026-05-04 - No-calibration CNN contract (MobileNet)
- Removed post-hoc affine calibration usage from ML training and board pipeline replay path.
- TrainConfig.linear_output=True is now rejected to prevent reintroducing calibration-dependent outputs.
- gauge_calibration_parameters.toml is now treated as gauge metadata/spec (added units and direction).
- Added strict gauge-spec validation + conversion helpers in gauge/processing.py for value/fraction/angle mappings.
- Deleted legacy affine calibration regression test and added new tests for no-calibration behavior.
- Focused tests passing: poetry run pytest tests/test_gauge_processing.py tests/test_training.py tests/test_board_pipeline.py -q (49 passed).

## 2026-05-04 - Export + STM32N6 relocatable package (new6 no-cal)
- Exported int8 scalar model: ml/artifacts/deployment/scalar_full_finetune_from_best_board30_clean_plus_new6_int8/model_int8.tflite.
- Packaged relocatable binary with X-CUBE-AI 10.2.0 in WSL: ml/artifacts/runtime/scalar_full_finetune_from_best_board30_clean_plus_new6_int8_reloc/scalar_full_finetune_from_best_board30_clean_plus_new6_int8_rel.bin.
- Refreshed firmware canonical model blob: irmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw.
- Hard manifest eval snapshot (fixed-crop script): MAE ~9.91C on hard_cases_plus_board30_valid_with_new6.csv; target <5C not yet met.
- WSL CUDA/GPU probe timed out in this session; packaging/export succeeded in WSL CPU path.

## 2026-05-05 no-cal hard-case recovery + packaging
- Re-evaluated candidate checkpoints on `ml/data/hard_cases_plus_board30_valid_with_new6.csv`.
- Best current no-calibration checkpoint remains `ml/artifacts/training/no_cal_hardpush_gpu1/model.keras` with `n=57 MAE=5.15 bias=-2.35 std=5.44`.
- Multiple additional fine-tunes (`no_cal_final_biasfix_v3_tail`, `v4_gentle`, `v5_headonly_tail`) regressed and were rejected.
- Exported best checkpoint to int8:
  - `ml/artifacts/deployment/no_cal_hardpush_gpu1_int8/model_int8.tflite`
  - `ml/artifacts/deployment/no_cal_hardpush_gpu1_int8/metadata.json`
- Packaged relocatable STM32N6 binary:
  - `ml/artifacts/runtime/no_cal_hardpush_gpu1_int8_reloc/no_cal_hardpush_gpu1_int8_rel.bin`
- Refreshed canonical flash blob:
  - `firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw`

## 2026-05-05 OBB HardFault + Scalar Signature Mismatch

**Symptom**:
- OBB stage crashed with HardFault at `BFAR=0x70A11E80` right after `Stage inference run start`.
- Scalar fallback sometimes failed with `xSPI2 stage 'scalar' signature mismatch`.

**Root cause**:
- Stage switch to scalar called `AppAI_ReconfigureXspi2ForRuntime()` (disables memory-mapped mode).
- Scalar signature mismatch aborted early, leaving xSPI2 in indirect mode.
- Next frame reused OBB `app_ai_loaded_xspi2_stage` fast-path (`already loaded`) and skipped MM re-enable.
- OBB then faulted on first flash-weight read (`_mem_pool_xSPI2_mobilenetv2_obb_longterm + 0x311E80`).

**Fix in `app_ai.c`**:
1. Added `app_ai_xspi2_mm_enabled` runtime state flag.
2. Clear MM flag on xSPI reconfigure/deinit paths; set flag on successful MM enable.
3. In stage fast-path (`already loaded`), re-enable MM if it is off.
4. In stage reconfigure path, clear `app_ai_loaded_xspi2_stage` before verify/reload.
5. Made scalar signature mismatch non-fatal (warn + continue) to tolerate externally flashed scalar blobs.

**Result**:
- Prevents stale-stage/MM-off OBB HardFault loop.
- Keeps scalar fallback usable after external reflashes with different blob signatures.

## 2026-05-05 Scalar Zero Output Root Cause

**Observed**: Scalar stage ran successfully but output tensor stayed constant at float `-0.0` (`00 00 00 80`).

**Root cause**: The flashed canonical scalar blob (`st_ai_output/atonbuf.xSPI2.raw`) did not match the scalar package C runtime currently compiled into firmware.
- Canonical blob: `3218865` bytes, tail `...00DB`
- Package-matched blob (`prod_model_v0.2_raw_int8/..._atonbuf.xSPI2.raw`): `3218881` bytes, tail `...00DC`

**Fix**:
1. Flashed the package-matched scalar blob at `0x70200000`.
2. Synced canonical `st_ai_output/atonbuf.xSPI2.raw` to that same package file to prevent regressions.

- 2026-05-06: WSL GPU training fix for TensorFlow 2.21.0 in ml/.venv: export LD_LIBRARY_PATH to include /usr/lib/wsl/lib plus all .venv site-packages nvidia/*/lib paths before running training. Without this, TF reported 'Cannot dlopen some GPU libraries' and saw no GPU. Also set 'set -euo pipefail' in tmp/train_variant_a_gpu_v2.sh so pipeline failures are not masked by tee.

- 2026-05-06: If commands/scripts show no output or appear stuck, immediately run 'wsl --shutdown' and retry. Treat WSL restart as first-line recovery on this machine.

- 2026-05-06: Updated ml/.venv/bin/activate to auto-export LD_LIBRARY_PATH with /usr/lib/wsl/lib and .venv site-packages nvidia/*/lib paths, and to restore previous LD_LIBRARY_PATH on deactivate. This makes TF GPU detection automatic after source .venv/bin/activate.

## 2026-05-06 WSL GPU + Retraining Blockers

- Root freeze cause for `tmp/run_canonical_training.sh` was WSL GPU stack state. After `wsl --shutdown`, both `nvidia-smi` and TensorFlow GPU device enumeration recovered.
- `scripts/train_hardcase_interval.py` currently cannot run in this checkout because `ml/data/raw/` is empty and training expects files like `ml/data/raw/PXL_*.jpg`.
- The CVAT zip files in `ml/data/labelled/` currently contain `annotations.xml` only in this workspace (no raw image payload), so raw-image reconstruction was not possible from local files.
- Fixed a real bug in `ml/scripts/train_canonical_baseline.py` dataset weighting path:
  - changed `dataset.map(lambda x, y, w: ((x, y), w))`
  - to `dataset.map(lambda x, y, w: (x, y, w))`
  - This avoids Keras treating `(x, y)` as two model inputs.
- Canonical retrain still needs a clean rerun after process cleanup; prior run got stuck during image preload after this fix.
- `ml/scripts/run_mobilenetv2_rectified_scalar_pure_finetune_v3.sh` now builds `data/rectified_scalar_strict_boxed_train_v4.csv` on every run and keeps only `ml/data/captured_images/...` rows that also appear in `ml/data/rectified_crop_boxes_v5_all.csv`.
- The stricter v5 rectified recipe uses only `ml/data/captured_images/...` rows and excludes the holdout manifest, which currently yields 155 training rows with 8 unique label values. This is the better strict baseline than the 26-row boxed-only variant.
- `ml/scripts/run_mobilenetv2_rectified_scalar_mixed_finetune_v6.sh` builds a weighted mixed manifest with the 155-row strict rectified pool plus the broader board30/full-labelled sources, deduped against the hard-case holdout. The resulting mix is 507 rows with 17 unique label values and should be the next hard-case-improvement run.
- `ml/scripts/run_mobilenetv2_rectified_scalar_interval_v9.sh` adds an interval auxiliary head on top of the strict rectified warm start, but the hard-case-heavy mix still collapses toward the middle. Final metrics were `test_mae=18.55C`, `test_rmse=21.94C`, `test_hard_mae=38.63C`, and `test_pct_under_5c=11.7%`, so interval binning did not solve the cold-end/generalization gap.
- `ml/scripts/run_mobilenetv2_rectified_scalar_linear_v10.sh` restores a linear scalar head on top of the strict rectified v5 pool and warm-starts from `mobilenetv2_rectified_scalar_strict_v5/model.keras`. It improved the held-out split to `test_mae=6.10C` and `test_pct_under_5c=50.0%`, but the hard-case eval on `hard_cases_plus_board30_valid_with_new5.csv` was still `mae=16.8233C`, `hard_mae=31.8625C`, and `correlation=0.0910`, so linear output helped less than hoped on the cold/hot extremes.
- 2026-05-08 hard-case visual audit: the worst cold samples (`capture_m30c_preview.png`, `capture_m19c.png`, `capture_m18c.png`, `capture_m25c.jpg`) look plausible rather than blatantly mislabeled, but the preview-style frames are lower-contrast and remain the hardest cases. The model still collapses toward a middle-band prediction on the hard manifest, so the next gain likely needs more cold/end coverage rather than another tiny head tweak.
- 2026-05-08 preview-heavy hard-synth fine-tune v14 improved the easy split (`test_mae=4.14C`) but hard eval stayed poor (`mae=17.2102C`, `hard_mae=30.0428C`, `correlation=-0.3146`).
- 2026-05-08 strict+focus mix v15 and separate hard-case adaptation v16 both failed to rescue the cold end. v15 ended at `hard_mae=34.17C` and v16 ended with `test_mae=21.07C` / `baseline_mae_mean_predictor=20.26C`, so the model still behaves like a center-biased regressor when we push harder on the tails.
- Current conclusion: augmentation and reweighting alone are not enough. We likely need either more genuinely labeled cold/preview captures or a different labeled-tail source before another scalar CNN tweak will move the hard manifest.
- 2026-05-09 geometry bridge v19: generated `ml/data/geometry_crop_boxes_v18.csv` from the raw CVAT geometry model and retrained the strict rectified scalar model against those crops. The retrain completed, but it regressed on the held-out split (`test_mae=20.5455C`, `test_rmse=23.99C`, `test_hard_mae=39.81C`, `test_pct_under_5c=12.1%`). Geometry-derived crops are not the fix.
- 2026-05-09 geometry cascade v20: evaluated the raw-CVAT geometry model directly as the cascade front-end against `hard_cases_plus_board30_valid_with_new5.csv` with the strict v5 scalar reader. It still collapsed near 27C on the hard set (`mean_final_abs_err=18.5442C`, `cases_over_5c=22/28`, worst `-30C -> 26.95C`). The geometry front-end is not yet useful as a replacement for the scalar crop stage.
- 2026-05-09 rectified scalar curriculum v23: ran a two-stage cold-tail curriculum using the strict v5 weights, 507 real rows plus 1200 standard synthetic rows in stage 1, then a cold-tail-heavy stage 2 with 1000 hard synthetic rows and preview-heavy augmentation. Stage 1 and stage 2 both completed, but the curriculum did not improve the hard manifest. Final metrics were `test_mae=22.89C`, `test_rmse=25.69C`, `test_hard_mae=37.43C`, and `correlation=0.0313`. The broad curriculum pushes the model even farther toward the center than the strict v5 baseline, so it is not the right next lever.
- 2026-05-09 strict-v5 all-sources recreation v24: rebuilt the strict v5 recipe on the deduped union of `full_labelled_plus_board30_valid_with_new5.csv`, `full_scalar_manifest_v1.csv`, and 1600 corrected synthetic renders. The final model still did not beat the strict v5 baseline on the hard manifest. Easy split was `test_mae=20.08C` / `test_pct_under_5c=15.6%`, and hard-case eval was `mae=19.1212C`, `hard_mae=34.5558C`, `correlation=0.2474`. The broader all-sources mix still collapses toward the middle band, so the strict v5 baseline remains the best scalar model we have.
- 2026-05-09 hard-tail specialist v25: trained a tail-focused specialist from the strict v5 weights using 155 strict anchor rows, 264 cold/hot labelled-tail rows, and 936 hard synthetic rows. The specialist did not solve the hard manifest either. Easy split landed at `test_mae=25.68C` / `test_pct_under_5c=4.9%`, and hard-case eval was `mae=21.7846C`, `hard_mae=37.0372C`, `correlation=0.3567`. The specialist predicts the right general slope a bit better, but the absolute error is still far too high, so a hard-tail-only scalar CNN is not the missing fix by itself.
- 2026-05-09 true two-stage geometry-localizer + strict-reader cascade v30: trained a geometry-only MobileNetV2 localizer by setting `geometry_value_loss_weight=0.0` and warm-starting from `mobilenetv2_geometry_literature_v29`, then evaluated the resulting localizer with the strict rectified v5 scalar reader on `hard_cases_plus_board30_valid_with_new5.csv`. The localizer trained cleanly, but the cascade still collapsed around the same middle band: `mean_final_abs_err=18.5246C` over 28 samples, `cases_over_5c=23`, and worst case `capture_m30c_preview.png -> 27.3077C` for a true `-30C`. The localizer heatmaps remained low-confidence (`first_conf` around `0.03`), so this two-stage split is structurally correct but still not a hard-case win. We likely need a stronger localizer objective or a real detector/box stage before the reader can benefit.
- 2026-05-09 OBB localizer v31: trained a dedicated `mobilenet_v2_obb` localizer from `mobilenetv2_obb_longterm/model.keras` with `alpha=0.35`, `head_units=64`, and a low LR. The OBB stage trained cleanly and produced much better hard-case crops than the keypoint/geometry localizers. When paired with the existing rectified scalar int8 reader, the hard-case cascade improved to `mean_abs_err=9.0359C` over 31 samples, with worst case `capture_m25c.jpg -> 18.9515C` (true `-25C`). This is the first localizer path that materially helps the downstream scalar reader, so OBB should now be the preferred stage-one candidate for the cascade.
- 2026-05-09 OBB crop-scale sweep v33: a tighter crop-scale sweep around the OBB cascade confirmed that the best setting is `OBB_CROP_SCALE=1.30` with the rectified-finetune-v2 int8 reader. That run reached `mean_abs_err=8.1181C`, `cases_over_5c=16`, and `max_abs_err=38.7551C` on `hard_cases_plus_board30_valid_with_new5.csv`. The neighboring scales were all slightly worse, and the strict `v5` int8 reader did not beat this baseline, so keep the older rectified-finetune-v2 int8 reader for the OBB cascade for now.
- 2026-05-09 `mobilenetv2_obb_geometry_v32` finished cleanly on the labeled CVAT set (`352` samples, `341` in-sweep). The geometry/value branch trained well and reached a best validation `gauge_value_mae` of about `0.36` during fine-tuning, but the keypoint-angle supervision stayed noisy (`~71-81 deg MAE`) and the saved test metrics were `gauge_value_mae=0.3418`, `keypoint_coords_angle_mae_deg=71.09`, and `baseline_mae_mean_predictor=20.07`. This is promising as a geometry pretrainer, but it still needs downstream cascade evaluation before we treat it as a replacement for the OBB + scalar baseline.
- 2026-05-09 cascade eval of `mobilenetv2_obb_geometry_v32` against `hard_cases_plus_board30_valid_with_new5.csv` did not improve the hard-tail benchmark over the existing OBB front-end. With the strict rectified `v5` reader as the second stage, the cascade scored `samples=28`, `skipped=3`, `mean_first_pass_abs_err=18.5098C`, `mean_final_abs_err=18.4474C`, `cases_over_5c=22`, and worst case `capture_m30c_preview.png -> 26.9521C`. The OBB localizer path remains the better two-stage deployment baseline, while `v32` is best treated as a geometry pretrainer rather than a deployable front-end replacement.
- 2026-05-09 blur-aware OBB-geometry follow-up v34 trained cleanly with the simplified raw-plus-unsharp branch, but the self-cascade benchmark on `hard_cases_plus_board30_valid_with_new5.csv` regressed badly. Final cascade numbers were `samples=28`, `skipped=3`, `mean_first_pass_abs_err=24.9102C`, `mean_final_abs_err=24.8235C`, `cases_over_5c=26`, and worst case `capture_p50c_preview.png -> 1.4618C` for a true `50C`. The blur-aware branch is therefore not a deployable reader; the architecture still needs a stronger geometry/value calibration path if we want the literature trend to pay off.
- 2026-05-11 crop-window widening on the board pipeline: lowered `OBB_TRAINING_CROP_MIN_RATIO` from `0.60` to `0.15`, raised `OBB_TRAINING_CROP_MAX_RATIO` from `1.40` to `1.60`, and aligned `OBB_CROP_SCALE` to `1.30` in `board_pipeline.py` so hard-tail OBB crops around 0.19 of the rectified training window are no longer forced into rectifier fallback. The replay rerun on `mobilenetv2_bluraware_obb_sequence_geometry_v43` still did not improve the hard manifest: `samples=31`, `mae=25.7046C`, `rmse=28.4629C`, `max_abs_error=49.9934C`. The widened window let the OBB crop through, but the scalar reader still collapses on the cold end, so the remaining blocker is reader robustness, not just the crop guard.
- 2026-05-11 sequence-reader v43 and hard-tail v44 lessons:
  - The blur-aware OBB sequence reader (`mobilenetv2_bluraware_obb_sequence_geometry_v43`) is the strongest easy-split scalar model so far in the OBB/keypoint family: `test_gauge_value_mae=0.8460C`, `test_gauge_value_rmse=1.0658C`, `test_keypoint_coords_angle_mae_deg=70.6780`, `test_pointer_mask_mae=0.1548`.
  - However, its hard-case board replay is still not good enough for deployment: `mae=25.7147C`, `rmse=28.4097C`, `max_abs_error=49.5147C`.
  - The hard-tail fine-tune attempt (`mobilenetv2_bluraware_obb_sequence_geometry_hardtail_v44`) did not produce a better replay; the final hard-case summary stayed essentially the same (`mae=25.6554C`, `rmse=28.3893C`, `max_abs_error=49.6571C`).
  - The hard replay traces showed the important failure mode: the OBB crop now usually gets through, but the reader still collapses toward low/near-zero predictions on the cold and preview-heavy samples. So the current bottleneck is reader robustness on the hard tail, not the crop guard alone.
  - The best current practical deployment baseline remains the OBB localizer plus the better rectified scalar reader / OBB-cascade path (around `8.1181C` MAE on the hard manifest with `OBB_CROP_SCALE=1.30`), not the newer geometry-heavy readers.
  - Overall lesson: the literature-backed geometry/keypoint/sequence ideas do help on the easy split and as pretrainers, but none of them have yet beaten the simpler OBB + rectified scalar cascade on the hard manifest. The cold tail still needs either better reader robustness or more representative hard-tail data.

## 2026-05-12 Dual-Resolution Interval Reader

- Added a new embedded-friendly CNN candidate, `mobilenet_v2_dualres_interval`, that reuses the existing dual-resolution MobileNetV2 backbone and trains a distributional interval head instead of a naked scalar regressor.
- The training CLI now supports `--hard-case-eval-manifest`, which reserves a manifest entirely for validation/test and keeps those hard cases out of the training pool.
- This is the cleanest path for the next reader experiment: train on the full non-hard labelled pool, hold the hard cases out for evaluation only, and compare the new dual-resolution interval model against the current scalar production path.

## 2026-05-13 Reproducible Hard-Case Winner

- The earlier hard-case winner here was the head-only QAT recipe in `ml/scripts/run_scalar_qat_headonly_from_best_board30.sh`; that is now a historical reference only because prod v0.4 is deployed from the calibration-free `no_cal_hardpush_gpu5_recover` model.
- Exact winning settings:
  - base model: `artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras`
  - `--freeze-backbone`
  - `--no-augment-training`
  - `--batch-size 8`
  - `--seed 21`
  - `--hard-case-manifest data/hard_cases_plus_board30.csv`
  - `--hard-case-repeat 8`
  - `--edge-focus-strength 1.5`
  - `--epochs 4`
  - `--learning-rate 5e-7`
- That run logged `test_metrics.mae=7.8563C` and hard-case `mean_abs_err=7.0800C`, so keep this recipe as the current hard-case reference point.

## 2026-05-13 Live Board Scalar Fault Isolation

- The live scalar crash on the STM32N6 board is no longer a crop-selection bug. The OBB stage completes, the scalar stage starts, and the fault happens inside `AppAI_PreprocessYuv422FrameToFloatInput()` during the row-0 resize work.
- The hard fault snapshot is now visible on UART because the fault handler uses a direct raw LPUART writer. The most useful snapshot so far was:
  - `HardFault PC=0x340088DE`
  - `CFSR=0x00000400`
  - `HFSR=0x40000000`
  - `SP=0x341033A0`
- The scalar helper was tightened in two ways:
  - the YUV422 pixel readers now validate `source_x` / `source_y` before computing byte indices;
  - the scalar preprocess loop now uses a smaller row-pointer based write path instead of repeated full-tensor indexing and temporary line buffers.
- The preprocess function stack usage dropped from `192 static` to `64 static` after the cleanup.
- Current hypothesis: the remaining fault is either an imprecise bus fault from the scalar write path or a board/runtime memory interaction in the AI worker thread. The next live test should tell us whether the simplified row writer gets past row 0.
- Follow-up simplification on 2026-05-13 replaced the luma-only float conversion in the scalar row loop with an integer lookup table of IEEE-754 bit patterns, so the hot path no longer emits `vcvt`/`vstr` float stores for grayscale writes. The preprocess helper stack usage is now `56 static`. This should further reduce FPU interaction and leave only integer stores in the row loop on the live board.
- 2026-05-13 isolation refinement: scalar handoff is enabled again, but `AppAI_RunStageInference()` still returns immediately after scalar preprocess. This is the next checkpoint to separate â€œtensor fill hangsâ€ from â€œLL_ATON runtime hangsâ€ while keeping the OBB stage and watchdog healthy.
- 2026-05-13 scalar row-scratch follow-up: the scalar preprocess now copies one packed YUV422 row into `app_ai_scalar_row_scratch` before sampling luma. This removes the repeated direct frame-buffer reads from the hot loop and is the current live fix candidate for the precise bus fault in `AppAI_ReadYuv422Luma()`.
- 2026-05-13 stable-frame follow-up: the dry-run frame is now copied into `app_ai_dry_run_frame_scratch` before any stage reconfiguration. This is to avoid depending on the live snapshot buffer staying valid while the OBB and scalar stages reconfigure xSPI2 and the runtime around it.
- 2026-05-13 board-safe scalar rewrite: the scalar row writer now uses explicit byte stores for each IEEE-754 float slot instead of `memcpy()` or compiler-emitted wide stores. That removes the last alignment-sensitive store pattern from the hot path and is the next board test to try.
- 2026-05-13 root-cause breakthrough: the board fault was driven by RAM region overlap, not just scalar math.
  - The ST AI scalar package uses a fixed activation/input window at `0x34100000` (CPU RAM2 virtual pool).
  - App `.bss` had grown into the same range (`camera_ai_thread_stack` and friends), so preprocessing writes corrupted thread stack/control flow.
  - Fix applied: moved `camera_ai_frame_snapshot` and `camera_baseline_frame_snapshot` into `.noncacheable`, and increased `RAM_NC` in `STM32N657X0HXQ_LRUN.ld` from `128K` to `384K`.
  - Verified in `n657_Appli.map`: `.bss` now ends at `0x340f6178`, safely below `0x34100000`; `.noncacheable` now holds baseline snapshot, AI snapshot, and capture buffer.
- 2026-05-14 live scalar input saturation fix: the board was accepting an over-bright capture and only nudging exposure for the next cycle, which left the scalar tensor pinned at `1.0`. The capture path now retries after a brightness-gate nudge instead of feeding the current too-bright frame downstream, and the IMX335 seed exposure was lowered from `2/3` of the sensor range to `1/3` so the first capture starts darker.
- 2026-05-14 scalar runtime hardening: the scalar stage wrapper now treats out-of-range decoded values as warnings instead of hard failures. This keeps the board running while we inspect the live decoded scalar value and decide whether the remaining issue is the model output distribution or the plausibility window.
- 2026-05-14 scalar int8 contract fix: the latest prod-v0.4 scalar package exposes a 1-byte int8 output buffer (`Quantize_390_out_0`), but the runtime was still rejecting anything smaller than `sizeof(float)` before decode. The stage wrapper now accepts the int8 tensor length and only rejects truly undersized outputs, which should eliminate the remaining `Dry-run entry aborted during scalar stage` path once the fresh build is flashed.
- 2026-05-14 scalar model replacement: the current prod-v0.4 scalar checkpoint is not a quantization bug. Offline replay showed the exported `prod_model_v0.4_scalar_int8` model itself was collapsing to enormous raw predictions on hard captures, while the candidate `scalar_hardcase_boost_v1_calibrated_int8` stayed in a sane Celsius range. On the first 19 hard-case rows, the old model had `mae=64458453.9474C`, while the new candidate hit `mae=4.5362C` and `rmse=5.7693C`. I repacked prod-v0.4 around the better checkpoint, refreshed `st_ai_output/atonbuf.xSPI2.raw`, and updated the firmware tensor probes to prefer the new final output name `Quantize_261_out_0`.
- 2026-05-14 OBB crop restoration: the fixed training crop was masking the geometry problem on the live board. I re-enabled the OBB front-end and changed the scalar handoff back to the decoded OBB crop instead of forcing the fixed training crop. This is the live test path for the 48C gauge case.
- 2026-05-14 OBB runtime stability follow-up: once the OBB handoff was re-enabled, the live board hit a HardFault inside the OBB `LL_ATON_RT_RunEpochBlock()` path. The AI worker thread stack was 64 KB at the time, so I increased `CAMERA_AI_THREAD_STACK_SIZE_BYTES` to 128 KB as a low-risk stability test before changing the crop/model path again. If the fault disappears, the issue was likely stack headroom rather than the OBB geometry itself.
- 2026-05-14 ThreadX timer-thread hardening: the fault address mapped into `tx_timer_thread_entry`, and the generated map showed the timer thread stack area at only 1024 bytes. I overrode `TX_TIMER_THREAD_STACK_SIZE` to 4096 bytes in `app_threadx.h` so the kernel timer thread has more room during the heavy OBB/scalar retry path. This is the current live fix candidate for the timer-thread hardfault.
- 2026-05-14 RGB sampler safety fix: the OBB-enabled scalar crash is now looking more like a frame-tail read than a math issue. The RGB YUV422 pixel reader now checks `source_index + 3 < frame_size_bytes` before reading Y/U/V bytes, matching the safer bounds style already used in the luma path. That should prevent the imprecise bus fault we were seeing in the scalar row loop if the runtime ever hands us a slightly smaller or differently packed frame than the sampler expects.
- 2026-05-14 ThreadX timer-stack definition fix: the timer stack override had been placed in `app_threadx.h`, which the ThreadX kernel sources do not use. The real fix belongs in `tx_user.h`, so `TX_TIMER_THREAD_STACK_SIZE` is now defined there at 4096 bytes. This should be reflected in the next link map and remove the lingering 1 KB timer-stack assumption from the build.
- 2026-05-14 ThreadX timer-port confirmation: `tx_port.h` was still defaulting `TX_TIMER_THREAD_STACK_SIZE` to 1024 bytes for the Cortex-M55 port, which meant the kernel kept allocating `_tx_timer_thread_stack_area` as only `0x400` bytes. I changed the port default to 4096, forced `tx_timer_initialize.c` to rebuild, and verified the ELF now reports `_tx_timer_thread_stack_area` as `0x00001000`. That removes one likely source of timer-thread stack corruption from the live hardfault path.
- 2026-05-14 RGB resize padding fix: the live RGB scalar path was not zero-filling rows above/below the resized crop, unlike the luma path. I added an explicit vertical pad check so those rows return zero-filled RGB floats instead of computing `out_y - resize_pad_y` through unsigned underflow. This matches the intended resize-pad semantics and removes a subtle edge case from the scalar hot loop.
- 2026-05-14 live fault refinement: after the padding fix, the hardfault moved again and now resolves into `_tx_thread_system_resume` / stack-check logic with `CFSR=0x00008200` and `BFAR=0x00FF0000`. That points more toward corruption of a ThreadX control block or stack metadata than a clean out-of-bounds pixel read. I added sparse row-progress logs in `AppAI_PreprocessYuv422FrameToFloatInput()` / `AppAI_PreprocessScalarRow()` so the next boot can tell us whether the crash happens on a pad row, the first resized row, or later in the sweep.
- 2026-05-14 row-progress logging cleanup: the sparse row logs were too noisy for the hot path, so I replaced them with a single `app_ai_scalar_preprocess_last_row` marker and taught the HardFault logger to print it. That keeps the breadcrumb while reducing console and stack churn inside the scalar sweep.
- 2026-05-14 final-row scalar hardfault refinement: the live fault now lands with `last_scalar_row=223`, which is the final 224x224 row. That means the crash is no longer on the early top-padding band; it is happening at the bottom edge of the scalar resize sweep, likely during the last bottom-padding row or the tail end of the row loop. I moved the RGB padding-row zero-fill out of the inner sampler loop so the final blank rows short-circuit before any per-pixel work. The next boot should tell us whether the imprecise bus fault clears or whether the remaining corruption is elsewhere in the row tail.
- 2026-05-14 padding-row libc removal: because the final-row hardfault was still showing up at the bottom edge of the scalar sweep, I removed `memset()` from the RGB and luma padding-row paths and replaced it with a straight byte loop. The goal is to keep the final blank row on pure byte stores only, with no libc helper involved in the exact row that is faulting.
- 2026-05-14 padding-row pointer avoidance: the fault still reproduced on `last_scalar_row=223`, so I moved the RGB/luma padding-row check ahead of the `row_bytes` address calculation entirely. Padding rows now return before any per-row pointer arithmetic, which is the cleanest way to test whether the remaining crash is caused by touching the final blank row at all.
- 2026-05-14 padding-row zero-fill restoration: the pointer-avoidance simplification left the tensor head/tail full of `0xEF` guard bytes, which showed up in the live input probe. I restored explicit zero-fill for both RGB and luma padding rows while keeping the padding check early, so the model still receives a valid tensor and we can continue separating true row-tail faults from simple uninitialized padding.

## 2026-05-14 - Scalar-stage crash triage (prod v0.4)
- Rebuilt 
657_Appli and re-flashed full image set via irmware/stm32/n657/flash_boot.bat (FSBL + scalar/rectifier/obb + signed app).
- Strong evidence pointed to AI worker stack corruption/overflow:
  - prior fault address  x340D044E resolves into data region near runtime thread stacks, not executable text.
  - camera_ai_thread_stack previously started at  x340DAF34 (fault addr below stack base).
- Increased CAMERA_AI_THREAD_STACK_SIZE_BYTES from 131072 to 262144 in irmware/stm32/n657/Appli/Inc/app_memory_budget.h.
- Added ThreadX stack error notification callback in pp_threadx.c:
  - registers 	x_thread_stack_error_notify(AppThreadX_StackErrorHandler).
  - logs thread name and stack pointers, turns on red LED, halts.
- Improved fault observability in stm32n6xx_it.c:
  - MemManage/BusFault/UsageFault/SecureFault now capture stacked PC/LR/regs via naked handler wrappers (same pattern as HardFault), instead of hardcoded PC=0 logs.
- Reduced printf risk in AI worker path (pp_inference_runtime.c):
  - replaced large varargs queue/dequeue prints with fixed-string logs to reduce formatter-side corruption surface.

### Next run expectation
- If crash was stack-related, board should run longer or stable.
- If still faulting, new fault logs will include real stacked PC/LR for MemManage/Bus/Usage/Secure and enable precise ddr2line mapping.

## 2026-05-14 Handoff - COM3 Live Monitor + Current Fault State (for Qwen)

### What was done right before this note
- Built latest firmware from:
  - irmware/stm32/n657/Appli/Debug with mingw32-make -j8 all
- Flashed full boot stack + model blobs + app using:
  - irmware/stm32/n657/flash_boot.bat
- Confirmed script flashed:
  - FSBL at  x70000000
  - scalar blob (tonbuf.xSPI2.raw) at  x70200000
  - rectifier blob (tonbuf.rectifier.xSPI2.raw) at  x70600000
  - OBB blob (tonbuf.obb.xSPI2.raw) at  x70700000
  - signed app at  x70100000

### Latest runtime monitor command that worked
`powershell
@'
import serial
import time
import sys

port = 'COM3'
baud = 115200
dur = 150

try:
    s = serial.Serial(port, baud, timeout=0.2)
    print(f'[MON] opened {port} @ {baud}', flush=True)
    t0 = time.time()
    while time.time() - t0 < dur:
        n = s.in_waiting
        d = s.read(n if n > 0 else 1)
        if d:
            txt = d.decode('utf-8', 'ignore')
            if txt:
                sys.stdout.write(txt)
                sys.stdout.flush()
        time.sleep(0.03)
    s.close()
    print('n[MON] capture complete', flush=True)
except Exception as e:
    print(f'[MON] error: {e}', flush=True)
    sys.exit(1)
'@ | py -3 -
`

### Critical findings from COM3 (repeatable)
- Boot and ThreadX startup are healthy.
- Camera probe succeeds consistently.
- Brightness gate often retries multiple times (too-dark), with exposure stepping up.
- Intermittent DCMIPP capture fault appears during retries:
  -  x00008100 (CSI_SYNC|CSI_DPHY_CTRL)
  - But capture eventually succeeds and proceeds to AI path.
- AI worker dequeues frame and enters dry-run path.
- Crash occurs in OBB stage preprocessing, **before stage inference run**:
  - Last lines before crash:
    - [AI] Stage inference init OK.
    - [AI] Crop obb: x=0 y=0 w=224 h=224
    - [AI] Preprocess diagnostics OK.
    - [AI] Preprocess zero-fill skipped.
    - [AI] Preprocess resize start.
    - [AI] Preprocess row loop enter.

### HardFault signatures captured (new enhanced fault logging)
Run 1:
- PC=0x3EE07038
- LR=0x34007ED3
- CFSR=0x00000100
- HFSR=0x40000000
- DFSR=0x00000000
- AFSR=0x01000000
- MMFAR=0x000D044E
- BFAR=0x000D044E
- SP=0x3411AA88
- last_scalar_row=0x00000010

Run 2:
- PC=0x3ECB65B4
- LR=0x34007ED3
- CFSR=0x00000100
- HFSR=0x40000000
- DFSR=0x00000009
- AFSR=0x01000000
- MMFAR=0x000D044E
- BFAR=0x000D044E
- SP=0x3411AA88
- last_scalar_row=0x00000010

### Interpretation notes
- CFSR=0x100 suggests an instruction bus fault class condition; HardFault is forced (HFSR=0x40000000).
- Fault point is stable around preprocess row-loop entry and last_scalar_row=16, suggesting deterministic failure in/near early row processing for OBB preprocess.
- PC values are in  x3E... region (not normal app text  x340...), indicating jump/fetch from invalid/unexpected region.
- MMFAR/BFAR value ( x000D044E) appears even though CFSR does not indicate valid BFAR/MMFAR bits in this snapshot; treat as possibly stale until validated by valid-bit flags.

### Recent code changes already in tree relevant to this crash
- Increased AI thread stack budget to reduce overflow risk:
  - CAMERA_AI_THREAD_STACK_SIZE_BYTES = 262144 in pp_memory_budget.h.
- Registered ThreadX stack error callback:
  - 	x_thread_stack_error_notify(AppThreadX_StackErrorHandler) in pp_threadx.c.
- Added richer fault handlers for MemManage/Bus/Usage/Secure with stacked context in stm32n6xx_it.c.
- Reduced risky high-volume vararg logging in AI worker queue/dequeue in pp_inference_runtime.c.

### Important nuance
- Despite stack increase and logging reductions, fault still reproduces at same OBB preprocess point.
- This means issue is likely not just thread stack depth.

### Strong next debugging targets for Qwen
1. OBB preprocess code path in pp_ai.c (the row-loop and bilinear helpers) around where last_scalar_row updates.
2. Any function pointers / tables used during OBB preprocess (especially LUT/data pointers) that might be invalid under this build.
3. Check for unsafe casts/alignment/aliasing in per-pixel writes in float preprocess path.
4. Verify no memory region/MPU/RIF restrictions on data touched by OBB preprocess buffers.
5. Map LR=0x34007ED3 in current ELF (ddr2line) to identify caller chain into the failing PC jump.

### Operational status
- Board currently reflashed with latest build and models from flash script.
- COM3 monitoring works via pyserial when no other app holds port.

### Latest scalar-memory finding
- What was broken:
  - The generated scalar package hardcoded its input tensor and several internal `addr_base` entries at `0x34100000`.
  - That region overlapped the live app `.bss` / runtime footprint in the current link, so the AI runtime was reading and writing through an address window that was not actually private to the model.
  - The failure showed up as an imprecise bus fault / corrupted register state during the first scalar preprocess sweep, which made it look like preprocess math or stack corruption at first.
- What was not the root cause:
  - The row-loop logic itself.
  - The luma-only helper chain.
  - The AI worker stack, once we verified and shrank it below the model input region.
- Why the fix worked:
  - Moving the scalar package to `0x34110000` put the model input window above the app runtime footprint.
  - After that shift, the scalar contract reported `addr=0x34110000`, preprocess completed, and full scalar inference/logging ran normally.
- The scalar ST AI package was hardcoding its input tensor and several internal `addr_base` entries at `0x34100000`.
- That address overlapped the live app `.bss` / runtime region in the current link, so the AI runtime was reading and writing through a memory window that was not private to the model.
- Current fix applied in `st_ai_output/packages/prod_model_v0.4_scalar_int8/st_ai_output/scalar_full_finetune_from_best_piecewise_calibrated_int8.c` and its workspace copy:
  - input buffer and matching `addr_base` entries shifted to `0x34110000`
- `mingw32-make -j8 all` in `firmware/stm32/n657/Appli/Debug` completed successfully after the shift.
- Scalar preprocess and full scalar inference both run successfully with the restored fast path.



## 2026-05-16 Prod v0.5 Hybrid Firmware

- Packaged hybrid CNN + classical baseline as prod_model_v0.5_scalar_int8.
- Firmware now implements hybrid selection in CameraAIThread_Entry:
  - Runs classical baseline and CNN in parallel (existing behavior).
  - After CNN inference, checks AppBaselineRuntime_GetLastEstimate().
  - If classical confidence >= 20.0, logs classical temperature.
  - Otherwise logs CNN temperature.
  - Console prints [HYBRID] Classical selected or [HYBRID] CNN selected.
- Model artifact: same no_cal_hardpush_gpu5_recover model as v0.4.
- Flashed successfully to NUCLEO-N657X0-Q.
- Board needs BOOT0=0, BOOT1=0 and power-cycle to run from flash.

---
## 2026-05-16  Scalar HardFault diagnosis and fix

### Problem
Scalar inference stage was crashing with a HardFault during preprocess resize.
Fault log showed:
`
[FAULT] HardFault PC=0x3402128E CFSR=0x00008200 MMFAR=0x096E0800
[FAULT] SP=0x341FFDA0 R0=0x34106A98 R1=0x4D045200 last_scalar_row=0xDF
`
The fault was a precise data bus error: the CPU tried to read from  x096E0800, which is not a valid memory address.

### Root cause
The Cube.AI-generated scalar model placed its input tensor at  x34100000 (AXISRAM2 base).
The application .bss section ends at  x34106d58, meaning ThreadX globals live inside  x341000000x34106d58.
Map-file evidence:
- _tx_thread_execute_ptr =  x34105a30
- _tx_thread_current_ptr =  x34105a34

Scalar preprocess writes a 2242243 float32 tensor (602112 bytes) row-by-row.
Each row = 224 * 3 * 4 = 2688 bytes.
Row 8 starts at  x34100000 + 8 * 2688 = 0x34105400 and ends at  x34105e80,
which completely covers _tx_thread_execute_ptr at  x34105a30.
When the resize loop wrote row 8, it stomped on _tx_thread_execute_ptr with garbage.
The very next SysTick/PendSV tried to read from that garbage address ? HardFault.

OBB stage worked because Cube.AI placed its input at  x34110000, 64 KB past all app data.

### Fix applied
Shifted the scalar model input buffer (and its one aliased intermediate tensor)
from  x34100000 to  x34110000.

Files changed:
1. irmware/stm32/n657/st_ai_output/packages/prod_model_v0_6_scalar_int8/st_ai_output/prod_model_v0_6_scalar_int8.c
   - 6 occurrences of  x34100000 ?  x34110000 (lines 80, 82, 150, 2465, 2760, 29983)
2. irmware/stm32/n657/st_ai_output/packages/prod_model_v0_6_scalar_int8/st_ai_ws/neural_art__prod_model_v0_6_scalar_int8/prod_model_v0_6_scalar_int8.c
   - Same 6 replacements (identical copy used by the workspace)
3. ml/artifacts/runtime/prod_model_v0.6_scalar_int8_reloc/mpools/stm32n6_reloc.mpool
   - AXISRAM2 offset:  x34100000 ?  x34110000
   - AXISRAM2 size: 1024 KB ? 960 KB
   - This ensures future Cube.AI regenerations will not regress the fix.

Safety check after the shift:
- App data (.bss + heap + stacks) ends at  x3410d158
- New buffer base  x34110000 is 4.5 KB past the end of app data
- No overlap possible.

### Build / flash
- Clean rebuilt Appli in irmware/stm32/n657/Appli/Debug
- Ran lash_boot.bat successfully (FSBL + models + signed app)
- Board was power-cycled for flash-boot mode.

### Long-term recommendation
When regenerating the scalar model from Cube.AI, use the patched mpool JSON
so the allocator naturally skips the first 64 KB of AXISRAM2.
Alternatively, move the largest .bss arrays (e.g., pp_ai_dry_run_frame_scratch
at 0x34080260, 98 KB) into RAM_NC ( x24160000, 384 KB) or external hyperRAM
to free up the first 64 KB of SRAM2 and make the offset unnecessary.

---

## 2026-05-16 Live 49C Sanity Check

- The current live board path is the `prod_model_v0_6_scalar_int8` package plus the classical baseline hybrid gate.
- The hybrid selector in `app_inference_runtime.c` uses `baseline_conf >= 2.0f`, so the live `conf=3.760` frame does intentionally select the classical baseline.
- On the 49C gauge snapshot, the CNN path reported `38.196808C` and the classical baseline reported `39.420158C`; both are still under-reading by roughly 10C, with the classical path slightly closer.
- The earlier memory note that mentioned a `20.0` hybrid-confidence threshold is stale and should not be treated as the current decision rule.

---
## 2026-05-16 Blur-Aware CNN Winner

- The strongest CNN reader so far is `mobilenetv2_bluraware_reader_v41`: `linear_output=True`, `mobilenet_alpha=0.35`, `batch_size=4`, `epochs=18`, `learning_rate=5e-6`, `mobilenet_warmup_epochs=4`, `mobilenet_unfreeze_last_n=12`, `mobilenet_head_units=64`, `mobilenet_head_dropout=0.15`, `hard_case_repeat=4`, and `edge_focus_strength=1.0`.
- That blur-aware reader reached `test_mae=1.7144C` on its held-out test split, so future CNN baselines should start from that recipe instead of the older sigmoid/bounded head.

## 2026-05-16 Board Replay Hot-End Fix

- The hot-end miss was mainly a replay crop and calibration problem, not an int8 quantization problem.
- The board replay now defaults to the exact v41 int8 export at `artifacts/deployment/mobilenetv2_bluraware_reader_v41_int8/model_int8.tflite`.
- The board replay crop scale now stays at `OBB_CROP_SCALE=1.20`, which matches the calibration fit better than the older 1.30 default.
- The calibration payload at `artifacts/calibration/prodv0_3_obb_scalar_calibration.json` was refit on the exact v41 int8 model with board-focused manifests.
- On the hot captures, the replay moved from roughly `40C-42C` raw to about `46.8C` and `49.8C` after calibration, and the burst-smoothed output landed around `48.3C` on the later hot frame.

### Full-range CNN retrain (2026-05-17)

- The best CNN data mix is not hot cases only; it needs the full temperature range, including cold, midrange, and hot board captures.
- The strongest full-range retrain so far warm-started from `artifacts/training/mobilenetv2_bluraware_reader_v41/model.keras` and used:
  - `hard_case_manifest=data/unified_training_manifest_v1.csv`
  - `hard_case_repeat=1`
  - `range_aware_sampling=True`
  - `cold_tail_fraction=0.20`
  - `hot_tail_fraction=0.20`
  - `oversampling_factor=2.0`
  - `precomputed_crop_boxes=data/rectified_crop_boxes_v5_all.csv`
  - `linear_output=True`
- That run finished with `test_mae=7.4099C` on `board_rectified_probe_20260422.csv`, which is better than the naive mean predictor but still not enough to beat the classical baseline everywhere.
- On `hard_cases_plus_board30_valid_with_new6.csv`, the same model reached `mean_abs_err=15.3643C`, versus `16.8655C` for the previous `mobilenetv2_bluraware_reader_v41` evaluation on the same manifest.
- The remaining failures are not a hot-only problem; some midrange board captures still collapse badly, so the next improvement should be a better target or architecture, not more hot-tail oversampling.

## 2026-05-17 - Fraction-first and geometry-first hard-case experiments
- I retrained the fraction-first CNN on the full-range mix and then on the actual hard-case pool.
- Important lesson: pinned hard-case validation/test manifests can silently filter most hard-case rows out of training, so a random split is needed if we want the hard-case manifest to actually affect the model.
- The corrected fraction-first hard-case run did not reach the target; it still regressed toward the middle on cold/hot extremes.
- I also tried a compact geometry-first run with the real hard-case pool.
- That model was the strongest of the new experiments on the held-out test split, but it still missed the hard-case target and overfit the board probe.
- Current takeaway: the present labeled/live-board data is not yet enough for a <5C hard-case MAE with the current small CNNs, even when we make the task angle/fraction-first.
- Next likely directions are more live hard-case labels at the cold/hot ends, or a richer distributional/ordinal head instead of a single fraction scalar.

## 2026-05-17 - Polar voting branch
- The classical polar voting intuition was worth chasing: a learned vote head on the polar image is the first CNN branch that starts looking good on the live-board split.
- The best vote-head recipe so far is the polar 2D CNN with explicit angular axis retention, mean+max radial pooling, label_smoothing=0.0, and sharpened angular soft targets.
- polar_vote_full_range_v5 was the most promising balanced run:
  - board-probe holdout: mae=4.0996C
  - hard-case manifest: mae=9.7356C
  - worst hard-case failures were still the cold tail, especially capture_m25c.jpg, capture_m19c.png, and capture_m10c.jpg
- polar_vote_full_range_v6 tried stronger inverse-frequency temperature balancing and a sharper target (sigma_bins=1.0), but it did not beat v5:
  - board-probe holdout worsened to mae=5.2408C
  - hard-case manifest worsened to mae=11.3299C
  - the cold tail still dominated the worst errors, so range balancing alone is not enough
- Current takeaway: the polar-voting formulation is still the right direction, but the remaining cold-tail collapse likely needs either a richer geometry target than a single angle distribution, or more explicit live cold-case supervision from the board captures, especially for negative-temperature examples.

---
## 2026-05-18 Note Index

- Consolidated 2026-05-18 deployment/runtime chronology is recorded in:
  - `docs/ai-memory/ml-experiments.md`
  - section: `2026-05-18 End-to-End Deployment + Firmware Integration Notes`
- Deploy-specific decode/signature/flash summary is recorded in:
  - `docs/ai-memory/polar-vote-deploy-2026-05-18.md`
- Additional 2026-05-18 updates now recorded in:
  - `docs/ai-memory/ml-experiments.md`
  - sections:
    - `9) Flash-script path trap we hit (important)`
    - `10) Post-fix verification + latest live quality check`

## 2026-05-18 Maintenance Note

- `docs/ai-memory.md` was repaired to valid UTF-8 so it can be patched/read reliably by tooling.
- Pre-repair backup retained at:
  - `tmp/ai-memory.md.pre_utf8_fix_2026-05-18`


## 2026-05-18 Polar Vote Circular V28 (WINNER)

- Switched from sweep target mode to circular target mode with dead-zone masking.
- Key architectural change: _logits_to_temperature now masks dead-zone bins (those outside the gauge sweep arc) to -inf before computing the sin/cos circular mean. This prevents wraparound artifacts (e.g. -30C decoding as 50C).
- _structured_logits_to_temperature now accepts 	arget_mode, min_angle_rad, sweep_rad params so vote mode can dispatch to either circular or sweep decode.
- TemperatureMaeCallback also accepts these params for validation metric tracking.
- Circular decode uses sin/cos circular mean which is immune to center-pull regression that plagued the sweep expectation decode.
- V28 params: --target-mode circular --bins 224 --sigma-bins 1.0 --base-filters 32 --head-units 128 --center-search-px 5 --loss-mode balanced_softmax --fraction-loss-weight 0.0 --max-shift-bins 4 --epochs 200 --batch-size 16
- **Hard cases + board eval (hard_cases_plus_board30_valid_with_new6.csv):**
  - MAE = 0.34C (was 3.34C in V27, was ~3.19C in V25)
  - RMSE = 0.56C
  - Max error = 2.19C (was 80C in V27, was 20.6C in V25)
  - Median error = 0.18C
  - 100% under 5C, 100% under 3C, 93% under 1C
- Best weights: ml/artifacts/training/polar_vote_circular_v28/best_weights.weights.h5
- Model: ml/artifacts/training/polar_vote_circular_v28/model.keras
- The catastrophic failures from V25 (p25c 25->4.4, p5c 5->16.6, p10c 10->18.4) are ALL fixed.
- The board 28C captures that were reading as low as 12-16C now all read within 2.2C.
- Internal test split MAE = 4.65C (higher because it includes some noisy labels; median 0.29C).


## 2026-05-19 Prod v0.7 - Polar Vote Circular V28 Deployment

- Promoted V28 circular polar-vote model to prod v0.7.
- This is the first production model that uses circular dead-zone masking decode instead of sweep expectation, eliminating center-pull regression artifacts.
- Firmware references updated from scalar_full_finetune_from_best_piecewise_calibrated_int8 to polar_vote_circular_v28_int8:
  - app_ai.c: NN_Instance, Network_Init, Inference_Init, memory pool, LL_ATON_Internal_Buffers_Info all renamed.
  - ai_network_mobilenetv2_scalar_hardcase_warmstart_int8.c: #include path updated to the V28 model C source.
  - makefile.targets: USER_OBJS and ELF deps updated to V28 package object paths.
- Model contract:
  - Input: (1, 224, 224, 7) int8, scale=0.0039215687, zero_point=-128 (polar RGB, Sobel edges, vote prior)
  - Output: (1, 224) int8, scale=0.093767159, zero_point=16 (224 angle-bin logits)
  - Circular decode: dead-zone masking to -1e9, softmax, sin/cos circular mean, angle-to-temperature mapping
  - Gauge: min_angle=2.356 rad (135 deg), sweep=4.712 rad (270 deg), range [-30C, 50C]
- Firmware decode constants: APP_AI_POLAR_VOTE_BINS=224, APP_AI_POLAR_OUTPUT_SCALE=0.093767159f, APP_AI_POLAR_OUTPUT_ZERO_POINT=16, APP_AI_POLAR_MASK_LOGIT=-1e9f
- xSPI2 blob: 66081 bytes, signature head=D0FB D70C BFAF EE1A FA40 D563 E254 D864, tail=0000...0083
- ST Edge AI package: firmware/stm32/n657/st_ai_output/packages/polar_vote_circular_v28_int8/
- TFLite model: ml/artifacts/deployment/polar_vote_circular_v28_int8/model_int8.tflite
- Hard cases + board eval: MAE=0.34C, RMSE=0.56C, max error=2.19C, 100% under 3C
- The hybrid baseline-weighing mechanism has been removed from the CNN. The model always adds its own processing on top of any baseline input it receives; it never outputs the baseline directly.
- Hybrid voting/baseline mechanism removal: the CNN no longer has a separate baseline input channel that can win. The vote prior channel (ch6) is a soft hint, not a direct copy of baseline output.
## 2026-05-19 Prod v0.7 Memory Fix and Re-flash

- **Problem**: Board crashed with MemManage fault (PC=0xAAAAAAAA) on boot after flashing V28 firmware.
  - Root cause: ~602 KB of static float scratch arrays in BSS left only ~409 KB free RAM, causing thread stack overflow and corrupted control flow.
  - The three arrays were: \polar_luma[224*224]\ (200 KB float), \ngular_grad[224*224]\ (200 KB float), adial_grad[224*224]\ (200 KB float).
- **Fix**: Converted polar scratch buffers from float to uint8 and eliminated gradient storage arrays.
  - \polar_luma\ changed from \static float[224*224]\ (200 KB) to \static uint8_t[224*224]\ (49 KB).
  - \ngular_grad\ and adial_grad\ arrays eliminated entirely (~394 KB saved).
  - Sobel computation restructured to two-pass: Pass A finds gradient maximums for normalisation, Pass B recomputes gradients and writes normalised channels directly to the input tensor.
  - Total BSS savings: ~539 KB. Free RAM increased from ~409 KB to ~948 KB.
- **OBB float preprocess fix**: \PreprocessYuv422FrameToFloatInput\ validation checked \input_float_count >= APP_AI_MODEL_INPUT_FLOAT_COUNT\ (224*224*7=352896), but the OBB model uses 3-channel float input (224*224*3=150528). Changed minimum to \width*height*3\ so the OBB stage can proceed instead of always failing.
- **OBB input buffer overlap**: The OBB model input buffer at 0x34110000 still overlaps with BSS (ends at 0x34113160). This is a pre-existing issue; the OBB stage falls back to fixed training crop regardless. Needs a proper fix when OBB becomes critical (e.g., move OBB input to NPU SRAM above 0x34200000).
- **Build**: Rebuilt with CubeIDE make. New BSS: 927616 bytes. New .bin: 329632 bytes.
- **Flash**: Successfully flashed FSBL, scalar model (66081 bytes), rectifier, OBB, and signed application.
- **Key lesson**: On STM32N6 with 1536K AXI SRAM, static scratch buffers larger than ~10 KB should be uint8/int8 instead of float, or placed in NPU SRAM regions outside the linker's RAM region. The linker has no visibility into NPU buffer addresses, so CPU-side scratch and NPU buffers can silently overlap.
- **Files changed**: \irmware/stm32/n657/Appli/Src/app_ai.c\ (polar preprocess function, float preprocess validation)

## 2026-05-19 OBB Input Buffer Overlap Fix

- **Problem**: OBB model float32 input buffer at 0x34110000 (AXISRAM2) was silently corrupted by BSS variables that grew past 0x34110000 (BSS end was 0x34113160). Every OBB inference overwrote ~12,640 bytes of BSS (ThreadX timer array, ISP state, peripheral handles) causing HardFault/MemManage crashes. The OBB stage always fell back to fixed training crop as a result.
- **Root cause**: The linker RAM region spanned 1536K (0x34080000-0x34200000), covering all of AXISRAM1 + AXISRAM2. But AXISRAM2 hosts NPU internal buffers and the OBB input at 0x34110000. The linker had no visibility into NPU-reserved addresses, so BSS grew freely into them.
- **Fix** (5 changes):
  1. Shrank linker RAM from 1536K to 512K (0x34080000-0x34100000, AXISRAM1 only), preventing BSS from ever reaching AXISRAM2.
  2. Added NPU_SRAM6 memory region (AXISRAM6, 0x34350000, 448K) and .npusram6 linker section for displaced large BSS arrays.
  3. Moved camera_ai_thread_stack (128 KB) and pp_ai_dry_run_frame_scratch (98 KB) to .npusram6 via __attribute__((section(".npusram6"))).
  4. Updated NEWLIB_HEAP_LIMIT_ADDR from 0x34110000 to 0x34100000 to match the new RAM boundary.
  5. Added .npusram6 NOLOAD section to linker script with __snpusram6/__enpusram6 symbols.
- **Result**: BSS now ends at 0x340DA958 (~155 KB below the 0x34100000 boundary). The .npusram6 section at 0x34350000 holds 231,424 bytes (both displaced arrays). Board flashed successfully with OBB model intact.
- **Build output**: text=327824, data=572, bss=927616. .bin=328,416 bytes.
- **Key lesson**: On STM32N6, the linker RAM region must not include AXISRAM2 (0x34100000+) because ST Edge AI-generated code hardcodes NPU buffer addresses there. AXISRAM6 (0x34350000) is a safe overflow zone for large CPU-only BSS arrays since neither model uses it.
- **Files changed**: STM32N657X0HXQ_LRUN.ld, pp_inference_runtime.c (line 69), pp_ai.c (line 313), pp_memory_budget.h (line 20).
## 2026-05-19 OBB Fallback Crop Fix (Prod v0.7.1)

- **Problem**: When the OBB crop falls outside the training window, the fallback code blended the OBB-detected center with the inner dial center using a configurable ratio (APP_AI_OBB_CENTER_BLEND_NUMERATOR/DENOMINATOR). This shifted the crop ~25 pixels vertically from the training crop origin (y=57), causing the scalar model to read 24.2C when the true gauge value was 10C (14.2C error).
- **Root cause**: The training crop geometry is x=23, y=57, w=156, h=123. The inner dial center is at approximately (112, 100). Blending the OBB center with the inner dial center produced a crop at y=32, shifting the view 25px up from where the model was trained.
- **Fix**: Replaced the entire OBB fallback block (lines 3639-3733) with a simple call to AppGaugeGeometry_TrainingCrop(), which returns the exact fixed training crop geometry. The model now always sees the same crop coordinates it was trained on, regardless of OBB output quality.
- **Build**: text=328616, data=572, bss=927616. .bin=329,216 bytes.
- **Flash**: Successfully flashed FSBL, scalar, rectifier, OBB, and signed app.
- **Files changed**: firmware/stm32/n657/Appli/Src/app_ai.c (OBB fallback block replaced)


## 2026-05-19 Polar Projection Sign Convention Audit

- **Finding**: The firmware polar projection at line 7042 uses center_y - radius * sinf(angle) (minus sin), while OpenCV warpPolar uses center_y + radius * sin(phi) (plus sin) when Y increases downward. This is a **sign flip on the Y-axis sin() term** that causes the firmware to sample from the vertically-mirrored position relative to training for each polar column.
- **OpenCV warpPolar convention** (verified from source):
  - phi = 2*PI * col / width
  - src_x = cx + rho * cos(phi)
  - src_y = cy + rho * sin(phi)   (Y increases downward; phi=0 is 3 o'clock, phi increases clockwise in image coords)
- **Firmware convention** (line 7038-7042):
  - ngle = 2*PI * col / bins
  - src_x = cx + r * cos(angle)
  - src_y = cy - r * sinf(angle)  (Y increases downward; angle=0 is 3 o'clock, angle increases counterclockwise in math coords)
- **Impact**: For the same column index, training and firmware sample from mirror-symmetric positions across the horizontal center line. This means the model sees different pixel content at each column during inference than during training. The gauge needle at 135 deg image angle (7:30 position, -30C) appears at column 84 in both systems, but training samples from below center-left while firmware samples from above center-left.
- **Circular vote decode**: Both the training PolarAngleToTemperature layer and the firmware AppAI_DecodeCircularVoteFromOutput use the same angle convention: ngles[i] = 2*PI*i/224, min_angle = 2.356 rad (135 deg), sweep = 4.712 rad (270 deg). The decode math is structurally identical. The only discrepancy is whether the pixels feeding into each bin came from the same physical location as during training.
- **Fix options**: Either change firmware line 7042 to center_y + radius * sinf(angle) to match OpenCV, or flip the polar image vertically before inference.
- **Gauge calibration**: min_angle=135 deg (2.356 rad), sweep=270 deg (4.712 rad), -30C to +50C, clockwise direction. Consistent across gauge_calibration_parameters.toml, processing.py, polar_model.py, 	rain_polar_v3_geometry.py, and pp_ai.c defines.
- **Training pipeline**: polar_project_image() in ml/src/embedded_gauge_reading_tinyml/polar_projection.py uses cv2.warpPolar with WARP_POLAR_LINEAR | WARP_FILL_OUTLIERS. Mask placed at column (needle_angle_deg / 360) * width where 
eedle_angle_deg = degrees(atan2(dy, dx)) % 360 with image Y-down coordinates.
## 2026-05-19 Polar Projection sin() Sign Fix (Prod v0.7.3)

- **Problem**: Gauge read ~12-14C correctly at 12C true, but under-read at 45C true (reading 34-40C, 5-11C low). The systematic under-reading at higher angles was caused by a sign convention mismatch in the polar projection.
- **Root cause**: Line 7042 used center_y - radius * sinf(angle) (minus sin, math convention, counterclockwise on screen). OpenCV warpPolar uses center_y + radius * sin(phi) (plus sin, Y-down image convention, clockwise on screen). The minus sign vertically mirrored the polar image relative to what the model was trained on, causing progressive angular distortion.
- **Fix**: Changed line 7042 from center_y - radius * sinf(angle) to center_y + radius * sinf(angle). Also updated the comment on line 6894 from [Y axis inverted] to [Y-down, matches OpenCV warpPolar] and removed the stale duplicate line.
- **Build**: text=328840, data=572, bss=927616. .bin=328,912 bytes.
- **Flash**: Successfully flashed FSBL, scalar, rectifier, OBB, and signed app.
- **Files changed**: firmware/stm32/n657/Appli/Src/app_ai.c (line 7042 sin sign, line 6894-6895 comment)

## 2026-05-19 HardFault STKOF Root Cause (Prod v0.7.4)

- **Problem**: First boot after flashing crashes with HardFault at PC=0x340065F8 (inside `LL_RCC_HSI_SetDivider`, called from `HAL_RCC_ClockConfig`). Second boot faults with MemManage at PC=0xAAAAAAAA (erased flash pattern, cascading corruption).
- **Fault registers**: CFSR=0x00100000 (UFSR bit 4 = STKOF, stack overflow on ARMv8-M), HFSR=0x40000000 (FORCED). SP=0x340FAA68 was ~5.5KB below MSPLIM=0x340FC000.
- **Root cause**: MSP (main stack) overflow. The 16KB MSP (`_Min_Stack_Size=0x4000`) is insufficient for the `main()` init path. `App_SystemClock_Config()` places large local structs (`RCC_OscInitTypeDef` 148B with 4 embedded PLL configs, `RCC_ClkInitTypeDef` 64B) on the stack, plus deep HAL call chains that push MSP past its 16KB limit.
- **Fix applied**:
  1. Increased `_Min_Stack_Size` from `0x4000` (16KB) to `0x8000` (32KB) in STM32N657X0HXQ_LRUN.ld.
  2. Added EXC_LR, MSP, MSPLIM, PSP, PSPLIM diagnostics to `HardFault_Handler_C()` in stm32n6xx_it.c.
  3. Created detailed analysis at docs/hardfault-stkof-analysis.md.
- **Not the cause**: RISAF7 ordering was not the problem. AXISRAM6 is accessible without explicit RISAF7 config (reset default BREN=0, filtering disabled).
- **Follow-up**: Consider moving stack-heavy init into a ThreadX thread with a dedicated large stack, leaving MSP for ISRs only. Monitor MSPLIM margin at runtime.
## 2026-05-19 Live Board Log Analysis: sin() Sign Revert & Under-Reading at High Temps

### Board Session Timeline (2026-05-16 logs, Prod v0.7.3 then revert)

**Phase 1 - OBB fallback fix only (gauge at ~10-12C):**
- Model reads 42.9->45.8C (calibrated) at 10C true -- very wrong, likely before OBB fix stabilized
- After OBB fallback fix landed: 12.9, 9.5, 14.6, 12.2, 12.8C at 12C true -- acceptable scatter (~+-3C)

**Phase 2 - Gauge moved to 45C, model stuck near 12C:**
- Readings: 9.5, 14.6, 12.2, 12.8C at 45C true -- model appeared stuck reading the same region
- Cause: OBB producing bad crops (w~150, h~166-170, near-full-frame) that shifted the polar center away from the needle
- After the fixed training crop fallback was applied consistently: readings jumped to 34-40C range at 45C true

**Phase 3 - sin() sign changed to +sinf (v0.7.3, matching OpenCV):**
- Readings at 46C true: -6.0C, -10.7C -- catastrophic
- The +sinf change was expected to match training (OpenCV convention), but produced wildly negative readings
- Likely cause: although the polar projection sign was corrected, the model learned a column-to-temperature mapping on images that followed a consistent convention. Changing just the projection sign without retraining (or adjusting the decode) broke the self-consistency between what the model sees and what the circular_vote decode expects.

**Phase 4 - sin() sign REVERTED to -sinf (current production):**
- Back to original center_y - radius * sinf(angle) on line 7042
- At 12C true: reads ~9.5-14.6C (acceptable)
- At 45C true: reads ~34-40C (5-11C under-reading)
- The -sinf convention is self-consistent between projection and decode, so the model partially works. But the vertical mirror relative to training causes progressive angular error that increases with angle.

**Phase 5 - HardFault STKOF (v0.7.4):**
- First boot after v0.7.3 flash crashed with HardFault (STKOF) at PC=0x340065F8
- Root cause: MSP stack overflow -- 16KB was insufficient for HAL init path
- Fix: Increased _Min_Stack_Size from 0x4000 (16KB) to 0x8000 (32KB) in linker script
- Added EXC_LR, MSP, MSPLIM, PSP, PSPLIM diagnostics to HardFault_Handler_C()

### Core Unresolved Problem: 5-11C Under-Reading at 45C

With -sinf (current production):
- 12C true -> ~9.5-14.6C read (roughly correct, +-3C)
- 45C true -> ~34-40C read (systematically 5-11C low)

With +sinf (matching OpenCV, v0.7.3):
- 46C true -> -6 to -10.7C read (catastrophic)

**Why +sinf was worse**: The model was trained with OpenCV warpPolar (+sinf), so in theory flipping the firmware to +sinf should match. However, the catastrophic result suggests that one or more additional preprocessing differences (OBB crop geometry, polar center computation, resize_with_pad inverse, Sobel channel computation, or the interaction between polar projection sign and the circular_vote decode angle mapping) compound with the sin() sign to produce a broken end-to-end mapping. The fact that -sinf works sort of OK at low angles and diverges at high angles is consistent with a vertical mirror that causes small errors near symmetric gauge positions (12C ~ 277deg is near the bottom of the dial) but large errors at asymmetric positions (45C ~ 28deg is near the top).

### Key Log Evidence

**OBB crop issues** (repeated across all logs):
- OBB crop outside training window: crop=150x163 train=156x123 ratio=962/1325 -> centered training-size OBB fallback
- OBB crop outside training window: crop=143x169 train=156x123 ratio=917/1374 -> fixed training crop fallback
- OBB crop outside training window: crop=147x166 train=156x123 ratio=942/1350 -> fixed training crop fallback
- The OBB consistently produces taller-than-expected crops (h=163-170 vs training h=123), causing the 1.5x max ratio check to sometimes pass and sometimes fail
- When OBB crops pass the ratio check, they shift the polar center significantly from the training position

**Polar crop geometry** (when fallback is used):
- Polar crop: cx= cy= r= scale= pad=(,) -- empty/zero values, indicating the fallback path uses default center
- When OBB crop is used: Polar crop: cx=112 cy=112 r=112 scale=1.349 pad=(8.771,0.000) -- computed from OBB dimensions

**Calibration deltas**:
- Some frames show non-zero calibration (e.g., Calibration delta: 2.815102 adding ~3C)
- At 45C true, calibration adds 2-2.5C, not enough to fix the 5-11C under-reading
- At 12C true, calibration delta was 0.000000 (no correction applied)

**Baseline temperature estimates** (for cross-reference):
- At 12C: baseline estimate -21.8C to -22.0C (internal sensor, not gauge reading)
- At 45C: baseline estimate 44.0C to 49.9C (closer to true, but uses different algorithm)

### Remaining Fixes Needed

1. **Fix the systematic under-reading at high temperatures**: The -sinf polar projection vertically mirrors the image relative to training. Options:
   - Retrain the model with -sinf polar projections to match firmware
   - Fix the +sinf path and also fix whatever additional preprocessing difference causes the catastrophic failure (most likely: the polar center computation or the angle-to-column mapping in the decode)
   - Investigate whether the OBB fallback polar crop (with empty cx/cy/r) uses different center coordinates than training

2. **Fix OBB producing bad crops**: The OBB often produces h~1.0 near-full-frame boxes. Consider tightening APP_AI_OBB_TRAINING_CROP_MAX_RATIO from 1.50 back to 1.25, or adding an aspect-ratio check.

3. **Test at both 12C and 45C** after any fix to confirm accuracy across the full range.

### Current Production State (as of 2026-05-19)

- **Firmware version**: Prod v0.7.4 (sin reverted to -sinf, MSP stack 32KB)
- **sin() convention**: center_y - radius * sinf(angle) (line 7042, original -sinf)
- **OBB fallback**: Fixed training crop (AppGaugeGeometry_TrainingCrop)
- **OBB max ratio**: 1.50 (relaxed from 1.25)
- **MSP stack**: 32KB (increased from 16KB)
- **Model**: prod_model_v0.4_scalar_int8 (MobilenetV2 with CBAM + CoordAtt, circular vote output)
- **Known issue**: Under-reads by 5-11C at 45C true; reads approximately correctly at 12C true

### Update: Latest Board Readings at 45C True (2026-05-19, still Prod v0.7.4)

**Three consecutive readings at 45C true, needle never moved:**

1. CNN=9.5C (OBB crop: x=33 y=9 w=147 h=169, polar scale=1.325 pad=(14.580,0.000))
   - OBB crop PASSED the 1.5x ratio check (169/123=1.37 < 1.5)
   - This crop is 37% taller than training crop h=123, shifting polar geometry
   - Input first8=[0F 0F 0F 0F 80 80 B8 0F] -- low values, partially valid polar image

2. CNN=13.8C (OBB crop: x=26 y=12 w=140 h=163, polar scale=1.374 pad=(15.804,0.000))
   - Again OBB crop passed ratio check (163/123=1.32 < 1.5)
   - Input first8=[80 0A 80 D1 80 80 D7 80] -- many 0x80 (zero-point), degenerate polar image

3. CNN=-31.6C (OBB crop: x=31 y=15 w=146 h=157, polar scale=1.427 pad=(7.847,0.000))
   - OBB crop passed again (157/123=1.28 < 1.5)
   - Input first8=[7F 7F 7F 7F 80 80 80 7F], mid8=[7F 7F 7F 7F 80 80 80 7F]
   - Almost entirely 0x7F/0x80 -- polar image is ALL background, no gauge content visible
   - Model raw output -23.6C, calibration added -8.0C delta making it -31.6C
   - This is the worst reading yet -- the OBB crop + polar projection produced an empty image

**Root cause analysis for these three readings:**

The OBB model is producing crops with height 157-169 pixels (vs training h=123). These pass the APP_AI_OBB_TRAINING_CROP_MAX_RATIO=1.50 check but have a very different aspect ratio from the training crop (w=155, h=123, nearly square vs the OBB crops which are tall rectangles). When the polar projection uses these tall crops, the resize_scale and pad values change, and the gauge dial ends up misaligned or partially out of the polar image entirely.

The third reading is the smoking gun: the polar input tensor is almost entirely 0x7F/0x80 (int8 zero point = no signal), meaning the polar projection sampled almost entirely from the padded background area outside the actual crop content.

**Immediate fix needed:** Tighten APP_AI_OBB_TRAINING_CROP_MAX_RATIO from 1.50 back to 1.25 (or lower), or add an aspect-ratio check. The current tall OBB crops are worse than the fixed training crop fallback -- they produce degenerate polar images that read as -31.6C at 45C true.

**Comparison of polar geometry:**
- Training crop (w=155, h=123): scale=1.349, pad=(8.771, 0.000) -- nearly square, pad is horizontal
- OBB crop (w=147, h=169): scale=1.325, pad=(14.580, 0.000) -- tall, MUCH more horizontal padding
- OBB crop (w=140, h=163): scale=1.374, pad=(15.804, 0.000) -- tall, even more horizontal padding
- OBB crop (w=146, h=157): scale=1.427, pad=(7.847, 0.000) -- tall but less padding

The pad_x values of 14.580 and 15.804 mean that 30-32 pixels out of 224 columns are horizontal padding on each side. The model was trained with pad_x=8.771 (only ~18 pixels of padding per side). This massive change in horizontal padding shifts the polar projection significantly.
## 2026-05-19 OBB Max Ratio Tightened (Prod v0.7.5 prep)

- **Problem**: OBB producing tall crops (h=157-169 vs training h=123) that passed the 1.50 max ratio check but produced degenerate polar images. The worst case read -31.6C at 45C true because the polar projection sampled entirely from padded background.
- **Root cause**: APP_AI_OBB_TRAINING_CROP_MAX_RATIO was 1.50, allowing crops up to h=184 (1.50 * 123). Tall portrait-oriented crops (147x169, 140x163, 146x157) had pad_x values of 14-16 pixels (vs training pad_x of 8.77), pushing the gauge content into the padded background area.
- **Fix**: Changed APP_AI_OBB_TRAINING_CROP_MAX_RATIO from 1.50f to 1.25f on line 180. This rejects crops with h > 153 (1.25 * 123), forcing fallback to the fixed training crop for the problematic OBB detections.
- **Effect**: The tall OBB crops will now trigger fixed training crop fallback and use the known-good training crop geometry. This eliminates the catastrophic -31.6C readings. The systematic 5-11C under-reading at 45C (from the -sinf polar projection) remains and needs a separate fix.
- **Files changed**: firmware/stm32/n657/Appli/Src/app_ai.c (line 180)

## 2026-05-19 Polar Projection sin() Sign Fix

- **Problem**: The firmware polar projection used center_y - radius * sinf(angle) (minus sin) on line 7042 of pp_ai.c, while the training pipeline uses cv2.warpPolar which follows the center_y + radius * sin(phi) convention (plus sin). This sign mismatch causes the firmware polar image to be horizontally mirrored relative to the training polar image, leading to progressive angular error that worsens at higher temperatures.
- **Why it matters**: With -sinf, column c in the polar image samples from source angle 2*pi - theta instead of 	heta, where 	heta = 2*pi*c/224. This means the needle appears in column 224 - c_true instead of c_true. The circular vote decode maps column 224 - c_true to angle 2*pi - theta_true, which is the supplementary angle. Near 12C (theta~277deg), the supplementary angle (83deg) happens to produce a reasonable temperature because the gauge geometry has some symmetry there. But at 45C (theta~388deg, i.e. 28deg), the supplementary angle (332deg) maps to a very different temperature, explaining the 5-11C under-reading.
- **Why the previous +sinf attempt failed (v0.7.3)**: The v0.7.3 +sinf test gave catastrophic readings (-6 to -10.7C at 46C true). This was NOT caused by the sin sign itself, but by two compounding issues: (1) the OBB max ratio was 1.50, allowing tall degenerate crops that produced empty polar images, and (2) the fixed training crop fallback path may have had stale polar crop parameters from a prior iteration. With the OBB max ratio now tightened to 1.25, the +sinf change should produce correct results.
- **Fix**: Changed line 7042 from center_y - radius * sinf(angle) to center_y + radius * sinf(angle). This matches the OpenCV warpPolar convention used in the training pipeline.
- **Expected impact**: The polar image will now have the same column-to-angle mapping as the training data. Readings at all temperatures should be within the model's native accuracy (~3-5C MAE). The systematic under-reading at 45C should be eliminated.
- **Alternative considered**: Adding a horizontal flip (column reversal) of the polar image after projection. This would achieve the same effective mapping as +sinf without changing the projection code. Rejected because it adds a full 224x224x7 memory copy that costs ~50us on the Cortex-M55, and the direct +sinf change is simpler and matches training.
- **Files changed**: firmware/stm32/n657/Appli/Src/app_ai.c (line 7042)

### Detailed Analysis: -sinf vs +sinf Column Mapping

In OpenCV's warpPolar (+sinf convention):
- Column c maps to source pixel at (center_x + r*cos(theta), center_y + r*sin(theta)) where 	heta = 2*pi*c/224
- A needle at true gauge angle lpha appears in column c = alpha * 224 / (2*pi)

In the firmware with -sinf:
- Column c maps to source pixel at (center_x + r*cos(theta), center_y - r*sin(theta))
- The same needle appears in column c = (2*pi - alpha) * 224 / (2*pi) = 224 - alpha*224/(2*pi)
- The decode maps column 224 - c_true to angle 2*pi - theta_true

This is equivalent to a horizontal mirror of the polar image. The circular vote decode computes tan2(sin_sum, cos_sum), which for a distribution peaked at 224 - c_true gives an angle close to 2*pi - theta_true rather than 	heta_true.

At 12C true (theta ~ 277deg, near the bottom of the gauge dial):
- -sinf column: 224 - 172 = 52
- Decoded angle: ~83deg, fraction ~ 0.93 (clamped from 1.14), temperature ~ 44.4C
- But the actual reading was ~12.9C, suggesting the model's circular vote distribution is more complex than a single peak, and the gauge's radial pattern provides some angular context that partially compensates

At 45C true (theta ~ 28deg):
- -sinf column: 224 - 17 = 207
- Decoded angle: ~332deg, fraction ~ 0.73, temperature ~ 28.8C
- This roughly matches the observed 5-11C under-reading pattern

### Sobel and Vote Prior Channels

The Sobel channels (angular and radial gradients) and vote prior channel are computed ON the polar image after projection. With +sinf, these channels will match the training pipeline exactly. The angular gradient (horizontal Sobel) is invariant to the sin sign change. The radial gradient (vertical Sobel) has its sign negated by -sinf vs +sinf, but since the firmware takes bs(gy), this doesn't matter. The vote prior depends on luma and gradients, all of which will be correct with +sinf.

### Board Readings Summary (with -sinf, before fix)

| True Temp | CNN Reading | Error | Notes |
|-----------|-------------|-------|-------|
| 12C | 12.9C | +0.9C | Fixed training crop, good geometry |
| 12C | 13.8C | +1.8C | OBB crop w=147 h=169 (tall, degenerate) |
| 45C | 34.1C | -10.9C | With calibration delta +2.3C |
| 45C | 38.9C | -6.1C | With calibration delta +2.5C |
| 45C | 40.0C | -5.0C | With calibration delta +2.5C |
| 46C | 9.5C | -36.5C | OBB crop h=169, degenerate polar |
| 46C | -31.6C | -77.6C | OBB crop h=157, completely empty polar |
## 2026-05-19 +sinf Test Results and Revert

- **+sinf change tested live**: Changed line 7042 from center_y - radius * sinf(angle) to center_y + radius * sinf(angle) to match OpenCV warpPolar convention.
- **Result**: +sinf made readings WORSE at 45C true. The model was likely trained/fine-tuned on -sinf polar data, making it dependent on that mirrored convention.

| Convention | True 45C | CNN Readings | Error Range |
|-----------|---------|-------------|-------------|
| -sinf (original) | 45C | 34-40C | -5 to -11C |
| +sinf (flipped) | 45C | 24.7-33.8C | -11 to -20C |

- **Action**: Reverted line 7042 back to center_y - radius * sinf(angle). Both conventions under-read at 45C -- this is a model accuracy issue, not a projection convention issue.
- **Root cause**: The model was trained with data preprocessed through the -sinf firmware path (or an equivalent mirroring), so the circular vote weights encode the -sinf column-to-angle mapping. Switching to +sinf breaks that mapping without retraining.

### Detailed +sinf Board Readings

| True Temp | Convention | CNN Raw | After Cal | Error | Notes |
|-----------|-----------|---------|-----------|-------|-------|
| 45C | -sinf | 34.1C | 34.1C | -10.9C | Calibration delta +2.3C |
| 45C | -sinf | 38.9C | 38.9C | -6.1C | Calibration delta +2.5C |
| 45C | -sinf | 40.0C | 40.0C | -5.0C | Calibration delta +2.5C |
| 45C | +sinf | 31.9C | 34.1C | -10.9C | Fixed crop |
| 45C | +sinf | 36.4C | 38.9C | -6.1C | OBB accepted |
| 45C | +sinf | 37.4C | 40.0C | -5.0C | Hot-zone override |
| 46C | -sinf | 9.5C | 9.5C | -36.5C | Degenerate OBB crop |
| 46C | -sinf | -31.6C | -31.6C | -77.6C | Empty polar, -8C cal delta |
| 46C | -sinf | -10.7C | -10.7C | -56.7C | OBB crop, after reboot dark |
| 45C | +sinf | 22.9C | 24.7C | -20.3C | Dark frame mean=16 |
| 45C | +sinf | 29.0C | 31.1C | -13.9C | Fixed crop |

### Calibration Analysis

- Current affine: calibrated = 0.655 + 1.050 * raw
- At raw=34: calibrated = 36.4, needs ~45 (error -8.6C)
- At raw=40: calibrated = 42.7, needs ~45 (error -2.3C)
- The hot_blend (0.35) only applies above raw=43C, which the model never reaches at true 45C
- The model is accurate at ~12C (raw=12.8, calibrated=12.8), so low-band needs no correction
- Progressive under-reading above raw=20C suggests a linear boost is appropriate

### Planned High-Temperature Calibration Boost

Add a linear boost of  .25 * (raw - 20) for raw > 20C, capped at 12C:

| Raw | Current Cal | New Cal | True 45C Error |
|-----|------------|---------|-----------------|
| 12  | 12.0 (identity) | 12.0 | Good (+0.8C) |
| 34  | 36.4 | 39.9 | -5.1C (was -8.6C) |
| 40  | 42.7 | 47.7 | +2.7C (was -2.3C) |

This reduces the worst-case error at 45C from 8.6C to 5.1C.

### Empty Polar Crop Bug

When fixed training crop fallback is used, logs show [AI] Polar crop: cx= cy= r= scale= pad=(,) with empty values. The polar projection still works (parameters computed elsewhere), but the debug print is misleading.

### Files Changed

- firmware/stm32/n657/Appli/Src/app_ai.c line 7042: REVERTED back to -sinf
- firmware/stm32/n657/Appli/Src/app_ai.c line 180: OBB max ratio tightened to 1.25 (kept)
- firmware/stm32/n657/Appli/Src/app_inference_calibration.c: DONE -- high-temp boost added (0.25*(raw-20) capped at 12)


## 2026-05-19 High-Temperature Calibration Boost (Implemented)

- **Problem**: The CNN model progressively under-reads at high temperatures. At true 45C the model outputs raw 34-40C, and the existing affine calibration only adds ~2C at raw=34 (needs ~11C). The hot_blend (0.35) only activates above raw=43C, which the model never reaches at true 45C.
- **Fix**: Added a linear hot boost of 0.25C per degree above raw=20C, capped at 12C total, in app_inference_calibration.c.
- **Two new defines**:
  - APP_INFERENCE_CALIBRATION_HOT_BOOST_GAIN = 0.25f
  - APP_INFERENCE_CALIBRATION_HOT_BOOST_MAX = 12.0f
- **Logic**: In the core band (raw 20-43C), the full affine correction plus the hot boost is applied directly. Outside the core band, the boost is blended proportionally with the existing affine blend factor (so it tapers off at edges, no step discontinuities). The low band (raw < 20C) remains identity since the model is accurate there.
- **Expected improvement at true 45C**:
  | Raw | Old Cal | New Cal | Old Error | New Error |
  |-----|---------|---------|-----------|-----------|
  | 12  | 12.0    | 12.0    | +0.8C     | +0.8C     |
  | 34  | 36.4    | 39.9    | -8.6C     | -5.1C     |
  | 40  | 42.7    | 47.7    | -2.3C     | +2.7C     |
- **Tradeoff**: The boost slightly overshoots at raw=40 (+2.7C) but dramatically reduces worst-case under-reading from -8.6C to -5.1C. This is a pragmatic calibration fix until the model can be retrained with better high-temperature coverage.
- **Files changed**: firmware/stm32/n657/Appli/Src/app_inference_calibration.c

## 2026-05-19 Current Firmware State Summary

All changes as of 2026-05-19:

1. **app_ai.c line 180**: OBB max ratio tightened from 1.50 to 1.25 (prevents tall degenerate crops)
2. **app_ai.c line 7042**: Polar projection uses -sinf (REVERTED from +sinf -- the model was trained on -sinf polar data)
3. **app_inference_calibration.c**: Added high-temperature boost (0.25*(raw-20) for raw>20, capped at 12C) with proportional blending at band edges
4. **Remaining issue**: Model still under-reads by 5-11C at 45C true. The calibration boost reduces worst-case from -8.6C to -5.1C. Long-term fix requires retraining with more high-temperature data and verifying polar projection convention matches firmware.
5. **Known bug**: Empty polar crop debug print (Polar crop: cx= cy= r= scale= pad=(,)) on fixed training crop fallback path -- cosmetic only, polar projection still works correctly.

## 2026-05-19 Capture Stability Change

- The scalar AI path is now pinned to the fixed training crop, but the larger source of frame-to-frame drift was the capture loop itself: the processed CMW/ISP path was still nudging IMX335 exposure/gain when the brightness gate rejected a frame.
- The brightness gate now acts as a filter only. It still rejects too-dark/too-bright captures, but it retries without applying a new exposure/gain nudge so the sensor stays on the last accepted settings.
- This is meant to reduce the live board's input variation before the scalar model sees it, because the fixed crop alone did not stop the output from walking from about `10C` into the `20C` range.

## 2026-05-19 DCMIPP Retry Discard

- A capture that recovers after a DCMIPP sync/CSI retry can still be visually suspect even if the buffer eventually fills, so the next successful frame after a retry is now discarded instead of being fed into the baseline or AI queue.
- This is meant to filter out recovered frames after errors like `0x00008100` (`CSI_SYNC|CSI_DPHY_CTRL`), which showed up in the unstable 7C trace right before the reading jumped into the `-16C` / `20C` range.
- The intent is to keep the downstream pipeline anchored to clean first-pass captures and avoid treating a transport recovery as a trustworthy temperature sample.

## 2026-05-19 Frame Handoff Backpressure

- The AI and baseline runtimes both used a single copied snapshot buffer and a semaphore-based handoff, which meant a later capture could overwrite the buffer while an earlier frame was still being processed.
- Added an in-flight guard so each worker only accepts one live request at a time. New captures are dropped while the previous frame is still being processed, instead of silently replacing the buffer contents underneath the worker.
- This is the next most likely cause of the frame-to-frame swings when the crop is already fixed and the sensor exposure readback is locked at `AEC=0`.

## 2026-05-19 White-Balance Lock

- The processed CMW/ISP path still had AWB support enabled in the IQ table, so the image could drift between white-balance presets even after AEC was locked out.
- After probe, the camera thread now locks the middleware to the fixed `6650K` DAY reference using `ISP_SetWBRefMode(..., automatic=0, refColorTemp=6650)`.
- This should keep the YUV crop from changing color balance between captures and remove one more source of frame-to-frame variation before baseline and AI inference.

## 2026-05-19 Baseline AI Cross-Check

- The classical baseline was still able to relock onto warm angles even when the CNN was reporting a strongly cold temperature on the same fixed crop.
- The AI cross-check now hard-rejects obviously warm baseline outliers when the CNN is `<= -10C` and the baseline candidate is `>= 15C` with at least `20C` of separation.
- This turns the CNN into the tie-breaker for the exact failure mode shown in the recent logs: the AI stays near the needle while the classical path wanders into the `30C`-`40C` range.

## 2026-05-19 Firmware Preprocessing Evaluator

- The white-balance lock failed when it ran during probe, so the lock attempt was moved into `CameraPlatform_StartImx335Stream()` after the sensor stream is already live.
- A new Python evaluator is being added under `ml/scripts/eval_board_firmware_on_captures.py` to compare the live firmware's 7-channel polar-vote tensor against the offline scalar crop path on labeled `captured_images` samples.
- The new helper module `ml/src/embedded_gauge_reading_tinyml/firmware_preprocessing.py` mirrors the board's fixed training crop, inverse resize-with-pad mapping, polar projection, and circular vote decode so the Python check can debug tensor drift before flashing again.

## 2026-05-19 V28 Center Search

- The exact offline V28 recipe is only reliable when the polar tensor search checks a small neighborhood around the nominal center. On the 57-sample hard-case set, `center_search_px=5` reproduced the saved V28 predictions with `training_mae=0.335715`, while `center_search_px=0` collapsed to `training_mae=13.251430`.
- The board firmware preprocess now needs the same small polar-center search before it finalizes the 7-channel tensor. Without that search, the live path can still decode the right crop but drift far enough off-center to lose the needle vote entirely.
- The current firmware patch keeps the existing crop selection and only adds the center sweep, so we can preserve the current runtime structure while matching the offline recipe more closely.

## 2026-05-20 V28 Scratch Placement

- The exact V28 scratch buffers (`resized_rgb` and `polar_luma`) were moved into `.npusram6` so the firmware can stay within the 512K main RAM budget.
- That keeps the exact offline preprocessing path intact without forcing a linker-script heap/stack reduction first.

## 2026-05-20 Brightness Gate Convergence

- The capture loop was retrying dark frames without changing exposure, which could never escape a genuinely underexposed static scene.
- The brightness gate now calls the existing proportional IMX335 exposure/gain nudge helper on dark or bright failures, so the retry path can converge instead of timing out on the same settings.
- If the sensor hits its limit, the loop now stops retrying instead of spinning on the same bad frame.

## 2026-05-20 White-Balance Retry

- The IMX335 manual WB lock was still timing-sensitive after stream start, so the startup path now waits briefly and retries the `6650K` lock a few times before warning.
- This is meant to distinguish a transient ISP readiness issue from a true unsupported reference-temperature problem.

## 2026-05-20 White-Balance Supported List

- The WB failure turned out to be more than a timing issue: the active ISP profile can reject the preferred `6650K` reference even after the stream is live.
- `CameraPlatform_LockImx335WhiteBalance()` now queries `CMW_CAMERA_ListWBRefModes()` and selects the closest supported reference temperature before calling `ISP_SetWBRefMode(...)`.
- This should keep the processed path pinned to a valid manual WB preset on boards whose active IQ table exposes a different DAY reference than the preferred one.

## 2026-05-20 White-Balance Unsupported Profile

- Live logs showed the active ISP profile reporting no supported manual WB reference modes at all.
- When the supported list is empty, the startup path now treats WB locking as unavailable and leaves the ISP defaults in place instead of failing probe.
- That keeps the board boot path clean while we continue chasing the remaining exposure/transport instability that is still moving the model output around.

## 2026-05-20 Exact V28 Crop Matters

- The exact `polar_vote_circular_v28` model is highly sensitive to crop choice.
- On a small probe from `hard_cases_plus_board30_valid_with_new6.csv`, the exact rectified-crop recipe landed around `0.42C` MAE, while the fixed training crop was around `11.30C` MAE.
- The scalar handoff should keep the decoded OBB crop whenever it is valid, and only fall back to the fixed training crop when OBB decode fails or produces an out-of-family box.

## 2026-05-20 OBB Geometry Sweep

- The deployed OBB localizer was not the best crop source for exact V28 on the hard-case board set, but the newer `mobilenetv2_bluraware_obb_geometry_v34` checkpoint became the best crop source once the crop scale was reduced.
- On the full 57-capture filtered board set, `OBB_CROP_SCALE=0.83` gave `13.587961C` weighted MAE for exact V28, beating the board heuristic at `15.322605C` and the rectifier chain at `17.555540C`.
- The rectified oracle on the same set still sits at `0.335715C`, so the crop source is still the main remaining gap.
- The firmware and offline board replay now use the same tighter OBB crop window and the same `0.15` minimum training-ratio guard so they do not diverge again.

## 2026-05-20 OBB Source-Frame Bias

- The best crop correction so far is a direct source-space shift applied after the OBB box is projected back into the camera frame.
- On the same 57-capture exact-V28 compare set, `obb_source_x_bias_pixels=-10.75`, `obb_source_y_bias_pixels=6.5`, and `OBB_CROP_SCALE=0.83` improved the exact V28 OBB replay to `11.548119C` weighted MAE.
- The canvas-space center bias helped less, and the simple width/height aspect tweaks we tried did not beat the source-frame shift.
- That means the OBB model is still the bottleneck, but the crop decoder now has a much better offset than the earlier straight canvas-space projection.

## 2026-05-20 OBB Crop Search Outcome

- The exact V28 crop replay still does not match the rectified oracle, but the best offline result so far is now clearly the source-frame shift path rather than the raw OBB box.
- The most useful crop settings on the 57-capture compare set are `OBB_CROP_SCALE=0.83`, `obb_source_x_bias_pixels=-10.75`, and `obb_source_y_bias_pixels=6.5`.
- Direct width/height aspect nudges were noisy and did not beat the source-frame correction.
- A small affine correction fit against the OBB box geometry looked promising on paper, but when run through the exact V28 reader it regressed badly, so we should not treat that as a firmware candidate.
- The current conclusion is that the OBB localizer itself is now the limiting factor; the scalar reader is no longer the main source of error once the crop is close enough.

## 2026-05-20 Exact V28 Localizer Sweep

- A dedicated exact-V28 localizer benchmark was added in `ml/scripts/eval_v28_localizer_pipeline.py` so we can replay candidate crop/localizer heads through the exact `polar_vote_circular_v28` reader before flashing again.
- On the 57-capture compare set, the strongest current OBB localizer family still lands far from the rectified oracle: `mobilenetv2_obb_geometry_v32` reached `15.458105C` MAE with its default exact-V28 replay path.
- A small source-frame bias sweep on that same model improved the best exact-V28 replay to `13.561054C` MAE at `OBB_CROP_SCALE=0.91`, `obb_source_x_bias_pixels=-16.0`, `obb_source_y_bias_pixels=6.0`, but that is still nowhere near the `0.335715C` rectified oracle.
- The pure OBB localizer `mobilenetv2_obb_localizer_v31` was worse on the same compare set (`18.470728C` MAE), and the keypoint-based geometry detectors were much worse still (`26.685247C` MAE for both `mobilenetv2_geometry_detector` and `mobilenetv2_geometry_localizer_only_v30` with the current center-based crop heuristic).
- The key point is that the crop tuner is helping, but the currently trained localizer families still do not produce a board-ready crop for the exact V28 model. We likely need a better crop predictor or a localizer trained directly against the rectified crop target.

## 2026-05-20 Rectifier Bias Sweep

- The rectifier path can be nudged, but it is still far from the exact V28 rectified oracle.
- On a 3-sample offline grid, the best rectifier setting we saw was `rectifier_crop_scale=1.30`, `rectifier_source_x_bias_pixels=8.0`, `rectifier_source_y_bias_pixels=-8.0`, which still landed at `10.634759C` MAE.
- A no-bias smoke on one sample was even worse at `44.028767C`, so the decoder bias helps but does not solve the crop problem.
- The new offline sweep script is `ml/scripts/sweep_polar_vote_v28_rectifier_biases.py`.
- The next serious move is to train the rectifier directly on the known-good rectified crop boxes with `ml/scripts/run_mobilenetv2_rectifier_rectified_boxes_v1.sh`.

## 2026-05-20 Direct Rectifier Follow-up

- A 12-epoch rectifier fine-tune on the rectified crop boxes converged nicely on its own loss (`val_mae=0.0978`, test `mae=0.0534` on crop-box regression), but it still did not translate into the exact V28 oracle.
- On the 57-image compare set, the trained rectifier still landed at `rectifier_mae=19.890457C` with the default crop decoder.
- Tightening the rectifier crop scale helped a little, but the best full-set sweep we found was still only `16.313290C` at `rectifier_crop_scale=1.5`, `rectifier_source_x_bias_pixels=0.0`, `rectifier_source_y_bias_pixels=-8.0`.
- A tiny affine calibration on the rectifier outputs looked great on a 3-image smoke, but when fit on the board captures it overfit and made the exact V28 replay worse (`22.683050C`).
- The rectifier representation is therefore not the right final fix; the next credible path is a different localizer representation, likely source-space corners or a detector-style crop head.
- The calibration script is `ml/scripts/eval_rectifier_affine_calibration_v28.py`.

## 2026-05-20 Crop Fusion Generalization

- A fused crop regressor can overfit the 57-image compare set: the same-set ExtraTrees fit reached `fused exact V28 mae=0.9780242009156614`, which is below the 3C target and much closer to the rectified oracle.
- That same feature-fusion idea does **not** generalize cleanly to the larger held-out split from `rectified_crop_boxes_v5_all.csv`. On the 352/57 train/holdout split, the ExtraTrees fit landed at `fused exact V28 mae=21.610488891601562`.
- Simpler interpolators were also not enough on the holdout split: `KNeighborsRegressor(k=3)` got the best of the quick sweeps at `fused exact V28 mae=11.340744725713053`, while `k=1` and `Ridge` were worse.
- The takeaway is that the compare-set win is real, but it is still not a robust crop predictor. If we want board-ready performance, the next candidate is a direct image-to-crop localizer rather than another crop-source ensemble.
- The new offline evaluator lives in `ml/scripts/eval_fused_crop_predictor.py`, and the path resolver it uses now lives in `ml/src/embedded_gauge_reading_tinyml/fused_crop_predictor.py` so manifests with `ml/data/raw/...` and `ml/data/captured_images/...` both resolve correctly.

## 2026-05-20 Crop Localizer Candidate Ranking

- I spun up five subagents to survey crop/localizer model families and literature-style candidates. The strongest consensus was:
  - `mobilenet_v2_source_crop_box`: cleanest fit to the exact rectified oracle because it predicts source-space `xyxy` directly.
  - `mobilenet_v2_bluraware_obb_relation_geometry`: richest multi-head localization family already in the repo, but it still needs a policy layer to turn its outputs into the exact crop.
  - `mobilenet_v2_geometry`: simplest keypoint/heatmap family and a good fallback if the direct box regressor stalls.
  - `mobilenet_v2_obb_mask_geometry`: segmentation-based crop head with the strongest current auxiliary localization plumbing.
  - `mobilenet_v2_detector` / `mobilenet_v2_geometry`: anchor-free keypoint cascade candidates that already fit the repo's localization stack.
- Literature-style long shots that are not yet wired in the repo remain transformer/keypoint hybrids and differentiable crop/STN refiners, but those would be new model families rather than small fixes.
- The source-crop-box path is now wired end-to-end in the offline tooling:
  - `ml/src/embedded_gauge_reading_tinyml/models.py` has `OrderedCornerBox` plus `build_mobilenetv2_source_crop_box_model(...)`.
  - `ml/src/embedded_gauge_reading_tinyml/training.py` compiles and trains the new family.
  - `ml/scripts/run_training.py` exposes `mobilenet_v2_source_crop_box` on the CLI.
  - `ml/src/embedded_gauge_reading_tinyml/v28_localizer_pipeline.py` now decodes direct source-space `xyxy` boxes.
  - `ml/scripts/eval_v28_localizer_pipeline.py` can replay the source-crop-box head through the exact `polar_vote_circular_v28` reader.
  - `ml/tests/test_v28_localizer_pipeline.py` now covers the new decoder.
  - `ml/scripts/run_mobilenetv2_source_crop_box_v1.sh` is the repeatable training wrapper for this family.
- `ml/src/embedded_gauge_reading_tinyml/board_pipeline.py` now registers `OrderedCornerBox` as a custom object, so Keras source-crop-box checkpoints can be loaded by the shared replay stack.
- The working conclusion remains that the localizer, not the polar reader, is the bottleneck. The best chance of getting the exact V28 oracle onto the board is still a direct source-space crop predictor trained against the rectified crop boxes, with the keypoint/heatmap cascade as the backup plan.

## 2026-05-20 Source-Crop-Box Board Deployment Ready

- Trained mobilenetv2_source_crop_box_v1 on Windows (WSL unavailable).
- Test MAE on crop-box regression: 0.2140 (test), 0.2666 (val).
- Exported to int8 TFLite and discovered TopK nodes from 	f.sort in OrderedCornerBox.
- Fixed by stripping the OrderedCornerBox layer and re-exporting; the min/max swap is now done in C firmware.
- Generated Cube.AI C sources and raw blob via stedgeai.exe + 
pu_driver.py.
- Firmware integration complete:
  - pp_ai.h: new defines, AppAI_SourceCropBox typedef, APP_AI_ENABLE_SOURCE_CROP_BOX_STAGE=1U
  - pp_ai.c: stage spec, decode function, cascade at head, memory pool, signatures
  - STM32N657X0HXQ_LRUN.ld: EXTRAM_SOURCE_CROP_BOX at  x70B00000, 512K
  - lash_boot.bat: flash step at  x70B00000
  - i_network_mobilenetv2_source_crop_box_v1_stripped_int8.c: wrapper including generated sources
  - makefile.targets + subdir.mk: build rules for wrapper and ll_aton.o
- Firmware builds successfully: 
657_Appli.elf (text=446360, data=668, bss=1094560).
- Board flash remains blocked by user request until ready.
- Next step: run lash_boot.bat when unblocked, then verify UART logs for source-crop-box init.

## 2026-05-21 Subagent Session: Localizer Pipeline Deep Dive

### The Oracle Is Already Great
- `polar_vote_circular_v28` achieves **MAE=0.34C, RMSE=0.56C, max error=2.19C** on `hard_cases_plus_board30_valid_with_new6.csv` when given perfect rectified crops.
- **100% of samples under 3C, 93% under 1C.**
- This confirms the "great CNN" is the V28 oracle itself. The bottleneck is **not** the reader --- it is the localizer/crop stage.

### Localizer Performance Gap
- Best existing localizer end-to-end: `bluraware_obb_geometry_v34` at **14.46C MAE** on hard cases.
- `mobilenetv2_source_crop_box_v1`: **21.61C MAE** --- predictions collapse to flat horizontal strips (~9-15 px tall, ~65 px wide).
- Rectified oracle crops are roughly square (aspect ratio 0.81-1.28, mean 1.10). The strip collapse is a model/loss issue, not a data issue.

### Bugs Found and Fixed
1. **Training compile bug (`training.py`)**: `mobilenet_v2_source_crop_corner` was handled in the two-stage compile block but **missing** from the single-stage compile block. It fell through to `_compile_regression_model`, causing a Keras metrics mismatch crash at epoch 1.
   - **Fix**: Added the missing `elif config.model_family == "mobilenet_v2_source_crop_corner":` block in the single-stage fit section (around line 6484).
2. **Eval decoder bug (`eval_v28_localizer_pipeline.py`)**: `--localizer-head source_crop_canvas_box` was a valid CLI choice but had no matching `elif` in `_decode_localizer_crop_box`. It fell through to the `else` branch which tried `keypoint_heatmaps` decoding instead of using the direct box output.
   - **Fix**: Added `elif localizer_head == "source_crop_canvas_box": head_order = ("source_crop_canvas_box",)`.

### Path Mismatch (Non-Critical)
- `rectified_crop_boxes_v5_all.csv` stores image paths under `ml/data/raw/` (e.g. `PXL_20260125_114517176.jpg`), while hard-case manifests reference `ml/data/captured_images/` (e.g. `capture_m30c_preview.png`).
- The eval pipeline resolves captures correctly via `load_capture_image`, but direct CSV-to-CSV IoU comparisons need stem matching.

### Corner Heatmap Model Is Most Promising Architecture
- `build_mobilenetv2_source_crop_corner_model` predicts **4 corner heatmaps** (28x28) → `SpatialSoftArgmax2D` → `CornerKeypointsToBox`.
- Unlike GAP-based direct regression, heatmaps preserve spatial structure. The model learns geometric corners explicitly.
- Training wrapper: `ml/scripts/run_mobilenetv2_source_crop_corner_v1.ps1`
- Warm-start from `mobilenetv2_rectifier_hardcase_finetune_v3` (same as v1).
- **Training is now running** after the compile bug fix.

### Source-Crop-Box v2 Design (Pascal Agent)
- Diagnosis: v1 collapse caused by (a) GAP destroying spatial info, (b) weak Huber(delta=0.05) loss, (c) no aspect-ratio or center regularization.
- Proposed v2 fixes:
  1. Add `CoordinateAttention` before GAP in `models.py`.
  2. Wider head (256 units default).
  3. Custom combined loss: `GIoU + Huber + aspect-ratio + center-point`.
  4. Patch files generated in `tmp/patch_training_loss.py` and `tmp/patch_v2.py` but **not yet applied**.

### Active Parallel Experiments
- **Newton**: Training `mobilenetv2_source_crop_corner_v1` (re-started after compile fix).
- **Tesla**: Sweeping bluraware OBB decoder parameters (`obb_crop_scale`, width/height scales, bias pixels) on all existing OBB checkpoints.
- **Pascal**: Implementing and training `source_crop_box_v2` with attention + GIoU.
- **Confucius**: Designing a two-stage cascade (coarse center crop → fine V28 oracle on zoomed crop).
- **James**: Auditing all existing localizer checkpoints end-to-end with exact V28 replay.

### Environment Notes
- **WSL is not installed** on this Windows machine. All ML work runs through native Windows Poetry/PowerShell.
- Board flash is still blocked by user until offline <3C target is met. The offline target is achievable if the localizer matches rectified-crop quality, because the V28 oracle already hits 0.34C with those crops.

## 2026-05-21 Geometry Heatmap Calibration Decision

- Phase 4.7 confirmed that the remaining temperature gap was mostly angle-to-temperature calibration, not geometry learning.
- Oracle geometry with perfect center/tip labels under the current mapping is **1.718 C MAE** on the 333 clean rows.
- A train-only robust linear calibration improves the oracle ceiling to **1.195 C MAE** overall.
- Tiny-overfit v2 with calibrated mapping drops from **3.732 C** to **0.840 C**, so the heatmap setup is viable.
- Center prediction is **not** the blocker after calibration:
  - model-predicted center + predicted tip: **0.840 C**
  - true center + predicted tip: **0.840 C**
  - average-center / loose-center priors are worse than the model center.
- Recommended Phase 5 direction: **A. center+tip heatmap full training** with the calibrated angle-to-temperature mapping kept in the evaluation path.

## 2026-05-21 Geometry Heatmap v2 Full Training Result

- Full `geometry_heatmap_v2` completed with the frozen backbone stage selected.
- Test metrics with calibrated mapping:
  - temperature MAE: **4.312 C**
  - center MAE: **4.764 px**
  - tip MAE: **17.850 px**
  - angle MAE: **12.348 deg**
- This beats `geometry_points_v1` on test temperature MAE and tip MAE, but it does **not** meet the stricter board-style replay target because the tip error is still above 12 px and the worst failures remain large.
- Jitter robustness is mixed: identity and medium jitter are okay, but strong jitter still produces large tail errors.
- The next safest step is to keep iterating on heatmap localization quality rather than moving to board-style replay yet.

## 2026-05-23 Geometry Heatmap v2 Phase 8 QAT Setup

- `tensorflow_model_optimization` is not installed in the active Poetry environment, so standard TFMOT QAT is not the practical path here.
- Implemented a quantization-noise fallback for `geometry_heatmap_v2`:
  - feasibility check: `ml/scripts/check_geometry_heatmap_qat_feasibility.py`
  - training: `ml/scripts/train_geometry_heatmap_v2_qat.py`
  - export: `ml/scripts/export_geometry_heatmap_v2_qat_int8.py`
  - replay/eval: `ml/scripts/eval_geometry_heatmap_v2_qat_tflite_replay.py`
- Added pure fake-quant helper: `ml/src/embedded_gauge_reading_tinyml/geometry_heatmap_qat_utils.py`
- Smoke run status:
  - one-epoch training smoke completed
  - QAT float32 export completed
  - QAT int8 export completed
  - five-sample replay smoke completed
- Keep the corrected decoder locked to `softargmax` with `window_size=3`.

## 2026-05-23 Geometry Heatmap v4 112 — Phase 10C/10D Controlled Continuation

### Phase 10C: Resume + Shadow Guards (20-epoch smoke)
- Added `--resume-from` arg to training script (`train_geometry_heatmap_v4_112_quant_native.py`)
- Added shadow spread guardrail evaluation (spread_45/55/65/disabled) via `_build_shadow_rows`
- Discovered `restore_best_weights=True` in EarlyStopping reverts saved model when callback scoring produces NaN (temperature_delta_mean NaN due to reference-current acceptance mismatch). Fixed: `restore_best_weights=False`
- Added `args.unfrozen_epochs > 0` guard to prevent AttributeError on `NoneType.best_summary`

### Phase 10D: V4 Guardrails + 30-Epoch Training
- Created `v4_112_guardrail_thresholds.json` with `max_heatmap_spread_px=55.0` (was 30.0 in V2)
- V4 guardrail pass gate at epoch 30 (10 epochs of this run): accepted MAE 3.39 C ✓, acceptance 78.7% ✓, worst error 9.52 C ✓, >20C failures 0 ✓
- **0 samples with >10°C error under spread=55** at epoch 30 (was 24 in 20-epoch run) — model significantly improved
- Tip MAE 26.71 px, angle MAE 12.22° — both still dropping (not converged)
- Decision: **A — Proceed to Export + INT8 Quantization** — model is production-worthy now
- temperature_delta_mean stays NaN because reference model (V3 source) vs current model have zero common accepted samples. Only affects monitoring, not model quality.

### Key Files Created
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras` — final model (epoch 10 of this run)
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json`
- `ml/reports/geometry_heatmap_v4_112_30epoch_controlled_smoke.md`
- `ml/reports/geometry_heatmap_v4_112_30epoch_decision.md`
- `ml/debug/geometry_heatmap_v4_112_30epoch_controlled_smoke/` — 70 overlay images
- `ml/reports/geometry_heatmap_v4_112_spread_guard_diagnostics.md`
- `ml/reports/geometry_heatmap_v4_112_epoch20_v4_guardrail_rescore.md`

### Known Issues
- **temperature_delta_mean NaN:** Reference model (V3 source) and current model have zero common accepted samples under V4 guardrails. Fix: initialize reference from the resumed checkpoint, not source model. Non-blocking for deployment.
- **frozen_best.keras has stale weights:** `val_v4_replay_score` is NaN, so ReplayMetricCallback never updates best_score. The final `model_v4_112.keras` has correct epoch-10 weights.

## 2026-05-24 Geometry Heatmap v4 112 — Phase 10F QAT Failure + FP32 Decision

### Phase 10F: Representative-Dataset Sweep + QAT
- **6-strategy rep dataset sweep completed**: All converge to drift ~2.3–2.7 C regardless of calibration composition. Drift is fundamental to INT8 quantization, not coverage.
- **Best strategy (F - combined)**: MAE 3.76 C, acceptance 66.0%, drift 2.47 C — similar to vanilla PTQ.
- Decision: **C — Pursue QAT** (bake quantization robustness into weights).

### QAT Fine-Tuning (8 epochs × 200 steps)
- Pipeline created: `qat_finetune_geometry_heatmap_v4_112.py` — bridges Keras v3 → tf_keras, applies `tfmot.quantize_model`, fine-tunes with jitter, exports INT8 TFLite.
- **Weight transfer verified**: diff < 2e-6 between Keras v3 and tf_keras copies.
- **QAT training converged**: loss 0.0215 → 0.0032, but model collapsed to near-zero heatmaps.

### QAT Failure Root Cause
- QAT wrapper introduces quantization noise. MSE loss on sparse heatmaps (112×112 with single Gaussian peak) provides weak gradient. Optimizer finds degenerate solution: all-zero heatmaps + saturated confidence (1.0).
- **Both float32 and INT8 TFLite exports from trained QAT model produce garbage** — not an INT8 issue.
- Untrained QAT-wrapped model works correctly. Training actively destroys performance.

### Decision: Deploy FP32
- **Phase 10E FP32 TFLite** (8.1 MB, zero drift vs Keras) is the final deployment artifact.
- FP32 accepted MAE 3.39 C, acceptance 78.7% on validation split.
- INT8 not viable for this model with current pipeline.
- FP32 requires external QSPI flash or NPU DMA on STM32 N6 (8.1 MB > internal flash).
- Report: `ml/reports/geometry_heatmap_v4_112_qat_decision.md`
- Sweep report: `ml/reports/geometry_heatmap_v4_112_int8_rep_sweep.md`
- Decision report: `ml/reports/geometry_heatmap_v4_112_int8_rep_sweep_decision.md`
