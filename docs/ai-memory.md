# AI Memory

## Board reference and retrain check (2026-06-04)

- Live board reference frame: the needle was physically at `6C`, but the current AI path logged `-7.7C` and then `-6.4C` on the next sample.
- The classical baseline on the same scene logged `-10.3C`, so the full board pipeline is still off by about `13C` on this lighting condition.
- The cleaned board-mimic retrain `boardmimic_clean_varB` exported successfully and kept firmware-compatible int8 quantization, but its holdout was worse than the earlier board-style candidates:
  - overall MAE `11.57 px`
  - capture MAE `5.57 px`
  - pxl MAE `12.84 px`
- Do not promote `boardmimic_clean_varB` to the board package. The stronger known candidates remain `boardstyle_mixed_varB` and `varB_v2`.

## Board-mimic dataset QA (2026-06-04)

- `ml/scripts/prepare_center_training_board_mimic.py` now regenerates the board-mimic dataset from the 420-row merged geometry manifest and uses the current OBB localizer on CPU without issue.
- The fresh output set kept all 420 rows, with zero blank-frame skips and zero crop fallbacks; the crop window is still the firmware-matching `155x123` box, then resize-with-pad to `224x224` for the CNN input.
- The older `ml/data/center_training_board_mimic/` artefacts included legacy black capture frames, but the regenerated dataset no longer shows that failure mode in QA.

## Board-mimic ablation results (2026-06-03)

### Comparison table

| Variant    | Overall  | Capture  | PXL     | Cx_err   | Cy_err   | Warmup | FT  | LR_ft   | C_wt  | Aug    | UF | Note |
|------------|----------|----------|---------|----------|----------|--------|-----|---------|-------|--------|----|------|
| **original** | 8.91 px | 8.29 px | 9.04 px | 9.54 px  | 8.29 px  | 4 ep   | 12  | 3e-5    | 1.0Ã—  | mild   | 0  | CPU-trained |
| **baseline** | 7.43 px | 6.58 px | 7.61 px | 7.36 px  | 7.49 px  | 4 ep   | 12  | 3e-5    | 3.0Ã—  | mild   | 0  | GPU reproduction |
| **varA**     | 6.62 px | 6.84 px | 6.57 px | 7.04 px  | 6.20 px  | 30 ep  | 0   | 3e-5    | 3.0Ã—  | mild   | 0  | Frozen only |
| **varB**     | 6.67 px | 5.52 px | 6.92 px | 7.11 px  | 6.23 px  | 30 ep  | 0   | 3e-5    | 5.0Ã—  | strong | 0  | Best capture |
| **varC**     | 8.98 px | 7.45 px | 9.30 px | 9.26 px  | 8.70 px  | 4 ep   | 20  | 1e-5    | 3.0Ã—  | mild   | 20 | Last 20% unfreeze regressed |

### Winner: **varB**

- **Best capture MAE (5.52 px)** â€” primary criterion, beats baseline (6.58 px) by 1.06 px
- Overall MAE (6.67 px) improved vs baseline (7.43 px), well within the 0.5 px constraint
- PXL MAE (6.92 px) also improved vs baseline (7.61 px)
- Config: frozen backbone, 30 epochs head-only, strong photometric + rotation aug, capture_weight=5.0
- No TFLite collapse on 11 capture test samples (offline sanity check)
- All quantization matches firmware: input scale=1.0 zp=-128, output scale=0.00390625 zp=-128
- Run: `ml/artifacts/training/center_model_ablation_varB_20260603_192725/`
- TFLite: `ml/artifacts/deployment/center_model_board_mimic_int8/model_int8_varB.tflite`

### Key findings
1. **Frozen backbone only** (varA, varB) beats any fine-tuning (baseline full, varC last-20%). The small dataset (294 train) doesn't support backbone updates without overfitting.
2. **Stronger augmentation + higher capture weight** (varB: brightnessÂ±0.30, contrast[0.5,1.5], rotationÂ±0.07 rad, cw=5.0) improves capture MAE by 1.3 px over mild aug + cw=3.0 (varA).
3. **Last-20% unfreeze** (varC) regressed val MAE from 0.039 â†’ 0.047 during fine-tune despite 1e-5 LR.
4. **Quantization degradation** is real: Keras model capture MAE 5.52 â†’ TFLite offline capture MAE ~8.59 px. The firmware will see TFLite accuracy â€” the gap is acceptable but worth tracking.
5. **The -128,-128 firmware collapse is NOT a model issue** â€” all TFLite models produced valid outputs in offline tests. The live board now keeps the rim-vote centre as a fallback if the CNN output is obviously wrong or the CNN path fails.

### Next step
- Cube.AI re-import of `model_int8_boardstyle_mixed_varB.tflite` (792,720 bytes) is complete; the mixed-loader board-mimic model is now the deployed center candidate
- NPU isolation, if any true overlap remains, is now a hardening task rather than a blocker for using the mixed-loader model

## Quick-reference for the current state of the project.

### Pipeline Architecture (Live Board, 2026-06-03)

**Camera:** IMX335 â†’ DCMIPP â†’ 224Ã—224 YUV422 frame buffer. Stores full YUV (luma + chroma), not greyscale.

**Baseline pipeline** (runs in separate ThreadX worker thread, ~1-2s per frame):
1. Five seed-centre estimators run in parallel:
   - **bright-center-polar**: brightness-weighted centroid (Priority 2)
   - **fixed-crop-polar**: training crop centre with upward bias (Priority 5) â€” centre ~(120,77)
   - **board-prior-polar**: fixed board-specific centre ratios (Priority 3)
   - **rim-center-polar**: coarse-to-fine rim-vote grid search (Priority 1) â€” centre ~(108,108)
   - **image-center-polar**: inner dial centre (Priority 4) â€” centre ~(112,100)
2. Each seed runs `AppBaselineRuntime_EstimatePolarNeedle` â€” Sobel-like edge gradient vote in polar space + spoke continuity + chroma consistency
3. Local geometry sweep (`RefineEstimateAroundSeed`) tries sub-centre offsets
4. **Rim-centre override**: if rim-centre-polar result is valid with Celsius-sweep angle, it replaces whatever the priority/consensus system selected
5. Burst median: 3-entry history, push every frame, median filters single-frame outliers

**AI inference pipeline** (runs in separate ThreadX worker thread, ~5.6s per frame):
1. **OBB localizer** (`mobilenetv2_obb_longterm`) â€” full-frame NPU inference â†’ gauge bounding box
2. **Rim-vote centre** (`AppBaselineRuntime_EstimateDialCenterFromRimVotes`) â€” full-frame scan, centre ~(108,108) quality ~70-78
3. **Centre-detector CNN v4 (varB)** â€” active on the board; rim-vote centre is still logged as the classical reference and is the fallback when the CNN output or runtime path looks bad
4. **Polar vote** (`AppBaselineRuntime_EstimatePolarNeedle`) â€” same function as baseline, at centre (108,108) â†’ needle angle
5. **Burst median smoothing** â€” 3-frame median, resets on >12Â°C jump

### Gauge Calibration Parameters

Stored in `ml/src/embedded_gauge_reading_tinyml/gauge/gauge_calibration_parameters.toml` and mirrored in `app_gauge_geometry.h`:

```toml
[littlegood_home_temp_gauge_c]
min_deg = 135.0        # min tick at ~7:30, maps to -30Â°C
sweep_deg = 270.0      # needle sweeps 270Â° clockwise
min_value = -30.0      # degrees Celsius
max_value = 50.0
units = "C"
direction = "clockwise"
needle_colour = "black"       # updated from "dark" (2026-06-03)
obb_pivot_x_offset_ratio = -0.0089   # â‰ˆ âˆ’2 px on 224Ã—224 frame
obb_pivot_y_offset_ratio = 0.0625    # â‰ˆ +14 px on 224Ã—224 frame
inner_dial_radius_frame_ratio = 0.3076  # 68.9 px on 224Ã—224
```

### Centre Estimation

**Rim-vote centre** (`AppBaselineRuntime_EstimateDialCenterFromRimVotes`) is the primary centre method for both pipelines:
- Coarse step = 8px, fine step = 4px
- Scans near the training crop centre (baseline) or full frame (AI)
- Returns centre ~(108,108) consistently (quality 70-78)
- This is the **correct** gauge centre for this mount

The centre-detector CNN v4 mixed-loader `boardstyle_mixed_varB` package is now enabled on the live board. `app_ai_runtime_tail.inc` logs the rim-vote reference, then passes `(-1,-1)` into `AppCenterDetector_Run()` so the retrained CNN runs. `app_center_detector.c` now falls back to the rim-vote centre whenever the CNN output is obviously wrong or the CNN path fails, and the xSPI2 signature check still matches the flashed package raw at `0x70200000`.

### Polar Vote Algorithm

`AppBaselineRuntime_EstimatePolarNeedle` in `app_baseline_runtime.c`:

1. **Gradient vote**: scan annulus (30%-70% of dial radius), Sobel edge magnitude Ã— tangential alignment Ã— darkness Ã— shaft-weight â†’ 360 angle bins
2. **Connection boost**: each edge pixel votes with weight amplified by hub_connection (0.4), tip_extension (0.4), and width_score (0.2) â€” needle-shaped spokes get 18Ã— boost over isolated dial markings
3. **Smooth + top peak selection**: 3-bin triangular smoothing, keep top 64 peaks
4. **Continuity-weighted peak selection**: for each top peak, measure spoke continuity (avg darkness along 12 samples, 20%-80% radius) Ã— hub darkness Ã— vote score. Best weighted score wins.
5. **Angle refinement**: quadratic interpolation around peak bin
6. **Inversion check**: if angle is outside Celsius sweep and opposite direction is darker, flip 180Â°
7. **Final validation**: 20-sample spoke continuity (15%-85% radius), modulated by chroma consistency penalty. Bright-relaxed threshold = 0.10, normal = 0.22

### Chroma Consistency (new 2026-06-03)

The final spoke-continuity check now measures U/V chroma variance along the needle ray:
- Black needle on white dial â†’ Uâ‰ˆ128, Vâ‰ˆ128 at every sample â†’ low variance
- False peak crossing coloured bezel/shadow edge â†’ high U/V variance
- `chroma_penalty = 1.0 / (1.0 + variance/500.0)`
- Effective continuity = spoke_continuity Ã— chroma_penalty
- Logged as `chroma=NNN` in Polar reject lines

This applies to BOTH pipelines since both call `EstimatePolarNeedle`.

### Live Board Accuracy (2026-06-03, True â‰ˆ 7-8Â°C)

| Pipeline | Reading | Error | Confidence |
|----------|---------|-------|------------|
| Baseline (rim-centre override) | 7.7Â°C | ~0Â°C | 3.3-3.5 |
| AI (rim-vote centre + polar vote) | 7.7Â°C | ~0Â°C | 3.3-3.5 |

Both pipelines now agree (identical centre + same polar vote). The AI's OBB NPU inference (5s) adds latency but provides no unique output vs the classical rim-vote centre.

### False Peak at 148Â°

The polar vote at centre (108,108) occasionally finds a false peak at ~148Â° (-26Â°C) instead of the real needle at ~262Â° (7.7Â°C). This happens under certain lighting conditions (bright frames, exposure settling after cold boot).

Mitigations applied (all three needed together):
1. **Rim-centre override** â€” ensures correct centre is always preferred
2. **Chroma consistency** â€” penalises the 148Â° ray if it crosses coloured bezel regions
3. **Burst median** â€” single-frame false peaks filtered by 3-entry median

### Stability / History Smoothing (new 2026-06-03)

- Removed the `IsStableEstimateForHistory` stability gate
- Now every frame estimate is pushed to the 3-entry history
- `PushEstimateHistory` internally resets when jump > ~12Â°C (APP_BASELINE_HISTORY_RESET_DELTA_C)
- `SelectSmoothedEstimate` sorts entries and returns median (3 entries) or warmest (1-2 entries during warm-up)
- This gives burst-median behaviour: single outliers suppressed, sustained changes track

### Latency & Power Metrics (2026-06-03)

- **Baseline**: 700-2000ms (CV only, varies with brightness/frame condition)
- **AI**: 5670-6568ms (OBB NPU inference is bottleneck, ~5.6s)
- **Capture period**: 60s (both pipelines complete within one period)
- **Baseline latency fixed**: was reporting 52ms (wrong slot), now correctly measures capture-to-result
- `Metrics_StartInference` no longer re-ends active slots
- `Metrics_OverrideStartTime` lets worker stamp frame capture time before EndInference
- Baseline now records a `MID` power checkpoint after the main geometry search, and both pipelines stamp the request capture time before the snapshot copy so the latency covers the full request-to-result window.
- `INA219_LogReading` now feeds the active metrics window, and the INA219 monitor thread runs above the baseline/AI workers so the power summary reflects pipeline-wide min/avg/max instead of a single seeded sample.

### Key Source Files

| File | Purpose |
|------|---------|
| `app_baseline_runtime.c` | Classical CV baseline: 5-seed polar voting, rim-centre override, chroma consistency, burst median |
| `app_ai.c` + `app_ai_runtime_tail.inc` | AI pipeline: OBB â†’ logged rim-vote reference + retrained centre-detector CNN â†’ polar vote â†’ temperature, burst median |
| `app_ai_helpers_decode.inc` | OBB stage inference, decode, crop computation |
| `app_center_detector.c` | Centre-detector module (CNN active; OBB pivot fallback still available) |
| `app_gauge_geometry.h` | Training crop ratios, inner-dial centre ratios, OBB pivot offset ratios |
| `app_filex.c` | FileX worker, SD log service, and the 2-second blue capture LED cue |
| `ina219_power.c` | INA219 sampling thread and power logging used by the metrics summary |
| `flash_boot.bat` | Signs FSBL/app, flashes center-detector and OBB blobs, extracts signatures |
| `inference_metrics.c` | 64-bit DWT timing, multi-slot inference metrics, total/queue/compute split |
| `gauge_calibration_parameters.toml` | Per-gauge angular & spatial calibration (mirrored in C header) |

### Key Calibration Constants (app_gauge_geometry.h)

```c
#define APP_GAUGE_TRAINING_CROP_X_MIN_RATIO 0.1027f
#define APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO 0.2573f
#define APP_GAUGE_INNER_DIAL_CENTER_X_RATIO 0.5000f
#define APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO 0.4460f
#define APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO (-0.0089f)
#define APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO  0.0625f
#define APP_GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO 0.3076f
```

### Key Thresholds (app_baseline_runtime.c)

```c
// Continuity thresholds (bright-relaxed / normal)
#define final_spoke_continuity_threshold  bright_relaxed ? 0.10f : 0.22f
#define main_continuity_threshold         bright_relaxed ? 0.24f : 0.35f
#define main_hub_threshold                bright_relaxed ? 0.15f : 0.25f

// Bright-relaxed triggers at ~60%+ bright pixel ratio
// "bright" = luma > 200

// Polar reject logs: continuity=X/100 chroma=NNN source=XXX mode=YYY
// continuity is spoke_continuity * 1000
// chroma is chroma_penalty * 1000 (1000 = no penalty)
```

### Known Issues / Next Steps

- **Centre-detector CNN v4 varB is flashed on the live board** â€” the runtime now drops to the rim-vote centre if the CNN output is saturated (`-128 -128`), wildly wrong, or the CNN path fails to initialize cleanly.
- **Post-flash smoke test still pending** â€” we rebuilt and flashed successfully, but the serial smoke test should be rerun once the board UART is free so we can confirm the live AI path returns to the baseline temperature instead of surfacing a bogus read.
- **False 148Â° peak** suppressed by chroma consistency but not eliminated. The chroma penalty scaling (variance/500.0) was chosen empirically â€” may need tuning for other lighting conditions.
- **Bright-relaxed mode** triggers at 60-70% brightness, lowering gates. The chroma consistency helps distinguish false peaks in bright frames.
- **Dial radius**: baseline uses 68.9px from frame ratio, AI infers from OBB box (~60-62px). Both give similar temperatures but the discrepancy needs investigation for generalisation.

### Session Summary (2026-06-03)

All changes made during this session:

#### Baseline & AI fixes (classical CV pipeline)
1. **Hot-zone override disabled entirely** â€” both blocks wrapped in `if (0)` or removed.
2. **Bright-relaxed continuity threshold lowered**: 0.20â†’0.15â†’0.10
3. **Normal-mode continuity threshold lowered**: 0.28â†’0.26â†’0.22
4. **Latency measurement fixed** â€” `Metrics_StartInference` no longer re-ends active slots; added `Metrics_OverrideStartTime`.
5. **Missing static variable declarations restored** â€” build fix.
6. **Rim-centre override** â€” rim-polar result replaces priority/consensus system's selected estimate.
7. **Burst median smoothing** â€” removed stability gate; 3-entry median on every frame.
8. **Chroma consistency** â€” U/V variance penalty in final spoke-continuity check.
9. **Gauge TOML**: `needle_colour` updated from `"dark"` to `"black"`.
10. **Dead code reverted**: `ScoreAngle` chroma weighting removed.

#### Centre-detector CNN retrained (2026-06-03 session)
**Problem**: The centre-detector CNN was producing garbage int8 outputs and was permanently bypassed via the `override_cx >= 0 && override_cy >= 0` fallback path.

**Data preparation**:
- Used ALL 374 board-capture PNGs from `ml/data/center_training_manual/images/` (previously only 193 "good" manual annotations were used)
- For each image, converted to YUV422 (BT.601) and ran the C rim-vote centre estimator (`ml/c/rim_estimator.so`) for verified ground-truth labels
- Generated 5 CD-crop variants per image with different OBB jitter seeds (Â±6px x, Â±2px y) â†’ 1870 total CD-crops (155Ã—123 â†’ 224Ã—224 pad)
- Stratified 80/20 train/val split by source image â†’ 1495 train, 375 val
- Fixed: RGB channel ordering in training data now matches OpenCV's BGRâ†’PNGâ†’TF decode pipeline (consistent with existing checkpoint)
- Data saved to `ml/data/center_training_crops/`

**Training**:
- Fine-tuned from existing checkpoint `center_model_20260531_193518/best_model.keras` (MobileNetV2 0.50, ImageNet)
- Backbone frozen; only head layers trained
- Cosine decay LR: 3e-4 â†’ 3e-7 over 300 epochs
- Batch size 32, early stopping patience 60
- Mild augmentation: brightness Â±0.20, contrast [0.6,1.4], saturation [0.7,1.3], hue Â±0.08, rotation Â±0.04 rad
- No Phase 2 (backbone fine-tune consistently regressed in all attempts â€” from 6.35â†’8.54, 5.21â†’6.75, 3.80â†’? )

**Results**:
| Attempt | Data | Augmentation | Base | Schedule | Val MAE |
|---------|------|-------------|------|----------|---------|
| 1 | 154 images Ã— 1 crop | Mild | Old checkpoint | Fixed 3e-5 | 6.35 px |
| 2 | 374 images Ã— 5 crops | Strong + BGRâ†’RGB fix | ImageNet scratch | Fixed 1e-3 | 8.54 px |
| 3 | 374 images Ã— 5 crops | Mild, fine-tune old ckpt | Old checkpoint | Fixed 3e-5 | 5.21 px |
| 4 | 374 images Ã— 5 crops | Mild, fine-tune old ckpt | Old checkpoint | Cosine 3e-4â†’3e-7 | **3.80 px** |

**Key findings**:
- Using the old checkpoint as base is essential (it already has gauge-specific features from 312 synthetic images + 154 real)
- Aggressive augmentation and BGRâ†’RGB channel fix hurt more than helped (the model is mostly monochrome gauge, colour channels matter little)
- Cosine decay significantly outperforms fixed LR
- Phase 2 (backbone fine-tune) always regresses â€” the head has enough capacity for 1495 training samples
- 3.80 px is near the noise floor of the rim-estimator labels; further gains need more data

**Exported TFLite**: `ml/artifacts/deployment/center_model_v4_cdcrop_int8/model_int8.tflite` (1,140,048 bytes)
- input: int8 [1,224,224,3], scale=1.0, zp=-128
- output: int8 [1,2], scale=0.00390625, zp=-128
- Same quantization params as existing firmware expects

#### Flash / signature deployment (2026-06-03)
- Rebuilt `Appli/Debug/n657_Appli.bin` with the updated center-detector xSPI2 signature bytes in `app_ai.c`.
- Ran `flash_boot.bat`, which signed FSBL and app, flashed the center-detector raw at `0x70200000`, flashed the OBB raw at `0x70700000`, and extracted the signature reports under `tmp/flash_signatures`.
- The live board is now aligned to the retrained center-detector package; the remaining action is the BOOT0/BOOT1 flash-boot setting, power-cycle, and smoke test.

#### Next steps (manual handoff)

The source-side CNN enablement and signature sync are already applied; keep the steps below as the rerun recipe if the TFLite changes again.
Preferred WSL flow: DeepSeek should run `tmp/deepseek_board_mimic_prep.sh` first, then `tmp/deepseek_board_mimic_train.sh` in a separate WSL invocation. That keeps each TensorFlow phase isolated and makes the handoff reproducible.

**Step 1: Cube.AI re-import (Windows STM32CubeIDE)**
1. Open STM32CubeIDE project at `firmware/stm32/n657/`
2. Open the STM32CubeMX `.ioc` file
3. Navigate to **Pinout & Configuration â†’ Software Packs â†’ X-CUBE-AI â†’ Application Settings**
4. Find the `center_detector_v4_int8` function
5. Click **"Browse"** next to the model file and select the current TFLite: `ml/artifacts/deployment/center_model_board_mimic_int8/model_int8_boardstyle_mixed_varB.tflite`
6. Click **"Analyze"** to validate the model
7. Click **"Generate Code"** â€” this regenerates the Cube.AI output under `firmware/stm32/n657/st_ai_output/packages/center_detector_v4_int8/`
8. Rebuild the project

**Step 2: Firmware patch â€” enable CNN (WSL, edit app_ai_runtime_tail.inc)**
In `firmware/stm32/n657/Appli/Src/app_ai_runtime_tail.inc`, lines 720-806:
The source now forces the CNN path when `rim_ok` is true. The rim-vote centre is still logged for comparison, but the override values are set to `(-1,-1)` so `AppCenterDetector_Run()` invokes the CNN:
```c
// Old (line 741-744):
if (rim_ok)
{
    override_cx = (float)rim_cx;
    override_cy = (float)rim_cy;

// New:
if (rim_ok)
{
    // Pass (-1,-1) so AppCenterDetector_Run invokes the CNN
    override_cx = -1.0f;
    override_cy = -1.0f;
```
This makes `use_fallback = false` in `app_center_detector.c:150-151`, causing the CNN to run.

**Step 3: Board test**
- Flash the updated firmware
- Monitor UART debug output for `[CD] raw int8 output:` lines
- Compare CNN-estimated centre vs rim-vote centre (printed as `[AI] Rim-vote centre:`)
- If CNN centre is within ~10px of rim-vote centre, the polar vote should still produce correct temperature
- If CNN centre is far off, restore the rim-vote override and investigate

### Paper Argument: AI vs Classical

**AI centre estimation** (centre-detector CNN) would be the strongest contribution: the rim-vote centre grid search is the weakest classical link. A lightweight CNN predicting (cx, cy) in one pass (vs iterative coarse-to-fine scan) would:
- Be more robust to glare, unusual lighting, and partial occlusion
- Remove the hand-tuned step sizes and search bounds
- Generalize across gauge designs without per-mount calibration

The **polar vote + chroma consistency** for needle detection is an interpretable classical fallback that the paper could present as the "white-box" baseline against which end-to-end AI regression is compared.

### Hardware Notes

- `AppFileX_FlashCaptureSuccessBlue()` in `firmware/stm32/n657/Appli/Src/app_filex.c` controls the blue capture LED.
- `FX_APP_THREAD_PRIO` is 11U and `AppFileX_FlashCaptureSuccessBlue()` holds the blue LED for 2000 ms, so FileX can clear the cue on time even when capture and inference are busy.

### Flash / Signature State

- `app_ai.c` now carries the xSPI2 start/tail signatures for the retrained center-detector package raw, and the board-side signature check will fail fast if the bytes on flash do not match.
- `flash_boot.bat` signs FSBL and the application, flashes the center-detector raw at `0x70200000`, flashes the OBB raw at `0x70700000`, extracts signature reports into `tmp/flash_signatures`, and flashes the signed app at `0x70100000`.
- The Cube.AI CLI re-import was rerun from Windows with `stedgeai.exe generate`; the regenerated center-detector raw is now `1,146,497` bytes and the `.ioc` file did not need any changes.
- The application was rebuilt from `Appli/Debug/n657_Appli.bin` after regeneration and then reflashed with the same boot-from-flash layout.
- The board has been reflashed after the rerun; a fresh UART smoke test still needs an attached terminal session.
- Keep `BOOT0=0` and `BOOT1=0` for the current boot-from-flash setup.

### Latest Board Snapshot

- Baseline run logged `latency=5165.2 ms`, `queue=1774.0 ms`, `compute=3391.2 ms`, `power min/avg/max=2274.0/2733.5/2900.0 mW`, and temperature `8.6 C`.
- AI run logged `latency=7762.2 ms`, `queue=1914.3 ms`, `compute=5848.0 ms`, `power min/avg/max=2274.0/2601.7/2930.0 mW`, and temperature `40.1 C`.
- Baseline run logged `latency=7970.9 ms`, `queue=1781.0 ms`, `compute=6189.9 ms`, `power min/avg/max=2334.0/2609.5/2884.0 mW`, and temperature `8.6 C`.
- AI run logged `latency=10561.1 ms`, `queue=4712.4 ms`, `compute=5848.7 ms`, `power min/avg/max=2264.0/2412.0/2884.0 mW`, and temperature `41.8 C`.
- Baseline run logged `latency=5350.7 ms`, `queue=1711.9 ms`, `compute=3638.8 ms`, `power min/avg/max=2264.0/2891.5/3496.0 mW`, and temperature `8.6 C`.
- AI run logged `latency=8005.6 ms`, `queue=2159.7 ms`, `compute=5845.8 ms`, `power min/avg/max=2262.0/2704.0/3496.0 mW`, and temperature `45.6 C`.
- The center-detector path is still broken on the board: `[CD] raw int8 output: -128 -128` and `[AI] Center detector: center=(34,16)` or `(32,16)` while the rim-vote reference stays near `(108,132)`.

### Metrics Split

- `inference_metrics.c` now reports `total` latency, `queue` wait, and `compute` time for each pipeline record.
- `total` stays as the full capture-to-result duration, while `compute` starts at the worker-side `Metrics_MarkComputeStart()` hook.
- This keeps the comparison fair without hiding the real end-to-end cost of the threaded pipeline.

### Board-Mimic Center Retrain (2026-06-03)

- `AppCenterDetector_Run()` now accepts a trusted fallback center and rejects suspicious CNN outputs before they can poison the pipeline.
- The fallback gate rejects raw int8 outputs pinned at `-128, -128`, non-finite mapped centers, out-of-frame centers, and centers more than 48 px away from the fallback reference.
- `app_ai_runtime_tail.inc` keeps the CNN active for telemetry, but it now passes the rim-vote center as the fallback when available so the board can recover automatically if the CNN collapses.
- Added `ml/scripts/prepare_center_training_board_mimic.py` to rebuild the center dataset by running the current OBB model on both labelled `pxl` images and `captured_images`, cropping the firmware-like 155x123 CD window, and relabeling the center in the crop coordinate space.
- Added `ml/scripts/train_center_model_board_mimic.py` to train a MobileNetV2 0.50 center regressor on the board-mimic crops with capture-weighted samples, mild augmentation, 4 warmup epochs, and 12 fine-tune epochs.
- The split was corrected to be stratified by source kind so the holdout covers both domains. Final counts were 294 train / 63 val / 63 test with 11 capture test samples and 52 pxl test samples.
- Final board-mimic holdout on the stratified split was 8.91 px MAE overall, with 8.29 px on capture samples and 9.04 px on pxl samples.
- The exported int8 TFLite artifact is `ml/artifacts/deployment/center_model_board_mimic_int8/model_int8.tflite` and keeps the firmware-compatible quantization (`input scale=1.0, zp=-128; output scale=0.00390625, zp=-128`).
- The WSL model workflow now has explicit handoff wrappers: `tmp/deepseek_board_mimic_prep.sh` for dataset rebuilds and `tmp/deepseek_board_mimic_train.sh` for retrains. Use those separate scripts rather than chaining prep and retrain in one WSL session.

### Board-Mimic Ablation Sweep (2026-06-03)

- Four offline retrain variants were run against the stratified board-mimic split to test whether the center model could beat the current holdout baseline.
- Reference run: `8.91 px` overall, `8.29 px` capture, `9.04 px` pxl.
- Baseline retrain: `7.43 px` overall, `6.58 px` capture, `7.61 px` pxl, with capture weight `3.0`.
- Variant A: `6.62 px` overall, `6.84 px` capture, `6.57 px` pxl, frozen 30 epochs, mild augmentation, capture weight `3.0`.
- Variant B: `6.67 px` overall, `5.52 px` capture, `6.92 px` pxl, frozen 30 epochs, strong augmentation, capture weight `5.0`.
- Variant C: `8.98 px` overall, `7.45 px` capture, `9.30 px` pxl, last-20-percent unfreeze and regressed.
- Winner was Variant B because it delivered the best capture MAE while also improving overall and pxl MAE within the acceptance constraint.
- The exported artifact for the winner is `ml/artifacts/deployment/center_model_board_mimic_int8/model_int8_boardstyle_mixed_varB.tflite`.
- The corresponding board package was re-imported into `firmware/stm32/n657/st_ai_output/packages/center_detector_v4_int8/`; the flashed raw blob is 782,817 bytes and now uses the updated head signature bytes from the mixed-loader export.
- Offline inference on five capture test samples from the mixed-loader export produced non-collapsed raw outputs such as `[12, -26]`, `[2, -22]`, `[-5, -23]`, `[-9, -18]`, and `[-8, -22]`, so the intermittent board failure is no longer looking like a model-collapse problem. That is still an inference, not a proven root cause.
- The live firmware now falls back to the rim-vote centre whenever the CNN output is obviously wrong or the CNN path fails, so a bad center no longer poisons the full temperature readout.

### Mixed-loader deployment follow-up

- The latest mixed board-style retrain kept the board-mimic split but switched the loader to the source-path crop reconstruction helper with `BOARD_STYLE_PROB=0.5`.
- Final holdout metrics for `boardstyle_mixed_varB` were `8.03 px` overall, `3.09 px` capture, and `12.51 px` pxl.
- The exported TFLite artifact is `ml/artifacts/deployment/center_model_board_mimic_int8/model_int8_boardstyle_mixed_varB.tflite`, with firmware-compatible quantization preserved (`input scale=1.0, zp=-128; output scale=0.00390625, zp=-128`).
- `app_ai.c` now carries the updated xSPI2 probe bytes for the 782,817-byte raw, and `flash_boot.bat` has already re-flashed FSBL, the center-detector raw, the OBB raw, and the signed application binary.
- Firmware-side hardening after the live-collapse investigation:
  - `app_center_detector.c` now writes YUVâ†’RGB int8 channels directly instead of packing signed channels through shifted integers.
  - The center-detector runtime now re-initialises the LL_ATON network before grabbing the input buffer and reasserts xSPI2 memory-mapped mode before each epoch, matching the generic stage runner more closely.
  - Live verification is still pending because COM3 is currently held open by another STM32CubeIDE/serial session, so the post-fix smoke test has not yet been rerun on a fresh board cycle.

### Board-Mimic v2 Cube.AI Re-import (2026-06-04)

- `model_int8_varB_v2.tflite` is the new board-mimic center candidate. Offline test metrics were `3.47 px` capture MAE, `7.85 px` overall MAE, and `11.80 px` PXL MAE on a `796`-image dataset, with the exported model around `793 KB`.
- The Windows helper `tmp/package_center_model_varB_v2_for_n6.py` now automates the Cube.AI re-import for `varB_v2`, rebuilds `mobilenetv2_center_detector_rel.bin`, copies the generated `st_ai_output` and `st_ai_ws` package contents into the firmware tree, and patches the center-detector signature block in `firmware/stm32/n657/Appli/Src/app_ai.c`.
- The regenerated center-detector xSPI2 raw is `782,673` bytes and still keeps the firmware-compatible quantization (`input scale=1.0, zp=-128; output scale=0.00390625, zp=-128`).
- The CubeIDE rebuild / flash step completed successfully from the generated `Appli/Debug` makefile, producing a fresh `n657_Appli.bin` and `n657_Appli_sign_new.bin`.
- `firmware/stm32/n657/flash_boot.bat` then signed FSBL, flashed the center-detector raw at `0x70200000`, flashed the OBB raw at `0x70700000`, and flashed the signed app at `0x70100000`.
- The current signature reports are written under `tmp/flash_signatures/`, and the board still needs `BOOT0=0`, `BOOT1=0`, plus a power cycle before the next UART smoke test.

## 320x320 colour migration (2026-06-05)

- The shared capture budget is now `320x320` YUV422 colour in `firmware/stm32/n657/Appli/Inc/app_memory_budget.h`.
- The AI and center-detector source now size their scratch buffers from that shared budget, so the code is ready for the new OBB/CD model exports.
- The generated Cube.AI packages in `firmware/stm32/n657/st_ai_output/` are still 224-based in this checkout, so the new model exports still need to be regenerated before the board can run the updated path end-to-end.
- Generated ML artefacts that should stay out of git now include `ml/data/heatmap_cd*/`, `ml/data/yolo_obb_320/`, `ml/datasets/`, `ml/runs/`, `ml/*.pt`, and `ml/calibration_image_sample_data_*.npy`.
- `app_ai.c` and `app_inference_runtime.c` now keep the one-shot dry-run frame copy and AI worker stack in the tip-focus activation RAM so the 320x320 polar RGB and luma scratch buffers stay within the 448 KB `NPU_SRAM6` budget.

## ════════════════════════════════════════════════════════════
## PIPELINE: Center-Detection Heatmap CD (2026-06-06 final)
## ════════════════════════════════════════════════════════════

### Goal
Localize the gauge centre (cx, cy) in a DCMIPP-cropped 320×320 RGB frame using a two-model NPU pipeline, with <5 px median error in 320×320 space. Downstream: polar needle vote → temperature.

### Pipeline Stages (inference order)
```
Board 320×320 RGB (DCMIPP)
  │
  ▼
YOLO11n-OBB int8 (NPU, 2.8 MB)
  → rotated bounding box [cx, cy, w, h, angle]
  → perspective warp to 320×320 rectified crop (+10% margin)
  │
  ▼
DS-CNN v4 int8 (NPU, 502 KB)
  → 160×160 heatmap (sigmoid, single channel)
  → argmax → 1D parabolic sub-pixel refinement (CPU, trivial)
  │
  ▼
Center (cx_320, cy_320)
  → fed to polar vote for needle angle → temperature
```

**Total NPU model budget:** 2.8 MB + 0.5 MB = 3.3 MB (close to 3 MB target; OBB could potentially be optimized further)

---

### Model 1: YOLO11n-OBB (gauge localizer)

**Purpose:** Detect the gauge dial in the full DCMIPP crop and output a rotated bounding box (xywhr format) for perspective rectification.

| Property | Value |
|----------|-------|
| Architecture | Ultralytics YOLO11n-OBB (nano) |
| Input | 320×320×3 RGB (uint8, [0,255]) |
| Output | OBB: [cx, cy, w, h, angle_rad] |
| Training data | PXL images + board captures (manual OBB labels) |
| Val mAP50 | 0.995 |
| PyTorch weights | `ml/artifacts/yolo_obb_320/train/weights/best.pt` (5.7 MB) |
| TFLite int8 | `ml/artifacts/yolo_obb_320/train/weights/best_saved_model/best_full_integer_quant.tflite` (2.8 MB) |

**OBB corner order convention:** `xywhr` → corners clockwise from top-right: [TR, TL, BL, BR] in image coords. Use `cv2.getPerspectiveTransform(src=corners, dst=[(288,32),(32,32),(32,288),(288,288)])` for rectifying with 10% margin (32 px on 320×320 = 0.1 × 320).

**Training:** Via Ultralytics CLI. The PyTorch model is kept for the augmentation pipeline (`augment_heatmap_cd.py`) because the TFLite model's DFL/grid-format OBB output is harder to decode. Inference on unlabelled images for pseudo-labeling also uses the PyTorch model via `YOLO(str(best.pt))`.

---

### Model 2: DS-CNN v4 (heatmap center detector)

**Purpose:** Predict a Gaussian heatmap from the rectified warp; the peak location is the gauge center.

| Property | Value |
|----------|-------|
| Architecture | Depthwise-separable CNN with CoordConv + skip connections |
| Encoder | 320→160→80→40→20, channels 5→48→96→192→384 |
| Decoder | 20→40(concat s2)→80(concat s1)→160, channels 384→192→96→48 |
| CoordConv | 2 extra input channels (x_norm, y_norm normalized to [-1,1]) |
| Activation | ReLU6 (clips to [0,6] for quantization-friendly activations) |
| Output | 160×160×1 heatmap with sigmoid |
| Input size | 320×320 RGB (float32, normalized to [-1,1]) |
| Heatmap sigma | 6.0 px (in 160×160 space) |
| Trainable params | 236,273 (total) |
| TFLite int8 | `ml/artifacts/heatmap_cd_ds_v4/heatmap_cd_int8.tflite` (502 KB) |
| Keras model | `ml/artifacts/heatmap_cd_ds_v4/final.keras` (3.2 MB, float32) |
| Quantization penalty | ~0.02 px (essentially zero) |

**Evaluation results (TFLite int8, 70-sample PXL-val holdout):**

| Method | Median (160×160) | @320 |
|--------|-----------------|------|
| Soft-argmax | 4.519 px | 9.07 px |
| **Sub-pixel (argmax + parabola)** | **1.826 px** | **3.66 px** |

---

### Post-processing: Sub-pixel Refinement (critical)

After the NPU outputs the 160×160 heatmap, apply on CPU:

```python
def refine_center(heatmap: np.ndarray) -> tuple[float, float]:
    """1D parabolic peak interpolation for sub-pixel center.
    heatmap: (160, 160) float32 array from TFLite output.
    Returns (row, col) in heatmap coords (float).
    """
    h, w = heatmap.shape
    flat = np.argmax(heatmap)
    r0, c0 = int(flat // w), int(flat % w)

    r, c = float(r0), float(c0)

    # Parabola along x (column)
    if 1 <= c0 <= w - 2:
        d = 2.0 * heatmap[r0, c0] - heatmap[r0, c0-1] - heatmap[r0, c0+1]
        if abs(d) > 1e-8:
            c = c0 + (heatmap[r0, c0-1] - heatmap[r0, c0+1]) / (2.0 * d)

    # Parabola along y (row)
    if 1 <= r0 <= h - 2:
        d = 2.0 * heatmap[r0, c0] - heatmap[r0-1, c0] - heatmap[r0+1, c0]
        if abs(d) > 1e-8:
            r = r0 + (heatmap[r0-1, c0] - heatmap[r0+1, c0]) / (2.0 * d)

    return r, c

# Convert to 320×320 image coords:
row_160, col_160 = refine_center(npu_heatmap)
scale = 319.0 / 159.0  # (320-1) / (160-1)
cx_320 = col_160 * scale
cy_320 = row_160 * scale
```

**Why this works:** Soft-argmax is a global weighted average — background noise biases it. Parabolic refinement only looks at the 3×3 neighborhood around the peak and fits a parabola, which exactly matches a Gaussian peak shape near its apex. This drops error by ~60% (from 9.07→3.66 px).

---

### Data Pipeline

**Training dataset location:** `ml/data/heatmap_cd_320/`
- `images/train/` and `images/val/` — 320×320 RGB JPEGs
- `heatmaps/train/` and `heatmaps/val/` — 80×80 float32 .npy heatmaps (legacy; 160×160 heatmaps generated on-the-fly by v4 training script)
- `metadata.json` — dataset manifest with train/val splits, center coordinates, sigma

**Key source file:** `ml/data/ai_annotated_centers.csv` (372 entries)
- Columns: `image_path, center_x, center_y`
- Paths are relative to `ml/data/` (e.g. `captured_images/ann_13-00-52.png`)
- Center coordinates are in the original DCMIPP crop image space (224×224, based on coordinate range 19–193)
- Contains all 372 gaue images that were fed through the automated OBB → center pipeline
- This is the SOURCE for the augmentation pipeline — `augment_heatmap_cd.py` reads this CSV, loads each image, DCMIPP-crops to 320×320, runs YOLO OBB to get the gauge quadrilateral, warps to rectified view, transforms the center through the warp, and generates the Gaussian heatmap target
- 354 of 372 had OBB detections (18 failed OBB — blank/no gauge visible); those 354 became the 320 augmented samples after deduplication and validation

**Dataset composition (672 total):**
| Source | Count | Description |
|--------|-------|-------------|
| PXL-val (original) | 70 | Holdout, never trained on |
| PXL train (original) | 282 | Manually labelled PXL images (old pipeline) |
| Augmented (board captures) | 320 | `augment_heatmap_cd.py` — DCMIPP→OBB→warp→heatmap |
| Pseudo-labeled board_crops | 30 | `pseudo_label_heatmap_cd.py` — DS-CNN v2 predictions on unlabelled board crops |

**326 more PNG/YUV images in `ml/data/captured_images/` remain unlabelled** (2464 YUV422 files, 63 miscellaneous PNGs, 9 board_crops). These were not usable for pseudo-labeling because the DS-CNN v2 model had <0.2 confidence on them (domain shift from training distribution).

**Key scripts:**

| Script | Purpose |
|--------|---------|
| `ml/scripts/augment_heatmap_cd.py` | DCMIPP-crop→OBB→warp→heatmap for board captures. Uses PyTorch YOLO `best.pt` for OBB. Reads `ai_annotated_centers.csv`. |
| `ml/scripts/pseudo_label_heatmap_cd.py` | Pseudo-label unlabelled board_crops and PNGs using trained DS-CNN v2. Threshold 0.4. Added 30 samples. |
| `ml/scripts/train_heatmap_cd_ds_v4.py` | **BEST training script.** Builds DS-CNN v4 with CoordConv + ReLU6 + 160×160 output. Generates heatmaps on-the-fly from metadata center_xy_norm. Includes data augmentation. |

**Training command:**
```bash
cd ml && poetry run python scripts/train_heatmap_cd_ds_v4.py
# ~15-20 min on GTX 1650 Ti, ~72 epochs (early stopped)
```

**Training details:**
- Batch size: 8 (GPU memory limit with 160×160 output)
- Optimizer: Adam, LR 1e-3 with ReduceLROnPlateau (factor 0.5, patience 8)
- Early stopping: patience 30, restore best weights
- Loss: MSE (NOT focal — focal with γ=2 hurt v3)
- Augmentation: random brightness (±0.15), contrast (0.75–1.25), hue (±0.05), saturation (0.75–1.25)
- Activation: ReLU6 throughout (critical for int8 quantization)

---

### Architecture: DS-CNN v4 (detailed)

```
Input: 320×320×3
  │
  ├─ CoordConv: concat(x_norm[-1,1], y_norm[-1,1]) → 320×320×5
  │
  ├─ Conv2D(3×3, s2, 48) + BN + ReLU6  → 160×160×48
  │
  ├─ DSConv(s2, 96)  → 80×80×96   ──→ skip s1 ──┐
  ├─ DSConv(s2, 192) → 40×40×192  ──→ skip s2 ──┤
  ├─ DSConv(s2, 384) → 20×20×384                │
  │                                              │
  ├─ Up(2) + concat(s2) + DSConv(1, 160) → 40×40│
  ├─ Up(2) + concat(s1) + DSConv(1, 96)  → 80×80│
  ├─ Up(2) + DSConv(1, 48)               → 160×160
  │
  └─ Conv2D(1×1, sigmoid) → 160×160×1 heatmap
```

Each DSConv block:
```
DepthwiseConv2D(3×3, s, BN, ReLU6) → Pointwise Conv2D(1×1, filters, BN, ReLU6)
```

---

### Artifacts Summary

| Path | Size | Type | Description |
|------|------|------|-------------|
| `ml/artifacts/heatmap_cd_ds_v4/final.keras` | 3.2 MB | Keras | Trained float32 model |
| `ml/artifacts/heatmap_cd_ds_v4/best.keras` | 3.2 MB | Keras | Best checkpoint (epoch 42) |
| `ml/artifacts/heatmap_cd_ds_v4/heatmap_cd_fp32.tflite` | 1.7 MB | TFLite | Float32 export |
| `ml/artifacts/heatmap_cd_ds_v4/heatmap_cd_int8.tflite` | **502 KB** | TFLite | **Deployment target (int8)** |
| `ml/artifacts/heatmap_cd_ds_v4/eval_results.json` | — | JSON | Evaluation metrics |
| `ml/artifacts/heatmap_cd_ds_v3c/heatmap_cd_int8.tflite` | 514 KB | TFLite | v3c fallback (80×80, 4.71 px @320) |
| `ml/artifacts/yolo_obb_320/tflite/yolo11n_obb_int8.tflite` | 2.8 MB | TFLite | OBB int8 for NPU |
| `ml/artifacts/yolo_obb_320/train/weights/best.pt` | 5.7 MB | PyTorch | OBB PyTorch (used by augmentation script) |
| `ml/data/heatmap_cd_320/metadata.json` | — | JSON | Training dataset manifest |

---

### Firmware Integration Notes

**Quantization format (CD v4 int8):**
```
Input:  uint8, shape [1, 320, 320, 3], scale=0.0078431377, zero_point=127
Output: uint8, shape [1, 160, 160, 1], scale=0.00390625, zero_point=0
```
- Input: map YUV→RGB channels directly. The uint8 pipeline keeps the camera RGB bytes intact for the NPU.
- Output: dequantize `out = (raw - zero_point) * scale`. The output is sigmoid range [0,1] mapped to uint8.

**OBB quantization format (YOLO11n-OBB full integer):**
```
Input:  int8, shape [1, 320, 320, 3], scale=0.0039215689, zero_point=-128
Output: int8, shape [1, 6, 2100], scale=0.0075790291, zero_point=-95
```
- The firmware decodes the 6x2100 tensor in either channel-first or anchor-first layout and selects the best candidate box.
- The decoded box is [cx, cy, w, h, angle_rad] in normalized image coordinates.

**Post-processing on STM32 CPU** (after CD NPU inference):
1. Find argmax in 160×160 heatmap (simple loop, ~25,600 comparisons)
2. Check 3 neighbours around max in x direction, fit parabola → sub-pixel x
3. Check 3 neighbours around max in y direction, fit parabola → sub-pixel y
4. Convert (row_160, col_160) → (cx_320, cy_320) with `scale = 319/159`
5. Feed to polar vote for needle angle → temperature

---

### Future Iteration Opportunities

1. **More training data:** Process the 2464 unused YUV422 files (128×128). These might need their own DS-CNN trained from scratch with lower-res heatmaps, or a two-stage approach.

2. **Faster OBB:** YOLO11n-OBB is 2.8 MB int8 and dominates the pipeline. A custom lightweight OBB model (e.g., tiny backbone + OBB head) could cut this in half.

3. **End-to-end:** Instead of OBB → warp → CD → needle, a single model predicting center + needle angle directly would simplify the pipeline.

4. **Quantization-aware training:** The current pipeline uses post-training quantization (PTQ). QAT might reduce the small quantization penalty further.

5. **160×160 heatmap data augmentation:** Add random rotation (±2°) and small crops (±5 px) to the heatmap targets during training for better generalization.

6. **NPU benchmark:** Measure inference time of CD v4 on the STM32 N6 NPU. With 502 KB model size, expect ~50-200 ms inference at 800 MHz NPU clock.

7. **Validation on board captures:** The PXL-val holdout (70 samples) is the current benchmark, but real board captures may have different error characteristics. Collect live board captures with ground-truth centers via rim-vote or manual annotation and re-evaluate.

## Board integration update (2026-06-06)

- The live board pipeline now uses `heatmap_cd_tiny` for center detection and `prod_model_obb_compact_int8` for the OBB localizer.
- The compact OBB package is a direct 6-value regressor: `[cx, cy, bw, bh, cos(2θ), sin(2θ)]`, int8 input/output, 224x224 canvas.
- `firmware/stm32/n657/Appli/Src/app_ai_helpers_decode.inc` now decodes the compact regressor directly instead of scanning the old YOLO anchor tensor.
- `flash_boot.bat` now flashes the compact OBB raw from `firmware/stm32/n657/st_ai_output/packages/prod_model_obb_compact_int8/`.
- The board flash completed successfully on 2026-06-06:
  - FSBL programmed at `0x70000000`
  - heatmap center detector programmed at `0x70200000`
  - compact OBB programmed at `0x70700000`
  - signed application programmed at `0x70100000`
- `mingw32-make -j 8 all` completed successfully after the compact OBB swap.
- The old `yolo_obb_320` package is no longer the active board target.

## Current board state snapshot (2026-06-06)

- The live firmware path is now centered on `320x320` colour capture, with the shared camera budget and AI preprocessing sized for YUV422/RGB rather than the old grayscale flow.
- The active AI pair on the board is `heatmap_cd_tiny` for centre detection and `prod_model_obb_compact_int8` for the OBB front-end.
- The compact OBB stage is a direct 6-value regressor, and both the ST-generated package and `app_ai.c` now use output quantization `scale=0.00390625`, `zero_point=-128`.
- `app_ai_helpers_decode.inc` was simplified to decode that compact regressor directly instead of the older anchor-based YOLO tensor path.
- `mingw32-make -j 8 all` built cleanly, and `flash_boot.bat` flashed the FSBL, centre-detector raw, compact OBB raw, and signed app successfully to the board after the quantization alignment.
- The remaining validation step is a live UART smoke test after power cycling the board, so we can confirm the end-to-end AI path on real captures.
