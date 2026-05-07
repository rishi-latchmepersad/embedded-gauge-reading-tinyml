# Retrain Plan For Kimi (Handoff)

**Date:** 2026-05-05  
**Status:** Draft - Ready for Implementation  
**Goal:** Fix the 46°C regression (true 46°C → pred ~36.3°C) without post-hoc calibration

> **Document Purpose:** This is a handoff document for Kimi to execute a focused retraining effort. Follow sections in order. Update `docs/ai-memory.md` with findings.

---

## 1. Lock Reproducible Baseline

### 1.1 Capture Current State

```powershell
# Create baseline snapshot directory
$BASELINE_DIR = "d:/Projects/embedded-gauge-reading-tinyml/tmp/retrain_baseline_snapshot"
New-Item -ItemType Directory -Force -Path $BASELINE_DIR

# Save firmware/app hash
cd d:/Projects/embedded-gauge-reading-tinyml/firmware/stm32/n657
git rev-parse HEAD > $BASELINE_DIR/firmware_hash.txt
git diff --stat >> $BASELINE_DIR/firmware_hash.txt

# Save scalar blob hash
Get-FileHash firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw -Algorithm SHA256 > $BASELINE_DIR/scalar_blob_hash.txt

# Record the 46°C regression
@"
Baseline Regression Target:
- True temperature: 46°C
- Current prediction: ~36.3°C
- Error: -9.7°C (under-reading hot band)
- Calibration status: APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION = 1 (enabled)
- Calibration type: Affine with cold-tail correction
"@ | Out-File -FilePath $BASELINE_DIR/regression_target.txt

# Copy hard-case report
Copy-Item ml/artifacts/hard_cases_predictions.csv $BASELINE_DIR/
Copy-Item ml/artifacts/hard_cases_predictions_classical.csv $BASELINE_DIR/
```

### 1.2 Baseline Artifacts to Preserve

| Component | Current Path | Snapshot Location |
|-----------|--------------|-------------------|
| Firmware hash | `firmware/stm32/n657/` (git) | `tmp/retrain_baseline_snapshot/firmware_hash.txt` |
| Scalar blob | `firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw` | `tmp/retrain_baseline_snapshot/scalar_blob_hash.txt` |
| Hard-case predictions | `ml/artifacts/hard_cases_predictions.csv` | `tmp/retrain_baseline_snapshot/` |
| Calibration params | `firmware/stm32/n657/Appli/Src/app_inference_calibration.c` | Documented in Section 4 |

---

## 2. Rebuild Training Dataset

### 2.1 Data Sources

```
ml/data/labelled/           # Core labeled dataset (CVAT format)
ml/data/captured_images/              # Board captures with metadata
ml/data/captured_images/         # Additional board captures
```

### 2.2 Merge Manifest with Source Tags

Create a unified manifest with the following schema:

```csv
image_path,label,source,crop_mode,brightness_bin,temp_bin
ml/data/labelled/gauge_001.png,25.0,core,fixed,medium,medium
ml/data/labelled/gauge_002.png,30.0,core,fixed,medium,medium
ml/data/captured_images/capture_2026-04-24_22-24-04.png,10.0,hard_case,obb_centered_fallback,low,low
ml/data/captured_images/capture_2026-04-30_12-45-08.png,42.0,board_capture,fixed,high,high
```

**Source tags:**
- `core`: Original labeled dataset from CVAT
- `hard_case`: Known difficult samples (close-ups, extreme temps)
- `board_capture`: Live board captures with verified labels

**Metadata fields:**
- `crop_mode`: `fixed`, `obb`, `obb_centered_fallback`, `rectifier`
- `brightness_bin`: `dark`, `medium`, `bright` (based on mean luma)
- `temp_bin`: `cold` (<0°C), `low` (0-20°C), `mid` (20-35°C), `hot` (35-50°C)

### 2.3 Re-verify Labels for Hot Band (35°C–50°C)

Priority files to re-inspect:
- All captures with true temp > 35°C
- Files where current model under-reads by > 5°C
- Close-up captures from 2026-04-24 and 2026-04-30

```bash
# Generate list of hot-band samples needing verification
cd d:/Projects/embedded-gauge-reading-tinyml/ml
python scripts/_analyze_all.py --filter "label>=35" --output hot_band_verify.csv
```

---

## 3. Training Crop Distribution

### 3.1 Match Firmware Crop Policy

The firmware uses this cascade (from `app_ai.c`):

1. **OBB crop** when OBB output is valid and within `APP_AI_OBB_TRAINING_CROP_MIN_RATIO` (0.60) to `APP_AI_OBB_TRAINING_CROP_MAX_RATIO` (1.25)
2. **OBB-centered training-size fallback** when OBB shape is out-of-window
3. **Fixed training crop** as final fallback

### 3.2 Training Crop Generation Policy

```python
# Pseudocode for crop generation
def generate_training_crop(obb_params, image_shape):
    """
    Generate training crop matching firmware behavior.
    
    Args:
        obb_params: (cx, cy, w, h, angle_deg) or None
        image_shape: (H, W)
    
    Returns:
        crop_box: (x_min, y_min, x_max, y_max) in normalized coords
        crop_mode: str describing the policy used
    """
    TRAINING_CROP_X_MIN = 0.1027
    TRAINING_CROP_X_MAX = 0.7987
    TRAINING_CROP_Y_MIN = 0.2573
    TRAINING_CROP_Y_MAX = 0.8071
    
    if obb_params is not None:
        cx, cy, w, h, angle = obb_params
        
        # Check if OBB shape is within acceptable ratio
        training_w = TRAINING_CROP_X_MAX - TRAINING_CROP_X_MIN
        training_h = TRAINING_CROP_Y_MAX - TRAINING_CROP_Y_MIN
        w_ratio = w / training_w
        h_ratio = h / training_h
        
        if 0.60 <= w_ratio <= 1.25 and 0.60 <= h_ratio <= 1.25:
            # Use OBB crop directly
            return compute_obb_crop(obb_params), "obb"
        else:
            # OBB-centered fallback: use OBB center with training crop size
            x_min = cx - training_w / 2
            x_max = cx + training_w / 2
            y_min = cy - training_h / 2
            y_max = cy + training_h / 2
            return clamp_crop((x_min, y_min, x_max, y_max)), "obb_centered_fallback"
    
    # Final fallback: fixed training crop
    return (TRAINING_CROP_X_MIN, TRAINING_CROP_Y_MIN, 
            TRAINING_CROP_X_MAX, TRAINING_CROP_Y_MAX), "fixed"
```

### 3.3 Persist Crop Boxes

Crop boxes must be pre-computed and saved to ensure train/eval use identical geometry:

```python
# Save crop boxes alongside the manifest
manifest_with_crops = []
for sample in all_samples:
    crop_box, crop_mode = generate_training_crop(sample.obb_params, sample.image_shape)
    manifest_with_crops.append({
        **sample,
        "crop_x_min": crop_box[0],
        "crop_y_min": crop_box[1],
        "crop_x_max": crop_box[2],
        "crop_y_max": crop_box[3],
        "crop_mode": crop_mode
    })

# Save to CSV for reproducibility
save_csv(manifest_with_crops, "ml/data/unified_manifest_with_crops.csv")
```

---

## 4. Experiment Matrix

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | MobileNetV2 | Proven architecture, runs on N6 NPU |
| Input size | 224×224 | Matches current deployment |
| Alpha | 1.0 | Standard width multiplier |
| Seed | Fixed (e.g., 42) | Reproducibility |

### 4.2 Experiment Variants

| Variant | Description | Key Parameters |
|---------|-------------|----------------|
| **A** | Baseline + new crop distribution | `hard_case_repeat=0`, standard loss |
| **B** | A + hard-case oversampling | `hard_case_repeat=3` (or tuned) |
| **C** | B + hot-band loss weighting | Add temperature-aware loss weight |
| **D** | C + lower LR fine-tune stage | Stage 2: LR=1e-5 for last 20% epochs |

### 4.3 Hot-Band Loss Weighting (Variant C)

```python
def temperature_aware_loss_weight(true_temp):
    """
    Up-weight hot band (35-50°C) samples in loss.
    """
    if 35 <= true_temp <= 50:
        return 2.0  # Double weight for hot band
    elif true_temp < 0:
        return 1.5  # 1.5x for cold band
    else:
        return 1.0  # Normal weight for mid range

# In training loop
loss = mse_loss(pred, true) * temperature_aware_loss_weight(true_temp)
```

### 4.4 Two-Stage Training (Variant D)

```python
# Stage 1: Standard training
config_stage1 = TrainConfig(
    epochs=int(0.8 * total_epochs),
    learning_rate=1e-4,
    # ... other params
)

# Stage 2: Fine-tune with lower LR
config_stage2 = TrainConfig(
    epochs=int(0.2 * total_epochs),
    learning_rate=1e-5,
    init_model_path=stage1_output_path,
    # ... other params
)
```

---

## 5. Selection Criteria

### 5.1 Primary Metrics (Must Pass All)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Hard-case MAE | < 5.0°C | Core requirement from AGENTS.md |
| **Per-band MAE** | **< 4.0°C each** | Must work across full gauge range |
| - Cold band (-30 to 0°C) | < 4.0°C | 5 hard cases: -30, -19, -18, -10, 0 |
| - Low band (0-20°C) | < 4.0°C | 5 hard cases: 5, 10, 18, 19, 20 |
| - Mid band (20-35°C) | < 4.0°C | 3 hard cases: 22, 30, 35 |
| - Hot band (35-50°C) | < 3.5°C | 5 hard cases: 45, 45, 46, 50 |
| Hard-case max abs error | < 8.0°C | Prevent any catastrophic failures |
| No calibration needed | Raw output meets above | Remove calibration dependency |

**Critical:** The model must work across the ENTIRE gauge range (-30°C to 50°C), not just at 46°C. Each temperature band must independently meet its MAE threshold.

### 5.2 Secondary Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Overall validation MAE | < 2.0°C | General accuracy |
| Stability (3 reruns) | σ < 0.5°C | Reproducibility |

### 5.3 Evaluation Protocol

```bash
# For each trained model, run:
python scripts/eval_keras_on_manifest.py \
    --model $MODEL_PATH \
    --manifest ml/data/hard_cases_manifest.csv \
    --output $RESULTS_DIR/eval_hard_cases.json

# Check hot band specifically
python scripts/eval_keras_on_manifest.py \
    --model $MODEL_PATH \
    --manifest ml/data/hot_band_manifest.csv \
    --output $RESULTS_DIR/eval_hot_band.json
```

---

## 6. Remove Calibration by Design

### 6.1 Current Calibration (to be disabled)

From `app_inference_calibration.c`:

```c
// Current affine calibration parameters
static const float kCalibrationAffineScale = 1.0502802133560180f;
static const float kCalibrationAffineBias = 0.6553916335105896f;
```

### 6.2 Evaluation Without Calibration

During model selection, evaluate with:

```c
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0
```

Only select models that meet criteria 5.1 with raw model output.

### 6.3 Keep Calibration for Emergency Rollback

Calibration code remains in firmware but gated:

```c
#ifndef APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0  // Disabled for release candidate
#endif
```

---

## 7. Export and Package

### 7.1 Export Flow

```bash
# 1. Export selected Keras model to TFLite (int8 quantized)
cd d:/Projects/embedded-gauge-reading-tinyml/ml
python scripts/export_tflite.py \
    --model $SELECTED_MODEL_PATH \
    --output artifacts/deployment/${MODEL_NAME}_int8/model_int8.tflite

# 2. Package for STM32N6
python scripts/package_scalar_model_for_n6.py \
    --model artifacts/deployment/${MODEL_NAME}_int8/model_int8.tflite \
    --output-dir artifacts/runtime/${MODEL_NAME}_int8_reloc \
    --workspace-dir firmware/stm32/n657/st_ai_output/packages/${MODEL_NAME}/st_ai_ws

# 3. Sync canonical file
Copy-Item artifacts/runtime/${MODEL_NAME}_int8_reloc/atonbuf.xSPI2.raw \
    firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw
```

### 7.2 Avoid Blob/Package Mismatch

Critical: Use the `*_atonbuf.xSPI2.raw` file that matches the package, not a stale copy.

```powershell
# Verify hash before flashing
Get-FileHash firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw
# Should match: artifacts/runtime/${MODEL_NAME}_int8_reloc/atonbuf.xSPI2.raw
```

---

## 8. Board Validation Protocol

### 8.1 Fixed Checkpoints

Test at these true temperatures:
- `-25°C`, `-10°C`, `0°C`, `20°C`, `30°C`, `35°C`, `42°C`, `46°C`, `50°C`

### 8.2 Capture Protocol

For each checkpoint:
1. Set gauge to target temperature
2. Wait 2 minutes for thermal stabilization
3. Capture at least 5 frames
4. Record: mean error, MAE, bias, std, max error

### 8.3 UART Log Format

```
[VALIDATION] Checkpoint: 46C
[VALIDATION] Frame 1: true=46.0C, pred=45.2C, err=-0.8C
[VALIDATION] Frame 2: true=46.0C, pred=46.1C, err=+0.1C
...
[VALIDATION] Summary: n=5, mean_err=-0.3C, MAE=0.5C, bias=-0.3C, std=0.4C, max_err=0.8C
```

### 8.4 Per-Band Validation Report

After testing all checkpoints, generate a per-band summary:

```
[VALIDATION] Band Summary:
[VALIDATION] Cold (-30-0°C):  n=5, MAE=2.1C, max_err=3.8C [PASS]
[VALIDATION] Low (0-20°C):    n=5, MAE=1.8C, max_err=3.2C [PASS]
[VALIDATION] Mid (20-35°C):   n=3, MAE=2.5C, max_err=4.1C [PASS]
[VALIDATION] Hot (35-50°C):   n=5, n=5, MAE=2.9C, max_err=4.5C [PASS]
[VALIDATION] Overall:         n=18, MAE=2.3C, max_err=4.5C [PASS]
```

**All bands must PASS independently.**

---

## 9. Commands Reference (WSL GPU Flow)

### 9.1 Pre-session Setup

```powershell
# Restart WSL (critical - prevents hangs)
wsl --shutdown
Start-Sleep 5

# Verify GPU availability
wsl -d Ubuntu bash -c "nvidia-smi"
```

### 9.2 Training

```bash
# In WSL
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

# Variant A: Baseline with new crop distribution
python scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs 100 \
    --seed 42 \
    --manifest data/unified_manifest_with_crops.csv \
    --output-dir artifacts/training/variant_a_baseline

# Variant B: Add hard-case oversampling
python scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs 100 \
    --seed 42 \
    --manifest data/unified_manifest_with_crops.csv \
    --hard-case-manifest data/hard_cases_manifest.csv \
    --hard-case-repeat 3 \
    --output-dir artifacts/training/variant_b_hardcase_boost

# Variant C: Add hot-band weighting (requires custom training script)
python scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs 100 \
    --seed 42 \
    --manifest data/unified_manifest_with_crops.csv \
    --hard-case-manifest data/hard_cases_manifest.csv \
    --hard-case-repeat 3 \
    --enable-temp-weighting \
    --output-dir artifacts/training/variant_c_hot_weighted

# Variant D: Two-stage with lower LR fine-tune
python scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs 80 \
    --seed 42 \
    --manifest data/unified_manifest_with_crops.csv \
    --hard-case-manifest data/hard_cases_manifest.csv \
    --hard-case-repeat 3 \
    --enable-temp-weighting \
    --output-dir artifacts/training/variant_d_stage1

python scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs 20 \
    --learning-rate 1e-5 \
    --seed 42 \
    --init-model-path artifacts/training/variant_d_stage1/model.keras \
    --manifest data/unified_manifest_with_crops.csv \
    --output-dir artifacts/training/variant_d_final
```

### 9.3 Evaluation

```bash
# Evaluate on hard cases
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
python scripts/eval_keras_on_manifest.py \
    --model artifacts/training/variant_a_baseline/model.keras \
    --manifest data/hard_cases_manifest.csv

# Evaluate on hot band specifically
python scripts/eval_keras_on_manifest.py \
    --model artifacts/training/variant_a_baseline/model.keras \
    --manifest data/hot_band_manifest.csv
```

### 9.4 Packaging and Flashing

```powershell
# Package the selected model
$MODEL_NAME = "variant_d_final"
cd d:/Projects/embedded-gauge-reading-tinyml/ml
python scripts/package_scalar_model_for_n6.py `
    --model artifacts/training/$MODEL_NAME/model_int8.tflite `
    --output-dir artifacts/runtime/${MODEL_NAME}_int8_reloc

# Sync canonical file
Copy-Item artifacts/runtime/${MODEL_NAME}_int8_reloc/atonbuf.xSPI2.raw `
    firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw

# Flash (from firmware/stm32/n657 directory)
cd d:/Projects/embedded-gauge-reading-tinyml/firmware/stm32/n657
# Set FLASH_MODEL=1 in flash_boot.bat, then run:
.\flash_boot.bat
```

---

## 10. Definition of Done

### Full-Range Accuracy (All Must Pass)
- [ ] **Cold band** (-30 to 0°C): MAE < 4.0°C, max error < 8.0°C
- [ ] **Low band** (0-20°C): MAE < 4.0°C, max error < 8.0°C  
- [ ] **Mid band** (20-35°C): MAE < 4.0°C, max error < 8.0°C
- [ ] **Hot band** (35-50°C): MAE < 3.5°C, max error < 8.0°C
- [ ] **Overall** hard-case MAE < 5.0°C without calibration

### Artifacts & Documentation
- [ ] Model + blob + firmware package hashes recorded in `tmp/retrain_baseline_snapshot/`
- [ ] One-page release note with exact commands and artifact paths
- [ ] Calibration code disabled (`APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0`) in release candidate
- [ ] All validation checkpoints pass (Section 8.1)

**Critical:** The model must work across the ENTIRE gauge range (-30°C to 50°C), not just at 46°C. Each band must independently meet its threshold.

---

## 11. Release Note Template

```markdown
## Release: Gauge CNN Retrain vYYYY.MM.DD

### Artifacts
- Model: `ml/artifacts/training/variant_d_final/model.keras`
- TFLite: `ml/artifacts/deployment/variant_d_final_int8/model_int8.tflite`
- Scalar blob: `firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw`
- Firmware: `firmware/stm32/n657/Appli/Debug/n657_Appli.bin`

### Hashes
- Model SHA256: `abc123...`
- Blob SHA256: `def456...`
- Firmware git: `githash...`

### Performance
- Hard-case MAE: X.XX°C
- Hot band MAE: X.XX°C
- 46°C checkpoint error: ±X.X°C

### Commands
```bash
# Flash
wsl --shutdown
cd firmware/stm32/n657
# Edit flash_boot.bat: FLASH_MODEL=1
.\flash_boot.bat
```

### Calibration Status
Disabled (`APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0`)
```

---

## Appendix: File Paths Summary

| Purpose | Path |
|---------|------|
| Baseline snapshot | `d:/Projects/embedded-gauge-reading-tinyml/tmp/retrain_baseline_snapshot/` |
| Unified manifest | `ml/data/unified_manifest_with_crops.csv` |
| Training outputs | `ml/artifacts/training/variant_*/` |
| Deployment TFLite | `ml/artifacts/deployment/*_int8/` |
| Runtime packages | `ml/artifacts/runtime/*_reloc/` |
| Canonical blob | `firmware/stm32/n657/st_ai_output/atonbuf.xSPI2.raw` |
| Calibration config | `firmware/stm32/n657/Appli/Src/app_inference_calibration.c` |
| Flash script | `firmware/stm32/n657/flash_boot.bat` |
