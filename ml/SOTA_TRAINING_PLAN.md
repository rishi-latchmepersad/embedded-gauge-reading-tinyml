# SOTA Model Training Plan

**Date:** 2026-05-12  
**Goal:** Train a CNN model that beats prod v0.3 on hard cases  
**Target MAE:** <4.0°C on `hard_cases.csv` (prod v0.3: ~5.5°C)

---

## Phase 1: Baseline Training (Week 1)

### 1.1 Data Preparation ✓
- [x] Combine all labelled data (`ml/data/labelled/*.zip`)
- [x] Add board captures (`ml/data/captured_images/`)
- [x] Create `combined_training_manifest.csv` (~538 samples)
- [x] Verify hard cases are held out (`hard_cases.csv`)

### 1.2 Train Multiscale Model
```bash
cd ml
poetry run python scripts/train_sota_model_v1.py \
    --variant multiscale_attn \
    --output-dir artifacts/training/sota_v1_multiscale \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --seed 42
```

**Expected duration:** 2-3 hours on GPU, 8-10 hours on CPU

**Success criteria:**
- Validation MAE < 4.5°C
- Test MAE (hard cases) < 5.0°C
- No overfitting (val loss tracks train loss)

### 1.3 Initial Evaluation
```bash
poetry run python scripts/eval_sota_vs_prod.py \
    --sota-model artifacts/training/sota_v1_multiscale/best_model.keras \
    --prod-model artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite \
    --manifest data/hard_cases.csv
```

**Decision point:**
- If MAE < 4.0°C → proceed to export
- If MAE 4.0-5.0°C → try ensemble variant
- If MAE > 5.0°C → debug (see troubleshooting)

---

## Phase 2: Model Refinement (Week 2)

### 2.1 Ensemble Training (if needed)
```bash
poetry run python scripts/train_sota_model_v1.py \
    --variant ensemble \
    --num-heads 3 \
    --output-dir artifacts/training/sota_v1_ensemble \
    --epochs 100 \
    --batch-size 12
```

**Expected improvement:** 10-15% MAE reduction

### 2.2 Uncertainty Model (optional)
```bash
poetry run python scripts/train_sota_model_v1.py \
    --variant uncertainty \
    --output-dir artifacts/training/sota_v1_uncertainty \
    --epochs 100
```

**Use case:** Confidence-gated deployment, hard case detection

### 2.3 Hyperparameter Tuning

| Parameter | Try | Expected Impact |
|-----------|-----|-----------------|
| Learning rate | 5e-4, 2e-3 | Convergence speed |
| Batch size | 8, 24 | Gradient stability |
| Dropout | 0.2, 0.4 | Overfitting control |
| Epochs | 150, 200 | Final accuracy |

---

## Phase 3: Export & Deployment (Week 3)

### 3.1 TFLite Export
```bash
# Export to float32 first (debugging)
poetry run python scripts/export_sota_model.py \
    --model artifacts/training/sota_v1_multiscale/best_model.keras \
    --output-dir artifacts/deployment/sota_v1_float32

# Export to int8 (deployment)
poetry run python scripts/export_sota_model.py \
    --model artifacts/training/sota_v1_multiscale/best_model.keras \
    --output-dir artifacts/deployment/sota_v1_int8 \
    --quantize-int8
```

### 3.2 Offline Validation
```bash
# Validate int8 model
poetry run python scripts/eval_sota_vs_prod.py \
    --sota-model artifacts/deployment/sota_v1_int8/model_float32.keras \
    --prod-model artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite \
    --manifest data/hard_cases.csv
```

**Acceptance criteria:**
- Int8 MAE < 4.5°C (slight degradation acceptable)
- Max error < 15°C
- Cases > 5°C: < 4/19

### 3.3 Board Deployment
1. Copy model to firmware:
   ```bash
   cp artifacts/deployment/sota_v1_int8/model_int8.tflite \
      firmware/stm32/n657/st_ai_output/packages/sota_v1_int8/
   ```

2. Rebuild firmware in STM32CubeIDE

3. Flash to board:
   ```powershell
   cd firmware/stm32/n657
   .\flash_boot.bat FLASH_MODEL=1 FLASH_APP=1
   ```

4. Test on live board captures

---

## Phase 4: Validation & Iteration (Week 4)

### 4.1 Board Testing
- Capture 10-20 new board images across temperature range
- Compare SOTA predictions vs classical baseline
- Document failure modes

### 4.2 Hard Case Analysis
```bash
# Analyze worst predictions
poetry run python scripts/analyze_hard_cases.py \
    --predictions artifacts/eval_sota_vs_prod/sota_predictions.csv \
    --output-dir artifacts/analysis/hard_cases
```

**Focus areas:**
- Temperature bands with highest error
- Common visual patterns (glare, close-up, low contrast)
- Recapture candidates

### 4.3 Final Decision
- **If SOTA beats prod v0.3 by >20% on MAE:** Promote to production
- **If SOTA beats prod v0.3 by 10-20%:** Deploy as fallback option
- **If SOTA < 10% better:** Keep prod v0.3, iterate on SOTA

---

## Troubleshooting Guide

### Problem: Training hangs on GPU
**Solution:**
```bash
# Restart WSL
wsl --shutdown

# Preflight GPU check
nvidia-smi

# Train on CPU instead
poetry run python scripts/train_sota_model_v1.py --device cpu
```

### Problem: Overfitting (val loss increases)
**Solutions:**
1. Increase dropout: Edit `models.py`, set `head_dropout=0.4`
2. Reduce learning rate: `--learning-rate 5e-4`
3. Add more augmentation: Edit `_augment_image()` in training script
4. Early stopping: Built-in (patience=15)

### Problem: Underfitting (train loss high)
**Solutions:**
1. Train longer: `--epochs 150`
2. Increase learning rate: `--learning-rate 2e-3`
3. Use pretrained backbone: `--pretrained` (default)
4. Make backbone trainable: Already default

### Problem: MAE > 5.0°C on hard cases
**Diagnostic steps:**
1. Check worst predictions:
   ```bash
   poetry run python scripts/analyze_worst_cases.py \
       --predictions artifacts/eval_sota_vs_prod/sota_predictions.csv
   ```

2. Inspect crop boxes:
   ```bash
   poetry run python scripts/visualize_crops.py \
       --manifest data/hard_cases.csv \
       --output-dir tmp/hard_case_crops
   ```

3. Verify data quality:
   - Are labels correct?
   - Are images corrupted?
   - Is crop geometry correct?

### Problem: Int8 export degrades accuracy
**Solutions:**
1. Use float32 export (larger, better accuracy)
2. Try post-training calibration:
   ```bash
   poetry run python scripts/calibrate_int8.py \
       --model artifacts/deployment/sota_v1_int8/model_int8.tflite \
       --manifest data/combined_training_manifest.csv
   ```
3. Use ensemble (more robust to quantization)

---

## Success Metrics

| Metric | Target | Stretch Goal |
|--------|--------|---------------|
| MAE (hard cases) | < 4.0°C | < 3.5°C |
| RMSE (hard cases) | < 5.5°C | < 4.5°C |
| Max Error | < 15°C | < 12°C |
| Cases > 5°C | < 4/19 | < 2/19 |
| Cases > 10°C | 0/19 | 0/19 |

---

## Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Baseline training | `sota_v1_multiscale` model |
| 2 | Refinement | `sota_v1_ensemble` (if needed) |
| 3 | Export | Int8 TFLite model |
| 4 | Validation | Board deployment + report |

---

## Notes

- **Always restart WSL** before training: `wsl --shutdown`
- **Use bash scripts** for long-running jobs
- **Tail logs** to detect hangs early
- **Document findings** in `docs/ai-memory.md`
- **Hard cases are for evaluation only** — never train on them

---

## Contact

For questions, refer to:
- `SOTA_MODEL_README.md` — Architecture details
- `docs/ai-memory.md` — Accumulated learnings
- `firmware/stm32/n657/flash_boot.bat` — Flash boot script and board-mode notes
