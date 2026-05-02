# Range-Aware Training Experiment

This document describes the new training experiment to build a more sustainable gauge-reading model.

## Problem Statement

The current firmware calibrations are tactical and not sustainable. We need to fix the model itself to reduce calibration to a tiny residual correction.

## Solution Overview

The experiment implements three key improvements:

1. **Range-aware sampling**: Oversample cold/hot tail regions for better range coverage
2. **Linear output head**: Unbounded regression without saturating activation
3. **Enhanced augmentation**: Crop jitter and brightness/exposure matching board reality

## Changes Made

### 1. Range-Aware Sampling (`training.py`)

Added `_range_aware_weight()` and `_compute_range_aware_weights()` functions that:
- Oversample cold tail (default 15% of range) by 3x
- Oversample hot tail (default 15% of range) by 3x
- Keep middle region at weight=1.0

This ensures the model sees more examples from the extremes where calibration is most critical.

### 2. Linear Output Head (`models.py`)

Added `linear_output` parameter to `build_mobilenetv2_regression_model()`:
- When `linear_output=True`: Uses linear Dense(1) without activation
- When `linear_output=False`: Uses sigmoid activation (default, backward compatible)

Linear output allows unbounded regression, which is then calibrated post-training.

### 3. Enhanced Augmentation (`training.py`)

Updated `_augment_image()` with:
- **Crop jitter**: 5% max offset to simulate dial position variations
- **Brighter brightness range**: ±20% (standard) / ±35% (heavy)
- **Wider contrast range**: (0.75, 1.25) / (0.55, 1.45)
- **Gamma augmentation**: Simulates non-linear camera exposure response
- **Higher glare rate**: 25% (was 20%)

### 4. Post-Training Calibration (`training.py`)

Added `_fit_calibration()` function that:
- Fits affine transformation: `output = slope * prediction + bias`
- Uses sklearn LinearRegression on model predictions vs ground truth
- Stores calibration parameters in model metadata
- Updates test metrics with calibrated MAE

### 5. Training Configuration (`training.py`)

Added new config options to `TrainConfig`:
- `range_aware_sampling: bool = False`
- `cold_tail_fraction: float = 0.15`
- `hot_tail_fraction: float = 0.15`
- `oversampling_factor: float = 3.0`
- `linear_output: bool = False`

### 6. Training Script (`scripts/run_training_range_aware.py`)

New script with CLI arguments:
- `--range-aware-sampling`: Enable range-aware sampling
- `--linear-output`: Use linear output head
- `--cold-tail-fraction`: Cold tail fraction (default 0.15)
- `--hot-tail-fraction`: Hot tail fraction (default 0.15)
- `--oversampling-factor`: Oversampling factor (default 3.0)

## Usage

### Run Training

```bash
# WSL
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
poetry run python scripts/run_training_range_aware.py \
    --model-family mobilenet_v2 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --range-aware-sampling \
    --linear-output \
    --artifacts-dir artifacts/training/range_aware_linear \
    --run-name range_aware_linear_20260502
```

### Run with Bash Script

```bash
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
bash run_training_range_aware_wsl.sh
```

## Expected Results

With range-aware sampling and linear output:
- **Cold/hot tails**: Better coverage → lower MAE at extremes
- **Linear output**: Unbounded predictions → smaller calibration residuals
- **Enhanced augmentation**: More board-like training data → better generalization

## Calibration

After training with `linear_output=True`, the model outputs need calibration:

```python
# Calibration parameters stored in model.metadata["calibration"]
{
    "slope": 1.0234,
    "bias": -1.2345,
    "method": "affine_fit"
}

# Apply calibration: calibrated_value = slope * prediction + bias
```

## Next Steps

1. Run the training experiment
2. Evaluate on hard cases
3. Compare MAE before/after calibration
4. Export TFLite model with calibration
5. Deploy to STM32N6 and validate

## Files Modified

- `ml/src/embedded_gauge_reading_tinyml/models.py`: Added `linear_output` parameter
- `ml/src/embedded_gauge_reading_tinyml/training.py`: 
  - Added range-aware sampling functions
  - Enhanced augmentation
  - Added calibration refitting
  - Updated config and training pipeline
- `ml/scripts/run_training_range_aware.py`: New training script
- `ml/run_training_range_aware_wsl.sh`: New bash script
