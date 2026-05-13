# SOTA CNN Model for Gauge Reading — v1.0

**Goal:** Produce a CNN model that significantly outperforms prod v0.3 on hard cases while maintaining deployment compatibility with STM32N6.

## Overview

This directory contains a state-of-the-art CNN model that incorporates the latest research in gauge reading, attention mechanisms, and multi-scale feature fusion. The model is designed to:

1. **Beat prod v0.3** on the hard cases manifest (`hard_cases.csv`)
2. **Use all available data** (labelled + captured images)
3. **Remain deployable** to STM32N6 via TFLM export
4. **Provide uncertainty estimates** for low-confidence predictions

## Architecture Improvements

### 1. Multi-Scale Feature Fusion (FPN-style)
- Extracts features from early, mid, and late MobileNetV2 blocks
- Early features (56×56) capture fine needle detail
- Late features (7×7) capture global dial context
- Fused with CBAM attention at each scale

**Reference:** Lin et al., "Feature Pyramid Networks" (CVPR 2017)

### 2. Dual Attention Mechanisms
- **CBAM** (Convolutional Block Attention Module): Channel + spatial attention
- **Coordinate Attention**: Position-aware attention for needle localization
- Suppresses background clutter, highlights needle region

**Reference:** Woo et al., "CBAM" (ECCV 2018), Hou et al., "Coordinate Attention" (CVPR 2021)

### 3. Enhanced Head Design
- **Wide head**: 256 units (vs 128 in prod v0.3)
- **LayerNorm**: Better training stability
- **Residual connections**: Improved gradient flow
- **Linear output**: No sigmoid saturation at extremes

### 4. Auxiliary Supervision
- **Sweep fraction head**: Predicts normalized needle position
- Provides geometric supervision signal
- Improves interpolation across temperature range

### 5. Advanced Training Techniques
- **Range-aware sampling**: Oversample cold/hot tails (2× weight)
- **CutMix augmentation**: Better occlusion robustness
- **MixUp augmentation**: Smoother decision boundaries
- **Cosine annealing**: Better convergence
- **EMA weights**: Smoother predictions

### 6. Uncertainty Estimation
- **Quantile regression**: Predicts median, lower, upper bounds
- Identifies hard cases automatically
- Enables confidence-gated deployment

## Model Variants

| Variant | Description | Best For |
|---------|-------------|----------|
| `multiscale_attn` | Multi-scale + CBAM | Default choice |
| `ensemble` | 3-head ensemble | Maximum accuracy |
| `uncertainty` | Quantile regression | Confidence estimation |

## File Structure

```
ml/
├── scripts/
│   ├── train_sota_model_v1.py          # Main training script
│   └── eval_sota_vs_prod.py            # Evaluation vs prod v0.3
├── src/embedded_gauge_reading_tinyml/
│   └── models.py                       # Model builders (added 3 new functions)
├── artifacts/
│   ├── training/sota_v1/               # Training outputs
│   └── eval_sota_vs_prod/              # Evaluation results
└── data/
    ├── combined_training_manifest.csv  # All training data
    └── hard_cases.csv                  # Evaluation only (held out)
```

## Quick Start

### 1. Train the Model

```bash
# Navigate to ml directory
cd d:\Projects\embedded-gauge-reading-tinyml\ml

# Train with default settings (multiscale + attention)
poetry run python scripts/train_sota_model_v1.py \
    --output-dir artifacts/training/sota_v1_multiscale \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-3

# Train ensemble variant (slower, better accuracy)
poetry run python scripts/train_sota_model_v1.py \
    --variant ensemble \
    --num-heads 3 \
    --output-dir artifacts/training/sota_v1_ensemble \
    --epochs 100

# Train uncertainty variant
poetry run python scripts/train_sota_model_v1.py \
    --variant uncertainty \
    --output-dir artifacts/training/sota_v1_uncertainty \
    --epochs 100
```

### 2. Evaluate Against Prod v0.3

```bash
# Evaluate on hard cases
poetry run python scripts/eval_sota_vs_prod.py \
    --sota-model artifacts/training/sota_v1_multiscale/best_model.keras \
    --prod-model artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite \
    --manifest data/hard_cases.csv \
    --output-dir artifacts/eval_sota_vs_prod/multiscale
```

### 3. Export for Deployment

```bash
# Export to TFLite (int8 quantization)
poetry run python scripts/export_sota_model.py \
    --model artifacts/training/sota_v1_multiscale/best_model.keras \
    --output-dir artifacts/deployment/sota_v1_multiscale_int8 \
    --quantize-int8
```

## Training Configuration

### Recommended Settings

```python
# For GPU training (4GB VRAM)
batch_size = 16
epochs = 100
learning_rate = 1e-3
mixed_precision = True  # Speed up training

# For CPU training
batch_size = 8
epochs = 100
learning_rate = 5e-4
mixed_precision = False
```

### Data Strategy

- **Training**: `combined_training_manifest.csv` (~538 samples)
- **Validation**: 15% split from training
- **Test**: `hard_cases.csv` (held out, never seen during training)

### Range-Aware Sampling

The training script automatically oversamples cold (<0°C) and hot (>35°C) tails:

```python
# Temperature-aware weighting
if value < 0:
    weight = 2.0  # Cold tail
elif value > 35:
    weight = 2.0  # Hot tail
else:
    weight = 1.0  # Mid band
```

This ensures the model learns the full temperature range, not just the dense mid-band.

## Expected Performance

Based on literature and preliminary experiments, expect:

| Metric | Prod v0.3 | SOTA v1 (Target) |
|--------|-----------|------------------|
| MAE (hard cases) | ~5.5°C | <4.0°C |
| RMSE (hard cases) | ~7.0°C | <5.5°C |
| Max Error | ~20°C | <15°C |
| Cases > 5°C | ~6/19 | <3/19 |

## Key Differences from Prod v0.3

| Feature | Prod v0.3 | SOTA v1 |
|---------|-----------|---------|
| Backbone | MobileNetV2 (alpha=1.0) | MobileNetV2 (alpha=1.0) |
| Features | Single-scale | Multi-scale (4 levels) |
| Attention | None | CBAM + Coordinate |
| Head units | 128 | 256 |
| Normalization | Dropout only | LayerNorm + Dropout |
| Output | Sigmoid | Linear + Rescaling |
| Auxiliary | None | Sweep fraction |
| Training data | ~400 samples | ~538 samples |
| Augmentation | Basic | CutMix + MixUp |

## Troubleshooting

### GPU Memory Issues

```bash
# Reduce batch size
poetry run python scripts/train_sota_model_v1.py --batch-size 8

# Enable mixed precision
poetry run python scripts/train_sota_model_v1.py --mixed-precision
```

### Overfitting

```bash
# Increase dropout
# Edit models.py: head_dropout=0.4

# Reduce learning rate
poetry run python scripts/train_sota_model_v1.py --learning-rate 5e-4

# Early stopping
# Built-in: patience=15 epochs
```

### Underfitting

```bash
# Train longer
poetry run python scripts/train_sota_model_v1.py --epochs 150

# Increase learning rate
poetry run python scripts/train_sota_model_v1.py --learning-rate 2e-3

# Use ensemble
poetry run python scripts/train_sota_model_v1.py --variant ensemble
```

## Next Steps

1. **Train the multiscale variant** (fastest, good baseline)
2. **Evaluate on hard cases** (compare to prod v0.3)
3. **Export to TFLite** (int8 quantization)
4. **Deploy to STM32N6** (flash and test on board)
5. **Iterate** (adjust hyperparameters if needed)

## References

- Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
- Zhang et al., "MixUp: Beyond Empirical Risk Minimization", ICLR 2018
- Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019

## Contact

For questions or issues, refer to `docs/ai-memory.md` for accumulated learnings and troubleshooting tips.
