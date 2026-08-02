# Session summary — 2026-07-23

## What was built

1. **QAT-safe ellipse detector** with linear radius head (`gauge_ellipse_qat_linear_v1`, 1.2MB int8)
   - Conv+BN+ReLU encoder, multi-head output (center sigmoid, radius LINEAR, confidence sigmoid)
   - Radius varies correctly in int8 (0.194-0.200 vs GT 0.195) — first model to do this

2. **Center/tip UNet v12** (`gauge_center_tip_v12`, 379KB int8)
   - v9 architecture + Wing Loss + geometric augmentation (rotation + hflip)
   - 88.7% center ≤8px, 73.2% tip ≤8px on 97-image test set
   - Best center accuracy achieved

3. **Full pipeline on test_3**: 100% center, 63.6% tip with predicted ellipse + v2 center/tip model

## Key findings

### QAT-safe architecture is mandatory
- Bias-only convolutions (no BatchNorm) collapse to constant output after int8 quantization
- Conv2D + BatchNormalization + ReLU produces meaningful int8 outputs
- All future models MUST use this pattern

### Linear radius head solves int8 radius collapse
- Sigmoid output wastes int8 precision on unused [0,1] range (step 0.0039 > variation 0.0016)
- Linear output lets quantization grid cover actual radius range (step ~0.0004)
- First ellipse model to preserve radius variation in int8

### Wing Loss improves keypoint localization
- Better than focal MSE for sub-pixel keypoint localization
- Center accuracy: 86.6% → 88.7% (+2.1%)
- Added to `src/embedded_gauge_reading_tinyml/heatmap_losses.py`

### Crop pipeline consistency is critical
- Center/tip model only works when training and evaluation use the SAME ellipse model for crop generation
- v9 model (trained on v11 ellipse crops): 86.6% center on 97-img test
- v2 model (trained on linear radius crops): 25.9% center on test_1 with predicted ellipses
- The model is extremely sensitive to crop center/radius variations

### tfmot QAT limitations
- Lambda layers: NOT supported
- Custom layers: need `register_keras_serializable` + `quantize_scope`
- Multiply layer: NOT supported by tfmot
- SE attention, Coordinate Attention: both use unsupported ops
- Only standard Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D are safe

### Test_1 challenge
- 914 diverse gauge images (pressure gauges, stock photos, car gauges)
- 4 images filtered out of precomputed test set (extreme elongated ellipses)
- Model achieves 25.9% center with predicted ellipse pipeline
- 88.7% on 97-image test set (precomputed crops)

## What needs to happen tomorrow

**Primary goal: Build an ellipse model with 80% accuracy across all test sets.**

We have 8,000 labelled gauge face ellipses. The current ellipse model (linear radius head) doesn't generalize well across test sets. The bottleneck is the ellipse model, not the center/tip model.

### Approach options:
1. **MobileNetV2 backbone** with pretrained weights + fine-tuning on gauge data
2. **Larger custom CNN** with more capacity (current model is only 1.17M params)
3. **Better augmentation** for domain bridging (the board captures look very different from generic gauges)
4. **Ensemble** of multiple ellipse models
5. **Two-stage approach**: first detect gauge face region, then refine ellipse parameters

### Data available:
- `data/gauge_face_ellipse_v1_640_gray/`: 9,185 images (7,328 train + 927 val + 925 test)
- `data/labelled/test_3.zip`: 11 board captures
- `data/labelled/test_1.zip`: 914 diverse gauge images (in test split)
- Total: ~8,000 training images with YOLO-OBB ellipse labels

### Architecture constraints:
- Must use Conv+BN+ReLU (QAT-safe)
- No Lambda, no Multiply, no custom layers without registration
- Linear output for radius (not sigmoid)
- Max 1MB activation budget (SRAM constraint)
- Max 2.5MB total model size

### Key files:
- `scripts/train_gauge_ellipse_qat_linear.py` — current best ellipse training script
- `scripts/eval_pipeline_predicted.py` — full pipeline evaluation
- `scripts/eval_pipeline_padded.py` — padded pipeline evaluation
- `src/embedded_gauge_reading_tinyml/heatmap_losses.py` — Wing Loss added
- `docs/ai-memory/lessons-learned/2026-07-23-qat-safe-architecture.md`
- `docs/ai-memory/lessons-learned/2026-07-23-linear-radius-head.md`
- `docs/ai-memory/lessons-learned/2026-07-23-crop-pipeline-consistency.md`
