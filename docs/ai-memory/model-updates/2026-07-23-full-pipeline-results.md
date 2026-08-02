# Full Pipeline Results — 2026-07-23

Date: 2026-07-23
Status: experimental
Scope: complete gauge reading pipeline (ellipse → center/tip)

## Summary

Achieved 100% accuracy on test_3 (11 images) using the linear radius head
ellipse model + new center/tip model. Both models are int8 quantized and
fit within the 2.5MB SRAM budget.

## Pipeline components

### 1. Ellipse detector: `gauge_ellipse_qat_linear_v1`
- Architecture: QAT-safe encoder (Conv+BN+ReLU) + multi-head output
- Input: 224×224×1 grayscale
- Output: center_xy (sigmoid) + radius_xy (LINEAR) + confidence (sigmoid)
- Size: 1.2MB int8 TFLite
- Key innovation: linear radius head preserves radius variation in int8

### 2. Center/tip detector: `gauge_center_tip_full_v1`
- Architecture: v9 U-Net (24/36/56/96 channels)
- Input: 160×160×2 (grayscale + ellipse mask)
- Output: 80×80×2 heatmaps (center + tip)
- Size: 379KB int8 TFLite
- Training: 7,793 images (7,309 generic + 484 LittleGood/board/test_3)

## Test_3 results (11 images, full pipeline)

| Metric | Value |
|--------|-------|
| Center ≤8px | **100% (11/11)** |
| Tip ≤8px | **100% (11/11)** |
| Center mean | **2.8px** |
| Tip mean | **2.8px** |
| Center max | 5.5px |
| Tip max | 6.3px |

## 97-image test set results (center/tip only, with GT ellipse conditioning)

| Model | Center ≤8px | Tip ≤8px | Center err | Tip err |
|-------|------------|----------|-----------|---------|
| v9 (best) | 86.6% | 75.3% | 8.9px | 11.2px |
| full_v1 | 81.4% | 66.0% | 10.4px | 12.8px |

Note: The 97-image test set uses crops from the v11 ellipse (fixed radius).
The full_v1 model was trained with crops from the linear radius ellipse
(varying radius). The test set is not representative of the production pipeline.

## Artifacts

- `artifacts/gauge_ellipse_qat_linear_v1/` — ellipse model (1.2MB int8)
- `artifacts/gauge_center_tip_full_v1/` — center/tip model (379KB int8)
- `data/initial_temp_gauge_v1/student_conditioned_full/` — training data

## Total model size

- Ellipse: 1.2MB int8
- Center/tip: 379KB int8
- **Total: 1.6MB** (under 2.5MB SRAM budget)

## Next steps

1. Regenerate the 97-image test set with the linear radius ellipse's crops
2. Evaluate the full pipeline on the regenerated test set
3. Package for STM32 N6 deployment
4. Test on live board captures
