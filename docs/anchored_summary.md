# Project Summary

## Goal
Read analog gauges on low-power embedded hardware (STM32 N6 NPU). Pipeline: DCMIPP crop → YOLO OBB → rectified warp → heatmap center detection → needle geometry → reading.

## What we've done

### YOLO OBB — Gauge detection (DONE)
- **Model:** YOLO11n-OBB trained on PXL phone images
- **Input:** 320×320 after DCMIPP center-crop
- **Performance:** mAP50=0.995 on val set
- **Export:** TFLite int8 (quantized), saves last epoch
- **Artifacts:** `ml/artifacts/yolo_obb_320/`
- **YUV→RGB:** STM32 IMX335 outputs YUV422; DCMIPP YCbCr→RGB handling fixed
- **Status:** Trained, exported, YUV pipeline handled

### Heatmap Center Detection (DONE — new 320×320 pipeline)
- **Encoder:** MobileNetV3-Small, α=0.75, ImageNet-pretrained, frozen
- **Decoder:** Tiny (upsample 10→20→40→80, 2× Conv3×3(16→8), BN, ReLU, then 1×1 conv to 1-ch heatmap)
- **Input:** 320×320 rectified gauge crop (10% margin around OBB)
- **Output:** 80×80 sigmoid heatmap → soft-argmax for sub-pixel center
- **Parameters:** 0.635M trainable (from 2.68M total, backbone frozen)
- **Training data:** 282 train / 70 val from PXL images
- **Training result (val):** mean=1.94, median=1.31 px error in 80×80 heatmap
- **Export:** TFLite float32 (2.45 MB) + int8 (855 KB)
- **Artifacts:** `ml/artifacts/heatmap_cd_320/`
- **Dataset:** `ml/data/heatmap_cd_320/` (rectified 320×320 crops + 80×80 Gaussian heatmaps)

### Pipeline Validation (DONE)
Validated the full flow (DCMIPP → GT OBB → rectified warp → heatmap CD) on 70 PXL val images:

| Model | Heatmap px (median) | Input px (median) | Est. angle error |
|---|---|---|---|
| f32 TFLite | 3.00 px | 12.1 px | ~5.4° |
| int8 TFLite | 4.35 px | 17.6 px | ~7.9° |

Quantization cost: ~1.3-1.8 px additional error.

Cross-domain test on 57 board captures (224→320 upsampled, no OBB):
- f32: 4.23 px heatmap median (17.1 px input)
- Domain shift adds ~1.2 px heatmap error

### Model sizes
- **YOLO OBB int8 TFLite:** ~1.5 MB
- **Heatmap CD int8 TFLite:** 855 KB
- **Combined:** ~2.4 MB

## Remaining work

1. **Validate full pipeline end-to-end** — Combine YOLO OBB int8 TFLite + heatmap CD int8 TFLite on PXL images; compare predicted center/tip against ground truth; measure end-to-end angle/reading error
2. **Improve heatmap CD accuracy** — The 3.0 px heatmap error translates to ~5.4° angle error (≈4.4°C); consider training longer, higher-resolution model, or unfreezing backbone
3. **Geometry reader** — Compute needle angle from center + tip and convert to temperature reading
4. **STM32 integration** — Run both TFLite models on N6 NPU; handle YUV→RGB→320×320→OBB→rectify→CD→geometry pipeline on-device

## Key files
- `ml/scripts/train_yolo_obb_320.py` — YOLO OBB training
- `ml/scripts/train_heatmap_cd_320.py` — Heatmap CD training
- `ml/scripts/prepare_heatmap_cd_320_dataset.py` — CD dataset prep
- `ml/scripts/validate_pipeline.py` — Pipeline validator
- `ml/src/embedded_gauge_reading_tinyml/heatmap_utils.py` — Heatmap generation/decoding utilities
- `ml/src/embedded_gauge_reading_tinyml/dataset.py` — Dataset loading
