# AI Memory

## Quick-reference for the current state of the project.

### Pipeline Architecture (Prod v0.9)

**Camera:** IMX335 → DCMIPP → 224×224 YUV422 frame buffer

**Inference pipeline** (run in `App_AI_RunDryInferenceFromYuv422`):
1. **OBB localizer** (`mobilenetv2_obb_longterm`) — full-frame 224×224 → gauge bounding box (cx, cy, w, h)
2. **Luma refiner** (CPU heuristic, optional) — refines OBB crop using bright-centroid
3. **Center detector** (`mobilenetv2_center_detector`) — OBB crop → 224×224 int8 RGB → NPU → (cx, cy) in crop coordinates → map to full-frame coords
4. **Polar vote** (`AppBaselineRuntime_EstimatePolarNeedle`) — full-frame 224×224 YUV, NN-predicted center → needle angle, confidence
5. **Angle→Temperature** (`AppBaselineRuntime_ConvertAngleToTemperature`) — calibrated gauge arc mapping

### Models & Flash Slots

| Model | Flash Address | Weights | Input | Output |
|-------|--------------|---------|-------|--------|
| OBB (`mobilenetv2_obb_longterm`) | 0x70700000 | 3.07 MB | 224×224×3 float32 | 24 bytes (6×float32: cx,cy,w,h,cos,sin) |
| Center Detector (`mobilenetv2_center_detector`) | 0x70200000 | 3.08 MB | 224×224×3 int8 | 2 bytes (cx_int8, cy_int8) |
| Rectifier (`mobilenetv2_rectifier_hardcase_finetune`) | 0x70600000 | 118 KB | 224×224×3 float32 | crop box |
| Scalar CNN (deprecated, kept for diagnostics) | 0x70200000 | 3.07 MB | — | — |

xSPI2 memory map: FSBL (0x70000000), App (0x70100000, 1 MB), Center-Detector (0x70200000, 3.2 MB), Tip-Focus (0x70400000, 4 MB), Rectifier (0x70600000, 256 KB), OBB (0x70700000, 4 MB)

### Key Files

| File | Purpose |
|------|---------|
| `Appli/Src/app_ai.c` | Main inference pipeline (OBB → luma → center-detector → polar-vote) |
| `Appli/Src/app_center_detector.c` | Center detector module: crop, YUV→int8 RGB, NPU inference, dequantize, polar vote |
| `Appli/Inc/app_center_detector.h` | Public API for center detector pipeline |
| `Appli/Src/app_baseline_runtime.c` | Classical baseline worker + polar vote engine |
| `Appli/Inc/app_baseline_runtime.h` | Public API: `AppBaselineRuntime_EstimatePolarNeedle`, `ConvertAngleToTemperature` |
| `Appli/Src/ai_network_mobilenetv2_center_detector.c` | Thin wrapper `#include`ing Cube.AI generated .c |
| `Appli/Src/ai_network_mobilenetv2_obb_longterm.c` | Thin wrapper `#include`ing OBB generated .c |
| `st_ai_output/packages/center_detector_v1_int8/` | Cube.AI generated center detector files |
| `st_ai_output/packages/prod_model_v0.3_obb_int8/` | Cube.AI generated OBB files |

### Performance (Python eval on 52 test samples)

| Pipeline | MAE (°C) | Center Dist (px) |
|----------|----------|-------------------|
| Center Detector + Polar Vote | **19.84** | 7.00±4.3 |
| OBB + Polar Vote | 32.76 | 8.08±4.9 |
| Firmware Baseline (ported) | 41.14 | 17.87±10.5 |
| GT Oracle + Polar Vote | 15.42 | — |

### Center Detector Details

- **Architecture:** MobileNetV2-based regression (2 output neurons)
- **Training inputs:** 224×224×3 int8 RGB, zero_point=-128, scale=0.00392156886
- **Output:** 2×int8 (cx, cy) normalized to [0, 1] in crop space
  - Dequantize: `val_norm = (val_int8 + 128) * 0.00390625`
  - Full-frame: `ff_cx = crop.x_min + cx_norm * crop.width`
- **Weight blob:** `mobilenetv2_center_detector_atonbuf.xSPI2.raw` (~3.08 MB)
- **Pool symbol alias:** `_mem_pool_xSPI2_mobilenetv2_center_detector` is an asm alias for `_mem_pool_xSPI2_scalar_full_finetune_from_best_piecewise_calibrated_int8` (both at 0x70200000)

### Pipeline Flow in `App_AI_RunDryInferenceFromYuv422`

```
[frame 224x224 YUV422]
       ↓
  OBB stage (NPU)
       ↓
  Luma refiner (CPU, optional)
       ↓
  Center detector (AppCenterDetector_Run):
    → Crop OBB box from frame
    → Resize to 224x224 int8 RGB
    → NPU inference → (cx_int8, cy_int8)
    → Dequantize → (cx, cy) in full-frame coords
    → Polar vote (AppBaselineRuntime_EstimatePolarNeedle)
    → Angle → temperature
       ↓
  app_ai_last_inference_value = temperature
```

### Build Notes

- Cube.AI generated pool symbols are 32-byte NOLOAD placeholders. The actual weight data (~3 MB per model) is pre-flashed to xSPI2 via `flash_boot.bat` before boot.
- The center detector pool is aliased to the scalar pool via `__asm__(".set ...")` to avoid a second placeholder at a different address, which would shift every weight offset in the generated NPU code.
- The scalar CNN model is still compiled in (thin wrapper + NN instance macro) for diagnostic/reference, but its inference call is disabled in the pipeline.

### Known Issues / Next Steps

- **~15% catastrophic polar voter failure rate** — about 1 in 7 frames where the polar vote finds no clear needle. Root cause: the scanned polar ring may contain glare, shadows, or be outside the gauge face.
- **~2/3 of angle error is center-estimation**, ~1/3 is the polar voter itself.
- **Spoked-arc voter** improves GT/oracle by 40% but slightly hurts center detector (+13%) due to Hough sensitivity to center errors.
- The polar vote uses `camera_baseline_current_frame_is_bright` state from the baseline runtime thread — this may be stale if the baseline thread is not actively processing the same frame.
