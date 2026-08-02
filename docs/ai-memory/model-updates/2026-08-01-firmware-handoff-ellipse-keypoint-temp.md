# FIRMWARE HANDOFF: ellipse + keypoint -> temperature models for STM32 N6 — 2026-08-01

Date: 2026-08-01
Status: current
Scope: STM32N657 board integration of the two-stage gauge-reading pipeline
Evidence: `ml/scripts/pipeline_ellipse_keypoint_temperature.py`,
`ml/data/labelled/test_3_pipeline_results.csv`,
`ml/artifacts/ellipse_iter8_universal_wide_deep/`,
`ml/artifacts/keypoint_unet_224g_stride2/`

## WHAT TO INTEGRATE (GPT: read this first)

Two int8 TFLite models run in sequence, replacing the old scalar pipeline:

### Model 1: ellipse localizer (gauge face)
- File: `ml/artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite`
- Input: 384x384x1 grayscale, int8 (scale=0.00392157, zero=-128)
- Output: single flat vector of 24,207 int8 values (dequantize by
  output scale 0.00390625 / zero -128). Layout:
  - 3 scale heads x (center heatmap + rim heatmap):
    sizes 24x24, 48x48, 96x96 (MAP_VALUES = 576/2304/9216 each pair)
  - 3 x 4 geometry values (cx, cy, rx, ry) per head
  - 3 scale-confidence values (softmax over heads)
  - GEOMETRY_OFFSET = 12096, CONFIDENCE_OFFSET = 12108
- Decode: pick argmax(confidence) head; soft-argmax the center heatmap
  (weights=(map-0.05)^4); geometry gives (rx, ry). Normalized [0,1].
- Int8 size: ~0.7 MB. Params: 709K.

### Model 2: center/tip keypoint model (BOARD-FIT: 56x56 output, alpha=1.5)
- File: `ml/artifacts/keypoint_unet_224g_wide_aug/model_int8.tflite`
- Input: 224x224x1 grayscale, int8 (scale=0.00392157, zero=-128)
- Output: 56x56x2 int8 heatmaps [center, tip] (scale=0.00390625, zero=-128)
  — the deployed tip_focus 56x56 contract
- Decode: soft-argmax each heatmap ((map-0.05)^4 weights) -> 224-crop px.
- Weights: 2.30 MB (xSPI2 flash — flash is not the constraint)
- **Peak int8 activation: 1.87 MiB (calibrated to GPT's N6 allocator)** —
  fits the 2.75 MiB no-HyperRAM pool with margin (under the 2.5 MiB rule)
- Accuracy: tip 8.88px screen slice; 0.82C MAE on 16 unseen captures.

## CRITICAL: activation budget rules (GPT's N6 compiler probes, 2026-08-01)

- The N6 internal (no-HyperRAM) activation pool is 2,883,584 bytes
  (2.75 MiB). Peak activation must stay under ~2.5 MiB.
- **Stride-2 (112x112 output) does NOT fit at any width**: alpha=1.0 needs
  3.12 MiB (incl. 980 KiB HyperRAM); lean decoders fit but cost accuracy
  (tip 11.7px vs 8.9px).
- The 56x56 output at alpha=1.5 is the sweet spot: 1.87 MiB peak, best tip.
- Weights live in xSPI2 flash — weight size is NOT the deployment
  constraint; activation memory is.

Do NOT attempt to package stride-2 (112x112 output) keypoint models for
the N6.

### Pipeline (per frame)
1. Resize/letterbox camera frame to 384x384 grayscale -> ellipse model ->
   (cx, cy, rx, ry) normalized.
2. Crop 1.35x square around the ellipse (side = max(2rx, 2ry) * 1.35 * W)
   -> resize to 224x224 grayscale -> keypoint model -> center + tip in
   crop pixels.
3. Map crop px back to source: center_src = crop_left + px * (actual_side/224).
4. angle = atan2(tip_y - center_y, tip_x - center_x) in degrees.
5. temperature = -30 + ((angle - 135) % 360 / 270) * 80, clamped to [0,1]
   fraction (LittleGood temp gauge: min_deg=135, sweep_deg=270,
   -30C..+50C, clockwise).

## VALIDATED ACCURACY

### Unseen captures (16 images, never in training; GT in filename)
| Model | Weights | Peak act | MAE | Median | Max | ≤2C | ≤5C |
|---|---|---|---|---|---|---|---|
| **keypoint wide_aug (BOARD-FIT)** | 2.30 MB | **1.87 MiB** | **0.82C** | 0.76C | 1.81C | **100%** | 100% |
| keypoint stride2_s (no-fit) | 1.05 MB | 3.12 MiB ❌ | 1.14C | 1.24C | 2.55C | 88% | 100% |

### test_3 (17 images with GT; capture_m25c.jpg is duplicated in training)
| Model | MAE | Median | Max | ≤2C |
|---|---|---|---|---|
| **wide_aug (BOARD-FIT)** | **1.52C** | 1.55C | 4.05C | 71% |
| stride2_s (no-fit) | 1.72C | 1.74C | 3.74C | 53% |

Board-fit best reads (unseen): p10c 0.1C, p15c 0.2C, p30c.png 0.2C,
p35c 0.3C, p31c 0.7C, p42c 1.1C. Worst: p50c ~3.4-4.1C (needle near
sweep end). 100% within 5C on every set.

## FIRMWARE NOTES (GPT)

- The camera path already produces 224x224 YUV422 captures; the old
  preprocess (`AppAI_PreprocessYuv422FrameTo...`) converts to grayscale
  float32. For the ellipse stage you need a 384x384 letterboxed grayscale
  input; for the keypoint stage a 1.35x ellipse crop resized to 224x224.
- Both models are QAT int8 with TFLite-BUILTINS_INT8 ops; export via
  ST Edge AI relocatable package like the current `tip_focus_v18_int8_n6_npu`
  (keep `c_info.json` + `network.csv` beside the xSPI2 blob).
- Memory: keep the two weight blobs in separate xSPI2 slots; the runtime
  arena must not overlap app RAM (see existing OBB/scalar placement
  lessons: arena at 0x34110000).
- The old scalar head + piecewise calibration in
  `app_inference_calibration.c` is NOT needed for this pipeline — the
  temperature comes from geometry + the fixed sweep calibration above.
- Keep the OBB localizer as the front-end; if OBB crop decoding fails, fall
  back to the fixed training crop (per AGENTS.md live-contracts).

## HOW TO REPRODUCE OFFLINE

```bash
cd ml
poetry run python scripts/pipeline_ellipse_keypoint_temperature.py \
  --ellipse artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite \
  --keypoint artifacts/keypoint_unet_224g_stride2/model_int8.tflite \
  --images data/labelled/test_3.zip
```

## KEY LESSONS (see model-updates/2026-07-31-* and 2026-08-01-*)

- Geometry-first (detect ellipse -> crop -> center/tip -> calibrated angle)
  beats end-to-end scalar regression: MAE 1.5C vs the old ~4C scalar best
  on the same hard cases.
- Stride-2 (112x112) heatmap output halves keypoint quantization error and
  broke the tip <10px wall (9.0px vs 10.7px) in screens.
- The LittleGood calibration TOML in the repo was OVERWRITTEN by a
  firmware-specific gauge_1 spec; the correct values are min_deg=135,
  sweep_deg=270, -30..50C (recovered from git history).
- Board-capture temperatures are encoded in filenames (`capture_p42c.jpg`
  = +42C, `capture_m10c.png` = -10C) — useful as GT.
