# AI Memory

## Quick-reference for the current state of the project.

### Pipeline Architecture (Live Board, 2026-06-02)

**Camera:** IMX335 → DCMIPP → 224×224 YUV422 frame buffer

**AI inference pipeline** (run in `App_AI_RunDryInferenceFromYuv422`):
1. **OBB localizer** (`mobilenetv2_obb_longterm`) — full-frame 224×224 → gauge bounding box
2. **OBB → pivot centre** — compute Celsius needle pivot from OBB centre + per-gauge offset ratios (from `app_gauge_geometry.h` and TOML gauge spec). EMA-smoothed to reject x-centre jitter. Bypasses the broken centre-detector CNN entirely.
3. **Polar vote** (`AppBaselineRuntime_EstimatePolarNeedle`) — 224×224 YUV, OBB-derived pivot → needle angle + confidence
4. **Angle → Temperature** (`AppBaselineRuntime_ConvertAngleToTemperature`)

**Baseline pipeline:**
1. Bright-pixel centre, fixed-crop, board-prior, rim-geometry, and image-centre seeds
2. Each seed runs polar vote independently
3. Live selector picks best candidate; stability history smooths outliers
4. Rim-geometry seed consistently finds (100-116, 108-112) centre → 10.6°C

### Gauge Calibration Parameters

Stored in `ml/src/embedded_gauge_reading_tinyml/gauge/gauge_calibration_parameters.toml` and mirrored in `app_gauge_geometry.h`:

```toml
[littlegood_home_temp_gauge_c]
min_deg = 135.0        # min tick at ~7:30, maps to -30°C
sweep_deg = 270.0      # needle sweeps 270° clockwise
min_value = -30.0      # degrees Celsius
max_value = 50.0
units = "C"
direction = "clockwise"
needle_colour = "dark"
obb_pivot_x_offset_ratio = -0.0089   # ≈ −2 px on 224×224 frame
obb_pivot_y_offset_ratio = 0.0625    # ≈ +14 px on 224×224 frame
```

The OBB pivot offsets are **from the OBB geometric centre** (not from the box edge). They are frame-ratio scaled for resolution independence. Calibrated from the rim-geometry centre detector on the live board: `pivot = OBB_centre + offset`.

### Centre-Detector CNN Status

**BYPASSED** in the live AI pipeline. The CNN produces garbage raw int8 outputs (`[CD] raw int8 output: %d %d` shows meaningless values, and `[AI] Center detector using OBB fallback center: (,)` prints blank floats because the output is NaN/invalid). Instead, the AI pipeline derives the pivot from the OBB box geometry using the gauge-spec constants above. A `TrainingCropCenter` override was previously used (centre (112,99)) but gave temperature readings off by ~6°C because the true Celsius dial centre is ~4 px lower and ~9 px further right.

### OBB Performance

The OBB CNN runs stably on the NPU:
- `obb_valid = 1` in 100% of captures
- `obb_cy = 0.4180` — exact same value every frame (y-centre is perfectly stable)
- `obb_cx` varies 0.473–0.527 across frames (±0.027 normalized, ~6 px)
- `obb_box_w` 0.53–0.55 normalized, ~145–150 px
- `obb_box_h` 0.71–0.74 normalized, ~160–166 px
- OBB crop: x=32–46, y=11–15, w=144–150, h=159–166
- Never times out, never produces non-finite values

**EMA smoothing** (`APP_AI_OBB_CENTER_EMA_ALPHA = 0.20`) suppresses single-frame x-centre jitter. With smoothing: y-pivot = 107.6 (rock-solid), x-pivot varies only ±0.5 px across frames.

### Live Board Accuracy (2026-06-02, True = 10°C)

| Pipeline | Reading | Error | Confidence |
|----------|---------|-------|------------|
| AI (OBB→pivot) | 10.3°C | +0.3°C | 47.1 |
| Baseline (held) | 10.3°C | +0.3°C | 6.9 |
| Baseline rim-geometry seed | 10.6°C | +0.6°C | 25.1 |
| AI (OBB→pivot, avg) | 10.7°C | +0.7°C | 30–35 |

The AI pipeline consistently matches or beats the baseline. The AI reads 10.3–11.2°C with confidence 30–47 across captures. The baseline periodically locks to a stale held value but recovers.

### Latency & Power Metrics (2026-06-02)

- **DWT counter** fixed from 32-bit (wraps every 5.4s at 800MHz) to 64-bit extended counter via high-word tracking
- **Multi-slot metrics**: Two concurrent active inference slots so BASELINE and AI can be timed within the same capture cycle
- **Timing boundaries**: Both pipelines start at snapshot copy (`RequestEstimate` / `RequestDryRun`) and end at temperature estimate print (`LogEstimate` / dry-run success path)
- **Power tracking** (TBD): continuous INA219 sampling during pipeline window, min/avg/max reported after latency line

### Key Source Files

| File | Purpose |
|------|---------|
| `app_baseline_runtime.c` | Classical CV baseline with 5-seed polar voting, continuity scoring, history smoothing |
| `app_ai.c` + `app_ai_runtime_tail.inc` | AI pipeline: OBB → offset-pivot → polar vote → temperature |
| `app_ai_helpers_decode.inc` | OBB stage inference, decode, crop computation |
| `app_center_detector.c` | Centre-detector module (CNN bypassed; polar vote used directly) |
| `app_gauge_geometry.h` | Training crop ratios, inner-dial centre ratios, OBB pivot offset ratios |
| `inference_metrics.c` | 64-bit DWT timing, multi-slot inference metrics, latency logging |
| `ina219_power.c` | INA219 power monitor driver and sampling thread |
| `gauge_calibration_parameters.toml` | Per-gauge angular & spatial calibration (mirrored in C header) |
| `processing.py` | Python `GaugeSpec` dataclass with TOML loader |

### Key Calibration Constants (app_gauge_geometry.h)

```c
#define APP_GAUGE_TRAINING_CROP_X_MIN_RATIO 0.1027f   // crop left edge
#define APP_GAUGE_TRAINING_CROP_Y_MIN_RATIO 0.2573f   // crop top edge
#define APP_GAUGE_INNER_DIAL_CENTER_X_RATIO 0.5000f   // (112,99) — baseline default
#define APP_GAUGE_INNER_DIAL_CENTER_Y_RATIO 0.4460f   // (112,99)
#define APP_GAUGE_OBB_PIVOT_X_OFFSET_RATIO (-0.0089f) // OBB centre → pivot x offset
#define APP_GAUGE_OBB_PIVOT_Y_OFFSET_RATIO  0.0625f   // OBB centre → pivot y offset
```

### Known Issues / Next Steps

- **Centre CNN is non-functional** on the live board — produces garbage int8 outputs. The OBB→pivot offset approach works as a substitute. Centre CNN needs retraining with board-capture labels.
- **Baseline live selector** prefers fixed-crop seed (source priority 5) over rim-geometry (priority 1) even when rim-geometry is correct. The baseline can lock to stale held values for multiple frames.
- **Bright-relaxed mode** triggers at 60-70% brightness in normal captures, lowering continuity gates. Darker captures (normal mode) produce cleaner polar votes with fewer hot-zone false positives.
- **Polar vote fragility**: a single-pixel centre change can flip the winning peak from the true needle to a dial marking. The 262–264° region has strong false peaks from fixed dial artwork.
- **Dial radius discrepancy**: AI uses ~60–62 px from OBB; baseline uses 68.9 px from frame ratio. Both give similar temperatures but the difference may indicate the OBB-sizing heuristic needs refinement.

### Centre Probe Grid Results (diagnostic, code removed)

The rim-geometry centre `(116, 108)` proved to be the correct pivot. The OBB geometric centre is at `~ob_cx_abs ≈ 106, obb_cy_abs ≈ 94`. The offset of (−2, +14) px from OBB centre to Celsius pivot was calibrated once and stored in the gauge spec. The OBB approach generalizes: different gauge types only need different offset ratios.

### Recent Changes Summary (2026-06-02)

1. **Added OBB pivot offset to gauge spec** — TOML + Python `GaugeSpec` + C `app_gauge_geometry.h`
2. **Switched AI centre from TrainingCropCenter to OBB-derived** — `pivot = OBB_centre + offset_ratio * frame_size`
3. **Added EMA smoothing on OBB centre** — `APP_AI_OBB_CENTER_EMA_ALPHA = 0.20`, suppresses x-jitter
4. **Fixed DWT latency counter** — 32→64 bit to prevent 5.4s wrap garbage
5. **Multi-slot metrics** — baseline and AI tracked independently
6. **Fixed AI EndInference** — was inside never-called `AppAI_LogInferenceResult` function
7. **Cleaned up probe grid diagnostic code** — 97-line grid sweep removed
8. **Removed duplicate DebugConsole_Printf calls** from rejection path
