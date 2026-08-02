# Current state: ellipse + center/tip pipeline — 2026-07-23

Date: 2026-07-23
Status: current
Scope: full gauge reading pipeline (ellipse → center/tip)

## Summary

Built a working int8 ellipse detector (QAT encoder) and center/tip model (v9).
The QAT encoder produces meaningful int8 outputs (not collapsed), but the
radius prediction is fixed at 0.2539 for all images (GT: 0.195-0.233).
The v9 center/tip model achieves 86.6% center, 75.3% tip on the 97-image
test set with ground truth ellipse conditioning.

## Model performance

| Component | Model | Metric | Value |
|-----------|-------|--------|-------|
| Ellipse (int8) | QAT encoder | Center error on test_3 | 3.7px mean |
| Ellipse (int8) | QAT encoder | Radius | Fixed 0.2539 (GT: 0.195-0.233) |
| Center/tip (int8) | v9 | Center ≤8px | 86.6% |
| Center/tip (int8) | v9 | Tip ≤8px | 75.3% |
| Center/tip (int8) | v9 | Center error | 8.9px mean |
| Center/tip (int8) | v9 | Tip error | 11.2px mean |
| Center/tip (int8) | v9 | Angle ≤5° | 78.4% |
| Full pipeline | QAT ellipse + v9 CT | Center on test_3 | 18.2% (bad due to radius) |

## Why the radius collapses

The sigmoid output range is [0,1] mapped to int8 [-128,127]. The quantization
step is 1/255 ≈ 0.0039. For rx≈0.195, the int8 value is ~50. The variation
in rx across training images (0.1947 to 0.1963) is less than 1 int8 step,
so the quantized model predicts the same value for all inputs.

## What was tried

1. **Multi-head approach** (separate center/radius heads with 3x radius loss weight):
   - Improved center accuracy (3.7px vs 6.5px on test_3)
   - Radius still fixed at 0.2539 (not improved)

2. **Full dataset training** (8,244 images including board_captures_2 + test_3):
   - Center: 86.6% (same as v9)
   - Tip: 71.1% (worse than v9's 75.3%)
   - Extra images may have introduced noise

3. **QAT-conditioned center/tip training** (using QAT ellipse for crops):
   - Much worse: 26.8% center, 8.2% tip
   - Fixed radius creates bad crops that confuse the model

## Best models

- **Ellipse**: `artifacts/gauge_ellipse_qat_multihead_v1/` (1.2MB int8)
- **Center/tip**: `artifacts/gauge_center_tip_littlegood_v9/` (379KB int8)

## Next steps

1. Use fixed radius for known gauge types (rx=0.195, ry=0.233 for LittleGood)
2. Or use a regression head with linear output (not sigmoid) for radius
3. Or normalize radius prediction (predict rx/ry ratio instead of absolute)
4. Or use a separate quantization scale for radius output
