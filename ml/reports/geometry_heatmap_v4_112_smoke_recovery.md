# Geometry Heatmap v4 112 Smoke Recovery

## Source
- Directory: `/tmp/geomq_v4_112_smoke_skip2`
- Previous run: `train_geometry_heatmap_v4_112_quant_native.py`

## Completion Status
- **Smoke completed**: Yes — all artifacts present (model, config, predictions, summary)
- **No leftover processes**: Confirmed via `ps aux`
- **History CSV**: Empty (headers only) — `_write_history` received a dict with no `"loss"` key; per-epoch training dynamics not recoverable

## Config
| Parameter | Value |
|-----------|-------|
| frozen_epochs | 3 |
| unfrozen_epochs | 0 |
| batch_size | 8 |
| frozen_learning_rate | 3e-6 |
| sigma_pixels | 2.5 |
| decoder | softargmax |
| source_model | v3 quant-native canonical |
| initialization | source_model |

## Validation Results (n=47, frozen stage)

| Metric | Value |
|--------|-------|
| Acceptance rate | **0.0000** (0/47) |
| Accepted MAE | NaN |
| Worst accepted error | NaN |
| Accepted >20 C failures | 0 |
| Under 2/5/10 C | 0% / 0% / 0% |
| Center MAE (px@224) | 11.24 |
| Tip MAE (px@224) | **75.32** |
| Angle MAE (deg) | **99.21** |
| Confidence mean | 0.19 |
| Center heatmap peak mean | 0.90 |
| Tip heatmap peak mean | 0.95 |
| Center heatmap spread mean (px@112) | **44.51** |
| Tip heatmap spread mean (px@112) | **46.40** |

## Rejection Analysis (47/47 rejected)

| Reason | Count |
|--------|-------|
| center_heatmap_too_spread_out | 47 |
| center_tip_distance_ratio_implausible | 47 |
| confidence_too_low | 47 |
| tip_heatmap_too_spread_out | 47 |
| temperature_outside_physical_margin | 7 |

## Key Finding: Heatmap Spread

All 47 samples are rejected because heatmap spread (~44–46 px on 112x112) exceeds the guardrail threshold (`max_heatmap_spread_px: 30.0`). This is **~40% of the image dimension**, genuinely too diffuse — not just a threshold-calibration issue.

## Calibration (D_robust_linear)
Despite poor geometry predictions, calibration on angle-derived temperatures shows:
- Validation MAE: 1.14 C (fitted on D_robust_linear)
- Validation max error: 3.00 C
This is misleading — the calibration was fitted on the angle predictions, but angle MAE of 99 degrees means the temperature predictions from those angles are unreliable.

## NaNs/Infs
- **None detected** — all values are finite

## Loss Progression
- Not recoverable (history CSV empty)
- End-state validation was evaluated

## Diagnosis
The v4 112 skip decoder head produces overly diffuse heatmaps (spread ~45px, target sigma=2.5 → expected ~6px FWHM). Center is coarsely localised (11 px @224 scale) but tip is near-random (75 px). Only 3 frozen epochs at 3e-6 LR is insufficient for the 112 head to converge.
