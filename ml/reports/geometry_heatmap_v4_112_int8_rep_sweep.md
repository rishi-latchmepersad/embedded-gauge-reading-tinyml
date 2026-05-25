# Geometry Heatmap v4 112 INT8 Representative-Dataset Sweep

- Validation samples: 47
- Decoder: softargmax w3
- Calibration: D_robust_linear
- Keras MAE: 3.8927 C, accept rate: 0.787

## Gate Criteria
- MAE ≤ 4.5 C
- Acceptance ≥ 65%
- Worst error < 20.0 C
- >20 C failures = 0
- Temp drift ≤ 1.0 C
- Tip drift < 14.82 px

## Results

| Strat | Name | MAE | Accept | Worst | >20C | Drift | Tip Drift | Spread | Gate |
|-------|------|-----|--------|-------|------|-------|-----------|--------|------|
| Ref | Phase 10E | 4.13 | 66.0% | 16.05 | 0 | 2.29 | 13.64 | 49.8 | FAIL |
| A | baseline_identity | 3.99 | 61.7% | 16.59 | 0 | 2.44 | 13.67 | 49.7 | FAIL |
| B | identity_mild | 4.14 | 66.0% | 15.17 | 0 | 2.47 | 14.10 | 50.0 | FAIL |
| C | identity_mild_medium | 4.10 | 63.8% | 16.78 | 0 | 2.65 | 13.89 | 49.6 | FAIL |
| D | stratified | 4.04 | 66.0% | 15.91 | 0 | 2.61 | 14.27 | 50.5 | FAIL |
| E | spread_boundary | 4.07 | 63.8% | 15.69 | 0 | 2.70 | 14.09 | 49.9 | FAIL |
| F | combined | 3.76 | 66.0% | 14.78 | 0 | 2.47 | 13.48 | 49.9 | FAIL |

**No strategy passed all gates.**

## Per-Gate Breakdown

| Strat | MAE Pass | Accept Pass | Worst Pass | >20C Pass | Drift Pass | Tip Drift Pass |
|-------|----------|-------------|------------|-----------|------------|----------------|
| Ref Phase 10E | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| A baseline_identity | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| B identity_mild | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| C identity_mild_medium | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| D stratified | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| E spread_boundary | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| F combined | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |