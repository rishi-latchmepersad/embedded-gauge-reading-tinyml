# Phase 11E Aux INT8 Recovery Matrix

- Candidates: 12
- Training schedule: 3 warmup + 15 frozen + 5 unfrozen = 23 epochs
- Baseline config: 08_tip_focus (peak_target=0.25, tip_weight=0.2)
- Source checkpoint: model_v4_112.keras

## Gates

| Gate | Threshold |
|------|-----------|
| accepted_mae_c | 4.5 |
| acceptance_rate | 0.65 |
| worst_accepted_error_c | 20.0 |
| accepted_gt20_failures | 0 |
| temperature_delta_mean | 1.0 |

## Results

| # | Candidate | Weight | Head | Loss | Status | MAE | Accept | Drift | Tip Drift | Gates |
|---|-----------|--------|------|------|--------|-----|--------|-------|-----------|-------|
| 1 | 01_w02_small_mse | 0.2 | small | mse | fail | 3.79 | 72.34% | 1.8554 | 12.53 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 2 | 02_w05_small_mse | 0.5 | small | mse | fail | 3.81 | 72.34% | 1.8495 | 12.57 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 3 | 03_w10_small_mse | 1.0 | small | mse | fail | 3.82 | 72.34% | 1.8629 | 12.53 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 4 | 04_w02_small_huber | 0.2 | small | huber | fail | 3.82 | 72.34% | 1.8510 | 12.57 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 5 | 05_w05_small_huber | 0.5 | small | huber | fail | 3.81 | 72.34% | 1.8600 | 12.58 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 6 | 06_w10_small_huber | 1.0 | small | huber | fail | 3.80 | 72.34% | 1.8472 | 12.53 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 7 | 07_w02_large_mse | 0.2 | large | mse | fail | 3.83 | 72.34% | 1.8654 | 12.52 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 8 | 08_w05_large_mse | 0.5 | large | mse | fail | 3.80 | 72.34% | 1.8651 | 12.54 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 9 | 09_w10_large_mse | 1.0 | large | mse | fail | 3.80 | 72.34% | 1.8606 | 12.53 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 10 | 10_w02_large_huber | 0.2 | large | huber | fail | 3.80 | 72.34% | 1.8653 | 12.55 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 11 | 11_w05_large_huber | 0.5 | large | huber | fail | 3.76 | 72.34% | 1.8423 | 12.60 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |
| 12 | 12_w10_large_huber | 1.0 | large | huber | fail | 3.81 | 72.34% | 1.8458 | 12.58 | accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL |

## Baseline Comparison

| Baseline | INT8 Drift |
|----------|------------|
| Phase 10E original | ~1.99 C |
| Phase 11B 08_tip_focus | 1.8405 C |
| Phase 11D aux smoke | 1.89 C |

## Best Non-Champion

- **Candidate**: 11_w05_large_huber
- **Weight**: 0.5
- **Head**: large
- **Loss**: huber
- **INT8 MAE**: 3.7639 C
- **INT8 Acceptance**: 72.34%
- **INT8 Drift**: 1.8423 C
- **Tip Drift**: 12.60 px
- **Gates**: accepted_mae_c=PASS | acceptance_rate=PASS | worst_accepted_error_c=PASS | accepted_gt20_failures=PASS | temperature_delta_mean=FAIL
