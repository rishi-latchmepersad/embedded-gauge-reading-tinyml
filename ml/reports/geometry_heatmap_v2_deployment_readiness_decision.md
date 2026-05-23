# Geometry Heatmap v2 Guarded Evaluation

## Selected Guardrail Thresholds

| threshold | value |
| --- | ---: |
| center_peak_min | 0.400 |
| tip_peak_min | 0.400 |
| confidence_min | 0.400 |
| max_heatmap_entropy | 1.000 |
| max_heatmap_spread_px | 25.000 |
| center_tip_distance_ratio_min | 0.400 |
| center_tip_distance_ratio_max | 1.400 |
| edge_margin_px | 4.000 |
| temperature_physical_margin_c | 2.000 |

## Per-Level Metrics

| level | total | accepted | rejected | clamped | acceptance_rate | accepted_mae | worst_accepted | under_2c | under_5c | under_10c | rejected_mean_raw | rejected_worst_raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 59 | 45 | 14 | 0 | 0.763 | 3.157 | 10.812 | 44.4 | 77.8 | 97.8 | 8.026 | 59.428 |
| mild | 59 | 47 | 12 | 0 | 0.797 | 2.806 | 9.077 | 42.6 | 85.1 | 100.0 | 12.896 | 73.604 |
| medium | 59 | 42 | 17 | 0 | 0.712 | 3.180 | 8.469 | 35.7 | 76.2 | 100.0 | 4.943 | 18.727 |
| strong | 59 | 42 | 17 | 0 | 0.712 | 3.253 | 10.812 | 40.5 | 73.8 | 97.6 | 11.718 | 105.195 |

## Baseline Comparison

- geometry_points_v1 test temperature MAE: 7.91 C
- geometry_points_v1 test center MAE: 11.30 px
- geometry_points_v1 test tip MAE: 21.82 px
- Oracle calibrated geometry ceiling: 1.195 C

## Rejected Reasons

- Rejected reason counts: {'tip_peak_too_low': 53, 'tip_heatmap_too_spread_out': 15, 'predicted_angle_outside_valid_sweep': 7, 'center_tip_distance_ratio_implausible': 11, 'temperature_outside_physical_margin': 5}

## Worst Remaining Accepted Predictions

| rank | level | image | abs_err | temp_true | temp_guard | status | center_err | tip_err | ratio | confidence | reasons |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | identity | PXL_20260125_114520969.jpg | 10.812 | -29.00 | -18.19 | accepted | 3.94 | 34.34 | 0.536 | 0.653 |  |
| 2 | strong | PXL_20260125_114520969.jpg | 10.812 | -29.00 | -18.19 | accepted | 3.94 | 34.34 | 0.536 | 0.653 |  |
| 3 | identity | PXL_20260125_120220792.jpg | 9.625 | 49.00 | 39.38 | accepted | 5.77 | 38.68 | 0.761 | 0.698 |  |
| 4 | identity | PXL_20260125_114911288.jpg | 9.112 | -17.50 | -26.61 | accepted | 10.98 | 29.39 | 0.756 | 0.658 |  |
| 5 | mild | PXL_20260125_115615761.jpg | 9.077 | 15.00 | 5.92 | accepted | 9.02 | 37.01 | 0.822 | 0.682 |  |
| 6 | medium | PXL_20260125_114520969.jpg | 8.469 | -29.00 | -20.53 | accepted | 5.61 | 25.79 | 0.585 | 0.665 |  |
| 7 | mild | PXL_20260125_114520969.jpg | 8.436 | -29.00 | -20.56 | accepted | 3.54 | 25.81 | 0.563 | 0.648 |  |
| 8 | identity | PXL_20260125_120119590.jpg | 8.069 | 37.50 | 29.43 | accepted | 2.70 | 25.40 | 0.685 | 0.719 |  |
| 9 | medium | PXL_20260125_120045775.jpg | 7.952 | 37.50 | 45.45 | accepted | 8.95 | 27.60 | 0.553 | 0.766 |  |
| 10 | identity | PXL_20260125_115615761.jpg | 7.743 | 15.00 | 7.26 | accepted | 7.16 | 34.00 | 0.784 | 0.683 |  |
| 11 | strong | PXL_20260125_115735852.jpg | 7.505 | 26.50 | 34.01 | accepted | 3.55 | 23.95 | 0.644 | 0.649 |  |
| 12 | medium | PXL_20260125_115735138.jpg | 7.341 | 26.50 | 33.84 | accepted | 2.57 | 27.50 | 0.632 | 0.630 |  |
| 13 | mild | PXL_20260125_114911288.jpg | 7.043 | -17.50 | -24.54 | accepted | 7.82 | 21.43 | 0.765 | 0.657 |  |
| 14 | identity | PXL_20260125_115612552.jpg | 7.026 | 15.00 | 22.03 | accepted | 6.84 | 20.72 | 0.635 | 0.720 |  |
| 15 | strong | PXL_20260125_115325407.jpg | 6.737 | 5.00 | -1.74 | accepted | 5.09 | 16.64 | 0.688 | 0.664 |  |
| 16 | medium | PXL_20260125_115325407.jpg | 6.661 | 5.00 | -1.66 | accepted | 4.62 | 18.54 | 0.736 | 0.646 |  |
| 17 | medium | PXL_20260125_114911288.jpg | 6.523 | -17.50 | -24.02 | accepted | 9.33 | 19.69 | 0.727 | 0.651 |  |
| 18 | identity | PXL_20260125_115239306.jpg | 6.229 | -7.50 | -1.27 | accepted | 1.67 | 21.70 | 0.840 | 0.599 |  |
| 19 | strong | PXL_20260125_120041517.jpg | 6.118 | 37.50 | 31.38 | accepted | 8.40 | 21.10 | 0.791 | 0.783 |  |
| 20 | identity | PXL_20260125_115735138.jpg | 6.030 | 26.50 | 32.53 | accepted | 3.60 | 22.98 | 0.648 | 0.641 |  |
| 21 | identity | PXL_20260125_114850916.jpg | 5.963 | -17.50 | -23.46 | accepted | 5.10 | 15.34 | 0.730 | 0.661 |  |
| 22 | mild | PXL_20260125_114850916.jpg | 5.963 | -17.50 | -23.46 | accepted | 5.10 | 15.34 | 0.730 | 0.661 |  |
| 23 | strong | PXL_20260125_114850916.jpg | 5.963 | -17.50 | -23.46 | accepted | 5.10 | 15.34 | 0.730 | 0.661 |  |
| 24 | mild | PXL_20260125_115239306.jpg | 5.870 | -7.50 | -1.63 | accepted | 3.41 | 23.11 | 0.869 | 0.597 |  |
| 25 | medium | PXL_20260125_115328820.jpg | 5.762 | 5.00 | -0.76 | accepted | 2.05 | 19.31 | 0.821 | 0.621 |  |
| 26 | strong | PXL_20260125_115328820.jpg | 5.678 | 5.00 | -0.68 | accepted | 4.18 | 17.18 | 0.819 | 0.617 |  |
| 27 | strong | PXL_20260125_115327587.jpg | 5.672 | 5.00 | -0.67 | accepted | 4.05 | 13.34 | 0.781 | 0.628 |  |
| 28 | medium | PXL_20260125_114850916.jpg | 5.667 | -17.50 | -23.17 | accepted | 5.42 | 15.27 | 0.698 | 0.654 |  |
| 29 | mild | PXL_20260125_115612552.jpg | 5.605 | 15.00 | 20.60 | accepted | 5.43 | 16.43 | 0.642 | 0.728 |  |
| 30 | identity | PXL_20260125_114914961.jpg | 5.562 | -17.50 | -23.06 | accepted | 2.77 | 30.88 | 0.574 | 0.651 |  |

## Decision

- Proceed to board-style replay: yes
- Identity accepted MAE: 3.157 C
- Medium accepted MAE: 3.180 C
- Strong accepted MAE: 3.253 C
- Worst accepted error after gating: 10.812 C
- Medium acceptance rate: 0.712
- Rejection rate (identity/medium/strong): 0.237 / 0.288 / 0.288

- The guardrails are strict enough to remove the catastrophic tail while preserving enough coverage for board-style replay.
- The remaining accepted predictions are materially better than the coordinate baseline.

## Explicit Answers

- Does guarded heatmap_v2 reduce the catastrophic tail enough? yes
- What is the accepted MAE on identity, mild, medium, and strong jitter? 3.157 / 2.806 / 3.180 / 3.253 C
- What is the worst accepted error after gating? 10.812 C
- What percentage of predictions are rejected? identity 0.237, mild 0.203, medium 0.288, strong 0.288
- Are rejections reasonable, or too aggressive? reasonable
- Should we proceed to board-style replay? yes
- Or should we train heatmap_v3 with stronger jitter and/or better tip supervision? no
