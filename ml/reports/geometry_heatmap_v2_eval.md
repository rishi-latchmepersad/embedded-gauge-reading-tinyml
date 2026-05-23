# Geometry Heatmap v2 Evaluation

## Run Summary

- Model: `D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v2\model.keras`
- Selected stage: frozen
- Calibration candidate: D_robust_linear (robust_linear)
- Oracle calibrated geometry ceiling: 1.195 C

## Split Metrics

| split | count | center_px_mae_224 | tip_px_mae_224 | angle_mae_degrees | temp_mae_current | temp_mae_calibrated | rmse_calibrated | under_2c_% | under_5c_% | under_10c_% | center_peak_mean | center_peak_median | tip_peak_mean | tip_peak_median | confidence_mean | confidence_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 227 | 2.691 | 9.621 | 6.034 | 2.551 | 2.235 | 4.494 | 62.1 | 91.6 | 99.1 | 0.6660 | 0.6651 | 0.7457 | 0.8296 | 0.6820 | 0.6819 |
| val | 47 | 4.347 | 13.713 | 8.045 | 5.204 | 4.785 | 15.580 | 46.8 | 85.1 | 97.9 | 0.6699 | 0.6601 | 0.6436 | 0.6710 | 0.6792 | 0.6710 |
| test | 59 | 4.764 | 17.850 | 12.348 | 4.584 | 4.312 | 8.978 | 45.8 | 74.6 | 94.9 | 0.6624 | 0.6590 | 0.6431 | 0.6384 | 0.6721 | 0.6732 |

## Baseline Comparison

- geometry_points_v1 test temperature MAE: 7.91 C
- geometry_points_v1 test center MAE: 11.30 px
- geometry_points_v1 test tip MAE: 21.82 px
- Heatmap v2 test calibrated MAE: 4.312 C
- Heatmap v2 test tip MAE: 17.850 px
- Heatmap v2 test center MAE: 4.764 px
- Heatmap v2 is within 3.117 C of the oracle geometry ceiling.

## Decision Checks

- Beats geometry_points_v1 test MAE: yes
- Beats geometry_points_v1 tip MAE: yes
- Reduces catastrophic errors: partially
- Good enough for board-style replay: no

## Worst 30 Predictions

| image | split | abs_err_calibrated | temp_true | temp_calibrated | center_err | tip_err | confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PXL_20260125_114529783.jpg | val | 104.396 | -29.00 | 75.40 | 3.93 | 41.09 | 0.6597 |
| PXL_20260125_115125409.jpg | test | 59.428 | -7.50 | 51.93 | 7.89 | 116.25 | 0.7035 |
| PXL_20260125_120305774.jpg | train | 53.946 | 49.00 | -4.95 | 3.78 | 106.12 | 0.7037 |
| PXL_20260125_115957052.jpg | test | 14.114 | 37.50 | 23.39 | 3.86 | 39.15 | 0.7726 |
| PXL_20260125_114520969.jpg | test | 10.812 | -29.00 | -18.19 | 3.94 | 34.34 | 0.6526 |
| PXL_20260125_120246531.jpg | train | 10.475 | 49.00 | 59.47 | 1.20 | 35.70 | 0.6355 |
| PXL_20260125_120220792.jpg | test | 9.625 | 49.00 | 39.38 | 5.77 | 38.68 | 0.6983 |
| PXL_20260125_120228002.jpg | test | 9.124 | 49.00 | 58.12 | 4.93 | 44.18 | 0.8087 |
| PXL_20260125_114911288.jpg | test | 9.112 | -17.50 | -26.61 | 10.98 | 29.39 | 0.6584 |
| PXL_20260125_114909671.jpg | val | 8.960 | -17.50 | -26.46 | 7.32 | 26.90 | 0.6600 |
| PXL_20260125_115647174.jpg | val | 8.241 | 15.00 | 6.76 | 3.77 | 27.39 | 0.7901 |
| PXL_20260125_120119590.jpg | test | 8.069 | 37.50 | 29.43 | 2.70 | 25.40 | 0.7189 |
| PXL_20260125_120256738.jpg | test | 7.899 | 49.00 | 41.10 | 4.72 | 26.68 | 0.6723 |
| PXL_20260125_114926493.jpg | train | 7.748 | -17.50 | -25.25 | 3.70 | 30.97 | 0.6622 |
| PXL_20260125_114845174.jpg | train | 7.748 | -17.50 | -25.25 | 1.27 | 24.76 | 0.6479 |
| PXL_20260125_115615761.jpg | test | 7.743 | 15.00 | 7.26 | 7.16 | 34.00 | 0.6831 |
| PXL_20260125_114538442.jpg | train | 7.695 | -29.00 | -21.30 | 3.95 | 18.93 | 0.5803 |
| PXL_20260125_120227628.jpg | train | 7.590 | 49.00 | 56.59 | 1.54 | 31.78 | 0.8100 |
| PXL_20260125_115758313.jpg | train | 7.393 | 26.50 | 33.89 | 2.06 | 27.28 | 0.7359 |
| PXL_20260125_120020529.jpg | test | 7.278 | 37.50 | 30.22 | 5.07 | 35.75 | 0.7008 |
| PXL_20260125_120026856.jpg | train | 7.112 | 37.50 | 30.39 | 2.80 | 11.71 | 0.7826 |
| PXL_20260125_115612552.jpg | test | 7.026 | 15.00 | 22.03 | 6.84 | 20.72 | 0.7195 |
| PXL_20260125_115807658.jpg | train | 6.592 | 26.50 | 19.91 | 2.48 | 24.75 | 0.6536 |
| PXL_20260125_120202892.jpg | val | 6.427 | 49.00 | 55.43 | 4.89 | 27.20 | 0.7002 |
| PXL_20260125_115950638.jpg | train | 6.395 | 37.50 | 31.10 | 2.71 | 4.22 | 0.6236 |
| PXL_20260125_114524322.jpg | train | 6.367 | -29.00 | -22.63 | 4.45 | 16.17 | 0.6407 |
| PXL_20260125_115239306.jpg | test | 6.229 | -7.50 | -1.27 | 1.67 | 21.70 | 0.5986 |
| PXL_20260125_115822580.jpg | val | 6.105 | 26.50 | 32.61 | 5.14 | 16.51 | 0.6847 |
| PXL_20260125_120216879.jpg | train | 6.078 | 49.00 | 55.08 | 3.21 | 25.11 | 0.7122 |
| PXL_20260125_115735138.jpg | test | 6.030 | 26.50 | 32.53 | 3.60 | 22.98 | 0.6414 |

## Interpretation

- The calibrated temperature gap versus the oracle ceiling is 3.117 C on test.
- The model is not yet below the coordinate baseline on the metrics that matter for board-style replay.
