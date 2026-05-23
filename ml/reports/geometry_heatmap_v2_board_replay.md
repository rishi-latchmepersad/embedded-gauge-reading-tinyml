# Geometry Heatmap v2 Board Replay

## Run Summary

- Model: `D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v2\model.keras`
- Calibration candidate: `D_robust_linear` (robust_linear)
- Selected preprocessing mode: `python_training_rgb_bilinear`
- Oracle calibrated geometry ceiling: 1.195 C

## Metrics by Mode and Split

| mode | split | total | accepted | clamped | rejected | acceptance_rate | accepted_mae_c | accepted_rmse_c | worst_accepted_error_c | under_2c_% | under_5c_% | under_10c_% | center_px_mae_224 | tip_px_mae_224 | angle_mae_degrees | center_peak_mean | tip_peak_mean | confidence_mean | top_rejection_reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| python_training_rgb_bilinear | train | 227 | 153 | 2 | 72 | 0.683 | 2.832 | 3.968 | 20.433 | 47.1 | 85.8 | 96.8 | 4.509 | 11.966 | 8.295 | 0.6570 | 0.6539 | 0.6883 | tip_peak_too_low:59, tip_heatmap_too_spread_out:26, center_tip_distance_ratio_implausible:19, predicted_angle_outside_valid_sweep:13, temperature_outside_physical_margin:12 |
| python_training_rgb_bilinear | val | 47 | 25 | 0 | 22 | 0.532 | 2.787 | 3.566 | 9.876 | 36.0 | 88.0 | 100.0 | 4.425 | 12.053 | 7.663 | 0.6612 | 0.5610 | 0.6872 | tip_peak_too_low:16, tip_heatmap_too_spread_out:8, center_tip_distance_ratio_implausible:7, predicted_angle_outside_valid_sweep:5, temperature_outside_physical_margin:5 |
| python_training_rgb_bilinear | test | 59 | 37 | 1 | 21 | 0.644 | 2.517 | 3.240 | 9.060 | 55.3 | 84.2 | 100.0 | 5.393 | 14.102 | 8.212 | 0.6573 | 0.5962 | 0.6757 | tip_peak_too_low:14, tip_heatmap_too_spread_out:11, center_tip_distance_ratio_implausible:7, predicted_angle_outside_valid_sweep:3, temperature_outside_physical_margin:2 |
| board_like_rgb_nearest | train | 227 | 117 | 1 | 109 | 0.520 | 3.325 | 4.485 | 21.776 | 39.0 | 80.5 | 96.6 | 6.059 | 14.855 | 10.337 | 0.6645 | 0.5296 | 0.7332 | tip_peak_too_low:99, tip_heatmap_too_spread_out:63, center_tip_distance_ratio_implausible:28, temperature_outside_physical_margin:20, predicted_angle_outside_valid_sweep:18 |
| board_like_rgb_nearest | val | 47 | 22 | 0 | 25 | 0.468 | 3.243 | 4.883 | 17.395 | 50.0 | 81.8 | 95.5 | 6.152 | 14.878 | 9.011 | 0.6697 | 0.5001 | 0.7276 | tip_peak_too_low:23, tip_heatmap_too_spread_out:15, center_tip_distance_ratio_implausible:9, predicted_angle_outside_valid_sweep:4, temperature_outside_physical_margin:3 |
| board_like_rgb_nearest | test | 59 | 28 | 0 | 31 | 0.475 | 4.139 | 4.950 | 11.980 | 17.9 | 67.9 | 96.4 | 5.831 | 18.052 | 12.831 | 0.6632 | 0.5062 | 0.7236 | tip_peak_too_low:29, tip_heatmap_too_spread_out:22, center_tip_distance_ratio_implausible:13, predicted_angle_outside_valid_sweep:5, temperature_outside_physical_margin:5 |
| board_like_luma_nearest_if_supported | train | 227 | 115 | 1 | 111 | 0.511 | 3.651 | 5.388 | 22.463 | 44.8 | 75.0 | 94.0 | 6.748 | 17.434 | 11.320 | 0.6456 | 0.5551 | 0.7170 | tip_peak_too_low:91, tip_heatmap_too_spread_out:68, center_tip_distance_ratio_implausible:41, predicted_angle_outside_valid_sweep:21, temperature_outside_physical_margin:18 |
| board_like_luma_nearest_if_supported | val | 47 | 25 | 1 | 21 | 0.553 | 3.280 | 4.343 | 9.417 | 46.2 | 76.9 | 100.0 | 5.549 | 15.048 | 9.972 | 0.6507 | 0.5464 | 0.7071 | tip_peak_too_low:16, tip_heatmap_too_spread_out:13, center_tip_distance_ratio_implausible:10, temperature_outside_physical_margin:5, predicted_angle_outside_valid_sweep:5 |
| board_like_luma_nearest_if_supported | test | 59 | 27 | 1 | 31 | 0.475 | 3.647 | 4.548 | 9.796 | 39.3 | 71.4 | 100.0 | 5.601 | 16.865 | 11.377 | 0.6418 | 0.5514 | 0.7100 | tip_peak_too_low:26, tip_heatmap_too_spread_out:24, center_tip_distance_ratio_implausible:17, predicted_angle_outside_valid_sweep:7, temperature_outside_physical_margin:7 |

## Comparison Against Guarded Replay

- Guarded identity accepted MAE: 3.157 C
- Guarded medium accepted MAE: 3.180 C
- Selected board replay test accepted MAE: 2.517 C
- Selected board replay test acceptance rate: 0.644
- Selected board replay test worst accepted error: 9.060 C
- Gap vs geometry_points_v1 test MAE: -5.393 C
- Gap vs calibrated oracle ceiling: 1.322 C

## Board Replay Interpretation

- Does board replay preserve guarded performance? partially
- Closest mode to training: `python_training_rgb_bilinear`
- Accuracy loss from board-like preprocessing: compare selected mode to `python_training_rgb_bilinear` in the table above.
- Worst accepted errors stay below 20 C: yes

## Worst 30 Accepted Predictions

| image | mode | split | abs_err_guarded | true_temp | guarded_temp | status | confidence | center_err | tip_err |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| PXL_20260125_120201512.jpg | board_like_luma_nearest_if_supported | train | 22.463 | 49.00 | 26.54 | accepted | 0.7063 | 12.66 | 84.60 |
| PXL_20260125_120201512.jpg | board_like_rgb_nearest | train | 21.776 | 49.00 | 27.22 | accepted | 0.7652 | 9.43 | 87.94 |
| PXL_20260125_120201512.jpg | python_training_rgb_bilinear | train | 20.433 | 49.00 | 28.57 | accepted | 0.6913 | 7.32 | 78.68 |
| PXL_20260125_120201091.jpg | board_like_luma_nearest_if_supported | train | 19.660 | 49.00 | 29.34 | accepted | 0.6952 | 4.18 | 78.58 |
| PXL_20260125_115240763.jpg | board_like_rgb_nearest | val | 17.395 | -7.50 | 9.90 | accepted | 0.7063 | 11.63 | 64.56 |
| PXL_20260125_115617076.jpg | board_like_luma_nearest_if_supported | train | 16.131 | 15.00 | 31.13 | accepted | 0.6606 | 5.38 | 64.13 |
| PXL_20260125_115150984.jpg | board_like_luma_nearest_if_supported | train | 15.061 | -7.50 | 7.56 | accepted | 0.7360 | 7.81 | 46.25 |
| PXL_20260125_115242004.jpg | board_like_luma_nearest_if_supported | train | 14.035 | -7.50 | 6.54 | accepted | 0.7102 | 7.19 | 43.91 |
| PXL_20260125_120201091.jpg | board_like_rgb_nearest | train | 13.124 | 49.00 | 35.88 | accepted | 0.7828 | 0.97 | 57.21 |
| PXL_20260125_120206223.jpg | board_like_luma_nearest_if_supported | train | 13.079 | 49.00 | 35.92 | accepted | 0.7375 | 6.15 | 55.95 |
| PXL_20260125_115335432.jpg | board_like_luma_nearest_if_supported | train | 13.019 | 5.00 | 18.02 | accepted | 0.7135 | 12.07 | 55.17 |
| PXL_20260125_120201091.jpg | python_training_rgb_bilinear | train | 12.627 | 49.00 | 36.37 | accepted | 0.7140 | 1.53 | 51.48 |
| PXL_20260125_115541259.jpg | python_training_rgb_bilinear | train | 12.008 | 15.00 | 2.99 | accepted | 0.6472 | 9.72 | 39.26 |
| PXL_20260125_115239306.jpg | board_like_rgb_nearest | test | 11.980 | -7.50 | 4.48 | accepted | 0.6795 | 4.69 | 45.86 |
| PXL_20260125_115242004.jpg | board_like_rgb_nearest | train | 11.611 | -7.50 | 4.11 | accepted | 0.7471 | 11.20 | 36.29 |
| PXL_20260125_115800564.jpg | board_like_rgb_nearest | train | 10.656 | 26.50 | 37.16 | accepted | 0.8049 | 5.06 | 47.84 |
| PXL_20260125_120034080.jpg | python_training_rgb_bilinear | train | 10.413 | 37.50 | 27.09 | accepted | 0.7627 | 4.38 | 27.91 |
| PXL_20260125_115948135.jpg | python_training_rgb_bilinear | train | 10.089 | 37.50 | 27.41 | accepted | 0.5776 | 9.45 | 35.92 |
| PXL_20260125_115647174.jpg | python_training_rgb_bilinear | val | 9.876 | 15.00 | 5.12 | accepted | 0.7832 | 4.27 | 35.04 |
| PXL_20260125_120024549.jpg | python_training_rgb_bilinear | train | 9.801 | 37.50 | 27.70 | accepted | 0.6687 | 11.63 | 35.16 |
| PXL_20260125_115948135.jpg | board_like_luma_nearest_if_supported | train | 9.798 | 37.50 | 27.70 | accepted | 0.5803 | 3.86 | 40.22 |
| PXL_20260125_115735138.jpg | board_like_luma_nearest_if_supported | test | 9.796 | 26.50 | 36.30 | accepted | 0.6895 | 5.29 | 32.49 |
| PXL_20260125_120024549.jpg | board_like_luma_nearest_if_supported | train | 9.598 | 37.50 | 27.90 | accepted | 0.7601 | 12.20 | 34.38 |
| PXL_20260125_120021544.jpg | board_like_luma_nearest_if_supported | val | 9.417 | 37.50 | 46.92 | accepted | 0.7533 | 12.51 | 32.55 |
| PXL_20260125_115115395.jpg | board_like_rgb_nearest | train | 9.364 | -7.50 | 1.86 | accepted | 0.7188 | 1.23 | 34.27 |
| PXL_20260125_120159765.jpg | board_like_luma_nearest_if_supported | val | 9.324 | 49.00 | 39.68 | accepted | 0.7211 | 1.38 | 37.19 |
| PXL_20260125_115615761.jpg | python_training_rgb_bilinear | test | 9.060 | 15.00 | 5.94 | accepted | 0.6928 | 10.07 | 38.06 |
| PXL_20260125_115735852.jpg | board_like_luma_nearest_if_supported | test | 9.003 | 26.50 | 35.50 | accepted | 0.6805 | 6.73 | 27.23 |
| PXL_20260125_115949522.jpg | python_training_rgb_bilinear | train | 8.939 | 37.50 | 28.56 | accepted | 0.6814 | 4.30 | 25.97 |
| PXL_20260125_115615761.jpg | board_like_rgb_nearest | test | 8.934 | 15.00 | 6.07 | accepted | 0.6905 | 9.98 | 35.58 |

## Export Readiness

- Meets export gate: no
- Required threshold summary: accepted MAE <= 4.5 C, acceptance rate >= 0.65, worst accepted error < 20 C.
