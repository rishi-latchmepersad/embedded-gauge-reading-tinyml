# Geometry Heatmap v2 Failure Analysis

## Run Summary

- Model: `D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v2\model.keras`
- Calibration candidate: D_robust_linear (robust_linear)
- Guardrail thresholds: center_peak>=0.40, tip_peak>=0.40, confidence>=0.40, ratio=[0.50, 1.50], edge_margin=4.0px, temp_margin=5.0C

- Overall calibrated MAE: 4.638 C
- Overall clamped calibrated MAE: 4.343 C
- Overall current-mapping MAE: 4.780 C
- Overall clamped current-mapping MAE: 4.536 C
- Worst calibrated error: 105.195 C
- Worst clamped calibrated error: 79.000 C

## Jitter-Level Summary

| level | count | calibrated_mae | clamped_mae | worst | clamped_worst | accepted_fraction | rejected_fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 59 | 4.312 | 4.135 | 59.428 | 57.500 | 0.763 | 0.237 |
| mild | 59 | 4.858 | 4.338 | 73.604 | 57.500 | 0.797 | 0.203 |
| medium | 59 | 3.688 | 3.652 | 18.727 | 18.727 | 0.729 | 0.271 |
| strong | 59 | 5.692 | 5.247 | 105.195 | 79.000 | 0.729 | 0.271 |

## What Broke

- The worst cases are dominated by tip localization failures and center-tip distance implausibility.
- Several failures also push the calibrated temperature outside the physical gauge range, which indicates extrapolation instead of safe interpolation.
- Peak confidence alone is not sufficient, so the guardrail also checks geometry consistency and temperature plausibility.

## Top 30 Worst Predictions

| rank | level | image | abs_err_calibrated | clamped_err | mode | guard | temp_calibrated | temp_clamped | center_err | tip_err | ratio | peaks c/t | confidence | reasons |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | strong | PXL_20260125_114606035.jpg | 105.195 | 79.000 | calibration_extrapolation | rejected | 76.19 | 50.00 | 0.98 | 40.45 | 0.341 | 0.602 / 0.563 | 0.539 | predicted_angle_outside_valid_sweep|temperature_outside_physical_margin|center_tip_distance_ratio_implausible |
| 2 | mild | PXL_20260125_115125409.jpg | 73.604 | 57.500 | calibration_extrapolation | rejected | 66.10 | 50.00 | 7.85 | 96.05 | 0.078 | 0.590 / 0.340 | 0.694 | tip_peak_too_low|tip_heatmap_too_spread_out|predicted_angle_outside_valid_sweep|temperature_outside_physical_margin|center_tip_distance_ratio_implausible |
| 3 | identity | PXL_20260125_115125409.jpg | 59.428 | 57.500 | angle_wrap_or_opposite_side | rejected | 51.93 | 50.00 | 7.89 | 116.25 | 0.250 | 0.586 / 0.341 | 0.704 | tip_peak_too_low|predicted_angle_outside_valid_sweep|center_tip_distance_ratio_implausible |
| 4 | medium | PXL_20260125_115957052.jpg | 18.727 | 18.727 | bad_tip_localization | rejected | 18.77 | 18.77 | 1.38 | 56.64 | 0.264 | 0.656 / 0.378 | 0.804 | tip_peak_too_low|tip_heatmap_too_spread_out|center_tip_distance_ratio_implausible |
| 5 | mild | PXL_20260125_115957052.jpg | 15.083 | 15.083 | bad_tip_localization | rejected | 22.42 | 22.42 | 4.43 | 42.31 | 0.444 | 0.677 / 0.515 | 0.779 | tip_heatmap_too_spread_out|center_tip_distance_ratio_implausible |
| 6 | strong | PXL_20260125_120256738.jpg | 14.425 | 14.425 | bad_tip_localization | rejected | 34.57 | 34.57 | 6.57 | 39.75 | 0.619 | 0.611 / 0.390 | 0.681 | tip_peak_too_low |
| 7 | identity | PXL_20260125_115957052.jpg | 14.114 | 14.114 | bad_tip_localization | rejected | 23.39 | 23.39 | 3.86 | 39.15 | 0.496 | 0.684 / 0.497 | 0.773 | tip_heatmap_too_spread_out|center_tip_distance_ratio_implausible |
| 8 | medium | PXL_20260125_114925728.jpg | 12.673 | 12.500 | bad_tip_localization | rejected | -30.17 | -30.00 | 3.88 | 54.89 | 0.276 | 0.646 / 0.537 | 0.605 | center_tip_distance_ratio_implausible |
| 9 | strong | PXL_20260125_114911288.jpg | 12.532 | 12.500 | bad_tip_localization | rejected | -30.03 | -30.00 | 7.87 | 47.27 | 0.570 | 0.668 / 0.255 | 0.663 | tip_peak_too_low |
| 10 | mild | PXL_20260125_120045775.jpg | 12.165 | 12.165 | bad_tip_localization | rejected | 49.66 | 49.66 | 11.55 | 40.08 | 0.421 | 0.622 / 0.339 | 0.765 | tip_peak_too_low|center_tip_distance_ratio_implausible |
| 11 | identity | PXL_20260125_114520969.jpg | 10.812 | 10.812 | bad_tip_localization | accepted | -18.19 | -18.19 | 3.94 | 34.34 | 0.536 | 0.619 / 0.654 | 0.653 |  |
| 12 | strong | PXL_20260125_114520969.jpg | 10.812 | 10.812 | bad_tip_localization | accepted | -18.19 | -18.19 | 3.94 | 34.34 | 0.536 | 0.619 / 0.654 | 0.653 |  |
| 13 | strong | PXL_20260125_120119590.jpg | 10.730 | 10.730 | bad_tip_localization | rejected | 26.77 | 26.77 | 2.71 | 40.65 | 0.416 | 0.625 / 0.295 | 0.736 | tip_peak_too_low|center_tip_distance_ratio_implausible |
| 14 | strong | PXL_20260125_114534732.jpg | 10.229 | 10.229 | bad_tip_localization | rejected | -18.77 | -18.77 | 2.00 | 28.32 | 0.671 | 0.620 / 0.343 | 0.657 | tip_peak_too_low |
| 15 | identity | PXL_20260125_120220792.jpg | 9.625 | 9.625 | bad_tip_localization | accepted | 39.38 | 39.38 | 5.77 | 38.68 | 0.761 | 0.659 / 0.528 | 0.698 |  |
| 16 | strong | PXL_20260125_120220792.jpg | 9.546 | 9.546 | bad_tip_localization | rejected | 39.45 | 39.45 | 6.71 | 40.55 | 0.591 | 0.733 / 0.363 | 0.689 | tip_peak_too_low |
| 17 | identity | PXL_20260125_120228002.jpg | 9.124 | 1.000 | calibration_extrapolation | rejected | 58.12 | 50.00 | 4.93 | 44.18 | 0.482 | 0.605 / 0.273 | 0.809 | tip_peak_too_low|predicted_angle_outside_valid_sweep|temperature_outside_physical_margin|center_tip_distance_ratio_implausible |
| 18 | mild | PXL_20260125_120119590.jpg | 9.123 | 9.123 | bad_tip_localization | rejected | 28.38 | 28.38 | 3.50 | 30.23 | 0.720 | 0.738 / 0.397 | 0.726 | tip_peak_too_low |
| 19 | strong | PXL_20260125_120159244.jpg | 9.113 | 9.113 | bad_tip_localization | rejected | 39.89 | 39.89 | 4.25 | 56.23 | 0.263 | 0.722 / 0.278 | 0.758 | tip_peak_too_low|center_tip_distance_ratio_implausible |
| 20 | identity | PXL_20260125_114911288.jpg | 9.112 | 9.112 | bad_tip_localization | accepted | -26.61 | -26.61 | 10.98 | 29.39 | 0.756 | 0.636 / 0.701 | 0.658 |  |
| 21 | mild | PXL_20260125_115615761.jpg | 9.077 | 9.077 | bad_tip_localization | accepted | 5.92 | 5.92 | 9.02 | 37.01 | 0.822 | 0.742 / 0.805 | 0.682 |  |
| 22 | mild | PXL_20260125_120302481.jpg | 8.555 | 1.000 | calibration_extrapolation | rejected | 57.55 | 50.00 | 4.70 | 46.18 | 0.371 | 0.550 / 0.277 | 0.729 | tip_peak_too_low|predicted_angle_outside_valid_sweep|temperature_outside_physical_margin|center_tip_distance_ratio_implausible |
| 23 | medium | PXL_20260125_114520969.jpg | 8.469 | 8.469 | bad_tip_localization | accepted | -20.53 | -20.53 | 5.61 | 25.79 | 0.585 | 0.639 / 0.755 | 0.665 |  |
| 24 | mild | PXL_20260125_114520969.jpg | 8.436 | 8.436 | bad_tip_localization | accepted | -20.56 | -20.56 | 3.54 | 25.81 | 0.563 | 0.620 / 0.710 | 0.648 |  |
| 25 | medium | PXL_20260125_120020529.jpg | 8.164 | 8.164 | bad_tip_localization | rejected | 29.34 | 29.34 | 7.19 | 40.40 | 0.681 | 0.662 / 0.349 | 0.741 | tip_peak_too_low |
| 26 | identity | PXL_20260125_120119590.jpg | 8.069 | 8.069 | bad_tip_localization | accepted | 29.43 | 29.43 | 2.70 | 25.40 | 0.685 | 0.753 / 0.419 | 0.719 |  |
| 27 | medium | PXL_20260125_120045775.jpg | 7.952 | 7.952 | bad_tip_localization | accepted | 45.45 | 45.45 | 8.95 | 27.60 | 0.553 | 0.570 / 0.431 | 0.766 |  |
| 28 | identity | PXL_20260125_120256738.jpg | 7.899 | 7.899 | bad_tip_localization | rejected | 41.10 | 41.10 | 4.72 | 26.68 | 0.515 | 0.670 / 0.353 | 0.672 | tip_peak_too_low |
| 29 | identity | PXL_20260125_115615761.jpg | 7.743 | 7.743 | bad_tip_localization | accepted | 7.26 | 7.26 | 7.16 | 34.00 | 0.784 | 0.788 / 0.863 | 0.683 |  |
| 30 | mild | PXL_20260125_120020529.jpg | 7.583 | 7.583 | bad_tip_localization | rejected | 29.92 | 29.92 | 8.59 | 37.52 | 0.720 | 0.673 / 0.349 | 0.742 | tip_peak_too_low |

## Catastrophic Tail Check

- Worst calibrated error row: PXL_20260125_114606035.jpg at strong jitter.
- Raw calibrated temperature: 76.19 C.
- Clamped calibrated temperature: 50.00 C.
- True temperature: -29.00 C.
- Clamping changes the error from 105.19 C to 79.00 C, so it helps a little but does not solve the tail.
- This row is rejected by the reference guardrails.

## Guardrail Coverage

- Total >20 C errors: 3
- >20 C errors rejected by the reference guardrails: 3
- >20 C errors clamped rather than rejected: 0
- Top-30 failure-mode counts: {'calibration_extrapolation': 4, 'angle_wrap_or_opposite_side': 1, 'bad_tip_localization': 25}

## Interpretation

- The tail is mostly a geometry/tip-localization problem, not a pure heatmap-confidence problem.
- Physical clamping alone is not enough because the worst errors remain far from the target even after clipping.
- The reference guardrails are able to identify the worst rows well enough to reject them instead of returning a wildly wrong temperature.
