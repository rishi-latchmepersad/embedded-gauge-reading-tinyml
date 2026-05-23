# Inner Dial Angle Calibration v1

## Run Summary

- Clean rows fitted: 333
- Selected candidate: D_robust_linear
- Selection basis: lowest train MAE among fitted candidates B/C/D

## Candidate Comparison

| candidate | kind | train_mae_c | val_mae_c | test_mae_c | overall_mae_c |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_current_mapping | current | 1.742 | 1.765 | 1.587 | 1.718 |
| B_constrained_cold_sweep | cold_sweep | 1.392 | 1.527 | 1.345 | 1.403 |
| C_linear_unwrapped | linear | 1.209 | 1.160 | 1.187 | 1.198 |
| D_robust_linear | robust_linear | 1.208 | 1.140 | 1.190 | 1.195 |

## Selected Candidate Metrics

- Selected candidate: D_robust_linear
- Train MAE: 1.208 C
- Val MAE: 1.140 C
- Test MAE: 1.190 C
- Overall MAE: 1.195 C

## Temperature Label Residuals

| temperature_c | count | mae_c |
| --- | ---: | ---: |
| -29.000 | 27 | 1.020 |
| -17.500 | 40 | 1.100 |
| -7.500 | 41 | 1.370 |
| 5.000 | 40 | 0.939 |
| 15.000 | 45 | 0.941 |
| 26.500 | 37 | 0.896 |
| 37.500 | 54 | 1.424 |
| 49.000 | 49 | 1.641 |

## Image Dimension Residuals

| image_dims | count | mae_c |
| --- | ---: | ---: |
| 3472x4624 | 333 | 1.195 |

## Source Batch Residuals

| source_manifest | count | mae_c |
| --- | ---: | ---: |
| gauge_1_batch_5.zip | 50 | 0.873 |
| gauge_1_batch_7.zip | 50 | 1.578 |
| gauge_1_batch_6.zip | 49 | 1.441 |
| gauge_1_batch_4.zip | 48 | 0.944 |
| gauge_1_batch_3.zip | 46 | 1.303 |
| gauge_1_batch_2.zip | 45 | 0.984 |
| gauge_1_batch_1.zip | 43 | 1.189 |
| gauge_1_batch_8.zip | 2 | 2.115 |

## Dial Radius Residuals

- Bin edges: 568.2, 833.2, 958.4, 1165.8, 1680.0

| dial_radius_bin | count | mae_c |
| --- | ---: | ---: |
| [568.2, 833.2] | 84 | 1.184 |
| [1165.8, 1680.0] | 83 | 1.064 |
| [833.2, 958.4] | 83 | 1.400 |
| [958.4, 1165.8] | 83 | 1.133 |

## Visual Diagnostics

- Angle scatter: `D:\Projects\embedded-gauge-reading-tinyml\ml\debug\inner_dial_angle_calibration_v1\angle_vs_temperature_scatter.png`
- Prediction scatter: `D:\Projects\embedded-gauge-reading-tinyml\ml\debug\inner_dial_angle_calibration_v1\predicted_vs_manifest_temperature_scatter.png`
- Residuals by temperature: `D:\Projects\embedded-gauge-reading-tinyml\ml\debug\inner_dial_angle_calibration_v1\residuals_by_temperature_label.png`
- Residuals by source batch: `D:\Projects\embedded-gauge-reading-tinyml\ml\debug\inner_dial_angle_calibration_v1\residuals_by_source_batch.png`
- Worst overlays: `D:\Projects\embedded-gauge-reading-tinyml\ml\debug\inner_dial_angle_calibration_v1\oracle_mismatches`

## Interpretation

- Candidate C (linear unwrapped angle) is the best train-fitted model in this run.
- The remaining error after perfect geometry is largely calibration/label mismatch, not center/tip geometry error.
- Keep the calibration artifact and reuse it when judging the tiny-overfit heatmap outputs.
