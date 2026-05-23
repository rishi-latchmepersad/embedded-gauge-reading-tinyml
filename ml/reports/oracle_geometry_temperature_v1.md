# Oracle Geometry Temperature v1

## Run Summary

- Clean rows evaluated: 333
- Current mapping: cold_angle=135.0, sweep=270.0, min=-30.0, max=50.0
- Oracle MAE with perfect geometry: 1.718 C

## Split Metrics

| split | count | mae_c | max_error_c |
| --- | ---: | ---: | ---: |
| train | 227 | 1.742 | 4.914 |
| val | 47 | 1.765 | 4.712 |
| test | 59 | 1.587 | 4.895 |

## Temperature Label Metrics

| temperature_c | count | mae_c | max_error_c |
| --- | ---: | ---: | ---: |
| -29.000 | 27 | 3.414 | 4.914 |
| -17.500 | 40 | 1.574 | 3.361 |
| -7.500 | 41 | 2.851 | 4.862 |
| 5.000 | 40 | 1.091 | 3.005 |
| 15.000 | 45 | 1.225 | 2.871 |
| 26.500 | 37 | 0.883 | 4.535 |
| 37.500 | 54 | 1.562 | 4.856 |
| 49.000 | 49 | 1.720 | 4.886 |

## Image Dimension Metrics

| image_dims | count | mae_c | max_error_c |
| --- | ---: | ---: | ---: |
| 3472x4624 | 333 | 1.718 | 4.914 |

## Source Batch Metrics

| source_manifest | count | mae_c | max_error_c |
| --- | ---: | ---: | ---: |
| gauge_1_batch_5.zip | 50 | 0.994 | 4.535 |
| gauge_1_batch_7.zip | 50 | 1.627 | 4.886 |
| gauge_1_batch_6.zip | 49 | 1.576 | 4.856 |
| gauge_1_batch_4.zip | 48 | 1.344 | 3.005 |
| gauge_1_batch_3.zip | 46 | 1.902 | 4.862 |
| gauge_1_batch_2.zip | 45 | 2.128 | 4.783 |
| gauge_1_batch_1.zip | 43 | 2.562 | 4.914 |
| gauge_1_batch_8.zip | 2 | 2.974 | 3.058 |

## Dial Radius Bins

- Bin edges: 568.2, 833.2, 958.4, 1165.8, 1680.0

| dial_radius_bin | count | mae_c | mean_radius |
| --- | ---: | ---: | ---: |
| [568.2, 833.2] | 84 | 1.767 | 719.889 |
| [1165.8, 1680.0] | 83 | 2.143 | 1337.513 |
| [833.2, 958.4] | 83 | 1.494 | 899.691 |
| [958.4, 1165.8] | 83 | 1.469 | 1053.786 |

## Worst 10 Mismatches

| image_path | split | temperature_c | current_temperature_c | current_absolute_error_c | source_manifest |
| --- | ---: | ---: | ---: | ---: | ---: |
| ml/data/raw/PXL_20260125_114554365.jpg | train | -29.000 | -24.086 | 4.914 | gauge_1_batch_1.zip |
| ml/data/raw/PXL_20260125_114552322.jpg | train | -29.000 | -24.094 | 4.906 | gauge_1_batch_1.zip |
| ml/data/raw/PXL_20260125_114534732.jpg | test | -29.000 | -24.105 | 4.895 | gauge_1_batch_1.zip |
| ml/data/raw/PXL_20260125_120246531.jpg | train | 49.000 | 53.886 | 4.886 | gauge_1_batch_7.zip |
| ml/data/raw/PXL_20260125_115209157.jpg | test | -7.500 | -2.638 | 4.862 | gauge_1_batch_3.zip |
| ml/data/raw/PXL_20260125_115950996.jpg | train | 37.500 | 32.644 | 4.856 | gauge_1_batch_6.zip |
| ml/data/raw/PXL_20260125_115129858.jpg | train | -7.500 | -2.717 | 4.783 | gauge_1_batch_2.zip |
| ml/data/raw/PXL_20260125_115242004.jpg | train | -7.500 | -2.769 | 4.731 | gauge_1_batch_3.zip |
| ml/data/raw/PXL_20260125_114532964.jpg | train | -29.000 | -24.288 | 4.712 | gauge_1_batch_1.zip |
| ml/data/raw/PXL_20260125_114529783.jpg | val | -29.000 | -24.288 | 4.712 | gauge_1_batch_1.zip |

## Interpretation

- This is the irreducible temperature error from geometry alone under the current mapping.
- Any model that predicts the same center/tip labels perfectly cannot beat this ceiling without a better angle-to-temperature calibration or cleaner labels.
