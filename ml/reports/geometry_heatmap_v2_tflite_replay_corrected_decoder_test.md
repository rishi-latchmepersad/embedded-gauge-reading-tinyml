# Geometry Heatmap v2 TFLite Replay

## Decode Selection

- Selected decode name: `softargmax_w3`
- Selected decode method: `softargmax`
- Selected decode window size: `3`

## Replay Summary

| model | split | accepted MAE | acceptance rate | worst accepted error | rejected | clamped | center MAE | tip MAE | angle MAE | center peak mean | tip peak mean | confidence mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keras_fp32 | test | 3.555 | 0.814 | 17.459 | 11 | 1 | 5.551 | 22.931 | 14.457 | 0.6573 | 0.5962 | 0.6757 |
| tflite_fp32 | test | 3.555 | 0.814 | 17.459 | 11 | 1 | 5.551 | 22.931 | 14.457 | 0.6573 | 0.5962 | 0.6757 |
| tflite_int8 | test | 3.706 | 0.712 | 14.588 | 17 | 0 | 6.337 | 22.208 | 12.802 | 0.6709 | 0.5093 | 0.7143 |

## Drift Against Keras

| split | Keras vs FP32 temp delta mean | Keras vs FP32 temp delta median | Keras vs INT8 temp delta mean | Keras vs INT8 temp delta median | INT8 center delta mean | INT8 tip delta mean | INT8 center peak delta mean | INT8 tip peak delta mean | rejection disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | nan | nan | nan | nan | nan | nan | nan | nan | 0 |
| val | nan | nan | nan | nan | nan | nan | nan | nan | 0 |
| test | 0.000 | 0.000 | 1.735 | 1.442 | 2.855 | 14.020 | 0.0391 | 0.0970 | 12 |

## Notes

- Accepted metrics are computed on accepted and clamped predictions.
- Center/tip/angle/peak/confidence metrics are reported over all rows.
