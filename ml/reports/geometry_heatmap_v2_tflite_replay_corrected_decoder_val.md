# Geometry Heatmap v2 TFLite Replay

## Decode Selection

- Selected decode name: `softargmax_w3`
- Selected decode method: `softargmax`
- Selected decode window size: `3`

## Replay Summary

| model | split | accepted MAE | acceptance rate | worst accepted error | rejected | clamped | center MAE | tip MAE | angle MAE | center peak mean | tip peak mean | confidence mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keras_fp32 | val | 3.260 | 0.660 | 11.334 | 16 | 0 | 5.570 | 21.116 | 14.608 | 0.6612 | 0.5610 | 0.6872 |
| tflite_fp32 | val | 3.260 | 0.660 | 11.333 | 16 | 0 | 5.570 | 21.116 | 14.608 | 0.6612 | 0.5610 | 0.6872 |
| tflite_int8 | val | 3.293 | 0.638 | 11.501 | 17 | 2 | 6.176 | 26.312 | 21.174 | 0.6675 | 0.5164 | 0.7211 |

## Drift Against Keras

| split | Keras vs FP32 temp delta mean | Keras vs FP32 temp delta median | Keras vs INT8 temp delta mean | Keras vs INT8 temp delta median | INT8 center delta mean | INT8 tip delta mean | INT8 center peak delta mean | INT8 tip peak delta mean | rejection disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | nan | nan | nan | nan | nan | nan | nan | nan | 0 |
| val | 0.000 | 0.000 | 1.638 | 1.236 | 2.995 | 14.693 | 0.0351 | 0.0750 | 8 |
| test | nan | nan | nan | nan | nan | nan | nan | nan | 0 |

## Notes

- Accepted metrics are computed on accepted and clamped predictions.
- Center/tip/angle/peak/confidence metrics are reported over all rows.
