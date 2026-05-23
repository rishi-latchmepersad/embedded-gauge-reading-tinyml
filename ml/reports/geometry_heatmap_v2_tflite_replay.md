# Geometry Heatmap v2 TFLite Replay

## Replay Summary

| model | split | accepted MAE | acceptance rate | worst accepted error | rejected | clamped | center MAE | tip MAE | angle MAE | center peak mean | tip peak mean | confidence mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keras_fp32 | train | 2.953 | 0.797 | 20.433 | 46 | 4 | 4.764 | 17.768 | 11.773 | 0.6570 | 0.6539 | 0.6883 |
| tflite_fp32 | train | 2.953 | 0.797 | 20.433 | 46 | 4 | 4.764 | 17.768 | 11.773 | 0.6570 | 0.6539 | 0.6883 |
| tflite_int8 | train | 3.036 | 0.731 | 19.667 | 61 | 5 | 5.372 | 20.219 | 14.073 | 0.6654 | 0.5570 | 0.7244 |
| keras_fp32 | val | 3.260 | 0.660 | 11.334 | 16 | 0 | 5.491 | 20.937 | 14.608 | 0.6612 | 0.5610 | 0.6872 |
| tflite_fp32 | val | 3.260 | 0.660 | 11.334 | 16 | 0 | 5.491 | 20.937 | 14.608 | 0.6612 | 0.5610 | 0.6872 |
| tflite_int8 | val | 3.293 | 0.638 | 11.501 | 17 | 2 | 6.102 | 26.378 | 21.174 | 0.6675 | 0.5164 | 0.7211 |
| keras_fp32 | test | 3.555 | 0.814 | 17.459 | 11 | 1 | 5.528 | 22.739 | 14.457 | 0.6573 | 0.5962 | 0.6757 |
| tflite_fp32 | test | 3.555 | 0.814 | 17.459 | 11 | 1 | 5.528 | 22.739 | 14.457 | 0.6573 | 0.5962 | 0.6757 |
| tflite_int8 | test | 3.706 | 0.712 | 14.588 | 17 | 0 | 6.403 | 22.183 | 12.802 | 0.6709 | 0.5093 | 0.7143 |

## Drift Against Keras

| split | Keras vs FP32 temp delta mean | Keras vs FP32 temp delta median | Keras vs INT8 temp delta mean | Keras vs INT8 temp delta median | INT8 center delta mean | INT8 tip delta mean | INT8 center peak delta mean | INT8 tip peak delta mean | rejection disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.000 | 0.000 | 1.669 | 1.124 | 3.350 | 11.679 | 0.0353 | 0.1107 | 37 |
| val | 0.000 | 0.000 | 1.638 | 1.236 | 3.008 | 14.759 | 0.0351 | 0.0750 | 8 |
| test | 0.000 | 0.000 | 1.735 | 1.442 | 2.868 | 14.083 | 0.0391 | 0.0970 | 12 |

## Notes

- Accepted metrics are computed on accepted and clamped predictions.
- Center/tip/angle/peak/confidence metrics are reported over all rows.
