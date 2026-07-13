# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 4.0050 C
- Acceptance rate: 0.7447
- Worst accepted error: 9.7973 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 25.53% / 51.06% / 74.47%
- Center MAE px: 15.7478
- Tip MAE px: 19.8779
- Angle MAE deg: 26.3185
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 3.9269 C
- Acceptance rate: 0.7447
- Worst accepted error: 9.6681 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.2045 / 0.1293 / 0.3483
- Center drift mean/median: 0.4149 / 0.2804
- Tip drift mean/median: 0.9589 / 0.7977
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 14.1663 / 12.5787
- Tip drift mean/median: 23.6155 / 20.9755
- Guardrail disagreements: 35

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]