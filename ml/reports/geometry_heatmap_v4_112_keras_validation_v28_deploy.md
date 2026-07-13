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
- Accepted MAE: 4.0050 C
- Acceptance rate: 0.7447
- Worst accepted error: 9.7974 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0001 / 0.0001
- Tip drift mean/median: 0.0002 / 0.0002
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 2.1213 C
- Acceptance rate: 0.0213
- Worst accepted error: 2.1213 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.7544 / 0.7544 / 0.7544
- Center drift mean/median: 7.3033 / 6.4299
- Tip drift mean/median: 12.9243 / 9.1420
- Guardrail disagreements: 34

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]