# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 3.3857 C
- Acceptance rate: 0.7872
- Worst accepted error: 9.5193 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 27.66% / 61.70% / 78.72%
- Center MAE px: 8.3359
- Tip MAE px: 26.7123
- Angle MAE deg: 26.6005
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 79.5665 / 82.5152
- Tip drift mean/median: 69.9278 / 68.4313
- Guardrail disagreements: 37

## TFLite INT8
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 86.0650 / 88.7294
- Tip drift mean/median: 73.1431 / 70.4483
- Guardrail disagreements: 37

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [0, 2, 1]