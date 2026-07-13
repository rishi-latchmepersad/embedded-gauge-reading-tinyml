# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 3.1011 C
- Acceptance rate: 0.8085
- Worst accepted error: 9.6058 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 31.91% / 63.83% / 80.85%
- Center MAE px: 8.2567
- Tip MAE px: 22.8177
- Angle MAE deg: 25.5831
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 43.0418 / 45.8485
- Tip drift mean/median: 55.9265 / 56.1657
- Guardrail disagreements: 38

## TFLite INT8
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 51.1620 / 54.2955
- Tip drift mean/median: 57.3655 / 57.4199
- Guardrail disagreements: 38

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [0, 2, 1]