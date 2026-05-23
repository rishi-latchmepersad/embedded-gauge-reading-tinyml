# Geometry Heatmap v3 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras v3
- Accepted MAE: 3.5472 C
- Acceptance rate: 0.7021
- Worst accepted error: 12.2516 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 21.28% / 55.32% / 68.09%
- Center MAE px: 5.5131
- Tip MAE px: 20.4532
- Angle MAE deg: 36.4667
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 3.5472 C
- Acceptance rate: 0.7021
- Worst accepted error: 12.2514 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0001 / 0.0000
- Tip drift mean/median: 0.0003 / 0.0003
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 3.3839 C
- Acceptance rate: 0.5745
- Worst accepted error: 11.8365 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 2.0145 / 1.1273 / 4.2997
- Center drift mean/median: 3.0235 / 2.3285
- Tip drift mean/median: 14.8213 / 12.4642
- Guardrail disagreements: 7

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]