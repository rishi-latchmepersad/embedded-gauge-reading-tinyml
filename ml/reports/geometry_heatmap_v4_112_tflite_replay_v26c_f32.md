# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 3.8411 C
- Acceptance rate: 0.6809
- Worst accepted error: 9.9647 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 23.40% / 51.06% / 68.09%
- Center MAE px: 16.3111
- Tip MAE px: 20.7387
- Angle MAE deg: 34.7808
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 3.8411 C
- Acceptance rate: 0.6809
- Worst accepted error: 9.9647 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0001 / 0.0001
- Tip drift mean/median: 0.0002 / 0.0002
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 15.4916 / 13.5586
- Tip drift mean/median: 23.2153 / 20.6740
- Guardrail disagreements: 32

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]