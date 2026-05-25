# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 6.9625 C
- Acceptance rate: 0.7660
- Worst accepted error: 54.2412 C
- Accepted >20 C failures: 1
- Under 2/5/10 C: 10.64% / 31.91% / 65.96%
- Center MAE px: 22.5324
- Tip MAE px: 27.8956
- Angle MAE deg: 32.4460
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 6.9624 C
- Acceptance rate: 0.7660
- Worst accepted error: 54.2411 C
- Accepted >20 C failures: 1
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0000 / 0.0000
- Tip drift mean/median: 0.0003 / 0.0002
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 7.3914 C
- Acceptance rate: 0.7660
- Worst accepted error: 42.4801 C
- Accepted >20 C failures: 2
- Temp drift mean/median/p90: 1.7108 / 1.3472 / 3.0880
- Center drift mean/median: 2.7714 / 2.7810
- Tip drift mean/median: 13.7627 / 10.8498
- Guardrail disagreements: 6

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32, float32
- INT8 output dtypes: int8, int8, int8, int8
- Semantic output reorder: [2, 0, 3, 1]