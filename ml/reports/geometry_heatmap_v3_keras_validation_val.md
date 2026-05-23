# Geometry Heatmap v3 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras v3
- Accepted MAE: 3.6002 C
- Acceptance rate: 0.7021
- Worst accepted error: 13.5297 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 21.28% / 55.32% / 68.09%
- Center MAE px: 5.5034
- Tip MAE px: 20.4946
- Angle MAE deg: 36.6646
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 3.6002 C
- Acceptance rate: 0.7021
- Worst accepted error: 13.5295 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0001 / 0.0001
- Tip drift mean/median: 0.0003 / 0.0003
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 3.3048 C
- Acceptance rate: 0.5957
- Worst accepted error: 11.8713 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 1.9923 / 1.1288 / 4.2724
- Center drift mean/median: 3.0205 / 2.3763
- Tip drift mean/median: 14.7833 / 13.0247
- Guardrail disagreements: 7

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]