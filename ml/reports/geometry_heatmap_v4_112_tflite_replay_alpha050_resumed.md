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
- Accepted MAE: 3.8360 C
- Acceptance rate: 0.8936
- Worst accepted error: 11.2365 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 2.5620 / 1.6219 / 5.4127
- Center drift mean/median: 3.4942 / 3.3716
- Tip drift mean/median: 19.3039 / 17.0342
- Guardrail disagreements: 8

## TFLite INT8
- Accepted MAE: 3.8616 C
- Acceptance rate: 0.7021
- Worst accepted error: 16.9250 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 3.9710 / 2.5102 / 7.2653
- Center drift mean/median: 4.0292 / 3.8944
- Tip drift mean/median: 23.2390 / 22.2668
- Guardrail disagreements: 13

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]