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
- Accepted MAE: 3.3857 C
- Acceptance rate: 0.7872
- Worst accepted error: 9.5194 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0000 / 0.0000
- Tip drift mean/median: 0.0002 / 0.0002
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 4.1346 C
- Acceptance rate: 0.6596
- Worst accepted error: 16.0458 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 1.9943 / 1.5482 / 3.8006
- Center drift mean/median: 0.6812 / 0.6493
- Tip drift mean/median: 11.5348 / 10.0840
- Guardrail disagreements: 9

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]