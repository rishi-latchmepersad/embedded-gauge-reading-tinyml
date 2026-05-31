# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 4.8450 C
- Acceptance rate: 0.4255
- Worst accepted error: 13.3422 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 10.64% / 25.53% / 36.17%
- Center MAE px: 9.5874
- Tip MAE px: 48.6812
- Angle MAE deg: 100.4136
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: 4.8450 C
- Acceptance rate: 0.4255
- Worst accepted error: 13.3420 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Center drift mean/median: 0.0000 / 0.0000
- Tip drift mean/median: 0.0002 / 0.0002
- Guardrail disagreements: 0

## TFLite INT8
- Accepted MAE: 5.2775 C
- Acceptance rate: 0.4255
- Worst accepted error: 19.2520 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 2.7654 / 2.4181 / 4.2774
- Center drift mean/median: 1.1518 / 1.0291
- Tip drift mean/median: 12.5746 / 11.2602
- Guardrail disagreements: 6

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]