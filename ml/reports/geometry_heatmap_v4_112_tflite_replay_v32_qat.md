# Geometry Heatmap v4 112 TFLite Replay

- Split: val
- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation gate passed: no

## Keras (val)
- Accepted MAE: 4.0174 C
- Acceptance rate: 0.7447
- Worst accepted error: 9.8009 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 25.53% / 51.06% / 74.47%
- Center MAE px: 15.7254
- Tip MAE px: 19.9242
- Angle MAE deg: 26.3508
- Temp drift mean/median/p90: nan / nan / nan

## TFLite FP32
- Accepted MAE: nan C
- Acceptance rate: 0.0000
- Worst accepted error: nan C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: nan / nan / nan
- Center drift mean/median: 14.2577 / 12.6174
- Tip drift mean/median: 17.6513 / 14.5986
- Guardrail disagreements: 35

## TFLite INT8
- Accepted MAE: 2.0980 C
- Acceptance rate: 0.0213
- Worst accepted error: 2.0980 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 0.7575 / 0.7575 / 0.7575
- Center drift mean/median: 7.2186 / 6.4863
- Tip drift mean/median: 12.8646 / 9.0004
- Guardrail disagreements: 34

## Tensor Contract
- FP32 input dtype: float32
- INT8 input dtype: int8
- FP32 output dtypes: float32, float32, float32
- INT8 output dtypes: int8, int8, int8
- Semantic output reorder: [1, 0, 2]