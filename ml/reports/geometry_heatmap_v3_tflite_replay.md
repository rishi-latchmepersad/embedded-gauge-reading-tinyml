# Geometry Heatmap v3 TFLite Replay

## Validation Replay

The exported v3 TFLite pair was replayed on the validation split only.

### Keras v3

- accepted MAE: `3.6002 C`
- acceptance rate: `0.7021`
- worst accepted error: `13.5297 C`
- accepted >20 C failures: `0`

### TFLite FP32

- accepted MAE: `3.6002 C`
- acceptance rate: `0.7021`
- worst accepted error: `13.5295 C`
- accepted >20 C failures: `0`
- temperature drift mean/median/p90: `0.0000 C / 0.0000 C / 0.0001 C`
- center drift mean/median: `0.0001 px / 0.0001 px`
- tip drift mean/median: `0.0003 px / 0.0003 px`
- guardrail disagreements: `0`

### TFLite INT8

- accepted MAE: `3.3048 C`
- acceptance rate: `0.5957`
- worst accepted error: `11.8713 C`
- accepted >20 C failures: `0`
- temperature drift mean/median/p90: `1.9923 C / 1.1288 C / 4.2724 C`
- center drift mean/median: `3.0205 px / 2.3763 px`
- tip drift mean/median: `14.7833 px / 13.0247 px`
- guardrail disagreements: `7`

## Tensor Contract

- Decoder: `softargmax w3`
- Semantic output reorder: `[1, 0, 2]`
- FP32 input dtype: `float32`
- INT8 input dtype: `int8`
- FP32 output dtypes: `float32, float32, float32`
- INT8 output dtypes: `int8, int8, int8`

## Decision

The validation gate failed because the INT8 model did not meet the required acceptance and temperature-drift thresholds.

Test was not run.
