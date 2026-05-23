# Geometry Heatmap v2 Corrected Decoder Selected Variant Test

## Locked Decoder

- Decoder: `softargmax`
- Window size: `3`
- Decode artifact: `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json`

## Selected Variant

- Variant: `variant_d_dynamic_range`
- Tensor contract input dtype: `float32`
- Tensor contract output dtypes: `float32|float32|float32`
- Tensor contract output names: `StatefulPartitionedCall_1:1|StatefulPartitionedCall_1:0|StatefulPartitionedCall_1:2`
- Requires dequantization: `False`

## Test Metrics

- Accepted MAE: `3.4466 C`
- Acceptance rate: `0.7288`
- Worst accepted error: `13.9186 C`
- Accepted >20 C failures: `0`
- Keras-vs-selected temperature delta mean: `1.3152 C`
- Keras-vs-selected temperature delta median: `0.9367 C`
- Keras-vs-selected center delta mean: `1.0725 px`
- Keras-vs-selected center delta median: `0.9663 px`
- Keras-vs-selected tip delta mean: `5.4413 px`
- Keras-vs-selected tip delta median: `4.2174 px`
- Guardrail disagreements: `15`

## Comparison To Current INT8 Baseline

- Keras accepted MAE: `3.5554 C`
- Keras acceptance rate: `0.8136`
- Keras worst accepted error: `17.4587 C`
- INT8 accepted MAE: `3.7062 C`
- INT8 acceptance rate: `0.7119`
- INT8 worst accepted error: `14.5879 C`
- INT8 Keras-vs-INT8 temp delta mean: `1.7350 C`
- INT8 Keras-vs-INT8 tip delta mean: `14.0198 px`
- INT8 guardrail disagreements: `12`

## Decision

- The selected variant improves drift materially over the old INT8 baseline, especially tip drift.
- The selected variant still misses the temperature-drift target of `<= 1.0 C`, so Cube.AI packaging is not allowed.
- Because no exported variant passed validation and the corrected decoder already restored Keras quality, QAT is the next justified fix.