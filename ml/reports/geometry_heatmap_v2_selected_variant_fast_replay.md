# Geometry Heatmap v2 Selected Variant Fast Replay

## Selection

- Selected decode method: `peak_weighted_centroid_w5`
- Decoded method: `peak_weighted_centroid`
- Window size: `5`
- Selected variant: `variant_b_float_io_internal_int8`
- Split: `test`
- Sample count: `59`
- Max samples: `none`

## Contract

- Input: dtype `float32`, shape `[1, 224, 224, 3]`, quantized `False`
- Input quantization: scale `0.0`, zero_point `0`
- Requires output dequantization? `False`
- Outputs:
  - `0`: name `StatefulPartitionedCall_1:1`, dtype `float32`, shape `[1, 56, 56, 1]`, quantized `False`
  - `1`: name `StatefulPartitionedCall_1:0`, dtype `float32`, shape `[1, 56, 56, 1]`, quantized `False`
  - `2`: name `StatefulPartitionedCall_1:2`, dtype `float32`, shape `[1, 1]`, quantized `False`

## Metrics

- Candidate accepted MAE: `8.3357 C`
- Candidate acceptance rate: `0.0508`
- Candidate worst accepted error: `17.3718 C`
- Candidate accepted >20 C failures: `0`
- Keras-vs-candidate temperature delta mean: `20.9925 C`
- Keras-vs-candidate temperature delta median: `13.6565 C`
- Keras-vs-candidate center delta mean: `114.8088 px`
- Keras-vs-candidate center delta median: `115.8521 px`
- Keras-vs-candidate tip delta mean: `130.5696 px`
- Keras-vs-candidate tip delta median: `134.5196 px`
- Guardrail disagreements: `42`

## Baseline Context

- Cached Keras acceptance rate: `0.4746`
- Cached current INT8 acceptance rate: `0.7119`

## Notes

This fast path uses cached replay rows for Keras and the current INT8 baseline and only executes fresh inference for the selected candidate variant.