# Geometry Heatmap v2 Corrected Decoder Quantization Readiness

## Decision

Cube.AI packaging is **not allowed**.

## Corrected Decoder Lock

- Selected decoder: `softargmax`
- Window size: `3`
- Selected on split: `val`
- Guardrails changed: `no`
- Corrected decode artifact: `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json`

## Variants Evaluated On Validation

- `variant_a_full_int8_identity`
- `variant_a_full_int8_identity_mild`
- `variant_a_full_int8_identity_mild_medium`
- `variant_a_full_int8_stratified`
- `variant_b_float_io_internal_int8`
- `variant_c_int8_input_float_output`
- `variant_d_dynamic_range`

## Validation Winner

- Selected validation winner: `variant_d_dynamic_range`
- Validation gate passed: `False`
- Reason it won: lowest Keras-vs-variant temperature drift among exported variants in this sweep

### Validation Metrics For The Winner

- Accepted MAE: `3.0062 C`
- Acceptance rate: `0.5957`
- Worst accepted error: `9.8088 C`
- Accepted >20 C failures: `0`
- Keras-vs-selected temperature delta mean: `1.1830 C`
- Keras-vs-selected temperature delta median: `0.9498 C`
- Keras-vs-selected tip delta mean: `5.0398 px`
- Keras-vs-selected tip delta median: `4.2854 px`
- Guardrail disagreements: `8`
- Tensor contract: float32 input, float32 outputs, no dequantization required

## Test Check For The Winner

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

## Current INT8 Reference On Test

- Accepted MAE: `3.7062 C`
- Acceptance rate: `0.7119`
- Worst accepted error: `14.5879 C`
- Accepted >20 C failures: `0`
- Keras-vs-INT8 temperature delta mean: `1.7350 C`
- Keras-vs-INT8 temperature delta median: `1.4424 C`
- Keras-vs-INT8 tip delta mean: `14.0198 px`
- Keras-vs-INT8 tip delta median: `9.9321 px`
- Guardrail disagreements: `12`

## What Improved

- The corrected decoder `softargmax w3` restored Keras board-replay quality.
- The exported variants did not reduce INT8 drift enough to meet the packaging target.
- The dynamic-range variant improved tip drift materially versus the old INT8 baseline, but its temperature drift still stayed above the `1.0 C` goal and its validation acceptance rate missed the gate.

## Why Cube.AI Is Still Blocked

- No exported variant passed the validation gate.
- The selected test winner improved drift, but the Keras-vs-selected temperature delta mean remained above `1.0 C`.
- Guardrail disagreements on the selected test winner were still above the old INT8 baseline.

## Required Next Fix

- QAT is now justified if we want to keep pushing INT8-quality drift lower after the export/decode options have been exhausted.
