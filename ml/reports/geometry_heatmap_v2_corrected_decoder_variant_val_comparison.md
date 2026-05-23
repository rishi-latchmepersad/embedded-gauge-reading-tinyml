# Geometry Heatmap v2 Corrected Decoder Variant Validation Comparison

## Decode Lock

- Selected decode artifact: `/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json`
- Locked decode method: `softargmax`
- Locked window size: `3`

## Validation Results

| variant | accepted MAE | acceptance rate | worst accepted | >20 C fails | temp delta mean | temp delta median | tip delta mean | guardrail disagreements | input dtype | output dtypes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| variant_d_dynamic_range | 3.0062 | 0.5957 | 9.8088 | 0 | 1.1830 | 0.9498 | 5.0398 | 8 | float32 | float32|float32|float32 |
| variant_a_full_int8_identity | 3.2415 | 0.6383 | 11.4549 | 0 | 7.5826 | 1.8615 | 14.3834 | 8 | int8 | int8|int8|int8 |
| variant_a_full_int8_stratified | 3.2415 | 0.6383 | 11.4549 | 0 | 7.5826 | 1.8615 | 14.3834 | 8 | int8 | int8|int8|int8 |
| variant_b_float_io_internal_int8 | 3.2415 | 0.6383 | 11.4549 | 0 | 7.5826 | 1.8615 | 14.3834 | 8 | float32 | float32|float32|float32 |
| variant_c_int8_input_float_output | 3.2415 | 0.6383 | 11.4549 | 0 | 7.5826 | 1.8615 | 14.3834 | 8 | int8 | float32|float32|float32 |
| variant_a_full_int8_identity_mild_medium | 3.1398 | 0.5957 | 12.0301 | 0 | 7.6137 | 1.8684 | 14.4501 | 9 | int8 | int8|int8|int8 |
| variant_a_full_int8_identity_mild | 2.9570 | 0.5957 | 10.8307 | 0 | 7.9397 | 1.8725 | 15.1204 | 10 | int8 | int8|int8|int8 |

## Selection

- Selected variant: `variant_d_dynamic_range`
- Validation accepted MAE: `3.0062 C`
- Validation acceptance rate: `0.5957`
- Validation worst accepted error: `9.8088 C`
- Validation Keras-vs-selected temperature delta mean: `1.1830 C`
- Validation Keras-vs-selected tip delta mean: `5.0398 px`
- Validation guardrail disagreements: `8`
- Validation gate passed: `False`

## Current INT8 Baseline (Validation Reference)

- Current INT8 accepted MAE: `3.2926 C`
- Current INT8 acceptance rate: `0.6383`
- Current INT8 worst accepted error: `11.5008 C`
- Current INT8 accepted >20 C failures: `0`

## Selection Notes

- The variant gate requires accepted MAE <= 4.5 C, acceptance rate >= 0.65, worst accepted error < 20 C, zero >20 C failures, temperature drift <= 1.0 C, and a materially better tip drift than the old INT8 baseline.
- Selected contract input/output dtypes: `float32` / `float32|float32|float32`

- Validation output CSV: `/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/corrected_decoder_variant_val_summary.csv`