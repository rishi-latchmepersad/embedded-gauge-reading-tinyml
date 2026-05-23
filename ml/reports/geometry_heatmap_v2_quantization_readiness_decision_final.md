# Geometry Heatmap v2 Quantization Readiness Decision

## Decision

Cube.AI packaging is **not allowed**.

## Corrected Decoder Selection

- Selected decoder: `softargmax`
- Selected window size: `3`
- Selected on split: `val`
- Guardrails re-swept: `False`
- Guardrail thresholds path: `ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json`

### Validation Metrics

- Accepted MAE: `3.2602 C`
- Acceptance rate: `0.6596`
- Worst accepted error: `11.3336 C`
- Accepted >20 C failures: `0`
- Under 2 C / 5 C / 10 C: `35.5` / `77.4` / `96.8`
- Center MAE: `5.57 px`
- Tip MAE: `21.12 px`
- Angle MAE: `14.61 deg`

### Test Metrics

- Accepted MAE: `3.5554 C`
- Acceptance rate: `0.8136`
- Worst accepted error: `17.4588 C`
- Accepted >20 C failures: `0`
- Under 2 C / 5 C / 10 C: `45.8` / `77.1` / `91.7`
- Center MAE: `5.55 px`
- Tip MAE: `22.93 px`
- Angle MAE: `14.46 deg`

## Current INT8 Check With The Selected Decoder

- Accepted MAE: `3.7062 C`
- Acceptance rate: `0.7119`
- Worst accepted error: `14.5879 C`
- Accepted >20 C failures: `0`
- Keras-vs-INT8 temperature delta mean: `1.7350 C`
- Keras-vs-INT8 temperature delta median: `1.4424 C`
- Keras-vs-INT8 center delta mean: `2.8548 px`
- Keras-vs-INT8 center delta median: `2.4118 px`
- Keras-vs-INT8 tip delta mean: `14.0198 px`
- Keras-vs-INT8 tip delta median: `9.9321 px`
- Guardrail disagreement count: `12`

## What The 5-Sample Probe Showed

- The TFLite input tensor is `int8` with scale `0.003921568859368563` and zero point `-128`.
- The output tensors are also `int8` with scale `0.00390625` and zero point `-128`.
- Raw output order is `tip_heatmap`, `center_heatmap`, `confidence`, and the evaluator reorders it to `center_heatmap`, `tip_heatmap`, `confidence` using `[1, 0, 2]`.
- Keras and INT8 raw heatmap statistics are broadly similar on the inspected samples.
- That makes a tensor-order or dequantization bug unlikely.
- The remaining problem is decode sensitivity: `softargmax` restores Keras board replay quality, but the INT8 drift is still above the preferred threshold.

## Why The Earlier Selected-Variant Result Is Invalid

- The earlier selected-variant replay used a broken coordinate projection path that overwrote the scaled 224-space coordinates.
- That made the old selected-variant drift numbers untrustworthy.
- The corrected evaluator now shows that the tensor contract is sane and the decoder is no longer the main blocker for Keras, but INT8 drift is still too high to approve packaging.

## Can We Proceed To Cube.AI?

No.

The selected decoder now passes the Keras board-replay gate, and current INT8 passes accuracy, but the Keras-vs-INT8 temperature drift is still above the `1.0 C` target.

## Single Next Fix

Rerun the selected TFLite variants under the corrected decoder if we want to reduce INT8 drift further before any packaging decision.
