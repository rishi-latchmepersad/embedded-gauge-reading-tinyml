# Geometry Heatmap v2 Corrected Decoder INT8 Baseline

## Run Summary

- Replay script: `scripts/eval_geometry_heatmap_v2_tflite_replay.py`
- Split: `test`
- Sample count: `59`
- Output suffix used for the rerun: `corrected_decoder_test`
- Selected decode: `peak_weighted_centroid_w5`
- Selected decode method parsed from lock: `peak_weighted_centroid`
- Selected decode window size: `5`

## Tensor Contract

- Input dtype: `int8`
- Input quantization: scale `0.003921568859368563`, zero point `-128`
- Raw output order from the TFLite interpreter:
  - `StatefulPartitionedCall_1:1`
  - `StatefulPartitionedCall_1:0`
  - `StatefulPartitionedCall_1:2`
- Raw output dtypes: all `int8`
- Raw output quantization: scale `0.00390625`, zero point `-128` for each output
- Semantic reorder mapping: `[1, 0, 2]`
- Semantic output order after reorder: `center_heatmap`, `tip_heatmap`, `confidence`

## Full Test Metrics

### Keras

- Accepted MAE: `9.3622 C`
- Acceptance rate: `0.5085`
- Worst accepted error: `32.1407 C`
- Accepted >20 C failures: `5`

### INT8

- Accepted MAE: `10.5313 C`
- Acceptance rate: `0.3051`
- Worst accepted error: `38.6716 C`
- Accepted >20 C failures: `4`

### Keras vs INT8 Drift

- Temperature delta mean: `2.9181 C`
- Temperature delta median: `0.7480 C`
- Center delta mean: `49.1282 px`
- Center delta median: `8.9909 px`
- Tip delta mean: `53.5685 px`
- Tip delta median: `12.7802 px`
- Guardrail disagreement count: `14`

## Five-Sample Raw-Output Probe

The five-sample probe was run directly against the Keras model and the current INT8 TFLite model with the corrected decoder lock.

Observed tensor behavior:

- The raw INT8 tensors are quantized as expected and dequantize cleanly.
- The Keras and INT8 heatmaps have similar raw statistics on the inspected samples.
- The output contract ordering is consistent with the documented `[1, 0, 2]` reorder.
- The remaining quality problem is therefore not an obvious tensor-order or dequantization bug.

Representative probe observations:

- Sample 0: Keras and INT8 both produce plausible heatmap peaks, and the decoded temperatures are close to each other, but both are still off the physical truth.
- Samples 1 to 4: the decoded geometry is often far from the true center/tip even though the raw heatmap statistics between Keras and INT8 remain broadly similar.
- This points to the selected decoder, `peak_weighted_centroid_w5`, as the more likely source of the poor absolute accuracy rather than a tensor contract mismatch.

## Interpretation

This corrected baseline is materially different from the earlier invalid selected-variant result.

- The old selected-variant result was invalid because the geometry projection bug was overwriting the scaled 224-space coordinates.
- The corrected baseline shows that the tensor handling is now sane, but the chosen decoder is still too weak for the current heatmaps.
- In other words, the replay evaluator is now much more credible, but the current selected decoder is not a good deployment decoder for this model.

## Readiness Decision

- Cube.AI packaging is **not allowed**.
- The current INT8 baseline does not meet the accuracy gate, and the Keras-vs-INT8 drift is still too high for a packaging decision.
- The next fix should be to revisit decode selection before considering selected TFLite variants again.
