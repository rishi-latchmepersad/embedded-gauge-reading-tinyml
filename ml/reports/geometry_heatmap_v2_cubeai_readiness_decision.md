# Geometry Heatmap v2 Cube.AI Readiness Decision

## Decision

Do **not** proceed to Cube.AI packaging yet.

## Test Split Replay Metrics

### Keras FP32

- accepted MAE: `3.555 C`
- acceptance rate: `0.814`
- worst accepted error: `17.459 C`
- accepted `>20 C` failures: `0`

### TFLite FP32

- accepted MAE: `3.555 C`
- acceptance rate: `0.814`
- worst accepted error: `17.459 C`
- accepted `>20 C` failures: `0`

### TFLite INT8

- accepted MAE: `3.706 C`
- acceptance rate: `0.712`
- worst accepted error: `14.588 C`
- accepted `>20 C` failures: `0`

## Keras vs INT8 Drift

- accepted temperature delta mean: `1.735 C`
- accepted temperature delta median: `1.442 C`
- center point delta mean: `2.868 px`
- tip point delta mean: `14.083 px`
- rejection-status disagreement count: `12`

## Verdict Against The Cube.AI Gate

The INT8 model satisfies the replay accuracy gate on its own:

- accepted MAE <= 4.5 C: yes
- acceptance rate >= 0.65: yes
- worst accepted error < 20 C: yes
- accepted `>20 C` failures = 0: yes

However, the Keras-to-INT8 temperature drift still exceeds the allowed `1.0 C` mean threshold. That means the quantized export is not yet close enough to the floating-point reference to be considered stable for board packaging.

## Smallest Recommended Fix

Use one of the following next steps before retrying Cube.AI packaging:

1. Keep the same architecture but export with less aggressive output quantization, or float32 outputs if Cube.AI can support the integration path.
2. If INT8 outputs are required, retrain with quantization-aware training and a slightly richer representative dataset so the heatmap peaks survive quantization with less drift.
3. If the board path must stay INT8 end-to-end, add more representative train samples from the full temperature range and confirm whether the drift drops below `1.0 C` on the accepted subset.

## Contract Status

The tensor contract is clear and stable:

- input: RGB, bilinear resize, 224x224, float32 normalization to `0..1`
- output tensors: center heatmap, tip heatmap, confidence
- semantic output order: center heatmap, tip heatmap, confidence
- raw TFLite output order: tip heatmap, center heatmap, confidence
- quantized outputs require dequantization with `(tensor - zero_point) * scale`

## Recommendation

Hold Phase 7 packaging. The model is accurate enough, but the INT8 drift still needs one more tightening step before Cube.AI export.
