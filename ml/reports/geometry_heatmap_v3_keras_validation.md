# Geometry Heatmap v3 Keras Validation

This report covers the selected `best_model.keras` checkpoint exported from the full v3 quantization-native training run.

## Trainer-Side Selection Replay

The training callback selected a checkpoint with replay-style validation metrics of:

- accepted MAE: `2.2700 C`
- acceptance rate: `0.8511`
- worst accepted error: `8.2211 C`
- accepted >20 C failures: `0`
- temperature drift mean: `0.4595 C`
- tip drift mean: `3.7389 px`
- guardrail disagreements: `4`

## Exported Checkpoint Replay

The exported checkpoint replay on the validation split measured:

- accepted MAE: `3.6002 C`
- acceptance rate: `0.7021`
- worst accepted error: `13.5297 C`
- accepted >20 C failures: `0`
- under 2/5/10 C: `21.28% / 55.32% / 68.09%`
- center MAE px: `5.5034`
- tip MAE px: `20.4946`
- angle MAE deg: `36.6646`

## Interpretation

- The checkpoint is usable as a Keras baseline, but the raw replay path is noticeably weaker than the trainer-side quantized replay metric.
- That discrepancy is one reason the exported INT8 checkpoint was not allowed to proceed to test.
