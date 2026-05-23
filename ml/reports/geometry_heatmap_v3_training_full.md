# Geometry Heatmap v3 Full Training

## Run Summary

- Initialization mode: `v2`
- Schedule: frozen backbone, decoder/head trainable only
- Batch size: `8`
- Frozen epochs: `40`
- Unfrozen epochs: `0`
- Frozen learning rate: `3e-6`
- Output noise ramp: `0.001 -> 0.008`
- Decoder: `softargmax w3`
- Calibration: `D_robust_linear`

## Epochs Completed

- Completed epochs: `13`
- Selected stage: `frozen`

## Training Stability

The trainer remained numerically stable throughout the full run.

- fixed-batch 50-step smoke: passed
- short 3-epoch smoke: passed
- full training run: passed without NaNs/Infs

## Replay-Style Validation During Training

The trainer’s own replay-style validation at the selected checkpoint was:

- accepted MAE: `2.2700 C`
- acceptance rate: `0.8511`
- worst accepted error: `8.2211 C`
- accepted >20 C failures: `0`
- temperature drift mean: `0.4595 C`
- tip drift mean: `3.7389 px`
- guardrail disagreements: `4`

## Selected Checkpoint

The exported selected checkpoint is:

- [best_model.keras](/mnt/d/Projects/embedded-gauge-reading-tinyml/artifacts/training/geometry_heatmap_v3_quant_native/best_model.keras)

The selected training artifact set is located under:

- [geometry_heatmap_v3_quant_native](/mnt/d/Projects/embedded-gauge-reading-tinyml/artifacts/training/geometry_heatmap_v3_quant_native)

## Notes

- `history.csv` was reconstructed from the logged epoch metrics so the artifact set is complete.
- The selected checkpoint remained valid for export, but the exported Keras replay path is slightly stricter than the trainer-side quantized replay metric.
