# Geometry Heatmap v3 Next Architecture Decision

## Decision

**A. Rerun v3 training with corrected canonical replay checkpoint selection.**

## Why

- The parity audit shows that the exported FP32 TFLite path is effectively identical to the loaded Keras checkpoint.
- The checkpoint reload parity is exact, so serialization is not the problem.
- The large gap is between the trainer-side selection metric and the canonical validation replay.
- Trainer-side validation used the legacy trainer preprocessing path, while the canonical validation replay uses the board-replay preprocessing path.
- The trainer-side selection metric was therefore optimistic and should not be trusted for export decisions.

## Evidence

- Trainer-style replay accepted MAE: `2.3358 C`
- Standalone canonical Keras accepted MAE: `3.6002 C`
- TFLite FP32 accepted MAE: `3.6002 C`
- TFLite INT8 accepted MAE: `3.3048 C`
- Trainer vs canonical temperature drift mean: `2.1192 C`
- Trainer vs canonical tip drift mean: `14.1735 px`
- Canonical Keras vs FP32 temperature drift mean: `0.0000 C`
- Reload parity passed: `yes`

## Implication

The next useful step is not a new architecture or a larger heatmap yet. We should rerun v3 training with:

- canonical validation replay only
- the corrected decoder lock (`softargmax w3`)
- the same board guardrails and calibration

That will give us a checkpoint selected against the same path used for export and deployment decisions.

