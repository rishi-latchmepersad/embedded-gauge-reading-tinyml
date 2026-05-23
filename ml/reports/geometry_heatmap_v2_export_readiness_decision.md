# Geometry Heatmap v2 Export Readiness Decision

- Selected preprocessing mode: `python_training_rgb_bilinear`
- Test accepted MAE: 2.517 C
- Test acceptance rate: 0.644
- Test worst accepted error: 9.060 C
- Mild/medium jitter tails are represented in the board replay summary table.

## Decision

- Proceed to int8 export: no

## Why

- The board replay gate requires accepted MAE <= 4.5 C, acceptance rate >= 0.65, and worst accepted error < 20 C.
- Selected mode accepted MAE versus geometry_points_v1: -5.393 C.
- Selected mode accepted MAE versus oracle ceiling: 1.322 C.

## Recommendation

- If the gate passes, move to int8 export with the same preprocessing contract and guardrails.
- If the gate fails, align preprocessing first rather than retraining the model blind.
