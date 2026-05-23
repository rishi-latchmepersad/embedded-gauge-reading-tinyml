# Geometry Heatmap v3 Canonical Keras Validation

- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Validation split: val

## Selected Canonical Checkpoint

- Accepted MAE: 3.5472 C
- Acceptance rate: 0.7021
- Worst accepted error: 12.2516 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 21.28% / 55.32% / 68.09%
- Center MAE px: 5.5131
- Tip MAE px: 20.4532
- Angle MAE deg: 36.4667
- Guardrail disagreements: 14
- Rejection reasons: tip_peak_too_low:12;predicted_angle_outside_valid_sweep:5;temperature_outside_physical_margin:5;center_tip_distance_ratio_implausible:3

## Comparison To Previous Noncanonical-Selected Checkpoint

- Previous canonical val accepted MAE: 3.6002 C
- Previous canonical val acceptance rate: 0.7021
- Previous canonical val worst accepted error: 13.5297 C
- Previous canonical val tip MAE px: 20.4946

## Decision

- The canonical validation checkpoint is better than the previous noncanonical-selected checkpoint on accepted MAE and worst accepted error.
- Export to TFLite is allowed.
- Canonical INT8 validation still needs to clear the drift/acceptance gate before Cube.AI can be considered.

