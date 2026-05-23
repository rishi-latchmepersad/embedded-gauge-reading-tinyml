# Geometry Architecture Decision v1

## Answers

- Oracle geometry temperature MAE using current mapping: 1.718 C
- Oracle geometry temperature MAE using calibrated mapping: 1.195 C
- Tiny-overfit v2 current mapping MAE: 3.732 C
- Tiny-overfit v2 calibrated mapping MAE: 0.840 C
- Tiny-overfit v2 passes under calibrated mapping: yes
- Center prediction is the blocker: no
- Center-prior still looks better after calibration: yes

## Calibration Summary

- Train MAE: 1.208 C
- Val MAE: 1.140 C
- Test MAE: 1.190 C

## Tiny Overfit Geometry

- Center MAE: 0.013 px
- Tip MAE: 0.540 px
- Angle MAE: 0.211 deg

## Center-Prior Ablation Under Calibration

| Mode | Center Source | Temp MAE (C) |
| --- | --- | ---: |
| A | model-predicted center | 0.840 |
| B | true / manifest center | 0.840 |
| C | average train-set center | 2.072 |
| D | loose-crop geometric center | 2.026 |

## Recommendation

Phase 5 should be: **A. center+tip heatmap full training**.

Why:

- The oracle geometry ceiling is already only about 1-2 C, so the remaining gap is calibration, not a geometry failure.
- The calibrated tiny-overfit v2 run is comfortably below the 3 C gate, which means the heatmap setup is viable.
- The center-prior ablation does not beat the full center+tip prediction after calibration, so dropping the center branch is not supported by this evidence.
