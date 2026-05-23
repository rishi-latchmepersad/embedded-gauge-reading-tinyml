# Geometry Heatmap Tiny Overfit v2

## Run Summary

- Samples: 8
- Sigma: 5.00 pixels at 56x56
- Center heatmap loss weight: 3.0
- Tip heatmap loss weight: 1.0
- Confidence loss weight: 0.0
- Final mode: unfrozen
- Frozen backbone passed: no
- Unfreezing needed: yes

## Final Mode Metrics

- Center MAE: 0.013 px
- Tip MAE: 0.540 px
- Angle MAE: 0.211 deg
- Temperature MAE: 3.732 C
- Mean center peak: 0.9123
- Mean tip peak: 0.8706
- Under 2 C: 0.0%
- Under 5 C: 100.0%
- Under 10 C: 100.0%
- Argmax center MAE: 2.116 px
- Argmax tip MAE: 2.894 px
- Argmax temperature MAE: 3.733 C
- Argmax angle MAE: 1.433 deg

## Frozen Backbone Check

- Frozen pass: no
- Frozen epoch: n/a
- Frozen center MAE: 0.076 px
- Frozen tip MAE: 0.565 px
- Frozen temperature MAE: 3.718 C
- Frozen mean center peak: 0.9102
- Frozen mean tip peak: 0.8612

## Unfrozen Fine-Tune Check

- Unfrozen pass: no
- Unfrozen epoch: n/a
- Unfrozen center MAE: 0.013 px
- Unfrozen tip MAE: 0.540 px
- Unfrozen temperature MAE: 3.732 C
- Unfrozen mean center peak: 0.9123
- Unfrozen mean tip peak: 0.8706

## Center-Prior Ablation

| Mode | Center Source | Tip Source | Temp MAE (C) |
| --- | --- | --- | ---: |
| A | model-predicted center | model-predicted tip | 3.732 |
| B | true / manifest center | model-predicted tip | 3.732 |
| C | average train-set center | model-predicted tip | 3.608 |
| D | loose-crop geometric center | model-predicted tip | 2.995 |

- Average train-set center: (111.89, 115.34)
- Loose-crop geometric center: (112.00, 112.00)

## Worst 3 Predictions

| Image | Abs Error (C) | Center Err (px) | Tip Err (px) | Peak Center | Peak Tip |
| --- | ---: | ---: | ---: | ---: | ---: |
| PXL_20260125_114538442.jpg | 4.571 | 0.028 | 1.263 | 0.9312 | 0.8719 |
| PXL_20260125_114532964.jpg | 4.476 | 0.005 | 0.373 | 0.9039 | 0.8637 |
| PXL_20260125_114524322.jpg | 4.218 | 0.013 | 1.466 | 0.9221 | 0.8708 |

## Interpretation

- Tiny-overfit gate passed: no
- Backbone unfreezing used: yes
- Recommendation: The center-prior ablation is better than the full center+tip prediction, so a simpler tip-only or fixed-center architecture is worth testing before scaling up.
