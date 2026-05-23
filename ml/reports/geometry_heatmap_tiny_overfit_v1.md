# Geometry Heatmap Tiny Overfit v1

## Run Summary

- Samples: 8
- Sigma: 2.50 pixels
- Epochs trained: 17
- Final loss: 1.342412
- Final center softargmax MAE: 0.688193
- Final tip softargmax MAE: 0.423232
- Final center peak: 0.994034
- Final tip peak: 0.999984

## Holdout-On-Same-8 Metrics

- Center MAE: 5.855 px
- Tip MAE: 2.188 px
- Temperature MAE: 3.844 C
- Angle MAE: 1.806 deg
- Under 2 C: 0.0%
- Under 5 C: 75.0%
- Under 10 C: 100.0%
- Mean center peak: 0.9974
- Mean tip peak: 1.0000

## Success Criteria

- Center MAE < 3 px: no
- Tip MAE < 5 px: yes
- Temperature MAE < 3 C: no
- Mean center peak > 0.5: yes
- Mean tip peak > 0.5: yes
- Overall result: FAIL

## Worst 3 Samples

| Image | Abs Error (C) | Center Err (px) | Tip Err (px) | Peak Center | Peak Tip |
| --- | ---: | ---: | ---: | ---: | ---: |
| PXL_20260125_114538442.jpg | 5.559 | 9.368 | 2.364 | 0.9998 | 1.0000 |
| PXL_20260125_114524322.jpg | 5.217 | 5.846 | 2.335 | 0.9994 | 1.0000 |
| PXL_20260125_114544203.jpg | 4.369 | 2.669 | 1.412 | 0.9964 | 1.0000 |

## Interpretation

- If this pass/fail gate fails, the heatmap loss is still not giving the model enough signal to fit even eight examples.
- If it passes, the next step is a controlled v2 training run with the same weighted heatmap objective and a small auxiliary coordinate penalty.
