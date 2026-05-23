# Geometry Heatmap v2 Jitter Robustness

## Run Summary

- Model: `D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v2\model.keras`
- Selected stage: frozen

| level | count | calibrated_mae | worst_error | under_5c_% | under_10c_% | center_mae | tip_mae | center_peak_mean | tip_peak_mean | confidence_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 59 | 4.312 | 59.428 | 74.6 | 94.9 | 4.764 | 17.850 | 0.6624 | 0.6431 | 0.6721 |
| mild | 59 | 4.858 | 73.604 | 72.9 | 94.9 | 4.872 | 17.785 | 0.6622 | 0.6499 | 0.6742 |
| medium | 59 | 3.688 | 18.727 | 72.9 | 96.6 | 4.883 | 17.976 | 0.6557 | 0.6429 | 0.6721 |
| strong | 59 | 5.692 | 105.195 | 69.5 | 89.8 | 5.009 | 17.736 | 0.6451 | 0.6270 | 0.6673 |

## Interpretation

- Identity crops establish the clean baseline for board-style replay.
- Mild, medium, and strong jitter show whether the crop pipeline is robust or brittle.
- If the strong-jitter MAE remains close to the identity crop MAE, the heatmap model is tolerant to practical localizer noise.
