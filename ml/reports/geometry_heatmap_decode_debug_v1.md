# Geometry Heatmap Decode Debug v1

## Run Summary

- Model: D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v1\model.keras
- Samples inspected: 30
- Split: test
- Decode methods: argmax and softargmax

## Heatmap Peak Statistics

| Map | Mean Max | Std Dev of Max | Mean Heatmap Std Dev |
| --- | ---: | ---: | ---: |
| Center | 0.119923 | 0.032641 | 0.005769 |
| Tip | 0.079997 | 0.032091 | 0.003324 |

## Decode Error

- Center argmax error: 158.146 px
- Tip argmax error: 128.131 px
- Center softargmax error: 45.919 px
- Tip softargmax error: 62.671 px
- Mean argmax-softargmax gap, center: 119.786 px
- Mean argmax-softargmax gap, tip: 96.709 px

## Collapse / Flatness Checks

- Mean center distance from crop center (softargmax): 43.999 px
- Mean tip distance from crop center (softargmax): 67.525 px
- Predictions collapsing to center: no
- Heatmaps nearly flat: yes

## Worst 5 Predictions

| Rank | Image | Abs Error (C) | Center Argmax Error (px) | Tip Argmax Error (px) | Peak Center | Peak Tip |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PXL_20260125_114534732.jpg | 34.947 | 172.384 | 170.494 | 0.1114 | 0.0723 |
| 2 | PXL_20260125_115239306.jpg | 30.877 | 159.174 | 175.054 | 0.0603 | 0.0271 |
| 3 | PXL_20260125_114520969.jpg | 29.143 | 154.114 | 154.010 | 0.1628 | 0.1213 |
| 4 | PXL_20260125_114606035.jpg | 28.669 | 161.739 | 162.733 | 0.0840 | 0.0609 |
| 5 | PXL_20260125_114911288.jpg | 28.604 | 171.081 | 116.752 | 0.0977 | 0.0609 |

## Interpretation

- If the heatmaps are diffuse, argmax and softargmax often diverge or stay near the crop center while peak values remain low.
- A healthy heatmap should show a sharp peak, with argmax and softargmax landing close to the same point and close to the true label.
