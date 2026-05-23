# Geometry Heatmap Target Debug v1

## Run Summary

- Samples inspected: 30
- Heatmap size: 56x56
- Sigma: 2.50 pixels
- Selection: balanced clean rows across train/val/test

## Heatmap Statistics

| Map | Min | Max | Mean | Mean Std Dev |
| --- | ---: | ---: | ---: | ---: |
| Center | 0.000000 | 0.999876 | 0.012522 | 0.078130 |
| Tip | 0.000000 | 0.999468 | 0.012513 | 0.078130 |

## Argmax Alignment

- Center mean argmax error: 0.333 heatmap px
- Tip mean argmax error: 0.412 heatmap px
- Center mean softargmax error: 0.000 heatmap px
- Tip mean softargmax error: 0.005 heatmap px
- Center swapped-order mean error: 3.914 heatmap px
- Tip swapped-order mean error: 31.531 heatmap px

## Coordinate Conversion Checks

- Center normalized-to-224 conversion mean abs error: 0.000000 px
- Tip normalized-to-224 conversion mean abs error: 0.000000 px
- x/y ordering looks correct for center: yes
- x/y ordering looks correct for tip: yes
- Heatmap argmax is within 1-2 px of the label: yes

## Interpretation

- The target generator is behaving if argmax stays near the expected point and the swapped-order error is much worse.
- The crop-space conversion should be nearly zero because the crop metadata and the target generation use the same normalized coordinates.
