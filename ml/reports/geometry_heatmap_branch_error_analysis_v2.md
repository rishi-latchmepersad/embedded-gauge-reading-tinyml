# Geometry Heatmap Branch Error Analysis v2

## Run Summary

- Model: D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_tiny_overfit_v1\model.keras
- Samples: 8
- Heatmap size: 56x56
- Same 8 clean train examples used by the tiny-overfit gate

## Target Heatmap Statistics

| Branch | Mean Max | Mean | Mean Std Dev |
| --- | ---: | ---: | ---: |
| Center | 0.986537 | 0.012522 | 0.078130 |
| Tip | 0.984013 | 0.012522 | 0.078130 |

## Predicted Heatmap Statistics

| Branch | Mean Peak | Peak Std Dev | Mean Softargmax Error (px) | Mean Argmax Error (px) |
| --- | ---: | ---: | ---: | ---: |
| Center | 0.997439 | 0.001803 | 5.855 | 48.345 |
| Tip | 0.999988 | 0.000012 | 2.188 | 5.224 |

## Center Ambiguity Check

- Mean center distance from crop center: 12.754 px
- Mean tip distance from crop center: 75.556 px
- Center distance range: 6.351 px to 21.511 px
- Tip distance range: 70.509 px to 81.583 px
- Predicted center distance from crop center (softargmax): 9.972 px
- Predicted tip distance from crop center (softargmax): 76.605 px

## Interpretation

- The center target is much closer to the crop hub than the tip target, so even small center mistakes can disturb the angle a lot more than a similar tip mistake.
- On these 8 samples, the center labels live in the hub zone while the tip labels sit far out on the needle, which makes the center branch the more ambiguous branch visually.
- If the v2 tiny-overfit gate still struggles after stronger center weighting, the cleanest simplification may be to keep the tip heatmap and substitute a prior or fixed center for the first board version.
