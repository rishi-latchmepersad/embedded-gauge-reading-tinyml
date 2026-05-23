# Geometry Heatmap v2 Board Guardrail Micro Sweep

- Selected preprocessing mode: `python_training_rgb_bilinear`
- Predictions source: `D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training\geometry_heatmap_v2_board_replay\board_replay_predictions.csv`

## Selection Rule

- Search the train/val grid only.
- Keep center peak, confidence, and edge margin fixed.
- Choose the least-relaxed candidate that passes validation.
- Tie-break by validation worst accepted error, then MAE, then acceptance rate.

## Selected Thresholds

| threshold | value |
| --- | ---: |
| center_peak_min | 0.4 |
| tip_peak_min | 0.35 |
| confidence_min | 0.4 |
| max_heatmap_entropy | 1.0 |
| max_heatmap_spread_px | 30.0 |
| center_tip_distance_ratio_min | 0.35 |
| center_tip_distance_ratio_max | 1.4 |
| edge_margin_px | 4.0 |
| temperature_physical_range_margin_c | 2.0 |
| minimum_celsius | -30.0 |
| maximum_celsius | 50.0 |
| clamp_temperature_to_physical_range | True |

## Validation Summary

- accepted MAE: 3.260 C
- acceptance rate: 0.660
- worst accepted error: 11.334 C
- accepted >20 C failures: 0

## Train Summary

- accepted MAE: 2.953 C
- acceptance rate: 0.797
- worst accepted error: 20.433 C
- accepted >20 C failures: 1

## Top Validation Rejection Reasons

| reason | count |
| --- | ---: |
| tip_peak_too_low | 12 |
| center_tip_distance_ratio_implausible | 5 |
| predicted_angle_outside_valid_sweep | 5 |
| temperature_outside_physical_margin | 5 |