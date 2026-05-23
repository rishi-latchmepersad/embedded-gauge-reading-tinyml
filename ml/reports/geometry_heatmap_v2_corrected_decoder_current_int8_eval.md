# Geometry Heatmap v2 Decode Method Comparison, Corrected

## Selection

- Selected decoder: `softargmax`
- Selected window size: `3`
- Selected on split: `val`
- Guardrails re-swept? `False`
- Selected guardrail thresholds path: `/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json`

## Selected Thresholds

- center_peak_min: `0.4`
- tip_peak_min: `0.35`
- confidence_min: `0.4`
- max_heatmap_entropy: `1.0`
- max_heatmap_spread_px: `30.0`
- center_tip_distance_ratio_min: `0.35`
- center_tip_distance_ratio_max: `1.4`
- edge_margin_px: `4.0`
- temperature_physical_margin_c: `2.0`

## Summary

| decode method | window | split | accepted MAE | acceptance rate | worst accepted | accepted >20 C | under 2 C | under 5 C | under 10 C | center MAE | tip MAE | angle MAE | top rejection reasons |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| argmax | 3 | test | 4.8868 | 0.3898 | 16.5964 | 0 | 34.8 | 60.9 | 82.6 | 67.93 | 59.00 | 39.67 | predicted_point_near_edge:33;center_tip_distance_ratio_implausible:28;tip_peak_too_low:7;predicted_angle_outside_valid_sweep:6;temperature_outside_physical_margin:6 |
| local_window_softargmax_w3 | 3 | test | 4.8867 | 0.3898 | 16.6011 | 0 | 34.8 | 60.9 | 82.6 | 67.21 | 57.27 | 39.63 | predicted_point_near_edge:33;center_tip_distance_ratio_implausible:27;tip_peak_too_low:7;predicted_angle_outside_valid_sweep:6;temperature_outside_physical_margin:6 |
| local_window_softargmax_w5 | 5 | test | 9.3622 | 0.5085 | 32.1407 | 5 | 26.7 | 46.7 | 63.3 | 66.55 | 55.94 | 39.50 | center_tip_distance_ratio_implausible:23;tip_peak_too_low:7;predicted_angle_outside_valid_sweep:6;temperature_outside_physical_margin:6;predicted_point_near_edge:2 |
| peak_weighted_centroid_w3 | 3 | test | 4.8867 | 0.3898 | 16.6011 | 0 | 34.8 | 60.9 | 82.6 | 67.21 | 57.27 | 39.63 | predicted_point_near_edge:33;center_tip_distance_ratio_implausible:27;tip_peak_too_low:7;predicted_angle_outside_valid_sweep:6;temperature_outside_physical_margin:6 |
| peak_weighted_centroid_w5 | 5 | test | 9.3622 | 0.5085 | 32.1407 | 5 | 26.7 | 46.7 | 63.3 | 66.55 | 55.94 | 39.50 | center_tip_distance_ratio_implausible:23;tip_peak_too_low:7;predicted_angle_outside_valid_sweep:6;temperature_outside_physical_margin:6;predicted_point_near_edge:2 |
| softargmax | 3 | test | 3.5554 | 0.8136 | 17.4588 | 0 | 45.8 | 77.1 | 91.7 | 5.55 | 22.93 | 14.46 | tip_peak_too_low:7;center_tip_distance_ratio_implausible:4;predicted_angle_outside_valid_sweep:3;temperature_outside_physical_margin:2;tip_heatmap_too_spread_out:1 |
| argmax | 3 | train | 7.3589 | 0.3480 | 55.1212 | 10 | 31.6 | 62.0 | 81.0 | 70.99 | 50.15 | 43.18 | predicted_point_near_edge:116;center_tip_distance_ratio_implausible:101;tip_peak_too_low:32;temperature_outside_physical_margin:21;predicted_angle_outside_valid_sweep:19 |
| local_window_softargmax_w3 | 3 | train | 8.2724 | 0.3612 | 55.1064 | 13 | 30.5 | 59.8 | 78.0 | 70.24 | 48.79 | 43.12 | predicted_point_near_edge:112;center_tip_distance_ratio_implausible:98;tip_peak_too_low:32;temperature_outside_physical_margin:21;predicted_angle_outside_valid_sweep:20 |
| local_window_softargmax_w5 | 5 | train | 10.3192 | 0.4890 | 55.0869 | 24 | 25.2 | 52.3 | 69.4 | 69.54 | 47.76 | 42.72 | center_tip_distance_ratio_implausible:96;tip_peak_too_low:32;temperature_outside_physical_margin:22;predicted_angle_outside_valid_sweep:20;predicted_point_near_edge:7 |
| peak_weighted_centroid_w3 | 3 | train | 8.2724 | 0.3612 | 55.1064 | 13 | 30.5 | 59.8 | 78.0 | 70.24 | 48.79 | 43.12 | predicted_point_near_edge:112;center_tip_distance_ratio_implausible:98;tip_peak_too_low:32;temperature_outside_physical_margin:21;predicted_angle_outside_valid_sweep:20 |
| peak_weighted_centroid_w5 | 5 | train | 10.3192 | 0.4890 | 55.0869 | 24 | 25.2 | 52.3 | 69.4 | 69.54 | 47.76 | 42.72 | center_tip_distance_ratio_implausible:96;tip_peak_too_low:32;temperature_outside_physical_margin:22;predicted_angle_outside_valid_sweep:20;predicted_point_near_edge:7 |
| softargmax | 3 | train | 2.9534 | 0.7974 | 20.4335 | 1 | 46.4 | 82.9 | 95.6 | 4.82 | 17.99 | 11.77 | tip_peak_too_low:32;center_tip_distance_ratio_implausible:14;predicted_angle_outside_valid_sweep:13;temperature_outside_physical_margin:12;tip_heatmap_too_spread_out:5 |
| argmax | 3 | val | 7.3813 | 0.3617 | 36.3963 | 2 | 17.6 | 70.6 | 82.4 | 70.74 | 65.64 | 54.09 | predicted_point_near_edge:27;center_tip_distance_ratio_implausible:24;tip_peak_too_low:12;temperature_outside_physical_margin:7;predicted_angle_outside_valid_sweep:6 |
| local_window_softargmax_w3 | 3 | val | 7.3808 | 0.3617 | 36.3882 | 2 | 17.6 | 70.6 | 82.4 | 69.92 | 63.75 | 54.02 | predicted_point_near_edge:27;center_tip_distance_ratio_implausible:22;tip_peak_too_low:12;temperature_outside_physical_margin:7;predicted_angle_outside_valid_sweep:6 |
| local_window_softargmax_w5 | 5 | val | 10.6340 | 0.4468 | 39.2889 | 4 | 14.3 | 57.1 | 66.7 | 69.13 | 62.33 | 53.71 | center_tip_distance_ratio_implausible:22;tip_peak_too_low:12;temperature_outside_physical_margin:7;predicted_angle_outside_valid_sweep:6;predicted_point_near_edge:1 |
| peak_weighted_centroid_w3 | 3 | val | 7.3808 | 0.3617 | 36.3882 | 2 | 17.6 | 70.6 | 82.4 | 69.92 | 63.75 | 54.02 | predicted_point_near_edge:27;center_tip_distance_ratio_implausible:22;tip_peak_too_low:12;temperature_outside_physical_margin:7;predicted_angle_outside_valid_sweep:6 |
| peak_weighted_centroid_w5 | 5 | val | 10.6340 | 0.4468 | 39.2889 | 4 | 14.3 | 57.1 | 66.7 | 69.13 | 62.33 | 53.71 | center_tip_distance_ratio_implausible:22;tip_peak_too_low:12;temperature_outside_physical_margin:7;predicted_angle_outside_valid_sweep:6;predicted_point_near_edge:1 |
| softargmax | 3 | val | 3.2602 | 0.6596 | 11.3336 | 0 | 35.5 | 77.4 | 96.8 | 5.57 | 21.12 | 14.61 | tip_peak_too_low:12;center_tip_distance_ratio_implausible:5;predicted_angle_outside_valid_sweep:5;temperature_outside_physical_margin:5 |

## Test Check

- Test accepted MAE: `3.5554 C`
- Test acceptance rate: `0.8136`
- Test worst accepted error: `17.4588 C`
- Test accepted >20 C failures: `0`
- Test under 2 C / 5 C / 10 C: `45.8` / `77.1` / `91.7`
- Test center MAE: `5.55 px`
- Test tip MAE: `22.93 px`
- Test angle MAE: `14.46 deg`

## Current INT8 Check

- INT8 accepted MAE: `3.7062 C`
- INT8 acceptance rate: `0.7119`
- INT8 worst accepted error: `14.5879 C`
- INT8 accepted >20 C failures: `0`
- Keras-vs-INT8 temperature delta mean: `1.7350 C`
- Keras-vs-INT8 temperature delta median: `1.4424 C`
- Keras-vs-INT8 center delta mean: `2.8548 px`
- Keras-vs-INT8 center delta median: `2.4118 px`
- Keras-vs-INT8 tip delta mean: `14.0198 px`
- Keras-vs-INT8 tip delta median: `9.9321 px`
- Guardrail disagreement count: `12`