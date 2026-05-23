# Geometry Heatmap v2 Guardrail Sweep

## Selected Thresholds

| threshold | value |
| --- | ---: |
| center_peak_min | 0.400 |
| tip_peak_min | 0.400 |
| confidence_min | 0.400 |
| max_heatmap_entropy | 1.000 |
| max_heatmap_spread_px | 25.000 |
| center_tip_distance_ratio_min | 0.400 |
| center_tip_distance_ratio_max | 1.400 |
| edge_margin_px | 4.000 |
| temperature_physical_range_margin_c | 2.000 |
| minimum_celsius | -30.000 |
| maximum_celsius | 50.000 |

## Selected Candidate Metrics

| metric | value |
| --- | ---: |
| accepted_fraction | 0.746 |
| rejected_fraction | 0.254 |
| clamped_fraction | 0.000 |
| accepted_mae_c | 3.092 |
| accepted_worst_error_c | 10.812 |
| accepted_percentage_under_5c | 78.409 |
| accepted_percentage_under_10c | 98.864 |
| false_rejection_rate_good_5c | 0.193 |
| all_gt20_rejected | 1.000 |
| identity_accepted_fraction | 0.763 |
| identity_accepted_mae_c | 3.157 |
| identity_accepted_worst_error_c | 10.812 |
| medium_accepted_fraction | 0.712 |
| medium_accepted_mae_c | 3.180 |
| medium_accepted_worst_error_c | 8.469 |
| strong_accepted_fraction | 0.712 |
| strong_accepted_mae_c | 3.253 |
| strong_accepted_worst_error_c | 10.812 |

## Selection Notes

- Valid candidates found: 648
- The selected guardrail set keeps identity and medium jitter above the requested acceptance floor while rejecting the catastrophic tail.
- The selected candidate's worst accepted error is 10.812 C.
- False rejection rate on <=5 C predictions: 0.193.

## Top 10 Candidates

| rank | accepted_mae | worst | identity_accept | medium_accept | strong_accept | false_reject_good | rejected_fraction | center_peak | tip_peak | conf | entropy | spread | ratio_min | ratio_max | edge | temp_margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 4.0 | 2.0 |
| 2 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 8.0 | 2.0 |
| 3 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 12.0 | 2.0 |
| 4 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 4.0 | 5.0 |
| 5 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 8.0 | 5.0 |
| 6 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 12.0 | 5.0 |
| 7 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 4.0 | 10.0 |
| 8 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 8.0 | 10.0 |
| 9 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.40 | 12.0 | 10.0 |
| 10 | 3.092 | 10.812 | 0.763 | 0.712 | 0.712 | 0.193 | 0.254 | 0.40 | 0.40 | 0.40 | 1.00 | 25.0 | 0.40 | 1.50 | 4.0 | 2.0 |