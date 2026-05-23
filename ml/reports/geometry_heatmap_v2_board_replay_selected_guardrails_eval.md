# Geometry Heatmap v2 Board Replay Selected Guardrails

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

## Test Metrics

- accepted MAE: 3.555 C
- acceptance rate: 0.814
- worst accepted error: 17.459 C
- accepted >20 C failures: 0
- under 2 C: 45.8%
- under 5 C: 77.1%
- under 10 C: 91.7%

## Comparison To Previous Test Result

- Previous accepted MAE: 2.517 C
- Previous acceptance: 0.644
- Previous worst accepted error: 9.060 C
- Delta accepted MAE: 1.038 C
- Delta acceptance: 0.170
- Delta worst accepted error: 8.399 C

## Top Rejection Reasons

| reason | count |
| --- | ---: |
| tip_peak_too_low | 7 |
| center_tip_distance_ratio_implausible | 4 |
| predicted_angle_outside_valid_sweep | 3 |
| temperature_outside_physical_margin | 2 |
| tip_heatmap_too_spread_out | 1 |

- Export gate status: pass
