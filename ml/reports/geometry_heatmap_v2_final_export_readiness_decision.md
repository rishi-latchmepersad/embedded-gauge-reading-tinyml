# Geometry Heatmap v2 Final Export Readiness Decision

## Selected Guardrails

- `center_peak_min = 0.40`
- `tip_peak_min = 0.35`
- `confidence_min = 0.40`
- `max_heatmap_entropy = 1.00`
- `max_heatmap_spread_px = 30.00`
- `center_tip_distance_ratio_min = 0.35`
- `center_tip_distance_ratio_max = 1.40`
- `edge_margin_px = 4.00`
- `temperature_physical_range_margin_c = 2.00`

## Validation Summary

- accepted MAE: `3.260 C`
- acceptance rate: `0.660`
- worst accepted error: `11.334 C`
- accepted >20 C failures: `0`

## Test Summary

- accepted MAE: `3.555 C`
- acceptance rate: `0.814`
- worst accepted error: `17.459 C`
- accepted >20 C failures: `0`
- under 2 C: `45.8%`
- under 5 C: `77.1%`
- under 10 C: `91.7%`

## Gate Comparison

- Previous test accepted MAE: `2.517 C`
- Previous test acceptance rate: `0.644`
- Previous test worst accepted error: `9.060 C`
- The selected guardrails increase acceptance from `0.644` to `0.814`.
- The selected guardrails keep the worst accepted error below `20 C`.

## Decision

- Proceed to int8 export: **yes**
- Next phase: **Phase 7, int8 export and quantization replay**

## Why

- The selected thresholds satisfy the export gate on test.
- The board input contract is now fixed to the RGB bilinear training path.
- The known bad nearest/luma alternatives are explicitly excluded from firmware.

## Notes For Export

- Preserve the selected guardrails in deployment replay.
- Preserve the calibration coefficients from the robust linear candidate.
- Re-run quantization replay against the same board input contract before firmware handoff.

