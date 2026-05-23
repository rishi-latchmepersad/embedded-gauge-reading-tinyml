# Geometry Heatmap v3 Cube.AI Readiness Decision

## Decision

Cube.AI packaging is **not allowed** for geometry_heatmap_v3 at this stage.

## Why

The exported INT8 checkpoint did not pass the validation gate:

- accepted MAE: `3.3048 C` `<= 4.5 C` by itself, but
- acceptance rate: `0.5957` `(< 0.65)`
- worst accepted error: `11.8713 C` `(< 20 C)`
- accepted >20 C failures: `0`
- temperature drift mean: `1.9923 C` `(> 1.0 C)`
- tip drift mean: `14.7833 px` `(not materially improved)`
- guardrail disagreements: `7`

The corrected decoder remains:

- `softargmax w3`

The tensor contract is documented in:

- [geometry_heatmap_v3_tflite_replay.md](/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/reports/geometry_heatmap_v3_tflite_replay.md)

## Next Fix

Recommend exactly one next action:

- **A. increase heatmap resolution to 112x112**

## Phase 10

Do not proceed to STM32/Cube.AI packaging yet.
