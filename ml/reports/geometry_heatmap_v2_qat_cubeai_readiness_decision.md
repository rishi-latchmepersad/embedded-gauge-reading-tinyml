# Geometry Heatmap v2 QAT Cube.AI Readiness Decision

## Decision

- Cube.AI packaging is **not allowed**.

## Why

- Corrected decoder remains `softargmax` with `window_size=3`.
- Tensor contract is documented by the QAT export:
  - input: float32 for the QAT float export, int8 for the QAT int8 export
  - semantic output order: `center_heatmap`, `tip_heatmap`, `confidence`
  - raw TFLite output order is reordered with `[1, 0, 2]`
- QAT Keras validation behavior is good:
  - accepted MAE: `2.4923 C`
  - acceptance rate: `0.7660`
  - worst accepted error: `11.9105 C`
- QAT INT8 validation behavior is not good enough:
  - accepted MAE: `4.1516 C`
  - acceptance rate: `0.5957`
  - worst accepted error: `29.2107 C`
  - accepted >20 C failures: `1`
- The QAT INT8 model fails the replay gates before we even consider test:
  - acceptance rate is below `0.65`
  - worst accepted error is above `20 C`
  - there is at least one accepted >20 C failure

## What Worked

- The corrected decoder restored Keras replay quality.
- The QAT float32 export preserved that quality.

## What Failed

- The QAT INT8 export did not preserve the replay behavior closely enough.
- Compared with the prior current INT8 baseline, the QAT INT8 export is worse on validation drift and much worse on tip stability.

## Next Action

- Recommended next action: **A. train heatmap_v3 with quantization-noise loss baked in from the start**

## Final Note

- Because the QAT INT8 validation gate failed, no final test-split QAT replay was run.
