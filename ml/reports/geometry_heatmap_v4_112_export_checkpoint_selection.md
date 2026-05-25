# Geometry Heatmap v4 112 — Export Checkpoint Selection

## Available Checkpoints

| File | Path | Size | Timestamp | Contents |
|------|------|------|-----------|----------|
| `model_v4_112.keras` | `.../30epoch_smoke/model_v4_112.keras` | 8,583,737 | May 23 17:35 | Final epoch-10 weights (30 total) |
| `best_model.keras` | `.../30epoch_smoke/best_model.keras` | 8,583,737 | May 23 17:29 | Same as above (saved post-training) |
| `model_v4_112_frozen_best.keras` | `.../30epoch_smoke/model_v4_112_frozen_best.keras` | 8,583,737 | May 23 17:29 | Stale — epoch-1 weights (NaN scoring prevented best update) |

## Selected Checkpoint

**`model_v4_112.keras`** from the 30-epoch smoke directory.

- **Timestamp:** May 23 17:35
- **Effective training:** 10 epochs from V3-initialized checkpoint (restore_best_weights reverted intermediate runs, so the model trained ~10 effective epochs)
- **Validation metrics** (from canonical_summary.json):
  - Accepted MAE: **3.39 C**
  - Acceptance rate: **78.7%**
  - Worst accepted error: **9.52 C**
  - >20C failures: **0**
  - Tip MAE: **26.71 px**
  - Center MAE: **8.34 px**
  - Angle MAE: **12.22°**
  - Center spread: **44.35 px**
  - Tip spread: **46.61 px**

## Guardrail Profile

- **JSON:** `artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json`
- **Family:** `geometry_heatmap_v4_112`
- **Spread threshold:** 55.0 px (was 30.0 in V2)
- **Do not use for V2 or V3 models**

## Decoder

- **Method:** softargmax
- **Window size:** 3 (corrected decoder lock)
