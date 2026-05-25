# Geometry Heatmap v4 112 — INT8 Validation Decision

## Validation Gate Check

| Criterion | Required | INT8 Actual | Status |
|-----------|----------|-------------|--------|
| Accepted MAE | ≤ 4.5 C | 4.13 C | ✅ |
| Acceptance rate | ≥ 65% | 66.0% | ✅ |
| Worst accepted error | < 20 C | 16.05 C | ✅ |
| >20C failures | = 0 | 0 | ✅ |
| Temp drift mean | ≤ 1.0 C | 1.99 C | ❌ |
| Tip drift mean | < 14.82 px | 11.53 px | ✅ |

**Validation: FAILED** — INT8 temperature drift (1.99 C) exceeds the 1.0 C limit.

## Readiness Assessment

| Question | Answer |
|----------|--------|
| Did export succeed? | ✅ Yes — FP32 and INT8 TFLite artifacts created cleanly |
| Does FP32 TFLite match Keras? | ✅ Yes — Δ = 3.27e-05 C drift, 0 guardrail disagreements |
| Does INT8 pass validation? | ❌ No — temp drift 1.99 C exceeds 1.0 C gate |
| Is the tensor contract clear? | ✅ Yes — documented in tflite_tensor_contract.json |
| Is test replay allowed? | ❌ No — validation gate failed |
| Is Cube.AI allowed? | ❌ No — blocked until INT8 validation passes |

## Root Cause

INT8 quantization noise widens heatmap distributions, causing:
1. **Tip spread increase:** 46.61 px → 49.84 px (+7%), causing 7 samples to exceed spread=55 guardrail (Keras-accepted → INT8-rejected)
2. **Softargmax coordinate shift:** 11.53 px mean tip drift between Keras and INT8
3. **Temperature prediction shift:** 1.99 C mean drift from heatmap quality changes

The FP32 TFLite matches Keras perfectly (drift < 0.001 C), confirming the model export itself is clean. The INT8 quantization calibration is insufficient.

## Decision: B — Improve Representative Dataset and Re-Export

**What failed:** INT8 quantization drift (1.99 C) exceeds drift tolerance (1.0 C).

**Why this fix:** The current representative dataset uses identity + mild jitter (shift 3-7 px, scale 0.97-1.03) × 2 on 227 train samples = 454 images. The quantization may not capture the full range of heatmap activations seen during inference, leading to poor calibration of the heatmap outputs.

**Next steps:**

1. **Augment representative dataset:**
   - Increase jitter diversity: add moderate jitter (shift 8-15 px, scale 0.94-1.06)
   - Ensure temperature-stratified coverage (currently has cold/cool/warm/hot bins)
   - Increase per-sample variants from 2 to 4-6
   - Add synthetic brightness/contrast variation if feasible
   - Target: 1000-2000+ representative samples

2. **Re-export INT8** with improved representative dataset
3. **Re-run validation replay**
4. If INT8 drift still exceeds 1.0 C after improved dataset, consider **Option D: Dynamic range fallback** (FP32 on NPU, or per-channel quantization)

## Why Not the Other Choices

- **A (proceed to test replay):** Not allowed — validation gate failed
- **C (inspect guardrails/overlays):** Guardrails are correct (V4 spread=55). The issue is quantization drift, not guardrail tuning
- **D (auxiliary head / dynamic range):** Premature — representative dataset improvement is a lower-effort fix
- **E (export/evaluator bug):** FP32 matches Keras perfectly. Export pipeline is clean

## Files

- `ml/reports/geometry_heatmap_v4_112_int8_validation.md` — full validation report
- `ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/v4_112_tflite_replay_summary_val.csv` — per-model summaries
- `ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/v4_112_tflite_replay_predictions_val.csv` — per-sample predictions
- `ml/debug/geometry_heatmap_v4_112_tflite_validation/` — 60 validation overlays
