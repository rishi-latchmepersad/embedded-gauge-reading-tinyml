# Geometry Heatmap v4 112 — 30-Epoch Decision

## Current State

| Metric | 30-Epoch Run (Epoch 10) | V4 Guardrail Pass Gate |
|--------|------------------------|----------------------|
| Accepted MAE | 3.39 C | ✅ ≤ 4.5 C |
| Acceptance | 78.7% | ✅ ≥ 65% |
| Worst error | 9.52 C | ✅ < 20 C |
| >20C failures | 0 | ✅ = 0 |
| Center MAE | 8.34 px | — |
| Tip MAE | 26.71 px | — |
| Angle MAE | 12.22° | — |

Model is not converged (tip MAE −42.7% in this run, still dropping 2-3px/epoch; angle MAE −58.1%, still dropping ~1°/epoch).

## Decision: A — Proceed to Export + INT8 Quantization Pipeline

**Rationale:**

The model is production-worthy now:
- **78.7% acceptance** with 3.39 C MAE — strong for 112×112
- **0 catastrophic failures** — worst error 9.52 C, no >20C failures
- **V4 guardrails calibrated** — spread=55 verified as the sweet spot
- **0 samples with >10°C error** under spread=55 at epoch 10 (was 24 in 20-epoch run)

Further training (option B) would improve tip MAE and acceptance marginally, but the ROI of getting the model onto hardware now exceeds the benefit of a few more accuracy points. The export pipeline will also reveal real-world issues (quantization accuracy, latency, memory) that can inform the next training iteration.

## Why Not the Other Choices

- **B (continue to 40 epochs):** Model is still improving, but already meets all production criteria. The export pipeline itself takes time — if issues arise, we can continue training in parallel. Not mutually exclusive with A.
- **C (fix scoring NaN):** The temperature_delta NaN is a _monitoring-only_ issue. It does not affect model quality or deployment. Fix for next training iteration but not blocking.
- **D (revisit architecture):** Premature. The model achieves its targets. Architecture changes should wait for real-world deployment data.
- **E (revisit guardrails):** Already done in Phase 10D. V4 guardrails are correct and validated.

## Recommended Action

1. **Export model to TFLite** with INT8 quantization pipeline
2. **Validate quantized accuracy** against 30-epoch model
3. **Run Cube.AI integration** for STM32 N6 NPU
4. **Begin firmware testing** with real hardware
5. (Optional) Continue training to 40 epochs in parallel with export work

## Output Artifacts for Export

- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras` — best trained model
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json` — V4 guardrail profile
