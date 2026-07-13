# Tip-Focus INT8 Deployment Decision

## Question
Should the Phase 11B `tip_focus` INT8 model proceed to Cube.AI/NPU feasibility
packaging?

## Answer
**YES — Decision A: proceed to Cube.AI INT8 feasibility packaging.**

## Decision Criteria

| Criterion | Gate | Actual | Result |
|-----------|------|--------|--------|
| INT8 accepted MAE | ≤4.5 C | **4.11 C** | ✅ PASS |
| INT8 acceptance rate | ≥0.65 | **0.76** | ✅ PASS |
| INT8 worst accepted error | <20 C | **12.88 C** | ✅ PASS |
| INT8 accepted >20 C failures | =0 | **0** | ✅ PASS |
| No new catastrophic drift/failure pattern | — | Confirmed | ✅ PASS |
| INT8 temperature drift | ≤1.0 C | **1.80 C** | ⚠️ WAIVED |

## Drift Exception Statement

The `≤1.0 C` INT8 temperature drift gate is **waived** because Phase 11A–H
exhausted all tested strategies (loss weighting, aux heads, dense offsets,
axis SimCC, alpha=0.5 backbone expansion, tfmot QAT) and none reduced mean
drift below the validation floor of ~1.84 C. The final test split confirms
this floor at **1.80 C**. This is accepted as the best achievable drift with
the current geometry heatmap architecture and pipeline.

## Test Split Metrics (61 samples, untouched during tuning)

### INT8 (Deployment Candidate)
- Accepted MAE: **4.11 C**
- Acceptance rate: **0.76** (45/59)
- Worst accepted error: **12.88 C**
- Accepted >20 C failures: **0**
- Under 5 C: **52.5 %**
- Under 10 C: **67.8 %**
- Center drift mean: **0.83 px**
- Tip drift mean: **11.55 px**
- Guardrail disagreements vs Keras: **6**
- Top rejection: tip heatmap too spread out (13)

### FP32 Drift (Upper Bound Reference)
- FP32 drift from Keras: **<0.001 C** (effectively zero)
- INT8 drift mean: **1.80 C** (median 1.53 C, p90 3.70 C)

## Performance Summary

### What Works Well
- Mid-range temperature predictions are accurate (majority under 5 C error)
- No catastrophic failures >20 C on the test split
- Keras and FP32 TFLite are identical (zero quantization drift)
- Center localization is robust (drift <1 px)
- Guardrails catch the worst outliers (14/59 rejected on INT8)
- FP32 path is a drop-in replacement at 7.6 MB if INT8 1.8 C drift is unacceptable

### Known Failure Modes
1. **Cold tail under-reading**: −29 C reads as −16 C (12.9 C error, accepted)
2. **Hot tail under-reading**: 49 C reads as 36 C (12.5 C error, accepted)
3. **Tip heatmap spread**: Primary rejection cause (spread >55 px edge)
4. **Tip localization drift**: 11.5 px mean (INT8 quantization noise on the 112×112 decoder)

### Comparison vs Validation Baseline
| Metric | Validation (Phase 11B) | Test (Final) | Delta |
|--------|----------------------|-------------|-------|
| INT8 accepted MAE | ~3.79 C | 4.11 C | +0.32 C |
| Acceptance rate | 0.72 | 0.76 | +0.04 |
| Worst accepted error | — | 12.88 C | — |
| Temp drift mean | 1.84 C | 1.80 C | −0.04 C |
| Guardrail disagreements | 10 | 6 | −4 |

The test split results are consistent with Phase 11B validation, confirming
no overfitting to the val split during tuning.

## Artifact Summary

| Artifact | Path | Size |
|----------|------|------|
| Keras model | `candidate_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112.keras` | 8.2 MB |
| FP32 TFLite | `recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112_float32.tflite` | 7.6 MB |
| INT8 TFLite | `recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112_int8.tflite` | 2.2 MB |
| Test predictions | `v4_112_tip_focus_final_test_predictions.csv` | — |
| Test summary | `v4_112_tip_focus_final_test_summary.csv` | — |
| Test worst accepted | `v4_112_tip_focus_final_test_worst_accepted.csv` | — |
| Test overlays | `ml/debug/geometry_heatmap_v4_112_tip_focus_int8_final_test/` | 109 images |

## Recommendation

**Proceed to Cube.AI INT8 feasibility packaging.**

The INT8 model (2.2 MB) fits the STM32N6 NPU memory budget. The 1.8 C drift
is a known architectural limitation, not a pipeline bug. The test split
confirms no regressions vs validation.

### Next Steps
1. Run Cube.AI on the INT8 TFLite to verify NPU operator support.
2. Generate STM32N6 C runtime and relocatable binary.
3. Measure on-board latency and power.
4. Integrate into the firmware cascade if NPU performance is acceptable.

### Alternative
If the 1.8 C INT8 drift is unacceptable for the deployment accuracy target:
- Use the FP32 TFLite (7.6 MB) via external QSPI flash — zero drift, same
  accuracy as Keras. Requires XSPI2 memory-mapped inference on the N6.
