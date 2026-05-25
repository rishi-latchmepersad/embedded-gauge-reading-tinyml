# Geometry Heatmap v4 112 — TFLite INT8 Validation Report

## Checkpoint
- **Source:** `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras`
- **Metrics:** Accepted MAE 3.39 C, Acceptance 78.7%, Worst error 9.52 C
- **Decoder:** softargmax w3
- **Guardrails:** V4-specific (spread=55)
- **Calibration:** D_robust_linear

## Export Artifacts

| Artifact | Path | Size |
|----------|------|------|
| FP32 TFLite | `model_v4_112_float32.tflite` | 7.6 MB |
| INT8 TFLite | `model_v4_112_int8.tflite` | 2.2 MB |
| Export config | `export_config.json` | — |
| Tensor contract | `tflite_tensor_contract.json` | — |
| Rep dataset manifest | `representative_dataset_manifest.csv` | 454 rows |

## Tensor Contract
- **Input:** 1×224×224×3, FP32 (→INT8 for int8 model), normalization `uint8_to_float32_0_1`
- **Outputs:** 3 tensors — center_heatmap (112×112), tip_heatmap (112×112), confidence (scalar)
- **Semantic reorder:** [1, 0, 2] (TFLite outputs: tip_heatmap, center_heatmap, confidence → reordered to center, tip, confidence)
- **Decoder:** softargmax w3 on 112×112 heatmaps
- **Guardrail JSON:** `v4_112_guardrail_thresholds.json` (spread=55)

## Validation Replay — Val Split (47 samples, V4 guardrails)

### Keras (Reference)
| Metric | Value |
|--------|-------|
| Accepted MAE | 3.39 C |
| Acceptance rate | 78.7% (37/47) |
| Worst accepted error | 9.52 C |
| >20C failures | 0 |
| Under 2/5/10C | 27.7% / 61.7% / 78.7% |
| Center MAE | 8.34 px |
| Tip MAE | 26.71 px |
| Center spread | 44.35 px |
| Tip spread | 46.61 px |
| Top rejections | tip spread>55:8, center-tip ratio:7, angle sweep:2, temp margin:1 |

### FP32 TFLite (vs Keras)
| Metric | Value | vs Keras |
|--------|-------|----------|
| Accepted MAE | 3.39 C | Δ = 0.00003 C ✓ |
| Acceptance rate | 78.7% | Δ = 0.0% ✓ |
| Worst accepted error | 9.52 C | Δ = 0.0001 C ✓ |
| Temp drift mean | 3.27e-05 C | ✓ |
| Center drift mean | 1.68e-05 px | ✓ |
| Tip drift mean | 2.42e-04 px | ✓ |
| Guardrail disagreements | 0 | ✓ |

**FP32 TFLite matches Keras to machine precision.** Export pipeline is clean.

### INT8 TFLite (vs Keras)
| Metric | Value | vs Keras | Gate |
|--------|-------|----------|------|
| Accepted MAE | 4.13 C | Δ = +0.75 C | ✅ ≤ 4.5 |
| Acceptance rate | 66.0% (31/47) | Δ = −12.7 pp | ✅ ≥ 65% |
| Worst accepted error | 16.05 C | Δ = +6.53 C | ✅ < 20 |
| >20C failures | 0 | 0 | ✅ = 0 |
| Under 2/5/10C | 17.0% / 51.1% / 61.7% | −10.6pp / −10.6pp / −17.0pp | — |
| Center MAE | 8.45 px | Δ = +0.12 px | — |
| Tip MAE | 32.07 px | Δ = +5.36 px | — |
| Center spread | 44.38 px | Δ = +0.03 px | — |
| Tip spread | 49.84 px | Δ = +3.23 px | — |
| **Temp drift mean** | **1.99 C** | — | **❌ ≤ 1.0 C** |
| Temp drift p90 | 3.80 C | — | — |
| Center drift mean | 0.68 px | — | — |
| **Tip drift mean** | **11.53 px** | — | ✅ < 14.82 |
| Guardrail disagreements | 9 / 47 | — | — |

**Validation gate: FAILED** — temperature drift (1.99 C) exceeds limit (1.0 C).

## Drift Analysis

### Guardrail Status Changes
- **Keras-accepted → INT8-rejected:** 7 samples
  - Root cause: INT8 quantization increases tip heatmap spread, causing more samples to exceed spread=55 guardrail
  - Tip spread increase: ~46.6 px (Keras) → ~49.8 px (INT8), +7%
- **Keras-rejected → INT8-accepted:** 1 sample
- **No change:** 39 samples (30 accepted by both, 9 rejected by both)

### Temperature Drift (Keras vs INT8, both accepted samples)
- Mean: 1.99 C — exceeds 1.0 C drift gate
- Median: 1.55 C
- P90: 3.80 C
- Top drift: 5.16 C (true 5.5°C → Keras 5.5°C → INT8 10.6°C)

### INT8 Accepted >10°C Errors
- 2 samples with |ΔT| > 10°C under INT8 (both near 16°C worst case)

### Tip Drift (Keras vs INT8, both accepted samples)
- Mean: 11.53 px — below v3 canonical INT8 (14.82 px) but above v2 dynamic range (5.44 px)
- Median: 10.08 px

## Overlays
Generated in `ml/debug/geometry_heatmap_v4_112_tflite_validation/`:
- 30 worst accepted INT8 predictions by temperature error
- 2 accepted errors >10°C (INT8)
- 7 Keras-accepted → INT8-rejected cases
- 1 Keras-rejected → INT8-accepted case
- 10 largest temperature drifts (Keras vs INT8)
- 10 largest tip drifts (Keras vs INT8)
