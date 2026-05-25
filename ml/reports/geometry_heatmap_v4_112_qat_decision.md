# QAT Decision: Deploy FP32

## Summary
QAT fine-tuning (8 epochs, 200 steps/epoch) failed to produce a working INT8 model.
The QAT training caused the model to collapse: heatmap outputs are near-zero,
confidence saturated at 1.0. Recommendation: deploy FP32 from Phase 10E.

## QAT Training Results
| Metric | Initial | Final (epoch 8) |
|--------|---------|-----------------|
| Total loss | 0.0215 | 0.0032 |
| Center heatmap MSE | 0.0119 | 0.0016 |
| Tip heatmap MSE | 0.0080 | 0.0016 |
| Confidence BCE | 0.0152 | 0.00063 |

## Root Cause
The QAT-wrapped model's MSE loss on sparse heatmaps is not robust to quantization
noise. The optimizer finds a degenerate solution: output all-zero heatmaps with
max confidence. The per-pixel MSE for all-zero prediction (~0.001) is close to
the QAT-achieved loss (0.0016), confirming the collapse.

## Key Evidence
1. Untrained QAT-wrapped model produces reasonable heatmaps (mean=0.52)
2. After 1600 fine-tuning steps, heatmaps are near-zero (mean=0.0001)
3. Float32 TFLite from trained QAT model is also garbage (not just INT8)
4. QAT Keras model cannot be loaded directly (tfmot deserialization failure)

## Phase 10E Validation Baseline
| Model | MAE | Acceptance | Drift |
|-------|-----|-----------|-------|
| Keras FP32 | 3.39 C | 78.7% | — |
| INT8 (PTQ) | 4.13 C | 66.0% | 2.0 C mean |
| QAT INT8 | NaN | 0.0% | N/A |

## Recommendation
- Deploy Phase 10E **FP32 TFLite** (8.1 MB, zero drift vs Keras)
- FP32 can run on STM32 N6 with external QSPI flash or via NPU DMA
- Accept INT8 is not viable for this model architecture + quantization pipeline
- Future work: try different loss (e.g., focal MSE), different QAT library (e.g.,
  QAT from tensorflow directly rather than tfmot), or use smaller model

## Files
- Phase 10E FP32: `ml/artifacts/deployment/geometry_heatmap_v4_112_tflite/model_v4_112_float32.tflite`
- QAT artifacts: `ml/artifacts/deployment/geometry_heatmap_v4_112_qat/`
- QAT script: `ml/scripts/qat_finetune_geometry_heatmap_v4_112.py`
