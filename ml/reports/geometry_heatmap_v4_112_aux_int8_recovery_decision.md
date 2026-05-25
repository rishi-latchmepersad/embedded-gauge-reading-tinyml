# Phase 11E Aux INT8 Recovery Decision

## Question

Can an INT8-friendly auxiliary coordinate head reduce Keras-vs-INT8 temperature drift below 1.0 C?

## Answer

**NO** -- no candidate passed all validation gates.

Best candidate: 11_w05_large_huber
INT8 drift: 1.8423 C (baseline: 1.8405 C)

### Analysis

Drift did not materially improve vs baseline. The GAP-only aux head architecture
is fundamentally limited for INT8 robustness. The pooled features do not provide
enough spatial resolution to resist INT8 quantization noise in the tip heatmap.

### Next steps

1. Move to a spatially-aware point head that reads from the decoder feature maps
   (e.g., a small conv head that predicts offsets from the heatmap peak locations).
2. Or add a tip-only aux head with dedicated spatial features.
3. Or accept the 1.8-1.9 C drift as the architectural limit for GAP-based aux and
   focus on other improvements (larger backbone, better calibration, etc.).

## All Candidates

| # | Candidate | Drift | MAE | Accept | Status |
|---|-----------|-------|-----|--------|--------|
| 1 | 11_w05_large_huber | 1.8423 | 3.76 | 72.34% | fail |
| 2 | 12_w10_large_huber | 1.8458 | 3.81 | 72.34% | fail |
| 3 | 06_w10_small_huber | 1.8472 | 3.80 | 72.34% | fail |
| 4 | 02_w05_small_mse | 1.8495 | 3.81 | 72.34% | fail |
| 5 | 04_w02_small_huber | 1.8510 | 3.82 | 72.34% | fail |
| 6 | 01_w02_small_mse | 1.8554 | 3.79 | 72.34% | fail |
| 7 | 05_w05_small_huber | 1.8600 | 3.81 | 72.34% | fail |
| 8 | 09_w10_large_mse | 1.8606 | 3.80 | 72.34% | fail |
| 9 | 03_w10_small_mse | 1.8629 | 3.82 | 72.34% | fail |
| 10 | 08_w05_large_mse | 1.8651 | 3.80 | 72.34% | fail |
| 11 | 10_w02_large_huber | 1.8653 | 3.80 | 72.34% | fail |
| 12 | 07_w02_large_mse | 1.8654 | 3.83 | 72.34% | fail |