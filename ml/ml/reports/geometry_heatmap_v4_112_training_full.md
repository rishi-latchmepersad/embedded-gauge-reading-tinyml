# Geometry Heatmap v4 112x112 Quantization-Native Training

- Decoder: softargmax w3
- Calibration candidate: D_robust_linear
- Selected stage: frozen
- Selected replay candidate: shadow_spread_45

## Frozen Stage Val Replay
- Accepted MAE: 3.9080 C
- Acceptance rate: 0.8085
- Worst accepted error: 9.8009 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 29.79% / 61.70% / 95.74%
- Center MAE px: 15.7254
- Tip MAE px: 19.9242
- Angle MAE deg: 15.4195

## Final Val Replay
- Accepted MAE: 3.9080 C
- Acceptance rate: 0.8085
- Worst accepted error: 9.8009 C
- Accepted >20 C failures: 0
- Under 2/5/10 C: 29.79% / 61.70% / 95.74%
- Center MAE px: 15.7254
- Tip MAE px: 19.9242
- Angle MAE deg: 15.4195