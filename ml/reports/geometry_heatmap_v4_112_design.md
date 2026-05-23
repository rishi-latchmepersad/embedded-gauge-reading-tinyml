# Geometry Heatmap v4 112x112 Design

## Why v3 Helped

- v3 fixed the replay parity bug by making trainer-side checkpoint selection use the canonical board-replay validation path.
- The export and reload paths were already trustworthy, and v3 confirmed that Keras reload parity and FP32 TFLite parity were clean.

## Why v3 Still Failed INT8

- Even after the canonical replay fix, the INT8 model still showed too much temperature drift and especially too much tip localization drift.
- The main problem is not model serialization or evaluation drift anymore.
- The remaining blocker is spatial quantization sensitivity in the tip heatmap.

## Why Tip Drift Is the Main Blocker

- The tip heatmap drives the angle and calibrated temperature.
- On a 56x56 heatmap, one heatmap pixel corresponds to roughly 4 crop pixels in 224-space.
- Small INT8 changes in the heatmap shape can therefore move the decoded tip by a large amount in crop space.

## Why 112x112 Is the Next Smallest Serious Change

- A 112x112 heatmap halves the spatial step in crop space from about 4 pixels to about 2 pixels.
- That should make softargmax w3 and peak localization less fragile under INT8 quantization.
- This is the smallest architecture change that directly attacks the remaining failure mode without changing the input contract or decoder policy.

## Expected Cost

- The heatmap outputs are 4x larger than 56x56 outputs, so output memory and export size increase.
- The decoder remains lightweight: MobileNetV2 backbone with a modest progressive upsampling head.
- The model should still be feasible to inspect for STM32N6 deployment, but the larger output tensors may increase latency and memory pressure.

## What Stays Fixed

- RGB bilinear input
- 224x224 input crop
- center and tip heatmaps
- softargmax w3
- robust calibration
- board guardrails
- canonical validation checkpoint selection

