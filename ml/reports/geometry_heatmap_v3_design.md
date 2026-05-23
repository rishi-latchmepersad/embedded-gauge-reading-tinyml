# Geometry Heatmap v3 Design

## Why v2 Keras Worked

- The corrected decoder `softargmax w3` restored Keras replay quality.
- The v2 heatmap architecture and board guardrails were already aligned with the
  RGB bilinear crop contract.
- Base Keras replay showed that the geometry signal itself was learnable.

## Why Full INT8 Drift Blocked Packaging

- The exported INT8 model drifted away from the Keras geometry even when replay
  accuracy looked acceptable on average.
- The most visible failure mode was tip instability, not just raw temperature
  error.
- Guardrail disagreements increased because the post-quantized heatmaps were
  sometimes flatter or more spatially shifted than the float model.

## Why Late QAT Did Not Solve It

- The quantization-noise fine-tuning pass improved the float checkpoint, but the
  exported INT8 model still diverged on validation replay.
- That showed the deployment objective was not being optimized early enough.
- In other words, quantization had to be part of the training objective from the
  beginning, not bolted on at the end.

## What v3 Changes

- Fake int8 output round-trips are active from epoch 1.
- Tip stability is weighted more heavily than in v2.
- Coordinate, angle, and temperature losses are computed after fake-quantized
  outputs so the training objective better matches deployment.
- Validation checkpoint selection is based on replay-style metrics, not training
  loss alone.
- Representative datasets for export stay train-only and preserve the same RGB
  bilinear crop contract.

## What Stays Fixed

- Input contract: RGB, bilinear resize, same crop preprocessing.
- Corrected decoder: `softargmax` with `window_size=3`.
- Board guardrails: same selected thresholds unless a later validation-only
  sweep proves a change is justified.
- Existing v2 artifacts remain untouched.
