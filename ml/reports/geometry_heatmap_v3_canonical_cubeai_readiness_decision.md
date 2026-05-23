# Geometry Heatmap v3 Canonical CubeAI Readiness Decision

## Summary

- Canonical trainer contract confirmed: yes
- Canonical Keras validation passed: yes
- Canonical INT8 validation passed: no
- Cube.AI packaging allowed: no

## Canonical Keras Validation

- Accepted MAE: 3.5472 C
- Acceptance rate: 0.7021
- Worst accepted error: 12.2516 C
- Accepted >20 C failures: 0

## Canonical INT8 Validation

- Accepted MAE: 3.3839 C
- Acceptance rate: 0.5745
- Worst accepted error: 11.8365 C
- Accepted >20 C failures: 0
- Temp drift mean/median/p90: 2.0145 / 1.1273 / 4.2997 C
- Tip drift mean/median: 14.8213 / 12.4642 px
- Guardrail disagreements: 7

## Why It Is Blocked

- The INT8 model misses the acceptance threshold.
- The INT8 model misses the temperature drift threshold.
- Tip drift remains materially worse than the prior dynamic-range result.

## Next Step

**A. Increase heatmap resolution to 112x112.**

The canonical Keras model is now trustworthy enough to export, but the INT8 path still loses too much localization precision in the tip heatmap. Increasing heatmap resolution is the most direct next step.

