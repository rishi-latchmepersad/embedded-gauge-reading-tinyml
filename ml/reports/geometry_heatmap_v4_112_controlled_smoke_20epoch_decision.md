# Geometry Heatmap v4 112 — Next Step After 20-Epoch Smoke

## Current State

| Metric | Epoch 3 | Epoch 10 | Epoch 20 | Trend |
|--------|---------|----------|----------|-------|
| Center MAE | 8.41 | 8.41 | 8.37 | Flat |
| Tip MAE | 57.07 | 30.52 | 28.56 | ↓ still dropping |
| Angle MAE | 48.82° | 13.96° | 13.17° | Plateauing |
| Center spread | ~44.2 | 44.2 | 44.3 | Flat |
| Tip spread | ~48.2 | 48.2 | 47.5 | Flat |
| Acceptance | 0% | 0% | 0% | — |

## Shadow (spread=55) Performance at Epoch 20
- Acceptance: 74.5%
- Accepted MAE: 3.64 C
- Worst accepted: 9.80 C
- >20C failures: 0

## Decision: D — Recalibrate 112-specific spread guardrail

**Rationale:**

The model is learning good geometry — tip MAE at 28.6px (still improving), angle MAE at 13.2° (plateauing). Under a relaxed spread threshold of 55px, 74.5% of samples would be accepted with 3.64 C MAE and no >20C failures. The sole reason for 0% acceptance is the 30px spread threshold, which was tuned for 56×56 heatmaps and is structurally incompatible with 112×112 heatmaps.

**Why not the other choices:**
- **A (full training):** Premature — the spread issue must be fixed first or full training will waste compute on a model with proven geometry but no acceptance.
- **B (30 epochs):** Tip MAE is still dropping but spread is flat — more epochs won't fix the guardrail mismatch.
- **C (adjust tip loss):** Tip MAE is improving (28.6px and dropping). The loss weighting is not the bottleneck.
- **E (fix architecture):** Spread is a structural property of 112×112 heatmaps, not a bug. The model finds correct tip locations (sub-30px MAE) despite wide spreads.

## Recommended Action

1. Update `max_heatmap_spread_px` from 30.0 to 55.0 in the threshold JSON for 112×112 models
2. Re-run validation with new thresholds to confirm acceptance ≥ 70% and MAE ≤ 4.5 C
3. If acceptance confirmed, proceed to full training (choice A)
