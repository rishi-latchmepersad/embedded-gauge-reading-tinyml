# Geometry Heatmap v4 112 Next-Step Decision

## Smoke Outcome Summary

| Check | Status |
|-------|--------|
| Smoke completed | ✓ |
| NaNs/Infs | None |
| Acceptance > 0 | ✗ (0/47) |
| Heatmap spreads sharp | ✗ (~45px vs target ~6px) |
| Sigma/evaluator mismatch | Fixed (sigma_pixels=2.5) |
| Guardrail threshold mismatch | Present (30px threshold on 112x112) — but secondary; primary issue is undertraining |

## Decision Tree Evaluation

**A. Run 8–10 epoch controlled smoke?**
- ✓ 3-epoch smoke is stable (no NaNs, finite losses)
- ✓ Model is undertrained, not structurally broken (center coarse-localised at 11px)
- ✓ Heatmaps are diffuse but peak values are high (0.90/0.95) — suggest capacity exists
- ✗ Cannot confirm *improvement* (no per-epoch history), but tip is the main failure (75px vs 11px center)

**B. Run full v4 training?**
- ✗ Acceptance is zero; not ready for full training

**C. Fix architecture?**
- ✗ Center localisation (11px) suggests the 112 head is learning *something*; tip being worse is consistent with the tip having higher weight (2x) and needing sharper spatial features — more epochs may help

**D. Fix guardrails for 112?**
- ✗ Even relaxing `max_heatmap_spread_px=30`, the tip MAE of 75px and angle MAE of 99 degrees make predictions unusable; guardrails aren't the primary problem

**E. Return to v3?**
- ✗ v4 112 is stable and shows some initial localisation; abandoning after 3 epochs is premature

## Decision: **A — Run 8 to 10 epoch controlled smoke**

### Rationale
The v4 112 architecture is stable (no NaNs, finite losses after 3 epochs). Center coarse-localisation (11px at 224 scale) shows the skip decoder is learning. The main failure — tip prediction (75px) and diffuse heatmaps (~45px spread) — is consistent with a severely undertrained decoder. An 8–10 epoch smoke will reveal whether the heatmaps sharpen and tip converges with more training.

### Command
```bash
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
timeout 14400s bash -lc 'TF_CPP_MIN_LOG_LEVEL=2 poetry run python scripts/train_geometry_heatmap_v4_112_quant_native.py --frozen-epochs 8 --unfrozen-epochs 0 --batch-size 8 --frozen-learning-rate 3e-6 --sigma-pixels 2.5 --output-dir /tmp/geomq_v4_112_smoke_8epoch'
```

### Pass/Fail Criteria
- **Pass**: acceptance rate > 0, heatmap spread < 20px, tip MAE < 20px, angle MAE < 20°
- **Marginal**: acceptance > 0, spread 20–30px, center MAE improves
- **Fail**: acceptance still 0, spread > 30px, angle MAE > 60°
