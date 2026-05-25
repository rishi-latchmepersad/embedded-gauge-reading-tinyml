# Geometry Heatmap v4 112 Controlled Smoke Decision

## 10-Epoch Smoke Summary

| Check | Status |
|-------|--------|
| 10 epochs completed | ✓ |
| No NaNs/Infs | ✓ |
| Acceptance > 0 | ✗ (0/47 all epochs) |
| Tip MAE trend | ↓ 57→30 (46% improvement) |
| Angle MAE trend | ↓ 49→14 (71% improvement) |
| Spread trend | Flat (~44-48px, no change) |
| Logging fixed (history.csv) | ✓ |
| Process stability | ✓ (with batch_size=4, eager eval) |

## Decision Tree

**A. Run full v4 training?**
- ✗ Spread not trending downward; acceptance still zero

**B. Continue controlled smoke to 20 epochs?**
- ✓ Tip MAE is clearly improving (57→30 and still trending down at epoch 10)
- ✓ Angle MAE is dramatically improving (49→14 and still trending down)
- ✓ Distance ratio implausibility rejections dropping (37→10)
- ✓ No signs of convergence plateau
- ✗ Spread is flat — may need more epochs to sharpen
- → **This is selected**

**C. Fix architecture?**
- Defer: tip MAE is still improving. If spread is flat at epoch 20, then fix.

**D. Adjust loss weighting?**
- Not indicated: tip (weight=2.0) is improving faster than center (weight=1.0)

**E. Revisit guardrail spread threshold?**
- Not yet: predictions at 30px tip MAE / 14° angle are still too poor for deployment even if guardrails were relaxed

## Decision: **B — Continue controlled smoke to 20 epochs**

### Rationale
The v4 112 head is learning: tip MAE dropped 46%, angle MAE dropped 71%, and the distance-ratio rejection count improved 73%. These trends are still progressing at epoch 10. The flat spread may begin to shrink as coordinate regression converges further. 20 epochs will determine whether spread can sharpen with extended training or requires an architecture change.

### Command
```bash
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
timeout 28800s bash -lc 'TF_CPP_MIN_LOG_LEVEL=2 poetry run python scripts/train_geometry_heatmap_v4_112_quant_native.py --frozen-epochs 20 --unfrozen-epochs 0 --batch-size 4 --frozen-learning-rate 3e-6 --sigma-pixels 2.5 --output-dir /tmp/geomq_v4_112_smoke_20epoch'
```

### Pass/Fail Criteria
- **Pass**: spread < 30px or acceptance > 0
- **Marginal**: spread 30–40px with tip MAE < 20px
- **Fail**: spread > 40px and tip MAE > 25px with no downward trend over epochs 10–20
