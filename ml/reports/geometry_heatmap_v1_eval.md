# Geometry Heatmap v1 Evaluation Report

## Overview

This report evaluates the first heatmap-based geometry model for gauge reading.
The model predicts center and tip heatmaps (56x56) from 224x224 crops, then decodes
coordinates using softargmax.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | MobileNetV2 (alpha=0.35, frozen) |
| Input size | 224x224x3 |
| Heatmap size | 56x56 |
| Sigma (Gaussian) | 2.5 pixels |
| Batch size | 16 |
| Epochs | 100 |
| Learning rate | 1e-4 to 1e-6 (decay) |
| Training samples | 206 |
| Validation samples | 42 |

## Metrics Comparison

### Temperature Error

| Split | MAE (C) | RMSE (C) | Under 2C | Under 5C | Under 10C |
|-------|----------|-----------|-----------|-----------|------------|
| Train | 26.00 | 30.49 | 0.9% | 7.5% | 19.0% |
| Val | 25.29 | 29.23 | 0.0% | 6.4% | 14.9% |
| Test | 24.10 | 28.45 | 3.4% | 11.9% | 27.1% |

### Coordinate Error (224x224 space)

| Split | Center MAE (px) | Tip MAE (px) |
|-------|-----------------|--------------|
| Train | 72.3 | 89.7 |
| Val | 49.5 | 89.7 |
| Test | 45.4 | 80.2 |

### Angle Error

| Split | Angle MAE (degrees) |
|-------|---------------------|
| Train | 82.8 |
| Val | 81.2 |
| Test | 78.3 |

## Comparison with Phase 3 Coordinate Model

| Metric | Phase 3 (Coordinates) | Phase 4 (Heatmaps) | Delta |
|--------|----------------------|-------------------|-------|
| Test Temp MAE | 7.91C | 24.10C | +16.2C worse |
| Test Center MAE | 11.30 px | 45.4 px | +34 px worse |
| Test Tip MAE | 21.82 px | 80.2 px | +58 px worse |
| Test Angle MAE | 25.16 | 78.3 | +53 worse |
| Under 10C | 72.9% | 27.1% | -46% worse |

## Diagnosis

The heatmap model performs significantly worse than the coordinate regression baseline.
Key observations:

1. Low training loss but poor generalization: The heatmap MSE loss converged to ~0.0125,
   but this does not translate to accurate coordinate predictions.

2. Weak heatmap peaks: Predicted heatmap peak values are very low (0.06-0.17),
   suggesting the model outputs diffuse heatmaps rather than sharp Gaussian peaks.

3. Possible causes:
   - Frozen backbone may be too restrictive for heatmap learning
   - Sigma=2.5 may be too narrow for the model to learn precise localization
   - MSE loss on heatmaps does not directly optimize for coordinate accuracy
   - Model may be predicting near-uniform or centrally-located heatmaps as a safe minimum

## Recommendations

1. Unfreeze backbone: Allow fine-tuning of at least the last few MobileNetV2 blocks
2. Increase sigma: Try sigma=4-6 pixels for more forgiving heatmap targets
3. Add coordinate loss: Add auxiliary MSE loss on decoded coordinates
4. Try argmax instead of softargmax: May be more robust to diffuse predictions
5. Consider U-Net decoder: Current upsampling may lose spatial precision

## Files Generated

- Model: ml/artifacts/training/geometry_heatmap_v1/model.keras
- Config: ml/artifacts/training/geometry_heatmap_v1/config.json
- History: ml/artifacts/training/geometry_heatmap_v1/history.csv
- Test predictions: ml/artifacts/training/geometry_heatmap_v1/test_predictions.csv
- Worst 30: ml/artifacts/training/geometry_heatmap_v1/worst_30_predictions.csv
- Metrics: ml/artifacts/training/geometry_heatmap_v1/eval_metrics.json

## Conclusion

The heatmap v1 model is not ready for deployment. The coordinate regression baseline
(Phase 3) significantly outperforms this approach. Before proceeding to board-style replay,
we need to:

1. Investigate why heatmaps are not learning sharp peaks
2. Try architectural modifications (unfrozen backbone, better decoder)
3. Consider hybrid approaches (heatmaps + coordinate heads)
4. Verify heatmap ground truth generation is correct
