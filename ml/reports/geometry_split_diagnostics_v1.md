# Geometry Split Diagnostics v1

## Purpose

This report investigates the validation anomaly observed in Phase 3:
- Val MAE: 13.99C
- Test MAE: 7.91C

This is unexpected since val and test should have similar distributions.

## Data Overview

| Split | Clean Rows | Percentage |
|-------|------------|------------|
| Train | 227 | 68.2% |
| Val | 47 | 14.1% |
| Test | 59 | 17.7% |
| Total | 333 | 100% |

## Temperature Distribution Analysis

### Phase 3 Coordinate Model Results

| Split | Temp MAE (C) | Center MAE (px) | Tip MAE (px) | Angle MAE (deg) |
|-------|--------------|-----------------|--------------|-----------------|
| Train | 7.37 | 9.8 | 18.5 | 22.4 |
| Val | 13.99 | 15.2 | 28.1 | 38.5 |
| Test | 7.91 | 11.3 | 21.8 | 25.2 |

### Phase 4 Heatmap Model Results

| Split | Temp MAE (C) | Center MAE (px) | Tip MAE (px) | Angle MAE (deg) |
|-------|--------------|-----------------|--------------|-----------------|
| Train | 26.00 | 72.3 | 89.7 | 82.8 |
| Val | 25.29 | 49.5 | 89.7 | 81.2 |
| Test | 24.10 | 45.4 | 80.2 | 78.3 |

## Key Observations

### Phase 3 Anomaly
1. Val error is ~1.8x higher than test error
2. Val center/tip errors are also elevated
3. This suggests val split has harder samples or distribution shift

### Phase 4 Observations
1. All splits perform poorly (heatmap model not learning well)
2. Test slightly better than train/val (unusual)
3. Val center MAE is lower than train but temp MAE similar

## Potential Causes

### 1. Small Validation Set
- Only 47 samples in validation
- High variance in metric estimates
- A few outliers can significantly impact MAE

### 2. Non-Random Split
- Original splits may have been created by source batch/folder
- Different batches may have different characteristics:
  - Lighting conditions
  - Gauge angles
  - Image quality
  - Temperature range coverage

### 3. Label Quality Variation
- Some source manifests may have less accurate labels
- If val has more samples from problematic sources, metrics suffer

### 4. Temperature Range Imbalance
- Val may have more extreme temperatures (harder to predict)
- Check temperature distribution per split

## Recommended Next Steps

1. **Analyze temperature histograms per split**
   - Plot distribution of temperature_c for train/val/test
   - Check for range imbalances

2. **Check source manifest distribution**
   - Count samples per source_manifest in each split
   - Identify if val is dominated by one source

3. **Review worst val predictions**
   - Manually inspect images with highest val errors
   - Check for annotation issues or image quality problems

4. **Consider stratified re-splitting**
   - Re-split data ensuring temperature distribution match
   - Maintain similar source manifest proportions

## Files for Further Analysis

- Manifest: ml/data/geometry_reader_manifest_v2_clean.csv
- Phase 3 predictions: ml/artifacts/training/geometry_points_v1/
- Phase 4 predictions: ml/artifacts/training/geometry_heatmap_v1/

## Conclusion

The validation anomaly is likely due to:
1. Small validation set size (47 samples)
2. Possible non-random split by source batch
3. Potential label quality variation across sources

For Phase 5, consider:
- Stratified re-splitting by temperature and source
- Increasing validation set to 10-15% of data
- Manual review of high-error validation samples
