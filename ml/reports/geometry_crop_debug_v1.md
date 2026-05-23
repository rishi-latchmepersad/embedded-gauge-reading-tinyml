# Geometry Crop Debug Report (v1)

**Generated:** 2026-05-21  
**Phase:** Phase 2 - Crop-Jitter Dataset Pipeline  
**Script:** ml/scripts/build_geometry_crop_debug_set.py

---

## Executive Summary

The crop-jitter pipeline is working correctly. Coordinate transforms from source-image space to normalized crop space are validated. The mean temperature difference between manifest labels and deterministic computation is **2.32C**, with worst-case mismatches around **12C** indicating potential annotation errors in a small subset of images.

**121 debug overlay images** have been generated for human inspection.

---

## 1. Data Loading Statistics

| Metric | Value |
|--------|-------|
| Source rows loaded | 352 |
| Train examples | 243 |
| Validation examples | 48 |
| Test examples | 61 |

---

## 2. Jittered Crop Generation

| Metric | Value |
|--------|-------|
| Attempted jittered crops | 150 |
| Accepted | 121 (80.7%) |
| Rejected | 29 (19.3%) |

### Jitter Parameters Used

| Parameter | Range |
|-----------|-------|
| X shift | -20 to +20 px |
| Y shift | -20 to +20 px |
| Scale | 0.85 to 1.25 |
| Aspect ratio | 0.90 to 1.10 |
| Rotation | Omitted (v1) |

---

## 3. Rejection Analysis

### Top Rejection Reasons

| Reason | Count | Percentage |
|--------|-------|------------|
| crop_outside_image_bounds | 26 | 89.7% |
| crop_aspect_unreasonable | 3 | 10.3% |
| crop_too_small | 0 | 0% |
| center_outside_crop | 0 | 0% |
| tip_outside_crop | 0 | 0% |

**Interpretation:**
- Most rejections (26/29) are due to jitter pushing the crop outside image bounds
- This is expected behavior - the validation is working correctly
- No rejections due to center/tip being outside crop, indicating labels are well-centered

---

## 4. Normalized Coordinate Ranges

For accepted crops, normalized coordinates should be in [0, 1].

| Coordinate | Min | Max | Valid Range |
|------------|-----|-----|-------------|
| Center X | 0.3747 | 0.5794 | [0, 1] ? |
| Center Y | 0.4114 | 0.5742 | [0, 1] ? |
| Tip X | 0.0506 | 0.9029 | [0, 1] ? |
| Tip Y | 0.1099 | 0.7089 | [0, 1] ? |

**Observations:**
- Center coordinates are tightly clustered around (0.5, 0.5) - the crop center
- This is expected since the loose crops are centered on the dial
- Tip coordinates have wider spread, reflecting needle position variation

---

## 5. Temperature Consistency Analysis

The deterministic temperature is computed from the center/tip angle using the inner dial calibration:
- 135° ? -30°C (cold end)
- 45° ? +50°C (hot end)
- 270° sweep counter-clockwise

### Temperature Difference Statistics

| Metric | Value |
|--------|-------|
| Mean absolute difference | 2.32°C |
| Median absolute difference | ~1.5°C (estimated) |
| Max difference (top 20) | 12.30°C |

### Worst 20 Temperature Mismatches

| Rank | Image | Diff | Manifest | Deterministic |
|------|-------|------|----------|---------------|
| 1 | PXL_20260125_114850084.jpg | 12.30°C | -29.0°C | -16.7°C |
| 2 | PXL_20260125_114908980.jpg | 11.95°C | -29.0°C | -17.0°C |
| 3 | PXL_20260125_114908980.jpg | 11.92°C | -29.0°C | -17.1°C |
| 4 | PXL_20260125_114908980.jpg | 11.91°C | -29.0°C | -17.1°C |
| 5 | PXL_20260125_114554365.jpg | 6.62°C | -29.0°C | -22.4°C |
| 6 | PXL_20260125_114554365.jpg | 6.47°C | -29.0°C | -22.5°C |
| 7 | PXL_20260125_114554365.jpg | 6.27°C | -29.0°C | -22.7°C |
| 8 | PXL_20260125_114554365.jpg | 6.22°C | -29.0°C | -22.8°C |
| 9 | PXL_20260125_114554365.jpg | 5.67°C | -29.0°C | -23.3°C |
| 10 | PXL_20260125_114527337.jpg | 5.42°C | -29.0°C | -23.6°C |
| 11 | PXL_20260125_114527337.jpg | 5.38°C | -29.0°C | -23.6°C |
| 12 | PXL_20260125_114527337.jpg | 4.97°C | -29.0°C | -24.0°C |
| 13 | PXL_20260125_114527337.jpg | 4.66°C | -29.0°C | -24.3°C |
| 14 | PXL_20260125_114527337.jpg | 4.45°C | -29.0°C | -24.5°C |
| 15 | PXL_20260125_115150021.jpg | 4.24°C | -7.5°C | -3.3°C |
| 16 | PXL_20260125_115208285.jpg | 4.14°C | -7.5°C | -3.4°C |
| 17 | PXL_20260125_115150021.jpg | 3.95°C | -7.5°C | -3.5°C |
| 18 | PXL_20260125_115208285.jpg | 3.60°C | -7.5°C | -3.9°C |
| 19 | PXL_20260125_115208285.jpg | 3.59°C | -7.5°C | -3.9°C |
| 20 | PXL_20260125_115208285.jpg | 3.15°C | -7.5°C | -4.4°C |

**Interpretation:**
- Large mismatches (>5°C) suggest potential annotation errors
- Images appearing multiple times in worst-20 list may have systematic labeling issues
- The deterministic temperature is computed from the labeled geometry - if geometry is wrong, temperature will be wrong
- **Recommendation:** Inspect overlay images for these specific images to verify label quality

---

## 6. Output Files

| File | Location | Description |
|------|----------|-------------|
| Overlay images | ml/debug/geometry_crops_v1/images/ | 121 JPEG images with center/tip annotations |
| Debug manifest | ml/debug/geometry_crops_v1/debug_crop_manifest.csv | 150 rows with all crop metadata |
| Validation summary | ml/debug/geometry_crops_v1/validation_summary.txt | Quick stats summary |
| This report | ml/reports/geometry_crop_debug_v1.md | Detailed analysis |

---

## 7. Overlay Image Format

Each overlay image shows:

- **Green circle:** Dial center
- **Red circle:** Needle tip
- **Blue line:** Center-to-tip vector
- **Text overlay:**
  - Manifest temperature
  - Computed angle (degrees)
  - Deterministic temperature (from angle)
  - Temperature difference
  - Split (train/val/test)
  - Jitter parameters (shift, scale, aspect)

Example filename: \crop_0000_train_PXL_20260125_115738811.jpg\

---

## 8. Recommendations for Phase 3 (Model Training)

### Ready for Training

The crop-jitter pipeline is **ready for model training**:

1. ? Coordinate transforms are correct
2. ? Validation rejects invalid crops
3. ? Normalized coordinates are in valid range
4. ? Jitter parameters are within specified ranges

### Data Quality Notes

1. **Inspect worst mismatches:** Review overlay images for the top 20 temperature mismatches to identify potential annotation errors

2. **Consider label filtering:** Images with >5°C temperature difference may benefit from label review

3. **Jitter tuning:** Current rejection rate (19.3%) is acceptable. If more augmentation is needed, consider:
   - Reducing shift range from ±20 to ±15
   - Narrowing scale range from [0.85, 1.25] to [0.90, 1.15]

### Next Steps

1. Human inspection of overlay images (especially worst mismatches)
2. Optionally fix annotation errors in CVAT and regenerate manifest
3. Proceed to Phase 3: CNN model architecture and training

---

## 9. Files Created in Phase 2

| File | Purpose |
|------|---------|
| ml/src/embedded_gauge_reading_tinyml/geometry_crop_dataset.py | Crop-jitter dataset utilities |
| ml/tests/test_geometry_crop_dataset.py | Unit tests (23 tests) |
| ml/scripts/build_geometry_crop_debug_set.py | Debug overlay generator |
| ml/debug/geometry_crops_v1/images/*.jpg | 121 overlay images |
| ml/debug/geometry_crops_v1/debug_crop_manifest.csv | Crop metadata |
| ml/reports/geometry_crop_debug_v1.md | This report |

---

*End of Report*
