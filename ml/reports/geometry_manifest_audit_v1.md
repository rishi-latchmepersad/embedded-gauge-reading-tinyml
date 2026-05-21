# Geometry Manifest Audit Report (v1)

**Generated:** 2026-05-21  
**Phase:** Phase 1 - Geometry Foundation  
**Target:** Inner Celsius dial center and needle tip prediction

---

## Executive Summary

**Key Finding:** The repo has **352 images with complete geometry labels** (dial center, needle tip, ellipse, temperature) in CVAT XML format, but these labels have **NOT been exported to a training manifest CSV**. 

**Recommendation:** We CAN train center/tip geometry models. The labels exist but need to be converted from CVAT XML to a geometry manifest CSV format.

---

## 1. Candidate Manifest/Label Files Found

### 1.1 CVAT Annotation Files (Primary Geometry Source)

| Source | Location | Images | Geometry Complete |
|--------|----------|--------|-------------------|
| gauge_1_batch_1 | ml/data/labelled/gauge_1_batch_1.zip | 50 | Yes (100%) |
| gauge_1_batch_2 | ml/data/labelled/gauge_1_batch_2.zip | 50 | Yes (100%) |
| gauge_1_batch_3 | ml/data/labelled/gauge_1_batch_3.zip | 50 | Yes (100%) |
| gauge_1_batch_4 | ml/data/labelled/gauge_1_batch_4.zip | 50 | Yes (100%) |
| gauge_1_batch_5 | ml/data/labelled/gauge_1_batch_5.zip | 50 | Yes (100%) |
| gauge_1_batch_6 | ml/data/labelled/gauge_1_batch_6.zip | 50 | Yes (100%) |
| gauge_1_batch_7 | ml/data/labelled/gauge_1_batch_7.zip | 50 | Yes (100%) |
| gauge_1_batch_8 | ml/data/labelled/gauge_1_batch_8.zip | 2  | Yes (100%) |
| **Total** | | **352** | **352 (100%)** |

### 1.2 Existing CSV Manifests (Temperature Only)

Found 50 CSV files in ml/data/. **None contain center/tip geometry columns.**

Key manifests with temperature labels:

| Manifest | Rows | Has Value | Has Crop | Has Center/Tip |
|----------|------|-----------|----------|----------------|
| full_scalar_manifest_v1.csv | 538 | Yes | No | No |
| weighted_full_range_v2.csv | 1204 | Yes | No | No |
| rectified_scalar_raw_tail_v17.csv | 1060 | Yes | No | No |
| canonical_manifest_v1.csv | 141 | Yes | Yes (crop) | No |
| geometry_crop_boxes_v18.csv | 383 | Yes | Yes (x0,y0,x1,y1) | No |
| unified_training_manifest_v1.csv | 409 | Yes | No | No |

**Note:** The file named \geometry_crop_boxes_v18.csv\ contains only **crop boxes** (x0,y0,x1,y1), NOT center/tip geometry.

---

## 2. Available Columns Per File Type

### 2.1 CVAT XML Format (Rich Geometry)

Each CVAT annotation contains:

| Column | Type | Description |
|--------|------|-------------|
| image_name | string | Image filename (e.g., PXL_20260125_114517176.jpg) |
| width, height | int | Source image dimensions |
| cx, cy | float | **Dial center X, Y** (ellipse center) |
| rx, ry | float | Ellipse semi-major and semi-minor radii |
| rotation | float | Ellipse rotation angle (degrees) |
| temp_c | float | **Temperature label** (-29 to 49 C) |
| center_x, center_y | float | **Needle center point** (temp_center label) |
| tip_x, tip_y | float | **Needle tip point** (temp_tip label) |

### 2.2 Existing CSV Manifests (Limited)

Common columns found:

| Column | Frequency | Description |
|--------|-----------|-------------|
| image_path | 50/50 files | Image path |
| value | 50/50 files | Temperature in Celsius |
| crop_x_min, crop_y_min, crop_x_max, crop_y_max | 7 files | Rectified crop box |
| x0, y0, x1, y1 | 7 files | Alternative crop box format |
| source_file, source_tag | ~10 files | Provenance tracking |
| split | 0 files | **No train/val/test splits defined** |
| center_x, center_y, tip_x, tip_y | 0 files | **NOT PRESENT** |

---

## 3. Label Statistics

### 3.1 Temperature Distribution (CVAT Data)

| Statistic | Value |
|-----------|-------|
| Total labeled images | 352 |
| Temperature range | -29 C to 49 C |
| Unique temperature values | 9 |
| Most common temperatures | 37.5 C (55), 49 C (49), -7.5 C (46), 15 C (45), -17.5 C (44) |

**Note:** Temperature labels are discretized (step of 7.5 C or 8 C approximately).

### 3.2 Geometry Label Completeness

| Label Type | Count | Percentage |
|------------|-------|------------|
| Images with ellipse (temp_dial) | 352 | 100% |
| Images with center (temp_center) | 352 | 100% |
| Images with tip (temp_tip) | 352 | 100% |
| **Images with ALL geometry** | **352** | **100%** |
| Images with temperature | 352 | 100% |

---

## 4. Duplicate Analysis

### 4.1 Within CVAT Data

| Metric | Value |
|--------|-------|
| Total image entries | 352 |
| Unique image names | 352 |
| Duplicate image names | 0 |

**No duplicates** in CVAT batches - each image is unique.

### 4.2 Across CSV Manifests

| Metric | Value |
|--------|-------|
| Total image path entries | 10,216 |
| Unique image paths | 538 |
| Images in multiple manifests | 538 (100%) |

**Heavy overlap** across CSV manifests - same images appear in many files with different weighting or filtering strategies.

### 4.3 CVAT vs CSV Overlap

Preliminary check shows CVAT images (PXL_20260125_*.jpg pattern) are from a different capture session than many CSV manifest images (capture_*.png pattern). **Cross-referencing needed** to determine overlap.

---

## 5. Conflicting Labels

### 5.1 Temperature Conflicts

The file \ull_scalar_manifest_conflicts_v1.csv\ (65 rows) explicitly tracks conflicts where the same image has different temperature values across source manifests.

**Action needed:** When building geometry manifest, resolve conflicts by:
1. Preferring CVAT temperature labels (manually verified)
2. Using precedence rules from canonical_manifest_v1.csv
3. Flagging unresolved conflicts for review

### 5.2 Geometry Conflicts

**No geometry conflicts detected** - each image appears only once in CVAT data with a single set of center/tip labels.

---

## 6. Crop Box Availability

| Manifest | Rows | Crop Format | Usable for Geometry |
|----------|------|-------------|---------------------|
| geometry_crop_boxes_v18.csv | 383 | x0,y0,x1,y1 | Yes (loose crop) |
| rectified_crop_boxes_v5_all.csv | 409 | x0,y0,x1,y1 | Yes (loose crop) |
| rectified_crop_boxes_full_scalar_v1.csv | 538 | x0,y0,x1,y1 | Yes (loose crop) |
| canonical_manifest_v1.csv | 141 | crop_x_min, etc. | Yes (loose crop) |

**Note:** These crop boxes are for the **full gauge face** or **rectified crops**, NOT specifically for the inner Celsius dial. They can be used as:
- Starting point for loose crops
- Validation that geometry predictions are within reasonable bounds

---

## 7. Recommendation: Can We Train Center/Tip Geometry Now?

### YES - With Caveats

| Requirement | Status | Notes |
|-------------|--------|-------|
| Dial center labels | **READY** | 352 images with temp_center points |
| Needle tip labels | **READY** | 352 images with temp_tip points |
| Temperature labels | **READY** | 352 images with temp_c attribute |
| Image paths | **NEEDS WORK** | CVAT has filenames, need full paths |
| Train/val/test split | **MISSING** | Must define splits |
| Manifest CSV format | **MISSING** | Must convert from CVAT XML |
| Crop boxes for inner dial | **MISSING** | Need to compute from ellipse or label |

### Required Next Steps

1. **Convert CVAT XML to geometry manifest CSV**
   - Extract center_x, center_y, tip_x, tip_y, temp_c, image_path
   - Add source image dimensions (width, height)
   - Add ellipse parameters (cx, cy, rx, ry, rotation)

2. **Define train/val/test splits**
   - Ensure temperature distribution is balanced
   - Consider hardness tags from existing manifests

3. **Compute loose crop boxes**
   - Option A: Use ellipse parameters to bound the inner dial
   - Option B: Label new crop boxes specifically for inner dial
   - Option C: Use existing full-gauge crops and filter to inner dial region

4. **Resolve image paths**
   - Map CVAT filenames to actual image locations
   - Verify images exist and are readable

---

## 8. Proposed Geometry Manifest Schema

\\\csv
image_path,temperature_c,split,source_width,source_height,loose_crop_x1,loose_crop_y1,loose_crop_x2,loose_crop_y2,center_x_source,center_y_source,tip_x_source,tip_y_source,dial_radius_source,label_quality,source_manifest,notes
\\\

| Column | Source | Notes |
|--------|--------|-------|
| image_path | Derived | Must resolve from CVAT filename |
| temperature_c | CVAT temp_c | From ellipse attribute |
| split | To define | train/val/test assignment |
| source_width, source_height | CVAT | From image element |
| loose_crop_* | To compute | Bounding box around inner dial |
| center_x_source, center_y_source | CVAT | From temp_center points |
| tip_x_source, tip_y_source | CVAT | From temp_tip points |
| dial_radius_source | CVAT | From ellipse rx/ry (average) |
| label_quality | To compute | Based on annotation confidence |
| source_manifest | Derived | CVAT batch filename |
| notes | Optional | Flags for edge cases |

---

## 9. Files Generated in This Audit

| File | Location | Purpose |
|------|----------|---------|
| geometry_manifest_audit_v1.md | ml/reports/ | This report |
| cvat_geometry_audit_all.csv | tmp/ | Full CVAT geometry extraction (352 rows) |
| manifest_audit_summary.csv | ml/data/ | Summary of all 50 CSV manifests |
| audit_geometry_labels.py | tmp/ | Audit script (batch 1-2) |
| full_geometry_audit.py | tmp/ | Full audit script (all batches) |
| manifest_audit.py | tmp/ | CSV manifest analysis script |

---

## 10. Conclusion

**The repo has sufficient center/tip labels for Phase 2 (model training).**

The 352 CVAT-labeled images provide:
- Complete geometry (center + tip + ellipse) for every image
- Temperature labels spanning the full range (-29 C to 49 C)
- No duplicate images within the labeled set

**Blocker:** The labels exist only in CVAT XML format and must be converted to a training manifest CSV before model training can begin.

**Recommended Phase 2 tasks:**
1. Write CVAT-to-manifest converter script
2. Define train/val/test splits (e.g., 70/15/15)
3. Compute loose crop boxes from ellipse parameters
4. Validate geometry labels (check for outliers)
5. Create geometry_reader_manifest_v1.csv

---

*End of Report*
