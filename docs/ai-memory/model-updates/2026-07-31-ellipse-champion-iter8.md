# ELLIPSE CHAMPION: universal_wide_deep — all 3 gates passed — 2026-07-31

Date: 2026-07-31
Status: current
Scope: ellipse detector — FINAL full-data validation
Evidence: `ml/artifacts/ellipse_iter8_universal_wide_deep/report.json`,
`/tmp/iter8_export.log` (export-only resume after memory-guard kill)

## Final full-data int8 results (30,050 samples, tiny200, hflip, QAT)

| Split | iter7 | **iter8 (champion)** | gate |
|---|---|---|---|
| test_1 center MAE | 12.67px | **7.15px** (median 5.24, 72.3% ≤8px) | ✓ <10px |
| test_2 center MAE | 6.27px | **8.32px** (median 4.87, 81.8% ≤8px) | ✓ <10px |
| test_3 center MAE | 6.10px | **4.38px** (median 3.64, 86.4% ≤8px) | ✓ <10px |
| test_1 radius | — | 7.82px | ✓ |
| test_2 radius | — | 32.46px (weak: tiny gauges) | — |
| test_3 radius | — | 8.33px | ✓ |

## The champion

- Architecture: `universal_wide_deep` — 6-stage encoder
  (24,32,48,64,96,128), 3-scale center+rim heatmap heads, geometry +
  scale-confidence heads (universal contract, 24,207 outputs). ~709K params.
- Recipe: all labelled data (8,233 generic + 22 tiny×200 + 598 board×4),
  50% random hflip augmentation, board_captures_2 (test_3 dup) excluded,
  uint8 storage + capped shuffle + preflight, 25 FP32 + 10 QAT epochs.
- Selected by screen #3 (8.72px mean, the only arch under 10px on all
  three slices).

## Notes

- The first iter8 run died at export: the call site materialized a
  17.7 GB float32 copy (`images_u8.astype(np.float32)/255.0`), tripping
  the memory guard ("memory floor breached: 1880 MiB"). Fixed by passing
  uint8 directly (export converts per-sample) and adding an `--export-only`
  resume path (loads `model_qat.weights.h5`, exports, evaluates).
- test_2 radius (32.46px) remains the weak spot — tiny-gauge radius is
  harder than center. Known and acceptable for now.

## Artifacts

- `ml/artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite`
- `ml/artifacts/ellipse_iter8_universal_wide_deep/model_qat.weights.h5`
- `ml/artifacts/ellipse_iter8_universal_wide_deep/report.json`
