# Ellipse detector winner: universal_wide_deep — all gates under 10px — 2026-07-31

Date: 2026-07-31
Status: current
Scope: ellipse detector architecture (final selection)
Evidence: `ml/artifacts/ellipse_screen_fast/leaderboard.json` (screen #3, seed-reset),
`ml/artifacts/ellipse_iter7_universal_hflip/report.json`

## Screen #3 leaderboard (10 architectures, 14 epochs, ~6,000 samples, FP32, per-arch seed reset)

| Rank | Arch | mean | t1 | t2 | t3 | params |
|---|---|---|---|---|---|---|
| **1** | **universal_wide_deep** (24,32,48,64,96,128) | **8.72** | 9.1 | 9.1 | 8.0 | 708,853 |
| 2 | universal_wide_eca | 8.95 | 12.2 | 5.2 | 9.5 | 375,944 |
| 3 | universal_noskip | 10.15 | 16.1 | 7.3 | 7.0 | 178,893 |
| 4 | universal_eca | 10.66 | 14.2 | 8.7 | 9.1 | 215,328 |
| 5 | universal_v1 (iter7 arch) | 13.48 | 13.9 | 18.7 | 7.8 | 214,029 |
| 6 | universal_offset (CenterNet) | 17.29 | 13.7 | 26.9 | 11.3 | 214,679 |
| 7 | simcc_quad_wide | 25.13 | 11.7 | 38.1 | 25.6 | 737,642 |
| 8 | simcc_quad | 25.69 | 11.6 | 35.0 | 30.5 | 577,346 |
| 9 | universal_deep6 | 36.34 | 9.1 | 90.3 | 9.5 | 373,469 |
| 10 | universal_wider | 37.18 | 13.5 | 92.1 | 5.9 | 374,069 |

## Selection

**universal_wide_deep**: 6-stage encoder (24,32,48,64,96,128), 3-scale
center+rim heatmap heads, geometry + scale-confidence heads (the universal
contract). Only architecture with ALL THREE slices under 10px. 708K params
fits the 2.5 MB SRAM budget comfortably.

Training recipe (from iter7, which cleared t2/t3 on full data):
- all labelled data (8,233 generic + 22 tiny×200 + 598 board×4 = 30,050 samples)
- 50% random hflip in `_augment_uint8` (fixed the 178px hflip failure)
- test_3 duplicate (`board_captures_2`) excluded from training
- uint8 storage + capped shuffle + preflight + `run_wsl_guarded.sh`

## Full-data confirmation run

Launcher: `tmp/run_iter8_wide_deep.sh`
Output: `ml/artifacts/ellipse_iter8_universal_wide_deep/`
Command: `--channels 24,32,48,64,96,128 --tiny-repeats 200 --board-repeats 4`

## Next

1. Confirm on full data (25+10 epochs, QAT, int8 eval on all 3 test zips).
2. Then sweep architectures for the center/tip keypoint model (same
   harness pattern, `tip_focus` contract: 224×224 → 56×56 heatmaps).
