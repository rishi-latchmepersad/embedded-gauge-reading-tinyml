# Fast architecture screen #1 — universal heatmap beats SimCC decisively — 2026-07-31

Date: 2026-07-31
Status: validated
Scope: ellipse detector architecture choice (iter5-6)
Evidence: `ml/artifacts/ellipse_screen_fast/leaderboard.json`, `/tmp/screen_fast.log`

## Method

New fast-screening harness `ml/scripts/screen_ellipse_archs.py`: one process,
capped training subset (2,500 generic + 22×60 tiny + 598×2 board ≈ 6,000
samples, ×2 augment), 8 FP32 epochs, fixed 150-image test_1 slice + full
test_2/test_3, FP32-only ranking. ~10 min for 4 architectures (vs ~40 min
per full run). Launched via `run_wsl_guarded.sh` (all memory safeguards).

## Leaderboard (mean center MAE over the 3 slices)

| Arch | test_1 | test_2 | test_3 | mean | params |
|---|---|---|---|---|---|
| **universal_v1** (heatmaps) | 15.39 | 78.82 | 9.74 | **34.65px** | 214K |
| simcc_v1 (GAP head) | 12.49 | 188.52 | 47.95 | 82.99px | 516K |
| simcc_eca | 12.27 | 189.50 | 48.81 | 83.52px | 517K |
| simcc_wider | 12.67 | 190.48 | 51.09 | 84.75px | 676K |

## Conclusions

1. **SimCC-with-GAP is a dead end for tiny/off-center gauges.** Global
   average pooling erases spatial position; fine for centered gauges
   (test_1 ~12px, near image-center prior) but catastrophic on test_2/test_3
   (~188-190px). This explains the iter5 full-run collapse (195px test_2)
   — the architecture was wrong, not just QAT.
2. **Universal heatmap architecture is the family to keep.** It already
   passed test_3 (6.23px) on the full-data iter4 run.
3. The screen's absolute numbers are NOT comparable to full runs (smaller
   set, 8 epochs, FP32); use it only for ranking.

## Next

iter6 = universal_v1 architecture + `--tiny-repeats 200` (fixes the iter4
test_2 dilution where the tiny IMG_144x family share fell from 15.7% to
8.6%). Launcher: `tmp/run_iter6_universal_tiny200.sh`.
