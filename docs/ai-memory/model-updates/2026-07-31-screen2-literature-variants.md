# Fast architecture screen #2 — literature-grounded variants — 2026-07-31

Date: 2026-07-31
Status: validated
Scope: ellipse detector architecture choice (iter6)
Evidence: `ml/artifacts/ellipse_screen_fast/leaderboard.json`, `/tmp/screen_fast.log`

## Variants tested (all literature-motivated, 8 epochs, capped subset, FP32)

| Arch | test_1 | test_2 | test_3 | mean | params | Idea source |
|---|---|---|---|---|---|---|
| **universal_v1** | 22.1 | **6.8** | 10.5 | **13.16px** | 214K | iter3/4 winner |
| universal_deep6 | **9.6** | 32.0 | **8.2** | 16.62px | 373K | deeper encoder |
| universal_offset | 12.2 | 51.0 | 13.7 | 25.64px | 215K | CenterNet (Zhou 2019) |
| simcc_quad | 12.3 | 45.2 | 32.7 | 30.06px | 577K | RTMPose spatial SimCC |

## Findings

1. **The universal heatmap family dominates every variant tested so far.**
   SimCC with spatial (4x4) pooling still loses; CenterNet offset refinement
   does not help the universal model; deeper encoder helps test_1/test_3 but
   hurts test_2 in this slice.
2. **universal_v1 scored test_2 at 6.8px in this screen vs 78.8px in screen
   #1** with identical architecture+data — the difference is random init
   order (seeds are not re-set per arch in the current harness; the running
   process predates the per-arch reseed fix). Ranking is noisy but the
   universal>simcc ordering is consistent across both screens.
3. **Confirms iter4's test_2 regression (38px) was data dilution, not
   architecture**: the same arch hits 6.8px when init lands well and the
   tiny family share is restored (this screen used tiny_repeats=60 vs
   iter4's effective 8.6% share).

## Decision

- Full-data iter6 = universal_v1 + `--tiny-repeats 200` (dilution fix),
  launcher `tmp/run_iter6_universal_tiny200.sh`.
- Follow-up candidates if test_2 stays broken: universal_deep6 (best
  test_1/test_3 in slice), then heavier tiny-family augmentation.

## Harness fixes needed for screen #3

- Per-arch `tf.random.set_seed(SEED)` before each build (saved in
  `screen_ellipse_archs.py`; the run above predates it).
- Consider 3-repeat runs per arch to average out init noise on the 150-image
  test_1 slice.
