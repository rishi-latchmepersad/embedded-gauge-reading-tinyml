# Classical CV Baseline

This document tracks the classical computer vision baseline for gauge reading.
It is the reference floor that any ML model must beat to justify its complexity.

## Pipeline

The live STM32 baseline is a deterministic multi-hypothesis polar-vote
pipeline over the 224x224 preview frame:

1. Capture a YUV422 preview snapshot and profile luma/brightness. If the
   frame is bright, the baseline switches to the more permissive
   `bright-relaxed` thresholds.
2. Build five center hypotheses: bright centroid, fixed training crop, board
   prior, rim geometry, and the inner image center.
3. For each accepted seed, vote over 360 polar bins for the darkest
   needle-like spoke. Saturated pixels and the subdial mask are rejected, the
   middle shaft is weighted more heavily, and the vote score is boosted by hub
   continuity, tip extension, and radial/tangential alignment. A hot-zone
   wrap-around rescue handles needles near the sweep boundary.
4. When the local geometry sweep is enabled, each accepted seed is refined
   over a tiny 5x5 offset grid and the best-quality estimate is kept.
   Remaining candidates are collapsed with a consensus selector.
5. Apply the confidence and peak-separation gate, then either publish the new
   estimate or hold/smooth the small 3-frame history buffer if the frame is
   ambiguous or unstable.
6. Convert the accepted angle to temperature with the calibrated 135° / 270°
   sweep map spanning -30°C to +50°C. A small angle offset is applied before
   the final value is logged.

This is the firmware flow implemented in
[firmware/stm32/n657/Appli/Src/app_baseline_runtime.c](../firmware/stm32/n657/Appli/Src/app_baseline_runtime.c).
The benchmark tables below are historical evaluation results from the older
classical baseline harness.

Key source files:
- [ml/src/embedded_gauge_reading_tinyml/baseline_classical_cv.py](../ml/src/embedded_gauge_reading_tinyml/baseline_classical_cv.py)
- [ml/src/embedded_gauge_reading_tinyml/single_image_baseline.py](../ml/src/embedded_gauge_reading_tinyml/single_image_baseline.py)
- [ml/src/embedded_gauge_reading_tinyml/gauge/gauge_calibration_parameters.toml](../ml/src/embedded_gauge_reading_tinyml/gauge/gauge_calibration_parameters.toml)

Eval script: [ml/scripts/eval_classical_baseline_on_manifest.py](../ml/scripts/eval_classical_baseline_on_manifest.py)

Runner: [ml/scripts/run_classical_baseline_eval.sh](../ml/scripts/run_classical_baseline_eval.sh)

## Benchmark Results (2026-04-19)

Artifacts: `ml/artifacts/baseline/classical_cv_20260419_073921/`

### hard_cases.csv (19 samples, all successful)

| Metric | Value |
|--------|-------|
| MAE | **13.99°C** |
| RMSE | 24.21°C |
| Max error | 66.15°C (`capture_0073.png`, true=46°C, pred=−20°C) |
| Cases > 5°C | 10 / 19 |

Worst failures:
- `capture_0073.png` 46°C → −20.2°C (needle detected inverted)
- `capture_2026-04-03_08-20-49.png` 45°C → −20.3°C (same flip)
- `capture_p5c.png` 5°C → 40.8°C (35.8°C error)
- `capture_p20c_preview.png` 20°C → 44.8°C (24.8°C error)

Good predictions (< 4°C):
- `capture_m30c_preview.png`, `capture_m19c.png`, `capture_m18c.png`, `capture_m10c_preview.png`, `capture_0c_preview.png`, `capture_0008.png`, `capture_p30c.png`, `capture_p50c_preview.png`

### hard_cases_plus_board30_valid_with_new5.csv (31 samples, 21 successful)

| Metric | Value |
|--------|-------|
| MAE | **12.79°C** (computed on successful only) |
| RMSE | 23.03°C |
| Max error | 66.15°C |
| Cases > 5°C | 10 / 21 successful |
| Failed (no needle) | 10 |

The 10 failed samples are all close-up `.jpg` board captures where the HoughCircles dial localiser fails to find the gauge circle — the frame fills too much of the image. The `.png` hard cases all produce a prediction.

## Comparison vs prod_model_v0.2

| Manifest | Classical MAE | prod_model_v0.2 MAE |
|----------|--------------|----------------------|
| hard_cases.csv | 13.99°C | ~5–6°C (offline) |
| hard_cases_plus_board30_valid_with_new5.csv | 12.79°C* | ~4–5°C (offline) |
| 33°C board captures | — | 0.62°C |

*Computed only on successful detections; 10 close-up images fail entirely.

The ML model is roughly **2–3× better** on the hard-case manifests and handles close-up frames the classical pipeline cannot.

## Known Failure Modes

1. **Needle inversion** — Two images at 45–46°C get a ~−20°C prediction (needle detected tail-first). The scoring heuristic favours the wrong segment when the needle is very close to the minimum tick mark.
2. **Close-up framing** — When the gauge fills >~60% of the frame, `HoughCircles` with the current radius constraints (`minRadius = 18%`, `maxRadius = 48%` of image min-dimension) fails to find the dial circle. All 10 `.jpg` board captures fail this way.
3. **Mid-band ambiguity** — `capture_p5c.png` (5°C), `capture_p20c_preview.png` (20°C), and `capture_p10c_preview.png` (10°C) are badly wrong (7–36°C errors), likely due to low needle contrast at those dial positions.

## Potential Improvements (not yet tried)

- Adaptive `HoughCircles` radius window based on detected gauge size, to handle close-up captures
- Second-pass needle scoring using the gauge sweep arc as a prior (reject candidates outside the calibrated sweep)
- Inversion detection: if predicted value is near the limits (−30°C or 50°C) and confidence is low, try flipping the needle direction

## How to Re-run

```bash
wsl -d Ubuntu-24.04 -e bash /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/scripts/run_classical_baseline_eval.sh
```

Results are written to `ml/artifacts/baseline/classical_cv_<timestamp>/`.

