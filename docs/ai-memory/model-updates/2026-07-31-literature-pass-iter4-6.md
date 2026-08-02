# Literature pass for ellipse-detector iterations 4-6 — 2026-07-31

Date: 2026-07-31
Status: current
Scope: gauge/ellipse detection architecture direction for the STM32 N6 NPU
Evidence: arXiv review of gauge-reading and keypoint papers (2024-2026)

## Papers reviewed this pass

### "Under pressure: learning-based analog gauge reading in the wild" (ICRA 2024, arXiv:2404.08785, ETH Zurich)
- Staged, interpretable pipeline with per-step failure detection; no prior on
  gauge type/range; <2% relative reading error.
- Validates our staged design and per-domain test gates. Reading value is
  computed from explicit geometry, not end-to-end scalar regression.

### "DialBench: pointer meter reading with large foundation models" (Nov 2025, arXiv:2511.21982)
- MRLM: vision-language model with physical-relation injection (pointer-scale
  geometry), RPM-10K dataset (10,730 images).
- VLM is far too heavy for STM32 N6; the pointer-to-scale *relation* idea is
  already embodied in our explicit center/rim heatmap + geometry heads.

### "Learning to Read Analog Gauges from Synthetic Data" (WACV 2024, arXiv:2308.14583)
- Two-stage CNN; keypoints = min marker, max marker, center, tip; angle from
  geometry. 52% error reduction over prior SOTA.
- Supports keypoint/marker supervision as a next lever if heatmap MAE stalls.

### SimCC (arXiv:2107.03332) — in repo research doc
- Coordinate classification (x/y axis logits) outperforms heatmaps at low
  resolution and quantizes cleanly (softmax logits). Our repo already has
  `_build_simcc_head` in `ml/src/embedded_gauge_reading_tinyml/models_geometry_v2.py`.
- Prior SimCC v4/v5/v6 runs (2026-06) failed on PXL→board domain shift, not
  on the head design; the board pool is now 598 images (was 201), so the
  domain-shift blocker is largely addressed.

## Implications for iteration 5+

1. Iteration 4 = same universal multiscale arch as iter3 + ALL labelled data
   (25,650 samples vs 14,008). Evaluate it first; data-coverage lesson
   predicts gains on test_1/test_3.
2. If heatmap center MAE is still the bottleneck, replace the fine center
   head with SimCC coordinate classification (384 bins, sub-pixel, INT8
   friendly) while keeping coarse proposal + rim + geometry + scale heads.
3. Keep 384² input unless a 512² experiment clearly beats it AND still fits
   the N6 NPU budget (2.5 MB SRAM; 384² iter3 tflite was 257 KB).
4. Marker supervision (min/max scale markers) is the next lever if SimCC
   does not clear the <10px gates.

## Decisions

- Iteration 5 will be a big architectural change from the pure heatmap
  universal model (per user instruction), first candidate: SimCC-fine-center
  hybrid at 384², memory-safe (uint8 + capped shuffle + preflight + guard).
