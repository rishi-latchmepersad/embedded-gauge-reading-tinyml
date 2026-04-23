# ML Experiments and Research Notes

This file holds the model-training history and research direction.
See `archive.md` for the full chronology.

## Classical CV Baseline

- The baseline started as a radial-spoke voting system.
- Bright-center, training-crop, and image-center heuristics were useful for debugging, but glare could mislead them.
- The baseline should act as a rough sanity check, not the final answer.
- A conservative mode is better than a brittle "guess anyway" baseline.

## Scalar Fine-Tuning

- The scalar model improved with board-style calibration and crop-box fitting.
- Rectified-crop experiments showed that the input domain matters more than another small calibration tweak.
- A crop-box affine correction was the best board-safe postprocess for a while.
- The current scalar deployment path should stay simple unless a new model clearly beats it on the board split.

## Geometry Experiments

- Geometry-first MobileNetV2 models are wired up in the repo.
- The long-term geometry run learned too narrow a pose prior and collapsed toward a near-fixed angle.
- The long-term direction run also collapsed and did not beat a trivial baseline.
- The detector-first MobileNetV2 geometry run also failed to become a useful reader. On the held-out test split it reached `gauge_value_mae=23.8880` while the baseline mean predictor was `20.1698`, so the detector branch is not yet better than a trivial predictor.
- The inspection script showed that the keypoint branch is the weak link, not just the final temperature mapping.
- A pure heatmap or pure direction head is not enough if the spatial supervision is too weak.

## Current Research Direction

- The next model family should make the geometry more explicit.
- Stronger choices include detector-style localization, oriented boxes, or direct needle-direction supervision with a more constrained output space.
- The best path is likely a compact geometry model with deterministic postprocessing, not a monolithic scalar regressor.

## Active Training Split

- Training uses the broad sweep set in the train pool.
- Board close-up hard cases are held out for validation.
- Rectified board probe crops are held out for final test.
- This pinned split has been useful for separating "looks good offline" from "works on the board".

## What to Remember

- The current geometry branch is a learning signal, not the production baseline.
- If a new run collapses to a narrow angle or a trivial predictor, stop and rethink the geometry target rather than just lowering the learning rate.
- Keep the pinned split and inspect sample-level predictions before declaring a geometry model viable.
- Do not treat the detector-first result as a recovery path unless the geometry formulation changes in a meaningful way.
