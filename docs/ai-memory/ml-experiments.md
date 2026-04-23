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

## Rectifier Chain

- The rectifier + scalar reader chain remains useful as a front-end normalizer, but it is not the full answer by itself.
- On the rectified board probe set, `mobilenetv2_rectifier_zoom_aug_v4` paired with the rectified scalar deployment reached `mean_abs_err=12.4529` and `max_abs_err=27.3887`.
- The same probe set with `mobilenetv2_rectifier_hardcase_finetune_v3` improved to `mean_abs_err=9.8036`, which is the best rectifier + scalar result so far on this board probe set.
- The exported int8 rectifier `mobilenetv2_rectifier_hardcase_finetune_v3_int8` improved that again. At `RECTIFIER_CROP_SCALE=1.80`, the board-probe chain reached `mean_abs_err=6.1574` and `max_abs_err=21.2753`, which is the best rectifier-based board result so far.
- Even that best rectifier chain is still far from board-ready, and it remains worse on many samples than a strong dedicated scalar fit on the same board domain.
- The rectifier is still the more promising part of the cascade; the reader still needs stronger spatial context or a better downstream geometry target.

## OBB Cascade

- The new MobileNetV2 OBB localizer long-term run trained cleanly on the labeled split and reached `val_mae=0.1435` and `test_mae=0.1786` on the OBB parameters. That makes it the strongest explicit localizer proxy so far.
- The OBB + scalar board-probe cascade using `mobilenetv2_obb_longterm` and the rectified scalar deployment reached `mean_abs_err=3.6617`, `max_abs_err=11.8603`, and `cases_over_5c=11` at `OBB_CROP_SCALE=1.20`.
- That beats the best rectifier + scalar board result (`mean_abs_err=6.1574` at `RECTIFIER_CROP_SCALE=1.80`), so the OBB cascade is now the strongest board-probe benchmark we have.
- The OBB localizer is now wired into the firmware candidate path as `prodv0.3`, and the board project builds successfully with the OBB wrapper plus the shared scalar runtime bundle.
- The OBB cascade should be the next comparison target for board-style work, while the rectifier chain remains the fallback benchmark.

## Geometry Experiments

- Geometry-first MobileNetV2 models are wired up in the repo.
- The long-term geometry run learned too narrow a pose prior and collapsed toward a near-fixed angle.
- The long-term direction run also collapsed and did not beat a trivial baseline.
- The detector-first MobileNetV2 geometry run also failed to become a useful reader. On the held-out test split it reached `gauge_value_mae=23.8880` while the baseline mean predictor was `20.1698`, so the detector branch is not yet better than a trivial predictor.
- The latest `mobilenetv2_detector_geometry` run repeated that failure mode. It finished with `test gauge_value_mae=24.2626` while the baseline mean predictor stayed at `20.1698`, so the detector-style geometry head still does not beat a trivial predictor on the pinned test split.
- The geometry keypoint-only run improved the spatial branch but still did not beat baseline on temperature. It finished with `test gauge_value_mae=23.1730` against the mean predictor's `20.1698`, while keypoint MAE dropped to `6.6727`.
- The uncertainty-aware geometry run was the best of the latest geometry variants, but it still only reached `test gauge_value_mae=18.9273` while the baseline mean predictor was `20.1698`. That means the uncertainty head helped some, but it still did not produce a board-ready reader.
- The compact geometry long-term run is the best tiny-detector-style proxy so far. It reached `gauge_value_mae=7.4751` on the pinned board test split, which beats the mean predictor (`11.2892`), but the geometry branch is still weak with `keypoint_coords_angle_mae_deg=48.0940` and `keypoint_coords_mae=24.2516`.
- On the rectified-board probe manifest, that same compact geometry checkpoint regressed badly: `mean_abs_err=17.5719` over 39 samples, with the worst `28C` samples down in the single digits and the worst `14C` samples up around `31C`. The angle branch is still collapsing to a narrow, unstable pose prior.
- The inspection script showed that the keypoint branch is the weak link, not just the final temperature mapping.
- A pure heatmap or pure direction head is not enough if the spatial supervision is too weak.

## Current Research Direction

- The next model family should make the geometry more explicit.
- Stronger choices include detector-style localization, oriented boxes, or direct needle-direction supervision with a more constrained output space.
- The best path is likely a compact geometry model with deterministic postprocessing, not a monolithic scalar regressor.
- A rectifier-style front end is supported by recent gauge-reading research, but the stronger papers pair localization with keypoints, OBBs, segmentation, or multi-stage reasoning rather than relying on a bare crop-box regressor.
- For multiple gauges, the rectifier idea scales best when it is trained as a gauge-agnostic localizer over a diverse dataset, then paired with a downstream reader or gauge-spec-aware postprocess.
- If the gauge family changes a lot, a single rectifier is usually not enough by itself; it needs either gauge metadata, richer localization targets, or a second-stage reader that adapts per gauge type.
- The next concrete experiment to try is a tiny detector/localizer variant, ideally YOLO-style or OBB-style, used as a front-end rectifier rather than a replacement for the reader itself.
- The compact geometry long-term proxy is worth keeping as a reference point, but it still needs a more explicit geometry target before it can replace the scalar path.
- The compact geometry proxy is no longer the next experiment to bet on for the board path; use it only as a reference when comparing more explicit localizers.
- The next live follow-up is the compact geometry cascade-localizer long-term run, which keeps the model small but emphasizes hard-case localization and pins the board-style split.
- The compact geometry cascade-localizer long-term run finished with `gauge_value_mae=6.7352` on the pinned test split and `keypoint_coords_angle_mae_deg=51.0065`. It is a mild improvement on scalar accuracy, but the geometry branch is still not tight enough to stand on its own.
- The keypoint-reader cascade using that compact localizer still missed badly on the board probe set: `mean_final_abs_err=14.5682` across 39 samples and `cases_over_5c=37`. The localizer helped a little, but it did not make the cascade board-ready.
- The explicit MobileNetV2 geometry cascade-localizer long-term run is a better localizer than the compact proxy. It reached `gauge_value_mae=5.7287` and `keypoint_coords_angle_mae_deg=45.0024` on the pinned test split, and its cascade eval improved to `mean_final_abs_err=13.2531` on the board probe set.
- Even so, that is still far from board-ready: `cases_over_5c=33` and the worst board probe sample was still off by `18.8867C`.
- Do not spend more cycles on MobileNetV2 scalar/direction heads unless the geometry target changes materially.
- The current in-repo proxy for that tiny detector/localizer idea is `compact_geometry_longterm`, which keeps the model small while pinning validation and test against the board split.
- The OBB long-term launcher now uses explicit `--val-fraction` and `--test-fraction` splits and should stay off the board manifest hard-case path.

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
- A rectifier/localizer remains worthwhile, but the literature suggests it should normalize the scene for a downstream geometry reader instead of being the full reader on its own.
- The OBB cascade has now shown the strongest board-probe result so far, so future board-style comparisons should start there before trying more scalar-only variants.
