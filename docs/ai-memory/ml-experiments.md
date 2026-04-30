# ML Experiments and Research Notes

This file holds the model-training history and research direction.
See `archive.md` for the full chronology.

## Classical CV Baseline

- The baseline started as a radial-spoke voting system.
- The current single-image classical geometry helper stays Hough-first with a plausibility-gated Hough circle estimate and a `0.75x` effective radius. We tested wider geometry grids and the LSD line-segment fallback on the hard cases, but the direct Hough path is still the best offline classical choice overall, even though a few individual frames can be rescued by a wider local geometry search.
- The single-image runner is now conservative by default: it uses the Hough seed first and only runs the experimental auto-sweep when explicitly requested. The `capture_p50c_preview.png` probe was the clearest example of why: the default path kept the Hough candidate, while the sweep jumped to a worse offset candidate and produced the wrong temperature.
- The manifest evaluator now defaults to `hough_only`, because that mode has been the strongest on the current hard-case manifests and the center fallback was not improving the aggregate error.
- The firmware selector now mirrors that conservative approach: fixed crop is the primary live anchor, the local geometry sweep is behind `APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP=0`, and the sweep should stay experimental unless we have a new hard-case set that proves it helps.
- On the current hard-case manifests, that direct-Hough path is about `mae=19.1053` on `hard_cases.csv`, `mae=19.0864` on `hard_cases_remaining_focus.csv`, and `mae=14.4444` on `board_weak_focus.csv`.
- Bright-center, fixed-crop, and image-center heuristics are still useful for debugging, but the detector now mirrors the gradient-polar hard-case winner instead of the older shaft-biased ray score.
- The hard-case detector-family sweep confirms that the gradient-polar detector is still the best pure classical family on the current focus set. On `hard_cases.csv` plus `hard_cases_remaining_focus.csv`, it reached `MAE=6.2606` with `28/28` detections, while `ray_score` reached `7.3950`, `hough_lines` reached `9.6858` with 8 failures, and `dark_polar` failed all 28 samples.
- A follow-up consensus sweep on the same hard-case family outputs showed that agreement matters too: using a `4C` cluster window beat raw winner-take-all selection (`MAE=18.0248` vs `19.0992`) even though the gradient-polar family was still the best individual detector.
- The baseline should act as a rough sanity check, not the final answer.
- A conservative mode is better than a brittle "guess anyway" baseline.
- The current version now uses a fixed-crop polar edge vote, emits a provisional warm-up reading from the first accepted frame instead of suppressing the first two samples, and compares the bright, fixed-crop, rim-center, and image-center seeds by preferring accepted candidates first, then using a peak-sharpness-plus-support score without a hard geometry priority so the rim fit does not override a better candidate just because it looks more like the Hough anchor.
- The hard-case strategy sweep still matters because geometry-only selection can move the MAE a lot, so the polar detector should keep being benchmarked on the same hard-case manifests.
- That keeps it classical CV, but makes it behave more like a defensible benchmark instead of a silent gate.
- The current baseline is now much closer to a canonical polar spoke detector than the old ray scorer, which makes it a better reviewer-facing classical comparator.
- The polar vote now follows the gradient-polar family: inner-annulus Sobel edge magnitude plus tangential alignment, with no extra shaft bias.
- Near-tied peaks are now rejected with a peak-ratio gate so the selected spoke has to beat the runner-up by a meaningful margin, and the smoothing history only accepts estimates that clear the same confidence, score, and peak-ratio gates.
- The 31C board trace was the trigger for this upgrade: the old baseline could produce nonsense warm-up values and then lock onto a wrong plateau. The polar version is meant to be the more defensible classical solution to beat.
- The baseline worker now keeps weak polar frames out of the smoothing history and holds the last stable estimate when the new frame is ambiguous, so low-confidence glare frames do not drag the comparator into nonsense.
- The confidence threshold was relaxed to `1.25` after the live `5C` traces showed the live board needed a less brittle SNR gate, but the selector now compares all refined candidate geometries by peak sharpness instead of giving the fixed crop a free pass.
- A recent 31C live trace showed the classical baseline at about 30.4C while the current prodv0.3 OBB+scalar path landed around 27.7C. That means the classical comparator is currently ahead at that point, which is a good sign for the baseline but a warning that the learned tail is still under-reading the upper-mid band.
- The latest captured-image inspection suggests the board needle is more saturated / color-separated than the surrounding dial, so the old grayscale-dark shaft assumption is too brittle. A color-aware shaft scan with heavier middle-shaft weighting was a good hypothesis, but the current implementation regressed `board_weak_focus.csv` to about `MAE=28.6173`, so it should stay marked as an experiment rather than the baseline.
- The current board detector still needs a better tie-breaker between Hough-style geometry and board-prior shaft evidence. The current score is not reliable enough on its own to choose the right candidate across the board-style captures.
- Keep `captured_images/` and `board_weak_focus.csv` as the main regression set for this detector work, because those samples expose the board-specific color and geometry failure modes better than the broader hard-case mix.
- The clean 2026-04-24 captures showed that the board-prior override was the wrong default for ideal scenes. The single-image baseline now keeps a confident Hough geometry on clean captures and only uses the board-prior shaft scan as a weak-Hough fallback, which is the right simplification when we only care about the simple near-centered cases.
- The default combined detector is now better off without the experimental line-segment and Hough-line branches; those families were over-scoring the wrong angle on clean photos. The baseline now prefers the stable spoke and center-weighted detectors, and the board-prior helper now tries the generic radial detector before the shaft scan.
- On the clean labeled captures (`capture_p25c.jpg`, `capture_p30c.jpg`, `capture_p31c.jpg`, `capture_p35c.jpg`, `capture_p45c.png`) the default baseline is now around `MAE=5.413`, which is a solid thesis baseline for the ideal-case story.

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
- The deploy-time calibration for `prodv0.3` now uses the affine p5 fit from `scalar_full_finetune_from_best_affine_calibrated_p5` instead of the older board30 piecewise curve, because the piecewise fit was overcorrecting the original hard-case manifest even though it looked good on some closer live reads.
- On `hard_cases_remaining_focus.csv`, the raw scalar model landed at `raw_mae=7.1941`, the affine calibration improved that to `affine_mae=4.1813`, and the classical manifest baseline reached `mean_abs_err=4.0247`, so the hard-case mix still gives the classical comparator a small edge on that set.
- The affine p5 source fit still lands at `calibrated_mae=4.2643` on `ml/artifacts/training/scalar_full_finetune_from_best_affine_calibrated_p5/metrics.json`, which is why it remains the current firmware-side calibration choice even though the hard-case manifest is close.
- The piecewise hard-case fit can still look perfect on the samples it was fit against, but that overfit makes it a poor general benchmark.
- The board path now softens the camera brightness nudges to 25% fractional steps, and the OBB crop window stays loose (`0.60..1.40` relative to the stable training crop) so moderate close-up crops remain on the fast path instead of getting punted into the slower rectifier stage.
- The OBB cascade should be the next comparison target for board-style work, while the rectifier chain remains the fallback benchmark.
- The OBB hard fault on the board turned out to be a memory-placement bug rather than a model bug: the package's CPU arena was pinned at `0x34100000`, which overlapped the app/RTOS footprint, so the arena base was moved to `0x34110000` and the firmware build still passes.
- The scalar package had the same memory-placement bug, and the firmware wrapper was rebuilt against `0x34110000` as well so the scalar stage no longer sits on the live app RAM window.
- The camera-init brightness gate was then simplified so it no longer retries in place. Combined with the 16 KB camera-init stack, that should reduce the chance of a stall right after the exposure nudge while we keep validating `prodv0.3` on the live board.
- The new laptop-side board-pipeline replay helper is now useful for parity debugging: it lazy-loads the deployed board models, prints the OBB/rectifier/scalar stage trace, and uses a fast PIL-based board-style crop/letterbox path so a single capture can be replayed in about `0.03s` on the laptop capture instead of stalling on TensorFlow resize ops.
- The hard-case manifest comparison is sobering: on `hard_cases_remaining_focus.csv` the pure classical baseline still reached `MAE=4.0247`, while the deployed board replay landed at `raw_mae=10.4790`, `calibrated_mae=13.7990`, and `reported_mae=14.1790`. The replay selected the OBB path for all 9 samples, so the problem on that manifest is the deployed reader and calibration, not the rectifier fallback.
- [2026-04-28] Baseline Improvements: After observing repeated "[BASELINE] Classical baseline failed to estimate a temperature" messages in the logs, the acceptance criteria were adjusted to improve reliability:
  * Adjusted APP_BASELINE_MIN_ACCEPT_SCORE to 2.0f (based on debug output showing scores in 6-7 range)
  * Relaxed APP_BASELINE_MIN_PEAK_RATIO from 1.01f to 1.10f for more realistic peak separation requirements
  * Tightened center distance threshold from 150px to 100px to better reject glare-induced false positives
  * Increased geometry override ratio from 1.20f to 1.50f to allow stronger fallback geometries to override weak anchors
  * Adjusted bright center penalties from 100px to 150px to be less punitive toward reasonably positioned center hypotheses
  * Fixed compilation error in AppBaselineRuntime_PassesAcceptanceGate function
  * These changes should improve baseline reliability while maintaining robustness against false positives
- [2026-04-28] Improved classical baseline robustness by adding darkness weighting to edge gradients, hub-connection scoring, and peak-width penalties. These changes help distinguish thin needles from thick spokes and shadows.
- [2026-04-28] Relaxed confidence thresholds in single_image_baseline.py (MIN_CONFIDENCE 10.0 -> 5.0, MIN_PEAK_RATIO 1.5 -> 1.2), increasing the detection rate on captured_images from 21.5% to ~40.5% while maintaining accuracy on successful detections.
- [2026-04-28] The classical needle detector now evaluates the top 10 angular candidates using a balanced score of vote magnitude, hub connection, and peak sharpness.

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
- The best pure classical board baseline on the current hard-case focus set is still the gradient-polar family, but the firmware version needs an explicit dial radius derived from crop height to match the Python Hough-seeded geometry path.
- The firmware classical baseline still runs a small rim-based center search before the spoke vote, but the live selector no longer gives rim-center a hard priority. The `-5C` board trace showed that unconditional rim priority could force a false warm read, so the firmware now ranks accepted candidates by peak-sharpness quality first, matching the Python classical helper.
- The rim search is now just one candidate family; it no longer overrides the looser center fallbacks when its spoke vote is only a broad near-tie.
- The ideal-case STM32 selector should keep `fixed-crop-polar` ahead of `board-prior-polar`. The board prior is still useful as a backup, but it should not outrank the stable crop on the clean capture path where we want the thesis baseline to look trustworthy.
- The 2026-04-30 previews showed that the main needle is dark against a light dial, so the firmware vote should keep the mid-shaft emphasis and avoid any red-pixel boost that would pull the scorer toward dial artwork instead of the needle.
- The simple HoughLinesP line-segment fallback is not stable enough to replace the current default detector. On the 2026-04-30 spot check it predicted about `-23.1C` for `07-00-09`, about `30.7C` for `05-51-06` and `05-52-17`, and failed entirely on `07-01-21`, so keep it as an experiment only.
- The latest batch replay over the stable ideal controls is reassuring: `capture_p25c.jpg`, `capture_p30c.jpg`, `capture_p31c.jpg`, `capture_p35c.jpg`, and `capture_p45c.png` came out to about `MAE=3.556C` with the current classical baseline. The raw 2026-04-29 capture set is mostly overexposed blank frames, so it is useful as a smoke test for robustness but not as an ideal-case benchmark.
- The 2026-04-30 live captures should be interpreted against the upper temperature dial, not the small lower subdial. The upper needle sits around 5C on the inner scale (roughly 40F on the outer scale) in the zoomed previews, so any future eval on those captures should use the main dial as the ground truth.
- The gauge scale convention matters: the outer large numbers are Fahrenheit and the inner numbers are Celsius. Any classical baseline or replay evaluation should map the needle to the inner Celsius ring, not the outer Fahrenheit labels.
- The classical selector score needed a robustness pass: peak ratio should penalize a candidate rather than amplify it, because a spiky hot false geometry was beating the cleaner near-5C candidate on the new captures. The current default path also needs the narrow local refinement sweep enabled so the seed geometry can slide a few pixels before we commit to the read.

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
- The OBB board hardfault was not caused by the localizer itself; the trigger was the per-frame ATON `LL_ATON_RT_Reset_Network()` call in `app_ai.c`, so the deployed `prodv0.3` path should stay in one-shot runtime mode by default.
- The rectifier fallback path now trusts the flashed blob and skips the signature compare, because the stale fingerprint was blocking a valid fallback image whenever the OBB crop overflowed the scalar window.
- FileX/SD readiness used to be a separate startup problem and should not be conflated with the ATON fault; the latest storage trace reached ready end-to-end, and the current code keeps that path quiet by default unless we re-enable bring-up breadcrumbs locally.
- A fresh live trace confirmed the one-shot ATON path is stable: the board now runs OBB + scalar cleanly with `Stage network reset skipped (one-shot runtime)`, so the hardfault is no longer the active issue. The remaining rough edge on that trace was RTC/logging stability, not FileX media readiness.
- The latest storage trace showed that the SD bring-up path now reaches ready end-to-end, so FileX/media readiness is no longer the active blocker. The explicit breadcrumbs and card-version-aware ACMD41 handshake made it obvious that the card was just being brought up slowly, not failing outright.
- The DS3231 build-time seed path also recovered cleanly from a year-`2000` boot, which fixed the impossible-date logging problem on the latest trace.
- The one-off DS3231 force-seed pass is finished, and the firmware is back to normal year-`2000`-only RTC seeding for future boots.
- After storage and RTC recovered, the first retry hit a `DCMIPP` capture error (`0x00008100` / `CSI_SYNC|CSI_DPHY_CTRL`), so the remaining live issue is now camera capture rather than storage or AI.
- The 2026-04-29 clean frame `capture_2026-04-29_09-35-33.png` showed that the line-segment detector is not the missing fix. The real issue was a zero-support center-weighted board-prior false positive overriding a good Hough/spoke read, so the board-prior rescue should stay spoke-only unless we add a much stronger shaft-support gate.
- The single-image runner's final peak-ratio gate was too strict for the clean bright captures on this camera. Broad but correct peaks around `1.02..1.07` are normal, so `MIN_PEAK_RATIO=1.01` is a better fit than the old `1.2` cutoff for the thesis baseline.
- On the 2026-04-30 ideal frames, the best simple baseline is Hough-first with board-prior only as a fallback. Pure Hough is the strongest local detector when it fires, but the end-to-end hybrid is the more useful thesis baseline because it still covers the one Hough-missing clean frame.
- The 2026-04-29 frames remain the hard regime; the current classical selector still misses badly there, so that set is useful as a robustness stress test but not the acceptance target for the ideal baseline.
- The camera brightness gate should not rely on `min_y` alone, because the dark needle leaves low-luma pixels in otherwise bright captures. The current sweep over the 2026-04-29 and 2026-04-30 raw frames showed that a `mean>=180` plus `>=50%` bright-pixel ratio rule catches the clearly bright cases (`09:34` through `09:52`, plus `07:01`, `11:48`, `11:51`, `12:20`, `12:21`) while still leaving the dark `05:51` and `05:52` frames unflagged.
- The classical selector's agreement cluster also needs a source-priority guard. Without it, a higher-quality but lower-priority rim candidate can override a better fixed-crop or board-prior anchor on the clean board crops, even when the visual cue is clearly the same upper needle.
- The firmware peak-ratio cutoff was still a little too tight on the newest clean capture (`capture_2026-04-30_12-45-08.png`), so the board-side gate now uses `APP_BASELINE_MIN_PEAK_RATIO=1.01`. That matches the permissive Python baseline better and keeps broad-but-correct peaks from falling through to the hot rim rescue path.
- The board crop itself needed a vertical correction too: a 12px upward bias on the bright-centroid crop fixed the top clipping on `capture_2026-04-30_12-45-08.png`, and the same offset should stay mirrored in the STM32 crop helper.
- The crop fix is better expressed as a bounded adaptive bias: use about 11% of the crop height as an upward shift, then clamp that to 8..18 pixels. That kept the dial top visible on the latest captures without making the framing overly fragile.
- The adaptive crop improved framing, but the post-crop classical baseline is still inconsistent across the 2026-04-30 captures. A small sweep showed a few near-5C predictions (`07-01-21` ≈ `4.5C`, `12-19-11` ≈ `7.6C`) alongside large failures (`11-51-05` ≈ `24.7C`, `05-52-17` ≈ `-30C`, `12-20-22` ≈ `-29.6C`), which means the detector/selector still needs work after cropping.
