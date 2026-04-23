# AI Memory

This is the entry point for durable project memory.
Keep this file short, and put detailed notes in the topical files below.

## Current State

- The current firmware board baseline is still the scalar reader path with firmware-side calibration, but the best board-probe benchmark so far is now the OBB + scalar cascade.
- `prodv0.3` is the current firmware integration candidate for the OBB localizer, and the board project now builds cleanly with the OBB wrapper plus the shared scalar runtime bundle.
- The long-term MobileNetV2 geometry, direction, and detector-first experiments are exploratory; the OBB localizer is the first one that clearly improved the board-probe cascade.
- The latest `mobilenetv2_detector_geometry` run also missed badly: `test gauge_value_mae=24.2626` versus `baseline_mae_mean_predictor=20.1698`, so it is still not a usable reader.
- The geometry keypoint-only MobileNetV2 run also missed the baseline: `test gauge_value_mae=23.1730` versus `baseline_mae_mean_predictor=20.1698`, even though its keypoint MAE improved to `6.6727`.
- The uncertainty-aware geometry run was the least-bad of the new geometry variants, but it still only reached `test gauge_value_mae=18.9273` versus the baseline mean predictor at `20.1698`, so it is not board-ready yet.
- The new MobileNetV2 OBB localizer run trained cleanly on the labeled split and reached `val_mae=0.1435` and `test_mae=0.1786` on the OBB parameters. That makes it the strongest explicit localizer proxy so far, even though it is still a localization model rather than a reader.
- The OBB + scalar board-probe cascade using `mobilenetv2_obb_longterm` and the rectified scalar deployment reached `mean_abs_err=3.6617`, `max_abs_err=11.8603`, and `cases_over_5c=11` at `OBB_CROP_SCALE=1.20`. That is the best board-probe result so far and beats the rectifier chain.
- The compact geometry long-term localizer is not board-ready on the rectified probe set, and the cascade-localizer long-term run only modestly improved value MAE while the geometry branch stayed weak.
- The explicit MobileNetV2 geometry cascade-localizer long-term run is better than the compact proxy, but still not board-ready on the rectified probe set.
- The latest cascade eval with that explicit localizer reached `mean_final_abs_err=13.2531` on 39 board-probe samples. That is an improvement over the compact cascade (`14.5682`), but still too far from the target.
- The rectifier + scalar chain using `mobilenetv2_rectifier_zoom_aug_v4` on the board probe set reached `mean_abs_err=12.4529` and `max_abs_err=27.3887`.
- The same board-probe eval with `mobilenetv2_rectifier_hardcase_finetune_v3` was better at `mean_abs_err=9.8036` and is now the best rectifier + scalar result on that probe set, but it still is not board-ready.
- The exported int8 rectifier `mobilenetv2_rectifier_hardcase_finetune_v3_int8` improved the board-probe chain further. With `RECTIFIER_CROP_SCALE=1.80`, the rectifier + scalar chain reached `mean_abs_err=6.1574` and `max_abs_err=21.2753`, which is still the best rectifier-based board result so far, but it is now behind the OBB cascade.
- The next live check should move toward an even more explicit localizer or detector/OBB target rather than another small refinement of the same geometry stack.
- The OBB long-term experiment should stay on the labeled dataset split with `val_fraction` and `test_fraction`; do not feed it the board manifest hard-case path.
- For WSL jobs, restart before the run and shut WSL down again afterward.

## Topic Files

- [Foundation notes](ai-memory/foundation.md)
- [Workflow and WSL notes](ai-memory/workflow.md)
- [Firmware and board notes](ai-memory/firmware-board.md)
- [ML experiments and research notes](ai-memory/ml-experiments.md)
- [Legacy archive](ai-memory/archive.md)

## How To Use This

- Write new durable facts into the topical file that matches the area.
- Update this index when a new topic file is added.
- Use the archive only for older chronology or deep detail.
