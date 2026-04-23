# AI Memory

This is the entry point for durable project memory.
Keep this file short, and put detailed notes in the topical files below.

## Current State

- The current board baseline is still the scalar reader path with firmware-side calibration.
- The long-term MobileNetV2 geometry, direction, and detector-first experiments are exploratory; none has beaten the scalar baseline on the pinned board split.
- The compact geometry long-term localizer is not board-ready on the rectified probe set, and the cascade-localizer long-term run only modestly improved value MAE while the geometry branch stayed weak.
- The explicit MobileNetV2 geometry cascade-localizer long-term run is better than the compact proxy, but still not board-ready on the rectified probe set.
- The latest cascade eval with that explicit localizer reached `mean_final_abs_err=13.2531` on 39 board-probe samples. That is an improvement over the compact cascade (`14.5682`), but still too far from the target.
- The rectifier + scalar chain using `mobilenetv2_rectifier_zoom_aug_v4` on the board probe set reached `mean_abs_err=12.4529` and `max_abs_err=27.3887`.
- The same board-probe eval with `mobilenetv2_rectifier_hardcase_finetune_v3` was better at `mean_abs_err=9.8036` and is now the best rectifier + scalar result on that probe set, but it still is not board-ready.
- The exported int8 rectifier `mobilenetv2_rectifier_hardcase_finetune_v3_int8` improved the board-probe chain further. With `RECTIFIER_CROP_SCALE=1.80`, the rectifier + scalar chain reached `mean_abs_err=6.1574` and `max_abs_err=21.2753`, which is the best rectifier-based board result so far.
- The next live check should move toward an even more explicit localizer or detector/OBB target rather than another small refinement of the same geometry stack.
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
