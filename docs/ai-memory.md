# AI Memory

This is the entry point for durable project memory.
Keep this file short, and put detailed notes in the topical files below.

## Current State

- The current firmware board baseline is still the OBB + scalar path with firmware-side calibration, but the classical Hough baseline remains a useful comparator and the best board-probe benchmark so far has been the OBB + scalar cascade.
- The classical CV baseline now uses a fixed-crop Hough-style edge vote over the stable training crop, still emits a provisional warm-up reading from the first accepted frame, and prefers the stronger fallback heuristic when the stable training crop is absent.
- The classical baseline is now closer to a canonical Hough-line classical architecture, which makes it a better paper benchmark than the old ray scorer.
- A true 31C trace was the reason for this upgrade: the old baseline could drift into wrong plateaus, so the new Hough-style version is meant to be the more defensible classical comparator.
- The classical baseline now refuses to let weak Hough estimates poison the tiny smoothing history: low-confidence frames are held against the last stable reading, and a new seed must have a strong enough vote total before it can start the history.
- On a recent true 31C board trace, the Hough baseline reported about 30.4C while the current OBB+scalar prodv0.3 path reported about 27.7C, so the classical baseline is currently outperforming the learned path at that point.
- The DS3231 RTC now seeds itself from the current firmware build timestamp automatically when booting at year `2000`, then re-reads the clock.
- The one-off forced RTC seed has now been turned back off; the normal rule is back to year-`2000` boot seeding only.
- `prodv0.3` is the current firmware integration candidate for the OBB localizer, and the board project now builds cleanly with the OBB wrapper plus the shared scalar runtime bundle.
- The OBB hardfault cause is now pinned down: the per-frame ATON `LL_ATON_RT_Reset_Network()` call in `app_ai.c` was the trigger, so `prodv0.3` should stay in one-shot runtime mode by default and only re-enable the reset path behind an explicit test flag.
- FileX/SD bring-up was a separate issue, but the latest boot trace shows it now reaches ready end-to-end: CMD0 -> CMD8 -> ACMD41 -> CMD58 -> partition -> FileX media open -> test file -> capture directory ready.
- The SD bring-up path is now quiet by default: FileX only emits concise ready/error lines, the ACMD41 handshake still only requests HCS when CMD8 proves the card is v2, and the storage wait path no longer prints periodic breadcrumbs.
- The watchdog heartbeat thread now prints a single `[WATCHDOG] pulse` line per cycle again, so UART liveness stays visible without reintroducing the old startup chatter.
- The latest boot trace showed the storage path finally succeeding end-to-end: CMD0 -> CMD8 -> ACMD41 -> CMD58 -> partition -> FileX media open -> test file -> capture directory ready, so the FileX/media readiness issue is no longer the active blocker.
- The DS3231 boot seeding is also working again: when the RTC comes up at year `2000`, the firmware seeds it from the build timestamp and the logger resumes with a sane date instead of the old impossible timestamps.
- The DS3231 override used for the one-off clock sync has been disabled again, so future boots should only reseed the RTC if it falls back to year `2000`.
- After storage came up, the next trace exposed a camera-pipeline error (`DCMIPP` `0x00008100` / `CSI_SYNC|CSI_DPHY_CTRL`) on the first retry, so the remaining live boot issue is now camera capture, not storage or AI.
- The latest live trace confirms the ATON fault is still gone with reset skipped, and `prodv0.3` can run OBB + scalar cleanly; the remaining boot issue on that trace is camera capture after storage comes up, plus the RTC/logging recovery path that still needs to stay sane.
- The OBB package originally pinned its CPU input arena at `0x34100000`, which overlapped the app's live `.bss` / heap / ThreadX footprint and was the likely source of the hard faults; the arena base was moved up to `0x34107000`, above `__bss_end__` / `_end = 0x34106b58`, and the build still passes with that safer placement.
- The scalar package had the same overlap bug and is now rebuilt against `0x34107000` as well, so both model stages are finally off the live app RAM window.
- The newlib heap is now capped below `0x34110000` in `sysmem.c` so libc allocations cannot grow into the AI arena even if a later runtime path uses `malloc`.
- The camera init thread stack was doubled to 16 KB after the brightness-gate path proved too deep for the old 8 KB budget, and the brightness gate no longer retries in place; it now nudges exposure for the next capture cycle and keeps the current frame moving.
- The classical baseline worker was also bumped from 8 KB to 16 KB after the Hough-style sweep and its logging looked like the next likely stack pressure point in the board freeze sequence.
- The debug console is now fail-fast under contention instead of spinning, so a lower-priority logger can finish without a higher-priority thread starving it; the heartbeat thread still avoids per-pulse UART logging so LED liveness is not tied to console throughput.
- The visible green LED is now the normal heartbeat, and red is reserved for fault state so a solid red LED is no longer just a startup indicator.
- The long camera-loop sleep, image-cleanup retry/period waits, and storage-readiness polling were moved off the ThreadX sleep/timer queue path and onto a cooperative delay helper to reduce the last timer-heavy background waits that could still wedge the board after the baseline log.
- While FileX is still not ready, the camera loop now retries sooner instead of waiting a full minute between capture attempts, so the board keeps visibly producing progress during bring-up and debug.
- The cooperative delay helper now sleeps one tick at a time instead of only yielding, because the pure-yield version starved the lower-priority heartbeat thread and broke the visible green pulse.
- The latest board hard fault traced into `HAL_TIM_IRQHandler()` from `TIM5_IRQHandler()`, with `BFAR=0x10`, which points to a bad TIM5 handle or a nearby overwrite rather than a model-stage bug; the IRQ now has a small guard so the board stays alive while we keep hunting the corruption source.
- The next hard fault moved into ThreadX timer service at `PC=0x34000948` inside `_tx_timer_interrupt` / `__tx_timer_no_time_slice`, which suggests `_tx_timer_current_ptr` was null or corrupted; the ThreadX timer assembly now self-heals null, out-of-range, or misaligned timer pointers by restoring the list pointer and skipping expiration work.
- The later freeze then moved into `_tx_timer_system_deactivate()` at `PC=0x34020130`, where the timer list-head pointer looked corrupted; that path now bails out early if the timer pointer wanders outside the timer list range instead of dereferencing garbage.
- The image cleanup worker is now deferred until after camera probe success to remove one source of startup contention while we debug the boot stall.
- The camera init thread is now the highest-priority app thread and no longer waits on the startup sleep before probing, so bring-up does not depend on the timer path before the probe starts.
- FileX was raised above the camera thread during startup so it can finish mounting the SD card before the first capture loop blocks on storage readiness.
- The capture path now logs when it begins waiting for FileX media readiness, and it prints a periodic wait breadcrumb so we can tell whether the board is really frozen or just waiting on storage.
- The hard capture-time storage gate turned out to be too aggressive during bring-up, so the capture path now skips SD saves when FileX is not ready instead of stalling the live camera / inference loop.
- The FileX thread startup blinks were removed again because they were an unnecessary blocking delay during boot and made the board look stuck during startup.
- The updated process diagram now uses per-step SVG thumbnails from `docs/process_diagrams_assets/`, so keep those assets in sync if the step text or flow changes again.
- The long-term MobileNetV2 geometry, direction, and detector-first experiments are exploratory; the OBB localizer is the first one that clearly improved the board-probe cascade.
- The latest `mobilenetv2_detector_geometry` run also missed badly: `test gauge_value_mae=24.2626` versus `baseline_mae_mean_predictor=20.1698`, so it is still not a usable reader.
- The geometry keypoint-only MobileNetV2 run also missed the baseline: `test gauge_value_mae=23.1730` versus `baseline_mae_mean_predictor=20.1698`, even though its keypoint MAE improved to `6.6727`.
- The uncertainty-aware geometry run was the least-bad of the new geometry variants, but it still only reached `test gauge_value_mae=18.9273` versus the baseline mean predictor at `20.1698`, so it is not board-ready yet.
- The new MobileNetV2 OBB localizer run trained cleanly on the labeled split and reached `val_mae=0.1435` and `test_mae=0.1786` on the OBB parameters. That makes it the strongest explicit localizer proxy so far, even though it is still a localization model rather than a reader.
- The OBB + scalar board-probe cascade using `mobilenetv2_obb_longterm` and the rectified scalar deployment reached `mean_abs_err=3.6617`, `max_abs_err=11.8603`, and `cases_over_5c=11` at `OBB_CROP_SCALE=1.20`. That is the best board-probe result so far and beats the rectifier chain.
- The firmware calibration was re-fit with `ml/scripts/calibrate_obb_scalar_firmware.py` using `mid_band_focus_18_42.csv` for fitting and `hard_cases_plus_board30_valid_with_new6.csv` plus `board_weak_focus.csv` for stress testing. The piecewise fit won on the aggregate hard-case test set (`affine_test_mae=15.6630`, `piecewise_test_mae=12.8421`) and the deployed firmware tail now uses that piecewise calibration.
- When the board probe holdout was added as an extra test, the new affine fit was still slightly better on the board probe (`test2_affine_mean_abs_err=5.1965` vs `test2_piecewise_mean_abs_err=5.5528`), so the current piecewise deployment is a hard-case-tuned tradeoff rather than a universal win.
- The compact geometry long-term localizer is not board-ready on the rectified probe set, and the cascade-localizer long-term run only modestly improved value MAE while the geometry branch stayed weak.
- The explicit MobileNetV2 geometry cascade-localizer long-term run is better than the compact proxy, but still not board-ready on the rectified probe set.
- The latest cascade eval with that explicit localizer reached `mean_final_abs_err=13.2531` on 39 board-probe samples. That is an improvement over the compact cascade (`14.5682`), but still too far from the target.
- The rectifier + scalar chain using `mobilenetv2_rectifier_zoom_aug_v4` on the board probe set reached `mean_abs_err=12.4529` and `max_abs_err=27.3887`.
- The same board-probe eval with `mobilenetv2_rectifier_hardcase_finetune_v3` was better at `mean_abs_err=9.8036` and is now the best rectifier + scalar result on that probe set, but it still is not board-ready.
- The exported int8 rectifier `mobilenetv2_rectifier_hardcase_finetune_v3_int8` improved the board-probe chain further. With `RECTIFIER_CROP_SCALE=1.80`, the rectifier + scalar chain reached `mean_abs_err=6.1574` and `max_abs_err=21.2753`, which is still the best rectifier-based board result so far, but it is now behind the OBB cascade.
- The next live check should move toward an even more explicit localizer or detector/OBB target rather than another small refinement of the same geometry stack.
- The OBB long-term experiment should stay on the labeled dataset split with `val_fraction` and `test_fraction`; do not feed it the board manifest hard-case path.
- For WSL jobs, restart before the run and shut WSL down again afterward.
- The `docs/process_diagrams.drawio` file now reflects the current OBB + scalar cascade and the current Hough-style classical baseline, so it should be kept in sync with future runtime changes.

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
