# Firmware and Board Notes

This file holds board-specific runtime, deployment, and firmware rules.
See `archive.md` for the full chronology.

## xSPI2 / Flash Lessons

- `DUMMY_CYCLES_READ_OCTAL = 20U` is a recurring bug to treat as a canary.
- The rectifier flash address is permanently `0x70600000`.
- Binary/flash timing races should be checked by verifying timestamps before flashing.
- The rectifier stage may report a signature mismatch, but it is currently treated as non-blocking during bring-up.

## Two-Stage Pipeline

- The current board path is a two-stage pipeline: rectifier first, scalar second.
- Keep the rectifier and scalar artifacts in sync with the firmware wrappers.
- The rectifier stage should stay small and predictable; the scalar stage should do the final readout.

## Deployment State

- The live board path remains the scalar reader with firmware-side calibration.
- The current board candidate is `prod_model_v0.3_obb_int8`, wired through the OBB wrapper in `app_ai.c`.
- The firmware build is currently green with the OBB wrapper linked against the shared scalar runtime bundle.
- The OBB package originally pointed its CPU input arena at `0x34100000`, which collided with the app's live RAM footprint in `.bss`, heap, and ThreadX globals; the arena base has been moved up to `0x34107000`, which sits above `_end = 0x34106b58` in the current link map.
- The scalar package had the same overlap issue, and the wrapper has now been rebuilt against `0x34107000` too, so both model stages are clear of the live app RAM window.
- `sysmem.c` now caps the newlib heap below `0x34110000` so future `malloc` activity cannot climb into the AI arena.
- The camera init thread stack is now 16 KB instead of 8 KB, and the brightness-gate logic no longer retries the same capture in place. It now nudges IMX335 for the next cycle and keeps the current frame flowing, which is safer for the camera init thread.
- The debug console is now fail-fast under contention instead of spinning, so a lower-priority logger can finish without a higher-priority thread starving it; the heartbeat thread still avoids per-pulse UART logging so LED liveness is not tied to console throughput.
- The visible green LED is now the normal heartbeat, and red is reserved for fault state so a solid red LED should mean a real fault instead of the startup indicator.
- The long camera-loop sleep, image-cleanup retry/period waits, and storage-readiness polling now use a cooperative delay helper instead of `tx_thread_sleep()` so the remaining long background waits do not keep exercising the timer queue while the freeze is under investigation.
- When FileX is not ready yet, the camera loop now uses a shorter retry cadence instead of a full 60-second pause, which keeps the debug build visibly active while storage is still coming up.
- The cooperative delay helper was adjusted to sleep one tick at a time instead of only yielding, because the yield-only version starved the lower-priority heartbeat thread and made the green LED appear dead.
- The detailed crop-box and calibration history is archived, but the active rule is to keep the deployed board path conservative and well logged.
- The deployed `prodv0.3` calibration tail is now piecewise, fit from `mid_band_focus_18_42.csv` and selected after hard-case testing against `hard_cases_plus_board30_valid_with_new6.csv` and `board_weak_focus.csv`.
- The same fit still leaves the affine tail slightly ahead on the rectified board probe holdout, so the piecewise tail is a hard-case-tuned deployment choice rather than the absolute best board-probe metric.
- Model updates still need the generated code, include path, and makefile targets kept in sync.

## Model Update Process

- The board-facing model swap is only safe when the generated files agree on the current artifact set.
- Check the three-way sync between generated source, include path, and the makefile target before flashing.
- When verifying a new model, confirm the UART log and the artifact SHA, not just the file copy.

## Image Cleanup Worker

- The cleanup worker should keep every image from the last 24 hours for diagnostics.
- For files older than 24 hours, keep only the newest image in each 10-minute bucket.
- If the live RTC reads year `2000`, the cleanup worker should skip deletion entirely.
- If the RTC is unavailable, cleanup should bias toward preserving files rather than pruning them.

## RTC Boot Policy

- When the DS3231 boots at year `2000`, the firmware should seed it from the current build timestamp automatically.
- After seeding, re-read the RTC and use the refreshed value for filenames and retention decisions.

## SD Card / Storage

- SD card speed and storage readiness still matter for the camera and logging flows.
- Storage work should respect FileX synchronization and avoid holding the wrong mutex while draining logs.

## Board-Calibrated Readout

- Firmware-side calibration is the current board-safe correction layer for the scalar readout.
- The live board trace should log raw output, calibrated output, and delta so regressions are visible.
- A small burst smoother is acceptable if it stabilizes the reported board value without hiding raw failures.
- The AI worker stack was bumped to 32 KB after a freeze during the OBB -> scalar handoff, and `app_ai.c` now logs explicit markers around preprocess, cache clean, reset, and inference run so the next fault can be localized faster.
- If the board faults again, capture the full `[FAULT] HardFault ...` line, because the stacked PC/LR and CFSR bits should tell us whether the crash is in the NPU runtime, cache maintenance, or the stage transition glue.
- The latest hard fault resolved to `HAL_TIM_IRQHandler()` via `TIM5_IRQHandler()`, with `BFAR=0x10`, which strongly suggests a corrupted `htim5.Instance` or nearby memory overwrite rather than a model/runtime bug.
- `stm32n6xx_it.c` now guards the TIM5 ISR against a bad handle by clearing the update flag directly and incrementing the HAL tick instead of letting the board hard-fault inside the timer interrupt.
- The fault then moved into ThreadX timer service: `PC=0x34000948` resolved into `_tx_timer_interrupt` / `__tx_timer_no_time_slice`, and the null-dereference shape there points at `_tx_timer_current_ptr` being bad or zeroed.
- `tx_timer_interrupt.S` now guards null, out-of-range, and misaligned timer-pointer cases by restoring `_tx_timer_current_ptr` from `_tx_timer_list_start` and skipping expiration processing instead of faulting on `LDR r2, [r0, #0]`.
- The follow-up freeze landed in `_tx_timer_system_deactivate()` at `PC=0x34020130`, and the timer list-head pointer looked corrupted; that helper now returns early if the list head falls outside the timer list range instead of dereferencing a garbage pointer.
- The OBB hardfault later resolved to the ATON epoch runner path, so `app_ai.c` now skips the per-frame `LL_ATON_RT_Reset_Network()` call by default while we test whether one-shot network init avoids the runtime fault.
- That reset skip is now the confirmed fix for the hardfault path; if `prodv0.3` ever regresses here, check for the per-frame ATON reset first before blaming the model memory layout or the board stacks.
- FileX/SD bring-up is now reaching ready end-to-end on the latest trace: CMD0 -> CMD8 -> ACMD41 -> CMD58 -> partition -> FileX media open -> test file -> capture directory ready.
- The SD init path is now quiet by default: FileX only emits concise ready/error lines, `SPI_SendACMD41_UntilReady()` still only asserts HCS after CMD8 proves the card understands the v2 handshake, and the storage wait path no longer prints periodic breadcrumbs.
- The RTC seed-from-build-time path is also working again, so the logger now recovers from a year-`2000` boot with sane filenames instead of impossible dates.
- The RTC force-seed override used for the one-off clock sync has been disabled again, so normal boots should only reseed the DS3231 when it falls back to year `2000`.
- Once storage came up, the next issue surfaced as a camera capture error on the first retry (`DCMIPP` `0x00008100` / `CSI_SYNC|CSI_DPHY_CTRL`), so the remaining live boot problem is now in the capture pipeline rather than FileX.
- The latest board trace showed the OBB + scalar path running cleanly with `Stage network reset skipped (one-shot runtime)`, which confirms the ATON hardfault is fixed; the remaining boot pain on that trace is the camera capture retry path plus RTC/logging recovery, not the model runtime.
- To reduce startup interference while we debug the stall, the image cleanup worker now starts only after the camera probe succeeds instead of during `App_ThreadX_Start()`.
- The camera init thread now runs at the highest app priority and skips its startup sleep, so bring-up no longer depends on `tx_thread_sleep()` before the probe begins.
- The classical baseline thread now has a 16 KB stack instead of 8 KB, because the Hough-style sweep plus its logging looked like the next likely stack-pressure source during the freeze.
- The classical baseline worker now holds the last stable reading whenever a new Hough vote is too weak, and it only lets a brand-new seed into the history if the absolute score is strong enough. That keeps low-confidence frames from dragging the median to nonsense values like the old `-19C` outliers.
- FileX now runs above the camera thread during startup so it can finish mounting the SD card before the first capture call waits on media readiness.
- `AppStorage_WaitForMediaReady()` still logs when it starts waiting and on timeout, but it no longer prints periodic “still waiting” breadcrumbs during normal bring-up.
- The live capture path should not block on FileX readiness during bring-up; if media is not ready yet, skip the SD save for that cycle so camera and inference keep running.
- The FileX thread startup LED blinks were removed because they were a blocking startup delay that made the board appear frozen before storage came online.
- The FileX app thread stack was bumped from 2 KB to 8 KB after the state machine started looking like the next likely source of a silent freeze during SD/card bring-up.
- The watchdog heartbeat thread now prints a single `[WATCHDOG] pulse` line per cycle again, so UART liveness is visible without restoring the older FileX/SD spam.
