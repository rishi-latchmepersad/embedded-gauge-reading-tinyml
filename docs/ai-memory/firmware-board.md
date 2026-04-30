# Firmware and Board Notes

This file holds board-specific runtime, deployment, and firmware rules.
See `archive.md` for the full chronology.

## xSPI2 / Flash Lessons

- `DUMMY_CYCLES_READ_OCTAL = 20U` is a recurring bug to treat as a canary.
- The rectifier flash address is permanently `0x70600000`.
- Binary/flash timing races should be checked by verifying timestamps before flashing.
- The rectifier stage now trusts the flashed blob and skips its signature gate, because the stale fingerprint was blocking a valid fallback image and contributing to the live freeze.
- The live AI cascade no longer enters the rectifier runtime by default; if the OBB crop is out of range or the OBB scalar pass fails, the board falls back to the fixed training crop instead so the capture loop keeps moving.
- The classical manifest-side Hough helper now rejects implausible circles and uses a `0.75x` effective radius, which is what keeps the hard-case and board-style weak-focus manifests under `10C` offline.

## Two-Stage Pipeline

- The current board path is a two-stage pipeline: rectifier first, scalar second.
- Keep the rectifier and scalar artifacts in sync with the firmware wrappers.
- The rectifier stage should stay small and predictable; the scalar stage should do the final readout.

## Deployment State

- The live board path remains the scalar reader, but the current A/B view bypasses firmware-side calibration so we can compare the raw model output directly.
- The current board candidate is `prod_model_v0.3_obb_int8`, wired through the OBB wrapper in `app_ai.c`.
- The live polar baseline now keeps the strongest raw angular peak instead of re-ranking the top bins with hub/width heuristics. That extra re-ranking was promoting unrelated fixed-crop peaks on the hard live traces, so the firmware is staying closer to the Python Hough-first reference.
- The classical firmware selector now defaults to a conservative fixed-crop-first branch with the local geometry sweep behind `APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP=0`. The sweep is still present for experiments, but the live board no longer trusts it by default after the hard live traces showed that nearby offset refinement could jump to the wrong plateau.
- The firmware build is currently green with the OBB wrapper linked against the shared scalar runtime bundle.
- The OBB package originally pointed its CPU input arena at `0x34100000`, which collided with the app's live RAM footprint in `.bss`, heap, and ThreadX globals; the arena base has been moved up to `0x34107000`, which sits above `_end = 0x34106b58` in the current link map.
- The scalar package had the same overlap issue, and the wrapper has now been rebuilt against `0x34107000` too, so both model stages are clear of the live app RAM window.
- `sysmem.c` now caps the newlib heap below `0x34110000` so future `malloc` activity cannot climb into the AI arena.
- The scalar output calibration helper is currently compiled with `APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION=0`, so the live board reports the raw model output while we compare it against the calibrated and smoothed paths.
- The camera init thread stack is now 16 KB instead of 8 KB, and the brightness-gate logic no longer retries the same capture in place. It now nudges IMX335 for the next cycle and keeps the current frame flowing, which is safer for the camera init thread. The active nudge is a 25% fractional exposure/gain step instead of a 2x jump, which should reduce the old bright/dark oscillation.
- The debug console is now fail-fast under contention instead of spinning, so a lower-priority logger can finish without a higher-priority thread starving it; the heartbeat thread still avoids per-pulse UART logging so LED liveness is not tied to console throughput.
- The visible green LED is now the normal heartbeat, and red is reserved for fault state so a solid red LED should mean a real fault instead of the startup indicator.
- The long camera-loop sleep, image-cleanup retry/period waits, and storage-readiness polling now use a cooperative delay helper instead of `tx_thread_sleep()` so the remaining long background waits do not keep exercising the timer queue while the freeze is under investigation.
- When FileX is not ready yet, the camera loop now uses a shorter retry cadence instead of a full 60-second pause, which keeps the debug build visibly active while storage is still coming up.
- The cooperative delay helper was adjusted to sleep one tick at a time instead of only yielding, because the yield-only version starved the lower-priority heartbeat thread and made the green LED appear dead.
- The detailed crop-box and calibration history is archived, but the active rule is to keep the deployed board path conservative and well logged.
- The deployed `prodv0.3` calibration tail now uses the board30 closer-camera piecewise spline from `scalar_full_finetune_from_best_board30_piecewise_calibrated`.
- That closer-camera spline matches the live cold-end reads much better than the older mid-band tail, even though the rectified board probe still leaves the affine line slightly ahead on the holdout.
- The stored calibration fit is still the affine p5 from `scalar_full_finetune_from_best_affine_calibrated_p5` with `scale=1.1630995` and `bias=0.7423046`, but the current board A/B view bypasses it by setting `APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION=0`.
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
- The classical baseline thread now has a 16 KB stack instead of 8 KB, because the polar spoke-voting sweep plus its logging looked like the next likely stack-pressure source during the freeze.
- The classical baseline worker now holds the last stable reading whenever a new polar vote is too weak, and it only lets a brand-new seed into the history if the absolute score is strong enough. That keeps low-confidence frames from dragging the median to nonsense values like the old `-19C` outliers.
- The hard-case detector-family sweep says the gradient-polar family is still the best pure classical baseline on the current focus set, so the firmware baseline now follows that detector family instead of the older shaft-biased heuristic.
- The classical baseline geometry selector now refines the bright, fixed-crop, and image-center seeds, prefers the candidates that clear the acceptance gate, and then keeps whichever refined geometry has the best blended peak-sharpness-plus-support score instead of letting the fixed crop win by default. Each candidate still gets a tiny local geometry refinement pass so the live geometry can slide a few pixels instead of staying locked to the first anchor.
- The classical baseline polar scorer now keeps the Sobel-edge/tangential vote but adds a smooth middle-shaft weight so the dark needle beats the surrounding dial artwork more reliably.
- The classical baseline now requires both SNR-like confidence and absolute vote support before any estimate can seed the smoothing history, and weak near-ties stay out of the history so a borderline frame cannot overwrite a stable reading.
- The classical baseline now also has a continuity-aware escape hatch for strong fixed-crop and image-center reads: if the peak ratio is only slightly soft but the temperature stays close to the last stable estimate, the firmware can still seed history instead of holding stale output.
- The board selector now has a `4C` agreement window too: if several refined geometry hypotheses cluster within that range, the firmware keeps the best candidate from the cluster instead of letting a lone high-score outlier win the frame.
- The live firmware source-priority order now keeps `fixed-crop-polar` ahead of `board-prior-polar`. That makes the stable crop win on the clean ideal captures we care about most, while still leaving the board prior available as a fallback on awkward framings.
- The latest 2026-04-30 board captures confirmed the needle is dark, not red, so the polar vote now emphasizes the middle shaft and does not give red-ish pixels an extra boost.
- The default STM32 selector now also applies the temperature-agreement consensus rule outside the experimental local sweep path, so a small cluster of agreeing geometry hypotheses can beat a lone fixed-crop outlier. Calibration is still disabled (`APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION=0`), so the live board is now comparing raw model output against the classical baseline independently.
- The current firmware baseline choice is the consensus-enabled fixed-crop-first selector with the dark-needle, middle-shaft polar vote. On the ideal controls it matches the laptop baseline well enough to be a defensible thesis baseline, while the new 2026-04-29 raw frames remain a robustness smoke test rather than the target regime.
- The live 2026-04-30 frames should be read from the upper dial. The small lower subdial is not the temperature target; it was the source of an earlier mix-up. The upper needle on those frames sits around 5C on the inner scale (roughly 40F on the outer scale), so baseline tuning should keep the upper needle as the target and treat the lower gauge as clutter.
- The gauge’s outer large numbers are Fahrenheit, while the inner numbers are Celsius. The firmware baseline should continue to map the needle position to the inner Celsius scale when we judge whether a reading is correct.
- On the 2026-04-24 live `14C` trace, the CNN output was the right answer and the classical baseline still missed the spoke, so the classical baseline is not the authoritative live read on that setup.
- On the 2026-04-24 boot trace, the same hold-last logic made the baseline look stale at about 18.4C. The underlying cause was capture quality, not a new baseline bug: the first camera attempt hit `DCMIPP` `CSI_SYNC|CSI_DPHY_CTRL`, the retry came through over-bright, and the polar vote stayed too close to the acceptance cliff to displace the existing history.
- The classical baseline confidence gate now sits around `1.25` because the detector reports a peak-vs-background ratio instead of the old tiny margin fraction, and a separate peak-ratio gate rejects near-tied peaks before they can seed history or report a live value.
- The fixed gauge crop is intentionally loose now (`0.60..1.40` relative to the stable crop), because the live close-up frames were still healthy enough to keep on the fast path and I did not want to send moderate crops into a slower fallback unnecessarily.
- FileX now runs above the camera thread during startup so it can finish mounting the SD card before the first capture call waits on media readiness.
- `AppStorage_WaitForMediaReady()` still logs when it starts waiting and on timeout, but it no longer prints periodic “still waiting” breadcrumbs during normal bring-up.
- The live capture path should not block on FileX readiness during bring-up; if media is not ready yet, skip the SD save for that cycle so camera and inference keep running.
- The FileX thread startup LED blinks were removed because they were a blocking startup delay that made the board appear frozen before storage came online.
- The FileX app thread stack was bumped from 2 KB to 8 KB after the state machine started looking like the next likely source of a silent freeze during SD/card bring-up.
- The watchdog heartbeat thread now prints a single `[WATCHDOG] pulse` line per cycle again, so UART liveness is visible without restoring the older FileX/SD spam.
- The classical board baseline now uses an explicit dial radius derived from the training crop height, not the old inscribed crop radius. That radius is still the right way to keep the polar vote aligned with the gauge ring.
- The firmware selector no longer gives the rim-center hypothesis an unconditional geometry win. The `-5C` live trace showed that bias forcing a warm false positive, so the board now compares accepted candidates by peak-sharpness quality first, the same way the Python classical helper does.
- The classical board baseline still adds a small rim-based center search ahead of the spoke vote, but it is now only one candidate family rather than a hard winner.
- The latest board-prior shaft-scan experiment is not yet board-ready: the color-aware, middle-shaft-weighted version still regressed `board_weak_focus.csv` badly, so do not promote it into the live firmware selector until it beats the current Hough-first path on the captured-image regression set.
- The selector score on the firmware side was too eager to reward extreme peak-ratio outliers, so it now treats peak ratio as a penalty (`confidence / peak_ratio`) and runs the narrow local refinement sweep by default. That gives the live board a chance to slide the seed geometry before it locks onto a hot false peak.
- The latest ideal-capture sweep suggests the board should keep the Hough seed in charge whenever it is confident, and treat the board-prior geometry as a fallback rather than a higher-priority override. That matches the cleaner 2026-04-30 captures much better than letting a board-prior candidate replace a good Hough read.
- The camera brightness gate was too aggressive for the newer processed frames. On the last two days of raw captures, the old `200/20` bright gate falsely flagged `capture_2026-04-30_11-51-05.yuv422` as too-bright, but the relaxed `230/100` gate leaves that frame as `ok` while still keeping the truly dim frames in `too-dark`.
- The brightness gate now also uses a bright-pixel ratio because the thin dark needle keeps `min_y` low on otherwise overexposed frames. The current firmware rule treats `mean>=180` plus at least `50%` of pixels above `220` as too bright, with the old `220/45` solid-overexposure fallback still catching the near-white `11:51` frame.
- The consensus step now respects source priority before peak-shape quality. That keeps a high-quality rim-center agreement cluster from overriding a better fixed-crop or board-prior anchor just because it is self-consistent.
- The latest clean `capture_2026-04-30_12-45-08` frame showed the firmware peak-ratio gate was still too strict for broad-but-correct peaks. Lowering `APP_BASELINE_MIN_PEAK_RATIO` to `1.01` keeps the correct fixed/board candidates alive so the hot rim rescue does not steal a frame that should stay near the inner Celsius needle.
- The board crop was also clipping the dial top. A 12px upward bias on the bright-centroid crop fixed the framing on the newest capture, so the live STM32 crop should keep that upward nudge instead of centering exactly on the bright centroid.
- The same crop fix is now bounded and adaptive in both Python and firmware: the upward bias scales with crop height at about 11%, but is clamped to 8..18 pixels so the crop can follow modest framing variation without dropping the dial top.
- The adaptive crop is an improvement in framing, but the baseline still needs detector work after the crop. On the 2026-04-30 sweep, some frames stayed close to the expected inner-Celsius reading, while others still jumped to obvious hot/cold outliers, so the board should treat the crop fix as necessary-but-not-sufficient.
