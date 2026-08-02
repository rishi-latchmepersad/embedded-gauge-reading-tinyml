# Board center/tip stage fed stale (non-camera) input — all-(-128) heatmaps — 2026-08-02

Date: 2026-08-02
Status: experimental
Scope: STM32N657 live ellipse -> center/tip pipeline (`keypoint_unet_224g_wide_aug_int8`)
Evidence: UART logs 2026-08-01 20:xx and 2026-08-02 06:xx sessions,
`tmp/board_2026-08-01/*.png`, `docs/ai-memory/model-updates/2026-08-01-live-board-validation.md`

## Symptoms

- Ellipse stage matches offline exactly (conf 0.996, qgeo/px/r reproduce the
  validated 2026-08-01 numbers) — the ellipse model and its wrapper are fine.
- Center/tip stage output is `raw=[-127,-128,...]` = quantized zero
  (scale 0.00390625, zero_point -128) on every frame; heatmap gate
  (floor 0.06) rejects all but one frame. The one pass (capture_30-32) read
  17.8C vs 12C ground truth.
- Same keypoint model on the same frames offline gives center/tip peaks
  0.68-0.76 and 11.0-11.5C (validated 2026-08-01).

## Smoking gun: the center/tip input signature is constant, not camera data

`[AI][CENTER_TIP] input sig ...` values are BIT-IDENTICAL across sessions,
frames, crops, and builds:

| capture (session 1, 2026-08-01 20:xx) | capture (session 2, 2026-08-02 06:xx) | sig (first/mid/last/min/max/sum) |
|---|---|---|
| 29-24 | 02-02 | -122/2/10/-128/51/-2826920 |
| 30-32 | 03-26 | -126/2/34/-128/66/-2816488 |
| 31-39 | 04-33 | -124/3/24/-128/54/-2903661 |

The crops differ between those pairs (e.g. crop side 370.7px vs 377.4px) and
the exposure/gain at save differed (gain 43931 vs 57303 mdB), so the input
cannot be the current camera frame. Offline recomputation of the firmware
crop from the same PNGs gives sum ≈ +418K..+898K (bright dial), not -2.8M.

## What the constant data looks like

The sigs are "mostly -128 with small peaks" — exactly the character of the
ELLIPSE stage's leftover output/activations at the shared AXISRAM5 pool
(0x342e0000). Both models' input AND the ellipse output share
`addr_base = 0x342e0000` (generated reloc.c buffer info). The keypoint
output sits at 0x342e0000+426496 = 0x34348200, inside the same pool the
ellipse activations use, so the printed `raw` output bytes are consistent
with ellipse leftovers too.

## Hypotheses (ranked; probe below settles it)

1. Center/tip fill loop writes never reach the buffer the sig/NPU read
   (alias or cache/coherency problem at 0x342e0000, or the fill is not the
   code being executed).
2. The keypoint NPU run does not write its output (epoch list/graph not
   scheduled; output buffer keeps ellipse leftovers).
3. The camera buffer the AI worker reads is not the saved frame (CMW stream
   keeps running; worker runs ~4s after save), and the worker's frame is
   some other constant content.

## RESOLVED 2026-08-02 — the ISP loop overwrites the capture buffer mid-inference

Diagnostic build with buffer-address prints, a frame probe, and SD dumps of
the exact center/tip input/output tensors (`diag_ct_in_*.bin`,
`diag_ct_out_*.bin`, first 8 frames) settled it:

- **NPU + decode chain is perfect**: the deployed TFLite model run offline on
  the board's exact dumped input reproduces the board's NPU output
  bit-for-bit (peaks 0.062/0.238 both sides).
- **The center/tip input is NOT the camera crop**: dumped tensors are a
  deterministic broken pattern — top ~308 source rows black, bottom a dim
  low-contrast 0.52 field, boundary fixed across 3 dumps. The correct crop
  recomputed from a real board gray8 is a bright dial (sum +4.07M vs board
  -2.85M). The frame probe (0.596 at the dial centre) proves the buffer
  holds the dial.
- **The buffer content changes between the ellipse fill and the center/tip
  fill** (~2s apart): the ellipse input has min=-109 (no black anywhere),
  the center/tip input min=-128 with a black band.
- Root cause: `AppCameraCapture_CaptureAndStoreSingleFrame()` cleanup reset
  `camera_capture_isp_loop_paused = false` immediately after queueing the
  AI worker. The ISP thread (`CMW_CAMERA_Run()`) then kept streaming
  processed frames into the shared capture buffer (0x24160000) while the
  worker was reading it, and the post-capture streamed frames are damaged
  (top rows never written; matches the recurring CSI_SYNC|CSI_DPHY_CTRL
  errors).
- Fix: keep `isp_loop_paused` set while
  `AppInferenceRuntime_IsInferenceInFlight()` is true, so the ISP only runs
  while a snapshot is armed. Rebuilt 2026-08-02 07:00 (0 errors).

## Update 2026-08-02 07:14 — ISP pause alone did NOT stop the overwrite

The 07:00 build (ISP-pause fix) still produced the same top-black band in
the dumped tensors. The gate-vs-save mismatch (gate read mean=52, saved
gray8 measures mean=170) proved the buffer is rewritten even during the
capture flow — the writer is inside the DCMIPP/CMW machinery, not gated by
the app's `isp_loop_paused` flag. The saved gray8 frames themselves are
bright and perfect (mean 151-170), so the captured frames are good.

## Definitive fix 2026-08-02 07:14 — restore the frozen snapshot copy

- `AppCameraBuffers_CopyCaptureToSnapshot()` (new, in app_camera_buffers.c):
  plain 32-bit word-loop copy of the fresh DMA frame into the existing
  `camera_inference_frame_snapshot` (409600 B, `.tip_focus_activations` at
  0x24024c00, app-side RAM — safe from the NPU pools). Word loop avoids the
  libc memcpy that HardFaulted on the old legacy-alias copy.
- `AppCameraCapture_CaptureAndStoreSingleFrame()` copies the frame to the
  snapshot BEFORE the slow SD save, then queues the AI worker with the
  snapshot pointer instead of the live DMA buffer.
- `app_inference_runtime.c` now logs "Queueing frozen snapshot frame."
- Keep the ISP-pause change (design intent; reduces streaming during the
  worker) plus the diag dumps for one more verification session.
- Rebuilt 2026-08-02 07:14 (0 errors).

Expected next session: ellipse and center/tip stages read the SAME frozen
frame; center/tip input shows a structured dark dial (no 308-row black
band); peaks ~0.6-0.7 (offline darkness test). Then disable
`APP_AI_DIAG_DUMP_CENTER_TIP_TENSORS`.

## Verification of the fix (next board session)

- Expect the `[AI][CENTER_TIP] input sig` to show a real dark dial crop
  (min still ~-128 at night, but structured rows, not the fixed 308-row
  black band) and heatmap peaks well above the 0.06 floor (the offline
  darkness test gave peaks 0.68-0.72 on a darkened real crop).
- Then set `APP_AI_DIAG_DUMP_CENTER_TIP_TENSORS` to 0 and re-flash.
- Camera secondary issues (exposure oscillation, WB 6650K lock status=81,
  DCMIPP 0x8100) remain open; night-time frames are very dark and may still
  read wrong until exposure is fixed.

## Next steps (one diagnostic build)

- Print all four I/O buffer addresses + sizes at pipeline entry.
- Dump the 50176-byte center/tip input tensor to SD
  (`AppFileX_WriteCapturedImage`) and compare offline with the firmware crop
  recomputed from the matching `.gray8`.
- Run the keypoint stage standalone on the fixed training crop (bypass the
  ellipse crop) to isolate NPU execution from the crop/fill path.
- Check the `.gray8` files offline with the deployed keypoint model: healthy
  peaks prove the camera frames are usable.

## Camera secondary issue (independent of the model bug)

Brightness gate oscillates (mean 14..231), DCMIPP error 0x00008100
(CSI_SYNC|CSI_DPHY_CTRL), WB 6650K lock fails (status=81). Most captured
frames are very dark (mean 14-98); the validated 2026-08-01 PNGs were better
exposed. Fixing the keypoint stage may still leave poor accuracy on dark
frames, so re-validate on well-exposed frames.

## Decision

Do not retrain or re-export the keypoint model yet. The model is proven good
offline on the exact board frames. Find the input-buffer/execution bug with
the probes above before touching the ML side.
