# Foundation Notes

This file holds the stable, high-level project context.
See `archive.md` for the full chronology.

## Project Goal

- Build a gauge-reading pipeline for low-power embedded hardware.
- Start with a baseline CV model, then CNNs, then a vision transformer.
- Keep the firmware small enough to fit and run well on STM32 N6 hardware.

## Core Hardware

- Target board: STM32 N6 NPU Nucleo board.
- MCU family: STM32N657.
- Camera sensor: IMX335.
- RTC: DS3231.
- Storage: FileX on SD card.
- Model blob storage: external xSPI flash used by the board boot flow.

## Important Data Locations

- Labeled training data lives in `ml/data/labelled`.
- Captured images live in `captured_images/`.
- Board-generated model artifacts live in `st_ai_output/`.
- Deployment-ready TFLite and evaluation artifacts live under `ml/artifacts/`.

## Runtime Layout

- `main.c` should stay a bootstrapper and startup coordinator.
- `app_threadx.c` should stay a thread orchestration layer.
- Feature logic should live in small modules with narrow APIs.
- Generated AI runtime code and vendor glue should stay behind wrappers.
- The current board image is easier to maintain when model logic stays out of `main.c`.

## Memory Lessons We Learned

- Do not assume "unused SRAM" can be used for executable image space.
- The linker `ROM` region is separate from general-purpose RAM.
- The board image is constrained more by executable `.text` and `.rodata` size than by total SRAM availability.
- Large AI runtime tables and generated kernels are the main ROM consumers.
- Verbose AI bring-up and debug logging strings can also tip the `ROM` image over the limit.
- The capture path uses large frame buffers and snapshots, so memory ownership must stay explicit.
- The board has a secure/noncacheable memory story, so DMA and capture buffers must be placed deliberately.

## Secure Buffer / Capture Buffer Lesson

- The capture pipeline uses a large YUV422 frame buffer.
- We also keep a copied snapshot for dry-run inference.
- Those buffers must stay in the right RAM region for DMA and cache coherency.
- If a buffer is moved casually, the camera path can appear to work while inference or logging silently breaks.

## Current Camera / Inference Facts

- The capture pipeline is built around the IMX335 and the STM32 DCMIPP/CSI path.
- The board can run a live optical path, not just test-pattern input.
- The AI path consumes a 224x224 YUV422 capture path.
- The capture buffer size is 100,352 bytes for the current 224x224 YUV422 frame.
- The inference output is logged as a floating-point reading with one decimal place.
- Storage readiness is coordinated by `app_storage.*`, which owns the event flag group used to wait for FileX media.
- The camera middleware is not safe to enter from the ISP background thread and the probe/capture thread at the same time.
- A hardfault we saw in `ISP_Algo_Process()` / `_ISP_BackgroundProcess()` went away after serializing those camera middleware entry points with the mutex.
- The FileX thread should not hold the media mutex while draining the debug log queue.
- Intermittent DCMIPP error `0x00008100` decodes to `CSI_SYNC | CSI_DPHY_CTRL`, which points at the camera link/CSI side rather than the AI worker or FileX path.
- When `0x00008100` shows up after a full frame buffer has already been reported, the current capture path retries once because it often behaves like a late CSI/DPHY rearm hiccup rather than a hard failure.

## RTC Facts

- The DS3231 is the live time source for retention and cleanup decisions.
- If the live RTC reports year `2000` during boot, the firmware should seed it from the current build timestamp and then re-read it.
- If the live RTC reports year `2000`, cleanup logic should not delete anything.
- RTC failures should bias toward preserving data rather than pruning it.

## File / Module Responsibilities

### `main.c`

- Keep startup wiring and BSP bootstrap logic there.

### `app_threadx.c`

- Keep thread startup, priority, and orchestration logic there.

### `threadx_utils.*`

- Keep small ThreadX helpers there.

### `app_camera_buffers.*`

- Own camera buffer allocation and lifecycle.

### `app_camera_platform.*`

- Own the camera platform glue.

### `app_storage.*`

- Own FileX media readiness and storage synchronization.

### `app_inference_runtime.*`

- Own the AI runtime wrapper and stage transitions.

### `app_camera_diagnostics.*`

- Keep camera debugging and probe helpers there.

### `app_camera_capture.*`

- Keep capture sequencing and snapshot handling there.

### `ds3231_clock.*`

- Own the live RTC access helpers.

### `app_inference_log_utils.*`

- Own inference logging helpers and formatting.
