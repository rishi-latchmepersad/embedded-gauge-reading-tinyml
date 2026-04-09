# AI Memory

This file is the stable memory for future AI agents working in this repo.
It should hold facts that matter across sessions, not the active task list.

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
- The deployment-ready TFLite model has been tested from `ml/artifacts/deployment/`.

## Runtime Layout

- `main.c` should stay a bootstrapper and startup coordinator.
- `app_threadx.c` should stay a thread orchestration layer.
- Feature logic should live in small modules with narrow APIs.
- Generated AI runtime code and vendor glue should stay behind wrappers.

Already split out:

- `ds3231_clock.*`
- `app_camera_buffers.*`
- `app_camera_platform.*`
- `app_storage.*`
- `app_inference_runtime.*`
- `app_camera_diagnostics.*`
- `app_camera_capture.*`
- `app_memory_budget.h`
- `app_camera_config.h`
- `app_threadx_config.h`
- `app_inference_log_config.h`
- `app_inference_log_utils.*`
- `threadx_utils.*`

## Memory Lessons We Learned

- Do not assume "unused SRAM" can be used for executable image space.
- The linker `ROM` region is separate from general-purpose RAM.
- The current board image is constrained more by executable `.text` and `.rodata` size than by total SRAM availability.
- Large AI runtime tables and generated kernels are the main ROM consumers.
- The capture path uses large frame buffers and snapshots, so memory ownership must stay explicit.
- The board has a secure/noncacheable memory story, so DMA/capture buffers must be placed deliberately.

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
- The camera middleware is not safe to enter from the ISP background thread and the probe/capture thread at the same time; pause the ISP background loop and also take the shared camera middleware mutex while probe and snapshot setup touch `CMW_CAMERA_*` / `ISP_*` state.
- A hardfault we saw in `ISP_Algo_Process()` / `_ISP_BackgroundProcess()` went away after serializing those camera middleware entry points with the mutex.
- The FileX thread should not hold the media mutex while draining the debug log queue; the SD log service already serializes its own file access and the outer lock can make capture writes look stalled.
- Intermittent DCMIPP error `0x00008100` decodes to `CSI_SYNC | CSI_DPHY_CTRL`, which points at the camera link/CSI side rather than the AI worker or FileX path.
- When `0x00008100` shows up after a full frame buffer has already been reported, the current capture path retries once because it often behaves like a late CSI/DPHY rearm hiccup rather than a hard failure.

## RTC Facts

- The DS3231 should not be silently overwritten on every boot.
- We previously removed the auto-write-from-build-time behavior.
- RTC timestamp generation is centralized in `ds3231_clock.*`.
- The old DS3231 implementation block has been removed from `main.c`; it only calls into `ds3231_clock.*` now.

## File / Module Responsibilities

### `main.c`

- Boot and system bring-up only.
- No camera, AI, or storage policy should live here.

### `app_threadx.c`

- Thread creation, startup ordering, and high-level orchestration.
- It should not own generic utility code if it can live elsewhere.
- Legacy camera capture/state helpers were removed from the active build; keep this file as orchestration only.

### `threadx_utils.*`

- Generic ThreadX helpers only.
- Delay conversion, mutex helpers, tick helpers, and byte-pool reporting belong here.

### `app_camera_buffers.*`

- Large persistent camera buffers and snapshot storage.

### `app_camera_platform.*`

- Low-level camera board support
- IMX335 chip-ID and reset helpers
- Camera enable/shutdown pin control
- Active DCMIPP handle selection
- IMX335 probe/init helpers and sensor register access
- DCMIPP arm/start helpers for snapshot capture
- IMX335 stream start sequencing
- Compatibility wrappers for camera-related tick helpers while the refactor is in progress
- This module now owns the low-level sensor probe and middleware init path that used to live in `app_threadx.c`.
- `app_threadx.c` should call this module for camera bring-up instead of implementing probe/init/register access inline.

### `app_storage.*`

- FileX media readiness coordination
- Storage-ready event signaling
- The event flag group and sync state live inside `app_storage.c`
- RTC-backed capture filename generation with FileX fallback

### `app_inference_runtime.*`

- AI worker thread
- Inference request queueing
- Inference log thread
- One-shot dry-run frame handling
- The AI/runtime boundary should stay behind this module

### `app_camera_diagnostics.*`

- Camera debug dumps, state snapshots, and capture diagnostics.

### `app_camera_capture.*`

- High-level camera capture flow.
- Capture-state snapshots and frame-acquisition helpers now live here.
- Frame save / SD handoff.
- Dry-run inference queueing after a successful processed capture.
- This module now owns the capture/save orchestration that used to live in `app_threadx.c`.
- The capture single-frame logic and capture-state logging are the active implementations now, not the old `CameraPlatform_*` copies.

### `ds3231_clock.*`

- RTC read/write, timestamp formatting, and RTC status behavior.

### `app_inference_log_utils.*`

- Small reusable inference log formatting helpers.

## Current Build / Tooling

- Use `poetry` for Python environment and scripts.
- Use `pytest` for Python tests.
- Use Unity for C tests when needed.
- Prefer WSL for ML work.
- Use STM32CubeIDE / CubeMX for board code and BSP work.
- Keep Python code typed.

## Memory / Build Gotchas

- Debug builds can overflow the `ROM` region even when RAM still looks available.
- Some log formatting choices are not cheap on embedded C builds.
- RTC filename logging can hide capture progress if it is too verbose, so keep the capture hot path breadcrumb-only unless we are actively debugging timestamping.
- Avoid holding the FileX media mutex around the SD debug-log drain; that can deadlock against the capture save path when both sides try to log and touch media at the same time.
- When isolating FileX/capture deadlocks, it is useful to disable the SD debug-log drain entirely so capture save can be tested without concurrent media writes from the log service.
- Keep the ISP background loop paused until the full capture/save/inference handoff is complete; resuming CMW/ISP too early can hardfault inside the ISP middleware.
- For hardfault isolation, it is valid to disable `CMW_CAMERA_Run()` entirely and keep the background ISP path off until the capture/save path is proven stable.
- A feature that feels "small" can still consume ROM through generated tables or verbose logging.
- The model package, runtime, and app binary must stay in sync.

## Current Refactor Direction

- Keep peeling `app_threadx.c` into smaller modules.
- Turn `main.c` and `app_threadx.c` into coordinators.
- Keep AI request/logging plumbing in `app_inference_runtime.*`.
- Split camera bring-up, capture, AI runtime, and storage into separate modules when they grow.
- Keep each refactor slice small enough to build and board-test on its own.

## Stable Working References

- Roadmap: `PLANS.md`
- Repo working rules: `AGENTS.md`

## Things To Preserve

- Boot still succeeds.
- Camera probe and capture still work.
- AI inference still produces a nonzero value on test images.
- FileX logging still writes inference rows.
- The build must stay small enough to fit the current linker layout until we intentionally expand it.
