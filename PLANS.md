# Refactor Plan

This project is already working, so the goal now is to make it easier to grow.
The long-term shape we want is:

- `main.c` and `app_threadx.c` act as coordinators.
- Feature logic lives in small modules with clear ownership.
- Generated or vendor-specific code stays behind thin wrappers.
- Each slice is small enough to build and test on its own.

## Current Direction

We are not trying to rewrite the firmware in one shot.
We are peeling out the biggest files one module at a time so the codebase stays testable.

Already split out:

- `ds3231_clock.*`
- `app_camera_buffers.*`
- `app_camera_platform.*`
- `app_storage.*`
- `app_inference_runtime.*`
- `app_camera_diagnostics.*`
- `app_memory_budget.h`
- `app_camera_config.h`
- `app_threadx_config.h`
- `app_inference_log_config.h`
- `app_inference_log_utils.*`
- `threadx_utils.*`

## Refactor Rules

1. Keep changes small.
2. Move one concern at a time.
3. Preserve boot, capture, inference, and logging behavior after each step.
4. Make coordinator files thinner, not smarter.
5. Prefer modules with narrow APIs and few cross-dependencies.
6. If a change adds risk, stop after that slice and test.

## Target Architecture

### Coordinator Files

These should mostly schedule work and forward to modules:

- `main.c`
- `app_threadx.c`
- `app_azure_rtos.c`
- `app_ai.c`
- `app_filex.c`

### Feature Modules

These should own the actual logic:

- `ds3231_clock.*`
- `app_camera_buffers.*`
- `app_camera_diagnostics.*`
- `threadx_utils.*`
- `app_inference_log_utils.*`
- future `app_camera_platform.*`
- future `app_camera_capture.*`
- future `app_ai_runtime.*`
- future `app_storage.*`

## Module-by-Module Plan

### 1. `main.c`

Role:

- Boot setup
- Clock and peripheral bring-up
- App handoff

Next steps:

1. Move any remaining peripheral-specific helpers into their own modules.
2. Keep `main()` focused on initialization order only.
3. Avoid adding feature logic here.
4. Keep DS3231 startup behavior limited to the RTC module.

Done when:

- `main.c` only orchestrates startup.
- No camera, AI, or storage logic lives in `main.c`.

### 2. `app_threadx.c`

Role:

- Thread creation and scheduling
- Camera/inference loop coordination
- Error propagation

Next steps:

1. Keep peeling utility helpers into shared modules.
2. Move remaining camera bring-up helpers into a future `app_camera_platform.*`.
3. Move capture flow helpers into a future `app_camera_capture.*`.
4. Keep the file as the thread orchestration layer.

Done when:

- `app_threadx.c` mostly creates threads and calls module APIs.
- No large helper blocks remain inline.

### 3. `threadx_utils.*`

Role:

- Generic ThreadX helpers

Current ownership:

- Delay conversion
- Mutex helpers
- Tick helpers
- Byte pool reporting

Next steps:

1. Keep generic RTOS helpers here.
2. Move any other reusable ThreadX utilities here before creating new one-off helpers.

Done when:

- `app_threadx.c` no longer owns generic RTOS utility code.

### 4. `app_camera_buffers.*`

Role:

- Shared capture buffers and camera working memory

Next steps:

1. Keep all large camera buffers here.
2. Split any future temporary scratch buffers into this module first.
3. Avoid reintroducing large arrays into `app_threadx.c`.

Done when:

- Camera frame storage is centralized.
- `app_threadx.c` only references buffer APIs or externs.

### 5. `app_camera_diagnostics.*`

Role:

- Camera debug dumps
- State snapshots
- Capture diagnostics

Next steps:

1. Keep all formatted camera logging here.
2. Move any remaining register dump helpers here.
3. Keep output formatting consistent so logs stay readable.

Done when:

- `app_threadx.c` no longer owns camera diagnostic print logic.

### 6. `app_camera_platform.*` future module

Role:

- Camera bring-up
- Sensor register access
- Vendor/BSP integration

Next steps:

1. Create this module when the remaining camera helper blocks get too large.
2. Move IMX335 probe, init, stream start, and register access helpers here.
3. Keep `app_threadx.c` as the caller, not the owner, of sensor setup.

Done when:

- `app_threadx.c` only asks for camera actions instead of implementing them.

### 7. `app_camera_capture.*` future module

Role:

- Capture state machine
- Frame acquisition
- Capture retries and timeouts

Next steps:

1. Move capture loop logic out of `app_threadx.c`.
2. Keep the capture state machine separate from diagnostics.
3. Let this module own the detailed capture flow.

Done when:

- Camera capture can be read without digging through thread startup code.

### 8. `app_inference_runtime.*`

Role:

- AI inference invocation
- Model I/O handling
- Postprocessing of outputs

Next steps:

1. Move inference request and result handling out of `app_threadx.c`.
2. Keep the model boundary narrow.
3. Make it easy to swap models without rewriting orchestration code.

Done when:

- `app_threadx.c` only feeds frames into an AI API and receives a value back.

### 9. `app_filex.c` and future `app_storage.*`

Role:

- Storage readiness
- SD logging
- File creation and rotation

Next steps:

1. Keep `app_filex.c` focused on FileX setup and readiness.
2. Move CSV/log formatting into a storage/logging module if it grows.
3. Avoid mixing storage policy with camera policy.

Done when:

- FileX setup and log formatting are separated cleanly.

### 10. `ds3231_clock.*`

Role:

- RTC read/write
- Timestamp formatting
- RTC status reporting

Next steps:

1. Keep all RTC policy in this module.
2. Do not reintroduce auto-write-on-boot behavior unless it is an explicit toggle.
3. Add host-set or manual-set paths later if needed.

Done when:

- RTC behavior is controlled by one module only.

## Suggested Execution Order

When we continue, the safest order is:

1. Finish trimming any remaining generic helpers from `app_threadx.c`.
2. Move camera bring-up helpers into `app_camera_platform.*`.
3. Move the capture loop into `app_camera_capture.*`.
4. Move inference invocation into `app_inference_runtime.*`.
5. Split storage/logging policy if it still grows.
6. Revisit memory budgets and linker layout once the code is cleaner.

## Memory Growth Plan

The project is an AI project, so model growth is expected.
That means we should track two budgets separately:

- Executable/constant space
- Runtime RAM / pools / stacks

We should not assume all free SRAM can be used for the executable image.
If the current `ROM` window becomes the limiter again, the next question is whether to:

1. shrink the runtime footprint,
2. move constants or generated code to a different region,
3. or expand the linker layout intentionally.

## Acceptance Criteria For Each Slice

Every refactor slice should end with:

- a successful syntax check or build,
- a boot test on the board,
- capture still working,
- inference still producing a value,
- logging still writing rows when storage is available.

## Notes For Future Work

- Keep `app_threadx.c` as the coordinator.
- Keep `main.c` as the bootstrapper.
- Keep generated/vendor glue behind thin wrappers.
- If a module starts to feel like "everything camera-related," split again.
