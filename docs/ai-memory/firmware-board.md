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
- The detailed crop-box and calibration history is archived, but the active rule is to keep the deployed board path conservative and well logged.
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

## SD Card / Storage

- SD card speed and storage readiness still matter for the camera and logging flows.
- Storage work should respect FileX synchronization and avoid holding the wrong mutex while draining logs.

## Board-Calibrated Readout

- Firmware-side calibration is the current board-safe correction layer for the scalar readout.
- The live board trace should log raw output, calibrated output, and delta so regressions are visible.
- A small burst smoother is acceptable if it stabilizes the reported board value without hiding raw failures.
