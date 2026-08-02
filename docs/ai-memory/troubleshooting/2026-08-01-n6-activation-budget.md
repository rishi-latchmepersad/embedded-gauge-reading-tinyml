# N6 NPU activation budget: stride-2 needs HyperRAM, lean variant fits — 2026-08-01

Date: 2026-08-01
Status: current
Scope: STM32N657 NPU keypoint model deployment constraint
Evidence: GPT N6 packaging probe (`package_board_geometry_models_for_n6.py`),
`ml/scripts/screen_keypoint_archs.py::build_unet_stride2_lean`

## The constraint (GPT's N6 compiler probe)

The N6 NPU internal (no-HyperRAM) activation pool is **2,883,584 bytes
(2.75 MiB)**.  The stride-2 keypoint model at alpha=1.0
(`keypoint_unet_224g_stride2_s`, 112x112 output) needs **3,268,496 bytes
(3.12 MiB)**, including 980 KiB placed at 0x90000000 (HyperRAM).  With
HyperRAM removed, 1.40 MiB stays unallocated → the package is
**unbuildable for the board**.

Rule: **peak activation must be < 2.5 MiB (2,621,440 bytes) with margin
for the allocator**.  Weight size (xSPI2) is NOT the constraint — weights
live in flash; activations are the SRAM budget.

## What fits

| Architecture | Output | Calibrated peak | Params | Fits? |
|---|---|---|---|---|
| unet_stride2 alpha=1.0 (stride2_s) | 112x112 | 3.12 MiB (measured) | 1.02M | ❌ |
| unet_stride2 alpha=1.5 | 112x112 | ~3.5 MiB | 2.30M | ❌ |
| **unet_stride2_lean alpha=1.0** | 112x112 | **2.18 MiB** (calibrated) | 515K | ✅ |
| **unet_stride2_lean alpha=1.1** | 112x112 | **2.38 MiB** (calibrated) | 613K | ✅ |
| unet_s alpha=1.0 (56x56) | 56x56 | ~0.8 MiB | 1.02M | ✅ |

Calibration factor: local allocator simulation ~1.63x smaller than GPT's
N6 compiler measurement (verified on stride2_s: sim 2.01 MiB vs measured
3.12 MiB).  Use `sim * 1.63` as the safe estimate.

## The lean architecture

`build_unet_stride2_lean` keeps the 112x112x2 output (the tip-accuracy
win) but cuts the high-res decoder channels: e1 24ch (was 32), d3 32ch
(was 48), d4/head 24ch (was 32).  The 112x112 concat drops from 80ch
(~1.0 MiB) to 56ch (~0.7 MiB) — that concat was the dominant tensor.

## Decision

- Train `unet_stride2_lean` (alpha=1.1 if it fits, else 1.0) full-data
  with rotation augmentation as the board keypoint candidate.
- Package it with the N6 tool; verify the measured peak < 2.5 MiB before
  wiring into firmware.
- Keep the 56x56 alpha=1.0 model as the known-good fallback contract
  (matches the deployed tip_focus 56x56 contract).
