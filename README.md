# Embedded Gauge Reading Using TinyML

## Overview
This project prototypes a robust embedded vision system that infers readings from analog industrial gauges across varied environmental conditions using an on-device camera, an STM32 Nucleo board with an NPU, and a compact Convolutional Neural Network (CNN).

Target constraints for a deployable, low-cost prototype (under $200 USD total):
- Offline inference (no Internet required)
- Power budget target: 1–2 W average under periodic inference
- Inference latency target: < 100 ms average on-device

This enables remote deployments (rural or offshore) using solar power, creating value without retrofitting legacy equipment or relying on expensive backhaul connectivity.

## Project Demo

[![Project Demo Video](https://img.youtube.com/vi/EDLs6GXLhhM/maxresdefault.jpg)](https://youtu.be/EDLs6GXLhhM)

The video shows the STM32 N6 prototype capturing a gauge image, running on-device CNN inference, and producing a temperature estimate without relying on a PC or cloud service.

**Current prototype result:**
- CNN inference is running on STM32 N6 hardware.
- A controlled 11-point test achieved approximately 2.4 °C MAE for the CNN compared with approximately 10.9 °C MAE for the classical CV baseline.
- Next measurements: NPU latency, CPU-vs-NPU comparison, and power draw using INA219.

Watch the full demo: [https://youtu.be/EDLs6GXLhhM](https://youtu.be/EDLs6GXLhhM)

## Problem
Many industrial sites still rely on analog gauges and legacy indicators that are not digitally instrumented. Manual readings are costly, slow, error-prone, and sometimes unsafe.  
This project aims to deliver affordable, low-power, on-device gauge digitization with reliable outputs and clear “no-read” behavior when confidence is low.

## Why it matters
- Safety and access: reduces human exposure in hazardous or hard-to-reach environments
- Cost: avoids expensive retrofits and reduces OPEX from manual rounds
- Reliability: enables continuous monitoring and alarming where connectivity is limited or expensive
- Practicality: minimal infrastructure change, deployable with commodity parts

## What I’m building
A deployable TinyML pipeline for gauge reading on microcontrollers:
1. Data: a labeled dataset of gauge images captured in realistic conditions (glare, distance, angle, dirt, low light)
2. Model: a compact CNN for gauge type detection plus needle angle or value estimation
3. Deployment: int8 quantized inference on STM32 NPU hardware with measured latency, memory, power, and accuracy
4. System: camera → on-device inference → confidence checks / no-read → local logging, optional wireless telemetry

## Risks identified
- Primary risk: domain shift and image quality variability (lighting changes, glare, reflections, oblique viewing angles, blur, dirt, distance), which can degrade model accuracy.
- Mitigation plan: dataset design that captures realistic conditions, targeted data augmentation, and an explicit “no-read” policy (the model refuses to output a reading when confidence is low or the image is out-of-distribution).

## Objective for my thesis-based master's
I want to pursue a thesis in embedded computer vision and TinyML, building on this prototype and leveraging university lab resources to:
- Improve robustness under domain shift (new gauge types, lighting, grime, oblique angles, occlusion)
- Quantify reliability (uncertainty, calibration, failure detection, confidence thresholds)
- Optimize and benchmark on-device performance (int8 quantization, pruning, NPU acceleration, memory scheduling)
- Validate end-to-end behavior with reproducible protocols and field-like deployments

## Status and Links
For the latest updates, check the project timeline at: https://github.com/users/rishi-latchmepersad/projects/1

## ML Workflow

The ML work is WSL-first and documented in [ml/README.md](ml/README.md).
Use that path for the classical baseline, CNN training, and future export/evaluation runs.

## Flashing the STM32N657 (boot from external flash)

The board boots from xSPI2 external flash via a two-stage process: ROM → FSBL → Application.
Both images must be signed with the ST signing tool before flashing.

### Prerequisites

- **STM32CubeProgrammer v2.21+** installed at the default path
- **STM32CubeIDE** to build both projects
- ST-Link connected via USB

### Jumper positions

There are two BOOT jumpers on the NUCLEO-N657X0-Q board. "Down" means the jumper is in the lower position (closer to the bottom edge of the board).

| Mode                            | BOOT0 | BOOT1  | When to use                                                           |
|---------------------------------|-------|--------|-----------------------------------------------------------------------|
| **Flash boot** (run from flash) | down  | down   | Normal operation — board loads firmware from xSPI2 flash              |
| **Dev / programming**           | down  | **up** | Flashing new firmware — move BOOT1 up before running the flash script |

> The BOOT pins are sampled only at power-on, so any jumper change requires a full power-cycle (not just a reset).

### Build

1. Open `firmware/stm32/n657/FSBL/` in STM32CubeIDE → Build → produces `FSBL/Debug/n657_FSBL.bin`
2. Open `firmware/stm32/n657/Appli/` in STM32CubeIDE → Build → produces `Appli/Debug/n657_Appli.bin`

### Flash new firmware

1. Move the **BOOT1 jumper up** (dev / programming mode) and power-cycle the board
2. From `firmware/stm32/n657/` with ST-Link connected, run:

   ```bat
   flash_boot.bat
   ```

   `FLASH_APP=1` is set by default — signs and flashes both the FSBL and the application.
   Set `FLASH_MODEL=1` to also flash the neural network model blob.

3. Move the **BOOT1 jumper back down** (both BOOT0 and BOOT1 down = flash boot mode)
4. Press the **reset button** — the board will now boot from the freshly flashed firmware

### Expected serial output (LPUART1, 115200 8N1, TX=PE5)

```
[FSBL] === FSBL started ===
...
[FSBL] Jumping to app: SP=0x34200000  Reset=0x3403xxxx
[BOOT] UART console initialized.
...
[AR] Calling App_ThreadX_Start().
```

### How it works

The STM32N657 ROM bootloader reads the signed FSBL from `0x70000000` in xSPI2 flash and executes it.
The FSBL initialises the MX25UM51245G flash chip into OctoSPI mode, copies the signed application
from `0x70100400` in flash to `0x34000400` in AXISRAM1, then jumps to it (LRUN — load and run).
