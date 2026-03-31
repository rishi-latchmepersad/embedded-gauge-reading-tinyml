# Embedded Gauge Reading Using TinyML

## Overview
I am prototyping a robust embedded vision system that infers readings from analog industrial gauges across varied environmental conditions using an on-device camera, an STM32 Nucleo board with an NPU, and a compact Convolutional Neural Network (CNN).

Target constraints for a deployable, low-cost prototype (under $200 USD total):
- Offline inference (no Internet required)
- Power budget target: 1–2 W average under periodic inference
- Inference latency target: < 100 ms average on-device

This enables remote deployments (rural or offshore) using solar power, creating value without retrofitting legacy equipment or relying on expensive backhaul connectivity.

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

## STM32 N657 Flashing

The N657 application target is built in `firmware/stm32/n657/Appli/Debug/n657_Appli.elf`.
CubeIDE's Debug launch uses ST-LINK over SWD with the `MX25UM51245G_STM32N6570-NUCLEO.stldr` external loader.
For this board, CubeProgrammer's normal SWD connect mode worked reliably from the command line.
If you just want a visible LED smoke test, the FSBL build already contains a blink loop and runs from RAM.

To flash from the command line on Windows, run:

```powershell
.\scripts\flash_n657.ps1
```

For the RAM-resident FSBL blink demo:

```powershell
.\scripts\run_fsbl_blink.ps1
```

For the one-command combined smoke test:

```powershell
.\scripts\run_n657_boot_chain.ps1
```

If you want the raw ST command, the script is wrapping the same CubeProgrammer flow:

```powershell
& "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" `
  -c port=SWD ap=1 mode=NORMAL `
  -el "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr" `
  -d "firmware\stm32\n657\Appli\Debug\n657_Appli.elf" -v -rst
```

And for the FSBL blink demo:

```powershell
& "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" `
  -c port=SWD mode=NORMAL `
  -d "firmware\stm32\n657\FSBL\Debug\n657_FSBL.elf" `
  -g 0x34180400
```
