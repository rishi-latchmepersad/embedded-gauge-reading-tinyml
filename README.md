# Embedded Gauge Reading Using TinyML

## Overview 
Embedded TinyML gauge reading on STM32 NPU hardware, including camera capture, compact CNN baselines, int8 quantization, NPU accelerated inference benchmarks, and an end to end prototype for reliable industrial analog gauge digitization.

## Problem
Many industrial sites around the world still rely on analog gauges and legacy indicators that are not digitally instrumented. Reading them manually is expensive, slow, error-prone, and sometimes unsafe. 
I am building a low-power, affordable embedded vision system that can read common industrial gauges locally and produce reliable digital measurements quickly.

## Why it matters
- Safety and access: reduces human exposure in hazardous or hard-to-reach environments
- Cost: avoids expensive retrofits and downtime to digitize old equipment, and also reduces the OPEX required for human operators taking manual measurements
- Reliability: enables continuous monitoring and alarming where connectivity is limited, intermittent or expensive
- Practicality: delivers value with minimal infrastructure changes

## What I’m building
A deployable TinyML pipeline for gauge reading on microcontrollers:
1. Data: a labeled dataset of gauge images captured in realistic conditions (considering factors such as glare, distance, angle, dirt, low light etc.)
2. Model: a compact CNN for gauge type detection and needle angle or value estimation
3. Deployment: quantized inference on STM32 NPU hardware with measured latency, memory, and power and accuracy
4. System: On-board camera → on-board inference → confidence checks → on-board logging and optionally telemetry using a wireless link

## Current status (end of January 2026)
- Repo scaffolding, roadmap, and documentation structure in progress
- Next objective: Dataset v0 and a baseline model that produces measurable results
- Target hardware: STM32 N657X0 NPU board + B-CAMS-IMX camera module

## Objective for my thesis-based masters
I want to pursue a thesis on embedded computer vision and TinyML, focused on robust gauge reading under real-world constraints.
I plan to build upon the prototype that I have been developing, and will utilize the university lab to:
- Improve robustness across domain shift (handle new gauges and different gauge types, lighting, grime, oblique angles, occlusion etc.)
- Quantify reliability (uncertainty, calibration, failure detection, confidence thresholds)
- Optimize and quantify on-device performance (int8 quantization, pruning, NPU acceleration, memory scheduling)
- Validate end to end behavior with reproducible benchmarks and real deployments

## Links
- Roadmap and milestones: docs/roadmap.md
- Results log and benchmark table: docs/results.md
- Demos: docs/demos.md
- Dataset notes: docs/dataset.md
