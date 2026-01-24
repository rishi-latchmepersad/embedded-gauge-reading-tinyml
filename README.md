# Embedded Gauge Reading Using TinyML

## Overview 
Embedded TinyML gauge reading on STM32 NPU hardware, including camera capture, compact CNN baselines, int8 quantization, NPU accelerated inference benchmarks, and an end to end prototype for reliable industrial analog gauge digitization.

## Problem
Industrial sites still rely on analog gauges and legacy indicators that are not digitally instrumented. Reading them manually is slow, error-prone, and sometimes unsafe. I am building a low-power embedded vision system that can read common industrial gauges locally and produce reliable digital measurements.

## Why it matters
- Safety and access: reduces human exposure in hazardous or hard-to-reach environments
- Cost: avoids expensive retrofits and downtime to digitize old equipment
- Reliability: enables continuous monitoring where connectivity is limited or intermittent
- Practicality: delivers value with minimal infrastructure changes

## What I’m building
A deployable TinyML pipeline for gauge reading on microcontrollers:
1. Data: a labeled dataset of gauge images captured in realistic conditions (glare, angle, dirt, low light)
2. Model: a compact CNN for gauge type detection and needle angle or value estimation
3. Deployment: quantized inference on STM32 NPU hardware with measured latency, memory, and power
4. System: camera → inference → confidence checks → logging and telemetry

## Current status (end of January 2026)
- Repo scaffolding, roadmap, and documentation structure in progress
- Next objective: Dataset v0 and a baseline model that produces measurable results
- Target hardware: STM32 NPU board + camera module

## What I want to do in your lab
I want to pursue a thesis on embedded computer vision and TinyML, focused on robust gauge reading under real-world constraints:
- Improve robustness across domain shift (new gauges, lighting, grime, oblique angles, occlusion)
- Quantify reliability (uncertainty, calibration, failure detection, confidence thresholds)
- Optimize on-device performance (int8 quantization, pruning, NPU acceleration, memory scheduling)
- Validate end to end behavior with reproducible benchmarks and real deployments

## Links
- Roadmap and milestones: docs/roadmap.md
- Results log and benchmark table: docs/results.md
- Demos: docs/demos.md
- Dataset notes: docs/dataset.md

---

# Milestones (February 2026 to June 2027)

This is a comfortable, thesis-oriented timeline with planned breaks for supervisor outreach, October applications, and Christmas.

## February 2026: M0, Portfolio front door
Deliverables
- README completed (this page)
- docs/roadmap.md, docs/results.md, docs/demos.md created
- Clear folder structure for ml/, firmware/, docs/

Done when
- A new reader understands the problem, approach, and plan in under 2 minutes
- Results log exists, even if it only contains placeholders

## March 2026: M1, Dataset v0 and labeling workflow
Deliverables
- Dataset v0 defined: gauge types, label format, train/val/test split rules
- First labeled batch completed (target: 200–500 images, even if imperfect)
- A simple dataset loader and sanity-check scripts

Done when
- You can regenerate splits and reproduce labels format consistently
- You can train a toy model without manual fiddling

## April 2026: M2, Baseline model v0 on laptop
Deliverables
- Baseline model trained (simple CNN or transfer learning baseline)
- Metrics chosen and recorded (example: MAE for needle angle, accuracy within tolerance for value)
- Initial failure modes documented

Done when
- A benchmark row exists in docs/results.md
- You can point to 3–5 concrete failure modes with example images

## May 2026: M3, Quantization and deployment feasibility
Deliverables
- TFLite int8 quantized model produced, accuracy delta measured
- Memory and latency profiling on laptop and estimated MCU footprint
- Deployment plan for STM32 NPU toolchain, inputs, pre-processing

Done when
- You have a clear go/no-go statement based on size, RAM, and expected latency
- You know what must change to fit on-device if it does not fit yet

## June 2026: M4, On-device bring-up path
Deliverables
- Firmware skeleton: camera capture path and frame pipeline stub
- One end-to-end “dry run” where a frame buffer reaches an inference call site
- Documented build steps and environment notes

Done when
- Firmware builds reproducibly
- Camera frames can be captured or replayed from stored images

## July to August 2026: Supervisor outreach window (planned focus shift)
Primary objective
- Email supervisors with a tight pitch and live links to:
  - README one-pager
  - Roadmap
  - Results log with at least one real benchmark row
  - One short demo clip if available

Keep building, but at reduced scope
- Maintain small weekly progress so the repo stays alive
- Focus on clarity, reproducibility, and evidence

## September 2026: M5, On-device inference v1
Deliverables
- Quantized model runs on the STM32 NPU board
- Inference timing measured on-device
- Simple output reporting over UART or logging

Done when
- You can run inference repeatedly and record latency and memory usage
- A short demo exists (gif or short clip)

## October 2026: Applications break (intentional bandwidth reduction)
Primary objective
- Applications and supporting materials

Maintenance objective
- Only small, low-stress tasks:
  - documentation cleanup
  - one minor experiment if time permits
  - keep results log current

## November 2026: M6, Robustness improvements and controlled evaluation
Deliverables
- Improvement pass based on documented failure modes
- Controlled comparisons: baseline vs improved approach
- Expanded dataset v1 (target: 1,000–3,000 images total if feasible)

Done when
- You can defend why v1 is better with evidence, not vibes
- You have at least one ablation, what helped, what did not

## December 2026: Christmas break (planned)
Primary objective
- Rest

Optional light tasks
- Organize dataset, clean docs, backlog grooming
- No heavy engineering required

## January 2027: M7, End-to-end prototype v1
Deliverables
- Camera → pre-processing → inference → post-processing pipeline integrated
- Confidence checks and “no-read” behavior implemented
- Logging of predictions and confidence to SD or serial

Done when
- System produces outputs reliably over a sustained run
- You have a simple reliability summary from logs

## February 2027: M8, Power and performance characterization
Deliverables
- Latency, memory, and power measurements (as feasible with available tools)
- Performance report table updated
- Identify bottlenecks and next optimizations

Done when
- You can state realistic runtime and resource requirements
- You know the top 2 constraints limiting performance

## March 2027: M9, Reliability and failure mode hardening
Deliverables
- Add robustness tests: lighting variation, angle, blur, partial occlusion
- Improve calibration or uncertainty behavior
- Expanded “known failures” section with mitigations

Done when
- You have a clear, testable definition of when the model should refuse to answer
- Demo does not silently output nonsense in obvious edge cases

## April 2027: M10, Field-like evaluation
Deliverables
- Longer-run dataset collection or replay tests
- Drift and stability analysis from logs
- Documented evaluation protocol

Done when
- You have a repeatable evaluation run that takes the same inputs and produces comparable outputs
- You can discuss stability over time and conditions

## May 2027: M11, Thesis-ready documentation package
Deliverables
- Polished README, roadmap, and results report
- Reproducible instructions for training and deployment
- Clean demo set, with captions and failure notes

Done when
- A supervisor can browse the repo and understand what you did, how you measured it, and what remains open

## June 2027: M12, Final pre-departure consolidation
Deliverables
- “Snapshot release” tagged in GitHub
- Final benchmark tables and demo assets
- Short research summary document in docs/ (2–4 pages)

Done when
- The work is packaged as a coherent thesis launchpad
- You have a clean story for interviews, supervisors, and applications

---

# Cadence and sanity rules
- Weekly: at least one measurable update in docs/results.md
- Monthly: one milestone deliverable, even if small
- Always document failure modes, not just wins
- Breaks are part of the plan, not a failure
