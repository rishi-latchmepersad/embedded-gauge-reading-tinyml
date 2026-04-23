# AI Memory

This is the entry point for durable project memory.
Keep this file short, and put detailed notes in the topical files below.

## Current State

- The current board baseline is still the scalar reader path with firmware-side calibration.
- The long-term MobileNetV2 geometry, direction, and detector-first experiments are exploratory; none has beaten the scalar baseline on the pinned board split.
- For WSL jobs, restart before the run and shut WSL down again afterward.

## Topic Files

- [Foundation notes](ai-memory/foundation.md)
- [Workflow and WSL notes](ai-memory/workflow.md)
- [Firmware and board notes](ai-memory/firmware-board.md)
- [ML experiments and research notes](ai-memory/ml-experiments.md)
- [Legacy archive](ai-memory/archive.md)

## How To Use This

- Write new durable facts into the topical file that matches the area.
- Update this index when a new topic file is added.
- Use the archive only for older chronology or deep detail.
