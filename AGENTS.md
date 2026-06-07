# AGENTS.md instructions for d:\Projects\embedded-gauge-reading-tinyml

<INSTRUCTIONS>
## Project goals
- Train a baseline CV model, a CNN and then a vision transformer to read analog gauges on low-power embedded hardware.
- Initial hardware will be an STM32 N6 NPU nucleo board.
- Data has been labelled in the CVAT for images 1.1 format, using CVAT, and is stored in the /ml/data/labelled directory.

## Expectations
- Prefer small, testable changes. Don't change code that you don't need to.
- Prefer modular files and folders over large monolithic files. Split firmware stages into separate .c/.h modules (e.g., `app_center_detector.c`, `app_baseline_runtime.c`) rather than adding thousands of lines to a single file.
- We will use Pytest for Python code and Unity for C code. 
- Explain your suggestions to me with code and I'll do the implementation. Teach me.
- All of our Python code should be typed.
- Each block of code should have a docstring or comment explaining what it does. Every few lines should have inline comments explaining why we're doing the lines.
- We will use keras and tensorflow for ML.
- We will use TFLM for export, and STM32 Cube.AI for integration into the board.
- We will use STM32 Cube IDE extension for development of the C code.
- We will use STM32 Cube MX for development of the C BSP packages etc.
- Favor `src/` layout conventions and Poetry tooling.
- Write any important details for yourself in /docs/ai-memory.md

## Intended Folder Layout
- Keep Python and ML code under `ml/`, using `ml/src/` for importable code, `ml/scripts/` for runnable jobs, and `ml/tests/` for pytest coverage.
- Keep firmware and board integration under `firmware/`, with STM32CubeIDE project files staying inside the matching board/app subdirectory.
- Keep project docs, notes, and design writeups under `docs/`.
- Keep generated or scratch files under `d:/Projects/embedded-gauge-reading-tinyml/tmp/` only.
- Do not place temporary scripts, exports, or experiment outputs in the repo root, `docs/`, or `firmware/` unless they are meant to be checked in there permanently.
- If a new file does not clearly belong in `ml/`, `firmware/`, `docs/`, or `tmp/`, stop and choose the smallest existing home rather than inventing a new top-level folder.

## Commands
- The firmware build runs on **Windows via STM32CubeIDE only**.
  * Never modify build-system files: `makefile`, `makefile.targets`, `subdir.mk` (any subdir), or any file under `Debug/` — these are generated/managed by CubeIDE.
  * Never convert Windows paths (`C:/Users/...`) to WSL paths (`/mnt/c/...`) in firmware files. The firmware is never built under WSL.
  * Fix only `.c` and `.h` source files in `Appli/Src/` and `Appli/Inc/` for C bugs.
- Use `poetry` for env management and scripts.
- Prefer `pytest` for tests.
- Use WSL for ML work, with the GPU preferred.
- Prepare explicit WSL handoff scripts in `tmp/` for model jobs and let DeepSeek run those directly; keep the workflow script-driven instead of manual and stateful.
- Always run jobs in bash scripts inside WSL, and tail the logs so you can see when they hang or fail.
- Use the `d:/Projects/embedded-gauge-reading-tinyml/tmp/` directory for all temporary files and folders (e.g., `tmp_*`, `artifacts/tmp_*`). This replaces any `tmp/` or `tmp_*/` folders that were previously in the project root.

## Notes
</INSTRUCTIONS>
