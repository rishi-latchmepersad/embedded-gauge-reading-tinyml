# AGENTS.md instructions for d:\Projects\embedded-gauge-reading-tinyml

<INSTRUCTIONS>
## Project goals
- Train a CNN to read analog gauges on low-power embedded hardware.
- Initial hardware will be an STM32 N6 NPU nucleo board.
- Keep the ML pipeline reproducible and lightweight.
- Data has been labelled in the COCO format, using CVAT, and is stored in the /ml/data/labelled directory.

## Expectations
- Prefer small, testable changes.
- We will use Pytest for Python code and Unity for C code. 
- Explain your suggestions to me with code and I'll do the implementation. Teach me.
- All of our code should be typed.
- Each block of code should have a docstring or comment explaining what it does. Each line or few lines should have inline comments explaining why we're doing it.
- We will use keras and tensorflow for ML.
- We will use TFLM for export, and STM32 Cube.AI for integration into the board.
- We will use STM32 VSCode extension for development.
- We will use STM32 Cube MX for development.
- Favor `src/` layout conventions and Poetry tooling.
- Keep dependencies minimal and pinned when needed.

## Commands
- Use `poetry` for env management and scripts.
- Prefer `pytest` for tests.

## Notes
</INSTRUCTIONS>