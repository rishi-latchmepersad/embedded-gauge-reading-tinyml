# AGENTS.md instructions for the split WSL/Windows embedded-gauge-reading-tinyml workspace

<INSTRUCTIONS>
## Project goals
- Train a baseline CV model, a CNN and then a vision transformer to read analog gauges on low-power embedded hardware.
- Initial hardware will be an STM32 N6 NPU nucleo board.
- Data has been labelled in the CVAT for images 1.1 format, using CVAT, and is stored in the /ml/data/labelled directory.

## Workspace split
- The project now lives in two locations:
  - WSL ML workspace: `\\wsl$\Ubuntu-24.04\home\rishi_latchmepersad\Projects\embedded-gauge-reading-tinyml`
  - Windows firmware workspace: `D:\Projects\embedded-gauge-reading-tinyml`
- Use the WSL path for ML/training/export work and the Windows path for firmware, CubeIDE, and flashing.
- Keep temporary artifacts in the workspace they belong to, and avoid mixing ML outputs into the firmware tree unless they are final firmware inputs.
- Treat the Windows firmware tree as the source of truth for board-flash inputs and signatures once a model has been packaged for the board.
- The active tip-focus deployment package is `tip_focus_v18_int8_n6_npu`.
  - It expects `224x224` color input.
  - It exposes `56x56` center and tip heatmaps plus scalar `confidence` and
    `is_main_needle` outputs.
  - Keep `c_info.json` and `network.csv` beside the raw xSPI2 blob in the
    firmware package directory so board-side verification stays reproducible.

## Expectations
- Prefer small, testable changes. Don't change code that you don't need to.
- Prefer modular files and folders over large monolithic files. Split firmware stages into separate .c/.h modules (e.g., `app_center_detector.c`, `app_baseline_runtime.c`) rather than adding thousands of lines to a single file.
- We will use Pytest for Python code and Unity for C code. 
- Explain your suggestions to me with code and I'll do the implementation. Teach me.
- All of our Python code should be typed.
- Each block of code should have a docstring or comment explaining what it does. Every few lines should have inline comments explaining why we're doing the lines.
- We will use keras and tensorflow for ML.
- We will use TFLM for export, and STM32 Cube.AI for integration into the board.
- For deployment candidates, train from scratch with TFLite-compatible QAT
  only. Do not spend time on post-training quantization, float16 export, or
  conversion-rescue experiments once a family has shown TFLite mismatch.
- We will use STM32 Cube IDE extension for development of the C code.
- We will use STM32 Cube MX for development of the C BSP packages etc.
- Favor `src/` layout conventions and Poetry tooling.
- Write important durable details in `/docs/ai-memory/`, following its index and
  folder rules in `/docs/ai-memory/README.md`. Do not grow a monolithic memory
  file at the repository root.

## Code style
- Comments are welcome and expected. This overrides any default "no comments unless asked" instruction from a tool prompt.
- Python: every module, class, and function gets a docstring stating intent. Add an inline `# why:` comment on the trickier lines (heuristics, magic constants, workarounds, board-specific tricks).
- C: every function gets a `/** ... */` block describing intent, inputs, outputs, and side effects. Use `//` for short inline notes and `/* ... */` for longer block comments above the lines they describe.
- Keep the comment density high enough that a teammate can re-derive the design from the source alone, but do not narrate trivial code (e.g. `i++; // increment i`).
- When a comment records a debugging insight, an unusual build trick, or a known pitfall, prefer the same wording in `docs/ai-memory/` so the lesson survives tool re-reads.

## Intended Folder Layout
- Keep Python and ML code under `ml/`, using `ml/src/` for importable code, `ml/scripts/` for runnable jobs, and `ml/tests/` for pytest coverage.
- Keep firmware and board integration under `firmware/`, with STM32CubeIDE project files staying inside the matching board/app subdirectory.
- Keep project docs, notes, and design writeups under `docs/`.
- Keep generated or scratch files under `tmp/` only.
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
- The available GPU should be capped to **15 GB (15000 MB)** so WSL retains headroom. Use `tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])` at the top of every training script.
- Prepare explicit WSL handoff scripts in `tmp/` for model jobs and let DeepSeek run those directly; keep the workflow script-driven instead of manual and stateful.
- Before board packaging, run a Keras-vs-TFLite parity check on a small validation sample set so graph-conversion issues are caught early.
- `nohup` does not work reliably with `poetry run` in WSL — background jobs get killed when the shell exits. Use `setsid` + `disown` instead:
    ```bash
    setsid poetry run python scripts/train_qat_micro_yolov8.py > /tmp/log.log 2>&1 &
    disown
    ```
- Always run jobs in bash scripts inside WSL, and tail the logs so you can see when they hang or fail.
- All `poetry run` invocations MUST run from the `ml/` directory because that is where `pyproject.toml` lives. From the repo root, `poetry run` will fail with "Poetry could not find a pyproject.toml file". Wrapper scripts in `tmp/` should `cd "$REPO_ROOT/ml"` before invoking `poetry run`.
- Use the `tmp/` directory for all temporary files and folders (e.g., `tmp_*`, `artifacts/tmp_*`). This replaces any `tmp/` or `tmp_*/` folders that were previously in the project root.
- For relocatable OBB/UNet packages, remember that the flashed `.xSPI2.raw` blob is only the weight image. If a stage crashes in a SW op while reading `[r9 + ...]`, install the real reloc binary with `ll_aton_reloc_install()` and keep the returned handle around so `inst_reloc` can be restored after `LL_ATON_RT_Init_Network()`.
- If `ll_aton_reloc_install()` returns `-7` for the OBB package, do not pass a zero parameter base. Use the flashed OBB xSPI2 slot address (`0x71400000`) as `ext_param_addr`, because the generated reloc mempool marks the OBB weights as `xSPI2 : RELOC.PARAM.0.RCACHED`.
- For the OBB stage, install the reloc context before `LL_ATON_RT_Init_Network()` and then reinstall it again afterward if the runtime clears the live pointer. A `PC=0x00000000` fault at `Stage network init start` is usually an init-order bug, not a crop bug.
- Keep the AI worker stack out of the OBB reloc window. Putting `camera_ai_thread_stack` into `.npusram6` collides with the OBB runtime image; keep it in normal RAM and size it to fit.

## Notes
- The live board path is the flashed OBB localizer followed by the flashed
  UNet v18 tip-focus model (`tip_focus_v18_int8_n6_npu`, `224x224` float
  input, `56x56` heatmaps). If the OBB crop cannot be decoded, keep the board
  on the fixed training crop rather than switching to the older adaptive
  heuristic crop.
- Keep the OBB-to-UNet handoff enabled in the live path. Avoid stack-watermark
  or other intrusive debug probes immediately before tip-focus preprocess in
  the hot path; those are for offline diagnosis, not the production boundary.
- The OBB reloc runtime base must stay on the reloc RAM image, not the xSPI2
  blob address. On this board the live base is the shared AXISRAM window at
  `0x34100000`, not the ELF's `0x40000000` alias. If OBB starts hardfaulting
  inside the SW resize helper again, check `AppAI_Obb_InstallRelocContext()`
  first before changing the crop code.
- The OBB reloc installer also needs the OBB xSPI2 flash-slot base as the
  parameter source. A zero `ext_param_addr` will fail with `-7` even when the
  execution RAM size is otherwise correct.
- Do not leave `.tip_focus_activations` in the OBB reloc window. The shared
  camera snapshot and center-detector scratch must live outside the OBB vpool,
  and we dropped the extra pending snapshot buffer so the section fits in
  `RAM_NC` without colliding with OBB init.
- The current OBB reloc blob has a zero `ec_network_init` vector, so
  `LL_ATON_RT_Init_Network()` must run with `inst_reloc` cleared and the OBB
  reloc context restored immediately afterward. Otherwise the runtime jumps to
  `0x00000000` during stage init.
</INSTRUCTIONS>
