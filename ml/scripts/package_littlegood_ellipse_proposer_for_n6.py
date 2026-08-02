#!/usr/bin/env python3
"""Package the 320x320 exact-int8 ellipse proposer for STM32N6."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import package_yolo_obb_320_for_n6 as generic  # noqa: E402

ORIGINAL_RUN = generic._run
MAKE_BIN = Path("/mnt/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615/tools/bin")
GCC_BIN = Path("/mnt/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin")


def run_with_cubeide_make(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run the NPU driver with CubeIDE's native build tools available."""
    if not any("npu_driver.py" in item for item in command):
        ORIGINAL_RUN(command, env=env)
        return
    for directory in (MAKE_BIN, GCC_BIN):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    shim = ROOT / "tmp" / "run_littlegood_proposer_npu_driver.cmd"
    shim.write_text(
        "@echo off\n"
        f"set \"PATH={generic._to_windows_path(MAKE_BIN)};{generic._to_windows_path(GCC_BIN)};%PATH%\"\n"
        f"\"{generic._to_windows_path(Path(command[0]))}\" \"{command[1]}\" %*\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    subprocess.run(["cmd.exe", "/d", "/c", generic._to_windows_path(shim), *command[2:]], check=True, env=env)


def main() -> None:
    """Generate and stage the proposer package without touching active packages."""
    name = "ocdet_ellipse_320_v2"
    package_name = "ocdet_ellipse_320_v2_int8_n6_npu"
    package = ROOT / "firmware" / "stm32" / "n657" / "st_ai_output" / "packages" / package_name
    generic.MODEL_NAME = name
    generic.PACKAGE_DIRNAME = package_name
    generic.TFLITE_MODEL = ROOT / "ml" / "artifacts" / name / "model_int8.tflite"
    generic.FIRMWARE_PACKAGE = package
    generic.STAI_OUTPUT_DIR = package / "st_ai_output"
    generic.STAI_WS_DIR = package / "st_ai_ws"
    generic.STAGING_BUILD = ROOT / "tmp" / f"stedgeai_{name}_build"
    generic.TEMP_WORKSPACE = ROOT / "tmp" / f"stedgeai_{name}_ws"
    generic.TEMP_OUTPUT = ROOT / "tmp" / f"stedgeai_{name}_out"
    generic.EXPECTED_XSPI2_RAW = generic.STAI_OUTPUT_DIR / f"{name}_atonbuf.xSPI2.raw"
    generic._run = run_with_cubeide_make
    generic.main()
    generated = generic.TEMP_WORKSPACE / f"neural_art__{name}"
    for filename in ("c_info.json", "network.csv"):
        source = generated / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, generic.STAI_OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
