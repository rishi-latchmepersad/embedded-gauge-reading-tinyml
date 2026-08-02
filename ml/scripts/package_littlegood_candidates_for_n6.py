#!/usr/bin/env python3
"""Package the validated LittleGood ellipse and center-tip models for STM32N6.

The script reuses the repository's established ST Edge AI/N6 relocatable flow
and stages each candidate in a new firmware package directory. Existing board
packages are deliberately left untouched until the generated artifacts pass
contract and hardware checks.
"""

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
MAKE_BIN = Path(
    "/mnt/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615/"
    "tools/bin"
)
GCC_BIN = Path(
    "/mnt/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/"
    "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/"
    "tools/bin"
)


def run_with_cubeide_make(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run packaging commands with CubeIDE's native make and GCC on PATH."""
    if not any("npu_driver.py" in item for item in command):
        ORIGINAL_RUN(command, env=env)
        return
    for directory in (MAKE_BIN, GCC_BIN):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    shim = ROOT / "tmp" / "run_littlegood_candidate_npu_driver.cmd"
    shim.write_text(
        "@echo off\n"
        f"set \"PATH={generic._to_windows_path(MAKE_BIN)};{generic._to_windows_path(GCC_BIN)};%PATH%\"\n"
        f"\"{generic._to_windows_path(Path(command[0]))}\" \"{command[1]}\" %*\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cmd.exe", "/d", "/c", generic._to_windows_path(shim), *command[2:]],
        check=True,
        env=env,
    )


def package_candidate(name: str, artifact: Path, package_name: str) -> None:
    """Generate one N6 relocatable package from an exact-int8 TFLite file."""
    package = ROOT / "firmware" / "stm32" / "n657" / "st_ai_output" / "packages" / package_name
    generic.MODEL_NAME = name
    generic.PACKAGE_DIRNAME = package_name
    generic.TFLITE_MODEL = artifact
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


def main() -> None:
    """Package both candidates without replacing the active firmware inputs."""
    package_candidate(
        "gauge_ellipse_littlegood_v2_gamma070",
        ROOT / "ml" / "artifacts" / "ocdet_refiner_256_littlegood_v2" / "model_int8.tflite",
        "gauge_ellipse_littlegood_v2_gamma070_int8_n6_npu",
    )
    package_candidate(
        "gauge_center_tip_littlegood_unet_v1",
        ROOT / "ml" / "artifacts" / "gauge_center_tip_littlegood_unet_v1" / "model_int8.tflite",
        "gauge_center_tip_littlegood_unet_v1_int8_n6_npu",
    )


if __name__ == "__main__":
    main()
