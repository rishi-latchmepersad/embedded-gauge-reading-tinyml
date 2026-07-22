"""Package gauge_ellipse_v1 through the local STM32N6 ST Edge AI flow.

This thin wrapper reuses the repository's tested relocatable packaging flow,
but points it at the new full-int8 grayscale ellipse detector and stages the
result under a model-specific firmware package directory.
"""

from __future__ import annotations

import sys
import subprocess
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import package_yolo_obb_320_for_n6 as generic_package


ORIGINAL_RUN = generic_package._run


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


def _run_with_cubeide_make(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run packaging commands, exposing CubeIDE's make.exe to npu_driver.py."""
    if not any("npu_driver.py" in item for item in command):
        ORIGINAL_RUN(command, env=env)
        return
    for tool_dir in (MAKE_BIN, GCC_BIN):
        if not tool_dir.is_dir():
            raise FileNotFoundError(f"STM32CubeIDE tool directory not found: {tool_dir}")

    # The Windows Python runtime sanitizes WSL PATH entries.  A temporary batch
    # shim preserves the native Windows PATH while forwarding npu_driver args.
    make_win = generic_package._to_windows_path(MAKE_BIN)
    gcc_win = generic_package._to_windows_path(GCC_BIN)
    python_win = generic_package._to_windows_path(Path(command[0]))
    driver_win = command[1]
    shim = REPO_ROOT / "tmp" / "run_gauge_ellipse_v1_npu_driver.cmd"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "@echo off\n"
        f"set \"PATH={make_win};{gcc_win};%PATH%\"\n"
        f"\"{python_win}\" \"{driver_win}\" %*\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    shim_win = generic_package._to_windows_path(shim)
    forwarded = ["cmd.exe", "/d", "/c", shim_win, *command[2:]]
    print(f"[PACKAGE] $ {' '.join(forwarded)}", flush=True)
    subprocess.run(forwarded, check=True, env=env)


def main() -> None:
    """Run ST Edge AI generation and N6 relocatable packaging for v1."""
    artifact_dir = REPO_ROOT / "ml" / "artifacts" / "gauge_ellipse_v1"
    package_dir = (
        REPO_ROOT
        / "firmware"
        / "stm32"
        / "n657"
        / "st_ai_output"
        / "packages"
        / "gauge_ellipse_v1_int8_n6_npu"
    )
    # Override only package identity and paths; generation flags stay aligned
    # with the existing YOLO/N6 wrapper used elsewhere in this checkout.
    generic_package.MODEL_NAME = "gauge_ellipse_v1_int8"
    generic_package.PACKAGE_DIRNAME = "gauge_ellipse_v1_int8_n6_npu"
    generic_package.TFLITE_MODEL = artifact_dir / "gauge_ellipse_v1_int8.tflite"
    generic_package.FIRMWARE_PACKAGE = package_dir
    generic_package.STAI_OUTPUT_DIR = package_dir / "st_ai_output"
    generic_package.STAI_WS_DIR = package_dir / "st_ai_ws"
    generic_package.STAGING_BUILD = REPO_ROOT / "tmp" / "stedgeai_gauge_ellipse_v1_build"
    generic_package.TEMP_WORKSPACE = REPO_ROOT / "tmp" / "stedgeai_gauge_ellipse_v1_ws"
    generic_package.TEMP_OUTPUT = REPO_ROOT / "tmp" / "stedgeai_gauge_ellipse_v1_out"
    generic_package.EXPECTED_XSPI2_RAW = (
        generic_package.STAI_OUTPUT_DIR / "gauge_ellipse_v1_int8_atonbuf.xSPI2.raw"
    )
    generic_package._run = _run_with_cubeide_make
    generic_package.main()

    # The shared helper looks for metadata under the final package workspace,
    # while ST Edge AI emits it under TEMP_WORKSPACE during generation.
    generated_workspace = (
        generic_package.TEMP_WORKSPACE
        / "neural_art__gauge_ellipse_v1_int8"
    )
    for filename in ("c_info.json", "network.csv"):
        source = generated_workspace / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing generated metadata: {source}")
        shutil.copy2(source, generic_package.STAI_OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
