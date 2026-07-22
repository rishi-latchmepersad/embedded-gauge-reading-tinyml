"""Package gauge_center_tip_v1 through the tested STM32N6 relocatable flow."""

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


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run npu_driver with CubeIDE make and GCC visible in Windows PATH."""
    if not any("npu_driver.py" in item for item in command):
        ORIGINAL_RUN(command, env=env)
        return
    for directory in (MAKE_BIN, GCC_BIN):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    shim = ROOT / "tmp" / "run_gauge_center_tip_v1_npu_driver.cmd"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "@echo off\n"
        f"set \"PATH={generic._to_windows_path(MAKE_BIN)};{generic._to_windows_path(GCC_BIN)};%PATH%\"\n"
        f"\"{generic._to_windows_path(Path(command[0]))}\" \"{command[1]}\" %*\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    subprocess.run(["cmd.exe", "/d", "/c", generic._to_windows_path(shim), *command[2:]], check=True, env=env)


def main() -> None:
    """Generate the N6 NPU blob, metadata, and C driver package."""
    artifact = ROOT / "ml" / "artifacts" / "gauge_center_tip_v1"
    package = ROOT / "firmware" / "stm32" / "n657" / "st_ai_output" / "packages" / "gauge_center_tip_v1_int8_n6_npu"
    generic.MODEL_NAME = "gauge_center_tip_v1_int8"
    generic.PACKAGE_DIRNAME = "gauge_center_tip_v1_int8_n6_npu"
    generic.TFLITE_MODEL = artifact / "gauge_center_tip_v1_int8.tflite"
    generic.FIRMWARE_PACKAGE = package
    generic.STAI_OUTPUT_DIR = package / "st_ai_output"
    generic.STAI_WS_DIR = package / "st_ai_ws"
    generic.STAGING_BUILD = ROOT / "tmp" / "stedgeai_gauge_center_tip_v1_build"
    generic.TEMP_WORKSPACE = ROOT / "tmp" / "stedgeai_gauge_center_tip_v1_ws"
    generic.TEMP_OUTPUT = ROOT / "tmp" / "stedgeai_gauge_center_tip_v1_out"
    generic.EXPECTED_XSPI2_RAW = generic.STAI_OUTPUT_DIR / "gauge_center_tip_v1_int8_atonbuf.xSPI2.raw"
    generic._run = _run
    generic.main()
    generated = generic.TEMP_WORKSPACE / "neural_art__gauge_center_tip_v1_int8"
    for filename in ("c_info.json", "network.csv"):
        shutil.copy2(generated / filename, generic.STAI_OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
