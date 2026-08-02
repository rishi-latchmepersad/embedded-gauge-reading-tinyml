"""Generate relocatable STM32N6 packages for the selected geometry models.

The training artifacts are TFLite files, while firmware needs the complete
ST Edge AI output set: generated C/H, relocatable metadata, memory pools, and
the xSPI2 weight image.  This wrapper keeps both models on the same tested
N6 packaging path and stages them under distinct package directories.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import json
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


def _no_hyperram_profile() -> Path:
    """Create the temporary ST Edge AI profile that forbids HyperRAM."""
    profile = ROOT / "tmp" / "neural_art_no_hyperram.json"
    pool = generic._to_windows_path(
        generic.PACK_ROOT
        / "scripts"
        / "N6_reloc"
        / "test"
        / "mpools"
        / "stm32n6_no_hyperram.mpool"
    )
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps({
        "Profiles": {
            "test-nohyper": {
                "memory_pool": pool,
                "options": "--native-float --mvei --cache-maintenance --Ocache-opt "
                "--enable-virtual-mem-pools --Os --optimization 3 --Oauto-sched "
                "--all-buffers-info --csv-file network.csv"
            }
        }
    }, indent=2), encoding="utf-8")
    return profile


def _run_with_cubeide_tools(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run the NPU driver with native CubeIDE make/GCC visible on PATH."""
    # The N6 board has no HyperRAM, so every geometry model must be allocated
    # against the internal-only pool before its generated files are accepted.
    adjusted = list(command)
    # Select the internal-only neural-art profile as well as the internal pool;
    # the default test profile allows HyperRAM during final compilation.
    is_keypoint = any("keypoint_unet" in item for item in adjusted)
    no_hyper_profile = _no_hyperram_profile() if is_keypoint else None
    adjusted = [
        f"test-nohyper@{generic._to_windows_path(no_hyper_profile)}"
        if is_keypoint and item.startswith("test@")
        else item
        for item in adjusted
    ]
    if "--memory-pool" in adjusted:
        pool_index = adjusted.index("--memory-pool") + 1
        if is_keypoint:
            adjusted[pool_index] = generic._to_windows_path(
                generic.PACK_ROOT
                / "scripts"
                / "N6_reloc"
                / "test"
                / "mpools"
                / "stm32n6_no_hyperram.mpool"
            )
    command = adjusted
    if not any("npu_driver.py" in item for item in command):
        ORIGINAL_RUN(command, env=env)
        return
    for tool_dir in (MAKE_BIN, GCC_BIN):
        if not tool_dir.is_dir():
            raise FileNotFoundError(f"CubeIDE tool directory not found: {tool_dir}")
    shim = ROOT / "tmp" / "run_geometry_npu_driver.cmd"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "@echo off\n"
        f"set \"PATH={generic._to_windows_path(MAKE_BIN)};{generic._to_windows_path(GCC_BIN)};%PATH%\"\n"
        f"\"{generic._to_windows_path(Path(command[0]))}\" \"{command[1]}\" %*\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    forwarded = ["cmd.exe", "/d", "/c", generic._to_windows_path(shim), *command[2:]]
    subprocess.run(forwarded, check=True, env=env)


def _package(model_name: str, package_name: str, artifact: Path) -> None:
    """Generate and stage one TFLite geometry model as an N6 package."""
    package = ROOT / "firmware" / "stm32" / "n657" / "st_ai_output" / "packages" / package_name
    generic.MODEL_NAME = model_name
    generic.PACKAGE_DIRNAME = package_name
    generic.TFLITE_MODEL = artifact
    generic.FIRMWARE_PACKAGE = package
    generic.STAI_OUTPUT_DIR = package / "st_ai_output"
    generic.STAI_WS_DIR = package / "st_ai_ws"
    generic.STAGING_BUILD = ROOT / "tmp" / f"stedgeai_{model_name}_build"
    generic.TEMP_WORKSPACE = ROOT / "tmp" / f"stedgeai_{model_name}_ws"
    generic.TEMP_OUTPUT = ROOT / "tmp" / f"stedgeai_{model_name}_out"
    generic.EXPECTED_XSPI2_RAW = generic.STAI_OUTPUT_DIR / f"{model_name}_atonbuf.xSPI2.raw"
    generic._run = _run_with_cubeide_tools
    generic.main()

    # Keep both metadata files next to the raw blob for reproducible firmware review.
    generated_workspace = generic.TEMP_WORKSPACE / f"neural_art__{model_name}"
    build_network = generic.STAI_WS_DIR / "build_network"
    build_network.mkdir(parents=True, exist_ok=True)
    # The firmware wrapper includes the relocatable C source and contract;
    # object files alone are not a complete board package.
    for pattern in ("*_reloc.c", "*_reloc_conf.h", "*_reloc_mempools.c"):
        for source in generic.STAGING_BUILD.glob(pattern):
            shutil.copy2(source, build_network / source.name)
    for filename in ("c_info.json", "network.csv"):
        source = generated_workspace / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing generated metadata: {source}")
        shutil.copy2(source, generic.STAI_OUTPUT_DIR / filename)


def main() -> None:
    """Package the ellipse and compact keypoint models in sequence."""
    _package(
        "ellipse_iter8_universal_wide_deep_int8",
        "ellipse_iter8_universal_wide_deep_int8_n6_npu",
        ROOT / "ml" / "artifacts" / "ellipse_iter8_universal_wide_deep" / "model_int8.tflite",
    )
    _package(
        "keypoint_unet_224g_wide_aug_int8",
        "keypoint_unet_224g_wide_aug_int8_n6_npu",
        ROOT / "ml" / "artifacts" / "keypoint_unet_224g_wide_aug" / "model_int8.tflite",
    )


if __name__ == "__main__":
    main()
