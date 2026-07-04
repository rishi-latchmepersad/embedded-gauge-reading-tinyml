#!/usr/bin/env python3
"""Package the v16_160 compact UNet heatmap model for STM32N6 NPU deployment.

Generates the relocatable NPU C sources and xSPI2 weight blob with proper
data/bss segments, places them into the firmware package tree, and prints
the new xSPI2 signature bytes so app_ai.c can be updated.

This script runs from WSL and launches stedgeai.exe on Windows via
subprocess (WSL→Windows interop). All filesystem paths are converted to
Windows paths with wslpath.

Usage (from WSL):
    cd ~/Projects/embedded-gauge-reading-tinyml
    python ml/scripts/package_v16_160_for_n6.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PACK_ROOT: Path = Path(
    os.environ.get(
        "X_CUBE_AI_PACK_ROOT",
        "/mnt/c/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0",
    )
)

# ---------------------------------------------------------------------------
# Input model (v16_160 compact UNet heatmap, int8 PTQ).
# ---------------------------------------------------------------------------
TFLITE_MODEL: Path = (
    REPO_ROOT
    / "tmp"
    / "clean_obb_geo_sweep"
    / "unet_v16_160"
    / "model_int8.tflite"
)

# ---------------------------------------------------------------------------
# Firmware package destination.  We create a new package tree under the
# existing st_ai_output/packages/ layout so the firmware build system can
# pick it up.
# ---------------------------------------------------------------------------
FIRMWARE_PACKAGE: Path = (
    REPO_ROOT
    / "firmware"
    / "stm32"
    / "n657"
    / "st_ai_output"
    / "packages"
    / "tip_focus_v16_160_int8_n6_npu"
)
STAI_OUTPUT_DIR: Path = FIRMWARE_PACKAGE / "st_ai_output"
STAI_WS_DIR: Path = FIRMWARE_PACKAGE / "st_ai_ws"

# ---------------------------------------------------------------------------
# Temp staging areas (under WSL tmp/, Windows-writable).
# ---------------------------------------------------------------------------
STAGING_BUILD: Path = REPO_ROOT / "tmp" / "v16_160_n6_reloc_build"
TEMP_WORKSPACE: Path = REPO_ROOT / "tmp" / "stedgeai_v16_160_reloc_ws"
TEMP_OUTPUT: Path = REPO_ROOT / "tmp" / "stedgeai_v16_160_reloc_out"

XSPI2_PROBE_BYTES: int = 16


def _to_windows_path(path: Path) -> str:
    """Convert a WSL/Linux path to a Windows path."""
    if os.name == "nt":
        return str(path.resolve())
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())], text=True
    ).strip()


def _print_signature(raw_path: Path) -> None:
    """Print the xSPI2 signature bytes for app_ai.c."""
    data = raw_path.read_bytes()
    start = data[:XSPI2_PROBE_BYTES]
    tail = data[-XSPI2_PROBE_BYTES:]
    blob_hash = hashlib.sha256(data).hexdigest()[:16]
    print(f"\n[PACKAGE] New xSPI2 raw blob hash (sha256[:16]): {blob_hash}")
    print(f"[PACKAGE] Size: {len(data)} bytes")
    print(f"\n[PACKAGE] xSPI2 signature start ({XSPI2_PROBE_BYTES} bytes):")
    print("    " + " ".join(f"0x{b:02X}U," for b in start))
    print(f"\n[PACKAGE] xSPI2 signature tail ({XSPI2_PROBE_BYTES} bytes):")
    print("    " + " ".join(f"0x{b:02X}U," for b in tail))
    print()


def main() -> None:
    stedgeai_exe = PACK_ROOT / "Utilities" / "windows" / "stedgeai.exe"
    pack_python = PACK_ROOT / "Utilities" / "windows" / "python.exe"
    npu_driver = PACK_ROOT / "scripts" / "N6_reloc" / "npu_driver.py"
    # Use the no-HyperRAM mpool that the v16_160 model expects.
    mpool = REPO_ROOT / "tmp" / "stm32n6_noramic_no_hyperram.mpool"
    neural_art_profile = (
        PACK_ROOT / "scripts" / "N6_reloc" / "test" / "neural_art_reloc.json"
    )

    # Fall back to the standard reloc mpool if the custom one doesn't exist.
    if not mpool.is_file():
        print("[WARN] No-HyperRAM mpool not found, falling back to standard "
              "reloc mpool (may allocate HyperRAM).", flush=True)
        mpool = (
            PACK_ROOT / "scripts" / "N6_reloc" / "test" / "mpools"
            / "stm32n6_reloc.mpool"
        )

    for path, desc in [
        (TFLITE_MODEL, "v16_160 TFLite"),
        (stedgeai_exe, "stedgeai.exe"),
        (pack_python, "python.exe (pack)"),
        (npu_driver, "npu_driver.py"),
        (mpool, "memory pool"),
        (neural_art_profile, "neural_art_reloc.json"),
    ]:
        if not path.exists():
            print(f"[ERROR] Required file not found: {path} ({desc})", flush=True)
            sys.exit(1)

    # Clean temp dirs.
    for d in (TEMP_WORKSPACE, TEMP_OUTPUT, STAGING_BUILD):
        if d.exists():
            shutil.rmtree(d)

    TEMP_WORKSPACE.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    STAGING_BUILD.mkdir(parents=True, exist_ok=True)

    os.environ["STEDGEAI_CORE_DIR"] = _to_windows_path(PACK_ROOT)

    # -----------------------------------------------------------------------
    # Step 1: stedgeai generate with relocatable + Neural-ART NPU profile.
    # -----------------------------------------------------------------------
    generate_cmd = [
        str(stedgeai_exe),
        "generate",
        "--model", _to_windows_path(TFLITE_MODEL),
        "--target", "stm32n6",
        "--type", "tflite",
        "--name", "tip_focus_v16_160_int8",
        "--compression", "lossless",
        "--optimization", "balanced",
        "--input-data-type", "int8",
        "--output-data-type", "int8",
        "--inputs-ch-position", "chlast",
        "--outputs-ch-position", "chlast",
        "--memory-pool", _to_windows_path(mpool),
        "--st-neural-art", f"test@{_to_windows_path(neural_art_profile)}",
        "--workspace", _to_windows_path(TEMP_WORKSPACE),
        "--output", _to_windows_path(TEMP_OUTPUT),
        "--relocatable",
        "--no-report",
    ]
    print("[PACKAGE] Step 1: stedgeai generate...", flush=True)
    print(f"  $ {' '.join(generate_cmd)}", flush=True)
    subprocess.run(generate_cmd, check=True)

    # -----------------------------------------------------------------------
    # Step 2: Find the generated C entry point.
    # -----------------------------------------------------------------------
    candidates = sorted(TEMP_OUTPUT.glob("**/tip_focus_v16_160_int8.c"))
    if not candidates:
        candidates = sorted(TEMP_WORKSPACE.glob("**/tip_focus_v16_160_int8.c"))
    if not candidates:
        print("[ERROR] Could not find generated C file", flush=True)
        sys.exit(1)
    generated_c = candidates[0]
    print(f"[PACKAGE] Generated C entry: {generated_c}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Build the relocatable binary via npu_driver.py.
    # This compiles the C file with arm-clang and links the .data/.bss
    # segments into a relocatable ELF. The resulting xSPI2.raw includes
    # the proper reloc binary header that ll_aton_reloc_install expects.
    # -----------------------------------------------------------------------
    npu_cmd = [
        _to_windows_path(pack_python),
        _to_windows_path(npu_driver),
        "-i", _to_windows_path(generated_c),
        "-o", _to_windows_path(STAGING_BUILD),
        "-n", "tip_focus_v16_160_int8",
        "--no-clean",
        "--no-dbg-info",
        "--verbosity", "0",
    ]
    npu_env = os.environ.copy()
    npu_env["PYTHONUTF8"] = "1"
    npu_env["PYTHONIOENCODING"] = "utf-8"
    npu_env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    print("[PACKAGE] Step 3: Building relocatable binary...", flush=True)
    subprocess.run(npu_cmd, check=True, env=npu_env)

    # -----------------------------------------------------------------------
    # Step 4: Find the generated xSPI2 raw blob.
    # -----------------------------------------------------------------------
    raw_candidates = list(STAGING_BUILD.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        raw_candidates = list(TEMP_OUTPUT.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        raw_candidates = list(TEMP_WORKSPACE.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        print("[ERROR] Could not find generated atonbuf.xSPI2.raw", flush=True)
        sys.exit(1)
    generated_raw = raw_candidates[0]
    print(
        f"[PACKAGE] xSPI2 raw blob: {generated_raw} "
        f"({generated_raw.stat().st_size} bytes)", flush=True,
    )

    # -----------------------------------------------------------------------
    # Step 5: Copy generated files to the firmware package tree.
    # -----------------------------------------------------------------------
    STAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ext in (".c", ".h"):
        for src in TEMP_OUTPUT.glob(f"tip_focus_v16_160_int8*{ext}"):
            dst = STAI_OUTPUT_DIR / src.name
            shutil.copy2(src, dst)
            print(f"[PACKAGE]   {src.name} -> {dst}", flush=True)

    for lic in TEMP_OUTPUT.glob("LICENSE*"):
        shutil.copy2(lic, STAI_OUTPUT_DIR / lic.name)

    # Copy the xSPI2 raw blob as the canonical tip_focus_v16_160_int8_atonbuf.xSPI2.raw.
    dst_raw = STAI_OUTPUT_DIR / "tip_focus_v16_160_int8_atonbuf.xSPI2.raw"
    shutil.copy2(generated_raw, dst_raw)
    print(f"[PACKAGE]   {generated_raw.name} -> {dst_raw}", flush=True)

    # Also update the workspace copy.
    ws_raw = STAI_WS_DIR / "neural_art__tip_focus_v16_160_int8" / "atonbuf.xSPI2.raw"
    ws_raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_raw, ws_raw)

    # -----------------------------------------------------------------------
    # Step 5b: Copy the build_network/ runtime objects so the firmware
    # link line picks up objects from the same npu_driver run.
    # -----------------------------------------------------------------------
    BUILD_NET_DIR = STAI_WS_DIR / "build_network"
    BUILD_NET_DIR.mkdir(parents=True, exist_ok=True)
    for src_obj in STAGING_BUILD.glob("*.o"):
        dst_obj = BUILD_NET_DIR / src_obj.name
        shutil.copy2(src_obj, dst_obj)
        print(f"[PACKAGE]   {src_obj.name} -> {dst_obj}", flush=True)

    # -----------------------------------------------------------------------
    # Step 6: Print the xSPI2 signature so the user can paste it into
    # app_ai.c as the xspi2_signature_start array.
    # -----------------------------------------------------------------------
    _print_signature(dst_raw)

    print("[PACKAGE] Done.", flush=True)


if __name__ == "__main__":
    main()
