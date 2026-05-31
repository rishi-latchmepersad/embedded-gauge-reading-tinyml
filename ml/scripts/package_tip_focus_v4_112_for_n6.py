"""Package the locked tip-focus model for STM32N6 NPU deployment.

Generates the relocatable NPU C sources and xSPI2 weight blob, places them
into the firmware package tree, and prints the new xSPI2 signature bytes so
the caller can update app_ai.c.
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
TFLITE_MODEL: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "deployment"
    / "geometry_heatmap_v4_112_tflite"
    / "recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3"
    / "model_v4_112_int8.tflite"
)

# The existing firmware package home — we replace the st_ai_output/ contents.
FIRMWARE_PACKAGE: Path = (
    REPO_ROOT
    / "firmware"
    / "stm32"
    / "n657"
    / "st_ai_output"
    / "packages"
    / "tip_focus_v4_112_int8_n6_npu"
)
STAI_OUTPUT_DIR: Path = FIRMWARE_PACKAGE / "st_ai_output"
STAI_WS_DIR: Path = FIRMWARE_PACKAGE / "st_ai_ws"

# Staging area for the NPU build (Windows-writable).
STAGING_BUILD: Path = REPO_ROOT / "tmp" / "tip_focus_reloc_build"
# Temp workspace for the stedgeai generator.
TEMP_WORKSPACE: Path = REPO_ROOT / "tmp" / "stedgeai_tip_focus_ws"
# Temp output for the intermediate C sources.
TEMP_OUTPUT: Path = REPO_ROOT / "tmp" / "stedgeai_tip_focus_out"

# Path to the new xSPI2 raw blob after the build.
EXPECTED_XSPI2_RAW: Path = STAI_OUTPUT_DIR / "network_atonbuf.xSPI2.raw"

XSPI2_PROBE_BYTES: int = 16


def _to_windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path.resolve())
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())], text=True
    ).strip()


def _print_signature(raw_path: Path) -> None:
    """Print the xSPI2 signature bytes so the user can update app_ai.c."""
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
    mpool = (
        PACK_ROOT
        / "scripts"
        / "N6_reloc"
        / "test"
        / "mpools"
        / "stm32n6_reloc.mpool"
    )
    neural_art_profile = (
        PACK_ROOT / "scripts" / "N6_reloc" / "test" / "neural_art_reloc.json"
    )

    for path in (TFLITE_MODEL, stedgeai_exe, pack_python, npu_driver, mpool, neural_art_profile):
        if not path.is_file():
            print(f"[ERROR] Required file not found: {path}", flush=True)
            sys.exit(1)

    # Clean temp dirs.
    for d in (TEMP_WORKSPACE, TEMP_OUTPUT, STAGING_BUILD):
        if d.exists():
            shutil.rmtree(d)

    TEMP_WORKSPACE.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    STAGING_BUILD.mkdir(parents=True, exist_ok=True)

    os.environ["STEDGEAI_CORE_DIR"] = _to_windows_path(PACK_ROOT)

    # Step 1: stedgeai generate with relocatable + neural-art
    generate_cmd = [
        str(stedgeai_exe),
        "generate",
        "--model", _to_windows_path(TFLITE_MODEL),
        "--target", "stm32n6",
        "--type", "tflite",
        "--name", "network",
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

    # Step 2: Find the generated C entry point for the relocatable build.
    candidates = sorted(TEMP_OUTPUT.glob("**/network.c"))
    if not candidates:
        candidates = sorted(TEMP_WORKSPACE.glob("**/network.c"))
    if not candidates:
        print("[ERROR] Could not find generated network.c", flush=True)
        sys.exit(1)
    generated_c = candidates[0]
    print(f"[PACKAGE] Generated C entry: {generated_c}", flush=True)

    # Step 3: Build the relocatable binary via npu_driver.py
    npu_cmd = [
        str(pack_python),
        _to_windows_path(npu_driver),
        "-i", _to_windows_path(generated_c),
        "-o", _to_windows_path(STAGING_BUILD),
        "-n", "network",
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

    # Step 4: Find the generated xSPI2 raw blob.
    raw_candidates = list(STAGING_BUILD.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        raw_candidates = list(TEMP_OUTPUT.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        raw_candidates = list(TEMP_WORKSPACE.glob("**/atonbuf.xSPI2.raw"))
    if not raw_candidates:
        print("[ERROR] Could not find generated atonbuf.xSPI2.raw", flush=True)
        sys.exit(1)
    generated_raw = raw_candidates[0]
    print(f"[PACKAGE] xSPI2 raw blob: {generated_raw} ({generated_raw.stat().st_size} bytes)", flush=True)

    # Step 5: Copy the generated st_ai_output files to the firmware package.
    # We keep the existing files but replace the C sources and the raw blob.
    STAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy generated C/H files from TEMP_OUTPUT to STAI_OUTPUT_DIR.
    for ext in (".c", ".h"):
        for src in TEMP_OUTPUT.glob(f"network*{ext}"):
            dst = STAI_OUTPUT_DIR / src.name
            shutil.copy2(src, dst)
            print(f"[PACKAGE]   {src.name} -> {dst}", flush=True)

    # Copy LICENSE.
    for lic in TEMP_OUTPUT.glob("LICENSE*"):
        shutil.copy2(lic, STAI_OUTPUT_DIR / lic.name)

    # Copy the generated compiler metadata so the firmware package carries
    # the same model provenance as the freshly built binary blob.
    c_info_src = STAI_WS_DIR / "neural_art__network" / "c_info.json"
    if c_info_src.is_file():
        shutil.copy2(c_info_src, STAI_OUTPUT_DIR / "tip_focus_v4_112_int8_c_info.json")

    # Copy the raw blob to the canonical firmware location.
    dst_raw = STAI_OUTPUT_DIR / "network_atonbuf.xSPI2.raw"
    shutil.copy2(generated_raw, dst_raw)
    print(f"[PACKAGE]   {generated_raw.name} -> {dst_raw}", flush=True)

    # Also update the workspace copy.
    ws_raw = STAI_WS_DIR / "neural_art__network" / "atonbuf.xSPI2.raw"
    ws_raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_raw, ws_raw)

    # Step 5b: Refresh the build_network/ runtime objects so the firmware link
    # line picks up objects compiled from the same npu_driver run that produced
    # network.c and the xSPI2 blob.  npu_driver.py places them in the staging
    # directory alongside the generated ELF.
    BUILD_NET_DIR = STAI_WS_DIR / "build_network"
    BUILD_NET_DIR.mkdir(parents=True, exist_ok=True)
    for src_obj in STAGING_BUILD.glob("*.o"):
        dst_obj = BUILD_NET_DIR / src_obj.name
        shutil.copy2(src_obj, dst_obj)
        print(f"[PACKAGE]   {src_obj.name} -> {dst_obj}", flush=True)

    # Step 6: Print the signature for app_ai.c.
    _print_signature(dst_raw)

    print("[PACKAGE] Done.", flush=True)


if __name__ == "__main__":
    main()
