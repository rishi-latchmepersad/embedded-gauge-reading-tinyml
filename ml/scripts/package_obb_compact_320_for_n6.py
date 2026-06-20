"""Package the compact OBB 320 full-frame model for STM32N6 (no HyperRAM)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_NAME: str = "obb_compact_320_int8"
PACKAGE_DIRNAME: str = "obb_compact_320_int8"
TFLITE_MODEL: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "training"
    / "obb_fullframe_320_20260607_183447"
    / "model_int8.tflite"
)
FIRMWARE_PACKAGE: Path = (
    REPO_ROOT
    / "firmware"
    / "stm32"
    / "n657"
    / "st_ai_output"
    / "packages"
    / PACKAGE_DIRNAME
)
STAI_OUTPUT_DIR: Path = FIRMWARE_PACKAGE / "st_ai_output"
STAI_WS_DIR: Path = FIRMWARE_PACKAGE / "st_ai_ws"
STAGING_BUILD: Path = REPO_ROOT / "tmp" / "stedgeai_obb_compact_320_build"
TEMP_WORKSPACE: Path = REPO_ROOT / "tmp" / "stedgeai_obb_compact_320_ws"
TEMP_OUTPUT: Path = REPO_ROOT / "tmp" / "stedgeai_obb_compact_320_out"
EXPECTED_XSPI2_RAW: Path = STAI_OUTPUT_DIR / f"{MODEL_NAME}_atonbuf.xSPI2.raw"
LOCAL_MPOOL: Path = REPO_ROOT / "tmp" / "mpools" / "stm32n6_onchip.mpool"
XSPI2_PROBE_BYTES: int = 16


def _default_pack_root() -> Path:
    """Return the local X-CUBE-AI pack root for Windows or WSL."""
    if os.name == "nt":
        return Path(
            os.environ.get(
                "X_CUBE_AI_PACK_ROOT",
                r"C:\Users\rishi_latchmepersad\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0",
            )
        )
    return Path(
        os.environ.get(
            "X_CUBE_AI_PACK_ROOT",
            "/mnt/c/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0",
        )
    )


PACK_ROOT: Path = _default_pack_root()


def _to_windows_path(path: Path) -> str:
    """Convert a local path to a Windows path for the ST tools."""
    if os.name == "nt":
        return str(path.resolve())
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run one subprocess and stream the command line to the console."""
    print(f"[PACKAGE] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def _find_generated_c(model_name: str) -> Path:
    """Find the generated C entry point emitted by the ST generator."""
    for search_dir in (TEMP_OUTPUT, TEMP_WORKSPACE):
        candidate = search_dir / f"{model_name}.c"
        if candidate.is_file():
            return candidate
        matches = sorted(search_dir.glob(f"**/{model_name}.c"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find generated {model_name}.c")


def _find_generated_raw() -> Path:
    """Find the relocatable xSPI2 blob produced by the NPU driver."""
    for search_dir in (STAGING_BUILD, TEMP_OUTPUT, TEMP_WORKSPACE):
        matches = sorted(search_dir.glob("**/atonbuf.xSPI2.raw"))
        if matches:
            return matches[0]
    raise FileNotFoundError("Could not find generated atonbuf.xSPI2.raw")


def _print_signature(raw_path: Path) -> None:
    """Print the xSPI2 prefix and suffix so app_ai.c can be updated."""
    data = raw_path.read_bytes()
    start = data[:XSPI2_PROBE_BYTES]
    tail = data[-XSPI2_PROBE_BYTES:]
    blob_hash = hashlib.sha256(data).hexdigest()[:16]
    print(f"\n[PACKAGE] sha256[:16] = {blob_hash}")
    print(f"[PACKAGE] size = {len(data)} bytes")
    print(f"[PACKAGE] start ({XSPI2_PROBE_BYTES} bytes):")
    print("    " + " ".join(f"0x{byte:02X}U," for byte in start))
    print(f"[PACKAGE] tail ({XSPI2_PROBE_BYTES} bytes):")
    print("    " + " ".join(f"0x{byte:02X}U," for byte in tail))


def main() -> None:
    """Run the packaging flow and stage the generated firmware artifacts."""
    stedgeai_exe = PACK_ROOT / "Utilities" / "windows" / "stedgeai.exe"
    pack_python = PACK_ROOT / "Utilities" / "windows" / "python.exe"
    npu_driver = PACK_ROOT / "scripts" / "N6_reloc" / "npu_driver.py"
    neural_art_profile = PACK_ROOT / "scripts" / "N6_reloc" / "test" / "neural_art_reloc.json"

    for path in (TFLITE_MODEL, stedgeai_exe, pack_python, npu_driver, LOCAL_MPOOL, neural_art_profile):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    if FIRMWARE_PACKAGE.exists():
        shutil.rmtree(FIRMWARE_PACKAGE)
    for path in (STAGING_BUILD, TEMP_WORKSPACE, TEMP_OUTPUT):
        if path.exists():
            shutil.rmtree(path)
    for path in (STAI_OUTPUT_DIR, STAI_WS_DIR, STAGING_BUILD, TEMP_WORKSPACE, TEMP_OUTPUT):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["STEDGEAI_CORE_DIR"] = _to_windows_path(PACK_ROOT)

    generate_cmd = [
        str(stedgeai_exe),
        "generate",
        "--model",
        _to_windows_path(TFLITE_MODEL),
        "--target",
        "stm32n6",
        "--type",
        "tflite",
        "--name",
        MODEL_NAME,
        "--compression",
        "lossless",
        "--optimization",
        "balanced",
        "--input-data-type",
        "int8",
        "--output-data-type",
        "int8",
        "--inputs-ch-position",
        "chlast",
        "--outputs-ch-position",
        "chlast",
        "--memory-pool",
        _to_windows_path(LOCAL_MPOOL),
        "--st-neural-art",
        f"test@{_to_windows_path(neural_art_profile)}",
        "--workspace",
        _to_windows_path(TEMP_WORKSPACE),
        "--output",
        _to_windows_path(TEMP_OUTPUT),
        "--relocatable",
        "--no-report",
    ]
    print("[PACKAGE] Step 1: stedgeai generate...", flush=True)
    _run(generate_cmd)

    generated_c = _find_generated_c(MODEL_NAME)
    print(f"[PACKAGE] generated C: {generated_c}", flush=True)

    npu_cmd = [
        str(pack_python),
        _to_windows_path(npu_driver),
        "-i",
        _to_windows_path(generated_c),
        "-o",
        _to_windows_path(STAGING_BUILD),
        "-n",
        MODEL_NAME,
        "--no-clean",
        "--no-dbg-info",
        "--verbosity",
        "0",
    ]
    npu_env = os.environ.copy()
    npu_env["PYTHONUTF8"] = "1"
    npu_env["PYTHONIOENCODING"] = "utf-8"
    npu_env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    print("[PACKAGE] Step 2: npu_driver relocatable build...", flush=True)
    _run(npu_cmd, env=npu_env)

    generated_raw = _find_generated_raw()
    print(f"[PACKAGE] raw blob: {generated_raw} ({generated_raw.stat().st_size} bytes)", flush=True)

    STAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ext in (".c", ".h"):
        for src in TEMP_OUTPUT.glob(f"{MODEL_NAME}*{ext}"):
            dst = STAI_OUTPUT_DIR / src.name
            shutil.copy2(src, dst)
            print(f"[PACKAGE]   {src.name} -> {dst}", flush=True)

    for lic in TEMP_OUTPUT.glob("LICENSE*"):
        shutil.copy2(lic, STAI_OUTPUT_DIR / lic.name)

    c_info_src = STAI_WS_DIR / f"neural_art__{MODEL_NAME}" / "c_info.json"
    if c_info_src.is_file():
        shutil.copy2(c_info_src, STAI_OUTPUT_DIR / f"{MODEL_NAME}_c_info.json")

    shutil.copy2(generated_raw, EXPECTED_XSPI2_RAW)
    print(f"[PACKAGE]   {generated_raw.name} -> {EXPECTED_XSPI2_RAW}", flush=True)

    ws_raw = STAI_WS_DIR / f"neural_art__{MODEL_NAME}" / "atonbuf.xSPI2.raw"
    ws_raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_raw, ws_raw)

    build_net_dir = STAI_WS_DIR / "build_network"
    build_net_dir.mkdir(parents=True, exist_ok=True)
    for src_obj in STAGING_BUILD.glob("*.o"):
        dst_obj = build_net_dir / src_obj.name
        shutil.copy2(src_obj, dst_obj)

    _print_signature(EXPECTED_XSPI2_RAW)
    print("[PACKAGE] Done.", flush=True)


if __name__ == "__main__":
    main()
