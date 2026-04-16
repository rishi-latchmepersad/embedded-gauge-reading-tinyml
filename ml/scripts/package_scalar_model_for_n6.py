"""Package the calibrated scalar CNN into an STM32N6 relocatable binary.

This script wraps the X-CUBE-AI / ST Edge AI Core relocatable flow that ships
with the STM32Cube pack. It takes the already-quantized scalar TFLite model,
generates the intermediate Neural-ART C sources, then converts them into a
`network_rel.bin` bundle that the STM32N6 board loader can flash at runtime.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
ML_ROOT: Path = REPO_ROOT / "ml"
DEFAULT_MODEL: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "deployment"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "runtime"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8_reloc"
)
DEFAULT_CANONICAL_XSPI2_RAW: Path = REPO_ROOT / "st_ai_output" / "atonbuf.xSPI2.raw"
DEFAULT_WORKSPACE_DIR: Path = (
    REPO_ROOT
    / "st_ai_output"
    / "packages"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "st_ai_ws"
)
DEFAULT_STAI_OUTPUT_DIR: Path = (
    REPO_ROOT
    / "st_ai_output"
    / "packages"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "st_ai_output"
)
DEFAULT_WINDOWS_BUILD_ROOT: Path = Path("/mnt/c/Users/rishi_latchmepersad/ml_reloc_build")
DEFAULT_PACK_ROOT: Path = Path(
    os.environ.get(
        "X_CUBE_AI_PACK_ROOT",
        "/mnt/c/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0",
    )
)
DEFAULT_MODEL_NAME: str = "scalar_full_finetune_from_best_piecewise_calibrated_int8"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the packaging flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and package the calibrated scalar CNN as an STM32N6 relocatable binary."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the int8 TFLite model to package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the final relocatable bundle should be written.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=DEFAULT_WORKSPACE_DIR,
        help="Workspace folder used by the ST Edge AI generator.",
    )
    parser.add_argument(
        "--stai-output-dir",
        type=Path,
        default=DEFAULT_STAI_OUTPUT_DIR,
        help="Output folder for the intermediate Neural-ART C files.",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=DEFAULT_PACK_ROOT,
        help="Root folder of the X-CUBE-AI pack installation.",
    )
    parser.add_argument(
        "--windows-build-root",
        type=Path,
        default=DEFAULT_WINDOWS_BUILD_ROOT,
        help="Windows-writable staging directory for the relocatable NPU build.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Basename used for the generated model bundle.",
    )
    parser.add_argument(
        "--canonical-raw-path",
        type=Path,
        default=DEFAULT_CANONICAL_XSPI2_RAW,
        help="Path where the generated xSPI2 raw blob should be copied.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "lossless", "low", "medium", "high"],
        default="high",
        help="ST Edge AI compression level for the generated model.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["time", "ram", "balanced"],
        default="balanced",
        help="Global optimization objective for model generation.",
    )
    parser.add_argument(
        "--all-buffers-info",
        action="store_true",
        help=(
            "Emit the full buffer-info table in the generated package. "
            "Leave off for smaller debug builds."
        ),
    )
    return parser.parse_args()


def _run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess and stream output to the console."""
    printable = " ".join(command)
    print(f"[RELOC] $ {printable}", flush=True)
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def _ensure_file(path: Path, description: str) -> None:
    """Raise a helpful error if an expected file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _to_windows_path(path: Path) -> str:
    """Convert a WSL path to a Windows path for Windows-hosted pack tools."""
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def _find_generated_c_file(*search_dirs: Path, model_name: str) -> Path:
    """Locate the generated C entry point emitted by the ST generator."""
    for search_dir in search_dirs:
        direct = search_dir / f"{model_name}.c"
        if direct.is_file():
            return direct

        matches = sorted(search_dir.glob(f"**/{model_name}.c"))
        if matches:
            return matches[0]

    joined_dirs = ", ".join(str(path) for path in search_dirs)
    raise FileNotFoundError(
        f"Unable to find the generated C file for '{model_name}' under {joined_dirs}"
    )


def _find_generated_xspi2_raw_file(*search_dirs: Path, model_name: str) -> Path:
    """Locate the generated xSPI2 raw blob emitted by the ST generator."""
    for search_dir in search_dirs:
        direct = search_dir / f"{model_name}_atonbuf.xSPI2.raw"
        if direct.is_file():
            return direct

        nested = sorted(search_dir.glob("**/atonbuf.xSPI2.raw"))
        if nested:
            return nested[0]

    joined_dirs = ", ".join(str(path) for path in search_dirs)
    raise FileNotFoundError(
        f"Unable to find the generated xSPI2 raw blob for '{model_name}' under {joined_dirs}"
    )


def _write_reloc_profile_without_all_buffers_info(source: Path, destination: Path) -> Path:
    """Create a local reloc profile that omits the debug-heavy buffer flag."""
    text = source.read_text(encoding="utf-8", errors="ignore")

    def _strip_flag(match: re.Match[str]) -> str:
        options = match.group(2)
        cleaned_options = " ".join(
            token for token in options.split() if token != "--all-buffers-info"
        )
        return f'{match.group(1)}{cleaned_options}{match.group(3)}'

    rewritten = re.sub(
        r'("options"\s*:\s*")(.*?)(")',
        _strip_flag,
        text,
    )
    destination.write_text(rewritten, encoding="utf-8")
    return destination


def main() -> None:
    """Run the generation and relocatable packaging flow."""
    args = _parse_args()

    stedgeai_exe = args.pack_root / "Utilities" / "windows" / "stedgeai.exe"
    pack_python = args.pack_root / "Utilities" / "windows" / "python.exe"
    npu_driver = args.pack_root / "scripts" / "N6_reloc" / "npu_driver.py"

    _ensure_file(args.model, "TFLite model")
    _ensure_file(stedgeai_exe, "ST Edge AI generator")
    _ensure_file(pack_python, "Pack Python runtime")
    _ensure_file(npu_driver, "STM32N6 relocatable driver")

    mpool_path = (
        args.pack_root
        / "scripts"
        / "N6_reloc"
        / "test"
        / "mpools"
        / "stm32n6_reloc.mpool"
    ).resolve()
    _ensure_file(mpool_path, "STM32N6 relocatable memory pool")

    # These paths are generated artifacts, so it is safe to start from a clean
    # slate and avoid stale float-model outputs from earlier attempts.
    for path in (args.output_dir, args.workspace_dir, args.stai_output_dir):
        if path.exists():
            shutil.rmtree(path)

    staging_build_dir = args.windows_build_root / args.name
    if staging_build_dir.exists():
        shutil.rmtree(staging_build_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workspace_dir.mkdir(parents=True, exist_ok=True)
    args.stai_output_dir.mkdir(parents=True, exist_ok=True)
    staging_build_dir.mkdir(parents=True, exist_ok=True)

    # The copied reloc profile still expects its memory-pool file in a sibling
    # `mpools/` directory, so stage that layout alongside the rewritten JSON.
    output_mpools_dir = args.output_dir / "mpools"
    output_mpools_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mpool_path, output_mpools_dir / mpool_path.name)

    # The relocatable driver resolves the ST Edge AI core from this env var.
    os.environ["STEDGEAI_CORE_DIR"] = _to_windows_path(args.pack_root)
    windows_model = _to_windows_path(args.model)
    windows_workspace = _to_windows_path(args.workspace_dir)
    windows_output = _to_windows_path(args.stai_output_dir)
    windows_mpool = _to_windows_path(mpool_path)
    reloc_profile_source = (
        args.pack_root / "scripts" / "N6_reloc" / "test" / "neural_art_reloc.json"
    )
    reloc_profile_path = args.output_dir / "neural_art_reloc_no_dbg.json"
    _write_reloc_profile_without_all_buffers_info(reloc_profile_source, reloc_profile_path)
    windows_neural_art = _to_windows_path(reloc_profile_path)

    generate_cmd: list[str] = [
        str(stedgeai_exe),
        "generate",
        "--model",
        windows_model,
        "--target",
        "stm32n6",
        "--type",
        "tflite",
        "--name",
        args.name,
        "--compression",
        args.compression,
        "--optimization",
        args.optimization,
        "--input-data-type",
        "float32",
        "--output-data-type",
        "float32",
        "--inputs-ch-position",
        "chlast",
        "--outputs-ch-position",
        "chlast",
        "--memory-pool",
        windows_mpool,
        "--st-neural-art",
        f"test@{windows_neural_art}",
        "--workspace",
        windows_workspace,
        "--output",
        windows_output,
        "--relocatable",
        "--no-report",
    ]
    if args.all_buffers_info:
        # The full buffer table is useful for deep inspection, but it bloats
        # the generated package enough to push the Debug firmware over ROM.
        generate_cmd.append("--all-buffers-info")

    print("[RELOC] Generating Neural-ART C sources...", flush=True)
    _run_command(generate_cmd, cwd=REPO_ROOT)

    generated_c_file = _find_generated_c_file(
        args.stai_output_dir,
        args.workspace_dir,
        model_name=args.name,
    )
    print(f"[RELOC] Generated C file: {generated_c_file}", flush=True)

    npu_cmd: list[str] = [
        str(pack_python),
        _to_windows_path(npu_driver),
        "-i",
        _to_windows_path(generated_c_file),
        "-o",
        _to_windows_path(staging_build_dir),
        "-n",
        args.name,
        "--no-clean",
        "--no-dbg-info",
        "--verbosity",
        "0",
    ]

    # The Windows pack Python runtime defaults to a legacy console encoding
    # that cannot print the box-drawing tables emitted by the N6 reloc driver.
    # Force UTF-8 so the driver can finish its summary logging cleanly under WSL.
    npu_env = os.environ.copy()
    npu_env["PYTHONUTF8"] = "1"
    npu_env["PYTHONIOENCODING"] = "utf-8"
    npu_env["PYTHONLEGACYWINDOWSSTDIO"] = "1"

    print("[RELOC] Building relocatable binary...", flush=True)
    _run_command(npu_cmd, cwd=REPO_ROOT, env=npu_env)

    reloc_bin = staging_build_dir / f"{args.name}_rel.bin"
    if not reloc_bin.is_file():
        raise FileNotFoundError(f"Relocatable binary not found after packaging: {reloc_bin}")

    final_bin = args.output_dir / reloc_bin.name
    shutil.copy2(reloc_bin, final_bin)
    print(f"[RELOC] Final relocatable binary: {final_bin}", flush=True)
    print(f"[RELOC] Size: {final_bin.stat().st_size} bytes", flush=True)

    canonical_raw = args.canonical_raw_path
    canonical_raw.parent.mkdir(parents=True, exist_ok=True)
    generated_raw = _find_generated_xspi2_raw_file(
        args.stai_output_dir,
        args.workspace_dir,
        model_name=args.name,
    )
    shutil.copy2(generated_raw, canonical_raw)
    print(
        "[RELOC] Canonical xSPI2 raw blob refreshed: "
        f"{canonical_raw} (source={generated_raw})",
        flush=True,
    )


if __name__ == "__main__":
    main()
