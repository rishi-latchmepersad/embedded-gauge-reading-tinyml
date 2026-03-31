"""Package the calibrated scalar CNN into an STM32N6 relocatable binary.

This script wraps the X-CUBE-AI / ST Edge AI Core relocatable flow that ships
with the STM32Cube pack. It takes the already-quantized scalar TFLite model,
generates the intermediate Neural-ART C sources, then converts them into a
`network_rel.bin` bundle that the STM32N6 board loader can flash at runtime.
"""

from __future__ import annotations

import argparse
import os
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
    / "scalar_hardcase_boost_v1_calibrated_int8"
    / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "runtime"
    / "scalar_hardcase_boost_v1_calibrated_int8_reloc"
)
DEFAULT_WORKSPACE_DIR: Path = (
    REPO_ROOT
    / "st_ai_output"
    / "packages"
    / "scalar_hardcase_boost_v1_calibrated_int8"
    / "st_ai_ws"
)
DEFAULT_STAI_OUTPUT_DIR: Path = (
    REPO_ROOT
    / "st_ai_output"
    / "packages"
    / "scalar_hardcase_boost_v1_calibrated_int8"
    / "st_ai_output"
)
DEFAULT_PACK_ROOT: Path = Path(
    os.environ.get(
        "X_CUBE_AI_PACK_ROOT",
        r"C:\Users\rishi_latchmepersad\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0",
    )
)
DEFAULT_MODEL_NAME: str = "scalar_hardcase_boost_v1_calibrated_int8"


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
        "--name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Basename used for the generated model bundle.",
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
    return parser.parse_args()


def _run_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess and stream output to the console."""
    printable = " ".join(command)
    print(f"[RELOC] $ {printable}", flush=True)
    subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else None)


def _ensure_file(path: Path, description: str) -> None:
    """Raise a helpful error if an expected file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workspace_dir.mkdir(parents=True, exist_ok=True)
    args.stai_output_dir.mkdir(parents=True, exist_ok=True)

    # The relocatable driver resolves the ST Edge AI core from this env var.
    os.environ["STEDGEAI_CORE_DIR"] = str(args.pack_root.resolve())

    generate_cmd: list[str] = [
        str(stedgeai_exe),
        "generate",
        "--model",
        str(args.model.resolve()),
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
        str(mpool_path),
        "--st-neural-art",
        f"test@{(args.pack_root / 'scripts' / 'N6_reloc' / 'test' / 'neural_art_reloc.json').resolve()}",
        "--workspace",
        str(args.workspace_dir.resolve()),
        "--output",
        str(args.stai_output_dir.resolve()),
        "--relocatable",
        "--no-report",
    ]

    print("[RELOC] Generating Neural-ART C sources...", flush=True)
    _run_command(generate_cmd, cwd=REPO_ROOT)

    generated_c_file = _find_generated_c_file(
        args.stai_output_dir,
        args.workspace_dir,
        model_name=args.name,
    )
    print(f"[RELOC] Generated C file: {generated_c_file}", flush=True)

    build_dir = args.output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    npu_cmd: list[str] = [
        str(pack_python),
        str(npu_driver),
        "-i",
        str(generated_c_file.resolve()),
        "-o",
        str(build_dir.resolve()),
        "-n",
        args.name,
        "--no-clean",
    ]

    print("[RELOC] Building relocatable binary...", flush=True)
    _run_command(npu_cmd, cwd=REPO_ROOT)

    reloc_bin = build_dir / f"{args.name}_rel.bin"
    if not reloc_bin.is_file():
        raise FileNotFoundError(f"Relocatable binary not found after packaging: {reloc_bin}")

    final_bin = args.output_dir / reloc_bin.name
    shutil.copy2(reloc_bin, final_bin)
    print(f"[RELOC] Final relocatable binary: {final_bin}", flush=True)
    print(f"[RELOC] Size: {final_bin.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
