"""Probe MobileNetV2 widths against the STM32N6 relocatable memory budget."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any


# Add `ml/src` to sys.path so this script works from the `ml/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.export import (  # noqa: E402
    ExportConfig,
    export_board_tflite_artifacts,
)
from embedded_gauge_reading_tinyml.models import (  # noqa: E402
    build_mobilenetv2_regression_model,
)


DEFAULT_PACK_ROOT: str = os.environ.get(
    "X_CUBE_AI_PACK_ROOT",
    r"C:\Users\rishi_latchmepersad\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0",
)
DEFAULT_HARD_CASE_MANIFEST: Path = PROJECT_ROOT / "data" / "hard_cases.csv"
DEFAULT_OUTPUT_ROOT: Path = PROJECT_ROOT / "artifacts" / "fit_search"
DEFAULT_IMAGE_HEIGHT: int = 224
DEFAULT_IMAGE_WIDTH: int = 224
DEFAULT_REPRESENTATIVE_COUNT: int = 8
DEFAULT_ALPHA_CANDIDATES: tuple[float, ...] = (2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5)


@dataclass(frozen=True)
class FitProbeResult:
    """Summary of one MobileNetV2 fit probe."""

    alpha: float
    head_units: int
    model_name: str
    model_params: int
    tflite_size_bytes: int
    hyperram_bytes: int
    octoflash_bytes: int
    internal_sram_bytes: int
    generate_dir: Path
    tflite_path: Path

    @property
    def fits_internal(self) -> bool:
        """Return True when the candidate stays off external hyperRAM."""
        return self.hyperram_bytes == 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fit probe."""
    parser = argparse.ArgumentParser(
        description=(
            "Probe MobileNetV2 widths and report the largest candidate that stays "
            "inside the STM32N6 on-chip relocatable memory pools."
        )
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_CANDIDATES),
        help="Candidate MobileNetV2 alpha values to probe, in descending order.",
    )
    parser.add_argument(
        "--head-units",
        type=int,
        default=128,
        help="Dense head width to use while probing the backbone size.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.2,
        help="Dropout rate to use in the candidate head.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Candidate input height.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Candidate input width.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=DEFAULT_REPRESENTATIVE_COUNT,
        help="Representative images to use when exporting int8 TFLite.",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=DEFAULT_HARD_CASE_MANIFEST,
        help="CSV manifest used to calibrate the export quantizer.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives per-candidate probe artifacts.",
    )
    parser.add_argument(
        "--pack-root",
        type=str,
        default=DEFAULT_PACK_ROOT,
        help="Root of the X-CUBE-AI pack installation.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet weights while probing. The default is faster and still fit-representative.",
    )
    parser.add_argument(
        "--stop-at-first-fit",
        action="store_true",
        help="Stop once the first candidate stays out of hyperRAM.",
    )
    return parser.parse_args()


def _quote_ps(value: str) -> str:
    """Quote a string for a PowerShell command line."""
    return "'" + value.replace("'", "''") + "'"


def _to_windows_path(path: Path) -> str:
    """Convert a WSL path to a Windows path when needed."""
    if os.name == "nt":
        return str(path)
    completed = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess and stream output to the console."""
    printable = " ".join(command)
    print(f"[FIT] $ {printable}", flush=True)
    subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else None)


def _build_candidate_model(
    *,
    image_height: int,
    image_width: int,
    alpha: float,
    head_units: int,
    head_dropout: float,
    pretrained: bool,
) -> tuple[Any, int, str]:
    """Build a MobileNetV2 regressor and return the model, param count, and name."""
    model = build_mobilenetv2_regression_model(
        image_height=image_height,
        image_width=image_width,
        pretrained=pretrained,
        backbone_trainable=False,
        alpha=alpha,
        head_units=head_units,
        head_dropout=head_dropout,
    )
    return model, model.count_params(), model.name


def _parse_network_csv(network_csv: Path) -> dict[str, int]:
    """Parse the Total row from an ATON network.csv file."""
    with network_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("epochs", "").strip() != "Total":
                continue

            totals: dict[str, int] = {}
            for key, value in row.items():
                if not key.endswith(" (r+w)"):
                    continue
                if value is None or value.strip() in {"", "?"}:
                    totals[key.removesuffix(" (r+w)")] = 0
                    continue
                totals[key.removesuffix(" (r+w)")] = int(round(float(value)))
            return totals

    raise FileNotFoundError(f"Total row not found in network summary: {network_csv}")


def _run_stedgeai_generate(
    *,
    model_path: Path,
    name: str,
    pack_root: PureWindowsPath,
    workspace_dir: Path,
    output_dir: Path,
) -> None:
    """Run the ST Edge AI generator so we can inspect relocatable memory fit."""
    stedgeai_exe = str(
        pack_root / "Utilities" / "windows" / "stedgeai.exe"
    )
    mpool_path = str(
        pack_root / "scripts" / "N6_reloc" / "test" / "mpools" / "stm32n6_reloc.mpool"
    )
    neural_art_path = str(
        pack_root / "scripts" / "N6_reloc" / "test" / "neural_art_reloc.json"
    )
    model_win_path = _to_windows_path(model_path)
    workspace_win_path = _to_windows_path(workspace_dir)
    output_win_path = _to_windows_path(output_dir)

    # The relocatable generator resolves the ST Edge AI core from this env var.
    os.environ["STEDGEAI_CORE_DIR"] = str(pack_root)

    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            "& "
            f"{_quote_ps(stedgeai_exe)} generate "
            f"--model {_quote_ps(model_win_path)} "
            "--target stm32n6 "
            "--type tflite "
            f"--name {_quote_ps(name)} "
            "--compression lossless "
            "--optimization balanced "
            "--input-data-type float32 "
            "--output-data-type float32 "
            "--inputs-ch-position chlast "
            "--outputs-ch-position chlast "
            f"--memory-pool {_quote_ps(mpool_path)} "
            f"--st-neural-art {_quote_ps(f'test@{neural_art_path}')} "
            f"--workspace {_quote_ps(workspace_win_path)} "
            f"--output {_quote_ps(output_win_path)} "
            "--relocatable "
            "--no-report"
        ),
    ]
    _run_command(command, cwd=PROJECT_ROOT)


def _probe_candidate(
    *,
    output_root: Path,
    pack_root: str,
    hard_case_manifest: Path,
    image_height: int,
    image_width: int,
    alpha: float,
    head_units: int,
    head_dropout: float,
    representative_count: int,
    pretrained: bool,
) -> FitProbeResult:
    """Build, export, and generate a single candidate."""
    alpha_tag: str = f"a{int(round(alpha * 100.0)):03d}"
    candidate_name = f"mobilenetv2_fit_{alpha_tag}_h{head_units:03d}"
    candidate_dir = output_root / candidate_name
    training_dir = candidate_dir / "training"
    export_dir = candidate_dir / "deployment"
    workspace_dir = candidate_dir / "st_ai_ws"
    stai_output_dir = candidate_dir / "st_ai_output"

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    stai_output_dir.mkdir(parents=True, exist_ok=True)

    model, model_params, model_name = _build_candidate_model(
        image_height=image_height,
        image_width=image_width,
        alpha=alpha,
        head_units=head_units,
        head_dropout=head_dropout,
        pretrained=pretrained,
    )

    model_path = training_dir / "model.keras"
    model.save(model_path)
    print(
        "[FIT] Candidate built: "
        f"name={model_name} alpha={alpha} head_units={head_units} "
        f"params={model_params}",
        flush=True,
    )

    export_result = export_board_tflite_artifacts(
        ExportConfig(
            model_path=model_path,
            output_dir=export_dir,
            hard_case_manifest=hard_case_manifest,
            representative_count=representative_count,
            image_height=image_height,
            image_width=image_width,
        )
    )

    _run_stedgeai_generate(
        model_path=export_result.tflite_path,
        name=candidate_name,
        pack_root=PureWindowsPath(pack_root),
        workspace_dir=workspace_dir,
        output_dir=stai_output_dir,
    )

    network_csv = (
        workspace_dir / f"neural_art__{candidate_name}" / "network.csv"
    )
    if not network_csv.is_file():
        raise FileNotFoundError(
            f"Expected network summary not found after generation: {network_csv}"
        )

    totals = _parse_network_csv(network_csv)
    hyperram_bytes = totals.get("hyperRAM", 0)
    octoflash_bytes = totals.get("octoFlash", 0)
    internal_sram_bytes = sum(
        value
        for pool_name, value in totals.items()
        if pool_name not in {"hyperRAM", "octoFlash"}
    )

    return FitProbeResult(
        alpha=alpha,
        head_units=head_units,
        model_name=model_name,
        model_params=model_params,
        tflite_size_bytes=export_result.tflite_path.stat().st_size,
        hyperram_bytes=hyperram_bytes,
        octoflash_bytes=octoflash_bytes,
        internal_sram_bytes=internal_sram_bytes,
        generate_dir=workspace_dir / f"neural_art__{candidate_name}",
        tflite_path=export_result.tflite_path,
    )


def _print_result(result: FitProbeResult) -> None:
    """Print a concise, human-readable summary for one candidate."""
    fit_state = "FIT" if result.fits_internal else "SPILL"
    print(
        "[FIT] "
        f"{fit_state} alpha={result.alpha:.2f} "
        f"params={result.model_params} "
        f"tflite={result.tflite_size_bytes}B "
        f"internal={result.internal_sram_bytes}B "
        f"hyperRAM={result.hyperram_bytes}B "
        f"octoFlash={result.octoflash_bytes}B "
        f"artifacts={result.generate_dir}",
        flush=True,
    )


def main() -> None:
    """Probe the candidate widths and report the best board-fit model."""
    args = _parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[FitProbeResult] = []
    for alpha in sorted(args.alphas, reverse=True):
        result = _probe_candidate(
            output_root=output_root,
            pack_root=args.pack_root,
            hard_case_manifest=args.hard_case_manifest,
            image_height=args.image_height,
            image_width=args.image_width,
            alpha=alpha,
            head_units=args.head_units,
            head_dropout=args.head_dropout,
            representative_count=args.representative_count,
            pretrained=args.pretrained,
        )
        results.append(result)
        _print_result(result)

        if args.stop_at_first_fit and result.fits_internal:
            break

    if not results:
        raise RuntimeError("No fit candidates were probed.")

    best_fit = next((result for result in results if result.fits_internal), None)
    if best_fit is None:
        print("[FIT] No candidate stayed inside the internal SRAM pools.", flush=True)
        best_fit = max(results, key=lambda result: result.alpha)
    else:
        print(
            "[FIT] Best fit: "
            f"alpha={best_fit.alpha:.2f} "
            f"params={best_fit.model_params} "
            f"internal={best_fit.internal_sram_bytes}B "
            f"hyperRAM={best_fit.hyperram_bytes}B",
            flush=True,
        )

    summary_path = output_root / "mobilenetv2_fit_search_summary.json"
    summary: dict[str, Any] = {
        "best_fit": {
            "alpha": best_fit.alpha,
            "head_units": best_fit.head_units,
            "model_name": best_fit.model_name,
            "model_params": best_fit.model_params,
            "tflite_size_bytes": best_fit.tflite_size_bytes,
            "hyperram_bytes": best_fit.hyperram_bytes,
            "octoflash_bytes": best_fit.octoflash_bytes,
            "internal_sram_bytes": best_fit.internal_sram_bytes,
            "tflite_path": str(best_fit.tflite_path),
            "generate_dir": str(best_fit.generate_dir),
        },
        "probed": [
            {
                "alpha": result.alpha,
                "head_units": result.head_units,
                "model_name": result.model_name,
                "model_params": result.model_params,
                "tflite_size_bytes": result.tflite_size_bytes,
                "hyperram_bytes": result.hyperram_bytes,
                "octoflash_bytes": result.octoflash_bytes,
                "internal_sram_bytes": result.internal_sram_bytes,
                "fits_internal": result.fits_internal,
                "tflite_path": str(result.tflite_path),
                "generate_dir": str(result.generate_dir),
            }
            for result in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[FIT] Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
