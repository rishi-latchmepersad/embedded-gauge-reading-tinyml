"""Replay the current STM32 board pipeline on laptop captures.

This command is meant to answer one question: "what would the board have
done on this exact image?"  It mirrors the board path closely enough to show
whether the OBB stage, rectifier fallback, scalar crop, calibration, or burst
history is the thing drifting away from the offline tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import sys
import time
from typing import Final

import numpy as np

# Make the package importable when this script is run from the `ml/` directory.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_OBB_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "prod_model_v0.3_obb_int8"
    / "model_int8.tflite"
)
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "mobilenetv2_rectifier_hardcase_finetune_v3_int8"
    / "model_int8.tflite"
)
DEFAULT_SCALAR_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "model_int8.tflite"
)

SUPPORTED_CAPTURE_SUFFIXES: Final[set[str]] = {".png", ".jpg", ".jpeg", ".yuv422"}


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One capture path plus an optional ground-truth value."""

    capture_path: Path
    true_value: float | None = None


def _parse_args() -> argparse.Namespace:
    """Parse the command line for the board-pipeline replay."""
    parser = argparse.ArgumentParser(
        description="Replay the STM32 board pipeline on raw board captures."
    )
    parser.add_argument(
        "--capture-path",
        action="append",
        type=Path,
        default=[],
        help="Capture to evaluate. Repeat this flag to choose specific files.",
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=REPO_ROOT / "captured_images",
        help="Directory to scan when --capture-path is not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of newest captures to replay when no paths are given.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV manifest with image_path,value rows. When provided, "
            "the script evaluates the manifest rows instead of raw capture files."
        ),
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        default=DEFAULT_OBB_MODEL,
        help="Path to the OBB localizer model.",
    )
    parser.add_argument(
        "--obb-model-kind",
        choices=["auto", "keras", "tflite"],
        default="auto",
        help="Backend used to load the OBB localizer.",
    )
    parser.add_argument(
        "--rectifier-model",
        type=Path,
        default=DEFAULT_RECTIFIER_MODEL,
        help="Path to the rectifier model.",
    )
    parser.add_argument(
        "--rectifier-model-kind",
        choices=["auto", "keras", "tflite"],
        default="auto",
        help="Backend used to load the rectifier model.",
    )
    parser.add_argument(
        "--scalar-model",
        type=Path,
        default=DEFAULT_SCALAR_MODEL,
        help="Path to the quantized scalar reader.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "board_pipeline_eval",
        help="Directory where JSON reports should be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square canvas size used by all three stages.",
    )
    parser.add_argument(
        "--obb-crop-scale",
        type=float,
        default=1.20,
        help="Safety margin applied to the OBB box before crop decoding.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=float,
        default=48.0,
        help="Minimum crop edge length after OBB expansion.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Report the scalar output without applying the board calibration.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="Size of the firmware-style burst history.",
    )
    parser.add_argument(
        "--history-reset-delta",
        type=float,
        default=12.0,
        help="Reset the burst history when values jump by more than this amount.",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Print stage-by-stage progress markers while replaying each capture.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load capture/value pairs from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            items.append(
                EvalItem(
                    capture_path=_resolve_image_path(row["image_path"]),
                    true_value=float(row["value"]),
                )
            )
    return items


def _find_latest_captures(captures_dir: Path, *, limit: int) -> list[EvalItem]:
    """Return the newest raw captures and PNG previews from the capture folder."""
    candidates = [
        path
        for path in captures_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_CAPTURE_SUFFIXES
        and path.stat().st_size > 0
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [EvalItem(capture_path=path) for path in candidates[:limit]]


def _format_bytes(byte_values: tuple[int, ...]) -> str:
    """Format a tiny byte window the same way the firmware logs do."""
    return " ".join(f"{value:02X}" for value in byte_values)


def _format_probe(probe: TensorProbe) -> str:
    """Render a compact, firmware-like tensor fingerprint."""
    return (
        f"name={probe.name} dtype={probe.dtype} shape={probe.shape} "
        f"len={probe.byte_length} hash={probe.crc32_hex} "
        f"first8=[{_format_bytes(probe.first8)}] "
        f"mid8=[{_format_bytes(probe.mid8)}] "
        f"last8=[{_format_bytes(probe.last8)}]"
    )


def _print_result(result: BoardPipelineResult) -> None:
    """Print one pipeline result in a readable stage-by-stage form."""
    print(f"[PIPE] Capture: {result.capture_path}")
    print(
        f"[PIPE] Source: kind={result.source_kind} shape={result.source_shape}",
        flush=True,
    )
    print(f"[PIPE] Full-frame probe: {_format_probe(result.full_frame_probe)}")
    print(f"[PIPE] OBB output probe: {_format_probe(result.obb_output_probe)}")
    if result.obb_decision.accepted:
        print(
            "[PIPE] OBB crop: "
            f"x={result.obb_decision.crop_box_xyxy[0]:.1f} "
            f"y={result.obb_decision.crop_box_xyxy[1]:.1f} "
            f"w={result.obb_decision.crop_box_xyxy[2] - result.obb_decision.crop_box_xyxy[0]:.1f} "
            f"h={result.obb_decision.crop_box_xyxy[3] - result.obb_decision.crop_box_xyxy[1]:.1f}"
        )
    else:
        print(
            "[PIPE] OBB crop outside training window: "
            f"crop={result.obb_decision.crop_box_xyxy[2] - result.obb_decision.crop_box_xyxy[0]:.1f}x"
            f"{result.obb_decision.crop_box_xyxy[3] - result.obb_decision.crop_box_xyxy[1]:.1f} "
            f"ratio={result.obb_decision.details['crop_width_ratio']:.3f}/"
            f"{result.obb_decision.details['crop_height_ratio']:.3f} "
            f"-> rectifier fallback."
        )

    if result.rectifier_output_probe is not None:
        print(f"[PIPE] Rectifier output probe: {_format_probe(result.rectifier_output_probe)}")
    if result.rectifier_decision is not None:
        rect_w = result.rectifier_decision.crop_box_xyxy[2] - result.rectifier_decision.crop_box_xyxy[0]
        rect_h = result.rectifier_decision.crop_box_xyxy[3] - result.rectifier_decision.crop_box_xyxy[1]
        print(
            "[PIPE] Rectifier crop: "
            f"x={result.rectifier_decision.crop_box_xyxy[0]:.1f} "
            f"y={result.rectifier_decision.crop_box_xyxy[1]:.1f} "
            f"w={rect_w:.1f} h={rect_h:.1f}"
        )

    print(f"[PIPE] Scalar input probe: {_format_probe(result.scalar_input_probe)}")
    print(f"[PIPE] Scalar output probe: {_format_probe(result.scalar_output_probe)}")
    print(
        f"[PIPE] Model output before calibration: {result.raw_prediction:.6f}"
    )
    print(
        f"[PIPE] Model output after calibration: {result.calibrated_prediction:.6f}"
    )
    print(f"[PIPE] Inference value: {result.reported_prediction:.6f}")
    print(
        f"[PIPE] Burst history: count={result.burst_history_count} "
        f"reset={'yes' if result.burst_history_reset else 'no'}"
    )


def _load_items(args: argparse.Namespace) -> list[EvalItem]:
    """Choose manifest rows or raw captures based on the CLI flags."""
    if args.manifest is not None:
        return _load_manifest(args.manifest)

    if args.capture_path:
        return [EvalItem(capture_path=path) for path in args.capture_path]

    return _find_latest_captures(args.captures_dir, limit=args.limit)


def main() -> None:
    """Entry point for the board-pipeline replay CLI."""
    args = _parse_args()
    items = _load_items(args)
    if not items:
        raise FileNotFoundError("No captures were selected for replay.")

    print("[PIPE] Importing board pipeline module...", flush=True)
    import_start = time.perf_counter()
    from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
        DEFAULT_OBB_MODEL,
        DEFAULT_RECTIFIER_MODEL,
        DEFAULT_SCALAR_MODEL,
        InferenceBurstHistory,
        TensorProbe,
        load_model_session,
        predict_board_pipeline_on_capture,
    )

    print(
        f"[PIPE] Board pipeline module imported in {time.perf_counter() - import_start:.3f}s",
        flush=True,
    )
    print("[PIPE] Loading models...", flush=True)
    model_start = time.perf_counter()
    obb_session = load_model_session(args.obb_model, args.obb_model_kind)
    print(
        f"[PIPE] OBB model loaded in {time.perf_counter() - model_start:.3f}s: {args.obb_model}",
        flush=True,
    )
    rectifier_start = time.perf_counter()
    rectifier_session = load_model_session(args.rectifier_model, args.rectifier_model_kind)
    print(
        f"[PIPE] Rectifier model loaded in {time.perf_counter() - rectifier_start:.3f}s: {args.rectifier_model}",
        flush=True,
    )
    scalar_start = time.perf_counter()
    scalar_session = load_model_session(args.scalar_model, "auto")
    print(
        f"[PIPE] Scalar model loaded in {time.perf_counter() - scalar_start:.3f}s: {args.scalar_model}",
        flush=True,
    )
    history = InferenceBurstHistory(
        size=args.history_size,
        reset_delta_c=args.history_reset_delta,
    )
    progress_callback = (
        (lambda message: print(f"[PIPE] {message}", flush=True))
        if args.trace_stages
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    abs_errors: list[float] = []

    for item in items:
        print(f"[PIPE] Replaying {item.capture_path.name}...", flush=True)
        item_start = time.perf_counter()
        result = predict_board_pipeline_on_capture(
            item.capture_path,
            obb_session=obb_session,
            rectifier_session=rectifier_session,
            scalar_session=scalar_session,
            history=history,
            progress=progress_callback,
            image_size=args.image_size,
            obb_crop_scale=args.obb_crop_scale,
            min_crop_size=args.min_crop_size,
            use_calibration=not args.no_calibration,
        )
        _print_result(result)
        print(
            f"[PIPE] Replay completed in {time.perf_counter() - item_start:.3f}s",
            flush=True,
        )

        report_dir = args.output_dir / item.capture_path.stem
        report_dir.mkdir(parents=True, exist_ok=True)
        report_json = report_dir / "report.json"
        payload: dict[str, object] = asdict(result)
        if item.true_value is not None:
            error = abs(result.reported_prediction - item.true_value)
            abs_errors.append(error)
            payload["true_value"] = item.true_value
            payload["abs_error"] = error
        report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        report_rows.append(payload)

    summary: dict[str, object] = {
        "items": len(report_rows),
        "output_dir": str(args.output_dir),
        "obb_model": str(args.obb_model),
        "rectifier_model": str(args.rectifier_model),
        "scalar_model": str(args.scalar_model),
        "calibration_enabled": not args.no_calibration,
        "history_size": args.history_size,
        "history_reset_delta": args.history_reset_delta,
    }
    if abs_errors:
        abs_errors_array = np.asarray(abs_errors, dtype=np.float32)
        summary["mae"] = float(np.mean(abs_errors_array))
        summary["rmse"] = float(np.sqrt(np.mean(np.square(abs_errors_array))))
        summary["max_abs_error"] = float(np.max(abs_errors_array))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[PIPE] Wrote summary: {summary_path}", flush=True)
    if abs_errors:
        print(
            f"[PIPE] samples={len(abs_errors)} mae={summary['mae']:.4f} "
            f"rmse={summary['rmse']:.4f} max_abs_error={summary['max_abs_error']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
