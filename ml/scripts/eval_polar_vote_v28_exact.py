#!/usr/bin/env python3
"""Evaluate the exact offline V28 polar-vote path on labeled captures.

This script answers three related questions:
1. Does the training-time float32 preprocessing reproduce the saved V28 CSVs?
2. How far is the current firmware-style int8 tensor from that float32 path?
3. What happens if we run the exact V28 model on the firmware tensor anyway?

The goal is to keep the offline recipe honest before we touch the board again.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import sys
from typing import Any, Final

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    build_legacy_firmware_polar_vote_tensor,
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
    firmware_training_crop_box,
    load_capture_image,
    probe_tensor,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.polar_vote_v28 import (  # noqa: E402
    build_polar_vote_v28_model,
)

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_WEIGHTS: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "best_weights.weights.h5"
)
DEFAULT_REFERENCE_PREDICTIONS: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "hard_cases_plus_board30_valid_with_new6_predictions.csv"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "polar_vote_v28_exact_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True, slots=True)
class RowResult:
    """One scored row from the exact offline parity run."""

    image_path: str
    source_kind: str
    value: float
    sample_weight: float
    crop_box_x0: float
    crop_box_y0: float
    crop_box_x1: float
    crop_box_y1: float
    training_prediction: float
    firmware_prediction: float
    reference_prediction: float | None
    training_abs_error: float
    firmware_abs_error: float
    reference_abs_error: float | None
    prediction_delta: float
    reference_delta: float | None
    tensor_mean_abs_diff: float
    tensor_max_abs_diff: float
    training_input_crc32: str
    firmware_input_crc32: str
    training_output_crc32: str
    firmware_output_crc32: str


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the exact offline V28 polar-vote pipeline."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest with image_path,value rows.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help="Only keep manifest rows that live under this directory.",
    )
    parser.add_argument(
        "--crop-boxes",
        type=Path,
        default=DEFAULT_CROP_BOXES,
        help="CSV file with rectified crop boxes.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the exact V28 best-weights checkpoint.",
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=DEFAULT_REFERENCE_PREDICTIONS,
        help="Optional CSV with saved V28 predictions to compare against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and JSON outputs should be written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to score. Use 0 for all rows.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before truncation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Expected capture size used for yuv422/raw16 decoding.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge spec identifier from the calibration TOML.",
    )
    parser.add_argument(
        "--center-search-px",
        type=int,
        default=5,
        help="Angular center search radius in pixels used by the offline V28 recipe.",
    )
    parser.add_argument(
        "--center-mode",
        type=str,
        choices=["image_center", "classical_baseline"],
        default="image_center",
        help="How to pick the polar projection center before optional offset search.",
    )
    parser.add_argument(
        "--allow-missing-crop-boxes",
        action="store_true",
        help="Keep manifest rows even when no rectified crop box exists.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw_path: str) -> Path:
    """Resolve a manifest path against the repository root when needed."""
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when path resolves inside root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_manifest(manifest_path: Path, capture_root: Path) -> list[EvalItem]:
    """Load labeled rows and keep only the captures under the requested root."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _resolve_repo_path(str(row["image_path"]))
            if not _is_under_root(image_path, capture_root):
                continue
            value = float(row["value"])
            sample_weight = float(row.get("sample_weight", 1.0) or 1.0)
            items.append(
                EvalItem(
                    image_path=image_path,
                    value=value,
                    sample_weight=sample_weight,
                )
            )
    return items


def _select_items(
    items: list[EvalItem],
    *,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> list[EvalItem]:
    """Select a deterministic subset of the filtered samples."""
    if shuffle:
        rng = random.Random(seed)
        items = list(items)
        rng.shuffle(items)
    if max_samples > 0:
        return items[:max_samples]
    return items


def _load_crop_boxes(
    crop_boxes_path: Path,
) -> dict[str, tuple[float, float, float, float]]:
    """Load rectified crop boxes keyed by resolved absolute image path."""
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "x0", "y0", "x1", "y1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Crop-box CSV must contain {sorted(required)}: {crop_boxes_path}"
            )
        for row in reader:
            boxes[_resolve_repo_path(row["image_path"]).as_posix()] = (
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
    return boxes


def _load_reference_predictions(
    reference_predictions_path: Path,
) -> dict[str, float]:
    """Load saved reference predictions keyed by repo-relative image path."""
    reference: dict[str, float] = {}
    with reference_predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return reference
        if "image_path" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            return reference
        for row in reader:
            image_path = str(row["image_path"])
            reference[image_path] = float(row["prediction"])
            reference[_resolve_repo_path(image_path).as_posix()] = float(row["prediction"])
    return reference


def _quantize_unit_float_to_int8(tensor: np.ndarray) -> np.ndarray:
    """Match the firmware quantization rule q = round(x * 255) - 128."""
    quantized = np.floor(np.asarray(tensor, dtype=np.float32) * 255.0 + 0.5) - 128.0
    return np.clip(quantized, -128.0, 127.0).astype(np.int8)


def _dequantize_int8_to_unit_float(tensor: np.ndarray) -> np.ndarray:
    """Invert the firmware quantization rule back into float32 [0, 1] space."""
    return (np.asarray(tensor, dtype=np.float32) + 128.0) / 255.0


def _predict_exact(
    model: tf.keras.Model,
    tensor: np.ndarray,
    gauge_spec: Any,
) -> tuple[float, str]:
    """Run the exact V28 model and decode the logits into Celsius."""
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    prediction = decode_circular_vote_logits(logits, gauge_spec)
    return float(prediction), probe_tensor("v28_logits", logits).crc32_hex


def _tensor_diff_stats(training_tensor: np.ndarray, firmware_tensor: np.ndarray) -> tuple[float, float]:
    """Return mean and max absolute differences for two matching int8 tensors."""
    diff = np.abs(training_tensor.astype(np.int16) - firmware_tensor.astype(np.int16))
    return float(np.mean(diff)), float(np.max(diff))


def _evaluate_one(
    item: EvalItem,
    *,
    model: tf.keras.Model,
    crop_boxes: dict[str, tuple[float, float, float, float]],
    reference_predictions: dict[str, float],
    gauge_spec: Any,
    image_size: int,
    center_search_px: int,
    center_mode: str,
) -> RowResult:
    """Evaluate one capture with the exact offline and firmware-style paths."""
    source_image, source_kind = load_capture_image(
        item.image_path,
        image_width=image_size,
        image_height=image_size,
    )
    crop_box = crop_boxes.get(
        item.image_path.as_posix(),
        firmware_training_crop_box(source_image.shape[1], source_image.shape[0]),
    )

    training_tensor = build_training_style_polar_vote_float32(
        source_image,
        crop_box_xyxy=crop_box,
        output_dim=image_size,
        input_mode="rgb_edge6_vote7",
        center_search_px=center_search_px,
        center_mode=center_mode,
        gauge_spec=gauge_spec,
    )
    training_tensor_int8 = _quantize_unit_float_to_int8(training_tensor)
    firmware_tensor = build_legacy_firmware_polar_vote_tensor(
        source_image,
        crop_box_xyxy=crop_box,
        output_dim=image_size,
    )
    mean_abs_diff, max_abs_diff = _tensor_diff_stats(training_tensor_int8, firmware_tensor)

    training_prediction, training_output_crc32 = _predict_exact(
        model,
        training_tensor,
        gauge_spec,
    )
    firmware_prediction, firmware_output_crc32 = _predict_exact(
        model,
        _dequantize_int8_to_unit_float(firmware_tensor),
        gauge_spec,
    )

    reference_prediction = reference_predictions.get(item.image_path.as_posix())
    if reference_prediction is None:
        reference_prediction = reference_predictions.get(str(item.image_path))

    reference_abs_error = (
        abs(reference_prediction - item.value)
        if reference_prediction is not None
        else None
    )

    return RowResult(
        image_path=item.image_path.as_posix(),
        source_kind=source_kind,
        value=item.value,
        sample_weight=item.sample_weight,
        crop_box_x0=float(crop_box[0]),
        crop_box_y0=float(crop_box[1]),
        crop_box_x1=float(crop_box[2]),
        crop_box_y1=float(crop_box[3]),
        training_prediction=training_prediction,
        firmware_prediction=firmware_prediction,
        reference_prediction=reference_prediction,
        training_abs_error=abs(training_prediction - item.value),
        firmware_abs_error=abs(firmware_prediction - item.value),
        reference_abs_error=reference_abs_error,
        prediction_delta=float(firmware_prediction - training_prediction),
        reference_delta=(
            float(training_prediction - reference_prediction)
            if reference_prediction is not None
            else None
        ),
        tensor_mean_abs_diff=mean_abs_diff,
        tensor_max_abs_diff=max_abs_diff,
        training_input_crc32=probe_tensor("training_input", training_tensor_int8).crc32_hex,
        firmware_input_crc32=probe_tensor("firmware_input", firmware_tensor).crc32_hex,
        training_output_crc32=training_output_crc32,
        firmware_output_crc32=firmware_output_crc32,
    )


def _summarize(
    rows: list[RowResult],
    manifest_path: Path,
    crop_boxes_path: Path,
    weights_path: Path,
    gauge_id: str,
) -> dict[str, Any]:
    """Aggregate row-wise results into a compact metrics payload."""
    values = np.array([row.value for row in rows], dtype=np.float32)
    weights = np.array([row.sample_weight for row in rows], dtype=np.float32)
    training_preds = np.array([row.training_prediction for row in rows], dtype=np.float32)
    firmware_preds = np.array([row.firmware_prediction for row in rows], dtype=np.float32)
    training_abs = np.array([row.training_abs_error for row in rows], dtype=np.float32)
    firmware_abs = np.array([row.firmware_abs_error for row in rows], dtype=np.float32)
    tensor_mean_abs = np.array([row.tensor_mean_abs_diff for row in rows], dtype=np.float32)
    tensor_max_abs = np.array([row.tensor_max_abs_diff for row in rows], dtype=np.float32)

    weighted_total = float(np.sum(weights))
    if weighted_total <= 0.0:
        weighted_total = float(len(rows))
        weights = np.ones_like(weights)

    def _weighted_mean(series: np.ndarray) -> float:
        return float(np.sum(series * weights) / weighted_total)

    reference_deltas = [row.reference_delta for row in rows if row.reference_delta is not None]
    reference_abs = [row.reference_abs_error for row in rows if row.reference_abs_error is not None]

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "crop_boxes": str(crop_boxes_path),
        "weights": str(weights_path),
        "gauge_id": gauge_id,
        "samples": int(len(rows)),
        "training_mae": float(np.mean(training_abs)),
        "training_rmse": float(np.sqrt(np.mean(training_abs**2))),
        "training_weighted_mae": _weighted_mean(training_abs),
        "firmware_mae": float(np.mean(firmware_abs)),
        "firmware_rmse": float(np.sqrt(np.mean(firmware_abs**2))),
        "firmware_weighted_mae": _weighted_mean(firmware_abs),
        "training_bias": float(np.mean(training_preds - values)),
        "firmware_bias": float(np.mean(firmware_preds - values)),
        "firmware_vs_training_mae": float(np.mean(np.abs(firmware_preds - training_preds))),
        "tensor_mean_abs_diff": float(np.mean(tensor_mean_abs)),
        "tensor_max_abs_diff": float(np.max(tensor_max_abs)),
    }
    if reference_deltas:
        summary["training_vs_reference_mae"] = float(np.mean(np.abs(np.asarray(reference_deltas, dtype=np.float32))))
    else:
        summary["training_vs_reference_mae"] = None
    if reference_abs:
        summary["reference_mae"] = float(np.mean(np.asarray(reference_abs, dtype=np.float32)))
    else:
        summary["reference_mae"] = None
    return summary


def _write_outputs(
    rows: list[RowResult],
    metrics: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    """Write the per-sample CSV and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    """Run the exact offline V28 parity evaluation."""
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = _load_manifest(args.manifest, args.capture_root)
    crop_boxes = _load_crop_boxes(args.crop_boxes)
    if not args.allow_missing_crop_boxes:
        manifest_items = [
            item for item in manifest_items if item.image_path.as_posix() in crop_boxes
        ]
    selected_items = _select_items(
        manifest_items,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not selected_items:
        raise SystemExit("No samples were selected for evaluation.")

    print(f"[V28] Loaded {len(selected_items)} capture samples from {args.manifest}", flush=True)
    print(f"[V28] Weights: {args.weights}", flush=True)
    print(f"[V28] Crop boxes: {args.crop_boxes}", flush=True)

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights checkpoint not found: {args.weights}")

    gauge_spec = load_gauge_specs()[args.gauge_id]

    model = build_polar_vote_v28_model(
        polar_size=args.image_size,
        input_channels=7,
        base_filters=32,
        head_units=128,
        dropout=0.2,
    )
    model.load_weights(str(args.weights))

    reference_predictions: dict[str, float] = {}
    if args.reference_predictions.exists():
        reference_predictions = _load_reference_predictions(args.reference_predictions)
        print(
            f"[V28] Reference predictions: {args.reference_predictions}",
            flush=True,
        )

    rows: list[RowResult] = []
    for index, item in enumerate(selected_items, start=1):
        row = _evaluate_one(
            item,
            model=model,
            crop_boxes=crop_boxes,
            reference_predictions=reference_predictions,
            gauge_spec=gauge_spec,
            image_size=args.image_size,
            center_search_px=args.center_search_px,
            center_mode=args.center_mode,
        )
        rows.append(row)

        ref_text = ""
        if row.reference_prediction is not None:
            ref_text = f" ref={row.reference_prediction:7.3f}"
        print(
            f"[V28] {index:03d}/{len(selected_items):03d} {Path(row.image_path).name}: "
            f"true={row.value:6.2f} train={row.training_prediction:7.3f} "
            f"firmware={row.firmware_prediction:7.3f}{ref_text} "
            f"tensor_mae={row.tensor_mean_abs_diff:6.2f}",
            flush=True,
        )

    metrics = _summarize(
        rows,
        args.manifest,
        args.crop_boxes,
        args.weights,
        args.gauge_id,
    )
    _write_outputs(rows, metrics, output_dir=output_dir)

    print("[V28] Summary:", flush=True)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"[V28]   {key}: {value:.6f}", flush=True)
        else:
            print(f"[V28]   {key}: {value}", flush=True)
    print(f"[V28] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
