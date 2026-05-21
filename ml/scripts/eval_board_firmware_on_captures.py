"""Evaluate the live-board firmware path on labeled capture samples.

This script compares two pipelines on the same captures:
- the firmware-style 7-channel polar vote tensor that mirrors the STM32 path,
- a conventional 3-channel offline scalar crop path.

It is meant to answer the practical question: "does the board behave like the
offline model on the same captured_images set?"
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import sys
from typing import Any, Final

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

import tensorflow as tf

from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    POLAR_VOTE_BINS,
    POLAR_VOTE_MAX_VALUE_C,
    POLAR_VOTE_MIN_VALUE_C,
    build_firmware_polar_vote_tensor,
    decode_circular_vote_logits,
    firmware_training_crop_box,
    load_capture_image,
    resize_with_pad_rgb,
    probe_tensor,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs  # noqa: E402

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "weighted_full_range_v2.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "board_firmware_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"
DEFAULT_FIRMWARE_MODEL: Final[Path] = (
    PROJECT_ROOT / "artifacts" / "deployment" / "polar_vote_circular_v28_int8" / "model_int8.tflite"
)
DEFAULT_OFFLINE_MODEL: Final[Path] = (
    PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.6_scalar_int8" / "model_int8.tflite"
)


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    """Loaded TFLite runtime and the tensor metadata needed to feed it."""

    model_path: Path
    interpreter: tf.lite.Interpreter
    input_details: dict[str, Any]
    output_details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One per-sample comparison row for the output CSV."""

    image_path: str
    source_kind: str
    value: float
    sample_weight: float
    firmware_prediction: float
    firmware_abs_error: float
    offline_prediction: float
    offline_abs_error: float
    firmware_input_crc32: str
    offline_input_crc32: str
    firmware_output_crc32: str
    offline_output_crc32: str


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the STM32 firmware-style polar pipeline on capture images."
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
        "--firmware-model",
        type=Path,
        default=DEFAULT_FIRMWARE_MODEL,
        help="Firmware-style 7-channel TFLite model.",
    )
    parser.add_argument(
        "--offline-model",
        type=Path,
        default=DEFAULT_OFFLINE_MODEL,
        help="Offline scalar TFLite model for comparison.",
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
        default=32,
        help="Maximum number of samples to evaluate after filtering.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the filtered samples before truncating to --max-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used when --shuffle is enabled.",
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


def _load_runtime(model_path: Path) -> ModelRuntime:
    """Load one quantized TFLite model and its tensor metadata."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return ModelRuntime(
        model_path=model_path,
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
    )


def _prepare_tflite_input(
    batch: np.ndarray,
    input_details: dict[str, Any],
) -> np.ndarray:
    """Convert a numpy batch into the dtype expected by a TFLite interpreter."""
    input_dtype = np.dtype(input_details["dtype"])
    batch_array = np.ascontiguousarray(batch)

    if batch_array.dtype == input_dtype:
        return batch_array

    if np.issubdtype(input_dtype, np.floating):
        return batch_array.astype(input_dtype, copy=False)

    if input_dtype == np.uint8 and batch_array.dtype == np.int8:
        return (batch_array.astype(np.int16) + 128).astype(np.uint8)
    if input_dtype == np.int8 and batch_array.dtype == np.uint8:
        return (batch_array.astype(np.int16) - 128).astype(np.int8)

    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(input_dtype).min, np.iinfo(input_dtype).max
    quantized = np.round(batch_array.astype(np.float32) / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(input_dtype)


def _dequantize_output(
    output_tensor: np.ndarray,
    output_details: dict[str, Any],
) -> np.ndarray:
    """Convert a quantized output tensor back into float32."""
    output_dtype = np.dtype(output_details["dtype"])
    if np.issubdtype(output_dtype, np.floating):
        return np.asarray(output_tensor, dtype=np.float32)

    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return scale * (np.asarray(output_tensor, dtype=np.float32) - zero_point)


def _predict_temperature(
    runtime: ModelRuntime,
    input_batch: np.ndarray,
    gauge_spec: GaugeSpec,
) -> tuple[float, TensorProbe, TensorProbe]:
    """Run one model and return the decoded temperature plus tensor probes."""
    prepared_input = _prepare_tflite_input(input_batch, runtime.input_details)
    runtime.interpreter.set_tensor(int(runtime.input_details["index"]), prepared_input)
    runtime.interpreter.invoke()
    raw_output = runtime.interpreter.get_tensor(int(runtime.output_details["index"]))

    input_probe = probe_tensor("input", prepared_input)
    output_probe = probe_tensor("output", raw_output)

    output_flat = np.asarray(raw_output).reshape(-1)
    if output_flat.size == POLAR_VOTE_BINS:
        output_dequant = _dequantize_output(raw_output, runtime.output_details).reshape(-1)
        prediction = decode_circular_vote_logits(
            output_dequant,
            gauge_spec,
            bins=POLAR_VOTE_BINS,
        )
        return prediction, input_probe, output_probe

    output_dequant = _dequantize_output(raw_output, runtime.output_details).reshape(-1)
    if output_dequant.size == 0:
        raise ValueError("Model output was empty.")
    return float(output_dequant[0]), input_probe, output_probe


def _evaluate_one(
    item: EvalItem,
    *,
    firmware_runtime: ModelRuntime,
    offline_runtime: ModelRuntime,
    gauge_spec: GaugeSpec,
    image_size: int,
) -> PredictionRow:
    """Evaluate one capture with both the firmware and offline paths."""
    source_image, source_kind = load_capture_image(
        item.image_path,
        image_width=image_size,
        image_height=image_size,
    )
    crop_box = firmware_training_crop_box(source_image.shape[1], source_image.shape[0])

    # Firmware path: 7-channel polar vote tensor that mirrors app_ai.c.
    firmware_tensor = build_firmware_polar_vote_tensor(
        source_image,
        crop_box_xyxy=crop_box,
        output_dim=image_size,
    )
    firmware_batch = np.expand_dims(firmware_tensor, axis=0)
    firmware_prediction, firmware_input_probe, firmware_output_probe = _predict_temperature(
        firmware_runtime,
        firmware_batch,
        gauge_spec,
    )

    # Offline comparison path: the conventional 3-channel crop that the
    # scalar models were trained around.
    offline_crop = resize_with_pad_rgb(source_image, crop_box, image_size=image_size)
    offline_batch = np.expand_dims(offline_crop.astype(np.float32) / 255.0, axis=0)
    offline_prediction, offline_input_probe, offline_output_probe = _predict_temperature(
        offline_runtime,
        offline_batch,
        gauge_spec,
    )

    return PredictionRow(
        image_path=str(item.image_path.relative_to(REPO_ROOT)),
        source_kind=source_kind,
        value=item.value,
        sample_weight=item.sample_weight,
        firmware_prediction=firmware_prediction,
        firmware_abs_error=abs(firmware_prediction - item.value),
        offline_prediction=offline_prediction,
        offline_abs_error=abs(offline_prediction - item.value),
        firmware_input_crc32=firmware_input_probe.crc32_hex,
        offline_input_crc32=offline_input_probe.crc32_hex,
        firmware_output_crc32=firmware_output_probe.crc32_hex,
        offline_output_crc32=offline_output_probe.crc32_hex,
    )


def _summarize(rows: list[PredictionRow], manifest_path: Path, gauge_spec: GaugeSpec) -> dict[str, Any]:
    """Aggregate the row-wise predictions into a compact metrics payload."""
    values = np.array([row.value for row in rows], dtype=np.float32)
    weights = np.array([row.sample_weight for row in rows], dtype=np.float32)
    firmware_preds = np.array([row.firmware_prediction for row in rows], dtype=np.float32)
    offline_preds = np.array([row.offline_prediction for row in rows], dtype=np.float32)
    firmware_abs = np.array([row.firmware_abs_error for row in rows], dtype=np.float32)
    offline_abs = np.array([row.offline_abs_error for row in rows], dtype=np.float32)

    weighted_total = float(np.sum(weights))
    if weighted_total <= 0.0:
        weighted_total = float(len(rows))
        weights = np.ones_like(weights)

    def _weighted_mean(series: np.ndarray) -> float:
        return float(np.sum(series * weights) / weighted_total)

    return {
        "manifest": str(manifest_path),
        "gauge_id": gauge_spec.gauge_id,
        "samples": int(len(rows)),
        "firmware_mae": float(np.mean(firmware_abs)),
        "firmware_rmse": float(np.sqrt(np.mean(firmware_abs**2))),
        "firmware_weighted_mae": _weighted_mean(firmware_abs),
        "offline_mae": float(np.mean(offline_abs)),
        "offline_rmse": float(np.sqrt(np.mean(offline_abs**2))),
        "offline_weighted_mae": _weighted_mean(offline_abs),
        "firmware_bias": float(np.mean(firmware_preds - values)),
        "offline_bias": float(np.mean(offline_preds - values)),
        "firmware_offline_mae": float(np.mean(np.abs(firmware_preds - offline_preds))),
        "value_min": float(POLAR_VOTE_MIN_VALUE_C),
        "value_max": float(POLAR_VOTE_MAX_VALUE_C),
    }


def _write_outputs(
    rows: list[PredictionRow],
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

    metrics_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    """Run the firmware-style evaluation on a slice of captured images."""
    args = _parse_args()
    gauge_specs = load_gauge_specs()
    gauge_spec = gauge_specs[args.gauge_id]

    items = _load_manifest(args.manifest, args.capture_root)
    items = _select_items(
        items,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not items:
        raise SystemExit(
            f"No samples matched {args.capture_root} in {args.manifest}. "
            "Check the manifest path or capture root."
        )

    print(f"[EVAL] Loaded {len(items)} capture samples from {args.manifest}", flush=True)
    print(f"[EVAL] Firmware model: {args.firmware_model}", flush=True)
    print(f"[EVAL] Offline model: {args.offline_model}", flush=True)

    firmware_runtime = _load_runtime(args.firmware_model)
    offline_runtime = _load_runtime(args.offline_model)

    rows: list[PredictionRow] = []
    for index, item in enumerate(items, start=1):
        row = _evaluate_one(
            item,
            firmware_runtime=firmware_runtime,
            offline_runtime=offline_runtime,
            gauge_spec=gauge_spec,
            image_size=args.image_size,
        )
        rows.append(row)
        print(
            f"[EVAL] {index:03d}/{len(items):03d} {Path(row.image_path).name}: "
            f"true={row.value:.1f} firmware={row.firmware_prediction:.2f} "
            f"offline={row.offline_prediction:.2f}",
            flush=True,
        )

    metrics = _summarize(rows, args.manifest, gauge_spec)
    _write_outputs(rows, metrics, output_dir=args.output_dir)

    print("[EVAL] Summary:", flush=True)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"[EVAL]   {key}: {value:.4f}", flush=True)
        else:
            print(f"[EVAL]   {key}: {value}", flush=True)
    print(f"[EVAL] Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
