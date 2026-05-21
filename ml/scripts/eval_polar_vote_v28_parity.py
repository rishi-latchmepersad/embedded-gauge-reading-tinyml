#!/usr/bin/env python3
"""Compare the offline V28 polar-vote tensor against the firmware tensor.

This evaluator answers a narrower question than the board replay scripts:
"If we run the exact training-style preprocessing and the current firmware
preprocessing on the same capture, how different are the tensors and the model
predictions?"

The output is designed to help debug board/firmware preprocessing drift before
we flash anything.
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
    build_training_style_polar_vote_tensor,
    decode_circular_vote_logits,
    firmware_training_crop_box,
    load_capture_image,
    probe_tensor,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs  # noqa: E402

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "weighted_full_range_v2.csv"
DEFAULT_CAPTURE_ROOT: Final[Path] = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_CROP_BOXES: Final[Path] = PROJECT_ROOT / "data" / "rectified_crop_boxes_v5_all.csv"
DEFAULT_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "polar_vote_circular_v28_int8"
    / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "polar_vote_v28_parity_eval"
DEFAULT_GAUGE_ID: Final[str] = "littlegood_home_temp_gauge_c"


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One labeled capture and its target value."""

    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True, slots=True)
class RowResult:
    """One scored comparison row."""

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
    training_abs_error: float
    firmware_abs_error: float
    prediction_delta: float
    tensor_mean_abs_diff: float
    tensor_max_abs_diff: float
    training_input_crc32: str
    firmware_input_crc32: str
    training_output_crc32: str
    firmware_output_crc32: str


class ReduceMeanAxis(tf.keras.layers.Layer):
    """Deserialize the custom mean-pool layer from the repacked V28 model."""

    def __init__(self, axis: int = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ReduceMaxAxis(tf.keras.layers.Layer):
    """Deserialize the custom max-pool layer from the repacked V28 model."""

    def __init__(self, axis: int = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_max(inputs, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare the offline V28 polar tensor against the firmware tensor."
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
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Quantized V28 polar-vote TFLite model.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["auto", "keras", "tflite"],
        default="auto",
        help="Model backend to use for the comparison run.",
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
        help="Maximum number of samples to score.",
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


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    """Loaded model plus the backend-specific object needed for inference."""

    kind: str
    interpreter: tf.lite.Interpreter | None = None
    keras_model: tf.keras.Model | None = None


def _load_model_runtime(model_path: Path, model_kind: str) -> ModelRuntime:
    """Load either the Keras source model or the int8 TFLite deployment."""
    chosen_kind = model_kind
    if chosen_kind == "auto":
        chosen_kind = "keras" if model_path.suffix.lower() == ".keras" else "tflite"

    if chosen_kind == "keras":
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "ReduceMeanAxis": ReduceMeanAxis,
                "ReduceMaxAxis": ReduceMaxAxis,
            },
            compile=False,
            safe_mode=False,
        )
        return ModelRuntime(kind="keras", keras_model=model)

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    return ModelRuntime(kind="tflite", interpreter=interpreter)


def _run_model(
    runtime: ModelRuntime,
    tensor: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Run one tensor through the chosen backend and return raw logits."""
    if runtime.kind == "keras":
        if runtime.keras_model is None:
            raise RuntimeError("Keras runtime is missing the loaded model.")
        input_tensor = (tensor.astype(np.float32) + 128.0) / 255.0
        pred_raw = runtime.keras_model.predict(input_tensor[None, ...], verbose=0)
        if isinstance(pred_raw, dict):
            pred_raw = pred_raw.get("gauge_value", next(iter(pred_raw.values())))
        output_tensor = np.asarray(pred_raw).reshape(-1)
        return output_tensor, probe_tensor("v28_output", output_tensor).crc32_hex

    if runtime.interpreter is None:
        raise RuntimeError("TFLite runtime is missing the loaded interpreter.")

    input_details = runtime.interpreter.get_input_details()[0]
    output_details = runtime.interpreter.get_output_details()[0]

    input_tensor = np.ascontiguousarray(tensor)
    if input_tensor.dtype != np.dtype(input_details["dtype"]):
        if input_details["dtype"] == np.int8 and input_tensor.dtype == np.uint8:
            input_tensor = (input_tensor.astype(np.int16) - 128).astype(np.int8)
        elif input_details["dtype"] == np.uint8 and input_tensor.dtype == np.int8:
            input_tensor = (input_tensor.astype(np.int16) + 128).astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(np.dtype(input_details["dtype"]))

    runtime.interpreter.set_tensor(int(input_details["index"]), input_tensor[None, ...])
    runtime.interpreter.invoke()
    output_tensor = runtime.interpreter.get_tensor(int(output_details["index"]))[0]
    return np.asarray(output_tensor), probe_tensor("v28_output", output_tensor).crc32_hex


def _tensor_diff_stats(training_tensor: np.ndarray, firmware_tensor: np.ndarray) -> tuple[float, float]:
    """Return mean and max absolute differences for two matching tensors."""
    diff = np.abs(training_tensor.astype(np.int16) - firmware_tensor.astype(np.int16))
    return float(np.mean(diff)), float(np.max(diff))


def main() -> None:
    """Entry point for the parity evaluator."""
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

    gauge_specs = load_gauge_specs()
    gauge_spec = gauge_specs[args.gauge_id]
    runtime = _load_model_runtime(args.model, args.model_kind)

    rows: list[RowResult] = []
    training_abs_errors: list[float] = []
    firmware_abs_errors: list[float] = []
    tensor_mean_abs_diffs: list[float] = []
    tensor_max_abs_diffs: list[float] = []
    training_predictions: list[float] = []
    firmware_predictions: list[float] = []
    hard_mask: list[bool] = []

    for item in selected_items:
        source_image, source_kind = load_capture_image(
            item.image_path,
            image_width=args.image_size,
            image_height=args.image_size,
        )
        crop_box = crop_boxes.get(
            item.image_path.as_posix(),
            firmware_training_crop_box(source_image.shape[1], source_image.shape[0]),
        )

        training_tensor = build_training_style_polar_vote_tensor(
            source_image,
            crop_box_xyxy=crop_box,
        )
        firmware_tensor = build_legacy_firmware_polar_vote_tensor(
            source_image,
            crop_box_xyxy=crop_box,
        )
        mean_abs_diff, max_abs_diff = _tensor_diff_stats(training_tensor, firmware_tensor)

        training_logits, training_output_crc32 = _run_model(runtime, training_tensor)
        firmware_logits, firmware_output_crc32 = _run_model(runtime, firmware_tensor)
        training_value = decode_circular_vote_logits(training_logits, gauge_spec)
        firmware_value = decode_circular_vote_logits(firmware_logits, gauge_spec)
        training_prediction = training_value
        firmware_prediction = firmware_value

        training_abs_error = abs(training_value - item.value)
        firmware_abs_error = abs(firmware_value - item.value)

        training_abs_errors.append(training_abs_error)
        firmware_abs_errors.append(firmware_abs_error)
        tensor_mean_abs_diffs.append(mean_abs_diff)
        tensor_max_abs_diffs.append(max_abs_diff)
        training_predictions.append(training_value)
        firmware_predictions.append(firmware_value)
        hard_mask.append(item.value <= -20.0 or item.value >= 40.0)

        rows.append(
            RowResult(
                image_path=item.image_path.as_posix(),
                source_kind=source_kind,
                value=item.value,
                sample_weight=item.sample_weight,
                crop_box_x0=float(crop_box[0]),
                crop_box_y0=float(crop_box[1]),
                crop_box_x1=float(crop_box[2]),
                crop_box_y1=float(crop_box[3]),
                training_prediction=float(training_value),
                firmware_prediction=float(firmware_value),
                training_abs_error=float(training_abs_error),
                firmware_abs_error=float(firmware_abs_error),
                prediction_delta=float(firmware_value - training_value),
                tensor_mean_abs_diff=float(mean_abs_diff),
                tensor_max_abs_diff=float(max_abs_diff),
                training_input_crc32=probe_tensor("training_input", training_tensor).crc32_hex,
                firmware_input_crc32=probe_tensor("firmware_input", firmware_tensor).crc32_hex,
                training_output_crc32=training_output_crc32,
                firmware_output_crc32=firmware_output_crc32,
            )
        )

        print(
            f"[PARITY] {item.image_path.name}: true={item.value:6.2f} "
            f"training={training_value:7.2f} firmware={firmware_value:7.2f} "
            f"tensor_mae={mean_abs_diff:7.2f} tensor_max={max_abs_diff:7.2f}"
        )
        print(
            f"[PARITY]     decode_hint training={training_prediction:7.2f} "
            f"firmware={firmware_prediction:7.2f}"
        )

    if not rows:
        raise ValueError("No samples were scored after filtering.")

    training_errors = np.asarray(training_abs_errors, dtype=np.float32)
    firmware_errors = np.asarray(firmware_abs_errors, dtype=np.float32)
    hard_mask_array = np.asarray(hard_mask, dtype=bool)
    tensor_mean = float(np.mean(np.asarray(tensor_mean_abs_diffs, dtype=np.float32)))
    tensor_max = float(np.max(np.asarray(tensor_max_abs_diffs, dtype=np.float32)))

    summary: dict[str, Any] = {
        "manifest": args.manifest.as_posix(),
        "crop_boxes": args.crop_boxes.as_posix(),
        "model": args.model.as_posix(),
        "samples": len(rows),
        "training_mae": float(np.mean(training_errors)),
        "firmware_mae": float(np.mean(firmware_errors)),
        "training_rmse": float(np.sqrt(np.mean(np.square(training_errors)))),
        "firmware_rmse": float(np.sqrt(np.mean(np.square(firmware_errors)))),
        "training_hard_mae": float(np.mean(training_errors[hard_mask_array])) if np.any(hard_mask_array) else None,
        "firmware_hard_mae": float(np.mean(firmware_errors[hard_mask_array])) if np.any(hard_mask_array) else None,
        "tensor_mean_abs_diff": tensor_mean,
        "tensor_max_abs_diff": tensor_max,
        "training_prediction_std": float(np.std(np.asarray(training_predictions, dtype=np.float32))),
        "firmware_prediction_std": float(np.std(np.asarray(firmware_predictions, dtype=np.float32))),
    }

    print("\n=== Parity Summary ===")
    for key in [
        "samples",
        "training_mae",
        "firmware_mae",
        "training_rmse",
        "firmware_rmse",
        "tensor_mean_abs_diff",
        "tensor_max_abs_diff",
        "training_prediction_std",
        "firmware_prediction_std",
    ]:
        print(f"{key}={summary[key]}")
    if summary["training_hard_mae"] is not None:
        print(f"training_hard_mae={summary['training_hard_mae']}")
    if summary["firmware_hard_mae"] is not None:
        print(f"firmware_hard_mae={summary['firmware_hard_mae']}")

    rows_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PARITY] Wrote {rows_path}")
    print(f"[PARITY] Wrote {summary_path}")


if __name__ == "__main__":
    main()
