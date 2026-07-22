#!/usr/bin/env python3
"""Train a QAT-friendly full-frame OBB box model on the grouped manifest.

The goal is a deployable first-stage localizer that can find the gauge anywhere
in the field of view. We now train on the clean pxl box labels first, apply
translation and photometric augmentation in full-frame canvas space, and export
an int8 model with the same ``conf`` + ``box`` outputs used by the OBB crop
manifest generator. Board captures are kept for later evaluation and fine-tune
passes once the pxl baseline is stable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf

_GPU_MEMORY_LIMIT_MB = 15000
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
        )
    except Exception:
        pass

import tf_keras as keras
import tensorflow_model_optimization as tfmot

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import resize_with_pad_rgb  # noqa: E402
from embedded_gauge_reading_tinyml.board_pipeline import load_capture_image  # noqa: E402
from embedded_gauge_reading_tinyml.obb_simcc_tf_models import (  # noqa: E402
    build_mobilenetv2_obb_box_model,
)

DEFAULT_MANIFEST: Path = PROJECT_ROOT / "data" / "labelled_captured_images.json"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "artifacts" / "training" / "obb_box_grouped_qat"
IMAGE_SIZE: int = 224
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_WARMUP_EPOCHS: int = 10
DEFAULT_FINETUNE_EPOCHS: int = 8
DEFAULT_QAT_EPOCHS: int = 8
DEFAULT_ALPHA: float = 0.35
DEFAULT_SPATIAL_CHANNELS: int = 64
DEFAULT_HEAD_UNITS: int = 96
DEFAULT_HEAD_DROPOUT: float = 0.15
DEFAULT_LEARNING_RATE: float = 6e-4
DEFAULT_QAT_LEARNING_RATE: float = 1e-4
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.10
DEFAULT_TRANSLATION_RATIO: float = 0.22
DEFAULT_SCALE_JITTER: float = 0.12
DEFAULT_BOARD_SAMPLE_WEIGHT: float = 3.0
DEFAULT_SOURCE_KINDS: tuple[str, ...] = ("pxl_geometry",)


@dataclass(frozen=True, slots=True)
class OBBBoxSample:
    """One grouped-manifest image and its axis-aligned gauge-face box."""

    image_path: Path
    source_width: int
    source_height: int
    box_xyxy: tuple[float, float, float, float]
    is_board: bool
    source_kind: str


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the OBB training job."""
    parser = argparse.ArgumentParser(description="Train a grouped-manifest OBB box model with QAT.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--spatial-channels", type=int, default=DEFAULT_SPATIAL_CHANNELS)
    parser.add_argument("--head-units", type=int, default=DEFAULT_HEAD_UNITS)
    parser.add_argument("--head-dropout", type=float, default=DEFAULT_HEAD_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--qat-learning-rate", type=float, default=DEFAULT_QAT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--translation-ratio",
        type=float,
        default=DEFAULT_TRANSLATION_RATIO,
        help="Maximum random translation as a fraction of the 224x224 canvas.",
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        default=DEFAULT_SCALE_JITTER,
        help="Maximum isotropic scale jitter around the canvas center.",
    )
    parser.add_argument(
        "--board-sample-weight",
        type=float,
        default=DEFAULT_BOARD_SAMPLE_WEIGHT,
        help="Relative sample weight for board captures during training.",
    )
    parser.add_argument(
        "--source-kinds",
        nargs="+",
        default=list(DEFAULT_SOURCE_KINDS),
        help="Source kinds to keep from the grouped manifest, e.g. pxl_geometry reviewed_geometry.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    """Coerce one JSON scalar into a finite float if possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _is_original_capture(filename: str) -> bool:
    """Skip preview derivatives and other non-source images."""
    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True


def _load_source_size(image_path: Path, annotations: list[dict[str, Any]]) -> tuple[int, int]:
    """Resolve the source dimensions from labels or from the image itself."""
    for annotation in annotations:
        row = annotation["source_row"]
        source_width = _as_float(row.get("source_width"))
        source_height = _as_float(row.get("source_height"))
        if source_width is not None and source_height is not None:
            width = int(round(source_width))
            height = int(round(source_height))
            if width > 0 and height > 0:
                return width, height

    absolute_path = (REPO_ROOT / image_path).resolve() if not image_path.is_absolute() else image_path
    suffix = absolute_path.suffix.lower()
    if suffix == ".yuv422":
        file_size = absolute_path.stat().st_size
        inferred_pixels = file_size / 2.0
        inferred_dim = int(round(math.sqrt(inferred_pixels)))
        if inferred_dim > 0 and inferred_dim * inferred_dim * 2 == file_size:
            return inferred_dim, inferred_dim
        raise ValueError(f"{image_path} is a raw YUV capture without source dimensions.")

    with Image.open(absolute_path) as image:
        width, height = image.size
    return int(width), int(height)


def _manifest_rows(
    manifest_path: Path,
    *,
    allowed_source_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the grouped manifest and keep only source captures."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in payload["images"]:
        image_path = Path(str(entry["image_path"]))
        if not _is_original_capture(image_path.name):
            continue
        annotations = list(entry["annotations"])
        source_width, source_height = _load_source_size(image_path, annotations)
        source_row: dict[str, Any] | None = None
        source_kind = ""
        for annotation in annotations:
            annotation_kind = str(annotation.get("source_kind", "")).strip()
            candidate = annotation["source_row"]
            crop_x_min = _as_float(candidate.get("crop_x_min"))
            crop_y_min = _as_float(candidate.get("crop_y_min"))
            crop_x_max = _as_float(candidate.get("crop_x_max"))
            crop_y_max = _as_float(candidate.get("crop_y_max"))
            if None not in (crop_x_min, crop_y_min, crop_x_max, crop_y_max):
                source_row = candidate
                source_kind = annotation_kind
                break
            center_x = _as_float(candidate.get("center_x_source"))
            center_y = _as_float(candidate.get("center_y_source"))
            radius = _as_float(candidate.get("dial_radius_source"))
            if None not in (center_x, center_y, radius):
                source_row = candidate
                source_kind = annotation_kind
                crop_x_min = center_x - radius
                crop_y_min = center_y - radius
                crop_x_max = center_x + radius
                crop_y_max = center_y + radius
                break
        if source_row is None:
            continue
        if source_kind == "":
            continue
        if allowed_source_kinds is not None and source_kind not in allowed_source_kinds:
            continue
        rows.append(
            {
                "image_path": image_path,
                "source_width": int(source_width),
                "source_height": int(source_height),
                "box_xyxy": (
                    float(crop_x_min),
                    float(crop_y_min),
                    float(crop_x_max),
                    float(crop_y_max),
                ),
                "is_board": "/captured_images/" in image_path.as_posix(),
                "source_kind": source_kind,
            }
        )
    return rows


def _fullframe_box_to_canvas_norm(
    box_xyxy: tuple[float, float, float, float],
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Map a source-space box into the resized 224x224 canvas."""
    x_min, y_min, x_max, y_max = box_xyxy
    scale = min(float(IMAGE_SIZE) / float(source_width), float(IMAGE_SIZE) / float(source_height))
    scaled_w = float(source_width) * scale
    scaled_h = float(source_height) * scale
    pad_x = 0.5 * (float(IMAGE_SIZE) - scaled_w)
    pad_y = 0.5 * (float(IMAGE_SIZE) - scaled_h)
    canvas_cx = (((x_min + x_max) * 0.5) * scale + pad_x) / float(IMAGE_SIZE)
    canvas_cy = (((y_min + y_max) * 0.5) * scale + pad_y) / float(IMAGE_SIZE)
    canvas_w = ((x_max - x_min) * scale) / float(IMAGE_SIZE)
    canvas_h = ((y_max - y_min) * scale) / float(IMAGE_SIZE)
    return np.array(
        [
            np.clip(canvas_cx, 0.0, 1.0),
            np.clip(canvas_cy, 0.0, 1.0),
            np.clip(canvas_w, 0.0, 1.0),
            np.clip(canvas_h, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _translate_canvas(
    image: np.ndarray,
    box: np.ndarray,
    *,
    max_shift_px: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a canvas image and update the normalized box center."""
    if max_shift_px <= 0:
        return image, box

    shift_x = int(rng.integers(-max_shift_px, max_shift_px + 1))
    shift_y = int(rng.integers(-max_shift_px, max_shift_px + 1))
    if shift_x == 0 and shift_y == 0:
        return image, box

    shifted = np.zeros_like(image)
    src_x0 = max(0, -shift_x)
    src_y0 = max(0, -shift_y)
    dst_x0 = max(0, shift_x)
    dst_y0 = max(0, shift_y)
    width = IMAGE_SIZE - abs(shift_x)
    height = IMAGE_SIZE - abs(shift_y)
    if width > 0 and height > 0:
        shifted[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = image[
            src_y0 : src_y0 + height,
            src_x0 : src_x0 + width,
        ]

    updated = box.copy()
    updated[0] = float(np.clip(updated[0] + (shift_x / float(IMAGE_SIZE)), 0.0, 1.0))
    updated[1] = float(np.clip(updated[1] + (shift_y / float(IMAGE_SIZE)), 0.0, 1.0))
    return shifted, updated


def _scale_canvas(
    image: np.ndarray,
    box: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a canvas image around its center and update the normalized box."""
    if not math.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-3:
        return image, box

    resized_size = max(1, int(round(IMAGE_SIZE * scale)))
    pil_image = Image.fromarray(image)
    scaled_image = np.asarray(
        pil_image.resize((resized_size, resized_size), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    if resized_size >= IMAGE_SIZE:
        offset = (resized_size - IMAGE_SIZE) // 2
        scaled_image = scaled_image[offset : offset + IMAGE_SIZE, offset : offset + IMAGE_SIZE]
    else:
        canvas = np.zeros_like(image)
        offset = (IMAGE_SIZE - resized_size) // 2
        canvas[offset : offset + resized_size, offset : offset + resized_size] = scaled_image
        scaled_image = canvas

    center_px = 0.5 * float(IMAGE_SIZE)
    updated = box.copy()
    center_x_px = float(updated[0] * IMAGE_SIZE)
    center_y_px = float(updated[1] * IMAGE_SIZE)
    width_px = float(updated[2] * IMAGE_SIZE)
    height_px = float(updated[3] * IMAGE_SIZE)
    center_x_px = (center_x_px - center_px) * scale + center_px
    center_y_px = (center_y_px - center_px) * scale + center_px
    width_px *= scale
    height_px *= scale
    updated[0] = float(np.clip(center_x_px / float(IMAGE_SIZE), 0.0, 1.0))
    updated[1] = float(np.clip(center_y_px / float(IMAGE_SIZE), 0.0, 1.0))
    updated[2] = float(np.clip(width_px / float(IMAGE_SIZE), 0.05, 1.0))
    updated[3] = float(np.clip(height_px / float(IMAGE_SIZE), 0.05, 1.0))
    return scaled_image, updated


def _photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply light photometric jitter while keeping the geometry intact."""
    image_f = image.astype(np.float32)
    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.90, 1.10))
    noise = rng.normal(0.0, 2.0, size=image_f.shape).astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)
    image_f = (image_f - mean) * contrast + mean
    image_f = image_f * brightness + noise
    return np.clip(image_f, 0.0, 255.0).astype(np.uint8)


def _load_canvas_image(sample: OBBBoxSample) -> np.ndarray:
    """Load one source capture and resize it onto the shared canvas."""
    absolute_path = (REPO_ROOT / sample.image_path).resolve() if not sample.image_path.is_absolute() else sample.image_path
    source_image, _kind = load_capture_image(
        absolute_path,
        image_width=sample.source_width,
        image_height=sample.source_height,
    )
    return np.asarray(
        resize_with_pad_rgb(
            source_image,
            (0.0, 0.0, float(sample.source_width), float(sample.source_height)),
            image_size=IMAGE_SIZE,
        ),
        dtype=np.uint8,
    )


class OBBBoxSequence(keras.utils.Sequence):
    """Load grouped-manifest OBB samples into a small in-memory training queue."""

    def __init__(
        self,
        samples: list[OBBBoxSample],
        *,
        batch_size: int,
        augment: bool,
        seed: int,
        translation_ratio: float,
        scale_jitter: float,
        board_sample_weight: float,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.translation_ratio = float(translation_ratio)
        self.scale_jitter = float(scale_jitter)
        self.board_sample_weight = float(board_sample_weight)
        self.indices = np.arange(len(self.samples), dtype=np.int32)
        self._epoch = 0
        self._image_cache = [_load_canvas_image(sample) for sample in self.samples]
        self._box_cache = [
            _fullframe_box_to_canvas_norm(sample.box_xyxy, source_width=sample.source_width, source_height=sample.source_height)
            for sample in self.samples
        ]
        self._weight_cache = [
            np.float32(self.board_sample_weight if sample.is_board else 1.0)
            for sample in self.samples
        ]

    def __len__(self) -> int:
        return int(math.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.samples))
        batch_indices = self.indices[start:stop]
        images: list[np.ndarray] = []
        conf_targets: list[np.ndarray] = []
        box_targets: list[np.ndarray] = []
        conf_weights: list[np.ndarray] = []
        box_weights: list[np.ndarray] = []

        for order_index, sample_index in enumerate(batch_indices):
            sample_idx = int(sample_index)
            sample = self.samples[sample_idx]
            rng = np.random.default_rng(self.seed + index * 1009 + sample_idx * 97 + order_index)
            image = self._image_cache[sample_idx]
            box = self._box_cache[sample_idx]
            if self.augment:
                image, box = _translate_canvas(
                    image,
                    box,
                    max_shift_px=max(1, int(round(self.translation_ratio * IMAGE_SIZE))),
                    rng=rng,
                )
                scale = float(rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
                image, box = _scale_canvas(image, box, scale=scale)
                image = _photometric_augment(image, rng)
            image_f = image.astype(np.float32) / 127.5 - 1.0
            images.append(image_f)
            conf_targets.append(np.array([1.0], dtype=np.float32))
            box_targets.append(box.astype(np.float32))
            weight = np.array(self._weight_cache[sample_idx], dtype=np.float32)
            conf_weights.append(weight)
            box_weights.append(weight)

        x = np.stack(images, axis=0).astype(np.float32)
        y = {
            "conf": np.stack(conf_targets, axis=0).astype(np.float32),
            "box": np.stack(box_targets, axis=0).astype(np.float32),
        }
        w = {
            "conf": np.asarray(conf_weights, dtype=np.float32).reshape(-1),
            "box": np.asarray(box_weights, dtype=np.float32).reshape(-1),
        }
        return x, y, w

    def on_epoch_end(self) -> None:
        """Shuffle training order after each epoch when augmentation is enabled."""
        if not self.augment or len(self.indices) == 0:
            return
        rng = np.random.default_rng(self.seed + self._epoch)
        rng.shuffle(self.indices)
        self._epoch += 1


def _weighted_huber(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    weights: tuple[float, float, float, float],
    delta: float,
) -> tf.Tensor:
    """Compute a per-dimension weighted Huber loss for box regression."""
    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    per_dim = 0.5 * tf.square(quadratic) / delta + linear
    weighted = per_dim * tf.constant(weights, dtype=tf.float32)[tf.newaxis, :]
    return tf.reduce_mean(weighted, axis=-1)


def _load_model(pretrained: bool, *, alpha: float, spatial_channels: int, head_units: int, head_dropout: float) -> keras.Model:
    """Build the full-frame box model."""
    return build_mobilenetv2_obb_box_model(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        alpha=alpha,
        pretrained=pretrained,
        backbone_trainable=False,
        spatial_channels=spatial_channels,
        head_units=head_units,
        head_dropout=head_dropout,
    )


def _evaluate_sequence(model: keras.Model, sequence: OBBBoxSequence) -> dict[str, float]:
    """Evaluate box MAE and confidence on a sequence, split by board/pxl."""
    center_errors: list[float] = []
    size_errors: list[float] = []
    conf_errors: list[float] = []
    board_center_errors: list[float] = []
    board_size_errors: list[float] = []
    pxl_center_errors: list[float] = []
    pxl_size_errors: list[float] = []

    for batch_index in range(len(sequence)):
        batch_x, batch_y, _batch_w = sequence[batch_index]
        predictions = model.predict(batch_x, verbose=0)
        if not isinstance(predictions, dict):
            predictions = {name: predictions[i] for i, name in enumerate(model.output_names)}
        pred_box = np.asarray(predictions["box"], dtype=np.float32)
        pred_conf = np.asarray(predictions["conf"], dtype=np.float32).reshape(-1)
        true_box = np.asarray(batch_y["box"], dtype=np.float32)
        true_conf = np.asarray(batch_y["conf"], dtype=np.float32).reshape(-1)
        batch_center_error = 0.5 * (
            np.abs(pred_box[:, 0] - true_box[:, 0]) + np.abs(pred_box[:, 1] - true_box[:, 1])
        ) * float(IMAGE_SIZE - 1)
        batch_size_error = 0.5 * (
            np.abs(pred_box[:, 2] - true_box[:, 2]) + np.abs(pred_box[:, 3] - true_box[:, 3])
        ) * float(IMAGE_SIZE - 1)
        center_errors.extend(float(value) for value in batch_center_error)
        size_errors.extend(float(value) for value in batch_size_error)
        conf_errors.append(float(np.mean(np.abs(pred_conf - true_conf))))

        batch_start = batch_index * sequence.batch_size
        for offset, sample in enumerate(sequence.samples[batch_start : batch_start + len(batch_x)]):
            if sample.is_board:
                board_center_errors.append(float(batch_center_error[offset]))
                board_size_errors.append(float(batch_size_error[offset]))
            else:
                pxl_center_errors.append(float(batch_center_error[offset]))
                pxl_size_errors.append(float(batch_size_error[offset]))

    report = {
        "box_center_mae_px": float(np.mean(center_errors)) if center_errors else float("nan"),
        "box_size_mae_px": float(np.mean(size_errors)) if size_errors else float("nan"),
        "conf_mae": float(np.mean(conf_errors)) if conf_errors else float("nan"),
    }
    if board_center_errors:
        report["board_box_center_mae_px"] = float(np.mean(board_center_errors))
        report["board_box_size_mae_px"] = float(np.mean(board_size_errors))
    if pxl_center_errors:
        report["pxl_box_center_mae_px"] = float(np.mean(pxl_center_errors))
        report["pxl_box_size_mae_px"] = float(np.mean(pxl_size_errors))
    return report


def _quantize_model(model: keras.Model) -> keras.Model:
    """Wrap a trained model in a QAT clone."""
    with tfmot.quantization.keras.quantize_scope():
        return tfmot.quantization.keras.quantize_model(model)


def main() -> None:
    """Train, quantize, and export the field-robust OBB box model."""
    args = _parse_args()
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1).")
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in (0, 1).")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed_source_kinds = {str(kind).strip() for kind in args.source_kinds if str(kind).strip()}
    rows = _manifest_rows(args.manifest, allowed_source_kinds=allowed_source_kinds)
    if not rows:
        raise ValueError(f"No usable images were found in {args.manifest}.")

    samples = [
        OBBBoxSample(
            image_path=row["image_path"],
            source_width=int(row["source_width"]),
            source_height=int(row["source_height"]),
            box_xyxy=tuple(row["box_xyxy"]),
            is_board=bool(row["is_board"]),
            source_kind=str(row.get("source_kind", "")),
        )
        for row in rows
    ]

    split_labels = np.array([1 if sample.is_board else 0 for sample in samples], dtype=np.int32)
    sample_indices = np.arange(len(samples), dtype=np.int32)
    train_val_indices, test_indices = train_test_split(
        sample_indices,
        test_size=max(args.test_fraction, 0.05),
        random_state=args.seed,
        shuffle=True,
        stratify=split_labels if len(np.unique(split_labels)) > 1 else None,
    )
    relative_val_fraction = args.val_fraction / max(1.0 - args.test_fraction, 1e-6)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=max(relative_val_fraction, 0.05),
        random_state=args.seed + 1,
        shuffle=True,
        stratify=split_labels[train_val_indices] if len(np.unique(split_labels[train_val_indices])) > 1 else None,
    )

    train_samples = [samples[int(index)] for index in train_indices]
    val_samples = [samples[int(index)] for index in val_indices]
    test_samples = [samples[int(index)] for index in test_indices]

    train_sequence = OBBBoxSequence(
        train_samples,
        batch_size=args.batch_size,
        augment=True,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
    )
    val_sequence = OBBBoxSequence(
        val_samples,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
    )
    test_sequence = OBBBoxSequence(
        test_samples,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
    )

    source_kind_counts: dict[str, int] = {}
    for sample in samples:
        source_kind_counts[sample.source_kind] = source_kind_counts.get(sample.source_kind, 0) + 1
    print(
        f"Loaded {len(samples)} samples from {args.manifest} "
        f"({sum(source_kind_counts.values())} kept, source_kinds={sorted(allowed_source_kinds)})."
    )
    print(json.dumps({"source_kind_counts": source_kind_counts}, indent=2, sort_keys=True))
    print(
        f"Split sizes: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}."
    )

    model = _load_model(
        pretrained=True,
        alpha=args.alpha,
        spatial_channels=args.spatial_channels,
        head_units=args.head_units,
        head_dropout=args.head_dropout,
    )

    losses: dict[str, Any] = {
        "conf": keras.losses.BinaryCrossentropy(),
        "box": lambda y_true, y_pred: _weighted_huber(
            y_true,
            y_pred,
            weights=(2.0, 2.0, 1.0, 1.0),
            delta=0.04,
        ),
    }
    metrics = {
        "conf": [keras.metrics.MeanAbsoluteError(name="mae")],
        "box": [keras.metrics.MeanAbsoluteError(name="mae")],
    }

    def _compile_for_lr(learning_rate: float) -> None:
        """Compile the model for one training phase."""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=losses,
            loss_weights={"conf": 0.05, "box": 1.0},
            metrics=metrics,
        )

    _compile_for_lr(args.learning_rate)

    warmup_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(3, args.warmup_epochs // 2),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]
    print("Phase 1: warmup training...")
    model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.warmup_epochs,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    backbone_layers = getattr(model, "_mobilenet_backbone_layers", None)
    if isinstance(backbone_layers, dict):
        for layer in backbone_layers.values():
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    _compile_for_lr(args.learning_rate * 0.5)
    print("Phase 2: fine-tuning the backbone...")
    model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.finetune_epochs,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    print("Phase 3: QAT fine-tuning...")
    qat_model = _quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.qat_learning_rate, clipnorm=1.0),
        loss=losses,
        loss_weights={"conf": 0.05, "box": 1.0},
        metrics=metrics,
    )
    qat_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(2, args.qat_epochs // 2),
            restore_best_weights=True,
        ),
    ]
    qat_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.qat_epochs,
        callbacks=qat_callbacks,
        verbose=2,
    )

    test_report = _evaluate_sequence(qat_model, test_sequence)
    print(json.dumps(test_report, indent=2, sort_keys=True))

    float_model_path = args.output_dir / "obb_box_float.keras"
    qat_model_path = args.output_dir / "obb_box_qat.keras"
    qat_tflite_path = args.output_dir / "obb_box_qat.tflite"
    model.save(float_model_path)
    qat_model.save(qat_model_path)

    representative_samples = train_sequence[0][0][:4]
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([np.asarray(sample[np.newaxis, ...], dtype=np.float32)] for sample in representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    qat_tflite_path.write_bytes(tflite_bytes)

    summary = {
        "manifest": args.manifest.as_posix(),
        "sample_count": len(samples),
        "board_count": int(sum(1 for sample in samples if sample.is_board)),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "float_model_path": float_model_path.as_posix(),
        "qat_model_path": qat_model_path.as_posix(),
        "qat_tflite_path": qat_tflite_path.as_posix(),
        "test_report": test_report,
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "spatial_channels": args.spatial_channels,
            "head_units": args.head_units,
            "head_dropout": args.head_dropout,
            "learning_rate": args.learning_rate,
            "qat_learning_rate": args.qat_learning_rate,
            "translation_ratio": args.translation_ratio,
            "scale_jitter": args.scale_jitter,
            "board_sample_weight": args.board_sample_weight,
            "source_kinds": sorted(allowed_source_kinds),
        },
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
