#!/usr/bin/env python3
"""Train a compact needle-direction model for board gauge captures.

This script replaces the heavier center/tip heatmap path with a direct 2D
needle direction regressor. The model learns a unit vector from the gauge
center to the needle tip, which is easier to quantize and cheaper to deploy on
the STM32 N6 board.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_model_optimization as tfmot

_GPU_MEMORY_LIMIT_MB = 3900
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
        )
    except Exception:
        pass

keras = tf.keras

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_pipeline import load_capture_image  # noqa: E402
from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    resolve_absolute_image_path,
    to_repo_relative_path,
)
from embedded_gauge_reading_tinyml.models_geometry import (  # noqa: E402
    build_qat_friendly_needle_direction_geometry_model,
    set_needle_direction_encoder_trainable,
)
from embedded_gauge_reading_tinyml.obb_crop_manifest import (  # noqa: E402
    load_obb_crop_overrides,
    resolve_crop_box_override,
)
from embedded_gauge_reading_tinyml.training._compilation import _direction_cosine_loss  # noqa: E402

DEFAULT_MANIFEST: Path = REPO_ROOT / "tmp" / "labelled_captured_images_board_center_tip_v2.json"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "board_needle_direction_qat_v1"
DEFAULT_OBB_CROP_MANIFEST: Path = REPO_ROOT / "tmp" / "obb_output_board_tip_v2" / "obb_crop_manifest.json"
IMAGE_SIZE: int = 224
DEFAULT_BATCH_SIZE: int = 12
DEFAULT_WARMUP_EPOCHS: int = 8
DEFAULT_FINETUNE_EPOCHS: int = 6
DEFAULT_QAT_EPOCHS: int = 4
DEFAULT_LEARNING_RATE: float = 5e-4
DEFAULT_FINETUNE_LEARNING_RATE: float = 2.5e-4
DEFAULT_QAT_LEARNING_RATE: float = 1e-4
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.10
DEFAULT_BOARD_SAMPLE_WEIGHT: float = 3.0
DEFAULT_SOURCE_KINDS: tuple[str, ...] = ("pxl_geometry", "reviewed_geometry")


@dataclass(frozen=True, slots=True)
class NeedleDirectionSample:
    """One cropped image paired with a unit needle direction target."""

    image_path: Path
    source_width: int
    source_height: int
    crop_x_min: float
    crop_y_min: float
    crop_x_max: float
    crop_y_max: float
    needle_x: float
    needle_y: float
    temperature_c: float | None
    source_kind: str

    @property
    def is_board(self) -> bool:
        """Return ``True`` when this sample came from a reviewed board capture."""

        return self.source_kind == "reviewed_geometry"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a compact needle-direction model with QAT.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--obb-crop-manifest", type=Path, default=DEFAULT_OBB_CROP_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--finetune-learning-rate", type=float, default=DEFAULT_FINETUNE_LEARNING_RATE)
    parser.add_argument("--qat-learning-rate", type=float, default=DEFAULT_QAT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--board-sample-weight", type=float, default=DEFAULT_BOARD_SAMPLE_WEIGHT)
    parser.add_argument(
        "--source-kinds",
        nargs="+",
        default=list(DEFAULT_SOURCE_KINDS),
        help="Source kinds to keep from the grouped manifest.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    """Coerce a JSON scalar into a finite float if possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _parse_optional_text(row: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string from the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_box_xyxy(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract a crop box from the grouped-manifest field names."""

    candidates = (
        ("crop_x_min", "crop_y_min", "crop_x_max", "crop_y_max"),
        ("loose_crop_x1", "loose_crop_y1", "loose_crop_x2", "loose_crop_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
    )
    for x1_key, y1_key, x2_key, y2_key in candidates:
        x1 = _as_float(row.get(x1_key))
        y1 = _as_float(row.get(y1_key))
        x2 = _as_float(row.get(x2_key))
        y2 = _as_float(row.get(y2_key))
        if None not in (x1, y1, x2, y2):
            return float(x1), float(y1), float(x2), float(y2)
    raise ValueError("Could not find a crop box in the manifest row.")


def _load_source_image(image_path: Path, *, source_width: int, source_height: int) -> np.ndarray:
    """Load one capture as an RGB array."""

    absolute_path = resolve_absolute_image_path(image_path)
    rgb, _kind = load_capture_image(
        absolute_path,
        image_width=source_width,
        image_height=source_height,
    )
    return np.asarray(rgb, dtype=np.uint8)


def _clip_crop_box(
    crop_x_min: float,
    crop_y_min: float,
    crop_x_max: float,
    crop_y_max: float,
    *,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    """Clip a floating-point crop box to valid integer pixel coordinates."""

    x1 = int(math.floor(max(0.0, min(float(source_width - 1), crop_x_min))))
    y1 = int(math.floor(max(0.0, min(float(source_height - 1), crop_y_min))))
    x2 = int(math.ceil(max(x1 + 1.0, min(float(source_width), crop_x_max))))
    y2 = int(math.ceil(max(y1 + 1.0, min(float(source_height), crop_y_max))))
    return x1, y1, x2, y2


def _crop_and_resize(
    image_rgb: np.ndarray,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Crop one source image and resize it to the shared training resolution."""

    x1, y1, x2, y2 = _clip_crop_box(
        *crop_box_xyxy,
        source_width=source_width,
        source_height=source_height,
    )
    image = Image.fromarray(image_rgb, mode="RGB")
    crop = image.crop((x1, y1, x2, y2))
    resized = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _direction_from_center_tip(
    center_x: float,
    center_y: float,
    tip_x: float,
    tip_y: float,
) -> tuple[float, float]:
    """Compute a unit direction vector from the center to the needle tip."""

    dx = float(tip_x) - float(center_x)
    dy = float(tip_y) - float(center_y)
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        raise ValueError("Center and tip are too close together to form a direction.")
    return float(dx / norm), float(dy / norm)


def _load_samples(
    manifest_path: Path,
    *,
    allowed_source_kinds: set[str],
    obb_crop_overrides: dict[Path, Any] | None,
) -> list[NeedleDirectionSample]:
    """Flatten the grouped manifest into direction-regression samples."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[NeedleDirectionSample] = []
    for image_entry in payload.get("images", []):
        annotations = list(image_entry.get("annotations", []))
        if not annotations:
            continue
        chosen: dict[str, Any] | None = None
        for annotation in annotations:
            source_kind = str(annotation.get("source_kind", "")).strip()
            if source_kind not in allowed_source_kinds:
                continue
            source_row = annotation.get("source_row", {})
            if not isinstance(source_row, dict):
                continue
            if _as_float(source_row.get("center_x_source")) is None:
                continue
            if _as_float(source_row.get("tip_x_source")) is None:
                continue
            chosen = annotation
            break
        if chosen is None:
            continue

        source_row = chosen["source_row"]
        image_path = to_repo_relative_path(
            _parse_optional_text(source_row, "image_path") or _parse_optional_text(chosen, "image_path")
        )
        source_width = int(round(_as_float(source_row.get("source_width")) or 0.0))
        source_height = int(round(_as_float(source_row.get("source_height")) or 0.0))
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid source size for {image_path}")

        crop_box_xyxy = _extract_box_xyxy(source_row)
        if obb_crop_overrides is not None:
            crop_box_xyxy, _record = resolve_crop_box_override(
                image_path,
                crop_box_xyxy,
                obb_crop_overrides,
                require_accepted=True,
            )

        center_x = _as_float(source_row.get("center_x_source"))
        center_y = _as_float(source_row.get("center_y_source"))
        tip_x = _as_float(source_row.get("tip_x_source"))
        tip_y = _as_float(source_row.get("tip_y_source"))
        temperature_c = _as_float(source_row.get("temperature_c"))
        if None in (center_x, center_y, tip_x, tip_y):
            continue

        needle_x, needle_y = _direction_from_center_tip(
            float(center_x),
            float(center_y),
            float(tip_x),
            float(tip_y),
        )
        samples.append(
            NeedleDirectionSample(
                image_path=image_path,
                source_width=source_width,
                source_height=source_height,
                crop_x_min=float(crop_box_xyxy[0]),
                crop_y_min=float(crop_box_xyxy[1]),
                crop_x_max=float(crop_box_xyxy[2]),
                crop_y_max=float(crop_box_xyxy[3]),
                needle_x=float(needle_x),
                needle_y=float(needle_y),
                temperature_c=float(temperature_c) if temperature_c is not None else None,
                source_kind=str(chosen["source_kind"]),
            )
        )
    return samples


def _build_arrays(
    samples: list[NeedleDirectionSample],
    *,
    board_sample_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preload the training images, direction targets, scalar targets, and weights."""

    images: list[np.ndarray] = []
    direction_targets: list[tuple[float, float]] = []
    temperature_targets: list[tuple[float]] = []
    temperature_weights: list[tuple[float]] = []
    weights: list[float] = []
    for sample in samples:
        source_image = _load_source_image(
            sample.image_path,
            source_width=sample.source_width,
            source_height=sample.source_height,
        )
        crop_rgb = _crop_and_resize(
            source_image,
            (sample.crop_x_min, sample.crop_y_min, sample.crop_x_max, sample.crop_y_max),
            source_width=sample.source_width,
            source_height=sample.source_height,
        )
        images.append(crop_rgb)
        direction_targets.append((sample.needle_x, sample.needle_y))
        temperature_targets.append((float(sample.temperature_c) if sample.temperature_c is not None else 0.0,))
        temperature_weights.append((1.0 if sample.temperature_c is not None else 0.0,))
        weights.append(float(board_sample_weight) if sample.is_board else 1.0)
    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(direction_targets, dtype=np.float32),
        np.asarray(temperature_targets, dtype=np.float32),
        np.asarray(temperature_weights, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def _rotate_direction(direction_xy: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a 2D direction vector by the same angle used for image augmentation."""

    theta = math.radians(float(angle_degrees))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x = float(direction_xy[0])
    y = float(direction_xy[1])
    # PIL rotates images counter-clockwise in display space. With image coordinates
    # (x right, y down), the matching vector transform uses the flipped off-diagonal terms.
    rotated_x = x * cos_theta + y * sin_theta
    rotated_y = -x * sin_theta + y * cos_theta
    return np.asarray([rotated_x, rotated_y], dtype=np.float32)


class DirectionSequence(keras.utils.Sequence):
    """Batch generator that can rotate crops and direction labels together."""

    def __init__(
        self,
        images: np.ndarray,
        direction_targets: np.ndarray,
        temperature_targets: np.ndarray,
        temperature_weights: np.ndarray,
        weights: np.ndarray,
        *,
        batch_size: int,
        augment: bool,
        seed: int,
        rotation_range_degrees: float,
    ) -> None:
        self._images = images
        self._direction_targets = direction_targets
        self._temperature_targets = temperature_targets
        self._temperature_weights = temperature_weights
        self._weights = weights
        self._batch_size = int(batch_size)
        self._augment = bool(augment)
        self._rotation_range_degrees = float(rotation_range_degrees)
        self._rng = np.random.default_rng(seed)
        self._indices = np.arange(len(images), dtype=np.int32)

    def __len__(self) -> int:
        return int(math.ceil(len(self._indices) / max(1, self._batch_size)))

    def on_epoch_end(self) -> None:
        if self._augment:
            self._rng.shuffle(self._indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        start = index * self._batch_size
        stop = min(len(self._indices), start + self._batch_size)
        batch_indices = self._indices[start:stop]
        batch_images = np.empty((len(batch_indices), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        batch_direction = np.empty((len(batch_indices), 2), dtype=np.float32)
        batch_temperature = np.empty((len(batch_indices), 1), dtype=np.float32)
        batch_direction_weights = np.empty((len(batch_indices),), dtype=np.float32)
        batch_temperature_weights = np.empty((len(batch_indices), 1), dtype=np.float32)
        for batch_index, sample_index in enumerate(batch_indices):
            image = self._images[int(sample_index)]
            direction_target = self._direction_targets[int(sample_index)]
            temperature_target = self._temperature_targets[int(sample_index)].astype(np.float32, copy=False)
            if self._augment:
                rotation = float(self._rng.uniform(-self._rotation_range_degrees, self._rotation_range_degrees))
                image = np.asarray(
                    Image.fromarray(image, mode="RGB").rotate(
                        rotation,
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.uint8,
                )
                direction_target = _rotate_direction(direction_target, rotation)
            batch_images[batch_index] = image
            batch_direction[batch_index] = direction_target
            batch_temperature[batch_index, :] = temperature_target
            batch_direction_weights[batch_index] = float(self._weights[int(sample_index)])
            batch_temperature_weights[batch_index, :] = self._temperature_weights[int(sample_index)]
        return (
            batch_images,
            {"needle_xy": batch_direction, "gauge_value": batch_temperature},
            {"needle_xy": batch_direction_weights, "gauge_value": batch_temperature_weights},
        )


def _direction_mae_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Report mean absolute angular error in degrees for unit direction vectors."""

    true_xy = tf.math.l2_normalize(tf.cast(y_true, tf.float32), axis=-1)
    pred_xy = tf.math.l2_normalize(tf.cast(y_pred, tf.float32), axis=-1)
    true_angle = tf.atan2(true_xy[..., 1], true_xy[..., 0])
    pred_angle = tf.atan2(pred_xy[..., 1], pred_xy[..., 0])
    delta = tf.atan2(tf.sin(pred_angle - true_angle), tf.cos(pred_angle - true_angle))
    return tf.reduce_mean(tf.abs(delta) * (180.0 / math.pi))


def _evaluate_arrays(
    model: keras.Model,
    images: np.ndarray,
    direction_targets: np.ndarray,
    temperature_targets: np.ndarray,
    temperature_weights: np.ndarray,
) -> dict[str, float]:
    """Evaluate angle and temperature MAE on a numpy batch."""

    preds = model.predict(images, batch_size=32, verbose=0)
    pred_xy = np.asarray(preds["needle_xy"], dtype=np.float32)
    pred_xy = pred_xy / np.clip(np.linalg.norm(pred_xy, axis=-1, keepdims=True), 1e-6, None)
    true_xy = np.asarray(direction_targets, dtype=np.float32)
    true_xy = true_xy / np.clip(np.linalg.norm(true_xy, axis=-1, keepdims=True), 1e-6, None)
    true_angle = np.degrees(np.arctan2(true_xy[:, 1], true_xy[:, 0]))
    pred_angle = np.degrees(np.arctan2(pred_xy[:, 1], pred_xy[:, 0]))
    angle_error = np.abs(
        np.degrees(np.arctan2(np.sin(np.radians(pred_angle - true_angle)), np.cos(np.radians(pred_angle - true_angle))))
    )
    temp_pred = np.asarray(preds["gauge_value"], dtype=np.float32).reshape(-1)
    temp_true = np.asarray(temperature_targets, dtype=np.float32).reshape(-1)
    temp_mask = np.asarray(temperature_weights, dtype=np.float32).reshape(-1) > 0.0
    if np.any(temp_mask):
        temp_mae = float(np.mean(np.abs(temp_pred[temp_mask] - temp_true[temp_mask])))
    else:
        temp_mae = float("nan")
    return {
        "angle_mae_degrees": float(np.mean(angle_error)),
        "angle_median_degrees": float(np.median(angle_error)),
        "temperature_mae_c": temp_mae,
    }


def _temperature_mae_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Report scalar temperature MAE."""

    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))


def _compile_model(model: keras.Model, *, learning_rate: float) -> None:
    """Compile the needle-direction model with angular agreement loss."""

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={
            "needle_xy": _direction_cosine_loss,
            "gauge_value": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "needle_xy": 1.0,
            "gauge_value": 0.25,
        },
        metrics={
            "needle_xy": [_direction_mae_metric],
            "gauge_value": [_temperature_mae_metric],
        },
    )


def _representative_dataset(images: np.ndarray, *, count: int) -> Any:
    """Yield representative batches for PTQ/QAT conversion."""

    limit = min(int(count), len(images))
    for index in range(limit):
        batch = np.asarray(images[index : index + 1], dtype=np.float32)
        yield [batch]


def _save_model_or_weights(model: keras.Model, output_path: Path) -> dict[str, str]:
    """Save a Keras model when possible, otherwise fall back to weights."""

    try:
        model.save(output_path)
        return {"artifact_kind": "model", "artifact_path": output_path.as_posix()}
    except Exception as exc:
        fallback_path = output_path.with_suffix(".weights.h5")
        model.save_weights(fallback_path)
        return {
            "artifact_kind": "weights_only",
            "artifact_path": fallback_path.as_posix(),
            "save_error": str(exc),
        }


def _normalize_obb_crop_manifest_path(path: Path) -> Path:
    """Resolve a crop manifest path to a local file path."""

    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> None:
    """Train the compact direction regressor and export PTQ/QAT artifacts."""

    args = _parse_args()
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed_source_kinds = {str(kind).strip() for kind in args.source_kinds if str(kind).strip()}
    obb_crop_overrides: dict[Path, Any] | None = None
    if args.obb_crop_manifest is not None:
        resolved_crop_manifest = _normalize_obb_crop_manifest_path(args.obb_crop_manifest)
        if not resolved_crop_manifest.exists():
            raise FileNotFoundError(resolved_crop_manifest)
        obb_crop_overrides = load_obb_crop_overrides(resolved_crop_manifest)
        print(
            f"Loaded {len(obb_crop_overrides)} OBB crop overrides from {resolved_crop_manifest}.",
            flush=True,
        )

    samples = _load_samples(
        args.manifest,
        allowed_source_kinds=allowed_source_kinds,
        obb_crop_overrides=obb_crop_overrides,
    )
    if not samples:
        raise ValueError(f"No usable samples were found in {args.manifest}.")

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

    train_images, train_direction_targets, train_temperature_targets, train_temperature_weights, train_weights = _build_arrays(
        train_samples,
        board_sample_weight=args.board_sample_weight,
    )
    val_images, val_direction_targets, val_temperature_targets, val_temperature_weights, val_weights = _build_arrays(
        val_samples,
        board_sample_weight=args.board_sample_weight,
    )
    test_images, test_direction_targets, test_temperature_targets, test_temperature_weights, _test_weights = _build_arrays(
        test_samples,
        board_sample_weight=args.board_sample_weight,
    )

    train_sequence = DirectionSequence(
        train_images,
        train_direction_targets,
        train_temperature_targets,
        train_temperature_weights,
        train_weights,
        batch_size=args.batch_size,
        augment=True,
        seed=args.seed,
        rotation_range_degrees=35.0,
    )
    val_sequence = DirectionSequence(
        val_images,
        val_direction_targets,
        val_temperature_targets,
        val_temperature_weights,
        val_weights,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        rotation_range_degrees=0.0,
    )

    source_kind_counts: dict[str, int] = {}
    for sample in samples:
        source_kind_counts[sample.source_kind] = source_kind_counts.get(sample.source_kind, 0) + 1
    print(
        f"Loaded {len(samples)} samples from {args.manifest} "
        f"(source_kinds={sorted(allowed_source_kinds)}).",
        flush=True,
    )
    print(json.dumps({"source_kind_counts": source_kind_counts}, indent=2, sort_keys=True), flush=True)
    print(
        f"Split sizes: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}.",
        flush=True,
    )

    base_model = build_qat_friendly_needle_direction_geometry_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        encoder_width_multiplier=1.0,
        head_units=64,
        head_dropout=0.15,
    )
    set_needle_direction_encoder_trainable(base_model, trainable=False)
    _compile_model(base_model, learning_rate=args.learning_rate)

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

    print("Phase 1: warmup training with the encoder frozen...", flush=True)
    base_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.warmup_epochs,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    print("Phase 2: fine-tuning the full model...", flush=True)
    set_needle_direction_encoder_trainable(base_model, trainable=True)
    _compile_model(base_model, learning_rate=args.finetune_learning_rate)
    finetune_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(2, args.finetune_epochs // 2),
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
    base_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.finetune_epochs,
        callbacks=finetune_callbacks,
        verbose=2,
    )

    float_test_report = _evaluate_arrays(
        base_model,
        test_images,
        test_direction_targets,
        test_temperature_targets,
        test_temperature_weights,
    )
    print(json.dumps({"float_test_report": float_test_report}, indent=2, sort_keys=True), flush=True)

    float_model_path = args.output_dir / "needle_direction_float.keras"
    float_artifact = _save_model_or_weights(base_model, float_model_path)

    ptq_tflite_path = args.output_dir / "needle_direction_ptq.tflite"
    ptq_converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ptq_converter.representative_dataset = lambda: _representative_dataset(train_images, count=min(32, len(train_images)))
    ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    ptq_converter.inference_input_type = tf.int8
    ptq_converter.inference_output_type = tf.int8
    ptq_tflite_path.write_bytes(ptq_converter.convert())

    print("Phase 3: QAT fine-tuning...", flush=True)
    qat_model_path = args.output_dir / "needle_direction_qat.keras"
    qat_tflite_path = args.output_dir / "needle_direction_qat.tflite"
    qat_test_report: dict[str, float] | None = None
    try:
        qat_base = tfmot.quantization.keras.quantize_model(base_model)
        _compile_model(qat_base, learning_rate=args.qat_learning_rate)
        qat_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=max(2, args.qat_epochs // 2),
                restore_best_weights=True,
            )
        ]
        qat_base.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=args.qat_epochs,
            callbacks=qat_callbacks,
            verbose=2,
        )
        qat_test_report = _evaluate_arrays(
            qat_base,
            test_images,
            test_direction_targets,
            test_temperature_targets,
            test_temperature_weights,
        )
        print(json.dumps({"qat_test_report": qat_test_report}, indent=2, sort_keys=True), flush=True)
        qat_artifact = _save_model_or_weights(qat_base, qat_model_path)
        qat_converter = tf.lite.TFLiteConverter.from_keras_model(qat_base)
        qat_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        qat_converter.representative_dataset = lambda: _representative_dataset(train_images, count=min(32, len(train_images)))
        qat_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        qat_converter.inference_input_type = tf.int8
        qat_converter.inference_output_type = tf.int8
        qat_tflite_path.write_bytes(qat_converter.convert())
    except Exception as exc:
        print(f"QAT path skipped: {exc}", flush=True)
        qat_artifact = {"artifact_kind": "skipped", "error": str(exc)}

    summary = {
        "manifest": args.manifest.as_posix(),
        "obb_crop_manifest": args.obb_crop_manifest.as_posix() if args.obb_crop_manifest is not None else None,
        "sample_count": len(samples),
        "board_count": int(sum(1 for sample in samples if sample.is_board)),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "source_kind_counts": source_kind_counts,
        "float_artifact": float_artifact,
        "float_model_path": float_model_path.as_posix(),
        "ptq_tflite_path": ptq_tflite_path.as_posix(),
        "qat_artifact": qat_artifact,
        "qat_model_path": qat_model_path.as_posix(),
        "qat_tflite_path": qat_tflite_path.as_posix(),
        "float_test_report": float_test_report,
        "qat_test_report": qat_test_report,
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "finetune_learning_rate": args.finetune_learning_rate,
            "qat_learning_rate": args.qat_learning_rate,
            "board_sample_weight": args.board_sample_weight,
            "source_kinds": sorted(allowed_source_kinds),
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
