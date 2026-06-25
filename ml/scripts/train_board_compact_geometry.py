#!/usr/bin/env python3
"""Train a compact keypoint geometry model for board gauge captures.

This trainer uses the smaller compact geometry architecture with explicit
center/tip heatmaps, explicit coordinate supervision, and an auxiliary scalar
temperature head. It is designed to learn from the reviewed board captures and
the pxl geometry rows while normalizing the board images through the stronger
OBB crop manifest.
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

import keras

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
from embedded_gauge_reading_tinyml.gauge_geometry import angle_degrees_from_center_to_tip  # noqa: E402
from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps  # noqa: E402
from embedded_gauge_reading_tinyml.models import build_compact_geometry_model  # noqa: E402
from embedded_gauge_reading_tinyml.obb_crop_manifest import (  # noqa: E402
    load_obb_crop_overrides,
    resolve_crop_box_override,
)

DEFAULT_MANIFEST: Path = REPO_ROOT / "tmp" / "labelled_captured_images_board_center_tip_v2.json"
DEFAULT_OBB_CROP_MANIFEST: Path = REPO_ROOT / "tmp" / "obb_output_board_tip_v2" / "obb_crop_manifest.json"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "board_compact_geometry_v1"
IMAGE_SIZE: int = 224
HEATMAP_SIZE: int = 56
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 24
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.10
DEFAULT_BOARD_SAMPLE_WEIGHT: float = 4.0
DEFAULT_SIGMA_PIXELS: float = 2.0
DEFAULT_HEATMAP_LOSS_WEIGHT: float = 1.0
DEFAULT_COORD_LOSS_WEIGHT: float = 2.0
DEFAULT_VALUE_LOSS_WEIGHT: float = 0.0
DEFAULT_MODEL_FAMILY: str = "compact_geometry"
DEFAULT_MOBILENET_ALPHA: float = 0.35
DEFAULT_BOARD_MEAN_LUMA_MIN: float = 40.0
DEFAULT_BOARD_LAPLACIAN_MIN: float = 20.0
DEFAULT_TIP_LOSS_WEIGHT: float = 3.0
DEFAULT_BOARD_FINETUNE_EPOCHS: int = 4
DEFAULT_BOARD_FINETUNE_LR: float = 5e-5
DEFAULT_SOURCE_KINDS: tuple[str, ...] = ("pxl_geometry", "reviewed_geometry")


@dataclass(frozen=True, slots=True)
class GeometrySample:
    """One crop-normalized geometry example."""

    image_path: Path
    source_width: int
    source_height: int
    crop_x_min: float
    crop_y_min: float
    crop_x_max: float
    crop_y_max: float
    center_x_224: float
    center_y_224: float
    tip_x_224: float
    tip_y_224: float
    temperature_c: float | None
    source_kind: str

    @property
    def is_board(self) -> bool:
        """Return true for reviewed board captures."""

        return self.source_kind == "reviewed_geometry"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a compact geometry keypoint model on board captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--obb-crop-manifest", type=Path, default=DEFAULT_OBB_CROP_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-family",
        choices=["compact_geometry", "mobilenetv2_geometry"],
        default=DEFAULT_MODEL_FAMILY,
    )
    parser.add_argument("--mobilenet-alpha", type=float, default=DEFAULT_MOBILENET_ALPHA)
    parser.add_argument(
        "--mobilenet-backbone-trainable",
        action="store_true",
        help="Fine-tune the MobileNetV2 backbone instead of freezing it.",
    )
    parser.add_argument("--board-sample-weight", type=float, default=DEFAULT_BOARD_SAMPLE_WEIGHT)
    parser.add_argument("--board-mean-luma-min", type=float, default=DEFAULT_BOARD_MEAN_LUMA_MIN)
    parser.add_argument("--board-laplacian-min", type=float, default=DEFAULT_BOARD_LAPLACIAN_MIN)
    parser.add_argument("--sigma-pixels", type=float, default=DEFAULT_SIGMA_PIXELS)
    parser.add_argument("--heatmap-loss-weight", type=float, default=DEFAULT_HEATMAP_LOSS_WEIGHT)
    parser.add_argument("--coord-loss-weight", type=float, default=DEFAULT_COORD_LOSS_WEIGHT)
    parser.add_argument("--tip-loss-weight", type=float, default=DEFAULT_TIP_LOSS_WEIGHT)
    parser.add_argument("--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT)
    parser.add_argument("--board-finetune-epochs", type=int, default=DEFAULT_BOARD_FINETUNE_EPOCHS)
    parser.add_argument("--board-finetune-lr", type=float, default=DEFAULT_BOARD_FINETUNE_LR)
    parser.add_argument(
        "--source-kinds",
        nargs="+",
        default=list(DEFAULT_SOURCE_KINDS),
        help="Source kinds to keep from the grouped manifest.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    """Parse a JSON scalar into a finite float."""

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
    """Clip a floating-point crop box to valid pixel coordinates."""

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
    """Crop one source image and resize it to the training resolution."""

    x1, y1, x2, y2 = _clip_crop_box(
        *crop_box_xyxy,
        source_width=source_width,
        source_height=source_height,
    )
    image = Image.fromarray(image_rgb, mode="RGB")
    crop = image.crop((x1, y1, x2, y2))
    resized = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _point_norm_in_crop(
    x_source: float,
    y_source: float,
    crop_x_min: float,
    crop_y_min: float,
    crop_x_max: float,
    crop_y_max: float,
) -> tuple[float, float]:
    """Map a source-space point into crop-normalized coordinates."""

    crop_w = max(1.0, float(crop_x_max - crop_x_min))
    crop_h = max(1.0, float(crop_y_max - crop_y_min))
    x_norm = (float(x_source) - float(crop_x_min)) / crop_w
    y_norm = (float(y_source) - float(crop_y_min)) / crop_h
    return float(np.clip(x_norm, 0.0, 1.0)), float(np.clip(y_norm, 0.0, 1.0))


def _crop_quality_stats(crop_rgb: np.ndarray) -> tuple[float, float]:
    """Return simple brightness and sharpness statistics for a crop."""

    crop = np.asarray(crop_rgb, dtype=np.float32)
    luma = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    mean_luma = float(np.mean(luma))
    center = luma[1:-1, 1:-1]
    laplacian = (
        luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        - 4.0 * center
    )
    laplacian_var = float(np.var(laplacian))
    return mean_luma, laplacian_var


def _load_samples(
    manifest_path: Path,
    *,
    allowed_source_kinds: set[str],
    obb_crop_overrides: dict[Path, Any] | None,
    board_mean_luma_min: float,
    board_laplacian_min: float,
) -> list[GeometrySample]:
    """Flatten the grouped manifest into geometry samples."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[GeometrySample] = []
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

        if source_kind == "reviewed_geometry":
            source_image = _load_source_image(
                image_path,
                source_width=source_width,
                source_height=source_height,
            )
            crop_rgb = _crop_and_resize(
                source_image,
                crop_box_xyxy,
                source_width=source_width,
                source_height=source_height,
            )
            mean_luma, laplacian_var = _crop_quality_stats(crop_rgb)
            if mean_luma < board_mean_luma_min or laplacian_var < board_laplacian_min:
                continue

        center_x = _as_float(source_row.get("center_x_source"))
        center_y = _as_float(source_row.get("center_y_source"))
        tip_x = _as_float(source_row.get("tip_x_source"))
        tip_y = _as_float(source_row.get("tip_y_source"))
        temperature_c = _as_float(source_row.get("temperature_c"))
        if None in (center_x, center_y, tip_x, tip_y):
            continue

        center_x_224, center_y_224 = _point_norm_in_crop(
            float(center_x),
            float(center_y),
            float(crop_box_xyxy[0]),
            float(crop_box_xyxy[1]),
            float(crop_box_xyxy[2]),
            float(crop_box_xyxy[3]),
        )
        tip_x_224, tip_y_224 = _point_norm_in_crop(
            float(tip_x),
            float(tip_y),
            float(crop_box_xyxy[0]),
            float(crop_box_xyxy[1]),
            float(crop_box_xyxy[2]),
            float(crop_box_xyxy[3]),
        )
        samples.append(
            GeometrySample(
                image_path=image_path,
                source_width=source_width,
                source_height=source_height,
                crop_x_min=float(crop_box_xyxy[0]),
                crop_y_min=float(crop_box_xyxy[1]),
                crop_x_max=float(crop_box_xyxy[2]),
                crop_y_max=float(crop_box_xyxy[3]),
                center_x_224=float(center_x_224 * (HEATMAP_SIZE - 1)),
                center_y_224=float(center_y_224 * (HEATMAP_SIZE - 1)),
                tip_x_224=float(tip_x_224 * (HEATMAP_SIZE - 1)),
                tip_y_224=float(tip_y_224 * (HEATMAP_SIZE - 1)),
                temperature_c=temperature_c,
                source_kind=str(chosen["source_kind"]),
            )
        )
    return samples


def _build_arrays(
    samples: list[GeometrySample],
    *,
    board_sample_weight: float,
    sigma_pixels: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build model input arrays and targets, repeating board rows for emphasis."""

    config = HeatmapConfig(
        heatmap_height=HEATMAP_SIZE,
        heatmap_width=HEATMAP_SIZE,
        input_height=IMAGE_SIZE,
        input_width=IMAGE_SIZE,
        sigma_pixels=sigma_pixels,
    )

    images: list[np.ndarray] = []
    gauge_values: list[tuple[float]] = []
    keypoint_heatmaps: list[np.ndarray] = []
    keypoint_coords: list[np.ndarray] = []
    repeat_factor = max(1, int(round(float(board_sample_weight))))

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
        images.append(crop_rgb.astype(np.float32) / 255.0)
        gauge_values.append((float(sample.temperature_c) if sample.temperature_c is not None else 0.0,))
        center_x_norm = sample.center_x_224 / max(HEATMAP_SIZE - 1, 1)
        center_y_norm = sample.center_y_224 / max(HEATMAP_SIZE - 1, 1)
        tip_x_norm = sample.tip_x_224 / max(HEATMAP_SIZE - 1, 1)
        tip_y_norm = sample.tip_y_224 / max(HEATMAP_SIZE - 1, 1)
        heatmap_pair = generate_center_tip_heatmaps(
            center_x_norm,
            center_y_norm,
            tip_x_norm,
            tip_y_norm,
            config=config,
        )
        keypoint_heatmaps.append(np.stack(heatmap_pair, axis=-1))
        keypoint_coords.append(
            np.asarray(
                [
                    [sample.center_x_224, sample.center_y_224],
                    [sample.tip_x_224, sample.tip_y_224],
                ],
                dtype=np.float32,
            )
        )
        if sample.is_board:
            for _ in range(repeat_factor - 1):
                images.append(crop_rgb.astype(np.float32) / 255.0)
                gauge_values.append((float(sample.temperature_c) if sample.temperature_c is not None else 0.0,))
                keypoint_heatmaps.append(np.stack(heatmap_pair, axis=-1))
                keypoint_coords.append(
                    np.asarray(
                        [
                            [sample.center_x_224, sample.center_y_224],
                            [sample.tip_x_224, sample.tip_y_224],
                        ],
                        dtype=np.float32,
                    )
                )
    inputs = np.asarray(images, dtype=np.float32)
    targets = {
        "gauge_value": np.asarray(gauge_values, dtype=np.float32),
        "keypoint_heatmaps": np.asarray(keypoint_heatmaps, dtype=np.float32),
        "keypoint_coords": np.asarray(keypoint_coords, dtype=np.float32),
    }
    return inputs, targets


def _build_board_only_arrays(
    samples: list[GeometrySample],
    *,
    sigma_pixels: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build arrays for a board-only fine-tuning pass."""

    board_samples = [sample for sample in samples if sample.is_board]
    return _build_arrays(
        board_samples,
        board_sample_weight=1.0,
        sigma_pixels=sigma_pixels,
    )


def _compile_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_COORD_LOSS_WEIGHT,
    tip_loss_weight: float = DEFAULT_TIP_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT,
) -> None:
    """Compile the compact geometry model."""

    def _weighted_coords_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Weight the tip keypoint more than the center keypoint."""

        error = tf.square(y_true - y_pred)
        point_weights = tf.constant([1.0, float(tip_loss_weight)], dtype=y_pred.dtype)
        point_weights = tf.reshape(point_weights, (1, 2, 1))
        return tf.reduce_mean(error * point_weights)

    def _weighted_heatmap_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Weight the tip heatmap channel more than the center channel."""

        error = tf.square(y_true - y_pred)
        channel_weights = tf.constant([1.0, float(tip_loss_weight)], dtype=y_pred.dtype)
        channel_weights = tf.reshape(channel_weights, (1, 1, 2))
        return tf.reduce_mean(error * channel_weights)

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": _weighted_heatmap_mse,
            "keypoint_coords": _weighted_coords_mse,
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
        },
        metrics={
            "gauge_value": [keras.metrics.MeanAbsoluteError(name="mae")],
            "keypoint_coords": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )


def _build_callbacks(patience: int) -> list[keras.callbacks.Callback]:
    """Build fresh callbacks for one training phase."""

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]


def _evaluate_model(
    model: keras.Model,
    samples: list[GeometrySample],
) -> dict[str, float]:
    """Evaluate the model on held-out samples."""

    center_errors: list[float] = []
    tip_errors: list[float] = []
    angle_errors: list[float] = []
    temp_errors: list[float] = []
    board_center_errors: list[float] = []
    board_tip_errors: list[float] = []
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
        pred = model.predict(crop_rgb[np.newaxis].astype(np.float32) / 255.0, verbose=0)
        pred_coords = np.asarray(pred["keypoint_coords"], dtype=np.float32).reshape(2, 2)
        true_center_x = sample.crop_x_min + (sample.center_x_224 / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_x_max - sample.crop_x_min)
        true_center_y = sample.crop_y_min + (sample.center_y_224 / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_y_max - sample.crop_y_min)
        true_tip_x = sample.crop_x_min + (sample.tip_x_224 / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_x_max - sample.crop_x_min)
        true_tip_y = sample.crop_y_min + (sample.tip_y_224 / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_y_max - sample.crop_y_min)
        pred_center_x = sample.crop_x_min + (float(pred_coords[0, 0]) / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_x_max - sample.crop_x_min)
        pred_center_y = sample.crop_y_min + (float(pred_coords[0, 1]) / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_y_max - sample.crop_y_min)
        pred_tip_x = sample.crop_x_min + (float(pred_coords[1, 0]) / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_x_max - sample.crop_x_min)
        pred_tip_y = sample.crop_y_min + (float(pred_coords[1, 1]) / max(HEATMAP_SIZE - 1, 1)) * (sample.crop_y_max - sample.crop_y_min)
        center_mae = 0.5 * (abs(pred_center_x - true_center_x) + abs(pred_center_y - true_center_y))
        tip_mae = 0.5 * (abs(pred_tip_x - true_tip_x) + abs(pred_tip_y - true_tip_y))
        true_angle = angle_degrees_from_center_to_tip(true_center_x, true_center_y, true_tip_x, true_tip_y)
        pred_angle = angle_degrees_from_center_to_tip(pred_center_x, pred_center_y, pred_tip_x, pred_tip_y)
        angle_error = float(abs((pred_angle - true_angle + 180.0) % 360.0 - 180.0))
        center_errors.append(center_mae)
        tip_errors.append(tip_mae)
        angle_errors.append(angle_error)
        if sample.temperature_c is not None:
            pred_temp = float(np.asarray(pred["gauge_value"]).reshape(-1)[0])
            temp_errors.append(abs(pred_temp - sample.temperature_c))
        if sample.is_board:
            board_center_errors.append(center_mae)
            board_tip_errors.append(tip_mae)
    return {
        "center_mae_px": float(np.mean(center_errors)),
        "tip_mae_px": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "temperature_mae_c": float(np.mean(temp_errors)) if temp_errors else float("nan"),
        "board_center_mae_px": float(np.mean(board_center_errors)) if board_center_errors else float("nan"),
        "board_tip_mae_px": float(np.mean(board_tip_errors)) if board_tip_errors else float("nan"),
    }


def main() -> None:
    """Train the compact geometry model and export PTQ artifacts."""

    args = _parse_args()
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed_source_kinds = {str(kind).strip() for kind in args.source_kinds if str(kind).strip()}
    obb_crop_overrides: dict[Path, Any] | None = None
    if args.obb_crop_manifest is not None:
        if not args.obb_crop_manifest.exists():
            raise FileNotFoundError(args.obb_crop_manifest)
        obb_crop_overrides = load_obb_crop_overrides(args.obb_crop_manifest)
        print(
            f"Loaded {len(obb_crop_overrides)} OBB crop overrides from {args.obb_crop_manifest}.",
            flush=True,
        )

    samples = _load_samples(
        args.manifest,
        allowed_source_kinds=allowed_source_kinds,
        obb_crop_overrides=obb_crop_overrides,
        board_mean_luma_min=args.board_mean_luma_min,
        board_laplacian_min=args.board_laplacian_min,
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

    train_x, train_y = _build_arrays(
        train_samples,
        board_sample_weight=args.board_sample_weight,
        sigma_pixels=args.sigma_pixels,
    )
    val_x, val_y = _build_arrays(
        val_samples,
        board_sample_weight=args.board_sample_weight,
        sigma_pixels=args.sigma_pixels,
    )
    test_x, test_y = _build_arrays(
        test_samples,
        board_sample_weight=args.board_sample_weight,
        sigma_pixels=args.sigma_pixels,
    )
    board_train_samples = [sample for sample in train_samples if sample.is_board]
    board_val_samples = [sample for sample in val_samples if sample.is_board]
    board_train_x, board_train_y = _build_board_only_arrays(
        board_train_samples,
        sigma_pixels=args.sigma_pixels,
    )
    board_val_x, board_val_y = _build_board_only_arrays(
        board_val_samples,
        sigma_pixels=args.sigma_pixels,
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
    print(f"Split sizes: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}.", flush=True)

    if args.model_family == "compact_geometry":
        model = build_compact_geometry_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            value_min=-30.0,
            value_max=50.0,
            min_angle_rad=math.radians(135.0),
            sweep_rad=math.radians(270.0),
            heatmap_size=HEATMAP_SIZE,
        )
    elif args.model_family == "mobilenetv2_geometry":
        from embedded_gauge_reading_tinyml.models import build_mobilenetv2_geometry_model  # noqa: E402

        model = build_mobilenetv2_geometry_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            value_min=-30.0,
            value_max=50.0,
            min_angle_rad=math.radians(135.0),
            sweep_rad=math.radians(270.0),
            heatmap_size=HEATMAP_SIZE,
            pretrained=True,
            backbone_trainable=bool(args.mobilenet_backbone_trainable),
            alpha=float(args.mobilenet_alpha),
        )
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")
    _compile_model(
        model,
        learning_rate=args.learning_rate,
        heatmap_loss_weight=args.heatmap_loss_weight,
        coord_loss_weight=args.coord_loss_weight,
        tip_loss_weight=args.tip_loss_weight,
        value_loss_weight=args.value_loss_weight,
    )

    callbacks = _build_callbacks(max(3, args.epochs // 4))
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    if args.board_finetune_epochs > 0 and len(board_train_samples) > 0:
        print(
            f"Starting board-only fine-tune on {len(board_train_samples)} clean board samples.",
            flush=True,
        )
        learning_rate = model.optimizer.learning_rate
        if hasattr(learning_rate, "assign"):
            learning_rate.assign(float(args.board_finetune_lr))
        else:
            model.optimizer.learning_rate = float(args.board_finetune_lr)
        fine_tune_history = model.fit(
            board_train_x,
            board_train_y,
            validation_data=(board_val_x, board_val_y) if len(board_val_samples) > 0 else None,
            epochs=args.board_finetune_epochs,
            batch_size=max(4, args.batch_size // 2),
            callbacks=_build_callbacks(max(2, args.board_finetune_epochs // 2)),
            verbose=2,
        )
        for key, values in fine_tune_history.history.items():
            history.history.setdefault(key, [])
            history.history[key].extend(values)

    float_test_report = _evaluate_model(model, test_samples)
    print(json.dumps({"float_test_report": float_test_report}, indent=2, sort_keys=True), flush=True)

    float_model_path = args.output_dir / "compact_geometry_float.keras"
    model.save(float_model_path)

    ptq_tflite_path = args.output_dir / "compact_geometry_ptq.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [train_x[index : index + 1].astype(np.float32)] for index in range(min(32, len(train_x)))
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    ptq_tflite_path.write_bytes(converter.convert())

    summary = {
        "manifest": args.manifest.as_posix(),
        "obb_crop_manifest": args.obb_crop_manifest.as_posix() if args.obb_crop_manifest is not None else None,
        "sample_count": len(samples),
        "board_count": int(sum(1 for sample in samples if sample.is_board)),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "source_kind_counts": source_kind_counts,
        "float_model_path": float_model_path.as_posix(),
        "ptq_tflite_path": ptq_tflite_path.as_posix(),
        "float_test_report": float_test_report,
        "config": {
            "image_size": IMAGE_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "model_family": args.model_family,
            "mobilenet_alpha": args.mobilenet_alpha,
            "mobilenet_backbone_trainable": bool(args.mobilenet_backbone_trainable),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "board_sample_weight": args.board_sample_weight,
            "board_mean_luma_min": args.board_mean_luma_min,
            "board_laplacian_min": args.board_laplacian_min,
            "sigma_pixels": args.sigma_pixels,
            "heatmap_loss_weight": args.heatmap_loss_weight,
            "coord_loss_weight": args.coord_loss_weight,
            "tip_loss_weight": args.tip_loss_weight,
            "value_loss_weight": args.value_loss_weight,
            "board_finetune_epochs": args.board_finetune_epochs,
            "board_finetune_lr": args.board_finetune_lr,
            "source_kinds": sorted(allowed_source_kinds),
        },
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
