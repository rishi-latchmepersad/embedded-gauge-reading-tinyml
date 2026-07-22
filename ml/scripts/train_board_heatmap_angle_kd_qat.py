#!/usr/bin/env python3
"""Train a board-focused heatmap model for needle geometry.

This script mirrors the OBB recipe we just used for the gauge-face localizer:
transfer learning from ImageNet MobileNetV2, teacher distillation, and a final
QAT/PTQ export path for STM32N6 deployment. The training mix comes from the
combined grouped manifest that already contains:

* clean PXL geometry rows, and
* reviewed board center/tip rows.

The model predicts a center heatmap, a tip heatmap, and a confidence scalar.
Those outputs are enough to recover the needle angle for the downstream polar
vote stage.
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
import keras as standalone_keras
import tensorflow_model_optimization as tfmot

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_pipeline import load_capture_image  # noqa: E402
from embedded_gauge_reading_tinyml.capture_labeling import resolve_absolute_image_path, to_repo_relative_path  # noqa: E402
from embedded_gauge_reading_tinyml.gauge_geometry import angle_degrees_from_center_to_tip  # noqa: E402
from embedded_gauge_reading_tinyml.heatmap_losses import (  # noqa: E402
    focal_heatmap_loss,
    softargmax_coordinate_mae,
    weighted_center_heatmap_loss,
    weighted_tip_heatmap_loss,
)
from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps  # noqa: E402
from embedded_gauge_reading_tinyml.obb_crop_manifest import (  # noqa: E402
    load_obb_crop_overrides,
    resolve_crop_box_override,
)
from embedded_gauge_reading_tinyml.models_geometry import (  # noqa: E402
    BilinearUpsamplingInitializer,
    Identity3x3Initializer,
    build_qat_friendly_heatmap_angle_model,
    set_heatmap_encoder_trainable,
)

DEFAULT_MANIFEST: Path = REPO_ROOT / "tmp" / "labelled_captured_images_board_center_tip_v2.json"
DEFAULT_TEACHER_MODEL_PATH: Path = PROJECT_ROOT / "artifacts" / "training" / "heatmap_angle_board_pxl_v1" / "model.keras"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "board_heatmap_angle_kd_qat_v1"
IMAGE_SIZE: int = 224
HEATMAP_SIZE: int = 112
DEFAULT_BATCH_SIZE: int = 12
DEFAULT_PRETRAIN_EPOCHS: int = 8
DEFAULT_KD_EPOCHS: int = 4
DEFAULT_QAT_EPOCHS: int = 4
DEFAULT_ALPHA: float = 0.35
DEFAULT_LEARNING_RATE: float = 5e-4
DEFAULT_KD_LEARNING_RATE: float = 2.5e-4
DEFAULT_QAT_LEARNING_RATE: float = 1e-4
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.10
DEFAULT_BOARD_SAMPLE_WEIGHT: float = 3.0
DEFAULT_TRANSLATION_RATIO: float = 0.10
DEFAULT_SCALE_JITTER: float = 0.08
DEFAULT_SIGMA_PIXELS: float = 3.0
DEFAULT_SOURCE_KINDS: tuple[str, ...] = ("pxl_geometry", "reviewed_geometry")


@dataclass(frozen=True, slots=True)
class NeedleSample:
    """One cropped gauge image and its center/tip supervision."""

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
    source_kind: str
    confidence: float

    @property
    def is_board(self) -> bool:
        """Return ``True`` when the sample comes from a reviewed board capture."""

        return self.source_kind == "reviewed_geometry"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the board heatmap trainer."""

    parser = argparse.ArgumentParser(description="Train a board-focused heatmap angle model with KD and QAT.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS)
    parser.add_argument("--kd-epochs", type=int, default=DEFAULT_KD_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--kd-learning-rate", type=float, default=DEFAULT_KD_LEARNING_RATE)
    parser.add_argument("--qat-learning-rate", type=float, default=DEFAULT_QAT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translation-ratio", type=float, default=DEFAULT_TRANSLATION_RATIO)
    parser.add_argument("--scale-jitter", type=float, default=DEFAULT_SCALE_JITTER)
    parser.add_argument("--board-sample-weight", type=float, default=DEFAULT_BOARD_SAMPLE_WEIGHT)
    parser.add_argument("--sigma-pixels", type=float, default=DEFAULT_SIGMA_PIXELS)
    parser.add_argument("--obb-crop-manifest", type=Path, default=None)
    parser.add_argument(
        "--source-kinds",
        nargs="+",
        default=list(DEFAULT_SOURCE_KINDS),
        help="Source kinds to keep from the grouped manifest.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    """Coerce one scalar into a finite float if possible."""

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
    """Return the first non-empty string found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_box_xyxy(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract a crop box from the known grouped-manifest field names."""

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


def _load_source_image(image_path: Path, *, source_width: int, source_height: int) -> np.ndarray:
    """Load one capture as an RGB array."""

    absolute_path = resolve_absolute_image_path(image_path)
    rgb, _kind = load_capture_image(
        absolute_path,
        image_width=source_width,
        image_height=source_height,
    )
    return np.asarray(rgb, dtype=np.uint8)


def _resize_crop_to_input(crop_rgb: np.ndarray) -> np.ndarray:
    """Resize one crop to the shared 224x224 input geometry."""

    image = Image.fromarray(crop_rgb).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _build_samples(
    manifest_path: Path,
    *,
    allowed_source_kinds: set[str],
    obb_crop_overrides: dict[Path, Any] | None = None,
) -> list[NeedleSample]:
    """Flatten the grouped manifest into needle heatmap samples."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[NeedleSample] = []
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
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = _extract_box_xyxy(source_row)
        center_x = _as_float(source_row.get("center_x_source"))
        center_y = _as_float(source_row.get("center_y_source"))
        tip_x = _as_float(source_row.get("tip_x_source"))
        tip_y = _as_float(source_row.get("tip_y_source"))
        if None in (center_x, center_y, tip_x, tip_y):
            continue
        if obb_crop_overrides is not None:
            override_crop, _record = resolve_crop_box_override(
                image_path,
                (crop_x_min, crop_y_min, crop_x_max, crop_y_max),
                obb_crop_overrides,
                require_accepted=True,
            )
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = override_crop
        center_x_224, center_y_224 = _point_norm_in_crop(
            float(center_x),
            float(center_y),
            crop_x_min,
            crop_y_min,
            crop_x_max,
            crop_y_max,
        )
        tip_x_224, tip_y_224 = _point_norm_in_crop(
            float(tip_x),
            float(tip_y),
            crop_x_min,
            crop_y_min,
            crop_x_max,
            crop_y_max,
        )
        samples.append(
            NeedleSample(
                image_path=image_path,
                source_width=source_width,
                source_height=source_height,
                crop_x_min=float(crop_x_min),
                crop_y_min=float(crop_y_min),
                crop_x_max=float(crop_x_max),
                crop_y_max=float(crop_y_max),
                center_x_224=float(center_x_224 * (IMAGE_SIZE - 1)),
                center_y_224=float(center_y_224 * (IMAGE_SIZE - 1)),
                tip_x_224=float(tip_x_224 * (IMAGE_SIZE - 1)),
                tip_y_224=float(tip_y_224 * (IMAGE_SIZE - 1)),
                source_kind=str(chosen["source_kind"]),
                confidence=1.0,
            )
        )
    return samples


def _load_crop_image(sample: NeedleSample) -> np.ndarray:
    """Load the base crop image for one sample."""

    source_rgb = _load_source_image(
        sample.image_path,
        source_width=sample.source_width,
        source_height=sample.source_height,
    )
    x1 = max(0, int(round(sample.crop_x_min)))
    y1 = max(0, int(round(sample.crop_y_min)))
    x2 = min(sample.source_width, int(round(sample.crop_x_max)))
    y2 = min(sample.source_height, int(round(sample.crop_y_max)))
    crop = Image.fromarray(source_rgb).crop((x1, y1, x2, y2)).convert("RGB")
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def _translate_canvas(
    image: np.ndarray,
    points: np.ndarray,
    *,
    max_shift_px: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a 224x224 crop and update the keypoints."""

    if max_shift_px <= 0:
        return image, points

    shift_x = int(rng.integers(-max_shift_px, max_shift_px + 1))
    shift_y = int(rng.integers(-max_shift_px, max_shift_px + 1))
    if shift_x == 0 and shift_y == 0:
        return image, points

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

    updated = points.copy()
    updated[:, 0] = np.clip(updated[:, 0] + float(shift_x), 0.0, float(IMAGE_SIZE - 1))
    updated[:, 1] = np.clip(updated[:, 1] + float(shift_y), 0.0, float(IMAGE_SIZE - 1))
    return shifted, updated


def _scale_canvas(
    image: np.ndarray,
    points: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a 224x224 crop around the canvas center and update keypoints."""

    if not math.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-3:
        return image, points

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
    updated = points.copy()
    updated[:, 0] = (updated[:, 0] - center_px) * scale + center_px
    updated[:, 1] = (updated[:, 1] - center_px) * scale + center_px
    updated[:, 0] = np.clip(updated[:, 0], 0.0, float(IMAGE_SIZE - 1))
    updated[:, 1] = np.clip(updated[:, 1], 0.0, float(IMAGE_SIZE - 1))
    return scaled_image, updated


def _photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply light photometric jitter."""

    image_f = image.astype(np.float32)
    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.90, 1.10))
    noise = rng.normal(0.0, 2.0, size=image_f.shape).astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)
    image_f = (image_f - mean) * contrast + mean
    image_f = image_f * brightness + noise
    return np.clip(image_f, 0.0, 255.0).astype(np.uint8)


def _make_heatmaps(points_224: np.ndarray, *, sigma_pixels: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate center and tip heatmaps from 224-space points."""

    heatmap_config = HeatmapConfig(
        heatmap_height=HEATMAP_SIZE,
        heatmap_width=HEATMAP_SIZE,
        input_height=IMAGE_SIZE,
        input_width=IMAGE_SIZE,
        sigma_pixels=sigma_pixels,
    )
    center_x_norm = float(points_224[0, 0]) / float(IMAGE_SIZE - 1)
    center_y_norm = float(points_224[0, 1]) / float(IMAGE_SIZE - 1)
    tip_x_norm = float(points_224[1, 0]) / float(IMAGE_SIZE - 1)
    tip_y_norm = float(points_224[1, 1]) / float(IMAGE_SIZE - 1)
    center_heatmap, tip_heatmap = generate_center_tip_heatmaps(
        center_x_norm,
        center_y_norm,
        tip_x_norm,
        tip_y_norm,
        config=heatmap_config,
    )
    return center_heatmap.astype(np.float32), tip_heatmap.astype(np.float32)


class HeatmapSequence(keras.utils.Sequence):
    """Deterministic batch generator for the needle heatmap model."""

    def __init__(
        self,
        samples: list[NeedleSample],
        *,
        batch_size: int,
        augment: bool,
        seed: int,
        translation_ratio: float,
        scale_jitter: float,
        board_sample_weight: float,
        sigma_pixels: float,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.translation_ratio = float(translation_ratio)
        self.scale_jitter = float(scale_jitter)
        self.board_sample_weight = float(board_sample_weight)
        self.sigma_pixels = float(sigma_pixels)
        self.indices = np.arange(len(self.samples), dtype=np.int32)
        self._epoch = 0
        self._image_cache = [_load_crop_image(sample) for sample in self.samples]
        self._point_cache = [
            np.asarray(
                [[sample.center_x_224, sample.center_y_224], [sample.tip_x_224, sample.tip_y_224]],
                dtype=np.float32,
            )
            for sample in self.samples
        ]
        self._weight_cache = [np.float32(self.board_sample_weight if sample.is_board else 1.0) for sample in self.samples]

    def __len__(self) -> int:
        return int(math.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[np.ndarray]]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.samples))
        batch_indices = self.indices[start:stop]
        images: list[np.ndarray] = []
        center_targets: list[np.ndarray] = []
        tip_targets: list[np.ndarray] = []
        confidence_targets: list[np.ndarray] = []

        for order_index, sample_index in enumerate(batch_indices):
            sample_idx = int(sample_index)
            sample = self.samples[sample_idx]
            rng = np.random.default_rng(self.seed + self._epoch * 100_000 + sample_idx * 97 + order_index)
            image = self._image_cache[sample_idx]
            points = self._point_cache[sample_idx]
            if self.augment:
                image, points = _translate_canvas(
                    image,
                    points,
                    max_shift_px=max(1, int(round(self.translation_ratio * IMAGE_SIZE))),
                    rng=rng,
                )
                scale = float(rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
                image, points = _scale_canvas(image, points, scale=scale)
                image = _photometric_augment(image, rng)
            center_heatmap, tip_heatmap = _make_heatmaps(points, sigma_pixels=self.sigma_pixels)
            images.append(image.astype(np.float32) / 255.0)
            center_targets.append(center_heatmap)
            tip_targets.append(tip_heatmap)
            confidence_targets.append(np.array([sample.confidence], dtype=np.float32))

        x = np.stack(images, axis=0).astype(np.float32)
        y = [
            np.stack(center_targets, axis=0).astype(np.float32),
            np.stack(tip_targets, axis=0).astype(np.float32),
            np.stack(confidence_targets, axis=0).astype(np.float32),
        ]
        return x, y

    def on_epoch_end(self) -> None:
        """Shuffle sample order after each epoch when augmentation is enabled."""

        if not self.augment or len(self.indices) == 0:
            return
        rng = np.random.default_rng(self.seed + self._epoch)
        rng.shuffle(self.indices)
        self._epoch += 1


def _normalize_outputs(outputs: Any) -> dict[str, np.ndarray]:
    """Convert a prediction bundle into a stable name -> ndarray mapping."""

    if isinstance(outputs, dict):
        return {name: np.asarray(value, dtype=np.float32) for name, value in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return {f"output_{index}": np.asarray(value, dtype=np.float32) for index, value in enumerate(outputs)}
    return {"output": np.asarray(outputs, dtype=np.float32)}


def _extract_heatmap_outputs(outputs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, tip, and confidence outputs from a model prediction bundle."""

    output_map = _normalize_outputs(outputs)
    center = None
    tip = None
    confidence = None
    for key, tensor in output_map.items():
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) == 4 and shape[-1] == 1 and center is None:
            center = tensor
            continue
        if len(shape) == 4 and shape[-1] == 1 and tip is None:
            tip = tensor
            continue
        if len(shape) == 2 and shape[-1] == 1 and confidence is None:
            confidence = tensor
    if center is None or tip is None or confidence is None:
        ordered = list(output_map.values())
        if len(ordered) < 3:
            raise ValueError("Could not identify center, tip, and confidence outputs.")
        center, tip, confidence = ordered[:3]
    return np.asarray(center, dtype=np.float32), np.asarray(tip, dtype=np.float32), np.asarray(confidence, dtype=np.float32)


def _ordered_heatmap_tensors(outputs: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return center, tip, and confidence tensors in model output order."""

    if isinstance(outputs, dict):
        return outputs["center_heatmap"], outputs["tip_heatmap"], outputs["confidence"]
    if isinstance(outputs, (list, tuple)):
        if len(outputs) < 3:
            raise ValueError("Expected at least three outputs for heatmap prediction.")
        return outputs[0], outputs[1], outputs[2]
    raise TypeError(f"Unsupported output bundle type: {type(outputs)!r}")


def _decode_heatmap_points(heatmap: np.ndarray) -> tuple[float, float]:
    """Decode one heatmap into 224-space pixel coordinates."""

    heatmap = np.squeeze(np.asarray(heatmap, dtype=np.float32))
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape={heatmap.shape!r}")
    heatmap_sum = float(np.sum(heatmap))
    if not math.isfinite(heatmap_sum) or heatmap_sum <= 0.0:
        return 0.0, 0.0
    y_coords, x_coords = np.meshgrid(
        np.arange(heatmap.shape[0], dtype=np.float32),
        np.arange(heatmap.shape[1], dtype=np.float32),
        indexing="ij",
    )
    normalized = heatmap / heatmap_sum
    x_idx = float(np.sum(normalized * x_coords))
    y_idx = float(np.sum(normalized * y_coords))
    x_px = x_idx * float(IMAGE_SIZE - 1) / float(heatmap.shape[1] - 1)
    y_px = y_idx * float(IMAGE_SIZE - 1) / float(heatmap.shape[0] - 1)
    return float(x_px), float(y_px)


def _evaluate_sequence(model: keras.Model, sequence: HeatmapSequence) -> dict[str, float]:
    """Compute needle geometry errors on a sequence."""

    center_errors: list[float] = []
    tip_errors: list[float] = []
    angle_errors: list[float] = []
    board_center_errors: list[float] = []
    board_tip_errors: list[float] = []
    pxl_center_errors: list[float] = []
    pxl_tip_errors: list[float] = []

    for batch_index in range(len(sequence)):
        batch = sequence[batch_index]
        batch_x = batch[0]
        batch_y = batch[1]
        predictions = model.predict(batch_x, verbose=0)
        pred_center, pred_tip, pred_conf = _extract_heatmap_outputs(predictions)
        _ = pred_conf
        true_center = np.asarray(batch_y[0], dtype=np.float32)
        true_tip = np.asarray(batch_y[1], dtype=np.float32)
        batch_start = batch_index * sequence.batch_size
        for offset, sample in enumerate(sequence.samples[batch_start : batch_start + len(batch_x)]):
            pred_center_xy = _decode_heatmap_points(pred_center[offset])
            pred_tip_xy = _decode_heatmap_points(pred_tip[offset])
            true_center_xy = _decode_heatmap_points(true_center[offset])
            true_tip_xy = _decode_heatmap_points(true_tip[offset])
            center_error = float(math.hypot(pred_center_xy[0] - true_center_xy[0], pred_center_xy[1] - true_center_xy[1]))
            tip_error = float(math.hypot(pred_tip_xy[0] - true_tip_xy[0], pred_tip_xy[1] - true_tip_xy[1]))
            pred_angle = angle_degrees_from_center_to_tip(
                pred_center_xy[0], pred_center_xy[1], pred_tip_xy[0], pred_tip_xy[1]
            )
            true_angle = angle_degrees_from_center_to_tip(
                true_center_xy[0], true_center_xy[1], true_tip_xy[0], true_tip_xy[1]
            )
            angle_error = float(abs(((pred_angle - true_angle + 180.0) % 360.0) - 180.0))
            center_errors.append(center_error)
            tip_errors.append(tip_error)
            angle_errors.append(angle_error)
            if sample.is_board:
                board_center_errors.append(center_error)
                board_tip_errors.append(tip_error)
            else:
                pxl_center_errors.append(center_error)
                pxl_tip_errors.append(tip_error)

    report = {
        "center_mae_px": float(np.mean(center_errors)) if center_errors else float("nan"),
        "tip_mae_px": float(np.mean(tip_errors)) if tip_errors else float("nan"),
        "angle_mae_degrees": float(np.mean(angle_errors)) if angle_errors else float("nan"),
    }
    if board_center_errors:
        report["board_center_mae_px"] = float(np.mean(board_center_errors))
        report["board_tip_mae_px"] = float(np.mean(board_tip_errors))
    if pxl_center_errors:
        report["pxl_center_mae_px"] = float(np.mean(pxl_center_errors))
        report["pxl_tip_mae_px"] = float(np.mean(pxl_tip_errors))
    return report


def _load_teacher_model(model_path: Path) -> keras.Model:
    """Load a heatmap teacher model for distillation."""

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    custom_objects = {
        "Identity3x3Initializer": Identity3x3Initializer,
        "BilinearUpsamplingInitializer": BilinearUpsamplingInitializer,
        "embedded_gauge_reading_tinyml>Identity3x3Initializer": Identity3x3Initializer,
        "embedded_gauge_reading_tinyml>BilinearUpsamplingInitializer": BilinearUpsamplingInitializer,
    }
    try:
        return standalone_keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects,
        )
    except Exception:
        return keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects,
        )


class HeatmapKDTrainingModel(keras.Model):
    """Wrap a heatmap model with supervised and teacher distillation losses."""

    def __init__(
        self,
        *,
        base_model: keras.Model,
        teacher_model: keras.Model,
        distillation_weight: float,
        heatmap_distillation_weight: float,
        confidence_distillation_weight: float,
    ) -> None:
        super().__init__(name="board_heatmap_kd_training_model")
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.distillation_weight = float(distillation_weight)
        self.heatmap_distillation_weight = float(heatmap_distillation_weight)
        self.confidence_distillation_weight = float(confidence_distillation_weight)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> list[tf.Tensor]:
        """Forward inputs through the base model."""

        outputs = self.base_model(inputs, training=training)
        if isinstance(outputs, (list, tuple)):
            return [outputs[0], outputs[1], outputs[2]]
        if isinstance(outputs, dict):
            return [outputs["center_heatmap"], outputs["tip_heatmap"], outputs["confidence"]]
        raise TypeError(f"Unsupported model output type: {type(outputs)!r}")

    def _distillation_loss(self, pred: Any, teacher: Any) -> tf.Tensor:
        """Match the student outputs against frozen teacher outputs."""

        pred_center, pred_tip, pred_conf = _ordered_heatmap_tensors(pred)
        teacher_center, teacher_tip, teacher_conf = _ordered_heatmap_tensors(teacher)
        center_loss = tf.reduce_mean(tf.square(tf.cast(pred_center, tf.float32) - tf.cast(teacher_center, tf.float32)))
        tip_loss = tf.reduce_mean(tf.square(tf.cast(pred_tip, tf.float32) - tf.cast(teacher_tip, tf.float32)))
        confidence_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(
                tf.cast(teacher_conf, tf.float32),
                tf.cast(pred_conf, tf.float32),
            )
        )
        return (
            self.heatmap_distillation_weight * (center_loss + tip_loss)
            + self.confidence_distillation_weight * confidence_loss
        )

    def train_step(self, data: Any) -> dict[str, float]:
        """Run one optimization step with teacher distillation."""

        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
            teacher_outputs = self.teacher_model(x, training=False)
            distillation_loss = self._distillation_loss(pred, teacher_outputs)
            total_loss = main_loss + self.distillation_weight * distillation_loss
        gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return {
            "loss": tf.cast(total_loss, tf.float32),
            "main_loss": tf.cast(main_loss, tf.float32),
            "distillation_loss": tf.cast(distillation_loss, tf.float32),
        }

    def test_step(self, data: Any) -> dict[str, float]:
        """Run one validation step without updates."""

        x, y = data
        pred = self(x, training=False)
        main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
        return {"loss": tf.cast(main_loss, tf.float32), "main_loss": tf.cast(main_loss, tf.float32)}


def _compile_model(model: keras.Model, *, learning_rate: float) -> None:
    """Compile a heatmap model with supervised losses and coordinate metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=[
            weighted_center_heatmap_loss,
            focal_heatmap_loss,
            keras.losses.BinaryCrossentropy(),
        ],
        loss_weights=[1.0, 2.0, 0.01],
    )


def _quantize_model(model: keras.Model) -> keras.Model:
    """Create a QAT clone of a heatmap model."""

    with tfmot.quantization.keras.quantize_scope():
        return tfmot.quantization.keras.quantize_model(model)


def _representative_dataset(sequence: HeatmapSequence, count: int) -> Any:
    """Yield a small representative calibration set for TFLite export."""

    seen = 0
    for batch_index in range(len(sequence)):
        batch = sequence[batch_index]
        batch_x = batch[0]
        for sample in batch_x:
            yield [np.asarray(sample[np.newaxis, ...], dtype=np.float32)]
            seen += 1
            if seen >= count:
                return


def _load_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    """Persist a training summary JSON file."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_model_or_weights(model: keras.Model, model_path: Path) -> dict[str, str]:
    """Save a model if possible, otherwise fall back to weights only.

    The mixed tf_keras / standalone-keras environment can make full Functional
    serialization brittle even when training and inference are fine.  For this
    pipeline, preserving the trained weights is enough because the downstream
    deliverable is the TFLite export and the evaluation summary.
    """

    try:
        model.save(model_path)
        return {"artifact_path": model_path.as_posix(), "artifact_kind": "keras_model"}
    except Exception as exc:
        fallback_path = model_path.with_suffix(".weights.h5")
        model.save_weights(fallback_path)
        return {
            "artifact_path": fallback_path.as_posix(),
            "artifact_kind": "weights_only",
            "save_error": str(exc),
        }


def main() -> None:
    """Train, distill, quantize, and export the board heatmap model."""

    args = _parse_args()
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1).")
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in (0, 1).")
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

    samples = _build_samples(
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

    train_sequence = HeatmapSequence(
        train_samples,
        batch_size=args.batch_size,
        augment=True,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
        sigma_pixels=args.sigma_pixels,
    )
    val_sequence = HeatmapSequence(
        val_samples,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
        sigma_pixels=args.sigma_pixels,
    )
    test_sequence = HeatmapSequence(
        test_samples,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        translation_ratio=args.translation_ratio,
        scale_jitter=args.scale_jitter,
        board_sample_weight=args.board_sample_weight,
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

    base_model = build_qat_friendly_heatmap_angle_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        heatmap_size=HEATMAP_SIZE,
        encoder_width_multiplier=1.0,
        decoder_width_multiplier=1.0,
    )
    model = base_model
    set_heatmap_encoder_trainable(model, trainable=False)

    _compile_model(model, learning_rate=args.learning_rate)

    warmup_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(3, args.pretrain_epochs // 2),
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
    print("Phase 1: warmup training with the backbone frozen...", flush=True)
    model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.pretrain_epochs,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    print("Phase 2: fine-tuning with teacher distillation...", flush=True)
    set_heatmap_encoder_trainable(model, trainable=True)
    _compile_model(model, learning_rate=args.kd_learning_rate)
    teacher_model = _load_teacher_model(args.teacher_model_path)
    teacher_model.trainable = False
    kd_model = HeatmapKDTrainingModel(
        base_model=model,
        teacher_model=teacher_model,
        distillation_weight=0.15,
        heatmap_distillation_weight=0.25,
        confidence_distillation_weight=0.05,
    )
    _compile_model(kd_model, learning_rate=args.kd_learning_rate)
    kd_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.kd_epochs,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    float_test_report = _evaluate_sequence(model, test_sequence)
    print(json.dumps({"float_test_report": float_test_report}, indent=2, sort_keys=True), flush=True)

    float_model_path = args.output_dir / "heatmap_float.keras"
    float_artifact = _save_model_or_weights(model, float_model_path)

    ptq_tflite_path = args.output_dir / "heatmap_ptq.tflite"
    ptq_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ptq_converter.representative_dataset = lambda: _representative_dataset(
        train_sequence,
        count=min(32, len(train_samples)),
    )
    ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    ptq_converter.inference_input_type = tf.int8
    ptq_converter.inference_output_type = tf.int8
    ptq_tflite_path.write_bytes(ptq_converter.convert())
    ptq_interpreter = tf.lite.Interpreter(model_path=str(ptq_tflite_path), num_threads=1)
    ptq_interpreter.allocate_tensors()

    print("Phase 3: QAT fine-tuning...", flush=True)
    qat_model_path = args.output_dir / "heatmap_qat.keras"
    qat_tflite_path = args.output_dir / "heatmap_qat.tflite"
    qat_test_report: dict[str, float] | None = None
    try:
        qat_base = _quantize_model(base_model)
        _compile_model(qat_base, learning_rate=args.qat_learning_rate)
        qat_teacher = _load_teacher_model(args.teacher_model_path)
        qat_teacher.trainable = False
        qat_model = HeatmapKDTrainingModel(
            base_model=qat_base,
            teacher_model=qat_teacher,
            distillation_weight=0.15,
            heatmap_distillation_weight=0.25,
            confidence_distillation_weight=0.05,
        )
        _compile_model(qat_model, learning_rate=args.qat_learning_rate)
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
        qat_test_report = _evaluate_sequence(qat_base, test_sequence)
        print(json.dumps({"qat_test_report": qat_test_report}, indent=2, sort_keys=True), flush=True)

        qat_artifact = _save_model_or_weights(qat_base, qat_model_path)
        qat_converter = tf.lite.TFLiteConverter.from_keras_model(qat_base)
        qat_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        qat_converter.representative_dataset = lambda: _representative_dataset(
            train_sequence,
            count=min(32, len(train_samples)),
        )
        qat_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        qat_converter.inference_input_type = tf.int8
        qat_converter.inference_output_type = tf.int8
        qat_tflite_path.write_bytes(qat_converter.convert())
    except Exception as exc:
        print(f"QAT path skipped: {exc}", flush=True)
        qat_artifact = {"artifact_kind": "skipped", "error": str(exc)}

    summary = {
        "manifest": args.manifest.as_posix(),
        "teacher_model_path": args.teacher_model_path.as_posix(),
        "sample_count": len(samples),
        "board_count": int(sum(1 for sample in samples if sample.is_board)),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "float_artifact": float_artifact,
        "float_model_path": float_model_path.as_posix(),
        "ptq_tflite_path": ptq_tflite_path.as_posix(),
        "qat_artifact": qat_artifact,
        "qat_model_path": qat_model_path.as_posix(),
        "qat_tflite_path": qat_tflite_path.as_posix(),
        "float_test_report": float_test_report,
        "qat_test_report": qat_test_report,
        "source_kind_counts": source_kind_counts,
        "config": {
            "image_size": IMAGE_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "kd_learning_rate": args.kd_learning_rate,
            "qat_learning_rate": args.qat_learning_rate,
            "translation_ratio": args.translation_ratio,
            "scale_jitter": args.scale_jitter,
            "board_sample_weight": args.board_sample_weight,
            "sigma_pixels": args.sigma_pixels,
            "source_kinds": sorted(allowed_source_kinds),
        },
    }
    _load_summary(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
