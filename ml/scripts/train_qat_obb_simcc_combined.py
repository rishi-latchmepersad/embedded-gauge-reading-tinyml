#!/usr/bin/env python3
"""Train a QAT-friendly MobileNetV2 center detector + SimCC gauge localizer.

The grouped manifest now starts from the clean PXL geometry CSV plus the hard
cases and the two smaller manual annotation sets. Those geometry rows carry the
center, tip, radius, and angle labels directly, and every temperature-labeled
row also exposes ``true_angle_degrees``. The center detector is distilled from
an existing teacher model so the first-stage crop remains robust while the
student model stays small enough for TFLite QAT.
"""

from __future__ import annotations

import argparse
import json
import math
import os as _os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3800"))
_SKIP_EXPLICIT_GPU_CONFIG = _os.environ.get("TF_SKIP_EXPLICIT_GPU_CONFIG", "0") == "1"
import tensorflow as tf

if not _SKIP_EXPLICIT_GPU_CONFIG:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
        )
del _os, _GPU_MEMORY_LIMIT_MB, _SKIP_EXPLICIT_GPU_CONFIG

import keras as standalone_keras
import tf_keras as keras
import tensorflow_model_optimization as tfmot

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    load_rgb_image,
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.gauge.processing import (  # noqa: E402
    GaugeSpec,
    fraction_to_angle_rad,
    load_gauge_specs,
    value_to_fraction,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    firmware_training_crop_box,
)
from embedded_gauge_reading_tinyml.obb_simcc_tf_models import (  # noqa: E402
    build_mobilenetv2_center_simcc_model,
)

MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "labelled_captured_images.json"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "artifacts" / "training" / "center_simcc_combined_qat"
# Standardize every training and evaluation sample onto the same 224x224
# crop-plus-pad canvas so the student model sees one consistent input geometry.
IMAGE_SIZE: int = 224
NUM_BINS: int = 112
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_WARMUP_EPOCHS: int = 12
DEFAULT_FINETUNE_EPOCHS: int = 8
DEFAULT_QAT_EPOCHS: int = 8
DEFAULT_ALPHA: float = 0.35
DEFAULT_SPATIAL_CHANNELS: int = 64
DEFAULT_HEAD_UNITS: int = 96
DEFAULT_HEAD_DROPOUT: float = 0.15
DEFAULT_LEARNING_RATE: float = 5e-4
DEFAULT_QAT_LEARNING_RATE: float = 1e-4
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.10
DEFAULT_SIMCC_SIGMA_BINS: float = 1.75
DEFAULT_CENTER_LOSS_WEIGHT: float = 2.0
DEFAULT_CENTER_DISTILL_WEIGHT: float = 0.25
DEFAULT_CENTER_DISTILL_TEMPERATURE: float = 1.0
DEFAULT_CENTER_DISTILL_HUBER_DELTA: float = 0.03
DEFAULT_TEMPERATURE_LOSS_WEIGHT: float = 0.25

_CENTER_PRIORITY: dict[str, int] = {
    "reviewed_geometry": 5,
    "pxl_geometry": 4,
    "board_tip_geometry": 3,
    "center_radii": 2,
    "center_only": 1,
    "board_geometry": 0,
}
_TIP_PRIORITY: dict[str, int] = {
    "reviewed_geometry": 3,
    "pxl_geometry": 2,
    "board_tip_geometry": 1,
}


@dataclass(frozen=True)
class CombinedSample:
    """One grouped manifest row with the geometry labels and crop box we trust."""

    image_path: Path
    source_width: int
    source_height: int
    crop_box_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float] | None
    center_weight: float
    tip_xy: tuple[float, float] | None
    tip_weight: float
    temperature_c: float | None
    temperature_weight: float
    source_kinds: tuple[str, ...]

    @property
    def has_geometry(self) -> bool:
        """Return ``True`` when the row can train at least one geometry head."""

        return self.center_xy is not None or self.tip_xy is not None

    @property
    def has_temperature(self) -> bool:
        """Return ``True`` when the row carries a temperature label."""

        return self.temperature_c is not None


def _as_float(value: Any) -> float | None:
    """Coerce a JSON scalar into ``float`` when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _float_from_row(row: dict[str, Any], *keys: str) -> float | None:
    """Return the first finite float value found under the requested row keys."""

    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _point_from_row(
    row: dict[str, Any],
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
) -> tuple[float, float] | None:
    """Return a row point when both coordinates are present."""

    x_value = _float_from_row(row, *x_keys)
    y_value = _float_from_row(row, *y_keys)
    if x_value is None or y_value is None:
        return None
    return x_value, y_value


def _is_original_capture(filename: str) -> bool:
    """Filter out derivative preview images that should not dominate training."""

    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True


def _has_center(annotation: dict[str, Any]) -> bool:
    """Return ``True`` when the annotation exposes a center point."""

    row = annotation["source_row"]
    return _point_from_row(
        row,
        ("center_x", "center_x_source"),
        ("center_y", "center_y_source"),
    ) is not None


def _has_tip(annotation: dict[str, Any]) -> bool:
    """Return ``True`` when the annotation exposes a tip point."""

    row = annotation["source_row"]
    return _point_from_row(
        row,
        ("tip_x", "tip_x_source"),
        ("tip_y", "tip_y_source"),
    ) is not None


def _has_radius(annotation: dict[str, Any]) -> bool:
    """Return ``True`` when the annotation exposes an explicit dial radius."""

    row = annotation["source_row"]
    return _float_from_row(row, "outer_radius", "dial_radius_source") is not None


def _has_temperature(annotation: dict[str, Any]) -> bool:
    """Return ``True`` when the annotation carries a temperature label."""

    row = annotation["source_row"]
    return _as_float(row.get("temperature_c")) is not None or _as_float(row.get("value")) is not None


def _annotation_temperature_c(annotation: dict[str, Any]) -> float | None:
    """Extract the temperature label from a grouped-manifest annotation."""

    row = annotation["source_row"]
    return _float_from_row(row, "temperature_c", "value")


def _annotation_true_angle_degrees(annotation: dict[str, Any]) -> float | None:
    """Extract the best available needle angle from a grouped-manifest annotation."""

    row = annotation["source_row"]
    angle = _float_from_row(row, "true_angle_degrees", "angle_degrees_from_labels", "angle_degrees")
    if angle is not None:
        return angle % 360.0

    temperature_c = _annotation_temperature_c(annotation)
    if temperature_c is not None:
        return _temperature_to_true_angle_degrees(temperature_c, _default_gauge_spec())
    return None


def _has_angle(annotation: dict[str, Any]) -> bool:
    """Return ``True`` when the annotation can supervise the angle head."""

    return _annotation_true_angle_degrees(annotation) is not None


def _pick_annotation(
    annotations: list[dict[str, Any]],
    *,
    predicate: Any,
    priority_map: dict[str, int],
) -> dict[str, Any] | None:
    """Pick the highest-priority annotation that satisfies ``predicate``."""

    candidates = [annotation for annotation in annotations if predicate(annotation)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda annotation: (
            priority_map.get(str(annotation["source_kind"]), -1),
            int(annotation.get("source_row_index", 0)),
        ),
    )


def _resolve_source_size(
    image_path: Path,
    annotations: list[dict[str, Any]],
) -> tuple[int, int]:
    """Resolve the source image dimensions from labels or from the file itself."""

    for annotation in annotations:
        row = annotation["source_row"]
        source_width = _as_float(row.get("source_width"))
        source_height = _as_float(row.get("source_height"))
        if source_width is not None and source_height is not None:
            width = int(round(source_width))
            height = int(round(source_height))
            if width > 0 and height > 0:
                return width, height

    absolute_path = REPO_ROOT / image_path
    suffix = absolute_path.suffix.lower()
    if suffix == ".yuv422":
        file_size = absolute_path.stat().st_size
        inferred_pixels = file_size / 2.0
        inferred_dim = int(round(math.sqrt(inferred_pixels)))
        if inferred_dim > 0 and inferred_dim * inferred_dim * 2 == file_size:
            return inferred_dim, inferred_dim
        raise ValueError(
            f"{image_path} is a raw YUV capture, but no source_width/source_height "
            "labels were present."
        )
    with Image.open(absolute_path) as image:
        width, height = image.size
    return int(width), int(height)


def _normalize_point(
    x: float,
    y: float,
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """Normalize a point into ``[0, 1]`` coordinates."""

    width = max(float(source_width), 1.0)
    height = max(float(source_height), 1.0)
    return (float(x) / width, float(y) / height)


def _map_point_to_canvas_norm(
    point_xy: tuple[float, float],
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    image_size: int = IMAGE_SIZE,
) -> tuple[float, float]:
    """Map a source-space point into the resized 224x224 canvas."""

    x_min, y_min, x_max, y_max = crop_box_xyxy
    x_min_i = max(0.0, float(math.floor(x_min)))
    y_min_i = max(0.0, float(math.floor(y_min)))
    x_max_i = min(float(source_width), float(math.ceil(x_max)))
    y_max_i = min(float(source_height), float(math.ceil(y_max)))
    if x_max_i <= x_min_i:
        x_max_i = min(float(source_width), x_min_i + 1.0)
    if y_max_i <= y_min_i:
        y_max_i = min(float(source_height), y_min_i + 1.0)

    crop_width = max(x_max_i - x_min_i, 1.0)
    crop_height = max(y_max_i - y_min_i, 1.0)
    scale = min(float(image_size) / crop_width, float(image_size) / crop_height)
    resized_width = crop_width * scale
    resized_height = crop_height * scale
    pad_x = 0.5 * (float(image_size) - resized_width)
    pad_y = 0.5 * (float(image_size) - resized_height)

    point_x, point_y = point_xy
    canvas_x = (float(point_x) - x_min_i) * scale + pad_x
    canvas_y = (float(point_y) - y_min_i) * scale + pad_y
    return (
        float(np.clip(canvas_x / float(image_size), 0.0, 1.0)),
        float(np.clip(canvas_y / float(image_size), 0.0, 1.0)),
    )


def _normalize_size(
    width_px: float,
    height_px: float,
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """Normalize a box size into ``[0, 1]`` coordinates."""

    return (
        float(width_px) / max(float(source_width), 1.0),
        float(height_px) / max(float(source_height), 1.0),
    )


def _make_simcc_target(
    coord_norm: float,
    *,
    num_bins: int = NUM_BINS,
    sigma_bins: float = DEFAULT_SIMCC_SIGMA_BINS,
) -> np.ndarray:
    """Create a 1D Gaussian SimCC target over ``num_bins`` bins."""

    coord_clamped = float(np.clip(coord_norm, 0.0, 1.0))
    bins = np.arange(num_bins, dtype=np.float32)
    center = coord_clamped * float(num_bins - 1)
    target = np.exp(-((bins - center) ** 2) / (2.0 * sigma_bins * sigma_bins))
    target_sum = float(target.sum())
    if target_sum > 0.0:
        target /= target_sum
    return target.astype(np.float32)


def _angle_to_sincos(angle_degrees: float) -> np.ndarray:
    """Convert an angle in degrees to a unit sine/cosine vector."""

    angle_rad = math.radians(angle_degrees)
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)


@lru_cache(maxsize=1)
def _default_gauge_spec() -> GaugeSpec:
    """Return the single gauge calibration spec used by this dataset."""

    specs = load_gauge_specs()
    if len(specs) == 1:
        return next(iter(specs.values()))
    if "littlegood_home_temp_gauge_c" in specs:
        return specs["littlegood_home_temp_gauge_c"]
    available = ", ".join(sorted(specs))
    raise ValueError(
        "Expected exactly one gauge calibration spec for the grouped manifest; "
        f"available gauge ids: {available}"
    )


def _temperature_to_true_angle_degrees(temperature_c: float, spec: GaugeSpec) -> float:
    """Map a temperature label to the corresponding needle angle in degrees."""

    fraction = value_to_fraction(temperature_c, spec)
    angle_rad = fraction_to_angle_rad(fraction, spec)
    return math.degrees(angle_rad) % 360.0


def _default_dial_diameter_ratio() -> float:
    """Return the dial diameter ratio used as a weak OBB size prior."""

    spec = _default_gauge_spec()
    return float(2.0 * spec.inner_dial_radius_frame_ratio)


def _select_combined_sample(
    entry: dict[str, Any],
    *,
    include_temperature_head: bool,
) -> CombinedSample | None:
    """Convert one grouped-manifest image entry into a training sample."""

    image_path = Path(str(entry["image_path"]))
    if not _is_original_capture(image_path.name):
        return None

    annotations = list(entry["annotations"])
    source_width, source_height = _resolve_source_size(image_path, annotations)
    crop_box_xyxy = firmware_training_crop_box(source_width, source_height)

    center_annotation = _pick_annotation(
        annotations,
        predicate=_has_center,
        priority_map=_CENTER_PRIORITY,
    )
    tip_annotation = _pick_annotation(
        annotations,
        predicate=_has_tip,
        priority_map=_TIP_PRIORITY,
    )

    center_xy: tuple[float, float] | None = None
    center_weight = 0.0
    if center_annotation is not None:
        row = center_annotation["source_row"]
        center_point = _point_from_row(
            row,
            ("center_x", "center_x_source"),
            ("center_y", "center_y_source"),
        )
        if center_point is not None:
            center_x, center_y = center_point
            center_xy = _normalize_point(
                center_x,
                center_y,
                source_width=source_width,
                source_height=source_height,
            )
            center_weight = {
                "reviewed_geometry": 1.0,
                "pxl_geometry": 1.0,
                "board_tip_geometry": 1.0,
                "center_radii": 1.0,
                "center_only": 0.9,
                "board_geometry": 0.6,
            }.get(str(center_annotation["source_kind"]), 0.5)

    tip_xy: tuple[float, float] | None = None
    tip_weight = 0.0
    if tip_annotation is not None:
        row = tip_annotation["source_row"]
        tip_point = _point_from_row(
            row,
            ("tip_x", "tip_x_source"),
            ("tip_y", "tip_y_source"),
        )
        if tip_point is not None:
            tip_x, tip_y = tip_point
            tip_xy = _normalize_point(
                tip_x,
                tip_y,
                source_width=source_width,
                source_height=source_height,
            )
            tip_weight = 1.0

    temperature_c = None
    temperature_weight = 0.0
    if include_temperature_head:
        temperature_annotation = _pick_annotation(
            annotations,
            predicate=_has_temperature,
            priority_map={
                "reviewed_geometry": 4,
                "temperature_only": 3,
                "board_tip_geometry": 2,
                "board_geometry": 2,
                "center_radii": 1,
                "center_only": 0,
                "pxl_geometry": 0,
            },
        )
        if temperature_annotation is not None:
            temperature_c = _annotation_temperature_c(temperature_annotation)
            if temperature_c is not None:
                temperature_weight = 1.0

    if (
        center_xy is None
        and tip_xy is None
        and not (include_temperature_head and temperature_c is not None)
    ):
        return None

    source_kinds = tuple(
        sorted(
            {
                str(annotation["source_kind"])
                for annotation in annotations
            }
        )
    )
    return CombinedSample(
        image_path=image_path,
        source_width=source_width,
        source_height=source_height,
        crop_box_xyxy=crop_box_xyxy,
        center_xy=center_xy,
        center_weight=center_weight,
        tip_xy=tip_xy,
        tip_weight=tip_weight,
        temperature_c=temperature_c,
        temperature_weight=temperature_weight,
        source_kinds=source_kinds,
    )


def load_combined_samples(
    manifest_path: Path,
    *,
    include_temperature_head: bool,
) -> list[CombinedSample]:
    """Load and filter the grouped manifest into training samples."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload["images"]
    samples: list[CombinedSample] = []
    for entry in images:
        sample = _select_combined_sample(
            entry,
            include_temperature_head=include_temperature_head,
        )
        if sample is not None:
            absolute_path = REPO_ROOT / sample.image_path
            if absolute_path.exists():
                samples.append(sample)
    return samples


def _load_image_array(sample: CombinedSample) -> np.ndarray:
    """Load one source image into an RGB ``uint8`` array."""

    absolute_path = REPO_ROOT / sample.image_path
    suffix = absolute_path.suffix.lower()
    if suffix == ".yuv422":
        image = load_yuv422_capture_as_rgb(
            absolute_path,
            image_width=sample.source_width,
            image_height=sample.source_height,
        )
    else:
        image = load_rgb_image(absolute_path)
    return np.asarray(image, dtype=np.uint8)


def _resize_image(
    image: np.ndarray,
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Crop one source image and resize it onto the shared 224x224 canvas."""

    return resize_with_pad_rgb(image, crop_box_xyxy, image_size=image_size)


def _augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply light photometric augmentation without changing label geometry."""

    image_f = image.astype(np.float32)
    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.90, 1.10))
    noise = rng.normal(0.0, 2.0, size=image_f.shape).astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)
    image_f = (image_f - mean) * contrast + mean
    image_f = image_f * brightness + noise
    return np.clip(image_f, 0.0, 255.0).astype(np.uint8)


def _encode_targets(
    sample: CombinedSample,
    *,
    include_temperature_head: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode one sample into model targets and sample weights."""

    targets: dict[str, np.ndarray] = {}
    sample_weights: dict[str, np.ndarray] = {}

    if sample.center_xy is not None:
        center_x, center_y = sample.center_xy
        center_x_canvas, center_y_canvas = _map_point_to_canvas_norm(
            (
                center_x * float(sample.source_width),
                center_y * float(sample.source_height),
            ),
            crop_box_xyxy=sample.crop_box_xyxy,
            source_width=sample.source_width,
            source_height=sample.source_height,
        )
        targets["center_xy"] = np.array([center_x_canvas, center_y_canvas], dtype=np.float32)
        targets["center_x_simcc"] = _make_simcc_target(center_x_canvas)
        targets["center_y_simcc"] = _make_simcc_target(center_y_canvas)
        sample_weights["center_xy"] = np.array(sample.center_weight, dtype=np.float32)
        sample_weights["center_x_simcc"] = np.array(sample.center_weight, dtype=np.float32)
        sample_weights["center_y_simcc"] = np.array(sample.center_weight, dtype=np.float32)
    else:
        targets["center_xy"] = np.zeros((2,), dtype=np.float32)
        targets["center_x_simcc"] = np.zeros((NUM_BINS,), dtype=np.float32)
        targets["center_y_simcc"] = np.zeros((NUM_BINS,), dtype=np.float32)
        sample_weights["center_xy"] = np.array(0.0, dtype=np.float32)
        sample_weights["center_x_simcc"] = np.array(0.0, dtype=np.float32)
        sample_weights["center_y_simcc"] = np.array(0.0, dtype=np.float32)

    if sample.tip_xy is not None:
        tip_x, tip_y = sample.tip_xy
        tip_x_canvas, tip_y_canvas = _map_point_to_canvas_norm(
            (
                tip_x * float(sample.source_width),
                tip_y * float(sample.source_height),
            ),
            crop_box_xyxy=sample.crop_box_xyxy,
            source_width=sample.source_width,
            source_height=sample.source_height,
        )
        targets["tip_x_simcc"] = _make_simcc_target(tip_x_canvas)
        targets["tip_y_simcc"] = _make_simcc_target(tip_y_canvas)
        sample_weights["tip_x_simcc"] = np.array(sample.tip_weight, dtype=np.float32)
        sample_weights["tip_y_simcc"] = np.array(sample.tip_weight, dtype=np.float32)
    else:
        targets["tip_x_simcc"] = np.zeros((NUM_BINS,), dtype=np.float32)
        targets["tip_y_simcc"] = np.zeros((NUM_BINS,), dtype=np.float32)
        sample_weights["tip_x_simcc"] = np.array(0.0, dtype=np.float32)
        sample_weights["tip_y_simcc"] = np.array(0.0, dtype=np.float32)

    if include_temperature_head:
        targets["gauge_value"] = np.array(
            [sample.temperature_c if sample.temperature_c is not None else 0.0],
            dtype=np.float32,
        )
        sample_weights["gauge_value"] = np.array(sample.temperature_weight, dtype=np.float32)

    return targets, sample_weights


class CombinedGeometrySequence(keras.utils.Sequence):
    """Small Python sequence that keeps the manifest labels aligned with images."""

    def __init__(
        self,
        samples: list[CombinedSample],
        *,
        batch_size: int,
        include_temperature_head: bool,
        augment: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.include_temperature_head = include_temperature_head
        self.augment = augment
        self.seed = int(seed)
        self.indices = np.arange(len(self.samples), dtype=np.int32)
        self._epoch = 0
        # Cache the resized 224x224 inputs once so we do not repeat the heavy
        # crop/resize work every epoch. The augmentation only changes colour,
        # so we can safely apply it after the resize step.
        self._image_cache = [
            _resize_image(
                _load_image_array(sample),
                crop_box_xyxy=sample.crop_box_xyxy,
                image_size=IMAGE_SIZE,
            )
            for sample in self.samples
        ]
        self._target_cache = [
            _encode_targets(sample, include_temperature_head=self.include_temperature_head)[0]
            for sample in self.samples
        ]
        self._weight_cache = [
            _encode_targets(sample, include_temperature_head=self.include_temperature_head)[1]
            for sample in self.samples
        ]

    def __len__(self) -> int:
        """Return the number of batches in the sequence."""

        return int(math.ceil(len(self.samples) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        """Shuffle sample order after each epoch when augmentation is enabled."""

        if len(self.indices) == 0:
            return
        rng = np.random.default_rng(self.seed + self._epoch)
        rng.shuffle(self.indices)
        self._epoch += 1

    def __getitem__(
        self,
        index: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load one batch of images, targets, and sample weights."""

        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.samples))
        batch_indices = self.indices[start:stop]

        images: list[np.ndarray] = []
        targets: dict[str, list[np.ndarray]] = {}
        sample_weights: dict[str, list[np.ndarray]] = {}

        for order_index, sample_index in enumerate(batch_indices):
            sample_idx = int(sample_index)
            sample = self.samples[sample_idx]
            image = self._image_cache[sample_idx]
            if self.augment:
                rng = np.random.default_rng(
                    self.seed + self._epoch * 100_000 + sample_idx * 997 + order_index
                )
                image = _augment_image(image, rng)
            image = image.astype(np.float32) / 127.5 - 1.0
            batch_targets = self._target_cache[sample_idx]
            batch_weights = self._weight_cache[sample_idx]
            images.append(image)
            for key, value in batch_targets.items():
                targets.setdefault(key, []).append(value)
            for key, value in batch_weights.items():
                sample_weights.setdefault(key, []).append(value)

        x = np.stack(images, axis=0).astype(np.float32)
        y = {key: np.stack(values, axis=0).astype(np.float32) for key, values in targets.items()}
        w = {key: np.asarray(values, dtype=np.float32).reshape(-1) for key, values in sample_weights.items()}
        return x, y, w


def _set_backbone_trainable(model: keras.Model, *, trainable: bool) -> None:
    """Flip the MobileNetV2 backbone trainability in-place."""

    backbone_layers = getattr(model, "_mobilenet_backbone_layers", None)
    if not isinstance(backbone_layers, dict):
        return
    for layer in backbone_layers.values():
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = trainable


def _prediction_dict(model: keras.Model, predictions: Any) -> dict[str, Any]:
    """Normalize Keras prediction output into a name->value dictionary."""

    if isinstance(predictions, dict):
        return dict(predictions)
    return {name: predictions[index] for index, name in enumerate(model.output_names)}


def _soft_argmax_1d(distribution: np.ndarray) -> np.ndarray:
    """Decode a SimCC distribution into a normalized coordinate."""

    bins = np.linspace(0.0, 1.0, distribution.shape[-1], dtype=np.float32)
    return np.sum(distribution * bins[np.newaxis, :], axis=-1)


def _soft_argmax_1d_tf(distribution: tf.Tensor) -> tf.Tensor:
    """Decode a SimCC distribution into a normalized coordinate tensor."""

    last_dim = tf.shape(distribution)[-1]
    bins = tf.cast(tf.linspace(0.0, 1.0, last_dim), tf.float32)
    return tf.reduce_sum(tf.cast(distribution, tf.float32) * bins[tf.newaxis, :], axis=-1)


def _weighted_mean(loss_values: tf.Tensor, sample_weight: tf.Tensor | None) -> tf.Tensor:
    """Average a per-sample loss with optional sample weights."""

    values = tf.reshape(tf.cast(loss_values, tf.float32), [-1])
    if sample_weight is None:
        return tf.reduce_mean(values)

    weights = tf.reshape(tf.cast(sample_weight, tf.float32), [-1])
    weighted_sum = tf.reduce_sum(values * weights)
    total_weight = tf.reduce_sum(weights)
    return tf.cond(
        total_weight > 0.0,
        lambda: weighted_sum / total_weight,
        lambda: tf.zeros((), dtype=tf.float32),
    )


def _huber_per_sample(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    delta: float,
) -> tf.Tensor:
    """Compute a coordinate-wise Huber loss and keep one scalar per sample."""

    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    per_coord = 0.5 * tf.square(quadratic) / delta + linear
    return tf.reduce_mean(per_coord, axis=-1)


def _extract_center_prediction(
    model: keras.Model,
    predictions: Any,
) -> tf.Tensor:
    """Resolve the center head from either a student or teacher model output."""

    prediction_dict = _prediction_dict(model, predictions)
    for key in ("center_xy", "obb_center_xy"):
        if key in prediction_dict:
            return tf.cast(prediction_dict[key], tf.float32)
    first_tensor = next(iter(prediction_dict.values()))
    return tf.cast(first_tensor, tf.float32)


def _discover_teacher_model_path() -> Path | None:
    """Pick the most recent teacher center model when the user does not pass one."""

    training_root = PROJECT_ROOT / "artifacts" / "training"
    candidates: list[Path] = []
    for pattern in ("center*/best_model.keras", "obb_center*/best_model.keras"):
        candidates.extend(training_root.glob(pattern))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_teacher_model(teacher_model_path: Path | None) -> keras.Model | None:
    """Load an optional teacher model for center distillation."""

    if teacher_model_path is None:
        return None
    if not teacher_model_path.exists():
        print(f"[KD] Teacher model not found: {teacher_model_path}")
        return None

    print(f"[KD] Loading teacher model from {teacher_model_path}")
    # Use standalone Keras here because several teacher checkpoints were
    # serialized with Keras 3 module paths that tf_keras cannot resolve.
    try:
        teacher_model = standalone_keras.models.load_model(
            teacher_model_path,
            compile=False,
            safe_mode=False,
        )
    except Exception:
        # Fall back to tf_keras for any older artifacts that were saved with the
        # legacy serializer. The validation gate below still rejects stale 320x320
        # teachers so we only keep compatible 224x224 checkpoints in the loop.
        teacher_model = keras.models.load_model(
            teacher_model_path,
            compile=False,
            safe_mode=False,
        )
    _validate_teacher_model_input_shape(teacher_model, teacher_model_path=teacher_model_path)
    return teacher_model


def _normalize_input_shape(raw_input_shape: Any) -> tuple[int | None, ...] | None:
    """Flatten the Keras input shape metadata into a comparable tuple."""

    input_shape = raw_input_shape
    if isinstance(input_shape, list):
        if not input_shape:
            return None
        input_shape = input_shape[0]
    if hasattr(input_shape, "as_list"):
        dims = input_shape.as_list()
    elif isinstance(input_shape, tuple):
        dims = list(input_shape)
    elif isinstance(input_shape, list):
        dims = input_shape
    else:
        return None
    return tuple(None if dim is None else int(dim) for dim in dims)


def _validate_teacher_model_input_shape(teacher_model: keras.Model, *, teacher_model_path: Path) -> None:
    """Reject teacher models that do not match the shared 224x224 geometry."""

    input_shape = _normalize_input_shape(getattr(teacher_model, "input_shape", None))
    expected_shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3)
    if input_shape is None or len(input_shape) < 4:
        raise ValueError(
            f"Teacher model {teacher_model_path} exposes an unsupported input_shape={input_shape!r}; "
            f"expected a single-image 224x224 RGB model with shape {expected_shape!r}."
        )
    spatial_shape = input_shape[1:3]
    channels = input_shape[3]
    if spatial_shape != (IMAGE_SIZE, IMAGE_SIZE) or channels not in (3, None):
        raise ValueError(
            f"Teacher model {teacher_model_path} has input_shape={input_shape!r}; "
            f"expected the shared 224x224 RGB geometry {expected_shape!r}. "
            "Point KD at a current combined-pipeline artifact rather than an older 320x320 model."
        )


class CenterSimCCDistillTrainer(keras.Model):
    """Wrap a center-detector + SimCC model with explicit teacher distillation."""

    def __init__(
        self,
        student_model: keras.Model,
        *,
        teacher_model: keras.Model | None = None,
        center_loss_weight: float = DEFAULT_CENTER_LOSS_WEIGHT,
        center_distill_weight: float = DEFAULT_CENTER_DISTILL_WEIGHT,
        center_huber_delta: float = DEFAULT_CENTER_DISTILL_HUBER_DELTA,
        include_temperature_head: bool = False,
        temperature_loss_weight: float = DEFAULT_TEMPERATURE_LOSS_WEIGHT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.center_loss_weight = float(center_loss_weight)
        self.center_distill_weight = float(center_distill_weight)
        self.center_huber_delta = float(center_huber_delta)
        self.include_temperature_head = bool(include_temperature_head)
        self.temperature_loss_weight = float(temperature_loss_weight)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> Any:
        """Forward inputs through the student model."""

        return self.student_model(inputs, training=training)

    def _compute_losses(
        self,
        x: tf.Tensor,
        y: dict[str, tf.Tensor],
        sample_weight: dict[str, tf.Tensor] | None,
        predictions: Any,
    ) -> dict[str, tf.Tensor]:
        """Compute the weighted student and distillation losses for one batch."""

        pred_dict = _prediction_dict(self.student_model, predictions)
        student_center = tf.cast(pred_dict["center_xy"], tf.float32)
        center_target = tf.cast(y["center_xy"], tf.float32)
        center_weight = None if sample_weight is None else sample_weight.get("center_xy")

        center_loss = _weighted_mean(
            _huber_per_sample(center_target, student_center, delta=self.center_huber_delta),
            center_weight,
        )

        if self.teacher_model is not None:
            teacher_predictions = self.teacher_model(x, training=False)
            teacher_center = _extract_center_prediction(self.teacher_model, teacher_predictions)
            center_distill_loss = _weighted_mean(
                _huber_per_sample(teacher_center, student_center, delta=self.center_huber_delta),
                None,
            )
        else:
            center_distill_loss = tf.zeros((), dtype=tf.float32)

        center_x_loss = _weighted_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.cast(y["center_x_simcc"], tf.float32),
                tf.cast(pred_dict["center_x_simcc"], tf.float32),
            ),
            None if sample_weight is None else sample_weight.get("center_x_simcc"),
        )
        center_y_loss = _weighted_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.cast(y["center_y_simcc"], tf.float32),
                tf.cast(pred_dict["center_y_simcc"], tf.float32),
            ),
            None if sample_weight is None else sample_weight.get("center_y_simcc"),
        )
        tip_x_loss = _weighted_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.cast(y["tip_x_simcc"], tf.float32),
                tf.cast(pred_dict["tip_x_simcc"], tf.float32),
            ),
            None if sample_weight is None else sample_weight.get("tip_x_simcc"),
        )
        tip_y_loss = _weighted_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.cast(y["tip_y_simcc"], tf.float32),
                tf.cast(pred_dict["tip_y_simcc"], tf.float32),
            ),
            None if sample_weight is None else sample_weight.get("tip_y_simcc"),
        )

        total_loss = (
            self.center_loss_weight * center_loss
            + self.center_distill_weight * center_distill_loss
            + center_x_loss
            + center_y_loss
            + tip_x_loss
            + tip_y_loss
        )

        center_mae_px = tf.reduce_mean(
            tf.abs(student_center - center_target) * tf.cast(IMAGE_SIZE - 1, tf.float32)
        )
        simcc_center_x = _soft_argmax_1d_tf(tf.cast(pred_dict["center_x_simcc"], tf.float32))
        simcc_center_y = _soft_argmax_1d_tf(tf.cast(pred_dict["center_y_simcc"], tf.float32))
        true_center_x = _soft_argmax_1d_tf(tf.cast(y["center_x_simcc"], tf.float32))
        true_center_y = _soft_argmax_1d_tf(tf.cast(y["center_y_simcc"], tf.float32))
        simcc_center_mae_px = 0.5 * (
            tf.reduce_mean(tf.abs(simcc_center_x - true_center_x))
            + tf.reduce_mean(tf.abs(simcc_center_y - true_center_y))
        ) * tf.cast(IMAGE_SIZE - 1, tf.float32)
        pred_tip_x = _soft_argmax_1d_tf(tf.cast(pred_dict["tip_x_simcc"], tf.float32))
        pred_tip_y = _soft_argmax_1d_tf(tf.cast(pred_dict["tip_y_simcc"], tf.float32))
        true_tip_x = _soft_argmax_1d_tf(tf.cast(y["tip_x_simcc"], tf.float32))
        true_tip_y = _soft_argmax_1d_tf(tf.cast(y["tip_y_simcc"], tf.float32))
        tip_mae_px = 0.5 * (
            tf.reduce_mean(tf.abs(pred_tip_x - true_tip_x))
            + tf.reduce_mean(tf.abs(pred_tip_y - true_tip_y))
        ) * tf.cast(IMAGE_SIZE - 1, tf.float32)

        losses: dict[str, tf.Tensor] = {
            "loss": total_loss,
            "center_loss": center_loss,
            "center_distill_loss": center_distill_loss,
            "center_xy_mae_px": center_mae_px,
            "simcc_center_mae_px": simcc_center_mae_px,
            "tip_mae_px": tip_mae_px,
            "center_x_loss": center_x_loss,
            "center_y_loss": center_y_loss,
            "tip_x_loss": tip_x_loss,
            "tip_y_loss": tip_y_loss,
        }

        if self.include_temperature_head and "gauge_value" in pred_dict and "gauge_value" in y:
            temp_weight = None if sample_weight is None else sample_weight.get("gauge_value")
            temp_loss = _weighted_mean(
                _huber_per_sample(
                    tf.cast(y["gauge_value"], tf.float32),
                    tf.cast(pred_dict["gauge_value"], tf.float32),
                    delta=2.5,
                ),
                temp_weight,
            )
            temp_mae = _weighted_mean(
                tf.abs(
                    tf.cast(pred_dict["gauge_value"], tf.float32)
                    - tf.cast(y["gauge_value"], tf.float32)
                ),
                temp_weight,
            )
            total_loss = total_loss + self.temperature_loss_weight * temp_loss
            losses["loss"] = total_loss
            losses["temperature_loss"] = temp_loss
            losses["temperature_mae_c"] = temp_mae

        return losses

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one optimization step with teacher distillation."""

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self.student_model(x, training=True)
            losses = self._compute_losses(
                x,
                y,
                sample_weight,
                predictions,
            )
        gradients = tape.gradient(losses["loss"], self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        return {key: tf.cast(value, tf.float32) for key, value in losses.items()}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Evaluate one batch without updating weights."""

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        predictions = self.student_model(x, training=False)
        losses = self._compute_losses(
            x,
            y,
            sample_weight,
            predictions,
        )
        return {key: tf.cast(value, tf.float32) for key, value in losses.items()}


def _evaluate_sequence(
    model: keras.Model,
    sequence: CombinedGeometrySequence,
    *,
    include_temperature_head: bool,
) -> dict[str, float]:
    """Evaluate center and tip quality on a held-out sequence."""

    center_errors: list[float] = []
    simcc_center_errors: list[float] = []
    tip_errors: list[float] = []
    temperature_errors: list[float] = []

    for batch_index in range(len(sequence)):
        batch_x, batch_y, _weights = sequence[batch_index]
        predictions = _prediction_dict(model, model.predict(batch_x, verbose=0))

        pred_center = np.asarray(predictions["center_xy"], dtype=np.float32)
        pred_center_x = _soft_argmax_1d(np.asarray(predictions["center_x_simcc"], dtype=np.float32))
        pred_center_y = _soft_argmax_1d(np.asarray(predictions["center_y_simcc"], dtype=np.float32))
        pred_tip_x = _soft_argmax_1d(np.asarray(predictions["tip_x_simcc"], dtype=np.float32))
        pred_tip_y = _soft_argmax_1d(np.asarray(predictions["tip_y_simcc"], dtype=np.float32))

        true_center = np.asarray(batch_y["center_xy"], dtype=np.float32)
        true_center_x = np.asarray(batch_y["center_x_simcc"], dtype=np.float32)
        true_center_y = np.asarray(batch_y["center_y_simcc"], dtype=np.float32)
        true_tip_x = np.asarray(batch_y["tip_x_simcc"], dtype=np.float32)
        true_tip_y = np.asarray(batch_y["tip_y_simcc"], dtype=np.float32)

        true_center_x_coord = _soft_argmax_1d(true_center_x)
        true_center_y_coord = _soft_argmax_1d(true_center_y)
        true_tip_x_coord = _soft_argmax_1d(true_tip_x)
        true_tip_y_coord = _soft_argmax_1d(true_tip_y)

        center_errors.append(
            float(
                np.mean(
                    np.abs(pred_center - true_center) * float(IMAGE_SIZE - 1),
                )
            )
        )
        simcc_center_errors.append(
            float(
                0.5
                * (
                    float(np.mean(np.abs(pred_center_x - true_center_x_coord)))
                    + float(np.mean(np.abs(pred_center_y - true_center_y_coord)))
                )
                * float(IMAGE_SIZE - 1)
            )
        )
        tip_errors.append(
            float(
                0.5
                * (
                    float(np.mean(np.abs(pred_tip_x - true_tip_x_coord)))
                    + float(np.mean(np.abs(pred_tip_y - true_tip_y_coord)))
                )
                * float(IMAGE_SIZE - 1)
            )
        )

        if include_temperature_head and "gauge_value" in predictions:
            pred_temp = np.asarray(predictions["gauge_value"], dtype=np.float32).reshape(-1)
            true_temp = np.asarray(batch_y["gauge_value"], dtype=np.float32).reshape(-1)
            temperature_errors.append(float(np.mean(np.abs(pred_temp - true_temp))))

    report = {
        "center_xy_mae_px": (
            float(np.mean(center_errors)) if center_errors else float("nan")
        ),
        "simcc_center_mae_px": (
            float(np.mean(simcc_center_errors)) if simcc_center_errors else float("nan")
        ),
        "tip_mae_px": float(np.mean(tip_errors)) if tip_errors else float("nan"),
    }
    if include_temperature_head:
        report["temperature_mae_c"] = (
            float(np.mean(temperature_errors)) if temperature_errors else float("nan")
        )
    return report


def _representative_dataset(sequence: CombinedGeometrySequence):
    """Yield a small representative dataset for TFLite export."""

    for batch_index in range(min(len(sequence), 4)):
        batch_x, _batch_y, _weights = sequence[batch_index]
        for sample in batch_x[:4]:
            yield [np.asarray(sample[np.newaxis, ...], dtype=np.float32)]


def _export_tflite(
    model: keras.Model,
    output_path: Path,
    *,
    representative_sequence: CombinedGeometrySequence,
) -> Path:
    """Convert the QAT model to TFLite and write it to disk."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(representative_sequence)
    tflite_bytes = converter.convert()
    output_path.write_bytes(tflite_bytes)
    return output_path


def _tflite_parity_check(
    model: keras.Model,
    tflite_path: Path,
    sequence: CombinedGeometrySequence,
) -> dict[str, float]:
    """Run a small Keras-vs-TFLite parity check on the first validation batch."""

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    batch_x, _batch_y, _weights = sequence[0]
    sample_x = np.asarray(batch_x[:1], dtype=np.float32)
    if tuple(input_details["shape"]) != tuple(sample_x.shape):
        interpreter.resize_tensor_input(input_details["index"], list(sample_x.shape))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()

    keras_raw_predictions = model.predict(sample_x, verbose=0)
    if isinstance(keras_raw_predictions, dict):
        keras_predictions = {
            str(key).removeprefix("quant_"): value
            for key, value in keras_raw_predictions.items()
        }
    else:
        keras_predictions = {
            str(name).removeprefix("quant_"): value
            for name, value in zip(model.output_names, keras_raw_predictions)
        }

    signature_list = interpreter.get_signature_list()
    if signature_list:
        signature_name = next(iter(signature_list))
        signature_info = signature_list[signature_name]
        input_name = signature_info["inputs"][0]
        runner = interpreter.get_signature_runner(signature_name)
        raw_tflite_predictions = runner(**{input_name: sample_x.astype(input_details["dtype"])})
        tflite_predictions = {
            str(key).removeprefix("quant_"): value
            for key, value in raw_tflite_predictions.items()
        }
    else:
        interpreter.set_tensor(input_details["index"], sample_x.astype(input_details["dtype"]))
        interpreter.invoke()
        tflite_predictions = {
            str(detail["name"]).removeprefix("quant_"): interpreter.get_tensor(detail["index"])
            for detail in output_details
        }

    diffs: list[float] = []
    for name, keras_pred in keras_predictions.items():
        tflite_pred = tflite_predictions[name]
        keras_pred = np.asarray(keras_pred, dtype=np.float32)
        tflite_pred = np.asarray(tflite_pred, dtype=np.float32)
        diffs.append(float(np.max(np.abs(keras_pred - tflite_pred))))
    return {
        "tflite_max_abs_diff": float(np.max(diffs)) if diffs else float("nan"),
        "tflite_mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train a quantizable center-detector + SimCC model with optional KD and QAT."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
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
        "--teacher-model-path",
        type=Path,
        default=None,
        help="Optional teacher .keras model for distilling the center head.",
    )
    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=DEFAULT_CENTER_LOSS_WEIGHT,
        help="Supervised Huber weight for the student center head.",
    )
    parser.add_argument(
        "--center-distill-weight",
        type=float,
        default=DEFAULT_CENTER_DISTILL_WEIGHT,
        help="Teacher-student Huber weight for the center head distillation term.",
    )
    parser.add_argument(
        "--center-huber-delta",
        type=float,
        default=DEFAULT_CENTER_DISTILL_HUBER_DELTA,
        help="Huber delta for the center supervision and distillation losses.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet MobileNetV2 weights when available.",
    )
    parser.add_argument(
        "--include-temperature-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the auxiliary temperature head so temp-only rows can contribute.",
    )
    parser.add_argument(
        "--export-tflite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the final QAT model to TFLite and run a small parity check.",
    )
    return parser.parse_args()


def main() -> None:
    """Train, fine-tune, quantize, and export the center-detector localizer."""

    args = _parse_args()
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1).")
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in (0, 1).")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_combined_samples(
        args.manifest,
        include_temperature_head=bool(args.include_temperature_head),
    )
    if not samples:
        raise ValueError(f"No usable samples were found in {args.manifest}.")

    geometry_samples = [sample for sample in samples if sample.has_geometry]
    if not geometry_samples and not args.include_temperature_head:
        raise ValueError("No geometry-labeled samples were found for center + SimCC training.")

    split_labels = np.array(
        [1 if sample.tip_xy is not None else 0 for sample in samples],
        dtype=np.int32,
    )
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

    train_sequence = CombinedGeometrySequence(
        train_samples,
        batch_size=args.batch_size,
        include_temperature_head=bool(args.include_temperature_head),
        augment=True,
        seed=args.seed,
    )
    val_sequence = CombinedGeometrySequence(
        val_samples,
        batch_size=args.batch_size,
        include_temperature_head=bool(args.include_temperature_head),
        augment=False,
        seed=args.seed,
    )
    test_sequence = CombinedGeometrySequence(
        test_samples,
        batch_size=args.batch_size,
        include_temperature_head=bool(args.include_temperature_head),
        augment=False,
        seed=args.seed,
    )

    print(
        f"Loaded {len(samples)} usable samples from {args.manifest} "
        f"({len(geometry_samples)} geometry-labeled, "
        f"{sum(1 for sample in samples if sample.has_temperature and not sample.has_geometry)} temperature-only)."
    )
    print(
        f"Split sizes: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}."
    )

    teacher_model_path = args.teacher_model_path or _discover_teacher_model_path()
    teacher_model = _load_teacher_model(teacher_model_path)
    if teacher_model is None:
        print("[KD] Teacher distillation disabled.")
    else:
        print(f"[KD] Teacher ready: {teacher_model_path}")

    try:
        student_model = build_mobilenetv2_center_simcc_model(
            image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            alpha=args.alpha,
            pretrained=bool(args.pretrained),
            backbone_trainable=False,
            num_bins=NUM_BINS,
            spatial_channels=args.spatial_channels,
            head_units=args.head_units,
            head_dropout=args.head_dropout,
            include_temperature_head=bool(args.include_temperature_head),
        )
    except Exception as exc:
        if not args.pretrained:
            raise
        print(f"Pretrained weights unavailable ({exc!r}); retrying without ImageNet initialization.")
        student_model = build_mobilenetv2_center_simcc_model(
            image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            alpha=args.alpha,
            pretrained=False,
            backbone_trainable=False,
            num_bins=NUM_BINS,
            spatial_channels=args.spatial_channels,
            head_units=args.head_units,
            head_dropout=args.head_dropout,
            include_temperature_head=bool(args.include_temperature_head),
        )

    trainer = CenterSimCCDistillTrainer(
        student_model,
        teacher_model=teacher_model,
        center_loss_weight=args.center_loss_weight,
        center_distill_weight=args.center_distill_weight if teacher_model is not None else 0.0,
        center_huber_delta=args.center_huber_delta,
        include_temperature_head=bool(args.include_temperature_head),
    )

    if args.pretrained and args.warmup_epochs > 0:
        _set_backbone_trainable(student_model, trainable=False)
        trainer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        )
        trainer.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=args.warmup_epochs,
            verbose=2,
        )

    _set_backbone_trainable(student_model, trainable=True)
    if args.finetune_epochs > 0:
        trainer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate * 0.2),
        )
        trainer.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=args.finetune_epochs,
            verbose=2,
        )

    float_model_path = args.output_dir / "center_simcc_float.keras"
    student_model.save(float_model_path)

    qat_student = tfmot.quantization.keras.quantize_model(student_model)
    qat_trainer = CenterSimCCDistillTrainer(
        qat_student,
        teacher_model=teacher_model,
        center_loss_weight=args.center_loss_weight,
        center_distill_weight=args.center_distill_weight if teacher_model is not None else 0.0,
        center_huber_delta=args.center_huber_delta,
        include_temperature_head=bool(args.include_temperature_head),
    )
    qat_trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.qat_learning_rate),
    )
    qat_trainer.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.qat_epochs,
        verbose=2,
    )

    qat_model_path = args.output_dir / "center_simcc_qat.keras"
    qat_student.save(qat_model_path)

    val_report = _evaluate_sequence(
        qat_student,
        val_sequence,
        include_temperature_head=bool(args.include_temperature_head),
    )
    test_report = _evaluate_sequence(
        qat_student,
        test_sequence,
        include_temperature_head=bool(args.include_temperature_head),
    )
    report: dict[str, Any] = {
        "manifest": args.manifest.as_posix(),
        "sample_count": len(samples),
        "geometry_sample_count": len(geometry_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "teacher_model_path": teacher_model_path.as_posix() if teacher_model_path is not None else None,
        "float_model_path": float_model_path.as_posix(),
        "qat_model_path": qat_model_path.as_posix(),
        "val_metrics": val_report,
        "test_metrics": test_report,
    }

    if args.export_tflite:
        tflite_path = args.output_dir / "center_simcc_qat.tflite"
        try:
            _export_tflite(
                qat_student,
                tflite_path,
                representative_sequence=train_sequence,
            )
            parity_report = _tflite_parity_check(qat_student, tflite_path, val_sequence)
            report["tflite_path"] = tflite_path.as_posix()
            report["tflite_parity"] = parity_report
        except Exception as exc:
            report["tflite_error"] = repr(exc)
            print(f"TFLite export/parity check failed: {exc!r}")

    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
