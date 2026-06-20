#!/usr/bin/env python3
"""Train polar needle-segmentation model with EXPLICIT geometry supervision.

This is the key improvement over previous polar attempts:
- We use the CVAT samples with full geometry labels as the backbone.
- We join the AI board-capture manifest plus the manual center manifests to
  scalar labels so we can synthesize additional polar training samples, but we
  down-weight those board/center-derived samples relative to the full-geometry
  CVAT samples.
- The annotate_30 manual notes stay in evaluation only, because they are noisy
  and some notes clamp outside the gauge sweep.
- We compute exact needle angles and generate exact needle masks in polar space.
- The board model sees the same 7-channel polar vote features that the firmware
  will build at inference time, so the learned representation matches export.
- The board model is trained with an angle-first profile loss, a deterministic
  angle-vector auxiliary loss, a small auxiliary scalar head, and teacher-driven
  hard-example mining so the value branch learns the horizontal needle position
  instead of overfitting vertical layout.
- Temperature is derived from the predicted mask angle via the known calibration.

Hard cases stay out of the training pool and are evaluated separately with the
best available center hints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os as _os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

# Configure the TensorFlow GPU memory limit before any GPU initialization happens.
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3800"))
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
    )
del _os, _GPU_MEMORY_LIMIT_MB
from sklearn.model_selection import train_test_split
from tensorflow import keras

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.dataset import EllipseLabel, PointLabel, Sample, load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    needle_angle_clockwise_rad,
    needle_value,
    GaugeSpec,
    needle_unit_xy_from_value,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import polar_rgb_to_training_features
from embedded_gauge_reading_tinyml.polar_projection import polar_project_image
from embedded_gauge_reading_tinyml.polar_model import (
    build_polar_needle_segmentation_model,
    build_polar_board_friendly_mask_model,
    build_polar_tiny_model,
    PolarAngleToTemperature,
)
from embedded_gauge_reading_tinyml.heatmap_losses import polar_profile_loss
from embedded_gauge_reading_tinyml.polar_qat import (
    PolarMaskQATTrainingModel,
    angle_vector_cosine_loss,
    angle_vector_mae_deg,
    angle_vector_from_polar_mask,
)
from embedded_gauge_reading_tinyml.polar_projection import angle_from_polar_prediction
from embedded_gauge_reading_tinyml.gauge.processing import needle_value_from_angle_deg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POLAR_SIZE = 160
VALUE_MIN = -30.0
VALUE_MAX = 50.0
MIN_ANGLE_DEG = 135.0
SWEEP_DEG = 270.0
DEFAULT_CENTER_JITTER_PX = 2.0
DEFAULT_MASK_SIGMA = 2.0
DEFAULT_RADIUS_SCALE_RANGE = (0.97, 1.03)
DEFAULT_AUGMENT_REPEATS = 3
DEFAULT_STEM_FILTERS = 48
DEFAULT_BASE_FILTERS = 48
DEFAULT_PROFILE_HEAD_UNITS = 64
DEFAULT_PROFILE_HEAD_DROPOUT = 0.0
DEFAULT_PROFILE_AUX_WEIGHT = 0.25
DEFAULT_ANGLE_AUX_WEIGHT = 4.0
DEFAULT_ANGLE_VECTOR_TEMPERATURE = 5.0
DEFAULT_HARD_EXAMPLE_BOOST = 1.25
DEFAULT_HARD_EXAMPLE_SCALE = 12.0
DEFAULT_CVAT_SOURCE_WEIGHT = 0.85
DEFAULT_AI_CENTER_SOURCE_WEIGHT = 0.80
DEFAULT_MANUAL_CENTER_SOURCE_WEIGHT = 1.00
DEFAULT_EDGE_VALUE_BOOST = 1.0
DEFAULT_FINETUNE_LR_MULTIPLIER = 0.05
DEFAULT_GEOMETRY_PRETRAIN_EPOCHS = 40
DEFAULT_DISTILL_BLEND_WEIGHT = 0.15
VALUE_BIN_SIZE = 10.0
VALUE_STATE_BIN_SIZE = 5.0
DEFAULT_STATE_HEAD_UNITS = 64
DEFAULT_STATE_HEAD_DROPOUT = 0.10
DEFAULT_STATE_AUX_WEIGHT = 0.25
DEFAULT_STATE_LABEL_SIGMA = 4.0
DEFAULT_SOURCE_VAL_FRACTION = 0.15
BOARD_FRIENDLY_INPUT_MODE: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb_edge6_vote7"
RGB_INPUT_MODE: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb"
BOARD_ACTIVATION_BUDGET_MB = 2.0

AI_CENTERS_CSV: Path = PROJECT_ROOT / "data" / "ai_annotated_board_captures.csv"
MANUAL_CENTERS_CSV: Path = PROJECT_ROOT / "data" / "manual_annotated_centers.csv"
ANNOTATE_30_MANUAL_CSV: Path = PROJECT_ROOT / "data" / "annotate_30" / "manual_annotations.csv"
FULL_SCALAR_MANIFEST_CSV: Path = PROJECT_ROOT / "data" / "full_scalar_manifest_v1.csv"
MERGED_GEOMETRY_MANIFEST_CSV: Path = PROJECT_ROOT / "data" / "merged_geometry_board_manifest.csv"
BOARD_LABELED_V2_CSV: Path = PROJECT_ROOT / "data" / "board_captures_labeled_v2.csv"
CANONICAL_MANIFEST_CSV: Path = PROJECT_ROOT / "data" / "canonical_manifest_v1.csv"


@dataclass(frozen=True)
class GeometryReference:
    """Approximate geometry needed to synthesize a full polar training sample."""

    dial_radius: float
    needle_radius: float


def _estimate_board_peak_activation_bytes(
    *,
    polar_size: int,
    base_filters: int,
) -> int:
    """Estimate the largest int8 activation tensor for the board-friendly decoder.

    The final full-resolution feature map is the conservative peak because the
    bridge widening happens at 20x20 and does not change that high-resolution tensor.
    """

    return int(polar_size) * int(polar_size) * int(base_filters)


def _estimate_feature_map_bytes(
    *,
    height: int,
    width: int,
    channels: int,
) -> int:
    """Estimate the byte size of a single feature map tensor."""

    return int(height) * int(width) * int(channels)


def _make_value_bin_edges(values: np.ndarray, bin_size: float = VALUE_BIN_SIZE) -> np.ndarray:
    """Build evenly spaced value-bin edges that cover the full label range."""

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    lower = math.floor(min_val / bin_size) * bin_size
    upper = math.ceil(max_val / bin_size) * bin_size + bin_size
    return np.arange(lower, upper + bin_size * 0.5, bin_size, dtype=np.float32)


def _compute_sample_weights(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Weight rare value bins more heavily so the model does not collapse to the mean."""

    bins = pd.cut(pd.Series(values), bins=bin_edges, include_lowest=True)
    counts = bins.value_counts()
    total = float(len(values))
    num_bins = float(len(counts)) if len(counts) > 0 else 1.0

    weight_map: dict[object, float] = {}
    for bin_label, count in counts.items():
        if count <= 0:
            weight_map[bin_label] = 0.0
        else:
            weight_map[bin_label] = float(min(total / (num_bins * float(count)), 5.0))

    weights = bins.map(weight_map).astype(np.float32).to_numpy()
    return weights


def _compute_edge_boost_weights(
    values: np.ndarray,
    *,
    min_value: float = VALUE_MIN,
    max_value: float = VALUE_MAX,
    boost: float = DEFAULT_EDGE_VALUE_BOOST,
) -> np.ndarray:
    """Boost examples near the sweep endpoints where the hard cases cluster."""

    if boost <= 0.0:
        return np.ones_like(values, dtype=np.float32)

    span = max(float(max_value) - float(min_value), 1e-6)
    normalized = np.clip((np.asarray(values, dtype=np.float32) - float(min_value)) / span, 0.0, 1.0)
    edge_distance = np.abs(normalized - 0.5) * 2.0
    return (1.0 + float(boost) * edge_distance).astype(np.float32)


def _make_state_bin_centers(
    min_value: float,
    max_value: float,
    bin_size: float,
) -> np.ndarray:
    """Build evenly spaced temperature-state bin centers for the ordinal head."""

    lower_edge = math.floor(float(min_value) / float(bin_size)) * float(bin_size)
    upper_edge = math.ceil(float(max_value) / float(bin_size)) * float(bin_size)
    centers = np.arange(
        lower_edge + float(bin_size) / 2.0,
        upper_edge + float(bin_size) / 2.0,
        float(bin_size),
        dtype=np.float32,
    )
    return centers


def _soft_value_state_distribution(
    values: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma: float = DEFAULT_STATE_LABEL_SIGMA,
) -> np.ndarray:
    """Convert scalar temperatures into a smooth ordinal target distribution."""

    clipped_values = np.clip(
        np.asarray(values, dtype=np.float32),
        float(bin_centers[0]),
        float(bin_centers[-1]),
    )
    deltas = clipped_values[:, np.newaxis] - np.asarray(bin_centers, dtype=np.float32)[np.newaxis, :]
    logits = -0.5 * np.square(deltas / max(float(sigma), 1e-6))
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-6)
    return probs.astype(np.float32)


def _angle_vector_from_degrees(angle_deg: float) -> np.ndarray:
    """Convert one angle in degrees into a unit [cos, sin] vector."""

    angle_rad = math.radians(float(angle_deg))
    return np.asarray([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)


def _extract_teacher_value_predictions(teacher_outputs: Any, spec: GaugeSpec) -> np.ndarray:
    """Normalize a teacher model output structure into scalar gauge values."""

    if isinstance(teacher_outputs, dict):
        if "gauge_value" in teacher_outputs:
            predicted_values = teacher_outputs["gauge_value"]
        elif "profile_value_aux" in teacher_outputs:
            predicted_values = teacher_outputs["profile_value_aux"]
        elif "needle_mask" in teacher_outputs:
            predicted_values = teacher_outputs["needle_mask"]
        else:
            predicted_values = next(iter(teacher_outputs.values()))
    elif isinstance(teacher_outputs, (list, tuple)):
        if len(teacher_outputs) == 0:
            raise ValueError("Teacher model returned no outputs")
        first_output = np.asarray(teacher_outputs[0], dtype=np.float32)
        if first_output.ndim >= 3:
            predicted_values = decode_mask_predictions_to_values(first_output, spec)
        else:
            predicted_values = first_output
    else:
        predicted_values = teacher_outputs

    predicted_values_array = np.asarray(predicted_values, dtype=np.float32)
    if predicted_values_array.ndim >= 3:
        return decode_mask_predictions_to_values(predicted_values_array, spec)
    return predicted_values_array.reshape(-1)


def _compute_teacher_hard_example_weights(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    teacher_model: keras.Model | None,
    batch_size: int,
    polar_size: int,
    mask_sigma: float,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = RGB_INPUT_MODE,
    hard_example_boost: float = DEFAULT_HARD_EXAMPLE_BOOST,
    hard_example_scale: float = DEFAULT_HARD_EXAMPLE_SCALE,
) -> np.ndarray | None:
    """Boost samples the teacher model still finds difficult."""

    if teacher_model is None or hard_example_boost <= 0.0:
        return None

    polar_images: list[np.ndarray] = []
    temperatures: list[float] = []
    kept_indices: list[int] = []

    for index, sample in enumerate(samples):
        try:
            polar, _, temp = generate_training_pair(
                sample,
                spec,
                polar_size=polar_size,
                center_jitter_px=0.0,
                radius_scale_range=(1.0, 1.0),
                mask_sigma=mask_sigma,
            )
            if input_mode != "rgb":
                polar = polar_rgb_to_training_features(polar, input_mode=input_mode)
            polar_images.append(np.asarray(polar, dtype=np.float32))
            temperatures.append(float(temp))
            kept_indices.append(index)
        except Exception as exc:
            logger.warning("Skip hard-example mining sample %s: %s", sample.image_path, exc)

    if not polar_images:
        return None

    teacher_outputs = teacher_model.predict(
        np.asarray(polar_images, dtype=np.float32),
        batch_size=batch_size,
        verbose=0,
    )
    teacher_values = _extract_teacher_value_predictions(teacher_outputs, spec)
    residuals = np.abs(teacher_values - np.asarray(temperatures, dtype=np.float32))
    normalized = np.clip(residuals / max(float(hard_example_scale), 1e-6), 0.0, 1.0)

    weights = np.ones(len(samples), dtype=np.float32)
    for local_index, sample_index in enumerate(kept_indices):
        weights[sample_index] = float(1.0 + float(hard_example_boost) * float(normalized[local_index]))

    return weights


def _wrapped_angle_error_deg(true_angles: np.ndarray, predicted_angles: np.ndarray) -> np.ndarray:
    """Return the smallest circular absolute error between two angle arrays."""

    delta = (np.asarray(predicted_angles, dtype=np.float32) - np.asarray(true_angles, dtype=np.float32) + 180.0) % 360.0
    return np.abs(delta - 180.0)


def _manifest_key(image_ref: str) -> str:
    """Return a stable key we can use to join data sources by filename."""

    return Path(image_ref).name


def _row_text(row: Mapping[str, str], fields: Sequence[str]) -> str | None:
    """Return the first non-empty string value for a row field list."""

    for field in fields:
        raw_value = row.get(field)
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if text:
            return text
    return None


def _row_float(row: Mapping[str, str], fields: Sequence[str]) -> float | None:
    """Return the first parseable float for a row field list."""

    text = _row_text(row, fields)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_image_path(image_ref: str) -> Path:
    """Resolve a CSV image reference against the repo's data directories."""

    raw_path = Path(image_ref)
    candidates: list[Path] = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend(
            [
                REPO_ROOT / image_ref,
                PROJECT_ROOT / image_ref,
                DATA_DIR / image_ref,
                DATA_DIR / raw_path.name,
                DATA_DIR / "captured_images" / raw_path.name,
                DATA_DIR / "annotate_30" / "images" / raw_path.name,
                DATA_DIR / "annotate_30" / raw_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Unable to resolve image path: {image_ref}")


def _build_value_lookup(manifest_paths: Sequence[Path]) -> dict[str, float]:
    """Build a filename -> gauge value lookup from one or more value manifests."""

    value_lookup: dict[str, float] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_ref = _row_text(row, ("image_path", "path", "resolved_path"))
                if image_ref is None:
                    continue
                key = _manifest_key(image_ref)
                if key in value_lookup:
                    continue
                value = _row_float(
                    row,
                    ("value", "temperature_c", "deterministic_temperature_c", "label"),
                )
                if value is None:
                    continue
                value_lookup[key] = value
    return value_lookup


def _build_center_lookup(manifest_paths: Sequence[Path]) -> dict[str, tuple[float, float]]:
    """Build a filename -> center lookup from one or more center annotation files."""

    center_lookup: dict[str, tuple[float, float]] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_ref = _row_text(row, ("image_path", "filename", "path"))
                if image_ref is None:
                    continue
                key = _manifest_key(image_ref)
                if key in center_lookup:
                    continue
                center_x = _row_float(row, ("center_x", "cx"))
                center_y = _row_float(row, ("center_y", "cy"))
                if center_x is None or center_y is None:
                    continue
                center_lookup[key] = (center_x, center_y)
    return center_lookup


def _build_manifest_key_set(manifest_path: Path) -> set[str]:
    """Collect the filenames present in a value manifest."""

    keys: set[str] = set()
    if not manifest_path.exists():
        return keys

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_ref = _row_text(row, ("image_path", "path", "resolved_path"))
            if image_ref is None:
                continue
            keys.add(_manifest_key(image_ref))
    return keys


def _build_geometry_lookup(manifest_paths: Sequence[Path]) -> dict[str, GeometryReference]:
    """Build a filename -> approximate geometry lookup from geometry manifests."""

    geometry_lookup: dict[str, GeometryReference] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_ref = _row_text(row, ("image_path", "path", "resolved_path"))
                if image_ref is None:
                    continue
                key = _manifest_key(image_ref)
                if key in geometry_lookup:
                    continue

                center_x = _row_float(row, ("center_x_source", "center_x", "cx"))
                center_y = _row_float(row, ("center_y_source", "center_y", "cy"))
                tip_x = _row_float(row, ("tip_x_source", "tip_x"))
                tip_y = _row_float(row, ("tip_y_source", "tip_y"))
                if center_x is None or center_y is None or tip_x is None or tip_y is None:
                    continue

                needle_radius = _row_float(row, ("center_tip_distance_pixels",))
                if needle_radius is None:
                    needle_radius = math.hypot(tip_x - center_x, tip_y - center_y)

                dial_radius = _row_float(
                    row,
                    ("dial_radius_source", "dial_radius", "outer_radius"),
                )
                if dial_radius is None or dial_radius <= 0.0:
                    dial_radius = max(needle_radius * 1.15, needle_radius)

                geometry_lookup[key] = GeometryReference(
                    dial_radius=dial_radius,
                    needle_radius=needle_radius,
                )
    return geometry_lookup


def _manual_note_value(notes: str | None) -> float | None:
    """Extract a temperature-like value from the manual notes field."""

    if notes is None:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*C\b", notes)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    """Read image width and height without decoding more than necessary."""

    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _default_dial_radius(image_path: Path, spec: GaugeSpec) -> float:
    """Estimate the dial radius when a geometry manifest is unavailable."""

    width, height = _image_dimensions(image_path)
    frame_dim = float(min(width, height))
    return frame_dim * float(spec.inner_dial_radius_frame_ratio)


def _build_synthetic_sample(
    image_path: Path,
    center_xy: tuple[float, float],
    value: float,
    spec: GaugeSpec,
    *,
    geometry: GeometryReference | None = None,
) -> Sample:
    """Build a full Sample from a center annotation plus a scalar value label."""

    center_x, center_y = center_xy
    dial_radius = geometry.dial_radius if geometry is not None else _default_dial_radius(image_path, spec)
    needle_radius = (
        geometry.needle_radius if geometry is not None else max(dial_radius * 0.9, 1.0)
    )
    unit_dx, unit_dy = needle_unit_xy_from_value(value, spec)

    return Sample(
        image_path=image_path,
        dial=EllipseLabel(
            cx=center_x,
            cy=center_y,
            rx=dial_radius,
            ry=dial_radius,
            rotation=0.0,
            label="temp_dial",
        ),
        center=PointLabel(x=center_x, y=center_y, label="temp_center"),
        tip=PointLabel(
            x=center_x + unit_dx * needle_radius,
            y=center_y + unit_dy * needle_radius,
            label="temp_tip",
        ),
    )


def _load_center_source_samples(
    csv_path: Path,
    *,
    source_name: str,
    spec: GaugeSpec,
    image_field: str,
    center_x_field: str,
    center_y_field: str,
    value_lookup: Mapping[str, float] | None = None,
    geometry_lookup: Mapping[str, GeometryReference] | None = None,
    notes_field: str | None = None,
    exclude_keys: set[str] | None = None,
) -> list[Sample]:
    """Load a center-annotated CSV and synthesize full geometry samples from it."""

    samples: list[Sample] = []
    skipped = 0
    out_of_range = 0

    if not csv_path.exists():
        logger.warning("%s missing: %s", source_name, csv_path)
        return samples

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_ref = _row_text(row, (image_field,))
            if image_ref is None:
                skipped += 1
                continue

            key = _manifest_key(image_ref)
            if exclude_keys is not None and key in exclude_keys:
                continue
            value = value_lookup.get(key) if value_lookup is not None else None
            if value is None and notes_field is not None:
                value = _manual_note_value(_row_text(row, (notes_field,)))
            if value is None:
                skipped += 1
                continue

            if value < spec.min_value or value > spec.max_value:
                out_of_range += 1

            center_x = _row_float(row, (center_x_field,))
            center_y = _row_float(row, (center_y_field,))
            if center_x is None or center_y is None:
                skipped += 1
                continue

            try:
                image_path = _resolve_image_path(image_ref)
            except FileNotFoundError as exc:
                logger.warning("%s: skip %s (%s)", source_name, image_ref, exc)
                skipped += 1
                continue

            geometry = geometry_lookup.get(key) if geometry_lookup is not None else None
            samples.append(
                _build_synthetic_sample(
                    image_path=image_path,
                    center_xy=(center_x, center_y),
                    value=value,
                    spec=spec,
                    geometry=geometry,
                )
            )

    if out_of_range > 0:
        logger.warning(
            "%s: %d samples fell outside the calibrated range and were clamped to the sweep",
            source_name,
            out_of_range,
        )
    logger.info("%s: loaded %d samples (%d skipped)", source_name, len(samples), skipped)
    return samples


def _load_value_manifest_samples(
    manifest_path: Path,
    *,
    source_name: str,
    spec: GaugeSpec,
    center_lookup: Mapping[str, tuple[float, float]] | None = None,
    geometry_lookup: Mapping[str, GeometryReference] | None = None,
    exclude_keys: set[str] | None = None,
) -> list[Sample]:
    """Load a value-labelled manifest and attach center hints when available."""

    samples: list[Sample] = []
    skipped = 0

    if not manifest_path.exists():
        logger.warning("%s missing: %s", source_name, manifest_path)
        return samples

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_ref = _row_text(row, ("image_path", "path", "resolved_path"))
            if image_ref is None:
                skipped += 1
                continue

            key = _manifest_key(image_ref)
            if exclude_keys is not None and key in exclude_keys:
                continue
            value = _row_float(
                row,
                ("value", "temperature_c", "deterministic_temperature_c", "label"),
            )
            if value is None:
                skipped += 1
                continue

            try:
                image_path = _resolve_image_path(image_ref)
            except FileNotFoundError as exc:
                logger.warning("%s: skip %s (%s)", source_name, image_ref, exc)
                skipped += 1
                continue

            if center_lookup is not None and key in center_lookup:
                center_xy = center_lookup[key]
            else:
                width, height = _image_dimensions(image_path)
                center_xy = (float(width) / 2.0, float(height) / 2.0)

            geometry = geometry_lookup.get(key) if geometry_lookup is not None else None
            samples.append(
                _build_synthetic_sample(
                    image_path=image_path,
                    center_xy=center_xy,
                    value=value,
                    spec=spec,
                    geometry=geometry,
                )
            )

    logger.info("%s: loaded %d samples (%d skipped)", source_name, len(samples), skipped)
    return samples


def preprocess_image_for_polar(
    image_path: str,
    center_xy: tuple[float, float],
    dial_radius: float,
    target_size: int = POLAR_SIZE,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Load image, crop around dial, resize, return resized image + scaled center + max_radius.

    The crop is a square 2.5x dial_radius centered on the dial center.
    After resize to target_size, center is at (target_size/2, target_size/2).
    """
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    cx, cy = center_xy
    crop_half = int(1.25 * dial_radius)  # 2.5x total diameter -> 1.25x radius each side

    # Ensure crop is within bounds
    x1 = max(0, int(cx) - crop_half)
    y1 = max(0, int(cy) - crop_half)
    x2 = min(w, int(cx) + crop_half)
    y2 = min(h, int(cy) + crop_half)

    # Make square crop
    crop_size = min(x2 - x1, y2 - y1)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for {image_path}")

    # Resize to target_size x target_size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # New center in resized image
    new_cx = target_size / 2.0
    new_cy = target_size / 2.0

    # Scale max_radius: original dial_radius mapped to resized space
    scale = target_size / crop_size if crop_size > 0 else 1.0
    new_radius = dial_radius * scale * 1.2  # slight margin

    return resized, (new_cx, new_cy), new_radius


def angle_deg_from_sample(sample: Sample, spec: GaugeSpec) -> float:
    """Compute needle angle in degrees [0, 360) from CVAT labels."""
    raw_angle = needle_angle_clockwise_rad(sample)
    # Convert to degrees, normalize to [0, 360)
    angle_deg = math.degrees(raw_angle) % 360.0
    return angle_deg


def create_needle_mask_polar(
    polar_image: np.ndarray,
    needle_angle_deg: float,
    mask_sigma: float = DEFAULT_MASK_SIGMA,
) -> np.ndarray:
    """Create soft Gaussian needle mask in polar space."""
    height, width = polar_image.shape[:2]
    center_x = (needle_angle_deg / 360.0) * float(width)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist_sq = (xx - center_x) ** 2
    mask = np.exp(-dist_sq / (2.0 * mask_sigma**2))
    return mask[..., np.newaxis].astype(np.float32)


def generate_training_pair(
    sample: Sample,
    spec: GaugeSpec,
    polar_size: int = POLAR_SIZE,
    center_jitter_px: float = 0.0,
    radius_scale_range: tuple[float, float] = DEFAULT_RADIUS_SCALE_RANGE,
    mask_sigma: float = DEFAULT_MASK_SIGMA,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate one training pair: (polar_image, needle_mask, temperature).

    Args:
        sample: CVAT sample with geometry labels.
        spec: Gauge calibration spec.
        center_jitter_px: Random pixel offset to apply to center for robustness.
        radius_scale_range: Random scale on max_radius.

    Returns:
        polar_image: (POLAR_SIZE, POLAR_SIZE, 3) float32
        needle_mask: (POLAR_SIZE, POLAR_SIZE, 1) float32
        temperature: scalar float
    """
    cx = sample.center.x + np.random.uniform(-center_jitter_px, center_jitter_px)
    cy = sample.center.y + np.random.uniform(-center_jitter_px, center_jitter_px)
    dial_radius = max(sample.dial.rx, sample.dial.ry)

    img, (new_cx, new_cy), new_radius = preprocess_image_for_polar(
        str(sample.image_path),
        (cx, cy),
        dial_radius,
        target_size=polar_size,
    )

    # Random radius scaling
    radius_scale = np.random.uniform(*radius_scale_range)
    new_radius *= radius_scale

    # Polar project
    polar = polar_project_image(
        img,
        center_xy=(new_cx, new_cy),
        max_radius=new_radius,
        polar_size=polar_size,
    )

    # Needle angle
    angle_deg = angle_deg_from_sample(sample, spec)
    # Adjust angle if we did horizontal flip (not doing here, but keep for symmetry)
    # For now, just generate mask
    mask = create_needle_mask_polar(polar, angle_deg, mask_sigma=mask_sigma)

    # Temperature label
    temp = needle_value(sample, spec, strict=False)

    return polar, mask, temp


def augment_training_pair(
    polar: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    polar_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply augmentation in polar space while keeping the angle label valid.

    Horizontal shift still acts like a dial rotation, but we also add stronger
    photometric noise so the model sees blur, darkness, glare, and contrast
    shifts that resemble the board hard cases.
    """
    # Horizontal shift amount (in pixels)
    max_shift_px = max(4, int(round(float(polar_size) / 16.0)))
    shift_px = np.random.randint(-max_shift_px, max_shift_px + 1)

    if shift_px != 0:
        polar = np.roll(polar, shift_px, axis=1)
        mask = np.roll(mask, shift_px, axis=1)
        # Update angle: shift right (positive) means angle increases
        angle_deg = (angle_deg + (shift_px / float(polar_size)) * 360.0) % 360.0

    # Photometric augmentations keep the geometry intact but stress the decoder.
    brightness = np.random.uniform(-0.15, 0.15)
    contrast = np.random.uniform(0.7, 1.3)
    gamma = np.random.uniform(0.8, 1.35)
    channel_scale = np.random.uniform(0.9, 1.1, size=(1, 1, polar.shape[-1])).astype(np.float32)
    polar = np.clip((polar - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    polar = np.power(np.clip(polar, 0.0, 1.0), gamma)
    polar = np.clip(polar * channel_scale, 0.0, 1.0)

    # Random blur mimics motion blur and soft focus from phone captures.
    if np.random.rand() < 0.35:
        blur_sigma = np.random.uniform(0.5, 1.8)
        ksize = max(3, int(blur_sigma * 2) * 2 + 1)
        polar = cv2.GaussianBlur(polar, (ksize, ksize), blur_sigma)

    # A subtle shadow band helps the model survive dark regions and glare.
    if np.random.rand() < 0.25:
        shadow_strength = np.random.uniform(0.15, 0.45)
        shadow_axis = int(np.random.randint(0, 2))
        if shadow_axis == 0:
            shadow_profile = np.linspace(1.0 - shadow_strength, 1.0, polar.shape[0], dtype=np.float32)
            if np.random.rand() < 0.5:
                shadow_profile = shadow_profile[::-1]
            polar = polar * shadow_profile[:, np.newaxis, np.newaxis]
        else:
            shadow_profile = np.linspace(1.0 - shadow_strength, 1.0, polar.shape[1], dtype=np.float32)
            if np.random.rand() < 0.5:
                shadow_profile = shadow_profile[::-1]
            polar = polar * shadow_profile[np.newaxis, :, np.newaxis]
        polar = np.clip(polar, 0.0, 1.0)

    # Random noise keeps the model from depending on a single clean texture.
    if np.random.rand() < 0.35:
        noise_std = np.random.uniform(0.01, 0.04)
        polar = np.clip(polar + np.random.normal(0.0, noise_std, polar.shape), 0.0, 1.0)

    # Small cutouts approximate partial occlusions from reflections or hands.
    if np.random.rand() < 0.15:
        cutout_h = max(4, int(round(polar_size * np.random.uniform(0.06, 0.16))))
        cutout_w = max(4, int(round(polar_size * np.random.uniform(0.06, 0.16))))
        top = int(np.random.randint(0, max(1, polar_size - cutout_h + 1)))
        left = int(np.random.randint(0, max(1, polar_size - cutout_w + 1)))
        polar[top : top + cutout_h, left : left + cutout_w, :] = np.random.uniform(0.0, 0.12)

    return polar, mask, angle_deg


def _generate_polar_examples(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    polar_size: int = POLAR_SIZE,
    center_jitter_px: float = DEFAULT_CENTER_JITTER_PX,
    radius_scale_range: tuple[float, float] = DEFAULT_RADIUS_SCALE_RANGE,
    mask_sigma: float = DEFAULT_MASK_SIGMA,
    augment: bool = True,
    augment_repeats: int = DEFAULT_AUGMENT_REPEATS,
    sample_weights: np.ndarray | None = None,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = RGB_INPUT_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, list[Sample]]:
    """Materialize the polar images, masks, and scalar targets for a sample list."""

    if sample_weights is not None and len(sample_weights) != len(samples):
        raise ValueError("sample_weights must match the number of samples")

    polar_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    temps: list[float] = []
    angle_vectors: list[np.ndarray] = []
    weights: list[float] = []
    kept_samples: list[Sample] = []

    logger.info("Generating polar projections for %d samples...", len(samples))
    for index, sample in enumerate(samples):
        base_weight = float(sample_weights[index]) if sample_weights is not None else 1.0
        repeat_count = augment_repeats if augment else 1
        for _ in range(repeat_count):
            try:
                polar, mask, temp = generate_training_pair(
                    sample,
                    spec,
                    polar_size=polar_size,
                    center_jitter_px=center_jitter_px if augment else 0.0,
                    radius_scale_range=radius_scale_range,
                    mask_sigma=mask_sigma,
                )
                angle_deg = angle_deg_from_sample(sample, spec)
                if augment:
                    polar, mask, angle_deg = augment_training_pair(
                        polar,
                        mask,
                        angle_deg,
                        polar_size=polar_size,
                    )
                if input_mode != "rgb":
                    polar = polar_rgb_to_training_features(polar, input_mode=input_mode)
                polar_images.append(np.asarray(polar, dtype=np.float32))
                masks.append(mask)
                temps.append(temp)
                angle_vectors.append(_angle_vector_from_degrees(angle_deg))
                weights.append(base_weight)
                kept_samples.append(sample)
            except Exception as exc:
                logger.warning("Skip %s: %s", sample.image_path, exc)

    polar_images_arr = np.asarray(polar_images, dtype=np.float32)
    masks_arr = np.asarray(masks, dtype=np.float32)
    temps_arr = np.asarray(temps, dtype=np.float32)
    angle_vectors_arr = np.asarray(angle_vectors, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32) if weights else None

    logger.info("Successfully generated %d training pairs", len(polar_images_arr))
    return polar_images_arr, masks_arr, temps_arr, angle_vectors_arr, weights_arr, kept_samples


def _load_teacher_model(teacher_model_path: Path | None) -> keras.Model | None:
    """Load an optional teacher model for soft-label distillation."""

    if teacher_model_path is None:
        return None
    if not teacher_model_path.exists():
        logger.warning("Distillation teacher model not found: %s", teacher_model_path)
        return None

    logger.info("Loading distillation teacher model from %s", teacher_model_path)
    return keras.models.load_model(
        teacher_model_path,
        custom_objects={
            "PolarAngleToTemperature": PolarAngleToTemperature,
        },
    )


def _distill_polar_supervision(
    polar_images: np.ndarray,
    masks: np.ndarray,
    angle_vectors: np.ndarray,
    *,
    teacher_model: keras.Model | None,
    batch_size: int,
    distill_blend_weight: float,
    angle_vector_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend hard targets with teacher predictions for a light distillation pass."""

    if teacher_model is None or distill_blend_weight <= 0.0:
        return masks, angle_vectors

    teacher_outputs = teacher_model.predict(polar_images, batch_size=batch_size, verbose=0)
    if isinstance(teacher_outputs, dict):
        teacher_masks = teacher_outputs.get("needle_mask")
        if teacher_masks is None:
            raise KeyError("Teacher model did not return a needle_mask output")
    elif isinstance(teacher_outputs, (list, tuple)):
        teacher_masks = teacher_outputs[0]
    else:
        teacher_masks = teacher_outputs

    teacher_masks_arr = np.asarray(teacher_masks, dtype=np.float32)
    if teacher_masks_arr.ndim == 3:
        teacher_masks_arr = teacher_masks_arr[..., np.newaxis]

    teacher_angle_vectors_arr = np.asarray(
        angle_vector_from_polar_mask(
            tf.convert_to_tensor(teacher_masks_arr, dtype=tf.float32),
            temperature=angle_vector_temperature,
        ),
        dtype=np.float32,
    )

    blend = float(np.clip(distill_blend_weight, 0.0, 1.0))
    blended_masks = ((1.0 - blend) * masks + blend * teacher_masks_arr).astype(np.float32)
    blended_angle_vectors = ((1.0 - blend) * angle_vectors + blend * teacher_angle_vectors_arr).astype(np.float32)
    norms = np.linalg.norm(blended_angle_vectors, axis=-1, keepdims=True)
    blended_angle_vectors = blended_angle_vectors / np.maximum(norms, 1e-6)

    return blended_masks, blended_angle_vectors


def create_tf_dataset(
    samples: list[Sample],
    spec: GaugeSpec,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    polar_size: int = POLAR_SIZE,
    center_jitter_px: float = DEFAULT_CENTER_JITTER_PX,
    radius_scale_range: tuple[float, float] = DEFAULT_RADIUS_SCALE_RANGE,
    mask_sigma: float = DEFAULT_MASK_SIGMA,
    augment_repeats: int = DEFAULT_AUGMENT_REPEATS,
    sample_weights: np.ndarray | None = None,
    include_angle_vector_output: bool = False,
    include_aux_value_output: bool = False,
    include_state_output: bool = False,
    value_state_bin_centers: np.ndarray | None = None,
    value_state_sigma: float = DEFAULT_STATE_LABEL_SIGMA,
    teacher_model: keras.Model | None = None,
    distill_blend_weight: float = 0.0,
    angle_vector_temperature: float = DEFAULT_ANGLE_VECTOR_TEMPERATURE,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = RGB_INPUT_MODE,
) -> tf.data.Dataset:
    """Create TensorFlow dataset from CVAT samples."""
    polar_images, masks, temps, angle_vectors, weights_arr, _ = _generate_polar_examples(
        samples,
        spec,
        polar_size=polar_size,
        center_jitter_px=center_jitter_px,
        radius_scale_range=radius_scale_range,
        mask_sigma=mask_sigma,
        augment=augment,
        augment_repeats=augment_repeats,
        sample_weights=sample_weights,
        input_mode=input_mode,
    )

    masks, angle_vectors = _distill_polar_supervision(
        polar_images,
        masks,
        angle_vectors,
        teacher_model=teacher_model,
        batch_size=batch_size,
        distill_blend_weight=distill_blend_weight,
        angle_vector_temperature=angle_vector_temperature,
    )

    state_targets: np.ndarray | None = None
    if include_state_output:
        if value_state_bin_centers is None:
            raise ValueError("value_state_bin_centers is required when include_state_output=True")
        state_targets = _soft_value_state_distribution(
            temps,
            value_state_bin_centers,
            sigma=value_state_sigma,
        )

    if weights_arr is not None:
        targets: dict[str, np.ndarray] = {
            "needle_mask": masks,
            "gauge_value": temps,
        }
        weight_targets: dict[str, np.ndarray] = {
            "needle_mask": weights_arr,
            "gauge_value": weights_arr,
        }
        if include_angle_vector_output:
            targets["needle_angle_vector"] = angle_vectors
            weight_targets["needle_angle_vector"] = weights_arr
        if include_aux_value_output:
            targets["profile_value_aux"] = temps
            weight_targets["profile_value_aux"] = weights_arr
        if include_state_output and state_targets is not None:
            targets["gauge_value_state"] = state_targets
            weight_targets["gauge_value_state"] = weights_arr
        ds = tf.data.Dataset.from_tensor_slices(
            (
                polar_images,
                targets,
                weight_targets,
            )
        )
    else:
        targets = {
            "needle_mask": masks,
            "gauge_value": temps,
        }
        if include_angle_vector_output:
            targets["needle_angle_vector"] = angle_vectors
        if include_aux_value_output:
            targets["profile_value_aux"] = temps
        if include_state_output and state_targets is not None:
            targets["gauge_value_state"] = state_targets
        ds = tf.data.Dataset.from_tensor_slices(
            (
                polar_images,
                targets,
            )
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(polar_images), 500))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_hard_cases_df(repo_root: Path) -> pd.DataFrame | None:
    """Load hard cases manifest."""
    hard_path = PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv"
    if not hard_path.exists():
        return None
    df = pd.read_csv(hard_path)
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})
    if "image_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "image_path"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def _split_samples_by_value(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[Sample], list[Sample]]:
    """Split a sample list into train/validation slices with value-aware stratification."""

    if len(samples) < 2:
        return samples, []

    indices = np.arange(len(samples))
    values = np.asarray([needle_value(sample, spec, strict=False) for sample in samples], dtype=np.float32)

    stratify: pd.Series | None = None
    if len(samples) >= 8:
        value_bins = pd.cut(
            pd.Series(values),
            bins=_make_value_bin_edges(values),
            include_lowest=True,
        )
        if (value_bins.value_counts() >= 2).all():
            stratify = value_bins

    if stratify is None:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=stratify,
        )

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples


def _split_sample_keys_by_value(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[set[str], set[str]]:
    """Split a combined sample pool by image key so duplicate sources stay aligned."""

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.image_path.name, []).append(sample)

    if len(grouped) < 2:
        return set(grouped.keys()), set()

    representative_samples = [grouped[key][0] for key in grouped]
    train_samples, val_samples = _split_samples_by_value(
        representative_samples,
        spec,
        seed=seed,
        val_fraction=val_fraction,
    )
    train_keys = {sample.image_path.name for sample in train_samples}
    val_keys = {sample.image_path.name for sample in val_samples}
    return train_keys, val_keys


def _predict_sample_values(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    batch_size: int,
    polar_size: int,
    mask_sigma: float,
    board_friendly: bool,
    model: keras.Model,
    export_model: keras.Model,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = RGB_INPUT_MODE,
) -> tuple[list[Sample], np.ndarray, np.ndarray, np.ndarray]:
    """Predict gauge values for a list of samples using the active deployment path."""

    if not samples:
        empty = np.asarray([], dtype=np.float32)
        return [], empty, empty, empty

    polar_images, _, temps, _, _, kept_samples = _generate_polar_examples(
        samples,
        spec,
        polar_size=polar_size,
        center_jitter_px=0.0,
        radius_scale_range=(1.0, 1.0),
        mask_sigma=mask_sigma,
        augment=False,
        augment_repeats=1,
        input_mode=input_mode,
    )

    if len(kept_samples) == 0:
        empty = np.asarray([], dtype=np.float32)
        return [], empty, empty, empty

    if board_friendly:
        predicted_masks = export_model.predict(
            polar_images,
            verbose=1,
            batch_size=batch_size,
        )
        predictions = decode_mask_predictions_to_values(predicted_masks, spec)
    else:
        predictions_dict = model.predict(
            polar_images,
            verbose=1,
            batch_size=batch_size,
        )
        predicted_masks = np.asarray(predictions_dict["needle_mask"], dtype=np.float32)
        predictions = np.asarray(predictions_dict["gauge_value"], dtype=np.float32).reshape(-1)

    predicted_angles = np.asarray(
        [angle_from_polar_prediction(prediction) for prediction in predicted_masks],
        dtype=np.float32,
    )

    return kept_samples, temps, predictions, predicted_angles


def _evaluate_sample_source(
    *,
    source_name: str,
    samples: list[Sample],
    spec: GaugeSpec,
    model: keras.Model,
    export_model: keras.Model,
    output_dir: Path,
    batch_size: int,
    polar_size: int,
    mask_sigma: float,
    board_friendly: bool,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = RGB_INPUT_MODE,
) -> dict[str, float]:
    """Evaluate one sample source and persist a CSV/JSON summary."""

    kept_samples, temps, predictions, predicted_angles = _predict_sample_values(
        samples,
        spec,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        model=model,
        export_model=export_model,
        input_mode=input_mode,
    )

    if len(kept_samples) == 0:
        logger.warning("No evaluable samples for %s", source_name)
        return {}

    true_angles = np.asarray([angle_deg_from_sample(sample, spec) for sample in kept_samples], dtype=np.float32)
    angle_errors = _wrapped_angle_error_deg(true_angles, predicted_angles)
    df = pd.DataFrame(
        {
            "image_path": [str(sample.image_path) for sample in kept_samples],
            "value": temps,
            "prediction": predictions,
            "true_angle_deg": true_angles,
            "predicted_angle_deg": predicted_angles,
        }
    )
    df["abs_error"] = np.abs(df["prediction"] - df["value"])
    df["angle_abs_error_deg"] = angle_errors
    df.to_csv(output_dir / f"{source_name}_predictions.csv", index=False)

    errors = df["abs_error"].to_numpy(dtype=np.float32)
    angle_error_values = df["angle_abs_error_deg"].to_numpy(dtype=np.float32)
    metrics = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
        "pct_under_5c": float(np.mean(errors < 5.0) * 100.0),
        "predicted_std": float(np.std(predictions)),
        "correlation": (
            float(np.corrcoef(df["value"], predictions)[0, 1])
            if len(predictions) > 1
            else 0.0
        ),
        "angle_mae_deg": float(np.mean(angle_error_values)),
        "angle_rmse_deg": float(np.sqrt(np.mean(angle_error_values**2))),
        "angle_median_error_deg": float(np.median(angle_error_values)),
    }
    with open(output_dir / f"{source_name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("\n=== %s Metrics ===", source_name)
    for key, value in metrics.items():
        logger.info("  %s: %.4f", key, float(value))
    return metrics


def decode_mask_predictions_to_values(
    predicted_masks: np.ndarray,
    spec: GaugeSpec,
) -> np.ndarray:
    """Convert predicted polar masks into gauge values via the geometric decode."""

    values = [
        needle_value_from_angle_deg(
            angle_from_polar_prediction(prediction),
            spec,
            strict=False,
        )
        for prediction in np.asarray(predicted_masks)
    ]
    return np.asarray(values, dtype=np.float32)


def preprocess_unlabelled_for_polar(image_path: str, polar_size: int = POLAR_SIZE) -> np.ndarray:
    """Preprocess an unlabelled image (board capture) for polar projection.

    Uses image center as approximate dial center.
    """
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    # Resize to square if needed
    if h != polar_size or w != polar_size:
        img = cv2.resize(img, (polar_size, polar_size), interpolation=cv2.INTER_LINEAR)

    # Use image center
    center = (polar_size / 2.0, polar_size / 2.0)
    max_radius = polar_size * 0.45  # Keep the dial rim safely inside the polar crop.

    polar = polar_project_image(img, center_xy=center, max_radius=max_radius, polar_size=polar_size)
    return polar


def train_polar_geometry_supervised(
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    tiny: bool = True,
    board_friendly: bool = True,
    polar_size: int = POLAR_SIZE,
    qat_output_noise_stddev: float = 0.01,
    mask_loss_weight: float = 0.5,
    temp_loss_weight: float = 1.0,
    center_jitter_px: float = DEFAULT_CENTER_JITTER_PX,
    radius_scale_range: tuple[float, float] = DEFAULT_RADIUS_SCALE_RANGE,
    mask_sigma: float = DEFAULT_MASK_SIGMA,
    augment_repeats: int = DEFAULT_AUGMENT_REPEATS,
    stem_filters: int = DEFAULT_STEM_FILTERS,
    base_filters: int = DEFAULT_BASE_FILTERS,
    bridge_filters: int | None = None,
    bridge_blocks: int = 3,
    decoder_mid_filters: int | None = None,
    decoder_mid_blocks: int = 2,
    profile_head_units: int = DEFAULT_PROFILE_HEAD_UNITS,
    profile_head_dropout: float = DEFAULT_PROFILE_HEAD_DROPOUT,
    profile_aux_loss_weight: float = DEFAULT_PROFILE_AUX_WEIGHT,
    state_head_units: int = DEFAULT_STATE_HEAD_UNITS,
    state_head_dropout: float = DEFAULT_STATE_HEAD_DROPOUT,
    state_aux_loss_weight: float = DEFAULT_STATE_AUX_WEIGHT,
    state_bin_size: float = VALUE_STATE_BIN_SIZE,
    state_label_sigma: float = DEFAULT_STATE_LABEL_SIGMA,
    angle_vector_aux_loss_weight: float = DEFAULT_ANGLE_AUX_WEIGHT,
    angle_vector_temperature: float = DEFAULT_ANGLE_VECTOR_TEMPERATURE,
    geometry_pretrain_epochs: int = DEFAULT_GEOMETRY_PRETRAIN_EPOCHS,
    finetune_learning_rate_multiplier: float = DEFAULT_FINETUNE_LR_MULTIPLIER,
    distill_teacher_model_path: Path | None = None,
    distill_blend_weight: float = DEFAULT_DISTILL_BLEND_WEIGHT,
    hard_example_boost: float = DEFAULT_HARD_EXAMPLE_BOOST,
    hard_example_scale: float = DEFAULT_HARD_EXAMPLE_SCALE,
) -> dict[str, Any]:
    """Train polar needle model with angle-first profile supervision."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gauge spec
    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    hard_case_manifest = PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv"
    hard_holdout_keys = _build_manifest_key_set(hard_case_manifest)

    value_lookup = _build_value_lookup(
        [
            AI_CENTERS_CSV,
            MERGED_GEOMETRY_MANIFEST_CSV,
            BOARD_LABELED_V2_CSV,
            FULL_SCALAR_MANIFEST_CSV,
            CANONICAL_MANIFEST_CSV,
        ]
    )
    geometry_lookup = _build_geometry_lookup(
        [
            MERGED_GEOMETRY_MANIFEST_CSV,
            BOARD_LABELED_V2_CSV,
        ]
    )

    manual30_samples = _load_center_source_samples(
        ANNOTATE_30_MANUAL_CSV,
        source_name="annotate_30_manual",
        spec=spec,
        image_field="filename",
        center_x_field="cx",
        center_y_field="cy",
        value_lookup=value_lookup,
        geometry_lookup=geometry_lookup,
        notes_field="notes",
        exclude_keys=hard_holdout_keys,
    )
    # Keep the note-derived manual-30 samples as an evaluation-only slice.
    # They are useful for regression tracking, but the notes are noisier than
    # the AI/manual center manifests, so they should not steer the value head.
    manual30_keys = {sample.image_path.name for sample in manual30_samples}
    training_holdout_keys = hard_holdout_keys | manual30_keys

    # Load the base CVAT geometry-labelled samples after the manual-30 holdout
    # is known so we can keep the noisy note-derived slice out of training.
    samples = [
        sample
        for sample in load_dataset()
        if sample.image_path.name not in training_holdout_keys
    ]
    logger.info("Loaded %d CVAT samples with geometry labels", len(samples))

    ai_center_samples = _load_center_source_samples(
        AI_CENTERS_CSV,
        source_name="ai_board_captures",
        spec=spec,
        image_field="image_path",
        center_x_field="center_x",
        center_y_field="center_y",
        value_lookup=value_lookup,
        geometry_lookup=geometry_lookup,
        exclude_keys=training_holdout_keys,
    )
    manual_center_samples = _load_center_source_samples(
        MANUAL_CENTERS_CSV,
        source_name="manual_centers",
        spec=spec,
        image_field="image_path",
        center_x_field="center_x",
        center_y_field="center_y",
        value_lookup=value_lookup,
        geometry_lookup=geometry_lookup,
        exclude_keys=training_holdout_keys,
    )

    hard_center_lookup = _build_center_lookup([MANUAL_CENTERS_CSV, AI_CENTERS_CSV])
    hard_case_samples = _load_value_manifest_samples(
        hard_case_manifest,
        source_name="hard_cases_holdout",
        spec=spec,
        center_lookup=hard_center_lookup,
        geometry_lookup=geometry_lookup,
    )

    logger.info(
        "Source counts: cvat=%d ai_board_captures=%d manual_centers=%d manual30_eval=%d hard_cases=%d",
        len(samples),
        len(ai_center_samples),
        len(manual_center_samples),
        len(manual30_samples),
        len(hard_case_samples),
    )

    feature_input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = (
        BOARD_FRIENDLY_INPUT_MODE if board_friendly else RGB_INPUT_MODE
    )
    feature_input_channels = 7 if board_friendly else 3
    if state_bin_size <= 0.0:
        raise ValueError("state_bin_size must be positive")
    if state_label_sigma <= 0.0:
        raise ValueError("state_label_sigma must be positive")
    state_bin_centers = _make_state_bin_centers(VALUE_MIN, VALUE_MAX, state_bin_size)
    include_state_output = board_friendly and state_aux_loss_weight > 0.0 and state_head_units > 0

    # Split the combined non-hard pool by image key so the same image never
    # leaks across train/validation when it appears in multiple center files.
    combined_samples = samples + ai_center_samples + manual_center_samples
    train_keys, val_keys = _split_sample_keys_by_value(
        combined_samples,
        spec,
        seed=seed,
        val_fraction=DEFAULT_SOURCE_VAL_FRACTION,
    )

    def _filter_by_keys(source_samples: list[Sample], keys: set[str]) -> list[Sample]:
        """Keep only samples whose image key is part of the requested split."""

        return [sample for sample in source_samples if sample.image_path.name in keys]

    train_cvat = _filter_by_keys(samples, train_keys)
    val_cvat = _filter_by_keys(samples, val_keys)
    train_ai = _filter_by_keys(ai_center_samples, train_keys)
    val_ai = _filter_by_keys(ai_center_samples, val_keys)
    train_manual = _filter_by_keys(manual_center_samples, train_keys)
    val_manual = _filter_by_keys(manual_center_samples, val_keys)

    train_samples = train_cvat + train_ai + train_manual
    val_samples = val_cvat + val_ai + val_manual
    train_source_weights = np.asarray(
        [
            *([DEFAULT_CVAT_SOURCE_WEIGHT] * len(train_cvat)),
            *([DEFAULT_AI_CENTER_SOURCE_WEIGHT] * len(train_ai)),
            *([DEFAULT_MANUAL_CENTER_SOURCE_WEIGHT] * len(train_manual)),
        ],
        dtype=np.float32,
    )
    train_values = np.asarray(
        [needle_value(sample, spec, strict=False) for sample in train_samples],
        dtype=np.float32,
    )
    bin_edges = _make_value_bin_edges(train_values, bin_size=VALUE_BIN_SIZE)
    train_weights = _compute_sample_weights(train_values, bin_edges) * train_source_weights
    train_weights = train_weights * _compute_edge_boost_weights(train_values)
    train_weights = np.clip(train_weights, 0.1, 8.0).astype(np.float32)

    clean_train_samples = train_cvat + train_manual
    clean_val_samples = val_cvat + val_manual
    clean_train_source_weights = np.asarray(
        [
            *([DEFAULT_CVAT_SOURCE_WEIGHT] * len(train_cvat)),
            *([DEFAULT_MANUAL_CENTER_SOURCE_WEIGHT] * len(train_manual)),
        ],
        dtype=np.float32,
    )
    clean_train_values = np.asarray(
        [needle_value(sample, spec, strict=False) for sample in clean_train_samples],
        dtype=np.float32,
    )
    clean_bin_edges = _make_value_bin_edges(clean_train_values, bin_size=VALUE_BIN_SIZE)
    clean_train_weights = _compute_sample_weights(clean_train_values, clean_bin_edges) * clean_train_source_weights
    clean_train_weights = clean_train_weights * _compute_edge_boost_weights(clean_train_values)
    clean_train_weights = np.clip(clean_train_weights, 0.1, 8.0).astype(np.float32)

    logger.info(
        "Split counts: train=%d val=%d | cvat=%d/%d ai=%d/%d manual=%d/%d manual30_eval=%d",
        len(train_samples),
        len(val_samples),
        len(train_cvat),
        len(val_cvat),
        len(train_ai),
        len(val_ai),
        len(train_manual),
        len(val_manual),
        len(manual30_samples),
    )
    logger.info(
        "Training sample weight range: "
        f"{float(np.min(train_weights)):.3f} to {float(np.max(train_weights)):.3f}"
    )
    logger.info(
        "Clean pretrain weight range: "
        f"{float(np.min(clean_train_weights)):.3f} to {float(np.max(clean_train_weights)):.3f}"
    )
    logger.info(
        "Training schedule: pretrain=%d epochs, finetune lr multiplier=%.3f, "
        "angle-vector temp=%.2f, angle-vector weight=%.2f, distill blend=%.2f, "
        "state bins=%d state loss=%.2f, hard-example boost=%.2f scale=%.2f",
        int(geometry_pretrain_epochs),
        float(finetune_learning_rate_multiplier),
        float(angle_vector_temperature),
        float(angle_vector_aux_loss_weight),
        float(distill_blend_weight),
        int(len(state_bin_centers)),
        float(state_aux_loss_weight),
        float(hard_example_boost),
        float(hard_example_scale),
    )

    teacher_model = _load_teacher_model(distill_teacher_model_path)
    if teacher_model is None:
        logger.info("Distillation teacher disabled.")
    else:
        logger.info(
            "Distillation teacher enabled at blend weight %.2f",
            float(distill_blend_weight),
        )

    train_hard_example_weights = _compute_teacher_hard_example_weights(
        train_samples,
        spec,
        teacher_model=teacher_model,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        input_mode=feature_input_mode,
        hard_example_boost=hard_example_boost,
        hard_example_scale=hard_example_scale,
    )
    clean_train_hard_example_weights = _compute_teacher_hard_example_weights(
        clean_train_samples,
        spec,
        teacher_model=teacher_model,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        input_mode=feature_input_mode,
        hard_example_boost=max(0.5, hard_example_boost * 0.75),
        hard_example_scale=hard_example_scale,
    )
    if train_hard_example_weights is not None:
        train_weights = train_weights * train_hard_example_weights
        logger.info(
            "Teacher hard-example boost range: %.3f to %.3f",
            float(np.min(train_hard_example_weights)),
            float(np.max(train_hard_example_weights)),
        )
    if clean_train_hard_example_weights is not None:
        clean_train_weights = clean_train_weights * clean_train_hard_example_weights
        logger.info(
            "Clean hard-example boost range: %.3f to %.3f",
            float(np.min(clean_train_hard_example_weights)),
            float(np.max(clean_train_hard_example_weights)),
        )
    else:
        logger.info("Teacher hard-example mining disabled.")

    train_weights = np.clip(train_weights, 0.1, 10.0).astype(np.float32)
    clean_train_weights = np.clip(clean_train_weights, 0.1, 10.0).astype(np.float32)

    # Build datasets for a clean geometry pretrain phase and a domain-adapt
    # fine-tune phase.  The clean phase focuses on the best labels first.
    include_angle_vector_output = board_friendly and angle_vector_aux_loss_weight > 0.0
    include_profile_aux_output = board_friendly and profile_head_units > 0

    clean_train_ds = create_tf_dataset(
        clean_train_samples,
        spec,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        polar_size=polar_size,
        center_jitter_px=center_jitter_px,
        radius_scale_range=radius_scale_range,
        mask_sigma=mask_sigma,
        augment_repeats=augment_repeats,
        sample_weights=clean_train_weights,
        include_angle_vector_output=include_angle_vector_output,
        include_aux_value_output=include_profile_aux_output,
        include_state_output=include_state_output,
        value_state_bin_centers=state_bin_centers,
        value_state_sigma=state_label_sigma,
        teacher_model=teacher_model,
        distill_blend_weight=distill_blend_weight,
        angle_vector_temperature=angle_vector_temperature,
        input_mode=feature_input_mode,
    )
    clean_val_ds = create_tf_dataset(
        clean_val_samples,
        spec,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        polar_size=polar_size,
        center_jitter_px=0.0,
        radius_scale_range=radius_scale_range,
        mask_sigma=mask_sigma,
        include_angle_vector_output=include_angle_vector_output,
        include_aux_value_output=include_profile_aux_output,
        include_state_output=include_state_output,
        value_state_bin_centers=state_bin_centers,
        value_state_sigma=state_label_sigma,
        input_mode=feature_input_mode,
    )
    train_ds = create_tf_dataset(
        train_samples,
        spec,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        polar_size=polar_size,
        center_jitter_px=center_jitter_px,
        radius_scale_range=radius_scale_range,
        mask_sigma=mask_sigma,
        augment_repeats=augment_repeats,
        sample_weights=train_weights,
        include_angle_vector_output=include_angle_vector_output,
        include_aux_value_output=include_profile_aux_output,
        include_state_output=include_state_output,
        value_state_bin_centers=state_bin_centers,
        value_state_sigma=state_label_sigma,
        teacher_model=teacher_model,
        distill_blend_weight=distill_blend_weight,
        angle_vector_temperature=angle_vector_temperature,
        input_mode=feature_input_mode,
    )
    val_ds = create_tf_dataset(
        val_samples,
        spec,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        polar_size=polar_size,
        center_jitter_px=0.0,
        radius_scale_range=radius_scale_range,
        mask_sigma=mask_sigma,
        include_angle_vector_output=include_angle_vector_output,
        include_aux_value_output=include_profile_aux_output,
        include_state_output=include_state_output,
        value_state_bin_centers=state_bin_centers,
        value_state_sigma=state_label_sigma,
        input_mode=feature_input_mode,
    )

    # Build the clean deployment model plus the optional QAT wrapper.
    if board_friendly:
        board_bridge_filters = (
            int(bridge_filters)
            if bridge_filters is not None
            else max(base_filters * 3, stem_filters * 2)
        )
        board_bridge_blocks = max(1, int(bridge_blocks))
        board_decoder_mid_filters = (
            int(decoder_mid_filters)
            if decoder_mid_filters is not None
            else max(base_filters * 2, 64)
        )
        board_decoder_mid_filters = max(board_decoder_mid_filters, int(base_filters))
        board_decoder_mid_blocks = max(1, int(decoder_mid_blocks))
        board_bridge_resolution = max(1, int(round(float(polar_size) / 8.0)))
        board_decoder_mid_resolution = max(1, int(round(float(polar_size) / 4.0)))
        peak_activation_bytes = _estimate_board_peak_activation_bytes(
            polar_size=polar_size,
            base_filters=base_filters,
        )
        peak_activation_mib = peak_activation_bytes / (1024.0 * 1024.0)
        bridge_activation_bytes = _estimate_feature_map_bytes(
            height=board_bridge_resolution,
            width=board_bridge_resolution,
            channels=board_bridge_filters,
        )
        bridge_activation_mib = bridge_activation_bytes / (1024.0 * 1024.0)
        decoder_mid_activation_bytes = _estimate_feature_map_bytes(
            height=board_decoder_mid_resolution,
            width=board_decoder_mid_resolution,
            channels=board_decoder_mid_filters,
        )
        decoder_mid_activation_mib = decoder_mid_activation_bytes / (1024.0 * 1024.0)
        logger.info(
            "Building board-friendly polar mask model "
            f"(polar_size={polar_size}, stem_filters={stem_filters}, base_filters={base_filters}, "
            f"bridge_filters={board_bridge_filters}, bridge_blocks={board_bridge_blocks}, "
            f"decoder_mid_filters={board_decoder_mid_filters}, decoder_mid_blocks={board_decoder_mid_blocks}, "
            f"input_channels={feature_input_channels}, input_mode={feature_input_mode})"
        )
        logger.info(
            "Estimated peak board activation: %.2f MiB (%d bytes) against a %.1f MiB SRAM budget",
            peak_activation_mib,
            peak_activation_bytes,
            BOARD_ACTIVATION_BUDGET_MB,
        )
        logger.info(
            "Estimated low-res activations: bridge %.2f MiB (%d bytes) at %dx%d, mid %.2f MiB (%d bytes) at %dx%d",
            bridge_activation_mib,
            bridge_activation_bytes,
            board_bridge_resolution,
            board_bridge_resolution,
            decoder_mid_activation_mib,
            decoder_mid_activation_bytes,
            board_decoder_mid_resolution,
            board_decoder_mid_resolution,
        )
        base_model = build_polar_board_friendly_mask_model(
            polar_size=polar_size,
            input_channels=feature_input_channels,
            stem_filters=stem_filters,
            base_filters=base_filters,
            bridge_filters=board_bridge_filters,
            bridge_blocks=board_bridge_blocks,
            decoder_mid_filters=board_decoder_mid_filters,
            decoder_mid_blocks=board_decoder_mid_blocks,
            dropout_rate=0.0,
        )
        model: keras.Model = PolarMaskQATTrainingModel(
            base_model=base_model,
            output_noise_stddev=qat_output_noise_stddev,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            min_angle_deg=MIN_ANGLE_DEG,
            sweep_deg=SWEEP_DEG,
            temperature=10.0,
            angle_vector_temperature=angle_vector_temperature,
            profile_head_units=profile_head_units,
            profile_head_dropout=profile_head_dropout,
            state_head_units=state_head_units,
            state_head_dropout=state_head_dropout,
            state_head_bins=len(state_bin_centers) if include_state_output else 0,
        )
    elif tiny:
        base_model = build_polar_tiny_model(
            polar_size=polar_size,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            min_angle_deg=MIN_ANGLE_DEG,
            sweep_deg=SWEEP_DEG,
        )
        model = base_model
    else:
        base_model = build_polar_needle_segmentation_model(
            polar_size=polar_size,
            base_filters=32,
            depth=4,
            dropout_rate=0.1,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            min_angle_deg=MIN_ANGLE_DEG,
            sweep_deg=SWEEP_DEG,
        )
        model = base_model

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_weights.weights.h5"),
            save_weights_only=True,
            monitor="val_gauge_value_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_gauge_value_mae",
            mode="min",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_gauge_value_mae",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # The value head stays the main objective, and the mask branch now learns
    # only the angular profile that matters for the decode path.
    def temp_huber(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Huber loss keeps the value head stable on hard cases."""

        return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=5.0))

    angle_vector_active = board_friendly and angle_vector_aux_loss_weight > 0.0
    profile_aux_active = board_friendly and profile_head_units > 0

    def _compile_board_model(
        *,
        learning_rate_value: float,
        temp_loss_weight_value: float,
        angle_vector_loss_weight_value: float,
        profile_aux_loss_weight_value: float,
        state_loss_weight_value: float,
        include_profile_aux_value: bool,
        include_state_value: bool,
    ) -> None:
        """Compile the board-friendly model with one curriculum phase's weights."""

        compile_loss: dict[str, Any] = {
            "needle_mask": polar_profile_loss,
            "gauge_value": temp_huber,
        }
        compile_loss_weights: dict[str, float] = {
            "needle_mask": mask_loss_weight,
            "gauge_value": temp_loss_weight_value,
        }
        compile_metrics: dict[str, list[Any]] = {
            "gauge_value": [keras.metrics.MeanAbsoluteError(name="mae")],
        }

        if angle_vector_active:
            compile_loss["needle_angle_vector"] = angle_vector_cosine_loss
            compile_loss_weights["needle_angle_vector"] = angle_vector_loss_weight_value
            compile_metrics["needle_angle_vector"] = [angle_vector_mae_deg]

        if include_profile_aux_value and profile_aux_active:
            compile_loss["profile_value_aux"] = temp_huber
            compile_loss_weights["profile_value_aux"] = profile_aux_loss_weight_value
            compile_metrics["profile_value_aux"] = [keras.metrics.MeanAbsoluteError(name="mae")]

        if include_state_value:
            compile_loss["gauge_value_state"] = keras.losses.CategoricalCrossentropy(from_logits=True)
            compile_loss_weights["gauge_value_state"] = state_loss_weight_value
            compile_metrics["gauge_value_state"] = [
                keras.metrics.CategoricalAccuracy(name="acc"),
                keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
            ]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate_value),
            loss=compile_loss,
            loss_weights=compile_loss_weights,
            metrics=compile_metrics,
            jit_compile=False,
        )

    if board_friendly:
        logger.info("Clean export model summary:")
        base_model.summary()

        phase1_epochs = max(1, min(int(geometry_pretrain_epochs), max(1, int(epochs) - 1)))
        phase2_epochs = max(1, int(epochs) - phase1_epochs)
        phase_histories: list[tuple[str, keras.callbacks.History]] = []

        logger.info(
            "\n=== Geometry pretrain phase (%d epochs, clean labels only) ===",
            phase1_epochs,
        )
        if hasattr(model, "output_noise_stddev"):
            model.output_noise_stddev = 0.0
        _compile_board_model(
            learning_rate_value=learning_rate,
            temp_loss_weight_value=max(0.1, temp_loss_weight * 0.2),
            angle_vector_loss_weight_value=max(angle_vector_aux_loss_weight, 6.0),
            profile_aux_loss_weight_value=max(profile_aux_loss_weight, 0.2),
            state_loss_weight_value=max(state_aux_loss_weight, 0.3),
            include_profile_aux_value=profile_aux_active,
            include_state_value=include_state_output,
        )
        pretrain_history = model.fit(
            clean_train_ds,
            validation_data=clean_val_ds,
            epochs=phase1_epochs,
            verbose=1,
        )
        phase_histories.append(("pretrain", pretrain_history))

        logger.info(
            "\n=== Board adaptation phase (%d epochs, mixed labels) ===",
            phase2_epochs,
        )
        if hasattr(model, "output_noise_stddev"):
            model.output_noise_stddev = qat_output_noise_stddev
        _compile_board_model(
            learning_rate_value=learning_rate * float(finetune_learning_rate_multiplier),
            temp_loss_weight_value=temp_loss_weight,
            angle_vector_loss_weight_value=angle_vector_aux_loss_weight,
            profile_aux_loss_weight_value=profile_aux_loss_weight,
            state_loss_weight_value=state_aux_loss_weight,
            include_profile_aux_value=include_profile_aux_output,
            include_state_value=include_state_output,
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase2_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        phase_histories.append(("finetune", history))
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "needle_mask": polar_profile_loss,
                "gauge_value": temp_huber,
            },
            loss_weights={
                "needle_mask": mask_loss_weight,
                "gauge_value": temp_loss_weight,
            },
            metrics={
                "gauge_value": [keras.metrics.MeanAbsoluteError(name="mae")],
            },
            jit_compile=False,
        )
        model.summary()
        phase_histories = []
        logger.info(f"\n=== Training polar geometry-supervised model ({epochs} epochs) ===")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        phase_histories.append(("single", history))

    # Evaluate each source slice separately so we can see where the model is
    # improving and where the remaining error is concentrated.
    metrics: dict[str, Any] = {}
    metrics["validation"] = _evaluate_sample_source(
        source_name="validation",
        samples=val_samples,
        spec=spec,
        model=model,
        export_model=base_model if board_friendly else model,
        output_dir=output_dir,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        input_mode=feature_input_mode,
    )
    metrics["ai_board_captures"] = _evaluate_sample_source(
        source_name="ai_board_captures",
        samples=val_ai,
        spec=spec,
        model=model,
        export_model=base_model if board_friendly else model,
        output_dir=output_dir,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        input_mode=feature_input_mode,
    )
    metrics["manual_centers"] = _evaluate_sample_source(
        source_name="manual_centers",
        samples=val_manual,
        spec=spec,
        model=model,
        export_model=base_model if board_friendly else model,
        output_dir=output_dir,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        input_mode=feature_input_mode,
    )
    metrics["manual_30"] = _evaluate_sample_source(
        source_name="manual_30",
        samples=manual30_samples,
        spec=spec,
        model=model,
        export_model=base_model if board_friendly else model,
        output_dir=output_dir,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        input_mode=feature_input_mode,
    )
    metrics["hard_cases"] = _evaluate_sample_source(
        source_name="hard_case",
        samples=hard_case_samples,
        spec=spec,
        model=model,
        export_model=base_model if board_friendly else model,
        output_dir=output_dir,
        batch_size=batch_size,
        polar_size=polar_size,
        mask_sigma=mask_sigma,
        board_friendly=board_friendly,
        input_mode=feature_input_mode,
    )

    # Save model
    if board_friendly:
        base_model.save(output_dir / "model.keras")
        logger.info(f"Saved clean deployment model to {output_dir / 'model.keras'}")
    else:
        model.save(output_dir / "model.keras")
        logger.info(f"Saved model to {output_dir / 'model.keras'}")

    combined_history: dict[str, list[float]] = {}
    for _, phase_history in phase_histories:
        for key, values in phase_history.history.items():
            combined_history.setdefault(key, []).extend(float(v) for v in values)

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            combined_history,
            f,
            indent=2,
        )

    return {
        "model": model,
        "export_model": base_model if board_friendly else model,
        "history": combined_history,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train polar needle model with geometry supervision")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "training" / "polar_v3_geometry")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--polar-size", type=int, default=POLAR_SIZE)
    parser.add_argument("--center-jitter-px", type=float, default=DEFAULT_CENTER_JITTER_PX)
    parser.add_argument(
        "--radius-scale-min", type=float, default=DEFAULT_RADIUS_SCALE_RANGE[0]
    )
    parser.add_argument(
        "--radius-scale-max", type=float, default=DEFAULT_RADIUS_SCALE_RANGE[1]
    )
    parser.add_argument("--mask-sigma", type=float, default=DEFAULT_MASK_SIGMA)
    parser.add_argument("--augment-repeats", type=int, default=DEFAULT_AUGMENT_REPEATS)
    parser.add_argument("--stem-filters", type=int, default=DEFAULT_STEM_FILTERS)
    parser.add_argument("--base-filters", type=int, default=DEFAULT_BASE_FILTERS)
    parser.add_argument("--bridge-filters", type=int, default=None)
    parser.add_argument("--bridge-blocks", type=int, default=3)
    parser.add_argument("--decoder-mid-filters", type=int, default=None)
    parser.add_argument("--decoder-mid-blocks", type=int, default=2)
    parser.add_argument("--profile-head-units", type=int, default=DEFAULT_PROFILE_HEAD_UNITS)
    parser.add_argument("--profile-head-dropout", type=float, default=DEFAULT_PROFILE_HEAD_DROPOUT)
    parser.add_argument("--board-friendly", action="store_true", dest="board_friendly")
    parser.add_argument("--no-board-friendly", action="store_false", dest="board_friendly")
    parser.set_defaults(board_friendly=True)
    parser.add_argument("--qat-output-noise-stddev", type=float, default=0.01)
    parser.add_argument("--angle-vector-temperature", type=float, default=DEFAULT_ANGLE_VECTOR_TEMPERATURE)
    parser.add_argument("--geometry-pretrain-epochs", type=int, default=DEFAULT_GEOMETRY_PRETRAIN_EPOCHS)
    parser.add_argument(
        "--finetune-learning-rate-multiplier",
        type=float,
        default=DEFAULT_FINETUNE_LR_MULTIPLIER,
    )
    parser.add_argument("--state-head-units", type=int, default=DEFAULT_STATE_HEAD_UNITS)
    parser.add_argument("--state-head-dropout", type=float, default=DEFAULT_STATE_HEAD_DROPOUT)
    parser.add_argument("--state-loss-weight", type=float, default=DEFAULT_STATE_AUX_WEIGHT)
    parser.add_argument("--state-bin-size", type=float, default=VALUE_STATE_BIN_SIZE)
    parser.add_argument("--state-label-sigma", type=float, default=DEFAULT_STATE_LABEL_SIGMA)
    parser.add_argument("--teacher-model-path", type=Path, default=None)
    parser.add_argument("--distill-blend-weight", type=float, default=DEFAULT_DISTILL_BLEND_WEIGHT)
    parser.add_argument("--hard-example-boost", type=float, default=DEFAULT_HARD_EXAMPLE_BOOST)
    parser.add_argument("--hard-example-scale", type=float, default=DEFAULT_HARD_EXAMPLE_SCALE)
    parser.add_argument("--tiny", action="store_true", default=True)
    parser.add_argument("--no-tiny", action="store_true", dest="tiny", default=False)
    parser.add_argument("--mask-loss-weight", type=float, default=0.5)
    parser.add_argument("--temp-loss-weight", type=float, default=1.0)
    parser.add_argument("--profile-aux-loss-weight", type=float, default=DEFAULT_PROFILE_AUX_WEIGHT)
    parser.add_argument("--angle-vector-aux-loss-weight", type=float, default=DEFAULT_ANGLE_AUX_WEIGHT)
    args = parser.parse_args()

    train_polar_geometry_supervised(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        tiny=args.tiny,
        board_friendly=args.board_friendly,
        polar_size=args.polar_size,
        qat_output_noise_stddev=args.qat_output_noise_stddev,
        mask_loss_weight=args.mask_loss_weight,
        temp_loss_weight=args.temp_loss_weight,
        center_jitter_px=args.center_jitter_px,
        radius_scale_range=(args.radius_scale_min, args.radius_scale_max),
        mask_sigma=args.mask_sigma,
        augment_repeats=args.augment_repeats,
          stem_filters=args.stem_filters,
          base_filters=args.base_filters,
          bridge_filters=args.bridge_filters,
          bridge_blocks=args.bridge_blocks,
          decoder_mid_filters=args.decoder_mid_filters,
          decoder_mid_blocks=args.decoder_mid_blocks,
          profile_head_units=args.profile_head_units,
        profile_head_dropout=args.profile_head_dropout,
        profile_aux_loss_weight=args.profile_aux_loss_weight,
        state_head_units=args.state_head_units,
        state_head_dropout=args.state_head_dropout,
        state_aux_loss_weight=args.state_loss_weight,
        state_bin_size=args.state_bin_size,
        state_label_sigma=args.state_label_sigma,
        angle_vector_aux_loss_weight=args.angle_vector_aux_loss_weight,
        angle_vector_temperature=args.angle_vector_temperature,
        geometry_pretrain_epochs=args.geometry_pretrain_epochs,
        finetune_learning_rate_multiplier=args.finetune_learning_rate_multiplier,
        distill_teacher_model_path=args.teacher_model_path,
        distill_blend_weight=args.distill_blend_weight,
        hard_example_boost=args.hard_example_boost,
        hard_example_scale=args.hard_example_scale,
    )


if __name__ == "__main__":
    main()



