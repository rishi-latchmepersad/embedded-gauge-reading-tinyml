"""Uncertainty-gated geometry cascade helpers for gauge reading."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf

from embedded_gauge_reading_tinyml.board_crop_compare import resize_with_pad_rgb
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec

RGBImage = NDArray[np.uint8]


@dataclass(frozen=True)
class CascadePassResult:
    """One pass through the detector plus the geometry-derived confidence."""

    value: float
    confidence: float
    crop_box_xyxy: tuple[float, float, float, float]
    keypoint_coords_xy: NDArray[np.float32]


@dataclass(frozen=True)
class CascadeResult:
    """Final cascade output, including the optional second-pass refinement."""

    first_pass: CascadePassResult
    second_pass: CascadePassResult | None
    final_value: float
    used_second_pass: bool


def _to_numpy(value: Any) -> NDArray[np.float32]:
    """Convert tensors or arrays into a float32 NumPy array."""
    if isinstance(value, tf.Tensor):
        return value.numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _extract_output_map(outputs: Any) -> dict[str, NDArray[np.float32]]:
    """Normalize Keras model outputs into a name->array mapping."""
    if isinstance(outputs, dict):
        return {key: _to_numpy(value) for key, value in outputs.items()}

    if isinstance(outputs, (list, tuple)):
        return {
            f"output_{index}": _to_numpy(value)
            for index, value in enumerate(outputs)
        }

    return {"output": _to_numpy(outputs)}


def _extract_scalar_value(outputs: Any) -> float:
    """Extract a scalar value from a model output bundle."""
    output_map = _extract_output_map(outputs)
    if "gauge_value" in output_map:
        tensor = output_map["gauge_value"]
    else:
        first_key = next(iter(output_map))
        tensor = output_map[first_key]

    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("Model output did not contain any scalar values.")
    return float(flat[0])


def _run_geometry_pass(
    *,
    localizer_model: tf.keras.Model,
    reader_model: tf.keras.Model,
    crop_batch: NDArray[np.float32],
) -> tuple[float, NDArray[np.float32], NDArray[np.float32], float]:
    """Run one localization pass and return the reader value plus geometry."""
    localizer_outputs = _extract_output_map(localizer_model(crop_batch, training=False))
    if "keypoint_heatmaps" not in localizer_outputs:
        raise ValueError("Model must expose keypoint_heatmaps outputs.")

    heatmaps = localizer_outputs["keypoint_heatmaps"][0]
    coords = (
        localizer_outputs["keypoint_coords"][0]
        if "keypoint_coords" in localizer_outputs
        else heatmaps_to_coords(heatmaps)
    )
    confidence = heatmap_confidence(heatmaps)
    if reader_model is localizer_model:
        value = _extract_scalar_value(localizer_outputs)
    else:
        reader_outputs = _extract_output_map(reader_model(crop_batch, training=False))
        value = _extract_scalar_value(reader_outputs)

    return value, coords, heatmaps, confidence


def _softmax_confidence(heatmap: NDArray[np.float32], temperature: float) -> float:
    """Compute a normalized confidence score from one keypoint heatmap."""
    flat = heatmap.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return 0.0
    centered = flat - np.max(flat)
    logits = centered * np.float32(temperature)
    logits -= np.max(logits)
    probs = np.exp(logits)
    denom = float(np.sum(probs))
    if denom <= 0.0:
        return 0.0
    probs /= np.float32(denom)
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
    max_entropy = float(math.log(float(flat.size)))
    if max_entropy <= 0.0:
        return 0.0
    normalized_entropy = entropy / max_entropy
    peakiness = float(np.max(probs))
    return float(np.clip(0.5 * (1.0 - normalized_entropy) + 0.5 * peakiness, 0.0, 1.0))


def heatmap_confidence(heatmaps: NDArray[np.float32], *, temperature: float = 10.0) -> float:
    """Return a confidence score from the model's keypoint heatmaps."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape (height, width, keypoints).")
    per_keypoint = [
        _softmax_confidence(heatmaps[..., keypoint_index], temperature)
        for keypoint_index in range(heatmaps.shape[-1])
    ]
    if not per_keypoint:
        return 0.0
    return float(np.mean(per_keypoint))


def heatmaps_to_coords(
    heatmaps: NDArray[np.float32],
    *,
    temperature: float = 10.0,
) -> NDArray[np.float32]:
    """Convert per-keypoint heatmaps into expected 2D coordinates."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape (height, width, keypoints).")

    height, width, keypoints = heatmaps.shape
    coords = np.zeros((keypoints, 2), dtype=np.float32)
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)

    for keypoint_index in range(keypoints):
        plane = heatmaps[..., keypoint_index].reshape(-1).astype(np.float32)
        if plane.size == 0:
            continue
        logits = (plane - np.max(plane)) * np.float32(temperature)
        logits -= np.max(logits)
        weights = np.exp(logits)
        denom = float(np.sum(weights))
        if denom <= 0.0:
            continue
        weights /= np.float32(denom)
        coords[keypoint_index, 0] = float(np.sum(weights * flat_x))
        coords[keypoint_index, 1] = float(np.sum(weights * flat_y))

    return coords


def crop_box_from_center(
    *,
    center_xy: tuple[float, float],
    base_crop_box_xyxy: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    shrink_scale: float = 0.75,
    min_size: float = 64.0,
) -> tuple[float, float, float, float]:
    """Build a tighter crop around a predicted center while staying in-bounds."""
    if shrink_scale <= 0.0:
        raise ValueError("shrink_scale must be > 0.")
    if min_size <= 0.0:
        raise ValueError("min_size must be > 0.")

    image_height, image_width = image_shape
    x_min, y_min, x_max, y_max = base_crop_box_xyxy
    base_width = max(x_max - x_min, 1.0)
    base_height = max(y_max - y_min, 1.0)
    target_width = max(min_size, base_width * shrink_scale)
    target_height = max(min_size, base_height * shrink_scale)

    center_x, center_y = center_xy
    new_x_min = center_x - 0.5 * target_width
    new_y_min = center_y - 0.5 * target_height
    new_x_max = new_x_min + target_width
    new_y_max = new_y_min + target_height

    if new_x_min < 0.0:
        new_x_max -= new_x_min
        new_x_min = 0.0
    if new_y_min < 0.0:
        new_y_max -= new_y_min
        new_y_min = 0.0
    if new_x_max > float(image_width):
        shift = new_x_max - float(image_width)
        new_x_min = max(0.0, new_x_min - shift)
        new_x_max = float(image_width)
    if new_y_max > float(image_height):
        shift = new_y_max - float(image_height)
        new_y_min = max(0.0, new_y_min - shift)
        new_y_max = float(image_height)

    return (new_x_min, new_y_min, new_x_max, new_y_max)


def source_xy_from_resized_xy(
    resized_xy: tuple[float, float],
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
) -> tuple[float, float]:
    """Map coordinates from a resize-with-pad crop back into source image space."""
    x_min, y_min, x_max, y_max = crop_box_xyxy
    crop_width = max(x_max - x_min, 1.0)
    crop_height = max(y_max - y_min, 1.0)
    scale = min(image_width / crop_width, image_height / crop_height)
    resized_width = crop_width * scale
    resized_height = crop_height * scale
    pad_x = 0.5 * float(image_width - resized_width)
    pad_y = 0.5 * float(image_height - resized_height)

    resized_x, resized_y = resized_xy
    source_x = x_min + (resized_x - pad_x) / scale
    source_y = y_min + (resized_y - pad_y) / scale
    return (source_x, source_y)


def run_geometry_cascade(
    *,
    model: tf.keras.Model,
    source_image: RGBImage,
    base_crop_box_xyxy: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
    input_size: int = 224,
    confidence_threshold: float = 0.55,
    recrop_scale: float = 0.75,
    min_recrop_size: float = 64.0,
    reader_model: tf.keras.Model | None = None,
) -> CascadeResult:
    """Run one localization pass and optionally a tighter second pass."""
    reader_model = model if reader_model is None else reader_model
    base_crop = resize_with_pad_rgb(
        source_image,
        base_crop_box_xyxy,
        image_size=input_size,
    )
    base_batch = np.expand_dims(base_crop.astype(np.float32) / 255.0, axis=0)
    first_value, first_coords, _first_heatmaps, first_confidence = _run_geometry_pass(
        localizer_model=model,
        reader_model=reader_model,
        crop_batch=base_batch,
    )
    first_pass = CascadePassResult(
        value=first_value,
        confidence=first_confidence,
        crop_box_xyxy=base_crop_box_xyxy,
        keypoint_coords_xy=first_coords,
    )

    if first_confidence >= confidence_threshold:
        return CascadeResult(
            first_pass=first_pass,
            second_pass=None,
            final_value=first_value,
            used_second_pass=False,
        )

    keypoint_source_coords = [
        source_xy_from_resized_xy(
            (float(coord[0]), float(coord[1])),
            crop_box_xyxy=base_crop_box_xyxy,
            image_height=image_height,
            image_width=image_width,
        )
        for coord in first_coords
    ]
    center_x = float(np.mean([coord[0] for coord in keypoint_source_coords]))
    center_y = float(np.mean([coord[1] for coord in keypoint_source_coords]))
    tighter_crop_box = crop_box_from_center(
        center_xy=(center_x, center_y),
        base_crop_box_xyxy=base_crop_box_xyxy,
        image_shape=source_image.shape[:2],
        shrink_scale=recrop_scale,
        min_size=min_recrop_size,
    )

    tighter_crop = resize_with_pad_rgb(
        source_image,
        tighter_crop_box,
        image_size=input_size,
    )
    tighter_batch = np.expand_dims(tighter_crop.astype(np.float32) / 255.0, axis=0)
    second_value, second_coords, _second_heatmaps, second_confidence = (
        _run_geometry_pass(
            localizer_model=model,
            reader_model=reader_model,
            crop_batch=tighter_batch,
        )
    )
    second_pass = CascadePassResult(
        value=second_value,
        confidence=second_confidence,
        crop_box_xyxy=tighter_crop_box,
        keypoint_coords_xy=second_coords,
    )

    if second_confidence >= first_confidence:
        return CascadeResult(
            first_pass=first_pass,
            second_pass=second_pass,
            final_value=second_value,
            used_second_pass=True,
        )

    return CascadeResult(
        first_pass=first_pass,
        second_pass=second_pass,
        final_value=first_value,
        used_second_pass=True,
    )
