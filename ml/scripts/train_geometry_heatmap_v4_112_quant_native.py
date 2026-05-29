#!/usr/bin/env python3
"""Train geometry_heatmap_v4_112 with quantization-native objectives.

The v4 schedule keeps fake int8 output round-trips active from the beginning so
the training objective matches the deployment path more closely than the v2/v3
late-finetune/QAT approach while using a 112x112 heatmap head.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    HeatmapSample,
    load_clean_geometry_examples,
    load_heatmap_sample,
    load_selected_calibration_candidate,
    sample_jitter_params,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_quantization_common import (
    DEFAULT_PREPROCESSING_MODE,
    load_split_samples,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v3_quant_native_utils import (
    angle_degrees_from_center_to_tip_tf,
    circular_angle_loss_tf,
    linear_temperature_from_angle_tf,
    normalize_scalar_tf,
    normalized_softargmax_coordinates_tf,
    normalized_temperature_huber_loss_tf,
    temperature_from_coords_tf,
)
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryDecodedPrediction,
    GeometryGuardrailThresholds,
    apply_geometry_guardrails,
    decode_heatmap_geometry_prediction,
)
from embedded_gauge_reading_tinyml.heatmap_losses import (
    softargmax_coordinate_loss,
    weighted_center_heatmap_loss,
    weighted_tip_heatmap_loss,
)
from embedded_gauge_reading_tinyml.models_geometry import build_mobilenetv2_geometry_heatmap_v4_112


TRAIN_JITTER_SHIFT_MIN_PX: int = 4
TRAIN_JITTER_SHIFT_MAX_PX: int = 8
TRAIN_JITTER_SCALE_MIN: float = 0.96
TRAIN_JITTER_SCALE_MAX: float = 1.05
TRAIN_JITTER_ASPECT_MIN: float = 0.98
TRAIN_JITTER_ASPECT_MAX: float = 1.02

DEFAULT_OUTPUT_DIR = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native")
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v3_quant_native_canonical/best_model.keras")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_TRAINING_REPORT_PATH = Path("ml/reports/geometry_heatmap_v4_112_training_full.md")
DEFAULT_STABILITY_REPORT_PATH = Path("ml/reports/geometry_heatmap_v4_112_training_stability.md")

LOSS_WEIGHTS = {
    "center_heatmap": 1.0,
    "tip_heatmap": 2.0,
    "center_coord": 1.0,
    "tip_coord": 2.0,
    "angle": 1.0,
    "temperature": 0.5,
    "confidence": 0.1,
    "distillation": 0.15,
    "peak_shape_center": 0.1,
    "peak_shape_tip": 0.2,
    "confidence_floor": 0.05,
    "aux_coord": 0.1,
    "local_offset": 0.5,
    "axis_simcc": 1.0,
}

NOISE_RAMP_START_STDDEV: float = 0.001
NOISE_RAMP_END_STDDEV: float = 0.008
NOISE_RAMP_EPOCHS: int = 8
SMOKE_DEBUG_BATCH_STEPS: int = 50

PEAK_TARGET: float = 0.3
CONFIDENCE_FLOOR: float = 0.5
EARLY_COLLAPSE_PEAK_THRESHOLD: float = 0.05
EARLY_COLLAPSE_PATIENCE: int = 3
WARMUP_EPOCHS: int = 5
WARMUP_START_LR_FRACTION: float = 0.01

LOCAL_OFFSET_LOSS_WEIGHT_DEFAULT: float = 0.5
LOCAL_OFFSET_SCALE_PX_DEFAULT: float = 8.0
LOCAL_OFFSET_SIGMA_PX_DEFAULT: float = 4.0
LOCAL_OFFSET_TIP_WEIGHT_DEFAULT: float = 1.0

AXIS_SIMCC_SIGMA_BINS_DEFAULT: float = 4.0
AXIS_SIMCC_LOSS_WEIGHT_DEFAULT: float = 1.0
AXIS_SIMCC_TIP_WEIGHT_DEFAULT: float = 2.0


@dataclass(frozen=True)
class GeometryV3Batch:
    """One deterministic mini-batch with heatmap and scalar supervision."""

    samples: list[HeatmapSample]
    x: np.ndarray
    y: dict[str, np.ndarray]


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else repo_root / path


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a stable JSON artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV table."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_history(history: dict[str, list[Any]], output_path: Path) -> None:
    """Persist a callback-friendly history dictionary to CSV.

    The custom train_step returns 'total_loss' rather than 'loss', so we
    derive epoch count from the first available metric list instead of
    relying on a 'loss' key that Keras may not create.
    """

    metric_names = list(history.keys())
    if not metric_names:
        raise ValueError("Refusing to write empty history — no metrics recorded.")
    first_key = metric_names[0]
    epochs = len(history[first_key])
    if epochs == 0:
        raise ValueError("Refusing to write empty history — metric lists are empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(metric_names) + "\n")
        for index in range(epochs):
            values: list[str] = []
            for name in metric_names:
                v = history[name][index]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    values.append("NaN")
                else:
                    values.append(str(v))
            handle.write(f"{index + 1}," + ",".join(values) + "\n")


def _copy_if_exists(source_path: Path, destination_path: Path) -> None:
    """Copy an artifact if it exists."""

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _as_output_dict(outputs: Any) -> dict[str, tf.Tensor]:
    """Normalize model outputs to the semantic heatmap dictionary.

    Supports 3-output (center, tip, confidence),
    4-output with aux_coords, and 4-output with aux_offset_map.
    """

    if isinstance(outputs, dict):
        result = {
            "center_heatmap": tf.cast(outputs["center_heatmap"], tf.float32),
            "tip_heatmap": tf.cast(outputs["tip_heatmap"], tf.float32),
            "confidence": tf.cast(outputs["confidence"], tf.float32),
        }
        if "aux_coords" in outputs:
            result["aux_coords"] = tf.cast(outputs["aux_coords"], tf.float32)
        if "aux_offset_map" in outputs:
            result["aux_offset_map"] = tf.cast(outputs["aux_offset_map"], tf.float32)
        if "axis_logits" in outputs:
            result["axis_logits"] = tf.cast(outputs["axis_logits"], tf.float32)
        return result
    center_heatmap, tip_heatmap, confidence, *extra = outputs
    result = {
        "center_heatmap": tf.cast(center_heatmap, tf.float32),
        "tip_heatmap": tf.cast(tip_heatmap, tf.float32),
        "confidence": tf.cast(confidence, tf.float32),
    }
    if extra:
        extra_tensor = extra[0]
        extra_rank = extra_tensor.shape.rank
        if extra_rank is not None:
            if extra_rank == 3:
                result["axis_logits"] = tf.cast(extra_tensor, tf.float32)
            elif extra_rank >= 4:
                result["aux_offset_map"] = tf.cast(extra_tensor, tf.float32)
            else:
                result["aux_coords"] = tf.cast(extra_tensor, tf.float32)
        else:
            result["aux_coords"] = tf.cast(extra_tensor, tf.float32)
    return result


def _quantize_outputs(outputs: dict[str, tf.Tensor], *, noise_stddev: float) -> dict[str, tf.Tensor]:
    """Inject light noise and fake-quantize all sigmoid-bounded outputs.

    aux_offset_map (tanh [-1, 1]) gets noise but no [0,1] fake-quant clamp.
    """

    TANH_RANGE_NAMES = {"aux_offset_map"}

    quantized: dict[str, tf.Tensor] = {}
    for name, tensor in outputs.items():
        values = tf.cast(tensor, tf.float32)
        if name in TANH_RANGE_NAMES:
            if noise_stddev > 0.0:
                values = values + tf.random.normal(tf.shape(values), stddev=noise_stddev, dtype=values.dtype)
            values = tf.clip_by_value(values, -1.0, 1.0)
        else:
            if noise_stddev > 0.0:
                values = values + tf.random.normal(tf.shape(values), stddev=noise_stddev, dtype=values.dtype)
            values = tf.clip_by_value(values, 0.0, 1.0)
            values = fake_quantize_01_tensor(values)
        quantized[name] = values
    return quantized


def _tensor_stats(tensor: tf.Tensor) -> dict[str, float]:
    """Summarize a tensor for one-batch stability diagnostics."""

    values = tf.cast(tf.convert_to_tensor(tensor), tf.float32)
    finite = tf.math.is_finite(values)
    safe_values = tf.where(finite, values, tf.zeros_like(values))
    return {
        "min": float(tf.reduce_min(safe_values).numpy()),
        "max": float(tf.reduce_max(safe_values).numpy()),
        "mean": float(tf.reduce_mean(safe_values).numpy()),
        "finite_fraction": float(tf.reduce_mean(tf.cast(finite, tf.float32)).numpy()),
    }


def _gradient_group_name(variable_name: str) -> str:
    """Map a trainable variable name to a stable diagnostic group."""

    lower_name = variable_name.lower()
    if "mobilenetv2" in lower_name or "backbone" in lower_name:
        return "backbone"
    if "center" in lower_name:
        return "center_head"
    if "tip" in lower_name:
        return "tip_head"
    if "confidence" in lower_name:
        return "confidence_head"
    return variable_name.split("/")[0]


def _gradient_group_norms(
    gradients: list[tf.Tensor | None],
    variables: list[tf.Variable],
) -> dict[str, float]:
    """Aggregate gradient norms by trainable variable group."""

    grouped: dict[str, list[tf.Tensor]] = {}
    for gradient, variable in zip(gradients, variables, strict=True):
        if gradient is None:
            continue
        grouped.setdefault(_gradient_group_name(variable.name), []).append(tf.cast(gradient, tf.float32))

    norms: dict[str, float] = {}
    for group_name, group_gradients in grouped.items():
        if not group_gradients:
            continue
        norm = tf.linalg.global_norm(group_gradients)
        norms[group_name] = float(norm.numpy())
    return dict(sorted(norms.items(), key=lambda item: item[0]))


def _print_loss_diagnostics(
    *,
    prefix: str,
    losses: dict[str, tf.Tensor],
    raw_outputs: dict[str, tf.Tensor],
    quantized_outputs: dict[str, tf.Tensor],
    gradients: list[tf.Tensor | None],
    variables: list[tf.Variable],
) -> None:
    """Print a compact, human-readable diagnostic block."""

    print(f"[{prefix}] loss diagnostics:", flush=True)
    for name in (
        "center_heatmap_loss",
        "tip_heatmap_loss",
        "center_coord_loss",
        "tip_coord_loss",
        "angle_loss",
        "temperature_loss",
        "confidence_loss",
        "peak_shape_center_loss",
        "peak_shape_tip_loss",
        "confidence_floor_loss",
        "distillation_loss",
        "total_loss",
    ):
        value = float(tf.cast(losses[name], tf.float32).numpy())
        print(f"  {name}: {value:.8f}", flush=True)
    global_gradient_norm = float(tf.linalg.global_norm([tf.cast(gradient, tf.float32) for gradient in gradients if gradient is not None]).numpy()) if any(gradient is not None for gradient in gradients) else 0.0
    print(f"  global_gradient_norm: {global_gradient_norm:.8f}", flush=True)
    print(f"  gradients_finite: {all(bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient in gradients if gradient is not None)}", flush=True)
    print(f"  losses_finite: {all(bool(tf.reduce_all(tf.math.is_finite(loss)).numpy()) for loss in losses.values())}", flush=True)
    print(f"  raw_outputs_finite: {all(bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()) for tensor in raw_outputs.values())}", flush=True)
    print(f"  quantized_outputs_finite: {all(bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()) for tensor in quantized_outputs.values())}", flush=True)
    for name, tensor in raw_outputs.items():
        stats = _tensor_stats(tensor)
        print(
            f"  raw_{name}: min={stats['min']:.8f} max={stats['max']:.8f} mean={stats['mean']:.8f} finite_fraction={stats['finite_fraction']:.8f}",
            flush=True,
        )
    for name, tensor in quantized_outputs.items():
        stats = _tensor_stats(tensor)
        print(
            f"  fake_quant_{name}: min={stats['min']:.8f} max={stats['max']:.8f} mean={stats['mean']:.8f} finite_fraction={stats['finite_fraction']:.8f}",
            flush=True,
        )
    for group_name, norm in _gradient_group_norms(gradients, variables).items():
        print(f"  grad_group[{group_name}]: {norm:.8f}", flush=True)


def _apply_gradients_with_clipping(
    *,
    optimizer: keras.optimizers.Optimizer,
    gradients: list[tf.Tensor | None],
    variables: list[tf.Variable],
    clipnorm: float,
) -> tuple[list[tf.Tensor | None], tf.Tensor]:
    """Clip gradients by global norm before applying them."""

    non_none_gradients = [gradient for gradient in gradients if gradient is not None]
    if non_none_gradients:
        global_norm = tf.linalg.global_norm([tf.cast(gradient, tf.float32) for gradient in non_none_gradients])
        clipped_gradients, _ = tf.clip_by_global_norm(non_none_gradients, clipnorm)
        clipped_iter = iter(clipped_gradients)
        clipped_with_none: list[tf.Tensor | None] = []
        for gradient in gradients:
            clipped_with_none.append(next(clipped_iter) if gradient is not None else None)
        optimizer.apply_gradients(
            [(gradient, variable) for gradient, variable in zip(clipped_with_none, variables, strict=True) if gradient is not None]
        )
        return clipped_with_none, global_norm
    return gradients, tf.constant(0.0, dtype=tf.float32)


def _masked_mean(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Compute the mean of values for rows where mask is true."""

    values = tf.cast(values, tf.float32)
    mask = tf.cast(mask, tf.bool)
    masked = tf.boolean_mask(values, mask)
    return tf.cond(
        tf.equal(tf.size(masked), 0),
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: tf.reduce_mean(masked),
    )


def _make_local_offset_targets(
    center_x_224: np.ndarray,
    center_y_224: np.ndarray,
    tip_x_224: np.ndarray,
    tip_y_224: np.ndarray,
    *,
    heatmap_size: int,
    offset_scale_px: float,
    offset_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate local offset map targets and Gaussian weight masks.

    Returns:
      - aux_offset_map: (batch, H, W, 4) with channels [center_dx, center_dy,
        tip_dx, tip_dy] in tanh range [-1, 1].
      - center_weight_mask: (batch, H, W, 1) Gaussian centered at true center.
      - tip_weight_mask: (batch, H, W, 1) Gaussian centered at true tip.
    """
    batch_size = len(center_x_224)
    hw = heatmap_size
    grid_col, grid_row = np.meshgrid(
        np.arange(hw, dtype=np.float32),
        np.arange(hw, dtype=np.float32),
        indexing="xy",
    )
    grid_col = grid_col[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
    grid_row = grid_row[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

    # True keypoints in heatmap pixel space (224-space / 2)
    scale = 224.0 / float(hw)
    cx_h = (center_x_224.reshape(batch_size, 1, 1, 1) / scale).astype(np.float32)
    cy_h = (center_y_224.reshape(batch_size, 1, 1, 1) / scale).astype(np.float32)
    tx_h = (tip_x_224.reshape(batch_size, 1, 1, 1) / scale).astype(np.float32)
    ty_h = (tip_y_224.reshape(batch_size, 1, 1, 1) / scale).astype(np.float32)

    # dx/dy target: (true_position - grid_position) / scale
    center_dx = ((cx_h - grid_col) / offset_scale_px).clip(-1.0, 1.0)
    center_dy = ((cy_h - grid_row) / offset_scale_px).clip(-1.0, 1.0)
    tip_dx = ((tx_h - grid_col) / offset_scale_px).clip(-1.0, 1.0)
    tip_dy = ((ty_h - grid_row) / offset_scale_px).clip(-1.0, 1.0)

    aux_offset_map = np.concatenate([center_dx, center_dy, tip_dx, tip_dy], axis=-1)

    # Gaussian weight: exp(-dist^2 / (2 * sigma^2))
    neg_two_sigma2 = -2.0 * offset_sigma_px * offset_sigma_px
    center_sq_dist = (grid_col - cx_h) ** 2 + (grid_row - cy_h) ** 2
    tip_sq_dist = (grid_col - tx_h) ** 2 + (grid_row - ty_h) ** 2
    center_weight_mask = np.exp(center_sq_dist / neg_two_sigma2).astype(np.float32)
    tip_weight_mask = np.exp(tip_sq_dist / neg_two_sigma2).astype(np.float32)

    return aux_offset_map, center_weight_mask, tip_weight_mask


def _make_axis_simcc_targets(
    center_x_224: np.ndarray,
    center_y_224: np.ndarray,
    tip_x_224: np.ndarray,
    tip_y_224: np.ndarray,
    *,
    num_bins: int = 112,
    sigma_bins: float = 4.0,
) -> np.ndarray:
    """Generate 1D Gaussian soft targets for axis_simcc logit head.

    Returns (batch, 4, num_bins) with soft targets summing to 1 along axis 2.
    Order: [center_x, center_y, tip_x, tip_y].
    """
    batch_size = len(center_x_224)
    # Convert 224-pixel coords to bin indices: bin = coord * (num_bins-1) / 223
    scale = (num_bins - 1) / 223.0
    coords = np.stack([
        center_x_224 * scale,  # (batch,)
        center_y_224 * scale,
        tip_x_224 * scale,
        tip_y_224 * scale,
    ], axis=1)  # (batch, 4)

    bins = np.arange(num_bins, dtype=np.float32)  # (num_bins,)
    coords = coords[:, :, np.newaxis]  # (batch, 4, 1)
    bins = bins[np.newaxis, np.newaxis, :]  # (1, 1, num_bins)

    neg_two_sigma2 = -2.0 * sigma_bins * sigma_bins
    sq_dist = (bins - coords) ** 2
    targets = np.exp(sq_dist / neg_two_sigma2)  # (batch, 4, num_bins)

    # Normalize to soft targets summing to 1
    targets_sum = targets.sum(axis=2, keepdims=True)
    targets_sum = np.maximum(targets_sum, 1e-8)
    targets = targets / targets_sum

    return targets.astype(np.float32)


def _build_targets(
    samples: list[HeatmapSample],
    *,
    include_local_offset_map: bool = False,
    local_offset_scale_px: float = LOCAL_OFFSET_SCALE_PX_DEFAULT,
    local_offset_sigma_px: float = LOCAL_OFFSET_SIGMA_PX_DEFAULT,
    include_axis_simcc_targets: bool = False,
    axis_simcc_sigma_bins: float = AXIS_SIMCC_SIGMA_BINS_DEFAULT,
) -> dict[str, np.ndarray]:
    """Build the full supervision dictionary for one split."""

    center_x_224 = np.asarray([[float(sample.metadata["center_x_224"])] for sample in samples], dtype=np.float32)
    center_y_224 = np.asarray([[float(sample.metadata["center_y_224"])] for sample in samples], dtype=np.float32)
    tip_x_224 = np.asarray([[float(sample.metadata["tip_x_224"])] for sample in samples], dtype=np.float32)
    tip_y_224 = np.asarray([[float(sample.metadata["tip_y_224"])] for sample in samples], dtype=np.float32)

    targets: dict[str, np.ndarray] = {
        "center_heatmap": np.stack([sample.center_heatmap[..., np.newaxis] for sample in samples], axis=0).astype(np.float32),
        "tip_heatmap": np.stack([sample.tip_heatmap[..., np.newaxis] for sample in samples], axis=0).astype(np.float32),
        "confidence": np.ones((len(samples), 1), dtype=np.float32),
        "true_center_x_norm": np.asarray([[float(sample.metadata["center_x_norm"])] for sample in samples], dtype=np.float32),
        "true_center_y_norm": np.asarray([[float(sample.metadata["center_y_norm"])] for sample in samples], dtype=np.float32),
        "true_tip_x_norm": np.asarray([[float(sample.metadata["tip_x_norm"])] for sample in samples], dtype=np.float32),
        "true_tip_y_norm": np.asarray([[float(sample.metadata["tip_y_norm"])] for sample in samples], dtype=np.float32),
        "true_center_x_224": center_x_224,
        "true_center_y_224": center_y_224,
        "true_tip_x_224": tip_x_224,
        "true_tip_y_224": tip_y_224,
        "true_angle_degrees": np.asarray([[float(sample.metadata["angle_degrees"])] for sample in samples], dtype=np.float32),
        "temperature_c": np.asarray([[float(sample.metadata["temperature_c"])] for sample in samples], dtype=np.float32),
        "aux_coords": np.concatenate(
            [
                np.asarray([[float(sample.metadata["center_x_norm"])] for sample in samples], dtype=np.float32),
                np.asarray([[float(sample.metadata["center_y_norm"])] for sample in samples], dtype=np.float32),
                np.asarray([[float(sample.metadata["tip_x_norm"])] for sample in samples], dtype=np.float32),
                np.asarray([[float(sample.metadata["tip_y_norm"])] for sample in samples], dtype=np.float32),
            ],
            axis=-1,
        ),
    }

    if include_local_offset_map:
        offset_map, center_w, tip_w = _make_local_offset_targets(
            center_x_224.squeeze(axis=-1),
            center_y_224.squeeze(axis=-1),
            tip_x_224.squeeze(axis=-1),
            tip_y_224.squeeze(axis=-1),
            heatmap_size=112,
            offset_scale_px=local_offset_scale_px,
            offset_sigma_px=local_offset_sigma_px,
        )
        targets["aux_offset_map"] = offset_map
        targets["aux_offset_center_weight"] = center_w
        targets["aux_offset_tip_weight"] = tip_w

    if include_axis_simcc_targets:
        targets["axis_logits_target"] = _make_axis_simcc_targets(
            center_x_224.squeeze(axis=-1),
            center_y_224.squeeze(axis=-1),
            tip_x_224.squeeze(axis=-1),
            tip_y_224.squeeze(axis=-1),
            num_bins=112,
            sigma_bins=axis_simcc_sigma_bins,
        )

    return targets


class GeometryV3Sequence(keras.utils.Sequence):
    """Jittered training batches generated on the fly from clean train rows."""

    def __init__(
        self,
        examples: list[Any],
        *,
        base_path: Path,
        batch_size: int,
        heatmap_size: int,
        sigma_pixels: float,
        seed: int,
        include_local_offset_map: bool = False,
        local_offset_scale_px: float = LOCAL_OFFSET_SCALE_PX_DEFAULT,
        local_offset_sigma_px: float = LOCAL_OFFSET_SIGMA_PX_DEFAULT,
        include_axis_simcc_targets: bool = False,
        axis_simcc_sigma_bins: float = AXIS_SIMCC_SIGMA_BINS_DEFAULT,
        inner_celsius_mask: bool = False,
    ) -> None:
        self.examples = list(examples)
        self.base_path = base_path
        self.batch_size = int(batch_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma_pixels = float(sigma_pixels)
        self.seed = int(seed)
        self.include_local_offset_map = include_local_offset_map
        self.local_offset_scale_px = float(local_offset_scale_px)
        self.local_offset_sigma_px = float(local_offset_sigma_px)
        self.include_axis_simcc_targets = include_axis_simcc_targets
        self.axis_simcc_sigma_bins = float(axis_simcc_sigma_bins)
        self.inner_celsius_mask = bool(inner_celsius_mask)
        self.indices = np.arange(len(self.examples))
        self.epoch = 0

    def __len__(self) -> int:
        return int(math.ceil(len(self.examples) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        """Shuffle the sample order between epochs."""

        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)
        self.epoch += 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build one jittered mini-batch."""

        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.examples))
        batch_indices = self.indices[batch_start:batch_end]

        batch_x: list[np.ndarray] = []
        batch_center: list[np.ndarray] = []
        batch_tip: list[np.ndarray] = []
        batch_center_x: list[np.ndarray] = []
        batch_center_y: list[np.ndarray] = []
        batch_tip_x: list[np.ndarray] = []
        batch_tip_y: list[np.ndarray] = []
        batch_center_x_norm: list[np.ndarray] = []
        batch_center_y_norm: list[np.ndarray] = []
        batch_tip_x_norm: list[np.ndarray] = []
        batch_tip_y_norm: list[np.ndarray] = []
        batch_angle: list[np.ndarray] = []
        batch_temperature: list[np.ndarray] = []

        for order_index, example_index in enumerate(batch_indices):
            example = self.examples[int(example_index)]
            jitter_rng = np.random.default_rng(self.seed + self.epoch * 100_000 + int(example_index) * 997 + order_index)
            jitter = sample_jitter_params(
                jitter_rng,
                shift_min_px=TRAIN_JITTER_SHIFT_MIN_PX,
                shift_max_px=TRAIN_JITTER_SHIFT_MAX_PX,
                scale_min=TRAIN_JITTER_SCALE_MIN,
                scale_max=TRAIN_JITTER_SCALE_MAX,
                aspect_min=TRAIN_JITTER_ASPECT_MIN,
                aspect_max=TRAIN_JITTER_ASPECT_MAX,
            )
            sample = load_heatmap_sample(
                example,
                self.base_path,
                heatmap_size=self.heatmap_size,
                sigma_pixels=self.sigma_pixels,
                jitter=jitter,
                inner_celsius_mask=self.inner_celsius_mask,
            )
            batch_x.append(sample.crop_image.astype(np.float32))
            batch_center.append(sample.center_heatmap.astype(np.float32)[..., np.newaxis])
            batch_tip.append(sample.tip_heatmap.astype(np.float32)[..., np.newaxis])
            batch_center_x.append(np.asarray([float(sample.metadata["center_x_224"])], dtype=np.float32))
            batch_center_y.append(np.asarray([float(sample.metadata["center_y_224"])], dtype=np.float32))
            batch_tip_x.append(np.asarray([float(sample.metadata["tip_x_224"])], dtype=np.float32))
            batch_tip_y.append(np.asarray([float(sample.metadata["tip_y_224"])], dtype=np.float32))
            batch_center_x_norm.append(np.asarray([float(sample.metadata["center_x_norm"])], dtype=np.float32))
            batch_center_y_norm.append(np.asarray([float(sample.metadata["center_y_norm"])], dtype=np.float32))
            batch_tip_x_norm.append(np.asarray([float(sample.metadata["tip_x_norm"])], dtype=np.float32))
            batch_tip_y_norm.append(np.asarray([float(sample.metadata["tip_y_norm"])], dtype=np.float32))
            batch_angle.append(np.asarray([float(sample.metadata["angle_degrees"])], dtype=np.float32))
            batch_temperature.append(np.asarray([float(sample.metadata["temperature_c"])], dtype=np.float32))

        x = np.stack(batch_x, axis=0)
        y = {
            "center_heatmap": np.stack(batch_center, axis=0),
            "tip_heatmap": np.stack(batch_tip, axis=0),
            "confidence": np.ones((len(batch_x), 1), dtype=np.float32),
            "true_center_x_224": np.stack(batch_center_x, axis=0),
            "true_center_y_224": np.stack(batch_center_y, axis=0),
            "true_tip_x_224": np.stack(batch_tip_x, axis=0),
            "true_tip_y_224": np.stack(batch_tip_y, axis=0),
            "true_center_x_norm": np.stack(batch_center_x_norm, axis=0),
            "true_center_y_norm": np.stack(batch_center_y_norm, axis=0),
            "true_tip_x_norm": np.stack(batch_tip_x_norm, axis=0),
            "true_tip_y_norm": np.stack(batch_tip_y_norm, axis=0),
            "true_angle_degrees": np.stack(batch_angle, axis=0),
            "temperature_c": np.stack(batch_temperature, axis=0),
            "aux_coords": np.concatenate(
                [
                    np.stack(batch_center_x_norm, axis=0),
                    np.stack(batch_center_y_norm, axis=0),
                    np.stack(batch_tip_x_norm, axis=0),
                    np.stack(batch_tip_y_norm, axis=0),
                ],
                axis=-1,
            ),
        }

        if self.include_local_offset_map:
            center_x_arr = np.stack(batch_center_x, axis=0).squeeze(axis=-1)
            center_y_arr = np.stack(batch_center_y, axis=0).squeeze(axis=-1)
            tip_x_arr = np.stack(batch_tip_x, axis=0).squeeze(axis=-1)
            tip_y_arr = np.stack(batch_tip_y, axis=0).squeeze(axis=-1)
            offset_map, center_w, tip_w = _make_local_offset_targets(
                center_x_arr,
                center_y_arr,
                tip_x_arr,
                tip_y_arr,
                heatmap_size=self.heatmap_size,
                offset_scale_px=self.local_offset_scale_px,
                offset_sigma_px=self.local_offset_sigma_px,
            )
            y["aux_offset_map"] = offset_map
            y["aux_offset_center_weight"] = center_w
            y["aux_offset_tip_weight"] = tip_w

        if self.include_axis_simcc_targets:
            center_x_arr = np.stack(batch_center_x, axis=0).squeeze(axis=-1)
            center_y_arr = np.stack(batch_center_y, axis=0).squeeze(axis=-1)
            tip_x_arr = np.stack(batch_tip_x, axis=0).squeeze(axis=-1)
            tip_y_arr = np.stack(batch_tip_y, axis=0).squeeze(axis=-1)
            y["axis_logits_target"] = _make_axis_simcc_targets(
                center_x_arr, center_y_arr, tip_x_arr, tip_y_arr,
                num_bins=self.heatmap_size,
                sigma_bins=self.axis_simcc_sigma_bins,
            )

        return x, y


class ReplayMetricCallback(keras.callbacks.Callback):
    """Evaluate replay-style metrics on the validation split after each epoch."""

    def __init__(
        self,
        *,
        metric_prefix: str,
        samples: list[HeatmapSample],
        inputs: np.ndarray,
        reference_model: keras.Model,
        calibration_candidate: Any,
        thresholds: GeometryGuardrailThresholds,
        decode_method: str,
        window_size: int,
        best_model_path: Path,
    ) -> None:
        super().__init__()
        self.metric_prefix = metric_prefix
        self.samples = list(samples)
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.reference_model = reference_model
        self.calibration_candidate = calibration_candidate
        self.thresholds = thresholds
        self.decode_method = decode_method
        self.window_size = int(window_size)
        self.best_model_path = best_model_path
        self.best_score: tuple[float, float, float, float, float, float] = (
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
        )
        self.best_summary: dict[str, float] | None = None
        self._epoch_summaries: list[dict[str, Any]] = []

    def _evaluate_model(
        self, model: keras.Model, *, quantized_like: bool
    ) -> tuple[list[dict[str, Any]], list[GeometryDecodedPrediction]]:
        """Decode one model on the stored split.

        Returns (guarded_rows, decoded_predictions).

        Uses __call__ directly (training=False) instead of model.predict()
        to avoid TF function retracing and GPU memory leaks across epochs.
        """

        outputs = _as_output_dict(model(self.inputs, training=False))

        rows: list[dict[str, Any]] = []
        decoded_predictions: list[GeometryDecodedPrediction] = []
        for index, sample in enumerate(self.samples):
            confidence = float(np.ravel(outputs["confidence"][index])[0])
            decoded = decode_heatmap_geometry_prediction(
                sample,
                outputs["center_heatmap"][index],
                outputs["tip_heatmap"][index],
                confidence,
                self.calibration_candidate,
                decode_method=self.decode_method,
                window_size=self.window_size,
            )
            guarded = apply_geometry_guardrails(decoded, self.thresholds)
            rows.append(
                {
                    "image_path": str(sample.metadata["image_path"]),
                    "true_temperature_c": float(sample.metadata["temperature_c"]),
                    "guardrail_status": guarded.status,
                    "guarded_temperature_c": float(guarded.temperature_c),
                    "predicted_temperature_c": float(decoded.predicted_temperature_c_calibrated),
                    "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
                    "predicted_center_x_224": float(decoded.predicted_center_x_224),
                    "predicted_center_y_224": float(decoded.predicted_center_y_224),
                    "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
                    "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
                    "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
                    "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
                    "center_heatmap_spread_px": float(guarded.quality_features.center_heatmap_spread_px),
                    "tip_heatmap_spread_px": float(guarded.quality_features.tip_heatmap_spread_px),
                    "confidence": float(confidence),
                    "rejection_reasons": ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
                }
            )
            decoded_predictions.append(decoded)
        return rows, decoded_predictions

    # Shadow spread-guard threshold configs for diagnostic evaluation.
    # "spread_disabled" uses an extreme max to effectively skip spread checks.
    SHADOW_SPREAD_VALUES: dict[str, float] = {
        "spread_45": 45.0,
        "spread_55": 55.0,
        "spread_65": 65.0,
        "spread_disabled": 999.0,
    }

    def _build_shadow_rows(
        self,
        decoded_predictions: list[GeometryDecodedPrediction],
        shadow_thresholds: GeometryGuardrailThresholds,
    ) -> list[dict[str, Any]]:
        """Build evaluation rows from decoded predictions using shadow guardrails."""

        rows: list[dict[str, Any]] = []
        for decoded in decoded_predictions:
            guarded = apply_geometry_guardrails(decoded, shadow_thresholds)
            rows.append(
                {
                    "image_path": decoded.image_path,
                    "true_temperature_c": decoded.true_temperature_c,
                    "guardrail_status": guarded.status,
                    "guarded_temperature_c": float(guarded.temperature_c),
                    "predicted_temperature_c": float(decoded.predicted_temperature_c_calibrated),
                    "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
                    "predicted_center_x_224": float(decoded.predicted_center_x_224),
                    "predicted_center_y_224": float(decoded.predicted_center_y_224),
                    "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
                    "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
                    "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
                    "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
                    "center_heatmap_spread_px": float(guarded.quality_features.center_heatmap_spread_px),
                    "tip_heatmap_spread_px": float(guarded.quality_features.tip_heatmap_spread_px),
                    "confidence": float(decoded.confidence),
                    "rejection_reasons": ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
                }
            )
        return rows

    def _shadow_summaries(
        self,
        decoded_predictions: list[GeometryDecodedPrediction],
    ) -> dict[str, dict[str, Any]]:
        """Evaluate all shadow spread configurations and return summaries keyed by name."""

        results: dict[str, dict[str, Any]] = {}
        for name, spread_px in self.SHADOW_SPREAD_VALUES.items():
            shadow_thresholds = replace(self.thresholds, max_heatmap_spread_px=spread_px)
            shadow_rows = self._build_shadow_rows(decoded_predictions, shadow_thresholds)
            results[name] = self._summarize(shadow_rows)
        return results

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Summarize replay metrics for one decoded model."""

        accepted = [row for row in rows if str(row["guardrail_status"]) in {"accepted", "clamped"}]
        accepted_errors = np.asarray(
            [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in accepted],
            dtype=np.float64,
        )
        all_errors = np.asarray(
            [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in rows],
            dtype=np.float64,
        )
        return {
            "count": float(len(rows)),
            "accepted_count": float(len(accepted)),
            "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
            "acceptance_rate": float(len(accepted) / len(rows)) if rows else math.nan,
            "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
            "accepted_gt20_failures": float(
                sum(
                    1
                    for row in accepted
                    if abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
                )
            ),
            "percentage_under_2c": float(np.mean(all_errors < 2.0) * 100.0) if all_errors.size else math.nan,
            "percentage_under_5c": float(np.mean(all_errors < 5.0) * 100.0) if all_errors.size else math.nan,
            "percentage_under_10c": float(np.mean(all_errors < 10.0) * 100.0) if all_errors.size else math.nan,
            "center_mae_px_224": float(
                np.mean(
                    [
                        math.hypot(
                            float(row["predicted_center_x_224"]) - float(sample.metadata["center_x_224"]),
                            float(row["predicted_center_y_224"]) - float(sample.metadata["center_y_224"]),
                        )
                        for row, sample in zip(rows, self.samples, strict=True)
                    ]
                )
            ),
            "tip_mae_px_224": float(
                np.mean(
                    [
                        math.hypot(
                            float(row["predicted_tip_x_224"]) - float(sample.metadata["tip_x_224"]),
                            float(row["predicted_tip_y_224"]) - float(sample.metadata["tip_y_224"]),
                        )
                        for row, sample in zip(rows, self.samples, strict=True)
                    ]
                )
            ),
            "angle_mae_degrees": float(
                np.mean(
                    [
                        abs(((float(row["predicted_angle_degrees"]) - float(sample.metadata["angle_degrees"]) + 180.0) % 360.0) - 180.0)
                        for row, sample in zip(rows, self.samples, strict=True)
                    ]
                )
            ),
            "center_heatmap_peak_mean": float(np.mean([float(row["center_heatmap_peak_value"]) for row in rows])),
            "tip_heatmap_peak_mean": float(np.mean([float(row["tip_heatmap_peak_value"]) for row in rows])),
            "center_heatmap_spread_mean": float(np.mean([float(row["center_heatmap_spread_px"]) for row in rows])),
            "tip_heatmap_spread_mean": float(np.mean([float(row["tip_heatmap_spread_px"]) for row in rows])),
            "confidence_mean": float(np.mean([float(row["confidence"]) for row in rows])),
            "guardrail_disagreement_count": float(sum(1 for row in rows if str(row["guardrail_status"]) != "accepted")),
            "top_rejection_reasons": _top_rejection_reason_string(rows),
        }

    def _compare_against_reference(
        self,
        current_rows: list[dict[str, Any]],
        reference_rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute current-vs-reference drift metrics."""

        base_by_image = {row["image_path"]: row for row in reference_rows}
        current_by_image = {row["image_path"]: row for row in current_rows}
        common_images = sorted(base_by_image.keys() & current_by_image.keys())
        temp_deltas: list[float] = []
        center_deltas: list[float] = []
        tip_deltas: list[float] = []
        guardrail_disagreements = 0
        for image_path in common_images:
            base_row = base_by_image[image_path]
            current_row = current_by_image[image_path]
            if str(base_row["guardrail_status"]) in {"accepted", "clamped"} and str(current_row["guardrail_status"]) in {"accepted", "clamped"}:
                temp_deltas.append(abs(float(base_row["guarded_temperature_c"]) - float(current_row["guarded_temperature_c"])))
            center_deltas.append(
                math.hypot(
                    float(base_row["predicted_center_x_224"]) - float(current_row["predicted_center_x_224"]),
                    float(base_row["predicted_center_y_224"]) - float(current_row["predicted_center_y_224"]),
                )
            )
            tip_deltas.append(
                math.hypot(
                    float(base_row["predicted_tip_x_224"]) - float(current_row["predicted_tip_x_224"]),
                    float(base_row["predicted_tip_y_224"]) - float(current_row["predicted_tip_y_224"]),
                )
            )
            if str(base_row["guardrail_status"]) != str(current_row["guardrail_status"]):
                guardrail_disagreements += 1
        return {
            "temperature_delta_mean": float(np.mean(temp_deltas)) if temp_deltas else math.nan,
            "temperature_delta_median": float(np.median(temp_deltas)) if temp_deltas else math.nan,
            "temperature_delta_p90": float(np.percentile(temp_deltas, 90)) if temp_deltas else math.nan,
            "center_delta_mean": float(np.mean(center_deltas)) if center_deltas else math.nan,
            "center_delta_median": float(np.median(center_deltas)) if center_deltas else math.nan,
            "tip_delta_mean": float(np.mean(tip_deltas)) if tip_deltas else math.nan,
            "tip_delta_median": float(np.median(tip_deltas)) if tip_deltas else math.nan,
            "guardrail_disagreements": float(guardrail_disagreements),
        }

    def _score_summary(self, summary: dict[str, float]) -> tuple[float, float, float, float, float, float]:
        """Rank checkpoints by deployment safety first, then drift."""

        return (
            0.0 if summary["accepted_gt20_failures"] <= 0.0 else 1.0,
            0.0 if summary["worst_accepted_error_c"] < 20.0 else 1.0,
            0.0 if summary["acceptance_rate"] >= 0.65 else 1.0,
            0.0 if summary["accepted_mae_c"] <= 4.5 else 1.0,
            float(summary["temperature_delta_mean"]),
            float(summary["tip_delta_mean"]),
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Record the replay-style validation metric and save the best weights."""

        logs = {} if logs is None else logs
        reference_rows, _ = self._evaluate_model(self.reference_model, quantized_like=False)
        current_rows, current_decoded = self._evaluate_model(self.model.base_model, quantized_like=True)
        reference_summary = self._summarize(reference_rows)
        current_summary = self._summarize(current_rows)
        drift = self._compare_against_reference(current_rows, reference_rows)

        # Shadow spread-guard diagnostic summaries.
        shadow_results = self._shadow_summaries(current_decoded)

        summary = {**current_summary, **drift}
        score = self._score_summary(summary)
        if score < self.best_score:
            self.best_score = score
            self.model.base_model.save(self.best_model_path)
            self.best_summary = summary

        logs[f"val_{self.metric_prefix}_accepted_mae"] = current_summary["accepted_mae_c"]
        logs[f"val_{self.metric_prefix}_acceptance_rate"] = current_summary["acceptance_rate"]
        logs[f"val_{self.metric_prefix}_worst_accepted_error"] = current_summary["worst_accepted_error_c"]
        logs[f"val_{self.metric_prefix}_accepted_gt20_failures"] = current_summary["accepted_gt20_failures"]
        logs[f"val_{self.metric_prefix}_temperature_delta_mean"] = drift["temperature_delta_mean"]
        logs[f"val_{self.metric_prefix}_temperature_delta_median"] = drift["temperature_delta_median"]
        logs[f"val_{self.metric_prefix}_temperature_delta_p90"] = drift["temperature_delta_p90"]
        logs[f"val_{self.metric_prefix}_center_delta_mean"] = drift["center_delta_mean"]
        logs[f"val_{self.metric_prefix}_tip_delta_mean"] = drift["tip_delta_mean"]
        logs[f"val_{self.metric_prefix}_guardrail_disagreements"] = drift["guardrail_disagreements"]
        logs[f"val_{self.metric_prefix}_score"] = float(sum(score))

        logs["val_center_mae_px"] = current_summary["center_mae_px_224"]
        logs["val_tip_mae_px"] = current_summary["tip_mae_px_224"]
        logs["val_angle_mae_deg"] = current_summary["angle_mae_degrees"]
        logs["val_center_heatmap_peak_mean"] = current_summary["center_heatmap_peak_mean"]
        logs["val_tip_heatmap_peak_mean"] = current_summary["tip_heatmap_peak_mean"]
        logs["val_center_heatmap_spread_mean"] = current_summary["center_heatmap_spread_mean"]
        logs["val_tip_heatmap_spread_mean"] = current_summary["tip_heatmap_spread_mean"]

        # Shadow spread-guard metrics to CSV logs (numeric only).
        for shadow_name, shadow_summary in shadow_results.items():
            logs[f"val_shadow_{shadow_name}_accepted_mae_c"] = shadow_summary["accepted_mae_c"]
            logs[f"val_shadow_{shadow_name}_acceptance_rate"] = shadow_summary["acceptance_rate"]
            logs[f"val_shadow_{shadow_name}_worst_accepted_error_c"] = shadow_summary["worst_accepted_error_c"]
            logs[f"val_shadow_{shadow_name}_accepted_gt20_failures"] = shadow_summary["accepted_gt20_failures"]
            logs[f"val_shadow_{shadow_name}_center_mae_px_224"] = shadow_summary["center_mae_px_224"]
            logs[f"val_shadow_{shadow_name}_tip_mae_px_224"] = shadow_summary["tip_mae_px_224"]
            logs[f"val_shadow_{shadow_name}_angle_mae_degrees"] = shadow_summary["angle_mae_degrees"]

        # Persist rejection-reason strings to a per-epoch JSON file so we
        # can track rejection trends without polluting numeric metric arrays.
        epoch_payload: dict[str, Any] = {
            **current_summary,
            **drift,
            "shadow": shadow_results,
        }
        self._epoch_summaries.append(epoch_payload)
        epoch_summaries_path = Path(self.best_model_path).parent / "epoch_summaries.json"
        epoch_summaries_path.write_text(
            json.dumps(self._epoch_summaries, indent=2, sort_keys=True), encoding="utf-8"
        )


class OutputNoiseRampCallback(keras.callbacks.Callback):
    """Ramp the output noise toward the deployment target over early epochs."""

    def __init__(self, *, start_stddev: float, end_stddev: float, ramp_epochs: int) -> None:
        super().__init__()
        self.start_stddev = float(start_stddev)
        self.end_stddev = float(end_stddev)
        self.ramp_epochs = max(int(ramp_epochs), 1)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Update the model noise level before each epoch."""

        del logs
        if self.ramp_epochs <= 1:
            current = self.end_stddev
        else:
            progress = min(max(epoch, 0), self.ramp_epochs - 1) / float(self.ramp_epochs - 1)
            current = self.start_stddev + progress * (self.end_stddev - self.start_stddev)
        if hasattr(self.model, "output_noise_stddev"):
            self.model.output_noise_stddev = float(current)
        print(f"[V4] output_noise_stddev={current:.6f}", flush=True)


class WarmupLRCallback(keras.callbacks.Callback):
    """Linearly ramp the learning rate from a fraction of the target to the target.

    After the warmup period the LR stays at the target value.  The optimizer's
    base learning rate must be set to the target *before* compile() so that
    this callback can read and scale it.
    """

    def __init__(self, *, warmup_epochs: int, start_lr_fraction: float) -> None:
        super().__init__()
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.start_lr_fraction = float(start_lr_fraction)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Update the learning rate before each epoch."""

        del logs
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return
        # In Keras 3, optimizer.learning_rate is a Variable that supports
        # direct assignment.  Use float() to extract the current value.
        target_lr = float(self.model.optimizer.learning_rate)
        progress = (epoch + 1) / float(self.warmup_epochs)
        warmup_lr = target_lr * (
            self.start_lr_fraction + progress * (1.0 - self.start_lr_fraction)
        )
        self.model.optimizer.learning_rate = warmup_lr
        print(
            f"[WARMUP] epoch={epoch + 1}/{self.warmup_epochs} lr={warmup_lr:.10f}",
            flush=True,
        )


class EarlyCollapseCallback(keras.callbacks.Callback):
    """Abort training if the mean heatmap peak stays below a threshold.

    Monitors the center and tip heatmap peak values after each validation epoch.
    If both means fall below `peak_threshold` for `patience` consecutive epochs,
    the run is terminated early to avoid wasting compute on a collapsed model.
    """

    def __init__(
        self,
        *,
        peak_threshold: float = EARLY_COLLAPSE_PEAK_THRESHOLD,
        patience: int = EARLY_COLLAPSE_PATIENCE,
    ) -> None:
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.patience = max(int(patience), 1)
        self._below_threshold_epochs = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Check heatmap peaks and abort if collapsed for too long."""

        logs = {} if logs is None else logs
        center_peak = float(logs.get("val_center_heatmap_peak_mean", 1.0))
        tip_peak = float(logs.get("val_tip_heatmap_peak_mean", 1.0))
        both_below = center_peak < self.peak_threshold and tip_peak < self.peak_threshold
        if both_below:
            self._below_threshold_epochs += 1
        else:
            self._below_threshold_epochs = 0

        print(
            f"[COLLAPSE] epoch={epoch + 1} center_peak={center_peak:.6f} "
            f"tip_peak={tip_peak:.6f} below_count={self._below_threshold_epochs}/{self.patience}",
            flush=True,
        )

        if self._below_threshold_epochs >= self.patience:
            print(
                f"[COLLAPSE] ABORTING: heatmap peaks below {self.peak_threshold} "
                f"for {self.patience} consecutive epochs — model collapsed.",
                flush=True,
            )
            self.model.stop_training = True


def _weighted_local_offset_loss(
    pred_offset_map: tf.Tensor,
    target_offset_map: tf.Tensor,
    center_weight_mask: tf.Tensor,
    tip_weight_mask: tf.Tensor,
    *,
    tip_weight: float = 1.0,
) -> tf.Tensor:
    """Weighted Huber loss for the local offset map.

    pred_offset_map / target_offset_map: (batch, H, W, 4) with channels
        [center_dx, center_dy, tip_dx, tip_dy].
    center_weight_mask / tip_weight_mask: (batch, H, W, 1) Gaussian weights
        centered at the true keypoint.
    """

    def _huber_1d(diff: tf.Tensor, delta: float = 0.1) -> tf.Tensor:
        """Element-wise Huber loss, preserving input shape."""
        abs_diff = tf.abs(diff)
        quadratic = 0.5 * tf.square(diff)
        linear = delta * abs_diff - 0.5 * delta * delta
        return tf.where(abs_diff <= delta, quadratic, linear)

    center_diff = pred_offset_map[..., :2] - target_offset_map[..., :2]
    tip_diff = pred_offset_map[..., 2:] - target_offset_map[..., 2:]

    center_huber = _huber_1d(center_diff)
    tip_huber = _huber_1d(tip_diff)

    center_loss = tf.reduce_mean(center_huber * center_weight_mask)
    tip_loss = tf.reduce_mean(tip_huber * tip_weight_mask) * tip_weight
    return center_loss + tip_loss


def _axis_simcc_logit_loss(
    axis_logits: tf.Tensor,
    axis_targets: tf.Tensor,
    *,
    tip_weight: float = 2.0,
) -> tf.Tensor:
    """Softmax cross-entropy loss for axis_simcc logit head.

    axis_logits: (batch, 4, 112) with [center_x, center_y, tip_x, tip_y].
    axis_targets: (batch, 4, 112) soft Gaussian targets summing to 1.
    tip_weight: multiplier for tip axis loss vs center axis loss.
    """
    log_probs = tf.nn.log_softmax(axis_logits, axis=-1)  # (batch, 4, 112)
    ce = -tf.reduce_sum(axis_targets * log_probs, axis=-1)  # (batch, 4)

    center_loss = tf.reduce_mean(ce[:, :2])  # center_x + center_y
    tip_loss = tf.reduce_mean(ce[:, 2:]) * tip_weight  # tip_x + tip_y
    return center_loss + tip_loss


class GeometryV3QuantNativeModel(keras.Model):
    """Wrap the base heatmap model with quantization-native training losses."""

    def __init__(
        self,
        *,
        base_model: keras.Model,
        reference_model: keras.Model,
        output_noise_stddev: float,
        distillation_weight: float,
        center_heatmap_weight: float,
        tip_heatmap_weight: float,
        center_coord_weight: float,
        tip_coord_weight: float,
        angle_weight: float,
        temperature_weight: float,
        confidence_weight: float,
        peak_shape_center_weight: float,
        peak_shape_tip_weight: float,
        confidence_floor_weight: float,
        aux_coord_weight: float,
        aux_loss_type: str = "mse",
        peak_target: float,
        confidence_floor: float,
        calibration_candidate: Any,
        temperature_min_celsius: float,
        temperature_max_celsius: float,
        local_offset_loss_weight: float = LOCAL_OFFSET_LOSS_WEIGHT_DEFAULT,
        local_offset_scale_px: float = LOCAL_OFFSET_SCALE_PX_DEFAULT,
        local_offset_sigma_px: float = LOCAL_OFFSET_SIGMA_PX_DEFAULT,
        local_offset_tip_weight: float = LOCAL_OFFSET_TIP_WEIGHT_DEFAULT,
        axis_simcc_loss_weight: float = AXIS_SIMCC_LOSS_WEIGHT_DEFAULT,
        axis_simcc_sigma_bins: float = AXIS_SIMCC_SIGMA_BINS_DEFAULT,
        axis_simcc_tip_weight: float = AXIS_SIMCC_TIP_WEIGHT_DEFAULT,
    ) -> None:
        super().__init__(name="geometry_heatmap_v4_112_quant_native_model")
        self.base_model = base_model
        self.reference_model = reference_model
        self.output_noise_stddev = float(output_noise_stddev)
        self.distillation_weight = float(distillation_weight)
        self.center_heatmap_weight = float(center_heatmap_weight)
        self.tip_heatmap_weight = float(tip_heatmap_weight)
        self.center_coord_weight = float(center_coord_weight)
        self.tip_coord_weight = float(tip_coord_weight)
        self.angle_weight = float(angle_weight)
        self.temperature_weight = float(temperature_weight)
        self.confidence_weight = float(confidence_weight)
        self.peak_shape_center_weight = float(peak_shape_center_weight)
        self.peak_shape_tip_weight = float(peak_shape_tip_weight)
        self.confidence_floor_weight = float(confidence_floor_weight)
        self.aux_coord_weight = float(aux_coord_weight)
        self.aux_loss_type = str(aux_loss_type)
        self.peak_target = float(peak_target)
        self.confidence_floor = float(confidence_floor)
        self.calibration_candidate = calibration_candidate
        self.temperature_min_celsius = float(temperature_min_celsius)
        self.temperature_max_celsius = float(temperature_max_celsius)
        self.local_offset_loss_weight = float(local_offset_loss_weight)
        self.local_offset_scale_px = float(local_offset_scale_px)
        self.local_offset_sigma_px = float(local_offset_sigma_px)
        self.local_offset_tip_weight = float(local_offset_tip_weight)
        self.axis_simcc_loss_weight = float(axis_simcc_loss_weight)
        self.axis_simcc_sigma_bins = float(axis_simcc_sigma_bins)
        self.axis_simcc_tip_weight = float(axis_simcc_tip_weight)
        self._temperature_slope = float(calibration_candidate.params.get("slope", 0.3118859767261175))
        self._temperature_intercept = float(calibration_candidate.params.get("intercept", -33.14101213857672))
        self._cold_angle_degrees = float(calibration_candidate.params.get("cold_angle_degrees", 135.0))

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return the semantic outputs, injecting quantization noise during training."""

        outputs = _as_output_dict(self.base_model(inputs, training=training))
        if training:
            outputs = _quantize_outputs(outputs, noise_stddev=self.output_noise_stddev)
        return outputs

    def quantized_outputs(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        """Return deployment-like outputs without training-time noise."""

        outputs = _as_output_dict(self.base_model(inputs, training=False))
        return _quantize_outputs(outputs, noise_stddev=0.0)

    def _distillation_loss(self, pred: dict[str, tf.Tensor], reference: dict[str, tf.Tensor]) -> tf.Tensor:
        """Match the current outputs against the frozen reference model."""

        center_loss = tf.reduce_mean(tf.square(pred["center_heatmap"] - reference["center_heatmap"]))
        tip_loss = tf.reduce_mean(tf.square(pred["tip_heatmap"] - reference["tip_heatmap"]))
        confidence_loss = tf.reduce_mean(keras.losses.binary_crossentropy(reference["confidence"], pred["confidence"]))
        return center_loss + tip_loss + confidence_loss

    def _supervised_losses(self, x: tf.Tensor, y: dict[str, tf.Tensor], pred: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Compute the deployment-aligned loss stack for one batch."""

        center_heatmap_loss = weighted_center_heatmap_loss(y["center_heatmap"], pred["center_heatmap"])
        tip_heatmap_loss = weighted_tip_heatmap_loss(y["tip_heatmap"], pred["tip_heatmap"])
        true_center_norm = tf.concat(
            [tf.cast(y["true_center_x_norm"], tf.float32), tf.cast(y["true_center_y_norm"], tf.float32)],
            axis=-1,
        )
        true_tip_norm = tf.concat(
            [tf.cast(y["true_tip_x_norm"], tf.float32), tf.cast(y["true_tip_y_norm"], tf.float32)],
            axis=-1,
        )
        pred_center_norm = normalized_softargmax_coordinates_tf(pred["center_heatmap"])
        pred_tip_norm = normalized_softargmax_coordinates_tf(pred["tip_heatmap"])
        center_coord_loss = tf.reduce_mean(tf.square(pred_center_norm - true_center_norm))
        tip_coord_loss = tf.reduce_mean(tf.square(pred_tip_norm - true_tip_norm))
        confidence_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y["confidence"], pred["confidence"]))

        pred_center = pred_center_norm * tf.constant([223.0, 223.0], dtype=tf.float32)
        pred_tip = pred_tip_norm * tf.constant([223.0, 223.0], dtype=tf.float32)
        pred_angle = angle_degrees_from_center_to_tip_tf(pred_center[:, 0], pred_center[:, 1], pred_tip[:, 0], pred_tip[:, 1])
        true_angle = tf.squeeze(tf.cast(y["true_angle_degrees"], tf.float32), axis=-1)
        angle_mask = tf.math.is_finite(true_angle)
        angle_loss = tf.cond(
            tf.reduce_any(angle_mask),
            lambda: circular_angle_loss_tf(tf.boolean_mask(pred_angle, angle_mask), tf.boolean_mask(true_angle, angle_mask)),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        true_temperature = tf.squeeze(tf.cast(y["temperature_c"], tf.float32), axis=-1)
        pred_temperature = temperature_from_coords_tf(
            pred_center[:, 0],
            pred_center[:, 1],
            pred_tip[:, 0],
            pred_tip[:, 1],
            slope=self._temperature_slope,
            intercept=self._temperature_intercept,
            cold_angle_degrees=self._cold_angle_degrees,
        )
        temperature_mask = tf.math.is_finite(true_temperature)
        temperature_loss = tf.cond(
            tf.reduce_any(temperature_mask),
            lambda: normalized_temperature_huber_loss_tf(
                tf.boolean_mask(pred_temperature, temperature_mask),
                tf.boolean_mask(true_temperature, temperature_mask),
                minimum_celsius=self.temperature_min_celsius,
                maximum_celsius=self.temperature_max_celsius,
            ),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        # Peak-shaping losses: penalize heatmap peaks that fall below self.peak_target
        # to discourage the all-zero collapse observed during QAT.
        center_flat = tf.reshape(pred["center_heatmap"], [tf.shape(pred["center_heatmap"])[0], -1])
        tip_flat = tf.reshape(pred["tip_heatmap"], [tf.shape(pred["tip_heatmap"])[0], -1])
        center_peak = tf.reduce_max(center_flat, axis=-1)
        tip_peak = tf.reduce_max(tip_flat, axis=-1)
        peak_shape_center_loss = tf.reduce_mean(
            tf.square(tf.maximum(self.peak_target - center_peak, 0.0))
        )
        peak_shape_tip_loss = tf.reduce_mean(
            tf.square(tf.maximum(self.peak_target - tip_peak, 0.0))
        )

        # Confidence floor: penalize confidence predictions that drop below the floor.
        confidence_flat = tf.squeeze(pred["confidence"], axis=-1)
        confidence_floor_loss = tf.reduce_mean(
            tf.square(tf.maximum(self.confidence_floor - confidence_flat, 0.0))
        )

        # Auxiliary coordinate loss: direct regression from pooled features.
        if "aux_coords" in pred:
            aux_coords_true = tf.concat(
                [
                    tf.cast(y["true_center_x_norm"], tf.float32),
                    tf.cast(y["true_center_y_norm"], tf.float32),
                    tf.cast(y["true_tip_x_norm"], tf.float32),
                    tf.cast(y["true_tip_y_norm"], tf.float32),
                ],
                axis=-1,
            )
            aux_diff = pred["aux_coords"] - aux_coords_true
            if self.aux_loss_type == "huber":
                aux_coord_loss = tf.reduce_mean(tf.keras.losses.huber(aux_coords_true, pred["aux_coords"], delta=0.1))
            else:
                aux_coord_loss = tf.reduce_mean(tf.square(aux_diff))
        else:
            aux_coord_loss = tf.constant(0.0, dtype=tf.float32)

        # Local offset loss: weighted Huber on the per-pixel offset map.
        if "aux_offset_map" in pred and "aux_offset_map" in y:
            local_offset_loss = _weighted_local_offset_loss(
                pred["aux_offset_map"],
                tf.cast(y["aux_offset_map"], tf.float32),
                tf.cast(y["aux_offset_center_weight"], tf.float32),
                tf.cast(y["aux_offset_tip_weight"], tf.float32),
                tip_weight=self.local_offset_tip_weight,
            )
        else:
            local_offset_loss = tf.constant(0.0, dtype=tf.float32)

        # Axis simcc loss: softmax cross-entropy against 1D Gaussian targets.
        if "axis_logits" in pred and "axis_logits_target" in y:
            axis_simcc_loss = _axis_simcc_logit_loss(
                pred["axis_logits"],
                tf.cast(y["axis_logits_target"], tf.float32),
                tip_weight=self.axis_simcc_tip_weight,
            )
        else:
            axis_simcc_loss = tf.constant(0.0, dtype=tf.float32)

        reference = _as_output_dict(self.reference_model(x, training=False))
        distillation_loss = self._distillation_loss(pred, reference)
        total_loss = (
            self.center_heatmap_weight * center_heatmap_loss
            + self.tip_heatmap_weight * tip_heatmap_loss
            + self.center_coord_weight * center_coord_loss
            + self.tip_coord_weight * tip_coord_loss
            + self.angle_weight * angle_loss
            + self.temperature_weight * temperature_loss
            + self.confidence_weight * confidence_loss
            + self.peak_shape_center_weight * peak_shape_center_loss
            + self.peak_shape_tip_weight * peak_shape_tip_loss
            + self.confidence_floor_weight * confidence_floor_loss
            + self.aux_coord_weight * aux_coord_loss
            + self.local_offset_loss_weight * local_offset_loss
            + self.axis_simcc_loss_weight * axis_simcc_loss
            + self.distillation_weight * distillation_loss
        )

        return {
            "total_loss": total_loss,
            "center_heatmap_loss": center_heatmap_loss,
            "tip_heatmap_loss": tip_heatmap_loss,
            "center_coord_loss": center_coord_loss,
            "tip_coord_loss": tip_coord_loss,
            "angle_loss": angle_loss,
            "temperature_loss": temperature_loss,
            "confidence_loss": confidence_loss,
            "aux_coord_loss": aux_coord_loss,
            "local_offset_loss": local_offset_loss,
            "axis_simcc_loss": axis_simcc_loss,
            "peak_shape_center_loss": peak_shape_center_loss,
            "peak_shape_tip_loss": peak_shape_tip_loss,
            "confidence_floor_loss": confidence_floor_loss,
            "distillation_loss": distillation_loss,
        }

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one optimized training step."""

        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            losses = self._supervised_losses(x, y, pred)
        gradients = tape.gradient(losses["total_loss"], self.base_model.trainable_variables)
        clipped_gradients, global_norm = _apply_gradients_with_clipping(
            optimizer=self.optimizer,
            gradients=list(gradients),
            variables=list(self.base_model.trainable_variables),
            clipnorm=1.0,
        )
        results = {name: tf.cast(value, tf.float32) for name, value in losses.items()}
        results["global_gradient_norm"] = tf.cast(global_norm, tf.float32)
        return results

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one validation step without gradient updates."""

        x, y = data
        pred = self.quantized_outputs(x)
        losses = self._supervised_losses(x, y, pred)
        return {name: tf.cast(value, tf.float32) for name, value in losses.items()}


def _set_backbone_trainability(model: keras.Model, *, trainable_last_block: bool) -> None:
    """Freeze the backbone or unfreeze only its last MobileNetV2 block."""

    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            backbone = layer
            break
    else:
        raise RuntimeError("Could not find the nested MobileNetV2 backbone.")

    if not trainable_last_block:
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False
        return

    backbone.trainable = True
    for layer in backbone.layers:
        is_last_block = layer.name.startswith("block_16") or layer.name in {"Conv_1", "Conv_1_bn", "out_relu"}
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = is_last_block


def _top_rejection_reason_string(rows: list[dict[str, Any]]) -> str:
    """Format the most common rejection reasons for reporting."""

    counts: dict[str, int] = {}
    for row in rows:
        if str(row["guardrail_status"]) == "accepted":
            continue
        for reason in str(row["rejection_reasons"]).split(";"):
            if reason and reason != "none":
                counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    return ";".join(f"{reason}:{count}" for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5])


def _evaluate_replay_split(
    model: keras.Model,
    samples: list[HeatmapSample],
    *,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Decode one split and summarize board-replay-style metrics."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    outputs = _as_output_dict(model.predict(x, verbose=0))

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        confidence = float(np.ravel(outputs["confidence"][index])[0])
        decoded = decode_heatmap_geometry_prediction(
            sample,
            outputs["center_heatmap"][index],
            outputs["tip_heatmap"][index],
            confidence,
            calibration_candidate,
            decode_method=decode_method,
            window_size=window_size,
        )
        guarded = apply_geometry_guardrails(decoded, thresholds)
        rows.append(
            {
                "image_path": str(sample.metadata["image_path"]),
                "true_temperature_c": float(sample.metadata["temperature_c"]),
                "guardrail_status": guarded.status,
                "guarded_temperature_c": float(guarded.temperature_c),
                "predicted_temperature_c": float(decoded.predicted_temperature_c_calibrated),
                "predicted_angle_degrees": float(decoded.predicted_angle_degrees),
                "predicted_center_x_224": float(decoded.predicted_center_x_224),
                "predicted_center_y_224": float(decoded.predicted_center_y_224),
                "predicted_tip_x_224": float(decoded.predicted_tip_x_224),
                "predicted_tip_y_224": float(decoded.predicted_tip_y_224),
                "center_heatmap_peak_value": float(decoded.center_heatmap_peak_value),
                "tip_heatmap_peak_value": float(decoded.tip_heatmap_peak_value),
                "center_heatmap_spread_px": float(guarded.quality_features.center_heatmap_spread_px),
                "tip_heatmap_spread_px": float(guarded.quality_features.tip_heatmap_spread_px),
                "confidence": float(confidence),
                "rejection_reasons": ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none",
            }
        )

    accepted = [row for row in rows if str(row["guardrail_status"]) in {"accepted", "clamped"}]
    accepted_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in accepted],
        dtype=np.float64,
    )
    all_errors = np.asarray(
        [abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) for row in rows],
        dtype=np.float64,
    )
    summary: dict[str, float] = {
        "count": float(len(rows)),
        "accepted_count": float(len(accepted)),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(len(accepted) / len(rows)) if rows else math.nan,
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": float(
            sum(
                1
                for row in accepted
                if abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
            )
        ),
        "percentage_under_2c": float(np.mean(all_errors < 2.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_5c": float(np.mean(all_errors < 5.0) * 100.0) if all_errors.size else math.nan,
        "percentage_under_10c": float(np.mean(all_errors < 10.0) * 100.0) if all_errors.size else math.nan,
        "center_mae_px_224": float(
            np.mean(
                [
                    math.hypot(
                        float(row["predicted_center_x_224"]) - float(sample.metadata["center_x_224"]),
                        float(row["predicted_center_y_224"]) - float(sample.metadata["center_y_224"]),
                    )
                    for row, sample in zip(rows, samples, strict=True)
                ]
            )
        ),
        "tip_mae_px_224": float(
            np.mean(
                [
                    math.hypot(
                        float(row["predicted_tip_x_224"]) - float(sample.metadata["tip_x_224"]),
                        float(row["predicted_tip_y_224"]) - float(sample.metadata["tip_y_224"]),
                    )
                    for row, sample in zip(rows, samples, strict=True)
                ]
            )
        ),
        "angle_mae_degrees": float(
            np.mean(
                [
                    abs(float(row["predicted_angle_degrees"]) - float(sample.metadata["angle_degrees"]))
                    for row, sample in zip(rows, samples, strict=True)
                ]
            )
        ),
        "center_heatmap_peak_mean": float(np.mean([float(row["center_heatmap_peak_value"]) for row in rows])),
        "tip_heatmap_peak_mean": float(np.mean([float(row["tip_heatmap_peak_value"]) for row in rows])),
        "center_heatmap_spread_mean": float(np.mean([float(row["center_heatmap_spread_px"]) for row in rows])),
        "tip_heatmap_spread_mean": float(np.mean([float(row["tip_heatmap_spread_px"]) for row in rows])),
        "confidence_mean": float(np.mean([float(row["confidence"]) for row in rows])),
        "guardrail_disagreement_count": float(sum(1 for row in rows if str(row["guardrail_status"]) != "accepted")),
        "top_rejection_reasons": _top_rejection_reason_string(rows),
    }
    return rows, summary


def _run_one_batch_debug(
    *,
    model: GeometryV3QuantNativeModel,
    batch_x: np.ndarray,
    batch_y: dict[str, np.ndarray],
    train_steps: int,
) -> None:
    """Run one-batch diagnostics and an optional fixed-batch optimization smoke."""

    x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
    y = {name: tf.convert_to_tensor(value, dtype=tf.float32) for name, value in batch_y.items()}
    model.output_noise_stddev = NOISE_RAMP_START_STDDEV
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-6))

    with tf.GradientTape() as tape:
        raw_outputs = _as_output_dict(model.base_model(x, training=False))
        quantized_outputs = model.quantized_outputs(x)
        losses = model._supervised_losses(x, y, model(x, training=True))
    gradients = tape.gradient(losses["total_loss"], model.base_model.trainable_variables)
    _print_loss_diagnostics(
        prefix="V3-DEBUG-LOSS",
        losses=losses,
        raw_outputs=raw_outputs,
        quantized_outputs=quantized_outputs,
        gradients=list(gradients),
        variables=list(model.base_model.trainable_variables),
    )

    if train_steps <= 0:
        return

    print(f"[V3-DEBUG-TRAIN] fixed_batch_steps={train_steps}", flush=True)
    first_step_losses: dict[str, float] | None = None
    for step in range(1, train_steps + 1):
        metrics = model.train_step((x, y))
        metric_values = {name: float(tf.cast(value, tf.float32).numpy()) for name, value in metrics.items()}
        if first_step_losses is None:
            first_step_losses = metric_values.copy()
        if step == 1 or step % 5 == 0:
            print(
                "[V3-DEBUG-TRAIN] "
                f"step={step} total_loss={metric_values['total_loss']:.8f} "
                f"center_heatmap_loss={metric_values['center_heatmap_loss']:.8f} "
                f"tip_heatmap_loss={metric_values['tip_heatmap_loss']:.8f} "
                f"center_coord_loss={metric_values['center_coord_loss']:.8f} "
                f"tip_coord_loss={metric_values['tip_coord_loss']:.8f} "
                f"angle_loss={metric_values['angle_loss']:.8f} "
                f"temperature_loss={metric_values['temperature_loss']:.8f} "
                f"confidence_loss={metric_values['confidence_loss']:.8f} "
                f"peak_shape_center_loss={metric_values['peak_shape_center_loss']:.8f} "
                f"peak_shape_tip_loss={metric_values['peak_shape_tip_loss']:.8f} "
                f"confidence_floor_loss={metric_values['confidence_floor_loss']:.8f} "
                f"distillation_loss={metric_values['distillation_loss']:.8f} "
                f"global_gradient_norm={metric_values['global_gradient_norm']:.8f}",
                flush=True,
            )
        if not all(math.isfinite(value) for value in metric_values.values()):
            raise FloatingPointError(f"Non-finite metric encountered at step {step}: {metric_values}")

    with tf.GradientTape() as final_tape:
        final_losses = model._supervised_losses(x, y, model(x, training=True))
    final_gradients = final_tape.gradient(final_losses["total_loss"], model.base_model.trainable_variables)
    final_global_norm = float(
        tf.linalg.global_norm([tf.cast(gradient, tf.float32) for gradient in final_gradients if gradient is not None]).numpy()
        if any(gradient is not None for gradient in final_gradients)
        else 0.0
    )
    final_metrics = {name: float(tf.cast(value, tf.float32).numpy()) for name, value in final_losses.items()}
    final_metrics["global_gradient_norm"] = final_global_norm
    print(
        "[V3-DEBUG-TRAIN] final_check "
        f"total_loss={final_metrics['total_loss']:.8f} "
        f"center_coord_loss={final_metrics['center_coord_loss']:.8f} "
        f"tip_coord_loss={final_metrics['tip_coord_loss']:.8f} "
        f"angle_loss={final_metrics['angle_loss']:.8f} "
        f"temperature_loss={final_metrics['temperature_loss']:.8f} "
        f"global_gradient_norm={final_metrics['global_gradient_norm']:.8f}",
        flush=True,
    )
    if first_step_losses is not None:
        improved = any(final_metrics[name] <= first_step_losses[name] for name in ("center_coord_loss", "tip_coord_loss", "angle_loss", "temperature_loss"))
        print(f"[V3-DEBUG-TRAIN] geometry_loss_improved={improved}", flush=True)
    if not all(math.isfinite(value) for value in final_metrics.values()):
        raise FloatingPointError(f"Non-finite metric encountered in final fixed-batch check: {final_metrics}")


def _write_markdown_report(
    *,
    output_path: Path,
    decode_method: str,
    window_size: int,
    selected_stage: str,
    calibration_name: str,
    frozen_summary: dict[str, float],
    final_summary: dict[str, float],
) -> None:
    """Write a concise training report for the v3 run."""

    lines = [
        "# Geometry Heatmap v4 112x112 Quantization-Native Training",
        "",
        f"- Decoder: {decode_method} w{window_size}",
        f"- Calibration candidate: {calibration_name}",
        f"- Selected stage: {selected_stage}",
        "",
        "## Frozen Stage Val Replay",
        f"- Accepted MAE: {frozen_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {frozen_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {frozen_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(frozen_summary['accepted_gt20_failures'])}",
        f"- Under 2/5/10 C: {frozen_summary['percentage_under_2c']:.2f}% / {frozen_summary['percentage_under_5c']:.2f}% / {frozen_summary['percentage_under_10c']:.2f}%",
        f"- Center MAE px: {frozen_summary['center_mae_px_224']:.4f}",
        f"- Tip MAE px: {frozen_summary['tip_mae_px_224']:.4f}",
        f"- Angle MAE deg: {frozen_summary['angle_mae_degrees']:.4f}",
        "",
        "## Final Val Replay",
        f"- Accepted MAE: {final_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {final_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {final_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(final_summary['accepted_gt20_failures'])}",
        f"- Under 2/5/10 C: {final_summary['percentage_under_2c']:.2f}% / {final_summary['percentage_under_5c']:.2f}% / {final_summary['percentage_under_10c']:.2f}%",
        f"- Center MAE px: {final_summary['center_mae_px_224']:.4f}",
        f"- Tip MAE px: {final_summary['tip_mae_px_224']:.4f}",
        f"- Angle MAE deg: {final_summary['angle_mae_degrees']:.4f}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _transfer_matching_layer_weights(source_model: keras.Model, target_model: keras.Model) -> int:
    """Copy any compatible named weights from one model tree into another."""

    def _flatten_weighted_layers(model: keras.Model) -> list[keras.layers.Layer]:
        """Collect weighted layers in a deterministic depth-first order."""

        flattened: list[keras.layers.Layer] = []
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                flattened.extend(_flatten_weighted_layers(layer))
            if layer.get_weights():
                flattened.append(layer)
        return flattened

    copied_layers = 0
    source_layers = _flatten_weighted_layers(source_model)
    target_layers = _flatten_weighted_layers(target_model)
    used_source_indices: set[int] = set()

    for target_layer in target_layers:
        target_weights = target_layer.get_weights()
        if not target_weights:
            continue

        exact_match_index: int | None = None
        fallback_match_index: int | None = None
        for source_index, source_layer in enumerate(source_layers):
            if source_index in used_source_indices:
                continue
            source_weights = source_layer.get_weights()
            if len(source_weights) != len(target_weights):
                continue
            if any(source_weight.shape != target_weight.shape for source_weight, target_weight in zip(source_weights, target_weights, strict=True)):
                continue
            if source_layer.name == target_layer.name and source_layer.__class__ is target_layer.__class__:
                exact_match_index = source_index
                break
            if fallback_match_index is None and source_layer.__class__ is target_layer.__class__:
                fallback_match_index = source_index

        selected_index = exact_match_index if exact_match_index is not None else fallback_match_index
        if selected_index is None:
            continue
        target_layer.set_weights(source_layers[selected_index].get_weights())
        used_source_indices.add(selected_index)
        copied_layers += 1

    return copied_layers


def _build_model_from_mode(
    *,
    mode: str,
    model_path: Path,
    resume_from: Path | None = None,
    include_aux_coords: bool = False,
    aux_head_size: str = "small",
    aux_head_type: str = "none",
    backbone_alpha: float = 0.35,
) -> keras.Model:
    """Load a source checkpoint, build a fresh ImageNet model, or resume from a v4 checkpoint."""

    if resume_from is not None:
        target_model = keras.models.load_model(str(resume_from))
        _set_backbone_trainability(target_model, trainable_last_block=False)
        return target_model
    if mode == "source_model":
        source_model = load_geometry_heatmap_keras_model(model_path)
        target_model = build_mobilenetv2_geometry_heatmap_v4_112(
            alpha=backbone_alpha, backbone_frozen=True,
            include_aux_coords=include_aux_coords,
            aux_head_size=aux_head_size, aux_head_type=aux_head_type,
        )
        _transfer_matching_layer_weights(source_model, target_model)
        return target_model
    if mode == "imagenet":
        return build_mobilenetv2_geometry_heatmap_v4_112(
            alpha=backbone_alpha, backbone_frozen=True,
            include_aux_coords=include_aux_coords,
            aux_head_size=aux_head_size, aux_head_type=aux_head_type,
        )
    raise ValueError(f"Unknown initialization mode: {mode}")


def main() -> None:
    """Train geometry_heatmap_v4_112 with quantization-native losses."""

    parser = argparse.ArgumentParser(description="Train geometry_heatmap_v4_112 quant-native")
    parser.add_argument("--initialization-mode", choices=("source_model", "imagenet"), default="source_model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frozen-epochs", type=int, default=40)
    parser.add_argument("--unfrozen-epochs", type=int, default=20)
    parser.add_argument("--frozen-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfrozen-learning-rate", type=float, default=5e-6)
    parser.add_argument("--heatmap-size", type=int, default=112)
    # Use a sharper target than the v2/v3 56x56 heatmaps so the 112x112 head
    # learns narrower peaks and a more stable soft-argmax decode.
    parser.add_argument("--sigma-pixels", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-noise-stddev", type=float, default=0.008)
    parser.add_argument("--output-noise-start-stddev", type=float, default=NOISE_RAMP_START_STDDEV)
    parser.add_argument("--output-noise-ramp-epochs", type=int, default=NOISE_RAMP_EPOCHS)
    parser.add_argument("--distillation-weight", type=float, default=LOSS_WEIGHTS["distillation"])
    parser.add_argument("--peak-shape-center-weight", type=float, default=LOSS_WEIGHTS["peak_shape_center"])
    parser.add_argument("--peak-shape-tip-weight", type=float, default=LOSS_WEIGHTS["peak_shape_tip"])
    parser.add_argument("--confidence-floor-weight", type=float, default=LOSS_WEIGHTS["confidence_floor"])
    parser.add_argument("--peak-target", type=float, default=PEAK_TARGET)
    parser.add_argument("--confidence-floor", type=float, default=CONFIDENCE_FLOOR)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-lr-fraction", type=float, default=WARMUP_START_LR_FRACTION)
    parser.add_argument("--early-collapse-peak-threshold", type=float, default=EARLY_COLLAPSE_PEAK_THRESHOLD)
    parser.add_argument("--early-collapse-patience", type=int, default=EARLY_COLLAPSE_PATIENCE)
    parser.add_argument("--debug-one-batch-losses", action="store_true")
    parser.add_argument("--debug-one-batch-train-steps", type=int, default=0)
    parser.add_argument("--debug-canonical-validation-contract", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--include-aux-coords", action="store_true", help="Add auxiliary coordinate regression head")
    parser.add_argument("--aux-coord-weight", type=float, default=LOSS_WEIGHTS["aux_coord"], help="Weight for auxiliary coordinate loss")
    parser.add_argument("--aux-head-size", type=str, choices=("small", "large"), default="small", help="Aux head capacity: small=Dense(64), large=Dense(128)->Dense(64)")
    parser.add_argument("--aux-loss-type", type=str, choices=("mse", "huber"), default="mse", help="Aux coordinate loss type")
    parser.add_argument("--aux-head-type", type=str, choices=("none", "gap", "local_offset", "axis_simcc"), default="none",
                        help="Aux head type: none, gap (GAP-based coords), local_offset (spatial offset map), or axis_simcc (1D axis logits)")
    parser.add_argument("--backbone-alpha", type=float, default=0.35,
                        help="MobileNetV2 width multiplier (default 0.35; 0.5 for larger backbone)")
    parser.add_argument("--local-offset-loss-weight", type=float, default=LOCAL_OFFSET_LOSS_WEIGHT_DEFAULT,
                        help="Weight for local offset map loss")
    parser.add_argument("--local-offset-scale-px", type=float, default=LOCAL_OFFSET_SCALE_PX_DEFAULT,
                        help="Max offset in heatmap pixels represented by tanh [-1,1]")
    parser.add_argument("--local-offset-sigma-px", type=float, default=LOCAL_OFFSET_SIGMA_PX_DEFAULT,
                        help="Gaussian sigma in heatmap pixels for offset loss weighting")
    parser.add_argument("--local-offset-tip-weight", type=float, default=LOCAL_OFFSET_TIP_WEIGHT_DEFAULT,
                        help="Multiplier for tip offset loss relative to center")
    parser.add_argument("--axis-simcc-loss-weight", type=float, default=AXIS_SIMCC_LOSS_WEIGHT_DEFAULT,
                        help="Weight for axis_simcc logit loss")
    parser.add_argument("--axis-simcc-sigma-bins", type=float, default=AXIS_SIMCC_SIGMA_BINS_DEFAULT,
                        help="Gaussian sigma in 112-bin space for axis soft targets")
    parser.add_argument("--axis-simcc-tip-weight", type=float, default=AXIS_SIMCC_TIP_WEIGHT_DEFAULT,
                        help="Multiplier for tip axis loss vs center axis loss")
    parser.add_argument("--inner-celsius-mask", action="store_true",
                        help="Apply inner-Celsius-only mask after crop+resize to exclude outer distractors")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    thresholds_path = _resolve_path(repo_root, args.thresholds_path)
    decoder_path = _resolve_path(repo_root, args.decoder_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with decoder_path.open("r", encoding="utf-8") as handle:
        decoder_payload = json.load(handle)
    decode_method = str(decoder_payload["decode_method"])
    window_size = int(decoder_payload["window_size"])
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected corrected decoder softargmax w3, found {decode_method} w{window_size}.")

    include_local_offset = args.aux_head_type == "local_offset"
    include_axis_simcc = args.aux_head_type == "axis_simcc"

    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    with thresholds_path.open("r", encoding="utf-8") as handle:
        thresholds_payload = json.load(handle)
    selected_thresholds = thresholds_payload["selected_thresholds"]
    thresholds = GeometryGuardrailThresholds(
        center_peak_min=float(selected_thresholds["center_peak_min"]),
        tip_peak_min=float(selected_thresholds["tip_peak_min"]),
        confidence_min=float(selected_thresholds["confidence_min"]),
        max_heatmap_entropy=float(selected_thresholds["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected_thresholds["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(selected_thresholds["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(selected_thresholds["center_tip_distance_ratio_max"]),
        edge_margin_px=float(selected_thresholds["edge_margin_px"]),
        temperature_physical_margin_c=float(selected_thresholds["temperature_physical_range_margin_c"]),
        clamp_temperature_to_physical_range=bool(selected_thresholds["clamp_temperature_to_physical_range"]),
        minimum_celsius=float(selected_thresholds["minimum_celsius"]),
        maximum_celsius=float(selected_thresholds["maximum_celsius"]),
    )

    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    val_examples = select_examples_from_split(examples, split="val")
    train_sequence = GeometryV3Sequence(
        train_examples,
        base_path=repo_root,
        batch_size=args.batch_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        seed=args.seed,
        include_local_offset_map=include_local_offset,
        local_offset_scale_px=args.local_offset_scale_px,
        local_offset_sigma_px=args.local_offset_sigma_px,
        include_axis_simcc_targets=include_axis_simcc,
        axis_simcc_sigma_bins=args.axis_simcc_sigma_bins,
        inner_celsius_mask=args.inner_celsius_mask,
    )
    val_samples = load_split_samples(
        manifest_path,
        repo_root,
        split="val",
        mode=DEFAULT_PREPROCESSING_MODE,
        input_size=224,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        inner_celsius_mask=args.inner_celsius_mask,
    ).samples
    val_x = np.stack([sample.crop_image for sample in val_samples], axis=0).astype(np.float32)
    val_y = _build_targets(
        val_samples,
        include_local_offset_map=include_local_offset,
        local_offset_scale_px=args.local_offset_scale_px,
        local_offset_sigma_px=args.local_offset_sigma_px,
        include_axis_simcc_targets=include_axis_simcc,
        axis_simcc_sigma_bins=args.axis_simcc_sigma_bins,
    )

    if args.debug_canonical_validation_contract:
        validation_contract = {
            "decode_method": decode_method,
            "window_size": window_size,
            "validation_split": "val",
            "validation_count": len(val_samples),
            "validation_preprocessing_mode": str(val_samples[0].metadata.get("preprocessing_mode", "")) if val_samples else "",
            "validation_resize_method": str(val_samples[0].metadata.get("resize_method", "")) if val_samples else "",
            "validation_channel_strategy": str(val_samples[0].metadata.get("channel_strategy", "")) if val_samples else "",
            "validation_normalization": str(val_samples[0].metadata.get("normalization", "")) if val_samples else "",
            "validation_source_kind": str(val_samples[0].metadata.get("source_kind", "board_replay")) if val_samples else "",
            "canonical_loader": "load_split_samples",
            "canonical_preprocess": DEFAULT_PREPROCESSING_MODE,
            "training_validation_scoring": "model(x, training=False) via ReplayMetricCallback",
            "legacy_validation_scoring_used": False,
            "uses_build_board_replay_sample": True,
            "uses_load_heatmap_sample_for_validation": False,
            "selected_board_guardrails_path": str(thresholds_path),
            "calibration_path": str(calibration_json_path),
            "fake_quant_noise_inference": False,
        }
        report_lines = [
        "# Geometry Heatmap v4 112x112 Canonical Trainer Validation Contract",
            "",
            f"- Decoder: {validation_contract['decode_method']} w{validation_contract['window_size']}",
            f"- Validation split: {validation_contract['validation_split']}",
            f"- Validation count: {validation_contract['validation_count']}",
            f"- Validation preprocessing mode: {validation_contract['validation_preprocessing_mode']}",
            f"- Validation resize method: {validation_contract['validation_resize_method']}",
            f"- Validation channel strategy: {validation_contract['validation_channel_strategy']}",
            f"- Validation normalization: {validation_contract['validation_normalization']}",
            f"- Validation source kind: {validation_contract['validation_source_kind']}",
            f"- Canonical loader: {validation_contract['canonical_loader']}",
            f"- Canonical preprocessing: {validation_contract['canonical_preprocess']}",
            f"- Validation scoring path: {validation_contract['training_validation_scoring']}",
            f"- Legacy validation scoring used: {'yes' if validation_contract['legacy_validation_scoring_used'] else 'no'}",
            f"- Uses build_board_replay_sample for validation: {'yes' if validation_contract['uses_build_board_replay_sample'] else 'no'}",
            f"- Uses load_heatmap_sample for validation: {'yes' if validation_contract['uses_load_heatmap_sample_for_validation'] else 'no'}",
            f"- Guardrails path: {validation_contract['selected_board_guardrails_path']}",
            f"- Calibration path: {validation_contract['calibration_path']}",
            f"- Fake quant/noise active at inference: {'yes' if validation_contract['fake_quant_noise_inference'] else 'no'}",
        ]
        validation_contract_path = output_dir / "canonical_validation_contract.json"
        _write_json(validation_contract, validation_contract_path)
        report_path = repo_root / "ml" / "reports" / "geometry_heatmap_v4_112_canonical_trainer_validation_contract.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"[V3 CONTRACT] Wrote {validation_contract_path}", flush=True)
        print(f"[V3 CONTRACT] Wrote {report_path}", flush=True)
        return

    base_model = _build_model_from_mode(
        mode=args.initialization_mode, model_path=model_path, resume_from=args.resume_from,
        include_aux_coords=args.include_aux_coords, aux_head_size=args.aux_head_size,
        aux_head_type=args.aux_head_type, backbone_alpha=args.backbone_alpha,
    )
    reference_model = _build_model_from_mode(
        mode=args.initialization_mode, model_path=model_path,
        include_aux_coords=args.include_aux_coords, aux_head_size=args.aux_head_size,
        aux_head_type=args.aux_head_type, backbone_alpha=args.backbone_alpha,
    )
    reference_model.trainable = False

    qat_model = GeometryV3QuantNativeModel(
        base_model=base_model,
        reference_model=reference_model,
        output_noise_stddev=args.output_noise_stddev,
        distillation_weight=args.distillation_weight,
        center_heatmap_weight=LOSS_WEIGHTS["center_heatmap"],
        tip_heatmap_weight=LOSS_WEIGHTS["tip_heatmap"],
        center_coord_weight=LOSS_WEIGHTS["center_coord"],
        tip_coord_weight=LOSS_WEIGHTS["tip_coord"],
        angle_weight=LOSS_WEIGHTS["angle"],
        temperature_weight=LOSS_WEIGHTS["temperature"],
        confidence_weight=LOSS_WEIGHTS["confidence"],
        peak_shape_center_weight=args.peak_shape_center_weight,
        peak_shape_tip_weight=args.peak_shape_tip_weight,
        confidence_floor_weight=args.confidence_floor_weight,
        aux_coord_weight=args.aux_coord_weight,
        aux_loss_type=args.aux_loss_type,
        peak_target=args.peak_target,
        confidence_floor=args.confidence_floor,
        calibration_candidate=calibration_candidate,
        temperature_min_celsius=float(selected_thresholds["minimum_celsius"]),
        temperature_max_celsius=float(selected_thresholds["maximum_celsius"]),
        local_offset_loss_weight=args.local_offset_loss_weight,
        local_offset_scale_px=args.local_offset_scale_px,
        local_offset_sigma_px=args.local_offset_sigma_px,
        local_offset_tip_weight=args.local_offset_tip_weight,
        axis_simcc_loss_weight=args.axis_simcc_loss_weight,
        axis_simcc_sigma_bins=args.axis_simcc_sigma_bins,
        axis_simcc_tip_weight=args.axis_simcc_tip_weight,
    )
    _set_backbone_trainability(base_model, trainable_last_block=False)

    if args.debug_one_batch_losses or args.debug_one_batch_train_steps > 0:
        debug_x, debug_y = train_sequence[0]
        qat_model.output_noise_stddev = float(args.output_noise_start_stddev)
        _run_one_batch_debug(
            model=qat_model,
            batch_x=debug_x,
            batch_y=debug_y,
            train_steps=args.debug_one_batch_train_steps,
        )
        return

    qat_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.frozen_learning_rate))

    frozen_best_path = output_dir / "model_v4_112_frozen_best.keras"
    frozen_callback = ReplayMetricCallback(
        metric_prefix="v4_replay",
        samples=val_samples,
        inputs=val_x,
        reference_model=reference_model,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
        best_model_path=frozen_best_path,
    )
    frozen_csv_logger = keras.callbacks.CSVLogger(
        str(output_dir / "frozen_training_log.csv"),
        separator=",",
        append=False,
    )
    frozen_history = qat_model.fit(
        train_sequence,
        validation_data=(val_x, val_y),
        epochs=args.frozen_epochs,
        callbacks=[
            OutputNoiseRampCallback(
                start_stddev=args.output_noise_start_stddev,
                end_stddev=args.output_noise_stddev,
                ramp_epochs=args.output_noise_ramp_epochs,
            ),
            WarmupLRCallback(
                warmup_epochs=args.warmup_epochs,
                start_lr_fraction=args.warmup_start_lr_fraction,
            ),
            frozen_callback,
            EarlyCollapseCallback(
                peak_threshold=args.early_collapse_peak_threshold,
                patience=args.early_collapse_patience,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_v4_replay_score",
                mode="min",
                patience=10,
                restore_best_weights=False,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_v4_replay_score",
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            keras.callbacks.TerminateOnNaN(),
            frozen_csv_logger,
        ],
        verbose=2,
    )
    frozen_rows, frozen_summary = _evaluate_replay_split(
        qat_model,
        val_samples,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
    )
    frozen_weights = base_model.get_weights()

    needs_unfreeze = (
        frozen_summary["accepted_mae_c"] > 4.5
        or frozen_summary["accepted_gt20_failures"] > 0
        or frozen_summary["acceptance_rate"] < 0.65
        or frozen_summary["worst_accepted_error_c"] >= 20.0
        or (
            frozen_callback.best_summary is not None
            and frozen_callback.best_summary.get("temperature_delta_mean", math.inf) > 1.0
        )
    )

    final_stage = "frozen"
    final_history = frozen_history.history
    final_rows = frozen_rows
    final_summary = frozen_summary
    selected_best_path = frozen_best_path
    if needs_unfreeze and args.initialization_mode != "imagenet" and args.unfrozen_epochs > 0:
        _set_backbone_trainability(base_model, trainable_last_block=True)
        qat_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.unfrozen_learning_rate))
        unfrozen_best_path = output_dir / "model_v4_112_unfrozen_best.keras"
        unfrozen_callback = ReplayMetricCallback(
        metric_prefix="v4_replay",
            samples=val_samples,
            inputs=val_x,
            reference_model=reference_model,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
            best_model_path=unfrozen_best_path,
        )
        unfrozen_history = qat_model.fit(
            train_sequence,
            validation_data=(val_x, val_y),
            epochs=args.unfrozen_epochs,
            callbacks=[
                OutputNoiseRampCallback(
                    start_stddev=args.output_noise_start_stddev,
                    end_stddev=args.output_noise_stddev,
                    ramp_epochs=args.output_noise_ramp_epochs,
                ),
                WarmupLRCallback(
                    warmup_epochs=args.warmup_epochs,
                    start_lr_fraction=args.warmup_start_lr_fraction,
                ),
                unfrozen_callback,
                EarlyCollapseCallback(
                    peak_threshold=args.early_collapse_peak_threshold,
                    patience=args.early_collapse_patience,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_v4_replay_score",
                    mode="min",
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_v4_replay_score",
                    mode="min",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                ),
                keras.callbacks.TerminateOnNaN(),
            ],
            verbose=2,
        )
        unfrozen_rows, unfrozen_summary = _evaluate_replay_split(
            qat_model,
            val_samples,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        if (
            unfrozen_summary["accepted_gt20_failures"] < final_summary["accepted_gt20_failures"]
            or (
                unfrozen_summary["accepted_gt20_failures"] == final_summary["accepted_gt20_failures"]
                and unfrozen_summary["worst_accepted_error_c"] < final_summary["worst_accepted_error_c"]
            )
            or (
                unfrozen_summary["accepted_gt20_failures"] == final_summary["accepted_gt20_failures"]
                and unfrozen_summary["worst_accepted_error_c"] == final_summary["worst_accepted_error_c"]
                and unfrozen_summary["acceptance_rate"] >= final_summary["acceptance_rate"]
                and unfrozen_summary["accepted_mae_c"] <= final_summary["accepted_mae_c"]
                and float(unfrozen_callback.best_summary.get("temperature_delta_mean", math.inf))
                <= float(frozen_callback.best_summary.get("temperature_delta_mean", math.inf))
            )
        ):
            final_stage = "unfrozen"
            final_history = unfrozen_history.history
            final_rows = unfrozen_rows
            final_summary = unfrozen_summary
            selected_best_path = unfrozen_best_path
        else:
            base_model.set_weights(frozen_weights)

    output_model_path = output_dir / "model_v4_112.keras"
    selected_model_path = output_dir / "model.keras"
    selected_best_model_path = output_dir / "best_model.keras"
    base_model.save(output_model_path)
    shutil.copy2(output_model_path, selected_model_path)
    _copy_if_exists(selected_best_path, selected_best_model_path)
    _write_csv(final_rows, output_dir / "val_predictions.csv")
    _write_history(final_history, output_dir / "history.csv")
    shutil.copy2(output_dir / "history.csv", output_dir / "canonical_validation_history.csv")
    summary_payload = {
        "selected_stage": final_stage,
        "selected_model_path": str(selected_model_path),
        "selected_best_model_path": str(selected_best_model_path),
        "decoder_method": decode_method,
        "window_size": window_size,
        "loss_weights": LOSS_WEIGHTS,
        "output_noise_stddev": args.output_noise_stddev,
        "distillation_weight": args.distillation_weight,
        "peak_shape_center_weight": args.peak_shape_center_weight,
        "peak_shape_tip_weight": args.peak_shape_tip_weight,
        "confidence_floor_weight": args.confidence_floor_weight,
        "peak_target": args.peak_target,
        "confidence_floor": args.confidence_floor,
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_lr_fraction": args.warmup_start_lr_fraction,
        "early_collapse_peak_threshold": args.early_collapse_peak_threshold,
        "early_collapse_patience": args.early_collapse_patience,
        "feasibility": {
            "tfmot_available": importlib.util.find_spec("tensorflow_model_optimization") is not None,
            "preferred_training_strategy": "quantization_native_training",
        },
        "selected_calibration_candidate": calibration_candidate.name,
        "selected_calibration_json": calibration_json,
        "frozen_val_summary": frozen_summary,
        "final_val_summary": final_summary,
    }
    _write_json(summary_payload, output_dir / "summary.json")
    _write_json(summary_payload, output_dir / "canonical_summary.json")
    _write_json(
        {
            "initialization_mode": args.initialization_mode,
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "calibration_json_path": str(calibration_json_path),
            "thresholds_path": str(thresholds_path),
            "decoder_path": str(decoder_path),
            "output_model_path": str(output_model_path),
            "selected_model_path": str(selected_model_path),
            "selected_best_model_path": str(selected_best_model_path),
            "selected_stage": final_stage,
            "decoder_method": decode_method,
            "window_size": window_size,
            "loss_weights": LOSS_WEIGHTS,
            "output_noise_stddev": args.output_noise_stddev,
            "distillation_weight": args.distillation_weight,
            "peak_shape_center_weight": args.peak_shape_center_weight,
            "peak_shape_tip_weight": args.peak_shape_tip_weight,
            "confidence_floor_weight": args.confidence_floor_weight,
            "peak_target": args.peak_target,
            "confidence_floor": args.confidence_floor,
            "warmup_epochs": args.warmup_epochs,
            "warmup_start_lr_fraction": args.warmup_start_lr_fraction,
            "early_collapse_peak_threshold": args.early_collapse_peak_threshold,
            "early_collapse_patience": args.early_collapse_patience,
            "frozen_epochs": args.frozen_epochs,
            "unfrozen_epochs": args.unfrozen_epochs,
            "frozen_learning_rate": args.frozen_learning_rate,
            "unfrozen_learning_rate": args.unfrozen_learning_rate,
        },
        output_dir / "config.json",
    )
    _write_markdown_report(
        output_path=output_dir / "replay_report.md",
        decode_method=decode_method,
        window_size=window_size,
        selected_stage=final_stage,
        calibration_name=calibration_candidate.name,
        frozen_summary=frozen_summary,
        final_summary=final_summary,
    )
    _write_markdown_report(
        output_path=DEFAULT_TRAINING_REPORT_PATH,
        decode_method=decode_method,
        window_size=window_size,
        selected_stage=final_stage,
        calibration_name=calibration_candidate.name,
        frozen_summary=frozen_summary,
        final_summary=final_summary,
    )

    print(f"[V4] Output model: {output_model_path}", flush=True)
    print(f"[V4] Selected stage: {final_stage}", flush=True)
    print(f"[V4] Val accepted MAE: {final_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[V4] Val acceptance rate: {final_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[V4] Val worst accepted error: {final_summary['worst_accepted_error_c']:.4f} C", flush=True)


if __name__ == "__main__":
    main()
