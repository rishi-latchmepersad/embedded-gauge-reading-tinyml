#!/usr/bin/env python3
"""
Unified training script for geometric reading candidates (A/B/C/D).

Supports four architectures for 224x224 gauge keypoint detection:
  A.  simcc          -- QAT-friendly custom CNN + SimCC classification heads
  B.  kd_simcc       -- Same as A, trained with KD from a frozen teacher
  C.  heatmap_dark   -- Custom encoder-decoder + 56x56 heatmap + DARK decoding
  D.  coordconv_direct -- Custom encoder + CoordConv + direct regression

The script manages the full workflow:
  1. Load geometry manifest (pxl + board combined)
  2. Pre-generate targets (SimCC bin distributions or heatmap gaussians)
  3. (Optional) Train teacher for Candidate B
  4. Train student/standalone model with optional KD loss
  5. (Optional) QAT fine-tune phase
  6. Export PTQ int8 and/or QAT int8 TFLite
  7. Evaluate on test split and save metrics

Usage:
    poetry run python scripts/train_geometry_candidates.py \
        --candidate simcc \
        --manifest ml/data/geometry_board_heatmap_manifest_v1.csv \
        --output-dir tmp/geometry_candidate_simcc_v1 \
        --epochs 100 --batch-size 16 --lr 1e-3 \
        --width-multiplier 1.0 \
        --backbone-variant standard

For KD:
    poetry run python scripts/train_geometry_candidates.py \
        --candidate kd_simcc \
        --teacher-path tmp/geometry_candidate_teacher/teacher_float.keras \
        --kd-temperature 3.0 \
        ...

Output per candidate:
    {output_dir}/
        model_float.keras        -- best float32 checkpoint
        model_qat.keras          -- QAT checkpoint (if QAT enabled)
        model_int8.tflite        -- PTQ int8 export
        model_qat_int8.tflite    -- QAT int8 export (if QAT enabled)
        history.json             -- training metrics
        test_metrics.json        -- held-out evaluation
        test_predictions.csv     -- per-sample predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# GPU memory cap must happen before TF is imported.
# ---------------------------------------------------------------------------
_GPU_MEMORY_LIMIT_MB = int(os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3900"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
        )
        print(f"[GPU] Memory limit set to {_GPU_MEMORY_LIMIT_MB} MB")
    except RuntimeError:
        print("[GPU] Could not set memory limit (TF already initialized)")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.models_geometry_v2 import (  # noqa: E402
    build_qat_simcc_model,
    build_teacher_model,
    build_heatmap_dark_model,
    build_coordconv_direct_model,
    build_mobilenetv2_spatial_simcc_model,
    build_spatial_simcc_attn_model,
    build_repvgg_simcc_model,
    build_mobilenetv2_hrnet_eca_simcc_model,
    build_mobilenetv2_unet_heatmap_model,
    build_mobilenetv2_compact_heatmap_model,
    dark_decode_heatmap,
    estimate_peak_int8_activation_bytes,
    export_tflite_int8,
    export_qat_tflite_int8,
    KDStudentWrapper,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (  # noqa: E402
    JitterParams,
    create_jittered_crop,
    generate_jitter_params,
    load_geometry_manifest,
)
from embedded_gauge_reading_tinyml.gauge_geometry import (  # noqa: E402
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for a candidate training run."""
    candidate: str = "simcc"
    manifest_path: str = ""
    output_dir: str = ""
    teacher_path: str = ""
    qat_finetune_steps: int = 0
    qat_lr: float = 1e-5
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_epochs: int = 5
    simcc_bins: int = 112
    simcc_sigma_bins: float = 1.5
    simcc_dense_units: int = 256
    heatmap_size: int = 112
    heatmap_sigma_px: float = 3.0
    width_multiplier: float = 1.0
    backbone_variant: str = "standard"
    dropout_rate: float = 0.15
    kd_temperature: float = 3.0
    kd_loss_weight: float = 0.5
    coordinate_loss_weight: float = 1.0
    tip_weight_multiplier: float = 2.0
    confidence_loss_weight: float = 0.1
    subpixel_loss_weight: float = 0.5
    label_smoothing: float = 0.0
    with_subpixel_refine: bool = False
    augment_jitter_max_shift_px: int = 16
    augment_scale_min: float = 0.92
    augment_scale_max: float = 1.08
    seed: int = 42
    early_stopping_patience: int = 20


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_yuv422_as_rgb(image_path: Path, source_width: int, source_height: int) -> np.ndarray:
    """Decode a packed YUV422 board capture into an RGB array via luma repetition."""
    raw = image_path.read_bytes()
    expected = source_height * (source_width // 2) * 4
    if len(raw) < expected:
        raise ValueError(f"{image_path} is too small for {source_width}x{source_height} YUV422")
    yuyv = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(source_height, source_width // 2, 4)
    luma = np.empty((source_height, source_width), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return np.repeat(luma[:, :, None], 3, axis=2)


def _load_source_image(image_path: str, source_width: int = 0, source_height: int = 0) -> np.ndarray:
    """Load any source image (PNG/JPG/YUV422) as uint8 RGB.

    For YUV422 files, source dimensions must be provided for correct decoding.
    """
    path = Path(image_path)
    if path.suffix.lower() == ".yuv422":
        sw = int(source_width) if source_width > 0 else 224
        sh = int(source_height) if source_height > 0 else 224
        return _load_yuv422_as_rgb(path, sw, sh)
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _crop_and_resize(
    image_array: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    target_size: int = 224,
) -> np.ndarray:
    """Crop an image array and resize to target_size × target_size."""
    crop = Image.fromarray(image_array).crop((x1, y1, x2, y2))
    crop_resized = crop.resize((target_size, target_size), Image.LANCZOS)
    return np.array(crop_resized, dtype=np.float32) / 255.0


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One prepared sample ready for model input."""
    image: np.ndarray
    center_x_224: float
    center_y_224: float
    tip_x_224: float
    tip_y_224: float
    angle_degrees: float
    temperature_c: float
    source_manifest: str


def prepare_samples_from_manifest(
    manifest_path: Path,
    split_filter: str,
    target_size: int = 224,
    augment: bool = False,
    rng: Optional[random.Random] = None,
    max_shift_px: int = 16,
    scale_min: float = 0.92,
    scale_max: float = 1.08,
) -> List[TrainingSample]:
    """Load geometry examples, crop them (OBB or loose crop), produce samples.

    If the manifest row has 'obb_crop_x1' field, use OBB crop; otherwise
    fall back to 'loose_crop_x1'.
    """
    repo_root = manifest_path.parent.parent
    if repo_root.name == "data":
        repo_root = repo_root.parent

    examples = load_geometry_manifest(manifest_path)
    # We also need the raw CSV rows for OBB crop fields (not in SourceGeometryExample).
    raw_rows: dict[str, dict] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_rows[str(row.get("image_path", "")).strip()] = row

    examples = [ex for ex in examples if ex.split == split_filter]

    if rng is None:
        rng = random.Random(42)

    samples: List[TrainingSample] = []

    for i, ex in enumerate(examples):
        image_path = _resolve_path_for_sample(ex.image_path, repo_root)
        if not Path(image_path).exists():
            continue

        try:
            source_img = _load_source_image(
                image_path,
                source_width=ex.source_width,
                source_height=ex.source_height,
            )
        except Exception:
            continue

        raw_row = raw_rows.get(ex.image_path, {})

        # Determine crop type: OBB crop or loose crop.
        use_obb = bool(raw_row.get("obb_crop_x1", "").strip())

        if use_obb:
            crop_x1 = int(float(raw_row["obb_crop_x1"]))
            crop_y1 = int(float(raw_row["obb_crop_y1"]))
            crop_x2 = int(float(raw_row["obb_crop_x2"]))
            crop_y2 = int(float(raw_row["obb_crop_y2"]))
            center_x_224_scale = 224.0 / max(1, crop_x2 - crop_x1)
            center_y_224_scale = 224.0 / max(1, crop_y2 - crop_y1)
            cx_224 = (ex.center_x_source - float(crop_x1)) * center_x_224_scale
            cy_224 = (ex.center_y_source - float(crop_y1)) * center_y_224_scale
            tx_224 = (ex.tip_x_source - float(crop_x1)) * center_x_224_scale
            ty_224 = (ex.tip_y_source - float(crop_y1)) * center_y_224_scale
            accepted = True
        else:
            # Use the existing loose crop logic.
            if augment:
                jitter = generate_jitter_params(
                    rng, max_shift_px=max_shift_px,
                    scale_min=scale_min, scale_max=scale_max,
                )
            else:
                jitter = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)

            crop = create_jittered_crop(ex, jitter)
            if not crop.accepted:
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = (
                crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2,
            )
            cx_224 = crop.center_x_224
            cy_224 = crop.center_y_224
            tx_224 = crop.tip_x_224
            ty_224 = crop.tip_y_224
            accepted = True

        try:
            cropped = _crop_and_resize(
                source_img,
                crop_x1, crop_y1, crop_x2, crop_y2,
                target_size=target_size,
            )
        except Exception:
            continue

        angle = angle_degrees_from_center_to_tip(cx_224, cy_224, tx_224, ty_224)
        temp_c = celsius_from_inner_dial_angle_degrees(angle)

        samples.append(TrainingSample(
            image=cropped,
            center_x_224=cx_224,
            center_y_224=cy_224,
            tip_x_224=tx_224,
            tip_y_224=ty_224,
            angle_degrees=angle,
            temperature_c=temp_c,
            source_manifest=ex.source_manifest,
        ))

    return samples


def _resolve_path_for_sample(raw_path: str, repo_root: Path) -> str:
    """Normalize a manifest image path into an absolute filesystem path."""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "ml":
        return str(repo_root.parent / path)
    return str(repo_root / "ml" / path)


# ---------------------------------------------------------------------------
# Target generation utilities
# ---------------------------------------------------------------------------

def make_simcc_targets(
    sample: TrainingSample,
    simcc_bins: int = 112,
    sigma_bins: float = 1.5,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate SimCC target distributions and sub-pixel offsets.

    Each coordinate is mapped to a bin index in [0, simcc_bins-1].
    The target is a Gaussian distribution centered on the true bin.
    Sub-pixel offsets are the difference between the exact coordinate and
    the bin center, in bin units (range [-0.5, 0.5]).

    Returns:
        Tuple of (simcc_target (4, simcc_bins), confidence_target scalar,
        subpixel_offsets (4,)).
    """
    coords = [
        sample.center_x_224,
        sample.center_y_224,
        sample.tip_x_224,
        sample.tip_y_224,
    ]
    num_keypoints = 4
    target = np.zeros((num_keypoints, simcc_bins), dtype=np.float32)
    subpixel_offsets = np.zeros((num_keypoints,), dtype=np.float32)

    for kp_idx in range(num_keypoints):
        # Map coordinate [0, image_size) → bin index [0, simcc_bins)
        coord = coords[kp_idx]
        bin_float = (coord / float(image_size)) * float(simcc_bins)
        bin_center_float = float(round(bin_float))
        # Sub-pixel offset in bin units: difference from nearest bin center.
        subpixel_offsets[kp_idx] = bin_float - bin_center_float
        # Gaussian smoothing around the true float position.
        bin_indices = np.arange(simcc_bins, dtype=np.float32)
        dist = np.exp(-0.5 * ((bin_indices - bin_float) / sigma_bins) ** 2)
        total = np.sum(dist)
        if total > 0:
            target[kp_idx] = dist / total

    confidence_target = np.array([1.0], dtype=np.float32)
    return target, confidence_target, subpixel_offsets


def make_heatmap_targets(
    sample: TrainingSample,
    heatmap_size: int = 112,
    sigma_px: float = 3.0,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D Gaussian heatmap targets for center and tip.

    Coordinates are scaled from 224-space to heatmap_size-space, then
    a 2D Gaussian is placed at the scaled position.

    Returns:
        (center_heatmap (1, H, W), tip_heatmap (1, H, W), confidence scalar).
    """
    scale = float(heatmap_size) / float(image_size)
    cx = int(round(sample.center_x_224 * scale))
    cy = int(round(sample.center_y_224 * scale))
    tx = int(round(sample.tip_x_224 * scale))
    ty = int(round(sample.tip_y_224 * scale))

    xs = np.arange(heatmap_size, dtype=np.float32)
    ys = np.arange(heatmap_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    def gaussian_2d(mx: int, my: int) -> np.ndarray:
        dist = (xx - mx) ** 2 + (yy - my) ** 2
        return np.exp(-dist / (2.0 * sigma_px ** 2))

    center_hm = gaussian_2d(cx, cy).astype(np.float32)[np.newaxis, :, :]
    tip_hm = gaussian_2d(tx, ty).astype(np.float32)[np.newaxis, :, :]
    conf = np.array([1.0], dtype=np.float32)

    return center_hm, tip_hm, conf


# ---------------------------------------------------------------------------
# Keras Sequence for memory-efficient training
# ---------------------------------------------------------------------------

class GeometrySequence(keras.utils.PyDataset):
    """Keras Sequence that yields batches of (image, targets) tuples.

    The target format depends on candidate_type:
      - "simcc" / "kd_simcc": (simcc_logits (4, bins), confidence [, subpixel_offsets])
      - "heatmap_dark": (center_heatmap, tip_heatmap, confidence)
      - "coordconv_direct": (cx_norm, cy_norm, tx_norm, ty_norm, conf)
    """

    def __init__(
        self,
        samples: List[TrainingSample],
        candidate_type: str,
        batch_size: int = 16,
        simcc_bins: int = 112,
        simcc_sigma_bins: float = 1.5,
        heatmap_size: int = 112,
        heatmap_sigma_px: float = 3.0,
        with_subpixel_refine: bool = False,
        shuffle: bool = True,
        augment: bool = False,
        rng_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._samples = list(samples)
        self._candidate_type = candidate_type
        self._batch_size = batch_size
        self._simcc_bins = simcc_bins
        self._simcc_sigma_bins = simcc_sigma_bins
        self._heatmap_size = heatmap_size
        self._heatmap_sigma_px = heatmap_sigma_px
        self._with_subpixel_refine = with_subpixel_refine
        self._shuffle = shuffle
        self._augment = augment
        self._rng = random.Random(rng_seed)
        self._indices = list(range(len(self._samples)))
        self.on_epoch_end()

    def __len__(self) -> int:
        return max(1, math.ceil(len(self._samples) / self._batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        batch_indices = self._indices[start:end]
        batch_samples = [self._samples[i] for i in batch_indices]

        images = np.stack([s.image for s in batch_samples], axis=0).astype(np.float32)

        if self._candidate_type in ("simcc", "kd_simcc", "mobilenetv2_simcc", "spatial_simcc", "spatial_simcc_attn", "repvgg_simcc", "mnv2_hrnet_eca"):
            simcc_targets = np.zeros(
                (len(batch_samples), 4, self._simcc_bins), dtype=np.float32,
            )
            conf_targets = np.zeros((len(batch_samples), 1), dtype=np.float32)
            subpixel_targets = np.zeros((len(batch_samples), 4), dtype=np.float32)
            for i, s in enumerate(batch_samples):
                simcc_t, conf_t, subp_t = make_simcc_targets(
                    s, simcc_bins=self._simcc_bins,
                    sigma_bins=self._simcc_sigma_bins,
                )
                simcc_targets[i] = simcc_t
                conf_targets[i] = conf_t
                if self._with_subpixel_refine:
                    subpixel_targets[i] = subp_t
            if self._with_subpixel_refine:
                return images, (simcc_targets, conf_targets, subpixel_targets)
            return images, (simcc_targets, conf_targets)

        elif (self._candidate_type == "heatmap_dark"
              or self._candidate_type == "mnv2_unet_heatmap"
              or self._candidate_type == "mnv2_compact_heatmap"):
            center_hms = np.zeros(
                (len(batch_samples), self._heatmap_size, self._heatmap_size, 1),
                dtype=np.float32,
            )
            tip_hms = np.zeros_like(center_hms)
            conf_targets = np.zeros((len(batch_samples), 1), dtype=np.float32)
            is_main_targets = np.ones((len(batch_samples), 1), dtype=np.float32)
            for i, s in enumerate(batch_samples):
                c_hm, t_hm, conf_t = make_heatmap_targets(
                    s, heatmap_size=self._heatmap_size,
                    sigma_px=self._heatmap_sigma_px,
                )
                # make_heatmap_targets returns (1, H, W) → (H, W, 1)
                center_hms[i, ..., 0] = c_hm.squeeze(0)
                tip_hms[i, ..., 0] = t_hm.squeeze(0)
                conf_targets[i] = conf_t
            return images, (center_hms, tip_hms, conf_targets, is_main_targets)

        else:  # coordconv_direct
            targets = np.zeros((len(batch_samples), 5), dtype=np.float32)
            for i, s in enumerate(batch_samples):
                targets[i] = [
                    s.center_x_224 / 224.0,
                    s.center_y_224 / 224.0,
                    s.tip_x_224 / 224.0,
                    s.tip_y_224 / 224.0,
                    1.0,
                ]
            return images, targets

    def on_epoch_end(self) -> None:
        self._indices = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._indices)


# ---------------------------------------------------------------------------
# Custom training loop with optional KD
# ---------------------------------------------------------------------------

def compile_model_for_candidate(
    model: keras.Model,
    candidate_type: str,
    learning_rate: float = 1e-3,
    coordinate_loss_weight: float = 1.0,
    confidence_loss_weight: float = 0.1,
    label_smoothing: float = 0.0,
    tip_weight_multiplier: float = 2.0,
    subpixel_loss_weight: float = 0.5,
) -> keras.Model:
    """Compile a model with candidate-appropriate losses and metrics."""
    if candidate_type in ("simcc", "kd_simcc", "mobilenetv2_simcc", "spatial_simcc", "spatial_simcc_attn", "repvgg_simcc", "mnv2_hrnet_eca"):
        def simcc_loss(y_true, y_pred):
            loss_per_axis = tf.keras.losses.categorical_crossentropy(
                y_true, y_pred, from_logits=True, label_smoothing=label_smoothing,
            )
            weights = tf.constant(
                [[1.0, 1.0, tip_weight_multiplier, tip_weight_multiplier]],
                dtype=tf.float32,
            )
            return tf.reduce_mean(loss_per_axis * weights)

        # Determine correct output name for SimCC logits.
        output_names = list(model.output_names) if hasattr(model, "output_names") else []
        if any("spatial_simcc" in name for name in output_names):
            simcc_output_name = "spatial_simcc_logits"
        elif any("repvgg_simcc" in name for name in output_names):
            simcc_output_name = "repvgg_simcc_logits"
        elif any("hr_simcc" in name for name in output_names):
            simcc_output_name = "hr_simcc_logits"
        elif any("teacher_simcc" in name for name in output_names):
            simcc_output_name = "teacher_simcc_reshape"
        else:
            simcc_output_name = "simcc_head_reshape"

        loss_dict = {
            simcc_output_name: simcc_loss,
            "confidence": keras.losses.BinaryCrossentropy(),
        }
        loss_weight_dict = {
            simcc_output_name: coordinate_loss_weight,
            "confidence": confidence_loss_weight,
        }

        # If model has subpixel refinement (3 outputs), add subpixel loss.
        num_outputs = len(model.outputs)
        if num_outputs >= 3:
            loss_dict["subpixel_offsets_scaled"] = keras.losses.Huber(delta=0.1)
            loss_weight_dict["subpixel_offsets_scaled"] = subpixel_loss_weight

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            loss_weights=loss_weight_dict,
        )

    elif (candidate_type == "heatmap_dark"
          or candidate_type == "mnv2_unet_heatmap"
          or candidate_type == "mnv2_compact_heatmap"):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "center_heatmap": keras.losses.MeanSquaredError(),
                "tip_heatmap": keras.losses.MeanSquaredError(),
                "confidence": keras.losses.BinaryCrossentropy(),
                "is_main_needle": keras.losses.BinaryCrossentropy(),
            },
            loss_weights={
                "center_heatmap": coordinate_loss_weight,
                "tip_heatmap": coordinate_loss_weight * tip_weight_multiplier,
                "confidence": confidence_loss_weight,
                "is_main_needle": 0.5,
            },
        )

    else:  # coordconv_direct
        def direct_loss(y_true, y_pred):
            """Huber on coordinates + BCE on confidence."""
            true_coords = y_true[..., :4]
            pred_coords = y_pred[..., :4]
            true_conf = y_true[..., 4]
            pred_conf = y_pred[..., 4]
            coord_loss = tf.reduce_mean(
                tf.keras.losses.huber(true_coords, pred_coords, delta=0.02),
            )
            conf_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(true_conf, pred_conf),
            )
            return (
                coordinate_loss_weight * coord_loss
                + confidence_loss_weight * conf_loss
            )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=direct_loss,
        )

    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Per-sample evaluation result."""
    center_mae_px: float
    tip_mae_px: float
    angle_mae_deg: float
    temperature_c: float
    temperature_error_c: float


def decode_simcc_prediction(
    simcc_logits: np.ndarray,
    confidence: float,
    simcc_bins: int = 112,
    image_size: int = 224,
    subpixel_offsets: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """Decode SimCC logits (and optional sub-pixel offsets) into 224-space coords.

    Returns: (center_x, center_y, tip_x, tip_y, confidence).
    """
    logits = np.asarray(simcc_logits, dtype=np.float32)
    if logits.ndim == 3:
        logits = logits[0]
    coords_list = []
    for kp_idx in range(4):
        probs = tf.nn.softmax(logits[kp_idx]).numpy()
        bin_indices = np.arange(simcc_bins, dtype=np.float32)
        expected_bin = float(np.sum(probs * bin_indices))
        # Add sub-pixel offset if available.
        offset = 0.0
        if subpixel_offsets is not None:
            offsets_arr = np.asarray(subpixel_offsets, dtype=np.float32)
            if offsets_arr.ndim == 2:
                offsets_arr = offsets_arr[0]
            if len(offsets_arr) > kp_idx:
                offset = float(offsets_arr[kp_idx])
        refined_bin = expected_bin + offset
        coord = (refined_bin / float(simcc_bins)) * float(image_size)
        coords_list.append(float(coord))
    cx, cy, tx, ty = coords_list
    return cx, cy, tx, ty, float(confidence)


def decode_heatmap_prediction(
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    confidence: float,
    heatmap_size: int = 112,
    image_size: int = 224,
    use_dark: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Decode heatmap outputs into 224-space coordinates.

    Args:
        center_heatmap: (1, H, W) or (H, W, 1) float array.
        tip_heatmap: Same shape.
        confidence: Scalar.
        heatmap_size: Spatial size of heatmap.
        image_size: Target image dimension.
        use_dark: If True, use DARK sub-pixel decoding.

    Returns: (center_x, center_y, tip_x, tip_y, confidence).
    """
    scale = float(image_size) / float(heatmap_size)

    c_hm = np.asarray(center_heatmap, dtype=np.float32)
    t_hm = np.asarray(tip_heatmap, dtype=np.float32)

    if c_hm.ndim == 3:
        c_hm = c_hm.squeeze(-1)
    if t_hm.ndim == 3:
        t_hm = t_hm.squeeze(-1)

    if use_dark and c_hm.size > 0:
        cx, cy = dark_decode_heatmap(c_hm)
        tx, ty = dark_decode_heatmap(t_hm)
    else:
        cy_raw, cx_raw = np.unravel_index(np.argmax(c_hm), c_hm.shape)
        ty_raw, tx_raw = np.unravel_index(np.argmax(t_hm), t_hm.shape)
        cx = float(cx_raw)
        cy = float(cy_raw)
        tx = float(tx_raw)
        ty = float(ty_raw)

    return (
        cx * scale, cy * scale,
        tx * scale, ty * scale,
        float(confidence),
    )


def evaluate_model_on_samples(
    model: keras.Model,
    samples: List[TrainingSample],
    candidate_type: str,
    simcc_bins: int = 112,
    heatmap_size: int = 112,
    use_dark: bool = True,
) -> List[EvalResult]:
    """Run model on a list of samples and compute geometry metrics."""
    if not samples:
        return []

    images = np.stack([s.image for s in samples], axis=0).astype(np.float32)
    preds = model.predict(images, batch_size=32, verbose=0)

    results: List[EvalResult] = []

    for i, sample in enumerate(samples):
        if candidate_type in ("simcc", "kd_simcc", "mobilenetv2_simcc", "spatial_simcc", "spatial_simcc_attn", "repvgg_simcc", "mnv2_hrnet_eca"):
            simcc_logits = preds[0][i]
            conf = float(preds[1][i][0]) if len(preds) > 1 else 1.0
            subp_offsets = preds[2][i] if len(preds) >= 3 else None
            cx, cy, tx, ty, _ = decode_simcc_prediction(
                simcc_logits, conf, simcc_bins=simcc_bins,
                subpixel_offsets=subp_offsets,
            )

        elif (candidate_type == "heatmap_dark"
              or candidate_type == "mnv2_unet_heatmap"
              or candidate_type == "mnv2_compact_heatmap"):
            c_hm = preds[0][i]
            t_hm = preds[1][i]
            conf = float(preds[2][i][0]) if len(preds) > 2 else 1.0
            cx, cy, tx, ty, _ = decode_heatmap_prediction(
                c_hm, t_hm, conf,
                heatmap_size=heatmap_size,
                use_dark=use_dark,
            )

        else:  # coordconv_direct
            pred = preds[i]
            cx = float(pred[0]) * 224.0
            cy = float(pred[1]) * 224.0
            tx = float(pred[2]) * 224.0
            ty = float(pred[3]) * 224.0
            conf = float(pred[4])

        center_err = math.sqrt(
            (cx - sample.center_x_224) ** 2 + (cy - sample.center_y_224) ** 2,
        )
        tip_err = math.sqrt(
            (tx - sample.tip_x_224) ** 2 + (ty - sample.tip_y_224) ** 2,
        )
        pred_angle = angle_degrees_from_center_to_tip(cx, cy, tx, ty)
        angle_err = circular_angle_error_degrees(pred_angle, sample.angle_degrees)
        pred_temp = celsius_from_inner_dial_angle_degrees(pred_angle)

        results.append(EvalResult(
            center_mae_px=center_err,
            tip_mae_px=tip_err,
            angle_mae_deg=angle_err,
            temperature_c=pred_temp,
            temperature_error_c=abs(pred_temp - sample.temperature_c),
        ))

    return results


def aggregate_metrics(results: List[EvalResult]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of EvalResult."""
    if not results:
        return {}
    n = len(results)
    return {
        "center_mae_px": sum(r.center_mae_px for r in results) / n,
        "tip_mae_px": sum(r.tip_mae_px for r in results) / n,
        "angle_mae_degrees": sum(r.angle_mae_deg for r in results) / n,
        "temperature_mae_c": sum(r.temperature_error_c for r in results) / n,
        "temperature_rmse_c": math.sqrt(
            sum(r.temperature_error_c ** 2 for r in results) / n,
        ),
        "num_samples": n,
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _lr_schedule(epoch: int, lr: float) -> float:
    """Learning rate schedule: warmup → cosine decay."""
    config = _lr_schedule._config  # type: ignore[attr-defined]
    initial_lr = config["initial_lr"]
    warmup_epochs = config["warmup_epochs"]
    total_epochs = config["total_epochs"]

    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train geometric reading candidates (A/B/C/D)",
    )

    # Candidate selection
    parser.add_argument(
        "--candidate", type=str, required=True,
        choices=["simcc", "kd_simcc", "heatmap_dark", "coordconv_direct", "mobilenetv2_simcc", "spatial_simcc", "spatial_simcc_attn", "repvgg_simcc", "mnv2_hrnet_eca", "mnv2_unet_heatmap", "mnv2_compact_heatmap"],
        help="Which architecture to train.",
    )

    # Paths
    parser.add_argument(
        "--manifest", type=str,
        default="ml/data/geometry_board_heatmap_manifest_v1.csv",
        help="CSV manifest path (relative to repo root).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for model artifacts and metrics.",
    )
    parser.add_argument(
        "--teacher-path", type=str, default="",
        help="Path to pre-trained teacher .keras for KD (Candidates B only).",
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)

    # Architecture hyperparams
    parser.add_argument("--simcc-bins", type=int, default=112)
    parser.add_argument("--simcc-sigma-bins", type=float, default=1.5)
    parser.add_argument("--simcc-dense-units", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=112)
    parser.add_argument("--heatmap-sigma-px", type=float, default=3.0)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--backbone-variant", type=str, default="standard",
        choices=["standard", "inverted", "tiny"],
    )
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--mnv2-alpha", type=float, default=0.75,
                        help="MobileNetV2 alpha for mobilenetv2_simcc candidate.")
    parser.add_argument("--mnv2-frozen", action="store_true",
                        help="Freeze MobileNetV2 backbone (transfer learning).")
    parser.add_argument("--spatial-channels", type=int, default=128,
                        help="Spatial SimCC bottleneck channels.")
    parser.add_argument("--attention-type", type=str, default="eca",
                        choices=["eca", "cbam", "coord", "none"],
                        help="Attention type for HRNet-ECA model.")
    parser.add_argument("--hr-filters", type=int, default=32,
                        help="HRNet-lite high-res branch filters.")

    # Loss weights
    parser.add_argument("--coordinate-loss-weight", type=float, default=1.0)
    parser.add_argument("--tip-weight-multiplier", type=float, default=2.0)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--subpixel-loss-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--with-subpixel-refine", action="store_true",
                        help="Enable sub-pixel refinement head on SimCC models.")

    # KD
    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--kd-loss-weight", type=float, default=0.5)

    # Augmentation
    parser.add_argument("--augment-jitter-max-shift", type=int, default=16)
    parser.add_argument("--augment-scale-min", type=float, default=0.92)
    parser.add_argument("--augment-scale-max", type=float, default=1.08)

    # QAT
    parser.add_argument("--qat-finetune-steps", type=int, default=0,
                        help="If > 0, run QAT fine-tuning for N steps.")
    parser.add_argument("--qat-lr", type=float, default=1e-5)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Action flags
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip final evaluation on test split.")
    parser.add_argument("--no-tflite-export", action="store_true",
                        help="Skip TFLite int8 export.")
    parser.add_argument("--train-teacher", action="store_true",
                        help="Train a teacher model first, then proceed with KD.")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Resolve paths relative to repo root.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    ml_dir = repo_root / "ml"

    manifest_path = repo_root / args.manifest
    if not manifest_path.exists():
        manifest_path = ml_dir / "data" / Path(args.manifest).name

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build config.
    config = TrainConfig(
        candidate=args.candidate,
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        teacher_path=args.teacher_path,
        qat_finetune_steps=args.qat_finetune_steps,
        qat_lr=args.qat_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        simcc_bins=args.simcc_bins,
        simcc_sigma_bins=args.simcc_sigma_bins,
        simcc_dense_units=args.simcc_dense_units,
        heatmap_size=args.heatmap_size,
        heatmap_sigma_px=args.heatmap_sigma_px,
        width_multiplier=args.width_multiplier,
        backbone_variant=args.backbone_variant,
        dropout_rate=args.dropout_rate,
        kd_temperature=args.kd_temperature,
        kd_loss_weight=args.kd_loss_weight,
        coordinate_loss_weight=args.coordinate_loss_weight,
        tip_weight_multiplier=args.tip_weight_multiplier,
        confidence_loss_weight=args.confidence_loss_weight,
        subpixel_loss_weight=args.subpixel_loss_weight,
        label_smoothing=args.label_smoothing,
        with_subpixel_refine=args.with_subpixel_refine,
        augment_jitter_max_shift_px=args.augment_jitter_max_shift,
        augment_scale_min=args.augment_scale_min,
        augment_scale_max=args.augment_scale_max,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Set seeds.
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # -----------------------------------------------------------------------
    # Log configuration.
    # -----------------------------------------------------------------------
    print("=" * 80)
    print(f"Geometric Reading Candidate: {config.candidate}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 80)
    print(f"  Manifest: {config.manifest_path}")
    print(f"  Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.learning_rate}")
    print(f"  Width multiplier: {config.width_multiplier}, Backbone: {config.backbone_variant}")
    print(f"  SimCC bins: {config.simcc_bins}, Sigma: {config.simcc_sigma_bins}")
    print(f"  Tip weight: {config.tip_weight_multiplier}")
    if config.candidate == "kd_simcc":
        print(f"  KD temperature: {config.kd_temperature}, KD loss weight: {config.kd_loss_weight}")
    print(f"  QAT steps: {config.qat_finetune_steps}")
    print(f"  GPU memory: {_GPU_MEMORY_LIMIT_MB} MB")
    print()

    # -----------------------------------------------------------------------
    # Load data.
    # -----------------------------------------------------------------------
    print("[1/6] Loading training data...")
    train_samples = prepare_samples_from_manifest(
        manifest_path, "train", target_size=224, augment=True,
        rng=random.Random(config.seed),
        max_shift_px=config.augment_jitter_max_shift_px,
        scale_min=config.augment_scale_min,
        scale_max=config.augment_scale_max,
    )
    val_samples = prepare_samples_from_manifest(
        manifest_path, "val", target_size=224, augment=False,
    )
    test_samples = prepare_samples_from_manifest(
        manifest_path, "test", target_size=224, augment=False,
    )

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    if len(train_samples) == 0:
        print("ERROR: No training samples loaded. Check manifest and image paths.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build model.
    # -----------------------------------------------------------------------
    print("[2/6] Building model...")
    input_shape = (224, 224, 3)

    if config.candidate in ("simcc", "kd_simcc", "mobilenetv2_simcc", "spatial_simcc", "spatial_simcc_attn", "repvgg_simcc", "mnv2_hrnet_eca"):
        if config.candidate == "mobilenetv2_simcc":
            model = build_teacher_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                alpha=args.mnv2_alpha,
                backbone_frozen=args.mnv2_frozen,
                simcc_dense_units=config.simcc_dense_units,
                dropout_rate=config.dropout_rate,
                model_name="mobilenetv2_simcc_geometry",
            )
        elif config.candidate == "spatial_simcc":
            model = build_mobilenetv2_spatial_simcc_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                spatial_channels=args.spatial_channels,
                alpha=args.mnv2_alpha,
                backbone_frozen=args.mnv2_frozen,
                dropout_rate=config.dropout_rate,
                model_name="spatial_simcc_geometry",
            )
        elif config.candidate == "spatial_simcc_attn":
            model = build_spatial_simcc_attn_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                spatial_channels=args.spatial_channels,
                alpha=args.mnv2_alpha,
                backbone_frozen=args.mnv2_frozen,
                dropout_rate=config.dropout_rate,
                attention_type=args.attention_type,
                model_name="spatial_simcc_attn_geometry",
            )
        elif config.candidate == "repvgg_simcc":
            model = build_repvgg_simcc_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                simcc_dense_units=config.simcc_dense_units,
                dropout_rate=config.dropout_rate,
                model_name="repvgg_simcc_geometry",
            )
        elif config.candidate == "mnv2_hrnet_eca":
            model = build_mobilenetv2_hrnet_eca_simcc_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                simcc_dense_units=config.simcc_dense_units,
                dropout_rate=config.dropout_rate,
                alpha=args.mnv2_alpha,
                backbone_frozen=args.mnv2_frozen,
                hr_filters=args.hr_filters,
                attention_type=args.attention_type,
                model_name="mnv2_hrnet_eca_simcc",
            )
        else:
            model = build_qat_simcc_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                simcc_sigma_bins=config.simcc_sigma_bins,
                width_multiplier=config.width_multiplier,
                backbone_variant=config.backbone_variant,
                simcc_dense_units=config.simcc_dense_units,
                dropout_rate=config.dropout_rate,
                with_subpixel_refine=config.with_subpixel_refine,
                model_name=f"{config.candidate}_geometry",
            )
    elif config.candidate == "mnv2_unet_heatmap":
        model = build_mobilenetv2_unet_heatmap_model(
            input_shape=input_shape,
            heatmap_size=config.heatmap_size,
            alpha=args.mnv2_alpha,
            backbone_frozen=args.mnv2_frozen,
            dropout_rate=config.dropout_rate,
            model_name="mnv2_unet_heatmap",
        )
    elif config.candidate == "mnv2_compact_heatmap":
        model = build_mobilenetv2_compact_heatmap_model(
            input_shape=input_shape,
            heatmap_size=config.heatmap_size,
            alpha=args.mnv2_alpha,
            dropout_rate=config.dropout_rate,
            model_name="mnv2_compact_heatmap",
        )
    elif config.candidate == "heatmap_dark":
        model = build_heatmap_dark_model(
            input_shape=input_shape,
            heatmap_size=config.heatmap_size,
            width_multiplier=config.width_multiplier,
            backbone_variant=config.backbone_variant,
            model_name="heatmap_dark_geometry",
        )
    else:  # coordconv_direct
        model = build_coordconv_direct_model(
            input_shape=input_shape,
            width_multiplier=config.width_multiplier,
            backbone_variant=config.backbone_variant,
            dense_units=config.simcc_dense_units,
            dropout_rate=config.dropout_rate,
        )

    # Activation budget check.
    peak_bytes = estimate_peak_int8_activation_bytes(model, input_shape)
    peak_mb = peak_bytes / (1024 * 1024)
    total_params = model.count_params()
    print(f"  Total params: {total_params:,}")
    print(f"  Peak int8 activation: {peak_bytes:,} bytes ({peak_mb:.2f} MB)")
    if peak_mb > 1.5:
        print("  WARNING: Exceeds 1.5 MB budget!")
    else:
        print("  OK: Under 1.5 MB budget.")

    # -----------------------------------------------------------------------
    # (Optional) Train teacher for KD.
    # -----------------------------------------------------------------------
    teacher_model: Optional[keras.Model] = None
    if config.candidate == "kd_simcc":
        if config.teacher_path and Path(config.teacher_path).exists():
            print("[KD] Loading pre-trained teacher...")
            teacher_model = keras.models.load_model(
                config.teacher_path, compile=False,
            )
            print(f"  Teacher loaded from {config.teacher_path}")
        elif args.train_teacher:
            print("[KD] Training teacher model first...")
            teacher_model = build_teacher_model(
                input_shape=input_shape,
                simcc_bins=config.simcc_bins,
                alpha=1.0,
                backbone_frozen=True,
                simcc_dense_units=512,
                dropout_rate=0.15,
            )
            # Compile teacher directly — it uses different output names
            # (teacher_simcc_reshape, confidence) vs the student names.
            teacher_loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=config.label_smoothing,
            )
            teacher_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    "teacher_simcc_reshape": teacher_loss,
                    "confidence": keras.losses.BinaryCrossentropy(),
                },
                loss_weights={
                    "teacher_simcc_reshape": config.coordinate_loss_weight,
                    "confidence": config.confidence_loss_weight,
                },
            )
            print(f"  Teacher params: {teacher_model.count_params():,}")
            # Train teacher on full data.
            teacher_train_seq = GeometrySequence(
                train_samples + val_samples, "simcc",
                batch_size=config.batch_size,
                simcc_bins=config.simcc_bins,
                simcc_sigma_bins=config.simcc_sigma_bins,
                shuffle=True, augment=True,
            )
            teacher_model.fit(
                teacher_train_seq,
                epochs=min(50, config.epochs // 2),
                verbose=1,
            )
            teacher_path = Path(config.output_dir) / "teacher_float.keras"
            teacher_model.save(str(teacher_path))
            print(f"  Teacher saved to {teacher_path}")
        else:
            print("ERROR: KD candidate requires --teacher-path or --train-teacher.")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Compile & train.
    # -----------------------------------------------------------------------
    print("[3/6] Compiling and training...")

    compile_model_for_candidate(
        model,
        candidate_type=config.candidate if config.candidate != "kd_simcc" else "simcc",
        learning_rate=config.learning_rate,
        coordinate_loss_weight=config.coordinate_loss_weight,
        confidence_loss_weight=config.confidence_loss_weight,
        label_smoothing=config.label_smoothing,
        tip_weight_multiplier=config.tip_weight_multiplier,
        subpixel_loss_weight=config.subpixel_loss_weight,
    )

    train_seq = GeometrySequence(
        train_samples, config.candidate if config.candidate != "kd_simcc" else "simcc",
        batch_size=config.batch_size,
        simcc_bins=config.simcc_bins,
        simcc_sigma_bins=config.simcc_sigma_bins,
        heatmap_size=config.heatmap_size,
        heatmap_sigma_px=config.heatmap_sigma_px,
        with_subpixel_refine=config.with_subpixel_refine,
        shuffle=True, augment=True,
    )
    val_seq = GeometrySequence(
        val_samples, config.candidate if config.candidate != "kd_simcc" else "simcc",
        batch_size=config.batch_size,
        simcc_bins=config.simcc_bins,
        simcc_sigma_bins=config.simcc_sigma_bins,
        heatmap_size=config.heatmap_size,
        heatmap_sigma_px=config.heatmap_sigma_px,
        with_subpixel_refine=config.with_subpixel_refine,
        shuffle=False, augment=False,
    )

    # For KD candidate, wrap student in KDStudentWrapper.
    fit_model = model
    if config.candidate == "kd_simcc" and teacher_model is not None:
        print("[KD] Wrapping student with KDStudentWrapper for distillation...")
        fit_model = KDStudentWrapper(
            student_model=model,
            teacher_model=teacher_model,
            simcc_output_name="simcc_head_reshape",
            teacher_simcc_name="teacher_simcc_reshape",
            kd_temperature=config.kd_temperature,
            kd_weight=config.kd_loss_weight,
        )
        fit_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=model.loss,
            loss_weights=model.loss_weights,
        )

    # Setup LR schedule config for the callback.
    _lr_schedule._config = {  # type: ignore[attr-defined]
        "initial_lr": config.learning_rate,
        "warmup_epochs": config.warmup_epochs,
        "total_epochs": config.epochs,
    }

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = fit_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save float checkpoint.
    float_path = Path(config.output_dir) / "model_float.keras"
    model.save(str(float_path))
    print(f"  Float model saved to {float_path}")

    # Save history.
    history_path = Path(config.output_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)
    print(f"  Training history saved to {history_path}")

    # -----------------------------------------------------------------------
    # (Optional) QAT fine-tune.
    # -----------------------------------------------------------------------
    if config.qat_finetune_steps > 0:
        print("[4/6] QAT fine-tuning...")
        try:
            import tensorflow_model_optimization as tfmot
            qat_model = tfmot.quantization.keras.quantize_model(model)
            qat_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.qat_lr),
                loss=model.loss,
            )
            qat_model.fit(
                train_seq,
                epochs=1,
                steps_per_epoch=config.qat_finetune_steps,
                validation_data=val_seq,
                validation_steps=max(1, len(val_seq) // 4),
                verbose=1,
            )
            qat_path = Path(config.output_dir) / "model_qat.keras"
            qat_model.save(str(qat_path))
            print(f"  QAT model saved to {qat_path}")

            # Export QAT int8.
            if not args.no_tflite_export:
                qat_tflite_path = Path(config.output_dir) / "model_qat_int8.tflite"
                # Gather representative data.
                rep_data = []
                for batch_idx in range(min(4, len(train_seq))):
                    batch_x, _ = train_seq[batch_idx]
                    rep_data.append(batch_x)
                export_qat_tflite_int8(qat_model, str(qat_tflite_path), representative_data=rep_data)
                print(f"  QAT int8 TFLite exported to {qat_tflite_path}")
        except Exception as exc:
            print(f"  QAT failed: {exc}")
            print("  Skipping QAT, proceeding with PTQ export.")
    else:
        step_label = 4
        print("[4/6] Skipping QAT fine-tune (--qat-finetune-steps=0).")

    # -----------------------------------------------------------------------
    # Export PTQ int8 TFLite.
    # -----------------------------------------------------------------------
    step_label = 5
    if not args.no_tflite_export:
        print(f"[{step_label}/6] Exporting PTQ int8 TFLite...")
        rep_data = []
        for batch_idx in range(min(8, len(train_seq))):
            batch_x, _ = train_seq[batch_idx]
            rep_data.append(batch_x)
        tflite_path = Path(config.output_dir) / "model_int8.tflite"
        try:
            export_tflite_int8(model, str(tflite_path), representative_data=rep_data)
            size_kb = tflite_path.stat().st_size / 1024
            print(f"  TFLite exported: {tflite_path} ({size_kb:.1f} KB)")
        except Exception as exc:
            print(f"  PTQ export failed: {exc}")
            print("  Trying fallback with float16...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            tflite_path.write_bytes(tflite_model)
            size_kb = len(tflite_model) / 1024
            print(f"  TFLite (float16 fallback) exported: {tflite_path} ({size_kb:.1f} KB)")
    else:
        print(f"[{step_label}/6] Skipping TFLite export.")

    # -----------------------------------------------------------------------
    # Evaluate on test split.
    # -----------------------------------------------------------------------
    print("[6/6] Evaluating on test split...")
    if not args.skip_eval and len(test_samples) > 0:
        test_results = evaluate_model_on_samples(
            model, test_samples, candidate_type=config.candidate,
            simcc_bins=config.simcc_bins,
            heatmap_size=config.heatmap_size,
            use_dark=True,
        )
        metrics = aggregate_metrics(test_results)

        print(f"\n  Test metrics ({len(test_results)} samples):")
        print(f"    Center MAE: {metrics.get('center_mae_px', 0):.2f} px")
        print(f"    Tip MAE: {metrics.get('tip_mae_px', 0):.2f} px")
        print(f"    Angle MAE: {metrics.get('angle_mae_degrees', 0):.2f} deg")
        print(f"    Temperature MAE: {metrics.get('temperature_mae_c', 0):.2f} C")
        print(f"    Temperature RMSE: {metrics.get('temperature_rmse_c', 0):.2f} C")

        # Save metrics.
        metrics_path = Path(config.output_dir) / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

        # Save per-sample predictions.
        pred_path = Path(config.output_dir) / "test_predictions.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sample_idx", "center_mae_px", "tip_mae_px",
                "angle_mae_deg", "temperature_error_c",
            ])
            writer.writeheader()
            for i, r in enumerate(test_results):
                writer.writerow({
                    "sample_idx": i,
                    "center_mae_px": f"{r.center_mae_px:.2f}",
                    "tip_mae_px": f"{r.tip_mae_px:.2f}",
                    "angle_mae_deg": f"{r.angle_mae_deg:.2f}",
                    "temperature_error_c": f"{r.temperature_error_c:.2f}",
                })
        print(f"  Predictions saved to {pred_path}")

    # -----------------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Training complete. Outputs in: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
