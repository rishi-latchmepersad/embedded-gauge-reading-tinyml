"""Micro-EdgeVGG (QARepVGG-style) joint OBB + center localizer with QAT.

Architecture: scaled-down RepVGG-A0 with reparameterizable multi-branch blocks.
              Training: 3×3 + 1×1 + identity branches for each block.
              Inference: all branches fused to single 3×3 conv (reparameterized).
              QAT applied after reparameterization for clean INT8 graph.

Target: <2.5 MB int8 TFLite, <10 px center MAE at 320×320.

Usage:
  cd ml && poetry run python scripts/train_qat_micro_edgevgg.py
"""

from __future__ import annotations

import os as _os
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3900"))

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
    )
del _os, _GPU_MEMORY_LIMIT_MB

import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import (
    TrainingExample,
    _compute_edge_weights,
    _compute_fullframe_obb_params,
)

# --- Paths and constants (same as V2 script) ---------------------------------
ML_ROOT: Path = PROJECT_ROOT
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
MANIFEST_PATH: Path = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"
AI_CENTERS_CSV: Path = ML_ROOT / "data" / "ai_annotated_board_captures.csv"
ANNOTATE_30_CSV: Path = ML_ROOT / "data" / "annotate_30" / "annotations.csv"
ANNOTATE_BATCH2_CSV: Path = ML_ROOT / "data" / "annotate_batch2" / "annotations_batch2.csv"
CAPTURED_IMAGES_DIR: Path = ML_ROOT / "data" / "captured_images_320"
YUV_LABELS_CSV: Path = ML_ROOT / "data" / "capture_2026-06-07_labels_v2.csv"

GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO: float = 0.3076

IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
SEED: int = 42
BATCH_SIZE: int = 8
EPOCHS_WARMUP: int = 60
EPOCHS_QAT: int = 20
LEARNING_RATE: float = 1e-3
QAT_LEARNING_RATE: float = 5e-5
VAL_FRACTION: float = 0.20
CENTER_LOSS_WEIGHT: float = 3.0

POSITIONAL_AUG_CY_MIN: float = 0.15
POSITIONAL_AUG_CY_MAX: float = 0.95
POSITIONAL_AUG_CX_MIN: float = 0.15
POSITIONAL_AUG_CX_MAX: float = 0.85


# ═════════════════════════════════════════════════════════════════════════════
# RepVGG / QARepVGG block
# ═════════════════════════════════════════════════════════════════════════════


class _RepVGGBlock(keras.layers.Layer):
    """RepVGG multi-branch block: 3×3 conv + 1×1 conv + identity.

    During training all three branches are active.
    Call ``reparameterize()`` after training to fuse into a single 3×3 conv.
    """

    def __init__(self, filters: int, stride: int = 1,
                 use_identity: bool = True, name: str = "", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.stride = stride
        self.use_identity = use_identity and stride == 1

    def build(self, input_shape: tuple[int, ...]) -> None:
        in_ch = input_shape[-1]

        # 3×3 branch
        self.conv_3x3 = keras.layers.Conv2D(
            self.filters, 3, strides=self.stride, padding="same",
            use_bias=False, name="conv_3x3",
        )
        self.bn_3x3 = keras.layers.BatchNormalization(name="bn_3x3")

        # 1×1 branch
        self.conv_1x1 = keras.layers.Conv2D(
            self.filters, 1, strides=self.stride, padding="same",
            use_bias=False, name="conv_1x1",
        )
        self.bn_1x1 = keras.layers.BatchNormalization(name="bn_1x1")

        # Identity branch (BN only, acts as a learned scaling)
        if self.use_identity:
            self.bn_id = keras.layers.BatchNormalization(name="bn_id")
        else:
            self.bn_id = None

        self.relu = keras.layers.ReLU(name="relu")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Forward pass: sum of three branches."""
        y_3x3 = self.bn_3x3(self.conv_3x3(x), training=training)

        if self.stride == 1:
            y_1x1 = self.bn_1x1(self.conv_1x1(x), training=training)
        else:
            # For stride > 1, 1×1 conv handles stride directly
            y_1x1 = self.bn_1x1(self.conv_1x1(x), training=training)

        if self.bn_id is not None:
            y_id = self.bn_id(x, training=training)
            y = y_3x3 + y_1x1 + y_id
        else:
            y = y_3x3 + y_1x1

        return self.relu(y)

    def reparameterize(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Fuse all branches into a single 3×3 kernel + bias.

        Returns:
            (kernel, bias) for a Conv2D(filters, 3, strides=stride, padding='same').
        """
        # Fuse 3×3 Conv → BN
        k3, b3 = self._fuse_conv_bn(self.conv_3x3, self.bn_3x3)

        # Fuse 1×1 Conv → BN, then pad from 1×1 to 3×3
        k1, b1 = self._fuse_conv_bn(self.conv_1x1, self.bn_1x1)
        k1 = self._pad_kernel_1x1_to_3x3(k1)

        if self.bn_id is not None:
            # Identity branch → 3×3 kernel with 1 at center
            kid, bid = self._fuse_identity_bn(self.bn_id, self.filters)
        else:
            kid, bid = tf.zeros_like(k3), tf.zeros_like(b3)

        fused_k = k3 + k1 + kid
        fused_b = b3 + b1 + bid
        return fused_k, fused_b

    def _fuse_conv_bn(self, conv: keras.layers.Conv2D,
                       bn: keras.layers.BatchNormalization) -> tuple[tf.Tensor, tf.Tensor]:
        """Fuse Conv2D → BatchNorm into a single kernel + bias."""
        gamma = bn.gamma
        beta = bn.beta
        mean = bn.moving_mean
        var = bn.moving_variance
        eps = bn.epsilon

        w = conv.kernel  # [KH, KW, C_in, C_out]
        std = tf.sqrt(var + eps)
        w_fused = w * (gamma / std)  # broadcast over kernel dims

        b_fused = beta - (gamma * mean / std)
        # If conv has bias, add it
        if conv.use_bias and conv.bias is not None:
            b_fused = b_fused + conv.bias

        return w_fused, b_fused

    def _pad_kernel_1x1_to_3x3(self, k: tf.Tensor) -> tf.Tensor:
        """Pad 1×1 kernel to 3×3."""
        # k shape: [1, 1, C_in, C_out] → [3, 3, C_in, C_out]
        return tf.pad(k, [[1, 1], [1, 1], [0, 0], [0, 0]])

    def _fuse_identity_bn(self, bn: keras.layers.BatchNormalization,
                           filters: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Convert identity BN to 3×3 kernel (1 at center, 0 elsewhere)."""
        gamma = bn.gamma
        beta = bn.beta
        mean = bn.moving_mean
        var = bn.moving_variance
        eps = bn.epsilon

        std = tf.sqrt(var + eps)
        w_scale = gamma / std  # [filters]
        b = beta - (gamma * mean / std)  # [filters]

        # Build identity kernel: 1 at center (1, 1), 0 elsewhere
        # Shape: [3, 3, filters, filters]
        w_id = tf.eye(filters, batch_shape=[3, 3])  # This creates [3, 3, filters, filters]
        # Actually, tf.eye with batch_shape creates a batched identity matrix
        # Let me construct it differently
        w_id = tf.zeros([3, 3, filters, filters], dtype=w_scale.dtype)
        center = tf.eye(filters, dtype=w_scale.dtype)[tf.newaxis, :, :]  # [1, filters, filters]
        w_id = tf.tensor_scatter_nd_update(w_id, [[1, 1]], center)

        w_fused = w_id * w_scale[tf.newaxis, tf.newaxis, tf.newaxis, :]
        return w_fused, b

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "use_identity": self.use_identity,
        })
        return config


def _build_reparam_model(image_height: int, image_width: int,
                          blocks_trained: _RepVGGBlock | None = None
                          ) -> tuple[keras.Model, list[_RepVGGBlock]]:
    """Build the multi-branch Micro-EdgeVGG and return (model, list of RepVGG blocks).

    RepVGG blocks are returned so the caller can call .reparameterize() later.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")
    x = inputs

    # --- Stem ---
    x = keras.layers.Conv2D(48, 3, strides=2, padding="same",
                             use_bias=False, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    repvgg_blocks: list[_RepVGGBlock] = []

    # --- Stages (scaled RepVGG-A0) ---
    # Config: (filters, num_blocks, stride_first)
    stages = [
        (48, 2, 1),    # Stage 1: 48 ch, 2 blocks, stride 1
        (96, 3, 2),    # Stage 2: 96 ch, 3 blocks, first stride 2
        (160, 4, 2),   # Stage 3: 160 ch, 4 blocks, first stride 2
        (320, 1, 2),   # Stage 4: 320 ch, 1 block, stride 2
    ]

    for stage_idx, (filters, num_blocks, first_stride) in enumerate(stages):
        for i in range(num_blocks):
            stride = first_stride if i == 0 else 1
            use_id = (stride == 1)
            block = _RepVGGBlock(
                filters, stride=stride, use_identity=use_id,
                name=f"stage{stage_idx}_block{i}",
            )
            x = block(x)
            repvgg_blocks.append(block)

    # --- Head: GAP + Dense ---
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(256, name="head_fc1")(x)
    x = keras.layers.ReLU(name="head_relu")(x)
    x = keras.layers.Dense(128, name="head_fc2")(x)
    x = keras.layers.ReLU(name="head_relu2")(x)
    x = keras.layers.Dropout(0.30, name="head_dropout")(x)

    obb_params = keras.layers.Dense(6, name="obb_params")(x)
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(
        inputs, {"obb_params": obb_params, "center_xy": center_xy},
        name="micro_edgevgg",
    )
    model._is_graph_network = True
    return model, repvgg_blocks


def _reparameterize_model(model: keras.Model,
                           repvgg_blocks: list[_RepVGGBlock]) -> keras.Model:
    """Convert multi-branch RepVGG blocks to single 3×3 convs.

    Builds a new model with single-branch 3×3 Conv2D layers using the
    fused weights from reparameterization. Preserves multi-output heads.
    """
    inputs = model.input
    x = inputs

    # Find the layer just before the two parallel Dense heads
    shared_feat = None
    obb_dense = None
    center_dense = None
    for layer in model.layers:
        if layer.name == "obb_params":
            obb_dense = layer
        elif layer.name == "center_xy":
            center_dense = layer
        elif layer.name == "head_dropout":
            shared_feat = layer   # last shared layer before heads

    # Walk through layers up to and including shared_feat
    for layer in model.layers:
        if isinstance(layer, _RepVGGBlock):
            k, b = layer.reparameterize()
            filters = layer.filters
            stride = layer.stride
            new_conv = keras.layers.Conv2D(
                filters, 3, strides=stride, padding="same",
                use_bias=True, name=f"{layer.name}_reparam",
            )
            _ = new_conv(x)
            new_conv.kernel = k
            if new_conv.bias is not None:
                new_conv.bias = b
            new_conv.trainable = False
            x = new_conv(x)
            x = keras.layers.ReLU(name=f"{layer.name}_relu")(x)
        else:
            x = layer(x)
            if layer is shared_feat:
                break   # stop after shared_feat; heads built fresh below

    # Rebuild the two parallel heads on the shared features
    obb_params = keras.layers.Dense(6, name="obb_params")(x)
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)
    # Copy weights from original heads
    if obb_dense is not None:
        obb_params.kernel = obb_dense.kernel
        if obb_params.bias is not None and obb_dense.bias is not None:
            obb_params.bias = obb_dense.bias
    if center_dense is not None:
        center_xy.kernel = center_dense.kernel
        if center_xy.bias is not None and center_dense.bias is not None:
            center_xy.bias = center_dense.bias

    reparam_model = keras.Model(
        inputs, {"obb_params": obb_params, "center_xy": center_xy},
        name="micro_edgevgg_reparam",
    )
    reparam_model._is_graph_network = True
    return reparam_model


# --- Image preprocessing (same as V2) ----------------------------------------

def _preprocess_colour(image: tf.Tensor, image_height: int, image_width: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    crop_h = tf.shape(image)[0]
    crop_w = tf.shape(image)[1]
    scale = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(image_width, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32)
    scaled_w = tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32)
    scaled_h = tf.maximum(scaled_h, 1)
    scaled_w = tf.maximum(scaled_w, 1)
    resized = tf.image.resize(image, [scaled_h, scaled_w], method="nearest")
    pad_y = (image_height - scaled_h) // 2
    pad_x = (image_width - scaled_w) // 2
    pad_bottom = image_height - scaled_h - pad_y
    pad_right = image_width - scaled_w - pad_x
    padded = tf.pad(resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]])
    padded = tf.ensure_shape(padded, [image_height, image_width, 3])
    return padded / 255.0


# --- Data loading (same as V2) -----------------------------------------------

def _is_original_capture(filename: str) -> bool:
    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True


def _load_fullframe_obb_data_colour(
    image_path: tf.Tensor, value: tf.Tensor, obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor, image_height: int, image_width: int, weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    is_yuv = tf.strings.regex_full_match(image_path, ".*\\.yuv422$")

    def _load_yuv():
        raw = tf.io.read_file(image_path)
        yuyv = tf.io.decode_raw(raw, tf.uint8)
        yuyv = tf.reshape(yuyv, [320, 640])
        y = tf.cast(yuyv[:, 0::2], tf.float32)
        u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
        v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
        u = tf.repeat(u, 2, axis=1)
        v = tf.repeat(v, 2, axis=1)
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        rgb = tf.stack([r, g, b], axis=-1)
        rgb = tf.clip_by_value(rgb, 0, 255)
        return tf.cast(rgb, tf.uint8)

    def _load_standard():
        return tf.io.decode_image(
            tf.io.read_file(image_path), channels=3, expand_animations=False,
        )

    image = tf.cond(is_yuv, _load_yuv, _load_standard)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, image_height, image_width)
    obb_target = tf.cast(obb_params, tf.float32)
    center_target = obb_target[:2]
    sample_weight = tf.cast(weight, tf.float32)
    return (
        image,
        {"obb_params": obb_target, "center_xy": center_target},
        {"obb_params": sample_weight, "center_xy": sample_weight * CENTER_LOSS_WEIGHT},
    )


def _obb_params_from_center_224(cx_224: float, cy_224: float, source_size: int = 224) -> np.ndarray:
    radius = source_size * GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
    return _compute_fullframe_obb_params(
        source_size, source_size, cx_224, cy_224,
        radius, radius, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH,
    )


def _load_manifest_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "captured_images" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            fpath = fpath.replace("/captured_images/", "/captured_images_320/")
            if fpath.endswith(".jpg"):
                fpath = fpath[:-4] + ".png"
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)
            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue
            obb_params = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_pxl_photo_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "PXL_" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)
            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue
            obb_params = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_annotate_csv(
    csv_path: Path, examples: list[TrainingExample], seen: set[str],
    *, source_size: int = 224,
) -> int:
    added = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if not _is_original_capture(fname):
                continue
            fpath = str(CAPTURED_IMAGES_DIR / fname)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)
            cx = float(row["cx"])
            cy = float(row["cy"])
            obb_params = _obb_params_from_center_224(cx, cy, source_size)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, float(source_size), float(source_size)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_ai_annotated_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    added = 0
    if not AI_CENTERS_CSV.exists():
        return 0
    with open(AI_CENTERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["image_path"]
            if not _is_original_capture(rel_path):
                continue
            rel_path = rel_path.replace("captured_images/", "captured_images_320/")
            fpath = str(ML_ROOT / "data" / rel_path)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            obb_params = _obb_params_from_center_224(cx, cy, 224)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_yuv_board_captures(examples: list[TrainingExample], seen: set[str]) -> int:
    added = 0
    if not YUV_LABELS_CSV.exists():
        return 0
    with open(YUV_LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yuv_file = row["image_path"]
            fpath = str(CAPTURED_IMAGES_DIR / yuv_file)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            outer_r = float(row["outer_radius"])
            obb_params = _compute_fullframe_obb_params(
                320, 320, cx, cy, outer_r, outer_r, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 320.0, 320.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _build_all_examples() -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    n1 = _load_pxl_photo_examples(examples, seen)
    print(f"  [1] PXL phone photos:         {n1:>4d}")
    n2 = _load_manifest_examples(examples, seen)
    print(f"  [2] Board captures (manifest): {n2:>4d}")
    n3 = _load_yuv_board_captures(examples, seen)
    print(f"  [3] YUV board captures:        {n3:>4d}")
    n4 = _load_annotate_csv(ANNOTATE_30_CSV, examples, seen)
    print(f"  [4] annotate_30:               {n4:>4d}")
    n5 = _load_annotate_csv(ANNOTATE_BATCH2_CSV, examples, seen)
    print(f"  [5] annotate_batch2:           {n5:>4d}")
    n6 = _load_ai_annotated_examples(examples, seen)
    print(f"  [6] AI annotated centers:      {n6:>4d}")
    print(f"  {'─' * 40}")
    print(f"  Total unique examples:         {len(examples):>4d}")
    return examples


# --- Augmentation (same as V2) -----------------------------------------------

_ROTATION_MAX_RAD: float = 0.05


def _augment_positional(image: tf.Tensor, obb_params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    orig_cx = obb_params[0]
    orig_cy = obb_params[1]
    target_cx = tf.random.uniform([], POSITIONAL_AUG_CX_MIN, POSITIONAL_AUG_CX_MAX)
    target_cy = tf.random.uniform([], POSITIONAL_AUG_CY_MIN, POSITIONAL_AUG_CY_MAX)
    shift_x = (target_cx - orig_cx) * img_w
    shift_y = (target_cy - orig_cy) * img_h
    transforms = tf.stack([1.0, 0.0, -shift_x, 0.0, 1.0, -shift_y, 0.0, 0.0])
    image_aug = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms[tf.newaxis, :],
        output_shape=[tf.shape(image)[0], tf.shape(image)[1]],
        interpolation="BILINEAR", fill_value=0.0,
    )[0]
    obb_aug = tf.stack([
        target_cx, target_cy,
        obb_params[2], obb_params[3], obb_params[4], obb_params[5],
    ])
    return image_aug, obb_aug


def _augment_photometric(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, max_delta=0.30)
    image = tf.image.random_contrast(image, lower=0.50, upper=1.50)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _augment_rotation(image: tf.Tensor, obb_params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    angle = tf.random.uniform([], -_ROTATION_MAX_RAD, _ROTATION_MAX_RAD)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    a0 = cos_a; a1 = -sin_a; a2 = 0.5 - 0.5 * cos_a + 0.5 * sin_a
    b0 = sin_a; b1 = cos_a; b2 = 0.5 - 0.5 * cos_a - 0.5 * sin_a
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    image_rot = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms[tf.newaxis, :],
        output_shape=[tf.shape(image)[0], tf.shape(image)[1]],
        interpolation="BILINEAR", fill_value=0.0,
    )[0]
    cx, cy = obb_params[0], obb_params[1]
    dx, dy = cx - 0.5, cy - 0.5
    new_cx = cos_a * dx - sin_a * dy + 0.5
    new_cy = sin_a * dx + cos_a * dy + 0.5
    cos2t, sin2t = obb_params[4], obb_params[5]
    cos2a, sin2a = tf.cos(2.0 * angle), tf.sin(2.0 * angle)
    obb_rot = tf.stack([
        tf.clip_by_value(new_cx, 0.0, 1.0), tf.clip_by_value(new_cy, 0.0, 1.0),
        obb_params[2], obb_params[3],
        cos2t * cos2a - sin2t * sin2a,
        sin2t * cos2a + cos2t * sin2a,
    ])
    return image_rot, obb_rot


def _augment_train(
    image: tf.Tensor, y: dict[str, tf.Tensor], w: dict[str, tf.Tensor],
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    obb = y["obb_params"]
    image, new_obb = _augment_positional(image, obb)
    image, new_obb = _augment_rotation(image, new_obb)
    image = _augment_photometric(image)
    new_center = new_obb[:2]
    return image, {"obb_params": new_obb, "center_xy": new_center}, w


# --- Loss (same as YOLOv8 script) --------------------------------------------

# --- Loss -------------------------------------------------------------------

class QATHuberLoss(keras.losses.Loss):
    def __init__(self, delta: float = 0.05, reduction: str = "sum_over_batch_size",
                 name: str = "qat_huber_loss"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = tf.abs(y_true - y_pred)
        quad = 0.5 * tf.square(diff)
        lin = self.delta * (diff - 0.5 * self.delta)
        return tf.reduce_mean(tf.where(diff <= self.delta, quad, lin), axis=-1)

    def get_config(self) -> dict:
        return {"delta": self.delta, "name": self.name}


# --- Main training routine ---------------------------------------------------

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qat_micro_edgevgg_{timestamp}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Micro-EdgeVGG (QARepVGG-style) + QAT")
    print(f"  {IMAGE_HEIGHT}x{IMAGE_WIDTH}, ReLU activations")
    print(f"  Multi-branch train → reparameterize → QAT → int8 TFLite")
    print(f"  FP32 warmup: {EPOCHS_WARMUP} epochs, QAT fine-tune: {EPOCHS_QAT}")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n[1/6] Loading all labelled examples...")
    examples = _build_all_examples()
    if len(examples) < 20:
        print("ERROR: Not enough examples.")
        return

    # --- 2. Split ---
    print("\n[2/6] Splitting data...")
    train_exs, temp_exs = train_test_split(
        examples, test_size=VAL_FRACTION * 2, random_state=SEED,
    )
    val_exs, test_exs = train_test_split(
        temp_exs, test_size=0.5, random_state=SEED,
    )
    print(f"  Train: {len(train_exs)}, Val: {len(val_exs)}, Test: {len(test_exs)}")

    # --- 3. Build datasets ---
    print("\n[3/6] Building tf.data datasets...")
    edge_weights = _compute_edge_weights(train_exs, strength=0.5)

    def make_dataset(
        exs: list[TrainingExample], shuffle: bool,
        weights: np.ndarray | None = None, augment: bool = False,
    ) -> tf.data.Dataset:
        paths = [ex.image_path for ex in exs]
        values = [ex.value for ex in exs]
        obb = [ex.obb_params for ex in exs]
        crops = [ex.crop_box_xyxy for ex in exs]
        w_arr = (
            weights.astype(np.float32)
            if weights is not None
            else np.ones(len(exs), dtype=np.float32)
        )
        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(paths), tf.constant(values, dtype=tf.float32),
            tf.constant(np.array(obb, dtype=np.float32)),
            tf.constant(np.array(crops, dtype=np.float32)),
            tf.constant(w_arr),
        ))
        if shuffle:
            ds = ds.shuffle(len(exs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, v, o, c, w: _load_fullframe_obb_data_colour(
                p, v, o, c, IMAGE_HEIGHT, IMAGE_WIDTH, w,
            ), num_parallel_calls=4,
        )
        if augment:
            ds = ds.map(_augment_train, num_parallel_calls=4)
        ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
        ds = ds.prefetch(2)
        return ds

    train_ds = make_dataset(
        train_exs, shuffle=True, weights=edge_weights, augment=True,
    )
    val_ds = make_dataset(val_exs, shuffle=False)
    test_ds = make_dataset(test_exs, shuffle=False)

    # --- 4. Build multi-branch model ---
    print("\n[4/6] Building Micro-EdgeVGG multi-branch model...")
    model, repvgg_blocks = _build_reparam_model(IMAGE_HEIGHT, IMAGE_WIDTH)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=LEARNING_RATE,
                decay_steps=EPOCHS_WARMUP * max(1, len(train_exs) // BATCH_SIZE),
                alpha=1e-4,
            ),
            clipnorm=1.0,
        ),
        loss={
            "obb_params": QATHuberLoss(delta=0.05),
            "center_xy": QATHuberLoss(delta=0.02),
        },
        loss_weights={
            "obb_params": 1.0,
            "center_xy": CENTER_LOSS_WEIGHT,
        },
        metrics={
            "obb_params": ["mae"],
            "center_xy": ["mae"],
        },
        weighted_metrics=[],
    )
    model.summary(print_fn=lambda s: print(f"  {s}"))
    total_params = model.count_params()
    print(f"  Total params: {total_params:,} (~{total_params/1e6:.1f}M → ~{total_params/1e6:.1f} MB int8)")

    # --- 5a. FP32 warmup with multi-branch ---
    print(f"\n[5a/6] FP32 warmup ({EPOCHS_WARMUP} epochs, multi-branch)...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_warmup.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1,
        ),
    ]
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_WARMUP, callbacks=callbacks, verbose=2,
    )
    model.load_weights(str(run_dir / "best_warmup.keras"))

    # --- 5b. Reparameterize: multi-branch → single 3×3 conv ---
    print("\n[5b/6] Reparameterizing RepVGG blocks → single 3×3 convs...")
    reparam_model = _reparameterize_model(model, repvgg_blocks)
    reparam_model.compile(
        loss={
            "obb_params": QATHuberLoss(delta=0.05),
            "center_xy": QATHuberLoss(delta=0.02),
        },
        loss_weights={
            "obb_params": 1.0,
            "center_xy": CENTER_LOSS_WEIGHT,
        },
        metrics={
            "obb_params": ["mae"],
            "center_xy": ["mae"],
        },
        weighted_metrics=[],
    )
    reparam_model.save(str(run_dir / "reparameterized.keras"))
    print(f"  Reparameterized model saved.")

    # Quick eval to verify reparameterization preserves accuracy
    pre_reparam = model.evaluate(val_ds, verbose=0)
    post_reparam = reparam_model.evaluate(val_ds, verbose=0)
    print(f"  Val loss before reparam: {pre_reparam[0]:.6f}")
    print(f"  Val loss after reparam:  {post_reparam[0]:.6f}")
    loss_diff = abs(post_reparam[0] - pre_reparam[0])
    if loss_diff > 0.01:
        print(f"  WARNING: Large reparam loss difference ({loss_diff:.6f})!")

    # --- 5c. Apply QAT on the single-branch model ---
    print(f"\n[5c/6] QAT fine-tune ({EPOCHS_QAT} epochs, single-branch)...")
    qat_model = tfmot.quantization.keras.quantize_model(reparam_model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE, clipnorm=1.0),
        loss={
            "obb_params": QATHuberLoss(delta=0.05),
            "center_xy": QATHuberLoss(delta=0.02),
        },
        loss_weights={
            "obb_params": 1.0,
            "center_xy": CENTER_LOSS_WEIGHT,
        },
        metrics={
            "obb_params": ["mae"],
            "center_xy": ["mae"],
        },
        weighted_metrics=[],
    )
    qat_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "qat_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_qat.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        ),
    ]
    qat_model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_QAT, callbacks=qat_callbacks, verbose=2,
    )

    # --- 6. Evaluate ---
    print("\n[6/6] Evaluating on test set...")
    test_results = qat_model.evaluate(test_ds, verbose=0)
    metric_names = qat_model.metrics_names
    for name, val in zip(metric_names, test_results):
        print(f"  {name}: {val:.6f}")

    test_preds = qat_model.predict(test_ds, verbose=0)
    test_obb_preds = test_preds["obb_params"]
    test_center_preds = test_preds["center_xy"]
    test_obb_true = np.array([ex.obb_params for ex in test_exs])
    test_center_true = test_obb_true[:, :2]

    obb_param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
    per_param_mae = np.mean(np.abs(test_obb_preds - test_obb_true), axis=0)
    print("\n  OBB per-parameter MAE (normalised):")
    for name, mae in zip(obb_param_names, per_param_mae):
        print(f"    {name}: {mae:.6f}")

    cx_err = np.mean(np.abs(test_center_preds[:, 0] - test_center_true[:, 0])) * IMAGE_WIDTH
    cy_err = np.mean(np.abs(test_center_preds[:, 1] - test_center_true[:, 1])) * IMAGE_HEIGHT
    euclidean = np.mean(np.sqrt(
        np.square((test_center_preds[:, 0] - test_center_true[:, 0]) * IMAGE_WIDTH)
        + np.square((test_center_preds[:, 1] - test_center_true[:, 1]) * IMAGE_HEIGHT)
    ))
    print(f"\n  Center error @320: cx={cx_err:.1f}px, cy={cy_err:.1f}px, "
          f"euclidean={euclidean:.1f}px")

    # --- Export TFLite int8 ---
    print("\n=== Exporting TFLite int8 ===")

    def representative_dataset():
        for ex in test_exs[:50]:
            if ex.image_path.endswith(".yuv422"):
                continue
            img = tf.io.read_file(ex.image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = _preprocess_colour(img, IMAGE_HEIGHT, IMAGE_WIDTH)
            yield [tf.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = run_dir / "micro_edgevgg_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  TFLite int8: {tflite_path}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb / 1024:.2f} MB)")

    # --- Save summary ---
    summary = {
        "run_name": run_name,
        "model": "micro_edgevgg_qarepvg_style",
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "batch_size": BATCH_SIZE,
        "epochs_warmup": EPOCHS_WARMUP,
        "epochs_qat": EPOCHS_QAT,
        "num_train": len(train_exs),
        "num_val": len(val_exs),
        "num_test": len(test_exs),
        "total_params": int(total_params),
        "test_center_mae_px": {
            "cx": float(cx_err), "cy": float(cy_err), "euclidean": float(euclidean),
        },
        "test_obb_mae": float(per_param_mae.mean()),
        "per_param_mae_obb": {
            name: float(mae) for name, mae in zip(obb_param_names, per_param_mae)
        },
        "tflite_size_kb": float(size_kb),
        "tflite_under_2_5_mb": size_kb < 2560,
    }
    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")
    print("\n=== Done ===")
    if size_kb >= 2560:
        print(f"WARNING: TFLite size {size_kb:.0f} KB exceeds 2.5 MB!")
    else:
        print(f"TFLite size {size_kb:.0f} KB is under the 2.5 MB target. ✓")


if __name__ == "__main__":
    main()
