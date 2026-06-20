"""Board-specific OBB localizer with MobileNetV2 transfer learning.

Uses ImageNet-pretrained MobileNetV2 backbone (alpha=0.25) for robust feature
extraction, then trains a small OBB head on top. The backbone is frozen initially
and fine-tuned after the head converges.

Target: <1MB int8 for OBB, leaving 1.5MB for CD model.
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Limit GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import (
    TrainingExample,
    _augment_image,
    _compute_edge_weights,
    _compute_fullframe_obb_params,
)

ML_ROOT: Path = PROJECT_ROOT
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
MANIFEST_PATH: Path = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"
AI_CENTERS_CSV: Path = ML_ROOT / "data" / "ai_annotated_board_captures.csv"
ANNOTATE_30_CSV: Path = ML_ROOT / "data" / "annotate_30" / "annotations.csv"
ANNOTATE_BATCH2_CSV: Path = ML_ROOT / "data" / "annotate_batch2" / "annotations_batch2.csv"
CAPTURED_IMAGES_DIR: Path = ML_ROOT / "data" / "captured_images"
YUV_LABELS_CSV: Path = ML_ROOT / "data" / "capture_2026-06-07_labels_v2.csv"

GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO: float = 0.3076

IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
SEED: int = 42
BATCH_SIZE: int = 16
EPOCHS: int = 300
LEARNING_RATE: float = 1e-3
VAL_FRACTION: float = 0.20

# MobileNetV2 alpha for OBB (alpha=0.25 ≈ 1MB int8)
MOBILENET_ALPHA: float = 0.35

POSITIONAL_AUG_CY_MIN: float = 0.30
POSITIONAL_AUG_CY_MAX: float = 0.95
POSITIONAL_AUG_CX_MIN: float = 0.30
POSITIONAL_AUG_CX_MAX: float = 0.70


def _preprocess_colour(
    image: tf.Tensor, image_height: int, image_width: int
) -> tf.Tensor:
    """Resize-with-pad in RGB colour space (no luma conversion)."""
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

    resized = tf.image.resize(
        image, [scaled_h, scaled_w], method="nearest"
    )

    pad_y = (image_height - scaled_h) // 2
    pad_x = (image_width - scaled_w) // 2
    pad_bottom = image_height - scaled_h - pad_y
    pad_right = image_width - scaled_w - pad_x
    padded = tf.pad(
        resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]]
    )
    padded = tf.ensure_shape(padded, [image_height, image_width, 3])

    return padded / 255.0


def _load_fullframe_obb_data_colour(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load full image, RGB resize-with-pad, attach OBB targets."""
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
        image_bytes = tf.io.read_file(image_path)
        return tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    image = tf.cond(is_yuv, _load_yuv, _load_standard)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, image_height, image_width)
    obb_target = tf.cast(obb_params, tf.float32)
    return (
        image,
        {"obb_params": obb_target},
        {"obb_params": tf.cast(weight, tf.float32)},
    )


def _is_original_capture(filename: str) -> bool:
    """Return True if *filename* is an original capture (not a derivative)."""
    name = Path(filename).name
    if ".gray." in name:
        return False
    if "_preview" in name:
        return False
    if "_yuy2" in name:
        return False
    if "_glare_" in name:
        return False
    return True


def _obb_params_from_center_224(
    cx_224: float,
    cy_224: float,
    source_size: int = 224,
) -> np.ndarray:
    """Compute 320x320-canvas OBB params from a center in 224 px space."""
    radius = source_size * GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
    return _compute_fullframe_obb_params(
        source_size, source_size,
        cx_224, cy_224,
        radius, radius,
        0.0,
        IMAGE_HEIGHT, IMAGE_WIDTH,
    )


def _load_manifest_examples(
    examples: list[TrainingExample],
    seen: set[str],
) -> int:
    """Load board-capture rows from the merged geometry manifest."""
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
                source_w, source_h,
                cx, cy, radius, radius,
                0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=float(row.get("temperature_c", 0)),
                    crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )
            added += 1
    return added


def _load_pxl_photo_examples(
    examples: list[TrainingExample],
    seen: set[str],
) -> int:
    """Load PXL phone photos from the manifest with full geometry labels."""
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
                source_w, source_h,
                cx, cy, radius, radius,
                0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=float(row.get("temperature_c", 0)),
                    crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )
            added += 1
    return added


def _load_annotate_csv(
    csv_path: Path,
    examples: list[TrainingExample],
    seen: set[str],
    *,
    source_size: int = 224,
) -> int:
    """Load center-label examples from an annotate-style CSV."""
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

            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=0.0,
                    crop_box_xyxy=(0.0, 0.0, float(source_size), float(source_size)),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )
            added += 1
    return added


def _load_ai_annotated_examples(
    examples: list[TrainingExample],
    seen: set[str],
) -> int:
    """Load center-label examples from the AI-annotated centers CSV."""
    added = 0
    with open(AI_CENTERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["image_path"]
            if not _is_original_capture(rel_path):
                continue

            fpath = str(ML_ROOT / "data" / rel_path)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            cx = float(row["center_x"])
            cy = float(row["center_y"])
            obb_params = _obb_params_from_center_224(cx, cy, source_size=224)

            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=0.0,
                    crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )
            added += 1
    return added


def _load_yuv422_image(yuv_path: str) -> np.ndarray:
    """Load a 320x320 YUV422 (YUYV) file and convert to RGB numpy array."""
    with open(yuv_path, "rb") as f:
        data = f.read()

    if len(data) != 204800:
        raise ValueError(f"Expected 204800 bytes, got {len(data)}")

    yuyv = np.frombuffer(data, dtype=np.uint8).reshape(320, 640)
    y = yuyv[:, 0::2].astype(np.float32)
    u = yuyv[:, 1::4].astype(np.float32) - 128.0
    v = yuyv[:, 3::4].astype(np.float32) - 128.0
    u = np.repeat(u, 2, axis=1)
    v = np.repeat(v, 2, axis=1)
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _load_yuv_board_captures(
    examples: list[TrainingExample],
    seen: set[str],
) -> int:
    """Load 320x320 YUV422 board captures with manual labels."""
    added = 0
    if not YUV_LABELS_CSV.exists():
        print(f"  WARNING: {YUV_LABELS_CSV} not found, skipping YUV captures")
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
                320, 320,
                cx, cy, outer_r, outer_r,
                0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )

            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=0.0,
                    crop_box_xyxy=(0.0, 0.0, 320.0, 320.0),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )
            added += 1
    return added


def _build_board_only_examples() -> list[TrainingExample]:
    """Build OBB examples from PXL photos and board captures."""
    examples: list[TrainingExample] = []
    seen: set[str] = set()

    n1 = _load_pxl_photo_examples(examples, seen)
    print(f"  [source 1] PXL photos:          {n1} examples")

    n2 = _load_manifest_examples(examples, seen)
    print(f"  [source 2] board captures:      {n2} examples")

    n3 = _load_yuv_board_captures(examples, seen)
    print(f"  [source 3] YUV board captures:  {n3} examples")

    n4 = _load_annotate_csv(ANNOTATE_30_CSV, examples, seen)
    print(f"  [source 4] annotate_30:         {n4} examples")

    n5 = _load_annotate_csv(ANNOTATE_BATCH2_CSV, examples, seen)
    print(f"  [source 5] annotate_batch2:     {n5} examples")

    n6 = _load_ai_annotated_examples(examples, seen)
    print(f"  [source 6] ai_annotated_centers: {n6} examples")

    print(f"  Total unique examples:          {len(examples)}")
    return examples


class OBBEqualLoss(keras.losses.Loss):
    """Huber loss with equal weights on all 6 OBB parameters."""

    def __init__(
        self,
        delta: float = 0.05,
        reduction: str = "sum_over_batch_size",
        name: str = "obb_equal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        quadratic = 0.5 * tf.square(diff)
        linear = self.delta * (diff - 0.5 * self.delta)
        loss = tf.where(diff <= self.delta, quadratic, linear)
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self) -> dict:
        return {"delta": self.delta}


def build_mobilenetv2_obb_model(
    image_height: int = 320,
    image_width: int = 320,
    *,
    alpha: float = 0.25,
    head_units: int = 96,
    head_dropout: float = 0.15,
    pretrained: bool = True,
) -> keras.Model:
    """Build OBB model with MobileNetV2 backbone (transfer learning).
    
    Uses ImageNet-pretrained MobileNetV2 for robust feature extraction,
    then adds a small OBB head on top.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")
    
    # MobileNetV2 expects [-1, 1], but our pipeline emits [0, 1]
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    
    # Load pretrained MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = False  # Start frozen — will unfreeze later
    
    x = base_model(x, training=False)  # Use inference mode for BN
    x = keras.layers.GlobalAveragePooling2D(name="obb_gap")(x)
    
    # OBB head
    x = keras.layers.LayerNormalization(name="obb_pooled_norm")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="obb_dense")(x)
    x = keras.layers.Dropout(head_dropout, name="obb_dropout")(x)
    
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="obb_center_xy")(x)
    size_wh = keras.layers.Dense(2, activation="sigmoid", name="obb_size_wh")(x)
    angle_raw = keras.layers.Dense(2, name="obb_angle_raw")(x)
    angle_sincos = keras.layers.UnitNormalization(axis=-1, name="obb_angle_sincos")(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")([center_xy, size_wh, angle_sincos])
    
    model = keras.Model(inputs=inputs, outputs={"obb_params": obb_params}, name=f"mobilenetv2_obb_alpha{alpha}")
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def _augment_obb_positional(
    image: tf.Tensor,
    obb_params: tf.Tensor,
    cy_min: float = POSITIONAL_AUG_CY_MIN,
    cy_max: float = POSITIONAL_AUG_CY_MAX,
    cx_min: float = POSITIONAL_AUG_CX_MIN,
    cx_max: float = POSITIONAL_AUG_CX_MAX,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply positional augmentation: shift gauge to random position."""
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)

    orig_cx = obb_params[0]
    orig_cy = obb_params[1]

    target_cx = tf.random.uniform([], cx_min, cx_max)
    target_cy = tf.random.uniform([], cy_min, cy_max)

    shift_x = (target_cx - orig_cx) * img_w
    shift_y = (target_cy - orig_cy) * img_h

    transforms = tf.stack([1.0, 0.0, -shift_x, 0.0, 1.0, -shift_y, 0.0, 0.0])

    image_aug = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms[tf.newaxis, :],
        output_shape=[tf.shape(image)[0], tf.shape(image)[1]],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]

    obb_aug = tf.stack([
        target_cx, target_cy,
        obb_params[2], obb_params[3], obb_params[4], obb_params[5],
    ])

    return image_aug, obb_aug


def _augment_obb(
    image: tf.Tensor, y: dict, w: tf.Tensor,
) -> tuple[tf.Tensor, dict, tf.Tensor]:
    """Full augmentation pipeline: positional + photometric."""
    image, new_obb = _augment_obb_positional(image, y["obb_params"])
    y_aug = {"obb_params": new_obb}
    image = _augment_image(image)
    return image, y_aug, w


def main() -> None:
    """Train board-specific OBB model with MobileNetV2 transfer learning."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"obb_mobilenetv2_{timestamp}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== MobileNetV2 OBB Training at 320x320 ===")
    print(f"Run directory: {run_dir}")

    print("\n[1/5] Loading examples...")
    examples = _build_board_only_examples()

    if len(examples) < 20:
        print("ERROR: Not enough examples to train.")
        return

    print("\n[2/5] Splitting data...")
    train_exs, temp_exs = train_test_split(
        examples, test_size=VAL_FRACTION * 2, random_state=SEED,
    )
    val_exs, test_exs = train_test_split(
        temp_exs, test_size=0.5, random_state=SEED,
    )
    print(f"  Train: {len(train_exs)}, Val: {len(val_exs)}, Test: {len(test_exs)}")

    train_cy_values = [ex.obb_params[1] for ex in train_exs]
    print(f"  Train cy range: [{min(train_cy_values):.3f}, {max(train_cy_values):.3f}]")
    print(f"  Train cy mean: {np.mean(train_cy_values):.3f}")

    edge_weights = _compute_edge_weights(train_exs, strength=0.5)

    print("\n[3/5] Building datasets...")

    def make_dataset(
        exs: list[TrainingExample],
        shuffle: bool,
        weights: np.ndarray | None = None,
        augment: bool = False,
    ) -> tf.data.Dataset:
        paths = [ex.image_path for ex in exs]
        values = [ex.value for ex in exs]
        obb = [ex.obb_params for ex in exs]
        crops = [ex.crop_box_xyxy for ex in exs]
        if weights is None:
            weights_arr = np.ones(len(exs), dtype=np.float32)
        else:
            weights_arr = weights.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(paths),
            tf.constant(values, dtype=tf.float32),
            tf.constant(np.array(obb, dtype=np.float32)),
            tf.constant(np.array(crops, dtype=np.float32)),
            tf.constant(weights_arr),
        ))
        if shuffle:
            ds = ds.shuffle(len(exs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, v, o, c, w: _load_fullframe_obb_data_colour(
                p, v, o, c, IMAGE_HEIGHT, IMAGE_WIDTH, w
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if augment:
            ds = ds.map(_augment_obb, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_exs, shuffle=True, weights=edge_weights, augment=True)
    val_ds = make_dataset(val_exs, shuffle=False)
    test_ds = make_dataset(test_exs, shuffle=False)

    print("\n[4/5] Building model...")
    model = build_mobilenetv2_obb_model(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        alpha=MOBILENET_ALPHA,
        head_units=96,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={"obb_params": OBBEqualLoss(delta=0.05)},
        metrics={"obb_params": [keras.metrics.MeanAbsoluteError(name="mae")]},
    )
    model.summary(print_fn=lambda s: print(f"  {s}"))

    # Phase 1: train head only, backbone frozen, full epochs
    print("\n[5/5] Phase 1 — training head (backbone frozen)...")
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=30, verbose=2,
    )
    print(f"  Phase 1 complete — best val_mae during head phase: {min(history1.history['val_mae']):.4f}")

    # Phase 2: unfreeze backbone, fine-tune with lower LR
    print("\n[5/5] Phase 2 — fine-tuning backbone...")
    backbone = getattr(model, "_mobilenet_backbone", None)
    if backbone is not None:
        backbone.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss={"obb_params": OBBEqualLoss(delta=0.05)},
        metrics={"obb_params": [keras.metrics.MeanAbsoluteError(name="mae")]},
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=30, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", mode="min", factor=0.5, patience=10, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_model.keras"),
            monitor="val_mae", mode="min", save_best_only=True, verbose=1,
        ),
    ]

    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, initial_epoch=30,
        callbacks=callbacks, verbose=2,
    )

    print("\n=== Test Set Evaluation ===")
    test_results = model.evaluate(test_ds, verbose=0)
    print(f"  Test MAE: {test_results[1]:.4f}")

    test_preds_raw = model.predict(test_ds, verbose=0)
    test_preds = test_preds_raw["obb_params"]
    test_targets = np.array([ex.obb_params for ex in test_exs])
    per_param_mae = np.mean(np.abs(test_preds - test_targets), axis=0)
    param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
    print("  Per-parameter MAE:")
    for name, mae in zip(param_names, per_param_mae):
        print(f"    {name}: {mae:.4f}")

    cx_err = np.mean(np.abs(test_preds[:, 0] - test_targets[:, 0])) * IMAGE_WIDTH
    cy_err = np.mean(np.abs(test_preds[:, 1] - test_targets[:, 1])) * IMAGE_HEIGHT
    euclidean_err = np.mean(np.sqrt(
        np.square((test_preds[:, 0] - test_targets[:, 0]) * IMAGE_WIDTH) +
        np.square((test_preds[:, 1] - test_targets[:, 1]) * IMAGE_HEIGHT)
    ))
    print(f"\n  Center error @320: cx={cx_err:.1f}px, cy={cy_err:.1f}px, euclidean={euclidean_err:.1f}px")

    model_path = run_dir / "final_model.keras"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}")

    print("\n=== Exporting TFLite int8 ===")
    def representative_dataset():
        for ex in test_exs[:50]:
            is_yuv = ex.image_path.endswith(".yuv422")
            if is_yuv:
                raw = tf.io.read_file(ex.image_path)
                yuyv = tf.io.decode_raw(raw, tf.uint8)
                yuyv = tf.reshape(yuyv, [320, 640])
                y = tf.cast(yuyv[:, 0::2], tf.float32)
                u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
                v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
                u = tf.repeat(u, 2, axis=1)
                v = tf.repeat(v, 2, axis=1)
                rgb = tf.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
                rgb = tf.clip_by_value(rgb, 0, 255) / 255.0
                rgb = tf.ensure_shape(rgb, [320, 320, 3])
                yield [tf.expand_dims(rgb, 0)]
            else:
                img = tf.io.read_file(ex.image_path)
                img = tf.io.decode_image(img, channels=3, expand_animations=False)
                img = _preprocess_colour(img, IMAGE_HEIGHT, IMAGE_WIDTH)
                yield [tf.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = run_dir / "obb_mobilenetv2_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"  TFLite int8 saved to: {tflite_path}")
    print(f"  Size: {len(tflite_model) / 1024:.1f} KB")

    summary = {
        "run_name": run_name,
        "num_train": len(train_exs),
        "num_val": len(val_exs),
        "num_test": len(test_exs),
        "test_mae": float(test_results[1]),
        "test_cx_err_px": float(cx_err),
        "test_cy_err_px": float(cy_err),
        "test_euclidean_err_px": float(euclidean_err),
        "per_param_mae": {name: float(mae) for name, mae in zip(param_names, per_param_mae)},
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "batch_size": BATCH_SIZE,
        "epochs_trained": len(history1.history["loss"]) + len(history2.history["loss"]),
        "final_val_mae": float(min(history2.history["val_mae"])),
        "mobilenet_alpha": MOBILENET_ALPHA,
    }

    import json
    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
