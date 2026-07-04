"""
Train a heatmap-based gauge center detector.

Pipeline:
  1. Load rectified 224×224 gauge crops + 56×56 Gaussian heatmap targets
  2. Build MobileNetV2 encoder + lightweight decoder
  3. Train with MSE loss on heatmap
  4. Evaluate with pixel distance between predicted and target center
  5. Export to TFLite (float32 and int8)

Usage:
  python scripts/train_heatmap_cd.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap, softargmax_2d  # noqa: E402

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
INPUT_SIZE = 224
HEATMAP_SIZE = 56
SIGMA_PIXELS = 2.5
EARLY_STOP_PATIENCE = 15
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd"

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model(
    input_shape: tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, 3),
    heatmap_size: int = HEATMAP_SIZE,
    alpha: float = 0.35,
) -> keras.Model:
    """MobileNetV2 encoder + simple decoder for center heatmap."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
    )
    backbone.trainable = False

    x = backbone.output  # (7, 7, 1280*alpha)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)  # 14×14

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)  # 28×28

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)  # 56×56

    heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="heatmap"
    )(x)

    model = keras.Model(inputs=backbone.input, outputs=heatmap)
    return model


def parse_metadata(data_dir: Path) -> tuple[list[Path], list[Path], list[dict]]:
    """Load image paths, heatmap paths, and center metadata from prepared dataset."""
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    img_paths: list[Path] = []
    hm_paths: list[Path] = []
    centers: list[dict] = []

    for split in ("train", "val"):
        for s in meta["samples"][split]:
            img_paths.append(data_dir / "images" / split / f"{s['stem']}.jpg")
            hm_paths.append(data_dir / "heatmaps" / split / f"{s['stem']}.npy")
            centers.append(s)

    return img_paths, hm_paths, centers


def load_sample(
    img_path: Path, hm_path: Path
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load and preprocess one training sample."""
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # normalise to [-1, 1]

    hm = np.load(str(hm_path)).astype(np.float32)
    hm = tf.convert_to_tensor(hm)
    hm = tf.expand_dims(hm, axis=-1)  # (56, 56, 1)

    return img, hm


def create_dataset(
    img_paths: list[Path],
    hm_paths: list[Path],
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from image/heatmap path lists."""
    ds = tf.data.Dataset.from_tensor_slices(
        (list(map(str, img_paths)), list(map(str, hm_paths)))
    )

    def _load(img_str, hm_str):
        img = tf.io.read_file(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        hm = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.float32)[..., None],
            [hm_str],
            tf.float32,
        )
        hm.set_shape((HEATMAP_SIZE, HEATMAP_SIZE, 1))
        return img, hm

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def center_pixel_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Compute mean and median Euclidean distance between predicted and true
    center in heatmap pixels (56×56 space)."""
    errors = []
    for gt, pred in zip(y_true, y_pred):
        gt_pt = softargmax_2d(gt.squeeze(-1))
        pred_pt = softargmax_2d(pred.squeeze(-1))
        dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        errors.append(dist)
    return float(np.mean(errors)), float(np.median(errors))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    train_samples = meta["samples"]["train"]
    val_samples = meta["samples"]["val"]

    train_img_paths = [DATA_DIR / "images" / "train" / f"{s['stem']}.jpg" for s in train_samples]
    train_hm_paths = [DATA_DIR / "heatmaps" / "train" / f"{s['stem']}.npy" for s in train_samples]
    val_img_paths = [DATA_DIR / "images" / "val" / f"{s['stem']}.jpg" for s in val_samples]
    val_hm_paths = [DATA_DIR / "heatmaps" / "val" / f"{s['stem']}.npy" for s in val_samples]

    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    train_ds = create_dataset(train_img_paths, train_hm_paths, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(val_img_paths, val_hm_paths, BATCH_SIZE, shuffle=False)

    # Build the model
    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
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
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACT_DIR / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(str(ARTIFACT_DIR / "final.keras"))
    print(f"Model saved to {ARTIFACT_DIR}")

    # Evaluate pixel error on validation set
    val_image_data = []
    for p in val_img_paths:
        img_data = tf.io.decode_jpeg(tf.io.read_file(str(p)), channels=3)
        img_data = tf.cast(img_data, tf.float32) / 127.5 - 1.0
        val_image_data.append(img_data)
    X_val = tf.stack(val_image_data, axis=0)
    y_val_pred = model.predict(X_val, verbose=0)

    y_val_true = np.stack([np.load(str(p))[..., None] for p in val_hm_paths], axis=0)

    mean_err, median_err = center_pixel_error(y_val_true, y_val_pred)
    print(f"\nValidation center error (56×56 heatmap pixels):")
    print(f"  Mean:   {mean_err:.3f} px")
    print(f"  Median: {median_err:.3f} px")
    print(f"  224×224 equivalent:")
    scale_224 = (INPUT_SIZE - 1) / (HEATMAP_SIZE - 1)
    print(f"  Mean:   {mean_err * scale_224:.2f} px")
    print(f"  Median: {median_err * scale_224:.2f} px")

    # Save evaluation results
    results = {
        "heatmap_pixels": {"mean": float(f"{mean_err:.4f}"), "median": float(f"{median_err:.4f}")},
        "input_pixels": {
            "mean": float(f"{mean_err * scale_224:.4f}"),
            "median": float(f"{median_err * scale_224:.4f}"),
        },
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(f"{min(history.history['val_loss']):.6f}"),
    }
    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {ARTIFACT_DIR / 'eval_results.json'}")

    # Export to TFLite
    print("\nExporting TFLite models...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
    print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = lambda: ({
        'keras_tensor': X_val[i:i+1].astype(np.float32)
    } for i in range(min(100, len(X_val))))
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_int8 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_int8.tflite").write_bytes(tflite_int8)
    print(f"  int8:   {len(tflite_int8) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
