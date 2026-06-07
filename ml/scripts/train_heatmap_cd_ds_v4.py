"""
DS-CNN v4: 160×160 heatmap output for 2× finer center localization.

Architecture: CoordConv + ReLU6 + wider encoder + 160×160 heatmap output
  - Encoder: 320→160→80→40→20 (channels 48→96→192→384)
  - Decoder: 20→40→80→160 with skip connections
  - Heatmap sigma: 6.0 px (doubled from 3.0 for 2× resolution)
  - Target: 2.39 px in 160×160 space = 4.78 px @ 320 (under 5 px)

Usage:
  python scripts/train_heatmap_cd_ds_v4.py
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
from embedded_gauge_reading_tinyml.heatmap_utils import softargmax_2d, make_gaussian_heatmap  # noqa: E402

SEED = 42
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-3
INPUT_SIZE = 320
HEATMAP_SIZE = 160  # 2× v3c
SIGMA_PIXELS = 6.0  # 2× v3c (scaled with heatmap resolution)
EARLY_STOP_PATIENCE = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd_ds_v4"
METADATA_PATH = DATA_DIR / "metadata.json"

tf.random.set_seed(SEED)
np.random.seed(SEED)


def add_coordconv(x: tf.Tensor) -> tf.Tensor:
    _, h, w, _ = x.shape
    x_range = tf.linspace(-1.0, 1.0, w)
    y_range = tf.linspace(-1.0, 1.0, h)
    x_grid, y_grid = tf.meshgrid(x_range, y_range)
    x_ch = x_grid[None, :, :, None]
    y_ch = y_grid[None, :, :, None]
    x_tiled = tf.tile(x_ch, [tf.shape(x)[0], 1, 1, 1])
    y_tiled = tf.tile(y_ch, [tf.shape(x)[0], 1, 1, 1])
    return tf.concat([x, tf.cast(x_tiled, x.dtype), tf.cast(y_tiled, x.dtype)], axis=-1)


def ds_conv_block(
    x: tf.Tensor, filters: int, stride: int = 1, name: str = ""
) -> tf.Tensor:
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(6., name=f"{name}_dw_relu6")(x)
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(6., name=f"{name}_pw_relu6")(x)
    return x


def build_model(
    input_shape: tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, 3),
    heatmap_size: int = HEATMAP_SIZE,
) -> keras.Model:
    """DS-CNN v4 with 160×160 heatmap output.

    Encoder: 320→160→80→40→20 (4 stages)
    Decoder: 20→40→80→160 (4 stages, skip connections at 80 and 40)
    """
    inputs = layers.Input(shape=input_shape, name="input")
    skips: dict[str, tf.Tensor] = {}

    x = layers.Lambda(lambda t: add_coordconv(t), name="coordconv")(inputs)

    # Head conv: 5→48
    x = layers.Conv2D(48, 3, strides=2, padding="same", use_bias=False, name="enc_head")(x)
    x = layers.BatchNormalization(name="enc_head_bn")(x)
    x = layers.ReLU(6., name="enc_head_relu6")(x)  # 160×160×48

    # Encoder stages (no skip at 160 — too early)
    x = ds_conv_block(x, 96, stride=2, name="enc_s1")   # 80×80×96
    skips["s1"] = x

    x = ds_conv_block(x, 192, stride=2, name="enc_s2")  # 40×40×192
    skips["s2"] = x

    x = ds_conv_block(x, 384, stride=2, name="enc_s3")  # 20×20×384

    # Decoder (narrower channels to save memory for 160×160 output)
    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_up1")(x)  # 40×40
    x = layers.Concatenate(name="dec_cat1")([x, skips["s2"]])
    x = ds_conv_block(x, 160, stride=1, name="dec_ds1")

    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_up2")(x)  # 80×80
    x = layers.Concatenate(name="dec_cat2")([x, skips["s1"]])
    x = ds_conv_block(x, 96, stride=1, name="dec_ds2")

    # Extra decoder stage for 160×160 output
    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_up3")(x)  # 160×160
    x = ds_conv_block(x, 48, stride=1, name="dec_ds3")

    # Output heatmap
    heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="heatmap")(x)

    model = keras.Model(inputs=inputs, outputs=heatmap)
    return model


def heatmap_from_center(cx_norm: float, cy_norm: float, size: int, sigma: float) -> np.ndarray:
    """Generate Gaussian heatmap at specified resolution."""
    return make_gaussian_heatmap(size, size, cx_norm, cy_norm, sigma)


def augment_image(img: tf.Tensor, hm: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.image.stateless_random_brightness(img, max_delta=0.15, seed=(SEED, 1))
    img = tf.image.stateless_random_contrast(img, lower=0.75, upper=1.25, seed=(SEED, 2))
    img = tf.image.stateless_random_hue(img, max_delta=0.05, seed=(SEED, 3))
    img = tf.image.stateless_random_saturation(img, lower=0.75, upper=1.25, seed=(SEED, 4))
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, hm


def center_pixel_error(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    errors = []
    for gt, pred in zip(y_true, y_pred):
        gt_pt = softargmax_2d(gt.squeeze(-1))
        pred_pt = softargmax_2d(pred.squeeze(-1))
        dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        errors.append(dist)
    return float(np.mean(errors)), float(np.median(errors))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    train_samples = meta["samples"]["train"]
    val_samples = meta["samples"]["val"]

    def load_split(samples: list[dict], split: str) -> tuple[np.ndarray, np.ndarray]:
        img_arrays, hm_arrays = [], []
        for s in samples:
            img = tf.io.decode_jpeg(
                tf.io.read_file(str(DATA_DIR / f"images/{split}/{s['stem']}.jpg")),
                channels=3,
            )
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            img_arrays.append(img.numpy())
            # Generate 160×160 heatmap from stored center_xy_norm
            cxn, cyn = s["center_xy_norm"]
            hm = heatmap_from_center(cxn, cyn, HEATMAP_SIZE, SIGMA_PIXELS)
            hm_arrays.append(hm.astype(np.float32))
        return np.stack(img_arrays, axis=0), np.stack(hm_arrays, axis=0)[..., None]

    print("Loading training data...")
    X_train, y_train = load_split(train_samples, "train")
    print(f"  Train: {X_train.shape}, {y_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_split(val_samples, "val")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()

    trainable = int(np.sum([keras.backend.count_params(p) for p in model.trainable_weights]))
    print(f"\nTotal trainable params: {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACT_DIR / "best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1,
    )

    model.save(str(ARTIFACT_DIR / "final.keras"))
    print(f"\nModel saved to {ARTIFACT_DIR}")

    y_val_pred = model.predict(X_val, verbose=0)
    mean_err, median_err = center_pixel_error(y_val, y_val_pred)
    scale_input = (INPUT_SIZE - 1) / (HEATMAP_SIZE - 1)
    print(f"\nKeras — Val center error ({HEATMAP_SIZE}×{HEATMAP_SIZE}): Mean={mean_err:.3f} Median={median_err:.3f}")
    print(f"  @{INPUT_SIZE}: Mean={mean_err * scale_input:.2f} Median={median_err * scale_input:.2f}")

    results = {
        "model": "depthwise_separable_cnn_v4_160",
        "params": trainable,
        "heatmap_size": HEATMAP_SIZE,
        "input_size": INPUT_SIZE,
        "loss": "mse",
        "coordconv": True,
        "activation": "relu6",
        "augmentation": "brightness_contrast_hue_saturation",
        "sigma_pixels": SIGMA_PIXELS,
        "keras_heatmap_pixels": {"mean": float(f"{mean_err:.4f}"), "median": float(f"{median_err:.4f}")},
        "keras_input_pixels": {
            "mean": float(f"{mean_err * scale_input:.4f}"),
            "median": float(f"{median_err * scale_input:.4f}"),
        },
    }

    print("\nExporting TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
    print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def representative_dataset():
        for i in range(min(100, len(X_val))):
            yield [X_val[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_int8 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_int8.tflite").write_bytes(tflite_int8)
    print(f"  int8:   {len(tflite_int8) / 1024:.1f} KB")

    print("\n--- Validating TFLite ---")

    def eval_tflite(tflite_path: str, label: str) -> dict:
        interp = tf.lite.Interpreter(str(tflite_path))
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        is_int8 = in_d["dtype"] == np.uint8
        in_scale, in_zero = in_d["quantization"] if is_int8 else (1.0, 0)
        out_scale, out_zero = out_d["quantization"] if is_int8 else (1.0, 0)
        errs = []
        for i in range(len(X_val)):
            inp = (np.round(X_val[i:i+1] / in_scale + in_zero).astype(np.uint8)
                   if is_int8 else X_val[i:i+1].astype(np.float32))
            interp.set_tensor(in_d["index"], inp)
            interp.invoke()
            raw = interp.get_tensor(out_d["index"])
            pred = (raw.astype(np.float32) - out_zero) * out_scale if is_int8 else raw
            gt_pt = softargmax_2d(y_val[i, :, :, 0])
            pred_pt = softargmax_2d(pred[0, :, :, 0])
            dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
            errs.append(dist)
        err_arr = np.array(errs)
        mean_px = float(err_arr.mean())
        median_px = float(np.median(err_arr))
        print(f"  {label}: Mean={mean_px:.3f} ({mean_px * scale_input:.2f} @320), Median={median_px:.3f} ({median_px * scale_input:.2f} @320)")
        return {"mean": mean_px, "median": median_px}

    results["tflite_fp32_kb"] = round(len(tflite_fp32) / 1024, 1)
    results["tflite_int8_kb"] = round(len(tflite_int8) / 1024, 1)
    fp32_err = eval_tflite(str(ARTIFACT_DIR / "heatmap_cd_fp32.tflite"), "TFLite fp32")
    int8_err = eval_tflite(str(ARTIFACT_DIR / "heatmap_cd_int8.tflite"), "TFLite int8")
    results["tflite_fp32"] = fp32_err
    results["tflite_int8"] = int8_err

    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone — results in {ARTIFACT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
