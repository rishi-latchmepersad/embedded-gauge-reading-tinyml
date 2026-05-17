"""Head surgery: replace gpu5_recover head with larger head and train.

Loads gpu5_recover, freezes backbone, adds 256-unit head, trains on
full labelled dataset + hard cases with heavy augmentation.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224


def load_luma_yuv422(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE // 2, 4)
    luma = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return luma


def load_rgb_png(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.uint8)


def crop_and_resize(img_hw3: np.ndarray) -> np.ndarray:
    h, w = img_hw3.shape[:2]
    x0, x1 = int(TRAINING_CROP_X_MIN * w), int(TRAINING_CROP_X_MAX * w)
    y0, y1 = int(TRAINING_CROP_Y_MIN * h), int(TRAINING_CROP_Y_MAX * h)
    crop = img_hw3[y0:y1, x0:x1]
    rgb = crop.astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def augment_image(img: tf.Tensor) -> tf.Tensor:
    """Apply photometric augmentation matching training distribution."""
    img = tf.image.random_brightness(img, max_delta=0.35)
    img = tf.image.random_contrast(img, lower=0.55, upper=1.45)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_saturation(img, lower=0.70, upper=1.30)
    img = tf.clip_by_value(img, 0.0, 1.0)
    gamma = tf.random.uniform([], minval=1.0, maxval=2.8, dtype=tf.float32)
    img = tf.where(tf.random.uniform([]) < 0.35, tf.pow(img, gamma), img)
    img = tf.clip_by_value(img, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def load_manifest_samples(manifest_path: Path, repo_root: Path):
    samples = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            img_col = next((k for k in row if "image" in k.lower() or "path" in k.lower() or "file" in k.lower()), list(row.keys())[0])
            val_col = next((k for k in row if "value" in k.lower() or "label" in k.lower() or "temp" in k.lower()), list(row.keys())[1])
            p = Path(row[img_col])
            if not p.is_absolute():
                p = repo_root / p
            samples.append((p, float(row[val_col])))
    return samples


def make_dataset(samples, batch_size=16, training=True):
    def generator():
        while True:  # infinite generator for repeat
            for path, value in samples:
                if not path.exists() or path.stat().st_size == 0:
                    continue
                if path.suffix == ".yuv422":
                    luma = load_luma_yuv422(path)
                    img = np.repeat(luma[:, :, None], 3, axis=2)
                else:
                    img = load_rgb_png(path)
                inp = crop_and_resize(img)
                yield inp.astype(np.float32), np.array([value], dtype=np.float32)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )
    if training:
        ds = ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("[SURGERY] Loading gpu5_recover...")
    base_model = tf.keras.models.load_model(
        str(PROJECT_ROOT / "artifacts/training/no_cal_hardpush_gpu5_recover/model.keras"),
        compile=False,
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
    )

    backbone_output = base_model.get_layer("global_average_pooling2d").output
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_1_new")(backbone_output)
    x = tf.keras.layers.Dense(256, activation="silu", name="head_dense_1_new")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2_new")(x)
    output = tf.keras.layers.Dense(1, activation="linear", name="gauge_value_new")(x)

    new_model = tf.keras.Model(inputs=base_model.input, outputs=output, name="mobilenetv2_head_surgery")

    for layer in new_model.layers:
        if layer.name in ("head_dropout_1_new", "head_dense_1_new", "head_dropout_2_new", "gauge_value_new"):
            layer.trainable = True
        else:
            layer.trainable = False

    trainable = sum([tf.keras.backend.count_params(w) for w in new_model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in new_model.non_trainable_weights])
    print(f"[SURGERY] Model: {new_model.count_params():,} total params")
    print(f"[SURGERY] Trainable: {trainable:,} | Non-trainable: {non_trainable:,}")

    print("[SURGERY] Loading data...")
    all_samples = []
    manifests = [
        PROJECT_ROOT / "data/hard_cases_plus_board30_valid_with_new6.csv",
        PROJECT_ROOT / "data/unified_training_manifest_v1.csv",
    ]
    for manifest in manifests:
        samples = load_manifest_samples(manifest, REPO_ROOT)
        print(f"  {manifest.name}: {len(samples)} samples")
        all_samples.extend(samples)
    print(f"[SURGERY] Total samples: {len(all_samples)}")

    np.random.seed(21)
    indices = np.random.permutation(len(all_samples))
    split = int(0.85 * len(all_samples))
    train_samples = [all_samples[i] for i in indices[:split]]
    val_samples = [all_samples[i] for i in indices[split:]]
    print(f"[SURGERY] Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = make_dataset(train_samples, batch_size=args.batch_size, training=True)
    val_ds = make_dataset(val_samples, batch_size=args.batch_size, training=False)

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mae",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    steps_per_epoch = max(1, len(train_samples) // args.batch_size)
    validation_steps = max(1, len(val_samples) // args.batch_size)
    print(f"[SURGERY] steps_per_epoch={steps_per_epoch} validation_steps={validation_steps}")
    print(f"[SURGERY] Training for {args.epochs} epochs...")
    history = new_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_mae", mode="min"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_mae", mode="min"),
        ],
        verbose=2,
    )

    out_dir = PROJECT_ROOT / "artifacts/training/head_surgery_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    new_model.save(out_dir / "model.keras")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist_dict, indent=2), encoding="utf-8")
    print(f"[SURGERY] Saved to {out_dir}")


if __name__ == "__main__":
    main()
