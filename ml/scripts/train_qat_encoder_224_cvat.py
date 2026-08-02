"""Train a 224x224 QAT encoder on the CVAT 9K ellipse dataset.

This is the AI-memory-proven int8-safe architecture (plain Conv+BN+ReLU,
single Dense(5, sigmoid) head). We previously verified that the existing
gauge_ellipse_qat_encoder_v1 (trained on different data) produces
varying int8 outputs on our 640x640 data resized to 224x224. This
script trains the same architecture on our 9K CVAT images to get a
better int8 model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.models_geometry_v2 import _build_qat_encoder  # noqa: E402
from tf_keras import layers, Model  # noqa: E402

IMAGE_SIZE = 224
SEED = 42
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_224g_cvat_v1"


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str):
    """Load a split, resizing 640x640 source images to 224x224."""
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / split / "labels.json").read_text())
    images = np.zeros((len(labels), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    targets = {
        "ellipse": np.zeros((len(labels), 5), dtype=np.float32),
    }
    img_dir = DATA_DIR / split / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = tf.image.resize(img[..., None], (IMAGE_SIZE, IMAGE_SIZE),
                                  method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["ellipse"][i] = [lab["cx"], lab["cy"], lab["rx"], lab["ry"], 1.0]
    return images, targets


def build_qat_encoder_224g():
    """224x224 QAT encoder with the AI-memory-proven architecture.

    Same pattern as gauge_ellipse_qat_encoder_v1: 5 stages of stride-2
    Conv+BN+ReLU, single Dense(5, sigmoid) head.
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    # _build_qat_encoder returns (bottleneck, skip_112, skip_56, skip_28, skip_14)
    out_tuple = _build_qat_encoder(
        inputs, width_multiplier=1.5, backbone_variant="standard", name_prefix="enc",
    )
    bottleneck = out_tuple[0]
    # 7x7x192 -> GAP -> Dense(128) -> Dense(5, sigmoid)
    x = layers.GlobalAveragePooling2D(name="gap")(bottleneck)
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="shared")(x)
    out = layers.Dense(5, activation="sigmoid", name="ellipse")(x)
    return Model(inputs=inputs, outputs=out, name="ellipse_qat_encoder_224g_cvat")


def main():
    configure_gpu()
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading data (resizing 640x640 to 224x224)...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    print("\nBuilding QAT encoder 224g (CVAT data)...")
    model = build_qat_encoder_224g()
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params/1e6:.2f}M)")

    # FP32 training
    losses = {"ellipse": keras.losses.Huber(delta=0.05)}
    loss_weights = {"ellipse": 1.0}
    steps_per_epoch = max(1, len(train_x) // 32)

    @keras.saving.register_keras_serializable()
    class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, peak_lr, total_steps, warmup_steps=0):
            super().__init__()
            self._peak, self._total, self._warmup = peak_lr, total_steps, warmup_steps
            self._cosine = keras.optimizers.schedules.CosineDecay(
                peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
            )
        def __call__(self, step):
            wf = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
            return tf.where(step < self._warmup, self._peak * wf, self._cosine(step - self._warmup))
        def get_config(self):
            return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}

    lr = WarmupCosineDecay(1e-3, steps_per_epoch * 40, steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=losses, loss_weights=loss_weights,
    )
    print("\nFP32 training (40 epochs)...")
    model.fit(
        train_x, train_y, batch_size=32, epochs=40,
        validation_data=(val_x, val_y),
        callbacks=[keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        )],
        verbose=2,
    )
    model.save(DEFAULT_OUTPUT / "model_fp32.keras")

    # QAT
    print("\nApplying QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_targets = {f"quant_ellipse": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_ellipse": v for k, v in val_y.items()}
    qat_loss = {"quant_ellipse": keras.losses.Huber(delta=0.05)}
    qat_weight = {"quant_ellipse": 1.0}
    qat_lr = WarmupCosineDecay(2e-4, steps_per_epoch * 15, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=qat_loss, loss_weights=qat_weight,
    )
    print("QAT training (15 epochs)...")
    qat_model.fit(
        train_x, qat_targets, batch_size=32, epochs=15,
        validation_data=(val_x, qat_val_targets),
        verbose=2,
    )
    qat_model.save(DEFAULT_OUTPUT / "model_qat.keras")

    # PTQ on FP32 too (sanity check, will likely collapse)
    print("\nExporting int8 TFLite (PTQ)...")
    def rep():
        rng = np.random.default_rng(42)
        for idx in rng.choice(len(train_x), size=min(512, len(train_x)), replace=False):
            yield [train_x[idx:idx+1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8  # try int8 output to match AI memory model
    blob = converter.convert()
    (DEFAULT_OUTPUT / "model_int8.tflite").write_bytes(blob)
    print(f"  TFLite size: {len(blob)/1024:.1f} KB")

    # Quick eval
    interp = tf.lite.Interpreter(model_path=str(DEFAULT_OUTPUT / "model_int8.tflite"))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    print("\n50-image variance check (TFLite int8):")
    vals = []
    for i in range(50):
        xq = np.clip(np.round(val_x[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(out_det["index"])
        dequant = (raw.astype(np.float32) - out_zp) * out_scale
        vals.append(dequant.flatten())
    arr = np.array(vals)
    print(f"  per-output std: {arr.std(axis=0)}")
    print(f"  first 3 predictions: {arr[:3].tolist()}")


if __name__ == "__main__":
    sys.exit(main())
