"""Train QAT encoder at 384x384 (or 512x512) grayscale.

The 224x224 model from the AI memory works in int8. The 640x640
model collapses. We test 384x384 and 512x512 to see if intermediate
resolutions preserve the int8 accuracy.

Activation budget at 384x384 with 5 stride-2 stages (channels 32-48-64-96-128):
  192x192x32 = 1.18 MB peak  <-- under 1.5 MB ✓
  96x96x48   = 442 KB
  48x48x64   = 147 KB
  24x24x96   =  55 KB
  12x12x128  =  18 KB

Activation budget at 512x512 with 5 stride-2 stages (channels 16-32-64-96-128):
  256x256x16 = 1.05 MB peak  <-- under 1.5 MB ✓
  128x128x32 = 524 KB
  64x64x64   = 262 KB
  32x32x96   =  98 KB
  16x16x128  =  33 KB

Both fit. We start with 384x384 and 512x512 in sequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from tf_keras import layers, Model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data" / "repvgg_ellipse"
SEED = 42


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _conv_bn_relu(x, filters, stride, name):
    """QAT-safe 3x3 conv block."""
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _channel_plan(input_size, alpha=1.5):
    """Pick channel widths based on the input size budget.

    384x384 (alpha=1.5): 32, 48, 64, 96, 128 (peak 1.18 MB)
    512x512 (alpha=1.0): 16, 32, 64, 96, 128 (peak 1.05 MB)
    640x640: needs stride-4 stem (peak 1.23 MB at stride-4)
    """
    if input_size <= 384:
        return {
            "s1": max(16, int(32 * alpha)),
            "s2": max(32, int(48 * alpha)),
            "s3": max(48, int(64 * alpha)),
            "s4": max(64, int(96 * alpha)),
            "s5": max(96, int(128 * alpha)),
        }
    elif input_size <= 512:
        return {
            "s1": max(8, int(16 * alpha)),
            "s2": max(16, int(32 * alpha)),
            "s3": max(32, int(64 * alpha)),
            "s4": max(48, int(96 * alpha)),
            "s5": max(96, int(128 * alpha)),
        }
    else:
        # 640x640 — use stride-4 stem (handled separately in the builder)
        return {
            "s1": max(24, int(32 * alpha)),
            "s2": max(48, int(72 * alpha)),
            "s3": max(64, int(96 * alpha)),
            "s4": max(96, int(144 * alpha)),
            "s5": max(128, int(192 * alpha)),
        }


def build_qat_encoder(input_size, alpha=1.5):
    """Build the AI-memory-proven QAT encoder at the given square input size.

    For 384x384 and 512x512: 5 stride-2 stages, all Conv+BN+ReLU.
    For 640x640: stride-4 stem + 4 stride-2 stages.
    """
    cp = _channel_plan(input_size, alpha)
    inputs = keras.Input(shape=(input_size, input_size, 1), name="image")

    if input_size >= 640:
        # Stride-4 stem to keep peak under 1.5 MB.
        x = _conv_bn_relu(inputs, filters=cp["s1"], stride=4, name="stem")
        # Remaining stages 1-4.
        stage_names = ["s2", "s3", "s4", "s5"]
        stage_filters = [cp["s2"], cp["s3"], cp["s4"], cp["s5"]]
    else:
        # Stride-2 stem: 5 stages total.
        x = _conv_bn_relu(inputs, filters=cp["s1"], stride=2, name="s1a")
        x = _conv_bn_relu(x, filters=cp["s1"], stride=1, name="s1b")
        stage_names = ["s2", "s3", "s4", "s5"]
        stage_filters = [cp["s2"], cp["s3"], cp["s4"], cp["s5"]]

    # Remaining stages: 2 Conv+BN+ReLU blocks each; first block downsamples.
    for name, filters in zip(stage_names, stage_filters):
        for i in range(2):
            stride = 2 if i == 0 else 1
            x = _conv_bn_relu(x, filters=filters, stride=stride, name=f"{name}b{i}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="shared")(x)
    out = layers.Dense(5, activation="sigmoid", name="ellipse")(x)
    return Model(inputs=inputs, outputs=out,
                name=f"ellipse_qat_encoder_{input_size}g_cvat_a{alpha}")


def _load_split(split, image_size):
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / split / "labels.json").read_text())
    images = np.zeros((len(labels), image_size, image_size, 1), dtype=np.float32)
    targets = {"ellipse": np.zeros((len(labels), 5), dtype=np.float32)}
    img_dir = DATA_DIR / split / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (image_size, image_size):
            img = tf.image.resize(img[..., None], (image_size, image_size),
                                  method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["ellipse"][i] = [lab["cx"], lab["cy"], lab["rx"], lab["ry"], 1.0]
    return images, targets


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
        return tf.where(step < self._warmup, self._peak * wf,
                        self._cosine(step - self._warmup))

    def get_config(self):
        return {"peak_lr": self._peak, "total_steps": self._total,
                "warmup_steps": self._warmup}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=384, choices=[224, 384, 512])
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--fp32-epochs", type=int, default=40)
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = ROOT / "artifacts" / f"gauge_ellipse_qat_encoder_{args.image_size}g_cvat_v1"
    args.output.mkdir(parents=True, exist_ok=True)

    configure_gpu()
    tf.random.set_seed(SEED); np.random.seed(SEED)

    print(f"Loading data (resizing 640x640 to {args.image_size}x{args.image_size})...")
    train_x, train_y = _load_split("train", args.image_size)
    val_x, val_y = _load_split("val", args.image_size)
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    print(f"\nBuilding QAT encoder {args.image_size}x{args.image_size} (alpha={args.alpha})...")
    model = build_qat_encoder(args.image_size, alpha=args.alpha)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params/1e6:.2f}M)")

    # FP32 training
    losses = {"ellipse": keras.losses.Huber(delta=0.05)}
    loss_weights = {"ellipse": 1.0}
    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(1e-3, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=losses, loss_weights=loss_weights,
    )
    print(f"\nFP32 training ({args.fp32_epochs} epochs)...")
    model.fit(
        train_x, train_y, batch_size=args.batch_size, epochs=args.fp32_epochs,
        validation_data=(val_x, val_y),
        callbacks=[keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        )],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    # QAT
    print("\nApplying QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_targets = {f"quant_ellipse": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_ellipse": v for k, v in val_y.items()}
    qat_loss = {"quant_ellipse": keras.losses.Huber(delta=0.05)}
    qat_weight = {"quant_ellipse": 1.0}
    qat_lr = WarmupCosineDecay(2e-4, steps_per_epoch * args.qat_epochs, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=qat_loss, loss_weights=qat_weight,
    )
    print(f"QAT training ({args.qat_epochs} epochs)...")
    qat_model.fit(
        train_x, qat_targets, batch_size=args.batch_size, epochs=args.qat_epochs,
        validation_data=(val_x, qat_val_targets),
        verbose=2,
    )
    qat_model.save(args.output / "model_qat.keras")

    # Export int8 TFLite
    print("\nExporting int8 TFLite...")
    def rep():
        rng = np.random.default_rng(42)
        for idx in rng.choice(len(train_x), size=min(512, len(train_x)), replace=False):
            yield [train_x[idx:idx+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    (args.output / "model_int8.tflite").write_bytes(blob)
    print(f"  TFLite size: {len(blob)/1024:.1f} KB")

    # Quick int8 variance check
    interp = tf.lite.Interpreter(model_path=str(args.output / "model_int8.tflite"))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    print(f"  Output scale={out_scale}, zp={out_zp}")

    print(f"\n50-image variance check (TFLite int8):")
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
    print(f"  per-output max-min: {arr.max(axis=0) - arr.min(axis=0)}")
    print(f"  first 5 predictions: {arr[:5].tolist()}")

    # Save report
    report = {
        "model": f"ellipse_qat_encoder_{args.image_size}g_cvat_a{args.alpha}",
        "input_shape": [args.image_size, args.image_size, 1],
        "alpha": args.alpha,
        "n_params": n_params,
        "int8_size_kb": round(len(blob) / 1024, 1),
        "per_output_std_int8": arr.std(axis=0).tolist(),
        "per_output_max_min_int8": (arr.max(axis=0) - arr.min(axis=0)).tolist(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
