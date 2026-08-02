#!/usr/bin/env python3
"""Train the QAT-encoder 640x640 grayscale ellipse detector.

Pipeline:
1. FP32 training on the staged repvgg_ellipse data (60 epochs, AdamW).
2. Apply tfmot.quantize_model() (QAT on the BN-equipped encoder).
3. QAT fine-tune (15 epochs, lower LR).
4. Export int8 TFLite with a representative dataset.
5. Activation budget check (peak intermediate < 1.5 MB int8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from tf_keras import layers

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.qat_encoder_640g import build_qat_encoder_640g  # noqa: E402


IMAGE_SIZE = 640
SEED = 42
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_640g_v1"


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / split / "labels.json").read_text())
    images = np.zeros((len(labels), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    targets = {
        "center_xy": np.zeros((len(labels), 2), dtype=np.float32),
        "radius_xy": np.zeros((len(labels), 2), dtype=np.float32),
        "confidence": np.ones((len(labels), 1), dtype=np.float32),
    }
    img_dir = DATA_DIR / split / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = tf.image.resize(img[..., None], (IMAGE_SIZE, IMAGE_SIZE),
                                  method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["center_xy"][i] = [lab["cx"], lab["cy"]]
        targets["radius_xy"][i] = [lab["rx"], lab["ry"]]
    return images, targets


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    @keras.saving.register_keras_serializable()
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
        )

    def __call__(self, step):
        warmup_frac = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(step < self._warmup, self._peak * warmup_frac,
                        self._cosine(step - self._warmup))

    def get_config(self):
        return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}


def _export_int8_tflite(model, sample, output_path):
    """Export a fully-quantized int8 TFLite model with a representative dataset."""
    def rep():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=min(512, len(sample)), replace=False):
            yield [sample[idx][None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    blob = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    return {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input": interp.get_input_details()[0]["shape"].tolist(),
        "outputs": [
            {"shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
            for d in interp.get_output_details()
        ],
    }


def _measure_peak_activation(model):
    from tensorflow.python.framework import convert_to_constants
    func = tf.function(lambda x: model(x, training=False))
    concrete = func.get_concrete_function(
        tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
    )
    frozen = convert_to_constants.convert_variables_to_constants_v2(concrete)

    peak_bytes = 0
    peak_name = ""
    for op in frozen.graph.get_operations():
        for out in op.outputs:
            shape = out.shape
            if not shape.is_fully_defined():
                continue
            n = int(np.prod(shape)) * out.dtype.size
            if n > peak_bytes:
                peak_bytes = n
                peak_name = f"{op.name}:{shape}"
    int8_peak_bytes = peak_bytes // 4  # int8 has 1 byte/elem
    return {
        "peak_bytes_fp32": int(peak_bytes),
        "peak_mb_fp32": round(peak_bytes / 1e6, 3),
        "peak_bytes_int8": int(int8_peak_bytes),
        "peak_mb_int8": round(int8_peak_bytes / 1e6, 3),
        "peak_tensor": peak_name,
        "within_int8_budget_1_5mb": bool(int8_peak_bytes <= 1.5 * 1024 * 1024),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--fp32-epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    print(f"\nBuilding QAT encoder 640g (alpha={args.alpha})...")
    model = build_qat_encoder_640g(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), alpha=args.alpha)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f} MB int8)")

    losses = {
        "center_xy": keras.losses.Huber(delta=0.05),
        "radius_xy": keras.losses.Huber(delta=0.05),
        "confidence": keras.losses.Huber(delta=0.05),
    }
    loss_weights = {"center_xy": 1.0, "radius_xy": 3.0, "confidence": 0.1}

    # FP32 training.
    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=losses,
        loss_weights=loss_weights,
    )
    print(f"\nFP32 training for {args.fp32_epochs} epochs...")
    model.fit(
        train_x, train_y,
        batch_size=args.batch_size,
        epochs=args.fp32_epochs,
        validation_data=(val_x, val_y),
        callbacks=[keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        )],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    # QAT.
    print("\nApplying tfmot QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_output_names = [o.name.split("/")[0] for o in qat_model.outputs]
    print(f"  QAT output names: {qat_output_names}")
    qat_losses = {qn: losses[old] for qn, old in zip(qat_output_names, losses)}
    qat_loss_weights = {qn: loss_weights[old] for qn, old in zip(qat_output_names, loss_weights)}
    qat_targets = {f"quant_{k}": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_{k}": v for k, v in val_y.items()}

    qat_lr = WarmupCosineDecay(args.qat_lr, steps_per_epoch * args.qat_epochs, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=qat_losses,
        loss_weights=qat_loss_weights,
    )
    print(f"QAT training for {args.qat_epochs} epochs...")
    qat_model.fit(
        train_x, qat_targets,
        batch_size=args.batch_size,
        epochs=args.qat_epochs,
        validation_data=(val_x, qat_val_targets),
        verbose=2,
    )
    qat_model.save(args.output / "model_qat.keras")

    # Activation budget.
    print("\nMeasuring peak activation...")
    peak = _measure_peak_activation(model)
    print(f"  Peak activation (fp32 graph): {peak['peak_mb_fp32']} MB")
    print(f"  Peak activation (int8 deployment): {peak['peak_mb_int8']} MB")
    print(f"  Peak tensor: {peak['peak_tensor']}")
    if not peak["within_int8_budget_1_5mb"]:
        print(f"  WARNING: peak int8 activation {peak['peak_mb_int8']} MB exceeds 1.5 MB budget")

    # Export int8.
    print("\nExporting int8 TFLite...")
    contract = _export_int8_tflite(qat_model, train_x, args.output / "model_int8.tflite")
    print(f"  TFLite size: {contract['kb']} KB")
    print(f"  Input: {contract['input']}")

    # Report.
    report = {
        "model": f"ellipse_qat_encoder_640g_a{args.alpha}",
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 1],
        "alpha": args.alpha,
        "n_params": n_params,
        "int8_size_mb": round(contract["bytes"] / 1e6, 3),
        "peak_activation_mb_int8": peak["peak_mb_int8"],
        "peak_tensor": peak["peak_tensor"],
        "tflite": contract,
        "fp32_epochs": args.fp32_epochs,
        "qat_epochs": args.qat_epochs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("\nReport:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
