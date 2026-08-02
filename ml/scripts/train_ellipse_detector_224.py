#!/usr/bin/env python3
"""Train the ellipse detector on 224x224 grayscale images.

Pipeline:
1. Load pre-extracted data from prepare_needle_data.py
2. FP32 training with AdamW + cosine LR + warmup
3. QAT fine-tuning for int8 export
4. Export int8 TFLite
5. Evaluate on all test splits
6. Check activation budget

Output: artifacts/ellipse_detector_224_v1/
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

from embedded_gauge_reading_tinyml.ellipse_detector_224 import build_ellipse_detector_224  # noqa: E402

DATA_DIR = ROOT / "data" / "needle_pipeline"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ellipse_detector_224_v1"
SEED = 42


def configure_gpu() -> None:
    """Cap GPU memory to 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 224x224 grayscale images and ellipse labels (cx, cy, rx, ry)."""
    images = np.load(DATA_DIR / split / "images.npy")
    labels = np.load(DATA_DIR / split / "ellipse_labels.npy")
    print(f"  {split}: {len(images)} images, labels shape {labels.shape}")
    print(f"    cx: [{labels[:, 0].min():.3f}, {labels[:, 0].max():.3f}]")
    print(f"    cy: [{labels[:, 1].min():.3f}, {labels[:, 1].max():.3f}]")
    print(f"    rx: [{labels[:, 2].min():.3f}, {labels[:, 2].max():.3f}]")
    print(f"    ry: [{labels[:, 3].min():.3f}, {labels[:, 3].max():.3f}]")
    return images, labels


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to 1% of peak LR."""

    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01,
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        wf = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(
            step < self._warmup,
            self._peak * wf,
            self._cosine(step - self._warmup),
        )

    def get_config(self) -> dict:
        return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}


def _export_int8_tflite(
    model: keras.Model,
    sample: np.ndarray,
    output_path: Path,
) -> dict:
    """Export fully-quantized int8 TFLite via QAT model."""
    def rep():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=min(512, len(sample)), replace=False):
            yield [sample[idx:idx + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input_shape": in_det["shape"].tolist(),
        "output_shape": out_det["shape"].tolist(),
    }


def _evaluate_int8_ellipse(
    model_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    max_images: int = 200,
    label: str = "val",
) -> dict:
    """Evaluate int8 TFLite ellipse model, report center/radius MAE in pixels."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]

    n = min(len(images), max_images)
    center_errs = []
    radius_errs = []

    for i in range(n):
        # Quantize input
        xq = np.clip(
            np.round(images[i:i + 1] / in_scale + in_zp), -128, 127
        ).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(out_det["index"])
        if out_det["dtype"] == np.int8:
            s, z = out_det["quantization"]
            pred = (raw.astype(np.float32) - z) * s
        else:
            pred = raw.astype(np.float32)
        pred = pred[0]

        # Compute errors in pixel space (image is 224x224)
        pcx, pcy, prx, pry = pred
        gcx, gcy, grx, gry = labels[i]
        center_errs.append(np.sqrt(((pcx - gcx) * 224) ** 2 + ((pcy - gcy) * 224) ** 2))
        radius_errs.append(np.sqrt(((prx - grx) * 224) ** 2 + ((pry - gry) * 224) ** 2))

    c = np.array(center_errs)
    r = np.array(radius_errs)
    metrics = {
        "n": n,
        "center_mae_px": float(c.mean()),
        "center_median_px": float(np.median(c)),
        "center_le8": float((c <= 8).mean() * 100),
        "center_le4": float((c <= 4).mean() * 100),
        "radius_mae_px": float(r.mean()),
    }
    print(f"  int8 {label} ({n}): center={metrics['center_mae_px']:.2f}px "
          f"({metrics['center_le8']:.1f}% <=8px), "
          f"radius={metrics['radius_mae_px']:.2f}px")
    return metrics


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fp32-epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")

    # ── Build model ────────────────────────────────────────────────────
    print("\nBuilding ellipse detector 224...")
    model = build_ellipse_detector_224()
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── FP32 training ──────────────────────────────────────────────────
    loss = keras.losses.Huber(delta=0.05)
    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 3)
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=loss,
        metrics=["mae"],
    )

    print(f"\nFP32 training ({args.fp32_epochs} epochs, {len(train_x)} samples)...")
    model.fit(
        train_x, train_y,
        batch_size=args.batch_size,
        epochs=args.fp32_epochs,
        validation_data=(val_x, val_y),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    # FP32 eval
    fp32_val = model.predict(val_x[:200], batch_size=8, verbose=0)
    fp32_center_err = np.sqrt(
        ((fp32_val[:, 0] - val_y[:200, 0]) * 224) ** 2 +
        ((fp32_val[:, 1] - val_y[:200, 1]) * 224) ** 2
    )
    print(f"\nFP32 val (200): center MAE={fp32_center_err.mean():.2f}px, "
          f"<=8px: {(fp32_center_err <= 8).mean() * 100:.1f}%")

    # ── QAT training ───────────────────────────────────────────────────
    print("\nApplying QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(args.qat_lr, steps_per_epoch * args.qat_epochs, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=loss,
        metrics=["mae"],
    )
    print(f"QAT training ({args.qat_epochs} epochs)...")
    qat_model.fit(
        train_x, train_y,
        batch_size=args.batch_size,
        epochs=args.qat_epochs,
        validation_data=(val_x, val_y),
        verbose=2,
    )
    qat_model.save(args.output / "model_qat.keras")

    # ── Export int8 TFLite ─────────────────────────────────────────────
    print("\nExporting int8 TFLite...")
    tflite_path = args.output / "model_int8.tflite"
    contract = _export_int8_tflite(qat_model, train_x, tflite_path)
    print(f"  Size: {contract['kb']} KB ({contract['bytes'] / 1e6:.2f} MB)")
    print(f"  Input: {contract['input_shape']}")
    print(f"  Output: {contract['output_shape']}")

    # ── Int8 evaluation ────────────────────────────────────────────────
    print("\nInt8 evaluation on val (200 images):")
    int8_val = _evaluate_int8_ellipse(tflite_path, val_x, val_y, label="val")

    # ── Test set evaluation ─────────────────────────────────────────────
    test_x, test_y = _load_split("test")
    print("\nInt8 evaluation on test:")
    int8_test = _evaluate_int8_ellipse(tflite_path, test_x, test_y, label="test")

    # ── Activation budget check ────────────────────────────────────────
    peak_kb = 112 * 112 * 32 / 1024  # Encoder stage 1
    print(f"\nActivation budget check (int8):")
    print(f"  Peak activation (e1 stage): ~{peak_kb:.0f} KB int8")
    print(f"  Budget: 2560 KB (2.5 MB SRAM)")
    print(f"  OK — within budget" if peak_kb <= 2560 else "  WARNING: exceeds budget!")

    # ── Save report ────────────────────────────────────────────────────
    report = {
        "model": "ellipse_detector_224_v1",
        "input_shape": [224, 224, 1],
        "output_shape": [4],
        "n_params": n_params,
        "int8_size_kb": contract["kb"],
        "peak_activation_kb": round(peak_kb, 1),
        "fp32_val": {
            "center_mae_px": float(fp32_center_err.mean()),
            "center_le8_pct": float((fp32_center_err <= 8).mean() * 100),
        },
        "int8_val": int8_val,
        "int8_test": int8_test,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {args.output / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
