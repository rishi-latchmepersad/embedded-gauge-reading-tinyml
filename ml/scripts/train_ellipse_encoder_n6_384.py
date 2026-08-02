#!/usr/bin/env python3
"""Train the conservative ellipse encoder for STM32N6 (no HyperRAM).

Matches deepseek_ellipse_encoder_n6_384_retrain_handoff.md:
- Input: 384x384x1 grayscale, int8, scale=1/255, zp=-128.
- Output: int8(1,5) [cx, cy, rx, ry, confidence], sigmoid, single Dense head.
- QAT only; no PTQ, no float16.
- Channels: [16,16,24,24,32,32,48,48,64,64].
- Peak int8 activation: 192x192x16 = 590 KB (under 1.5 MB).
- Cube.AI gate: no HyperRAM allocations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.ellipse_encoder_n6_384 import build_ellipse_encoder_n6_384  # noqa

DATA_DIR = ROOT / "data" / "repvgg_ellipse"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_384g_cvat_v2"
IMAGE_SIZE = 384
SEED = 42


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str):
    """Load 640x640 data resized to 384x384 with multi-output targets."""
    labels = json.loads((DATA_DIR / split / "labels.json").read_text())
    n = len(labels)
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    targets = {
        "center_xy": np.zeros((n, 2), dtype=np.float32),
        "radius_xy": np.zeros((n, 2), dtype=np.float32),
        "confidence": np.ones((n, 1), dtype=np.float32),
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


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps=0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
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


def _export_int8_tflite(model, sample, output_path):
    """Export fully-quantized int8 TFLite with multi-output heads."""
    def rep():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=min(512, len(sample)), replace=False):
            yield [sample[idx:idx+1].astype(np.float32)]

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
    out_dets = interp.get_output_details()

    return {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input_shape": in_det["shape"].tolist(),
        "input_dtype": str(in_det["dtype"]),
        "input_scale": float(in_det["quantization"][0]),
        "input_zp": int(in_det["quantization"][1]),
        "outputs": [
            {
                "shape": d["shape"].tolist(),
                "dtype": str(d["dtype"]),
                "scale": float(d["quantization"][0]),
                "zp": int(d["quantization"][1]),
            }
            for d in out_dets
        ],
    }


def _parity_check(fp32_model, tflite_path, sample_x, sample_y):
    """Run Keras fp32 and TFLite int8 on the same sample, report per-output differences."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]

    # Keras outputs.
    y_fp32 = fp32_model.predict(sample_x, batch_size=8, verbose=0)

    # TFLite: map outputs by dimension to match the Keras order
    # (center_xy=2, radius_xy=2, confidence=1).
    keras_by_dim = {y.shape[1]: y for y in y_fp32}
    tflite_by_dim = {}
    for d in out_dets:
        dim = int(d["shape"][-1])
        tflite_by_dim[dim] = d

    y_int8 = {dim: np.zeros_like(keras_by_dim[dim]) for dim in tflite_by_dim}
    for i in range(len(sample_x)):
        xq = np.clip(np.round(sample_x[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for dim, det in tflite_by_dim.items():
            raw = interp.get_tensor(det["index"])
            if det["dtype"] == np.int8:
                s, z = det["quantization"]
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            y_int8[dim][i] = dequant[0]

    diffs = {}
    for name, y_k in zip(["center_xy", "radius_xy", "confidence"], y_fp32):
        dim = int(y_k.shape[1])
        d = np.max(np.abs(y_k - y_int8[dim]))
        diffs[name] = float(d)
    return {
        "max_abs_diff": float(max(diffs.values())),
        "per_output_max_abs_diff": diffs,
        "parity_ok": bool(max(diffs.values()) < 0.02),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channels", type=int, nargs=10,
                        default=[16, 16, 24, 24, 32, 32, 48, 48, 64, 64])
    parser.add_argument("--fp32-epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data (resizing 640x640 to 384x384)...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    print(f"  train: {train_x.shape}")
    print(f"  val:   {val_x.shape}")

    # Verify target distribution.
    for name in ["center_xy", "radius_xy", "confidence"]:
        arr = train_y[name]
        print(f"  {name}: train min={arr.min():.3f}, max={arr.max():.3f}, "
              f"val min={val_y[name].min():.3f}, max={val_y[name].max():.3f}")

    print(f"\nBuilding conservative encoder n6_384 (channels={args.channels})...")
    model = build_ellipse_encoder_n6_384(channels=args.channels)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Estimate peak int8 activation.
    peak_pixels = 192 * 192 * args.channels[0]
    peak_kb = peak_pixels / 1024
    print(f"Est. peak int8 activation: {peak_kb:.0f} KB (192x192x{args.channels[0]})")

    loss = {
        "center_xy": keras.losses.Huber(delta=0.05),
        "radius_xy": keras.losses.Huber(delta=0.05),
        "confidence": keras.losses.Huber(delta=0.05),
    }
    loss_weights = {"center_xy": 1.0, "radius_xy": 3.0, "confidence": 0.1}
    steps_per_epoch = max(1, len(train_x) // args.batch_size)

    # FP32 training.
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 3)
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
                  loss=loss, loss_weights=loss_weights)
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

    # FP32 eval.
    preds = model.predict(val_x[:200], batch_size=8, verbose=0)
    center_err_px = np.sqrt(
        (preds[0][:, 0] * 384 - val_y["center_xy"][:200, 0] * 384) ** 2 +
        (preds[0][:, 1] * 384 - val_y["center_xy"][:200, 1] * 384) ** 2
    )
    radius_err_px = np.sqrt(
        (preds[1][:, 0] * 384 - val_y["radius_xy"][:200, 0] * 384) ** 2 +
        (preds[1][:, 1] * 384 - val_y["radius_xy"][:200, 1] * 384) ** 2
    )
    print(f"\nFP32 eval on first 200 val images:")
    print(f"  Center MAE: {center_err_px.mean():.2f} px")
    print(f"  Center % ≤8px: {(center_err_px <= 8).mean() * 100:.1f}%")
    print(f"  Radius MAE: {radius_err_px.mean():.2f} px")

    # QAT.
    print("\nApplying QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_output_names = [o.name.split("/")[0] for o in qat_model.outputs]
    print(f"  QAT output names: {qat_output_names}")
    qat_losses = {qn: loss[old] for qn, old in zip(qat_output_names, loss)}
    qat_loss_weights = {qn: loss_weights[old] for qn, old in zip(qat_output_names, loss_weights)}
    qat_targets = {f"quant_{k}": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_{k}": v for k, v in val_y.items()}
    qat_lr = WarmupCosineDecay(args.qat_lr, steps_per_epoch * args.qat_epochs, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=qat_losses, loss_weights=qat_loss_weights,
    )
    print(f"QAT training ({args.qat_epochs} epochs)...")
    qat_model.fit(
        train_x, qat_targets, batch_size=args.batch_size,
        epochs=args.qat_epochs,
        validation_data=(val_x, qat_val_targets),
        verbose=2,
    )
    qat_model.save(args.output / "model_qat.keras")

    # Export int8 TFLite.
    print("\nExporting int8 TFLite (QAT → int8)...")
    tflite_path = args.output / "model_int8.tflite"
    contract = _export_int8_tflite(qat_model, train_x, tflite_path)
    print(f"  Size: {contract['kb']} KB ({contract['bytes'] / 1e6:.2f} MB)")
    print(f"  Input:  {contract['input_shape']}, dtype={contract['input_dtype']}, "
          f"scale={contract['input_scale']}, zp={contract['input_zp']}")
    for i, out in enumerate(contract["outputs"]):
        print(f"  Output {i}: shape={out['shape']}, dtype={out['dtype']}, "
              f"scale={out['scale']}, zp={out['zp']}")

    # Parity check.
    print("\nKeras vs TFLite parity check (200 val images)...")
    parity = _parity_check(model, tflite_path, val_x[:200], {
        k: v[:200] for k, v in val_y.items()
    })
    print(f"  Max abs diff: {parity['max_abs_diff']:.5f}")
    print(f"  Per-output: {json.dumps(parity['per_output_max_abs_diff'])}")
    print(f"  Parity OK (< 0.02): {parity['parity_ok']}")

    # TFLite int8 eval.
    print("\nTFLite int8 eval on first 200 val images...")
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]

    # Map TFLite outputs by dimension.
    tflite_by_dim = {}
    for d in out_dets:
        dim = int(d["shape"][-1])
        tflite_by_dim[dim] = d

    int8_center = np.zeros((200, 2), dtype=np.float32)
    int8_radius = np.zeros((200, 2), dtype=np.float32)
    for i in range(200):
        xq = np.clip(np.round(val_x[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for dim, det in tflite_by_dim.items():
            raw = interp.get_tensor(det["index"])
            if det["dtype"] == np.int8:
                s, z = det["quantization"]
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            dequant = dequant[0]
            if dim == 2:
                # Determine center vs radius by comparing index order.
                if det["index"] < [dd["index"] for dd in out_dets if int(dd["shape"][-1]) == 2][1]:
                    int8_center[i] = dequant
                else:
                    int8_radius[i] = dequant

    # Safety: ensure center and radius got assigned.
    if int8_center.sum() == 0:
        # First 2-dim output is center (matching Keras output order).
        dim2_outputs = [d for d in out_dets if int(d["shape"][-1]) == 2]
        for i in range(200):
            xq = np.clip(np.round(val_x[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], xq)
            interp.invoke()
            for j, det in enumerate(dim2_outputs):
                raw = interp.get_tensor(det["index"])
                if det["dtype"] == np.int8:
                    s, z = det["quantization"]
                    dequant = (raw.astype(np.float32) - z) * s
                else:
                    dequant = raw.astype(np.float32)
                (int8_center if j == 0 else int8_radius)[i] = dequant[0]

    center_err_px_i = np.sqrt(
        (int8_center[:, 0] * 384 - val_y["center_xy"][:200, 0] * 384) ** 2 +
        (int8_center[:, 1] * 384 - val_y["center_xy"][:200, 1] * 384) ** 2
    )
    radius_err_px_i = np.sqrt(
        (int8_radius[:, 0] * 384 - val_y["radius_xy"][:200, 0] * 384) ** 2 +
        (int8_radius[:, 1] * 384 - val_y["radius_xy"][:200, 1] * 384) ** 2
    )
    print(f"  Center MAE: {center_err_px_i.mean():.2f} px")
    print(f"  Center % ≤8px: {(center_err_px_i <= 8).mean() * 100:.1f}%")
    print(f"  Radius MAE: {radius_err_px_i.mean():.2f} px")

    # Metadata.
    metadata = {
        "model": f"ellipse_encoder_n6_384_c{'_'.join(str(c) for c in args.channels)}",
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 1],
        "output_shape": [5],
        "channels": args.channels,
        "peak_activation_kb": round(peak_kb, 1),
        "n_params": n_params,
        "fp32_epochs": args.fp32_epochs,
        "qat_epochs": args.qat_epochs,
        "tflite_contract": contract,
        "parity_check": parity,
        "fp32_eval_200": {
            "center_mae_px": float(center_err_px.mean()),
            "center_pct_le_8px": float((center_err_px <= 8).mean()),
            "radius_mae_px": float(radius_err_px.mean()),
        },
        "int8_eval_200": {
            "center_mae_px": float(center_err_px_i.mean()),
            "center_pct_le_8px": float((center_err_px_i <= 8).mean()),
            "radius_mae_px": float(radius_err_px_i.mean()),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved to {args.output / 'metadata.json'}")


if __name__ == "__main__":
    sys.exit(main())
