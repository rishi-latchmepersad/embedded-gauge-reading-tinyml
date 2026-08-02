#!/usr/bin/env python3
"""Convert the multi-branch model to int8 TFLite using PTQ.

Why this works (and fused + QAT doesn't):
- The fused model has Conv+ReLU per block (no BN). Without BN the
  activation ranges drift across a wide range that depends on the input.
  The TFLite calibrator picks a single (min, max) from the representative
  dataset and bakes it into a static int8 grid. Without BN to normalize
  the inputs, the chosen grid is too coarse and the int8 output collapses
  to a constant (the bias-only failure mode in
  docs/ai-memory/lessons-learned/2026-07-23-qat-safe-architecture.md).
- The multi-branch model has BN after every conv. BN normalises each
  layer's activations to roughly zero mean and unit variance, so the
  representative dataset sees a tight, repeatable range. PTQ can
  calibrate the int8 grid correctly.

The TFLite converter can fold the BN into the conv at compile time, so
the deployed NPU graph is still a single 3x3 conv per block (RepVGG's
whole point). The TFLite file is slightly larger (~10%) because the
folded weights live as separate tensors in the file, but the runtime
graph is the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.ellipse_repvgg import build_repvgg_ellipse_multi  # noqa: E402


IMAGE_SIZE = 640
SEED = 42
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
ARTIFACTS = ROOT / "artifacts" / "gauge_ellipse_repvgg_640g_v1"


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str):
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
    def __init__(self, peak_lr, total_steps, warmup_steps=0):
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


def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()

    print("Loading data...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    print("\nLoading multi-branch model...")
    custom = {"WarmupCosineDecay": WarmupCosineDecay}
    multi = keras.models.load_model(
        ARTIFACTS / "model_fp32_multi.keras", custom_objects=custom,
    )
    print("  multi-branch loaded")

    # Sanity: does it produce varying output?
    print("\nMulti-branch on 5 sample inputs:")
    sample5 = train_x[:5]
    preds = multi.predict(sample5, verbose=0)
    for name, p in zip(["center_xy", "radius_xy", "confidence"], preds):
        print(f"  {name}: std={p.std():.6f}, max-min={p.max() - p.min():.6f}")

    # Export int8 TFLite with PTQ.
    def rep_dataset():
        rng = np.random.default_rng(SEED)
        # Use 256 random training images for calibration.
        for idx in rng.choice(len(train_x), size=256, replace=False):
            yield [train_x[idx:idx+1].astype(np.float32)]

    print("\nConverting to int8 TFLite (PTQ)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(multi)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    blob = converter.convert()
    out_path = ARTIFACTS / "model_int8_multibranch_ptq.tflite"
    out_path.write_bytes(blob)
    print(f"  TFLite size: {len(blob) / 1024:.1f} KB ({len(blob) / 1e6:.2f} MB)")
    print(f"  Saved to: {out_path}")

    # Sanity: does the TFLite produce varying output?
    print("\nTFLite model on 5 sample inputs (post-conversion):")
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]
    print(f"  Input: scale={in_scale}, zp={in_zp}")
    for d in out_dets:
        s, z = d["quantization"]
        print(f"  Output {d['name']}: dtype={d['dtype']}, scale={s}, zp={z}")

    all_out = {tuple(d["shape"]): [] for d in out_dets}
    for img in sample5:
        xq = np.clip(np.round(img[None] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for d in out_dets:
            raw = interp.get_tensor(d["index"])
            s, z = d["quantization"]
            if d["dtype"] == np.int8:
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            all_out[tuple(d["shape"])].append(dequant.flatten())
    for shape, outs in all_out.items():
        arr = np.array(outs)
        print(f"  {shape}: std={arr.std():.6f}, max-min={arr.max() - arr.min():.6f}")
        print(f"    values: {arr.tolist()}")


if __name__ == "__main__":
    sys.exit(main())
