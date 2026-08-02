#!/usr/bin/env python3
"""QAT + int8 export on the multi-branch model (with BatchNorm present).

The fused model has Conv+ReLU per block — no BatchNorm. tfmot collapses
its int8 output to a constant because the activation ranges drift
without BN to normalize them. The fix: apply QAT to the MULTI-branch
model (which has BN) and convert it to TFLite. The deployed NPU can
fuse the BN into the conv at compile time, so the on-device graph is
still a single 3x3 conv per block.

The TFLite model will be a bit larger than a true RepVGG-fused int8
(3.6 MB vs 3.2 MB) because it carries the BN scale/shift in addition
to the conv weights. For the 1.5 MB peak activation budget this is
fine — the peak is the same, only the weight count is slightly higher.
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

from embedded_gauge_reading_tinyml.ellipse_repvgg import (  # noqa: E402
    build_repvgg_ellipse_multi,
    _channel_plan,
)


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
    def rep():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=min(256, len(sample)), replace=False):
            yield [sample[idx][idx+1] if False else sample[idx:idx+1].astype(np.float32)]
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    args = parser.parse_args()

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

    # Sanity check: does the multi-branch model produce varying output?
    print("\nMulti-branch model on 5 sample inputs (sanity check):")
    sample5 = train_x[:5]
    preds = multi.predict(sample5, verbose=0)
    for name, p in zip(["center_xy", "radius_xy", "confidence"], preds):
        print(f"  {name}: std={p.std():.6f}, max-min={p.max() - p.min():.6f}")

    print("\nApplying tfmot QAT to multi-branch model...")
    # tfmot's default quantize_apply does not know how to wrap BatchNorm
    # layers. We annotate the model manually: default 8-bit QAT for Conv2D
    # and Dense, NoOp for BatchNorm (the BN gets folded into the conv
    # weight by the TFLite converter anyway, so we don't need fake-quant
    # nodes for it).
    from tensorflow_model_optimization.quantization.keras import (
        quantize_annotate_model,
        quantize_apply,
        QuantizeConfig,
    )
    from tensorflow_model_optimization.quantization.keras.default_8bit import Default8BitQuantizeRegistry

    class _NoOpQuantizeConfig(QuantizeConfig):
        """Pass-through quantize config: leaves the wrapped layer unchanged.

        Used for BatchNorm so the QAT graph contains Conv+BN+ReLU blocks
        but the BN itself does not get fake-quant wrappers. The TFLite
        converter folds the BN into the conv anyway, so the deployed
        graph is still a single 3x3 conv per block.
        """
        def get_weights_and_quantizers(self, layer):
            return []
        def get_activations_and_quantizers(self, layer):
            return []
        def set_quantize_weights(self, layer, quantize_weights):
            pass
        def set_quantize_activations(self, layer, quantize_activations):
            pass
        def get_output_quantizers(self, layer):
            return []
        def get_config(self):
            return {}

    Default8BitQuantizeRegistry._QUANTIZATION_CONFIGS[
        "BatchNormalization"
    ] = _NoOpQuantizeConfig

    annotated = quantize_annotate_model(multi)
    qat_model = quantize_apply(annotated)
    print("  QAT applied. Check varying output:")
    preds = qat_model.predict(sample5, verbose=0)
    for name, p in zip(["center_xy", "radius_xy", "confidence"], preds):
        print(f"  {name}: std={p.std():.6f}, max-min={p.max() - p.min():.6f}")

    if all(p.std() < 0.001 for p in preds):
        print("\nERROR: QAT model still produces constant output.")
        print("  This is unexpected for the multi-branch model (which has BN).")
        return 1

    # Compile and fine-tune the QAT model.
    losses = {
        "center_xy": keras.losses.Huber(delta=0.05),
        "radius_xy": keras.losses.Huber(delta=0.05),
        "confidence": keras.losses.Huber(delta=0.05),
    }
    loss_weights = {"center_xy": 1.0, "radius_xy": 3.0, "confidence": 0.1}
    qat_output_names = [o.name.split("/")[0] for o in qat_model.outputs]
    qat_losses = {qn: losses[old] for qn, old in zip(qat_output_names, losses)}
    qat_loss_weights = {qn: loss_weights[old] for qn, old in zip(qat_output_names, loss_weights)}
    qat_targets = {f"quant_{k}": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_{k}": v for k, v in val_y.items()}

    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    qat_lr = WarmupCosineDecay(args.qat_lr, steps_per_epoch * args.qat_epochs, steps_per_epoch)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(qat_lr, weight_decay=1e-5),
        loss=qat_losses,
        loss_weights=qat_loss_weights,
    )
    print(f"\nQAT training for {args.qat_epochs} epochs...")
    qat_model.fit(
        train_x, qat_targets,
        batch_size=args.batch_size,
        epochs=args.qat_epochs,
        validation_data=(val_x, qat_val_targets),
        verbose=2,
    )

    print("\nExporting int8 TFLite...")
    contract = _export_int8_tflite(qat_model, train_x, ARTIFACTS / "model_int8_multibranch.tflite")
    print(f"  TFLite size: {contract['kb']} KB")
    print(f"  Input: {contract['input']}")

    # Sanity check the TFLite output varies
    print("\nTFLite model on 5 sample inputs:")
    interp = tf.lite.Interpreter(model_path=str(ARTIFACTS / "model_int8_multibranch.tflite"))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]
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

    print(f"\nDone. TFLite model at {ARTIFACTS / 'model_int8_multibranch.tflite'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
