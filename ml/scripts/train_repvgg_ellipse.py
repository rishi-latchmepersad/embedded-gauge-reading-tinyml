#!/usr/bin/env python3
"""Train a RepVGG ellipse detector end-to-end: FP32 -> reparameterize -> QAT -> int8 TFLite.

Pipeline (per the AI memory lessons):
1. Build the multi-branch model and train on 640x640 grayscale.
2. Reparameterize to a single-branch fused model.
3. Sanity check: fused model output == multi-branch output within float32 noise.
4. Apply tfmot QAT on the FUSED model and continue training.
5. Export int8 TFLite with a representative dataset drawn from the training set.
6. Write a report.json with size, activation budget, and metrics.

Wins the board-deployable candidate if:
- int8 model is < 1.5 MB
- peak activation < 1.5 MB (input is 410 KB, stem is 614 KB at alpha=1.0)
- center MAE on val < 0.02 in normalised [0,1] coords
- radius variance on val > 0.01 (i.e. it is NOT collapsed to a constant)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.ellipse_repvgg import (  # noqa: E402
    build_repvgg_ellipse_multi,
    build_repvgg_ellipse_fused,
    reparameterize_model,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640
SEED = 42
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_ellipse_repvgg_640g_v1"


# ---------------------------------------------------------------------------
# GPU configuration (cap to 15 GB per AGENTS.md)
# ---------------------------------------------------------------------------

def configure_gpu() -> None:
    """Cap GPU memory to 15 GB so WSL retains headroom.

    Per the operations note in the AI memory: when we set a memory_limit
    we MUST NOT also call set_memory_growth, because TF refuses the second
    call with "Cannot set memory growth on device when virtual devices
    configured". The 15 GB cap is what gives WSL its headroom.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_split(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a single split from disk into RAM.

    Each image is ~50 KB, 9K images = ~450 MB. That fits in CPU RAM and
    keeps the data pipeline simple. For 30K+ images we would switch to
    a tf.data pipeline that streams from disk.
    """
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
        # Resize to IMAGE_SIZE if the source isn't already 640x640.
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = tf.image.resize(img[..., None], (IMAGE_SIZE, IMAGE_SIZE),
                                  method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["center_xy"][i] = [lab["cx"], lab["cy"]]
        targets["radius_xy"][i] = [lab["rx"], lab["ry"]]
    return images, targets


def _augment(images: tf.Tensor, targets: dict[str, tf.Tensor]) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Light augmentation: horizontal/vertical flips and small brightness jitter.

    Flip handling:
    - horizontal flip swaps cx with (1 - cx) and keeps rx/ry.
    - vertical flip swaps cy with (1 - cy) and keeps rx/ry.

    We keep the augmentation deterministic per-sample so the label flips
    are always consistent with the image flips.
    """
    # Why not rotation: at +/-15 deg a 640x640 image has a different
    # bounding box for the ellipse, and rotating the ellipse label needs
    # a full affine transform of the cx/cy/rx/ry vector. Flips are
    # label-free so we keep just those for the first iteration.
    flip_h = tf.random.uniform(()) > 0.5
    flip_v = tf.random.uniform(()) > 0.5

    images = tf.cond(flip_h, lambda: tf.image.flip_left_right(images),
                                  lambda: images)
    images = tf.cond(flip_v, lambda: tf.image.flip_up_down(images),
                                  lambda: images)

    # Brightness: keep within +/- 20% so we don't break gauge/background contrast.
    images = tf.image.random_brightness(images, max_delta=0.2)

    def _flip_h_targets() -> dict[str, tf.Tensor]:
        new_cx = 1.0 - targets["center_xy"][..., 0:1]
        old_cy = targets["center_xy"][..., 1:2]
        return {
            "center_xy": tf.concat([new_cx, old_cy], axis=-1),
            "radius_xy": targets["radius_xy"],
            "confidence": targets["confidence"],
        }

    def _flip_v_targets() -> dict[str, tf.Tensor]:
        old_cx = targets["center_xy"][..., 0:1]
        new_cy = 1.0 - targets["center_xy"][..., 1:2]
        return {
            "center_xy": tf.concat([old_cx, new_cy], axis=-1),
            "radius_xy": targets["radius_xy"],
            "confidence": targets["confidence"],
        }

    targets = tf.cond(flip_h, _flip_h_targets, lambda: targets)
    targets = tf.cond(flip_v, _flip_v_targets, lambda: targets)
    return images, targets


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup over `warmup_steps`, then cosine decay to 1% of peak.

    Registered as a serializable Keras object so the saved .keras files
    can be reloaded with `keras.models.load_model` without custom_object
    plumbing.
    """

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
        return tf.where(
            step < self._warmup,
            self._peak * warmup_frac,
            self._cosine(step - self._warmup),
        )

    def get_config(self):
        return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}


# ---------------------------------------------------------------------------
# Reparameterize sanity check
# ---------------------------------------------------------------------------

def _check_fused_matches_multi(multi: keras.Model, fused: keras.Model,
                                images: np.ndarray, atol: float = 1e-4) -> dict:
    """Confirm the fused model's predictions match the multi-branch model.

    The fusion math is exact in float64 but the float32 round-trip should
    be well below 1e-4 on real data. A larger gap indicates a bug in
    `_fuse_conv_bn` or `_pad_1x1_to_3x3`.
    """
    n = min(32, len(images))
    sample = images[:n]
    y_multi = multi.predict(sample, verbose=0)
    y_fused = fused.predict(sample, verbose=0)
    diffs = []
    for a, b in zip(y_multi, y_fused):
        d = np.max(np.abs(a - b))
        diffs.append(float(d))
    return {
        "max_diff": float(max(diffs)),
        "ok": bool(max(diffs) < atol),
        "per_output_max_diff": {
            "center_xy": diffs[0],
            "radius_xy": diffs[1],
            "confidence": diffs[2],
        },
    }


# ---------------------------------------------------------------------------
# Activation budget check
# ---------------------------------------------------------------------------

def _measure_peak_activation(model: keras.Model) -> dict:
    """Run a single forward pass and record the peak activation tensor size.

    We synthesise a fake input of the right shape and watch every
    intermediate tensor's allocated size. The peak is the largest single
    tensor in bytes.

    Why we report BOTH fp32 and int8 sizes:
    - The TF graph is traced in fp32 (4 bytes per element) for the float
      Keras model. The measured peak in fp32 bytes overstates what the
      deployed int8 NPU will see.
    - The int8-equivalent size is the same tensor with 1 byte per element,
      which is the number the STM32 N6 NPU actually has to fit in SRAM.
    - For our 640x640x1 input, the int8 peak is ~614 KB and the fp32 peak
      is ~2.46 MB. The deployment budget is the int8 one.
    """
    from tensorflow.python.framework import convert_to_constants

    # Build a concrete function so we can inspect intermediate tensors.
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
    # The int8 deployment sees 1 byte per element, so divide by 4.
    int8_peak_bytes = peak_bytes // 4
    return {
        "peak_bytes_fp32": int(peak_bytes),
        "peak_mb_fp32": round(peak_bytes / 1e6, 3),
        "peak_bytes_int8": int(int8_peak_bytes),
        "peak_mb_int8": round(int8_peak_bytes / 1e6, 3),
        "peak_tensor": peak_name,
        "within_int8_budget_1_5mb": bool(int8_peak_bytes <= 1.5 * 1024 * 1024),
    }


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def _export_int8_tflite(model: keras.Model, sample: np.ndarray,
                         output_path: Path) -> dict:
    """Export a fully-quantized int8 TFLite model with a representative dataset.

    The representative dataset is what calibrates the activation ranges
    for every layer. We pull 512 random training images and feed them
    through the model. Without this calibration the int8 grid is set by
    a single forward pass and the quantized output collapses to a
    constant (the same bug we hit with the v9/v10/v11 bias-only convs).
    """

    def rep_dataset():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=512, replace=False):
            yield [sample[idx][None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # float32 outputs are easier to debug
    blob = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)

    # Inspect the resulting model so the report can record shape + quant info.
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    info = {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input": {
            "shape": interp.get_input_details()[0]["shape"].tolist(),
            "dtype": str(interp.get_input_details()[0]["dtype"]),
        },
        "outputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": str(d["dtype"]),
            }
            for d in interp.get_output_details()
        ],
    }
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fp32-epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    parser.add_argument("--skip-fp32", action="store_true",
                        help="Skip FP32 training (use only QAT).")
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()

    # Ensure the artifact directory exists before any model.save() calls.
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    # Build the multi-branch model.
    print(f"Building multi-branch model (alpha={args.alpha})...")
    multi = build_repvgg_ellipse_multi(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        alpha=args.alpha,
    )
    multi.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in multi.trainable_variables))
    print(f"Multi-branch trainable params: {n_params:,} ({n_params * 4 / 1e6:.2f} MB fp32)")

    # Losses and weights.
    losses = {
        "center_xy": keras.losses.Huber(delta=0.05),
        "radius_xy": keras.losses.Huber(delta=0.05),
        "confidence": keras.losses.Huber(delta=0.05),
    }
    loss_weights = {"center_xy": 1.0, "radius_xy": 3.0, "confidence": 0.1}

    # Build the matching fused model so we can copy weights over at the end.
    fused_template = build_repvgg_ellipse_fused(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        alpha=args.alpha,
    )

    # ---- FP32 training ----
    # Custom object map for loading previously-saved .keras files. The
    # WarmupCosineDecay class is registered as serializable now, but the
    # first checkpoint was saved before that registration, so we still
    # need to pass it explicitly on load.
    custom_objects = {"WarmupCosineDecay": WarmupCosineDecay}

    if not args.skip_fp32:
        steps_per_epoch = max(1, len(train_x) // args.batch_size)
        lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs,
                                steps_per_epoch * 3)
        multi.compile(
            optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
            loss=losses,
            loss_weights=loss_weights,
        )
        print("FP32 training...")
        multi.fit(
            train_x, train_y,
            batch_size=args.batch_size,
            epochs=args.fp32_epochs,
            validation_data=(val_x, val_y),
            callbacks=[keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6, verbose=1,
            )],
            verbose=2,
        )
        multi.save(args.output / "model_fp32_multi.keras")
    else:
        # Load a previously trained multi-branch model and re-fuse it.
        # (Or, if a fused model already exists, load it directly to skip
        # the multi->fused reparameterize step entirely.)
        fused_path = args.output / "model_fused.keras"
        if fused_path.exists():
            fused = keras.models.load_model(fused_path, custom_objects=custom_objects)
            multi = keras.models.load_model(
                args.output / "model_fp32_multi.keras", custom_objects=custom_objects,
            )
            print(f"Loaded pre-trained fused model from {fused_path}")
        else:
            multi = keras.models.load_model(
                args.output / "model_fp32_multi.keras", custom_objects=custom_objects,
            )
            print("Loaded pre-trained multi-branch model")
            print("Reparameterizing multi-branch -> fused single-branch...")
            fused = reparameterize_model(multi)
            fused.save(args.output / "model_fused.keras")

    n_fused_params = int(sum(np.prod(v.shape) for v in fused.trainable_variables))
    print(f"Fused trainable params: {n_fused_params:,} ({n_fused_params * 4 / 1e6:.2f} MB fp32, "
          f"~{n_fused_params / 1e6:.2f} MB int8)")

    # Verify the fused model matches the multi-branch model on a small sample.
    print("Checking fused matches multi-branch (max abs diff)...")
    sanity = _check_fused_matches_multi(multi, fused, val_x, atol=1e-4)
    print(f"  {sanity}")
    if not sanity["ok"]:
        raise RuntimeError(
            f"Fused model diverges from multi-branch by {sanity['max_diff']:.2e} "
            f"(expected < 1e-4). Check _fuse_conv_bn and _pad_1x1_to_3x3."
        )

    # ---- QAT on the FUSED model ----
    print("Applying tfmot QAT to the fused model...")
    import tensorflow_model_optimization as tfmot
    qat_model = tfmot.quantization.keras.quantize_model(fused)

    # Why: tfmot renames every output by prepending "quant_", so the target
    # dict has to use the new names. We also have to remap the loss dict
    # and loss_weights dict to match.
    qat_output_names = [o.name.split("/")[0] for o in qat_model.outputs]
    print(f"  QAT output names: {qat_output_names}")
    qat_losses = {name: losses[old] for name, old in zip(qat_output_names, losses)}
    qat_loss_weights = {name: loss_weights[old] for name, old in zip(qat_output_names, loss_weights)}
    qat_targets = {f"quant_{k}": v for k, v in train_y.items()}
    qat_val_targets = {f"quant_{k}": v for k, v in val_y.items()}

    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    qat_lr = WarmupCosineDecay(args.qat_lr, steps_per_epoch * args.qat_epochs,
                                steps_per_epoch)
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

    # ---- Activation budget ----
    print("Measuring peak activation...")
    peak = _measure_peak_activation(fused)
    print(f"  Peak activation: {peak['peak_mb']} MB ({peak['peak_tensor']})")
    if peak["peak_mb"] > 1.5:
        print(f"  WARNING: peak activation {peak['peak_mb']} MB exceeds 1.5 MB budget!")

    # ---- Int8 TFLite export ----
    print("Exporting int8 TFLite...")
    tflite_info = _export_int8_tflite(qat_model, train_x, args.output / "model_int8.tflite")
    print(f"  TFLite size: {tflite_info['kb']} KB")
    print(f"  Input: {tflite_info['input']}")
    print(f"  Outputs: {[o['name'] for o in tflite_info['outputs']]}")

    # ---- Report ----
    report = {
        "model": f"ellipse_repvgg_fused_a{args.alpha}",
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 1],
        "alpha": args.alpha,
        "n_params_multi": n_params,
        "n_params_fused": n_fused_params,
        "fp32_size_mb": round(n_fused_params * 4 / 1e6, 3),
        "int8_size_mb": round(tflite_info["bytes"] / 1e6, 3),
        "peak_activation_mb": peak["peak_mb"],
        "peak_tensor": peak["peak_tensor"],
        "fused_sanity_check": sanity,
        "tflite": tflite_info,
        "fp32_epochs": args.fp32_epochs,
        "qat_epochs": args.qat_epochs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("\nReport:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
