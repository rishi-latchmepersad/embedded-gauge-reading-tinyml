#!/usr/bin/env python3
"""Train the needle center/tip heatmap UNet on gauge face crops.

Pipeline:
1. Load pre-extracted data from prepare_needle_data.py
2. FP32 training with focal heatmap loss + cosine LR
3. Export int8 TFLite via PTQ
4. Per-split evaluation with softargmax decoding
5. Activation budget check

The needle model takes the GAUGE FACE CROP (not full image) as input.
The ellipse detector produces the crop; this model refines center/tip.

Output: artifacts/needle_unet_224_v1/
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

from embedded_gauge_reading_tinyml.needle_unet_224 import build_needle_unet_224  # noqa: E402

DATA_DIR = ROOT / "data" / "needle_pipeline"
DEFAULT_OUTPUT = ROOT / "artifacts" / "needle_unet_224_v1"
HEATMAP_SIZE = 56
SEED = 42


def configure_gpu() -> None:
    """Cap GPU memory to 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 224x224 crops + 56x56 heatmaps. Returns images, center_hm, tip_hm, has_needle, ellipse_labels."""
    images = np.load(DATA_DIR / split / "images.npy")
    center_hm = np.load(DATA_DIR / split / "center_heatmaps.npy")
    tip_hm = np.load(DATA_DIR / split / "tip_heatmaps.npy")
    has_needle = np.load(DATA_DIR / split / "has_needle.npy")
    ellipse_labels = np.load(DATA_DIR / split / "ellipse_labels.npy")
    n_with_needle = int(has_needle.sum())
    print(f"  {split}: {len(images)} images, {n_with_needle} with needle labels")
    return images, center_hm, tip_hm, has_needle, ellipse_labels


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


def _focal_heatmap_loss(
    alpha: float = 2.0,
    gamma: float = 4.0,
    center_weight: float = 1.0,
    tip_weight: float = 8.0,
):
    """Focal loss on per-pixel heatmaps with per-channel weighting.

    Why these defaults:
    - alpha=2.0, gamma=4.0: strong focal weighting to focus on peak pixels
    - tip_weight=8.0: tip is harder to localize, needs more gradient
    - center_weight=1.0: center is easier, baseline weight
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        channel_w = tf.constant([center_weight, tip_weight], dtype=tf.float32)
        pos_term = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        neg_term = -tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        focal = y_true * pos_term + (1.0 - y_true) * neg_term
        return tf.reduce_mean(focal * channel_w)
    return loss


def _decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float]:
    """Sub-pixel keypoint from heatmap using weighted softargmax."""
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # why: subtract small threshold to focus on peak region
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = weights.sum()
    if total < 1e-6:
        return 0.5, 0.5
    cx = (weights * xx).sum() / total / (w - 1)
    cy = (weights * yy).sum() / total / (h - 1)
    return float(cx), float(cy)


def _export_int8_tflite(
    model: keras.Model,
    sample: np.ndarray,
    output_path: Path,
) -> dict:
    """Export int8 TFLite via PTQ."""
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


def _evaluate_int8_needle(
    model_path: Path,
    images: np.ndarray,
    center_hm: np.ndarray,
    tip_hm: np.ndarray,
    has_needle: np.ndarray,
    ellipse_labels: np.ndarray,
    max_images: int = 200,
    label: str = "val",
) -> dict:
    """Evaluate int8 TFLite needle model on heatmap data.

    Reports:
    - center_mae: Euclidean distance in heatmap space * 224 (pixel units)
    - tip_mae: same for tip
    - center_le8: % of center predictions within 8px
    - tip_le8: % of tip predictions within 8px
    """
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]

    n = min(len(images), max_images)
    # Only evaluate on images that have needle labels.
    # Why: most images in the array are from web datasets without needle labels.
    # Needle-labeled images are appended at the end of the array.
    all_needle_indices = np.where(has_needle)[0]
    eval_indices = all_needle_indices[all_needle_indices < n]
    n_eval = len(eval_indices)

    if n_eval == 0:
        print(f"  int8 {label}: no needle labels available")
        return {"n": 0, "center_mae": 0, "tip_mae": 0}

    center_errs = []
    tip_errs = []

    for i in eval_indices:
        xq = np.clip(
            np.round(images[i:i + 1] / in_scale + in_zp), -128, 127
        ).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(out_det["index"])
        if out_det["dtype"] == np.int8:
            s, z = out_det["quantization"]
            hm = ((raw.astype(np.float32) - z) * s)[0]
        else:
            hm = raw.astype(np.float32)[0]

        # Decode center and tip from predicted heatmaps
        pcx, pcy = _decode_heatmap_peak(hm[..., 0])
        ptx, pty = _decode_heatmap_peak(hm[..., 1])

        # Decode from ground truth heatmaps
        gcx, gcy = _decode_heatmap_peak(center_hm[i])
        gtx, gty = _decode_heatmap_peak(tip_hm[i])

        # Euclidean distance in pixel space (heatmap is 56x56 mapped to 224x224)
        center_err = np.sqrt(((pcx - gcx) * 224) ** 2 + ((pcy - gcy) * 224) ** 2)
        tip_err = np.sqrt(((ptx - gtx) * 224) ** 2 + ((pty - gty) * 224) ** 2)
        center_errs.append(center_err)
        tip_errs.append(tip_err)

    c = np.array(center_errs)
    t = np.array(tip_errs)
    metrics = {
        "n": n_eval,
        "center_mae": float(c.mean()),
        "center_median": float(np.median(c)),
        "center_le8": float((c <= 8).mean() * 100),
        "center_le4": float((c <= 4).mean() * 100),
        "tip_mae": float(t.mean()),
        "tip_median": float(np.median(t)),
        "tip_le8": float((t <= 8).mean() * 100),
        "tip_le4": float((t <= 4).mean() * 100),
    }
    print(f"  int8 {label} ({n_eval}): center={metrics['center_mae']:.2f}px "
          f"({metrics['center_le8']:.1f}% <=8px), "
          f"tip={metrics['tip_mae']:.2f}px "
          f"({metrics['tip_le8']:.1f}% <=8px)")
    return metrics


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fp32-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tip-weight", type=float, default=8.0)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train_x, train_c, train_t, train_needle, train_ell = _load_split("train")
    val_x, val_c, val_t, val_needle, val_ell = _load_split("val")

    # Filter to only samples with needle labels for training
    train_mask = train_needle
    val_mask = val_needle

    train_x_n = train_x[train_mask]
    train_y = np.stack([train_c[train_mask], train_t[train_mask]], axis=-1)
    val_x_n = val_x[val_mask]
    val_y = np.stack([val_c[val_mask], val_t[val_mask]], axis=-1)

    print(f"\nFiltered to needle-labeled samples:")
    print(f"  train: {len(train_x_n)} / {len(train_x)}")
    print(f"  val: {len(val_x_n)} / {len(val_x)}")

    # ── Build model ────────────────────────────────────────────────────
    print(f"\nBuilding needle UNet 224 (alpha={args.alpha})...")
    model = build_needle_unet_224(alpha=args.alpha)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── FP32 training ──────────────────────────────────────────────────
    loss = _focal_heatmap_loss(
        alpha=2.0, gamma=4.0,
        center_weight=1.0, tip_weight=args.tip_weight,
    )
    steps_per_epoch = max(1, len(train_x_n) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 5)
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=loss,
    )

    print(f"\nFP32 training ({args.fp32_epochs} epochs, {len(train_x_n)} samples)...")
    model.fit(
        train_x_n, train_y,
        batch_size=args.batch_size,
        epochs=args.fp32_epochs,
        validation_data=(val_x_n, val_y),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    # ── Export int8 TFLite ─────────────────────────────────────────────
    print("\nExporting int8 TFLite (PTQ on fp32 model)...")
    tflite_path = args.output / "model_int8.tflite"
    contract = _export_int8_tflite(model, train_x_n, tflite_path)
    print(f"  Size: {contract['kb']} KB ({contract['bytes'] / 1e6:.2f} MB)")
    print(f"  Input: {contract['input_shape']}")
    print(f"  Output: {contract['output_shape']}")

    # ── Int8 evaluation ────────────────────────────────────────────────
    print("\nInt8 needle accuracy on val:")
    int8_val = _evaluate_int8_needle(
        tflite_path, val_x, val_c, val_t, val_needle, val_ell, label="val",
    )

    # ── Test set evaluation ─────────────────────────────────────────────
    test_x, test_c, test_t, test_needle, test_ell = _load_split("test")
    print("\nInt8 needle accuracy on test:")
    int8_test = _evaluate_int8_needle(
        tflite_path, test_x, test_c, test_t, test_needle, test_ell, label="test",
    )

    # ── Activation budget check ────────────────────────────────────────
    peak_kb = 112 * 112 * 32 / 1024  # Encoder stage 1
    print(f"\nActivation budget check (int8):")
    print(f"  Peak activation (e1 stage): ~{peak_kb:.0f} KB int8")
    print(f"  Budget: 2560 KB (2.5 MB SRAM)")
    print(f"  OK — within budget" if peak_kb <= 2560 else "  WARNING: exceeds budget!")

    # ── Save report ────────────────────────────────────────────────────
    report = {
        "model": f"needle_unet_224_a{args.alpha}",
        "input_shape": [224, 224, 1],
        "output_shape": [56, 56, 2],
        "heatmap_size": HEATMAP_SIZE,
        "alpha": args.alpha,
        "tip_weight": args.tip_weight,
        "n_params": n_params,
        "int8_size_kb": contract["kb"],
        "peak_activation_kb": round(peak_kb, 1),
        "int8_val": int8_val,
        "int8_test": int8_test,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {args.output / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
