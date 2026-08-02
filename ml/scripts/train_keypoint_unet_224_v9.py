#!/usr/bin/env python3
"""Train keypoint UNet with combined focal + decoded-keypoint L1 loss (v9).

Combines:
  1. Focal heatmap loss (existing, learns heatmap distribution)
  2. L1 loss on decoded keypoint coordinates (new, direct spatial supervision)

The decoded keypoint is computed from the heatmap via differentiable
softargmax, so gradients flow back through the heatmap to the model weights.
This gives the model direct supervision on the final spatial coordinates,
not just the heatmap distribution.

Uses v6 architecture (proven) with expanded training data (7525 images).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.keypoint_unet_224 import build_keypoint_unet_224  # noqa: E402

DATA_DIR = ROOT / "data" / "gauge_keypoint_224"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_keypoint_unet_224g_v9"
HEATMAP_SIZE = 56
SEED = 42


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 224×224 crops + 56×56 center/tip heatmaps."""
    img_dir = DATA_DIR / split / "images"
    center_npy = DATA_DIR / split / "center.npy"
    tip_npy = DATA_DIR / split / "tip.npy"
    if not center_npy.exists() or not tip_npy.exists():
        raise FileNotFoundError(f"Missing {center_npy}.")

    center_hm = np.load(center_npy)
    n = len(center_hm)
    images = np.zeros((n, 224, 224, 1), dtype=np.float32)
    tip_hm = np.load(tip_npy)

    for i in range(n):
        img = np.asarray(Image.open(img_dir / f"{i:06d}.jpg").convert("L"), dtype=np.float32)
        images[i, ..., 0] = img / 255.0

    print(f"  {split}: {n} images, heatmap peaks: c={center_hm.max():.3f}, t={tip_hm.max():.3f}")
    return images, center_hm, tip_hm


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0):
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
        return {"peak_lr": self._peak, "total_steps": self._total, "warmup_steps": self._warmup}


def _softargmax(heatmap: tf.Tensor) -> tf.Tensor:
    """Differentiable softargmax: weighted average of coordinates.

    Args:
        heatmap: (batch, H, W) tensor.
    Returns:
        (batch, 2) tensor of (x, y) coordinates in [0, 1].
    """
    h = tf.cast(tf.shape(heatmap)[1], tf.float32)
    w = tf.cast(tf.shape(heatmap)[2], tf.float32)
    yy, xx = tf.meshgrid(
        tf.linspace(0.0, 1.0, tf.shape(heatmap)[1]),
        tf.linspace(0.0, 1.0, tf.shape(heatmap)[2]),
        indexing="ij",
    )
    xx = tf.expand_dims(xx, axis=0)  # (1, H, W)
    yy = tf.expand_dims(yy, axis=0)  # (1, H, W)
    weights = tf.maximum(heatmap - 0.03, 0.0) ** 2
    total = tf.reduce_sum(weights, axis=[1, 2], keepdims=True) + 1e-6
    cx = tf.reduce_sum(weights * xx, axis=[1, 2]) / tf.squeeze(total)
    cy = tf.reduce_sum(weights * yy, axis=[1, 2]) / tf.squeeze(total)
    return tf.stack([cx, cy], axis=-1)  # (batch, 2)


def _heatmap_to_keypoints(hm: tf.Tensor) -> tf.Tensor:
    """Convert (batch, H, W, 2) heatmaps to (batch, 4) keypoint coords.

    Output: [center_x, center_y, tip_x, tip_y] in [0, 1].
    """
    center_coords = _softargmax(hm[..., 0])  # (batch, 2)
    tip_coords = _softargmax(hm[..., 1])     # (batch, 2)
    return tf.concat([center_coords, tip_coords], axis=-1)  # (batch, 4)


def _heatmap_to_gt_keypoints(hm: tf.Tensor) -> tf.Tensor:
    """Extract GT keypoint coords from heatmaps (argmax-based, non-differentiable).

    Used only for the L1 supervision target.
    Output: (batch, 4) — [center_x, center_y, tip_x, tip_y] in [0, 1].
    """
    batch_size = tf.shape(hm)[0]
    h = tf.cast(tf.shape(hm)[1], tf.float32)
    w = tf.cast(tf.shape(hm)[2], tf.float32)

    results = []
    for ch in range(2):
        channel_hm = hm[..., ch]  # (batch, H, W)
        flat = tf.reshape(channel_hm, (batch_size, -1))
        idx = tf.argmax(flat, axis=-1)  # (batch,)
        gy = tf.cast(idx // tf.cast(tf.shape(channel_hm)[1], tf.int64), tf.float32) / (h - 1.0)
        gx = tf.cast(idx % tf.cast(tf.shape(channel_hm)[1], tf.int64), tf.float32) / (w - 1.0)
        results.append(gx)
        results.append(gy)
    return tf.stack(results, axis=-1)  # (batch, 4)


def _combined_loss(
    alpha: float = 2.0,
    gamma: float = 4.0,
    center_weight: float = 1.0,
    tip_weight: float = 8.0,
    l1_weight: float = 5.0,
) -> callable:
    """Combined focal heatmap loss + decoded keypoint L1 loss.

    The focal loss teaches the model to produce correct heatmap distributions.
    The L1 loss on decoded keypoints teaches the model to produce heatmaps
    whose peaks decode to the correct spatial coordinates.
    """
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred_clipped = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Focal heatmap loss
        channel_w = tf.constant([center_weight, tip_weight], dtype=tf.float32)
        pos_term = -alpha * tf.pow(1.0 - y_pred_clipped, gamma) * tf.math.log(y_pred_clipped)
        neg_term = -tf.pow(y_pred_clipped, gamma) * tf.math.log(1.0 - y_pred_clipped)
        focal = y_true * pos_term + (1.0 - y_true) * neg_term
        focal_loss = tf.reduce_mean(focal * channel_w)

        # Decoded keypoint L1 loss
        pred_kpts = _heatmap_to_keypoints(y_pred)       # (batch, 4) — differentiable
        gt_kpts = _heatmap_to_gt_keypoints(y_true)      # (batch, 4) — argmax from GT
        l1_loss = tf.reduce_mean(tf.abs(pred_kpts - gt_kpts))

        return focal_loss + l1_weight * l1_loss
    return loss


def _export_int8_tflite(model: keras.Model, sample: np.ndarray,
                         output_path: Path) -> dict:
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
    out_det = interp.get_output_details()[0]
    return {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input_shape": in_det["shape"].tolist(),
        "output_shape": out_det["shape"].tolist(),
    }


def _decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float, float]:
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = weights.sum()
    if total < 1e-6:
        return 0.5, 0.5, 0.0
    cx = (weights * xx).sum() / total / (w - 1)
    cy = (weights * yy).sum() / total / (h - 1)
    return float(cx), float(cy), float(heatmap.max())


def _evaluate_int8(model_path, images, center_hm, tip_hm, max_images=200, label="val"):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    n = min(len(images), max_images)
    c_errs, t_errs = [], []
    for i in range(n):
        xq = np.clip(np.round(images[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(out_det["index"])
        hm = ((raw.astype(np.float32) - out_zp) * out_scale)[0]
        pcx, pcy, _ = _decode_heatmap_peak(hm[..., 0])
        tcx, tcy, _ = _decode_heatmap_peak(hm[..., 1])
        gcx, gcy, _ = _decode_heatmap_peak(center_hm[i])
        gtx, gty, _ = _decode_heatmap_peak(tip_hm[i])
        c_errs.append(np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2) * 224)
        t_errs.append(np.sqrt((tcx - gtx)**2 + (tcy - gty)**2) * 224)

    c, t = np.array(c_errs), np.array(t_errs)
    m = {
        "n": n, "center_mae": float(c.mean()), "center_le8": float((c <= 8).mean() * 100),
        "tip_mae": float(t.mean()), "tip_le8": float((t <= 8).mean() * 100),
    }
    print(f"  int8 {label} ({n}): center={m['center_mae']:.2f}px ({m['center_le8']:.1f}% ≤8px), "
          f"tip={m['tip_mae']:.2f}px ({m['tip_le8']:.1f}% ≤8px)")
    return m


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fp32-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-weight", type=float, default=5.0,
                        help="Weight for decoded keypoint L1 loss")
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_x, train_c, train_t = _load_split("train")
    val_x, val_c, val_t = _load_split("val")
    train_y = np.stack([train_c, train_t], axis=-1)
    val_y = np.stack([val_c, val_t], axis=-1)
    print(f"  train: {train_x.shape}, val: {val_x.shape}")

    print(f"\nBuilding keypoint UNet (v6 arch, combined loss)...")
    model = build_keypoint_unet_224()
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    loss = _combined_loss(alpha=2.0, gamma=4.0, center_weight=1.0,
                          tip_weight=8.0, l1_weight=args.l1_weight)
    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 5)
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=loss)

    print(f"\nFP32 training ({args.fp32_epochs} epochs, {len(train_x)} samples, l1_weight={args.l1_weight})...")
    model.fit(
        train_x, train_y, batch_size=args.batch_size, epochs=args.fp32_epochs,
        validation_data=(val_x, val_y),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    print("\nExporting int8 TFLite...")
    contract = _export_int8_tflite(model, train_x, args.output / "model_int8.tflite")
    print(f"  Size: {contract['kb']} KB ({contract['bytes'] / 1e6:.2f} MB)")

    print("\nInt8 val:")
    int8_val = _evaluate_int8(args.output / "model_int8.tflite", val_x, val_c, val_t, label="val")

    test_x, test_c, test_t = _load_split("test")
    print("Int8 test:")
    int8_test = _evaluate_int8(args.output / "model_int8.tflite", test_x, test_c, test_t, label="test")

    report = {
        "model": "keypoint_unet_224_v6_combined_loss",
        "l1_weight": args.l1_weight, "n_params": n_params,
        "int8_size_kb": contract["kb"], "int8_val": int8_val, "int8_test": int8_test,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.output / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
