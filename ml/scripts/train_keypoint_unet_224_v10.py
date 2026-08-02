#!/usr/bin/env python3
"""Train keypoint UNet with offset regression (v10).

Improvements:
  - Offset regression head: 56×56×4 predicting dx/dy for center and tip
  - Heatmap weighting loss: focuses gradient on keypoint pixels
  - Combined output: heatmaps(2) + offsets(4) = 56×56×6

The offset head recovers sub-pixel accuracy beyond 56×56 resolution.
At inference, only the heatmap channels (0:2) are used for export.
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

from embedded_gauge_reading_tinyml.keypoint_unet_224_v4 import build_keypoint_unet_v4  # noqa: E402

DATA_DIR = ROOT / "data" / "gauge_keypoint_224"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_keypoint_unet_224g_v10"
HEATMAP_SIZE = 56
SEED = 42


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split):
    img_dir = DATA_DIR / split / "images"
    center_npy = DATA_DIR / split / "center.npy"
    tip_npy = DATA_DIR / split / "tip.npy"
    center_hm = np.load(center_npy)
    n = len(center_hm)
    images = np.zeros((n, 224, 224, 1), dtype=np.float32)
    tip_hm = np.load(tip_npy)
    for i in range(n):
        img = np.asarray(Image.open(img_dir / f"{i:06d}.jpg").convert("L"), dtype=np.float32)
        images[i, ..., 0] = img / 255.0
    print(f"  {split}: {n} images")
    return images, center_hm, tip_hm


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps=0):
        super().__init__()
        self._peak = peak_lr
        self._total = total_steps
        self._warmup = warmup_steps
        self._cosine = keras.optimizers.schedules.CosineDecay(
            peak_lr, max(1, total_steps - warmup_steps), alpha=0.01)

    def __call__(self, step):
        wf = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(step < self._warmup, self._peak * wf,
                        self._cosine(step - self._warmup))

    def get_config(self):
        return {"peak_lr": self._peak, "total_steps": self._total,
                "warmup_steps": self._warmup}


def _build_gt_offsets(center_hm, tip_hm):
    """Build GT offset maps from heatmaps.

    At each heatmap peak, the offset is (gt_coord - peak_coord).
    For GT heatmaps, the peak IS at the GT coordinate, so offset ≈ 0.
    The model learns to predict offsets when its own peak is slightly off.
    """
    n = len(center_hm)
    h, w = HEATMAP_SIZE, HEATMAP_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= (h - 1)
    xx /= (w - 1)

    center_offsets = np.zeros((n, h, w, 2), dtype=np.float32)
    tip_offsets = np.zeros((n, h, w, 2), dtype=np.float32)

    for i in range(n):
        c_peak = np.unravel_index(np.argmax(center_hm[i]), (h, w))
        t_peak = np.unravel_index(np.argmax(tip_hm[i]), (h, w))

        c_y, c_x = c_peak[0] / (h - 1), c_peak[1] / (w - 1)
        t_y, t_x = t_peak[0] / (h - 1), t_peak[1] / (w - 1)

        # Set offsets in a small radius around each peak
        radius = 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py, px = c_peak[0] + dy, c_peak[1] + dx
                if 0 <= py < h and 0 <= px < w:
                    center_offsets[i, py, px, 0] = c_x - px / (w - 1)
                    center_offsets[i, py, px, 1] = c_y - py / (h - 1)
                py, px = t_peak[0] + dy, t_peak[1] + dx
                if 0 <= py < h and 0 <= px < w:
                    tip_offsets[i, py, px, 0] = t_x - px / (w - 1)
                    tip_offsets[i, py, px, 1] = t_y - py / (h - 1)

    return np.concatenate([center_offsets, tip_offsets], axis=-1)  # (N,H,W,4)


def _build_heatmap_weights(center_hm, tip_hm, sigma=3.0):
    """Build per-pixel weight map emphasizing keypoint regions."""
    n = len(center_hm)
    h, w = HEATMAP_SIZE, HEATMAP_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.ones((n, h, w, 1), dtype=np.float32)

    for i in range(n):
        for hm in [center_hm[i], tip_hm[i]]:
            peak = np.unravel_index(np.argmax(hm), (h, w))
            dist2 = (yy - peak[0])**2 + (xx - peak[1])**2
            w_map = 1.0 + 5.0 * np.exp(-dist2 / (2.0 * sigma**2))
            weights[i, :, :, 0] = np.maximum(weights[i, :, :, 0], w_map)

    return weights


class CombinedLoss(keras.losses.Loss):
    """Combined focal heatmap + offset L1 loss with spatial weighting.

    The model output is (batch, 56, 56, 6):
      channels 0-1: heatmaps (center, tip)
      channels 2-5: offsets (center_dx, center_dy, tip_dx, tip_dy)

    GT is (batch, 56, 56, 2) heatmaps.  Offsets and weights are
    precomputed from the GT heatmaps and stored as numpy arrays.
    """

    def __init__(self, gt_offsets, heatmap_weights, alpha=2.0, gamma=4.0,
                 center_weight=1.0, tip_weight=8.0, offset_weight=10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.gt_offsets = tf.constant(gt_offsets, dtype=tf.float32)
        self.hm_weights = tf.constant(heatmap_weights, dtype=tf.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.center_weight = center_weight
        self.tip_weight = tip_weight
        self.offset_weight = offset_weight

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        gt_hm = y_true[..., :2]  # (N,H,W,2) — heatmaps from GT
        gt_off = self.gt_offsets[:batch_size]
        hm_w = self.hm_weights[:batch_size]

        # Split prediction into heatmaps and offsets
        hm_pred = y_pred[..., :2]
        off_pred = y_pred[..., 2:]

        # 1. Focal heatmap loss with spatial weighting
        eps = 1e-7
        hm_clipped = tf.clip_by_value(hm_pred, eps, 1.0 - eps)
        channel_w = tf.constant([self.center_weight, self.tip_weight], dtype=tf.float32)
        pos_term = -self.alpha * tf.pow(1.0 - hm_clipped, self.gamma) * tf.math.log(hm_clipped)
        neg_term = -tf.pow(hm_clipped, self.gamma) * tf.math.log(1.0 - hm_clipped)
        focal = gt_hm * pos_term + (1.0 - gt_hm) * neg_term
        focal_loss = tf.reduce_mean(focal * channel_w * hm_w)

        # 2. Offset L1 loss (masked to keypoint neighborhoods)
        off_mask = tf.cast(tf.reduce_max(gt_hm, axis=-1, keepdims=True) > 0.3, tf.float32)
        off_loss = tf.reduce_mean(tf.abs(off_pred - gt_off) * off_mask)

        return focal_loss + self.offset_weight * off_loss


def _export_int8_tflite(model, sample, output_path):
    # Create inference model that outputs only heatmaps (channels 0:2)
    inp = model.input
    full_out = model.output[..., :2]  # heatmaps only
    infer_model = keras.Model(inputs=inp, outputs=full_out, name="keypoint_infer")

    def rep():
        rng = np.random.default_rng(SEED)
        for idx in rng.choice(len(sample), size=min(512, len(sample)), replace=False):
            yield [sample[idx:idx+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
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
    return {
        "bytes": len(blob),
        "kb": round(len(blob) / 1024, 1),
        "input_shape": interp.get_input_details()[0]["shape"].tolist(),
        "output_shape": interp.get_output_details()[0]["shape"].tolist(),
    }


def _decode_heatmap_peak(heatmap):
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
    hm_out = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = hm_out["quantization"]

    n = min(len(images), max_images)
    c_errs, t_errs = [], []
    for i in range(n):
        xq = np.clip(np.round(images[i:i+1] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(hm_out["index"])
        hm = ((raw.astype(np.float32) - out_zp) * out_scale)[0]
        pcx, pcy, _ = _decode_heatmap_peak(hm[..., 0])
        tcx, tcy, _ = _decode_heatmap_peak(hm[..., 1])
        gcx, gcy, _ = _decode_heatmap_peak(center_hm[i])
        gtx, gty, _ = _decode_heatmap_peak(tip_hm[i])
        c_errs.append(np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2) * 224)
        t_errs.append(np.sqrt((tcx - gtx)**2 + (tcy - gty)**2) * 224)

    c, t = np.array(c_errs), np.array(t_errs)
    m = {"n": n, "center_mae": float(c.mean()), "center_le8": float((c <= 8).mean() * 100),
         "tip_mae": float(t.mean()), "tip_le8": float((t <= 8).mean() * 100)}
    print(f"  int8 {label} ({n}): center={m['center_mae']:.2f}px ({m['center_le8']:.1f}% ≤8px), "
          f"tip={m['tip_mae']:.2f}px ({m['tip_le8']:.1f}% ≤8px)")
    return m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--offset-weight", type=float, default=10.0)
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_x, train_c, train_t = _load_split("train")
    val_x, val_c, val_t = _load_split("val")

    # Build GT targets
    train_hm_y = np.stack([train_c, train_t], axis=-1)
    val_hm_y = np.stack([val_c, val_t], axis=-1)
    train_gt_offsets = _build_gt_offsets(train_c, train_t)
    train_hm_weights = _build_heatmap_weights(train_c, train_t)

    # Full model output: (N, 56, 56, 6) = heatmaps(2) + offsets(4)
    train_y_full = np.concatenate([train_hm_y, train_gt_offsets], axis=-1).astype(np.float32)
    val_y_full = np.concatenate([val_hm_y, np.zeros((len(val_x), HEATMAP_SIZE, HEATMAP_SIZE, 4), dtype=np.float32)], axis=-1)

    print("Building model...")
    model = build_keypoint_unet_v4()
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Params: {n_params:,}")

    loss_fn = CombinedLoss(train_gt_offsets, train_hm_weights, offset_weight=args.offset_weight)

    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.epochs, steps_per_epoch * 5)
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=loss_fn)

    print(f"\nTraining ({args.epochs} epochs, {len(train_x)} samples, offset_weight={args.offset_weight})...")
    model.fit(
        train_x, train_y_full,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(val_x, val_y_full),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    model.save(args.output / "model_fp32.keras")

    print("\nExporting int8 TFLite (heatmap head only)...")
    contract = _export_int8_tflite(model, train_x, args.output / "model_int8.tflite")
    print(f"  Size: {contract['kb']} KB")

    print("\nInt8 val:")
    int8_val = _evaluate_int8(args.output / "model_int8.tflite", val_x, val_c, val_t, label="val")
    test_x, test_c, test_t = _load_split("test")
    print("Int8 test:")
    int8_test = _evaluate_int8(args.output / "model_int8.tflite", test_x, test_c, test_t, label="test")

    report = {"model": "keypoint_unet_v10_offset", "n_params": n_params,
              "int8_size_kb": contract["kb"], "int8_val": int8_val, "int8_test": int8_test}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.output / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
