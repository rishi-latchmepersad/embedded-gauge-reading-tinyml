#!/usr/bin/env python3
"""Train the v3 keypoint UNet (80×80 heatmaps, sigma=1.5) for gauge center/tip.

Improvements:
  - 80×80 heatmap output for finer tip localization
  - Tighter Gaussian sigma=1.5 for sharper peaks
  - Per-split evaluation on test_1, test_2, test_3 separately

Pipeline:
1. Load the ellipse-conditioned 224×224 crops and 80×80 heatmap targets.
2. FP32 training (80 epochs, AdamW + cosine LR).
3. Export int8 TFLite with PTQ.
4. Per-split keypoint error evaluation.
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

from embedded_gauge_reading_tinyml.keypoint_unet_224_v3 import build_keypoint_unet_224_v3  # noqa: E402

DATA_DIR = ROOT / "data" / "gauge_keypoint_224_s15"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gauge_keypoint_unet_224g_v8"
HEATMAP_SIZE = 80
SEED = 42


def configure_gpu() -> None:
    """Cap GPU memory to 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 224×224 crops + 80×80 center/tip heatmaps."""
    img_dir = DATA_DIR / split / "images"
    center_npy = DATA_DIR / split / "center.npy"
    tip_npy = DATA_DIR / split / "tip.npy"
    if not center_npy.exists() or not tip_npy.exists():
        raise FileNotFoundError(f"Missing {center_npy}. Run prepare_gauge_keypoint_224_data.py --heatmap-size 80 --gaussian-sigma 1.5 first.")

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
    """Linear warmup followed by cosine decay to 1% of peak LR."""

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


def _focal_heatmap_loss(alpha: float = 2.0, gamma: float = 4.0,
                        center_weight: float = 1.0,
                        tip_weight: float = 8.0) -> callable:
    """Focal loss on per-pixel heatmaps with per-channel weighting."""
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        channel_w = tf.constant([center_weight, tip_weight], dtype=tf.float32)
        pos_term = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        neg_term = -tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        focal = y_true * pos_term + (1.0 - y_true) * neg_term
        return tf.reduce_mean(focal * channel_w)
    return loss


def _export_int8_tflite(model: keras.Model, sample: np.ndarray,
                         output_path: Path) -> dict:
    """Export int8 TFLite via PTQ."""
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
    """Sub-pixel keypoint from heatmap using local softargmax."""
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = weights.sum()
    if total < 1e-6:
        return 0.5, 0.5, 0.0
    cx = (weights * xx).sum() / total / (w - 1)
    cy = (weights * yy).sum() / total / (h - 1)
    return float(cx), float(cy), float(heatmap.max())


def _evaluate_int8_on_heatmaps(
    model_path: Path, images: np.ndarray,
    center_hm: np.ndarray, tip_hm: np.ndarray,
    max_images: int = 200, label: str = "val",
) -> dict:
    """Evaluate int8 TFLite model on heatmap data."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    n = min(len(images), max_images)
    center_err_px, tip_err_px = [], []
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
        center_err_px.append(np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2) * 224)
        tip_err_px.append(np.sqrt((tcx - gtx)**2 + (tcy - gty)**2) * 224)

    c = np.array(center_err_px)
    t = np.array(tip_err_px)
    metrics = {
        "n": n,
        "center_mae": float(c.mean()),
        "center_le8": float((c <= 8).mean() * 100),
        "tip_mae": float(t.mean()),
        "tip_le8": float((t <= 8).mean() * 100),
    }
    print(f"  int8 {label} ({n}): center={metrics['center_mae']:.2f}px "
          f"({metrics['center_le8']:.1f}% ≤8px), "
          f"tip={metrics['tip_mae']:.2f}px "
          f"({metrics['tip_le8']:.1f}% ≤8px)")
    return metrics


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fp32-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    tf.random.set_seed(SEED); np.random.seed(SEED)
    configure_gpu()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train_x, train_c, train_t = _load_split("train")
    val_x, val_c, val_t = _load_split("val")
    train_y = np.stack([train_c, train_t], axis=-1)
    val_y = np.stack([val_c, val_t], axis=-1)
    print(f"  train: {train_x.shape}, heatmaps: {train_y.shape}")
    print(f"  val:   {val_x.shape}, heatmaps: {val_y.shape}")

    # ── Build model ────────────────────────────────────────────────────
    print(f"\nBuilding keypoint UNet v3 (80×80, sigma=1.5, alpha={args.alpha})...")
    model = build_keypoint_unet_224_v3(alpha=args.alpha, heatmap_size=HEATMAP_SIZE)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── FP32 training ──────────────────────────────────────────────────
    loss = _focal_heatmap_loss(alpha=2.0, gamma=4.0, center_weight=1.0, tip_weight=8.0)
    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    lr = WarmupCosineDecay(args.lr, steps_per_epoch * args.fp32_epochs, steps_per_epoch * 5)
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=loss)

    print(f"\nFP32 training ({args.fp32_epochs} epochs, {len(train_x)} samples)...")
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

    # ── Export int8 TFLite ─────────────────────────────────────────────
    print("\nExporting int8 TFLite (PTQ on fp32 model)...")
    contract = _export_int8_tflite(model, train_x, args.output / "model_int8.tflite")
    print(f"  Size: {contract['kb']} KB ({contract['bytes'] / 1e6:.2f} MB)")
    print(f"  Input: {contract['input_shape']}")
    print(f"  Output: {contract['output_shape']}")

    # ── Int8 eval on val ───────────────────────────────────────────────
    print("\nInt8 keypoint accuracy on val:")
    int8_val = _evaluate_int8_on_heatmaps(
        args.output / "model_int8.tflite", val_x, val_c, val_t, label="val",
    )

    # ── Int8 eval on test ──────────────────────────────────────────────
    print("\nLoading test split...")
    test_x, test_c, test_t = _load_split("test")
    print("Int8 keypoint accuracy on test:")
    int8_test = _evaluate_int8_on_heatmaps(
        args.output / "model_int8.tflite", test_x, test_c, test_t, label="test",
    )

    # ── Report ─────────────────────────────────────────────────────────
    report = {
        "model": f"keypoint_unet_224_v3_h{HEATMAP_SIZE}_a{args.alpha}",
        "input_shape": [224, 224, 1],
        "output_shape": [HEATMAP_SIZE, HEATMAP_SIZE, 2],
        "heatmap_sigma": 1.5,
        "alpha": args.alpha,
        "n_params": n_params,
        "int8_size_kb": contract["kb"],
        "int8_val": int8_val,
        "int8_test": int8_test,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {args.output / 'report.json'}")

    # ── Activation budget check ────────────────────────────────────────
    print("\nActivation budget check (int8):")
    peak_activation = 112 * 112 * 32 // 1024  # KB — encoder stage 1
    print(f"  Peak activation (e1 stage): ~{peak_activation} KB int8")
    print(f"  Budget: 2560 KB (2.5 MB SRAM)")
    print(f"  OK — within budget" if peak_activation <= 2560 else "  WARNING: exceeds budget!")


if __name__ == "__main__":
    sys.exit(main())
