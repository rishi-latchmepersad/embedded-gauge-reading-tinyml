#!/usr/bin/env python3
"""Train keypoint UNet variants to find the best architecture for tip accuracy.

Variants:
  v6 baseline:  2-channel heatmap (center, tip), tip_weight=8.0
  v11 needle:   3-channel heatmap (center, tip, needle_line), tip_weight=8.0
  v12 heavy:    2-channel heatmap, tip_weight=16.0
  v13 wide:     2-channel heatmap, wider decoder (64/96/128 instead of 48/64/96)

Each variant trains for 80 epochs and evaluates per-split.
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

DATA_DIR = ROOT / "data" / "gauge_keypoint_224"
SEED = 42
HEATMAP_SIZE = 56


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _load_split(split):
    img_dir = DATA_DIR / split / "images"
    center_hm = np.load(DATA_DIR / split / "center.npy")
    tip_hm = np.load(DATA_DIR / split / "tip.npy")
    n = len(center_hm)
    images = np.zeros((n, 224, 224, 1), dtype=np.float32)
    for i in range(n):
        img = np.asarray(Image.open(img_dir / f"{i:06d}.jpg").convert("L"), dtype=np.float32)
        images[i, ..., 0] = img / 255.0
    return images, center_hm, tip_hm


def _build_needle_line_heatmap(center_hm, tip_hm):
    """Build a 3rd heatmap channel: Gaussian along the line from center to tip.

    This forces the model to learn the full needle structure (center → tip),
    not just two independent points.  The line heatmap has high values along
    the needle shaft, which should prevent 180° flips because the model must
    predict the correct direction.
    """
    n = len(center_hm)
    h, w = HEATMAP_SIZE, HEATMAP_SIZE
    line_hm = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        # Find peaks
        c_peak = np.unravel_index(np.argmax(center_hm[i]), (h, w))
        t_peak = np.unravel_index(np.argmax(tip_hm[i]), (h, w))

        # Rasterize line from center to tip with Gaussian blur
        cy, cx = c_peak
        ty, tx = t_peak
        n_steps = max(abs(ty - cy), abs(tx - cx), 1)
        for s in range(n_steps + 1):
            t = s / n_steps
            py = cy + t * (ty - cy)
            px = cx + t * (tx - cx)
            # Gaussian at this point along the line
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            sigma = 1.5  # narrower than keypoint sigma
            dist2 = (yy - py)**2 + (xx - px)**2
            line_hm[i] = np.maximum(line_hm[i], np.exp(-dist2 / (2.0 * sigma**2)))

    return line_hm


def _conv_bn_relu(x, filters, stride=1, name=""):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _encoder_stage(x, filters, name, downsample=True):
    x = _conv_bn_relu(x, filters, stride=2 if downsample else 1, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def _decoder_stage(x, skip, filters, name):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_bn_relu(x, filters, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def build_model(n_channels=2, decoder_channels=None):
    """Build UNet with configurable output channels and decoder width.

    Args:
        n_channels: number of output heatmap channels (2 or 3)
        decoder_channels: list of channel counts for decoder stages.
                         Default: [48, 64, 96] (same as v6).
                         v13 uses [64, 96, 128] (wider).
    """
    if decoder_channels is None:
        decoder_channels = [48, 64, 96]

    inputs = keras.Input(shape=(224, 224, 1), name="image")

    # Encoder (same as v6)
    e1 = _encoder_stage(inputs, 32, "e1", downsample=True)   # 224→112
    e2 = _encoder_stage(e1, 48, "e2")                         # 112→56
    e3 = _encoder_stage(e2, 64, "e3")                         # 56→28
    e4 = _encoder_stage(e3, 96, "e4")                         # 28→14
    b = _encoder_stage(e4, 128, "e5")                         # 14→7

    # Decoder (configurable width)
    d1 = _decoder_stage(b, e4, decoder_channels[2], "d1")     # 7→14
    d2 = _decoder_stage(d1, e3, decoder_channels[1], "d2")    # 14→28
    d3 = _decoder_stage(d2, e2, decoder_channels[0], "d3")    # 28→56

    # Output head
    x = _conv_bn_relu(d3, 32, name="head")
    outputs = layers.Conv2D(n_channels, 1, padding="same",
                            activation="sigmoid", name="heatmaps")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name=f"keypoint_unet_c{n_channels}")


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


def _focal_loss(alpha=2.0, gamma=4.0, channel_weights=None):
    """Focal loss with per-channel weighting."""
    if channel_weights is None:
        channel_weights = [1.0, 1.0]

    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cw = tf.constant(channel_weights, dtype=tf.float32)
        pos = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        neg = -tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        focal = y_true * pos + (1.0 - y_true) * neg
        return tf.reduce_mean(focal * cw)
    return loss


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
    in_s, in_z = in_det["quantization"]
    out_s, out_z = hm_out["quantization"]

    n = min(len(images), max_images)
    c_errs, t_errs = [], []
    for i in range(n):
        xq = np.clip(np.round(images[i:i+1] / in_s + in_z), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(hm_out["index"])
        hm = ((raw.astype(np.float32) - out_z) * out_s)[0]
        pcx, pcy, _ = _decode_heatmap_peak(hm[..., 0])
        tcx, tcy, _ = _decode_heatmap_peak(hm[..., 1])
        gcx, gcy, _ = _decode_heatmap_peak(center_hm[i])
        gtx, gty, _ = _decode_heatmap_peak(tip_hm[i])
        c_errs.append(np.sqrt((pcx-gcx)**2+(pcy-gcy)**2)*224)
        t_errs.append(np.sqrt((tcx-gtx)**2+(tcy-gty)**2)*224)

    c, t = np.array(c_errs), np.array(t_errs)
    m = {"n": n, "c_mae": float(c.mean()), "c_le8": float((c<=8).mean()*100),
         "t_mae": float(t.mean()), "t_le8": float((t<=8).mean()*100)}
    print(f"    {label} ({n}): center={m['c_mae']:.2f}px ({m['c_le8']:.1f}% ≤8), "
          f"tip={m['t_mae']:.2f}px ({m['t_le8']:.1f}% ≤8)")
    return m


def _export_int8(model, sample, output_path):
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
    output_path.write_bytes(blob)
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    return {"kb": round(len(blob) / 1024, 1)}


def train_and_eval(name, model, train_x, train_y, val_x, val_y,
                   val_c, val_t, test_x, test_c, test_t,
                   tip_weight=8.0, epochs=80, batch_size=16, lr=1e-3):
    """Train a model variant and evaluate per-split."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  tip_weight={tip_weight}, output_channels={model.output_shape[-1]}")
    print(f"  params={int(sum(np.prod(v.shape) for v in model.trainable_variables)):,}")
    print(f"{'='*60}")

    out_dir = ROOT / "artifacts" / f"gauge_keypoint_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_channels = model.output_shape[-1]
    cw = [1.0, tip_weight] if n_channels == 2 else [1.0, tip_weight, tip_weight * 0.5]
    loss = _focal_loss(channel_weights=cw)

    steps = max(1, len(train_x) // batch_size)
    schedule = WarmupCosineDecay(lr, steps * epochs, steps * 5)
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-4), loss=loss)

    model.fit(
        train_x, train_y, batch_size=batch_size, epochs=epochs,
        validation_data=(val_x, val_y),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=0),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True, verbose=0),
        ],
        verbose=0,
    )
    model.save(out_dir / "model_fp32.keras")

    # Export int8 (only first 2 channels for deployment)
    if n_channels > 2:
        infer = keras.Model(inputs=model.input, outputs=model.output[..., :2], name="infer")
    else:
        infer = model
    contract = _export_int8(infer, train_x, out_dir / "model_int8.tflite")

    # Evaluate per-split
    print(f"    Int8 size: {contract['kb']} KB")
    val_m = _evaluate_int8(out_dir / "model_int8.tflite", val_x, val_c, val_t, label="val")
    test_m = _evaluate_int8(out_dir / "model_int8.tflite", test_x, test_c, test_t, label="test")

    # Per-split on test zips
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_keypoint_per_split import evaluate_split, _extract_records
    import tensorflow as tf_lite
    interp = tf.lite.Interpreter(model_path=str(out_dir / "model_int8.tflite"))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    per_split = {}
    for zip_name in ["test_1.zip", "test_2.zip", "test_3.zip"]:
        m = evaluate_split(interp, in_d, out_d, zip_name)
        if m:
            per_split[zip_name.replace(".zip", "")] = m

    report = {"name": name, "tip_weight": tip_weight,
              "n_channels": n_channels, "int8_kb": contract["kb"],
              "val": val_m, "test": test_m, "per_split": per_split}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    configure_gpu()
    np.random.seed(SEED); tf.random.set_seed(SEED)

    print("Loading data...")
    train_x, train_c, train_t = _load_split("train")
    val_x, val_c, val_t = _load_split("val")
    test_x, test_c, test_t = _load_split("test")

    # Build needle-line heatmap for v11
    needle_line = _build_needle_line_heatmap(train_c, train_t)
    train_y_3ch = np.stack([train_c, train_t, needle_line], axis=-1)
    train_y_2ch = np.stack([train_c, train_t], axis=-1)
    val_y_2ch = np.stack([val_c, val_t], axis=-1)

    results = []

    # v11: Needle-line heatmap (3 channels)
    m = build_model(n_channels=3)
    val_y_3ch = np.concatenate([val_y_2ch, np.zeros((len(val_x), HEATMAP_SIZE, HEATMAP_SIZE, 1), dtype=np.float32)], axis=-1)
    r = train_and_eval("v11_needle_line", m, train_x, train_y_3ch,
                       val_x, val_y_3ch, val_c, val_t, test_x, test_c, test_t,
                       tip_weight=8.0, epochs=80)
    results.append(r)

    # v12: Heavy tip weight (16.0)
    m = build_model(n_channels=2)
    r = train_and_eval("v12_heavy_tip", m, train_x, train_y_2ch,
                       val_x, val_y_2ch, val_c, val_t, test_x, test_c, test_t,
                       tip_weight=16.0, epochs=80)
    results.append(r)

    # v13: Wider decoder (64/96/128 instead of 48/64/96)
    m = build_model(n_channels=2, decoder_channels=[64, 96, 128])
    r = train_and_eval("v13_wide_decoder", m, train_x, train_y_2ch,
                       val_x, val_y_2ch, val_c, val_t, test_x, test_c, test_t,
                       tip_weight=8.0, epochs=80)
    results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    for r in results:
        ps = r.get("per_split", {})
        t1 = ps.get("test_1", {})
        t3 = ps.get("test_3", {})
        print(f"  {r['name']:25s}  test_1 tip={t1.get('tip_mae', 0):.1f}px "
              f"({t1.get('tip_le8', 0):.0f}% ≤8)  "
              f"test_3 tip={t3.get('tip_mae', 0):.1f}px "
              f"({t3.get('tip_le8', 0):.0f}% ≤8)")

    (ROOT / "artifacts" / "architecture_comparison.json").write_text(
        json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
