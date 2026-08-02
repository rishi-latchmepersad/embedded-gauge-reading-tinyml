#!/usr/bin/env python3
"""Train RepVGG ellipse detector with QAT for STM32 N6.

Single-branch RepVGG (fused from scratch) — no multi-branch reparameterization.
320x320 grayscale input, RepVGG backbone, multi-head ellipse regression.
"""

from __future__ import annotations

import sys, json, random, argparse, zipfile, io
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from tf_keras import layers, Model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.tf_models import _conv_bn_relu, _channel_plan

IMAGE_SIZE = 320
SEED = 42


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)],
        )


def build_fused_ellipse_repvgg(alpha=0.75):
    """Single-branch RepVGG (Conv+BN+ReLU only, no SE, no Multiply).

    Fully QAT-compatible. Uses the fused single-branch topology directly.
    """
    cp = _channel_plan(alpha)
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])

    # Stem: stride-2
    c0 = cp["stem"]
    x = keras.layers.Conv2D(c0, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    # 4 stages, each: Conv3x3(stride2, bias=False) → BN → ReLU → Conv3x3(stride1) → BN → ReLU
    for si, (filters, n_blocks) in enumerate([(cp["s1"], 2), (cp["s2"], 2), (cp["s3"], 2), (cp["s4"], 1)], 1):
        for bi in range(n_blocks):
            stride = 2 if si < 4 and bi == 0 else 1  # stride-2 on first block of non-final stages
            name = f"s{si}block{bi}"
            x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
            x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
            x = keras.layers.ReLU(name=f"{name}_relu")(x)

    # Regression head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    shared = keras.layers.Dense(128, activation="relu", name="shared")(x)

    # Center (sigmoid), Radius (LINEAR), Confidence (sigmoid)
    c = keras.layers.Dense(64, activation="relu", name="center_dense")(shared)
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(c)
    r = keras.layers.Dense(64, activation="relu", name="radius_dense")(shared)
    radius_xy = keras.layers.Dense(2, activation=None, name="radius_xy")(r)
    f = keras.layers.Dense(32, activation="relu", name="conf_dense")(shared)
    confidence = keras.layers.Dense(1, activation="sigmoid", name="confidence")(f)
    return Model(inputs, [center_xy, radius_xy, confidence], name=f"ellipse_repvgg_fused_a{alpha}")


def load_ellipse_labels(zip_paths):
    paths, targets = [], []
    for zp in zip_paths:
        if not zp.exists(): continue
        z = zipfile.ZipFile(zp)
        try: root = ET.parse(z.open("annotations.xml")).getroot()
        except Exception: continue
        for img in root.findall("image"):
            name = img.get("name"); w, h = int(img.get("width","640")), int(img.get("height","640"))
            target = None
            for e in img:
                if e.get("label","") in ("GaugeFace","temp_dial"):
                    if e.tag=="ellipse": target=np.array([float(e.get(c))/w for c in ("cx","cy","rx","ry")]+[1.0], np.float32)
                    elif e.tag=="box":
                        xtl,ytl=float(e.get("xtl")),float(e.get("ytl")); xbr,ybr=float(e.get("xbr")),float(e.get("ybr"))
                        target=np.array([(xtl+xbr)/(2*w),(ytl+ybr)/(2*h),(xbr-xtl)/(2*w),(ybr-ytl)/(2*h),1.0], np.float32)
                    break
            if target is None: continue
            paths.append((zp, name)); targets.append(target)
    return paths, np.stack(targets)


def preload_images(paths, size=320):
    """Decode all training images once, reusing open ZIP handles for speed."""
    images = []
    archives = {}
    for zp, name in paths:
        # why: reopening a ZIP for every image made this experiment appear to
        # hang before the GPU ever received a batch.
        z = archives.setdefault(str(zp), zipfile.ZipFile(zp))
        img_name = f"images/{name}"
        if img_name not in z.namelist():
            matches = [m for m in z.namelist() if name in m and m.endswith((".png",".jpg"))]
            if not matches: continue
            img_name = matches[0]
        data = z.read(img_name)
        img = tf.io.decode_image(data, channels=1, expand_animations=False)
        img = tf.image.resize(img, (size, size), method="bilinear")
        images.append(tf.cast(img.numpy(), tf.float32) / 255.0)
        if len(images) % 1000 == 0:
            print(f"Decoded {len(images)}/{len(paths)} images", flush=True)
    for z in archives.values():
        z.close()
    return np.stack(images, axis=0)


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak, total, warmup=0):
        super().__init__()
        self._peak, self._total, self._warmup = peak, total, warmup
        self._cosine = keras.optimizers.schedules.CosineDecay(peak, max(1, total - warmup), alpha=0.01)
    def __call__(self, step):
        p = tf.cast(step, tf.float32) / tf.cast(max(1, self._warmup), tf.float32)
        return tf.where(step < self._warmup, self._peak * p, self._cosine(step - self._warmup))
    def get_config(self):
        return {"peak": self._peak, "total": self._total, "warmup": self._warmup}


def export_int8(model, images, output):
    def rep():
        rng = np.random.default_rng(42)
        for idx in rng.permutation(len(images))[:512]:
            yield [images[idx][None].astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    blob = c.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    det = {"bytes": len(blob), "outputs": []}
    for d in interp.get_output_details():
        det["outputs"].append({"shape": d["shape"].tolist(), "quant": d["quantization"]})
    det["input"] = interp.get_input_details()[0]["shape"].tolist()
    return det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "gauge_ellipse_repvgg_v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-train-images", type=int, default=0,
                        help="Optional deterministic cap for quick architecture trials.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    parser.add_argument(
        "--clean-board-split",
        action="store_true",
        help="Exclude board_captures_2 and refreshed test_3 from training.",
    )
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    configure_gpu()

    labelled = ROOT / "data" / "labelled"
    # why: board_captures_2 is an exact image-basename duplicate of refreshed
    # test_3, and test_3 itself must remain an unseen acceptance set.
    train_zips = [labelled / "train_1.zip",
                  labelled / "initial_temp_gauge" / "board_captures_1.zip"]
    if not args.clean_board_split:
        train_zips += [labelled / "initial_temp_gauge" / "board_captures_2.zip",
                       labelled / "test_3.zip"]
    train_paths, train_t = load_ellipse_labels(train_zips)
    if args.max_train_images and len(train_paths) > args.max_train_images:
        # why: architecture screening should not spend minutes decoding a
        # large archive before we know the family is worth full training.
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(train_paths), args.max_train_images, replace=False)
        train_paths = [train_paths[i] for i in keep]
        train_t = train_t[keep]
    print(f"Train labels: {len(train_paths)}")
    print("Pre-loading images...")
    # preload_images already returns NHWC grayscale tensors; adding another
    # axis here creates an invalid 5-D input and breaks TFLite calibration.
    train_x = preload_images(train_paths, IMAGE_SIZE)
    print(f"Train images: {train_x.shape}")

    tc = train_t[:, :2].astype(np.float32)
    tr = train_t[:, 2:4].astype(np.float32)
    tconf = train_t[:, 4:5].astype(np.float32)

    steps = max(1, len(train_x) // args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)

    # Build fused single-branch model
    model = build_fused_ellipse_repvgg(alpha=args.alpha)
    model.summary(line_length=100)

    losses = [keras.losses.Huber(delta=0.05)] * 3
    loss_weights = [1.0, 3.0, 0.1]

    # FP32
    lr = WarmupCosineDecay(args.lr, steps * args.epochs, steps * 3)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=loss_weights)
    model.fit(train_x, (tc, tr, tconf), batch_size=args.batch_size, epochs=args.epochs, verbose=2)

    # QAT
    print("Starting QAT...")
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(args.qat_lr, steps * args.qat_epochs, steps)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=losses, loss_weights=loss_weights)
    qat.fit(train_x, (tc, tr, tconf), batch_size=args.batch_size, epochs=args.qat_epochs, verbose=2)

    # Export
    tflite_path = args.output / "ellipse_repvgg_int8.tflite"
    contract = export_int8(qat, train_x, tflite_path)

    report = {
        "model": f"ellipse_repvgg_fused_a{args.alpha}",
        "train_images": len(train_x),
        "fp32_epochs": args.epochs,
        "qat_epochs": args.qat_epochs,
        "tflite_int8": contract,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
