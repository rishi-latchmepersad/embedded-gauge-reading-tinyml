#!/usr/bin/env python3
"""Train 640x640 ellipse detector with stride-4 stem for STM32 N6.

Architecture: same as proven QAT encoder but at native 640px:
  - Stride-4 stem → 160x160x24 (614KB peak activation, under 1.5MB)
  - 4 progressive downsample stages (Conv+BN+ReLU)
  - GAP + Dense(128) → multi-head regression
"""

import sys, json, random, argparse, zipfile, io
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from embedded_gauge_reading_tinyml.tf_models import _channel_plan

IMAGE_SIZE = 640
BATCH_SIZE = 4  # 640x640 is big
SEED = 42


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model(alpha=1.0):
    """640x640 → ellipse regression. Stride-4 stem keeps activations under 1.5MB."""
    cp = _channel_plan(alpha)
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])

    # Stage 0: stride-4 → 160x160 (same as proven QAT encoder)
    x = keras.layers.Conv2D(cp["stem"], 3, strides=4, padding="same", use_bias=False, name="s0_conv")(x)
    x = keras.layers.BatchNormalization(name="s0_bn")(x)
    x = keras.layers.ReLU(name="s0_relu")(x)

    # Stages 1-4: progressive stride-2 downsample
    for si, (filters, repeat) in enumerate([(cp["s1"], 2), (cp["s2"], 2), (cp["s3"], 2), (cp["s4"], 1)], 1):
        for bi in range(repeat):
            s = 2 if bi == 0 else 1
            n = f"s{si}b{bi}"
            x = keras.layers.Conv2D(filters, 3, strides=s, padding="same", use_bias=False, name=f"{n}_conv")(x)
            x = keras.layers.BatchNormalization(name=f"{n}_bn")(x)
            x = keras.layers.ReLU(name=f"{n}_relu")(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    shared = keras.layers.Dense(128, activation="relu", name="shared")(x)
    c = keras.layers.Dense(64, activation="relu", name="c_dense")(shared)
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(c)
    r = keras.layers.Dense(64, activation="relu", name="r_dense")(shared)
    radius_xy = keras.layers.Dense(2, activation=None, name="radius_xy")(r)
    f = keras.layers.Dense(32, activation="relu", name="f_dense")(shared)
    confidence = keras.layers.Dense(1, activation="sigmoid", name="confidence")(f)
    return keras.Model(inputs, [center_xy, radius_xy, confidence], name=f"ellipse_640_a{alpha}")


def load_ellipse_labels(zip_paths):
    paths, targets = [], []
    for zp in zip_paths:
        if not zp.exists(): continue
        z = zipfile.ZipFile(zp)
        try: root = ET.parse(z.open("annotations.xml")).getroot()
        except: continue
        for img in root.findall("image"):
            name = img.get("name"); w, h = int(img.get("width","640")), int(img.get("height","640"))
            target = None
            for e in img:
                if e.get("label","") in ("GaugeFace","temp_dial"):
                    if e.tag == "ellipse":
                        target = np.array([float(e.get(c))/w for c in ("cx","cy","rx","ry")] + [1.0], np.float32)
                    elif e.tag == "box":
                        xtl, ytl = float(e.get("xtl")), float(e.get("ytl"))
                        xbr, ybr = float(e.get("xbr")), float(e.get("ybr"))
                        target = np.array([(xtl+xbr)/(2*w), (ytl+ybr)/(2*h), (xbr-xtl)/(2*w), (ybr-ytl)/(2*h), 1.0], np.float32)
                    break
            if target is None: continue
            paths.append((zp, name)); targets.append(target)
    return paths, np.stack(targets)


def preload_images(paths, size=640):
    images = []
    for zp, name in paths:
        z = zipfile.ZipFile(zp)
        img_name = f"images/{name}"
        if img_name not in z.namelist():
            matches = [m for m in z.namelist() if name in m and m.endswith((".png",".jpg"))]
            if not matches: continue
            img_name = matches[0]
        data = z.read(img_name)
        img = tf.io.decode_image(data, channels=1, expand_animations=False)
        img = tf.image.resize(img, (size, size), method="bilinear")
        images.append(tf.cast(img.numpy(), tf.float32) / 255.0)
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
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "gauge_ellipse_640_v2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    configure_gpu()

    labelled = ROOT / "data" / "labelled"
    train_zips = [labelled / "train_1.zip",
                  labelled / "initial_temp_gauge" / "board_captures_1.zip",
                  labelled / "initial_temp_gauge" / "board_captures_2.zip",
                  labelled / "test_3.zip"]
    train_paths, train_t = load_ellipse_labels(train_zips)
    print(f"Train labels: {len(train_paths)}")
    print("Pre-loading images...")
    train_x = preload_images(train_paths, IMAGE_SIZE)[:, :, :, None]
    print(f"Train images: {train_x.shape}")

    tc = train_t[:, :2].astype(np.float32)
    tr = train_t[:, 2:4].astype(np.float32)
    tconf = train_t[:, 4:5].astype(np.float32)

    steps = max(1, len(train_x) // BATCH_SIZE)
    args.output.mkdir(parents=True, exist_ok=True)

    model = build_model(alpha=args.alpha)
    model.summary(line_length=100)

    losses = [keras.losses.Huber(delta=0.05)] * 3
    loss_weights = [1.0, 3.0, 0.1]

    lr = WarmupCosineDecay(args.lr, steps * args.epochs, steps * 3)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=loss_weights)
    model.fit(train_x, (tc, tr, tconf), batch_size=BATCH_SIZE, epochs=args.epochs, verbose=2)

    print("Starting QAT...")
    qat = tfmot.quantization.keras.quantize_model(model)
    qat_lr = WarmupCosineDecay(args.qat_lr, steps * args.qat_epochs, steps)
    qat.compile(optimizer=keras.optimizers.Adam(qat_lr), loss=losses, loss_weights=loss_weights)
    qat.fit(train_x, (tc, tr, tconf), batch_size=BATCH_SIZE, epochs=args.qat_epochs, verbose=2)

    tflite_path = args.output / "ellipse_640_v2_int8.tflite"
    contract = export_int8(qat, train_x, tflite_path)

    report = {"model": f"ellipse_640_a{args.alpha}", "train_images": len(train_x),
              "fp32_epochs": args.epochs, "qat_epochs": args.qat_epochs, "tflite_int8": contract}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
