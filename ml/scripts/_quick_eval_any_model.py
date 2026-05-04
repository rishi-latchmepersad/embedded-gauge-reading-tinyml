"""Evaluate any Keras model on any CSV manifest."""

from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224


def load_rgb_png(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.uint8)


def crop_and_resize(img_hw3: np.ndarray) -> np.ndarray:
    h, w = img_hw3.shape[:2]
    x0, x1 = int(TRAINING_CROP_X_MIN * w), int(TRAINING_CROP_X_MAX * w)
    y0, y1 = int(TRAINING_CROP_Y_MIN * h), int(TRAINING_CROP_Y_MAX * h)
    crop = img_hw3[y0:y1, x0:x1]
    rgb = crop.astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def main():
    if len(sys.argv) < 3:
        print("Usage: python _quick_eval_any_model.py <model_path> <manifest_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    manifest_path = Path(sys.argv[2])

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    print(f"Loading model from {model_path}...", flush=True)
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        },
        compile=False,
    )
    print("Model loaded.", flush=True)

    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["image_path"].startswith("#"):
                continue
            p = Path(row["image_path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            rows.append((p, float(row["value"])))

    print(f"Manifest: {manifest_path.name}  ({len(rows)} images)\n", flush=True)

    errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            print(f"  SKIP (missing/empty): {path.name}", flush=True)
            continue
        img = load_rgb_png(path)
        inp = crop_and_resize(img)
        pred = float(np.asarray(model.predict(inp[None], verbose=0)).flatten()[0])
        err = pred - true_val
        errors.append((true_val, pred, err))
        print(
            f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {path.name}",
            flush=True,
        )

    if errors:
        arr = np.array([e[2] for e in errors])
        print(
            f"\nn={len(arr)}  MAE={np.abs(arr).mean():.2f}  bias={arr.mean():.2f}  std={arr.std():.2f}",
            flush=True,
        )
        print(f"Cases > 5C: {sum(1 for e in errors if abs(e[2]) > 5)}", flush=True)
        print(f"Cases > 10C: {sum(1 for e in errors if abs(e[2]) > 10)}", flush=True)
        # Show worst
        errors.sort(key=lambda x: -abs(x[2]))
        print("\nWorst failures:", flush=True)
        for e in errors[:10]:
            print(f"  err={e[2]:+.2f}  true={e[0]:6.1f}  pred={e[1]:7.2f}", flush=True)


if __name__ == "__main__":
    main()
