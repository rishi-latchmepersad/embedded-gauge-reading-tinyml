"""Evaluate gpu5_recover on a manifest and save predictions to CSV."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224


def load_luma_yuv422(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE // 2, 4)
    luma = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return luma


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        compile=False,
    )

    rows = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            img_col = next((k for k in row if "image" in k.lower() or "path" in k.lower() or "file" in k.lower()), list(row.keys())[0])
            val_col = next((k for k in row if "value" in k.lower() or "label" in k.lower() or "temp" in k.lower()), list(row.keys())[1])
            p = Path(row[img_col])
            if not p.is_absolute():
                p = args.repo_root / p
            rows.append((p, float(row[val_col])))

    results = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            continue
        if path.suffix == ".yuv422":
            luma = load_luma_yuv422(path)
            img = np.repeat(luma[:, :, None], 3, axis=2)
        else:
            img = load_rgb_png(path)
        inp = crop_and_resize(img)
        pred = float(np.asarray(model.predict(inp[None], verbose=0)).flatten()[0])
        results.append((str(path), true_val, pred))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true", "pred"])
        for r in results:
            writer.writerow(r)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
