"""Hybrid CNN + Classical baseline evaluator.

Uses classical baseline when confidence is high, CNN otherwise.
"""
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
    parser.add_argument("--model", type=Path, required=True, help="CNN model path")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data/hard_cases_plus_board30_valid_with_new6.csv")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--classical-thresh", type=float, default=20.0, help="Confidence threshold for classical baseline")
    args = parser.parse_args()

    # Load CNN model
    cnn_model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        compile=False,
    )

    # Import classical baseline
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from embedded_gauge_reading_tinyml.baseline_manifest_eval import evaluate_manifest, GeometryEvaluationConfig
    from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    geo_config = GeometryEvaluationConfig(
        mode="hough_only",
        confidence_threshold=4.0,
        center_radius_scale=0.45,
    )

    # Evaluate classical baseline on manifest
    print("[HYBRID] Running classical baseline...")
    classical_result = evaluate_manifest(
        args.manifest,
        spec,
        config=geo_config,
        repo_root=args.repo_root,
    )
    classical_preds = {}
    for pred in classical_result.result.predictions:
        name = Path(pred.image_path).name
        classical_preds[name] = {
            "true": pred.true_value,
            "pred": pred.predicted_value,
            "err": pred.abs_error,
            "conf": pred.confidence,
        }

    # Load manifest rows
    rows = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            img_col = next((k for k in row if "image" in k.lower() or "path" in k.lower() or "file" in k.lower()), list(row.keys())[0])
            val_col = next((k for k in row if "value" in k.lower() or "label" in k.lower() or "temp" in k.lower()), list(row.keys())[1])
            p = Path(row[img_col])
            if not p.is_absolute():
                p = args.repo_root / p
            rows.append((p, float(row[val_col])))

    print(f"[HYBRID] Manifest: {args.manifest.name} ({len(rows)} images)")
    print(f"[HYBRID] Classical succeeded: {len(classical_preds)}/{len(rows)}")
    print(f"[HYBRID] Classical threshold: {args.classical_thresh}")
    print()

    # Hybrid evaluation
    errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            continue
        name = path.name

        # Get CNN prediction
        if path.suffix == ".yuv422":
            luma = load_luma_yuv422(path)
            img = np.repeat(luma[:, :, None], 3, axis=2)
        else:
            img = load_rgb_png(path)
        inp = crop_and_resize(img)
        cnn_pred = float(np.asarray(cnn_model.predict(inp[None], verbose=0)).flatten()[0])

        # Decide which to use
        if name in classical_preds and classical_preds[name]["conf"] >= args.classical_thresh:
            pred = classical_preds[name]["pred"]
            source = "classical"
        else:
            pred = cnn_pred
            source = "cnn"

        err = pred - true_val
        errors.append((true_val, pred, err, source))
        marker = "*" if source == "classical" else " "
        print(f"{marker} true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  src={source:10s}  {name}")

    arr = np.array([e[2] for e in errors])
    classical_count = sum(1 for e in errors if e[3] == "classical")
    print(f"\nn={len(arr)}  MAE={np.abs(arr).mean():.2f}  bias={arr.mean():.2f}  std={arr.std():.2f}")
    print(f"Used classical on {classical_count}/{len(arr)} images")


if __name__ == "__main__":
    main()
