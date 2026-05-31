#!/usr/bin/env python3
"""Evaluate all geometry v2 models on board captures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
)

BOARD_CAPTURES_GT = {
    "capture_0007.png": 18.0,
    "capture_0008.png": 22.0,
    "capture_0073.png": 46.0,
    "capture_0075.png": 19.0,
    "capture_0c_preview.png": 0.0,
    "capture_2026-04-03_08-20-49.png": 45.0,
    "capture_2026-04-03_13-48-34.png": 30.0,
    "capture_2026-04-03_15-46-04.png": 19.0,
    "capture_2026-04-22_07-25-57.png": 28.0,
    "capture_2026-04-24_22-24-04.png": 0.0,
    "capture_p5c.png": 5.0,
    "capture_p35c_preview.png": 35.0,
    "capture_p45c.png": 45.0,
    "capture_p50c_preview.png": 50.0,
    "capture_p31c_preview.png": 31.0,
}


def load_and_crop(image_path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.uint8)
    centroid = estimate_bright_centroid(img_arr)
    h, w = img_arr.shape[:2]
    crop_box = compute_dynamic_crop(width=w, height=h, center_x=centroid.center_x, center_y=centroid.center_y)
    if crop_box is None:
        raise ValueError("Crop failed")
    crop = crop_and_resize(img_arr, crop_box, target_size=224)
    return crop.astype(np.float32) / 255.0


def coords_to_temp(cx, cy, tx, ty):
    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    tx = max(0.01, min(0.99, tx))
    ty = max(0.01, min(0.99, ty))
    angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
    temp = celsius_from_inner_dial_angle_degrees(angle)
    return max(-30.0, min(50.0, temp))


def evaluate_model(model_path: Path, base_dir: Path) -> dict | None:
    print(f"\nEvaluating: {model_path.parent.name}/{model_path.name}")
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    errors = []
    for img_name, gt_temp in BOARD_CAPTURES_GT.items():
        img_path = base_dir / "captured_images" / img_name
        if not img_path.exists():
            continue
        try:
            img = load_and_crop(img_path)
            batch = np.expand_dims(img, 0)
            coords = model.predict(batch, verbose=0)[0]
            pred_temp = coords_to_temp(coords[0], coords[1], coords[2], coords[3])
            errors.append(abs(pred_temp - gt_temp))
        except Exception as e:
            print(f"  Error {img_name}: {e}")

    if not errors:
        return None

    errors = np.array(errors)
    mae = float(errors.mean())
    print(f"  MAE: {mae:.2f}°C (n={len(errors)})")
    return {"temp_mae": mae, "temp_median": float(np.median(errors)), "count": len(errors)}


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data"
    exp_root = Path("/tmp/gauge_geometry_v2")
    old_root = Path("/tmp/gauge_geometry")

    print("=" * 60)
    print("EVALUATE ALL MODELS")
    print("=" * 60)

    results = {}

    for root in [exp_root, old_root]:
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("*/")):
            if not exp_dir.is_dir():
                continue
            keras_model = exp_dir / "best.keras"
            if keras_model.exists():
                r = evaluate_model(keras_model, base_dir)
                if r:
                    results[f"{root.name}/{exp_dir.name}"] = r

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("No models found")
        return

    sorted_results = sorted(results.items(), key=lambda x: x[1]["temp_mae"])
    print(f"{'Model':<45} {'MAE':>8} {'Median':>8}")
    print("-" * 65)
    for name, r in sorted_results:
        s = f"{r['temp_mae']:.2f}"
        if r['temp_mae'] < 5.0:
            s += " ✓"
        print(f"{name:<45} {s:>8} {r['temp_median']:>8.2f}")

    all_results_path = Path("/tmp/gauge_geometry_v2_all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {all_results_path}")

    best_name, best_r = sorted_results[0]
    print(f"\nBEST: {best_name} - {best_r['temp_mae']:.2f}°C MAE")
    if best_r["temp_mae"] < 5.0:
        print("✓ TARGET ACHIEVED!")
    else:
        print(f"✗ Need <5°C")


if __name__ == "__main__":
    main()
