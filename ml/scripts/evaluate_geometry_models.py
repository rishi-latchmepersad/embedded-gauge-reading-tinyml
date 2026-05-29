#!/usr/bin/env python3
"""Evaluate geometric models on board captures."""

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
    circular_angle_error_degrees,
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
    "capture_2026-04-22_07-15-36.png": 28.0,
    "capture_2026-04-22_07-25-57.png": 28.0,
    "capture_2026-04-22_07-33-52.png": 28.0,
    "capture_2026-04-22_07-37-33.png": 28.0,
    "capture_2026-04-22_07-38-47.png": 28.0,
    "capture_2026-04-24_22-24-04.png": 0.0,
    "capture_p31c_preview.png": 31.0,
    "capture_p35c_preview.png": 35.0,
    "capture_p45c.png": 45.0,
    "capture_p50c_preview.png": 50.0,
    "capture_p5c.png": 5.0,
}


def load_and_crop(image_path: Path, target_size: int = 224) -> np.ndarray:
    """Load image and apply luma crop."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.uint8)

    centroid = estimate_bright_centroid(img_arr)
    height, width = img_arr.shape[:2]
    crop_box = compute_dynamic_crop(
        width=width,
        height=height,
        center_x=centroid.center_x,
        center_y=centroid.center_y,
    )

    if crop_box is None:
        raise ValueError("Crop failed")

    crop = crop_and_resize(img_arr, crop_box, target_size=target_size)
    return crop.astype(np.float32) / 255.0


def predict_coordinates(model, img: np.ndarray) -> np.ndarray:
    """Predict coordinates from model."""
    batch = np.expand_dims(img, 0)
    coords = model.predict(batch, verbose=0)[0]
    return coords


def coords_to_temp(cx: float, cy: float, tx: float, ty: float, image_size: int = 224) -> float:
    """Convert coordinates to temperature."""
    angle = angle_degrees_from_center_to_tip(
        cx * image_size, cy * image_size, tx * image_size, ty * image_size
    )
    temp = celsius_from_inner_dial_angle_degrees(angle)
    return temp


def evaluate_model(model_path: Path, base_dir: Path) -> dict:
    """Evaluate model on board captures."""
    print(f"\nEvaluating: {model_path.name}")

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    angle_errors = []
    temp_errors = []
    results = []

    for img_name, gt_temp in BOARD_CAPTURES_GT.items():
        img_path = base_dir / "captured_images" / img_name
        if not img_path.exists():
            continue

        try:
            img = load_and_crop(img_path)
            coords = predict_coordinates(model, img)
            cx, cy, tx, ty = coords[0], coords[1], coords[2], coords[3]

            pred_temp = coords_to_temp(cx, cy, tx, ty)
            
            # We don't have GT coordinates, only temperature
            # So we can only compute temperature error
            temp_error = abs(pred_temp - gt_temp)
            temp_errors.append(temp_error)
            results.append((img_name, gt_temp, pred_temp, temp_error))

        except Exception as e:
            print(f"  Error on {img_name}: {e}")

    if not temp_errors:
        return None

    temp_errors = np.array(temp_errors)
    mae = float(temp_errors.mean())

    print(f"  Temp MAE: {mae:.2f}°C (n={len(temp_errors)})")

    return {
        "temp_mae": mae,
        "temp_median": float(np.median(temp_errors)),
        "temp_std": float(temp_errors.std()),
        "temp_max": float(temp_errors.max()),
        "count": len(temp_errors),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data"
    exp_root = Path("/tmp/gauge_geometry")

    if not exp_root.exists():
        print("No experiments found")
        return

    print("=" * 60)
    print("GEOMETRIC MODEL EVALUATION")
    print("=" * 60)

    results = {}

    for exp_dir in exp_root.glob("*/"):
        if not exp_dir.is_dir():
            continue

        keras_model = exp_dir / "best.keras"
        if keras_model.exists():
            result = evaluate_model(keras_model, base_dir)
            if result:
                results[exp_dir.name] = result
                result["model_path"] = str(keras_model)

        tflite_model = exp_dir / "model_int8.tflite"
        if tflite_model.exists():
            result = evaluate_model(tflite_model, base_dir)
            if result:
                results[f"{exp_dir.name}_int8"] = result
                result["model_path"] = str(tflite_model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("No models evaluated")
        return

    sorted_results = sorted(results.items(), key=lambda x: x[1]["temp_mae"])

    print(f"{'Model':<45} {'MAE':>8} {'Median':>8} {'Max':>8}")
    print("-" * 70)

    for name, res in sorted_results:
        mae_str = f"{res['temp_mae']:.2f}"
        if res["temp_mae"] < 5.0:
            mae_str += " ✓"
        print(f"{name:<45} {mae_str:>8} {res['temp_median']:>8.2f} {res['temp_max']:>8.2f}")

    with open(exp_root / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: {exp_root / 'all_results.json'}")

    best_name, best_res = sorted_results[0]
    print(f"\n{'='*60}")
    print(f"BEST: {best_name}")
    print(f"Board MAE: {best_res['temp_mae']:.2f}°C")
    if best_res["temp_mae"] < 5.0:
        print("✓ TARGET ACHIEVED!")
    else:
        print(f"✗ Need <5°C, got {best_res['temp_mae']:.2f}°C")


if __name__ == "__main__":
    main()
