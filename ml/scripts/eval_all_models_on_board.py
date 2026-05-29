#!/usr/bin/env python3
"""Evaluate all trained models on board captures."""

from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

# Board capture ground truth
BOARD_CAPTURES_GT = {
    # From board_captures_labeled_v2.csv
    "capture_0007.png": 18.0,
    "capture_0008.png": 22.0,
    "capture_0073.png": 46.0,
    "capture_0075.png": 19.0,
    "capture_0c_preview.png": 0.0,
    "capture_2026-04-03_08-20-49.png": 45.0,
    "capture_2026-04-03_13-48-34.png": 30.0,
    "capture_2026-04-03_15-46-04.png": 19.0,
    "capture_2026-04-22_07-15-36.png": 28.0,
    "capture_2026-04-22_07-16-50.png": 28.0,
    "capture_2026-04-22_07-18-03.png": 28.0,
    "capture_2026-04-22_07-19-17.png": 28.0,
    "capture_2026-04-22_07-20-31.png": 28.0,
    "capture_2026-04-22_07-21-44.png": 28.0,
    "capture_2026-04-22_07-22-58.png": 28.0,
    "capture_2026-04-22_07-24-43.png": 28.0,
    "capture_2026-04-22_07-25-57.png": 28.0,
    "capture_2026-04-22_07-27-11.png": 28.0,
    "capture_2026-04-22_07-28-25.png": 28.0,
    "capture_2026-04-22_07-29-39.png": 28.0,
    "capture_2026-04-22_07-30-53.png": 28.0,
    "capture_2026-04-22_07-32-07.png": 28.0,
    "capture_2026-04-22_07-33-52.png": 28.0,
    "capture_2026-04-22_07-35-06.png": 28.0,
    "capture_2026-04-22_07-36-19.png": 28.0,
    "capture_2026-04-22_07-37-33.png": 28.0,
    "capture_2026-04-22_07-38-47.png": 28.0,
    "capture_2026-04-22_07-40-01.png": 28.0,
    "capture_2026-04-24_22-24-04.png": 0.0,
    "capture_p31c_preview.png": 31.0,
    "capture_p35c_preview.png": 35.0,
    "capture_p42c.png": 42.0,
    "capture_p45c.png": 45.0,
    "capture_p50c_preview.png": 50.0,
    "capture_p5c.png": 5.0,
}

# Models to evaluate
MODELS = {
    "coord_simple": "/tmp/coord_regression/best.keras",
    "coord_augmented": "/tmp/coord_regression_augmented/best.keras",
    "coord_mobilenet_phase2": "/tmp/coord_regression_mobilenet/best_phase2.keras",
    "heatmap_v1": "/tmp/heatmap_angle_quick/best.keras",
    "heatmap_angle": "/tmp/heatmap_angle_train/checkpoint_01.keras",
    "geometry_transfer": "/tmp/geometry_cnn_transfer/best.keras",
}


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load image and preprocess to 224x224."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_arr, 0)


def predict_coord(model, batch: np.ndarray) -> tuple[float, float, float, float]:
    """Predict coordinates from model."""
    coords = model.predict(batch, verbose=0)[0]
    return coords[0], coords[1], coords[2], coords[3]


def predict_heatmap(model, batch: np.ndarray):
    """Predict heatmaps from model."""
    preds = model.predict(batch, verbose=0)
    if isinstance(preds, list):
        # Multiple outputs
        return preds[0][0], preds[1][0]  # center, tip
    else:
        # Single output
        return preds[0, ..., 0], preds[0, ..., 1]  # Assume channel split


def soft_argmax(heatmap: np.ndarray) -> tuple[float, float]:
    """Compute soft argmax of heatmap."""
    heatmap = np.clip(heatmap, 0, 1)
    total = heatmap.sum()
    if total < 1e-6:
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return float(idx[1]), float(idx[0])
    
    heatmap_norm = heatmap / total
    h, w = heatmap.shape
    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    p_x = heatmap_norm.sum(axis=0)
    p_y = heatmap_norm.sum(axis=1)
    x = (x_coords * p_x).sum()
    y = (y_coords * p_y).sum()
    return x / w, y / h  # Normalize to [0, 1]


def coords_to_temp(cx: float, cy: float, tx: float, ty: float) -> float:
    """Convert normalized coordinates to temperature."""
    angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
    temp = celsius_from_inner_dial_angle_degrees(angle)
    return temp


def evaluate_model(model_path: str, model_type: str, base_dir: Path):
    """Evaluate a model on board captures."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    errors = []
    results = []
    
    for img_name, gt_temp in BOARD_CAPTURES_GT.items():
        img_path = base_dir / "captured_images" / img_name
        if not img_path.exists():
            continue
        
        try:
            batch = load_and_preprocess(str(img_path))
            
            if "heatmap" in model_type.lower():
                center_hm, tip_hm = predict_heatmap(model, batch)
                cx, cy = soft_argmax(center_hm)
                tx, ty = soft_argmax(tip_hm)
            else:
                cx, cy, tx, ty = predict_coord(model, batch)
            
            pred_temp = coords_to_temp(cx, cy, tx, ty)
            error = abs(pred_temp - gt_temp)
            errors.append(error)
            results.append((img_name, gt_temp, pred_temp, error))
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    if not errors:
        print("No images processed!")
        return None
    
    errors = np.array(errors)
    print(f"\nResults ({len(errors)} images):")
    print(f"  MAE: {errors.mean():.2f}°C")
    print(f"  Median: {np.median(errors):.2f}°C")
    print(f"  Std: {errors.std():.2f}°C")
    print(f"  Max error: {errors.max():.2f}°C")
    
    # Show worst predictions
    results.sort(key=lambda x: -x[3])
    print("\n  Worst predictions:")
    for name, gt, pred, err in results[:5]:
        print(f"    {name}: GT={gt:5.1f}°C, Pred={pred:5.1f}°C, Error={err:5.1f}°C")
    
    return {
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "count": len(errors),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    
    print("Evaluating all models on board captures...")
    print(f"Testing {len(BOARD_CAPTURES_GT)} board captures")
    
    results = {}
    for name, path in MODELS.items():
        result = evaluate_model(path, name, base_dir)
        if result:
            results[name] = result
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'MAE':>8} {'Median':>8} {'Std':>8}")
    print("-"*60)
    for name, res in sorted(results.items(), key=lambda x: x[1]["mae"]):
        print(f"{name:<30} {res['mae']:>8.2f} {res['median']:>8.2f} {res['std']:>8.2f}")
    
    # Save results
    with open("/tmp/model_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/model_eval_results.json")


if __name__ == "__main__":
    main()
