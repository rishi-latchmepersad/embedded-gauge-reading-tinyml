#!/usr/bin/env python3
"""Inference for coordinate regression gauge reader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

# Gauge constants (from firmware)
COLD_END_ANGLE = 135.0
SWEEP_ANGLE = 270.0
MIN_TEMP = -30.0
MAX_TEMP = 50.0

def load_model(model_path: str):
    """Load trained coordinate regression model."""
    return tf.keras.models.load_model(model_path, compile=False)


def predict_temperature(model, image_path: str) -> tuple[float, float, float, float]:
    """Predict temperature from image.
    
    Returns:
        (temperature_c, center_x, center_y, tip_x, tip_y) all in normalized coords
    """
    img = Image.open(image_path).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, -1)
    elif img_arr.shape[-1] == 1:
        img_arr = np.repeat(img_arr, 3, -1)
    
    # Ensure 224x224
    if img_arr.shape[0] != 224 or img_arr.shape[1] != 224:
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        img = img.resize((224, 224), Image.BILINEAR)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
    
    batch = np.expand_dims(img_arr, 0)
    coords = model.predict(batch, verbose=0)[0]
    
    center_x, center_y, tip_x, tip_y = coords
    
    # Convert to angle then temperature (coords are in 224x224 pixel space)
    angle = angle_degrees_from_center_to_tip(
        center_x * 224, center_y * 224, tip_x * 224, tip_y * 224,
    )
    temp = celsius_from_inner_dial_angle_degrees(
        angle,
        cold_angle_degrees=COLD_END_ANGLE,
        sweep_degrees=SWEEP_ANGLE,
        minimum_celsius=MIN_TEMP,
        maximum_celsius=MAX_TEMP,
    )
    
    return temp, center_x, center_y, tip_x, tip_y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("images", nargs="+", help="Image files to process")
    args = parser.parse_args()
    
    model = load_model(args.model)
    
    for img_path in args.images:
        temp, cx, cy, tx, ty = predict_temperature(model, img_path)
        print(f"{img_path}: {temp:.1f}°C (center={cx:.3f},{cy:.3f}, tip={tx:.3f},{ty:.3f})")
