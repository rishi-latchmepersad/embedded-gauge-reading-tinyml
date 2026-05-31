#!/usr/bin/env python3
"""Ensemble inference - combine coordinate and direct temperature models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import celsius_from_inner_dial_angle_degrees, angle_degrees_from_center_to_tip

MIN_TEMP = -30.0
MAX_TEMP = 50.0
TEMP_RANGE = MAX_TEMP - MIN_TEMP


def predict_ensemble(coord_model, temp_model, image_path: str, board_weight: float = 0.7):
    """
    Ensemble prediction combining coordinate and direct temperature models.
    
    Args:
        coord_model: Coordinate regression model (good on phone photos)
        temp_model: Direct temperature model (good on board captures)
        image_path: Path to input image
        board_weight: Weight for temperature model (higher = trust temp model more)
    
    Returns:
        Ensemble temperature prediction
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(img_arr, 0)
    
    # Get coordinate prediction
    coords = coord_model.predict(batch, verbose=0)[0]
    cx, cy, tx, ty = coords
    angle = angle_degrees_from_center_to_tip(cx*224, cy*224, tx*224, ty*224)
    temp_coord = celsius_from_inner_dial_angle_degrees(angle)
    
    # Get direct temperature prediction
    temp_norm = temp_model.predict(batch, verbose=0)[0, 0]
    temp_direct = temp_norm * TEMP_RANGE + MIN_TEMP
    
    # Ensemble (weighted average)
    temp_ensemble = (1 - board_weight) * temp_coord + board_weight * temp_direct
    
    return temp_ensemble, temp_coord, temp_direct


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--coord-model", default="/tmp/coord_regression_augmented/best.keras")
    parser.add_argument("--temp-model", default="/tmp/direct_temp_regression/best.keras")
    parser.add_argument("--board-weight", type=float, default=0.5)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on validation set")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()
    
    # Load models
    coord_model = tf.keras.models.load_model(args.coord_model, compile=False)
    temp_model = tf.keras.models.load_model(args.temp_model, compile=False)
    
    if args.evaluate or not args.images:
        # Load metadata for evaluation
        with open('data/preprocessed_crops/metadata.json') as f:
            meta = json.load(f)
        
        val_samples = [s for s in meta if s['split']=='val']
        
        errors_all = []
        errors_board = []
        errors_phone = []
        
        for s in val_samples:
            img_path = 'data/preprocessed_crops/' + s['image_path']
            temp_ens, temp_coord, temp_direct = predict_ensemble(
                coord_model, temp_model, img_path, args.board_weight
            )
            
            gt_temp = s['temperature_c']
            error = abs(temp_ens - gt_temp)
            errors_all.append(error)
            
            is_board = 'capture_' in s['image_path']
            if is_board:
                errors_board.append(error)
            else:
                errors_phone.append(error)
        
        print(f'Ensemble Results (board_weight={args.board_weight}):')
        print(f'Overall ({len(errors_all)}): MAE={np.mean(errors_all):.2f}°C, Median={np.median(errors_all):.2f}°C')
        print(f'Phone ({len(errors_phone)}): MAE={np.mean(errors_phone):.2f}°C, Median={np.median(errors_phone):.2f}°C')
        print(f'Board ({len(errors_board)}): MAE={np.mean(errors_board):.2f}°C, Median={np.median(errors_board):.2f}°C')
    else:
        # Predict individual images
        for img_path in args.images:
            temp_ens, temp_coord, temp_direct = predict_ensemble(
                coord_model, temp_model, img_path, args.board_weight
            )
            print(f'{img_path}: ensemble={temp_ens:.1f}°C (coord={temp_coord:.1f}, direct={temp_direct:.1f})')
