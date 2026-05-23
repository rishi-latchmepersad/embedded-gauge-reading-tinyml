#!/usr/bin/env python3
"""
Generate prediction visualization overlays for geometry points model.

This script creates visual overlays showing:
- True vs predicted center and tip positions
- True vs predicted needle lines
- Temperature labels and errors

Usage:
    poetry run python ml/scripts/generate_geometry_predictions_viz.py

Output:
    ml/debug/geometry_points_v1_predictions/
        - best_20_overlays/
        - worst_30_overlays/
        - random_30_overlays/
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class PredictionRow:
    """Holds one prediction row from CSV."""
    image_path: str
    split: str
    true_temperature_c: float
    predicted_temperature_c: float
    absolute_error_c: float
    true_center_x_224: float
    true_center_y_224: float
    predicted_center_x_224: float
    predicted_center_y_224: float
    true_tip_x_224: float
    true_tip_y_224: float
    predicted_tip_x_224: float
    predicted_tip_y_224: float
    true_angle_degrees: float
    predicted_angle_degrees: float
    confidence: float


def load_predictions(pred_path: Path) -> List[PredictionRow]:
    """Load predictions from CSV file."""
    rows = []
    
    with open(pred_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            row = PredictionRow(
                image_path=row_dict["image_path"],
                split=row_dict["split"],
                true_temperature_c=float(row_dict["true_temperature_c"]),
                predicted_temperature_c=float(row_dict["predicted_temperature_c"]),
                absolute_error_c=float(row_dict["absolute_error_c"]),
                true_center_x_224=float(row_dict["true_center_x_224"]),
                true_center_y_224=float(row_dict["true_center_y_224"]),
                predicted_center_x_224=float(row_dict["predicted_center_x_224"]),
                predicted_center_y_224=float(row_dict["predicted_center_y_224"]),
                true_tip_x_224=float(row_dict["true_tip_x_224"]),
                true_tip_y_224=float(row_dict["true_tip_y_224"]),
                predicted_tip_x_224=float(row_dict["predicted_tip_x_224"]),
                predicted_tip_y_224=float(row_dict["predicted_tip_y_224"]),
                true_angle_degrees=float(row_dict["true_angle_degrees"]),
                predicted_angle_degrees=float(row_dict["predicted_angle_degrees"]),
                confidence=float(row_dict["confidence"]),
            )
            rows.append(row)
    
    return rows


def create_prediction_overlay(
    pred: PredictionRow,
    source_image: Image.Image,
    output_path: Path,
    base_path: Path,
) -> bool:
    """
    Create overlay showing true vs predicted geometry.

    Shows:
    - True center (green circle) and tip (red circle)
    - Predicted center (green square) and tip (red square)
    - True needle line (solid blue)
    - Predicted needle line (dashed cyan)
    - Temperature labels and error

    Args:
        pred: Prediction row
        source_image: Source PIL Image
        output_path: Output path for overlay
        base_path: Base path for loading images

    Returns:
        True if successful
    """
    if source_image is None:
        return False
    
    # Create copy for drawing
    img = source_image.copy()
    draw = ImageDraw.Draw(img)
    
    # We need to map 224x224 coordinates back to source image coordinates
    # First, we need to know the crop that was used
    # For simplicity, we'll show the full image with a crop box indicator
    
    # Draw points at 224 scale (we'll create a side-by-side view)
    # Left: true geometry, Right: predicted geometry
    
    # Actually, let's create a simpler overlay:
    # Show the source image with both true and predicted overlaid
    
    # Since we don't have the exact crop box in predictions, we'll show
    # a normalized representation
    
    # For now, let's create a visualization that shows:
    # 1. The source image resized to 448x448
    # 2. A 224x224 crop representation with points overlaid
    
    display_size = 448
    img_resized = source_image.resize((display_size, display_size), Image.LANCZOS)
    draw = ImageDraw.Draw(img_resized)
    
    # Scale 224 coordinates to display size
    scale = display_size / 224.0
    
    # Draw true center (green circle)
    true_center_radius = 8
    draw.ellipse(
        [
            pred.true_center_x_224 * scale - true_center_radius,
            pred.true_center_y_224 * scale - true_center_radius,
            pred.true_center_x_224 * scale + true_center_radius,
            pred.true_center_y_224 * scale + true_center_radius,
        ],
        fill="green",
        outline="white",
        width=2,
    )
    
    # Draw true tip (red circle)
    true_tip_radius = 8
    draw.ellipse(
        [
            pred.true_tip_x_224 * scale - true_tip_radius,
            pred.true_tip_y_224 * scale - true_tip_radius,
            pred.true_tip_x_224 * scale + true_tip_radius,
            pred.true_tip_y_224 * scale + true_tip_radius,
        ],
        fill="red",
        outline="white",
        width=2,
    )
    
    # Draw true needle line (solid blue)
    draw.line(
        [
            (pred.true_center_x_224 * scale, pred.true_center_y_224 * scale),
            (pred.true_tip_x_224 * scale, pred.true_tip_y_224 * scale),
        ],
        fill="blue",
        width=3,
    )
    
    # Draw predicted center (green square)
    pred_center_radius = 6
    draw.rectangle(
        [
            pred.predicted_center_x_224 * scale - pred_center_radius,
            pred.predicted_center_y_224 * scale - pred_center_radius,
            pred.predicted_center_x_224 * scale + pred_center_radius,
            pred.predicted_center_y_224 * scale + pred_center_radius,
        ],
        fill="lightgreen",
        outline="darkgreen",
        width=2,
    )
    
    # Draw predicted tip (red square)
    pred_tip_radius = 6
    draw.rectangle(
        [
            pred.predicted_tip_x_224 * scale - pred_tip_radius,
            pred.predicted_tip_y_224 * scale - pred_tip_radius,
            pred.predicted_tip_x_224 * scale + pred_tip_radius,
            pred.predicted_tip_y_224 * scale + pred_tip_radius,
        ],
        fill="pink",
        outline="darkred",
        width=2,
    )
    
    # Draw predicted needle line (dashed cyan)
    # PIL doesn't support dashed lines directly, so we draw multiple segments
    pred_center = (pred.predicted_center_x_224 * scale, pred.predicted_center_y_224 * scale)
    pred_tip = (pred.predicted_tip_x_224 * scale, pred.predicted_tip_y_224 * scale)
    
    # Draw as a thinner cyan line
    draw.line(
        [pred_center, pred_tip],
        fill="cyan",
        width=2,
    )
    
    # Add text overlay
    image_name = Path(pred.image_path).name
    text_lines = [
        f"Image: {image_name[:25]}",
        f"Split: {pred.split}",
        f"True Temp: {pred.true_temperature_c:.1f}C",
        f"Pred Temp: {pred.predicted_temperature_c:.1f}C",
        f"Error: {pred.absolute_error_c:.2f}C",
        f"Confidence: {pred.confidence:.3f}",
        f"True Angle: {pred.true_angle_degrees:.1f} deg",
        f"Pred Angle: {pred.predicted_angle_degrees:.1f} deg",
    ]
    
    # Draw text background
    text_height = 20 * len(text_lines) + 10
    draw.rectangle([(0, 0), (350, text_height)], fill=(0, 0, 0, 180))
    
    # Draw text
    for i, line in enumerate(text_lines):
        y = 5 + i * 20
        # Color-code error
        if pred.absolute_error_c < 2.0:
            text_color = "lightgreen"
        elif pred.absolute_error_c < 5.0:
            text_color = "yellow"
        elif pred.absolute_error_c < 10.0:
            text_color = "orange"
        else:
            text_color = "red"
        
        if "Error:" in line:
            draw.text((5, y), line, fill=text_color)
        else:
            draw.text((5, y), line, fill="white")
    
    # Save
    img_resized.save(output_path, "JPEG", quality=85)
    return True


def load_source_image(image_path: Path) -> Optional[Image.Image]:
    """Load source image from disk."""
    try:
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        return None
    except Exception:
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate prediction visualizations")
    parser.add_argument("--pred-path", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Geometry Points Prediction Visualizations")
    print("=" * 80)
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    
    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        pred_path = base_path / "ml" / "artifacts" / "training" / "geometry_points_v1" / "test_predictions.csv"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "ml" / "debug" / "geometry_points_v1_predictions"
    
    print(f"\nPredictions: {pred_path}")
    print(f"Output dir: {output_dir}")
    
    # Load predictions
    print("\nLoading predictions...")
    if not pred_path.exists():
        print(f"ERROR: Predictions not found at {pred_path}")
        print("Run eval_geometry_points_v1.py first.")
        sys.exit(1)
    
    predictions = load_predictions(pred_path)
    print(f"Loaded {len(predictions)} predictions")
    
    # Create output directories
    best_dir = output_dir / "best_20_overlays"
    worst_dir = output_dir / "worst_30_overlays"
    random_dir = output_dir / "random_30_overlays"
    
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort predictions by error
    sorted_by_error = sorted(predictions, key=lambda p: p.absolute_error_c)
    
    # Best 20 (lowest error)
    best_20 = sorted_by_error[:20]
    
    # Worst 30 (highest error)
    worst_30 = sorted_by_error[-30:][::-1]  # Reverse so worst is first
    
    # Random 30
    rng = random.Random(42)
    random_30 = rng.sample(predictions, min(30, len(predictions)))
    
    # Generate overlays
    print("\nGenerating best 20 overlays...")
    best_count = 0
    for i, pred in enumerate(best_20):
        image_path = base_path / pred.image_path
        source_image = load_source_image(image_path)
        
        if source_image is None:
            continue
        
        output_path = best_dir / f"best_{i:02d}_error{pred.absolute_error_c:.2f}C_{Path(pred.image_path).stem}.jpg"
        if create_prediction_overlay(pred, source_image, output_path, base_path):
            best_count += 1
    
    print(f"  Created {best_count} overlays")
    
    print("\nGenerating worst 30 overlays...")
    worst_count = 0
    for i, pred in enumerate(worst_30):
        image_path = base_path / pred.image_path
        source_image = load_source_image(image_path)
        
        if source_image is None:
            continue
        
        output_path = worst_dir / f"worst_{i:02d}_error{pred.absolute_error_c:.2f}C_{Path(pred.image_path).stem}.jpg"
        if create_prediction_overlay(pred, source_image, output_path, base_path):
            worst_count += 1
    
    print(f"  Created {worst_count} overlays")
    
    print("\nGenerating random 30 overlays...")
    random_count = 0
    for i, pred in enumerate(random_30):
        image_path = base_path / pred.image_path
        source_image = load_source_image(image_path)
        
        if source_image is None:
            continue
        
        output_path = random_dir / f"random_{i:02d}_error{pred.absolute_error_c:.2f}C_{Path(pred.image_path).stem}.jpg"
        if create_prediction_overlay(pred, source_image, output_path, base_path):
            random_count += 1
    
    print(f"  Created {random_count} overlays")
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  Best 20: {best_dir} ({best_count} images)")
    print(f"  Worst 30: {worst_dir} ({worst_count} images)")
    print(f"  Random 30: {random_dir} ({random_count} images)")


if __name__ == "__main__":
    main()
