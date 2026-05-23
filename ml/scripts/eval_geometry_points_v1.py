#!/usr/bin/env python3
"""
Evaluation script for geometry points prediction model (v1).

This script evaluates the trained model on train/val/test splits and generates
detailed metrics and prediction reports.

Usage:
    poetry run python ml/scripts/eval_geometry_points_v1.py

Output:
    ml/artifacts/training/geometry_points_v1/
        - test_predictions.csv
        - worst_30_predictions.csv
    ml/reports/geometry_points_v1_eval.md
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    JitterParams,
    load_geometry_manifest,
    SourceGeometryExample,
    create_jittered_crop,
    generate_jitter_params,
)
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)


@dataclass
class PredictionResult:
    """Holds prediction results for one sample."""
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


def load_clean_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load v2_clean manifest filtered to clean rows."""
    rows = []
    
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            if row_dict.get("quality_flag", "clean") != "clean":
                continue
            rows.append(row_dict)
    
    return rows


def prepare_example_from_row(row: Dict[str, Any]) -> SourceGeometryExample:
    """Convert manifest row to SourceGeometryExample."""
    return SourceGeometryExample(
        image_path=row["image_path"],
        temperature_c=float(row["temperature_c"]),
        split=row["split"],
        source_width=int(row["source_width"]),
        source_height=int(row["source_height"]),
        loose_crop_x1=float(row["loose_crop_x1"]),
        loose_crop_y1=float(row["loose_crop_y1"]),
        loose_crop_x2=float(row["loose_crop_x2"]),
        loose_crop_y2=float(row["loose_crop_y2"]),
        center_x_source=float(row["center_x_source"]),
        center_y_source=float(row["center_y_source"]),
        tip_x_source=float(row["tip_x_source"]),
        tip_y_source=float(row["tip_y_source"]),
        dial_radius_source=float(row["dial_radius_source"]),
    )


def create_identity_crop_input(
    example: SourceGeometryExample,
    input_size: int = 224,
    base_path: Path = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Create input for evaluation using identity crop (no jitter).

    Args:
        example: Source geometry example
        input_size: Target input size
        base_path: Base path for image loading

    Returns:
        Tuple of (image_array, label_vector, metadata) or None
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent
    
    # Create identity jitter (no shift, scale=1.0, aspect=1.0)
    jitter = JitterParams(
        shift_x=0,
        shift_y=0,
        scale=1.0,
        aspect=1.0,
    )
    
    crop = create_jittered_crop(example, jitter)
    
    if not crop.accepted:
        return None
    
    # Load image
    image_path = base_path / crop.source_image_path
    
    if not image_path.exists():
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    
    # Extract crop region
    crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
    crop_image = image.crop(crop_box)
    
    # Resize to input size
    crop_resized = crop_image.resize((input_size, input_size), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(crop_resized, dtype=np.float32) / 255.0
    
    # Create label vector
    label_vector = np.array([
        crop.center_x_normalized,
        crop.center_y_normalized,
        crop.tip_x_normalized,
        crop.tip_y_normalized,
        1.0,
    ], dtype=np.float32)
    
    # Metadata
    metadata = {
        "source_image_path": crop.source_image_path,
        "split": crop.split,
        "temperature_c": crop.temperature_c,
        "true_center_x_224": crop.center_x_224,
        "true_center_y_224": crop.center_y_224,
        "true_tip_x_224": crop.tip_x_224,
        "true_tip_y_224": crop.tip_y_224,
    }
    
    return image_array, label_vector, metadata


def compute_angle_from_coords(
    center_x: float,
    center_y: float,
    tip_x: float,
    tip_y: float,
) -> float:
    """Compute angle from center and tip coordinates."""
    dx = tip_x - center_x
    dy = tip_y - center_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg


def compute_temperature_from_angle(angle_deg: float) -> float:
    """Compute temperature from angle using default calibration."""
    return celsius_from_inner_dial_angle_degrees(angle_deg)


def evaluate_model(
    model: keras.Model,
    examples: List[SourceGeometryExample],
    base_path: Path,
    input_size: int = 224,
) -> List[PredictionResult]:
    """
    Evaluate model on a list of examples.

    Args:
        model: Trained Keras model
        examples: List of source geometry examples
        base_path: Base path for image loading
        input_size: Input image size

    Returns:
        List of PredictionResult objects
    """
    results = []
    
    for example in examples:
        sample = create_identity_crop_input(example, input_size, base_path)
        
        if sample is None:
            continue
        
        image_array, label_vector, metadata = sample
        
        # Run prediction
        pred = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
        
        # Extract predictions
        pred_center_x = float(pred[0])
        pred_center_y = float(pred[1])
        pred_tip_x = float(pred[2])
        pred_tip_y = float(pred[3])
        pred_confidence = float(pred[4])
        
        # True values (from 224x224 coordinates)
        true_center_x_224 = metadata["true_center_x_224"]
        true_center_y_224 = metadata["true_center_y_224"]
        true_tip_x_224 = metadata["true_tip_x_224"]
        true_tip_y_224 = metadata["true_tip_y_224"]
        
        # Convert normalized predictions to 224 scale
        pred_center_x_224 = pred_center_x * 224.0
        pred_center_y_224 = pred_center_y * 224.0
        pred_tip_x_224 = pred_tip_x * 224.0
        pred_tip_y_224 = pred_tip_y * 224.0
        
        # Compute angles
        true_angle = compute_angle_from_coords(
            true_center_x_224, true_center_y_224,
            true_tip_x_224, true_tip_y_224,
        )
        pred_angle = compute_angle_from_coords(
            pred_center_x_224, pred_center_y_224,
            pred_tip_x_224, pred_tip_y_224,
        )
        
        # Compute temperatures
        true_temp = metadata["temperature_c"]
        pred_temp = compute_temperature_from_angle(pred_angle)
        
        # Compute error
        abs_error = abs(true_temp - pred_temp)
        
        result = PredictionResult(
            image_path=metadata["source_image_path"],
            split=metadata["split"],
            true_temperature_c=true_temp,
            predicted_temperature_c=pred_temp,
            absolute_error_c=abs_error,
            true_center_x_224=true_center_x_224,
            true_center_y_224=true_center_y_224,
            predicted_center_x_224=pred_center_x_224,
            predicted_center_y_224=pred_center_y_224,
            true_tip_x_224=true_tip_x_224,
            true_tip_y_224=true_tip_y_224,
            predicted_tip_x_224=pred_tip_x_224,
            predicted_tip_y_224=pred_tip_y_224,
            true_angle_degrees=true_angle,
            predicted_angle_degrees=pred_angle,
            confidence=pred_confidence,
        )
        results.append(result)
    
    return results


def compute_metrics(results: List[PredictionResult]) -> Dict[str, Any]:
    """
    Compute evaluation metrics from prediction results.

    Args:
        results: List of PredictionResult objects

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {}
    
    # Temperature errors
    temp_errors = [r.absolute_error_c for r in results]
    temp_mae = sum(temp_errors) / len(temp_errors)
    temp_rmse = math.sqrt(sum(e**2 for e in temp_errors) / len(temp_errors))
    
    # Percentage under thresholds
    under_2c = sum(1 for e in temp_errors if e < 2.0) / len(temp_errors) * 100
    under_5c = sum(1 for e in temp_errors if e < 5.0) / len(temp_errors) * 100
    under_10c = sum(1 for e in temp_errors if e < 10.0) / len(temp_errors) * 100
    
    # Coordinate errors (in 224x224 pixel space)
    center_x_errors = [abs(r.predicted_center_x_224 - r.true_center_x_224) for r in results]
    center_y_errors = [abs(r.predicted_center_y_224 - r.true_center_y_224) for r in results]
    tip_x_errors = [abs(r.predicted_tip_x_224 - r.true_tip_x_224) for r in results]
    tip_y_errors = [abs(r.predicted_tip_y_224 - r.true_tip_y_224) for r in results]
    
    center_px_mae_224 = (
        sum(center_x_errors) + sum(center_y_errors)
    ) / (2 * len(results))
    tip_px_mae_224 = (
        sum(tip_x_errors) + sum(tip_y_errors)
    ) / (2 * len(results))
    
    # Angle errors
    angle_errors = [
        circular_angle_error_degrees(r.predicted_angle_degrees, r.true_angle_degrees)
        for r in results
    ]
    angle_mae_degrees = sum(angle_errors) / len(angle_errors)
    
    # Confidence stats
    confidences = [r.confidence for r in results]
    mean_confidence = sum(confidences) / len(confidences)
    
    return {
        "num_samples": len(results),
        "temperature_mae_c": temp_mae,
        "temperature_rmse_c": temp_rmse,
        "percentage_under_2c": under_2c,
        "percentage_under_5c": under_5c,
        "percentage_under_10c": under_10c,
        "center_px_mae_224": center_px_mae_224,
        "tip_px_mae_224": tip_px_mae_224,
        "angle_mae_degrees": angle_mae_degrees,
        "mean_confidence": mean_confidence,
    }


def save_predictions(
    results: List[PredictionResult],
    output_path: Path,
) -> None:
    """Save predictions to CSV."""
    fieldnames = [
        "image_path",
        "split",
        "true_temperature_c",
        "predicted_temperature_c",
        "absolute_error_c",
        "true_center_x_224",
        "true_center_y_224",
        "predicted_center_x_224",
        "predicted_center_y_224",
        "true_tip_x_224",
        "true_tip_y_224",
        "predicted_tip_x_224",
        "predicted_tip_y_224",
        "true_angle_degrees",
        "predicted_angle_degrees",
        "confidence",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                "image_path": r.image_path,
                "split": r.split,
                "true_temperature_c": f"{r.true_temperature_c:.2f}",
                "predicted_temperature_c": f"{r.predicted_temperature_c:.2f}",
                "absolute_error_c": f"{r.absolute_error_c:.2f}",
                "true_center_x_224": f"{r.true_center_x_224:.2f}",
                "true_center_y_224": f"{r.true_center_y_224:.2f}",
                "predicted_center_x_224": f"{r.predicted_center_x_224:.2f}",
                "predicted_center_y_224": f"{r.predicted_center_y_224:.2f}",
                "true_tip_x_224": f"{r.true_tip_x_224:.2f}",
                "true_tip_y_224": f"{r.true_tip_y_224:.2f}",
                "predicted_tip_x_224": f"{r.predicted_tip_x_224:.2f}",
                "predicted_tip_y_224": f"{r.predicted_tip_y_224:.2f}",
                "true_angle_degrees": f"{r.true_angle_degrees:.2f}",
                "predicted_angle_degrees": f"{r.predicted_angle_degrees:.2f}",
                "confidence": f"{r.confidence:.4f}",
            })


def generate_eval_report(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate evaluation report in Markdown format."""
    lines = [
        "# Geometry Points v1 - Evaluation Report",
        "",
        "## Summary",
        "",
        "This report evaluates the coordinate-regression baseline model (MobileNetV2 backbone)",
        "for predicting dial center and needle tip coordinates.",
        "",
        "## Metrics by Split",
        "",
        "### Training Set",
        "",
    ]
    
    if train_metrics:
        lines.extend([
            f"- **Samples:** {train_metrics.get('num_samples', 'N/A')}",
            f"- **Temperature MAE:** {train_metrics.get('temperature_mae_c', 0):.2f}°C",
            f"- **Temperature RMSE:** {train_metrics.get('temperature_rmse_c', 0):.2f}°C",
            f"- **Angle MAE:** {train_metrics.get('angle_mae_degrees', 0):.2f}°",
            f"- **Center Pixel MAE (224x224):** {train_metrics.get('center_px_mae_224', 0):.2f}px",
            f"- **Tip Pixel MAE (224x224):** {train_metrics.get('tip_px_mae_224', 0):.2f}px",
            f"- **Under 2°C:** {train_metrics.get('percentage_under_2c', 0):.1f}%",
            f"- **Under 5°C:** {train_metrics.get('percentage_under_5c', 0):.1f}%",
            f"- **Under 10°C:** {train_metrics.get('percentage_under_10c', 0):.1f}%",
            "",
        ])
    else:
        lines.append("*Not evaluated*", "")
    
    lines.extend([
        "### Validation Set",
        "",
    ])
    
    if val_metrics:
        lines.extend([
            f"- **Samples:** {val_metrics.get('num_samples', 'N/A')}",
            f"- **Temperature MAE:** {val_metrics.get('temperature_mae_c', 0):.2f}°C",
            f"- **Temperature RMSE:** {val_metrics.get('temperature_rmse_c', 0):.2f}°C",
            f"- **Angle MAE:** {val_metrics.get('angle_mae_degrees', 0):.2f}°",
            f"- **Center Pixel MAE (224x224):** {val_metrics.get('center_px_mae_224', 0):.2f}px",
            f"- **Tip Pixel MAE (224x224):** {val_metrics.get('tip_px_mae_224', 0):.2f}px",
            f"- **Under 2°C:** {val_metrics.get('percentage_under_2c', 0):.1f}%",
            f"- **Under 5°C:** {val_metrics.get('percentage_under_5c', 0):.1f}%",
            f"- **Under 10°C:** {val_metrics.get('percentage_under_10c', 0):.1f}%",
            "",
        ])
    else:
        lines.append("*Not evaluated*", "")
    
    lines.extend([
        "### Test Set",
        "",
    ])
    
    if test_metrics:
        lines.extend([
            f"- **Samples:** {test_metrics.get('num_samples', 'N/A')}",
            f"- **Temperature MAE:** {test_metrics.get('temperature_mae_c', 0):.2f}°C",
            f"- **Temperature RMSE:** {test_metrics.get('temperature_rmse_c', 0):.2f}°C",
            f"- **Angle MAE:** {test_metrics.get('angle_mae_degrees', 0):.2f}°",
            f"- **Center Pixel MAE (224x224):** {test_metrics.get('center_px_mae_224', 0):.2f}px",
            f"- **Tip Pixel MAE (224x224):** {test_metrics.get('tip_px_mae_224', 0):.2f}px",
            f"- **Under 2°C:** {test_metrics.get('percentage_under_2c', 0):.1f}%",
            f"- **Under 5°C:** {test_metrics.get('percentage_under_5c', 0):.1f}%",
            f"- **Under 10°C:** {test_metrics.get('percentage_under_10c', 0):.1f}%",
            "",
        ])
    else:
        lines.append("*Not evaluated*", "")
    
    lines.extend([
        "## Worst Failure Modes",
        "",
        "The worst failures typically occur when:",
        "",
        "1. **Annotation errors:** Ground truth center/tip labels are incorrect",
        "2. **Extreme temperatures:** Near the ends of the dial sweep (-30°C or +50°C)",
        "3. **Poor image quality:** Blur, motion, or lighting issues",
        "4. **Occlusion:** Needle partially obscured or hard to distinguish",
        "",
        "See `worst_30_predictions.csv` for detailed analysis of worst cases.",
        "",
        "## Recommendation",
        "",
        "This coordinate-regression baseline provides a sanity check for the full pipeline.",
        "",
    ])
    
    # Add recommendation based on test MAE
    if test_metrics:
        test_mae = test_metrics.get("temperature_mae_c", 999)
        if test_mae < 3.0:
            lines.append(
                "**Temperature MAE < 3°C:** Excellent baseline. Ready to proceed to heatmap-based approach.\n"
            )
        elif test_mae < 5.0:
            lines.append(
                "**Temperature MAE < 5°C:** Good baseline. Coordinate regression is working. "
                "Heatmap approach may improve further.\n"
            )
        elif test_mae < 10.0:
            lines.append(
                "**Temperature MAE < 10°C:** Acceptable for a first baseline. "
                "Some annotation errors or model capacity issues. "
                "Recommend reviewing worst cases before proceeding.\n"
            )
        else:
            lines.append(
                "**Temperature MAE > 10°C:** Baseline needs improvement. "
                "Likely annotation errors or data pipeline issues. "
                "Review worst cases and clean manifest before proceeding.\n"
            )
    
    lines.extend([
        "",
        "---",
        "",
        "*Report generated by eval_geometry_points_v1.py*",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate geometry points model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Geometry Points Evaluation (v1)")
    print("=" * 80)
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    manifest_path = base_path / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
    
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = base_path / "ml" / "artifacts" / "training" / "geometry_points_v1" / "model.keras"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "ml" / "artifacts" / "training" / "geometry_points_v1"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel: {model_path}")
    print(f"Output dir: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_geometry_points_v1.py first.")
        sys.exit(1)
    
    model = keras.models.load_model(model_path)
    print("Model loaded successfully")
    
    # Load manifest
    print("\nLoading clean manifest...")
    rows = load_clean_manifest(manifest_path)
    print(f"Loaded {len(rows)} clean rows")
    
    # Convert to examples
    examples = [prepare_example_from_row(row) for row in rows]
    
    # Split by train/val/test
    train_examples = [ex for ex in examples if ex.split == "train"]
    val_examples = [ex for ex in examples if ex.split == "val"]
    test_examples = [ex for ex in examples if ex.split == "test"]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")
    
    # Evaluate on each split
    print("\n" + "=" * 80)
    print("Evaluating on training set...")
    train_results = evaluate_model(model, train_examples, base_path)
    train_metrics = compute_metrics(train_results)
    print(f"  Temperature MAE: {train_metrics['temperature_mae_c']:.2f}°C")
    
    print("\nEvaluating on validation set...")
    val_results = evaluate_model(model, val_examples, base_path)
    val_metrics = compute_metrics(val_results)
    print(f"  Temperature MAE: {val_metrics['temperature_mae_c']:.2f}°C")
    
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_examples, base_path)
    test_metrics = compute_metrics(test_results)
    print(f"  Temperature MAE: {test_metrics['temperature_mae_c']:.2f}°C")
    
    # Save test predictions
    print("\nSaving test predictions...")
    test_pred_path = output_dir / "test_predictions.csv"
    save_predictions(test_results, test_pred_path)
    print(f"Saved {len(test_results)} predictions to {test_pred_path}")
    
    # Save worst 30 predictions
    print("\nSaving worst 30 predictions...")
    test_results_sorted = sorted(test_results, key=lambda r: r.absolute_error_c, reverse=True)
    worst_30_path = output_dir / "worst_30_predictions.csv"
    save_predictions(test_results_sorted[:30], worst_30_path)
    print(f"Saved worst 30 to {worst_30_path}")
    
    # Generate report
    print("\nGenerating evaluation report...")
    report_path = base_path / "ml" / "reports" / "geometry_points_v1_eval.md"
    generate_eval_report(train_metrics, val_metrics, test_metrics, report_path)
    print(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    
    print("\nTest set summary:")
    print(f"  Temperature MAE: {test_metrics['temperature_mae_c']:.2f}°C")
    print(f"  Temperature RMSE: {test_metrics['temperature_rmse_c']:.2f}°C")
    print(f"  Angle MAE: {test_metrics['angle_mae_degrees']:.2f}°")
    print(f"  Center Pixel MAE: {test_metrics['center_px_mae_224']:.2f}px")
    print(f"  Tip Pixel MAE: {test_metrics['tip_px_mae_224']:.2f}px")
    print(f"  Under 2°C: {test_metrics['percentage_under_2c']:.1f}%")
    print(f"  Under 5°C: {test_metrics['percentage_under_5c']:.1f}%")
    print(f"  Under 10°C: {test_metrics['percentage_under_10c']:.1f}%")
    
    return train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    main()
