"""Evaluate CNN gauge reader against ground truth and baseline.

This script evaluates the trained CNN model on labeled test data:
1. Load labeled manifest CSV with ground truth temperatures
2. Run CNN inference on each image
3. Compute error metrics (MAE, RMSE, accuracy within tolerance)
4. Compare against baseline polar voting method
5. Identify hard cases where CNN outperforms/underperforms baseline

Usage:
    # Evaluate on board captures
    python ml/scripts/evaluate_cnn.py \
        --labels ml/data/board_captures_labeled_v2.csv \
        --model /tmp/heatmap_angle_quick/final.keras \
        --output-dir /tmp/eval_results/
    
    # With baseline comparison
    python ml/scripts/evaluate_cnn.py \
        --labels ml/data/board_captures_labeled_v2.csv \
        --model /tmp/heatmap_angle_quick/final.keras \
        --compare-baseline \
        --output-dir /tmp/eval_results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
    angle_degrees_from_center_to_tip,
)

from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
    CropBox,
)

from read_gauge_cnn import (
    load_model,
    soft_argmax_2d,
    decode_heatmaps,
    run_inference,
)


# ---------------------------------------------------------------------------
# Baseline Implementation
# ---------------------------------------------------------------------------


def baseline_polar_voting_estimate_angle(
    image: NDArray[np.uint8],
) -> float | None:
    """Estimate gauge angle using baseline polar voting method.
    
    This is a simplified emulation of the C baseline algorithm.
    
    Args:
        image: RGB image array
        
    Returns:
        Estimated angle in degrees, or None if detection failed
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image
    
    # Threshold for bright pixels (matching C baseline)
    BRIGHT_THRESHOLD = 150
    bright_mask = gray > BRIGHT_THRESHOLD
    
    # Find centroid of bright region
    bright_coords = np.column_stack(np.where(bright_mask))
    if len(bright_coords) < 1024:  # MIN_BRIGHT_PIXELS
        return None
    
    center_y = bright_coords[:, 0].mean()
    center_x = bright_coords[:, 1].mean()
    
    # For now, return a placeholder - full baseline emulation is complex
    # In practice, we'd run the actual C code or a full Python port
    return None


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Results for one evaluated image."""
    image_path: str
    ground_truth_temp: float
    predicted_temp: float
    temp_error: float
    ground_truth_angle: float
    predicted_angle: float
    angle_error: float
    confidence: float
    crop_source: str
    baseline_temp: float | None = None
    baseline_error: float | None = None


def compute_metrics(results: list[EvaluationResult]) -> dict[str, Any]:
    """Compute aggregate evaluation metrics.
    
    Args:
        results: List of per-image evaluation results
        
    Returns:
        Dict with aggregate metrics
    """
    if len(results) == 0:
        return {"error": "No results to evaluate"}
    
    temp_errors = np.array([r.temp_error for r in results])
    angle_errors = np.array([r.angle_error for r in results])
    confidences = np.array([r.confidence for r in results])
    
    metrics = {
        "num_samples": len(results),
        "temperature": {
            "mae": float(np.mean(np.abs(temp_errors))),
            "rmse": float(np.sqrt(np.mean(temp_errors ** 2))),
            "max_error": float(np.max(np.abs(temp_errors))),
            "std": float(np.std(temp_errors)),
            "within_1c": float(np.mean(np.abs(temp_errors) <= 1.0)),
            "within_2c": float(np.mean(np.abs(temp_errors) <= 2.0)),
            "within_5c": float(np.mean(np.abs(temp_errors) <= 5.0)),
        },
        "angle": {
            "mae": float(np.mean(np.abs(angle_errors))),
            "rmse": float(np.sqrt(np.mean(angle_errors ** 2))),
            "max_error": float(np.max(np.abs(angle_errors))),
        },
        "confidence": {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
    }
    
    # Baseline comparison if available
    baseline_results = [r for r in results if r.baseline_temp is not None]
    if baseline_results:
        baseline_errors = np.array([r.baseline_error for r in baseline_results])
        cnn_errors = np.array([r.temp_error for r in baseline_results])
        
        metrics["baseline_comparison"] = {
            "num_samples": len(baseline_results),
            "baseline_mae": float(np.mean(np.abs(baseline_errors))),
            "cnn_mae": float(np.mean(np.abs(cnn_errors))),
            "improvement": float(np.mean(np.abs(baseline_errors)) - np.mean(np.abs(cnn_errors))),
            "cnn_better": float(np.mean(np.abs(cnn_errors) < np.abs(baseline_errors))),
            "baseline_better": float(np.mean(np.abs(baseline_errors) < np.abs(cnn_errors))),
        }
    
    return metrics


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------


def load_labels(csv_path: Path) -> list[dict[str, Any]]:
    """Load labeled manifest CSV.
    
    Args:
        csv_path: Path to labels CSV file
        
    Returns:
        List of label dicts
    """
    import csv
    
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                "image_path": row["image_path"],
                "temperature_c": float(row["temperature_c"]),
                "angle_degrees": float(row["angle_degrees"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "tip_x": float(row["tip_x"]),
                "tip_y": float(row["tip_y"]),
            })
    
    return labels


def evaluate_model(
    labels: list[dict[str, Any]],
    model,
    output_dir: Path | None = None,
    compare_baseline: bool = False,
) -> list[EvaluationResult]:
    """Evaluate model on labeled dataset.
    
    Args:
        labels: List of label dicts
        model: Loaded Keras model
        output_dir: Optional directory for saving results
        compare_baseline: Whether to compare against baseline
        
    Returns:
        List of evaluation results
    """
    results = []
    
    # Resolve paths relative to project root (parent of ml/)
    project_root = Path(__file__).parent.parent.parent
    
    for i, label in enumerate(labels):
        image_path = project_root / label["image_path"]
        
        # Check if image exists
        if not image_path.exists():
            print(f"  [{i+1}/{len(labels)}] {image_path.name} - NOT FOUND")
            continue
        
        print(f"  [{i+1}/{len(labels)}] {image_path.name}...", end=" ", flush=True)
        
        # Load image
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
        
        # Run CNN inference
        cnn_result = run_inference(image, model)
        predicted_temp = cnn_result["decoded"]["temperature_c"]
        predicted_angle = cnn_result["decoded"]["angle_degrees"]
        confidence = cnn_result["confidence"]
        crop_source = cnn_result["crop_source"]
        
        # Compute errors
        gt_temp = label["temperature_c"]
        gt_angle = label["angle_degrees"]
        temp_error = predicted_temp - gt_temp
        angle_error = circular_angle_error_degrees(predicted_angle, gt_angle)
        
        # Baseline comparison (placeholder for now)
        baseline_temp = None
        baseline_error = None
        if compare_baseline:
            # TODO: Implement full baseline emulation
            pass
        
        # Store result
        result = EvaluationResult(
            image_path=str(image_path),
            ground_truth_temp=gt_temp,
            predicted_temp=predicted_temp,
            temp_error=temp_error,
            ground_truth_angle=gt_angle,
            predicted_angle=predicted_angle,
            angle_error=angle_error,
            confidence=confidence,
            crop_source=crop_source,
            baseline_temp=baseline_temp,
            baseline_error=baseline_error,
        )
        results.append(result)
        
        # Print summary
        status = "OK" if abs(temp_error) <= 2.0 else "WARN"
        print(f"{status} CNN={predicted_temp:.1f}°C GT={gt_temp:.1f}°C err={temp_error:+.2f}°C")
        
        # Save visualization if requested
        if output_dir:
            from read_gauge_cnn import visualize_result
            vis_path = output_dir / f"{image_path.stem}_eval.png"
            visualize_result(image, cnn_result, vis_path)
    
    return results


def save_results(
    results: list[EvaluationResult],
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save evaluation results to files.
    
    Args:
        results: Per-image evaluation results
        metrics: Aggregate metrics
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # Save per-image results
    results_path = output_dir / "results.csv"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("image_path,gt_temp,cnn_temp,temp_error,gt_angle,cnn_angle,angle_error,confidence,crop_source\n")
        for r in results:
            f.write(f"{r.image_path},{r.ground_truth_temp:.2f},{r.predicted_temp:.2f},"
                    f"{r.temp_error:+.2f},{r.ground_truth_angle:.2f},{r.predicted_angle:.2f},"
                    f"{r.angle_error:+.2f},{r.confidence:.3f},{r.crop_source}\n")
    
    # Save worst cases
    worst_cases = sorted(results, key=lambda r: abs(r.temp_error), reverse=True)[:10]
    worst_path = output_dir / "worst_cases.json"
    worst_path.write_text(json.dumps([{
        "image_path": r.image_path,
        "ground_truth_temp": r.ground_truth_temp,
        "predicted_temp": r.predicted_temp,
        "temp_error": r.temp_error,
        "angle_error": r.angle_error,
        "confidence": r.confidence,
    } for r in worst_cases], indent=2), encoding="utf-8")
    
    print(f"\nResults saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate CNN gauge reader")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labeled manifest CSV",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare against baseline polar voting",
    )
    args = parser.parse_args()
    
    # Validate inputs
    if not args.labels.exists():
        parser.error(f"Labels file not found: {args.labels}")
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    
    print(f"Loading labels from {args.labels}...")
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labeled images")
    
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model)
    
    print("\nEvaluating model...")
    results = evaluate_model(
        labels,
        model,
        output_dir=args.output_dir,
        compare_baseline=args.compare_baseline,
    )
    
    print(f"\nEvaluated {len(results)} images")
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples: {metrics['num_samples']}")
    print(f"\nTemperature MAE: {metrics['temperature']['mae']:.2f}°C")
    print(f"Temperature RMSE: {metrics['temperature']['rmse']:.2f}°C")
    print(f"Max Error: {metrics['temperature']['max_error']:.2f}°C")
    print(f"Within 1°C: {metrics['temperature']['within_1c']*100:.1f}%")
    print(f"Within 2°C: {metrics['temperature']['within_2c']*100:.1f}%")
    print(f"Within 5°C: {metrics['temperature']['within_5c']*100:.1f}%")
    print(f"\nAngle MAE: {metrics['angle']['mae']:.2f}°")
    print(f"Mean Confidence: {metrics['confidence']['mean']:.3f}")
    
    if "baseline_comparison" in metrics:
        bc = metrics["baseline_comparison"]
        print(f"\nBaseline Comparison:")
        print(f"  Baseline MAE: {bc['baseline_mae']:.2f}°C")
        print(f"  CNN MAE: {bc['cnn_mae']:.2f}°C")
        print(f"  Improvement: {bc['improvement']:+.2f}°C")
        print(f"  CNN Better: {bc['cnn_better']*100:.1f}%")
    
    print("=" * 60)
    
    # Save results
    if args.output_dir:
        save_results(results, metrics, args.output_dir)


if __name__ == "__main__":
    main()
