#!/usr/bin/env python3
"""Geometry Heatmap v1 Evaluation Script."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    SourceGeometryExample,

    create_jittered_crop,
    load_geometry_manifest,
)
from embedded_gauge_reading_tinyml.heatmap_utils import (
    decode_heatmap_to_pixel_coords,
)


def create_identity_crop_input(
    example: SourceGeometryExample,
    base_path: Path,
    input_size: int = 224,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create 224x224 crop using identity jitter."""
    from embedded_gauge_reading_tinyml.geometry_crop_dataset import JitterParams
    from PIL import Image
    
    jitter = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
    crop_result = create_jittered_crop(example, jitter)
    
    if not crop_result.accepted:
        raise ValueError(f"Crop rejected: {crop_result.rejection_reason}")
    
    # Load image and extract crop
    image_path = base_path / crop_result.source_image_path
    image = Image.open(image_path).convert("RGB")
    crop_box = (crop_result.crop_x1, crop_result.crop_y1, crop_result.crop_x2, crop_result.crop_y2)
    crop_image = image.crop(crop_box)
    crop_image = crop_image.resize((input_size, input_size))
    
    crop_array = np.array(crop_image, dtype=np.float32) / 255.0
    
    # Build metadata dict from crop_result
    crop_meta = {
        "crop_x1": crop_result.crop_x1,
        "crop_y1": crop_result.crop_y1,
        "crop_x2": crop_result.crop_x2,
        "crop_y2": crop_result.crop_y2,
        "center_x_224": crop_result.center_x_224,
        "center_y_224": crop_result.center_y_224,
        "tip_x_224": crop_result.tip_x_224,
        "tip_y_224": crop_result.tip_y_224,
    }
    
    return crop_array, crop_meta


def predict_on_sample(
    model: Any,
    crop: np.ndarray,
    example: SourceGeometryExample,
    crop_metadata: dict[str, Any],
    input_size: int = 224,
    heatmap_size: int = 56,
) -> dict[str, Any]:
    """Run model prediction and decode heatmaps."""
    pred = model.predict(crop[np.newaxis, ...], verbose=0)
    
    center_heatmap = pred[0][0, ..., 0]
    tip_heatmap = pred[1][0, ..., 0]
    confidence = float(pred[2][0, 0])
    
    # decode_heatmap_to_pixel_coords with input_size=224 returns coords in 224-pixel space
    center_x_224, center_y_224 = decode_heatmap_to_pixel_coords(
        center_heatmap, method="softargmax", input_size=input_size
    )
    tip_x_224, tip_y_224 = decode_heatmap_to_pixel_coords(
        tip_heatmap, method="softargmax", input_size=input_size
    )
    
    predicted_angle = angle_degrees_from_center_to_tip(
        center_x_224, center_y_224, tip_x_224, tip_y_224
    )
    predicted_temp = celsius_from_inner_dial_angle_degrees(predicted_angle)
    
    true_center_x_224 = crop_metadata["center_x_224"]
    true_center_y_224 = crop_metadata["center_y_224"]
    true_tip_x_224 = crop_metadata["tip_x_224"]
    true_tip_y_224 = crop_metadata["tip_y_224"]
    
    true_angle = angle_degrees_from_center_to_tip(
        true_center_x_224, true_center_y_224, true_tip_x_224, true_tip_y_224
    )
    true_temp = example.temperature_c
    
    center_error = np.sqrt(
        (center_x_224 - true_center_x_224) ** 2 + 
        (center_y_224 - true_center_y_224) ** 2
    )
    tip_error = np.sqrt(
        (tip_x_224 - true_tip_x_224) ** 2 + 
        (tip_y_224 - true_tip_y_224) ** 2
    )
    angle_error = abs(circular_angle_error_degrees(predicted_angle, true_angle))
    temp_error = abs(predicted_temp - true_temp)
    
    return {
        "image_path": str(example.image_path),
        "split": example.split,
        "true_temperature_c": true_temp,
        "predicted_temperature_c": predicted_temp,
        "absolute_error_c": temp_error,
        "true_center_x_224": true_center_x_224,
        "true_center_y_224": true_center_y_224,
        "predicted_center_x_224": center_x_224,
        "predicted_center_y_224": center_y_224,
        "true_tip_x_224": true_tip_x_224,
        "true_tip_y_224": true_tip_y_224,
        "predicted_tip_x_224": tip_x_224,
        "predicted_tip_y_224": tip_y_224,
        "true_angle_degrees": true_angle,
        "predicted_angle_degrees": predicted_angle,
        "confidence": confidence,
        "center_heatmap_peak_value": float(np.max(center_heatmap)),
        "tip_heatmap_peak_value": float(np.max(tip_heatmap)),
        "center_pixel_error": center_error,
        "tip_pixel_error": tip_error,
        "angle_error_degrees": angle_error,
    }


def evaluate_model(
    model: Any,
    examples: list[SourceGeometryExample],
    base_path: Path,
    output_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate model on a list of examples."""
    predictions = []
    
    for i, example in enumerate(examples):
        try:
            crop, crop_meta = create_identity_crop_input(example, base_path, 224)
            pred = predict_on_sample(model, crop, example, crop_meta)
            predictions.append(pred)
            print(f"  {split_name} [{i+1}/{len(examples)}]: {example.image_path if isinstance(example.image_path, str) else example.image_path.name}")
        except Exception as e:
            print(f"  ERROR processing {example.image_path if isinstance(example.image_path, str) else example.image_path.name}: {e}")
            continue
    
    if not predictions:
        return {"error": "No predictions generated"}
    
    center_errors = [p["center_pixel_error"] for p in predictions]
    tip_errors = [p["tip_pixel_error"] for p in predictions]
    angle_errors = [p["angle_error_degrees"] for p in predictions]
    temp_errors = [p["absolute_error_c"] for p in predictions]
    
    metrics = {
        "split": split_name,
        "num_samples": len(predictions),
        "center_px_mae_224": float(np.mean(center_errors)),
        "tip_px_mae_224": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "temperature_mae_c": float(np.mean(temp_errors)),
        "temperature_rmse_c": float(np.sqrt(np.mean(np.square(temp_errors)))),
        "percentage_under_2c": float(np.mean([e < 2 for e in temp_errors]) * 100),
        "percentage_under_5c": float(np.mean([e < 5 for e in temp_errors]) * 100),
        "percentage_under_10c": float(np.mean([e < 10 for e in temp_errors]) * 100),
        "mean_confidence": float(np.mean([p["confidence"] for p in predictions])),
    }
    
    import csv
    csv_path = output_dir / f"{split_name}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"  Saved {len(predictions)} predictions to {csv_path}")
    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Geometry Heatmap v1 Model")
    parser.add_argument("--model-path", type=str, default="ml/artifacts/training/geometry_heatmap_v1/model.keras")
    parser.add_argument("--manifest-path", type=str, default="ml/data/geometry_reader_manifest_v2_clean.csv")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts/training/geometry_heatmap_v1")
    parser.add_argument("--base-path", type=str, default=None)
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    base_path = Path(args.base_path) if args.base_path else Path(__file__).parent.parent.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Geometry Heatmap v1 Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    print("\nLoading model...")
    from tensorflow import keras
    model = keras.models.load_model(str(model_path))
    print("Model loaded successfully")
    
    print("\nLoading clean manifest...")
    all_examples = load_geometry_manifest(manifest_path)
    clean_examples = [e for e in all_examples if getattr(e, "quality_flag", "clean") == "clean"]
    print(f"Loaded {len(clean_examples)} clean rows")
    
    train_examples = [e for e in clean_examples if e.split == "train"]
    val_examples = [e for e in clean_examples if e.split == "val"]
    test_examples = [e for e in clean_examples if e.split == "test"]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")
    
    all_metrics = {}
    
    print("\n" + "=" * 80)
    print("Evaluating on training set...")
    print("=" * 80)
    train_metrics = evaluate_model(model, train_examples, base_path, output_dir, "train")
    all_metrics["train"] = train_metrics
    print(f"\nTrain metrics: {json.dumps(train_metrics, indent=2)}")
    
    print("\n" + "=" * 80)
    print("Evaluating on validation set...")
    print("=" * 80)
    val_metrics = evaluate_model(model, val_examples, base_path, output_dir, "val")
    all_metrics["val"] = val_metrics
    print(f"\nVal metrics: {json.dumps(val_metrics, indent=2)}")
    
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    test_metrics = evaluate_model(model, test_examples, base_path, output_dir, "test")
    all_metrics["test"] = test_metrics
    print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")
    
    if "test" in all_metrics and test_examples:
        test_preds_path = output_dir / "test_predictions.csv"
        import csv
        test_preds = []
        with open(test_preds_path, "r") as f:
            reader = csv.DictReader(f)
            test_preds = list(reader)
        
        test_preds_sorted = sorted(test_preds, key=lambda x: float(x["absolute_error_c"]), reverse=True)
        worst_30 = test_preds_sorted[:30]
        
        worst_path = output_dir / "worst_30_predictions.csv"
        with open(worst_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test_preds[0].keys())
            writer.writeheader()
            writer.writerows(worst_30)
        
        print(f"\nSaved worst 30 predictions to {worst_path}")
    
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nSaved metrics to {metrics_path}")
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
