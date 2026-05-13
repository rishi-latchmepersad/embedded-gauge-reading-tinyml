#!/usr/bin/env python3
"""
Evaluate SOTA model vs prod v0.3 on hard cases.

This script provides a comprehensive comparison between the new SOTA model
and the current production baseline (prod v0.3) on the hard cases manifest.

Usage:
    poetry run python scripts/eval_sota_vs_prod.py \
        --sota-model artifacts/training/sota_v1/best_model.keras \
        --prod-model artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite \
        --manifest data/hard_cases.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ─── Constants ───────────────────────────────────────────────────────────────

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224

# ─── Data Loading ────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a CSV manifest with image_path,value columns."""
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_path": row["image_path"],
                    "value": float(row["value"]),
                }
            )
    return rows


def resolve_image_path(rel_path: str) -> Path:
    """Resolve a relative image path to an absolute path."""
    p = Path(rel_path)
    if p.is_absolute():
        return p

    # Strip leading 'ml/' prefix
    rel = rel_path
    if rel.startswith("ml/") or rel.startswith("ml\\"):
        rel = rel[3:]

    # Try relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / rel
    if candidate.exists():
        return candidate

    # Try REPO_ROOT
    candidate = REPO_ROOT / rel
    if candidate.exists():
        return candidate

    return PROJECT_ROOT / rel_path


def load_rgb_image(path: Path) -> np.ndarray:
    """Load image and resize to IMAGE_SIZE."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.uint8)


def crop_and_resize(img_hw3: np.ndarray) -> np.ndarray:
    """Apply the fixed training crop and resize with padding."""
    h, w = img_hw3.shape[:2]
    x0 = int(TRAINING_CROP_X_MIN * w)
    x1 = int(TRAINING_CROP_X_MAX * w)
    y0 = int(TRAINING_CROP_Y_MIN * h)
    y1 = int(TRAINING_CROP_Y_MAX * h)
    crop = img_hw3[y0:y1, x0:x1]

    # Resize with pad
    crop_h, crop_w = crop.shape[:2]
    scale = min(IMAGE_SIZE / crop_w, IMAGE_SIZE / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))

    from PIL import Image

    pil_crop = Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR)

    # Paste onto canvas
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    x_offset = (IMAGE_SIZE - new_w) // 2
    y_offset = (IMAGE_SIZE - new_h) // 2
    canvas.paste(pil_crop, (x_offset, y_offset))

    return np.array(canvas, dtype=np.float32) / 255.0


def load_and_prepare_input(path: Path) -> np.ndarray:
    """Load image and prepare for model input."""
    img = load_rgb_image(path)
    return crop_and_resize(img)


# ─── Model Evaluation ────────────────────────────────────────────────────────


def evaluate_keras_model(model_path: Path, manifest_rows: list[dict[str, Any]]) -> dict:
    """Evaluate a Keras model on the manifest."""
    import tensorflow as tf

    print(f"\n{'='*70}")
    print(f"  Evaluating Keras model: {model_path.name}")
    print(f"{'='*70}")

    model = tf.keras.models.load_model(str(model_path), compile=False)

    errors = []
    predictions = []

    for row in manifest_rows:
        img_path = resolve_image_path(row["image_path"])
        if not img_path.exists() or img_path.stat().st_size == 0:
            print(f"  SKIP: {img_path.name} (missing/empty)")
            continue

        true_val = row["value"]
        inp = load_and_prepare_input(img_path)

        # Handle multi-output models
        pred_out = model.predict(inp[None], verbose=0)

        if isinstance(pred_out, dict):
            if "gauge_value" in pred_out:
                pred = float(pred_out["gauge_value"][0][0])
            else:
                # Take first output
                first_key = list(pred_out.keys())[0]
                pred = float(pred_out[first_key][0][0])
        else:
            pred = float(np.asarray(pred_out).flatten()[0])

        err = pred - true_val
        errors.append(err)
        predictions.append(
            {
                "image_path": str(img_path),
                "true_value": true_val,
                "predicted_value": pred,
                "error": err,
            }
        )

        print(
            f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {img_path.name}"
        )

    return {
        "predictions": predictions,
        "errors": errors,
        "metrics": _compute_metrics(errors),
    }


def evaluate_tflite_model(
    model_path: Path, manifest_rows: list[dict[str, Any]]
) -> dict:
    """Evaluate a TFLite model on the manifest."""
    import tensorflow as tf

    print(f"\n{'='*70}")
    print(f"  Evaluating TFLite model: {model_path.name}")
    print(f"{'='*70}")

    interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    in_scale, in_zp = inp_details["quantization"]
    out_scale, out_zp = out_details["quantization"]

    print(f"Input: {inp_details['shape']} quant={inp_details['quantization']}")
    print(f"Output: {out_details['shape']} quant={out_details['quantization']}\n")

    errors = []
    predictions = []

    for row in manifest_rows:
        img_path = resolve_image_path(row["image_path"])
        if not img_path.exists() or img_path.stat().st_size == 0:
            print(f"  SKIP: {img_path.name} (missing/empty)")
            continue

        true_val = row["value"]
        img = load_rgb_image(img_path)
        inp_img = crop_and_resize(img)

        # Quantize input
        q = np.clip(np.round(inp_img / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_details["index"], q[None])
        interp.invoke()

        # Dequantize output
        raw = interp.get_tensor(out_details["index"]).flatten()[0]
        pred = (float(raw.astype(np.float32)) - out_zp) * out_scale

        err = pred - true_val
        errors.append(err)
        predictions.append(
            {
                "image_path": str(img_path),
                "true_value": true_val,
                "predicted_value": pred,
                "error": err,
            }
        )

        print(
            f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {img_path.name}"
        )

    return {
        "predictions": predictions,
        "errors": errors,
        "metrics": _compute_metrics(errors),
    }


def _compute_metrics(errors: list[float]) -> dict[str, Any]:
    """Compute aggregate metrics from per-sample errors."""
    if not errors:
        return {
            "mae": -1.0,
            "rmse": -1.0,
            "max_error": -1.0,
            "count": 0,
            "bias": 0.0,
            "std": 0.0,
        }

    arr = np.array(errors)

    mae = float(np.abs(arr).mean())
    rmse = float(np.sqrt(np.mean(arr**2)))
    bias = float(arr.mean())
    std = float(arr.std())
    max_error = float(np.abs(arr).max())

    over_5 = sum(1 for e in errors if abs(e) > 5.0)
    over_10 = sum(1 for e in errors if abs(e) > 10.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "count": len(errors),
        "bias": bias,
        "std": std,
        "cases_over_5c": over_5,
        "cases_over_10c": over_10,
    }


def print_metrics_summary(metrics: dict[str, Any], label: str = "") -> None:
    """Print a formatted metrics summary."""
    prefix = f"{label} " if label else ""
    print(f"\n{prefix}Metrics:")
    print(f"  n={metrics['count']}")
    print(f"  MAE={metrics['mae']:.4f}°C")
    print(f"  RMSE={metrics['rmse']:.4f}°C")
    print(f"  Max Error={metrics['max_error']:.2f}°C")
    print(f"  Bias={metrics['bias']:+.2f}°C")
    print(f"  Std={metrics['std']:.2f}°C")
    print(f"  Cases > 5°C: {metrics['cases_over_5c']}/{metrics['count']}")
    print(f"  Cases > 10°C: {metrics['cases_over_10c']}/{metrics['count']}")


def print_comparison(sota_metrics: dict, prod_metrics: dict) -> None:
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'SOTA':>12} {'Prod v0.3':>12} {'Improvement':>12}")
    print("-" * 70)

    for key in ["mae", "rmse", "max_error", "bias", "std"]:
        sota_val = sota_metrics[key]
        prod_val = prod_metrics[key]
        if prod_val != 0:
            improvement = ((prod_val - sota_val) / abs(prod_val)) * 100
        else:
            improvement = 0.0

        improvement_str = f"{improvement:+.1f}%"
        if key in ["mae", "rmse", "max_error"]:
            # Lower is better
            if improvement > 0:
                improvement_str = f"↑ {improvement:.1f}%"
            else:
                improvement_str = f"↓ {improvement:.1f}%"

        print(f"{key:<20} {sota_val:>12.4f} {prod_val:>12.4f} {improvement_str:>12}")

    print("-" * 70)
    print(
        f"{'Cases > 5°C':<20} {sota_metrics['cases_over_5c']:>12} {prod_metrics['cases_over_10c']:>12}"
    )
    print(
        f"{'Cases > 10°C':<20} {sota_metrics['cases_over_10c']:>12} {prod_metrics['cases_over_10c']:>12}"
    )

    # Overall improvement
    mae_improvement = (
        (prod_metrics["mae"] - sota_metrics["mae"]) / prod_metrics["mae"]
    ) * 100
    print("=" * 70)
    if mae_improvement > 0:
        print(f"✓ SOTA model is {mae_improvement:.1f}% better on MAE")
    else:
        print(f"✗ SOTA model is {abs(mae_improvement):.1f}% worse on MAE")
    print("=" * 70)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOTA model vs prod v0.3")

    parser.add_argument(
        "--sota-model",
        type=Path,
        required=True,
        help="Path to SOTA Keras model",
    )

    parser.add_argument(
        "--prod-model",
        type=Path,
        required=True,
        help="Path to prod v0.3 TFLite model",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "hard_cases.csv",
        help="Path to hard cases manifest",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval_sota_vs_prod",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    manifest_rows = load_manifest(args.manifest)
    print(f"Loaded {len(manifest_rows)} samples\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate SOTA model
    sota_results = evaluate_keras_model(args.sota_model, manifest_rows)

    # Evaluate prod model
    prod_results = evaluate_tflite_model(args.prod_model, manifest_rows)

    # Print comparison
    print_metrics_summary(sota_results["metrics"], "SOTA")
    print_metrics_summary(prod_results["metrics"], "Prod v0.3")
    print_comparison(sota_results["metrics"], prod_results["metrics"])

    # Save results
    results = {
        "sota": {
            "model_path": str(args.sota_model),
            "metrics": sota_results["metrics"],
            "predictions": sota_results["predictions"],
        },
        "prod": {
            "model_path": str(args.prod_model),
            "metrics": prod_results["metrics"],
            "predictions": prod_results["predictions"],
        },
        "comparison": {
            "mae_improvement_pct": (
                (prod_results["metrics"]["mae"] - sota_results["metrics"]["mae"])
                / prod_results["metrics"]["mae"]
            )
            * 100,
        },
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed predictions
    with open(args.output_dir / "sota_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "true_value", "predicted_value", "error"]
        )
        writer.writeheader()
        writer.writerows(sota_results["predictions"])

    with open(args.output_dir / "prod_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "true_value", "predicted_value", "error"]
        )
        writer.writeheader()
        writer.writerows(prod_results["predictions"])

    print(f"\nResults saved to {args.output_dir}")
    print(f"  - results.json")
    print(f"  - sota_predictions.csv")
    print(f"  - prod_predictions.csv")


if __name__ == "__main__":
    main()
