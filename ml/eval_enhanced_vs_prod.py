"""
Evaluate enhanced model vs prod v0.3 scalar on hard cases manifest.

This script establishes the baseline comparison needed to determine
if the enhanced model beats prod v0.3.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Try TF import, fall back gracefully
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224

# Model paths
MODELS = {
    "enhanced_multi_scale": PROJECT_ROOT / "artifacts/training/enhanced_multi_scale_e60_s42/best_model.keras",
    "prod_v0.3_scalar": PROJECT_ROOT / "artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite",
}

# Manifest - use the hard cases plus board30 manifest used for prod v0.3 eval
MANIFEST = PROJECT_ROOT / "ml/data/hard_cases_plus_board30_valid_with_new6.csv"


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
    rgb = crop.astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def load_and_prepare_input(path: Path) -> np.ndarray:
    """Load image and prepare for model input."""
    img = load_rgb_image(path)
    return crop_and_resize(img)


def evaluate_keras_model(model_path: Path, manifest_path: Path) -> dict:
    """Evaluate a Keras model on the hard cases manifest."""
    print(f"\n{'='*60}")
    print(f"  Evaluating Keras model: {model_path.name}")
    print(f"{'='*60}")
    
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )
    
    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            rows.append((p, float(row["value"])))
    
    print(f"Manifest: {manifest_path.name} ({len(rows)} images)\n")
    
    errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            print(f"  SKIP: {path.name} (missing/empty)")
            continue
        
        inp = load_and_prepare_input(path)
        pred_out = model.predict(inp[None], verbose=0)
        
        # Handle multi-output (dict) vs single-output (array)
        if isinstance(pred_out, dict):
            pred = float(pred_out["gauge_value"][0][0])
        else:
            pred = float(np.asarray(pred_out).flatten()[0])
        
        err = pred - true_val
        errors.append((true_val, pred, err))
        print(f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {path.name}")
    
    return _compute_metrics(errors)


def evaluate_tflite_model(model_path: Path, manifest_path: Path) -> dict:
    """Evaluate a TFLite model on the hard cases manifest."""
    print(f"\n{'='*60}")
    print(f"  Evaluating TFLite model: {model_path.name}")
    print(f"{'='*60}")
    
    interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    print(f"Input: {inp['shape']} quant={inp['quantization']}")
    print(f"Output: {out['shape']} quant={out['quantization']}\n")
    
    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            rows.append((p, float(row["value"])))
    
    print(f"Manifest: {manifest_path.name} ({len(rows)} images)\n")
    
    errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            print(f"  SKIP: {path.name} (missing/empty)")
            continue
        
        img = load_rgb_image(path)
        inp_img = crop_and_resize(img)
        q = np.clip(np.round(inp_img / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        raw = interp.get_tensor(out["index"]).flatten()[0]
        pred = (float(raw.astype(np.float32)) - out_zp) * out_scale
        
        err = pred - true_val
        errors.append((true_val, pred, err))
        print(f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {path.name}")
    
    return _compute_metrics(errors)


def _compute_metrics(errors: list[tuple[float, float, float]]) -> dict:
    """Compute aggregate metrics from per-sample errors."""
    if not errors:
        return {"mae": -1.0, "max_error": -1.0, "count": 0, "bias": 0.0, "std": 0.0}
    
    arr = np.array([e[2] for e in errors])
    preds = np.array([e[1] for e in errors])
    targets = np.array([e[0] for e in errors])
    
    mae = float(np.abs(arr).mean())
    bias = float(arr.mean())
    std = float(arr.std())
    max_error = float(np.abs(arr).max())
    
    # Per-band analysis
    cold_mask = targets < 0.0
    low_mask = (targets >= 0.0) & (targets < 20.0)
    mid_mask = (targets >= 20.0) & (targets < 35.0)
    hot_mask = targets >= 35.0
    
    cold_mae = float(np.abs(arr[cold_mask]).mean()) if np.any(cold_mask) else -1.0
    low_mae = float(np.abs(arr[low_mask]).mean()) if np.any(low_mask) else -1.0
    mid_mae = float(np.abs(arr[mid_mask]).mean()) if np.any(mid_mask) else -1.0
    hot_mae = float(np.abs(arr[hot_mask]).mean()) if np.any(hot_mask) else -1.0
    
    over_5 = sum(1 for e in errors if abs(e[2]) > 5.0)
    over_10 = sum(1 for e in errors if abs(e[2]) > 10.0)
    
    print(f"\nn={len(arr)}  MAE={mae:.2f}°C  bias={bias:+.2f}°C  std={std:.2f}°C")
    print(f"Cases > 5°C: {over_5}/{len(errors)}  > 10°C: {over_10}/{len(errors)}")
    print(f"Cold MAE: {cold_mae:.2f}°C  Low MAE: {low_mae:.2f}°C  Mid MAE: {mid_mae:.2f}°C  Hot MAE: {hot_mae:.2f}°C")
    
    # Sort and show worst
    errors.sort(key=lambda x: -abs(x[2]))
    print("\nWorst 10 failures:")
    for e in errors[:10]:
        print(f"  err={e[2]:+.2f}  true={e[0]:6.1f}  pred={e[1]:7.2f}")
    
    return {
        "mae": mae,
        "bias": bias,
        "std": std,
        "max_error": max_error,
        "count": len(errors),
        "over_5": over_5,
        "over_10": over_10,
        "cold_mae": cold_mae,
        "low_mae": low_mae,
        "mid_mae": mid_mae,
        "hot_mae": hot_mae,
    }


def main():
    if not HAS_TF:
        print("ERROR: TensorFlow not available on Windows. Need WSL for ML evaluation.")
        sys.exit(1)
    
    print("=" * 60)
    print("  Enhanced Model vs Prod v0.3 Hard-Case Evaluation")
    print("=" * 60)
    
    manifest = MANIFEST
    if not manifest.exists():
        print(f"ERROR: Manifest not found: {manifest}")
        sys.exit(1)
    
    results = {}
    
    # Evaluate enhanced model if it exists
    enhanced_path = MODELS["enhanced_multi_scale"]
    if enhanced_path.exists():
        results["enhanced_multi_scale"] = evaluate_keras_model(enhanced_path, manifest)
    else:
        print(f"\nEnhanced model not found: {enhanced_path}")
    
    # Evaluate prod v0.3 scalar TFLite
    scalar_path = MODELS["prod_v0.3_scalar"]
    if scalar_path.exists():
        results["prod_v0.3_scalar"] = evaluate_tflite_model(scalar_path, manifest)
    else:
        print(f"\nProd v0.3 scalar model not found: {scalar_path}")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("  COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'MAE':>8} {'MaxErr':>8} {'>5°C':>6}")
        print("-" * 60)
        for name, r in results.items():
            print(f"{name:<30} {r['mae']:>7.2f} {r['max_error']:>7.2f} {r['over_5']:>5d}/{r['count']}")
        print("-" * 60)
        
        # Check if enhanced beats prod v0.3
        if "enhanced_multi_scale" in results and "prod_v0.3_scalar" in results:
            enhanced_mae = results["enhanced_multi_scale"]["mae"]
            prod_mae = results["prod_v0.3_scalar"]["mae"]
            if enhanced_mae < prod_mae:
                improvement = prod_mae - enhanced_mae
                pct = 100 * improvement / prod_mae
                print(f"\n✓ ENHANCED MODEL BEATS PROD v0.3 by {improvement:.2f}°C ({pct:.1f}% reduction in MAE)")
            else:
                print(f"\n✗ Enhanced model does NOT beat prod v0.3")


if __name__ == "__main__":
    main()
