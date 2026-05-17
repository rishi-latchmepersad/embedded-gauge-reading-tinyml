"""Post-hoc calibration for the current best scalar model."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import (
    SpatialSoftArgmax2D,
    GaugeValueFromKeypoints,
)

def load_model(path: Path):
    custom = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    }
    return tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)

def load_manifest(path: Path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_path": row["image_path"],
                "value": float(row["value"]),
            })
    return rows

def resolve_image_path(rel_path: str, ml_root: Path) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    rel = rel_path
    if rel.startswith("ml/") or rel.startswith("ml\\"):
        rel = rel[3:]
    candidates = [
        ml_root / rel,
        ml_root / "data" / rel,
        ml_root / "data" / "captured_images" / Path(rel).name,
        ml_root / "data" / "raw" / Path(rel).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return ml_root / rel

def preprocess_image(path: Path, target_size: int = 224):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def main():
    model_path = Path("artifacts/training/no_cal_hardpush_gpu5_recover/model.keras")
    manifest_path = Path("data/full_scalar_manifest_v1.csv")
    output_path = Path("tmp/calibration_spline.json")

    print("[CALIB] Loading model...")
    model = load_model(model_path)
    print(f"[CALIB] Model outputs: {[o.name for o in model.outputs]}")

    print("[CALIB] Loading manifest...")
    rows = load_manifest(manifest_path)
    print(f"[CALIB] {len(rows)} samples")

    truths = []
    preds = []
    print("[CALIB] Running inference...")
    for i, row in enumerate(rows):
        img_path = resolve_image_path(row["image_path"], PROJECT_ROOT)
        if not img_path.exists():
            print(f"[CALIB] Missing: {img_path}")
            continue
        img = preprocess_image(img_path)
        pred = float(model.predict(img, verbose=0)[0][0])
        truths.append(row["value"])
        preds.append(pred)
        if i % 50 == 0:
            print(f"[CALIB] {i}/{len(rows)} done")

    truths = np.array(truths)
    preds = np.array(preds)

    mae_before = np.mean(np.abs(truths - preds))
    print(f"[CALIB] Before calibration: MAE = {mae_before:.3f}C")

    # Fit isotonic regression (monotonic, non-parametric)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(preds, truths)
    cal_iso = iso.predict(preds)
    mae_iso = np.mean(np.abs(truths - cal_iso))
    print(f"[CALIB] Isotonic calibration: MAE = {mae_iso:.3f}C")

    # Fit spline (smoother, allows non-monotonic if needed)
    sorted_idx = np.argsort(preds)
    sp = UnivariateSpline(preds[sorted_idx], truths[sorted_idx], s=len(truths)*2)
    cal_sp = sp(preds)
    mae_sp = np.mean(np.abs(truths - cal_sp))
    print(f"[CALIB] Spline calibration: MAE = {mae_sp:.3f}C")

    # Save isotonic as primary (safer, guaranteed monotonic)
    calibration = {
        "type": "isotonic",
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "n_samples": len(truths),
        "mae_before": float(mae_before),
        "mae_after": float(mae_iso),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "iso_x": [float(x) for x in iso.X_min_ + iso.X_thresholds_],
        "iso_y": [float(y) for y in iso.y_thresholds_],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"[CALIB] Saved calibration to {output_path}")

    # Also save a simple lookup table
    lut_path = output_path.with_name("calibration_lut.csv")
    with open(lut_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true", "iso_calibrated"])
        for p, t, c in zip(preds, truths, cal_iso):
            writer.writerow([p, t, c])
    print(f"[CALIB] Saved LUT to {lut_path}")

if __name__ == "__main__":
    main()
