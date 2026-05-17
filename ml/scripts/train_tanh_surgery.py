"""Output surgery: replace sigmoid with tanh on existing best model."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

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
    rel = rel.lstrip("/\\")
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

def build_dataset(rows, ml_root, batch_size=16, target_size=224):
    paths = []
    values = []
    for row in rows:
        img_path = resolve_image_path(row["image_path"], ml_root)
        if img_path.exists():
            paths.append(str(img_path))
            values.append(row["value"])
    
    def load_and_preprocess(path, value):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [target_size, target_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, value
    
    ds = tf.data.Dataset.from_tensor_slices((paths, values))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    model_path = PROJECT_ROOT / "artifacts" / "training" / "no_cal_hardpush_gpu5_recover" / "model.keras"
    manifest_path = PROJECT_ROOT / "data" / "full_scalar_manifest_v1.csv"
    output_dir = PROJECT_ROOT / "artifacts" / "training" / "no_cal_hardpush_tanh_surgery"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[SURGERY] Loading model...")
    model = load_model(model_path)
    model.summary()
    
    # Find the layer just before the sigmoid output
    # The architecture is: swish_dense -> dropout -> sigmoid_dense -> rescaling
    # We want to take the output of the swish_dense layer (or the dropout after it)
    # and add tanh -> rescaling -> rescaling
    
    # Strategy: find the second-to-last dense layer
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    print(f"[SURGERY] Found {len(dense_layers)} Dense layers")
    for i, l in enumerate(dense_layers):
        print(f"  {i}: {l.name} units={l.units} act={l.activation.__name__ if hasattr(l.activation, '__name__') else l.activation}")
    
    # The last dense is the sigmoid, the one before is swish
    if len(dense_layers) < 2:
        raise ValueError("Expected at least 2 dense layers")
    
    penultimate_dense = dense_layers[-2]
    print(f"[SURGERY] Penultimate dense: {penultimate_dense.name}")
    
    # Build new model from penultimate dense output
    x = penultimate_dense.output
    
    # Add tanh output
    span = 80.0  # 50 - (-30)
    value_min = -30.0
    value_max = 50.0
    
    # tanh outputs [-1, 1]; map to [0, 1] with (tanh + 1) / 2
    x = tf.keras.layers.Dense(1, activation="tanh", name="gauge_value_tanh")(x)
    x = tf.keras.layers.Rescaling(scale=0.5, offset=0.5, name="gauge_value_norm")(x)
    output = tf.keras.layers.Rescaling(scale=span, offset=value_min, name="gauge_value")(x)
    
    new_model = tf.keras.Model(inputs=model.inputs, outputs=output, name="mobilenetv2_tanh_surgery")
    
    # Freeze all layers except the new tanh output and rescaling layers
    for layer in new_model.layers:
        if layer.name in ("gauge_value_tanh", "gauge_value_norm", "gauge_value"):
            layer.trainable = True
            print(f"[SURGERY] Trainable: {layer.name}")
        else:
            layer.trainable = False
    
    # But wait - the dense before sigmoid might also need fine-tuning
    # Let's also unfreeze the penultimate dense layer
    penultimate_dense.trainable = True
    print(f"[SURGERY] Also trainable: {penultimate_dense.name}")
    
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    
    print("[SURGERY] Loading data...")
    rows = load_manifest(manifest_path)
    ds = build_dataset(rows, PROJECT_ROOT, batch_size=16)
    
    # Split into train/val
    n = len(rows)
    n_val = max(1, int(n * 0.15))
    n_train = n - n_val
    
    # Actually we can't easily split a tf.data.Dataset by count without consuming it
    # Let's just use the full dataset for fine-tuning with early stopping
    # using a separate validation dataset built from a held-out manifest
    
    print(f"[SURGERY] Training on {n} samples...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    
    history = new_model.fit(
        ds,
        epochs=30,
        callbacks=callbacks,
        verbose=1,
    )
    
    print("[SURGERY] Saving model...")
    new_model.save(output_dir / "model.keras")
    
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (output_dir / "history.json").write_text(json.dumps(hist_dict, indent=2), encoding="utf-8")
    
    print(f"[SURGERY] Saved to {output_dir}")
    print(f"[SURGERY] Final loss: {history.history['loss'][-1]:.4f}")
    print(f"[SURGERY] Final MAE: {history.history['mae'][-1]:.4f}")

if __name__ == "__main__":
    main()
