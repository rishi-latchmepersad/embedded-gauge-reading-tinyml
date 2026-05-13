"""Debug script to reproduce the exact training setup."""

import sys
from pathlib import Path

# Add ml/src to path.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras

from embedded_gauge_reading_tinyml.polar_model import (
    build_polar_needle_segmentation_model,
)

REPO_ROOT = PROJECT_ROOT.parent


def resolve_full_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


# Load manifest
manifest_path = PROJECT_ROOT / "artifacts" / "polar_masks" / "manifest.csv"
df = pd.read_csv(manifest_path)
print(f"Loaded manifest: {len(df)} rows")

# Load first 8 samples
polar_images = []
values = []
masks = []

for idx, row in df.head(8).iterrows():
    image_path = resolve_full_path(row["image_path"], REPO_ROOT)
    mask_path = resolve_full_path(row["mask_path"], REPO_ROOT)

    # Load polar image
    polar_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    polar_img = cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB)
    polar_img = polar_img.astype(np.float32) / 255.0
    polar_images.append(polar_img)
    values.append(float(row["value"]))

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    mask = mask[..., np.newaxis]
    masks.append(mask)

polar_images_arr = np.array(polar_images, dtype=np.float32)
values_arr = np.array(values, dtype=np.float32)
masks_arr = np.array(masks, dtype=np.float32)

print(f"polar_images_arr shape: {polar_images_arr.shape}")
print(f"values_arr shape: {values_arr.shape}")
print(f"masks_arr shape: {masks_arr.shape}")

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((polar_images_arr, values_arr, masks_arr))
dataset = dataset.map(lambda x, y, m: (x, {"gauge_value": y, "needle_mask": m}))
dataset = dataset.batch(8)

# Create model
model = build_polar_needle_segmentation_model(polar_size=224, base_filters=32, depth=4)
print(f"Model output_names: {model.output_names}")
for i, output in enumerate(model.outputs):
    print(f"  Output {i}: name={output.name}, shape={output.shape}")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "gauge_value": keras.losses.MeanSquaredError(),
        "needle_mask": keras.losses.BinaryCrossentropy(),
    },
    loss_weights={
        "gauge_value": 1.0,
        "needle_mask": 1.0,
    },
    metrics={
        "gauge_value": ["mae"],
    },
)

# Try training for 1 step
print("\nTrying model.fit() for 1 epoch...")
model.fit(dataset, epochs=1)
print("SUCCESS")
