"""Debug script to check data shapes before training."""
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
from embedded_gauge_reading_tinyml.polar_projection import polar_project_image_path

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
print(f"Columns: {list(df.columns)}")

# Load first 5 samples
polar_images = []
values = []
masks = []

for idx, row in df.head(5).iterrows():
    image_path = resolve_full_path(row["image_path"], REPO_ROOT)
    mask_path = resolve_full_path(row["mask_path"], REPO_ROOT)
    
    print(f"\nSample {idx}:")
    print(f"  image_path: {image_path}")
    print(f"  mask_path: {mask_path}")
    print(f"  value: {row['value']}")
    
    # Load polar image
    polar_img = polar_project_image_path(image_path, polar_size=224)
    print(f"  polar_img shape: {polar_img.shape}, dtype: {polar_img.dtype}")
    polar_images.append(polar_img)
    values.append(float(row["value"]))
    
    # Load mask
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"  mask raw shape: {mask.shape}, dtype: {mask.dtype}")
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.resize(mask, (224, 224))
    print(f"  mask resized shape: {mask.shape}")
    mask = mask[..., np.newaxis]
    print(f"  mask with channel shape: {mask.shape}")
    masks.append(mask)

# Convert to arrays
polar_images_arr = np.array(polar_images, dtype=np.float32)
values_arr = np.array(values, dtype=np.float32)
masks_arr = np.array(masks, dtype=np.float32)

print(f"\nFinal array shapes:")
print(f"  polar_images: {polar_images_arr.shape}")
print(f"  values: {values_arr.shape}")
print(f"  masks: {masks_arr.shape}")

# Create dataset and check batch
dataset = tf.data.Dataset.from_tensor_slices((polar_images_arr, values_arr, masks_arr))
dataset = dataset.map(lambda x, y, m: (x, {"gauge_value": y, "needle_mask": m}))
dataset = dataset.batch(2)

for batch_x, batch_y in dataset.take(1):
    print(f"\nBatch target keys: {list(batch_y.keys())}")
    for k, v in batch_y.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\nSUCCESS")
