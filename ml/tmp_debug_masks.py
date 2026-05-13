"""Debug script to check mask shapes from manifest."""

import sys
from pathlib import Path

# Add ml/src to path.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import cv2

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

# Load first 5 masks
for idx, row in df.head(5).iterrows():
    mask_path = resolve_full_path(row["mask_path"], REPO_ROOT)

    print(f"\nSample {idx}:")
    print(f"  mask_path: {mask_path}")

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"  mask raw shape: {mask.shape}, dtype: {mask.dtype}")
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.resize(mask, (224, 224))
    print(f"  mask resized shape: {mask.shape}")
    mask = mask[..., np.newaxis]
    print(f"  mask with channel shape: {mask.shape}")

print("\nSUCCESS")
