#!/usr/bin/env python3
"""Debug script to check manifest format."""

import csv
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"

manifest_path = DATA_DIR / "combined_training_manifest.csv"

print(f"Checking manifest: {manifest_path}")
print(f"File exists: {manifest_path.exists()}")
if manifest_path.exists():
    print(f"File size: {manifest_path.stat().st_size} bytes\n")

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(f"Field names: {reader.fieldnames}")

        # Read first 5 rows
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"Row {i}: {row}")
            if "image_path" not in row:
                print(f"  ERROR: 'image_path' not in row keys: {list(row.keys())}")
else:
    # Try alternative paths
    print("\nTrying alternative paths:")
    for alt_path in [
        PROJECT_ROOT / "ml/data/combined_training_manifest.csv",
        PROJECT_ROOT / "data/combined_training_manifest.csv",
        PROJECT_ROOT / "combined_training_manifest.csv",
    ]:
        if alt_path.exists():
            print(f"  Found at: {alt_path}")
            with open(alt_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                print(f"  Field names: {reader.fieldnames}")
                for i, row in enumerate(reader):
                    if i >= 5:
                        break
                    print(f"  Row {i}: {row}")
                    break
            break

print("\nDone!")
