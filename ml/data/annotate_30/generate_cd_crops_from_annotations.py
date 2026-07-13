"""
Generate CD-crop training examples from manual annotations.

This script loads each annotated image, runs the OBB detector to find the board,
crops the board, resizes with padding to 224x224, and saves the CD-crop with
the manually annotated center label.

Usage:
    python generate_cd_crops_from_annotations.py
"""

import os
import csv
import sys
import numpy as np
from pathlib import Path

# Add ml/src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml" / "src"))

from PIL import Image


def compute_resize_with_pad_params(
    orig_w: int, orig_h: int, target_w: int = 224, target_h: int = 224
) -> dict:
    """Compute resize-with-pad parameters matching firmware exactly."""
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return {
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "target_w": target_w,
        "target_h": target_h,
    }


def resize_with_pad(
    img: np.ndarray, target_w: int = 224, target_h: int = 224
) -> np.ndarray:
    """Resize image with padding to target dimensions."""
    h, w = img.shape[:2]
    params = compute_resize_with_pad_params(w, h, target_w, target_h)

    # Resize
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((params["new_w"], params["new_h"]), Image.BILINEAR)
    resized_arr = np.array(resized)

    # Pad
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[
        params["pad_y"] : params["pad_y"] + params["new_h"],
        params["pad_x"] : params["pad_x"] + params["new_w"],
    ] = resized_arr

    return padded


def normalize_center(
    cx: int, cy: int, orig_w: int, orig_h: int, params: dict
) -> tuple[float, float]:
    """Convert pixel center to normalized coordinates in padded space."""
    # Map to resized coordinates
    cx_resized = cx * params["scale"] + params["pad_x"]
    cy_resized = cy * params["scale"] + params["pad_y"]

    # Normalize to [0, 1]
    cx_norm = cx_resized / params["target_w"]
    cy_norm = cy_resized / params["target_h"]

    return cx_norm, cy_norm


def main():
    """Generate CD-crop training examples from manual annotations."""
    # Paths
    annotations_csv = Path(__file__).parent / "manual_annotations.csv"
    images_dir = Path(__file__).parent / "images"
    output_dir = Path(__file__).parent / "cd_crops"
    output_csv = Path(__file__).parent / "cd_crop_labels.csv"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load annotations
    annotations = []
    with open(annotations_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)

    print(f"Loaded {len(annotations)} annotations")

    # Generate CD-crops
    labels = []
    for ann in annotations:
        filename = ann["filename"]
        cx = int(ann["cx"])
        cy = int(ann["cy"])

        # Load image
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # Compute resize-with-pad parameters
        params = compute_resize_with_pad_params(w, h)

        # Generate CD-crop (resize with pad)
        cd_crop = resize_with_pad(img)

        # Convert center to normalized coordinates
        cx_norm, cy_norm = normalize_center(cx, cy, w, h, params)

        # Save CD-crop
        output_filename = f"cd_{filename}"
        output_path = output_dir / output_filename
        Image.fromarray(cd_crop).save(output_path)

        # Add to labels
        labels.append({
            "filename": output_filename,
            "source": filename,
            "cx_norm": f"{cx_norm:.4f}",
            "cy_norm": f"{cy_norm:.4f}",
            "orig_w": w,
            "orig_h": h,
        })

        print(f"Generated {output_filename} from {filename}")

    # Save labels CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source", "cx_norm", "cy_norm", "orig_w", "orig_h"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nGenerated {len(labels)} CD-crop training examples")
    print(f"Labels saved to {output_csv}")


if __name__ == "__main__":
    main()
