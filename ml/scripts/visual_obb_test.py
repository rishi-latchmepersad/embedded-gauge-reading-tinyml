#!/usr/bin/env python3
"""
Visual OBB test: run the OBB model on phone pictures and draw the bounding box.
This is an iterative tool for verifying the OBB model works correctly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from embedded_gauge_reading_tinyml.board_pipeline import (
    decode_obb_crop_box,
    load_model_session,
    _run_session,
    _predict_full_frame_batch,
)

DEFAULT_OBB_MODEL: Path = (
    PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
)
DEFAULT_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "obb_visual_test"
INPUT_SIZE: int = 224


def run_obb_on_image(
    image_path: Path,
    obb_session,
    *,
    input_size: int = INPUT_SIZE,
    obb_crop_scale: float = 1.2,
) -> dict:
    """Run OBB on a phone picture and return the results."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    # Resize to model input size for inference
    img_resized = img.resize((input_size, input_size), Image.BILINEAR)
    img_array = np.asarray(img_resized, dtype=np.uint8)
    
    # Run OBB inference
    full_batch = (img_array.astype(np.float32) / 255.0)[None, ...]
    obb_output, _ = _run_session(
        obb_session, full_batch,
        preferred_output_keys=("obb_params",),
    )
    obb_params = np.asarray(obb_output, dtype=np.float32).reshape(-1)
    
    # Decode crop box
    dec = decode_obb_crop_box(
        obb_params,
        source_width=orig_w,
        source_height=orig_h,
        input_size=input_size,
        obb_crop_scale=obb_crop_scale,
    )
    
    return {
        "image_path": image_path,
        "orig_size": (orig_w, orig_h),
        "obb_params": obb_params,
        "crop_box": dec.crop_box_xyxy,
        "accepted": dec.accepted,
        "details": dec.details,
    }


def draw_obb_on_image(
    image_path: Path,
    obb_result: dict,
    output_path: Path,
) -> None:
    """Draw the OBB bounding box on the image and save."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    draw = ImageDraw.Draw(img)
    
    # Crop box is in 224x224 space — scale to original image coordinates
    x1, y1, x2, y2 = obb_result["crop_box"]
    input_size = 224
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y
    
    # Draw rectangle (green for accepted, red for rejected)
    color = "green" if obb_result["accepted"] else "red"
    draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline=color, width=5)
    
    # Add text label
    label = f"OBB: {'ACCEPTED' if obb_result['accepted'] else 'REJECTED'}"
    draw.text((10, 10), label, fill=color)
    
    # Add OBB params info
    params = obb_result["obb_params"]
    info = f"cx={params[0]:.3f} cy={params[1]:.3f}"
    draw.text((10, 30), info, fill="white")
    info2 = f"w={params[2]:.3f} h={params[3]:.3f}"
    draw.text((10, 50), info2, fill="white")
    info3 = f"Image: {orig_w}x{orig_h}"
    draw.text((10, 70), info3, fill="white")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual OBB test on phone pictures")
    parser.add_argument("--obb-model", type=Path, default=DEFAULT_OBB_MODEL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--obb-crop-scale", type=float, default=1.2)
    args = parser.parse_args()
    
    print("=" * 60)
    print("VISUAL OBB TEST — Phone Pictures")
    print("=" * 60)
    
    # Load OBB model
    if not args.obb_model.exists():
        print(f"ERROR: OBB model not found: {args.obb_model}")
        sys.exit(1)
    obb_session = load_model_session(str(args.obb_model), "tflite")
    print(f"  OBB model: {args.obb_model}")
    
    # Find phone pictures
    pxl_files = sorted(args.raw_dir.glob("PXL_*.jpg"))
    if not pxl_files:
        print(f"ERROR: No PXL_*.jpg files found in {args.raw_dir}")
        sys.exit(1)
    pxl_files = pxl_files[:args.max_images]
    print(f"  Testing {len(pxl_files)} phone pictures")
    
    # Process each image
    for idx, img_path in enumerate(pxl_files, 1):
        print(f"\n  [{idx}/{len(pxl_files)}] {img_path.name}")
        
        # Run OBB
        result = run_obb_on_image(
            img_path, obb_session, obb_crop_scale=args.obb_crop_scale,
        )
        
        # Print results
        print(f"    Accepted: {result['accepted']}")
        print(f"    Crop box: {result['crop_box']}")
        print(f"    OBB params: {result['obb_params']}")
        
        # Draw and save
        output_path = args.output_dir / f"obb_{img_path.stem}.jpg"
        draw_obb_on_image(img_path, result, output_path)
    
    print(f"\n  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
