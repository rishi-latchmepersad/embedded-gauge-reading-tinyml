#!/usr/bin/env python3
"""Evaluate polar_vote_v28 model with luma crop detection.

This script tests whether the polar_vote_circular_v28 model can work
with the luma-based Cartesian crop detection instead of the firmware's
polar projection crop pipeline.

Pipeline:
1. Load image
2. Luma bright-centroid crop detection (same as read_gauge_cnn.py)
3. Build 7-channel polar tensor from crop
4. Run polar_vote_circular_v28 model
5. Decode circular logits to temperature
6. Evaluate on board captures
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    celsius_from_inner_dial_angle_degrees,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    build_firmware_polar_vote_tensor,
    decode_circular_vote_logits,
    firmware_training_crop_box,
    POLAR_VOTE_BINS,
    POLAR_VOTE_MIN_VALUE_C,
    POLAR_VOTE_MAX_VALUE_C,
)

# Import luma crop detector
from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
    CropBox,
)

# Board capture ground truth
BOARD_CAPTURES_GT = {
    "capture_0007.png": 18.0,
    "capture_0008.png": 22.0,
    "capture_0073.png": 46.0,
    "capture_0075.png": 19.0,
    "capture_0c_preview.png": 0.0,
    "capture_2026-04-03_08-20-49.png": 45.0,
    "capture_2026-04-03_13-48-34.png": 30.0,
    "capture_2026-04-03_15-46-04.png": 19.0,
    "capture_2026-04-22_07-15-36.png": 28.0,
    "capture_2026-04-22_07-16-50.png": 28.0,
    "capture_2026-04-22_07-18-03.png": 28.0,
    "capture_2026-04-22_07-19-17.png": 28.0,
    "capture_2026-04-22_07-20-31.png": 28.0,
    "capture_2026-04-22_07-21-44.png": 28.0,
    "capture_2026-04-22_07-22-58.png": 28.0,
    "capture_2026-04-22_07-24-43.png": 28.0,
    "capture_2026-04-22_07-25-57.png": 28.0,
    "capture_2026-04-22_07-27-11.png": 28.0,
    "capture_2026-04-22_07-28-25.png": 28.0,
    "capture_2026-04-22_07-29-39.png": 28.0,
    "capture_2026-04-22_07-30-53.png": 28.0,
    "capture_2026-04-22_07-32-07.png": 28.0,
    "capture_2026-04-22_07-33-52.png": 28.0,
    "capture_2026-04-22_07-35-06.png": 28.0,
    "capture_2026-04-22_07-36-19.png": 28.0,
    "capture_2026-04-22_07-37-33.png": 28.0,
    "capture_2026-04-22_07-38-47.png": 28.0,
    "capture_2026-04-22_07-40-01.png": 28.0,
    "capture_2026-04-24_22-24-04.png": 0.0,
    "capture_p31c_preview.png": 31.0,
    "capture_p35c_preview.png": 35.0,
    "capture_p42c.png": 42.0,
    "capture_p45c.png": 45.0,
    "capture_p50c_preview.png": 50.0,
    "capture_p5c.png": 5.0,
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "deployment" / "polar_vote_circular_v28_int8" / "model_int8.tflite"


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load image as RGB."""
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def luma_crop_detection(image_rgb: np.ndarray) -> CropBox:
    """Run luma-based crop detection."""
    height, width = image_rgb.shape[:2]
    
    # Estimate bright centroid
    centroid = estimate_bright_centroid(image_rgb)
    
    # Compute dynamic crop box
    crop_box = compute_dynamic_crop(
        width=width,
        height=height,
        center_x=centroid.center_x,
        center_y=centroid.center_y,
    )
    
    if crop_box is None:
        raise ValueError("Crop detection failed - crop box is None")
    
    return crop_box


def build_polar_input_from_crop(image_rgb: np.ndarray, crop_box: CropBox) -> np.ndarray:
    """Build 7-channel polar input from luma-detected crop."""
    # Crop and resize to 224x224
    crop = crop_and_resize(
        image_rgb,
        crop_box,
        target_size=224,
    )
    
    # Build firmware-style polar tensor
    polar_tensor = build_firmware_polar_vote_tensor(
        image_rgb=crop,
        gauge_spec=None,  # Use defaults
    )
    
    return polar_tensor


def load_tflite_model(model_path: Path):
    """Load TFLite model."""
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    return interpreter, input_details, output_details


def run_polar_model(interpreter, input_details, output_details, polar_tensor: np.ndarray) -> np.ndarray:
    """Run polar model inference."""
    # Quantize input if needed
    input_dtype = input_details["dtype"]
    if input_dtype == np.int8:
        # Quantize: float [0,1] -> int8
        scale, zero_point = input_details["quantization"]
        polar_int8 = np.round(polar_tensor / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], polar_int8)
    else:
        interpreter.set_tensor(input_details["index"], polar_tensor.astype(input_dtype))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details["index"])
    
    return output


def decode_output(logits: np.ndarray) -> float:
    """Decode polar model output to temperature."""
    # Dequantize if needed
    if logits.dtype == np.int8:
        # Check output_details for quantization params
        pass
    
    # Decode circular vote
    temp = decode_circular_vote_logits(logits)
    
    return temp


def evaluate_on_board_captures(base_dir: Path):
    """Evaluate polar model with luma crop on board captures."""
    print(f"Loading model from {MODEL_PATH}...")
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return None
    
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {input_details['shape']}, Output shape: {output_details['shape']}")
    
    errors = []
    results = []
    
    for img_name, gt_temp in BOARD_CAPTURES_GT.items():
        img_path = base_dir / "captured_images" / img_name
        if not img_path.exists():
            print(f"  Skipping {img_name} (not found)")
            continue
        
        try:
            # Load image
            image_rgb = load_image_rgb(img_path)
            
            # Luma crop detection
            crop_box = luma_crop_detection(image_rgb)
            
            # Build polar input
            polar_tensor = build_polar_input_from_crop(image_rgb, crop_box)
            
            # Run model
            logits = run_polar_model(interpreter, input_details, output_details, polar_tensor)
            
            # Decode to temperature
            pred_temp = decode_output(logits)
            
            # Calculate error
            error = abs(pred_temp - gt_temp)
            errors.append(error)
            results.append((img_name, gt_temp, pred_temp, error))
            
            print(f"  {img_name}: GT={gt_temp:5.1f}°C, Pred={pred_temp:5.1f}°C, Error={error:5.1f}°C")
            
        except Exception as e:
            print(f"  ERROR processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not errors:
        print("No images processed!")
        return None
    
    errors = np.array(errors)
    print(f"\n{'='*60}")
    print(f"Results ({len(errors)} images):")
    print(f"  MAE: {errors.mean():.2f}°C")
    print(f"  Median: {np.median(errors):.2f}°C")
    print(f"  Std: {errors.std():.2f}°C")
    print(f"  Max error: {errors.max():.2f}°C")
    print(f"  Min error: {errors.min():.2f}°C")
    
    # Show worst predictions
    results.sort(key=lambda x: -x[3])
    print("\n  Worst predictions:")
    for name, gt, pred, err in results[:5]:
        print(f"    {name}: GT={gt:5.1f}°C, Pred={pred:5.1f}°C, Error={err:5.1f}°C")
    
    # Show best predictions
    print("\n  Best predictions:")
    results_sorted_best = sorted(results, key=lambda x: x[3])[:5]
    for name, gt, pred, err in results_sorted_best:
        print(f"    {name}: GT={gt:5.1f}°C, Pred={pred:5.1f}°C, Error={err:5.1f}°C")
    
    return {
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "min": float(errors.min()),
        "count": len(errors),
        "results": results,
    }


def main():
    # Use correct path for captured images
    base_dir = Path(__file__).resolve().parent.parent / "data"
    
    print("Evaluating polar_vote_circular_v28 with luma crop detection...")
    print(f"Testing {len(BOARD_CAPTURES_GT)} board captures")
    print(f"{'='*60}")
    
    results = evaluate_on_board_captures(base_dir)
    
    if results:
        # Save results
        import json
        output_path = Path("/tmp/polar_v28_luma_crop_eval.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
