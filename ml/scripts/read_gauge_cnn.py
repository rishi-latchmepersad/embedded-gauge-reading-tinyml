"""End-to-end gauge reading inference with CNN heatmap model.

This script chains the full inference pipeline:
1. Load image (RGB PNG/JPG or YUV422 board capture)
2. Run luma bright-centroid crop detection
3. Resize crop to 224x224
4. Load trained heatmap model and predict center/tip heatmaps
5. Decode heatmaps via soft-argmax to get center and tip coordinates
6. Compute needle angle via atan2(dy, dx)
7. Map angle to temperature via celsius_from_inner_dial_angle_degrees()
8. Output result and save visualization

Usage:
    # Single image inference
    python ml/scripts/read_gauge_cnn.py --input image.png --model model.keras
    
    # With output directory for visualization
    python ml/scripts/read_gauge_cnn.py --input image.png --model model.keras --output-dir tmp/inference/
    
    # Batch inference on directory
    python ml/scripts/read_gauge_cnn.py --input-dir ml/data/captured_images/ --model model.keras --output-dir tmp/batch/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Import project modules
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)

# Import luma crop detector
import sys
sys.path.insert(0, str(Path(__file__).parent))
from luma_crop_detector import (
    estimate_bright_centroid,
    compute_dynamic_crop,
    crop_and_resize,
    CropBox,
)


# ---------------------------------------------------------------------------
# Heatmap Decoding
# ---------------------------------------------------------------------------


def soft_argmax_2d(heatmap: NDArray[np.float32]) -> tuple[float, float]:
    """Compute soft-argmax of a 2D heatmap.

    The soft-argmax is the expected value of the coordinate distribution
    defined by the heatmap (treated as a probability distribution).

    Args:
        heatmap: 2D heatmap array, shape (H, W), values in [0, 1]

    Returns:
        (x, y) coordinates of soft-argmax in pixel space
    """
    # Normalize to sum to 1 (treat as probability distribution)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    total = heatmap.sum()
    if total < 1e-6:
        # Fallback to hard argmax if heatmap is all zeros
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return float(idx[1]), float(idx[0])
    
    heatmap_norm = heatmap / total
    
    # Compute expected coordinates
    h, w = heatmap.shape
    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    
    # Marginal distributions
    p_x = heatmap_norm.sum(axis=0)  # Sum over rows
    p_y = heatmap_norm.sum(axis=1)  # Sum over columns
    
    # Expected values
    x = (x_coords * p_x).sum()
    y = (y_coords * p_y).sum()
    
    return x, y


def decode_heatmaps(
    center_heatmap: NDArray[np.float32],
    tip_heatmap: NDArray[np.float32],
) -> dict[str, Any]:
    """Decode center and tip heatmaps to coordinates and angle.

    Args:
        center_heatmap: Center heatmap, shape (112, 112)
        tip_heatmap: Tip heatmap, shape (112, 112)

    Returns:
        Dict with center_x, center_y, tip_x, tip_y, angle_degrees, temperature_c
    """
    # Soft-argmax decode
    center_x, center_y = soft_argmax_2d(center_heatmap)
    tip_x, tip_y = soft_argmax_2d(tip_heatmap)

    # Compute angle
    angle = angle_degrees_from_center_to_tip(center_x, center_y, tip_x, tip_y)

    # Compute temperature
    temp = celsius_from_inner_dial_angle_degrees(angle)

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "tip_x": float(tip_x),
        "tip_y": float(tip_y),
        "angle_degrees": float(angle),
        "temperature_c": float(temp),
    }


# ---------------------------------------------------------------------------
# Model Inference
# ---------------------------------------------------------------------------


def load_model(model_path: Path):
    """Load a trained Keras model.

    Args:
        model_path: Path to .keras or .h5 model file

    Returns:
        Loaded Keras model
    """
    from tensorflow import keras
    
    return keras.models.load_model(str(model_path))


def run_inference(
    image: NDArray[np.uint8],
    model,
    crop_detector_fn=None,
) -> dict[str, Any]:
    """Run full inference pipeline on one image.

    Args:
        image: RGB image array, shape (H, W, 3)
        model: Loaded Keras model with heatmap outputs
        crop_detector_fn: Optional custom crop detector function

    Returns:
        Dict with all inference results
    """
    import tensorflow as tf

    height, width = image.shape[:2]

    # Step 1: Run crop detection
    if crop_detector_fn is not None:
        centroid_result = crop_detector_fn(image)
    else:
        centroid_result = estimate_bright_centroid(image)

    # Step 2: Compute dynamic crop
    if centroid_result.detected:
        crop_box = compute_dynamic_crop(width, height, centroid_result.center_x, centroid_result.center_y)
        crop_source = "bright_centroid"
    else:
        # Fallback to fixed training crop
        from luma_crop_detector import (
            TRAINING_CROP_X_MIN_RATIO,
            TRAINING_CROP_Y_MIN_RATIO,
            TRAINING_CROP_X_MAX_RATIO,
            TRAINING_CROP_Y_MAX_RATIO,
        )
        x_min = int(width * TRAINING_CROP_X_MIN_RATIO)
        y_min = int(height * TRAINING_CROP_Y_MIN_RATIO)
        crop_w = int(width * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO))
        crop_h = int(height * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO))
        crop_box = CropBox(x_min=x_min, y_min=y_min, width=crop_w, height=crop_h)
        crop_source = "fixed_training_crop"

    # Step 3: Crop and resize
    cropped = crop_and_resize(image, crop_box, target_size=224)

    # Step 4: Run model inference
    input_tensor = tf.convert_to_tensor(cropped, dtype=tf.float32) / 255.0
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

    predictions = model.predict(input_tensor, verbose=0)

    # Model outputs: [center_heatmap, tip_heatmap, confidence]
    center_heatmap = predictions[0][0, :, :, 0]  # Remove batch and channel dims
    tip_heatmap = predictions[1][0, :, :, 0]
    confidence = float(predictions[2][0, 0])

    # Step 5: Decode heatmaps
    decoded = decode_heatmaps(center_heatmap, tip_heatmap)

    # Compile results
    results = {
        "image_size": f"{width}x{height}",
        "crop_source": crop_source,
        "crop_box": {
            "x_min": crop_box.x_min,
            "y_min": crop_box.y_min,
            "width": crop_box.width,
            "height": crop_box.height,
        },
        "centroid": {
            "x": round(centroid_result.center_x, 1),
            "y": round(centroid_result.center_y, 1),
            "detected": centroid_result.detected,
            "bright_count": centroid_result.bright_count,
        },
        "decoded": decoded,
        "confidence": confidence,
        "center_heatmap": center_heatmap,
        "tip_heatmap": tip_heatmap,
        "cropped_image": cropped,
    }

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_result(
    image: NDArray[np.uint8],
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Create visualization of inference result.

    Args:
        image: Original RGB image
        results: Inference results dict
        output_path: Path to save visualization
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image with crop overlay
    axes[0].imshow(image)
    crop = results["crop_box"]
    rect = patches.Rectangle(
        (crop["x_min"], crop["y_min"]),
        crop["width"],
        crop["height"],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label=f"crop ({results['crop_source']})",
    )
    axes[0].add_patch(rect)
    
    # Plot centroid if detected
    if results["centroid"]["detected"]:
        axes[0].plot(
            results["centroid"]["x"],
            results["centroid"]["y"],
            "r+",
            markersize=12,
            markeredgewidth=2,
            label="centroid",
        )
    
    axes[0].set_title(f"Input: {results['image_size']}\nTemp: {results['decoded']['temperature_c']:.1f}°C")
    axes[0].legend(loc="lower right")
    axes[0].axis("off")

    # Panel 2: Cropped image with needle overlay
    cropped = results["cropped_image"]
    axes[1].imshow(cropped)
    
    # Draw needle line
    center_x, center_y = results["decoded"]["center_x"], results["decoded"]["center_y"]
    tip_x, tip_y = results["decoded"]["tip_x"], results["decoded"]["tip_y"]
    axes[1].plot([center_x, tip_x], [center_y, tip_y], "r-", linewidth=2, label="needle")
    axes[1].plot(center_x, center_y, "go", markersize=8, label="center")
    axes[1].plot(tip_x, tip_y, "rx", markersize=10, markeredgewidth=2, label="tip")
    
    axes[1].set_title(f"Angle: {results['decoded']['angle_degrees']:.1f}°\nConf: {results['confidence']:.3f}")
    axes[1].legend(loc="lower right")
    axes[1].axis("off")

    # Panel 3: Heatmaps
    center_hm = results["center_heatmap"]
    tip_hm = results["tip_heatmap"]
    
    im0 = axes[2].imshow(center_hm, cmap="Reds")
    im1 = axes[2].imshow(tip_hm, cmap="Blues", alpha=0.5)
    axes[2].plot(results["decoded"]["center_x"], results["decoded"]["center_y"], "g+", markersize=12, markeredgewidth=2)
    axes[2].plot(results["decoded"]["tip_x"], results["decoded"]["tip_y"], "rx", markersize=12, markeredgewidth=2)
    axes[2].set_title("Heatmaps (Red=center, Blue=tip)")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="End-to-end gauge reading inference")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path (PNG/JPG)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory for batch inference",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained Keras model (.keras or .h5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.input is None and args.input_dir is None:
        parser.error("Must specify --input or --input-dir")
    if args.input and not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    if args.input_dir and not args.input_dir.exists():
        parser.error(f"Input directory not found: {args.input_dir}")
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    # Get input images
    if args.input:
        image_paths = [args.input]
    else:
        image_paths = sorted(
            list(args.input_dir.glob("*.png")) +
            list(args.input_dir.glob("*.jpg")) +
            list(args.input_dir.glob("*.jpeg"))
        )

    print(f"Processing {len(image_paths)} images...")

    # Process each image
    for image_path in image_paths:
        print(f"\n{image_path}:")

        # Load image
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # Run inference
        results = run_inference(image, model)

        # Output results
        temp = results["decoded"]["temperature_c"]
        angle = results["decoded"]["angle_degrees"]
        conf = results["confidence"]

        if args.format == "text":
            print(f"  Temperature: {temp:.2f}°C")
            print(f"  Angle: {angle:.2f}°")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Crop: {results['crop_source']}")
        else:
            # JSON output
            output = {
                "image_path": str(image_path),
                "temperature_c": round(temp, 2),
                "angle_degrees": round(angle, 2),
                "confidence": round(conf, 3),
                "crop_source": results["crop_source"],
            }
            print(json.dumps(output, indent=2))

        # Save outputs
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            vis_path = args.output_dir / f"{image_path.stem}_result.png"
            visualize_result(image, results, vis_path)
            
            # Save JSON report (without large arrays)
            report = {
                "image_path": str(image_path),
                "image_size": results["image_size"],
                "crop_source": results["crop_source"],
                "crop_box": results["crop_box"],
                "centroid": results["centroid"],
                "decoded": results["decoded"],
                "confidence": results["confidence"],
            }
            report_path = args.output_dir / f"{image_path.stem}_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDone.")


if __name__ == "__main__":
    main()
