"""Evaluate CNN baseline with slice metrics.

This script evaluates a trained model and reports:
- Overall MAE / RMSE
- MAE by source_tag
- MAE by value bins
- MAE on hard-case subset
- Error table for top N worst samples

Usage:
    python evaluate_canonical_baseline.py --model-path ml/artifacts/canonical_baseline/model.keras
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import keras
import tensorflow as tf

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_image(
    image_path: str,
    crop_box_ratios: tuple[float, float, float, float],
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load and preprocess an image for inference.

    Matches the training preprocessing pipeline:
    1. Load image
    2. Crop to dial ROI (using relative ratios)
    3. Resize with padding to preserve aspect ratio
    4. Normalize to [0, 1]

    Args:
        image_path: Path to the image file.
        crop_box_ratios: Crop box as relative ratios (x_min, y_min, x_max, y_max).
        target_size: Target (height, width) for resizing.

    Returns:
        Preprocessed image as numpy array.
    """
    # Load image using PIL (simpler than TensorFlow for single image)
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Convert relative ratios to absolute pixel coordinates
    x_min = int(crop_box_ratios[0] * img_w)
    y_min = int(crop_box_ratios[1] * img_h)
    x_max = int(crop_box_ratios[2] * img_w)
    y_max = int(crop_box_ratios[3] * img_h)

    # Ensure valid crop box
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(x_min + 1, min(x_max, img_w))
    y_max = max(y_min + 1, min(y_max, img_h))

    # Crop
    cropped = img.crop((x_min, y_min, x_max, y_max))

    # Resize with padding using TensorFlow (to match training)
    img_array = np.array(cropped)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
    resized = tf.image.resize_with_pad(
        tf.cast(img_tensor, tf.float32),
        target_size[0],
        target_size[1],
        method="bilinear",
    )

    # Normalize to [0, 1]
    return (resized / 255.0).numpy()


def evaluate_model(
    model: keras.Model,
    df: pd.DataFrame,
    repo_root: Path,
    image_size: tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Evaluate model on a dataset and return predictions.

    Args:
        model: Trained Keras model.
        df: DataFrame with image paths and true values.
        repo_root: Repository root for resolving paths.
        image_size: Input image size.

    Returns:
        DataFrame with predictions and errors added.
    """
    results = []

    for idx, row in df.iterrows():
        # Resolve image path
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = repo_root / img_path

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Get crop box from row or use default (full image)
        if "crop_x_min" in row and pd.notna(row["crop_x_min"]):
            crop_box = (
                float(row["crop_x_min"]),
                float(row["crop_y_min"]),
                float(row["crop_x_max"]),
                float(row["crop_y_max"]),
            )
        else:
            # Default crop box (use training crop ratios)
            # These match the training pipeline defaults
            crop_box = (0.1027, 0.2573, 0.7987, 0.8071)

        # Preprocess and predict
        try:
            img = preprocess_image(str(img_path), crop_box, image_size)
            img_batch = np.expand_dims(img, axis=0)
            pred_output = model.predict(img_batch, verbose=0)
            # Handle both scalar and array outputs
            if np.isscalar(pred_output) or pred_output.shape == ():
                prediction = float(pred_output)
            else:
                prediction = float(pred_output.flatten()[0])
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue

        true_value = float(row["value"])
        abs_error = abs(prediction - true_value)

        result = {
            "image_path": str(row["image_path"]),
            "true_value": true_value,
            "predicted_value": prediction,
            "abs_error": abs_error,
            "squared_error": abs_error**2,
        }

        # Add metadata columns if available
        for col in ["source_tag", "hardness_tag"]:
            if col in row:
                result[col] = row[col]

        results.append(result)

    return pd.DataFrame(results)


def compute_slice_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics sliced by different dimensions.

    Args:
        results_df: DataFrame with predictions and metadata.

    Returns:
        Dictionary with slice metrics.
    """
    metrics = {}

    # Overall metrics
    metrics["overall"] = {
        "mae": float(results_df["abs_error"].mean()),
        "rmse": float(np.sqrt(results_df["squared_error"].mean())),
        "count": int(len(results_df)),
    }

    # By source_tag
    if "source_tag" in results_df.columns:
        metrics["by_source"] = {}
        for source in results_df["source_tag"].unique():
            source_df = results_df[results_df["source_tag"] == source]
            metrics["by_source"][source] = {
                "mae": float(source_df["abs_error"].mean()),
                "rmse": float(np.sqrt(source_df["squared_error"].mean())),
                "count": int(len(source_df)),
            }

    # By hardness_tag
    if "hardness_tag" in results_df.columns:
        metrics["by_hardness"] = {}
        for hardness in results_df["hardness_tag"].unique():
            hardness_df = results_df[results_df["hardness_tag"] == hardness]
            metrics["by_hardness"][hardness] = {
                "mae": float(hardness_df["abs_error"].mean()),
                "rmse": float(np.sqrt(hardness_df["squared_error"].mean())),
                "count": int(len(hardness_df)),
            }

    # By value bins (5C bins)
    results_df["value_bin"] = pd.cut(
        results_df["true_value"],
        bins=np.arange(-35, 55, 5),
        include_lowest=True,
    )
    metrics["by_value_bin"] = {}
    for bin_val in results_df["value_bin"].unique():
        if pd.isna(bin_val):
            continue
        bin_df = results_df[results_df["value_bin"] == bin_val]
        metrics["by_value_bin"][str(bin_val)] = {
            "mae": float(bin_df["abs_error"].mean()),
            "rmse": float(np.sqrt(bin_df["squared_error"].mean())),
            "count": int(len(bin_df)),
        }

    return metrics


def get_worst_samples(results_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Get the N samples with highest absolute error.

    Args:
        results_df: DataFrame with predictions and errors.
        n: Number of worst samples to return.

    Returns:
        DataFrame with top N worst samples.
    """
    return results_df.nlargest(n, "abs_error")[
        ["image_path", "true_value", "predicted_value", "abs_error"]
    ]


def print_metrics_report(metrics: dict[str, Any]) -> None:
    """Print a formatted metrics report.

    Args:
        metrics: Dictionary with computed metrics.
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS REPORT")
    print("=" * 60)

    # Overall
    print("\n--- Overall Metrics ---")
    overall = metrics["overall"]
    print(f"  MAE:  {overall['mae']:.4f}°C")
    print(f"  RMSE: {overall['rmse']:.4f}°C")
    print(f"  Count: {overall['count']}")

    # By source
    if "by_source" in metrics:
        print("\n--- MAE by Source Tag ---")
        for source, source_metrics in sorted(metrics["by_source"].items()):
            print(
                f"  {source:20s}: MAE={source_metrics['mae']:.4f}°C, Count={source_metrics['count']}"
            )

    # By hardness
    if "by_hardness" in metrics:
        print("\n--- MAE by Hardness Tag ---")
        for hardness, hardness_metrics in sorted(metrics["by_hardness"].items()):
            print(
                f"  {hardness:20s}: MAE={hardness_metrics['mae']:.4f}°C, Count={hardness_metrics['count']}"
            )

    # By value bin
    if "by_value_bin" in metrics:
        print("\n--- MAE by Value Bin (5°C) ---")
        for bin_str, bin_metrics in sorted(metrics["by_value_bin"].items()):
            print(
                f"  {bin_str:20s}: MAE={bin_metrics['mae']:.4f}°C, Count={bin_metrics['count']}"
            )

    print("=" * 60)


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate CNN baseline with slice metrics."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Path to test split CSV (default: ml/data/splits/canonical_split_v1_test.csv)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: same as model directory)",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=30,
        help="Number of worst samples to report (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-detect paths
    if args.repo_root is None:
        script_dir = Path(__file__).resolve().parent
        args.repo_root = script_dir.parent.parent
    else:
        args.repo_root = args.repo_root.resolve()

    if args.test_csv is None:
        args.test_csv = (
            args.repo_root / "ml" / "data" / "splits" / "canonical_split_v1_test.csv"
        )
    if args.output_dir is None:
        args.output_dir = args.model_path.parent

    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test CSV: {args.test_csv}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load model
    logger.info("Loading model...")
    try:
        # Load without compiling first to avoid custom loss issues
        model = keras.models.load_model(args.model_path, compile=False)
        # Recompile with standard loss for evaluation
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Load test data
    logger.info("Loading test data...")
    try:
        test_df = pd.read_csv(args.test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return 1

    # Evaluate
    logger.info("Running evaluation...")
    results_df = evaluate_model(model, test_df, args.repo_root)
    logger.info(f"Evaluated {len(results_df)} samples")

    # Compute metrics
    logger.info("Computing slice metrics...")
    metrics = compute_slice_metrics(results_df)

    # Print report
    print_metrics_report(metrics)

    # Get worst samples
    worst_samples = get_worst_samples(results_df, args.worst_n)
    print(f"\n--- Top {len(worst_samples)} Worst Samples ---")
    print(worst_samples.to_string(index=False))

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = args.output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save detailed results
    results_path = args.output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved detailed results to {results_path}")

    # Save worst samples
    worst_path = args.output_dir / "worst_samples.csv"
    worst_samples.to_csv(worst_path, index=False)
    logger.info(f"Saved worst samples to {worst_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
