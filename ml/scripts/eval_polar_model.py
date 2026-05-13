"""Evaluate the polar needle-segmentation model on a manifest.

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/eval_polar_model.py \
        --model artifacts/training/polar_needle_model_v1/model.keras \
        --manifest data/hard_cases.csv \
        --output-dir artifacts/eval/polar_needle_v1_hard
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import sys

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.polar_projection import polar_project_image_path
from embedded_gauge_reading_tinyml.polar_model import PolarAngleToTemperature


def normalize_path(path_str: str, repo_root: Path) -> str:
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def resolve_full_path(normalized_path: str, repo_root: Path) -> Path:
    return repo_root / normalized_path


def load_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})
    if "image_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "image_path"})
    df["image_path"] = df["image_path"].apply(
        lambda p: normalize_path(str(p), repo_root)
    )
    df["image_path_resolved"] = df["image_path"].apply(
        lambda p: str(resolve_full_path(p, repo_root))
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def evaluate_polar_model(
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    polar_size: int = 224,
) -> dict[str, Any]:
    """Evaluate a trained polar model on a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    custom_objects = {
        "PolarAngleToTemperature": PolarAngleToTemperature,
    }
    model: keras.Model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
    )
    print(f"Loaded model from {model_path}")

    # Load manifest.
    df = load_manifest(manifest_path, REPO_ROOT)
    print(f"Evaluating on {len(df)} samples from {manifest_path}")

    # Precompute polar projections.
    polar_images = []
    for idx, row in df.iterrows():
        try:
            polar_img = polar_project_image_path(
                row["image_path_resolved"], polar_size=polar_size
            )
            polar_images.append(polar_img)
        except Exception as exc:
            print(f"  Skip failed projection: {row['image_path_resolved']} ({exc})")
            polar_images.append(np.zeros((polar_size, polar_size, 3), dtype=np.float32))

    polar_images = np.array(polar_images, dtype=np.float32)

    # Predict.
    predictions = model.predict(polar_images, verbose=1, batch_size=8)
    gauge_preds = predictions["gauge_value"].flatten()

    # Compute metrics.
    df = df.copy()
    df["prediction"] = gauge_preds
    df["abs_error"] = np.abs(df["prediction"] - df["value"])
    df["error"] = df["prediction"] - df["value"]

    errors = df["abs_error"].values
    hard_mask = (df["value"] <= -20) | (df["value"] >= 40)
    hard_errors = errors[hard_mask] if hard_mask.any() else np.array([])

    metrics = {
        "manifest": str(manifest_path),
        "model": str(model_path),
        "samples": int(len(df)),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
        "std": float(np.std(errors)),
        "pct_under_5c": float(np.mean(errors < 5.0) * 100),
        "hard_mae": float(np.mean(hard_errors)) if len(hard_errors) > 0 else None,
        "hard_max": float(np.max(hard_errors)) if len(hard_errors) > 0 else None,
        "predicted_std": float(np.std(gauge_preds)),
        "correlation": (
            float(np.corrcoef(df["value"], gauge_preds)[0, 1])
            if len(gauge_preds) > 1
            else 0.0
        ),
    }

    print("\n=== Evaluation Results ===")
    for key, val in metrics.items():
        if val is not None:
            print(f"  {key}: {val:.4f}")

    # Save results.
    df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate polar needle model.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model.keras.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval" / "polar_model",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--polar-size",
        type=int,
        default=224,
        help="Polar projection size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_polar_model(
        model_path=args.model,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        polar_size=args.polar_size,
    )


if __name__ == "__main__":
    main()
