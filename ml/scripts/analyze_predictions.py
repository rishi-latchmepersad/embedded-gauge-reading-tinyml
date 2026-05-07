#!/usr/bin/env python3
"""Analyze model predictions to understand hard case failures.

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/analyze_predictions.py \
        --predictions artifacts/training/all_data_baseline/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_predictions(predictions_path: Path) -> None:
    """Analyze prediction errors and identify patterns."""
    df = pd.read_csv(predictions_path)

    logger.info(f"Analyzing {len(df)} predictions")
    logger.info(f"Value range: {df['value'].min():.1f} to {df['value'].max():.1f}")

    # Overall metrics
    errors = df["abs_error"].values
    logger.info(f"\nOverall:")
    logger.info(f"  MAE: {np.mean(errors):.2f}°C")
    logger.info(f"  Median error: {np.median(errors):.2f}°C")
    logger.info(f"  Max error: {np.max(errors):.2f}°C")
    logger.info(f"  % under 5°C: {np.mean(errors < 5)*100:.1f}%")

    # By temperature range
    ranges = [
        ("Extreme cold (<=-25°C)", df["value"] <= -25),
        ("Cold (-25 to -15°C)", (df["value"] > -25) & (df["value"] <= -15)),
        ("Cool (-15 to 0°C)", (df["value"] > -15) & (df["value"] <= 0)),
        ("Mid (0 to 30°C)", (df["value"] > 0) & (df["value"] <= 30)),
        ("Hot (30 to 40°C)", (df["value"] > 30) & (df["value"] <= 40)),
        ("Extreme hot (>=40°C)", df["value"] >= 40),
    ]

    logger.info(f"\nBy temperature range:")
    for name, mask in ranges:
        if mask.any():
            subset = df[mask]
            errs = subset["abs_error"].values
            logger.info(f"  {name}: n={len(subset)}, MAE={np.mean(errs):.2f}°C, "
                       f"median={np.median(errs):.2f}°C, max={np.max(errs):.2f}°C, "
                       f"% under 5°C={np.mean(errs < 5)*100:.1f}%")
            # Show worst predictions
            worst = subset.nlargest(3, "abs_error")
            for _, row in worst.iterrows():
                logger.info(f"    {Path(row['image_path']).name}: "
                           f"true={row['value']:.1f}°C, pred={row['prediction']:.1f}°C, "
                           f"error={row['abs_error']:.1f}°C")

    # Check for systematic bias
    logger.info(f"\nSystematic bias analysis:")
    df["bias"] = df["prediction"] - df["value"]
    for name, mask in ranges:
        if mask.any():
            bias = df[mask]["bias"].values
            logger.info(f"  {name}: mean bias={np.mean(bias):.2f}°C "
                       f"(positive = overpredicting)")

    # Check if model is collapsing to mid-range
    logger.info(f"\nPrediction distribution:")
    logger.info(f"  Predicted range: {df['prediction'].min():.1f} to {df['prediction'].max():.1f}°C")
    logger.info(f"  Predicted mean: {df['prediction'].mean():.1f}°C")
    logger.info(f"  Predicted std: {df['prediction'].std():.1f}°C")
    logger.info(f"  True mean: {df['value'].mean():.1f}°C")
    logger.info(f"  True std: {df['value'].std():.1f}°C")

    # Check correlation
    corr = np.corrcoef(df["value"], df["prediction"])[0, 1]
    logger.info(f"  Correlation: {corr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    analyze_predictions(args.predictions)


if __name__ == "__main__":
    main()
