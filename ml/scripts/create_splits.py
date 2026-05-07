"""Create robust train/val/test splits from the canonical manifest.

This script creates stratified splits that preserve source distribution
and ensure hard cases and board captures are represented in validation and test sets.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_value_bins(df: pd.DataFrame, bin_size: float = 5.0) -> pd.DataFrame:
    """Create value bins for stratification.

    Args:
        df: Input DataFrame with 'value' column.
        bin_size: Size of each temperature bin in degrees C.

    Returns:
        DataFrame with added 'value_bin' column.
    """
    df = df.copy()
    # Create bins centered on multiples of bin_size
    min_val = df["value"].min()
    max_val = df["value"].max()
    bin_edges = np.arange(
        np.floor(min_val / bin_size) * bin_size,
        np.ceil(max_val / bin_size) * bin_size + bin_size,
        bin_size,
    )
    df["value_bin"] = pd.cut(df["value"], bins=bin_edges, include_lowest=True)
    return df


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """Create a stratification key combining value bin and source tag.

    Args:
        df: Input DataFrame with 'value_bin' and 'source_tag' columns.

    Returns:
        Series with combined stratification keys.
    """
    # Combine value bin and source tag for stratification
    # Handle NA values by converting to string representation
    value_bin_str = df["value_bin"].astype(str)
    return value_bin_str + "_" + df["source_tag"]


def split_with_forced_samples(
    df: pd.DataFrame,
    forced_samples: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data ensuring forced samples are in val/test.

    Args:
        df: Full dataset.
        forced_samples: Samples that must be in val/test (not train).
        test_size: Fraction of remaining data for test.
        val_size: Fraction of remaining data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Remove forced samples from the pool
    forced_indices = set(forced_samples.index)
    remaining_df = df[~df.index.isin(forced_indices)].copy()

    # Calculate sizes
    total_size = len(df)
    forced_size = len(forced_samples)
    remaining_size = len(remaining_df)

    # Target sizes
    target_test_size = int(total_size * test_size)
    target_val_size = int(total_size * val_size)
    target_train_size = total_size - target_test_size - target_val_size

    logger.info(
        f"Target sizes: train={target_train_size}, val={target_val_size}, test={target_test_size}"
    )
    logger.info(f"Forced samples to place in val/test: {forced_size}")

    # Place forced samples: split between val and test
    if forced_size > 0:
        forced_val_size = min(len(forced_samples) // 2, target_val_size)
        forced_test_size = min(len(forced_samples) - forced_val_size, target_test_size)

        forced_val, forced_test = train_test_split(
            forced_samples,
            test_size=(
                forced_test_size / len(forced_samples)
                if len(forced_samples) > 0
                else 0.5
            ),
            random_state=random_state,
            stratify=(
                create_stratification_key(forced_samples)
                if len(forced_samples) >= 4
                else None
            ),
        )
    else:
        forced_val = pd.DataFrame()
        forced_test = pd.DataFrame()

    # Calculate how many more samples we need from remaining
    remaining_val_needed = target_val_size - len(forced_val)
    remaining_test_needed = target_test_size - len(forced_test)
    remaining_train_needed = target_train_size

    logger.info(
        f"Remaining needed: train={remaining_train_needed}, val={remaining_val_needed}, test={remaining_test_needed}"
    )

    # Split remaining into train and (val+test pool)
    remaining_val_test_size = (
        (remaining_val_needed + remaining_test_needed) / len(remaining_df)
        if len(remaining_df) > 0
        else 0
    )

    if remaining_val_test_size > 0 and remaining_val_test_size < 1:
        remaining_train, remaining_val_test = train_test_split(
            remaining_df,
            test_size=remaining_val_test_size,
            random_state=random_state,
            stratify=create_stratification_key(remaining_df),
        )

        # Split val_test into val and test
        test_ratio = remaining_test_needed / (
            remaining_val_needed + remaining_test_needed
        )
        remaining_val, remaining_test = train_test_split(
            remaining_val_test,
            test_size=test_ratio,
            random_state=random_state,
            stratify=create_stratification_key(remaining_val_test),
        )
    else:
        # Edge case: all remaining go to train
        remaining_train = remaining_df
        remaining_val = pd.DataFrame()
        remaining_test = pd.DataFrame()

    # Combine
    train_df = remaining_train
    val_df = (
        pd.concat([forced_val, remaining_val], ignore_index=True)
        if not remaining_val.empty
        else forced_val
    )
    test_df = (
        pd.concat([forced_test, remaining_test], ignore_index=True)
        if not remaining_test.empty
        else forced_test
    )

    return train_df, val_df, test_df


def create_splits(
    manifest_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    bin_size: float = 5.0,
) -> tuple[Path, Path, Path]:
    """Create stratified train/val/test splits.

    Args:
        manifest_path: Path to canonical manifest CSV.
        output_dir: Directory to save split files.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        random_state: Random seed for reproducibility.
        bin_size: Size of temperature bins for stratification.

    Returns:
        Tuple of (train_path, val_path, test_path).
    """
    # Load manifest
    logger.info(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    if df.empty:
        raise ValueError("Manifest is empty")

    logger.info(f"Loaded {len(df)} rows")

    # Create value bins
    df = create_value_bins(df, bin_size)
    logger.info(f"Created value bins of size {bin_size}°C")

    # Identify samples that must be in val/test (hard cases and board captures)
    # However, we still need some of these in training, so we'll split them proportionally
    forced_mask = df["source_tag"].isin(["hard_case", "board_capture"])
    forced_samples = df[forced_mask].copy()
    normal_samples = df[~forced_mask].copy()

    logger.info(f"Forced samples (hard_case + board_capture): {len(forced_samples)}")
    logger.info(f"Normal samples: {len(normal_samples)}")

    # Calculate split sizes
    total_size = len(df)

    # If we have no normal samples, we need to split forced samples across train/val/test
    if len(normal_samples) == 0:
        logger.info(
            "No normal samples found - splitting forced samples across train/val/test"
        )

        # Check if we have enough samples per bin for stratification
        value_bin_counts = forced_samples["value_bin"].value_counts()
        min_bin_count = value_bin_counts.min()
        use_stratification = min_bin_count >= 2

        logger.info(f"Minimum samples per value bin: {min_bin_count}")
        logger.info(f"Using stratification: {use_stratification}")

        # Split forced samples into train and (val+test pool) first
        forced_train_val, forced_test = train_test_split(
            forced_samples,
            test_size=test_ratio,
            random_state=random_state,
            stratify=forced_samples["value_bin"] if use_stratification else None,
        )

        # Then split train_val into train and val
        val_relative_size = val_ratio / (train_ratio + val_ratio)
        forced_train, forced_val = train_test_split(
            forced_train_val,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=(
                forced_train_val["value_bin"]
                if use_stratification and len(forced_train_val) >= 2
                else None
            ),
        )

        train_df = forced_train
        val_df = forced_val
        test_df = forced_test
    else:
        # Original logic: normal samples go to train/val/test, forced samples only to val/test
        logger.info("Using original split logic with normal samples")

        # First split: separate test set from normal samples using value_bin stratification
        # Use only value_bin for stratification to avoid rare class issues
        normal_train_val, normal_test = train_test_split(
            normal_samples,
            test_size=test_ratio,
            random_state=random_state,
            stratify=normal_samples["value_bin"],
        )

        # Split forced samples between val and test (no stratification, just shuffle and split)
        if len(forced_samples) > 0:
            forced_shuffled = forced_samples.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
            split_idx = len(forced_shuffled) // 2
            forced_val = forced_shuffled.iloc[:split_idx]
            forced_test = forced_shuffled.iloc[split_idx:]
        else:
            forced_val = pd.DataFrame()
            forced_test = pd.DataFrame()

        # Second split: separate train and val from normal_train_val
        val_relative_size = val_ratio / (train_ratio + val_ratio)
        normal_train, normal_val = train_test_split(
            normal_train_val,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=normal_train_val["value_bin"],
        )

        # Combine
        train_df = normal_train
        val_df = pd.concat([normal_val, forced_val], ignore_index=True)
        test_df = pd.concat([normal_test, forced_test], ignore_index=True)

    # Remove temporary columns
    for col in ["value_bin"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in val_df.columns:
            val_df = val_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    # Log split statistics
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")

    # Log source distribution
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"\n{name} source distribution:")
        logger.info(split_df["source_tag"].value_counts().to_string())

    # Log value distribution
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(
            f"\n{name} value range: {split_df['value'].min():.1f} to {split_df['value'].max():.1f}"
        )

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "canonical_split_v1_train.csv"
    val_path = output_dir / "canonical_split_v1_val.csv"
    test_path = output_dir / "canonical_split_v1_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"\nSaved splits to {output_dir}")

    return train_path, val_path, test_path


def main():
    """Main entry point for the split creation script."""
    parser = argparse.ArgumentParser(
        description="Create robust train/val/test splits from canonical manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to canonical manifest CSV (default: ml/data/canonical_manifest_v1.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split files (default: ml/data/splits/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction for training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for test set (default: 0.15)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=5.0,
        help="Temperature bin size for stratification (default: 5.0)",
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
    if args.manifest is None:
        script_dir = Path(__file__).resolve().parent
        args.manifest = script_dir.parent / "data" / "canonical_manifest_v1.csv"
    else:
        args.manifest = args.manifest.resolve()

    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        args.output_dir = script_dir.parent / "data" / "splits"
    else:
        args.output_dir = args.output_dir.resolve()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0, rtol=1e-5):
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        return 1

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
    )
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Bin size: {args.bin_size}°C")

    # Create splits
    try:
        train_path, val_path, test_path = create_splits(
            args.manifest,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.random_state,
            args.bin_size,
        )

        print("\n=== Split Creation Complete ===")
        print(f"Train: {train_path}")
        print(f"Val: {val_path}")
        print(f"Test: {test_path}")

        return 0
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
