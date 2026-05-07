"""Build a canonical labeled manifest from multiple data sources.

This script merges multiple manifest files into a single canonical manifest,
handling deduplication, conflict detection, and source precedence.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Define source precedence (higher index = higher priority)
SOURCE_PRECEDENCE = [
    "all_captured_images_manifest",
    "unified_manifest_with_crops_v2",
    "full_labelled_plus_board30_valid_with_new5",
    "new_labelled_captures4",
    "hard_cases_plus_board30_valid_with_new6",
]

# Source tag mapping
SOURCE_TAG_MAP = {
    "all_captured_images_manifest": "board_capture",
    "unified_manifest_with_crops_v2": "core",
    "full_labelled_plus_board30_valid_with_new5": "core",
    "new_labelled_captures4": "board_capture",
    "hard_cases_plus_board30_valid_with_new6": "hard_case",
}


def normalize_path(path_str: str, repo_root: Path) -> str:
    """Normalize a path string to use forward slashes and be relative to repo root.

    Args:
        path_str: The path string to normalize.
        repo_root: The repository root directory.

    Returns:
        Normalized path string using forward slashes.
    """
    # First, normalize backslashes to forward slashes manually
    # This handles Windows-style paths regardless of the OS
    normalized = path_str.replace("\\", "/")

    # Convert to Path object and normalize
    path = Path(normalized)

    # If it's an absolute path, try to make it relative to repo root
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            # Path is not under repo_root, keep as-is
            pass

    # Convert to forward slashes for consistency
    return path.as_posix()


def resolve_full_path(normalized_path: str, repo_root: Path) -> Path:
    """Resolve a normalized path to a full Path object.

    Args:
        normalized_path: The normalized path string.
        repo_root: The repository root directory.

    Returns:
        Full Path object.
    """
    return repo_root / normalized_path


def load_manifest(
    file_path: Path,
    repo_root: Path,
    source_name: str,
) -> Optional[pd.DataFrame]:
    """Load a manifest CSV file and standardize its columns.

    Args:
        file_path: Path to the CSV file.
        repo_root: Repository root for path normalization.
        source_name: Name of the source for tagging.

    Returns:
        DataFrame with standardized columns, or None if file doesn't exist.
    """
    if not file_path.exists():
        logger.warning(f"Manifest file not found: {file_path}")
        return None

    logger.info(f"Loading manifest: {file_path}")

    # Read CSV, skipping comment lines
    try:
        df = pd.read_csv(file_path, comment="#")
    except pd.errors.EmptyDataError:
        logger.warning(f"Manifest file is empty: {file_path}")
        return None

    if df.empty:
        logger.warning(f"Manifest file is empty: {file_path}")
        return None

    # Standardize column names
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})

    # Ensure required columns exist
    if "image_path" not in df.columns:
        logger.error(f"Missing 'image_path' column in {file_path}")
        return None

    if "value" not in df.columns:
        logger.error(f"Missing 'value' column in {file_path}")
        return None

    # Normalize paths
    df["image_path"] = df["image_path"].apply(
        lambda x: normalize_path(str(x), repo_root)
    )

    # Add source metadata
    df["source_file"] = source_name
    df["source_tag"] = SOURCE_TAG_MAP.get(source_name, "unknown")
    df["hardness_tag"] = (
        "hard_case"
        if source_name == "hard_cases_plus_board30_valid_with_new6"
        else "normal"
    )

    # Ensure crop columns exist (may be NaN if not present in source)
    for col in ["crop_x_min", "crop_y_min", "crop_x_max", "crop_y_max"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Select and order columns
    columns = [
        "image_path",
        "value",
        "source_tag",
        "hardness_tag",
        "crop_x_min",
        "crop_y_min",
        "crop_x_max",
        "crop_y_max",
        "source_file",
    ]
    df = df[[col for col in columns if col in df.columns]]

    return df


def filter_valid_rows(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Filter out rows with missing files or invalid labels.

    Args:
        df: Input DataFrame.
        repo_root: Repository root for file existence checks.

    Returns:
        Filtered DataFrame.
    """
    initial_count = len(df)

    # Filter out rows with missing/non-numeric labels
    df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
    df["value"] = pd.to_numeric(df["value"])

    # Filter out rows with missing files
    def file_exists(row):
        full_path = resolve_full_path(row["image_path"], repo_root)
        return full_path.exists()

    df = df[df.apply(file_exists, axis=1)]

    final_count = len(df)
    logger.info(
        f"Filtered {initial_count - final_count} rows with missing files or invalid labels"
    )

    return df


def deduplicate_and_resolve_conflicts(
    df: pd.DataFrame,
    conflict_threshold: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate by image_path and resolve conflicts.

    Args:
        df: Input DataFrame with potentially duplicate image_paths.
        conflict_threshold: Maximum allowed label difference for auto-resolution.

    Returns:
        Tuple of (canonical_df, conflicts_df).
    """
    canonical_rows = []
    conflict_rows = []

    # Group by normalized image_path
    grouped = df.groupby("image_path")

    for image_path, group in grouped:
        if len(group) == 1:
            # No duplicate, keep as-is
            canonical_rows.append(group.iloc[0].to_dict())
            continue

        # Sort by source precedence (higher index = higher priority)
        group = group.copy()
        group["precedence"] = group["source_file"].apply(
            lambda x: SOURCE_PRECEDENCE.index(x) if x in SOURCE_PRECEDENCE else -1
        )
        group = group.sort_values("precedence")

        # Check for label conflicts
        values = group["value"].values
        max_diff = max(values) - min(values)

        if max_diff > conflict_threshold:
            # Conflict detected - move all rows to conflicts
            logger.warning(
                f"Conflict detected for {image_path}: values differ by {max_diff:.2f}"
            )
            for _, row in group.iterrows():
                conflict_rows.append(row.to_dict())
        else:
            # Keep the highest precedence (last) row
            canonical_rows.append(group.iloc[-1].to_dict())

    canonical_df = pd.DataFrame(canonical_rows)
    conflicts_df = pd.DataFrame(conflict_rows)

    logger.info(f"Canonical rows: {len(canonical_df)}, Conflicts: {len(conflicts_df)}")

    return canonical_df, conflicts_df


def build_canonical_manifest(
    data_dir: Path,
    repo_root: Path,
    conflict_threshold: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical manifest from all sources.

    Args:
        data_dir: Directory containing manifest files.
        repo_root: Repository root directory.
        conflict_threshold: Maximum allowed label difference for auto-resolution.

    Returns:
        Tuple of (canonical_df, conflicts_df).
    """
    all_dfs = []

    # Define manifest files to load
    manifest_files = [
        ("unified_manifest_with_crops_v2.csv", "unified_manifest_with_crops_v2"),
        (
            "full_labelled_plus_board30_valid_with_new5.csv",
            "full_labelled_plus_board30_valid_with_new5",
        ),
        (
            "hard_cases_plus_board30_valid_with_new6.csv",
            "hard_cases_plus_board30_valid_with_new6",
        ),
        ("new_labelled_captures4.csv", "new_labelled_captures4"),
        ("all_captured_images_manifest.csv", "all_captured_images_manifest"),
    ]

    for filename, source_name in manifest_files:
        file_path = data_dir / filename
        df = load_manifest(file_path, repo_root, source_name)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid manifest files found")

    # Combine all manifests
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} rows from {len(all_dfs)} sources")

    # Filter valid rows
    combined_df = filter_valid_rows(combined_df, repo_root)

    # Deduplicate and resolve conflicts
    canonical_df, conflicts_df = deduplicate_and_resolve_conflicts(
        combined_df, conflict_threshold
    )

    return canonical_df, conflicts_df


def main():
    """Main entry point for the manifest builder."""
    parser = argparse.ArgumentParser(
        description="Build a canonical labeled manifest from multiple data sources."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing manifest files (default: ml/data relative to repo root)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for canonical manifest (default: ml/data/canonical_manifest_v1.csv)",
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=None,
        help="Output path for conflicts (default: ml/data/canonical_manifest_conflicts_v1.csv)",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=float,
        default=1.0,
        help="Maximum label difference for auto-resolution (default: 1.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-detect repo root
    if args.repo_root is None:
        # Start from this script's location and go up to find the repo root
        script_dir = Path(__file__).resolve().parent
        # ml/scripts -> ml -> repo root
        args.repo_root = script_dir.parent.parent
    else:
        args.repo_root = args.repo_root.resolve()

    # Set default data dir
    if args.data_dir is None:
        args.data_dir = args.repo_root / "ml" / "data"
    else:
        args.data_dir = args.data_dir.resolve()

    # Set default output paths
    if args.output is None:
        args.output = args.data_dir / "canonical_manifest_v1.csv"
    if args.conflicts_output is None:
        args.conflicts_output = args.data_dir / "canonical_manifest_conflicts_v1.csv"

    logger.info(f"Repository root: {args.repo_root}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Conflicts output: {args.conflicts_output}")

    # Build canonical manifest
    canonical_df, conflicts_df = build_canonical_manifest(
        args.data_dir,
        args.repo_root,
        args.conflict_threshold,
    )

    # Save outputs
    canonical_df.to_csv(args.output, index=False)
    logger.info(f"Saved canonical manifest: {args.output} ({len(canonical_df)} rows)")

    if not conflicts_df.empty:
        conflicts_df.to_csv(args.conflicts_output, index=False)
        logger.info(
            f"Saved conflicts: {args.conflicts_output} ({len(conflicts_df)} rows)"
        )
    else:
        logger.info("No conflicts detected")

    # Print summary
    print("\n=== Canonical Manifest Summary ===")
    print(f"Total rows: {len(canonical_df)}")
    print(f"\nBy source_tag:")
    print(canonical_df["source_tag"].value_counts())
    print(f"\nBy hardness_tag:")
    print(canonical_df["hardness_tag"].value_counts())
    print(
        f"\nValue range: {canonical_df['value'].min():.1f} to {canonical_df['value'].max():.1f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
