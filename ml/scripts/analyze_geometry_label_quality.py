#!/usr/bin/env python3
"""
Analyze geometry label quality for the inner Celsius dial reader.

This script reads the geometry manifest, computes deterministic temperatures
from center/tip labels, and flags suspicious rows for review or exclusion.

The goal is to create a cleaned, training-ready manifest (v2) that marks
rows with quality issues rather than deleting them.

Usage:
    poetry run python ml/scripts/analyze_geometry_label_quality.py
"""

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)


@dataclass
class GeometryRow:
    """Represents one row from the geometry manifest."""
    image_path: str
    temperature_c: float
    split: str
    source_width: int
    source_height: int
    loose_crop_x1: float
    loose_crop_y1: float
    loose_crop_x2: float
    loose_crop_y2: float
    center_x_source: float
    center_y_source: float
    tip_x_source: float
    tip_y_source: float
    dial_radius_source: float
    label_quality: str
    source_manifest: str
    notes: str
    
    # Computed fields
    angle_degrees_from_labels: float = 0.0
    deterministic_temperature_c: float = 0.0
    absolute_temperature_difference_c: float = 0.0
    center_tip_distance_pixels: float = 0.0
    center_x_normalized: float = 0.0
    center_y_normalized: float = 0.0
    tip_x_normalized: float = 0.0
    tip_y_normalized: float = 0.0
    quality_flag: str = "clean"
    rejection_reasons: List[str] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


def load_geometry_manifest(manifest_path: Path) -> List[GeometryRow]:
    """
    Load geometry manifest from CSV file.
    
    Args:
        manifest_path: Path to the geometry manifest CSV
        
    Returns:
        List of GeometryRow objects
    """
    rows = []
    
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            row = GeometryRow(
                image_path=row_dict["image_path"],
                temperature_c=float(row_dict["temperature_c"]),
                split=row_dict["split"],
                source_width=int(row_dict["source_width"]),
                source_height=int(row_dict["source_height"]),
                loose_crop_x1=float(row_dict["loose_crop_x1"]),
                loose_crop_y1=float(row_dict["loose_crop_y1"]),
                loose_crop_x2=float(row_dict["loose_crop_x2"]),
                loose_crop_y2=float(row_dict["loose_crop_y2"]),
                center_x_source=float(row_dict["center_x_source"]),
                center_y_source=float(row_dict["center_y_source"]),
                tip_x_source=float(row_dict["tip_x_source"]),
                tip_y_source=float(row_dict["tip_y_source"]),
                dial_radius_source=float(row_dict["dial_radius_source"]),
                label_quality=row_dict.get("label_quality", ""),
                source_manifest=row_dict.get("source_manifest", ""),
                notes=row_dict.get("notes", ""),
            )
            rows.append(row)
    
    return rows


def compute_geometry_metrics(row: GeometryRow) -> None:
    """
    Compute geometry metrics for a row.
    
    This computes:
    - Angle from center to tip
    - Deterministic temperature from angle
    - Absolute temperature difference
    - Center-tip distance
    - Normalized coordinates in loose crop
    
    Args:
        row: GeometryRow to compute metrics for (modified in place)
    """
    # Compute angle from center to tip
    row.angle_degrees_from_labels = angle_degrees_from_center_to_tip(
        row.center_x_source,
        row.center_y_source,
        row.tip_x_source,
        row.tip_y_source,
    )
    
    # Compute deterministic temperature from angle
    row.deterministic_temperature_c = celsius_from_inner_dial_angle_degrees(
        row.angle_degrees_from_labels
    )
    
    # Compute absolute temperature difference
    row.absolute_temperature_difference_c = abs(
        row.temperature_c - row.deterministic_temperature_c
    )
    
    # Compute center-tip distance
    dx = row.tip_x_source - row.center_x_source
    dy = row.tip_y_source - row.center_y_source
    row.center_tip_distance_pixels = math.sqrt(dx * dx + dy * dy)
    
    # Compute normalized coordinates in loose crop
    crop_width = row.loose_crop_x2 - row.loose_crop_x1
    crop_height = row.loose_crop_y2 - row.loose_crop_y1
    
    if crop_width > 0 and crop_height > 0:
        row.center_x_normalized = (row.center_x_source - row.loose_crop_x1) / crop_width
        row.center_y_normalized = (row.center_y_source - row.loose_crop_y1) / crop_height
        row.tip_x_normalized = (row.tip_x_source - row.loose_crop_x1) / crop_width
        row.tip_y_normalized = (row.tip_y_source - row.loose_crop_y1) / crop_height
    else:
        row.center_x_normalized = float("nan")
        row.center_y_normalized = float("nan")
        row.tip_x_normalized = float("nan")
        row.tip_y_normalized = float("nan")


def check_geometry_validity(
    row: GeometryRow,
    base_path: Path,
    temp_diff_threshold_review: float = 5.0,
    temp_diff_threshold_exclude: float = 10.0,
) -> Tuple[str, List[str]]:
    """
    Check geometry validity and assign quality flag.
    
    Rules:
    - clean: absolute_temperature_difference_c <= 5C and geometry checks pass
    - review: 5C < absolute_temperature_difference_c <= 10C or minor geometry warning
    - exclude: absolute_temperature_difference_c > 10C or serious geometry failure
    
    Args:
        row: GeometryRow to check
        base_path: Base path for checking image file existence
        temp_diff_review: Threshold for review flag
        temp_diff_exclude: Threshold for exclude flag
        
    Returns:
        Tuple of (quality_flag, list of rejection reasons)
    """
    reasons = []
    
    # Check for non-finite values
    if not math.isfinite(row.angle_degrees_from_labels):
        reasons.append("non_finite_angle")
    if not math.isfinite(row.deterministic_temperature_c):
        reasons.append("non_finite_temperature")
    if not math.isfinite(row.center_tip_distance_pixels):
        reasons.append("non_finite_distance")
    if not math.isfinite(row.center_x_normalized) or not math.isfinite(row.center_y_normalized):
        reasons.append("non_finite_normalized_center")
    if not math.isfinite(row.tip_x_normalized) or not math.isfinite(row.tip_y_normalized):
        reasons.append("non_finite_normalized_tip")
    
    # Check temperature difference
    if row.absolute_temperature_difference_c > temp_diff_threshold_exclude:
        reasons.append(f"temp_diff_too_large_{row.absolute_temperature_difference_c:.2f}C")
    elif row.absolute_temperature_difference_c > temp_diff_threshold_review:
        reasons.append(f"temp_diff_elevated_{row.absolute_temperature_difference_c:.2f}C")
    
    # Check center outside expected loose crop bounds (normalized should be roughly [0, 1])
    # Allow some tolerance since loose crop may not perfectly contain the dial
    if row.center_x_normalized < -0.2 or row.center_x_normalized > 1.2:
        reasons.append("center_outside_crop_x")
    if row.center_y_normalized < -0.2 or row.center_y_normalized > 1.2:
        reasons.append("center_outside_crop_y")
    
    # Check tip outside expected loose crop bounds
    if row.tip_x_normalized < -0.2 or row.tip_x_normalized > 1.2:
        reasons.append("tip_outside_crop_x")
    if row.tip_y_normalized < -0.2 or row.tip_y_normalized > 1.2:
        reasons.append("tip_outside_crop_y")
    
    # Check center-tip distance too small (needle too short to be reliable)
    if row.center_tip_distance_pixels < 50.0:
        reasons.append(f"center_tip_distance_too_small_{row.center_tip_distance_pixels:.1f}px")
    
    # Check center-tip distance too large relative to dial radius
    if row.dial_radius_source > 0:
        distance_ratio = row.center_tip_distance_pixels / row.dial_radius_source
        # Needle should be roughly within dial radius (allowing some tolerance)
        if distance_ratio > 1.5:
            reasons.append(f"center_tip_distance_too_large_ratio_{distance_ratio:.2f}")
        elif distance_ratio < 0.1:
            reasons.append(f"center_tip_distance_too_small_ratio_{distance_ratio:.2f}")
    
    # Check if image file exists
    image_full_path = base_path / row.image_path
    if not image_full_path.exists():
        reasons.append("image_file_missing")
    
    # Determine quality flag
    if len(reasons) == 0:
        quality_flag = "clean"
    elif row.absolute_temperature_difference_c > temp_diff_threshold_exclude:
        quality_flag = "exclude"
    elif "non_finite" in " ".join(reasons):
        quality_flag = "exclude"
    elif "image_file_missing" in reasons:
        quality_flag = "exclude"
    elif len(reasons) >= 3:
        quality_flag = "exclude"
    else:
        quality_flag = "review"
    
    return quality_flag, reasons


def analyze_label_quality(
    manifest_path: Path,
    base_path: Path,
) -> List[GeometryRow]:
    """
    Analyze label quality for all rows in manifest.
    
    Args:
        manifest_path: Path to geometry manifest CSV
        base_path: Base path for image file checks
        
    Returns:
        List of GeometryRow objects with computed metrics and quality flags
    """
    rows = load_geometry_manifest(manifest_path)
    
    for row in rows:
        compute_geometry_metrics(row)
        quality_flag, reasons = check_geometry_validity(row, base_path)
        row.quality_flag = quality_flag
        row.rejection_reasons = reasons
    
    return rows


def save_clean_manifest(rows: List[GeometryRow], output_path: Path) -> None:
    """
    Save cleaned manifest with quality flags.
    
    Args:
        rows: List of GeometryRow objects with computed metrics
        output_path: Path to save the cleaned manifest CSV
    """
    fieldnames = [
        "image_path",
        "temperature_c",
        "split",
        "source_width",
        "source_height",
        "loose_crop_x1",
        "loose_crop_y1",
        "loose_crop_x2",
        "loose_crop_y2",
        "center_x_source",
        "center_y_source",
        "tip_x_source",
        "tip_y_source",
        "dial_radius_source",
        "label_quality",
        "source_manifest",
        "notes",
        "angle_degrees_from_labels",
        "deterministic_temperature_c",
        "absolute_temperature_difference_c",
        "center_tip_distance_pixels",
        "quality_flag",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            writer.writerow({
                "image_path": row.image_path,
                "temperature_c": row.temperature_c,
                "split": row.split,
                "source_width": row.source_width,
                "source_height": row.source_height,
                "loose_crop_x1": row.loose_crop_x1,
                "loose_crop_y1": row.loose_crop_y1,
                "loose_crop_x2": row.loose_crop_x2,
                "loose_crop_y2": row.loose_crop_y2,
                "center_x_source": row.center_x_source,
                "center_y_source": row.center_y_source,
                "tip_x_source": row.tip_x_source,
                "tip_y_source": row.tip_y_source,
                "dial_radius_source": row.dial_radius_source,
                "label_quality": row.label_quality,
                "source_manifest": row.source_manifest,
                "notes": row.notes,
                "angle_degrees_from_labels": f"{row.angle_degrees_from_labels:.4f}",
                "deterministic_temperature_c": f"{row.deterministic_temperature_c:.4f}",
                "absolute_temperature_difference_c": f"{row.absolute_temperature_difference_c:.4f}",
                "center_tip_distance_pixels": f"{row.center_tip_distance_pixels:.4f}",
                "quality_flag": row.quality_flag,
            })


def generate_quality_report(rows: List[GeometryRow], output_path: Path) -> None:
    """
    Generate quality analysis report in Markdown format.
    
    Args:
        rows: List of GeometryRow objects with computed metrics
        output_path: Path to save the report
    """
    total_rows = len(rows)
    clean_rows = sum(1 for r in rows if r.quality_flag == "clean")
    review_rows = sum(1 for r in rows if r.quality_flag == "review")
    exclude_rows = sum(1 for r in rows if r.quality_flag == "exclude")
    
    # Compute statistics for temperature differences
    temp_diffs = [r.absolute_temperature_difference_c for r in rows]
    temp_diffs_sorted = sorted(temp_diffs)
    
    mean_diff = sum(temp_diffs) / len(temp_diffs) if temp_diffs else 0.0
    median_diff = temp_diffs_sorted[len(temp_diffs) // 2] if temp_diffs else 0.0
    p90_idx = int(len(temp_diffs_sorted) * 0.9)
    p90_diff = temp_diffs_sorted[p90_idx] if temp_diffs else 0.0
    max_diff = max(temp_diffs) if temp_diffs else 0.0
    
    # Get suspicious rows (review + exclude), sorted by temp diff
    suspicious_rows = [r for r in rows if r.quality_flag != "clean"]
    suspicious_rows.sort(key=lambda r: r.absolute_temperature_difference_c, reverse=True)
    
    # Get rows that need manual correction
    exclude_images = [r for r in rows if r.quality_flag == "exclude"]
    review_images = [r for r in rows if r.quality_flag == "review"]
    
    # Build report
    report_lines = [
        "# Geometry Label Quality Analysis Report (v2)",
        "",
        "## Summary",
        "",
        f"- **Total rows:** {total_rows}",
        f"- **Clean rows:** {clean_rows} ({100*clean_rows/total_rows:.1f}%)",
        f"- **Review rows:** {review_rows} ({100*review_rows/total_rows:.1f}%)",
        f"- **Exclude rows:** {exclude_rows} ({100*exclude_rows/total_rows:.1f}%)",
        "",
        "## Temperature Difference Statistics",
        "",
        f"- **Mean:** {mean_diff:.2f}C",
        f"- **Median:** {median_diff:.2f}C",
        f"- **P90:** {p90_diff:.2f}C",
        f"- **Max:** {max_diff:.2f}C",
        "",
        "## Suspicious Rows (Worst First)",
        "",
        "Top 30 rows by absolute temperature difference:",
        "",
        "| # | Image | Temp (manifest) | Temp (deterministic) | Diff (C) | Flag | Reasons |",
        "|---|-------|-----------------|----------------------|-----------|------|---------|",
    ]
    
    for i, row in enumerate(suspicious_rows[:30], 1):
        image_name = Path(row.image_path).name
        reasons_str = "; ".join(row.rejection_reasons) if row.rejection_reasons else "elevated_temp_diff"
        report_lines.append(
            f"| {i} | {image_name} | {row.temperature_c:.1f} | {row.deterministic_temperature_c:.1f} | "
            f"{row.absolute_temperature_difference_c:.2f} | {row.quality_flag} | {reasons_str} |"
        )
    
    report_lines.extend([
        "",
        "## Recommendation for Phase 3 Training",
        "",
    ])
    
    # Determine if Phase 3 can proceed
    if clean_rows >= 100:
        report_lines.append(
            f"**Phase 3 can proceed** using only clean rows ({clean_rows} rows available).\n"
        )
        report_lines.append(
            "Training on clean rows only is recommended to ensure model learns correct geometry.\n"
        )
    elif clean_rows >= 50:
        report_lines.append(
            f"**Phase 3 can proceed with caution** using clean rows ({clean_rows} rows).\n"
        )
        report_lines.append(
            "Consider manual review of 'review' rows to potentially recover more training data.\n"
        )
    else:
        report_lines.append(
            f"**Phase 3 should wait** - only {clean_rows} clean rows available.\n"
        )
        report_lines.append(
            "Manual CVAT correction recommended for suspicious rows before training.\n"
        )
    
    report_lines.extend([
        "",
        "## Images Needing Manual CVAT Correction or Exclusion",
        "",
        "### Exclude List (serious issues)",
        "",
    ])
    
    if exclude_images:
        for row in exclude_images:
            image_name = Path(row.image_path).name
            reasons_str = "; ".join(row.rejection_reasons)
            report_lines.append(f"- `{image_name}` - {reasons_str}")
    else:
        report_lines.append("*None*")
    
    report_lines.extend([
        "",
        "### Review List (elevated temperature difference or minor geometry warnings)",
        "",
    ])
    
    if review_images:
        for row in review_images:
            image_name = Path(row.image_path).name
            reasons_str = "; ".join(row.rejection_reasons) if row.rejection_reasons else "elevated_temp_diff"
            report_lines.append(f"- `{image_name}` - {reasons_str}")
    else:
        report_lines.append("*None*")
    
    report_lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review the suspicious rows in CVAT to verify center/tip annotations",
        "2. Correct or exclude rows with serious annotation errors",
        "3. Re-run this analysis to generate updated v2_clean manifest",
        "4. Proceed to Phase 3 training using clean rows only",
        "",
        "---",
        "",
        "*Report generated by analyze_geometry_label_quality.py*",
    ])
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    """Main entry point for label quality analysis."""
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    manifest_path = base_path / "ml" / "data" / "geometry_reader_manifest_v1.csv"
    output_manifest_path = base_path / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
    report_path = base_path / "ml" / "reports" / "geometry_label_quality_v2.md"
    
    print(f"Loading manifest from: {manifest_path}")
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)
    
    # Analyze label quality
    rows = analyze_label_quality(manifest_path, base_path)
    
    # Print summary
    total_rows = len(rows)
    clean_rows = sum(1 for r in rows if r.quality_flag == "clean")
    review_rows = sum(1 for r in rows if r.quality_flag == "review")
    exclude_rows = sum(1 for r in rows if r.quality_flag == "exclude")
    
    print(f"\nAnalysis complete:")
    print(f"  Total rows: {total_rows}")
    print(f"  Clean: {clean_rows}")
    print(f"  Review: {review_rows}")
    print(f"  Exclude: {exclude_rows}")
    
    # Save cleaned manifest
    print(f"\nSaving cleaned manifest to: {output_manifest_path}")
    save_clean_manifest(rows, output_manifest_path)
    
    # Generate report
    print(f"Generating report: {report_path}")
    generate_quality_report(rows, report_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
