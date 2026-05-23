"""
Debug overlay generator for geometry crop dataset.

This script generates visual overlays for inspecting the crop-jitter pipeline.
It reads the geometry manifest, generates jittered crops, and saves images
with center/tip annotations for human verification.

Usage:
    poetry run python ml/scripts/build_geometry_crop_debug_set.py
    poetry run python ml/scripts/build_geometry_crop_debug_set.py --quality-filter clean

Output:
    ml/debug/geometry_crops_v1/
        - Crop images with overlays
        - debug_crop_manifest.csv
"""

import argparse
import csv
import math
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    load_geometry_manifest,
    SourceGeometryExample,
    generate_jitter_params,
    create_jittered_crop,
    JitteredCrop,
    load_and_generate_crops,
)
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)


def load_image(image_path: Path) -> Optional[Any]:
    """
    Load an image from disk.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object or None if not found
    """
    try:
        from PIL import Image
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        return None
    except ImportError:
        print("PIL not available, skipping image loading")
        return None


def create_crop_overlay(
    crop: JitteredCrop,
    source_image: Any,
    output_path: Path,
) -> bool:
    """
    Create a debug overlay image for a jittered crop.

    The overlay shows:
    - Cropped image region
    - Center point (green circle)
    - Tip point (red circle)
    - Line from center to tip (blue)
    - Temperature labels
    - Jitter parameters

    Args:
        crop: JitteredCrop object
        source_image: PIL Image of source image
        output_path: Path to save overlay image

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available")
        return False

    if source_image is None:
        return False

    if not crop.accepted:
        # Skip rejected crops
        return False

    # Extract crop region from source image
    crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
    crop_image = source_image.crop(crop_box)

    # Resize to display size (448x448 for better visibility)
    display_size = 448
    crop_display = crop_image.resize((display_size, display_size), Image.LANCZOS)

    # Create drawing context
    draw = ImageDraw.Draw(crop_display)

    # Scale factor from crop to display
    scale_x = display_size / (crop.crop_x2 - crop.crop_x1)
    scale_y = display_size / (crop.crop_y2 - crop.crop_y1)

    # Compute display coordinates for center and tip
    center_x_disp = (crop.center_x_224 / 224.0) * display_size
    center_y_disp = (crop.center_y_224 / 224.0) * display_size
    tip_x_disp = (crop.tip_x_224 / 224.0) * display_size
    tip_y_disp = (crop.tip_y_224 / 224.0) * display_size

    # Draw line from center to tip
    draw.line(
        [(center_x_disp, center_y_disp), (tip_x_disp, tip_y_disp)],
        fill="blue",
        width=2,
    )

    # Draw center point (green circle)
    center_radius = 6
    draw.ellipse(
        [
            center_x_disp - center_radius,
            center_y_disp - center_radius,
            center_x_disp + center_radius,
            center_y_disp + center_radius,
        ],
        fill="green",
        outline="white",
        width=2,
    )

    # Draw tip point (red circle)
    tip_radius = 6
    draw.ellipse(
        [
            tip_x_disp - tip_radius,
            tip_y_disp - tip_radius,
            tip_x_disp + tip_radius,
            tip_y_disp + tip_radius,
        ],
        fill="red",
        outline="white",
        width=2,
    )

    # Add text overlay with metadata
    text_lines = [
        f"Temp: {crop.temperature_c:.1f}C",
        f"Angle: {crop.angle_degrees:.1f} deg",
        f"Det Temp: {crop.deterministic_temperature_c:.1f}C",
        f"Diff: {crop.absolute_temperature_difference_c:.1f}C",
        f"Split: {crop.split}",
        f"Jitter: ({crop.jitter_shift_x:+d}, {crop.jitter_shift_y:+d})",
        f"Scale: {crop.jitter_scale:.2f}, Aspect: {crop.jitter_aspect:.2f}",
    ]

    # Draw text background
    text_height = 20 * len(text_lines) + 10
    draw.rectangle(
        [(0, 0), (display_size, text_height)],
        fill=(0, 0, 0, 180),
    )

    # Draw text
    for i, line in enumerate(text_lines):
        y = 5 + i * 20
        draw.text((5, y), line, fill="white")

    # Save image
    crop_display.save(output_path, "JPEG", quality=85)
    return True


def load_clean_manifest_with_quality(manifest_path: Path) -> List[SourceGeometryExample]:
    """
    Load v2_clean manifest and filter by quality_flag if present.
    
    This is a wrapper that can read the v2_clean manifest which has
    additional quality_flag column.
    
    Args:
        manifest_path: Path to manifest CSV
        
    Returns:
        List of SourceGeometryExample objects
    """
    examples = []
    
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            # Skip non-clean rows if quality_flag column exists
            if "quality_flag" in row_dict and row_dict["quality_flag"] != "clean":
                continue
            
            ex = SourceGeometryExample(
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
            )
            examples.append(ex)
    
    return examples


def sample_examples_by_split(
    examples: List[SourceGeometryExample],
    samples_per_split: int = 10,
    crops_per_example: int = 5,
) -> List[Tuple[SourceGeometryExample, int]]:
    """
    Sample examples from each split for debug visualization.

    Args:
        examples: List of source geometry examples
        samples_per_split: Number of examples to sample per split
        crops_per_example: Number of jittered crops to generate per example

    Returns:
        List of (example, num_crops) tuples
    """
    # Group by split
    by_split: Dict[str, List[SourceGeometryExample]] = {}
    for ex in examples:
        if ex.split not in by_split:
            by_split[ex.split] = []
        by_split[ex.split].append(ex)

    # Sample from each split
    sampled = []
    for split_name in ["train", "val", "test"]:
        if split_name not in by_split:
            continue
        
        split_examples = by_split[split_name]
        # Deterministic sampling
        rng = random.Random(42)
        sampled_indices = rng.sample(
            range(len(split_examples)),
            min(samples_per_split, len(split_examples)),
        )
        
        for idx in sampled_indices:
            sampled.append((split_examples[idx], crops_per_example))

    return sampled


def save_debug_manifest(all_crops: List[JitteredCrop], output_path: Path) -> int:
    """
    Save debug manifest CSV with all crop metadata.

    Args:
        all_crops: List of all jittered crops
        output_path: Path to save manifest

    Returns:
        Number of rows written
    """
    fieldnames = [
        "source_image_path",
        "split",
        "temperature_c",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "center_x_normalized",
        "center_y_normalized",
        "tip_x_normalized",
        "tip_y_normalized",
        "center_x_224",
        "center_y_224",
        "tip_x_224",
        "tip_y_224",
        "angle_degrees",
        "deterministic_temperature_c",
        "absolute_temperature_difference_c",
        "jitter_shift_x",
        "jitter_shift_y",
        "jitter_scale",
        "jitter_aspect",
        "accepted",
        "rejection_reason",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for crop in all_crops:
            writer.writerow({
                "source_image_path": crop.source_image_path,
                "split": crop.split,
                "temperature_c": crop.temperature_c,
                "crop_x1": crop.crop_x1,
                "crop_y1": crop.crop_y1,
                "crop_x2": crop.crop_x2,
                "crop_y2": crop.crop_y2,
                "center_x_normalized": f"{crop.center_x_normalized:.6f}",
                "center_y_normalized": f"{crop.center_y_normalized:.6f}",
                "tip_x_normalized": f"{crop.tip_x_normalized:.6f}",
                "tip_y_normalized": f"{crop.tip_y_normalized:.6f}",
                "center_x_224": f"{crop.center_x_224:.2f}",
                "center_y_224": f"{crop.center_y_224:.2f}",
                "tip_x_224": f"{crop.tip_x_224:.2f}",
                "tip_y_224": f"{crop.tip_y_224:.2f}",
                "angle_degrees": f"{crop.angle_degrees:.2f}",
                "deterministic_temperature_c": f"{crop.deterministic_temperature_c:.2f}",
                "absolute_temperature_difference_c": f"{crop.absolute_temperature_difference_c:.2f}",
                "jitter_shift_x": crop.jitter_shift_x,
                "jitter_shift_y": crop.jitter_shift_y,
                "jitter_scale": f"{crop.jitter_scale:.4f}",
                "jitter_aspect": f"{crop.jitter_aspect:.4f}",
                "accepted": crop.accepted,
                "rejection_reason": crop.rejection_reason or "",
            })

    return len(all_crops)


def generate_validation_stats(
    all_crops: List[JitteredCrop],
    rejection_counts: Dict[str, int],
) -> Dict[str, Any]:
    """
    Generate validation statistics from crops.

    Args:
        all_crops: List of all jittered crops
        rejection_counts: Count of rejections by reason

    Returns:
        Dictionary of statistics
    """
    accepted_crops = [c for c in all_crops if c.accepted]
    rejected_crops = [c for c in all_crops if not c.accepted]

    # Compute normalized coordinate ranges
    if accepted_crops:
        center_x_vals = [c.center_x_normalized for c in accepted_crops]
        center_y_vals = [c.center_y_normalized for c in accepted_crops]
        tip_x_vals = [c.tip_x_normalized for c in accepted_crops]
        tip_y_vals = [c.tip_y_normalized for c in accepted_crops]

        center_x_range = (min(center_x_vals), max(center_x_vals))
        center_y_range = (min(center_y_vals), max(center_y_vals))
        tip_x_range = (min(tip_x_vals), max(tip_x_vals))
        tip_y_range = (min(tip_y_vals), max(tip_y_vals))
    else:
        center_x_range = center_y_range = tip_x_range = tip_y_range = (0, 0)

    # Compute temperature differences
    temp_diffs = [c.absolute_temperature_difference_c for c in accepted_crops if c.absolute_temperature_difference_c is not None]
    mean_temp_diff = sum(temp_diffs) / len(temp_diffs) if temp_diffs else 0.0

    # Find worst mismatches
    worst_mismatches = sorted(
        [c for c in accepted_crops if c.absolute_temperature_difference_c is not None],
        key=lambda c: c.absolute_temperature_difference_c,
        reverse=True,
    )[:20]

    return {
        "total_attempted": len(all_crops),
        "accepted": len(accepted_crops),
        "rejected": len(rejected_crops),
        "rejection_counts": rejection_counts,
        "center_x_range": center_x_range,
        "center_y_range": center_y_range,
        "tip_x_range": tip_x_range,
        "tip_y_range": tip_y_range,
        "mean_temp_diff": mean_temp_diff,
        "worst_mismatches": worst_mismatches,
    }


def main():
    """Main entry point for debug overlay generation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate debug overlays for geometry crop dataset"
    )
    parser.add_argument(
        "--quality-filter",
        type=str,
        choices=["clean", "none"],
        default="none",
        help="Filter manifest by quality flag (requires v2_clean manifest)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Geometry Crop Debug Overlay Generator")
    print("=" * 80)

    # Configuration
    project_root = Path(__file__).parent.parent.parent
    
    # Choose manifest based on quality filter
    if args.quality_filter == "clean":
        manifest_path = project_root / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
        print(f"Using v2_clean manifest with quality filter: {args.quality_filter}")
    else:
        manifest_path = project_root / "ml" / "data" / "geometry_reader_manifest_v1.csv"
        print(f"Using v1 manifest (no quality filter)")
    
    output_dir = project_root / "ml" / "debug" / "geometry_crops_v1"
    images_dir = output_dir / "images"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nManifest: {manifest_path}")
    print(f"Output dir: {output_dir}")

    # Load manifest
    print("\nLoading geometry manifest...")
    if args.quality_filter == "clean" and manifest_path.exists():
        examples = load_clean_manifest_with_quality(manifest_path)
        print(f"Loaded {len(examples)} clean examples (filtered from v2_clean)")
    else:
        examples = load_geometry_manifest(manifest_path)
        print(f"Loaded {len(examples)} examples")

    # Count by split
    split_counts: Dict[str, int] = {}
    for ex in examples:
        split_counts[ex.split] = split_counts.get(ex.split, 0) + 1
    print(f"Split distribution: {split_counts}")

    # Sample examples for debug visualization
    print("\nSampling examples for debug visualization...")
    sampled = sample_examples_by_split(examples, samples_per_split=10, crops_per_example=5)
    print(f"Sampling {len(sampled)} examples, {sum(n for _, n in sampled)} total crops")

    # Generate crops for sampled examples
    print("\nGenerating jittered crops...")
    all_crops: List[JitteredCrop] = []
    rejection_counts: Dict[str, int] = {}

    for i, (example, num_crops) in enumerate(sampled):
        example_seed = 42 + i
        rng = random.Random(example_seed)

        for j in range(num_crops):
            jitter = generate_jitter_params(rng)
            crop = create_jittered_crop(example, jitter)
            all_crops.append(crop)

            if not crop.accepted and crop.rejection_reason:
                rejection_counts[crop.rejection_reason] = rejection_counts.get(crop.rejection_reason, 0) + 1

    print(f"Generated {len(all_crops)} crops")
    print(f"Accepted: {sum(1 for c in all_crops if c.accepted)}")
    print(f"Rejected: {sum(1 for c in all_crops if not c.accepted)}")

    # Generate overlays for accepted crops
    print("\nGenerating overlay images...")
    overlay_count = 0
    for i, crop in enumerate(all_crops):
        if not crop.accepted:
            continue

        # Load source image
        source_path = project_root / crop.source_image_path
        source_image = load_image(source_path)

        if source_image is None:
            continue

        # Create overlay
        output_path = images_dir / f"crop_{i:04d}_{crop.split}_{Path(crop.source_image_path).stem}.jpg"
        if create_crop_overlay(crop, source_image, output_path):
            overlay_count += 1

    print(f"Created {overlay_count} overlay images")

    # Save debug manifest
    print("\nSaving debug manifest...")
    manifest_output = output_dir / "debug_crop_manifest.csv"
    rows_written = save_debug_manifest(all_crops, manifest_output)
    print(f"Saved {rows_written} rows to {manifest_output}")

    # Generate validation stats
    print("\nComputing validation statistics...")
    stats = generate_validation_stats(all_crops, rejection_counts)

    # Save validation report
    report_path = output_dir / "validation_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Geometry Crop Debug Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total attempted crops: {stats['total_attempted']}\n")
        f.write(f"Accepted: {stats['accepted']}\n")
        f.write(f"Rejected: {stats['rejected']}\n\n")
        f.write("Rejection reasons:\n")
        for reason, count in sorted(stats['rejection_counts'].items(), key=lambda x: -x[1]):
            f.write(f"  {reason}: {count}\n")
        f.write("\n")
        f.write(f"Center X range: [{stats['center_x_range'][0]:.4f}, {stats['center_x_range'][1]:.4f}]\n")
        f.write(f"Center Y range: [{stats['center_y_range'][0]:.4f}, {stats['center_y_range'][1]:.4f}]\n")
        f.write(f"Tip X range: [{stats['tip_x_range'][0]:.4f}, {stats['tip_x_range'][1]:.4f}]\n")
        f.write(f"Tip Y range: [{stats['tip_y_range'][0]:.4f}, {stats['tip_y_range'][1]:.4f}]\n\n")
        f.write(f"Mean absolute temperature difference: {stats['mean_temp_diff']:.2f}C\n\n")
        f.write("Worst 20 temperature mismatches:\n")
        for i, crop in enumerate(stats['worst_mismatches']):
            f.write(f"  {i+1}. {crop.source_image_path}: diff={crop.absolute_temperature_difference_c:.2f}C "
                    f"(manifest={crop.temperature_c:.1f}C, deterministic={crop.deterministic_temperature_c:.1f}C)\n")

    print(f"Saved validation summary to {report_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Source rows loaded: {len(examples)}")
    print(f"Attempted jittered crops: {stats['total_attempted']}")
    print(f"Accepted: {stats['accepted']} ({100*stats['accepted']/stats['total_attempted']:.1f}%)")
    print(f"Rejected: {stats['rejected']} ({100*stats['rejected']/stats['total_attempted']:.1f}%)")
    print(f"Overlay images created: {overlay_count}")
    print(f"Debug manifest: {manifest_output}")
    print(f"Validation summary: {report_path}")
    print(f"Mean temperature difference: {stats['mean_temp_diff']:.2f}C")
    print("\nTop rejection reasons:")
    for reason, count in sorted(stats['rejection_counts'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {reason}: {count}")

    return stats


if __name__ == "__main__":
    main()
