#!/usr/bin/env python3
"""
Generate visual review package for suspicious geometry labels.

This script creates overlay images for rows flagged as "review" or "exclude"
in the v2_clean manifest, plus a contact sheet of the worst 30 rows.

Usage:
    poetry run python ml/scripts/generate_geometry_label_review.py

Output:
    ml/debug/geometry_label_quality_v2/
        - Overlay images for suspicious rows
        - worst_30_contact_sheet.jpg
"""

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Tuple

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
    angle_degrees_from_labels: float = 0.0
    deterministic_temperature_c: float = 0.0
    absolute_temperature_difference_c: float = 0.0
    center_tip_distance_pixels: float = 0.0
    quality_flag: str = "clean"
    rejection_reasons: List[str] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


def load_clean_manifest(manifest_path: Path) -> List[GeometryRow]:
    """Load v2_clean manifest with quality flags."""
    rows = []
    
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            # Handle the duplicate column name issue
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
                angle_degrees_from_labels=float(row_dict.get("angle_degrees_from_labels", 0)),
                deterministic_temperature_c=float(row_dict.get("deterministic_temperature_c", 0)),
                absolute_temperature_difference_c=float(row_dict.get("absolute_temperature_difference_c", 0)),
                center_tip_distance_pixels=float(row_dict.get("center_tip_distance_pixels", 0)),
                quality_flag=row_dict.get("quality_flag", "clean"),
            )
            rows.append(row)
    
    return rows


def load_image(image_path: Path) -> Optional[Any]:
    """Load an image from disk."""
    try:
        from PIL import Image
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        return None
    except ImportError:
        print("PIL not available")
        return None


def create_suspicious_overlay(
    row: GeometryRow,
    source_image: Any,
    output_path: Path,
    base_path: Path,
) -> bool:
    """
    Create overlay for a suspicious row.
    
    Shows:
    - Original source image with loose crop box
    - Center point (green)
    - Tip point (red)
    - Center-to-tip line (blue)
    - Manifest temperature, deterministic temperature, difference
    - Quality flag
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    
    if source_image is None:
        return False
    
    # Create a copy for drawing
    img = source_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Draw loose crop box
    crop_box = [row.loose_crop_x1, row.loose_crop_y1, row.loose_crop_x2, row.loose_crop_y2]
    draw.rectangle(crop_box, outline="yellow", width=3)
    
    # Draw center point (green)
    center_radius = 8
    draw.ellipse(
        [
            row.center_x_source - center_radius,
            row.center_y_source - center_radius,
            row.center_x_source + center_radius,
            row.center_y_source + center_radius,
        ],
        fill="green",
        outline="white",
        width=2,
    )
    
    # Draw tip point (red)
    tip_radius = 8
    draw.ellipse(
        [
            row.tip_x_source - tip_radius,
            row.tip_y_source - tip_radius,
            row.tip_x_source + tip_radius,
            row.tip_y_source + tip_radius,
        ],
        fill="red",
        outline="white",
        width=2,
    )
    
    # Draw line from center to tip
    draw.line(
        [(row.center_x_source, row.center_y_source), (row.tip_x_source, row.tip_y_source)],
        fill="blue",
        width=3,
    )
    
    # Add text overlay
    image_name = Path(row.image_path).name
    text_lines = [
        f"Image: {image_name}",
        f"Split: {row.split}",
        f"Manifest Temp: {row.temperature_c:.1f}C",
        f"Deterministic Temp: {row.deterministic_temperature_c:.1f}C",
        f"Diff: {row.absolute_temperature_difference_c:.2f}C",
        f"Flag: {row.quality_flag.upper()}",
        f"Angle: {row.angle_degrees_from_labels:.1f} deg",
    ]
    
    # Draw text background
    text_height = 22 * len(text_lines) + 10
    draw.rectangle([(0, 0), (400, text_height)], fill=(0, 0, 0, 180))
    
    # Draw text
    for i, line in enumerate(text_lines):
        y = 5 + i * 22
        draw.text((5, y), line, fill="white")
    
    # Save
    img.save(output_path, "JPEG", quality=85)
    return True


def create_contact_sheet(
    rows: List[GeometryRow],
    source_images: dict,
    output_path: Path,
    base_path: Path,
    max_rows: int = 30,
) -> bool:
    """
    Create a contact sheet of worst rows.
    
    Args:
        rows: List of suspicious rows sorted by temp diff (worst first)
        source_images: Dict mapping image_path to PIL Image
        output_path: Path to save contact sheet
        max_rows: Maximum number of rows to include
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    
    # Take top rows
    top_rows = rows[:max_rows]
    
    # Thumbnail size
    thumb_width = 200
    thumb_height = 200
    cols = 5
    rows_count = (len(top_rows) + cols - 1) // cols
    
    # Contact sheet dimensions
    sheet_width = cols * thumb_width
    sheet_height = rows_count * thumb_height + 100  # Extra space for title
    
    # Create sheet
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    
    # Add title
    title = "Worst 30 Geometry Label Mismatches"
    draw.text((10, 10), title, fill="white")
    draw.text((10, 30), f"Sorted by absolute temperature difference", fill="lightgray")
    
    # Add thumbnails
    for i, row in enumerate(top_rows):
        col = i % cols
        row_idx = i // cols
        
        x_offset = col * thumb_width
        y_offset = row_idx * thumb_height + 100
        
        # Load or get cached image
        source_path = base_path / row.image_path
        if row.image_path not in source_images:
            source_images[row.image_path] = load_image(source_path)
        
        source_image = source_images[row.image_path]
        if source_image is None:
            # Draw placeholder
            draw.rectangle(
                [(x_offset, y_offset), (x_offset + thumb_width, y_offset + thumb_height)],
                outline="red",
                width=2,
            )
            draw.text((x_offset + 10, y_offset + 90), "MISSING", fill="red")
            continue
        
        # Resize to thumbnail
        thumb = source_image.resize((thumb_width, thumb_height), Image.LANCZOS)
        
        # Draw on sheet
        sheet.paste(thumb, (x_offset, y_offset))
        
        # Draw crop box scaled to thumbnail
        scale_x = thumb_width / row.source_width
        scale_y = thumb_height / row.source_height
        
        crop_x1 = row.loose_crop_x1 * scale_x
        crop_y1 = row.loose_crop_y1 * scale_y
        crop_x2 = row.loose_crop_x2 * scale_x
        crop_y2 = row.loose_crop_y2 * scale_y
        
        thumb_draw = ImageDraw.Draw(thumb)
        thumb_draw.rectangle(
            [(crop_x1, crop_y1), (crop_x2, crop_y2)],
            outline="yellow",
            width=1,
        )
        
        # Add text below thumbnail
        image_name = Path(row.image_path).name[:18]
        draw.text(
            (x_offset + 5, y_offset + thumb_height + 5),
            f"{image_name}",
            fill="lightgray",
        )
        diff_color = "red" if row.absolute_temperature_difference_c > 10 else "orange"
        draw.text(
            (x_offset + 5, y_offset + thumb_height + 20),
            f"Diff: {row.absolute_temperature_difference_c:.1f}C",
            fill=diff_color,
        )
        draw.text(
            (x_offset + 5, y_offset + thumb_height + 35),
            f"Flag: {row.quality_flag}",
            fill="white",
        )
    
    # Save
    sheet.save(output_path, "JPEG", quality=85)
    return True


def main() -> None:
    """Main entry point."""
    base_path = Path(__file__).parent.parent.parent
    manifest_path = base_path / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
    output_dir = base_path / "ml" / "debug" / "geometry_label_quality_v2"
    
    print(f"Loading v2_clean manifest from: {manifest_path}")
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run analyze_geometry_label_quality.py first.")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    rows = load_clean_manifest(manifest_path)
    print(f"Loaded {len(rows)} rows")
    
    # Filter suspicious rows
    suspicious_rows = [r for r in rows if r.quality_flag != "clean"]
    suspicious_rows.sort(key=lambda r: r.absolute_temperature_difference_c, reverse=True)
    
    print(f"Found {len(suspicious_rows)} suspicious rows")
    print(f"  Exclude: {sum(1 for r in suspicious_rows if r.quality_flag == 'exclude')}")
    print(f"  Review: {sum(1 for r in suspicious_rows if r.quality_flag == 'review')}")
    
    # Generate overlays for suspicious rows
    print("\nGenerating overlay images...")
    overlay_count = 0
    source_images_cache = {}
    
    for i, row in enumerate(suspicious_rows):
        source_path = base_path / row.image_path
        
        # Load image
        if row.image_path not in source_images_cache:
            source_images_cache[row.image_path] = load_image(source_path)
        
        source_image = source_images_cache[row.image_path]
        
        if source_image is None:
            print(f"  WARNING: Image not found: {row.image_path}")
            continue
        
        # Create overlay
        image_name = Path(row.image_path).stem
        output_path = images_dir / f"suspicious_{i:03d}_{row.quality_flag}_{image_name}.jpg"
        
        if create_suspicious_overlay(row, source_image, output_path, base_path):
            overlay_count += 1
    
    print(f"Created {overlay_count} overlay images")
    
    # Create contact sheet of worst 30
    print("\nCreating contact sheet of worst 30 rows...")
    contact_sheet_path = output_dir / "worst_30_contact_sheet.jpg"
    
    if create_contact_sheet(suspicious_rows, source_images_cache, contact_sheet_path, base_path):
        print(f"Saved contact sheet: {contact_sheet_path}")
    else:
        print("ERROR: Failed to create contact sheet")
    
    print("\nDone!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
