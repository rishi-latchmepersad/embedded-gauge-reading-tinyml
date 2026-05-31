#!/usr/bin/env python3
"""Label board captures with temperature/angle ground truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

# Gauge constants
COLD_END_ANGLE = 135.0
SWEEP_ANGLE = 270.0
MIN_TEMP = -30.0
MAX_TEMP = 50.0


def find_board_captures(base_dir: Path):
    """Find all board capture directories with report.json."""
    probe_dir = base_dir / "_live_rectified_probe"
    captures = []
    
    for subdir in sorted(probe_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("capture_"):
            report_path = subdir / "report.json"
            crop_path = subdir / "board_crop.png"
            if report_path.exists() and crop_path.exists():
                captures.append({
                    "dir": subdir,
                    "report": report_path,
                    "crop": crop_path,
                })
    
    return captures


def label_interactive(captures: list[dict], output_csv: Path):
    """Interactive labeling interface."""
    print(f"Found {len(captures)} board captures to label")
    print("Commands: <temp> (e.g., 25.0), 's' skip, 'q' quit")
    print("=" * 80)
    
    labeled = []
    skipped = []
    
    # Load existing labels if any
    if output_csv.exists():
        with open(output_csv) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    labeled.append({
                        "image": parts[0],
                        "temp": float(parts[1]),
                        "cx": float(parts[2]),
                        "cy": float(parts[3]),
                        "tx": float(parts[4]),
                        "ty": float(parts[5]) if len(parts) > 5 else None,
                    })
        print(f"Loaded {len(labeled)} existing labels")
    
    for i, cap in enumerate(captures):
        img_name = cap["crop"].name
        if any(l["image"] == img_name for l in labeled):
            print(f"[{i+1}/{len(captures)}] {img_name} - ALREADY LABELED, skipping")
            continue
        
        # Load and display image
        img = Image.open(cap["crop"])
        img_arr = np.asarray(img)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_arr, cmap="gray" if len(img_arr.shape) == 2 else None)
        plt.title(f"{img_name} ({i+1}/{len(captures)})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        while True:
            try:
                user_input = input(f"Temperature °C (or s/q): ").strip()
            except EOFError:
                break
            
            if user_input.lower() == "q":
                print("Quitting...")
                break
            elif user_input.lower() == "s":
                print(f"Skipped {img_name}")
                skipped.append(img_name)
                break
            else:
                try:
                    temp = float(user_input)
                    if temp < MIN_TEMP or temp > MAX_TEMP:
                        print(f"Warning: {temp}°C outside valid range [{MIN_TEMP}, {MAX_TEMP}]")
                        confirm = input("Continue? (y/n): ").strip().lower()
                        if confirm != "y":
                            continue
                    
                    # Calculate angle from temperature
                    # temp = min_temp + (angle - cold_end) / sweep * (max_temp - min_temp)
                    # angle = cold_end + (temp - min_temp) / (max_temp - min_temp) * sweep
                    angle = COLD_END_ANGLE + (temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP) * SWEEP_ANGLE
                    
                    print(f"  Temperature: {temp}°C → Angle: {angle:.1f}°")
                    print(f"  Note: You'll need to manually mark center and tip coordinates")
                    
                    # For now, just save temperature - coordinates will be added later
                    labeled.append({
                        "image": img_name,
                        "temp": temp,
                        "angle": angle,
                        "cx": None,
                        "cy": None,
                        "tx": None,
                        "ty": None,
                    })
                    
                    # Save progress
                    with open(output_csv, "w") as f:
                        f.write("image,temp_c,angle_deg,center_x,center_y,tip_x,tip_y\n")
                        for l in labeled:
                            f.write(f"{l['image']},{l['temp']},{l.get('angle', '')},{l.get('cx', '')},{l.get('cy', '')},{l.get('tx', '')},{l.get('ty', '')}\n")
                    
                    break
                except ValueError:
                    print("Invalid input. Enter a number (e.g., 25.0)")
        
        plt.close()
    
    print(f"\nLabeled {len(labeled)} images, skipped {len(skipped)}")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("ml/data/captured_images"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/board_labels.csv"))
    args = parser.parse_args()
    
    captures = find_board_captures(args.base_dir)
    label_interactive(captures, args.output)
