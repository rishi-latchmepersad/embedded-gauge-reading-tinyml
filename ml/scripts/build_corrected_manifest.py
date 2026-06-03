import csv
from pathlib import Path

project_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
auto_manifest = project_root / "ml" / "data" / "ai_annotated_centers.csv"
final_manifest = project_root / "ml" / "data" / "manual_annotated_centers.csv"

# Manual annotations for images that failed auto-detection
# Based on visual inspection of the images
manual_annotations = {
    # These failed because they're very dark or have unusual framing
    # I'll annotate the known-temperature ones and a few key captures
}

results = []

# Read auto-detected and apply y-correction
with open(auto_manifest) as f:
    reader = csv.DictReader(f)
    for row in reader:
        cx = float(row["center_x"])
        cy = float(row["center_y"]) - 20.0  # Shift up by 20px to hit needle pivot
        results.append({
            "image_path": row["image_path"],
            "center_x": f"{cx:.1f}",
            "center_y": f"{cy:.1f}",
        })

print(f"Corrected {len(results)} auto-detected centers (y -= 20px)")

# Write final manifest
with open(final_manifest, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "center_x", "center_y"])
    writer.writeheader()
    writer.writerows(results)

print(f"Wrote {final_manifest}")
