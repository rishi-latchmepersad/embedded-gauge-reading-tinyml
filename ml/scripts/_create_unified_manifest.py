"""Create a unified training manifest from all available data sources."""

import csv
from pathlib import Path
from collections import Counter

repo_root = Path("d:/Projects/embedded-gauge-reading-tinyml")

# Source manifests
labelled_manifest = repo_root / "ml/data/full_labelled_plus_board30_valid_with_new5.csv"
all_captured = repo_root / "ml/data/all_captured_images_manifest.csv"
hard_cases = repo_root / "ml/data/hard_cases_plus_board30_valid_with_new6.csv"

all_rows = []
seen_paths = set()

# Add labelled data (most comprehensive)
with open(labelled_manifest) as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row["image_path"]
        if path not in seen_paths:
            all_rows.append((path, float(row["value"])))
            seen_paths.add(path)

print(f"After labelled data: {len(all_rows)} rows")

# Add hard cases
with open(hard_cases) as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row["image_path"]
        if not path.startswith("#") and path not in seen_paths:
            all_rows.append((path, float(row["value"])))
            seen_paths.add(path)

print(f"After hard cases: {len(all_rows)} rows")

# Add captured images
with open(all_captured) as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row["image_path"]
        if not path.startswith("#") and path not in seen_paths:
            full_path = repo_root / path
            if full_path.exists():
                all_rows.append((path, float(row["value"])))
                seen_paths.add(path)

print(f"After captured images: {len(all_rows)} rows")

# Write unified manifest
output_path = repo_root / "ml/data/unified_training_manifest_v1.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "value"])
    for path, value in all_rows:
        writer.writerow([path, value])

print(f"Wrote unified manifest to: {output_path}")
print(f"Total unique images: {len(all_rows)}")

# Show distribution
values = [v for _, v in all_rows]
counts = Counter(values)
print("Value distribution:")
for v in sorted(counts.keys()):
    print(f"  {v:5.1f}C: {counts[v]:3d} images")
