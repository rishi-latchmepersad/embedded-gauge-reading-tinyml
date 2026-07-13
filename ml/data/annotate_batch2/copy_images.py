#!/usr/bin/env python3
"""Copy 30 diverse images to annotate_batch2."""

import shutil
import os

# List of 30 diverse images (different dates, times, conditions)
SELECTED = [
    "capture_2026-04-03_11-44-35.png",
    "capture_2026-04-09_06-28-44.png",
    "capture_2026-04-15_11-54-49.png",
    "capture_2026-04-18_17-26-51.png",
    "capture_2026-04-19_13-05-42.png",
    "capture_2026-04-19_13-14-12.png",
    "capture_2026-04-19_13-22-41.png",
    "capture_2026-04-19_13-42-07.png",
    "capture_2026-04-19_16-46-57.png",
    "capture_2026-04-19_17-02-46.png",
    "capture_2026-04-19_18-43-21.png",
    "capture_2026-04-19_18-44-34.png",
    "capture_2026-04-19_18-54-51.png",
    "capture_2026-04-19_19-09-07.png",
    "capture_2026-04-19_19-10-20.png",
    "capture_2026-04-20_16-02-50.png",
    "capture_2026-04-22_07-16-50.png",
    "capture_2026-04-22_07-21-44.png",
    "capture_2026-04-22_07-33-19.png",
    "capture_2026-04-22_07-41-14.png",
    "capture_2026-04-24_22-25-14.png",
    "capture_2026-04-30_07-00-09.png",
    "capture_2026-04-30_13-08-26.png",
    "capture_2026-04-30_14-05-09.png",
    "capture_2026-04-30_14-05-49.png",
    "capture_2026-04-30_19-02-45.png",
    "capture_2026-04-30_20-09-44.png",
    "capture_2026-04-30_20-10-56.png",
    "capture_2026-04-30_20-12-09.png",
    "capture_2026-05-27_06-20-11.png",
]

src_dir = "/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/data/captured_images"
dst_dir = "/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp/annotate_batch2/images"

os.makedirs(dst_dir, exist_ok=True)

for fname in SELECTED:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  {fname}")
    else:
        print(f"  MISSING: {fname}")

print(f"\nDone! Copied files to {dst_dir}")

# Write image list for annotator
with open("/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp/annotate_batch2/image_list.txt", "w") as f:
    for fname in SELECTED:
        f.write(fname + "\n")

print(f"Image list written to annotate_batch2/image_list.txt")
