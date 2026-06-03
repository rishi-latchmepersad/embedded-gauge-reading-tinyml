import json
from pathlib import Path
import cv2
import numpy as np

project_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
metadata_path = project_root / "ml" / "data" / "center_training_manual" / "metadata.json"
img_dir = project_root / "ml" / "data" / "center_training_manual" / "images"
out_dir = project_root / "tmp" / "training_verify"
out_dir.mkdir(parents=True, exist_ok=True)

with open(metadata_path) as f:
    entries = json.load(f)

# Draw centers on a representative sample
sample = [
    'capture_m18c.png',
    'capture_m19c.png', 
    'capture_0c.png',
    'capture_p35c.png',
    'capture_p45c.png',
    'capture_2026-04-03_08-20-49.png',
    'capture_2026-04-22_07-15-36.png',
    'capture_2026-04-22_07-29-38.png',
]

for e in entries:
    if e["image"] in sample:
        img_path = img_dir / e["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        cx = int(e["center_x"] * 224)
        cy = int(e["center_y"] * 224)
        
        # Draw center
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.line(img, (cx-12, cy), (cx+12, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy-12), (cx, cy+12), (0, 255, 0), 2)
        
        # Label
        cv2.putText(img, f"({e['center_x']:.2f},{e['center_y']:.2f})", 
                    (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        out_path = out_dir / f"verify_{e['image']}"
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path.name} center=({e['center_x']:.3f},{e['center_y']:.3f})")
