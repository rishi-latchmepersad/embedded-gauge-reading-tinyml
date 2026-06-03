import cv2
from pathlib import Path
import csv

project_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
manifest = project_root / "ml" / "data" / "ai_annotated_centers.csv"
out_dir = project_root / "tmp" / "center_corrected"
out_dir.mkdir(parents=True, exist_ok=True)

with open(manifest) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Test correction on all known captures + a few randoms
for row in rows:
    if any(k in row["image_path"] for k in ['capture_m19c.png', 'capture_0c.png', 'capture_p45c.png',
         'capture_2026-04-03_08-20-49.png', 'capture_2026-04-22_07-15-36.png']):
        img_path = project_root / "ml" / "data" / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        cx = int(float(row["center_x"]))
        cy = int(float(row["center_y"]))
        
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy-20), 4, (255, 0, 0), -1)
        
        out_path = out_dir / f"corr_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path.name}")
