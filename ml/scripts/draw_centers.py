import cv2
from pathlib import Path
import csv

project_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
manifest = project_root / "ml" / "data" / "ai_annotated_centers.csv"
out_dir = project_root / "tmp" / "center_debug"
out_dir.mkdir(parents=True, exist_ok=True)

with open(manifest) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Draw centers on known captures
known = ['capture_m18c.png', 'capture_m19c.png', 'capture_0c.png', 
         'capture_p35c.png', 'capture_p45c.png']
for row in rows:
    if any(k in row["image_path"] for k in known):
        img_path = project_root / "ml" / "data" / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cx = int(float(row["center_x"]))
        cy = int(float(row["center_y"]))
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.line(img, (cx-12, cy), (cx+12, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy-12), (cx, cy+12), (0, 255, 0), 2)
        out_path = out_dir / f"debug_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path.name} center=({cx},{cy})")
