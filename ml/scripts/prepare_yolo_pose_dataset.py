"""
Prepare a YOLO pose dataset for gauge center detection on rectified 320×320 crops.

Uses the existing heatmap_cd_320 rectified crops + center annotations.
Converts to YOLO pose format with 1 keypoint (gauge_center).

Output layout:
  ml/data/heatmap_cd_320/yolo_pose/
    images/train/*.jpg      (symlinks to ../images/train/)
    images/val/*.jpg
    labels/train/*.txt      (YOLO pose label files)
    labels/val/*.txt
    dataset.yaml            (Ultralytics dataset config)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"
OUT_DIR = DATA_DIR / "yolo_pose"


def main() -> None:
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    for split in ("train", "val"):
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for s in meta["samples"][split]:
            stem = s["stem"]
            cx_norm, cy_norm = s["center_xy_norm"]

            # Symlink the image
            src_img = DATA_DIR / "images" / split / f"{stem}.jpg"
            dst_img = img_out / f"{stem}.jpg"
            if not dst_img.exists():
                dst_img.symlink_to(src_img)

            # Write YOLO pose label: class_id cx cy w h kp1_x kp1_y kp1_vis
            # Bbox = full image (gauge fills the rectified crop)
            bbox_cx, bbox_cy, bbox_w, bbox_h = 0.5, 0.5, 1.0, 1.0
            line = f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f} {cx_norm:.6f} {cy_norm:.6f} 1\n"
            (lbl_out / f"{stem}.txt").write_text(line)

    # Write dataset.yaml
    yaml_content = f"""
path: {OUT_DIR}
train: images/train
val: images/val

nc: 1
names: ['gauge_face']
kpt_shape: [1, 3]
"""
    (OUT_DIR / "dataset.yaml").write_text(yaml_content)

    # Count stats
    train_count = len(list((OUT_DIR / "images" / "train").iterdir()))
    val_count = len(list((OUT_DIR / "images" / "val").iterdir()))
    print(f"YOLO pose dataset ready: {train_count} train, {val_count} val")
    print(f"  Path: {OUT_DIR}")
    print(f"  Config: {OUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
