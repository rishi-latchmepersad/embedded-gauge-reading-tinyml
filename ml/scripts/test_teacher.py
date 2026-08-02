"""Test the YOLOv11m teacher model on a sample image."""
import os
os.environ["DISABLE_TORCHVISION_NMS"] = "1"

from ultralytics import YOLO
from pathlib import Path

teacher = YOLO("/mnt/d/Projects/cvat/training/gaugeface_segmentation/runs/gaugeface_yolo11m_v2_finetune/weights/best.pt")
print("Teacher model loaded")

img_path = list(Path("data/gauge_face_ellipse_v1_640_gray/images/train").glob("*.png"))[0]
print(f"Test image: {img_path}")

result = teacher(str(img_path), verbose=False)
r = result[0]
print(f"Detections: {len(r.boxes)}")
for box in r.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
    print(f"  ellipse: cx={cx:.1f} cy={cy:.1f} rx={rx:.1f} ry={ry:.1f} conf={conf:.3f}")
