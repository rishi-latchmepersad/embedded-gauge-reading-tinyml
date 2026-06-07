"""
Visualise YOLO11n-OBB predictions on the validation set.
Saves overlay images to artifacts/yolo_obb_320/val_viz/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "yolo_obb_320"
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "yolo_obb_320" / "train" / "weights" / "best.pt"
VIZ_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "yolo_obb_320" / "val_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

VAL_IMAGES = sorted((DATA_DIR / "images" / "val").glob("*.jpg"))
VAL_LABELS = DATA_DIR / "labels" / "val"


def load_obb_label(txt_path: Path) -> np.ndarray | None:
    """Load YOLO OBB label: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    if not txt_path.exists():
        return None
    vals = np.loadtxt(txt_path)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


def obb_to_corners(obb: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalized OBB coords to pixel coords."""
    pts = obb[1:].reshape(4, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    return pts.astype(np.int32)


def main() -> None:
    model = YOLO(str(MODEL_PATH))

    for img_path in VAL_IMAGES:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        gt_obbs = load_obb_label(label_path)
        if gt_obbs is not None:
            for obb in gt_obbs:
                pts = obb_to_corners(obb, w, h)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # Prediction
        results = model.predict(str(img_path), imgsz=320, conf=0.25, verbose=False)
        if results[0].obb is not None:
            pred_xyxyxyxyn = results[0].obb.xyxyxyxyn.cpu().numpy()
            for pts_norm in pred_xyxyxyxyn:
                pts = (pts_norm * np.array([w, h])).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)

        out_path = VIZ_DIR / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"Saved {len(VAL_IMAGES)} viz images to {VIZ_DIR}")


if __name__ == "__main__":
    main()
