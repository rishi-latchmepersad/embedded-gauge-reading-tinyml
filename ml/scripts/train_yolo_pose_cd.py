"""
Train YOLO11n-pose for gauge center detection on rectified 320×320 crops.

Pipeline:
  1. Load YOLO pose dataset (prepared by prepare_yolo_pose_dataset.py)
  2. Train YOLO11n-pose with 1 keypoint (gauge_center)
  3. Evaluate keypoint accuracy
  4. Export TFLite int8

Usage:
  python scripts/train_yolo_pose_cd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320" / "yolo_pose"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "yolo_pose_cd"

SEED = 42
IMG_SIZE = 320
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 30
LR = 0.001


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    # Load pretrained YOLO11n-pose
    model = YOLO("yolo11n-pose.pt")

    # Train
    results = model.train(
        data=str(DATA_DIR / "dataset.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LR,
        lrf=0.01,
        patience=PATIENCE,
        seed=SEED,
        project=str(ARTIFACT_DIR.parent),
        name=ARTIFACT_DIR.name,
        exist_ok=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=False,  # avoid AMP issues with small keypoint
        workers=4,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=10,
        val=True,
        save=True,
        save_period=50,
    )

    best_pt = ARTIFACT_DIR / "weights" / "best.pt"
    print(f"\nBest model: {best_pt}")

    # Evaluate keypoint accuracy in pixels
    from ultralytics.utils.metrics import ConfusionMatrix
    val_results = model.val(data=str(DATA_DIR / "dataset.yaml"))
    print(f"\nValidation results:")
    for k, v in val_results.results_dict.items():
        print(f"  {k}: {v:.4f}")

    # Export to TFLite int8
    print("\n=== Exporting TFLite ===")
    tflite_path = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        int8=True,
        data=str(DATA_DIR / "dataset.yaml"),
    )
    print(f"  TFLite saved to: {tflite_path}")

    # Also export float32 tflite
    f32_path = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        int8=False,
    )
    print(f"  TFLite f32 saved to: {f32_path}")

    # Validate TFLite model on val set
    print("\n=== Validating TFLite on val set ===")
    import tensorflow as tf
    import json
    import numpy as np
    import math

    meta = json.loads(
        (DATA_DIR.parent / "metadata.json").read_text()
    )
    val_samples = meta["samples"]["val"]

    interp = tf.lite.Interpreter(str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    # YOLO TFLite output format: [1, num_kpts*3, 8400] for pose
    # Or [1, 56, 8400] for 1 keypoint (x, y, vis)
    print(f"  Input:  {in_det['shape']} {in_det['dtype']}")
    print(f"  Output: {out_det['shape']} {out_det['dtype']}")

    err_px = []
    for s in val_samples:
        img_path = DATA_DIR.parent / "images" / "val" / f"{s['stem']}.jpg"
        img = tf.io.decode_jpeg(tf.io.read_file(str(img_path)), channels=3)

        # YOLO expects uint8 [0, 255] input
        u8 = tf.cast(img, tf.uint8).numpy()
        interp.set_tensor(in_det["index"], u8[None, ...])
        interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]  # (56, 8400) for 1 kpt

        # Parse keypoint output
        # YOLO pose output: first 4 rows are bbox, next rows are kpt x,y,vis
        # For 1 keypoint: rows 4,5 = x,y; row 6 = vis
        kpt_x = out[4, :]
        kpt_y = out[5, :]
        kpt_v = out[6, :]

        # Pick the highest-confidence detection's keypoint
        # Objectness is at row 4+3*nkpts = row 7
        obj = out[7, :] if out.shape[0] > 7 else out[4 + 3, :]
        best_idx = int(np.argmax(kpt_v * obj))  # vis * obj score

        cx_pred = float(kpt_x[best_idx])  # already in [0, 320]
        cy_pred = float(kpt_y[best_idx])
        cx_gt, cy_gt = s["center_xy_rectified"]
        err = math.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)
        err_px.append(err)

    err_arr = np.array(err_px)
    print(f"\n  Center error on {len(err_arr)} val samples:")
    print(f"    Mean:   {err_arr.mean():.2f} px")
    print(f"    Median: {np.median(err_arr):.2f} px")
    print(f"    Std:    {err_arr.std():.2f} px")

    # Save eval results
    results_dict = dict(val_results.results_dict)
    results_dict["center_error_px_mean"] = float(f"{err_arr.mean():.3f}")
    results_dict["center_error_px_median"] = float(f"{np.median(err_arr):.3f}")
    import json as j
    (ARTIFACT_DIR / "eval_results.json").write_text(j.dumps(results_dict, indent=2))
    print(f"\nResults saved to {ARTIFACT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
