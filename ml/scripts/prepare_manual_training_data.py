"""Prepare training data using manually annotated centers for board captures."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
CAPTURED_DIR = PROJECT_ROOT / "ml" / "data" / "captured_images"
MANIFEST = PROJECT_ROOT / "ml" / "data" / "manual_annotated_centers.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "center_training_manual"
OBB_MODEL = PROJECT_ROOT / "ml" / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"

INPUT_SIZE = 224
TC_W = 155
TC_H = 123


def load_obb_model():
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(OBB_MODEL))
    interp.allocate_tensors()
    return interp


def run_obb(interp, frame):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale = float(inp['quantization'][0])
    zp = int(inp['quantization'][1])
    batch = (frame.astype(np.float32) / 255.0)[None, ...]
    q = np.clip(np.round(batch / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(int(inp['index']), q)
    interp.invoke()
    qo = interp.get_tensor(int(out['index']))[0]
    oscale = float(out['quantization'][0])
    ozp = int(out['quantization'][1])
    return oscale * (np.asarray(qo, dtype=np.float32) - ozp)


def resize_with_pad_geom(crop_w, crop_h):
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    return scale, (INPUT_SIZE - crop_w * scale) * 0.5, (INPUT_SIZE - crop_h * scale) * 0.5


def frame_to_canvas_center(ff_cx, ff_cy, cd_x, cd_y, cd_w, cd_h):
    """Convert full-frame center to CD-crop-resized-pad canvas normalized coords."""
    scale, pad_x, pad_y = resize_with_pad_geom(cd_w, cd_h)
    # Reverse: ff_cx = cd_x + (canvas_cx - pad_x) / scale
    # So: canvas_cx = pad_x + (ff_cx - cd_x) * scale
    canvas_cx = pad_x + (ff_cx - cd_x) * scale
    canvas_cy = pad_y + (ff_cy - cd_y) * scale
    return canvas_cx / INPUT_SIZE, canvas_cy / INPUT_SIZE


def main():
    import tensorflow as tf
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    
    # Load manual annotations
    annotations = {}
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations[row["image_path"]] = (float(row["center_x"]), float(row["center_y"]))
    
    print(f"Loaded {len(annotations)} manual annotations")
    
    obb_interp = load_obb_model()
    entries = []
    
    for img_rel_path, (ff_cx, ff_cy) in annotations.items():
        img_path = PROJECT_ROOT / "ml" / "data" / img_rel_path
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        
        # Run OBB
        obb_out = run_obb(obb_interp, frame)
        obb_cx = float(obb_out[0]) * INPUT_SIZE
        obb_cy = float(obb_out[1]) * INPUT_SIZE
        
        # CD crop
        cd_x = int(max(0, min(obb_cx - TC_W / 2, INPUT_SIZE - TC_W)))
        cd_y = int(max(0, min(obb_cy - TC_H / 2, INPUT_SIZE - TC_H)))
        cd_crop = frame[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]
        
        # Resize with pad
        scale, pad_x, pad_y = resize_with_pad_geom(TC_W, TC_H)
        rw = int(round(TC_W * scale))
        rh = int(round(TC_H * scale))
        cd_resized = cv2.resize(cd_crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        x_off = int(round(pad_x))
        y_off = int(round(pad_y))
        canvas[y_off:y_off + rh, x_off:x_off + rw] = cd_resized
        
        # Convert full-frame center to canvas normalized coords
        cx_norm, cy_norm = frame_to_canvas_center(ff_cx, ff_cy, cd_x, cd_y, TC_W, TC_H)
        
        # Save
        out_name = img_path.stem + ".png"
        out_path = OUTPUT_DIR / "images" / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        entries.append({
            "image": out_name,
            "center_x": cx_norm,
            "center_y": cy_norm,
            "source": "manual_annotation",
        })
    
    # Write metadata
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(entries, f, indent=2)
    
    print(f"Generated {len(entries)} training examples in {OUTPUT_DIR}")
    print(f"Canvas center range: x=[{min(e['center_x'] for e in entries):.3f}, {max(e['center_x'] for e in entries):.3f}]")
    print(f"Canvas center range: y=[{min(e['center_y'] for e in entries):.3f}, {max(e['center_y'] for e in entries):.3f}]")


if __name__ == "__main__":
    main()
