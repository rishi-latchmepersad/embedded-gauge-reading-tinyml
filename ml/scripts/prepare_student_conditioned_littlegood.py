"""Prepare int8-ellipse-conditioned LittleGood center/tip training arrays.

Each source image is evaluated once by the deployed ellipse graph.  The
annotated points are transformed into that predicted crop, so the keypoint
student learns the real detector error distribution without oversampling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FULL_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ELLIPSE = ROOT / "artifacts" / "gauge_ellipse_vector_littlegood_v1" / "gauge_ellipse_vector_v1_int8.tflite"
OUTPUT = Path(os.environ["STUDENT_OUTPUT"]) if "STUDENT_OUTPUT" in os.environ else ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
SIZE = 160
HEATMAP = 80
# why: the ellipse detector is intentionally tolerant, so the keypoint crop
# must retain the full needle even when the detector ellipse is offset.
CROP_SCALE = float(os.environ.get("STUDENT_CROP_SCALE", "1.35"))


def ellipse_predictor() -> tuple[tf.lite.Interpreter, dict[str, object], dict[str, object]]:
    """Load the full-int8 ellipse graph and its tensor descriptors."""
    interpreter = tf.lite.Interpreter(model_path=str(ELLIPSE))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def predict_ellipse(interpreter: tf.lite.Interpreter, inp: dict[str, object], out: dict[str, object], image: np.ndarray) -> np.ndarray:
    """Predict [cx, cy, rx, ry] in the 640-pixel source frame."""
    gray = np.asarray(Image.fromarray(image).convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    scale, zero = inp["quantization"]
    tensor = np.clip(np.round(gray[None, ..., None] / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], tensor)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"]).astype(np.float32)
    values = (raw - out["quantization"][1]) * out["quantization"][0]
    return np.clip(values[0], 0.02, 0.98) * 640.0


def crop_transform(ellipse: np.ndarray) -> tuple[float, float, float]:
    """Return crop left/top/side for the runtime ellipse-to-head transform."""
    cx, cy, rx, ry = ellipse
    side = max(2.0 * rx, 2.0 * ry) * CROP_SCALE
    return float(cx - side / 2.0), float(cy - side / 2.0), float(side)


def full_ellipse(split: str, stem: str) -> np.ndarray:
    """Read the authoritative full-frame ellipse label in 640-pixel units."""
    values = np.fromstring((FULL_DATA / "labels" / split / f"{stem}.txt").read_text(), sep=" ")
    points = values[1:9].reshape(4, 2) * 640.0
    low, high = points.min(axis=0), points.max(axis=0)
    return np.concatenate(((low + high) / 2.0, (high - low) / 2.0)).astype(np.float32)


def local_points_to_full_640(row: dict[str, object], split: str) -> np.ndarray:
    """Map points from the saved 1.18x source crop into the 640 frame.

    The center/tip metadata is normalized in the original crop, while the
    ellipse metadata remains in that crop's source-pixel coordinates.  It is
    therefore not valid to multiply the normalized points directly by 640.
    """
    fixed = np.asarray(row["ellipse"], dtype=np.float32)
    side = max(2.0 * fixed[2], 2.0 * fixed[3]) * 1.18
    left_top = fixed[:2] - side / 2.0
    local = np.asarray((row["center_xy_norm"], row["tip_xy_norm"]), dtype=np.float32)
    source_points = left_top + local * side
    target = full_ellipse(split, str(row["stem"]))
    # why: this affine map preserves the annotated point's offset from the
    # ellipse center while changing from source ellipse units to 640 units.
    return target[:2] + (source_points - fixed[:2]) * target[2:] / fixed[2:]


def make_sample(image: np.ndarray, ellipse: np.ndarray, points_640: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create normalized two-channel input and center/tip heatmap targets."""
    left, top, side = crop_transform(ellipse)
    source = Image.fromarray(image).convert("L").crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    gray = np.asarray(source, dtype=np.float32) / 255.0
    axis = (np.arange(SIZE, dtype=np.float32) + 0.5) / SIZE * side
    xx, yy = np.meshgrid(axis + left, axis + top)
    cx, cy, rx, ry = ellipse
    mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    sample = np.stack((gray * 2.0 - 1.0, mask * 2.0 - 1.0), axis=-1)
    local = np.clip((points_640 - np.asarray((left, top), dtype=np.float32)) / side, 0.0, 1.0)
    heatmaps = np.zeros((HEATMAP, HEATMAP, 2), dtype=np.float32)
    yyh, xxh = np.mgrid[0:HEATMAP, 0:HEATMAP]
    for channel, point in enumerate(local):
        px, py = point * HEATMAP - 0.5
        heatmaps[..., channel] = np.exp(-((xxh - px) ** 2 + (yyh - py) ** 2) / (2.0 * 2.2**2))
    return sample, heatmaps, local.astype(np.float32)


def main() -> None:
    """Generate one NPZ per split with no source-image duplication."""
    predictor, inp, out = ellipse_predictor()
    metadata = json.loads((POINT_DATA / "metadata.json").read_text())["splits"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, rows in metadata.items():
        samples, targets, points, ellipses = [], [], [], []
        for row in rows:
            image_path = FULL_DATA / "images" / split / f"{row['stem']}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            predicted = predict_ellipse(predictor, inp, out, image)
            points_640 = local_points_to_full_640(row, split)
            sample, heatmaps, local = make_sample(image, predicted, points_640)
            samples.append(sample); targets.append(heatmaps); points.append(local); ellipses.append(predicted)
        np.savez_compressed(OUTPUT / f"{split}.npz", inputs=np.stack(samples), heatmaps=np.stack(targets), points=np.stack(points), ellipses=np.stack(ellipses))
        print(split, len(samples), "written")


if __name__ == "__main__":
    main()
