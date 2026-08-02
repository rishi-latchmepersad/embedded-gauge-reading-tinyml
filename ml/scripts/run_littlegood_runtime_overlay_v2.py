"""Render the deployed ellipse-to-keypoint pipeline on labelled LittleGood frames."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from prepare_student_conditioned_littlegood import full_ellipse, local_points_to_full_640
from train_gauge_center_tip_fullframe_v1 import decode


ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_vector_littlegood_v1" / "gauge_ellipse_vector_v1_int8.tflite"
KEYPOINT_MODEL = Path(os.environ.get("OVERLAY_KEYPOINT_MODEL", str(ROOT / "artifacts" / "gauge_center_tip_direction_radius_littlegood_v1" / "gauge_center_tip_direction_radius_v1_int8.tflite")))
OUTPUT = Path(os.environ.get("OVERLAY_OUTPUT", str(ROOT / "artifacts" / "littlegood_runtime_overlay_v2")))
SIZE = 160
CROP_SCALE = float(os.environ.get("OVERLAY_CROP_SCALE", "1.35"))
RADIUS_SCALE = 0.5


def interpreter(path: Path) -> tuple[tf.lite.Interpreter, dict[str, object], list[dict[str, object]]]:
    """Load a full-int8 graph and expose its tensor descriptors."""
    runner = tf.lite.Interpreter(model_path=str(path))
    runner.allocate_tensors()
    return runner, runner.get_input_details()[0], runner.get_output_details()


def predict(runner: tf.lite.Interpreter, input_detail: dict[str, object], outputs: list[dict[str, object]], sample: np.ndarray) -> list[np.ndarray]:
    """Run one int8 sample and dequantize all model outputs."""
    scale, zero = input_detail["quantization"]
    encoded = np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]
    runner.set_tensor(input_detail["index"], encoded)
    runner.invoke()
    values = []
    for detail in outputs:
        raw = runner.get_tensor(detail["index"]).astype(np.float32)[0]
        scale, zero = detail["quantization"]
        values.append((raw - zero) * scale)
    return values


def predicted_ellipse(runner: tf.lite.Interpreter, input_detail: dict[str, object], output: dict[str, object], image: np.ndarray) -> np.ndarray:
    """Predict a 640-frame ellipse from the resized grayscale frame."""
    gray = np.asarray(Image.fromarray(image).convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    prediction = predict(runner, input_detail, [output], gray[..., None])[0]
    return np.clip(prediction, 0.02, 0.98) * 640.0


def keypoint_input(image: np.ndarray, ellipse: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Build the exact runtime crop and ellipse-mask channels."""
    cx, cy, rx, ry = ellipse
    side = max(2.0 * rx, 2.0 * ry) * CROP_SCALE
    left, top = float(cx - side / 2.0), float(cy - side / 2.0)
    source = Image.fromarray(image).convert("L").crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    gray = np.asarray(source, dtype=np.float32) / 255.0
    axis = (np.arange(SIZE, dtype=np.float32) + 0.5) / SIZE * side
    xx, yy = np.meshgrid(axis + left, axis + top)
    mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    return np.stack((gray * 2.0 - 1.0, mask * 2.0 - 1.0), axis=-1), (left, top, side)


def draw_point(draw: ImageDraw.ImageDraw, point: np.ndarray, color: str) -> None:
    """Draw a labelled or predicted point with a contrasting outline."""
    x, y = (float(point[0]), float(point[1]))
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=3)


def main() -> None:
    """Render six deterministic random test frames and write JSON evidence."""
    rows = json.loads((POINT_DATA / "metadata.json").read_text())['splits']['test']
    selected = np.random.default_rng(42).choice(rows, size=6, replace=False).tolist()
    ellipse_runner, ellipse_input_detail, ellipse_outputs = interpreter(ELLIPSE_MODEL)
    keypoint_runner, keypoint_input_detail, keypoint_outputs = interpreter(KEYPOINT_MODEL)
    ellipse_output = ellipse_outputs[0]
    heat_output = next(item for item in keypoint_outputs if len(item["shape"]) == 4)
    radius_output = next(item for item in keypoint_outputs if len(item["shape"]) == 2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels, evidence = [], []
    for row in selected:
        stem = str(row["stem"])
        image = np.asarray(Image.open(ELLIPSE_DATA / "images" / "test" / f"{stem}.png").convert("RGB"))
        ellipse = predicted_ellipse(ellipse_runner, ellipse_input_detail, ellipse_output, image)
        sample, (left, top, side) = keypoint_input(image, ellipse)
        outputs = predict(keypoint_runner, keypoint_input_detail, [heat_output, radius_output], sample)
        local = decode(outputs[0][None])[0]
        direction = local[1] - local[0]
        direction /= np.linalg.norm(direction) + 1e-6
        local_tip = local[0] + direction * float(outputs[1][0]) * RADIUS_SCALE
        prediction = np.asarray([local[0], local_tip]) * side + np.asarray([left, top])
        target = local_points_to_full_640(row, "test")
        target_ellipse = full_ellipse("test", stem)
        view = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(view)
        draw.ellipse((ellipse[0] - ellipse[2], ellipse[1] - ellipse[3], ellipse[0] + ellipse[2], ellipse[1] + ellipse[3]), outline="red", width=5)
        draw.ellipse((target_ellipse[0] - target_ellipse[2], target_ellipse[1] - target_ellipse[3], target_ellipse[0] + target_ellipse[2], target_ellipse[1] + target_ellipse[3]), outline="lime", width=3)
        draw_point(draw, target[0], "lime"); draw_point(draw, target[1], "cyan")
        draw_point(draw, prediction[0], "orange"); draw_point(draw, prediction[1], "magenta")
        panels.append(view.resize((320, 320), Image.Resampling.LANCZOS))
        evidence.append({"stem": stem, "predicted_ellipse_640": ellipse.tolist(), "target_ellipse_640": target_ellipse.tolist(), "target_points_640": target.tolist(), "predicted_points_640": prediction.tolist()})
    sheet = Image.new("RGB", (960, 640), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 320, (index // 3) * 320))
    sheet.save(OUTPUT / "contact_sheet.png")
    (OUTPUT / "predictions.json").write_text(json.dumps(evidence, indent=2))
    print(OUTPUT / "contact_sheet.png")


if __name__ == "__main__":
    main()
