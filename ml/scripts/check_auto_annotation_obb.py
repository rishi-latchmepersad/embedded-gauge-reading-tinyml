"""Visually check the recovered CVAT auto-annotation OBB on LittleGood frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
DEFAULT_MODEL = Path("/mnt/c/Users/rishi_latchmepersad/Cloud-Drive/embedded-gauge-reading-tinyml/ml/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite")
OUTPUT = ROOT / "artifacts" / "auto_annotation_obb_littlegood_check_v1"


def load_interpreter(path: Path) -> tuple[tf.lite.Interpreter, dict, dict]:
    """Load the int8 six-parameter OBB model."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def predict(interpreter: tf.lite.Interpreter, input_detail: dict, output_detail: dict, image: Image.Image) -> np.ndarray:
    """Run one RGB frame through the affine int8 contract."""
    resized = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    input_scale, input_zero = input_detail["quantization"]
    quantized = np.clip(np.round(resized / input_scale + input_zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_detail["index"], quantized[None])
    interpreter.invoke()
    output_scale, output_zero = output_detail["quantization"]
    raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)[0]
    return (raw - output_zero) * output_scale


def corners(params: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized [cx,cy,w,h,cos2t,sin2t] into full-frame corners."""
    cx, cy, box_w, box_h, cos2t, sin2t = params
    angle = 0.5 * math.atan2(float(sin2t), float(cos2t))
    c, s = math.cos(angle), math.sin(angle)
    local = np.asarray([[-box_w / 2, -box_h / 2], [box_w / 2, -box_h / 2], [box_w / 2, box_h / 2], [-box_w / 2, box_h / 2]], dtype=np.float32)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (local @ rotation.T + np.asarray([cx, cy])) * np.asarray([width, height])


def target_box(label_path: Path, width: int, height: int) -> np.ndarray:
    """Read the labeled axis-aligned gauge-face box for visual comparison."""
    values = np.fromstring(label_path.read_text(), sep=" ")
    points = values[1:9].reshape(4, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    return np.asarray([[low[0] * width, low[1] * height], [high[0] * width, low[1] * height], [high[0] * width, high[1] * height], [low[0] * width, high[1] * height]], dtype=np.float32)


def main() -> None:
    """Render six deterministic random LittleGood OBB overlays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = sorted((args.data / "images" / "test").glob("*.png"))
    selected = np.random.default_rng(42).choice(rows, size=min(6, len(rows)), replace=False)
    interpreter, input_detail, output_detail = load_interpreter(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    report: list[dict[str, object]] = []
    for image_path in selected:
        image = Image.open(image_path).convert("RGB")
        prediction = predict(interpreter, input_detail, output_detail, image)
        predicted = corners(prediction, *image.size)
        label_path = args.data / "labels" / "test" / f"{image_path.stem}.txt"
        target = target_box(label_path, *image.size)
        view = image.copy()
        draw = ImageDraw.Draw(view)
        draw.line([tuple(point) for point in target] + [tuple(target[0])], fill="lime", width=4)
        draw.line([tuple(point) for point in predicted] + [tuple(predicted[0])], fill="red", width=4)
        panel = Image.new("RGB", (640, 680), "black")
        panel.paste(view.resize((640, 640)), (0, 0))
        ImageDraw.Draw(panel).text((8, 650), image_path.stem[:82], fill="white")
        panels.append(panel)
        report.append({"stem": image_path.stem, "prediction": prediction.tolist(), "predicted_corners_px": predicted.tolist(), "target_corners_px": target.tolist()})
    sheet = Image.new("RGB", (1280, 2040), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 640, (index // 2) * 680))
    sheet.save(args.output / "contact_sheet.png")
    (args.output / "predictions.json").write_text(json.dumps(report, indent=2))
    print(args.output / "contact_sheet.png")


if __name__ == "__main__":
    main()
