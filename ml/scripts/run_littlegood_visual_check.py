"""Run the retrained int8 models on held-out LittleGood gauges and draw overlays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
CENTER_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
OUTPUT = ROOT / "artifacts" / "littlegood_visual_check_fusion_v1"


def _interpreter(path: Path) -> tuple[tf.lite.Interpreter, dict, list[dict]]:
    """Load an int8 interpreter and return input/output tensor metadata."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()


def _predict(interpreter: tf.lite.Interpreter, input_detail: dict, output_details: list[dict], sample: np.ndarray) -> list[np.ndarray]:
    """Run one sample and dequantize every output tensor."""
    input_scale, input_zero = input_detail["quantization"]
    quantized = np.clip(np.round(sample / input_scale + input_zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_detail["index"], quantized[None])
    interpreter.invoke()
    predictions = []
    for output_detail in output_details:
        output_scale, output_zero = output_detail["quantization"]
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        predictions.append((raw[0] - output_zero) * output_scale)
    return predictions


def _ellipse_input(path: Path) -> np.ndarray:
    """Decode one prepared 640x640 grayscale ellipse input."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)[..., None] / 255.0


def _center_input(path: Path, row: dict[str, object]) -> np.ndarray:
    """Build the two-channel center/tip input used by training."""
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    cx, cy, rx, ry = np.asarray(row["ellipse"], dtype=np.float32)
    side = max(2.0 * rx, 2.0 * ry) * 1.18
    x = (np.arange(160, dtype=np.float32) + 0.5) / 160.0 * side + cx - side / 2.0
    y = (np.arange(160, dtype=np.float32) + 0.5) / 160.0 * side + cy - side / 2.0
    xx, yy = np.meshgrid(x, y)
    mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    return np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)


def _decode_ellipse(mask: np.ndarray) -> np.ndarray:
    """Decode a weighted 80x80 mask into normalized center and radii."""
    yy, xx = np.mgrid[0:80, 0:80].astype(np.float32)
    weight = np.maximum(mask[..., 0] - 0.20, 0.0)
    total = float(weight.sum())
    if total <= 1e-6:
        raise ValueError("Ellipse mask has no usable confidence mass")
    cx = float((weight * xx).sum() / total + 0.5) / 80.0
    cy = float((weight * yy).sum() / total + 0.5) / 80.0
    var_x = float((weight * (xx - cx * 80.0 + 0.5) ** 2).sum() / total)
    var_y = float((weight * (yy - cy * 80.0 + 0.5) ** 2).sum() / total)
    rx = float(np.clip(2.0 * np.sqrt(max(var_x, 1e-6)) / 80.0, 0.03, 0.50))
    ry = float(np.clip(2.0 * np.sqrt(max(var_y, 1e-6)) / 80.0, 0.03, 0.50))
    return np.asarray([cx, cy, rx, ry], dtype=np.float32)


def _decode_tip(heatmap: np.ndarray) -> np.ndarray:
    """Decode the hybrid tip heatmap with the trained local centroid rule."""
    heatmap = heatmap[..., 0]
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y0, y1 = max(0, y - 4), min(80, y + 5)
    x0, x1 = max(0, x - 4), min(80, x + 5)
    weight = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = float(weight.sum())
    if total <= 1e-6:
        return np.asarray([(x + 0.5) / 80.0, (y + 0.5) / 80.0], dtype=np.float32)
    return np.asarray([(xx * weight).sum() / total + 0.5, (yy * weight).sum() / total + 0.5], dtype=np.float32) / 80.0


def _draw_point(draw: ImageDraw.ImageDraw, point: tuple[float, float], color: str, radius: int = 5) -> None:
    """Draw a filled point with a contrasting outline."""
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)


def main() -> None:
    """Run six deterministic random held-out samples and save annotated panels."""
    rows = json.loads((CENTER_DATA / "metadata.json").read_text(encoding="utf-8"))["splits"]["test"]
    selected = np.random.default_rng(42).choice(rows, size=6, replace=False).tolist()
    ellipse_interpreter, ellipse_input, ellipse_outputs = _interpreter(ROOT / "artifacts" / "gauge_ellipse_mask_littlegood_v2" / "gauge_ellipse_mask_v1_int8.tflite")
    center_interpreter, center_input_detail, center_outputs = _interpreter(ROOT / "artifacts" / "gauge_center_tip_vector_littlegood_v2" / "gauge_center_tip_vector_v1_int8.tflite")
    tip_interpreter, tip_input_detail, tip_outputs = _interpreter(ROOT / "artifacts" / "gauge_center_tip_hybrid_littlegood_v1" / "gauge_center_tip_hybrid_v1_int8.tflite")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    report: list[dict[str, object]] = []
    for row in selected:
        stem = str(row["stem"])
        ellipse_path = ELLIPSE_DATA / "images" / "test" / f"{stem}.png"
        center_path = CENTER_DATA / str(row["image"])
        ellipse_image = np.asarray(Image.open(ellipse_path).convert("L"), dtype=np.float32)
        ellipse_image = tf.image.resize(ellipse_image[..., None], [320, 320]).numpy() / 255.0
        ellipse_prediction = _decode_ellipse(_predict(ellipse_interpreter, ellipse_input, ellipse_outputs, ellipse_image)[0])
        ellipse_target = np.asarray([row["ellipse"][0], row["ellipse"][1], row["ellipse"][2], row["ellipse"][3]], dtype=np.float32)
        # The ellipse label is recovered from the prepared YOLO target pixels.
        label = np.fromstring((ELLIPSE_DATA / "labels" / "test" / f"{stem}.txt").read_text(), sep=" ")[1:9].reshape(4, 2)
        target_ellipse = np.array([label[:, 0].min(), label[:, 1].min(), label[:, 0].max(), label[:, 1].max()]) * 640.0
        predicted_ellipse = np.array([ellipse_prediction[0] - ellipse_prediction[2], ellipse_prediction[1] - ellipse_prediction[3], ellipse_prediction[0] + ellipse_prediction[2], ellipse_prediction[1] + ellipse_prediction[3]]) * 640.0
        ellipse_view = Image.open(center_path).convert("RGB")
        draw = ImageDraw.Draw(ellipse_view)
        draw.rectangle(tuple(target_ellipse[[0, 1, 2, 3]]), outline="lime", width=4)
        draw.rectangle(tuple(predicted_ellipse[[0, 1, 2, 3]]), outline="red", width=4)

        center_input = _center_input(center_path, {**row, "ellipse": (ellipse_prediction * 640.0).tolist()})
        center_prediction = _predict(center_interpreter, center_input_detail, center_outputs, center_input)[0]
        tip_prediction = _predict(tip_interpreter, tip_input_detail, tip_outputs, center_input)
        local_center = center_prediction[:2]
        local_tip = _decode_tip(tip_prediction[1])
        crop_cx, crop_cy, crop_rx, crop_ry = ellipse_prediction * 640.0
        crop_side = max(2.0 * crop_rx, 2.0 * crop_ry) * 1.18
        predicted_points = np.asarray([[local_center[0] * crop_side + crop_cx - crop_side / 2.0, local_center[1] * crop_side + crop_cy - crop_side / 2.0], [local_tip[0] * crop_side + crop_cx - crop_side / 2.0, local_tip[1] * crop_side + crop_cy - crop_side / 2.0]], dtype=np.float32)
        center_view = Image.open(center_path).convert("RGB")
        center_draw = ImageDraw.Draw(center_view)
        target_points = np.asarray([row["center_xy_norm"], row["tip_xy_norm"]], dtype=np.float32) * 640.0
        _draw_point(draw, tuple(target_points[0]), "lime")
        _draw_point(draw, tuple(target_points[1]), "cyan")
        _draw_point(draw, tuple(predicted_points[0]), "red")
        _draw_point(draw, tuple(predicted_points[1]), "orange")
        _draw_point(center_draw, tuple(target_points[0]), "lime")
        _draw_point(center_draw, tuple(target_points[1]), "cyan")
        _draw_point(center_draw, tuple(predicted_points[0]), "red")
        _draw_point(center_draw, tuple(predicted_points[1]), "orange")
        panel = Image.new("RGB", (960, 320), "black")
        panel.paste(ellipse_view.resize((320, 320)), (0, 0))
        panel.paste(center_view.resize((320, 320)), (320, 0))
        ImageDraw.Draw(panel).text((650, 12), stem[:42], fill="white")
        panels.append(panel)
        report.append({"stem": stem, "ellipse_prediction": ellipse_prediction.tolist(), "ellipse_target": ellipse_target.tolist(), "center_tip_prediction_px": predicted_points.tolist(), "center_tip_target_px": target_points.tolist()})
    sheet = Image.new("RGB", (960, 320 * len(panels)), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 320))
    sheet.save(OUTPUT / "contact_sheet.png")
    (OUTPUT / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT / "contact_sheet.png")


if __name__ == "__main__":
    main()
