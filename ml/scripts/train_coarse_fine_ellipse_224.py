#!/usr/bin/env python3
"""Train and evaluate a two-stage coarse-to-fine ellipse reader.

Stage one is the current full-frame int8 proposer.  Stage two is trained on
noisy, randomly sized crops so it learns to correct proposal errors instead
of merely memorizing centered gauge crops.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_640_center import (
    IMAGE_SIZE,
    decode_masks,
    predict_int8 as predict_stage1,
)
from train_ellipse_robust_384 import SEED, _block, load_zips

LOCAL_SIZE = 224
STAGE1_MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")
# why: the local stage improves radius substantially but can over-correct the
# center on generic frames; a conservative residual blend is more stable.
CENTER_BLEND = 0.25


def configure_gpu() -> None:
    """Limit TensorFlow to the 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact local crop refiner with a scalar ellipse contract."""
    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"local_enc{stage}_down")
        x = _block(x, filters, 1, f"local_enc{stage}_refine")
    x = layers.GlobalAveragePooling2D(name="local_gap")(x)
    x = layers.Dense(32, activation="relu", name="local_hidden")(x)
    outputs = layers.Dense(4, activation="sigmoid", name="local_ellipse")(x)
    return keras.Model(inputs, outputs, name="coarse_fine_ellipse_refiner_224")


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a normalized square box and return image plus its source box."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1 * width)), int(np.floor(y1 * height))
    ix2, iy2 = int(np.ceil(x2 * width)), int(np.ceil(y2 * height))
    # why: padding lets intentionally noisy proposals remain valid training
    # examples even when the predicted crop reaches the frame boundary.
    pad = max(ix2 - ix1, iy2 - iy1, 1)
    canvas = np.zeros((pad, pad), dtype=np.float32)
    source_x1, source_y1 = max(0, ix1), max(0, iy1)
    source_x2, source_y2 = min(width, ix1 + pad), min(height, iy1 + pad)
    dst_x1, dst_y1 = source_x1 - ix1, source_y1 - iy1
    canvas[dst_y1:dst_y1 + source_y2 - source_y1, dst_x1:dst_x1 + source_x2 - source_x1] = image[source_y1:source_y2, source_x1:source_x2, 0]
    return cv2.resize(canvas, (LOCAL_SIZE, LOCAL_SIZE), interpolation=cv2.INTER_AREA)[..., None], np.asarray([ix1 / width, iy1 / height, (ix1 + pad) / width, (iy1 + pad) / height], dtype=np.float32)


def make_training_examples(images: np.ndarray, targets: np.ndarray, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy proposal crops and local ellipse targets."""
    rng = np.random.default_rng(SEED)
    local_images: list[np.ndarray] = []
    local_targets: list[np.ndarray] = []
    for image, target in zip(images, targets):
        for _ in range(repeats):
            cx, cy, rx, ry = target[:4]
            side = float(np.clip(2.2 * max(rx, ry) * rng.uniform(0.8, 1.8), 0.16, 1.4))
            proposal_cx = float(cx + rng.normal(0.0, 0.18 * side))
            proposal_cy = float(cy + rng.normal(0.0, 0.18 * side))
            box = np.asarray([proposal_cx - side / 2, proposal_cy - side / 2, proposal_cx + side / 2, proposal_cy + side / 2], dtype=np.float32)
            crop, source_box = crop_image(image, box)
            sx = source_box[2] - source_box[0]
            sy = source_box[3] - source_box[1]
            local_images.append(crop)
            local_targets.append([(cx - source_box[0]) / sx, (cy - source_box[1]) / sy, rx / sx, ry / sy])
    return np.asarray(local_images, dtype=np.float32), np.clip(np.asarray(local_targets, dtype=np.float32), 0.0, 1.0)


def make_proposal_training_examples(
    images: np.ndarray, targets: np.ndarray, stage1_model: Path, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    """Train on real stage-one proposals with mild proposal jitter."""
    proposals = stage1_decode(stage1_model, images)
    rng = np.random.default_rng(SEED + 1)
    local_images: list[np.ndarray] = []
    local_targets: list[list[float]] = []
    for image, target, proposal in zip(images, targets, proposals):
        for _ in range(repeats):
            side = float(np.clip(2.2 * max(proposal[2], proposal[3]) * rng.uniform(0.9, 1.2), 0.18, 1.4))
            cx = float(proposal[0] + rng.normal(0.0, 0.04 * side))
            cy = float(proposal[1] + rng.normal(0.0, 0.04 * side))
            box = np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)
            crop, source_box = crop_image(image, box)
            sx, sy = source_box[2] - source_box[0], source_box[3] - source_box[1]
            local_images.append(crop)
            local_targets.append([(target[0] - source_box[0]) / sx, (target[1] - source_box[1]) / sy, target[2] / sx, target[3] / sy])
    return np.asarray(local_images, dtype=np.float32), np.clip(np.asarray(local_targets, dtype=np.float32), 0.0, 1.0)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the local refiner as fully integer TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield local crops for bounded activation calibration."""
        for sample in images[: min(512, len(images))]:
            yield [sample[None].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the local int8 refiner on a batch of 224px crops."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        values.append((raw - out_zero) * out_scale)
    return np.asarray(values, dtype=np.float32)


def stage1_decode(model: Path, images: np.ndarray) -> np.ndarray:
    """Decode the current full-frame model into a generous proposal ellipse."""
    masks, heatmaps, _, geometry, scale = predict_stage1(model, images)
    proposal = decode_masks(masks)
    for index, heatmap in enumerate(heatmaps[..., 0]):
        if float(scale[index, 0]) >= 0.5 and float(np.max(heatmap)) >= 0.55:
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            proposal[index, :2] = [(x + 0.5) / heatmap.shape[1], (y + 0.5) / heatmap.shape[0]]
            proposal[index, 2:4] = geometry[index, 2:4] * np.asarray([0.487, 0.368], dtype=np.float32)
    # why: a coarse proposal must include the face even when its radius is
    # imperfect; the refiner is trained to correct this generous crop.
    proposal[:, 2:4] = np.maximum(proposal[:, 2:4] * 1.35, 0.06)
    return proposal


def refine_predictions(model: Path, stage1_model: Path, images: np.ndarray) -> np.ndarray:
    """Crop stage-one proposals, refine locally, and map ellipses to full frame."""
    proposals = stage1_decode(stage1_model, images)
    crops: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    for image, proposal in zip(images, proposals):
        side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
        box = np.asarray([proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2], dtype=np.float32)
        crop, source_box = crop_image(image, box)
        crops.append(crop)
        boxes.append(source_box)
    local = predict_local(model, np.asarray(crops, dtype=np.float32))
    outputs: list[list[float]] = []
    for value, box in zip(local, boxes):
        sx, sy = box[2] - box[0], box[3] - box[1]
        refined_center = np.asarray([box[0] + value[0] * sx, box[1] + value[1] * sy], dtype=np.float32)
        # why: retain the coarse detector's global landmark authority while
        # allowing the local crop to correct systematic proposal displacement.
        fused_center = (1.0 - CENTER_BLEND) * proposals[len(outputs), :2] + CENTER_BLEND * refined_center
        outputs.append([fused_center[0], fused_center[1], value[2] * sx, value[3] * sy])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train the refiner and evaluate the complete coarse-to-fine system."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--proposal-aware", action="store_true")
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:1000], np.repeat(tiny_images, 50, axis=0), np.repeat(board_images, 5, axis=0)])
    targets = np.concatenate([generic_targets[:1000], np.repeat(tiny_targets, 50, axis=0), np.repeat(board_targets, 5, axis=0)])
    if args.proposal_aware:
        local_images, local_targets = make_proposal_training_examples(images, targets, args.stage1_model, args.repeats)
    else:
        local_images, local_targets = make_training_examples(images, targets, args.repeats)
    dataset = tf.data.Dataset.from_tensor_slices((local_images, local_targets)).shuffle(len(local_images), seed=SEED).batch(16).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.05))
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=keras.losses.Huber(delta=0.05))
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, local_images, args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_crops": int(len(local_images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        predictions = np.concatenate([refine_predictions(args.output / "model_int8.tflite", args.stage1_model, resized), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
