#!/usr/bin/env python3
"""Train a 512px local ellipse refiner on top of a multiscale proposal stage.

The goal is to keep the best part of the current proposal family — the
multiscale center proposer that already does relatively well on test_2 — and
give it a much stronger local refiner that sees a larger crop around the
proposal.  This is a more aggressive two-stage jump than the earlier 224px
refiners because the failure mode on test_2 looked like "find the face anywhere
in the frame, then tighten it up", not "slightly improve an already centered
crop."
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
from train_ellipse_robust_384 import SEED, _block, load_zips

LOCAL_SIZE = 512
STAGE1_MODEL = Path("artifacts/multiscale_center_proposal_384_v1/model_int8.tflite")
CENTER_BLEND = 0.20


def configure_gpu() -> None:
    """Limit TensorFlow to the repo-approved 15 GB GPU budget."""

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact 512px local ellipse refiner."""

    layers = keras.layers
    inputs = keras.Input((LOCAL_SIZE, LOCAL_SIZE, 1), name="local_crop")
    x = inputs
    # why: the refiner needs enough downsampling to summarize the crop without
    # blowing up memory, but it should stay lightweight for int8 deployment.
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"ref512_enc{stage}_down")
        x = _block(x, filters, 1, f"ref512_enc{stage}_refine")
    x = layers.GlobalAveragePooling2D(name="ref512_gap")(x)
    x = layers.Dense(32, activation="relu", name="ref512_hidden")(x)
    outputs = layers.Dense(4, activation="sigmoid", name="ref512_ellipse")(x)
    return keras.Model(inputs, outputs, name="multiscale_refiner_512")


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a normalized square box and return the crop plus its source box."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1 * width)), int(np.floor(y1 * height))
    ix2, iy2 = int(np.ceil(x2 * width)), int(np.ceil(y2 * height))
    # why: we deliberately keep the square crop padded so edge-touching
    # proposals still train as valid examples instead of being discarded.
    side = max(ix2 - ix1, iy2 - iy1, 1)
    canvas = np.zeros((side, side), dtype=np.float32)
    source_x1, source_y1 = max(0, ix1), max(0, iy1)
    source_x2, source_y2 = min(width, ix1 + side), min(height, iy1 + side)
    dst_x1, dst_y1 = source_x1 - ix1, source_y1 - iy1
    canvas[
        dst_y1 : dst_y1 + source_y2 - source_y1,
        dst_x1 : dst_x1 + source_x2 - source_x1,
    ] = image[source_y1:source_y2, source_x1:source_x2, 0]
    resized = cv2.resize(canvas, (LOCAL_SIZE, LOCAL_SIZE), interpolation=cv2.INTER_AREA)
    source_box = np.asarray(
        [ix1 / width, iy1 / height, (ix1 + side) / width, (iy1 + side) / height],
        dtype=np.float32,
    )
    return resized[..., None], source_box


def _decode_heatmap(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode a heatmap batch into normalized centers and peak confidences."""

    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers: list[list[float]] = []
    confidences: list[float] = []
    for row in values:
        heatmap = row[..., 0]
        peak = float(np.max(heatmap))
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1 = max(0, y - 4), min(size, y + 5)
        x0, x1 = max(0, x - 4), min(size, x + 5)
        local = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
        total = float(local.sum())
        if total > 1e-6:
            centers.append(
                [
                    float((local * xx[y0:y1, x0:x1]).sum() / total),
                    float((local * yy[y0:y1, x0:x1]).sum() / total),
                ]
            )
        else:
            centers.append([float((x + 0.5) / size), float((y + 0.5) / size)])
        confidences.append(peak)
    return np.asarray(centers, dtype=np.float32), np.asarray(confidences, dtype=np.float32)


def stage1_decode(model: Path, images: np.ndarray) -> np.ndarray:
    """Decode the multiscale proposer into a generous square proposal."""

    interpreter = tf.lite.Interpreter(model_path=str(model), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    in_scale, in_zero = inp["quantization"]
    out_q = [out["quantization"] for out in outputs]
    proposals: list[list[float]] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        coarse = (interpreter.get_tensor(outputs[0]["index"])[0].astype(np.float32) - out_q[0][1]) * out_q[0][0]
        fine = (interpreter.get_tensor(outputs[1]["index"])[0].astype(np.float32) - out_q[1][1]) * out_q[1][0]
        radius = (interpreter.get_tensor(outputs[2]["index"])[0].astype(np.float32) - out_q[2][1]) * out_q[2][0]
        coarse_center, _ = _decode_heatmap(coarse[None], coarse.shape[0])
        fine_center, fine_conf = _decode_heatmap(fine[None], fine.shape[0])
        alpha = float(np.clip((fine_conf[0] - 0.20) / 0.55, 0.0, 1.0))
        center = (1.0 - alpha) * coarse_center[0] + alpha * fine_center[0]
        # why: the refiner needs to see the face even when the stage-one radius
        # is a bit tight, so we inflate the proposal before cropping.
        side = float(np.clip(2.25 * max(radius[0], radius[1]), 0.18, 1.40))
        side = max(side, 0.06)
        proposals.append([center[0], center[1], side, side])
    return np.asarray(proposals, dtype=np.float32)


def make_proposal_training_examples(
    images: np.ndarray,
    targets: np.ndarray,
    stage1_model: Path,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create proposal-aware crops using real stage-one predictions."""

    proposals = stage1_decode(stage1_model, images)
    rng = np.random.default_rng(SEED + 1)
    local_images: list[np.ndarray] = []
    local_targets: list[list[float]] = []
    for image, target, proposal in zip(images, targets, proposals):
        for _ in range(repeats):
            side = float(np.clip(proposal[2] * rng.uniform(0.9, 1.2), 0.18, 1.40))
            cx = float(proposal[0] + rng.normal(0.0, 0.04 * side))
            cy = float(proposal[1] + rng.normal(0.0, 0.04 * side))
            box = np.asarray(
                [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
                dtype=np.float32,
            )
            crop, source_box = crop_image(image, box)
            sx = source_box[2] - source_box[0]
            sy = source_box[3] - source_box[1]
            local_images.append(crop)
            local_targets.append(
                [
                    (target[0] - source_box[0]) / sx,
                    (target[1] - source_box[1]) / sy,
                    target[2] / sx,
                    target[3] / sy,
                ]
            )
    return np.asarray(local_images, dtype=np.float32), np.clip(np.asarray(local_targets, dtype=np.float32), 0.0, 1.0)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the local refiner as fully integer TFLite."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield calibration crops so the int8 graph matches training data."""

        for sample in images[: min(512, len(images))]:
            yield [sample[None].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_local(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the local int8 refiner on a batch of 512px crops."""

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


def refine_predictions(model: Path, stage1_model: Path, images: np.ndarray) -> np.ndarray:
    """Run the proposer, crop locally, then map the refiner output back."""

    proposals = stage1_decode(stage1_model, images)
    crops: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    for image, proposal in zip(images, proposals):
        side = float(np.clip(2.25 * max(proposal[2], proposal[3]), 0.18, 1.40))
        box = np.asarray(
            [proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2],
            dtype=np.float32,
        )
        crop, source_box = crop_image(image, box)
        crops.append(crop)
        boxes.append(source_box)
    local = predict_local(model, np.asarray(crops, dtype=np.float32))
    outputs: list[list[float]] = []
    for index, (value, box) in enumerate(zip(local, boxes)):
        sx, sy = box[2] - box[0], box[3] - box[1]
        refined_center = np.asarray([box[0] + value[0] * sx, box[1] + value[1] * sy], dtype=np.float32)
        fused_center = (1.0 - CENTER_BLEND) * proposals[index, :2] + CENTER_BLEND * refined_center
        outputs.append([fused_center[0], fused_center[1], value[2] * sx, value[3] * sy])
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    """Train the refiner and evaluate the full coarse-to-fine system."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--generic-count", type=int, default=1000)
    parser.add_argument("--tiny-limit", type=int, default=1000000)
    parser.add_argument("--board-limit", type=int, default=1000000)
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))

    # why: the refiner only needs a representative slice of each family, but
    # proposal-aware crops are expensive enough that we keep the source set
    # manageable.
    images = np.concatenate(
        [
            generic_images[: args.generic_count],
            tiny_images[: args.tiny_limit],
            board_images[: args.board_limit],
        ]
    )
    targets = np.concatenate(
        [
            generic_targets[: args.generic_count],
            tiny_targets[: args.tiny_limit],
            board_targets[: args.board_limit],
        ]
    )

    local_images, local_targets = make_proposal_training_examples(
        images, targets, args.stage1_model, args.repeats
    )
    dataset = (
        tf.data.Dataset.from_tensor_slices((local_images, local_targets))
        .shuffle(len(local_images), seed=SEED)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )

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
        resized = tf.image.resize(test_images, (384, 384)).numpy()
        predictions = np.concatenate(
            [
                refine_predictions(args.output / "model_int8.tflite", args.stage1_model, resized),
                np.ones((len(test_targets), 1), dtype=np.float32),
            ],
            axis=1,
        )
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
