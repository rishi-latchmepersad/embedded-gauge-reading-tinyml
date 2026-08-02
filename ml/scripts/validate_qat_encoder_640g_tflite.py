"""Validate the QAT-encoder 640x640 TFLite int8 export on 5 sample images.

Run this BEFORE training. If the int8 outputs are constant, the
architecture has the same TFLite mismatch the previous RepVGG had
and we should not waste time on full training.

This script:
1. Builds a fresh QAT encoder (random weights).
2. Converts to int8 TFLite with PTQ (no QAT yet, no training).
3. Runs the TFLite model on 5 different images.
4. Asserts the outputs vary across inputs.

The previous RepVGG attempt failed this test -- the int8 output was
identical for all inputs. The QAT encoder is expected to pass because
of the BatchNorm between every conv (lesson-learned
docs/ai-memory/lessons-learned/2026-07-23-qat-safe-architecture.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.qat_encoder_640g import build_qat_encoder_640g  # noqa: E402

DATA_DIR = ROOT / "data" / "repvgg_ellipse"


def _load_n_images(n: int = 5) -> np.ndarray:
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / "val" / "labels.json").read_text())[:n]
    images = np.zeros((n, 640, 640, 1), dtype=np.float32)
    img_dir = DATA_DIR / "val" / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (640, 640):
            img = tf.image.resize(img[..., None], (640, 640), method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
    return images


def main() -> int:
    print("Building QAT encoder 640x640 grayscale (alpha=1.5)...")
    model = build_qat_encoder_640g(input_shape=(640, 640, 1), alpha=1.5)
    model.summary(line_length=100, print_fn=print)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f} MB int8)")

    # Confirm the Keras fp32 model produces varying output.
    print("\nLoading 5 sample images...")
    images = _load_n_images(5)
    print(f"  images shape: {images.shape}")
    print("Keras fp32 forward pass:")
    preds = model.predict(images, verbose=0)
    for name, p in zip(["center_xy", "radius_xy", "confidence"], preds):
        print(f"  {name}: std={p.std():.6f}, max-min={p.max() - p.min():.6f}")

    # Convert to int8 TFLite via PTQ (no QAT yet).
    print("\nConverting to int8 TFLite with PTQ...")

    def rep_dataset():
        for img in images:
            yield [img[None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    blob = converter.convert()
    out_path = Path("/tmp/test_qat_encoder_640g_int8.tflite")
    out_path.write_bytes(blob)
    print(f"  TFLite size: {len(blob) / 1024:.1f} KB ({len(blob) / 1e6:.2f} MB)")

    # Run the TFLite model on the 5 different images.
    print("\nTFLite int8 forward pass on 5 different images:")
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]
    print(f"  Input: scale={in_scale}, zp={in_zp}")
    for d in out_dets:
        print(f"  Output {d['name']}: shape={d['shape']}, dtype={d['dtype']}, "
              f"scale={d['quantization'][0]}, zp={d['quantization'][1]}")

    # Collect outputs keyed by output index (not by shape, since multiple
    # outputs can share a shape).
    all_outputs = {d["index"]: [] for d in out_dets}
    for i, img in enumerate(images):
        xq = np.clip(np.round(img[None] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for d in out_dets:
            raw = interp.get_tensor(d["index"])
            s, z = d["quantization"]
            if d["dtype"] == np.int8:
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            all_outputs[d["index"]].append(dequant.flatten())
            if i == 0:
                print(f"  img 0 -> {d['name']} = {dequant.flatten()}")

    print("\nPer-output variance across 5 inputs:")
    all_vary = True
    for idx, outs in all_outputs.items():
        arr = np.array(outs)
        std = arr.std()
        max_min = arr.max() - arr.min()
        varies = std > 1e-4
        all_vary = all_vary and varies
        status = "OK" if varies else "COLLAPSED"
        print(f"  output {idx}: std={std:.6f}, max-min={max_min:.6f} [{status}]")
        print(f"    values: {arr.tolist()}")

    if all_vary:
        print("\nSUCCESS: TFLite int8 outputs vary across inputs.")
        print("  Safe to proceed with full training.")
        return 0
    else:
        print("\nFAILURE: TFLite int8 outputs collapsed to constants.")
        print("  Do NOT train. The architecture has the same TFLite mismatch")
        print("  as the previous RepVGG attempt.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
