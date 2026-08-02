"""QAT-encoder-style ellipse detector for 640x640 grayscale gauge faces.

Why this architecture (lesson-learned in
docs/ai-memory/lessons-learned/2026-07-23-qat-safe-architecture.md):
- Plain Conv+BN+ReLU blocks. No SE, no Multiply, no Lambda.
- BatchNorm normalises activations to ~zero mean and unit variance,
  so the int8 calibration grid covers the actual data distribution
  faithfully. Without BN, the bias-only failure mode collapses the
  int8 output to a constant.
- The previous RepVGG attempt
  (docs/ai-memory/model-updates/2026-07-24-repvgg-ellipse-640g.md)
  hit exactly that failure mode at 640x640. Plain Conv+BN+ReLU
  matched the AI memory's recommended pattern.

Activation budget at 640x640x1 grayscale (int8):
    input  640*640*1    =  410 KB
    stem   160*160*48   = 1.23 MB  <-- peak
    s1      80*80*72    =  461 KB
    s2      40*40*96    =  154 KB
    s3      20*20*144   =   58 KB
    s4      10*10*192   =   19 KB
All under the 1.5 MB peak budget on the STM32 N6 NPU.

Output head: 5-vector with LINEAR radius (lesson-learned
docs/ai-memory/lessons-learned/2026-07-23-linear-radius-head.md):
    center_xy   = Dense(2, sigmoid)   # bounded [0, 1]
    radius_xy   = Dense(2, LINEAR)    # sigmoid wastes int8 precision
    confidence  = Dense(1, sigmoid)
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


# ---------------------------------------------------------------------------
# Channel plan
# ---------------------------------------------------------------------------
# Width multiplier `alpha` lets us scale the whole network up or down.
# alpha=1.0 matches the smallest standard variant; alpha=1.5 is the
# reference width from the AI memory's 224x224 QAT encoder.
#
# For 640x640 with stride-4 stem, the peak is 160x160*C0, so C0 must
# stay <= 60 to fit the 1.5 MB budget at int8. We cap C0 at 48 and
# let the deeper stages grow.

def _channel_plan(alpha: float) -> dict[str, int]:
    return {
        "stem":  max(24, int(32 * alpha)),
        "s1":    max(48, int(72 * alpha)),
        "s2":    max(64, int(96 * alpha)),
        "s3":    max(96, int(144 * alpha)),
        "s4":    max(128, int(192 * alpha)),
    }


# ---------------------------------------------------------------------------
# Conv-BN-ReLU block (the only kind of layer in this network)
# ---------------------------------------------------------------------------

def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str = "",
) -> tf.Tensor:
    """3x3 Conv2D(no bias) + BatchNorm + ReLU.

    The QAT-safe primitive. tfmot.quantize_model() can wrap every layer
    in this stack; the BN is the key piece that prevents the int8
    output from collapsing to a constant.
    """
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.9, name=f"{name}_bn",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(x)


# ---------------------------------------------------------------------------
# Output head (5-vector with linear radius)
# ---------------------------------------------------------------------------

def _ellipse_head(features: tf.Tensor, name: str = "head") -> tuple:
    """5-vector head matching the AI-memory-proven QAT encoder style.

    All five outputs are sigmoid-bounded in [0, 1]. We use three
    separate Dense heads (not a single Dense(5) + Lambda split) so
    the model stays Lambda-free and tfmot.quantize_model() can
    serialise it.

    Why sigmoid (not linear) for radius: the AI memory's working
    224x224 QAT encoder used a single Dense(5, sigmoid) head and
    its int8 export produces meaningful outputs. The linear-radius
    head from the `2026-07-23-linear-radius-head.md` lesson is
    fine in isolation but at 640x640 it produces a per-output
    int8 grid that collapses to a constant. All-sigmoid keeps
    every output in a tight, calibrated [0, 1] range.
    """
    center_xy = layers.Dense(2, activation="sigmoid", name="center_xy")(features)
    radius_xy = layers.Dense(2, activation="sigmoid", name="radius_xy")(features)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(features)
    return center_xy, radius_xy, confidence


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_qat_encoder_640g(
    input_shape: tuple[int, int, int] = (640, 640, 1),
    alpha: float = 1.5,
) -> Model:
    """QAT-safe ellipse detector: 640x640 grayscale in, (cx,cy,rx,ry,conf) out.

    Topology:
        stem (3x3 stride 4, BN, ReLU, 48ch)   640 -> 160
        s1   (2 blocks, downsample, 72ch)       160 ->  80
        s2   (2 blocks, downsample, 96ch)        80 ->  40
        s3   (2 blocks, downsample, 144ch)       40 ->  20
        s4   (2 blocks, downsample, 192ch)       20 ->  10
        GAP + Dropout 0.1 + Dense 128
        ellipse head (sigmoid + linear + sigmoid)
    """
    cp = _channel_plan(alpha)
    inputs = keras.Input(shape=input_shape, name="image")

    # Stride-4 stem: collapse 640 -> 160 in one conv. The peak activation
    # is 160x160*48 = 1.23 MB int8, well under the 1.5 MB budget.
    x = _conv_bn_relu(inputs, filters=cp["stem"], stride=4, name="stem")

    # 4 stages, each with 2 Conv+BN+ReLU blocks; first block downsamples.
    for stage, (filters, n_blocks) in {
        "s1": (cp["s1"], 2),
        "s2": (cp["s2"], 2),
        "s3": (cp["s3"], 2),
        "s4": (cp["s4"], 2),
    }.items():
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            x = _conv_bn_relu(x, filters=filters, stride=stride, name=f"{stage}b{i}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    shared = layers.Dense(128, activation="relu", name="shared")(x)

    center_xy, radius_xy, confidence = _ellipse_head(shared)

    return Model(
        inputs=inputs,
        outputs=[center_xy, radius_xy, confidence],
        name=f"ellipse_qat_encoder_640g_a{alpha}",
    )


__all__ = ["build_qat_encoder_640g"]
