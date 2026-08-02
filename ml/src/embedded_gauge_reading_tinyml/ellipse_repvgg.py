"""RepVGG ellipse detector for 640x640 grayscale gauge faces.

Why RepVGG:
- Multi-branch 3x3 + 1x1 + identity blocks at training time give us the
  accuracy of a branched network.
- After training, we fuse each block into a single 3x3 Conv2D so the
  inference graph is purely feed-forward. The NPU can schedule a
  straight-line graph much more efficiently than a branched one, and the
  fused weights match the multi-branch output to within float32 noise.

Why this is QAT-safe:
- Only Conv2D, BatchNormalization, ReLU, Add, and Dense are used.
- No SE blocks, no Multiply, no Lambda, no Swish.
- tfmot.quantization.keras.quantize_model() can cleanly wrap this graph.

Activation budget at 640x640x1 grayscale (int8):
- Input:  640*640*1   = 410 KB  (this is the peak — see design note)
- Stem:   160*160*24  = 614 KB
- s1:     80*80*48    = 307 KB
- s2:     40*40*96    = 154 KB
- s3:     20*20*192   =  77 KB
- s4:     10*10*256   =  26 KB
All under the 1.5 MB peak budget the STM32 N6 NPU can hold in SRAM.

Output head:
- center_xy:    Dense(2, sigmoid)   — normalised [0,1] image coordinates
- radius_xy:    Dense(2, linear)    — normalised radii. LINEAR is critical;
  a sigmoid output would collapse to a constant after int8 quantisation
  because the radius variation (~0.001) is smaller than one int8 step.
- confidence:   Dense(1, sigmoid)   — for downstream gating.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


# ---------------------------------------------------------------------------
# Channel plan
# ---------------------------------------------------------------------------
# Width multiplier `alpha` lets us scale the whole network to fit the
# activation budget. alpha=1.0 targets ~1.1M params (~1.1MB int8 weights).
# alpha=0.75 shrinks the network ~44% if we need to come in well under
# the peak-activation cap.

def _channel_plan(alpha: float) -> dict[str, int]:
    """Return per-stage channel counts scaled by `alpha`."""
    return {
        "stem":  max(16, int(24 * alpha)),
        "s1":    max(32, int(48 * alpha)),
        "s2":    max(48, int(96 * alpha)),
        "s3":    max(96, int(192 * alpha)),
        "s4":    max(128, int(256 * alpha)),
    }


# ---------------------------------------------------------------------------
# Conv-BN helper (returns the conv + bn layers so we can fuse them later)
# ---------------------------------------------------------------------------

def _conv_bn(
    x: tf.Tensor,
    filters: int,
    kernel: int,
    stride: int,
    name: str,
) -> tuple[tf.Tensor, layers.Conv2D, layers.BatchNormalization]:
    """Apply Conv2D(no bias) + BatchNormalization and return all three for fusion."""
    conv = layers.Conv2D(
        filters,
        kernel,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )
    bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")
    return bn(conv(x)), conv, bn


# ---------------------------------------------------------------------------
# Multi-branch RepVGG block (training-time)
# ---------------------------------------------------------------------------

def _repvgg_block_multi(
    x: tf.Tensor,
    filters: int,
    stride: int,
    name: str,
) -> tf.Tensor:
    """Training-time RepVGG block: 3x3 + 1x1 + (identity BN if stride==1 and C matches).

    The identity branch is just a BatchNormalization with gamma initialised
    to zero in the caller. That way the identity contributes nothing at the
    start of training and the optimizer can decide how much to use it.
    """
    in_c = int(x.shape[-1])
    use_id = (stride == 1 and in_c == filters)

    # 3x3 branch
    b3, _, _ = _conv_bn(x, filters, 3, stride, name=f"{name}_3x3")

    # 1x1 branch
    b1, _, _ = _conv_bn(x, filters, 1, stride, name=f"{name}_1x1")

    branches = [b3, b1]
    if use_id:
        # Identity is modelled as a plain BatchNorm. The fusion code turns
        # this into a 3x3 identity kernel.
        branches.append(layers.BatchNormalization(name=f"{name}_id_bn")(x))

    y = layers.Add(name=f"{name}_add")(branches)
    return layers.ReLU(name=f"{name}_relu")(y)


def _stage_multi(
    x: tf.Tensor,
    filters: int,
    n_blocks: int,
    downsample: bool,
    name: str,
) -> tf.Tensor:
    """A stage of `n_blocks` RepVGG blocks. First block downsamples if requested."""
    for i in range(n_blocks):
        stride = 2 if (downsample and i == 0) else 1
        x = _repvgg_block_multi(x, filters, stride=stride, name=f"{name}b{i}")
    return x


# ---------------------------------------------------------------------------
# Single-branch RepVGG block (inference-time, after fusion)
# ---------------------------------------------------------------------------

def _repvgg_block_fused(
    x: tf.Tensor,
    filters: int,
    stride: int,
    name: str,
) -> tf.Tensor:
    """Inference-time RepVGG block: a single fused 3x3 conv + ReLU.

    The conv carries its own bias (it absorbs the fused BN scale + shift).
    """
    y = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=True,
        name=f"{name}_fused",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(y)


def _stage_fused(
    x: tf.Tensor,
    filters: int,
    n_blocks: int,
    downsample: bool,
    name: str,
) -> tf.Tensor:
    for i in range(n_blocks):
        stride = 2 if (downsample and i == 0) else 1
        x = _repvgg_block_fused(x, filters, stride=stride, name=f"{name}b{i}")
    return x


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

def _ellipse_head(
    features: tf.Tensor,
    name: str = "head",
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """5-vector ellipse head: center (sigmoid) + radius (LINEAR) + confidence (sigmoid).

    The radius head MUST be linear. Sigmoid wastes the int8 grid on the
    unused [0,1] range and the small radius variation collapses to a
    single quantized value.
    """
    # Center: bounded [0,1] so sigmoid is fine.
    c = layers.Dense(64, activation="relu", name=f"{name}_c_fc")(features)
    center_xy = layers.Dense(2, activation="sigmoid", name="center_xy")(c)

    # Radius: linear so the int8 grid covers the actual radius range.
    r = layers.Dense(64, activation="relu", name=f"{name}_r_fc")(features)
    radius_xy = layers.Dense(2, activation=None, name="radius_xy")(r)

    # Confidence: bounded [0,1] for downstream gating.
    f = layers.Dense(32, activation="relu", name=f"{name}_f_fc")(features)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(f)

    return center_xy, radius_xy, confidence


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_repvgg_ellipse_multi(
    input_shape: tuple[int, int, int] = (640, 640, 1),
    alpha: float = 1.0,
) -> Model:
    """Multi-branch RepVGG ellipse detector for training.

    Topology:
        stem (3x3 stride 4,  BN, ReLU)         640 -> 160
        s1   (2 blocks, downsample, 48c)        160 ->  80
        s2   (4 blocks, downsample, 96c)         80 ->  40
        s3   (6 blocks, downsample, 192c)        40 ->  20
        s4   (2 blocks, no downsample, 256c)     20 ->  20
        GAP
        Dropout 0.1
        ellipse head -> (center_xy, radius_xy, confidence)
    """
    cp = _channel_plan(alpha)
    inputs = keras.Input(shape=input_shape, name="image")

    # Stem: stride-4 single conv. The 3x3 stride 4 collapses 640x640 to
    # 160x160 in one shot, which is what keeps us under the 1.5 MB peak.
    x = layers.Conv2D(
        cp["stem"], 3, strides=4, padding="same", use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Backbone stages. Block counts [2, 4, 6, 2] keep the model under 1.2M
    # params at alpha=1.0 while giving stage 3 the depth it needs for the
    # mid-frequency dial cues.
    x = _stage_multi(x, cp["s1"], n_blocks=2, downsample=True, name="s1")
    x = _stage_multi(x, cp["s2"], n_blocks=4, downsample=True, name="s2")
    x = _stage_multi(x, cp["s3"], n_blocks=6, downsample=True, name="s3")
    x = _stage_multi(x, cp["s4"], n_blocks=2, downsample=False, name="s4")

    # Trunk -> head.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    center_xy, radius_xy, confidence = _ellipse_head(x)

    return Model(
        inputs=inputs,
        outputs=[center_xy, radius_xy, confidence],
        name=f"ellipse_repvgg_multi_a{alpha}",
    )


def build_repvgg_ellipse_fused(
    input_shape: tuple[int, int, int] = (640, 640, 1),
    alpha: float = 1.0,
) -> Model:
    """Fused (inference-time) RepVGG ellipse detector.

    Same topology as the multi-branch model, but every RepVGG block is
    a single 3x3 conv with bias. Weights are copied from the fused
    multi-branch model by `reparameterize_model`.
    """
    cp = _channel_plan(alpha)
    inputs = keras.Input(shape=input_shape, name="image")

    # Stem is identical in both models so it is copied verbatim.
    x = layers.Conv2D(
        cp["stem"], 3, strides=4, padding="same", use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Fused stages: one 3x3 conv per block, no 1x1, no Add, no per-block BN.
    x = _stage_fused(x, cp["s1"], n_blocks=2, downsample=True, name="s1")
    x = _stage_fused(x, cp["s2"], n_blocks=4, downsample=True, name="s2")
    x = _stage_fused(x, cp["s3"], n_blocks=6, downsample=True, name="s3")
    x = _stage_fused(x, cp["s4"], n_blocks=2, downsample=False, name="s4")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    center_xy, radius_xy, confidence = _ellipse_head(x)

    return Model(
        inputs=inputs,
        outputs=[center_xy, radius_xy, confidence],
        name=f"ellipse_repvgg_fused_a{alpha}",
    )


# ---------------------------------------------------------------------------
# Reparameterization: multi-branch -> single 3x3 conv per block
# ---------------------------------------------------------------------------

def _fuse_conv_bn(
    conv_w: np.ndarray,
    bn: layers.BatchNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into the preceding Conv2D weights.

    For each output channel i:
        scale_i = gamma_i / sqrt(var_i + eps)
        W'[..., i] = W[..., i] * scale_i
        b'[i]     = beta_i - gamma_i * mu_i / sqrt(var_i + eps)

    Returns (W_fused, b_fused) ready to assign to a Conv2D(use_bias=True).
    """
    gamma = np.array(bn.gamma)
    beta = np.array(bn.beta)
    mu = np.array(bn.moving_mean)
    var = np.array(bn.moving_variance)
    # Match the BN epsilon used in `_conv_bn`.
    sigma = np.sqrt(var + 1e-3)
    scale = gamma / sigma
    w = conv_w * scale.reshape(1, 1, 1, -1)
    b = beta - gamma * mu / sigma
    return w, b


def _pad_1x1_to_3x3(w: np.ndarray, stride: int) -> np.ndarray:
    """Pad a (1,1,in,out) kernel to (3,3,in,out) matching TF's SAME padding.

    TF's SAME padding for an odd kernel with stride > 1 places the kernel
    so the top-left input pixel lines up with the top-left output pixel,
    NOT the centre. So:
      - stride == 1:  weight goes at kernel index [1,1] (centre pad)
      - stride == 2:  weight goes at kernel index [0,0] (top-left pad)

    This is the trick that lets the 1x1 branch and the 3x3 branch be
    summed as if they were both 3x3 convs operating on the same input.
    """
    if stride == 1:
        return np.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]])
    return np.pad(w, [[0, 2], [0, 2], [0, 0], [0, 0]])


def _identity_3x3(c: int) -> np.ndarray:
    """3x3 identity kernel: weight 1 at centre pixel, 0 elsewhere."""
    w = np.zeros((3, 3, c, c), dtype=np.float32)
    for i in range(c):
        w[1, 1, i, i] = 1.0
    return w


def _fuse_block(
    multi_model: Model,
    block_name: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse one RepVGG block's 3x3 + 1x1 + identity branches into a single 3x3 (W, b)."""
    # 3x3 branch.
    conv3 = multi_model.get_layer(f"{block_name}_3x3_conv")
    bn3 = multi_model.get_layer(f"{block_name}_3x3_bn")
    w3, b3 = _fuse_conv_bn(np.array(conv3.kernel), bn3)

    # 1x1 branch (padded to 3x3).
    conv1 = multi_model.get_layer(f"{block_name}_1x1_conv")
    bn1 = multi_model.get_layer(f"{block_name}_1x1_bn")
    w1_raw, b1 = _fuse_conv_bn(np.array(conv1.kernel), bn1)
    w1 = _pad_1x1_to_3x3(w1_raw, stride=stride)

    w = w3 + w1
    b = b3 + b1

    # Identity branch if it exists (only when stride==1 and channels match).
    try:
        bn_id = multi_model.get_layer(f"{block_name}_id_bn")
        out_c = int(w3.shape[-1])
        w_id, b_id = _fuse_conv_bn(_identity_3x3(out_c), bn_id)
        w += w_id
        b += b_id
    except ValueError:
        # No identity branch — either stride>1 or channel mismatch.
        pass

    return w, b


def reparameterize_model(multi_model: Model) -> Model:
    """Build a single-branch fused model with weights transferred from the multi-branch model.

    The two models must have been created with the same `alpha` and the
    same `input_shape`. The multi-branch model must already be trained
    (or at least have BN running statistics populated).
    """
    # Recover alpha from the trained stem's filter count.
    stem_filters = int(multi_model.get_layer("stem_conv").filters)
    alpha = stem_filters / 24.0
    input_shape = multi_model.input_shape[1:]

    # Build the fused graph with the same topology.
    fused = build_repvgg_ellipse_fused(input_shape=input_shape, alpha=alpha)

    # Copy weights for layers that exist in both models (stem, head, dropout).
    for layer in fused.layers:
        try:
            src = multi_model.get_layer(layer.name)
        except ValueError:
            # Layer exists only in the fused graph (the per-block `_*_fused` convs).
            continue
        if "fused" in layer.name:
            continue
        if src.weights and len(src.weights) == len(layer.weights):
            try:
                layer.set_weights(src.get_weights())
            except ValueError:
                # Shape mismatch (e.g. dropout has no weights to copy). Skip.
                continue

    # Enumerate the RepVGG block names in the same order the fused model does.
    cp = _channel_plan(alpha)
    block_specs = [
        ("s1b0", cp["s1"], 2), ("s1b1", cp["s1"], 1),
        ("s2b0", cp["s2"], 2), ("s2b1", cp["s2"], 1), ("s2b2", cp["s2"], 1), ("s2b3", cp["s2"], 1),
        ("s3b0", cp["s3"], 2), ("s3b1", cp["s3"], 1), ("s3b2", cp["s3"], 1),
        ("s3b3", cp["s3"], 1), ("s3b4", cp["s3"], 1), ("s3b5", cp["s3"], 1),
        ("s4b0", cp["s4"], 1), ("s4b1", cp["s4"], 1),
    ]
    for block_name, _filters, stride in block_specs:
        w_fused, b_fused = _fuse_block(multi_model, block_name, stride=stride)
        target_name = f"{block_name}_fused"
        conv_layer = fused.get_layer(target_name)
        conv_layer.kernel.assign(w_fused)
        conv_layer.bias.assign(b_fused)

    return fused


__all__ = [
    "build_repvgg_ellipse_multi",
    "build_repvgg_ellipse_fused",
    "reparameterize_model",
    "_channel_plan",
]
