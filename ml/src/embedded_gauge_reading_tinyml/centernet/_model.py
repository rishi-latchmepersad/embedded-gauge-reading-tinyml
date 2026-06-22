"""CenterNet model builders for gauge center detection.

Provides:
  - build_centernet_resnet50(): Full teacher/parent CenterNet with ResNet-50.
  - build_centernet_mobilenetv2_student(): MobileNetV2-based student for KD.

Both follow the Objects as Points (Zhou et al. 2019) architecture:
  Input → Backbone → Upsampling decoder → Heatmap head + Offset head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import keras
import tensorflow as tf


@dataclass
class CenterNetConfig:
    """Configuration for a CenterNet model.

    Attributes:
        input_height: Input image height.
        input_width: Input image width.
        heatmap_height: Output heatmap height (at stride 4).
        heatmap_width: Output heatmap width.
        num_classes: Number of keypoint classes (1 for gauge center).
        backbone: Backbone type ('resnet50' or 'mobilenetv2').
        backbone_weights: Pretrained weights source (e.g. 'imagenet').
        backbone_trainable: Whether backbone is trainable.
        alpha: MobileNetV2 width multiplier.
        decoder_filters: Base filter count for the upsampling decoder.
        head_filters: Filter count for the prediction heads.
        use_skip_connections: Use encoder-decoder skip connections (student).
    """

    input_height: int = 384
    input_width: int = 384
    heatmap_height: int = 96
    heatmap_width: int = 96
    num_classes: int = 1
    backbone: str = "resnet50"
    backbone_weights: str = "imagenet"
    backbone_trainable: bool = True
    alpha: float = 0.35
    decoder_filters: int = 256
    head_filters: int = 64
    use_skip_connections: bool = False


# ---------------------------------------------------------------------------
# ResNet-50 CenterNet (parent / teacher)
# ---------------------------------------------------------------------------


def _upsample_block(
    x: keras.KerasTensor,
    filters: int,
    name: str,
) -> keras.KerasTensor:
    """One upsampling stage: Conv2DTranspose → BN → ReLU."""
    x = keras.layers.Conv2DTranspose(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False,
        name=f"{name}_upconv",
    )(x)
    x = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-5, name=f"{name}_bn"
    )(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    return x


def _heatmap_head(
    x: keras.KerasTensor,
    num_classes: int,
    head_filters: int,
    name: str = "heatmap_head",
) -> keras.KerasTensor:
    """CenterNet heatmap head: 3x3 Conv → ReLU → 1x1 Conv → Sigmoid."""
    x = keras.layers.Conv2D(
        head_filters,
        kernel_size=3,
        padding="same",
        use_bias=True,
        name=f"{name}_conv1",
    )(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    x = keras.layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="center_heatmap",
    )(x)
    return x


def _offset_head(
    x: keras.KerasTensor,
    head_filters: int,
    name: str = "offset_head",
) -> keras.KerasTensor:
    """CenterNet offset head: 3x3 Conv → ReLU → 1x1 Conv (linear)."""
    x = keras.layers.Conv2D(
        head_filters,
        kernel_size=3,
        padding="same",
        use_bias=True,
        name=f"{name}_conv1",
    )(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    x = keras.layers.Conv2D(
        2,
        kernel_size=1,
        padding="same",
        name="center_offset",
    )(x)
    return x


def build_centernet_resnet50(
    config: CenterNetConfig | None = None,
) -> keras.Model:
    """Build CenterNet with ResNet-50 backbone (teacher model).

    Architecture:
      Input (384, 384, 3)
        → ResNet-50 (stride 32 → 12×12×2048)
        → 3× Upsample (stride 32→16→8→4 → 96×96×64)
        → Heatmap head (96×96×1) + Offset head (96×96×2)

    Fits in ~3.5 GB GPU memory with batch size 2.

    Args:
        config: CenterNetConfig. Uses defaults if None.

    Returns:
        keras.Model with outputs [center_heatmap, center_offset].
    """
    if config is None:
        config = CenterNetConfig()

    # Input.
    inputs = keras.Input(
        shape=(config.input_height, config.input_width, 3),
        name="image",
    )

    # ResNet-50 backbone (pretrained on ImageNet).
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=config.backbone_weights,
        input_tensor=inputs,
        pooling=None,
    )
    backbone.trainable = config.backbone_trainable

    # The backbone output is at stride 32.
    # For 384×384 input: 12×12×2048.
    x = backbone.output

    # Initial refinement convolution.
    x = keras.layers.Conv2D(
        config.decoder_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="decoder_init_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-5, name="decoder_init_bn"
    )(x)
    x = keras.layers.ReLU(name="decoder_init_relu")(x)

    # Upsampling: stride 32 → 16 → 8 → 4.
    # 12×12 → 24×24.
    x = _upsample_block(x, 128, name="decoder_up1")
    # 24×24 → 48×48.
    x = _upsample_block(x, 64, name="decoder_up2")
    # 48×48 → 96×96.
    x = _upsample_block(x, 32, name="decoder_up3")

    # Prediction heads.
    heatmap = _heatmap_head(x, config.num_classes, config.head_filters)
    offset = _offset_head(x, config.head_filters)

    # Concatenate into a single output tensor for Keras loss compatibility:
    # [:,:,:1] = heatmap, [:,:,1:3] = offset.
    output = keras.layers.Concatenate(name="center_output")([heatmap, offset])

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name="centernet_resnet50",
    )
    return model


# ---------------------------------------------------------------------------
# MobileNetV2 Student CenterNet (for knowledge distillation)
# ---------------------------------------------------------------------------


def _mobilenetv2_backbone(
    inputs: keras.KerasTensor,
    alpha: float = 0.35,
    trainable: bool = True,
) -> tuple[keras.KerasTensor, list[keras.KerasTensor]]:
    """Build MobileNetV2 backbone and return output + skip features.

    Returns:
        (final_features, skip_features) where skip_features are from
        block_2, block_4, block_12 (at strides 4, 8, 16 respectively)
        and final_features is at stride 32.
    """
    # Use the alpha-scaled MobileNetV2 from keras.applications.
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=alpha,
        pooling=None,
    )
    base.trainable = trainable

    # Extract intermediate skip features by layer name.
    # These layer names are stable across Keras versions.
    # MobileNetV2 layer order: block_2 (stride4), block_4 (stride8), block_12 (stride16).
    # We reverse so that decoder merges: stride16 first → stride8 → stride4 last.
    skip_names_ordered = ["block_12_add", "block_4_add", "block_2_add"]
    skip_outputs: list[keras.KerasTensor] = []

    for layer in base.layers:
        if layer.name in skip_names_ordered:
            skip_outputs.append(layer.output)

    # Collect from low-res (block_12) to high-res (block_2).
    # The loop visits layers in forward order, so block_2 is first, block_12 last.
    # Reverse to get [block_12_add (14x14), block_4_add (28x28), block_2_add (56x56)].
    skip_outputs = list(reversed(skip_outputs))

    return base.output, skip_outputs


def _decoder_with_skips(
    x: keras.KerasTensor,
    skips: list[keras.KerasTensor],
    target_h: int,
    target_w: int,
    decoder_filters: int = 256,
) -> keras.KerasTensor:
    """Upsampling decoder with skip connections from encoder.

    The skips list is in order of increasing resolution:
      skips[0] = stride 16 (from block_12_add)
      skips[1] = stride 8  (from block_4_add)
      skips[2] = stride 4  (from block_2_add)

    x is at stride 32.
    We upsample in stages, merging with skips at each resolution.
    """
    # Initial refinement.
    x = keras.layers.Conv2D(
        decoder_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="stud_decoder_init",
    )(x)
    x = keras.layers.BatchNormalization(name="stud_decoder_init_bn")(x)
    x = keras.layers.ReLU(name="stud_decoder_init_relu")(x)

    # Up 32→16, merge with skip[0] (stride 16).
    x = _upsample_block(x, decoder_filters // 2, name="stud_up1")
    if len(skips) > 0 and skips[0] is not None:
        skip = keras.layers.Conv2D(
            decoder_filters // 2,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="stud_skip1_proj",
        )(skips[0])
        skip = keras.layers.BatchNormalization(name="stud_skip1_bn")(skip)
        x = keras.layers.Add(name="stud_skip1_add")([x, skip])
        x = keras.layers.ReLU(name="stud_skip1_relu")(x)

    # Up 16→8, merge with skip[1] (stride 8).
    x = _upsample_block(x, decoder_filters // 4, name="stud_up2")
    if len(skips) > 1 and skips[1] is not None:
        skip = keras.layers.Conv2D(
            decoder_filters // 4,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="stud_skip2_proj",
        )(skips[1])
        skip = keras.layers.BatchNormalization(name="stud_skip2_bn")(skip)
        x = keras.layers.Add(name="stud_skip2_add")([x, skip])
        x = keras.layers.ReLU(name="stud_skip2_relu")(x)

    # Up 8→4, merge with skip[2] (stride 4).
    x = _upsample_block(x, decoder_filters // 8, name="stud_up3")
    if len(skips) > 2 and skips[2] is not None:
        skip = keras.layers.Conv2D(
            decoder_filters // 8,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="stud_skip3_proj",
        )(skips[2])
        skip = keras.layers.BatchNormalization(name="stud_skip3_bn")(skip)
        x = keras.layers.Add(name="stud_skip3_add")([x, skip])
        x = keras.layers.ReLU(name="stud_skip3_relu")(x)

    # Final refinement to match heatmap spatial dimensions.
    x = keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=True,
        name="stud_final_refine",
    )(x)
    x = keras.layers.ReLU(name="stud_final_refine_relu")(x)

    return x


def build_centernet_mobilenetv2_student(
    config: CenterNetConfig | None = None,
) -> keras.Model:
    """Build MobileNetV2-based CenterNet student model for KD.

    Architecture:
      Input (H, W, 3)
        → MobileNetV2 (α-scaled) encoder
        → 3× Upsample decoder with skip connections
        → Heatmap head + Offset head

    Designed for <2.5 MB activations with α=0.35 and input 224×224.

    Args:
        config: CenterNetConfig. Uses student defaults if None.

    Returns:
        keras.Model with outputs [center_heatmap, center_offset].
    """
    if config is None:
        config = CenterNetConfig(
            backbone="mobilenetv2",
            input_height=224,
            input_width=224,
            heatmap_height=56,
            heatmap_width=56,
            alpha=0.35,
            use_skip_connections=True,
            decoder_filters=128,
            head_filters=32,
        )

    inputs = keras.Input(
        shape=(config.input_height, config.input_width, 3),
        name="image",
    )

    # MobileNetV2 encoder with skip features.
    final_features, skips = _mobilenetv2_backbone(
        inputs,
        alpha=config.alpha,
        trainable=config.backbone_trainable,
    )

    # Decoder with skip connections.
    x = _decoder_with_skips(
        final_features,
        skips,
        target_h=config.heatmap_height,
        target_w=config.heatmap_width,
        decoder_filters=config.decoder_filters,
    )

    # Prediction heads.
    heatmap = _heatmap_head(
        x, config.num_classes, config.head_filters, name="stud_hm_head"
    )
    offset = _offset_head(
        x, config.head_filters, name="stud_off_head"
    )

    # Concatenate into single output: [:,:,:1]=heatmap, [:,:,1:3]=offset.
    output = keras.layers.Concatenate(name="stud_center_output")([heatmap, offset])

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name="centernet_mobilenetv2_student",
    )
    return model
