# Geometry OBB + Center v1 Design

## Decision

Use **MobileNetV2 + CoordConv** as the pretrained geometry backbone, with the
`320x320` colour IMX input contract and a dual-head output:

- `obb_params = [cx, cy, w, h, cos(2θ), sin(2θ)]`
- `center_xy = [cx, cy]`

For the first real training pass, target the **α=0.75** variant:

- `alpha = 0.75`
- `head_units_1 = 384`
- `head_units_2 = 160`
- `head_dropout_1 = 0.30`
- `head_dropout_2 = 0.20`
- `center_loss_weight = 3.0`

This is the best documented pretrained OBB+center family in the repo and the
strongest match to the actual board problem: learn generic gauge geometry first,
then fine-tune on the single gauge.

## Why This Backbone

### 1) It is pretrained

MobileNetV2 gives us ImageNet features, so DeepSeek is not starting from random
weights. That matters a lot when the target dataset is only ~700 images.

### 2) It already solves the right geometry problem

The model is trained to predict both:

- the gauge face box and orientation, and
- the needle pivot / dial center

That matches the polar-vote pipeline much better than a scalar regressor.

### 3) It fits the size envelope

The best documented α=0.75 run in the repo exported to a **2.47 MB INT8 TFLite**
model and stayed under the 2.5 MB budget. Another α=0.75 run hit **9.6 px**
center MAE on the test split, which is the strongest center result we have for
this family.

### 4) It generalizes better than the tiny α=0.35 baseline

The α=0.35 variant was compact, but its center error was much worse
(`~23.9 px` Euclidean on the test split). That is too sloppy if we want the
polar vote to be stable across multiple gauge layouts.

## What We Should Not Bet On

- **MobileNetV3-Small**: attractive on paper, but the `hard-swish` / SE style
  is a recurring NPU portability risk.
- **EfficientNet / Swish-heavy backbones**: same issue, plus more activation
  pressure.
- **YOLO11n-OBB**: strong detector, but the model footprint and activation
  story are too heavy for this board path.
- **Training from scratch**: unnecessary when we already have a good pretrained
  geometry backbone.

## Output Contract

Keep the firmware contract simple and deterministic:

```text
obb_params  -> [cx, cy, w, h, cos(2θ), sin(2θ)]
center_xy   -> [cx, cy]
```

Recommended decode rules:

- `cx`, `cy`, `w`, `h` are normalized to `[0, 1]`
- `cos(2θ)`, `sin(2θ)` are decoded on the CPU side if the export graph keeps
  the angle branch raw
- `center_xy` is the value polar voting should use first
- `obb_params` is the geometry sanity check and future crop helper

## Training Recipe For DeepSeek

Use the existing training entrypoint:

- [train_obb_center_mobilenetv2.py](D:/Projects/embedded-gauge-reading-tinyml/ml/scripts/train_obb_center_mobilenetv2.py)

Recommended settings:

- `IMAGE_HEIGHT = 320`
- `IMAGE_WIDTH = 320`
- `MOBILENET_ALPHA = 0.75`
- `HEAD_UNITS_1 = 384`
- `HEAD_UNITS_2 = 160`
- `HEAD_DROPOUT_1 = 0.30`
- `HEAD_DROPOUT_2 = 0.20`
- `CENTER_LOSS_WEIGHT = 3.0`
- `pretrained = True`
- `backbone_trainable = False` for the first pass

Training strategy:

1. Pretrain / transfer on the broad multi-gauge geometry pool.
2. Keep the backbone frozen for the first pass so the model does not overfit
   the 700-image single-gauge set too quickly.
3. If a later validation sweep proves it helps, unfreeze only the last MobileNet
   stage with a very low learning rate.
4. Keep positional augmentation enabled so the gauge can move around the frame.
5. Preserve the clean/validated labels only.

Suggested augmentations:

- nearest-neighbor resize + pad
- brightness / contrast jitter
- small rotation jitter
- position jitter, especially in `cy`, so the model does not lock onto a
  centered-phone-photo prior

## NPU / Export Notes

The backbone is the right starting point, but the export graph should stay as
simple as possible:

- avoid `swish` in the final head if it causes fallback
- avoid unnecessary normalization layers in the deployed graph
- if the angle branch needs normalization, do it on the CPU postprocess side
- verify the final TFLite / Cube.AI graph against the exact firmware contract

If the α=0.75 export or on-device timing is too tight, the fallback is the
same architecture with `alpha=0.50`. Do **not** fall back to MobileNetV3 or a
larger detector unless we are deliberately trading NPU safety for capacity.

## Bottom Line

For the next training pass, I would bet on:

**MobileNetV2 + CoordConv, α=0.75, 320x320 RGB, dual-head OBB + center**

It is pretrained, it has the best documented geometry results in the repo, and
it still fits the practical board budget.

