# Geometry Heatmap v2 TFLite Tensor Contract

## Input Tensor

| model | tensor name | shape | dtype | scale | zero point | quantized |
| --- | --- | --- | --- | ---: | ---: | --- |
| float32 | serving_default_input_image:0 | [1, 224, 224, 3] | float32 | 0.0 | 0 | False |
| int8 | serving_default_input_image:0 | [1, 224, 224, 3] | int8 | 0.003921568859368563 | -128 | True |

## Output Tensors

| model | output name | shape | dtype | scale | zero point | quantized |
| --- | --- | --- | --- | ---: | ---: | --- |
| float32 | StatefulPartitionedCall_1:1 | [1, 56, 56, 1] | float32 | 0.0 | 0 | False |
| float32 | StatefulPartitionedCall_1:0 | [1, 56, 56, 1] | float32 | 0.0 | 0 | False |
| float32 | StatefulPartitionedCall_1:2 | [1, 1] | float32 | 0.0 | 0 | False |
| int8 | StatefulPartitionedCall_1:1 | [1, 56, 56, 1] | int8 | 0.00390625 | -128 | True |
| int8 | StatefulPartitionedCall_1:0 | [1, 56, 56, 1] | int8 | 0.00390625 | -128 | True |
| int8 | StatefulPartitionedCall_1:2 | [1, 1] | int8 | 0.00390625 | -128 | True |

## Dequantization Rules

- float32 model requires no output dequantization: True
- int8 model requires output dequantization: True
- For int8 tensors, decode values with `(tensor - zero_point) * scale` after reading the raw output buffer.
- Semantic output order is center_heatmap, tip_heatmap, confidence, even though the raw TFLite tensors arrive as tip_heatmap, center_heatmap, confidence.
- Keep the board input path as RGB + bilinear resize + 224x224 + float32 normalization to 0..1.
