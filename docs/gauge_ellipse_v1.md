# gauge_ellipse_v1 deployment contract

`gauge_ellipse_v1` is a single-gauge grayscale ellipse regressor for the
STM32N6/IMX/DCMIPP path. It is not a YOLO post-processing graph: there is one
fixed output vector and no NMS, resize, concat, or grid decoder in the model.

## Input

- Shape: `int8[1, 640, 640, 1]`
- Quantization: `scale=1/255`, `zero_point=-128`
- Firmware preprocessing: convert the DCMIPP grayscale pixel `gray` in
  `[0,255]` to `int8(gray - 128)`. The model input is one channel, so do not
  replicate grayscale to RGB.

## Output and ellipse drawing

The output is `int8[1,5]`, with `scale=1/256` and `zero_point=-128`.
Dequantize each value as:

```c
float value = ((float)output[i] + 128.0f) / 256.0f;
```

The five values are normalized:

```text
[center_x, center_y, radius_x, radius_y, confidence]
```

Convert to 640-pixel coordinates by multiplying the first four values by
`640.0f`, clamp the radii to positive image bounds, and draw an axis-aligned
ellipse centered at `(center_x, center_y)` with radii `(radius_x, radius_y)`.
The confidence output is trained as one for the supplied `GaugeFace` frames;
the firmware should apply its normal application confidence gate before using
the ellipse.

## N6 package

The ST Edge AI 10.2.0 package is staged at:

`firmware/stm32/n657/st_ai_output/packages/gauge_ellipse_v1_int8_n6_npu/`

The package contains the generated C/H wrapper, `c_info.json`, `network.csv`,
and `gauge_ellipse_v1_int8_atonbuf.xSPI2.raw`. The generated N6 report measured
`819,200` bytes of activations in two NPU SRAM segments and `274.77 KiB` of
octoFlash weights. The relocatable blob signature is SHA-256
`e70ddc1d3c936e99b8972e4bf70ba0dbdeb9885265d4f0c4027e4f3e0fd1290f`.

The model was accepted by `stedgeai generate --target stm32n6` and the
relocatable NPU driver built with the STM32CubeIDE GCC toolchain. It has not
been flashed or measured on physical board hardware in this task.
