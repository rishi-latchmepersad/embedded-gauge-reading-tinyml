# gauge_center_tip_v1

`gauge_center_tip_v1` is the second-stage needle localizer. It consumes a
square grayscale crop generated from the `gauge_ellipse_v1` prediction and a
filled ellipse mask generated from the same predicted `(cx, cy, rx, ry)`.

## Deployment contract

- Input: signed int8 `[1, 160, 160, 2]`, channel-last.
  - channel 0: grayscale crop normalized to `[-1, 1]`
  - channel 1: ellipse mask normalized to `[-1, 1]`
- Input quantization: scale `0.007843138`, zero point `0`
- Output: signed int8 `[1, 40, 40, 2]`, channel-last.
  - channel 0: center heatmap
  - channel 1: tip heatmap
- Output quantization: scale `0.00390625`, zero point `-128`
- Decode: select or soft-argmax each heatmap, map 40×40 coordinates back to
  the 160×160 crop, then transform through the ellipse crop to the 640×640
  DCMIPP frame and draw the center-to-tip needle line.

CVAT duplicates are handled deterministically during preparation: the center
box nearest the ellipse center is selected, and the tip box farthest from that
center is selected as the outward needle endpoint.

## N6 allocation

The ST Edge AI 10.2.0 relocatable build reports **819,200 bytes (800 KiB) of
activations**, below the 1.5 MB SRAM budget. It uses two activation segments;
the generated report places 1.0 MiB in CPU RAM2 and 448 KiB in NPU RAM3.
All 21 epochs are hardware except the tool-generated DepthToSpace and Concat
epochs, which are reported as software epochs and require firmware/runtime
validation before declaring the complete production path.

The packaged blob is beside `c_info.json` and `network.csv` at:

`firmware/stm32/n657/st_ai_output/packages/gauge_center_tip_v1_int8_n6_npu/st_ai_output/`

The QAT TFLite model uses only the exported Conv2D, max-pool, nearest-neighbor
resize, concatenation, logistic, and quantization operators.
