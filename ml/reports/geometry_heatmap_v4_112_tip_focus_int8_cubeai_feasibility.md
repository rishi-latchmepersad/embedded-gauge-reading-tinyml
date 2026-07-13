# Phase 12B: Cube.AI CPU-Only INT8 Feasibility — tip_focus_v4_112_int8

> **Classification: Generic STM32 Cube.AI CPU codegen feasibility (NOT NPU).**  
> See Phase 12C for the NPU (STM32N6 Neural-ART accelerator) feasibility assessment.

## Toolchain
- **CLI**: ST Edge AI Core v2.0.0 (STM32CubeAI 10.0.0)
- **Target**: `stm32` (generic STM32 Cortex-M CPU; no NPU-specific `stm32n6` target in v2.0.0)
- **Model**: `model_v4_112_int8.tflite` (2.2 MB INT8)

## Result: ✅ PASS — Cube.AI CPU operator support confirmed

All **153/153 operations** passed analysis and C code generation completed successfully.  
This confirms the model can run on **STM32 CPU** (Cortex-M55 in the STM32N6 series) using the generic Cube.AI runtime. NPU hardware acceleration is NOT confirmed here.

## Memory Footprint
| Resource | Size | Notes |
|----------|------|-------|
| MACC | 208,548,212 | |
| Weights (read-only) | 1,992,716 B (1.90 MiB) | Compressed 74.7% vs float model |
| Activations (read-write) | 632,384 B (617.56 KiB) | |
| RAM total | 632,384 B (617.56 KiB) | fits in STM32N6 NPU SRAM (4.2 MB) |
| Weights + Activations total | ~2.5 MiB | well within NPU budget |

## Tensor Contract (Cube.AI Output Order)
| C Output | TFLite Tensor | Semantic Name | Shape |
|----------|---------------|---------------|-------|
| `nl_78` (out_1) | `StatefulPartitionedCall_1:1` | tip_heatmap | int8 [1,112,112,1] |
| `nl_73` (out_2) | `StatefulPartitionedCall_1:0` | center_heatmap | int8 [1,112,112,1] |
| `nl_76` (out_3) | `StatefulPartitionedCall_1:2` | confidence | int8 [1,1] |

**Semantic order**: `[tip_heatmap, center_heatmap, confidence]` — Cube.AI output matches expected order directly. No remapping needed in firmware.

## Generated Artifacts
Location: `firmware/stm32/n657/st_ai_output/packages/tip_focus_v4_112_int8/st_ai_output/`

| File | Size | Purpose |
|------|------|---------|
| `tip_focus_v4_112_int8.c` | 463 KB | Network C implementation |
| `tip_focus_v4_112_int8.h` | 11 KB | Network API header |
| `tip_focus_v4_112_int8_config.h` | 1.5 KB | Configuration |
| `tip_focus_v4_112_int8_data.c` | 3.7 KB | Data section (activations config) |
| `tip_focus_v4_112_int8_data.h` | 2.5 KB | Data section header |
| `tip_focus_v4_112_int8_data_params.c` | 5.2 MB | Weights/biases data |
| `tip_focus_v4_112_int8_data_params.h` | 2 KB | Weights header |
| `tip_focus_v4_112_int8_c_info.json` | 637 KB | Per-layer analysis |
| `tip_focus_v4_112_int8_c_graph.json` | 752 KB | Graph structure |
| `tip_focus_v4_112_int8_report.json` | 78 KB | Summary report |

## Caveats
1. **Target `stm32`, not `stm32n6`**: The installed stedgeai v2.0.0 does not support the `stm32n6` NPU target. The analysis was run against the generic `stm32` target (Cortex-M CPU inference). NPU-specific operator mapping and offloading decisions require ST Edge AI Core ≥3.0.0 (or the cloud v4.0.0).
2. **No ONNX intermediate**: v2.0.0 `generate` produces C code directly without the ONNX + _Q.json intermediates seen in prior packages. This is normal for the `generate` command path.
3. **No arm-none-eabi-gcc available**: Final binary size estimate was skipped because the ARM GCC toolchain is not in PATH. Real `.text`/`.data`/`.bss` sizes will be known after compilation in STM32CubeIDE.

## Next Steps
1. ~~(Optional) Upgrade to ST Edge AI Core ≥3.0.0 for `stm32n6` NPU target analysis~~ → See Phase 12C
2. Integrate the generated C wrappers into the STM32CubeIDE firmware project (CPU-only path)
3. Compile with `arm-none-eabi-gcc` to get final binary size
4. Run on-target validation (board pipeline eval)
5. Measure NPU vs CPU inference latency

## File List
- `firmware/stm32/n657/st_ai_output/packages/tip_focus_v4_112_int8/`
  - `CUBEAI_ANALYSIS.md` - instructions (superseded by this report)
  - `st_ai_output/` - all generated artifacts (CPU-only C code for generic stm32 target)
