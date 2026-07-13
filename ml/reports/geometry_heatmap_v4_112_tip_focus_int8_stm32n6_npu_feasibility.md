# Phase 12C: STM32N6 Neural-ART NPU Feasibility — tip_focus_v4_112_int8

## Toolchain
- **CLI**: ST Edge AI Core v4.0.0 (Windows local install + cloud benchmark validation)
- **Target**: `stm32n6` (STM32N6 series with Neural-ART NPU)
- **Board**: STM32N6570-DK (Discovery Kit)
- **Model**: `model_v4_112_int8.tflite` (2.2 MB INT8)

## Result: ✅ PASS — NPU acceleration confirmed on real hardware

| Test | Status | Detail |
|------|--------|--------|
| **Compile (`--target stm32n6 --st-neural-art`)** | ✅ PASS | NPU-integrated C code + NBG binary produced |
| **On-target validate (real N6570-DK)** | ✅ PASS | 23.771 ms/inference, 42 FPS |
| **NPU utilization** | ✅ | **43.1% HW (NPU)**, 56.5% SW (CPU), 0.4% control |
| **Operator mapping** | ✅ CONFIRMED | 12 epoch blocks: 7 HW (EC), 4 SW (Resize), 1 HYBRID (Concat) |
| **Numerical accuracy** | ✅ | cos=0.996 (tip), 0.999 (center), 1.000 (conf) — excellent |

## Memory Footprint (from `analyze --target stm32n6`)

| Resource | Size | Notes |
|----------|------|-------|
| MACC | 208,548,212 | ~209M multiply-accumulate operations |
| Weights (read-only) | 1,992,716 B (1.90 MiB) | 1 segment, compressed 74.7% vs float |
| Activations (read-write) | 632,256 B (617.44 KiB) | 1 segment, fits in NPU SRAM (4.2 MB) |
| RAM total (incl. I/O) | 638,580 B (623.61 KiB) | Kernel RAM 6,324 B |
| FLASH total (RT + weights) | 2,083,263 B (2.03 MiB) | RT code 90,543 B + weights 1,992,720 B |
| Internal FLASH usage | ~84 KiB | RT runtime + toolchain objects |
| External FLASH needed | ~2 MiB | Weights + data segments |

## NBG (Neural Network Binary Graph) Generation

The `generate_nbg` endpoint (`/api/generate_nbg/optimize`) successfully produced an NPU binary graph:

- **File**: `model_v4_112_int8_4.nb` (2085.9 KiB)
- **Format**: VPMN (Visual Processor Model Network) — ST NPU binary format
- **Header**: `VPMN\x20\x00\x01\x00\x15\x00\x00\x00_NHWC...`
- **Layout**: NHWC (channel-last, consistent with model's TensorFlow Lite format)

This confirms the model CAN be compiled to the Neural-ART NPU instruction set. The NBG file contains the NPU microcode that maps AI operations to hardware acceleration when possible.

## On-Target Validation Results (STM32N6570-DK)

### Device

| Parameter | Value |
|-----------|-------|
| Board | STM32N6570-DK (Discovery Kit) |
| MCU | STM32N6xx @ 800/400 MHz |
| NPU | Neural-ART @ 1000 MHz |
| Protocol | Serial (COM16:921600) |
| Tools | ST Neural ART (LL_ATON API) v1.1.3 |
| Runtime lib | atonn-v1.1.3-262-g7cc65410 |

### Performance

| Metric | Value |
|--------|-------|
| Inference time | **23.771 ms** (42.07 inf/s) |
| HW (NPU) | **43.1%** |
| SW (CPU) | **56.5%** |
| SW control | 0.4% |
| Network init | 0.001 ms |

### Per-Epoch Breakdown

| # | Type | Duration | %Total | Description |
|---|------|----------|--------|-------------|
| 0 | EC (HW) | 8.867 ms | 37.30% | EpochBlock_1 → 55 (MobileNetV2 backbone bulk) |
| 1 | SW - Resize | 0.513 ms | 2.16% | EpochBlock_56 |
| 2 | EC (HW) | 0.223 ms | 0.94% | EpochBlock_57 → 58 |
| 3 | SW - Resize | 1.079 ms | 4.54% | EpochBlock_59 |
| 4 | EC (HW) | 0.126 ms | 0.53% | EpochBlock_60 → 61 |
| 5 | SW - Resize | 2.311 ms | 9.72% | EpochBlock_62 |
| 6 | HYBRID - Concat | 0.102 ms | 0.43% | EpochBlock_63 |
| 7 | extra HW | 0.107 ms | 0.45% | EpochBlock_63 (3) |
| 8 | extra HW | 0.054 ms | 0.23% | EpochBlock_63 (3) |
| 9 | EC (HW) | 0.402 ms | 1.69% | EpochBlock_64 → 65 |
| 10 | **SW - Resize** | **9.429 ms** | **39.66%** | **EpochBlock_66 — biggest single cost** |
| 11 | EC (HW) | 0.557 ms | 2.35% | EpochBlock_67 → 70 |

### Numerical Accuracy (vs float reference)

| Output | MAE | RMSE | Cosine Sim | Meaning |
|--------|-----|------|------------|---------|
| `nl_78` (tip_heatmap) | 0.0095 | 0.0154 | **0.996** | Excellent match |
| `nl_73` (center_heatmap) | 0.0232 | 0.0310 | **0.999** | Excellent match |
| `nl_76` (confidence) | 0.0000 | 0.0000 | **1.000** | Identical |

Output order: `[tip_heatmap, center_heatmap, confidence]` — confirmed.

### Operator Mapping (from on-target profiling)

The model is compiled to 12 epoch blocks (NPU scheduling units):

| Block Type | Count | % Total Time |
|------------|-------|-------------|
| EC (Epoch Controller = NPU HW) | 7 | 43.31% |
| SW - Resize (CPU fallback) | 4 | 56.08% |
| HYBRID - Concat (mixed) | 1 | 0.43% |
| extra HW (NPU extension) | 2 | 0.68% |

The 4 CPU-constrained RESIZE_BILINEAR operations consume 56.08% of inference time. The NPU handles the remaining 43.99% of operations across the MobileNetV2 backbone and decoder heads.

### Performance Analysis

The 23.771 ms (42 FPS) inference time is real-time capable for industrial gauge reading. The main bottleneck is the **EpochBlock_66** (9.429 ms = 39.66%) — a SW-managed RESIZE_BILINEAR in the final decoder stage. This is inherent to the model architecture and can only be reduced by:
1. Model redesign (replace RESIZE_BILINEAR with transposed convolution)
2. Larger NPU-batch scheduling (reduce SW sync points)

## Tensor Contract (STM32N6 `generate` Output Order)

Confirmed from `generate` output (`network_c_info.json`):

| C Output | TFLite Tensor | Semantic Name | Shape  |
|----------|---------------|---------------|--------|
| `nl_78` (out_1) | `StatefulPartitionedCall_1:1` | tip_heatmap | int8 [1,112,112,1] |
| `nl_73` (out_2) | `StatefulPartitionedCall_1:0` | center_heatmap | int8 [1,112,112,1] |
| `nl_76` (out_3) | `StatefulPartitionedCall_1:2` | confidence | int8 [1,1] |

**Semantic order**: `[tip_heatmap, center_heatmap, confidence]` — matches v2.0.0 CPU-only output. No remapping needed.

## Key Findings & Risks

1. **NPU compilation is confirmed**: The `--st-neural-art` flag successfully produces NPU-integrated C code. The model runs on real STM32N6570-DK hardware with **43.1% NPU acceleration**.

2. **CPU bottleneck identified**: RESIZE_BILINEAR operations consume **56.08% of inference time** across 4 epoch blocks. The final decoder upsampling stage (EpochBlock_66) alone takes **9.429 ms (39.66%)**. These SW sync points dominate the latency.

3. **Real-time capable**: 23.771 ms (42 FPS) is sufficient for industrial gauge reading at typical camera capture rates (15-30 FPS).

4. **Memory fits (confirmed)**: 617 KiB activations within NPU SRAM (4.2 MB). ~2 MiB weights in external Flash. No external RAM needed.

5. **Numerical accuracy is excellent**: Cosine similarity > 0.995 for all outputs. INT8 quantization loss is negligible.

6. **Cloud API limitation**: The REST API does not pass `--st-neural-art` on analyze/generate endpoints, but the benchmark/validate pipeline does include it for firmware build + on-target validation.

## Recommendations

1. **Proceed to firmware integration**: The model is confirmed feasible for STM32N6 NPU. Use the `--st-neural-art` compile output for NPU-scheduled C code.

2. **Optimization opportunity**: If higher frame rate is needed (>42 FPS), consider replacing RESIZE_BILINEAR with transposed convolutions (NPU-friendly) in the decoder to reduce SW sync points.

3. **Firmware integration approach**: Use the validate/benchmark pipeline output which includes both the NPU NBG binary (embedded) and the NPU-aware C scheduling code. The C API follows standard `stai_network_*` conventions:
   - `stai_network_init()` — init
   - `stai_network_run()` — inference (NPU + CPU coordinated)
   - Input/output buffer management via `stai_network_get_inputs/set_inputs`

## Artifacts

| File | Size | Description |
|------|------|-------------|
| `st_ai_output/network.c` | 396.6 KiB | CPU C implementation for stm32n6 |
| `st_ai_output/network.h` | 21.0 KiB | Network API header |
| `st_ai_output/network_data.c` | 5190.7 KiB | Weights and data tables |
| `st_ai_output/network_c_info.json` | 642.7 KiB | Per-node analysis (all NODE_SW) |
| `st_ai_output/NetworkRuntime1200_CM55_GCC.a` | 1257.3 KiB | Cortex-M55 CPU runtime library |
| `nbg_output/model_v4_112_int8_4.nb` | 2085.9 KiB | NPU Neural Binary Graph (VPMN format) |
| `cloud_analysis/analyze_raw_result.json` | - | Raw analyze output |

## References

- Phase 12B report: `ml/reports/geometry_heatmap_v4_112_tip_focus_int8_cubeai_feasibility.md`
- Cloud API script: `tmp/stm32ai_dc_phase12c/phase12c_npu_analysis.py`
- NBG download script: `tmp/stm32ai_dc_phase12c/phase12c_npu_download5.py`
- Benchmark attempt: `tmp/stm32ai_dc_phase12c/phase12c_benchmark.py`
