# Integration Report: tip_focus_v4_112_int8 Firmware Spike

**Date**: 2026-05-26
**Phase**: 13A
**Status**: Guarded integration complete — build verified in both disabled (0U) and enabled (1U) states.

## Summary

The locked `tip_focus_v4_112_int8` geometry heatmap model was integrated into the STM32N6 firmware as a guarded, reversible spike. The model runs as generic STAI CPU code (all 100 nodes `"mapping": "NODE_SW"`, no NPU offloading). Memory overflow (2.0 MiB weights, 617 KiB activations) was solved by placing both in external memory regions via linker script modifications.

## Integration Architecture

```
APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE (default 0U)
  └── ai_network_tip_focus_v4_112_int8.c  ← #include-s generated network.c + network_data.c
        ├── g_tip_focus_activations[]      → .tip_focus_activations @ 0x34200000 (NOLOAD, 632 KiB)
        ├── g_tip_focus_input[]            → .bss @ 0x34087A00 (main RAM, 150 KiB)
        ├── g_tip_focus_output_tip[]       → .bss (main RAM, 12.5 KiB)
        ├── g_tip_focus_output_center[]    → .bss (main RAM, 12.5 KiB)
        ├── g_tip_focus_ctx                → .bss (main RAM)
        └── g_network_weights_array[]      → .tip_focus_weights @ 0x70C00000 (xSPI2 flash, NOLOAD, 1.9 MiB)
```

No changes to existing scalar/rectifier/OBB/source-crop-box stages. The wrapper compiles to a standalone `.o` file and only appears in the link when `APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE=1U`.

## Memory Placement

### Before fix (build would overflow)

| Region | Capacity | Required | Status |
|--------|----------|----------|--------|
| Internal ROM (0x34000400) | 511 KiB | ~2.0 MiB (weights + text + rodata) | **OVERFLOW** |
| Main RAM (0x34080000) | 512 KiB | ~808 KiB (activations + IO + ctx + existing BSS) | **OVERFLOW** |

### After fix (external regions)

| Data | Size | Placement | Address |
|------|------|-----------|---------|
| `g_network_weights_array` | 1,992,720 bytes | `TIP_FOCUS_WEIGHTS` (xSPI2 flash) | 0x70C00000 |
| `g_tip_focus_activations` | 632,256 bytes | `TIP_FOCUS_RAM` (AXISRAM3+4) | 0x34200000 |
| Input, outputs, context, descriptors | ~28 KiB | Main RAM (`.bss` + `.data`) | 0x34080000+ |

### xSPI2 flash map (after placement)

| Stage | Address | Size |
|-------|---------|------|
| scalar | 0x70200000 | 256K |
| rectifier | 0x70600000 | 512K |
| OBB | 0x70700000 | 4096K |
| source-crop-box | 0x70B00000 | 512K |
| **tip-focus weights** | **0x70C00000** | **4096K** |

### AXISRAM usage (after placement)

| Region | Address | Size | Used by |
|--------|---------|------|---------|
| AXISRAM1 (main RAM) | 0x34080000 | 512K | App `.bss`, `.data`, heap, stacks, **tip-focus IO/ctx** |
| AXISRAM2 (NPU) | 0x34100000 | 1024K | OBB/scalar NPU buffers (untouched) |
| AXISRAM3+4 (**TIP_FOCUS_RAM**) | **0x34200000** | **896K** | **tip-focus activations** (632K used, 264K free) |
| AXISRAM6 (overflow) | 0x34350000 | 448K | Displaced large BSS arrays |

## Key Decisions

1. **Wrapper approach**: `#include` generated `.c` files into a single translation unit (same pattern as existing scalar/OBB wrappers). Avoids linker symbol collisions.

2. **NOLOAD for weights**: The `g_network_weights_array` symbol resolves to 0x70C00000 (xSPI2). NOLOAD prevents the linker from inserting padding bytes between internal ROM and xSPI2 in the ELF binary (would add ~1 GB). Weights are extracted via `objcopy` post-compile and flashed separately.

3. **Existing runtime library works**: `NetworkRuntime1020_CM55_GCC.a` in `Appli/Lib/` resolves all `forward_*` kernel function pointers. No additional lib needed from the newer `NetworkRuntime1200_CM55_GCC.a` shipped with the package.

4. **STAI runtime is header-only**: `forward_lite_*` functions are inline in `stedgeai-lib/Inc/lite_*.h`. No separate `.a`/`.o` runtime — just include path.

5. **Semantic output order**: `GetTipHeatmap()` → output[0], `GetCenterHeatmap()` → output[1] to match generated STAI order. Caller responsible for semantic reversal if needed.

## Build Verification

| Guard | Compiler | Errors | Warnings | Notes |
|-------|----------|--------|----------|-------|
| 0U | WSL `arm-none-eabi-gcc 13.2.1` | 0 | 0 | |
| 1U | WSL `arm-none-eabi-gcc 13.2.1` | 0 | 0 | Needs `-I/` + `/C:` → `/mnt/c` symlink for Windows absolute paths |
| Link | `arm-none-eabi-ld` + `NetworkRuntime1020_CM55_GCC.a` | 0 | 0 | All sections placed at expected addresses |

### Section verification (from link map)

```
.tip_focus_weights     0x70C00000  (NOLOAD) — g_network_weights_array
.tip_focus_activations 0x34200000  size=0x9a5c0 (632,256 bytes)
.rodata                0x340490b4+ (compound literals, layer descriptors — no weights)
.data                  0x34080000  (function pointer tables, layer configs)
.bss                   0x340817f8+ (input, outputs, context, flags)
```

### Weight extraction

```bash
arm-none-eabi-objcopy -O binary --only-section=.rodata.g_network_weights_array \
  ./Src/ai_network_tip_focus_v4_112_int8_enabled.o ./tip_focus_weights.raw
# → 1,992,720 bytes (matches 249,090 × 8)
```

## API Surface

The wrapper exposes 7 functions (matching existing stage pattern):

| Function | Returns | Description |
|----------|---------|-------------|
| `AppAI_TipFocus_Init()` | bool | Initialize STAI network, bind activations/IO |
| `AppAI_TipFocus_Run()` | bool | Run inference on current input buffer |
| `AppAI_TipFocus_GetInputBuffer()` | int8_t* | Get writable input tensor (224×224×3) |
| `AppAI_TipFocus_GetTipHeatmap()` | int8_t* | Get tip heatmap output (112×112) |
| `AppAI_TipFocus_GetCenterHeatmap()` | int8_t* | Get center heatmap output (112×112) |
| `AppAI_TipFocus_GetConfidence()` | int8_t* | Get confidence output (scalar) |
| `AppAI_TipFocus_DryRun()` | void | Self-test: fill input with 0, run, log results |

### Model contract

| Tensor | Shape | DType | Scale | Zero Point |
|--------|-------|-------|-------|------------|
| Input | [1, 224, 224, 3] | int8 | 0.003921569 | -128 |
| Output[0] — tip_heatmap | [1, 112, 112, 1] | int8 | 0.00390625 | -128 |
| Output[1] — center_heatmap | [1, 112, 112, 1] | int8 | 0.00390625 | -128 |
| Output[2] — confidence | [1, 1] | int8 | 0.00390625 | -128 |

## Files

### New files
- `firmware/stm32/n657/Appli/Inc/ai_network_tip_focus_v4_112_int8.h` — wrapper header
- `firmware/stm32/n657/Appli/Src/ai_network_tip_focus_v4_112_int8.c` — wrapper source

### Modified files
- `firmware/stm32/n657/Appli/makefile.targets` — compile rule, weight extraction, cleanup targets
- `firmware/stm32/n657/Appli/STM32N657X0HXQ_LRUN.ld` — TIP_FOCUS_WEIGHTS + TIP_FOCUS_RAM regions + output sections
- `firmware/stm32/n657/flash_boot.bat` — tip_focus_weights.raw flash step at 0x70C00000

### Untouched
- `app_ai.c` / `app_ai.h` — no changes (guard is 0U by default)
- All existing stage wrappers (scalar, rectifier, OBB, source-crop-box)
- All generated STAI files (`network.c`, `network.h`, `network_data.c`)
- `flash_boot.bat` adds a warning-only check (non-fatal if weights missing)

## WSL Build Workaround

The project headers include Windows absolute paths like:
```c
#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/..."
```

These fail on WSL GCC. The workaround is:
```bash
# Add to compiler flags:
-I/

# Create symlink:
sudo ln -s /mnt/c /C
```

This is a dev-environment issue only — the Windows-native CubeIDE build resolves these paths natively.

## Remaining Before Board Bring-Up

1. CubeIDE build test (Windows native) — verify 0 errors both 0U and 1U
2. Flash `tip_focus_weights.raw` at 0x70C00000 via `flash_boot.bat`
3. Enable guard (`APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE=1U`)
4. Wire `AppAI_TipFocus_DryRun()` into `app_ai.c` or a test harness
5. Verify UART logs show tip-focus init, inference run, and heatmap min/max/confidence
6. Map-file overlap check with existing scalar/rectifier/OBB/source-crop-box regions
