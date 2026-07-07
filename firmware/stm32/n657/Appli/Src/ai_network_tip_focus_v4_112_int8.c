/* Thin build wrapper for the board-fit compact geometry 224 NPU network.
 *
 * The generated package exposes a quantized-internal contract with float32
 * I/O at the firmware boundary:
 *   - output[0] center heatmap   [1,56,56,1]
 *   - output[1] confidence       [1,1]
 *   - output[2] tip heatmap      [1,56,56,1]
 *   - output[3] is_main_needle   [1,1]
 *
 * We keep the firmware-facing API stable by reusing the existing wrapper
 * filename while swapping the underlying generated package to tip_focus_v18.
 * No HyperRAM needed; 1.72 MiB on-chip activations, 815 KB xSPI2 weights.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif
#ifndef LL_ATON_RT_RELOC
#define LL_ATON_RT_RELOC 1
#endif
#ifdef MODEL_CONF
#undef MODEL_CONF
#endif
#define MODEL_CONF "../../st_ai_output/packages/tip_focus_v18_int8_n6_npu/st_ai_ws/build_tip_focus_v18_int8/tip_focus_v18_int8_reloc_conf.h"

#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_aton_reloc_network.h"
#include "debug_console.h"
#include "npu_cache.h"
#include "mcu_cache.h"

#include <string.h>

extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);
extern bool AppAI_VerifyTipFocusWeights(void);
extern const LL_Buffer_InfoTypeDef *LL_ATON_Input_Buffers_Info_tip_focus_v18_int8(void);
extern const LL_Buffer_InfoTypeDef *LL_ATON_Output_Buffers_Info_tip_focus_v18_int8(void);
extern const LL_Buffer_InfoTypeDef *LL_ATON_Internal_Buffers_Info_tip_focus_v18_int8(void);

#include "../../st_ai_output/packages/tip_focus_v18_int8_n6_npu/st_ai_output/tip_focus_v18_int8.h"

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(tip_focus_v18_int8);

#include "../../st_ai_output/packages/tip_focus_v18_int8_n6_npu/st_ai_ws/build_tip_focus_v18_int8/tip_focus_v18_int8_reloc.c"

static bool app_ai_tip_focus_outputs_valid = false;
static bool app_ai_tip_focus_logged_compiled_in_runtime = false;
static bool app_ai_tip_focus_logged_heatmap_source = false;
static bool app_ai_tip_focus_logged_missing_internal_buffer = false;
static bool app_ai_tip_focus_logged_confidence_source = false;
static float app_ai_tip_focus_center_heatmap_fallback[3136U];
static float app_ai_tip_focus_tip_heatmap_fallback[3136U];

/* xSPI2 base address of the tip-focus model weight blob. Mirrors
 * `APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR` from `app_ai.c`; duplicated here
 * because the wrapper does not see that translation unit. The value is
 * fixed by the linker script (`STM32N657X0HXQ_LRUN.ld` `TIP_FOCUS_WEIGHTS`
 * region at 0x70400000) and the `flash_boot.ps1` flash target. */
#define APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR 0x70400000UL

/* `ll_aton_reloc_network.c` references `LL_ATON_Cache_NPU_Invalidate` (the
 * full-cache invalidate, not the address-range variant), but the X-CUBE-AI
 * `ll_aton_reloc_callbacks.c` only implements the two range variants. Provide
 * the missing wrapper here so the reloc link resolves; the implementation
 * just delegates to the project's existing NPU cache helper. */
void LL_ATON_Cache_NPU_Invalidate(void)
{
    npu_cache_invalidate();
}

/* When `BUILD_AI_NETWORK_RELOC=1` is defined, the ATON cache-interface
 * header (`ll_aton_caches_interface.h`) changes the MCU/NPU cache helpers
 * from `static inline` wrappers to plain extern declarations. The per-model
 * build rule for the tip-focus now sets that define so `_network_rt_ctx` is
 * emitted, which means the model's SW quantize/dequantize epoch blocks
 * expect these functions to be resolved at link time. We provide thin
 * implementations that delegate to the project's existing cache primitives
 * and drop the callback layer — the callback bridge in `ll_aton_reloc_
 * callbacks.c` duplicates symbols already defined in `ll_aton_lib.c`, so
 * we avoid building that file entirely and define only what the reloc
 * runtime actually needs. */
void LL_ATON_Cache_MCU_Clean_Range(uintptr_t virtual_addr, uint32_t size)
{
    (void)mcu_cache_clean_range((uint32_t)virtual_addr,
                                 (uint32_t)(virtual_addr + size));
}

void LL_ATON_Cache_MCU_Invalidate_Range(uintptr_t virtual_addr, uint32_t size)
{
    (void)mcu_cache_invalidate_range((uint32_t)virtual_addr,
                                      (uint32_t)(virtual_addr + size));
}

void LL_ATON_Cache_MCU_Clean_Invalidate_Range(uintptr_t virtual_addr, uint32_t size)
{
    (void)mcu_cache_clean_invalidate_range((uint32_t)virtual_addr,
                                            (uint32_t)(virtual_addr + size));
}

void LL_ATON_Cache_NPU_Clean_Range(uintptr_t virtual_addr, uint32_t size)
{
    npu_cache_clean_range((uint32_t)virtual_addr,
                           (uint32_t)(virtual_addr + size));
}

void LL_ATON_Cache_NPU_Clean_Invalidate_Range(uintptr_t virtual_addr, uint32_t size)
{
    npu_cache_clean_invalidate_range((uint32_t)virtual_addr,
                                      (uint32_t)(virtual_addr + size));
}

/* RAM region for the tip-focus model's data/bss segment.
 *
 * The tip-focus package is relocatable and its runtime base must point at
 * the on-chip virtual pool start that the compiler targeted, not the xSPI2
 * flash address of the weight blob. The xSPI2 address is still tracked in
 * `file_addr` so the runtime can verify the flashed parameter blob, but the
 * live `ram_addr` value must reflect the AXISRAM window where the model's
 * relocated tables and activations live.
 *
 * The current v18 package uses the same 0x3410_0000 virtual pool base as the
 * other relocatable N6 models, so keep the runtime base aligned with that
 * region when re-installing the context after `LL_ATON_RT_Init_Network()`. */
__attribute__((aligned(32)))
static uint8_t app_ai_tip_focus_reloc_data[65536U]
    __attribute__((section(".tip_focus_activations")));

/* The per-model `AI_RELOC_NETWORK()` macro emits `_network_rt_ctx` into
 * the `.network_rt_ctx` linker section. `LL_ATON_RT_Init_Network` resets
 * `inst_reloc` to zero on every call, so this small helper re-wires the
 * per-model context into the NN instance after each init.
 *
 * For this package, `ram_addr` is the relocatable runtime base in on-chip
 * AXISRAM (the generated reloc mempool uses the 0x3410_0000 vpool base and
 * spans AXISRAM2..AXISRAM6). The xSPI2 address only identifies the flashed
 * parameter blob; the SW quantize/dequantize ops still read their tables
 * through the reloc runtime base. `app_ai_tip_focus_reloc_data` remains a
 * small on-chip scratch area for firmware-side staging, not the model's
 * reloc base itself. */

static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectInputInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectOutputInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectInternalInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_FindBufferByName(
    const LL_Buffer_InfoTypeDef *buffer_info,
    const char *name);

static bool AppAI_TipFocus_InstallRelocContext(void)
{
    /* The `AI_RELOC_NETWORK()` macro already set `.c_name`, `.acts_sz`,
     * `.params_sz`, etc. on `_network_rt_ctx`. Repoint the runtime base at
     * the on-chip pool the package was compiled for, keep the xSPI2 flash
     * address in `file_addr`, and wire the struct into the NN instance's
     * `inst_reloc` so the SW epoch blocks can restore the correct `r9`
     * value after each `LL_ATON_RT_Init_Network()` call. */
    _network_rt_ctx.ram_addr  = 0x34100000UL;
    _network_rt_ctx.file_addr = APP_AI_XSPI2_TIP_FOCUS_BASE_ADDR;
    _network_rt_ctx.state     = AI_RELOC_RT_STATE_INITIALIZED |
                                AI_RELOC_RT_STATE_XIP_MODE;

    NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc =
        (uint32_t)(uintptr_t)&_network_rt_ctx;

    return true;
}

static void AppAI_TipFocus_InvalidateBufferRange(
    const LL_Buffer_InfoTypeDef *buffer_info)
{
    if (buffer_info == NULL) {
        return;
    }

    const uintptr_t start = (uintptr_t)LL_Buffer_addr_start(buffer_info);
    const uintptr_t end = start + (uintptr_t)LL_Buffer_len(buffer_info);

    if ((start == 0U) || (end <= start)) {
        return;
    }

    (void)mcu_cache_invalidate_range((uint32_t)start, (uint32_t)end);
}

static void AppAI_TipFocus_InvalidateNamedInternalBuffer(
    const LL_Buffer_InfoTypeDef *internal_info,
    const char *name)
{
    const LL_Buffer_InfoTypeDef *buffer_info =
        AppAI_TipFocus_FindBufferByName(internal_info, name);

    if (buffer_info == NULL) {
        if (!app_ai_tip_focus_logged_missing_internal_buffer) {
            DebugConsole_Printf(
                "[AI][TIP_FOCUS] internal buffer missing during invalidate: %s\r\n",
                (name != NULL) ? name : "(null)");
            app_ai_tip_focus_logged_missing_internal_buffer = true;
        }
        return;
    }

    AppAI_TipFocus_InvalidateBufferRange(buffer_info);
}

static void AppAI_TipFocus_InvalidateOutputs(
    const LL_Buffer_InfoTypeDef *output_info)
{
    const LL_Buffer_InfoTypeDef *internal_info =
        AppAI_TipFocus_GetDirectInternalInfo();

    if (output_info == NULL) {
        goto invalidate_internal;
    }

    AppAI_TipFocus_InvalidateBufferRange(&output_info[0]);
    AppAI_TipFocus_InvalidateBufferRange(&output_info[1]);
    AppAI_TipFocus_InvalidateBufferRange(&output_info[2]);
    AppAI_TipFocus_InvalidateBufferRange(&output_info[3]);

invalidate_internal:
    AppAI_TipFocus_InvalidateNamedInternalBuffer(internal_info, "Sigmoid_227_out_0");
    AppAI_TipFocus_InvalidateNamedInternalBuffer(internal_info, "Sigmoid_241_out_0");
}

static uintptr_t AppAI_TipFocus_GetRuntimeR9(void)
{
    if (NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc != 0U)
    {
        const struct ai_reloc_rt_ctx *rt_ctx =
            (const struct ai_reloc_rt_ctx *)(uintptr_t)NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc;
        if ((rt_ctx != NULL) && (rt_ctx->ram_addr != 0U))
        {
            return (uintptr_t)rt_ctx->ram_addr;
        }
    }

    return 0U;
}

static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectInputInfo(void)
{
    return LL_ATON_Input_Buffers_Info_tip_focus_v18_int8();
}

static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectOutputInfo(void)
{
    return LL_ATON_Output_Buffers_Info_tip_focus_v18_int8();
}

static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_GetDirectInternalInfo(void)
{
    return LL_ATON_Internal_Buffers_Info_tip_focus_v18_int8();
}

static const LL_Buffer_InfoTypeDef *AppAI_TipFocus_FindBufferByName(
    const LL_Buffer_InfoTypeDef *buffer_info,
    const char *name)
{
    if ((buffer_info == NULL) || (name == NULL)) {
        return NULL;
    }

    while (buffer_info->name != NULL) {
        if (strcmp(buffer_info->name, name) == 0) {
            return buffer_info;
        }
        buffer_info++;
    }

    return NULL;
}

static bool AppAI_TipFocus_IsAllZeroFloatBuffer(
    const float *buffer,
    size_t element_count)
{
    if ((buffer == NULL) || (element_count == 0U)) {
        return true;
    }

    for (size_t index = 0U; index < element_count; ++index) {
        if (buffer[index] != 0.0f) {
            return false;
        }
    }

    return true;
}

static const float *AppAI_TipFocus_GetFloatBufferFromInfo(
    const LL_Buffer_InfoTypeDef *buffer_info)
{
    if (buffer_info == NULL) {
        return NULL;
    }
    if (LL_Buffer_addr_start(buffer_info) == NULL) {
        return NULL;
    }
    return (const float *)LL_Buffer_addr_start(buffer_info);
}

static const float *AppAI_TipFocus_DequantizeHeatmapFallback(
    const LL_Buffer_InfoTypeDef *buffer_info,
    float *scratch_buffer,
    size_t element_count,
    const char *label)
{
    const int8_t *source = NULL;
    float scale = 0.0f;
    int32_t zero_point = 0;

    if ((buffer_info == NULL) || (scratch_buffer == NULL) || (element_count == 0U)) {
        return NULL;
    }
    if ((buffer_info->scale == NULL) || (buffer_info->offset == NULL)) {
        return NULL;
    }
    if (LL_Buffer_len(buffer_info) < element_count) {
        return NULL;
    }

    source = (const int8_t *)LL_Buffer_addr_start(buffer_info);
    if (source == NULL) {
        return NULL;
    }

    scale = buffer_info->scale[0];
    zero_point = (int32_t)buffer_info->offset[0];
    for (size_t index = 0U; index < element_count; ++index) {
        scratch_buffer[index] = scale * ((float)((int32_t)source[index] - zero_point));
    }

    DebugConsole_Printf(
        "[AI][TIP_FOCUS] using %s quantized fallback %s addr=%p len=%lu scale=%ld zp=%ld\r\n",
        (label != NULL) ? label : "heatmap",
        (buffer_info->name != NULL) ? buffer_info->name : "(null)",
        (void *)LL_Buffer_addr_start(buffer_info),
        (unsigned long)LL_Buffer_len(buffer_info),
        (long)(scale * 1000000.0f),
        (long)zero_point);
    return scratch_buffer;
}

static float AppAI_TipFocus_DequantizeScalarFallback(
    const LL_Buffer_InfoTypeDef *buffer_info)
{
    const int8_t *source = NULL;
    float scale = 0.0f;
    int32_t zero_point = 0;
    float value = 0.0f;

    if (buffer_info == NULL) {
        return 0.0f;
    }
    if ((buffer_info->scale == NULL) || (buffer_info->offset == NULL)) {
        return 0.0f;
    }
    if (LL_Buffer_len(buffer_info) < 1U) {
        return 0.0f;
    }

    source = (const int8_t *)LL_Buffer_addr_start(buffer_info);
    if (source == NULL) {
        return 0.0f;
    }

    scale = buffer_info->scale[0];
    zero_point = (int32_t)buffer_info->offset[0];
    value = scale * ((float)((int32_t)source[0] - zero_point));
    if (!app_ai_tip_focus_logged_confidence_source) {
        DebugConsole_Printf(
            "[AI][TIP_FOCUS] confidence source: quantized scalar fallback %s addr=%p len=%lu scale=%ld zp=%ld value=%ld\r\n",
            (buffer_info->name != NULL) ? buffer_info->name : "(null)",
            (void *)LL_Buffer_addr_start(buffer_info),
            (unsigned long)LL_Buffer_len(buffer_info),
            (long)(scale * 1000000.0f),
            (long)zero_point,
            (long)lroundf(value * 1000.0f));
        app_ai_tip_focus_logged_confidence_source = true;
    }
    return value;
}

static const float *AppAI_TipFocus_GetQuantizedHeatmapPrimary(
    const LL_Buffer_InfoTypeDef *internal_info,
    size_t output_index)
{
    const char *sigmoid_name = NULL;
    float *scratch = NULL;
    const LL_Buffer_InfoTypeDef *buffer_info = NULL;
    const char *label = NULL;

    if (output_index == 0U) {
        sigmoid_name = "Sigmoid_227_out_0";
        scratch = app_ai_tip_focus_center_heatmap_fallback;
        label = "center";
    } else if (output_index == 2U) {
        sigmoid_name = "Sigmoid_241_out_0";
        scratch = app_ai_tip_focus_tip_heatmap_fallback;
        label = "tip";
    } else {
        return NULL;
    }

    buffer_info = AppAI_TipFocus_FindBufferByName(internal_info, sigmoid_name);
    return AppAI_TipFocus_DequantizeHeatmapFallback(
        buffer_info,
        scratch,
        3136U,
        label);
}

static const float *AppAI_TipFocus_ResolveHeatmapBuffer(
    size_t output_index)
{
    const LL_Buffer_InfoTypeDef *output_info = AppAI_TipFocus_GetDirectOutputInfo();
    const LL_Buffer_InfoTypeDef *internal_info = AppAI_TipFocus_GetDirectInternalInfo();
    const float *buffer = NULL;
    const char *sigmoid_name = NULL;
    float *scratch = NULL;

    if (output_index == 0U) {
        sigmoid_name = "Sigmoid_227_out_0";
        scratch = app_ai_tip_focus_center_heatmap_fallback;
    } else if (output_index == 2U) {
        sigmoid_name = "Sigmoid_241_out_0";
        scratch = app_ai_tip_focus_tip_heatmap_fallback;
    } else {
        return NULL;
    }

    /* Prefer the quantized sigmoid buffers and dequantize them on CPU.
     *
     * These tensors sit immediately before the final transpose stage and
     * preserve the 56x56 probability maps we actually need for decode.
     * Because the heatmap channel dimension is 1, the later transpose does
     * not change flattened order, so the CPU-dequantized buffer is spatially
     * equivalent while avoiding the board-local float output path. */
    buffer = AppAI_TipFocus_GetQuantizedHeatmapPrimary(internal_info, output_index);
    if (buffer != NULL) {
        if (!app_ai_tip_focus_logged_heatmap_source) {
            DebugConsole_WriteString(
                "[AI][TIP_FOCUS] heatmap source: quantized sigmoid fallback primary.\r\n");
            app_ai_tip_focus_logged_heatmap_source = true;
        }
        return buffer;
    }

    if ((output_info != NULL) &&
        (output_index < (size_t)LL_ATON_TIP_FOCUS_V18_INT8_OUT_NUM)) {
        buffer = AppAI_TipFocus_DequantizeHeatmapFallback(
            &output_info[output_index],
            scratch,
            3136U,
            (output_index == 0U) ? "center" : "tip");
        if ((buffer != NULL) && !AppAI_TipFocus_IsAllZeroFloatBuffer(buffer, 3136U)) {
            return buffer;
        }
    }

    return AppAI_TipFocus_DequantizeHeatmapFallback(
        AppAI_TipFocus_FindBufferByName(internal_info, sigmoid_name),
        scratch,
        3136U,
        (output_index == 0U) ? "center" : "tip");
}

static const float *AppAI_TipFocus_GetOutputBuffer(size_t output_index)
{
    const LL_Buffer_InfoTypeDef *output_info =
        AppAI_TipFocus_GetDirectOutputInfo();

    if (!app_ai_tip_focus_outputs_valid || (output_info == NULL)) {
        return NULL;
    }
    if (output_index >= (size_t)LL_ATON_TIP_FOCUS_V18_INT8_OUT_NUM) {
        return NULL;
    }
    if (LL_Buffer_addr_start(&output_info[output_index]) == NULL) {
        return NULL;
    }

    DebugConsole_Printf(
        "[AI][TIP_FOCUS] output[%lu] name=%s addr=%p len=%lu bits=%lu type=%lu\r\n",
        (unsigned long)output_index,
        (output_info[output_index].name != NULL) ? output_info[output_index].name : "(null)",
        (void *)LL_Buffer_addr_start(&output_info[output_index]),
        (unsigned long)LL_Buffer_len(&output_info[output_index]),
        (unsigned long)output_info[output_index].nbits,
        (unsigned long)output_info[output_index].type);

    return (const float *)LL_Buffer_addr_start(&output_info[output_index]);
}

bool AppAI_TipFocus_Init(void)
{
    if (!AppAI_VerifyTipFocusWeights()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Init aborted: xSPI2 weights not valid.\r\n");
        return false;
    }

    if (!AppAI_Xspi2EnsureMemoryMappedMode()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Init aborted: xSPI2 MM mode failed.\r\n");
        return false;
    }

    if (!LL_ATON_EC_Network_Init_tip_focus_v18_int8()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Init aborted: network init failed.\r\n");
        return false;
    }

    LL_ATON_RT_Init_Network(&NN_Instance_tip_focus_v18_int8);
    {
        /* Wire the per-model `_network_rt_ctx` into the NN instance so
         * the SW quantize/dequantize ops read `ram_addr` from a valid
         * `ai_reloc_rt_ctx` at `[r9 + 8]`. The context is re-wired again
         * inside `AppAI_TipFocus_Run()` after every `LL_ATON_RT_Init_
         * Network()` call. */
        (void)AppAI_TipFocus_InstallRelocContext();

        const struct ai_reloc_rt_ctx *init_rt_ctx = NULL;
        uintptr_t init_ram_addr = 0U;
        uintptr_t live_r9 = 0U;
        if (NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc != 0U) {
            init_rt_ctx = (const struct ai_reloc_rt_ctx *)(uintptr_t)
                NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc;
            if (init_rt_ctx != NULL) {
                init_ram_addr = (uintptr_t)init_rt_ctx->ram_addr;
            }
        }
        __asm volatile("mov %0, r9" : "=r"(live_r9));
        DebugConsole_Printf(
            "[AI][TIP_FOCUS] init: inst_reloc=%p ram_addr=%p live_r9=%p xspi2=%p\r\n",
            (const void *)init_rt_ctx,
            (const void *)init_ram_addr,
            (const void *)live_r9,
            (const void *)_mem_pool_xSPI2_tip_focus_v18_int8);
    }

    app_ai_tip_focus_outputs_valid = false;
    return true;
}

bool AppAI_TipFocus_Run(void)
{
    uintptr_t caller_r9 = 0U;
    const LL_Buffer_InfoTypeDef *input_info = NULL;
    const LL_Buffer_InfoTypeDef *output_info = NULL;
    const uint8_t *input_ptr = NULL;
    size_t input_len = 0U;

    __asm volatile("mov %0, r9" : "=r"(caller_r9));

    if (!AppAI_Xspi2EnsureMemoryMappedMode()) {
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    if (!LL_ATON_EC_Network_Init_tip_focus_v18_int8()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Network init failed before run.\r\n");
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    LL_ATON_RT_Init_Network(&NN_Instance_tip_focus_v18_int8);
    /* `LL_ATON_RT_Init_Network` resets `inst_reloc` to zero, so the
     * reloc context must be re-installed here before every inference. */
    if (!AppAI_TipFocus_InstallRelocContext()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Reloc re-install before run failed.\r\n");
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    if (!LL_ATON_EC_Inference_Init_tip_focus_v18_int8()) {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Inference init failed.\r\n");
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    input_info = AppAI_TipFocus_GetDirectInputInfo();
    output_info = AppAI_TipFocus_GetDirectOutputInfo();
    if ((input_info == NULL) || (output_info == NULL)) {
        DebugConsole_WriteString("[AI][TIP_FOCUS] Buffer info unavailable.\r\n");
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    input_ptr = (const uint8_t *)LL_Buffer_addr_start(input_info);
    input_len = (size_t)LL_Buffer_len(input_info);
    if ((input_ptr == NULL) || (input_len == 0U)) {
        DebugConsole_WriteString("[AI][TIP_FOCUS] Bad input buffer.\r\n");
        __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
        return false;
    }

    (void)mcu_cache_clean_range(
        (uint32_t)(uintptr_t)input_ptr,
        (uint32_t)((uintptr_t)input_ptr + input_len));

#if APP_AI_RESET_NETWORK_EACH_INFERENCE
    LL_ATON_RT_Reset_Network(&NN_Instance_tip_focus_v18_int8);
#endif

    {
        LL_ATON_RT_RetValues_t status = LL_ATON_RT_DONE;
        const struct ai_reloc_rt_ctx *rt_ctx = NULL;
        /* Always start the epoch loop from the model's own reloc base when the
         * installed instance reports one. When it does not, fall back to the
         * live r9 (e.g. carried over from the OBB logging pass) so the SW
         * dequantize path still sees a valid ai_array pointer. Doing this once
         * up front means we never have to branch on the runtime base inside
         * the hot loop, and the loop can restore r9 unconditionally before
         * every epoch block and again after each LL_ATON_OSAL_WFE(). */
        uintptr_t runtime_r9 = AppAI_TipFocus_GetRuntimeR9();

        if (NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc != 0U) {
            rt_ctx =
                (const struct ai_reloc_rt_ctx *)(uintptr_t)NN_Instance_tip_focus_v18_int8.exec_state.inst_reloc;
        }

        if (runtime_r9 == 0U) {
            __asm volatile("mov %0, r9" : "=r"(runtime_r9));
        }

        if (!app_ai_tip_focus_logged_compiled_in_runtime) {
            DebugConsole_Printf(
                "[AI][TIP_FOCUS] run base: inst_reloc=%p ram_addr=%p live_r9=%p\r\n",
                (const void *)rt_ctx,
                (const void *)((rt_ctx != NULL) ? (uintptr_t)rt_ctx->ram_addr : 0U),
                (const void *)runtime_r9);
            app_ai_tip_focus_logged_compiled_in_runtime = true;
        }

        for (;;) {
            __asm volatile("mov r9, %0" ::"r"(runtime_r9) : "r9");
            status = LL_ATON_RT_RunEpochBlock(&NN_Instance_tip_focus_v18_int8);
            if (status == LL_ATON_RT_DONE) {
                break;
            }
            if (status == LL_ATON_RT_WFE) {
                LL_ATON_OSAL_WFE();
                __asm volatile("mov r9, %0" ::"r"(runtime_r9) : "r9");
                continue;
            }
            if (status == LL_ATON_RT_NO_WFE) {
                continue;
            }

            DebugConsole_Printf(
                "[AI][TIP_FOCUS] Unexpected runtime status=%d\r\n",
                (int)status);
            __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
            return false;
        }
    }

    AppAI_TipFocus_InvalidateOutputs(output_info);
    app_ai_tip_focus_outputs_valid = true;
    DebugConsole_Printf(
        "[AI][TIP_FOCUS] outputs ready: 0=%s@%p 1=%s@%p 2=%s@%p 3=%s@%p\r\n",
        (output_info[0].name != NULL) ? output_info[0].name : "(null)",
        (void *)LL_Buffer_addr_start(&output_info[0]),
        (output_info[1].name != NULL) ? output_info[1].name : "(null)",
        (void *)LL_Buffer_addr_start(&output_info[1]),
        (output_info[2].name != NULL) ? output_info[2].name : "(null)",
        (void *)LL_Buffer_addr_start(&output_info[2]),
        (output_info[3].name != NULL) ? output_info[3].name : "(null)",
        (void *)LL_Buffer_addr_start(&output_info[3]));

    __asm volatile("mov r9, %0" ::"r"(caller_r9) : "r9");
    return true;
}

float *AppAI_TipFocus_GetInputBuffer(void)
{
    const LL_Buffer_InfoTypeDef *info = AppAI_TipFocus_GetDirectInputInfo();

    if (info == NULL) {
        return NULL;
    }
    return (float *)LL_Buffer_addr_start(info);
}

const void *AppAI_TipFocus_GetInputBufferInfo(void)
{
    return (const void *)AppAI_TipFocus_GetDirectInputInfo();
}

const float *AppAI_TipFocus_GetCenterHeatmap(void)
{
    return AppAI_TipFocus_ResolveHeatmapBuffer(0U);
}

const float *AppAI_TipFocus_GetTipHeatmap(void)
{
    return AppAI_TipFocus_ResolveHeatmapBuffer(2U);
}

float AppAI_TipFocus_GetConfidence(void)
{
    const LL_Buffer_InfoTypeDef *output_info = AppAI_TipFocus_GetDirectOutputInfo();
    float confidence_value = 0.0f;

    if (output_info != NULL) {
        confidence_value = AppAI_TipFocus_DequantizeScalarFallback(&output_info[1]);
        if (isfinite(confidence_value)) {
            return confidence_value;
        }
    }

    return confidence_value;
}

float AppAI_TipFocus_GetIsMainNeedle(void)
{
    const LL_Buffer_InfoTypeDef *output_info = AppAI_TipFocus_GetDirectOutputInfo();
    float is_main_value = 0.0f;

    if (output_info != NULL) {
        is_main_value = AppAI_TipFocus_DequantizeScalarFallback(&output_info[3]);
        if (isfinite(is_main_value)) {
            return is_main_value;
        }
    }

    return 0.0f;
}

bool AppAI_TipFocus_DryRun(void)
{
    float *input = AppAI_TipFocus_GetInputBuffer();
    const LL_Buffer_InfoTypeDef *info =
        (const LL_Buffer_InfoTypeDef *)AppAI_TipFocus_GetInputBufferInfo();
    size_t input_float_count = 0U;

    if ((input == NULL) || (info == NULL)) {
        return false;
    }

    input_float_count = (size_t)LL_Buffer_len(info) / sizeof(float);
    if (input_float_count == 0U) {
        return false;
    }

    for (size_t i = 0U; i < input_float_count; ++i) {
        input[i] = 0.0f;
    }

    return AppAI_TipFocus_Run();
}
