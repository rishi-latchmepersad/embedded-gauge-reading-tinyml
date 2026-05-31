/* Thin build wrapper for tip_focus_v4_112_int8 NPU network.
 *
 * The generated network.c is #included directly so the NPU scheduling
 * code and the _Default-suffixed buffer-info functions are compiled
 * into this translation unit.
 *
 * This file also defines the AppAI_TipFocus_* API that app_ai.c calls
 * to initialise, run, and read outputs from the tip-focus model.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LL_ATON_DBG_BUFFER_INFO_EXCLUDED
#define LL_ATON_DBG_BUFFER_INFO_EXCLUDED 1
#endif

#include "ll_aton_NN_interface.h"
#include "ll_aton_rt_user_api.h"
#include "ll_sw.h"
#include "debug_console.h"
#include "npu_cache.h"
#include <string.h>

/* Forward declaration for xSPI2 memory-mapped mode guard.
 * Defined in app_ai.c.  Called before inference to ensure the
 * tip-focus weight blob at 0x70400000 is visible to the CPU. */
extern bool AppAI_Xspi2EnsureMemoryMappedMode(void);
extern bool AppAI_VerifyTipFocusWeights(void);

/* Log the generated resize metadata right before ST's helper runs.
 * This keeps the diagnostic noise focused on the four resize nodes that
 * currently fault, without changing the model execution path. */
static void AppAI_TipFocus_LogResizeNode(const char *node_name,
                                         const Resize_integer_sw_info *sw_info)
{
    uintptr_t r9_before = 0U;

    if ((node_name == NULL) || (sw_info == NULL)) {
        DebugConsole_Printf("[AI][TIP_FOCUS][RESIZE] missing node metadata: name=%p info=%p\r\n",
                            (const void *)node_name, (const void *)sw_info);
        return;
    }

    /* Capture r9 at the call site before any formatting work can perturb it. */
    __asm volatile("mov %0, r9" : "=r"(r9_before));

    DebugConsole_Printf(
        "[AI][TIP_FOCUS][RESIZE] %s regs: r9=%p sw=%p\r\n",
        node_name,
        (const void *)r9_before,
        (const void *)sw_info);

    DebugConsole_Printf(
        "[AI][TIP_FOCUS][RESIZE] %s dims: in=%lu,%lu,%lu,%lu out=%lu,%lu,%lu,%lu mode=%u nmode=%u ctm=%u\r\n",
        node_name,
        (unsigned long)sw_info->general.input.dim.tensor_b,
        (unsigned long)sw_info->general.input.dim.tensor_h,
        (unsigned long)sw_info->general.input.dim.tensor_w,
        (unsigned long)sw_info->general.input.dim.tensor_c,
        (unsigned long)sw_info->general.output.dim.tensor_b,
        (unsigned long)sw_info->general.output.dim.tensor_h,
        (unsigned long)sw_info->general.output.dim.tensor_w,
        (unsigned long)sw_info->general.output.dim.tensor_c,
        (unsigned int)sw_info->mode,
        (unsigned int)sw_info->nearest_mode,
        (unsigned int)sw_info->coord_transf_mode);

    DebugConsole_Printf(
        "[AI][TIP_FOCUS][RESIZE] %s mem: in=%p out=%p scales=%p(%lu) roi=%p(%lu) sizes=%p(%lu) is=%p izp=%p os=%p ozp=%p\r\n",
        node_name,
        (const void *)sw_info->general.input.mem.start_offset,
        (const void *)sw_info->general.output.mem.start_offset,
        (const void *)sw_info->scales.mem.start_offset,
        (unsigned long)sw_info->scales.dim.num_elem,
        (const void *)sw_info->roi.mem.start_offset,
        (unsigned long)sw_info->roi.dim.num_elem,
        (const void *)sw_info->sizes.mem.start_offset,
        (unsigned long)sw_info->sizes.dim.num_elem,
        (const void *)sw_info->is.mem.start_offset,
        (const void *)sw_info->izp.mem.start_offset,
        (const void *)sw_info->os.mem.start_offset,
        (const void *)sw_info->ozp.mem.start_offset);

    /* Preserve the incoming r9 value for the downstream ST runtime. */
    __asm volatile("mov r9, %0" : : "r"(r9_before) : "r9");
}

/* Log the live r9 value at a small number of inference checkpoints.
 * The resize helper uses r9 as an internal ai_array base, so this lets us
 * locate the first call site that hands it a bad register value. */
static void AppAI_TipFocus_LogR9(const char *label)
{
    uintptr_t r9_before = 0U;

    if (label == NULL) {
        return;
    }

    __asm volatile("mov %0, r9" : "=r"(r9_before));
    DebugConsole_Printf("[AI][TIP_FOCUS][R9] %s=%p\r\n",
                        label, (const void *)r9_before);
    __asm volatile("mov r9, %0" : : "r"(r9_before) : "r9");
}

/* Declare and create the NN interface + instance for the Default network.
 * The _Default-suffixed functions (LL_ATON_EC_Network_Init_Default, ...)
 * are provided by the included network.c below.
 */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default);

#include "../../st_ai_output/packages/tip_focus_v4_112_int8_n6_npu/st_ai_output/network.c"

/* ---------------------------------------------------------------------------
 * AppAI_TipFocus_* public API
 * -------------------------------------------------------------------------*/

/**
 * @brief Initialize the tip-focus geometry network.
 * @retval true Initialization succeeded and xSPI2 weights are valid.
 * @retval false Initialization failed or xSPI2 flash not programmed.
 */
bool AppAI_TipFocus_Init(void)
{
    /* Verify that the xSPI2 flash has been programmed with tip-focus weights
     * before initializing the network. This prevents HardFaults during
     * inference when the flash is empty. */
    if (!AppAI_VerifyTipFocusWeights())
    {
        DebugConsole_WriteString(
            "[AI][TIP_FOCUS] Init aborted: xSPI2 weights not valid.\r\n");
        return false;
    }

    LL_ATON_RT_Init_Network(&NN_Instance_Default);
    return true;
}

bool AppAI_TipFocus_Run(void)
{
    /* Capture the incoming r9 before any helper calls inside this wrapper
     * can potentially disturb the register state. */
    AppAI_TipFocus_LogR9("entry");

    const LL_Buffer_InfoTypeDef *input_info =
        LL_ATON_Input_Buffers_Info(&NN_Instance_Default);
    const LL_Buffer_InfoTypeDef *output_info =
        LL_ATON_Output_Buffers_Info(&NN_Instance_Default);
    const uint8_t *input_ptr = NULL;
    const uint8_t *output_ptr = NULL;
    size_t input_len = 0U;
    size_t output_len = 0U;

    if ((input_info == NULL) || (output_info == NULL)) {
        return false;
    }

    if (!AppAI_Xspi2EnsureMemoryMappedMode()) {
        return false;
    }

    /* Check the post-xSPI2 state before any buffer-info helpers run. */
    AppAI_TipFocus_LogR9("post_xspi2");

    /* Keep this wrapper aligned with the generic stage runner: make sure
     * the CPU's writes are visible to the NPU, and only reset when the build
     * explicitly opts into per-inference resets. */
    input_ptr = (const uint8_t *)LL_Buffer_addr_start(input_info);
    output_ptr = (const uint8_t *)LL_Buffer_addr_start(output_info);
    input_len = (size_t)LL_Buffer_len(input_info);
    output_len = (size_t)LL_Buffer_len(output_info);

    if ((input_ptr == NULL) || (output_ptr == NULL) ||
        (input_len == 0U) || (output_len == 0U)) {
        DebugConsole_Printf(
            "[AI][TIP_FOCUS] Bad buffers: in=%p out=%p in_len=%lu out_len=%lu\r\n",
            (const void *)input_ptr, (const void *)output_ptr,
            (unsigned long)input_len, (unsigned long)output_len);
        return false;
    }

    /* Preserve r9 across the last pre-inference console log as a guard
     * against any variadic logger implementation that might not respect
     * the callee-saved register contract. */
    uintptr_t r9_before_log = 0U;
    __asm volatile("mov %0, r9" : "=r"(r9_before_log));
    DebugConsole_Printf(
        "[AI][TIP_FOCUS] Run: in=%p len=%lu out=%p len=%lu\r\n",
        (const void *)input_ptr, (unsigned long)input_len,
        (const void *)output_ptr, (unsigned long)output_len);
    __asm volatile("mov r9, %0" : : "r"(r9_before_log) : "r9");

    /* Flush the CPU D-cache so the NPU's AXI master sees our preprocessed
     * input data, then issue a full memory barrier to ensure the flash
     * memory-mapped window is visible before the NPU starts reading weights. */
    (void)mcu_cache_clean_range((uint32_t)(uintptr_t)input_ptr,
                                (uint32_t)((uintptr_t)input_ptr + input_len));
    __DSB();
    __ISB();

#if APP_AI_RESET_NETWORK_EACH_INFERENCE
    LL_ATON_RT_Reset_Network(&NN_Instance_Default);
#endif

    /* The ST-generated resize helpers in network.c use R9 as the base
     * pointer for the internal ai_array activation scratch buffer.
     * Cube.AI does not reliably initialise this register for networks that
     * contain resize nodes, so we load it explicitly from the linker-
     * defined activation section start before handing control to the NPU. */
    {
        extern uint8_t __stip_focus_activations[];
        const uintptr_t act_base = (uintptr_t)&__stip_focus_activations[0];
        __asm volatile("mov r9, %0" : : "r"(act_base) : "r9");
    }

    LL_ATON_RT_RetValues_t status;
    uint32_t epoch_count = 0U;
    do {
        /* The generated resize helpers read xSPI2-resident metadata on each
         * epoch, so reassert memory-mapped mode before every step. */
        if (!AppAI_Xspi2EnsureMemoryMappedMode()) {
            return false;
        }

        status = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
        epoch_count++;
        if (status == LL_ATON_RT_WFE) {
            LL_ATON_OSAL_WFE();
        }
    } while (status != LL_ATON_RT_DONE);

    (void)mcu_cache_invalidate_range((uint32_t)(uintptr_t)output_ptr,
                                     (uint32_t)((uintptr_t)output_ptr + output_len));
    return true;
}

/* This model uses internal NPU buffers (not user-allocated).
 * The actual buffer addresses are obtained from the buffer-info API
 * after LL_ATON_RT_Init_Network() has been called. */

int8_t *AppAI_TipFocus_GetInputBuffer(void)
{
    const LL_Buffer_InfoTypeDef *info =
        LL_ATON_Input_Buffers_Info(&NN_Instance_Default);
    if (info == NULL) return NULL;
    return (int8_t *)LL_Buffer_addr_start(info);
}

const void *AppAI_TipFocus_GetInputBufferInfo(void)
{
    return (const void *)LL_ATON_Input_Buffers_Info(&NN_Instance_Default);
}

/* Raw output order from the generated package:
 *   info[0] = tip_heatmap    [1,112,112,1] int8
 *   info[1] = center_heatmap [1,112,112,1] int8
 *   info[2] = confidence     [1,1]          int8
 *
 * The firmware-facing accessors below return semantic names, so the
 * caller always sees tip first and center second when decoding angle.
 */
const int8_t *AppAI_TipFocus_GetTipHeatmap(void)
{
    const LL_Buffer_InfoTypeDef *info =
        LL_ATON_Output_Buffers_Info(&NN_Instance_Default);
    if (info == NULL) return NULL;
    return (const int8_t *)LL_Buffer_addr_start(&info[0]);
}

const int8_t *AppAI_TipFocus_GetCenterHeatmap(void)
{
    const LL_Buffer_InfoTypeDef *info =
        LL_ATON_Output_Buffers_Info(&NN_Instance_Default);
    if (info == NULL) return NULL;
    return (const int8_t *)LL_Buffer_addr_start(&info[1]);
}

int8_t AppAI_TipFocus_GetConfidenceRaw(void)
{
    const LL_Buffer_InfoTypeDef *info =
        LL_ATON_Output_Buffers_Info(&NN_Instance_Default);
    if (info == NULL) return -128;
    const int8_t *buf = (const int8_t *)LL_Buffer_addr_start(&info[2]);
    if (buf == NULL) return -128;
    return *buf;
}

bool AppAI_TipFocus_DryRun(void)
{
    /* Zero the input buffer so the self-test starts from a known frame,
     * then run one inference and discard the published outputs. */
    const LL_Buffer_InfoTypeDef *input_info =
        LL_ATON_Input_Buffers_Info(&NN_Instance_Default);
    int8_t *input = NULL;
    size_t input_len = 0U;

    if (input_info == NULL) {
        return false;
    }

    input = (int8_t *)LL_Buffer_addr_start(input_info);
    input_len = (size_t)LL_Buffer_len(input_info);
    if ((input == NULL) || (input_len == 0U)) {
        return false;
    }

    memset(input, 0, input_len);
    return AppAI_TipFocus_Run();
}
