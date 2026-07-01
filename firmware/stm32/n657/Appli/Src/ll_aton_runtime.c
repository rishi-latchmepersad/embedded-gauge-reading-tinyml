/**
 * @file    ll_aton_runtime.c
 * @brief   Project-owned wrapper for the ST ATON runtime source.
 *
 * The pack runtime defaults to async WFE behavior. Keep that execution model
 * so the runtime's IRQ handler can advance epoch blocks exactly as ST expects.
 * We harden the ThreadX wait path separately instead of forcing polling mode.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_MODE LL_ATON_RT_ASYNC

#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Middlewares/ST/AI/Npu/ll_aton/ll_aton_runtime.c"
