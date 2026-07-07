/**
 * @file    ll_aton_runtime.c
 * @brief   Project-owned placeholder for the ST ATON runtime unit.
 *
 * Keep the generated runtime implementation in this translation unit so the
 * linker sees the `LL_ATON_RT_*` entry points, but do not fold in the extra
 * ST AI library or utility sources because those are already compiled as
 * separate objects elsewhere in the project.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX
#define LL_ATON_RT_MODE LL_ATON_RT_ASYNC
#ifndef NDEBUG
#define NDEBUG
#endif

#include "C:/Users/rishi_latchmepersad/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Middlewares/ST/AI/Npu/ll_aton/ll_aton_runtime.c"
