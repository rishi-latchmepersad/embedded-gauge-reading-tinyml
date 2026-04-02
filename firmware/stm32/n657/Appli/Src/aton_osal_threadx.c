/**
 *******************************************************************************
 * @file    aton_osal_threadx.c
 * @brief   User-owned ThreadX OSAL bridge for the ST ATON runtime.
 *******************************************************************************
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

#include "ll_aton_config.h"

#if (LL_ATON_OSAL == LL_ATON_OSAL_THREADX)

#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#include "ll_aton_osal_threadx.h"

/* Keep the OSAL implementation in a user-owned translation unit so the build
 * no longer depends on a generated `Debug/` object for this symbol. */
#include "ll_aton_osal_rtos_template.c"

#endif /* (LL_ATON_OSAL == LL_ATON_OSAL_THREADX) */
