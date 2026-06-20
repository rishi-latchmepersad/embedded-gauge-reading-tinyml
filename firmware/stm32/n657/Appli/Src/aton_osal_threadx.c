/**
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
#include "debug_console.h"

/* Keep the OSAL implementation in a user-owned translation unit so the build
 * no longer depends on a generated `Debug/` object for this symbol. */
#include "ll_aton_osal_rtos_template.c"

/**
 * @brief Drain any leaked WFE semaphore tokens before a new inference.
 *
 * The WFE semaphore can be left with count > 0 if a previous inference's last
 * epoch completed and signalled the semaphore, but RunEpochBlock returned DONE
 * before the app called WFE. This causes the next inference's first WFE to
 * return immediately without blocking, breaking the epoch loop.
 *
 * Call this before starting a new inference to ensure the semaphore is empty.
 */
void LL_ATON_OSAL_DrainWfeSemaphore(void)
{
	UINT ret;
	UINT drained_count = 0U;
	do {
		ret = tx_semaphore_get(&_wfe_sem, TX_NO_WAIT);
		if (ret == TX_SUCCESS)
		{
			++drained_count;
		}
	} while (ret == TX_SUCCESS);
	
	DebugConsole_Printf("[AI][OSAL] Drained %u leaked WFE semaphore tokens.\r\n", drained_count);
}

/**
 * @brief Read the current WFE semaphore count without consuming it.
 */
UINT LL_ATON_OSAL_GetWfeSemaphoreCount(void)
{
	ULONG current_count = 0;
	tx_semaphore_info_get(&_wfe_sem, NULL, &current_count, NULL, NULL, NULL);
	return (UINT)current_count;
}

#endif /* (LL_ATON_OSAL == LL_ATON_OSAL_THREADX) */
