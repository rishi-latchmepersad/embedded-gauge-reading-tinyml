/**
 * @file    aton_osal_threadx.c
 * @brief   Project-owned ThreadX bridge for the ST ATON runtime.
 *
 * The stock X-CUBE-AI RTOS template asserts hard inside `LL_ATON_OSAL_WFE()`
 * if ThreadX returns anything other than `TX_SUCCESS`. On this board we want
 * the runtime to keep making forward progress even if the wait primitive is
 * flaky, so we own the wait/event plumbing directly here.
 */

#define LL_ATON_PLATFORM LL_ATON_PLAT_STM32N6
#define LL_ATON_OSAL LL_ATON_OSAL_THREADX

#include "ll_aton_config.h"
#include "ll_aton_platform.h"

#if (LL_ATON_OSAL == LL_ATON_OSAL_THREADX)

#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#include "debug_console.h"
#include "main.h"
#include "ll_aton_osal_threadx.h"

/* Keep the OSAL implementation in a user-owned translation unit so the build
 * no longer depends on the generated `Debug/` object or the vendor template. */

extern void ATON_STD_IRQHandler(void);

static TX_MUTEX g_cache_mutex;
static TX_SEMAPHORE g_wfe_sem;
static volatile ULONG g_wfe_pending_count = 0UL;
static volatile ULONG g_wfe_signal_count = 0UL;
static bool g_wfe_semaphore_ready = false;
static bool g_wfe_use_semaphore_wait = true;
static bool g_osal_initialized = false;

/**
 * @brief Return true when the ATON interrupt controller has a real pending IRQ.
 */
static bool AtonOsalThreadx_HasPendingAtonIrq(void)
{
#if (ATON_INT_NR > 32)
	return (ATON_INTCTRL_INTREG_GET(0) != 0U) ||
		   (ATON_INTCTRL_INTREG_H_GET(0) != 0U);
#else
	return (ATON_INTCTRL_INTREG_GET(0) != 0U);
#endif
}

/**
 * @brief Service a latched ATON IRQ when the NVIC path failed to deliver it.
 */
static bool AtonOsalThreadx_ServicePendingAtonIrq(void)
{
	if (!AtonOsalThreadx_HasPendingAtonIrq())
	{
		return false;
	}

	ATON_STD_IRQHandler();
	return true;
}

/**
 * @brief Initialize the RTOS-facing ATON bridge.
 */
void aton_osal_threadx_init(void)
{
	UINT ret;

	ret = tx_mutex_create(&g_cache_mutex, (CHAR *)"aton_cache", TX_INHERIT);
	if (ret != TX_SUCCESS)
	{
		DebugConsole_Printf("[AI][OSAL] cache mutex create failed: %lu\r\n",
							(unsigned long)ret);
	}

	ret = tx_semaphore_create(&g_wfe_sem, (CHAR *)"aton_wfe", 0U);
	if (ret != TX_SUCCESS)
	{
		DebugConsole_Printf("[AI][OSAL] wfe semaphore create failed: %lu\r\n",
							(unsigned long)ret);
		g_wfe_use_semaphore_wait = false;
	}
	else
	{
		g_wfe_semaphore_ready = true;
	}

	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	g_wfe_pending_count = 0UL;
	g_wfe_signal_count = 0UL;
	TX_RESTORE

	g_osal_initialized = true;
}

/**
 * @brief Tear down the RTOS-facing ATON bridge.
 */
void aton_osal_threadx_deinit(void)
{
	if (!g_osal_initialized)
	{
		return;
	}

	if (g_wfe_semaphore_ready)
	{
		(void)tx_semaphore_delete(&g_wfe_sem);
	}
	(void)tx_mutex_delete(&g_cache_mutex);
	g_osal_initialized = false;
	g_wfe_semaphore_ready = false;
	g_wfe_use_semaphore_wait = true;
}

/**
 * @brief Lock ATON access.
 *
 * The project currently builds with `APP_HAS_PARALLEL_NETWORKS=0`, so there
 * is no concurrent ATON owner to arbitrate.
 */
void aton_osal_threadx_dao_lock(void)
{
}

/**
 * @brief Unlock ATON access.
 */
void aton_osal_threadx_dao_unlock(void)
{
}

/**
 * @brief Lock the MCU cache mutex used by the ATON runtime.
 */
void aton_osal_threadx_lock(void)
{
	UINT ret = tx_mutex_get(&g_cache_mutex, TX_WAIT_FOREVER);
	if (ret != TX_SUCCESS)
	{
		DebugConsole_Printf("[AI][OSAL] cache mutex get failed: %lu\r\n",
							(unsigned long)ret);
	}
}

/**
 * @brief Unlock the MCU cache mutex used by the ATON runtime.
 */
void aton_osal_threadx_unlock(void)
{
	UINT ret = tx_mutex_put(&g_cache_mutex);
	if (ret != TX_SUCCESS)
	{
		DebugConsole_Printf("[AI][OSAL] cache mutex put failed: %lu\r\n",
							(unsigned long)ret);
	}
}

/**
 * @brief Wait for the next ATON event.
 *
 * The ATON runtime expects a real wait primitive here. We still keep the
 * counters for diagnostics, but the semaphore is the canonical wakeup path.
 */
void aton_osal_threadx_wfe(void)
{
	for (;;)
	{
		TX_INTERRUPT_SAVE_AREA
		ULONG pending = 0UL;
		UINT ret = TX_SUCCESS;

		/* First preference: consume a real IRQ delivery from the ThreadX
		 * semaphore path without blocking forever. */
		if (g_wfe_use_semaphore_wait && g_wfe_semaphore_ready)
		{
			ret = tx_semaphore_get(&g_wfe_sem, TX_NO_WAIT);
			if (ret == TX_SUCCESS)
			{
				TX_DISABLE
				if (g_wfe_pending_count != 0UL)
				{
					g_wfe_pending_count--;
				}
				TX_RESTORE
				return;
			}

			if ((ret == TX_SEMAPHORE_ERROR) || (ret == TX_WAIT_ERROR))
			{
				DebugConsole_Printf("[AI][OSAL] wfe wait failed: %lu\r\n",
									(unsigned long)ret);
				g_wfe_use_semaphore_wait = false;
				DebugConsole_WriteString(
					"[AI][OSAL] wfe semaphore unavailable; using event counter fallback.\r\n");
			}
		}

		TX_DISABLE
		pending = g_wfe_pending_count;
		if (pending != 0UL)
		{
			g_wfe_pending_count--;
		}
		TX_RESTORE

		if (pending != 0UL)
		{
			return;
		}

		/* If the interrupt controller latched a real ATON completion but the
		 * NVIC or ThreadX handoff failed to wake us, service the handler
		 * directly here. This preserves the async runtime contract without
		 * inventing synthetic completions. */
		if (AtonOsalThreadx_ServicePendingAtonIrq())
		{
			continue;
		}

		tx_thread_relinquish();
	}
}

/**
 * @brief Signal the ATON wait path from interrupt context.
 */
void aton_osal_threadx_signal_event(void)
{
	TX_INTERRUPT_SAVE_AREA

	TX_DISABLE
	if (g_wfe_pending_count < ULONG_MAX)
	{
		g_wfe_pending_count++;
	}
	g_wfe_signal_count++;
	TX_RESTORE

	if (g_wfe_use_semaphore_wait && g_wfe_semaphore_ready)
	{
		UINT ret = tx_semaphore_put(&g_wfe_sem);
		if (ret != TX_SUCCESS)
		{
			g_wfe_use_semaphore_wait = false;
			DebugConsole_Printf("[AI][OSAL] wfe signal semaphore put failed: %lu\r\n",
								(unsigned long)ret);
		}
	}
}

/**
 * @brief Drain any leaked WFE events before a new inference.
 */
void LL_ATON_OSAL_DrainWfeSemaphore(void)
{
	while (tx_semaphore_get(&g_wfe_sem, TX_NO_WAIT) == TX_SUCCESS)
	{
	}

	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	g_wfe_pending_count = 0UL;
	TX_RESTORE

	DebugConsole_Printf("[AI][OSAL] Drained WFE event counter.\r\n");
}

/**
 * @brief Read the current WFE event count without consuming it.
 */
UINT LL_ATON_OSAL_GetWfeSemaphoreCount(void)
{
	return (UINT)g_wfe_pending_count;
}

#endif /* (LL_ATON_OSAL == LL_ATON_OSAL_THREADX) */
