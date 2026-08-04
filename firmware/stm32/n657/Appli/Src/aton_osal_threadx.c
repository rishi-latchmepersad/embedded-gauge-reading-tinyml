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

/* A failed model handoff must not strand the camera thread in WFE forever.
 * Normal ATON completions arrive immediately; this guard only releases the
 * caller when the interrupt path has stopped producing events. */
#define ATON_OSAL_WFE_GUARD_MS 2000U

extern void ATON_STD_IRQHandler(void);

static TX_MUTEX g_cache_mutex;
/* The no-HyperRAM product has one ATON owner.  Object creation can still fail
 * when the shared ThreadX byte pool is nearly exhausted, so remember which
 * primitives are actually live before any callback touches them. */
static bool g_cache_mutex_ready = false;
static volatile ULONG g_wfe_pending_count = 0UL;
static volatile ULONG g_wfe_signal_count = 0UL;
static volatile bool g_wfe_guard_expired = false;
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

	/* Re-initialization must start from a conservative state.  The NPU
	 * interrupt handler only updates the event counter; it never calls a
	 * ThreadX synchronization API from interrupt context. */
	g_cache_mutex_ready = false;

	ret = tx_mutex_create(&g_cache_mutex, (CHAR *)"aton_cache", TX_INHERIT);
	if (ret != TX_SUCCESS)
	{
		DebugConsole_Printf("[AI][OSAL] cache mutex create failed: %lu\r\n",
							(unsigned long)ret);
		g_cache_mutex_ready = false;
	}
	else
	{
		g_cache_mutex_ready = true;
	}

	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	g_wfe_pending_count = 0UL;
	g_wfe_signal_count = 0UL;
	g_wfe_guard_expired = false;
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

	if (g_cache_mutex_ready)
	{
		(void)tx_mutex_delete(&g_cache_mutex);
	}
	g_osal_initialized = false;
	g_cache_mutex_ready = false;
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
	/* No parallel ATON networks are enabled in this product.  If the optional
	 * mutex could not be allocated, the single worker remains safe without
	 * touching an uninitialized ThreadX object. */
	if (!g_cache_mutex_ready)
	{
		return;
	}

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
	if (!g_cache_mutex_ready)
	{
		return;
	}

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
 * The NPU ISR records completion in an interrupt-safe counter. The worker
 * consumes that counter here and sleeps in one-tick increments when the NPU
 * has not signalled yet. This deliberately avoids calling ThreadX semaphore
 * APIs from the NPU ISR, which can corrupt ThreadX timer state on this port.
 */
void aton_osal_threadx_wfe(void)
{
	const uint32_t wait_start_tick = HAL_GetTick();

	for (;;)
	{
		TX_INTERRUPT_SAVE_AREA
		ULONG pending = 0UL;

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

		if ((HAL_GetTick() - wait_start_tick) >= ATON_OSAL_WFE_GUARD_MS)
		{
			TX_DISABLE
			g_wfe_guard_expired = true;
			TX_RESTORE
			DebugConsole_Printf(
				"[AI][OSAL] WFE guard expired after %lu ms; returning to runtime.\r\n",
				(unsigned long)(HAL_GetTick() - wait_start_tick));
			return;
		}

		/* Sleep briefly rather than spin so the AI worker and camera watchdog
		 * continue to run while the interrupt line is being recovered. */
		tx_thread_sleep(1U);
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

	/* The worker polls at one ThreadX tick while waiting.  SEV is harmless on
	 * Cortex-M and provides an additional wake hint without entering ThreadX
	 * from interrupt context; it is not used as the correctness mechanism. */
	__SEV();
}

/**
 * @brief Drain any leaked WFE events before a new inference.
 */
void LL_ATON_OSAL_DrainWfeSemaphore(void)
{
	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	g_wfe_pending_count = 0UL;
	g_wfe_guard_expired = false;
	TX_RESTORE

	DebugConsole_Printf("[AI][OSAL] Drained WFE event counter.\r\n");
}

/**
 * @brief Report whether the most recent ATON wait exceeded its safety guard.
 * @return true when the caller must abort the current stage and reinitialize.
 */
bool LL_ATON_OSAL_WfeGuardExpired(void)
{
	return g_wfe_guard_expired;
}

/**
 * @brief Read the current WFE event count without consuming it.
 */
UINT LL_ATON_OSAL_GetWfeSemaphoreCount(void)
{
	return (UINT)g_wfe_pending_count;
}

#endif /* (LL_ATON_OSAL == LL_ATON_OSAL_THREADX) */
