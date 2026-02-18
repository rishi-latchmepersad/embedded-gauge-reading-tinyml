/*
 * threadx_utils.c
 *
 *  Created on: 10 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "tx_api.h"
#include <stdint.h>
#include "debug_console.h"

/**
 * @brief  Delay execution of the current ThreadX thread by a given number of milliseconds.
 * @param  delay_time_milliseconds The requested delay duration in milliseconds.
 * @return None.
 * @sideeffects Causes the calling thread to sleep, allowing other threads to run.
 * @preconditions ThreadX kernel must be running; must be called from thread context.
 * @concurrency Safe; affects only the calling thread.
 * @timing Delay resolution is limited by TX_TIMER_TICKS_PER_SECOND.
 * @errors None returned; extremely long delays may overflow tick conversion if not bounded.
 * @notes Use this instead of HAL_Delay() inside ThreadX threads to avoid dependence on HAL tick.
 */
void DelayMilliseconds_ThreadX(const uint32_t delay_time_milliseconds) {
	const uint32_t ticks_per_second = (uint32_t) TX_TIMER_TICKS_PER_SECOND;

	/* Round up milliseconds to at least 1 tick when delay_time_milliseconds > 0. */
	uint32_t delay_ticks = (delay_time_milliseconds * ticks_per_second + 999U)
			/ 1000U;

	if ((delay_time_milliseconds > 0U) && (delay_ticks == 0U)) {
		delay_ticks = 1U;
	}

	(void) tx_thread_sleep((ULONG) delay_ticks);
}

/**
 * @brief  Print basic usage statistics for a ThreadX byte pool.
 * @param  byte_pool_ptr Pointer to the ThreadX byte pool to query.
 * @param  pool_friendly_name_ptr Short label to identify the pool in logs (e.g. "TX", "FX").
 * @return None.
 * @sideeffects Prints to the debug console.
 * @preconditions ThreadX must be initialized; byte_pool_ptr must refer to a valid created pool.
 * @concurrency Safe; values may change if other threads allocate/free concurrently.
 * @timing Fast; suitable for startup diagnostics.
 * @errors If tx_byte_pool_info_get() fails, prints the returned status code.
 * @notes Useful to confirm remaining free bytes and fragmentation after middleware init.
 */
void PrintBytePoolUsage_ThreadX(TX_BYTE_POOL *byte_pool_ptr,
		const char *pool_friendly_name_ptr) {
	CHAR *pool_name_ptr = TX_NULL;
	ULONG available_bytes = 0UL;
	ULONG fragment_count = 0UL;
	TX_THREAD *first_suspended_thread_ptr = TX_NULL;
	ULONG suspended_thread_count = 0UL;
	TX_BYTE_POOL *next_pool_ptr = TX_NULL;

	if ((byte_pool_ptr == TX_NULL) || (pool_friendly_name_ptr == NULL)) {
		DebugConsole_Printf("Byte pool usage print skipped, invalid args.\r\n");
		return;
	}

	const UINT status = tx_byte_pool_info_get(byte_pool_ptr, &pool_name_ptr,
			&available_bytes, &fragment_count, &first_suspended_thread_ptr,
			&suspended_thread_count, &next_pool_ptr);

	if (status != TX_SUCCESS) {
		DebugConsole_Printf("%s pool info get failed, status=%lu\r\n",
				pool_friendly_name_ptr, (unsigned long) status);
		return;
	}

	DebugConsole_Printf(
			"%s pool '%s': free=%lu bytes, fragments=%lu, suspended=%lu\r\n",
			pool_friendly_name_ptr,
			(pool_name_ptr != TX_NULL) ? pool_name_ptr : "unknown",
			(unsigned long) available_bytes, (unsigned long) fragment_count,
			(unsigned long) suspended_thread_count);
}
