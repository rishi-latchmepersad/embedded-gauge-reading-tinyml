/*
 * threadx_utils.c
 *
 *  Created on: 10 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "tx_api.h"
#include <stdint.h>

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
