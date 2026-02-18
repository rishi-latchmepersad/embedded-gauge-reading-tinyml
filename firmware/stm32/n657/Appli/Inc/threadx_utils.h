#ifndef THREADX_UTILS_H
#define THREADX_UTILS_H

#include <stdint.h>
#include "tx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
void DelayMilliseconds_ThreadX(const uint32_t delay_time_milliseconds);

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
		const char *pool_friendly_name_ptr);

#ifdef __cplusplus
}
#endif

#endif /* THREADX_UTILS_H */
