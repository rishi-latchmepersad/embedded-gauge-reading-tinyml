#ifndef THREADX_UTILS_H
#define THREADX_UTILS_H

#include <stdint.h>
#include "tx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
void DelayMilliseconds_ThreadX(const uint32_t delay_time_milliseconds);

/**
 * @brief  Delay execution without using the ThreadX timer queue.
 * @param  delay_time_milliseconds The requested delay duration in milliseconds.
 * @return None.
 * @sideeffects Yields the CPU cooperatively while waiting on the HAL tick.
 * @preconditions ThreadX kernel must be running; must be called from thread context.
 * @concurrency Safe; affects only the calling thread.
 * @timing Delay resolution is limited by the HAL tick and thread scheduling.
 * @notes Use this for long background waits when the timer queue itself is part
 *        of the thing we are trying to avoid exercising.
 */
void DelayMilliseconds_Cooperative(const uint32_t delay_time_milliseconds);

/**
 * @brief  Lock a ThreadX mutex without blocking.
 * @param  mutex_ptr Pointer to the mutex to acquire.
 * @return None.
 * @notes This is intentionally best-effort so debug logging can avoid deadlock.
 */
void ThreadxUtils_LockMutex(TX_MUTEX *mutex_ptr);

/**
 * @brief  Release a ThreadX mutex.
 * @param  mutex_ptr Pointer to the mutex to release.
 * @return None.
 */
void ThreadxUtils_UnlockMutex(TX_MUTEX *mutex_ptr);

/**
 * @brief  Provide the current HAL tick count in milliseconds.
 * @return Current system tick in milliseconds.
 */
int32_t ThreadxUtils_GetTickMs(void);

/**
 * @brief  Convert milliseconds to ThreadX ticks, rounding up.
 * @param  timeout_ms Timeout in milliseconds.
 * @return Equivalent timeout in scheduler ticks.
 */
ULONG ThreadxUtils_MillisecondsToTicks(uint32_t timeout_ms);

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
