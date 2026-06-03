/* This file is included by app_ai_helpers.inc. */

static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms)
{
	const ULONG timeout_ticks = (ULONG)((timeout_ms + 9U) / 10U);
	const ULONG start_tick = tx_time_get();

	while (!AppFileX_IsMediaReady())
	{
		if ((tx_time_get() - start_tick) >= timeout_ticks)
		{
			return false;
		}

		tx_thread_sleep(1U);
	}

	return true;
}


