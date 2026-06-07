/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    inference_metrics.c
 * @brief   Unified inference metrics tracking for power and latency.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "inference_metrics.h"

/* Private includes ----------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "debug_console.h"
#include "ina219_power.h"
#include "ds3231_clock.h"
#include "sd_debug_log_service.h"

/* Private defines -----------------------------------------------------------*/
#define METRICS_TIMER_FREQ_HZ 1000000U /* 1 MHz = 1us resolution */
#define METRICS_ACTIVE_SLOTS 2U

/* Private variables ---------------------------------------------------------*/
static MetricsRecord_t s_metrics_buffer[METRICS_MAX_SAMPLES];
static uint32_t s_metrics_count = 0;
static uint32_t s_metrics_index = 0;

/* Active inference tracking — two slots so BASELINE and AI can be
 * timed independently within the same capture cycle. */
static struct
{
	bool active;
	char label[METRICS_LABEL_MAX_LEN];
	uint64_t start_time_us;
	uint64_t compute_start_time_us;
	uint64_t checkpoint_time_us;
	float power_pre_w;
	float power_mid_w;
	float power_post_w;
	/* Power accumulation during the active window (fed by INA219 thread). */
	uint32_t power_sample_count;
	float power_min_mw;
	float power_max_mw;
	float power_sum_mw;
} s_active_slots[METRICS_ACTIVE_SLOTS] = {0};

/* 64-bit DWT cycle-counter extension to avoid wrap every 5.4 s. */
static struct
{
	uint64_t high_cycles;
	uint32_t prev_cycles;
} s_dwt_state = {0, 0};

/* Private function prototypes -----------------------------------------------*/
static float Metrics_ReadPower(void);
static long Metrics_ToTenth(float value);
static int Metrics_FindActiveSlot(const char *label);

/* Private functions ---------------------------------------------------------*/

/**
 * @brief Read current power from INA219 in watts.
 */
static float Metrics_ReadPower(void)
{
    INA219_Measurement_t measurement;
    if (INA219_GetLastMeasurement(&measurement))
    {
        return measurement.power_w;
    }
    return 0.0f;
}

static long Metrics_ToTenth(float value)
{
    return (long)lroundf(value * 10.0f);
}

/* Public functions ----------------------------------------------------------*/

/**
 * @brief Initialize the metrics subsystem.
 */
void Metrics_Init(void)
{
	memset(s_metrics_buffer, 0, sizeof(s_metrics_buffer));
	s_metrics_count = 0;
	s_metrics_index = 0;
	memset(s_active_slots, 0, sizeof(s_active_slots));

	DebugConsole_Printf("[METRICS] Initialized (max %u samples)\r\n", METRICS_MAX_SAMPLES);
}

/**
 * @brief Get current timestamp in microseconds using DWT cycle counter.
 *
 * The DWT counter wraps every ~5.4 s at 800 MHz, so we extend it to 64
 * bits by tracking the high word across wraps.
 */
uint64_t Metrics_GetMicros(void)
{
	static bool dwt_initialized = false;
	if (!dwt_initialized)
	{
		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CYCCNT = 0;
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		s_dwt_state.prev_cycles = 0U;
		s_dwt_state.high_cycles = 0ULL;
		dwt_initialized = true;
	}

	uint32_t cycles = DWT->CYCCNT;
	if (cycles < s_dwt_state.prev_cycles)
	{
		s_dwt_state.high_cycles += (1ULL << 32);
	}
	s_dwt_state.prev_cycles = cycles;
	uint64_t total = s_dwt_state.high_cycles + (uint64_t)cycles;
	return total / 800ULL;
}

/**
 * @brief Find an active slot with the given label.  Returns -1 if not found.
 */
static int Metrics_FindActiveSlot(const char *label)
{
	if (label == NULL)
	{
		return -1;
	}
	for (size_t i = 0U; i < (size_t)METRICS_ACTIVE_SLOTS; i++)
	{
		if (s_active_slots[i].active &&
			(strcmp(s_active_slots[i].label, label) == 0))
		{
			return (int)i;
		}
	}
	return -1;
}

/**
 * @brief Start a new inference timing session.
 *
 * If a slot with the same label is already active it is re-stamped instead of
 * being ended, so a later capture can refresh the queued frame timing.
 * If all slots are busy the call is silently dropped.
 */
void Metrics_StartInference(const char *label)
{
	int slot;

	if (label == NULL)
	{
		return;
	}

	/* If a slot for this label is already active, just stamp the new
	 * capture time and reset power accumulators — do NOT end the slot.
	 * The async baseline worker may still be processing the previous
	 * frame; ending the slot would discard its latency measurement. */
	slot = Metrics_FindActiveSlot(label);
	if (slot >= 0)
	{
		s_active_slots[(size_t)slot].start_time_us = Metrics_GetMicros();
		s_active_slots[(size_t)slot].compute_start_time_us = 0ULL;
		s_active_slots[(size_t)slot].power_pre_w = Metrics_ReadPower();
		s_active_slots[(size_t)slot].power_mid_w = 0.0f;
		s_active_slots[(size_t)slot].power_post_w = 0.0f;
		s_active_slots[(size_t)slot].power_sample_count = 0U;
		s_active_slots[(size_t)slot].power_min_mw = 0.0f;
		s_active_slots[(size_t)slot].power_max_mw = 0.0f;
		s_active_slots[(size_t)slot].power_sum_mw = 0.0f;
		Metrics_PowerSample(s_active_slots[(size_t)slot].power_pre_w * 1000.0f);
		return;
	}

	/* Find a free slot. */
	slot = -1;
	for (int i = 0; i < (int)METRICS_ACTIVE_SLOTS; i++)
	{
		if (!s_active_slots[(size_t)i].active)
		{
			slot = i;
			break;
		}
	}
	if (slot < 0)
	{
		return;
	}

	s_active_slots[(size_t)slot].active = true;
	strncpy(s_active_slots[(size_t)slot].label, label,
			METRICS_LABEL_MAX_LEN - 1);
	s_active_slots[(size_t)slot].label[METRICS_LABEL_MAX_LEN - 1] = '\0';
	s_active_slots[(size_t)slot].start_time_us = Metrics_GetMicros();
	s_active_slots[(size_t)slot].compute_start_time_us = 0ULL;
	s_active_slots[(size_t)slot].power_pre_w = Metrics_ReadPower();
	s_active_slots[(size_t)slot].power_mid_w = 0.0f;
	s_active_slots[(size_t)slot].power_post_w = 0.0f;
	s_active_slots[(size_t)slot].power_sample_count = 0U;
	s_active_slots[(size_t)slot].power_min_mw = 0.0f;
	s_active_slots[(size_t)slot].power_max_mw = 0.0f;
	s_active_slots[(size_t)slot].power_sum_mw = 0.0f;

	/* Seed the accumulator immediately with the current power reading
	 * so the first sample doesn't lag up to 1 s behind the start. */
	Metrics_PowerSample(s_active_slots[(size_t)slot].power_pre_w * 1000.0f);
}

/**
 * @brief Mark the start of worker-side compute for an active inference slot.
 *
 * This is used to separate queue wait time from actual pipeline compute so the
 * logs can report both end-to-end latency and the worker execution time.
 */
void Metrics_MarkComputeStart(const char *label)
{
	int slot = Metrics_FindActiveSlot(label);

	if (slot < 0)
	{
		return;
	}

	if (s_active_slots[(size_t)slot].compute_start_time_us == 0ULL)
	{
		s_active_slots[(size_t)slot].compute_start_time_us =
			Metrics_GetMicros();
	}
}

/**
 * @brief Mark a checkpoint during inference.
 */
void Metrics_Checkpoint(const char *checkpoint_name)
{
	if (checkpoint_name == NULL)
	{
		return;
	}

	for (size_t i = 0U; i < (size_t)METRICS_ACTIVE_SLOTS; i++)
	{
		if (s_active_slots[i].active &&
			(strcmp(checkpoint_name, "MID") == 0))
		{
			s_active_slots[i].power_mid_w = Metrics_ReadPower();
			s_active_slots[i].checkpoint_time_us = Metrics_GetMicros();
			return;
		}
	}
}

/**
 * @brief Feed a power reading (milliwatts) to every active inference slot.
 *
 * Called by the INA219 monitoring thread at its sample rate.  Samples are
 * accumulated across the pipeline window and logged as min/avg/max when
 * the metric ends.
 */
void Metrics_PowerSample(float power_mw)
{
	for (size_t i = 0U; i < (size_t)METRICS_ACTIVE_SLOTS; i++)
	{
		if (s_active_slots[i].active)
		{
			s_active_slots[i].power_sum_mw += power_mw;
			s_active_slots[i].power_sample_count++;

			if ((s_active_slots[i].power_sample_count == 1U) ||
				(power_mw < s_active_slots[i].power_min_mw))
			{
				s_active_slots[i].power_min_mw = power_mw;
			}
			if (power_mw > s_active_slots[i].power_max_mw)
			{
				s_active_slots[i].power_max_mw = power_mw;
			}
		}
	}
}

/**
 * @brief Complete the inference and record metrics.
 *
 * Finds an active slot whose label matches @p label and closes it.
 * If no matching slot is active the function does nothing.
 */
void Metrics_EndInference(const char *label, float temperature_c)
{
	int slot = Metrics_FindActiveSlot(label);
	if (slot < 0)
	{
		return;
	}

	uint64_t end_time_us = Metrics_GetMicros();
	s_active_slots[(size_t)slot].power_post_w = Metrics_ReadPower();
	const bool temperature_is_finite = (isfinite(temperature_c) != 0);

	/* Calculate total latency, queue wait, and compute time. */
	const uint64_t request_start_us = s_active_slots[(size_t)slot].start_time_us;
	const uint64_t compute_start_us =
		s_active_slots[(size_t)slot].compute_start_time_us;
	uint64_t latency_us64 = end_time_us - request_start_us;
	uint64_t compute_us64 = latency_us64;
	uint64_t queue_wait_us64 = 0ULL;

	if ((compute_start_us >= request_start_us) &&
		(compute_start_us <= end_time_us))
	{
		compute_us64 = end_time_us - compute_start_us;
		queue_wait_us64 = compute_start_us - request_start_us;
	}

	uint32_t latency_us = (latency_us64 > (uint64_t)UINT32_MAX)
		? UINT32_MAX
		: (uint32_t)latency_us64;
	uint32_t compute_us = (compute_us64 > (uint64_t)UINT32_MAX)
		? UINT32_MAX
		: (uint32_t)compute_us64;

	/* Store in circular buffer */
	MetricsRecord_t *record = &s_metrics_buffer[s_metrics_index];
	strncpy(record->label, s_active_slots[(size_t)slot].label,
			METRICS_LABEL_MAX_LEN - 1);
	record->label[METRICS_LABEL_MAX_LEN - 1] = '\0';
	record->timestamp_ms = HAL_GetTick();
	record->latency_us = latency_us;
	record->compute_us = compute_us;
	record->power_pre_w = s_active_slots[(size_t)slot].power_pre_w;
	record->power_mid_w = s_active_slots[(size_t)slot].power_mid_w;
	record->power_post_w = s_active_slots[(size_t)slot].power_post_w;
	record->power_delta_w = s_active_slots[(size_t)slot].power_mid_w
		- s_active_slots[(size_t)slot].power_pre_w;
    record->temperature_c = temperature_is_finite ? temperature_c : NAN;
    record->valid = true;

    /* Update indices */
    s_metrics_index = (s_metrics_index + 1) % METRICS_MAX_SAMPLES;
    if (s_metrics_count < METRICS_MAX_SAMPLES)
    {
        s_metrics_count++;
    }

	/* Log immediately with total latency, queue wait, and compute time. */
	const float total_latency_ms = (float)latency_us / 1000.0f;
	const float compute_latency_ms = (float)compute_us / 1000.0f;
	const float queue_wait_ms = (float)queue_wait_us64 / 1000.0f;
	const long total_latency_tenth = Metrics_ToTenth(total_latency_ms);
	const long queue_wait_tenth = Metrics_ToTenth(queue_wait_ms);
	const long compute_latency_tenth = Metrics_ToTenth(compute_latency_ms);
	const long power_pre_tenth = Metrics_ToTenth(record->power_pre_w);
	const long power_mid_tenth = Metrics_ToTenth(record->power_mid_w);
	const long power_post_tenth = Metrics_ToTenth(record->power_post_w);
	const long power_delta_tenth = Metrics_ToTenth(record->power_delta_w);
	const long temp_tenth = temperature_is_finite ? Metrics_ToTenth(temperature_c) : 0L;
	char temp_field[16];
	if (temperature_is_finite)
	{
		DebugConsole_Snprintf(
			temp_field, sizeof(temp_field), "%ld.%01ld",
			temp_tenth / 10L, labs(temp_tenth % 10L));
	}
	else
	{
		DebugConsole_Snprintf(temp_field, sizeof(temp_field), "nan");
	}

	DebugConsole_Printf(
		"[METRICS] %s: total=%ld.%01ld ms, queue=%ld.%01ld ms, compute=%ld.%01ld ms, "
		"power_pre=%ld.%01ld W, power_mid=%ld.%01ld W, power_post=%ld.%01ld W, "
		"delta=%ld.%01ld W, temp=%sC\r\n",
		record->label,
		total_latency_tenth / 10L, labs(total_latency_tenth % 10L),
		queue_wait_tenth / 10L, labs(queue_wait_tenth % 10L),
		compute_latency_tenth / 10L, labs(compute_latency_tenth % 10L),
		power_pre_tenth / 10L, labs(power_pre_tenth % 10L),
		power_mid_tenth / 10L, labs(power_mid_tenth % 10L),
		power_post_tenth / 10L, labs(power_post_tenth % 10L),
		power_delta_tenth / 10L, labs(power_delta_tenth % 10L),
		temp_field);

	/* Log to SD card in CSV format with ISO 8601 timestamp. */
	char datetime_str[32];
	char csv_line[256];
	if (App_Clock_GetCurrentTimestamp(datetime_str, sizeof(datetime_str)))
	{
		/* Format: 2024-01-15 14:30:25 */
		DebugConsole_Snprintf(csv_line, sizeof(csv_line),
				 "%s,%s,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%s\r\n",
				 datetime_str,
				 record->label,
				 total_latency_tenth / 10L, labs(total_latency_tenth % 10L),
				 queue_wait_tenth / 10L, labs(queue_wait_tenth % 10L),
				 compute_latency_tenth / 10L, labs(compute_latency_tenth % 10L),
				 power_pre_tenth / 10L, labs(power_pre_tenth % 10L),
				 power_mid_tenth / 10L, labs(power_mid_tenth % 10L),
				 power_post_tenth / 10L, labs(power_post_tenth % 10L),
				 power_delta_tenth / 10L, labs(power_delta_tenth % 10L),
				 temp_field);
	}
	else
	{
		/* Fallback to tick timestamp if RTC unavailable. */
		DebugConsole_Snprintf(csv_line, sizeof(csv_line),
				 "%lu,%s,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%s\r\n",
				 (unsigned long)record->timestamp_ms,
				 record->label,
				 total_latency_tenth / 10L, labs(total_latency_tenth % 10L),
				 queue_wait_tenth / 10L, labs(queue_wait_tenth % 10L),
				 compute_latency_tenth / 10L, labs(compute_latency_tenth % 10L),
				 power_pre_tenth / 10L, labs(power_pre_tenth % 10L),
				 power_mid_tenth / 10L, labs(power_mid_tenth % 10L),
				 power_post_tenth / 10L, labs(power_post_tenth % 10L),
				 power_delta_tenth / 10L, labs(power_delta_tenth % 10L),
				 temp_field);
	}
	SdDebugLogService_EnqueueLine(csv_line);

	/* Power stats: log min / avg / max across the pipeline window. */
	if (s_active_slots[(size_t)slot].power_sample_count > 0U)
	{
		const uint32_t n = s_active_slots[(size_t)slot].power_sample_count;
		const float avg_mw =
			s_active_slots[(size_t)slot].power_sum_mw / (float)n;
		/* roundtrip through ToTenth keeps one-decimal-place consistency,
		 * reported in milliwatts */
		const long pmin = Metrics_ToTenth(
			s_active_slots[(size_t)slot].power_min_mw);
		const long pavg = Metrics_ToTenth(avg_mw);
		const long pmax = Metrics_ToTenth(
			s_active_slots[(size_t)slot].power_max_mw);
		DebugConsole_Printf(
			"[POWER][%s] min=%ld.%01ld mW avg=%ld.%01ld mW max=%ld.%01ld mW (%lu samples)\r\n",
			record->label,
			labs(pmin) / 10L, labs(pmin) % 10L,
			labs(pavg) / 10L, labs(pavg) % 10L,
			labs(pmax) / 10L, labs(pmax) % 10L,
			(unsigned long)n);
	}

	/* Reset the active slot */
	s_active_slots[(size_t)slot].active = false;
}

/**
 * @brief Override the start time of an active inference slot.
 *
 * The async baseline pipeline calls Metrics_StartInference at capture time,
 * but the worker thread may need to fix the slot's capture timestamp if a
 * subsequent frame's capture re-started the slot before the worker finished.
 */
void Metrics_OverrideStartTime(const char *label, uint64_t start_time_us)
{
    int slot = Metrics_FindActiveSlot(label);
    if (slot >= 0)
    {
        s_active_slots[(size_t)slot].start_time_us = start_time_us;
    }
}

/**
 * @brief Get the last completed metrics record.
 */
bool Metrics_GetLastRecord(MetricsRecord_t *record)
{
    if (record == NULL || s_metrics_count == 0)
    {
        return false;
    }

    uint32_t last_index = (s_metrics_index == 0) ? (METRICS_MAX_SAMPLES - 1) : (s_metrics_index - 1);
    *record = s_metrics_buffer[last_index];
    return record->valid;
}

/**
 * @brief Get summary statistics for all recorded inferences.
 */
bool Metrics_GetSummary(MetricsSummary_t *summary)
{
    if (summary == NULL || s_metrics_count == 0)
    {
        return false;
    }

    uint32_t valid_count = 0U;
    bool have_stats = false;
    float total_sum_ms = 0.0f;
    float total_min_ms = 0.0f;
    float total_max_ms = 0.0f;
    float queue_sum_ms = 0.0f;
    float queue_min_ms = 0.0f;
    float queue_max_ms = 0.0f;
    float compute_sum_ms = 0.0f;
    float compute_min_ms = 0.0f;
    float compute_max_ms = 0.0f;
    float delta_sum = 0.0f;
    float delta_min = 0.0f;
    float delta_max = 0.0f;
    float energy_sum = 0.0f;

    for (uint32_t i = 0; i < s_metrics_count; i++)
    {
        if (!s_metrics_buffer[i].valid)
        {
            continue;
        }

        const float total_ms = (float)s_metrics_buffer[i].latency_us / 1000.0f;
        const float compute_ms = (float)s_metrics_buffer[i].compute_us / 1000.0f;
        const float queue_ms = (total_ms > compute_ms) ? (total_ms - compute_ms) : 0.0f;
        const float delta = s_metrics_buffer[i].power_delta_w;
        const float energy_uj =
            (s_metrics_buffer[i].power_pre_w + s_metrics_buffer[i].power_mid_w) /
            2.0f * (float)s_metrics_buffer[i].latency_us;

        if (!have_stats)
        {
            total_min_ms = total_ms;
            total_max_ms = total_ms;
            queue_min_ms = queue_ms;
            queue_max_ms = queue_ms;
            compute_min_ms = compute_ms;
            compute_max_ms = compute_ms;
            delta_min = delta;
            delta_max = delta;
            have_stats = true;
        }
        else
        {
            if (total_ms < total_min_ms)
            {
                total_min_ms = total_ms;
            }
            if (total_ms > total_max_ms)
            {
                total_max_ms = total_ms;
            }
            if (queue_ms < queue_min_ms)
            {
                queue_min_ms = queue_ms;
            }
            if (queue_ms > queue_max_ms)
            {
                queue_max_ms = queue_ms;
            }
            if (compute_ms < compute_min_ms)
            {
                compute_min_ms = compute_ms;
            }
            if (compute_ms > compute_max_ms)
            {
                compute_max_ms = compute_ms;
            }
            if (delta < delta_min)
            {
                delta_min = delta;
            }
            if (delta > delta_max)
            {
                delta_max = delta;
            }
        }

        total_sum_ms += total_ms;
        queue_sum_ms += queue_ms;
        compute_sum_ms += compute_ms;
        delta_sum += delta;
        energy_sum += energy_uj;
        valid_count++;
    }

    if (!have_stats || (valid_count == 0U))
    {
        return false;
    }

    summary->count = valid_count;
    summary->latency_avg_ms = total_sum_ms / (float)valid_count;
    summary->latency_min_ms = total_min_ms;
    summary->latency_max_ms = total_max_ms;
    summary->queue_wait_avg_ms = queue_sum_ms / (float)valid_count;
    summary->queue_wait_min_ms = queue_min_ms;
    summary->queue_wait_max_ms = queue_max_ms;
    summary->compute_avg_ms = compute_sum_ms / (float)valid_count;
    summary->compute_min_ms = compute_min_ms;
    summary->compute_max_ms = compute_max_ms;
    summary->power_delta_avg_w = delta_sum / (float)valid_count;
    summary->power_delta_min_w = delta_min;
    summary->power_delta_max_w = delta_max;
    summary->energy_avg_uj = energy_sum / (float)valid_count;

    return true;
}

/**
 * @brief Log all recorded metrics to console in CSV format.
 */
void Metrics_LogAll(void)
{
    DebugConsole_Printf("\r\n[METRICS] CSV Export:\r\n");
    DebugConsole_Printf("timestamp_ms,label,total_latency_ms,queue_wait_ms,compute_ms,power_pre_W,power_mid_W,power_post_W,power_delta_W,temp_c\r\n");

    for (uint32_t i = 0; i < s_metrics_count; i++)
    {
        MetricsRecord_t *r = &s_metrics_buffer[i];
        if (!r->valid)
            continue;

        const float total_latency_ms = (float)r->latency_us / 1000.0f;
        const float compute_latency_ms = (float)r->compute_us / 1000.0f;
        const float queue_wait_ms =
            (total_latency_ms > compute_latency_ms)
            ? (total_latency_ms - compute_latency_ms)
            : 0.0f;
        const long total_latency_tenth = Metrics_ToTenth(total_latency_ms);
        const long queue_wait_tenth = Metrics_ToTenth(queue_wait_ms);
        const long compute_latency_tenth = Metrics_ToTenth(compute_latency_ms);
        const long power_pre_tenth = Metrics_ToTenth(r->power_pre_w);
        const long power_mid_tenth = Metrics_ToTenth(r->power_mid_w);
        const long power_post_tenth = Metrics_ToTenth(r->power_post_w);
        const long power_delta_tenth = Metrics_ToTenth(r->power_delta_w);
        char temp_field[16];
        if (isfinite(r->temperature_c))
        {
            const long temp_tenth = Metrics_ToTenth(r->temperature_c);
            DebugConsole_Snprintf(temp_field, sizeof(temp_field), "%ld.%01ld",
                                  temp_tenth / 10L, labs(temp_tenth % 10L));
        }
        else
        {
            DebugConsole_Snprintf(temp_field, sizeof(temp_field), "nan");
        }
        DebugConsole_Printf(
            "%lu,%s,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%ld.%01ld,%s\r\n",
            (unsigned long)r->timestamp_ms,
            r->label,
            total_latency_tenth / 10L, labs(total_latency_tenth % 10L),
            queue_wait_tenth / 10L, labs(queue_wait_tenth % 10L),
            compute_latency_tenth / 10L, labs(compute_latency_tenth % 10L),
            power_pre_tenth / 10L, labs(power_pre_tenth % 10L),
            power_mid_tenth / 10L, labs(power_mid_tenth % 10L),
            power_post_tenth / 10L, labs(power_post_tenth % 10L),
            power_delta_tenth / 10L, labs(power_delta_tenth % 10L),
            temp_field);
    }

    /* Log summary */
    MetricsSummary_t summary;
    if (Metrics_GetSummary(&summary))
    {
        DebugConsole_Printf("\r\n[METRICS] Summary (%lu samples):\r\n", (unsigned long)summary.count);
        const long latency_avg_tenth = Metrics_ToTenth(summary.latency_avg_ms);
        const long latency_min_tenth = Metrics_ToTenth(summary.latency_min_ms);
        const long latency_max_tenth = Metrics_ToTenth(summary.latency_max_ms);
        const long queue_avg_tenth = Metrics_ToTenth(summary.queue_wait_avg_ms);
        const long queue_min_tenth = Metrics_ToTenth(summary.queue_wait_min_ms);
        const long queue_max_tenth = Metrics_ToTenth(summary.queue_wait_max_ms);
        const long compute_avg_tenth = Metrics_ToTenth(summary.compute_avg_ms);
        const long compute_min_tenth = Metrics_ToTenth(summary.compute_min_ms);
        const long compute_max_tenth = Metrics_ToTenth(summary.compute_max_ms);
        const long delta_avg_tenth = Metrics_ToTenth(summary.power_delta_avg_w);
        const long delta_min_tenth = Metrics_ToTenth(summary.power_delta_min_w);
        const long delta_max_tenth = Metrics_ToTenth(summary.power_delta_max_w);
        const long energy_avg_tenth = Metrics_ToTenth(summary.energy_avg_uj);

        DebugConsole_Printf("  Total latency: avg=%ld.%01ld ms, min=%ld.%01ld ms, max=%ld.%01ld ms\r\n",
                            latency_avg_tenth / 10L, labs(latency_avg_tenth % 10L),
                            latency_min_tenth / 10L, labs(latency_min_tenth % 10L),
                            latency_max_tenth / 10L, labs(latency_max_tenth % 10L));
        DebugConsole_Printf("  Queue wait: avg=%ld.%01ld ms, min=%ld.%01ld ms, max=%ld.%01ld ms\r\n",
                            queue_avg_tenth / 10L, labs(queue_avg_tenth % 10L),
                            queue_min_tenth / 10L, labs(queue_min_tenth % 10L),
                            queue_max_tenth / 10L, labs(queue_max_tenth % 10L));
        DebugConsole_Printf("  Compute: avg=%ld.%01ld ms, min=%ld.%01ld ms, max=%ld.%01ld ms\r\n",
                            compute_avg_tenth / 10L, labs(compute_avg_tenth % 10L),
                            compute_min_tenth / 10L, labs(compute_min_tenth % 10L),
                            compute_max_tenth / 10L, labs(compute_max_tenth % 10L));
        DebugConsole_Printf("  Power Delta: avg=%ld.%01ld W, min=%ld.%01ld W, max=%ld.%01ld W\r\n",
                            delta_avg_tenth / 10L, labs(delta_avg_tenth % 10L),
                            delta_min_tenth / 10L, labs(delta_min_tenth % 10L),
                            delta_max_tenth / 10L, labs(delta_max_tenth % 10L));
        DebugConsole_Printf("  Energy per inference: %ld.%01ld uJ\r\n",
                            energy_avg_tenth / 10L, labs(energy_avg_tenth % 10L));
    }

    DebugConsole_Printf("\r\n");
}

/**
 * @brief Clear all recorded metrics.
 */
void Metrics_Clear(void)
{
    memset(s_metrics_buffer, 0, sizeof(s_metrics_buffer));
    s_metrics_count = 0;
    s_metrics_index = 0;
    DebugConsole_Printf("[METRICS] Cleared all samples\r\n");
}
