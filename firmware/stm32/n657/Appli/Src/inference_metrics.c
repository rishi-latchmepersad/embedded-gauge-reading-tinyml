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

/* Private variables ---------------------------------------------------------*/
static MetricsRecord_t s_metrics_buffer[METRICS_MAX_SAMPLES];
static uint32_t s_metrics_count = 0;
static uint32_t s_metrics_index = 0;

/* Active inference tracking */
static struct
{
    bool active;
    char label[METRICS_LABEL_MAX_LEN];
    uint32_t start_time_us;
    uint32_t checkpoint_time_us;
    float power_pre_mw;
    float power_mid_mw;
    float power_post_mw;
} s_active_inference = {0};

/* Private function prototypes -----------------------------------------------*/
static float Metrics_ReadPower(void);
static long Metrics_ToCenti(float value);
static long Metrics_ToMilli(float value);

/* Private functions ---------------------------------------------------------*/

/**
 * @brief Read current power from INA219 in milliwatts.
 */
static float Metrics_ReadPower(void)
{
    INA219_Measurement_t measurement;
    if (INA219_GetLastMeasurement(&measurement))
    {
        return measurement.power_mw;
    }
    return 0.0f;
}

static long Metrics_ToCenti(float value)
{
    return (long)lroundf(value * 100.0f);
}

static long Metrics_ToMilli(float value)
{
    return (long)lroundf(value * 1000.0f);
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
    memset(&s_active_inference, 0, sizeof(s_active_inference));

    DebugConsole_Printf("[METRICS] Initialized (max %u samples)\r\n", METRICS_MAX_SAMPLES);
}

/**
 * @brief Get current timestamp in microseconds using DWT cycle counter.
 */
uint32_t Metrics_GetMicros(void)
{
    /* Use DWT cycle counter for high-resolution timing */
    static bool dwt_initialized = false;
    if (!dwt_initialized)
    {
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
        DWT->CYCCNT = 0;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
        dwt_initialized = true;
    }

    /* Convert cycles to microseconds (assuming 800MHz clock) */
    uint32_t cycles = DWT->CYCCNT;
    return cycles / 800; /* 800 cycles = 1us at 800MHz */
}

/**
 * @brief Start a new inference timing session.
 */
void Metrics_StartInference(const char *label)
{
    if (label == NULL)
    {
        return;
    }

    /* End any active inference first */
    if (s_active_inference.active)
    {
        Metrics_EndInference(NAN);
    }

    /* Start new inference tracking */
    s_active_inference.active = true;
    strncpy(s_active_inference.label, label, METRICS_LABEL_MAX_LEN - 1);
    s_active_inference.label[METRICS_LABEL_MAX_LEN - 1] = '\0';
    s_active_inference.start_time_us = Metrics_GetMicros();
    s_active_inference.checkpoint_time_us = 0;
    s_active_inference.power_pre_mw = Metrics_ReadPower();
    s_active_inference.power_mid_mw = 0.0f;
    s_active_inference.power_post_mw = 0.0f;
}

/**
 * @brief Mark a checkpoint during inference.
 */
void Metrics_Checkpoint(const char *checkpoint_name)
{
    if (!s_active_inference.active || checkpoint_name == NULL)
    {
        return;
    }

    if (strcmp(checkpoint_name, "MID") == 0)
    {
        s_active_inference.power_mid_mw = Metrics_ReadPower();
        s_active_inference.checkpoint_time_us = Metrics_GetMicros();
    }
}

/**
 * @brief Complete the inference and record metrics.
 */
void Metrics_EndInference(float temperature_c)
{
    if (!s_active_inference.active)
    {
        return;
    }

    uint32_t end_time_us = Metrics_GetMicros();
    s_active_inference.power_post_mw = Metrics_ReadPower();

    /* Calculate latency */
    uint32_t latency_us = end_time_us - s_active_inference.start_time_us;

    /* Store in circular buffer */
    MetricsRecord_t *record = &s_metrics_buffer[s_metrics_index];
    strncpy(record->label, s_active_inference.label, METRICS_LABEL_MAX_LEN - 1);
    record->label[METRICS_LABEL_MAX_LEN - 1] = '\0';
    record->timestamp_ms = HAL_GetTick();
    record->latency_us = latency_us;
    record->power_pre_mw = s_active_inference.power_pre_mw;
    record->power_mid_mw = s_active_inference.power_mid_mw;
    record->power_post_mw = s_active_inference.power_post_mw;
    record->power_delta_mw = s_active_inference.power_mid_mw - s_active_inference.power_pre_mw;
    record->temperature_c = temperature_c;
    record->valid = true;

    /* Update indices */
    s_metrics_index = (s_metrics_index + 1) % METRICS_MAX_SAMPLES;
    if (s_metrics_count < METRICS_MAX_SAMPLES)
    {
        s_metrics_count++;
    }

    /* Log immediately (latency in ms for readability) */
    float latency_ms = (float)latency_us / 1000.0f;
    const long latency_ms_centi = Metrics_ToCenti(latency_ms);
    const long power_pre_centi = Metrics_ToCenti(record->power_pre_mw);
    const long power_mid_centi = Metrics_ToCenti(record->power_mid_mw);
    const long power_post_centi = Metrics_ToCenti(record->power_post_mw);
    const long power_delta_centi = Metrics_ToCenti(record->power_delta_mw);
    const long temp_centi = Metrics_ToCenti(temperature_c);

    DebugConsole_Printf(
        "[METRICS] %s: latency=%ld.%02ld ms, power_pre=%ld.%02ldmW, power_mid=%ld.%02ldmW, "
        "power_post=%ld.%02ldmW, delta=%ld.%02ldmW, temp=%ld.%02ldC\r\n",
        record->label,
        latency_ms_centi / 100L, labs(latency_ms_centi % 100L),
        power_pre_centi / 100L, labs(power_pre_centi % 100L),
        power_mid_centi / 100L, labs(power_mid_centi % 100L),
        power_post_centi / 100L, labs(power_post_centi % 100L),
        power_delta_centi / 100L, labs(power_delta_centi % 100L),
        temp_centi / 100L, labs(temp_centi % 100L));

    /* Log to SD card in CSV format with ISO 8601 timestamp */
    char datetime_str[32];
    char csv_line[256];
    if (App_Clock_GetCurrentTimestamp(datetime_str, sizeof(datetime_str)))
    {
        /* Format: 2024-01-15 14:30:25 */
        DebugConsole_Snprintf(csv_line, sizeof(csv_line),
                 "%s,%s,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld\r\n",
                 datetime_str,
                 record->label,
                 latency_ms_centi / 100L, labs(latency_ms_centi % 100L),
                 power_pre_centi / 100L, labs(power_pre_centi % 100L),
                 power_mid_centi / 100L, labs(power_mid_centi % 100L),
                 power_post_centi / 100L, labs(power_post_centi % 100L),
                 power_delta_centi / 100L, labs(power_delta_centi % 100L),
                 temp_centi / 100L, labs(temp_centi % 100L));
    }
    else
    {
        /* Fallback to tick timestamp if RTC unavailable */
        DebugConsole_Snprintf(csv_line, sizeof(csv_line),
                 "%lu,%s,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld,%ld.%02ld\r\n",
                 (unsigned long)record->timestamp_ms,
                 record->label,
                 latency_ms_centi / 100L, labs(latency_ms_centi % 100L),
                 power_pre_centi / 100L, labs(power_pre_centi % 100L),
                 power_mid_centi / 100L, labs(power_mid_centi % 100L),
                 power_post_centi / 100L, labs(power_post_centi % 100L),
                 power_delta_centi / 100L, labs(power_delta_centi % 100L),
                 temp_centi / 100L, labs(temp_centi % 100L));
    }
    SdDebugLogService_EnqueueLine(csv_line);

    /* Reset active inference */
    s_active_inference.active = false;
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

    float latency_sum = 0.0f;
    float latency_min = (float)s_metrics_buffer[0].latency_us;
    float latency_max = (float)s_metrics_buffer[0].latency_us;
    float delta_sum = 0.0f;
    float delta_min = s_metrics_buffer[0].power_delta_mw;
    float delta_max = s_metrics_buffer[0].power_delta_mw;
    float energy_sum = 0.0f;

    for (uint32_t i = 0; i < s_metrics_count; i++)
    {
        if (!s_metrics_buffer[i].valid)
            continue;

        float lat = (float)s_metrics_buffer[i].latency_us;
        latency_sum += lat;
        if (lat < latency_min)
            latency_min = lat;
        if (lat > latency_max)
            latency_max = lat;

        float delta = s_metrics_buffer[i].power_delta_mw;
        delta_sum += delta;
        if (delta < delta_min)
            delta_min = delta;
        if (delta > delta_max)
            delta_max = delta;

        /* Energy = Power * Time (convert mW * us to uJ) */
        float energy_uj = (s_metrics_buffer[i].power_pre_mw + s_metrics_buffer[i].power_mid_mw) / 2.0f * lat / 1000.0f;
        energy_sum += energy_uj;
    }

    summary->count = s_metrics_count;
    summary->latency_avg_ms = (latency_sum / s_metrics_count) / 1000.0f;
    summary->latency_min_ms = latency_min / 1000.0f;
    summary->latency_max_ms = latency_max / 1000.0f;
    summary->power_delta_avg_mw = delta_sum / s_metrics_count;
    summary->power_delta_min_mw = delta_min;
    summary->power_delta_max_mw = delta_max;
    summary->energy_avg_uj = energy_sum / s_metrics_count;

    return true;
}

/**
 * @brief Log all recorded metrics to console in CSV format.
 */
void Metrics_LogAll(void)
{
    DebugConsole_Printf("\r\n[METRICS] CSV Export:\r\n");
    DebugConsole_Printf("timestamp_ms,label,latency_ms,power_pre_mW,power_mid_mW,power_post_mW,power_delta_mW,temp_c\r\n");

    for (uint32_t i = 0; i < s_metrics_count; i++)
    {
        MetricsRecord_t *r = &s_metrics_buffer[i];
        if (!r->valid)
            continue;

        const long latency_ms_centi = Metrics_ToCenti((float)r->latency_us / 1000.0f);
        const long power_pre_milli = Metrics_ToMilli(r->power_pre_mw);
        const long power_mid_milli = Metrics_ToMilli(r->power_mid_mw);
        const long power_post_milli = Metrics_ToMilli(r->power_post_mw);
        const long power_delta_milli = Metrics_ToMilli(r->power_delta_mw);
        const long temp_milli = Metrics_ToMilli(r->temperature_c);
        DebugConsole_Printf(
            "%lu,%s,%ld.%02ld,%ld.%03ld,%ld.%03ld,%ld.%03ld,%ld.%03ld,%ld.%03ld\r\n",
            (unsigned long)r->timestamp_ms,
            r->label,
            latency_ms_centi / 100L, labs(latency_ms_centi % 100L),
            power_pre_milli / 1000L, labs(power_pre_milli % 1000L),
            power_mid_milli / 1000L, labs(power_mid_milli % 1000L),
            power_post_milli / 1000L, labs(power_post_milli % 1000L),
            power_delta_milli / 1000L, labs(power_delta_milli % 1000L),
            temp_milli / 1000L, labs(temp_milli % 1000L));
    }

    /* Log summary */
    MetricsSummary_t summary;
    if (Metrics_GetSummary(&summary))
    {
        DebugConsole_Printf("\r\n[METRICS] Summary (%lu samples):\r\n", (unsigned long)summary.count);
        const long latency_avg_centi = Metrics_ToCenti(summary.latency_avg_ms);
        const long latency_min_centi = Metrics_ToCenti(summary.latency_min_ms);
        const long latency_max_centi = Metrics_ToCenti(summary.latency_max_ms);
        const long delta_avg_centi = Metrics_ToCenti(summary.power_delta_avg_mw);
        const long delta_min_centi = Metrics_ToCenti(summary.power_delta_min_mw);
        const long delta_max_centi = Metrics_ToCenti(summary.power_delta_max_mw);
        const long energy_avg_centi = Metrics_ToCenti(summary.energy_avg_uj);

        DebugConsole_Printf("  Latency: avg=%ld.%02ld ms, min=%ld.%02ld ms, max=%ld.%02ld ms\r\n",
                            latency_avg_centi / 100L, labs(latency_avg_centi % 100L),
                            latency_min_centi / 100L, labs(latency_min_centi % 100L),
                            latency_max_centi / 100L, labs(latency_max_centi % 100L));
        DebugConsole_Printf("  Power Delta: avg=%ld.%02ld mW, min=%ld.%02ld mW, max=%ld.%02ld mW\r\n",
                            delta_avg_centi / 100L, labs(delta_avg_centi % 100L),
                            delta_min_centi / 100L, labs(delta_min_centi % 100L),
                            delta_max_centi / 100L, labs(delta_max_centi % 100L));
        DebugConsole_Printf("  Energy per inference: %ld.%02ld uJ\r\n",
                            energy_avg_centi / 100L, labs(energy_avg_centi % 100L));
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
