/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    inference_metrics.h
 * @brief   Unified inference metrics tracking for power and latency.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __INFERENCE_METRICS_H
#define __INFERENCE_METRICS_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdbool.h>
#include <stdint.h>
#include "stm32n6xx_hal.h"

/* Public defines ------------------------------------------------------------*/
#define METRICS_LABEL_MAX_LEN 32U
#define METRICS_MAX_SAMPLES 100U

    /* Public typedefs -----------------------------------------------------------*/

    /**
     * @brief Single inference metrics record
     */
    typedef struct
    {
        char label[METRICS_LABEL_MAX_LEN]; /* "CNN", "BASELINE", etc. */
        uint32_t timestamp_ms;             /* When sample was taken */
        uint32_t latency_us;               /* Inference duration in microseconds */
        float power_pre_w;                 /* Power before inference */
        float power_mid_w;                 /* Power during inference (if available) */
        float power_post_w;                /* Power after inference */
        float power_delta_w;               /* power_mid - power_pre */
        float temperature_c;               /* Inference result (if applicable) */
        bool valid;                        /* True if record is valid */
    } MetricsRecord_t;

    /**
     * @brief Metrics summary statistics
     */
    typedef struct
    {
        uint32_t count;
        float latency_avg_ms;
        float latency_min_ms;
        float latency_max_ms;
        float power_delta_avg_w;
        float power_delta_min_w;
        float power_delta_max_w;
        float energy_avg_uj; /* Average energy per inference */
    } MetricsSummary_t;

    /* Public API ----------------------------------------------------------------*/

    /**
     * @brief Initialize the metrics subsystem.
     */
    void Metrics_Init(void);

    /**
     * @brief Start a new inference timing session.
     * @param label Label for this inference (e.g., "CNN", "BASELINE")
     */
    void Metrics_StartInference(const char *label);

    /**
     * @brief Mark a checkpoint during inference (for mid-inference power).
     * @param checkpoint_name Name of checkpoint (e.g., "MID", "OBB-DONE")
     */
    void Metrics_Checkpoint(const char *checkpoint_name);

    /**
     * @brief Complete the inference and record metrics.
     * @param temperature_c Optional inference result (NaN if not applicable)
     */
    void Metrics_EndInference(float temperature_c);

    /**
     * @brief Get the last completed metrics record.
     * @param record Pointer to fill with last record
     * @retval true if a valid record exists
     */
    bool Metrics_GetLastRecord(MetricsRecord_t *record);

    /**
     * @brief Get summary statistics for all recorded inferences.
     * @param summary Pointer to fill with statistics
     * @retval true if summary calculated successfully
     */
    bool Metrics_GetSummary(MetricsSummary_t *summary);

    /**
     * @brief Log all recorded metrics to console in CSV format.
     */
    void Metrics_LogAll(void);

    /**
     * @brief Clear all recorded metrics.
     */
    void Metrics_Clear(void);

    /**
     * @brief Get current timestamp in microseconds.
     * @retval Microseconds since boot
     */
    uint32_t Metrics_GetMicros(void);

#ifdef __cplusplus
}
#endif

#endif /* __INFERENCE_METRICS_H */
