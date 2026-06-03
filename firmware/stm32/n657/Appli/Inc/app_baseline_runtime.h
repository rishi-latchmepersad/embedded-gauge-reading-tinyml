/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_runtime.h
 * @brief   Classical CV baseline worker for temperature estimation.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_BASELINE_RUNTIME_H
#define __APP_BASELINE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tx_api.h"

/** @brief A polar-vote estimate: center, angle, temperature, and quality. */
typedef struct
{
	bool valid;
	size_t center_x;
	size_t center_y;
	float angle_rad;
	float temperature_c;
	float confidence;
	float best_score;
	float runner_up_score;
	const char *source_label;
} AppBaselineRuntime_Estimate_t;



/**
 * @brief Initialize the baseline runtime synchronization objects.
 *
 * The baseline worker runs independently from the learned AI runtime so the
 * classical comparator can stay alive even when the model worker is busy.
 *
 * @retval TX_SUCCESS on success.
 */
UINT AppBaselineRuntime_Init(void);

/**
 * @brief Start the baseline worker thread.
 *
 * The worker consumes copied YUV422 frames from a private snapshot buffer and
 * emits a temperature estimate for each accepted camera frame.
 *
 * @retval TX_SUCCESS on success.
 */
UINT AppBaselineRuntime_Start(void);

/**
 * @brief Queue a frame for the baseline temperature estimate.
 *
 * @param frame_ptr Pointer to the captured frame bytes.
 * @param frame_length Number of valid bytes in the frame.
 * @retval true when the request was queued successfully.
 * @retval false when the runtime is unavailable or the frame is invalid.
 */
bool AppBaselineRuntime_RequestEstimate(const uint8_t *frame_ptr,
		ULONG frame_length);

/**
 * @brief Run the polar-vote needle detector at a given center point.
 */
bool AppBaselineRuntime_EstimatePolarNeedle(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, size_t center_x, size_t center_y,
	float dial_radius_px,
	const char *source_label, AppBaselineRuntime_Estimate_t *estimate_out);

/**
 * @brief Find the dial center by rim-edge alignment.
 *
 * Searches for the circular gauge rim using edge gradients and radial
 * alignment.  Used as the primary center detector in the AI pipeline.
 */
bool AppBaselineRuntime_EstimateDialCenterFromRimVotes(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, float dial_radius_px,
	size_t *center_x_out, size_t *center_y_out, float *center_quality_out);

/**
 * @brief Map angle to temperature.
 */
float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad);

#ifdef __cplusplus
}
#endif

bool AppBaselineRuntime_GetLastEstimate(float *temp_out,
													 float *confidence_out);

/**
 * @brief Retrieve the version counter for the last accepted baseline result.
 *
 * The counter increments each time the baseline worker stores a fresh value.
 * Callers can use it to distinguish a new estimate from a stale carry-over.
 */
ULONG AppBaselineRuntime_GetLastEstimateGeneration(void);

/**
 * @brief Retrieve the version of the most recently queued baseline request.
 */
ULONG AppBaselineRuntime_GetRequestGeneration(void);
#endif /* __APP_BASELINE_RUNTIME_H */
