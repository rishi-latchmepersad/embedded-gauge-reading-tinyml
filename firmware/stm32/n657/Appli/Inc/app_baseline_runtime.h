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

/**
 * @brief Temperature calibration profile for a specific gauge.
 *
 * The live firmware keeps the vision stages shared, then applies one of these
 * profiles to convert the detected needle angle into a board-specific
 * temperature.  That gives us a scalable per-gauge hook without baking the
 * correction directly into the vision pipeline.
 *
 * Profiles may use the affine fallback fields or provide explicit
 * angle/temperature anchors for piecewise interpolation.  Anchor points are
 * ordered in sweep-fraction order from hot to cold, and the runtime will use
 * them when a profile defines at least two valid anchors.
 */
#define APP_BASELINE_CALIBRATION_MAX_POINTS 4U

/** @brief One temperature anchor on a gauge calibration curve. */
typedef struct
{
	float angle_deg;
	float temperature_c;
} AppBaselineRuntime_CalibrationPoint_t;

typedef struct
{
	const char *profile_name;
	float angle_offset_deg;
	float temperature_pivot_c;
	float temperature_gain;
	size_t calibration_point_count;
	AppBaselineRuntime_CalibrationPoint_t
		calibration_points[APP_BASELINE_CALIBRATION_MAX_POINTS];
} AppBaselineRuntime_CalibrationProfile_t;

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
 * The worker consumes the immutable MONO_Y8 snapshot published by the camera
 * coordinator and emits a separately tagged temperature estimate for each
 * accepted camera frame.
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

/** @brief State of the classical baseline worker and its current request. */
typedef enum
{
	APP_BASELINE_WORKER_UNINITIALIZED = 0,
	APP_BASELINE_WORKER_WAITING,
	APP_BASELINE_WORKER_QUEUED,
	APP_BASELINE_WORKER_EXECUTING,
	APP_BASELINE_WORKER_PUBLISHING,
	APP_BASELINE_WORKER_FAILED,
} AppBaselineRuntime_WorkerState_t;

/** @brief Return the baseline worker state for bounded-wait diagnostics. */
AppBaselineRuntime_WorkerState_t AppBaselineRuntime_GetWorkerState(void);

/** @brief Return the last baseline worker progress tick. */
ULONG AppBaselineRuntime_GetWorkerProgressTick(void);

/** @brief Convert a baseline worker state to a diagnostic label. */
const char *AppBaselineRuntime_WorkerStateName(
		AppBaselineRuntime_WorkerState_t state);

/**
 * @brief Report whether the baseline worker is using the shared snapshot.
 * @retval true while a request is queued or being processed.
 */
bool AppBaselineRuntime_IsEstimateInFlight(void);

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
const AppBaselineRuntime_CalibrationProfile_t *
AppBaselineRuntime_GetCalibrationProfile(void);

/**
 * @brief Select the active gauge calibration profile.
 *
 * Pass NULL to restore the built-in default board profile.
 *
 * @param profile Calibration profile to use for later angle-to-temperature
 *        conversions.
 */
void AppBaselineRuntime_SetCalibrationProfile(
	const AppBaselineRuntime_CalibrationProfile_t *profile);

/**
 * @brief Select the active gauge calibration profile by name.
 *
 * Unknown names fall back to the built-in default profile so the board keeps
 * producing a temperature instead of hard-failing at boot.
 *
 * @param profile_name Registered profile name, or NULL for the default.
 */
void AppBaselineRuntime_SetCalibrationProfileByName(const char *profile_name);

/**
 * @brief Map a north-zero signed needle angle to a temperature using the
 * active calibration profile.
 *
 * Shared by the classical baseline and the learned AI path so a gauge swap
 * is a single profile registry change. Anchor points are ordered hot-to-cold
 * in north-zero signed degrees (the default two-point profile reproduces the
 * linear gauge-1 sweep exactly).
 * @param angle_deg North-zero signed needle angle in degrees.
 * @retval The mapped temperature in Celsius.
 */
float AppBaselineRuntime_MapAngleToTemperature(float angle_deg);

/**
 * @brief Map angle to temperature using the active gauge calibration profile.
 */
float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad);

/**
 * @brief Map an angle linearly between the active profile's extreme anchors.
 *
 * This intentionally ignores interior calibration anchors. It is used by the
 * AI geometry path when the model must interpolate consistently from the
 * cold-end to the hot-end gauge limits.
 */
float AppBaselineRuntime_ConvertAngleToTemperatureExtremes(float angle_rad);

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
