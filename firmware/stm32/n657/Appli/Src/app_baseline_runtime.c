/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_runtime.c
 * @brief   Classical CV baseline worker for gauge temperature estimation.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "main.h"
#include "app_baseline_runtime.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "app_camera_buffers.h"
#include "app_gauge_geometry.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_threadx_config.h"
#include "debug_console.h"
#include "threadx_utils.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef struct {
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

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_BASELINE_PI                     3.14159265358979323846f
#define APP_BASELINE_TWO_PI                 (2.0f * APP_BASELINE_PI)
#define APP_BASELINE_MIN_ANGLE_DEG          135.0f
#define APP_BASELINE_SWEEP_DEG              270.0f
#define APP_BASELINE_MIN_VALUE_C            -30.0f
#define APP_BASELINE_MAX_VALUE_C             50.0f
#define APP_BASELINE_BRIGHT_THRESHOLD       150U
/* Pixels above this luma are considered saturated/glare and excluded from
 * the bright-centroid calculation and ray scoring. */
#define APP_BASELINE_SATURATION_THRESHOLD   220U
#define APP_BASELINE_MIN_BRIGHT_PIXELS      1024U
#define APP_BASELINE_SCAN_BORDER_PIXELS      8U
#define APP_BASELINE_ANGLE_BINS             360U
#define APP_BASELINE_RAY_SAMPLES            32U
#define APP_BASELINE_RAY_START_FRACTION      0.20f
#define APP_BASELINE_RAY_END_FRACTION        0.78f
#define APP_BASELINE_SUBDIAL_X_FRACTION      0.35f
#define APP_BASELINE_SUBDIAL_Y_MIN_FRACTION  0.10f
#define APP_BASELINE_SUBDIAL_Y_MAX_FRACTION  0.58f
#define APP_BASELINE_LOCAL_BACKGROUND_OFFSETS 2U
#define APP_BASELINE_MIN_RADIUS_PIXELS      16U
/* The closer camera setup reduced the baseline margin a bit, so keep the gate
 * permissive enough to accept the still-useful training-crop candidates while
 * preserving the noise floor guardrails. */
#define APP_BASELINE_CONFIDENCE_THRESHOLD    0.055f
/* Scale factor for integer-encoded confidence in log output (avoids %f). */
#define APP_BASELINE_CONFIDENCE_LOG_SCALE    1000L
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static TX_THREAD camera_baseline_thread;
static ULONG camera_baseline_thread_stack[BASELINE_RUNTIME_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_baseline_thread_created = false;
static TX_SEMAPHORE camera_baseline_request_semaphore;
static bool camera_baseline_sync_created = false;
static volatile const uint8_t *camera_baseline_request_frame_ptr = NULL;
static volatile ULONG camera_baseline_request_frame_length = 0U;
static bool app_baseline_runtime_initialized = false;
static volatile bool camera_baseline_last_result_valid = false;
static volatile float camera_baseline_last_temperature_c = 0.0f;
static volatile float camera_baseline_last_angle_rad = 0.0f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static VOID CameraBaselineThread_Entry(ULONG thread_input);
static bool AppBaselineRuntime_EstimateFromFrame(const uint8_t *frame_bytes,
		size_t frame_size, AppBaselineRuntime_Estimate_t *estimate_out);
static bool AppBaselineRuntime_EstimateCenterFromBrightPixels(
		const uint8_t *frame_bytes, size_t frame_size, size_t *center_x_out,
		size_t *center_y_out, size_t *bright_count_out);
static bool AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
		const uint8_t *frame_bytes, size_t frame_size,
		AppBaselineRuntime_Estimate_t *estimate_out);
static bool AppBaselineRuntime_EstimateFromCenterHypothesis(
		const uint8_t *frame_bytes, size_t frame_size, size_t center_x,
		size_t center_y, const char *source_label,
		AppBaselineRuntime_Estimate_t *estimate_out);
static float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad);
static float AppBaselineRuntime_ConvertAngleToFraction(float angle_rad);
static float AppBaselineRuntime_ReadLuma(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t x, size_t y);
static bool AppBaselineRuntime_IsInSubdialMask(size_t center_x, size_t center_y,
		size_t x, size_t y, float radius_px);
static float AppBaselineRuntime_ScoreAngle(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t frame_height_pixels, size_t center_x,
		size_t center_y, float angle_rad);
static float AppBaselineRuntime_ClampFloat(float value, float min_value,
		float max_value);
static long AppBaselineRuntime_RoundToLong(float value);
static void AppBaselineRuntime_LogEstimate(
		const AppBaselineRuntime_Estimate_t *estimate);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

/**
 * @brief Create the synchronization objects used by the baseline worker.
 */
UINT AppBaselineRuntime_Init(void) {
	UINT status = TX_SUCCESS;

	if (app_baseline_runtime_initialized) {
		return TX_SUCCESS;
	}

	status = tx_semaphore_create(&camera_baseline_request_semaphore,
			"camera_baseline_request", 0U);
	if (status != TX_SUCCESS) {
		return status;
	}

	camera_baseline_sync_created = true;
	app_baseline_runtime_initialized = true;
	return TX_SUCCESS;
}

/**
 * @brief Start the baseline worker thread.
 */
UINT AppBaselineRuntime_Start(void) {
	if (!app_baseline_runtime_initialized) {
		const UINT init_status = AppBaselineRuntime_Init();
		if (init_status != TX_SUCCESS) {
			return init_status;
		}
	}

	if (!camera_baseline_thread_created) {
		const UINT create_status = tx_thread_create(
				&camera_baseline_thread, "camera_baseline",
				CameraBaselineThread_Entry, 0U, camera_baseline_thread_stack,
				sizeof(camera_baseline_thread_stack),
				BASELINE_RUNTIME_THREAD_PRIORITY,
				BASELINE_RUNTIME_THREAD_PRIORITY, TX_NO_TIME_SLICE,
				TX_AUTO_START);
		if (create_status != TX_SUCCESS) {
			return create_status;
		}

		camera_baseline_thread_created = true;
		DebugConsole_Printf(
				"[BASELINE][THREAD] Classical CV baseline thread created and started.\r\n");
	}

	return TX_SUCCESS;
}

/**
 * @brief Queue a YUV422 frame for the classical temperature estimate.
 */
bool AppBaselineRuntime_RequestEstimate(const uint8_t *frame_ptr,
		ULONG frame_length) {
	uint8_t first8[8] = { 0 };

	if (!camera_baseline_sync_created) {
		DebugConsole_Printf(
				"[BASELINE] Request dropped; baseline queue not initialized.\r\n");
		return false;
	}

	if ((frame_ptr == NULL) || (frame_length == 0U)) {
		DebugConsole_Printf(
				"[BASELINE] Request dropped; empty frame ptr=%p len=%lu.\r\n",
				(const void *) frame_ptr, (unsigned long) frame_length);
		return false;
	}

	if (frame_length > CAMERA_CAPTURE_BUFFER_SIZE_BYTES) {
		DebugConsole_Printf(
				"[BASELINE] Request dropped; frame too large len=%lu max=%lu.\r\n",
				(unsigned long) frame_length,
				(unsigned long) CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		return false;
	}

	(void) memcpy((void *) camera_baseline_frame_snapshot, frame_ptr,
			(size_t) frame_length);
	(void) memcpy(first8, camera_baseline_frame_snapshot,
			(size_t) ((frame_length < 8U) ? frame_length : 8U));
	DebugConsole_Printf(
			"[BASELINE] Snapshot copied: src=%p dst=%p len=%lu first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			(const void *) frame_ptr, (void *) camera_baseline_frame_snapshot,
			(unsigned long) frame_length, first8[0], first8[1], first8[2],
			first8[3], first8[4], first8[5], first8[6], first8[7]);

	camera_baseline_request_frame_ptr = camera_baseline_frame_snapshot;
	camera_baseline_request_frame_length = frame_length;

	if (tx_semaphore_put(&camera_baseline_request_semaphore) != TX_SUCCESS) {
		DebugConsole_Printf(
				"[BASELINE] Failed to signal baseline request semaphore.\r\n");
		return false;
	}

	return true;
}

/* USER CODE END 0 */

/**
 * @brief Low-priority worker that turns each accepted frame into a temperature.
 */
static VOID CameraBaselineThread_Entry(ULONG thread_input) {
	(void) thread_input;

	(void) DebugConsole_WriteString("[BASELINE] worker alive\r\n");

	while (1) {
		const UINT request_status = tx_semaphore_get(
				&camera_baseline_request_semaphore, TX_WAIT_FOREVER);
		const uint8_t *frame_ptr = NULL;
		ULONG frame_length = 0U;
		AppBaselineRuntime_Estimate_t estimate = { 0 };

		if (request_status != TX_SUCCESS) {
			continue;
		}

		frame_ptr = (const uint8_t *) camera_baseline_request_frame_ptr;
		frame_length = camera_baseline_request_frame_length;
		camera_baseline_request_frame_ptr = NULL;
		camera_baseline_request_frame_length = 0U;

		if ((frame_ptr == NULL) || (frame_length == 0U)) {
			DebugConsole_Printf(
					"[BASELINE] Worker woke without a queued frame; ignoring.\r\n");
			continue;
		}

		if (!AppBaselineRuntime_EstimateFromFrame(frame_ptr,
				(size_t) frame_length, &estimate)) {
			DebugConsole_Printf(
					"[BASELINE] Classical baseline failed to estimate a temperature.\r\n");
			continue;
		}

		camera_baseline_last_result_valid = true;
		camera_baseline_last_temperature_c = estimate.temperature_c;
		camera_baseline_last_angle_rad = estimate.angle_rad;
		AppBaselineRuntime_LogEstimate(&estimate);
	}
}

/**
 * @brief Run the classical CV baseline over one captured YUV422 frame.
 */
static bool AppBaselineRuntime_EstimateFromFrame(const uint8_t *frame_bytes,
		size_t frame_size, AppBaselineRuntime_Estimate_t *estimate_out) {
	AppBaselineRuntime_Estimate_t bright_hypothesis = { 0 };
	AppBaselineRuntime_Estimate_t training_crop_hypothesis = { 0 };
	AppBaselineRuntime_Estimate_t center_hypothesis = { 0 };
	const AppBaselineRuntime_Estimate_t *selected_estimate = NULL;
	bool bright_ok = false;
	bool training_crop_ok = false;
	bool center_ok = false;
	size_t center_x = 0U;
	size_t center_y = 0U;
	size_t bright_count = 0U;

	if ((frame_bytes == NULL) || (estimate_out == NULL)) {
		return false;
	}

	bright_ok = AppBaselineRuntime_EstimateCenterFromBrightPixels(frame_bytes,
			frame_size, &center_x, &center_y, &bright_count);
	if (bright_ok) {
		bright_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
				frame_size, center_x, center_y, "bright-center",
				&bright_hypothesis);
	}
	(void) bright_count;

	training_crop_ok = AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
			frame_bytes, frame_size, &training_crop_hypothesis);

	center_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
			frame_size, CAMERA_CAPTURE_WIDTH_PIXELS / 2U,
			CAMERA_CAPTURE_HEIGHT_PIXELS / 2U, "image-center",
			&center_hypothesis);

	if (!bright_ok && !training_crop_ok && !center_ok) {
		return false;
	}

	/* Prefer the stable training crop whenever it is available. The close-up
	 * setup makes the glare-prone center heuristics swing around too much, so
	 * the production baseline only trusts them when the training crop is absent. */
	if (training_crop_ok) {
		selected_estimate = &training_crop_hypothesis;
	} else if (bright_ok) {
		selected_estimate = &bright_hypothesis;
	} else if (center_ok) {
		selected_estimate = &center_hypothesis;
	}

	if (selected_estimate == NULL) {
		return false;
	}

	*estimate_out = *selected_estimate;

	{
		const long bright_conf_m   = AppBaselineRuntime_RoundToLong(
				(bright_ok ? bright_hypothesis.confidence : 0.0f)
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long train_conf_m    = AppBaselineRuntime_RoundToLong(
				(training_crop_ok ? training_crop_hypothesis.confidence : 0.0f)
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long image_conf_m    = AppBaselineRuntime_RoundToLong(
				(center_ok ? center_hypothesis.confidence : 0.0f)
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long selected_conf_m = AppBaselineRuntime_RoundToLong(
				estimate_out->confidence
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		DebugConsole_Printf(
				"[BASELINE] candidates: bright=%s(%ld) training=%s(%ld) image=%s(%ld) selected=%s(%ld)\r\n",
				bright_ok ? "ok" : "no", bright_conf_m,
				training_crop_ok ? "ok" : "no", train_conf_m,
				center_ok ? "ok" : "no", image_conf_m,
				(estimate_out->source_label != NULL) ? estimate_out->source_label : "unknown",
				selected_conf_m);
	}

	if (estimate_out->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD) {
		const long conf_m      = AppBaselineRuntime_RoundToLong(
				estimate_out->confidence
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long threshold_m = AppBaselineRuntime_RoundToLong(
				APP_BASELINE_CONFIDENCE_THRESHOLD
				* (float) APP_BASELINE_CONFIDENCE_LOG_SCALE);
		DebugConsole_Printf(
				"[BASELINE] Rejected low-confidence estimate: source=%s confidence=%ld threshold=%ld (x1000)\r\n",
				(estimate_out->source_label != NULL) ? estimate_out->source_label : "unknown",
				conf_m, threshold_m);
		return false;
	}

	return true;
}

/**
 * @brief Estimate the needle using the same stable crop that the AI runtime
 * falls back to when the adaptive rectifier is unreliable.
 */
static bool AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
		const uint8_t *frame_bytes, size_t frame_size,
		AppBaselineRuntime_Estimate_t *estimate_out) {
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	size_t center_x = 0U;
	size_t center_y = 0U;

	if ((estimate_out == NULL) || (frame_bytes == NULL)) {
		return false;
	}

	AppGaugeGeometry_TrainingCropCenter(width_pixels, height_pixels,
			&center_x, &center_y);

	return AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
			frame_size, center_x, center_y, "training-crop", estimate_out);
}

/**
 * @brief Estimate a bright dial center from the high-luma pixels.
 */
static bool AppBaselineRuntime_EstimateCenterFromBrightPixels(
		const uint8_t *frame_bytes, size_t frame_size, size_t *center_x_out,
		size_t *center_y_out, size_t *bright_count_out) {
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const size_t stride_bytes = width_pixels * CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const AppGaugeGeometry_Crop_t crop = AppGaugeGeometry_TrainingCrop(
			width_pixels, height_pixels);
	const size_t scan_x_min = crop.x_min;
	const size_t scan_x_max = crop.x_min + crop.width;
	const size_t scan_y_min = crop.y_min;
	const size_t scan_y_max = crop.y_min + crop.height;
	size_t bright_x_min = width_pixels;
	size_t bright_y_min = height_pixels;
	size_t bright_x_max = 0U;
	size_t bright_y_max = 0U;
	size_t bright_count = 0U;
	uint64_t bright_sum_x = 0U;
	uint64_t bright_sum_y = 0U;

	if ((frame_bytes == NULL) || (center_x_out == NULL) || (center_y_out == NULL)
			|| (bright_count_out == NULL)) {
		return false;
	}

	if (frame_size < (stride_bytes * height_pixels)) {
		return false;
	}

	for (size_t y = scan_y_min; y < scan_y_max; ++y) {
		for (size_t x = scan_x_min; x < scan_x_max; ++x) {
			const float luma = AppBaselineRuntime_ReadLuma(frame_bytes,
					width_pixels, x, y);

			if (luma < (float) APP_BASELINE_BRIGHT_THRESHOLD) {
				continue;
			}

			/* Exclude saturated/glare pixels — they skew the centroid away
			 * from the real dial face toward the blown-out reflection spot. */
			if (luma > (float) APP_BASELINE_SATURATION_THRESHOLD) {
				continue;
			}

			bright_count++;
			bright_sum_x += (uint64_t) x;
			bright_sum_y += (uint64_t) y;

			if (x < bright_x_min) {
				bright_x_min = x;
			}
			if (y < bright_y_min) {
				bright_y_min = y;
			}
			if (x > bright_x_max) {
				bright_x_max = x;
			}
			if (y > bright_y_max) {
				bright_y_max = y;
			}
		}
	}

	if (bright_count < APP_BASELINE_MIN_BRIGHT_PIXELS) {
		return false;
	}

	if ((bright_x_max <= bright_x_min) || (bright_y_max <= bright_y_min)) {
		return false;
	}

	*center_x_out = (size_t) (bright_sum_x / (uint64_t) bright_count);
	*center_y_out = (size_t) (bright_sum_y / (uint64_t) bright_count);
	*bright_count_out = bright_count;
	return true;
}

/**
 * @brief Evaluate the needle angle using one center hypothesis.
 */
static bool AppBaselineRuntime_EstimateFromCenterHypothesis(
		const uint8_t *frame_bytes, size_t frame_size, size_t center_x,
		size_t center_y, const char *source_label,
		AppBaselineRuntime_Estimate_t *estimate_out) {
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const size_t stride_bytes = width_pixels * CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG
			* (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	const float max_radius_x =
			(float) ((center_x < (width_pixels - 1U - center_x)) ? center_x
					: (width_pixels - 1U - center_x));
	const float max_radius_y =
			(float) ((center_y < (height_pixels - 1U - center_y)) ? center_y
					: (height_pixels - 1U - center_y));
	const float max_radius = (max_radius_x < max_radius_y) ? max_radius_x
			: max_radius_y;
	float best_score = -1.0f;
	float runner_up_score = -1.0f;
	float best_angle_rad = min_angle_rad;

	if ((estimate_out == NULL) || (frame_bytes == NULL) || (source_label == NULL)) {
		return false;
	}

	if (frame_size < (stride_bytes * height_pixels)) {
		return false;
	}

	if (max_radius < (float) APP_BASELINE_MIN_RADIUS_PIXELS) {
		return false;
	}

	for (size_t angle_index = 0U; angle_index < APP_BASELINE_ANGLE_BINS;
			++angle_index) {
		const float fraction = (APP_BASELINE_ANGLE_BINS > 1U) ?
				((float) angle_index
						/ (float) (APP_BASELINE_ANGLE_BINS - 1U)) : 0.0f;
		const float angle_rad = min_angle_rad + (fraction * sweep_rad);
		const float score = AppBaselineRuntime_ScoreAngle(frame_bytes,
				width_pixels, height_pixels, center_x, center_y, angle_rad);

		if (score > best_score) {
			runner_up_score = best_score;
			best_score = score;
			best_angle_rad = angle_rad;
		} else if (score > runner_up_score) {
			runner_up_score = score;
		}
	}

	if (best_score <= 0.0f) {
		return false;
	}

	estimate_out->valid = true;
	estimate_out->center_x = center_x;
	estimate_out->center_y = center_y;
	estimate_out->angle_rad = best_angle_rad;
	estimate_out->temperature_c =
			AppBaselineRuntime_ConvertAngleToTemperature(best_angle_rad);
	/* Margin-based confidence: absolute gap between best angle and runner-up,
	 * normalised to [0,1] by dividing by 100 (typical score range is 0..~50,
	 * so margin of 8 -> 0.08). SNR-style confidence floored at ~0.04 even
	 * for noise frames (best=32 runner=29 -> 0.094) and let random bright
	 * blobs through; raw margin is a cleaner noise gate. */
	{
		const float margin = ((best_score > runner_up_score) ?
				(best_score - runner_up_score) : 0.0f);
		estimate_out->confidence = AppBaselineRuntime_ClampFloat(
				margin / 100.0f, 0.0f, 1.0f);
	}
	estimate_out->best_score = best_score;
	estimate_out->runner_up_score = runner_up_score;
	estimate_out->source_label = source_label;

	return true;
}

/**
 * @brief Map an angle inside the gauge sweep to a temperature.
 */
static float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad) {
	return APP_BASELINE_MIN_VALUE_C
			+ (AppBaselineRuntime_ConvertAngleToFraction(angle_rad)
					* (APP_BASELINE_MAX_VALUE_C - APP_BASELINE_MIN_VALUE_C));
}

/**
 * @brief Convert an angle to the calibrated [0, 1] sweep fraction.
 */
static float AppBaselineRuntime_ConvertAngleToFraction(float angle_rad) {
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG
			* (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	float shifted = angle_rad - min_angle_rad;

	while (shifted < 0.0f) {
		shifted += APP_BASELINE_TWO_PI;
	}
	while (shifted >= APP_BASELINE_TWO_PI) {
		shifted -= APP_BASELINE_TWO_PI;
	}

	shifted = AppBaselineRuntime_ClampFloat(shifted, 0.0f, sweep_rad);
	return AppBaselineRuntime_ClampFloat(shifted / sweep_rad, 0.0f, 1.0f);
}

/**
 * @brief Read the Y component from one packed YUV422 pixel.
 */
static float AppBaselineRuntime_ReadLuma(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t x, size_t y) {
	const size_t row_stride_bytes = frame_width_pixels * 2U;
	const size_t pair_offset = (y * row_stride_bytes) + ((x & ~1U) * 2U);
	const size_t y_offset = pair_offset + (((x & 1U) != 0U) ? 2U : 0U);

	return (float) frame_bytes[y_offset];
}

/**
 * @brief Check whether a sampled point falls inside the subdial clutter mask.
 */
static bool AppBaselineRuntime_IsInSubdialMask(size_t center_x, size_t center_y,
		size_t x, size_t y, float radius_px) {
	const float dx = (float) ((x >= center_x) ? (x - center_x) : (center_x - x));
	const float dy = (float) ((y >= center_y) ? (y - center_y) : (center_y - y));

	return (dx < (APP_BASELINE_SUBDIAL_X_FRACTION * radius_px))
			&& ((float) y > ((float) center_y
					+ (APP_BASELINE_SUBDIAL_Y_MIN_FRACTION * radius_px)))
			&& ((float) y < ((float) center_y
					+ (APP_BASELINE_SUBDIAL_Y_MAX_FRACTION * radius_px)))
			&& (dy > (APP_BASELINE_SUBDIAL_Y_MIN_FRACTION * radius_px));
}

/**
 * @brief Score one ray candidate by favoring dark pixels on the needle line.
 */
static float AppBaselineRuntime_ScoreAngle(const uint8_t *frame_bytes,
		size_t frame_width_pixels, size_t frame_height_pixels, size_t center_x,
		size_t center_y, float angle_rad) {
	const float unit_dx = cosf(angle_rad);
	const float unit_dy = sinf(angle_rad);
	const float perp_dx = -unit_dy;
	const float perp_dy = unit_dx;
	const float center_x_f = (float) center_x;
	const float center_y_f = (float) center_y;
	const float max_radius_x =
			(float) ((center_x < (frame_width_pixels - 1U - center_x)) ? center_x
					: (frame_width_pixels - 1U - center_x));
	const float max_radius_y =
			(float) ((center_y < (frame_height_pixels - 1U - center_y)) ? center_y
					: (frame_height_pixels - 1U - center_y));
	const float max_radius = (max_radius_x < max_radius_y) ? max_radius_x
			: max_radius_y;
	const float start_radius = max_radius * APP_BASELINE_RAY_START_FRACTION;
	const float end_radius = max_radius * APP_BASELINE_RAY_END_FRACTION;
	const float radius_step =
			(APP_BASELINE_RAY_SAMPLES > 1U) ?
					((end_radius - start_radius)
							/ (float) (APP_BASELINE_RAY_SAMPLES - 1U)) : 0.0f;
	float score = 0.0f;
	size_t valid_sample_count = 0U;

	for (size_t sample_index = 0U; sample_index < APP_BASELINE_RAY_SAMPLES;
			++sample_index) {
		const float radius = start_radius + (radius_step * (float) sample_index);
		const float weight =
				0.50f + (0.50f * ((float) sample_index
						/ (float) (APP_BASELINE_RAY_SAMPLES - 1U)));
		const long sample_x = AppBaselineRuntime_RoundToLong(
				center_x_f + (unit_dx * radius));
		const long sample_y = AppBaselineRuntime_RoundToLong(
				center_y_f + (unit_dy * radius));
		float background_sum = 0.0f;
		size_t background_count = 0U;

		if ((sample_x < 0L) || (sample_y < 0L)
				|| ((size_t) sample_x >= frame_width_pixels)
				|| ((size_t) sample_y >= frame_height_pixels)) {
			continue;
		}

		if (AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
				(size_t) sample_x, (size_t) sample_y, max_radius)) {
			continue;
		}

		/* Skip this sample entirely if the line pixel itself is saturated —
		 * glare makes both the needle and its background equally bright, so
		 * the contrast score is meaningless and including it only dilutes
		 * valid_sample_count without adding useful signal. */
		{
			const float line_luma_check = AppBaselineRuntime_ReadLuma(
					frame_bytes, frame_width_pixels,
					(size_t) sample_x, (size_t) sample_y);
			if (line_luma_check > (float) APP_BASELINE_SATURATION_THRESHOLD) {
				continue;
			}
		}

		for (size_t offset_index = 0U;
				offset_index < APP_BASELINE_LOCAL_BACKGROUND_OFFSETS;
				++offset_index) {
			const float offset = 2.0f + (2.0f * (float) offset_index);
			const long left_x = AppBaselineRuntime_RoundToLong(
					((float) sample_x) + (perp_dx * offset));
			const long left_y = AppBaselineRuntime_RoundToLong(
					((float) sample_y) + (perp_dy * offset));
			const long right_x = AppBaselineRuntime_RoundToLong(
					((float) sample_x) - (perp_dx * offset));
			const long right_y = AppBaselineRuntime_RoundToLong(
					((float) sample_y) - (perp_dy * offset));
			const bool left_in_bounds = (left_x >= 0L)
					&& (left_y >= 0L)
					&& ((size_t) left_x < frame_width_pixels)
					&& ((size_t) left_y < frame_height_pixels);
			const bool right_in_bounds = (right_x >= 0L)
					&& (right_y >= 0L)
					&& ((size_t) right_x < frame_width_pixels)
					&& ((size_t) right_y < frame_height_pixels);

			if (left_in_bounds
					&& !AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
							(size_t) left_x, (size_t) left_y, max_radius)) {
				const float bg_luma = AppBaselineRuntime_ReadLuma(frame_bytes,
						frame_width_pixels, (size_t) left_x, (size_t) left_y);
				/* Skip saturated background samples — they don't reflect true
				 * dial-face brightness and would lower the measured contrast. */
				if (bg_luma <= (float) APP_BASELINE_SATURATION_THRESHOLD) {
					background_sum += bg_luma;
					background_count++;
				}
			}

			if (right_in_bounds
					&& !AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
							(size_t) right_x, (size_t) right_y, max_radius)) {
				const float bg_luma_r = AppBaselineRuntime_ReadLuma(frame_bytes,
						frame_width_pixels, (size_t) right_x, (size_t) right_y);
				if (bg_luma_r <= (float) APP_BASELINE_SATURATION_THRESHOLD) {
					background_sum += bg_luma_r;
					background_count++;
				}
			}
		}

		if (background_count == 0U) {
			continue;
		}

		{
			const float line_luma = AppBaselineRuntime_ReadLuma(frame_bytes,
					frame_width_pixels, (size_t) sample_x, (size_t) sample_y);
			const float local_background = background_sum
					/ (float) background_count;
			const float local_contrast = local_background - line_luma;

			if (local_contrast <= 0.0f) {
				continue;
			}

			score += (local_contrast * weight);
			valid_sample_count++;
		}
	}

	if (valid_sample_count == 0U) {
		return 0.0f;
	}

	return score / (float) valid_sample_count;
}

/**
 * @brief Clamp a float to a closed range.
 */
static float AppBaselineRuntime_ClampFloat(float value, float min_value,
		float max_value) {
	if (value < min_value) {
		return min_value;
	}
	if (value > max_value) {
		return max_value;
	}

	return value;
}

/**
 * @brief Round a float to the nearest long for logging and pixel lookup.
 */
static long AppBaselineRuntime_RoundToLong(float value) {
	if (value >= 0.0f) {
		return (long) (value + 0.5f);
	}

	return (long) (value - 0.5f);
}

/**
 * @brief Print the selected baseline estimate in a compact, thesis-friendly form.
 */
static void AppBaselineRuntime_LogEstimate(
		const AppBaselineRuntime_Estimate_t *estimate) {
	char temperature_line[96] = { 0 };
	long angle_tenths = 0L;
	long confidence_thousandths = 0L;
	long score_whole = 0L;
	long runner_up_whole = 0L;
	long angle_abs_tenths = 0L;
	long confidence_abs_thousandths = 0L;

	if ((estimate == NULL) || !estimate->valid) {
		return;
	}

	AppInferenceLog_FormatFloatTenths(temperature_line,
			sizeof(temperature_line),
			"[BASELINE] Temperature estimate: ", estimate->temperature_c);
	(void) DebugConsole_WriteString(temperature_line);

	angle_tenths = AppBaselineRuntime_RoundToLong(
			(estimate->angle_rad * 180.0f / APP_BASELINE_PI) * 10.0f);
	confidence_thousandths = AppBaselineRuntime_RoundToLong(
			estimate->confidence * 1000.0f);
	angle_abs_tenths = (angle_tenths < 0L) ? -angle_tenths : angle_tenths;
	confidence_abs_thousandths =
			(confidence_thousandths < 0L) ? -confidence_thousandths
					: confidence_thousandths;
	score_whole = AppBaselineRuntime_RoundToLong(estimate->best_score);
	runner_up_whole = AppBaselineRuntime_RoundToLong(
			estimate->runner_up_score);

	DebugConsole_Printf(
			"[BASELINE] details: center=(%lu,%lu) source=%s angle=%ld.%01lddeg confidence=%ld.%03ld score=%ld runner_up=%ld\r\n",
			(unsigned long) estimate->center_x,
			(unsigned long) estimate->center_y,
			(estimate->source_label != NULL) ? estimate->source_label : "unknown",
			(long) (angle_tenths / 10L), (long) (angle_abs_tenths % 10L),
			(long) (confidence_thousandths / 1000L),
			(long) (confidence_abs_thousandths % 1000L),
			score_whole, runner_up_whole);
}

/* USER CODE END 0 */
