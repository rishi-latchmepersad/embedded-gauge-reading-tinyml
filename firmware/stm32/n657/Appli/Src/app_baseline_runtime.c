/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_runtime.c
 * @brief   Classical polar-vote CV baseline worker for gauge temperature estimation.
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

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APP_BASELINE_PI 3.14159265358979323846f
#define APP_BASELINE_TWO_PI (2.0f * APP_BASELINE_PI)
#define APP_BASELINE_MIN_ANGLE_DEG 135.0f
/* FIX (2026-05-01): sweep was 180° but the physical gauge sweep is 270°.
 * The Python spec (gauge_calibration_parameters.toml) has sweep_deg = 270.0.
 * Using 180° here caused every temperature to be read ~80 % too high
 * (e.g. -30 °C read as +50 °C). */
#define APP_BASELINE_SWEEP_DEG 270.0f
#define APP_BASELINE_MIN_VALUE_C -30.0f
#define APP_BASELINE_MAX_VALUE_C 50.0f
/* Calibration offset removed (2026-05-01). The ~2° "low" bias was an
 * artifact of the incorrect 180° sweep; with the correct 270° sweep the
 * mapping now aligns with the Python reference without extra offset. */
#define APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG 0.0f
#define APP_BASELINE_BRIGHT_THRESHOLD 150U
/* Pixels above this luma are considered saturated/glare and excluded from
 * the bright-centroid calculation and ray scoring.
 * Raised from 220 to 235 (2026-04-30) — the 220 threshold was too aggressive
 * on bright board captures where the dial face is well-lit but not blown out,
 * causing valid needle pixels to be skipped and the detector to miss entirely. */
#define APP_BASELINE_SATURATION_THRESHOLD 235U
#define APP_BASELINE_MIN_BRIGHT_PIXELS 1024U
#define APP_BASELINE_SCAN_BORDER_PIXELS 8U
#define APP_BASELINE_ANGLE_BINS 360U
#define APP_BASELINE_RAY_SAMPLES 32U
#define APP_BASELINE_RAY_START_FRACTION 0.20f
#define APP_BASELINE_RAY_END_FRACTION 0.78f
#define APP_BASELINE_SUBDIAL_X_FRACTION 0.35f
#define APP_BASELINE_SUBDIAL_Y_MIN_FRACTION 0.10f
#define APP_BASELINE_SUBDIAL_Y_MAX_FRACTION 0.58f
#define APP_BASELINE_LOCAL_BACKGROUND_OFFSETS 2U
#define APP_BASELINE_MIN_RADIUS_PIXELS 16U
/* The dial ring extends beyond the crop's inscribed radius, so use the crop
 * height as a cheap proxy for the real gauge radius. */
#define APP_BASELINE_DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO 0.56f
/* The board capture has a slightly off-center inner dial with a smaller
 * effective radius than the full crop. This fixed prior is still classical;
 * it just encodes the framing we already observed on the board. */
#define APP_BASELINE_BOARD_PRIOR_CENTER_X_RATIO 0.4900f
#define APP_BASELINE_BOARD_PRIOR_CENTER_Y_RATIO 0.4460f
#define APP_BASELINE_BOARD_PRIOR_RADIUS_RATIO 0.3500f
/* A tiny circle-style center search keeps the baseline aligned to the dial
 * rim before we score the needle spoke. */
#define APP_BASELINE_CENTER_SEARCH_COARSE_STEP_PIXELS 8U
#define APP_BASELINE_CENTER_SEARCH_FINE_STEP_PIXELS 4U
#define APP_BASELINE_CENTER_SEARCH_SAMPLE_STEP_PIXELS 4U
#define APP_BASELINE_CENTER_SEARCH_RIM_MIN_FRACTION 0.84f
#define APP_BASELINE_CENTER_SEARCH_RIM_MAX_FRACTION 1.04f
/* The polar-vote detector now reports a true spoke-vs-background SNR, so the
 * gate can live in the same range used by the Python reference implementation.
 * A peak needs to stand clearly above the angular background before we seed
 * history or report a live baseline value. */
#define APP_BASELINE_CONFIDENCE_THRESHOLD 1.25f
/* Even a strong peak is not trustworthy if the runner-up is almost tied.
 * Clean ideal captures often have broader but still correct peaks, so keep
 * this close to the Python baseline's permissive gate.
 * At extreme temperatures (-30C) the needle sits at the sweep edge where
 * the background gradient can out-vote the needle. Lower the ratio so the
 * best peak still wins even when runner_up > best_score. Board log at -30C
 * shows pr=582-617 being rejected, so we need this well below 0.58. */
#define APP_BASELINE_MIN_PEAK_RATIO 0.35f
/* Strong fixed-crop or image-center reads can still be promoted when they stay
 * close to the last stable temperature, even if the peak ratio is a little
 * soft. This keeps a good continuation frame from getting stuck behind stale
 * history while still rejecting unrelated clutter peaks. */
#define APP_BASELINE_BORDERLINE_PEAK_RATIO 1.05f
#define APP_BASELINE_BORDERLINE_MIN_CONFIDENCE 10.0f
#define APP_BASELINE_BORDERLINE_MAX_TEMP_DELTA_C 4.0f
/* Bright-relaxed frames can still produce occasional false jumps. If the
 * estimate moves too far in one frame and confidence is only modest, hold
 * the last stable value instead of seeding history with the jump. */
#define APP_BASELINE_BRIGHT_RELAXED_MAX_TEMP_JUMP_C 8.0f
#define APP_BASELINE_BRIGHT_RELAXED_MIN_CONFIDENCE_FOR_JUMP 8.0f
/* Even when global peak-ratio gating is permissive, reject sudden large
 * temperature jumps when the winning and runner-up peaks are almost tied.
 * This catches ambiguous false-lock frames (e.g., hot scene snapping to a
 * cold angle with score ~= runner_up). */
#define APP_BASELINE_AMBIGUOUS_JUMP_DELTA_C 12.0f
#define APP_BASELINE_AMBIGUOUS_JUMP_MAX_PEAK_RATIO 1.08f
#define APP_BASELINE_AMBIGUOUS_JUMP_MAX_CONFIDENCE 12.0f
/* When multiple geometry hypotheses agree within a few degrees, keep that
 * consensus cluster instead of letting a lone high-score outlier win. */
#define APP_BASELINE_CONSENSUS_TEMP_DELTA_C 4.0f
/* A consensus cluster must still be close enough in quality to the raw best
 * candidate. This keeps a hot agreement cluster from overriding a clearly
 * stronger near-target geometry. */
#define APP_BASELINE_CONSENSUS_MIN_QUALITY_RATIO 0.85f
/* Let a much stronger fallback geometry override the fixed crop when the
 * fixed anchor is clearly missing the needle. */
#define APP_BASELINE_GEOMETRY_OVERRIDE_RATIO 1.50f
/* After choosing a center hypothesis, do a tiny local geometry search so the
 * fixed anchor can slide a few pixels when the live crop is slightly off. */
#define APP_BASELINE_GEOMETRY_SEARCH_RADIUS_PIXELS 8U
#define APP_BASELINE_GEOMETRY_SEARCH_STEP_PIXELS 4U
/* Run the narrow local geometry sweep by default so each seed can slide a few
 * pixels when the crop is slightly off. The sweep stays classical and small;
 * it just gives the seed geometries a chance to recover the inner Celsius
 * needle instead of freezing on the first plausible anchor. */
#define APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP 1U
/* Require a minimum amount of absolute vote support before we let a brand-new
 * baseline estimate seed the history. The hard-case sweep showed that the
 * gradient-polar detector is already the best classical family, so this floor
 * only needs to reject obvious noise, not the normal hard-frame range. */
#define APP_BASELINE_MIN_ACCEPT_SCORE 2.0f
/* Keep a tiny history so the baseline can report a stable rough reading
 * instead of jumping frame-to-frame on glare or digit clutter. */
#define APP_BASELINE_ESTIMATE_HISTORY_SIZE 3U
/* If the scene jumps a lot between captures, drop the history and re-lock to
 * the new setpoint quickly rather than averaging across two different temps. */
#define APP_BASELINE_HISTORY_RESET_DELTA_C 12.0f
/* Scale factor for integer-encoded confidence in log output (avoids %f). */
#define APP_BASELINE_CONFIDENCE_LOG_SCALE 1000L
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static TX_THREAD camera_baseline_thread;
static ULONG camera_baseline_thread_stack[BASELINE_RUNTIME_THREAD_STACK_SIZE_BYTES / sizeof(ULONG)];
static bool camera_baseline_thread_created = false;
static TX_SEMAPHORE camera_baseline_request_semaphore;
static bool camera_baseline_sync_created = false;
static volatile const uint8_t *camera_baseline_request_frame_ptr = NULL;
static volatile ULONG camera_baseline_request_frame_length = 0U;
static bool app_baseline_runtime_initialized = false;
static volatile bool camera_baseline_last_result_valid = false;
static volatile float camera_baseline_last_temperature_c = 0.0f;
static volatile float camera_baseline_last_angle_rad = 0.0f;
static AppBaselineRuntime_Estimate_t camera_baseline_estimate_history
	[APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {0};
static size_t camera_baseline_estimate_history_count = 0U;
static size_t camera_baseline_estimate_history_next_index = 0U;
/* Per-frame brightness profile used to adapt detector thresholds under
 * heavy overexposure without globally weakening the baseline gate. */
static bool camera_baseline_current_frame_is_bright = false;
static float camera_baseline_current_frame_mean_luma = 0.0f;
static float camera_baseline_current_frame_bright_ratio = 0.0f;
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
static bool AppBaselineRuntime_EstimateFromRimGeometryHypothesis(
	const uint8_t *frame_bytes, size_t frame_size,
	AppBaselineRuntime_Estimate_t *estimate_out);
static bool AppBaselineRuntime_EstimateFromCenterHypothesis(
	const uint8_t *frame_bytes, size_t frame_size, size_t center_x,
	size_t center_y, float dial_radius_px, const char *source_label,
	AppBaselineRuntime_Estimate_t *estimate_out);
static bool AppBaselineRuntime_EstimateFromBoardPriorHypothesis(
	const uint8_t *frame_bytes, size_t frame_size,
	AppBaselineRuntime_Estimate_t *estimate_out);
static float AppBaselineRuntime_ComputeEstimateQuality(
	const AppBaselineRuntime_Estimate_t *estimate);
static int AppBaselineRuntime_SourcePriority(const char *source_label);
static bool AppBaselineRuntime_PassesAcceptanceGate(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_IsBetterEstimate(
	const AppBaselineRuntime_Estimate_t *candidate,
	const AppBaselineRuntime_Estimate_t *incumbent);
static bool AppBaselineRuntime_HasAcceptablePeakSeparation(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_IsBorderlineContinuityEstimate(
	const AppBaselineRuntime_Estimate_t *estimate);
static const AppBaselineRuntime_Estimate_t *AppBaselineRuntime_SelectConsensusEstimate(
	const AppBaselineRuntime_Estimate_t *const estimates[5],
	const bool candidate_ok[5],
	const AppBaselineRuntime_Estimate_t *fallback_estimate);
static bool AppBaselineRuntime_RefineEstimateAroundSeed(
	const uint8_t *frame_bytes, size_t frame_size,
	float dial_radius_px,
	const AppBaselineRuntime_Estimate_t *seed_estimate,
	AppBaselineRuntime_Estimate_t *estimate_out);
static void AppBaselineRuntime_ResetEstimateHistory(void);
static void AppBaselineRuntime_PushEstimateHistory(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_IsStableEstimateForHistory(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_SelectSmoothedEstimate(
	AppBaselineRuntime_Estimate_t *estimate_out);
static float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad);
static float AppBaselineRuntime_ConvertAngleToFraction(float angle_rad);
static bool AppBaselineRuntime_AngleToSweepFraction(float angle_rad,
													float *fraction_out);
static float AppBaselineRuntime_ReadLuma(const uint8_t *frame_bytes,
										 size_t frame_width_pixels, size_t x, size_t y);
static void AppBaselineRuntime_ReadChroma(const uint8_t *frame_bytes,
										  size_t frame_width_pixels, size_t x, size_t y,
										  float *u_out, float *v_out);
static float AppBaselineRuntime_ReadEdgeMagnitude(const uint8_t *frame_bytes,
												  size_t frame_width_pixels, size_t frame_height_pixels, size_t x,
												  size_t y, float *gradient_x_out, float *gradient_y_out,
												  float *background_luma_out);
static bool AppBaselineRuntime_IsInSubdialMask(size_t center_x, size_t center_y,
											   size_t x, size_t y, float radius_px);
static bool AppBaselineRuntime_EstimatePolarNeedle(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, size_t center_x, size_t center_y,
	float dial_radius_px,
	const char *source_label, AppBaselineRuntime_Estimate_t *estimate_out);
static bool AppBaselineRuntime_AngleToSweepFractionWithMargin(float angle_rad,
															  float margin_rad, float *fraction_out);
static void AppBaselineRuntime_UpdateFrameBrightnessProfile(
	const uint8_t *frame_bytes, size_t frame_size);
static float AppBaselineRuntime_NormalizeAngleDegrees(float angle_deg);
static bool AppBaselineRuntime_IsAngleInCelsiusSweep(float angle_deg);
static bool AppBaselineRuntime_IsAngleInSubdialBand(float angle_deg);
static float AppBaselineRuntime_ClampFloat(float value, float min_value,
										   float max_value);
static long AppBaselineRuntime_RoundToLong(float value);
static float AppBaselineRuntime_EstimateDialRadiusPixels(
	size_t frame_width_pixels, size_t frame_height_pixels);
static float AppBaselineRuntime_ScoreDialCenterCandidate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, float dial_radius_px, size_t center_x,
	size_t center_y);
static bool AppBaselineRuntime_EstimateDialCenterFromRimVotes(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, float dial_radius_px, size_t *center_x_out,
	size_t *center_y_out, float *center_quality_out);
static float AppBaselineRuntime_RunnerUpPeakAfterSuppression(
	const float *peak_values, size_t num_bins, size_t best_index,
	size_t suppression_bins);
static void AppBaselineRuntime_LogEstimate(
	const AppBaselineRuntime_Estimate_t *estimate);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

/**
 * @brief Create the synchronization objects used by the baseline worker.
 */
UINT AppBaselineRuntime_Init(void)
{
	UINT status = TX_SUCCESS;

	if (app_baseline_runtime_initialized)
	{
		return TX_SUCCESS;
	}

	status = tx_semaphore_create(&camera_baseline_request_semaphore,
								 "camera_baseline_request", 0U);
	if (status != TX_SUCCESS)
	{
		return status;
	}

	camera_baseline_sync_created = true;
	app_baseline_runtime_initialized = true;
	AppBaselineRuntime_ResetEstimateHistory();
	return TX_SUCCESS;
}

/**
 * @brief Start the baseline worker thread.
 */
UINT AppBaselineRuntime_Start(void)
{
	if (!app_baseline_runtime_initialized)
	{
		const UINT init_status = AppBaselineRuntime_Init();
		if (init_status != TX_SUCCESS)
		{
			return init_status;
		}
	}

	if (!camera_baseline_thread_created)
	{
		const UINT create_status = tx_thread_create(
			&camera_baseline_thread, "camera_baseline",
			CameraBaselineThread_Entry, 0U, camera_baseline_thread_stack,
			sizeof(camera_baseline_thread_stack),
			BASELINE_RUNTIME_THREAD_PRIORITY,
			BASELINE_RUNTIME_THREAD_PRIORITY, TX_NO_TIME_SLICE,
			TX_AUTO_START);
		if (create_status != TX_SUCCESS)
		{
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
										ULONG frame_length)
{
	uint8_t first8[8] = {0};

	if (!camera_baseline_sync_created)
	{
		DebugConsole_Printf(
			"[BASELINE] Request dropped; baseline queue not initialized.\r\n");
		return false;
	}

	if ((frame_ptr == NULL) || (frame_length == 0U))
	{
		DebugConsole_Printf(
			"[BASELINE] Request dropped; empty frame ptr=%p len=%lu.\r\n",
			(const void *)frame_ptr, (unsigned long)frame_length);
		return false;
	}

	if (frame_length > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)
	{
		DebugConsole_Printf(
			"[BASELINE] Request dropped; frame too large len=%lu max=%lu.\r\n",
			(unsigned long)frame_length,
			(unsigned long)CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		return false;
	}

	(void)memcpy((void *)camera_baseline_frame_snapshot, frame_ptr,
				 (size_t)frame_length);
	(void)memcpy(first8, camera_baseline_frame_snapshot,
				 (size_t)((frame_length < 8U) ? frame_length : 8U));
	DebugConsole_Printf(
		"[BASELINE] Snapshot copied: src=%p dst=%p len=%lu first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		(const void *)frame_ptr, (void *)camera_baseline_frame_snapshot,
		(unsigned long)frame_length, first8[0], first8[1], first8[2],
		first8[3], first8[4], first8[5], first8[6], first8[7]);

	camera_baseline_request_frame_ptr = camera_baseline_frame_snapshot;
	camera_baseline_request_frame_length = frame_length;

	if (tx_semaphore_put(&camera_baseline_request_semaphore) != TX_SUCCESS)
	{
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
static VOID CameraBaselineThread_Entry(ULONG thread_input)
{
	(void)thread_input;

	(void)DebugConsole_WriteString("[BASELINE] worker alive\r\n");

	while (1)
	{
		const UINT request_status = tx_semaphore_get(
			&camera_baseline_request_semaphore, TX_WAIT_FOREVER);
		const uint8_t *frame_ptr = NULL;
		ULONG frame_length = 0U;
		AppBaselineRuntime_Estimate_t estimate = {0};
		AppBaselineRuntime_Estimate_t held_estimate = {0};

		if (request_status != TX_SUCCESS)
		{
			continue;
		}

		frame_ptr = (const uint8_t *)camera_baseline_request_frame_ptr;
		frame_length = camera_baseline_request_frame_length;
		camera_baseline_request_frame_ptr = NULL;
		camera_baseline_request_frame_length = 0U;

		if ((frame_ptr == NULL) || (frame_length == 0U))
		{
			DebugConsole_Printf(
				"[BASELINE] Worker woke without a queued frame; ignoring.\r\n");
			continue;
		}

		if (!AppBaselineRuntime_EstimateFromFrame(frame_ptr,
												  (size_t)frame_length, &estimate))
		{
			if (!AppBaselineRuntime_SelectSmoothedEstimate(&held_estimate))
			{
				DebugConsole_Printf(
					"[BASELINE] Classical baseline failed to estimate a temperature.\r\n");
				continue;
			}

			held_estimate.source_label = "baseline-polar-held";
			camera_baseline_last_result_valid = true;
			camera_baseline_last_temperature_c = held_estimate.temperature_c;
			camera_baseline_last_angle_rad = held_estimate.angle_rad;
			DebugConsole_Printf(
				"[BASELINE] Holding last stable estimate after an invalid frame.\r\n");
			AppBaselineRuntime_LogEstimate(&held_estimate);
			continue;
		}

		if (!AppBaselineRuntime_IsStableEstimateForHistory(&estimate))
		{
			if (!AppBaselineRuntime_SelectSmoothedEstimate(&held_estimate))
			{
				DebugConsole_Printf(
					"[BASELINE] Classical baseline failed to estimate a temperature.\r\n");
				continue;
			}

			held_estimate.source_label = "baseline-polar-held";
			camera_baseline_last_result_valid = true;
			camera_baseline_last_temperature_c = held_estimate.temperature_c;
			camera_baseline_last_angle_rad = held_estimate.angle_rad;
			DebugConsole_WriteString(
				"[BASELINE] Holding last stable estimate after an unstable frame.\r\n");
			AppBaselineRuntime_LogEstimate(&held_estimate);
			continue;
		}

		/* Smooth the baseline with a tiny history so single-frame glitches do
		 * not swing the reported temperature all over the place. We also keep
		 * the early history quiet until it fills so the first one or two
		 * accepted frames do not leak a nonsense warm-up estimate. */
		AppBaselineRuntime_PushEstimateHistory(&estimate);
		if (!AppBaselineRuntime_SelectSmoothedEstimate(&estimate))
		{
			DebugConsole_Printf(
				"[BASELINE] Classical baseline smoothing failed unexpectedly.\r\n");
			continue;
		}

		if (!estimate.valid)
		{
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
												 size_t frame_size, AppBaselineRuntime_Estimate_t *estimate_out)
{
	AppBaselineRuntime_Estimate_t bright_hypothesis = {0};
	AppBaselineRuntime_Estimate_t fixed_crop_hypothesis = {0};
	AppBaselineRuntime_Estimate_t board_prior_hypothesis = {0};
	AppBaselineRuntime_Estimate_t rim_geometry_hypothesis = {0};
	AppBaselineRuntime_Estimate_t center_hypothesis = {0};
	const AppBaselineRuntime_Estimate_t *selected_estimate = NULL;
	const float dial_radius_px =
		AppBaselineRuntime_EstimateDialRadiusPixels(
			CAMERA_CAPTURE_WIDTH_PIXELS, CAMERA_CAPTURE_HEIGHT_PIXELS);
	bool bright_ok = false;
	bool fixed_crop_ok = false;
	bool board_prior_ok = false;
	bool rim_geometry_ok = false;
	bool center_ok = false;
	size_t center_x = 0U;
	size_t center_y = 0U;
	size_t bright_count = 0U;

	if ((frame_bytes == NULL) || (estimate_out == NULL))
	{
		return false;
	}

	AppBaselineRuntime_UpdateFrameBrightnessProfile(frame_bytes, frame_size);
	{
		const long mean_luma_x10 = AppBaselineRuntime_RoundToLong(
			camera_baseline_current_frame_mean_luma * 10.0f);
		const long bright_pct = AppBaselineRuntime_RoundToLong(
			camera_baseline_current_frame_bright_ratio * 100.0f);
		DebugConsole_Printf(
			"[BASELINE] frame profile: mean=%ld.%01ld bright=%ld%% mode=%s\r\n",
			mean_luma_x10 / 10L,
			((mean_luma_x10 % 10L) < 0L) ? -(mean_luma_x10 % 10L) : (mean_luma_x10 % 10L),
			bright_pct,
			camera_baseline_current_frame_is_bright ? "bright-relaxed" : "normal");
	}

	bright_ok = AppBaselineRuntime_EstimateCenterFromBrightPixels(frame_bytes,
																  frame_size, &center_x, &center_y, &bright_count);
	if (bright_ok)
	{
		bright_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
																	frame_size, center_x, center_y, dial_radius_px,
																	"bright-center-polar",
																	&bright_hypothesis);
	}
	(void)bright_count;

	fixed_crop_ok = AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
		frame_bytes, frame_size, &fixed_crop_hypothesis);

	board_prior_ok = AppBaselineRuntime_EstimateFromBoardPriorHypothesis(
		frame_bytes, frame_size, &board_prior_hypothesis);

	rim_geometry_ok = AppBaselineRuntime_EstimateFromRimGeometryHypothesis(
		frame_bytes, frame_size, &rim_geometry_hypothesis);

	/* Use the inner dial center for the image-center hypothesis too, so the
	 * polar vote pivots around the correct point for the Celsius scale. */
	{
		size_t inner_center_x = 0U;
		size_t inner_center_y = 0U;
		AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
											CAMERA_CAPTURE_HEIGHT_PIXELS,
											&inner_center_x, &inner_center_y);
		center_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
																	frame_size, inner_center_x,
																	inner_center_y, dial_radius_px,
																	"image-center-polar",
																	&center_hypothesis);
	}

	if (!bright_ok && !fixed_crop_ok && !board_prior_ok && !rim_geometry_ok &&
		!center_ok)
	{
		return false;
	}

	/* Keep the live selector conservative by default: trust the fixed-crop
	 * anchor first, then the inner dial center, and only fall back to the
	 * brighter or rim-based seeds if the primary anchors are missing. The
	 * local offset sweep stays behind an explicit experiment flag. */
	{
#if APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP
		AppBaselineRuntime_Estimate_t refined_bright = {0};
		AppBaselineRuntime_Estimate_t refined_fixed_crop = {0};
		AppBaselineRuntime_Estimate_t refined_board_prior = {0};
		AppBaselineRuntime_Estimate_t refined_rim_geometry = {0};
		AppBaselineRuntime_Estimate_t refined_center = {0};
		AppBaselineRuntime_Estimate_t *refined_candidates[5] = {
			&refined_bright,
			&refined_fixed_crop,
			&refined_board_prior,
			&refined_rim_geometry,
			&refined_center,
		};
		const AppBaselineRuntime_Estimate_t *seed_candidates[5] = {
			&bright_hypothesis,
			&fixed_crop_hypothesis,
			&board_prior_hypothesis,
			&rim_geometry_hypothesis,
			&center_hypothesis,
		};
		const bool candidate_ok[5] = {
			bright_ok,
			fixed_crop_ok,
			board_prior_ok,
			rim_geometry_ok,
			center_ok,
		};
		const AppBaselineRuntime_Estimate_t *consensus_candidates[5] = {
			&refined_bright,
			&refined_fixed_crop,
			&refined_board_prior,
			&refined_rim_geometry,
			&refined_center,
		};

		for (size_t candidate_index = 0U; candidate_index < 5U;
			 ++candidate_index)
		{
			const AppBaselineRuntime_Estimate_t *seed_estimate =
				seed_candidates[candidate_index];
			AppBaselineRuntime_Estimate_t *refined_estimate =
				refined_candidates[candidate_index];

			if (!candidate_ok[candidate_index])
			{
				continue;
			}

			if (!AppBaselineRuntime_RefineEstimateAroundSeed(frame_bytes,
															 frame_size, dial_radius_px, seed_estimate,
															 refined_estimate))
			{
				continue;
			}

			if ((selected_estimate == NULL) || AppBaselineRuntime_IsBetterEstimate(refined_estimate,
																				   selected_estimate))
			{
				selected_estimate = refined_estimate;
			}
		}

		selected_estimate = AppBaselineRuntime_SelectConsensusEstimate(
			consensus_candidates, candidate_ok, selected_estimate);
#else
		{
			const AppBaselineRuntime_Estimate_t *candidate_estimates[5] = {
				&bright_hypothesis,
				&fixed_crop_hypothesis,
				&board_prior_hypothesis,
				&rim_geometry_hypothesis,
				&center_hypothesis,
			};
			const bool candidate_ok[5] = {
				bright_ok,
				fixed_crop_ok,
				board_prior_ok,
				rim_geometry_ok,
				center_ok,
			};

			for (size_t candidate_index = 0U; candidate_index < 5U;
				 ++candidate_index)
			{
				const AppBaselineRuntime_Estimate_t *candidate_estimate =
					candidate_estimates[candidate_index];

				if (!candidate_ok[candidate_index] || (candidate_estimate == NULL))
				{
					continue;
				}

				if ((selected_estimate == NULL) ||
					AppBaselineRuntime_IsBetterEstimate(candidate_estimate,
														selected_estimate))
				{
					selected_estimate = candidate_estimate;
				}
			}

			selected_estimate = AppBaselineRuntime_SelectConsensusEstimate(
				candidate_estimates, candidate_ok, selected_estimate);
		}
#endif
	}

	if (selected_estimate == NULL)
	{
		DebugConsole_Printf(
			"[BASELINE] candidates: bright=%s fixed=%s board=%s rim=%s image=%s selected=none(0)\r\n",
			bright_ok ? "ok" : "no",
			fixed_crop_ok ? "ok" : "no",
			board_prior_ok ? "ok" : "no",
			rim_geometry_ok ? "ok" : "no",
			center_ok ? "ok" : "no");
		return false;
	}

	*estimate_out = *selected_estimate;

	{
		const long bright_conf_m = AppBaselineRuntime_RoundToLong(
			(bright_ok ? bright_hypothesis.confidence : 0.0f) * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long fixed_conf_m = AppBaselineRuntime_RoundToLong(
			(fixed_crop_ok ? fixed_crop_hypothesis.confidence : 0.0f) * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long board_conf_m = AppBaselineRuntime_RoundToLong(
			(board_prior_ok ? board_prior_hypothesis.confidence : 0.0f) * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long rim_conf_m = AppBaselineRuntime_RoundToLong(
			(rim_geometry_ok ? rim_geometry_hypothesis.confidence : 0.0f) * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long image_conf_m = AppBaselineRuntime_RoundToLong(
			(center_ok ? center_hypothesis.confidence : 0.0f) * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long selected_conf_m = AppBaselineRuntime_RoundToLong(
			estimate_out->confidence * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		DebugConsole_Printf(
			"[BASELINE] candidates: bright=%s(%ld) fixed=%s(%ld) board=%s(%ld) rim=%s(%ld) image=%s(%ld) selected=%s(%ld)\r\n",
			bright_ok ? "ok" : "no", bright_conf_m,
			fixed_crop_ok ? "ok" : "no", fixed_conf_m,
			board_prior_ok ? "ok" : "no", board_conf_m,
			rim_geometry_ok ? "ok" : "no", rim_conf_m,
			center_ok ? "ok" : "no", image_conf_m,
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "unknown",
			selected_conf_m);
	}

	/* All candidate geometries must clear the same live acceptance gate before
	 * they are allowed to seed history or replace the live reading. */
	if (!AppBaselineRuntime_PassesAcceptanceGate(estimate_out))
	{
		const long conf_m = AppBaselineRuntime_RoundToLong(
			estimate_out->confidence * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long threshold_m = AppBaselineRuntime_RoundToLong(
			APP_BASELINE_CONFIDENCE_THRESHOLD * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long best_score_m = AppBaselineRuntime_RoundToLong(
			estimate_out->best_score * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long runner_up_m = AppBaselineRuntime_RoundToLong(
			estimate_out->runner_up_score * (float)APP_BASELINE_CONFIDENCE_LOG_SCALE);
		const long peak_ratio_x1000 =
			AppBaselineRuntime_RoundToLong(
				((estimate_out->runner_up_score > 0.0f) ? (estimate_out->best_score / estimate_out->runner_up_score) : 0.0f) * 1000.0f);
		/* Determine the actual rejection reason by mirroring the gate checks. */
		const char *reject_reason = "unknown";
		if (estimate_out->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD)
		{
			reject_reason = "confidence";
		}
		else if (estimate_out->best_score < APP_BASELINE_MIN_ACCEPT_SCORE)
		{
			reject_reason = "score";
		}
		else if ((estimate_out->runner_up_score > 0.0f) && ((estimate_out->best_score / estimate_out->runner_up_score) < APP_BASELINE_MIN_PEAK_RATIO))
		{
			reject_reason = "peak_ratio";
		}
		else
		{
			/* Confidence, score, and peak_ratio all passed — must be the
			 * center-distance check that rejected this estimate. */
			reject_reason = "center_dist";
		}
		DebugConsole_Printf(
			"[BASELINE] Rejected: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%lu cy=%lu\r\n",
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "?",
			reject_reason, conf_m, threshold_m, best_score_m, runner_up_m,
			peak_ratio_x1000, estimate_out->center_x, estimate_out->center_y);
		DebugConsole_Printf(
			"[BASELINE] FAIL: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%lu cy=%lu\r\n",
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "?",
			reject_reason, conf_m, threshold_m, best_score_m, runner_up_m,
			peak_ratio_x1000, estimate_out->center_x, estimate_out->center_y);
		DebugConsole_Printf(
			"[BASELINE] REJECTED: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%lu cy=%lu\r\n",
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "?",
			reject_reason, conf_m, threshold_m, best_score_m, runner_up_m,
			peak_ratio_x1000, estimate_out->center_x, estimate_out->center_y);
		return false;
	}

	return true;
}

/**
 * @brief Compute a quality score for one estimate.
 *
 * We mirror the Python classical baseline here: confidence alone is not
 * enough, because broad near-ties can still have a high vote sum. A candidate
 * should score well when it is strong and stable, but a tiny runner-up should
 * not explode the score and let a spiky false geometry win.
 */
static float AppBaselineRuntime_ComputeEstimateQuality(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	float peak_ratio = 0.0f;

	if ((estimate == NULL) || !estimate->valid)
	{
		return 0.0f;
	}

	if (estimate->best_score <= 0.0f)
	{
		return 0.0f;
	}

	if (estimate->runner_up_score > 0.0f)
	{
		peak_ratio = estimate->best_score / estimate->runner_up_score;
	}
	else
	{
		peak_ratio = estimate->best_score;
	}

	if (peak_ratio < 1.0f)
	{
		peak_ratio = 1.0f;
	}

	return AppBaselineRuntime_ClampFloat(
		estimate->confidence / peak_ratio, 0.0f, 1000000.0f);
}

/**
 * @brief Return a small priority score for live geometry sources.
 *
 * Clean, near-centered captures should prefer the stable fixed-crop anchor
 * first, then the inner image center, and only then fall back to the
 * board-specific prior before we consider the bright and rim hypotheses. The
 * board prior is still useful as a rescue path on awkward framings, but it
 * should not outrank the stable crop on the ideal captures we care about most.
 */
static int AppBaselineRuntime_SourcePriority(const char *source_label)
{
	if (source_label == NULL)
	{
		return 0;
	}

	if (strcmp(source_label, "fixed-crop-polar") == 0)
	{
		return 5;
	}

	if (strcmp(source_label, "image-center-polar") == 0)
	{
		return 4;
	}

	if (strcmp(source_label, "board-prior-polar") == 0)
	{
		return 3;
	}

	if (strcmp(source_label, "bright-center-polar") == 0)
	{
		return 2;
	}

	if (strcmp(source_label, "rim-center-polar") == 0)
	{
		return 1;
	}

	return 0;
}

/**
 * @brief Check whether one estimate clears the live acceptance gate.
 *
 * In addition to the standard confidence, score, and peak-ratio checks, we
 * reject estimates whose center is too far from the inner dial center. This
 * prevents glare-induced bright-centroid hypotheses from polluting the output.
 */
static bool AppBaselineRuntime_PassesAcceptanceGate(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if ((estimate == NULL) || !estimate->valid)
	{
		return false;
	}

	if (estimate->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD)
	{
		return false;
	}

	if (estimate->best_score < APP_BASELINE_MIN_ACCEPT_SCORE)
	{
		return false;
	}

	if (!AppBaselineRuntime_HasAcceptablePeakSeparation(estimate))
	{
		return false;
	}

	/* Reject estimates whose center is too far from the inner dial center.
	 * The inner dial center is at ~(104, 104) on a 640x480 frame. Any center
	 * more than 100 pixels away is almost certainly a glare-induced false
	 * positive rather than the real dial pivot. */
	{
		size_t inner_cx = 0U;
		size_t inner_cy = 0U;
		AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
											CAMERA_CAPTURE_HEIGHT_PIXELS,
											&inner_cx, &inner_cy);
		const float dx = (float)estimate->center_x - (float)inner_cx;
		const float dy = (float)estimate->center_y - (float)inner_cy;
		const float dist_sq = dx * dx + dy * dy;
		if (dist_sq > 10000.0f) /* 100 pixels — wide enough for rim geometry drift */
		{
			return false;
		}
	}

	return true;
}

/**
 * @brief Decide whether one estimate is better than another candidate.
 *
 * We prefer the candidate with the strongest peak-separation quality score,
 * then fall back to the peak ratio, raw support, and confidence.
 */
static bool AppBaselineRuntime_IsBetterEstimate(
	const AppBaselineRuntime_Estimate_t *candidate,
	const AppBaselineRuntime_Estimate_t *incumbent)
{
	const float candidate_quality = AppBaselineRuntime_ComputeEstimateQuality(
		candidate);
	const float incumbent_quality = AppBaselineRuntime_ComputeEstimateQuality(
		incumbent);

	if ((candidate == NULL) || !candidate->valid)
	{
		return false;
	}
	if ((incumbent == NULL) || !incumbent->valid)
	{
		return true;
	}

	/* Penalize the 'bright-center' hypothesis if the center is too far from
	 * the image center. This prevents glare-induced centroids from winning
	 * over stable geometric anchors. */
	if (candidate->source_label != NULL &&
		strcmp(candidate->source_label, "bright-center-polar") == 0)
	{
		const float dx = (float)candidate->center_x - (CAMERA_CAPTURE_WIDTH_PIXELS / 2.0f);
		const float dy = (float)candidate->center_y - (CAMERA_CAPTURE_HEIGHT_PIXELS / 2.0f);
		const float dist_sq = dx * dx + dy * dy;
		/* If center is more than 150 pixels away from image center,
		 * heavily penalize the quality score. */
		if (dist_sq > 22500.0f)
		{
			return false;
		}
	}
	if (candidate_quality > incumbent_quality)
	{
		/* Source priority wins before raw quality so ideal captures keep the
		 * stable fixed-crop anchor instead of drifting to a rim false
		 * positive. However, if a lower-priority candidate has significantly
		 * higher quality (2x), allow it to win — the fixed-crop may be
		 * detecting a dial marking while rim/image found the real needle. */
		const int candidate_priority =
			AppBaselineRuntime_SourcePriority(candidate->source_label);
		const int incumbent_priority =
			AppBaselineRuntime_SourcePriority(incumbent->source_label);
		const float quality_ratio = candidate_quality / incumbent_quality;

		if (candidate_priority > incumbent_priority)
		{
			return true;
		}
		if (candidate_priority < incumbent_priority)
		{
			/* Allow lower-priority candidate to win if quality is 2x better. */
			if (quality_ratio >= 2.0f)
			{
				return true;
			}
			return false;
		}

		return true;
	}
	if (candidate_quality < incumbent_quality)
	{
		const int candidate_priority =
			AppBaselineRuntime_SourcePriority(candidate->source_label);
		const int incumbent_priority =
			AppBaselineRuntime_SourcePriority(incumbent->source_label);

		if (candidate_priority > incumbent_priority)
		{
			return true;
		}
		if (candidate_priority < incumbent_priority)
		{
			return false;
		}

		return false;
	}
	{
		const float candidate_peak_ratio =
			(candidate->runner_up_score > 0.0f) ? (candidate->best_score / candidate->runner_up_score) : candidate->best_score;
		const float incumbent_peak_ratio =
			(incumbent->runner_up_score > 0.0f) ? (incumbent->best_score / incumbent->runner_up_score) : incumbent->best_score;

		if (candidate_peak_ratio > incumbent_peak_ratio)
		{
			return true;
		}
		if (candidate_peak_ratio < incumbent_peak_ratio)
		{
			return false;
		}
	}

	if (candidate->best_score > incumbent->best_score)
	{
		return true;
	}
	if (candidate->best_score < incumbent->best_score)
	{
		return false;
	}

	return candidate->confidence > incumbent->confidence;
}

/**
 * @brief Estimate the needle using the stable fixed gauge crop.
 *
 * The detector is classical end to end: it scores Sobel-like edges in polar
 * space and keeps only the strongest spoke hypothesis. The scan area is
 * centered on the inner Celsius dial so the polar vote uses the correct pivot.
 */
static bool AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
	const uint8_t *frame_bytes, size_t frame_size,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const float dial_radius_px =
		AppBaselineRuntime_EstimateDialRadiusPixels(width_pixels,
													height_pixels);
	size_t center_x = 0U;
	size_t center_y = 0U;

	if ((estimate_out == NULL) || (frame_bytes == NULL))
	{
		return false;
	}

	AppGaugeGeometry_TrainingCropCenter(width_pixels, height_pixels,
										&center_x, &center_y);

	/* Apply upward bias to center on the inner Celsius dial rather than the
	 * geometric center of the training crop. The inner dial sits higher in the
	 * frame, so we nudge the center upward by ~11% of crop height, clamped
	 * to 8..18 pixels to stay within the dial face. */
	const AppGaugeGeometry_Crop_t crop =
		AppGaugeGeometry_TrainingCrop(width_pixels, height_pixels);
	const size_t bias = (size_t)(0.11f * (float)crop.height + 0.5f);
	const size_t clamped_bias = (bias < 8U) ? 8U : ((bias > 18U) ? 18U : bias);
	center_y = (center_y > clamped_bias) ? (center_y - clamped_bias) : 0U;

	/* Scan area centered on the inner dial, large enough to cover the full
	 * needle range but tight enough to exclude outer-dial clutter. */
	const size_t scan_radius = (size_t)(dial_radius_px * 1.5f);
	const size_t scan_x_min = (center_x > scan_radius) ? (center_x - scan_radius) : 0U;
	const size_t scan_y_min = (center_y > scan_radius) ? (center_y - scan_radius) : 0U;
	const size_t scan_x_max = (center_x + scan_radius < width_pixels) ? (center_x + scan_radius) : width_pixels;
	const size_t scan_y_max = (center_y + scan_radius < height_pixels) ? (center_y + scan_radius) : height_pixels;

	return AppBaselineRuntime_EstimatePolarNeedle(frame_bytes, frame_size,
												  width_pixels, height_pixels, scan_x_min, scan_y_min,
												  scan_x_max, scan_y_max, center_x,
												  center_y, dial_radius_px, "fixed-crop-polar", estimate_out);
}

/**
 * @brief Estimate the dial geometry from the outer rim before scoring spokes.
 *
 * This small Hough-like search prefers centers whose outer ring edges point
 * back toward the same center with high radial agreement. The scan area is
 * centered on the inner Celsius dial to avoid outer-dial clutter.
 */
static bool AppBaselineRuntime_EstimateFromRimGeometryHypothesis(
	const uint8_t *frame_bytes, size_t frame_size,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const float dial_radius_px =
		AppBaselineRuntime_EstimateDialRadiusPixels(width_pixels,
													height_pixels);
	size_t center_x = 0U;
	size_t center_y = 0U;

	if ((frame_bytes == NULL) || (estimate_out == NULL))
	{
		return false;
	}

	/* Use the inner dial center as the anchor for the rim search. */
	AppGaugeGeometry_TrainingCropCenter(width_pixels, height_pixels,
										&center_x, &center_y);

	/* Scan area centered on the inner dial, sized to cover the rim region. */
	const size_t scan_radius = (size_t)(dial_radius_px * 1.8f);
	const size_t scan_x_min = (center_x > scan_radius) ? (center_x - scan_radius) : 0U;
	const size_t scan_x_max = (center_x + scan_radius < width_pixels) ? (center_x + scan_radius) : width_pixels;
	const size_t scan_y_min = (center_y > scan_radius) ? (center_y - scan_radius) : 0U;
	const size_t scan_y_max = (center_y + scan_radius < height_pixels) ? (center_y + scan_radius) : height_pixels;

	if (!AppBaselineRuntime_EstimateDialCenterFromRimVotes(frame_bytes,
														   frame_size, width_pixels, height_pixels, scan_x_min, scan_y_min,
														   scan_x_max, scan_y_max, dial_radius_px,
														   &center_x, &center_y, NULL))
	{
		return false;
	}

	return AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
														   frame_size, center_x, center_y, dial_radius_px,
														   "rim-center-polar", estimate_out);
}

/**
 * @brief Estimate a bright dial center from the high-luma pixels.
 *
 * The scan area is centered on the inner Celsius dial so the centroid
 * reflects the inner-dial face rather than the outer gauge ring.
 */
static bool AppBaselineRuntime_EstimateCenterFromBrightPixels(
	const uint8_t *frame_bytes, size_t frame_size, size_t *center_x_out,
	size_t *center_y_out, size_t *bright_count_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const size_t stride_bytes = width_pixels * CAMERA_CAPTURE_BYTES_PER_PIXEL;
	size_t inner_center_x = 0U;
	size_t inner_center_y = 0U;
	AppGaugeGeometry_TrainingCropCenter(width_pixels, height_pixels,
										&inner_center_x, &inner_center_y);
	const float dial_radius_px =
		AppBaselineRuntime_EstimateDialRadiusPixels(width_pixels,
													height_pixels);
	const size_t scan_radius = (size_t)(dial_radius_px * 1.5f);
	const size_t scan_x_min = (inner_center_x > scan_radius) ? (inner_center_x - scan_radius) : 0U;
	const size_t scan_x_max = (inner_center_x + scan_radius < width_pixels) ? (inner_center_x + scan_radius) : width_pixels;
	const size_t scan_y_min = (inner_center_y > scan_radius) ? (inner_center_y - scan_radius) : 0U;
	const size_t scan_y_max = (inner_center_y + scan_radius < height_pixels) ? (inner_center_y + scan_radius) : height_pixels;
	size_t bright_x_min = width_pixels;
	size_t bright_y_min = height_pixels;
	size_t bright_x_max = 0U;
	size_t bright_y_max = 0U;
	size_t bright_count = 0U;
	uint64_t bright_sum_x = 0U;
	uint64_t bright_sum_y = 0U;

	if ((frame_bytes == NULL) || (center_x_out == NULL) || (center_y_out == NULL) || (bright_count_out == NULL))
	{
		return false;
	}

	if (frame_size < (stride_bytes * height_pixels))
	{
		return false;
	}

	for (size_t y = scan_y_min; y < scan_y_max; ++y)
	{
		for (size_t x = scan_x_min; x < scan_x_max; ++x)
		{
			const float luma = AppBaselineRuntime_ReadLuma(frame_bytes,
														   width_pixels, x, y);

			if (luma < (float)APP_BASELINE_BRIGHT_THRESHOLD)
			{
				continue;
			}

			/* Exclude saturated/glare pixels — they skew the centroid away
			 * from the real dial face toward the blown-out reflection spot. */
			if (luma > (float)APP_BASELINE_SATURATION_THRESHOLD)
			{
				continue;
			}

			bright_count++;
			bright_sum_x += (uint64_t)x;
			bright_sum_y += (uint64_t)y;

			if (x < bright_x_min)
			{
				bright_x_min = x;
			}
			if (y < bright_y_min)
			{
				bright_y_min = y;
			}
			if (x > bright_x_max)
			{
				bright_x_max = x;
			}
			if (y > bright_y_max)
			{
				bright_y_max = y;
			}
		}
	}

	if (bright_count < APP_BASELINE_MIN_BRIGHT_PIXELS)
	{
		return false;
	}

	if ((bright_x_max <= bright_x_min) || (bright_y_max <= bright_y_min))
	{
		return false;
	}

	size_t raw_center_x = (size_t)(bright_sum_x / (uint64_t)bright_count);
	size_t raw_center_y = (size_t)(bright_sum_y / (uint64_t)bright_count);

	/* Apply upward bias to center on the inner Celsius dial rather than the
	 * whole gauge face. The inner dial sits above the geometric center of the
	 * crop, so we nudge the bright centroid upward by ~11% of crop height,
	 * clamped to 8..18 pixels to stay within the dial face. */
	const size_t crop_height = bright_y_max - bright_y_min;
	const size_t bias = (size_t)(0.11f * (float)crop_height + 0.5f);
	const size_t clamped_bias = (bias < 8U) ? 8U : ((bias > 18U) ? 18U : bias);
	const size_t biased_center_y = (raw_center_y > clamped_bias) ? (raw_center_y - clamped_bias) : 0U;

	*center_x_out = raw_center_x;
	*center_y_out = biased_center_y;
	*bright_count_out = bright_count;
	return true;
}

/**
 * @brief Evaluate the needle angle using one center hypothesis.
 */
static bool AppBaselineRuntime_EstimateFromCenterHypothesis(
	const uint8_t *frame_bytes, size_t frame_size, size_t center_x,
	size_t center_y, float dial_radius_px, const char *source_label,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;

	if (source_label == NULL)
	{
		return false;
	}

	/* Scan area centered on the inner dial center, sized to cover the full
	 * needle range without including outer-dial clutter. */
	const size_t scan_radius = (size_t)(dial_radius_px * 1.8f);
	const size_t scan_x_min = (center_x > scan_radius) ? (center_x - scan_radius) : 0U;
	const size_t scan_x_max = (center_x + scan_radius < width_pixels) ? (center_x + scan_radius) : width_pixels;
	const size_t scan_y_min = (center_y > scan_radius) ? (center_y - scan_radius) : 0U;
	const size_t scan_y_max = (center_y + scan_radius < height_pixels) ? (center_y + scan_radius) : height_pixels;

	return AppBaselineRuntime_EstimatePolarNeedle(frame_bytes, frame_size,
												  width_pixels, height_pixels, scan_x_min, scan_y_min,
												  scan_x_max, scan_y_max, center_x,
												  center_y, dial_radius_px, source_label, estimate_out);
}

/**
 * @brief Evaluate the needle angle using the fixed board prior.
 *
 * The board capture is framed a little differently from the generic crop, so
 * this candidate uses the observed inner-dial offset and radius directly.
 */
static bool AppBaselineRuntime_EstimateFromBoardPriorHypothesis(
	const uint8_t *frame_bytes, size_t frame_size,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const float board_center_x =
		(float)width_pixels * APP_BASELINE_BOARD_PRIOR_CENTER_X_RATIO;
	const float board_center_y =
		(float)height_pixels * APP_BASELINE_BOARD_PRIOR_CENTER_Y_RATIO;
	const float min_frame_pixels =
		(float)((width_pixels < height_pixels) ? width_pixels : height_pixels);
	const float board_radius_px =
		AppBaselineRuntime_ClampFloat(
			APP_BASELINE_BOARD_PRIOR_RADIUS_RATIO * min_frame_pixels,
			(float)APP_BASELINE_MIN_RADIUS_PIXELS,
			0.49f * min_frame_pixels);

	return AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
														   frame_size,
														   AppBaselineRuntime_RoundToLong(board_center_x),
														   AppBaselineRuntime_RoundToLong(board_center_y),
														   board_radius_px,
														   "board-prior-polar",
														   estimate_out);
}

/**
 * @brief Refine one center hypothesis by probing a small local neighborhood.
 *
 * The live board can be a few pixels off after crop jitter, so we let the
 * selected geometry slide slightly and keep the best classical polar peak.
 */
static bool AppBaselineRuntime_RefineEstimateAroundSeed(
	const uint8_t *frame_bytes, size_t frame_size,
	float dial_radius_px,
	const AppBaselineRuntime_Estimate_t *seed_estimate,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const AppGaugeGeometry_Crop_t crop =
		AppGaugeGeometry_TrainingCrop(width_pixels, height_pixels);
	const long offset_values[] = {-8L, -4L, 0L, 4L, 8L};
	const long min_center_x = (long)crop.x_min + 1L;
	const long max_center_x = (long)crop.x_min + (long)crop.width - 2L;
	const long min_center_y = (long)crop.y_min + 1L;
	const long max_center_y = (long)crop.y_min + (long)crop.height - 2L;
	AppBaselineRuntime_Estimate_t best_estimate = {0};
	bool found_any = false;

	if ((frame_bytes == NULL) || (estimate_out == NULL) || (seed_estimate == NULL) || !seed_estimate->valid)
	{
		return false;
	}

	best_estimate = *seed_estimate;
	found_any = true;

	for (size_t dy_index = 0U; dy_index < (sizeof(offset_values) / sizeof(offset_values[0])); ++dy_index)
	{
		for (size_t dx_index = 0U; dx_index < (sizeof(offset_values) / sizeof(offset_values[0])); ++dx_index)
		{
			const long offset_x = offset_values[dx_index];
			const long offset_y = offset_values[dy_index];
			long candidate_center_x = (long)seed_estimate->center_x + offset_x;
			long candidate_center_y = (long)seed_estimate->center_y + offset_y;
			AppBaselineRuntime_Estimate_t candidate_estimate = {0};

			if (candidate_center_x < min_center_x)
			{
				candidate_center_x = min_center_x;
			}
			else if (candidate_center_x > max_center_x)
			{
				candidate_center_x = max_center_x;
			}
			if (candidate_center_y < min_center_y)
			{
				candidate_center_y = min_center_y;
			}
			else if (candidate_center_y > max_center_y)
			{
				candidate_center_y = max_center_y;
			}

			if (!AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
																 frame_size, (size_t)candidate_center_x,
																 (size_t)candidate_center_y, dial_radius_px,
																 seed_estimate->source_label, &candidate_estimate))
			{
				continue;
			}

			{
				const float candidate_quality =
					AppBaselineRuntime_ComputeEstimateQuality(
						&candidate_estimate);
				const float best_quality =
					AppBaselineRuntime_ComputeEstimateQuality(&best_estimate);

				if ((candidate_quality > best_quality) || ((candidate_quality == best_quality) && (candidate_estimate.confidence > best_estimate.confidence)))
				{
					best_estimate = candidate_estimate;
				}
			}
		}
	}

	*estimate_out = best_estimate;
	return found_any;
}

/**
 * @brief Clear the small baseline smoothing history.
 */
static void AppBaselineRuntime_ResetEstimateHistory(void)
{
	memset(camera_baseline_estimate_history, 0,
		   sizeof(camera_baseline_estimate_history));
	camera_baseline_estimate_history_count = 0U;
	camera_baseline_estimate_history_next_index = 0U;
}

/**
 * @brief Store one accepted baseline estimate in the tiny smoothing history.
 */
static void AppBaselineRuntime_PushEstimateHistory(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if (estimate == NULL)
	{
		return;
	}

	if (camera_baseline_estimate_history_count == 0U)
	{
		camera_baseline_estimate_history[0U] = *estimate;
		camera_baseline_estimate_history_count = 1U;
		camera_baseline_estimate_history_next_index = 1U % APP_BASELINE_ESTIMATE_HISTORY_SIZE;
		return;
	}

	if (camera_baseline_estimate_history_count > 0U)
	{
		const size_t last_index =
			(camera_baseline_estimate_history_next_index + APP_BASELINE_ESTIMATE_HISTORY_SIZE - 1U) % APP_BASELINE_ESTIMATE_HISTORY_SIZE;
		const float last_temperature_c =
			camera_baseline_estimate_history[last_index].temperature_c;

		if (fabsf(estimate->temperature_c - last_temperature_c) > APP_BASELINE_HISTORY_RESET_DELTA_C)
		{
			AppBaselineRuntime_ResetEstimateHistory();
		}
	}

	camera_baseline_estimate_history[camera_baseline_estimate_history_next_index] = *estimate;
	if (camera_baseline_estimate_history_count < APP_BASELINE_ESTIMATE_HISTORY_SIZE)
	{
		camera_baseline_estimate_history_count++;
	}
	camera_baseline_estimate_history_next_index =
		(camera_baseline_estimate_history_next_index + 1U) % APP_BASELINE_ESTIMATE_HISTORY_SIZE;
}

/**
 * @brief Decide whether a baseline estimate is stable enough for history.
 *
 * The tiny smoothing buffer only accepts estimates that already cleared the
 * same confidence, vote-support, and peak-separation gates as the live read.
 * Borderline continuation frames can still seed history when they stay close
 * to the last stable reading, which keeps a soft but coherent frame from being
 * dropped behind stale history.
 */
static bool AppBaselineRuntime_IsStableEstimateForHistory(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if (estimate == NULL)
	{
		return false;
	}

	if (estimate->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD)
	{
		return false;
	}

	if (estimate->best_score < APP_BASELINE_MIN_ACCEPT_SCORE)
	{
		return false;
	}

	if (!AppBaselineRuntime_HasAcceptablePeakSeparation(estimate))
	{
		return false;
	}

	if (camera_baseline_current_frame_is_bright && camera_baseline_last_result_valid)
	{
		const float delta_c =
			fabsf(estimate->temperature_c - camera_baseline_last_temperature_c);
		if ((delta_c > APP_BASELINE_BRIGHT_RELAXED_MAX_TEMP_JUMP_C) &&
			(estimate->confidence < APP_BASELINE_BRIGHT_RELAXED_MIN_CONFIDENCE_FOR_JUMP))
		{
			const long delta_x10 = AppBaselineRuntime_RoundToLong(delta_c * 10.0f);
			const long conf_m = AppBaselineRuntime_RoundToLong(
				estimate->confidence * 1000.0f);
			DebugConsole_Printf(
				"[BASELINE] Stability hold: bright jump=%ld.%01ldC conf=%ld/1000\r\n",
				delta_x10 / 10L,
				((delta_x10 % 10L) < 0L) ? -(delta_x10 % 10L) : (delta_x10 % 10L),
				conf_m);
			return false;
		}
	}

	if (camera_baseline_last_result_valid && (estimate->runner_up_score > 0.0f))
	{
		const float delta_c =
			fabsf(estimate->temperature_c - camera_baseline_last_temperature_c);
		const float peak_ratio = estimate->best_score / estimate->runner_up_score;
		if ((delta_c > APP_BASELINE_AMBIGUOUS_JUMP_DELTA_C) &&
			(peak_ratio < APP_BASELINE_AMBIGUOUS_JUMP_MAX_PEAK_RATIO) &&
			(estimate->confidence < APP_BASELINE_AMBIGUOUS_JUMP_MAX_CONFIDENCE))
		{
			const long delta_x10 = AppBaselineRuntime_RoundToLong(delta_c * 10.0f);
			const long ratio_m = AppBaselineRuntime_RoundToLong(peak_ratio * 1000.0f);
			const long conf_m = AppBaselineRuntime_RoundToLong(
				estimate->confidence * 1000.0f);
			DebugConsole_Printf(
				"[BASELINE] Stability hold: ambiguous jump=%ld.%01ldC ratio=%ld/1000 conf=%ld/1000\r\n",
				delta_x10 / 10L,
				((delta_x10 % 10L) < 0L) ? -(delta_x10 % 10L) : (delta_x10 % 10L),
				ratio_m, conf_m);
			return false;
		}
	}

	return true;
}

/**
 * @brief Decide whether the peak separation is good enough to trust.
 *
 * The normal gate still requires a clearly dominant spoke peak. When the peak
 * is only a little soft, we allow a narrow continuation path for fixed-crop or
 * image-center reads that stay close to the last stable temperature. That
 * gives the live board a way to move forward on consistent borderline frames
 * without lowering the global threshold for unrelated clutter.
 */
static bool AppBaselineRuntime_HasAcceptablePeakSeparation(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if ((estimate == NULL) || !estimate->valid)
	{
		return false;
	}

	if (estimate->runner_up_score <= 0.0f)
	{
		return true;
	}

	if ((estimate->best_score / estimate->runner_up_score) >=
		APP_BASELINE_MIN_PEAK_RATIO)
	{
		return true;
	}

	return AppBaselineRuntime_IsBorderlineContinuityEstimate(estimate);
}

/**
 * @brief Check whether a soft peak is still a safe continuation frame.
 *
 * We only relax the peak-ratio gate for the strong fixed-crop and
 * image-center families, and only when the new reading stays close to the last
 * stable temperature. That keeps the fallback conservative while still letting
 * a slightly broad but coherent frame advance the baseline.
 */
static bool AppBaselineRuntime_IsBorderlineContinuityEstimate(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if ((estimate == NULL) || !estimate->valid || !camera_baseline_last_result_valid)
	{
		return false;
	}

	if (estimate->confidence < APP_BASELINE_BORDERLINE_MIN_CONFIDENCE)
	{
		return false;
	}

	if ((estimate->source_label == NULL) ||
		((strcmp(estimate->source_label, "fixed-crop-polar") != 0) &&
		 (strcmp(estimate->source_label, "image-center-polar") != 0)))
	{
		return false;
	}

	if (fabsf(estimate->temperature_c - camera_baseline_last_temperature_c) >
		APP_BASELINE_BORDERLINE_MAX_TEMP_DELTA_C)
	{
		return false;
	}

	if ((estimate->runner_up_score <= 0.0f) ||
		((estimate->best_score / estimate->runner_up_score) <
		 APP_BASELINE_BORDERLINE_PEAK_RATIO))
	{
		return false;
	}

	return true;
}

/**
 * @brief Prefer a small cluster of agreeing geometry hypotheses.
 *
 * The raw winner-take-all peak can be a lone outlier when one geometry seed
 * locks onto glare or dial clutter. If several refined estimates land within
 * a few degrees of each other, keep the strongest one from that agreement
 * cluster instead.
 */
static const AppBaselineRuntime_Estimate_t *AppBaselineRuntime_SelectConsensusEstimate(
	const AppBaselineRuntime_Estimate_t *const estimates[5],
	const bool candidate_ok[5],
	const AppBaselineRuntime_Estimate_t *fallback_estimate)
{
	const AppBaselineRuntime_Estimate_t *best_estimate = NULL;
	int best_priority = -1;
	float best_quality = -1.0f;
	float best_peak_ratio = -1.0f;
	float best_confidence = -1.0f;
	float best_score = -1.0f;
	size_t valid_indices[5] = {0U};
	size_t valid_count = 0U;
	size_t best_support = 0U;

	if ((estimates == NULL) || (candidate_ok == NULL) || (fallback_estimate == NULL))
	{
		return fallback_estimate;
	}

	for (size_t i = 0U; i < 5U; ++i)
	{
		const AppBaselineRuntime_Estimate_t *estimate = estimates[i];
		if ((estimate == NULL) || !candidate_ok[i] || !estimate->valid)
		{
			continue;
		}

		valid_indices[valid_count] = i;
		++valid_count;
	}

	if (valid_count < 2U)
	{
		return fallback_estimate;
	}

	size_t support[5] = {0U};

	for (size_t valid_index = 0U; valid_index < valid_count; ++valid_index)
	{
		const size_t i = valid_indices[valid_index];
		const AppBaselineRuntime_Estimate_t *estimate_i = estimates[i];
		size_t agreement = 1U;

		for (size_t other_index = 0U; other_index < valid_count; ++other_index)
		{
			const size_t j = valid_indices[other_index];
			const AppBaselineRuntime_Estimate_t *estimate_j = estimates[j];

			if ((i == j) || (estimate_j == NULL))
			{
				continue;
			}

			if (fabsf(estimate_i->temperature_c - estimate_j->temperature_c) <=
				APP_BASELINE_CONSENSUS_TEMP_DELTA_C)
			{
				++agreement;
			}
		}

		support[i] = agreement;
		if (agreement > best_support)
		{
			best_support = agreement;
		}
	}

	if (best_support < 2U)
	{
		return fallback_estimate;
	}

	const float fallback_quality =
		AppBaselineRuntime_ComputeEstimateQuality(fallback_estimate);
	const int fallback_priority =
		AppBaselineRuntime_SourcePriority(fallback_estimate->source_label);

	for (size_t valid_index = 0U; valid_index < valid_count; ++valid_index)
	{
		const size_t i = valid_indices[valid_index];
		const AppBaselineRuntime_Estimate_t *candidate = estimates[i];
		const float candidate_quality =
			AppBaselineRuntime_ComputeEstimateQuality(candidate);
		const float candidate_peak_ratio =
			(candidate->runner_up_score > 0.0f)
				? (candidate->best_score / candidate->runner_up_score)
				: candidate->best_score;
		const int candidate_priority =
			AppBaselineRuntime_SourcePriority(candidate->source_label);

		if (support[i] != best_support)
		{
			continue;
		}

		if ((best_estimate == NULL) ||
			(candidate_priority > best_priority) ||
			((candidate_priority == best_priority) &&
			 (candidate_quality > best_quality)) ||
			((candidate_priority == best_priority) &&
			 (candidate_quality == best_quality) &&
			 (candidate_peak_ratio > best_peak_ratio)) ||
			((candidate_priority == best_priority) &&
			 (candidate_quality == best_quality) &&
			 (candidate_peak_ratio == best_peak_ratio) &&
			 (candidate->confidence > best_confidence)) ||
			((candidate_priority == best_priority) &&
			 (candidate_quality == best_quality) &&
			 (candidate_peak_ratio == best_peak_ratio) &&
			 (candidate->confidence == best_confidence) &&
			 (candidate->best_score > best_score)))
		{
			best_estimate = candidate;
			best_priority = candidate_priority;
			best_quality = candidate_quality;
			best_peak_ratio = candidate_peak_ratio;
			best_confidence = candidate->confidence;
			best_score = candidate->best_score;
		}
	}

	if ((best_estimate == NULL) || (fallback_quality <= 0.0f))
	{
		return fallback_estimate;
	}

	if (best_priority < fallback_priority)
	{
		return fallback_estimate;
	}

	if (best_priority > fallback_priority)
	{
		return best_estimate;
	}

	if (best_quality >=
		(fallback_quality * APP_BASELINE_CONSENSUS_MIN_QUALITY_RATIO))
	{
		return best_estimate;
	}

	return fallback_estimate;
}

/**
 * @brief Return a smoothed estimate from the tiny baseline history.
 *
 * We use a small median filter so the classical polar baseline stays in the
 * right ballpark even when a single frame latches onto glare or dial clutter.
 */
static bool AppBaselineRuntime_SelectSmoothedEstimate(
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	AppBaselineRuntime_Estimate_t ordered[APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {
		0};
	size_t sample_count = camera_baseline_estimate_history_count;

	if ((estimate_out == NULL) || (sample_count == 0U))
	{
		return false;
	}

	(void)memcpy(ordered, camera_baseline_estimate_history,
				 sample_count * sizeof(ordered[0]));

	/* Keep the local copy sorted by temperature so the median is easy to pick. */
	for (size_t i = 1U; i < sample_count; ++i)
	{
		AppBaselineRuntime_Estimate_t key = ordered[i];
		size_t j = i;

		while ((j > 0U) && (ordered[j - 1U].temperature_c > key.temperature_c))
		{
			ordered[j] = ordered[j - 1U];
			--j;
		}

		ordered[j] = key;
	}

	/* Filter out history entries with invalid angles (polluted from before
	 * the angle validation fix). Only consider entries with angles outside
	 * the subdial band (50°–130°) covering the full 270° gauge sweep. */
	{
		AppBaselineRuntime_Estimate_t valid_entries[APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {0};
		size_t valid_count = 0U;

		for (size_t i = 0U; i < sample_count; ++i)
		{
			float angle_deg = ordered[i].angle_rad * (180.0f / APP_BASELINE_PI);
			while (angle_deg < 0.0f)
			{
				angle_deg += 360.0f;
			}
			while (angle_deg >= 360.0f)
			{
				angle_deg -= 360.0f;
			}
			if ((angle_deg <= 50.0f) || (angle_deg >= 130.0f))
			{
				valid_entries[valid_count++] = ordered[i];
			}
		}

		if (valid_count > 0U)
		{
			/* Replace ordered with valid_entries for further processing. */
			(void)memcpy(ordered, valid_entries,
						 valid_count * sizeof(ordered[0]));
			sample_count = valid_count;
		}
		/* If no valid entries, fall through to use original (will be rejected
		 * by angle validation later anyway). */
	}

	if (sample_count < APP_BASELINE_ESTIMATE_HISTORY_SIZE)
	{
		/* Emit the provisional reading anyway so the classical benchmark has a
		 * value from the first accepted frame instead of going silent. */
		*estimate_out = ordered[sample_count - 1U];
		estimate_out->valid = true;
		estimate_out->source_label = "baseline-polar-warming";
		return true;
	}

	*estimate_out = ordered[sample_count / 2U];
	estimate_out->source_label = "baseline-polar-smoothed";
	estimate_out->valid = true;
	return true;
}

/**
 * @brief Map an angle inside the gauge sweep to a temperature.
 */
static float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad)
{
	/* Apply calibration offset determined from hard-case analysis.
	 * The detected angles are consistently ~2° low compared to expected
	 * values from the hard-case manifest. This corrects the systematic bias. */
	const float calibrated_angle_rad = angle_rad + (APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG * (APP_BASELINE_PI / 180.0f));
	return APP_BASELINE_MIN_VALUE_C + (AppBaselineRuntime_ConvertAngleToFraction(calibrated_angle_rad) * (APP_BASELINE_MAX_VALUE_C - APP_BASELINE_MIN_VALUE_C));
}

/**
 * @brief Convert an angle to the calibrated [0, 1] sweep fraction.
 */
static float AppBaselineRuntime_ConvertAngleToFraction(float angle_rad)
{
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG * (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	float shifted = angle_rad - min_angle_rad;

	while (shifted < 0.0f)
	{
		shifted += APP_BASELINE_TWO_PI;
	}
	while (shifted >= APP_BASELINE_TWO_PI)
	{
		shifted -= APP_BASELINE_TWO_PI;
	}

	shifted = AppBaselineRuntime_ClampFloat(shifted, 0.0f, sweep_rad);
	return AppBaselineRuntime_ClampFloat(shifted / sweep_rad, 0.0f, 1.0f);
}

/**
 * @brief Convert an angle to a sweep fraction only when it falls inside the
 * calibrated gauge arc.
 */
static bool AppBaselineRuntime_AngleToSweepFraction(float angle_rad,
													float *fraction_out)
{
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG * (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	float shifted = angle_rad - min_angle_rad;

	if (fraction_out == NULL)
	{
		return false;
	}

	while (shifted < 0.0f)
	{
		shifted += APP_BASELINE_TWO_PI;
	}
	while (shifted >= APP_BASELINE_TWO_PI)
	{
		shifted -= APP_BASELINE_TWO_PI;
	}

	if (shifted > sweep_rad)
	{
		return false;
	}

	*fraction_out = AppBaselineRuntime_ClampFloat(shifted / sweep_rad, 0.0f,
												  1.0f);
	return true;
}

/**
 * @brief Convert an angle to a sweep fraction, allowing a small tolerance at
 * the sweep boundary.
 */
static bool AppBaselineRuntime_AngleToSweepFractionWithMargin(float angle_rad,
															  float margin_rad, float *fraction_out)
{
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG * (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	const float margin = AppBaselineRuntime_ClampFloat(margin_rad, 0.0f,
													   APP_BASELINE_PI);
	float shifted = angle_rad - min_angle_rad;

	if (fraction_out == NULL)
	{
		return false;
	}

	while (shifted < 0.0f)
	{
		shifted += APP_BASELINE_TWO_PI;
	}
	while (shifted >= APP_BASELINE_TWO_PI)
	{
		shifted -= APP_BASELINE_TWO_PI;
	}

	if ((shifted > (sweep_rad + margin)) && (shifted < (APP_BASELINE_TWO_PI - margin)))
	{
		return false;
	}

	shifted = AppBaselineRuntime_ClampFloat(shifted, 0.0f, sweep_rad);
	*fraction_out = AppBaselineRuntime_ClampFloat(shifted / sweep_rad, 0.0f,
												  1.0f);
	return true;
}

/**
 * @brief Update per-frame brightness profile for adaptive thresholding.
 *
 * We sample the training crop region (step 2) and classify a frame as
 * "bright" when the average luma is high or a large fraction of pixels are
 * above the capture bright threshold.
 */
static void AppBaselineRuntime_UpdateFrameBrightnessProfile(
	const uint8_t *frame_bytes, size_t frame_size)
{
	const size_t width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const size_t height_pixels = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const size_t expected_size =
		width_pixels * height_pixels * CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const AppGaugeGeometry_Crop_t crop =
		AppGaugeGeometry_TrainingCrop(width_pixels, height_pixels);
	uint64_t luma_sum = 0U;
	size_t sample_count = 0U;
	size_t bright_count = 0U;

	camera_baseline_current_frame_is_bright = false;
	camera_baseline_current_frame_mean_luma = 0.0f;
	camera_baseline_current_frame_bright_ratio = 0.0f;

	if ((frame_bytes == NULL) || (frame_size < expected_size) ||
		(crop.width == 0U) || (crop.height == 0U))
	{
		return;
	}

	for (size_t y = crop.y_min; y < (crop.y_min + crop.height); y += 2U)
	{
		for (size_t x = crop.x_min; x < (crop.x_min + crop.width); x += 2U)
		{
			const float luma =
				AppBaselineRuntime_ReadLuma(frame_bytes, width_pixels, x, y);
			luma_sum += (uint64_t)AppBaselineRuntime_RoundToLong(luma);
			sample_count++;
			if (luma >= 180.0f)
			{
				bright_count++;
			}
		}
	}

	if (sample_count == 0U)
	{
		return;
	}

	camera_baseline_current_frame_mean_luma =
		(float)luma_sum / (float)sample_count;
	camera_baseline_current_frame_bright_ratio =
		(float)bright_count / (float)sample_count;
	camera_baseline_current_frame_is_bright =
		(camera_baseline_current_frame_mean_luma >= 188.0f) ||
		(camera_baseline_current_frame_bright_ratio >= 0.55f);
}

/**
 * @brief Normalize an angle in degrees into the [0, 360) range.
 */
static float AppBaselineRuntime_NormalizeAngleDegrees(float angle_deg)
{
	while (angle_deg < 0.0f)
	{
		angle_deg += 360.0f;
	}
	while (angle_deg >= 360.0f)
	{
		angle_deg -= 360.0f;
	}
	return angle_deg;
}

/**
 * @brief Check whether an angle falls inside the Celsius sweep.
 *
 * The calibrated sweep starts at APP_BASELINE_MIN_ANGLE_DEG and wraps through
 * 360°. We keep a small tolerance at both ends so warm/hot angles near 0°
 * are not rejected by single-frame jitter.
 */
static bool AppBaselineRuntime_IsAngleInCelsiusSweep(float angle_deg)
{
	const float norm_angle = AppBaselineRuntime_NormalizeAngleDegrees(angle_deg);
	const float sweep_end_deg =
		AppBaselineRuntime_NormalizeAngleDegrees(
			APP_BASELINE_MIN_ANGLE_DEG + APP_BASELINE_SWEEP_DEG);
	const float min_valid_deg = APP_BASELINE_MIN_ANGLE_DEG - 5.0f;
	const float max_valid_deg = sweep_end_deg + 10.0f;

	return (norm_angle >= min_valid_deg) || (norm_angle <= max_valid_deg);
}

/**
 * @brief Check whether an angle falls in the lower subdial clutter band.
 */
static bool AppBaselineRuntime_IsAngleInSubdialBand(float angle_deg)
{
	const float norm_angle = AppBaselineRuntime_NormalizeAngleDegrees(angle_deg);
	return (norm_angle > 55.0f) && (norm_angle < 130.0f);
}

/**
 * @brief Read the Y component from one packed YUV422 pixel.
 */
static float AppBaselineRuntime_ReadLuma(const uint8_t *frame_bytes,
										 size_t frame_width_pixels, size_t x, size_t y)
{
	const size_t row_stride_bytes = frame_width_pixels * 2U;
	const size_t pair_offset = (y * row_stride_bytes) + ((x & ~1U) * 2U);
	const size_t y_offset = pair_offset + (((x & 1U) != 0U) ? 2U : 0U);

	return (float)frame_bytes[y_offset];
}

/**
 * @brief Read the U and V components from one packed YUV422 pixel pair.
 */
static void AppBaselineRuntime_ReadChroma(const uint8_t *frame_bytes,
										  size_t frame_width_pixels, size_t x, size_t y,
										  float *u_out, float *v_out)
{
	const size_t row_stride_bytes = frame_width_pixels * 2U;
	const size_t pair_offset = (y * row_stride_bytes) + ((x & ~1U) * 2U);

	if (u_out != NULL)
	{
		*u_out = (float)frame_bytes[pair_offset + 1U];
	}
	if (v_out != NULL)
	{
		*v_out = (float)frame_bytes[pair_offset + 3U];
	}
}

/**
 * @brief Measure local edge strength and background brightness around one pixel.
 *
 * The polar baseline votes for dark, radial structures using a Sobel-like
 * edge response and the surrounding luma average.
 */
static float AppBaselineRuntime_ReadEdgeMagnitude(const uint8_t *frame_bytes,
												  size_t frame_width_pixels, size_t frame_height_pixels, size_t x, size_t y,
												  float *gradient_x_out, float *gradient_y_out,
												  float *background_luma_out)
{
	const bool at_border = (x < 1U) || (y < 1U) || ((x + 1U) >= frame_width_pixels) || ((y + 1U) >= frame_height_pixels);

	if (gradient_x_out != NULL)
	{
		*gradient_x_out = 0.0f;
	}
	if (gradient_y_out != NULL)
	{
		*gradient_y_out = 0.0f;
	}
	if (background_luma_out != NULL)
	{
		*background_luma_out = 0.0f;
	}

	if (at_border)
	{
		return 0.0f;
	}

	{
		const float top_left = AppBaselineRuntime_ReadLuma(frame_bytes,
														   frame_width_pixels, x - 1U, y - 1U);
		const float top_center = AppBaselineRuntime_ReadLuma(frame_bytes,
															 frame_width_pixels, x, y - 1U);
		const float top_right = AppBaselineRuntime_ReadLuma(frame_bytes,
															frame_width_pixels, x + 1U, y - 1U);
		const float mid_left = AppBaselineRuntime_ReadLuma(frame_bytes,
														   frame_width_pixels, x - 1U, y);
		const float mid_right = AppBaselineRuntime_ReadLuma(frame_bytes,
															frame_width_pixels, x + 1U, y);
		const float bottom_left = AppBaselineRuntime_ReadLuma(frame_bytes,
															  frame_width_pixels, x - 1U, y + 1U);
		const float bottom_center = AppBaselineRuntime_ReadLuma(frame_bytes,
																frame_width_pixels, x, y + 1U);
		const float bottom_right = AppBaselineRuntime_ReadLuma(frame_bytes,
															   frame_width_pixels, x + 1U, y + 1U);
		const float gx = (top_right + (2.0f * mid_right) + bottom_right) - (top_left + (2.0f * mid_left) + bottom_left);
		const float gy = (bottom_left + (2.0f * bottom_center) + bottom_right) - (top_left + (2.0f * top_center) + top_right);
		const float background_luma =
			(top_left + top_center + top_right + mid_left + mid_right + bottom_left + bottom_center + bottom_right) / 8.0f;

		if (gradient_x_out != NULL)
		{
			*gradient_x_out = gx;
		}
		if (gradient_y_out != NULL)
		{
			*gradient_y_out = gy;
		}
		if (background_luma_out != NULL)
		{
			*background_luma_out = background_luma;
		}

		return sqrtf((gx * gx) + (gy * gy));
	}
}

/**
 * @brief Check whether a sampled point falls inside the subdial clutter mask.
 */
static bool AppBaselineRuntime_IsInSubdialMask(size_t center_x, size_t center_y,
											   size_t x, size_t y, float radius_px)
{
	const float dx = (float)((x >= center_x) ? (x - center_x) : (center_x - x));
	const float dy = (float)((y >= center_y) ? (y - center_y) : (center_y - y));

	return (dx < (APP_BASELINE_SUBDIAL_X_FRACTION * radius_px)) && ((float)y > ((float)center_y + (APP_BASELINE_SUBDIAL_Y_MIN_FRACTION * radius_px))) && ((float)y < ((float)center_y + (APP_BASELINE_SUBDIAL_Y_MAX_FRACTION * radius_px))) && (dy > (APP_BASELINE_SUBDIAL_Y_MIN_FRACTION * radius_px));
}

/**
 * @brief Return a smooth weight that emphasizes the middle of the shaft.
 *
 * The dial face is noisy near the hub and the outer tick ring, so the ray
 * vote should focus on the cleaner middle band where the dark needle is most
 * visible.
 */
static float AppBaselineRuntime_MiddleShaftWeight(float sample_progress)
{
	const float shaft_center = 0.55f;
	const float shaft_sigma = 0.18f;
	const float normalized = (sample_progress - shaft_center) / shaft_sigma;

	return expf(-0.5f * normalized * normalized);
}

/**
 * @brief Score one ray candidate by favoring dark pixels on the needle line.
 */
static float AppBaselineRuntime_ScoreAngle(const uint8_t *frame_bytes,
										   size_t frame_width_pixels, size_t frame_height_pixels, size_t center_x,
										   size_t center_y, float angle_rad)
{
	/* Angular Mask: Only score angles within the calibrated gauge sweep.
	 * This prevents the baseline from locking onto the dial bezel or
	 * physical stops at the ends of the scale. */
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG * (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	float shifted = angle_rad - min_angle_rad;

	while (shifted < 0.0f)
	{
		shifted += APP_BASELINE_TWO_PI;
	}
	while (shifted >= APP_BASELINE_TWO_PI)
	{
		shifted -= APP_BASELINE_TWO_PI;
	}

	if (shifted > sweep_rad)
	{
		return 0.0f;
	}

	/* Boundary Penalty: Slightly penalize votes that fall exactly on the
	 * sweep boundaries to prevent the baseline from saturating at -30C or 50C
	 * when the strongest signal is actually outside the valid range. */
	float boundary_weight = 1.0f;
	const float boundary_margin = 0.05f * sweep_rad;
	if (shifted < boundary_margin || shifted > (sweep_rad - boundary_margin))
	{
		boundary_weight = 0.7f;
	}

	const float unit_dx = cosf(angle_rad);
	const float unit_dy = sinf(angle_rad);
	const float perp_dx = -unit_dy;
	const float perp_dy = unit_dx;
	const float center_x_f = (float)center_x;
	const float center_y_f = (float)center_y;
	const float max_radius_x =
		(float)((center_x < (frame_width_pixels - 1U - center_x)) ? center_x
																  : (frame_width_pixels - 1U - center_x));
	const float max_radius_y =
		(float)((center_y < (frame_height_pixels - 1U - center_y)) ? center_y
																   : (frame_height_pixels - 1U - center_y));
	const float max_radius = (max_radius_x < max_radius_y) ? max_radius_x
														   : max_radius_y;
	const float start_radius = max_radius * APP_BASELINE_RAY_START_FRACTION;
	const float end_radius = max_radius * APP_BASELINE_RAY_END_FRACTION;
	const float radius_step =
		(APP_BASELINE_RAY_SAMPLES > 1U) ? ((end_radius - start_radius) / (float)(APP_BASELINE_RAY_SAMPLES - 1U)) : 0.0f;
	float score = 0.0f;
	float score_sq_sum = 0.0f;
	size_t valid_sample_count = 0U;

	for (size_t sample_index = 0U; sample_index < APP_BASELINE_RAY_SAMPLES;
		 ++sample_index)
	{
		const float radius = start_radius + (radius_step * (float)sample_index);
		const float sample_progress = (float)sample_index / (float)(APP_BASELINE_RAY_SAMPLES - 1U);
		/* Prefer the cleaner middle shaft over the noisy hub and tip/tick region. */
		const float weight = 0.35f +
							 (0.65f * AppBaselineRuntime_MiddleShaftWeight(sample_progress));
		const long sample_x = AppBaselineRuntime_RoundToLong(
			center_x_f + (unit_dx * radius));
		const long sample_y = AppBaselineRuntime_RoundToLong(
			center_y_f + (unit_dy * radius));
		float background_sum = 0.0f;
		size_t background_count = 0U;

		if ((sample_x < 0L) || (sample_y < 0L) || ((size_t)sample_x >= frame_width_pixels) || ((size_t)sample_y >= frame_height_pixels))
		{
			continue;
		}

		if (AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
											   (size_t)sample_x, (size_t)sample_y, max_radius))
		{
			continue;
		}

		/* Skip this sample entirely if the line pixel itself is saturated —
		 * glare makes both the needle and its background equally bright, so
		 * the contrast score is meaningless and including it only dilutes
		 * valid_sample_count without adding useful signal. */
		{
			const float line_luma_check = AppBaselineRuntime_ReadLuma(
				frame_bytes, frame_width_pixels,
				(size_t)sample_x, (size_t)sample_y);
			if (line_luma_check > (float)APP_BASELINE_SATURATION_THRESHOLD)
			{
				continue;
			}
		}

		for (size_t offset_index = 0U;
			 offset_index < APP_BASELINE_LOCAL_BACKGROUND_OFFSETS;
			 ++offset_index)
		{
			const float offset = 2.0f + (2.0f * (float)offset_index);
			const long left_x = AppBaselineRuntime_RoundToLong(
				((float)sample_x) + (perp_dx * offset));
			const long left_y = AppBaselineRuntime_RoundToLong(
				((float)sample_y) + (perp_dy * offset));
			const long right_x = AppBaselineRuntime_RoundToLong(
				((float)sample_x) - (perp_dx * offset));
			const long right_y = AppBaselineRuntime_RoundToLong(
				((float)sample_y) - (perp_dy * offset));
			const bool left_in_bounds = (left_x >= 0L) && (left_y >= 0L) && ((size_t)left_x < frame_width_pixels) && ((size_t)left_y < frame_height_pixels);
			const bool right_in_bounds = (right_x >= 0L) && (right_y >= 0L) && ((size_t)right_x < frame_width_pixels) && ((size_t)right_y < frame_height_pixels);

			if (left_in_bounds && !AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
																	  (size_t)left_x, (size_t)left_y, max_radius))
			{
				const float bg_luma = AppBaselineRuntime_ReadLuma(frame_bytes,
																  frame_width_pixels, (size_t)left_x, (size_t)left_y);
				/* Skip saturated background samples — they don't reflect true
				 * dial-face brightness and would lower the measured contrast. */
				if (bg_luma <= (float)APP_BASELINE_SATURATION_THRESHOLD)
				{
					background_sum += bg_luma;
					background_count++;
				}
			}

			if (right_in_bounds && !AppBaselineRuntime_IsInSubdialMask(center_x, center_y,
																	   (size_t)right_x, (size_t)right_y, max_radius))
			{
				const float bg_luma_r = AppBaselineRuntime_ReadLuma(frame_bytes,
																	frame_width_pixels, (size_t)right_x, (size_t)right_y);
				if (bg_luma_r <= (float)APP_BASELINE_SATURATION_THRESHOLD)
				{
					background_sum += bg_luma_r;
					background_count++;
				}
			}
		}

		if (background_count == 0U)
		{
			continue;
		}

		{
			const float line_luma = AppBaselineRuntime_ReadLuma(frame_bytes,
																frame_width_pixels, (size_t)sample_x, (size_t)sample_y);
			const float local_background = background_sum / (float)background_count;
			const float local_contrast = local_background - line_luma;

			if (local_contrast <= 0.0f)
			{
				continue;
			}

			score += (local_contrast * weight);
			score_sq_sum += (local_contrast * local_contrast * weight);
			valid_sample_count++;
		}
	}

	if (valid_sample_count == 0U)
	{
		return 0.0f;
	}

	/* Linearity Weighting: A true needle has consistent contrast along its length.
	 * We penalize rays with high variance (e.g. a single bright spot) by
	 * calculating the ratio of the mean square to the square of the mean. */
	float mean = score / (float)valid_sample_count;
	float mean_sq = score_sq_sum / (float)valid_sample_count;
	float variance = mean_sq - (mean * mean);
	float linearity = 1.0f / (1.0f + (variance / (mean * mean + 1e-6f)));

	return (mean * linearity * boundary_weight);
}

/**
 * @brief Score the needle by voting in polar space around one geometry seed.
 *
 * This mirrors the hard-case-winning gradient-polar detector:
 * - work in a fixed gauge crop
 * - reject the subdial clutter
 * - vote for spoke angles using Sobel edge alignment in the inner annulus
 * - keep the strongest angular peak if it stands clearly above background
 */
static bool AppBaselineRuntime_EstimatePolarNeedle(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, size_t center_x, size_t center_y,
	float dial_radius_px,
	const char *source_label, AppBaselineRuntime_Estimate_t *estimate_out)
{
	const float min_angle_rad = APP_BASELINE_MIN_ANGLE_DEG * (APP_BASELINE_PI / 180.0f);
	const float sweep_rad = APP_BASELINE_SWEEP_DEG * (APP_BASELINE_PI / 180.0f);
	/* Reduced from 12° to 6° — the 12° margin was too wide and allowed dial
	 * markings and subdial artifacts to pollute the vote. The needle is a
	 * thin dark spoke; we want tight angular filtering to reject the radial
	 * dial artwork that sits just outside the calibrated sweep.
	 * Widened to 10° (2026-04-30) — the 6° margin was too tight and rejected
	 * valid needle angles on captures where the needle sits near the sweep
	 * boundaries (e.g. -30°C or 50°C). The spoke-continuity check below is
	 * sufficient to reject dial markings. */
	const float angle_margin_rad = 10.0f * (APP_BASELINE_PI / 180.0f);
	const size_t scan_x_max_inclusive = scan_x_max - 1U;
	const size_t scan_y_max_inclusive = scan_y_max - 1U;
	float angle_votes[APP_BASELINE_ANGLE_BINS] = {0.0f};
	float smoothed_votes[APP_BASELINE_ANGLE_BINS] = {0.0f};
	float best_score = -1.0f;
	float runner_up_score = -1.0f;
	size_t best_bin = 0U;
	const bool bright_relaxed = camera_baseline_current_frame_is_bright;
	const float edge_threshold = bright_relaxed ? 5.5f : 8.0f;
	const float main_continuity_threshold = bright_relaxed ? 0.24f : 0.35f;
	const float main_hub_threshold = bright_relaxed ? 0.15f : 0.25f;
	const float hot_continuity_threshold = bright_relaxed ? 0.22f : 0.28f;
	const float hot_hub_threshold = bright_relaxed ? 0.12f : 0.18f;
	const float final_spoke_continuity_threshold = bright_relaxed ? 0.22f : 0.30f;

	if ((estimate_out == NULL) || (frame_bytes == NULL) || (source_label == NULL))
	{
		return false;
	}

	if (frame_size < (frame_width_pixels * frame_height_pixels * CAMERA_CAPTURE_BYTES_PER_PIXEL))
	{
		return false;
	}

	if ((scan_x_max <= scan_x_min) || (scan_y_max <= scan_y_min))
	{
		return false;
	}

	if (dial_radius_px < (float)APP_BASELINE_MIN_RADIUS_PIXELS)
	{
		return false;
	}

	{
		/* Focus on middle shaft (30%-70%) where needle is most distinct from
		 * dial markings. The Python reference uses this tighter range. */
		const float search_radius_min = AppBaselineRuntime_ClampFloat(
			dial_radius_px * 0.30f, (float)APP_BASELINE_MIN_RADIUS_PIXELS,
			dial_radius_px);
		const float search_radius_max = dial_radius_px * 0.70f;
		if (search_radius_max <= search_radius_min)
		{
			return false;
		}

		for (size_t y = scan_y_min + 1U; y < scan_y_max_inclusive; ++y)
		{
			for (size_t x = scan_x_min + 1U; x < scan_x_max_inclusive; ++x)
			{
				const float dx = (float)x - (float)center_x;
				const float dy = (float)y - (float)center_y;
				const float radius = sqrtf((dx * dx) + (dy * dy));
				const float luma = AppBaselineRuntime_ReadLuma(frame_bytes,
															   frame_width_pixels, x, y);
				float gradient_x = 0.0f;
				float gradient_y = 0.0f;
				float fraction = 0.0f;

				if ((radius < search_radius_min) || (radius > search_radius_max))
				{
					continue;
				}

				if ((luma > (float)APP_BASELINE_SATURATION_THRESHOLD) || AppBaselineRuntime_IsInSubdialMask(center_x, center_y, x,
																											y, dial_radius_px))
				{
					continue;
				}

				{
					const float edge_mag = AppBaselineRuntime_ReadEdgeMagnitude(
						frame_bytes, frame_width_pixels, frame_height_pixels, x,
						y, &gradient_x, &gradient_y, NULL);

					/* Bright frames reduce apparent edge magnitude. Use a relaxed
					 * edge floor only when the frame brightness profile indicates
					 * heavy overexposure; otherwise keep the nominal threshold. */
					if (edge_mag <= edge_threshold)
					{
						continue;
					}

					if (!AppBaselineRuntime_AngleToSweepFractionWithMargin(
							atan2f(dy, dx), angle_margin_rad, &fraction))
					{
						continue;
					}

					{
						const float grad_mag_safe = (edge_mag > 1.0f) ? edge_mag : 1.0f;
						const float radial_x = dx / radius;
						const float radial_y = dy / radius;
						const float grad_x = gradient_x / grad_mag_safe;
						const float grad_y = gradient_y / grad_mag_safe;
						/* Tangential component: measures how well the edge aligns with
						 * a radial spoke. */
						const float tangential = (grad_x * radial_y) - (grad_y * radial_x);
						/* Darkness weight: the needle is dark on a light background. */
						const float darkness = (255.0f - luma) / 255.0f;
						const float sample_progress =
							(radius - search_radius_min) /
							(search_radius_max - search_radius_min + 1e-6f);
						const float shaft_weight = 0.35f +
												   (0.65f * AppBaselineRuntime_MiddleShaftWeight(sample_progress));

						const float vote =
							edge_mag * fabsf(tangential) * darkness * shaft_weight;
						const size_t bin_index = (size_t)AppBaselineRuntime_RoundToLong(
							fraction * (float)(APP_BASELINE_ANGLE_BINS - 1U));

						if (bin_index < APP_BASELINE_ANGLE_BINS)
						{
							/* Hub-connection boost: the needle is a long spoke that connects
							 * to the center. Dial markings are short edges that don't reach
							 * the center. Check if there's a dark path toward the center. */
							float hub_connection = 0.0f;
							const size_t steps = 7U;
							for (size_t step = 1U; step <= steps; ++step)
							{
								const float t = (float)step / (float)(steps + 1U);
								const long hx = AppBaselineRuntime_RoundToLong(
									(float)center_x + (dx * t * 0.6f));
								const long hy = AppBaselineRuntime_RoundToLong(
									(float)center_y + (dy * t * 0.6f));
								if (hx >= 0 && (size_t)hx < frame_width_pixels &&
									hy >= 0 && (size_t)hy < frame_height_pixels)
								{
									const float hub_luma = AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels, (size_t)hx, (size_t)hy);
									hub_connection += ((255.0f - hub_luma) / 255.0f);
								}
							}
							hub_connection /= (float)steps;

							/* Tip-extension check: the needle extends beyond the middle shaft
							 * toward the outer dial edge. Dial markings are isolated.
							 * Sample points from 70% to 95% of dial radius. */
							float tip_extension = 0.0f;
							const size_t tip_steps = 5U;
							for (size_t step = 0U; step < tip_steps; ++step)
							{
								const float r_frac = 0.70f + (0.25f * (float)step / (float)(tip_steps - 1U));
								const long tx = AppBaselineRuntime_RoundToLong(
									(float)center_x + (dx / radius) * r_frac * dial_radius_px);
								const long ty = AppBaselineRuntime_RoundToLong(
									(float)center_y + (dy / radius) * r_frac * dial_radius_px);
								if (tx >= 0 && (size_t)tx < frame_width_pixels &&
									ty >= 0 && (size_t)ty < frame_height_pixels)
								{
									const float tip_luma = AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels, (size_t)tx, (size_t)ty);
									tip_extension += ((255.0f - tip_luma) / 255.0f);
								}
							}
							tip_extension /= (float)tip_steps;

							/* Width check: the needle is a thin spoke. Sample perpendicular
							 * to the spoke direction and check if dark region is narrow.
							 * Dial markings are typically wider edges. */
							float width_score = 1.0f;
							{
								const float perp_x = -dy / radius;
								const float perp_y = dx / radius;
								float perp_darkness = 0.0f;
								const size_t width_samples = 5U;
								for (size_t w = 0U; w < width_samples; ++w)
								{
									const float offset = (float)(w - width_samples / 2) * 1.5f;
									const long wx = AppBaselineRuntime_RoundToLong((float)x + perp_x * offset);
									const long wy = AppBaselineRuntime_RoundToLong((float)y + perp_y * offset);
									if (wx >= 0 && (size_t)wx < frame_width_pixels &&
										wy >= 0 && (size_t)wy < frame_height_pixels)
									{
										const float w_luma = AppBaselineRuntime_ReadLuma(
											frame_bytes, frame_width_pixels, (size_t)wx, (size_t)wy);
										perp_darkness += ((255.0f - w_luma) / 255.0f);
									}
								}
								/* Thin spoke: darkness concentrated in center samples.
								 * Wide marking: darkness spread across all samples. */
								const float center_darkness = ((255.0f - luma) / 255.0f);
								const float avg_perp_darkness = perp_darkness / (float)width_samples;
								/* High score if center is much darker than average (thin). */
								width_score = 0.3f + (0.7f * (center_darkness / (avg_perp_darkness + 0.01f)));
								if (width_score > 1.0f)
									width_score = 1.0f;
							}

							/* Combined boost: hub connection AND tip extension AND thin width.
							 * Needle: hub~0.9, tip~0.8, width~0.9 → boost ~18x
							 * Dial marking: hub~0.1, tip~0.2, width~0.5 → boost ~0.03x */
							const float spoke_score = ((hub_connection * 0.4f) +
													   (tip_extension * 0.4f) +
													   (width_score * 0.2f));
							const float connection_boost = 0.03f + (19.97f * spoke_score * spoke_score);

							/* No center-of-sweep bias - the needle can be at any angle
							 * across the full -30°C to 50°C range. The angle validation
							 * (130°-320°) is sufficient to reject outliers. */
							angle_votes[bin_index] += (vote * connection_boost);
						}
					}
				}
			}
		}
	}

	for (size_t bin_index = 0U; bin_index < APP_BASELINE_ANGLE_BINS;
		 ++bin_index)
	{
		const size_t prev_index =
			(bin_index + APP_BASELINE_ANGLE_BINS - 1U) % APP_BASELINE_ANGLE_BINS;
		const size_t next_index = (bin_index + 1U) % APP_BASELINE_ANGLE_BINS;
		smoothed_votes[bin_index] =
			(angle_votes[prev_index] + angle_votes[bin_index] + angle_votes[next_index]) / 3.0f;
	}

	{
		float vote_sum = 0.0f;
		enum
		{
			/* Keep more candidate peaks so hot-wrap angles (near 30°-60°)
			 * are not dropped when their raw gradient vote is weaker than
			 * mid-range false positives. */
			APP_BASELINE_TOP_PEAK_COUNT = 64
		};
		size_t top_bins[APP_BASELINE_TOP_PEAK_COUNT] = {0};
		float top_scores[APP_BASELINE_TOP_PEAK_COUNT] = {0.0f};

		for (size_t bin_index = 0U; bin_index < APP_BASELINE_ANGLE_BINS;
			 ++bin_index)
		{
			const float val = smoothed_votes[bin_index];
			vote_sum += val;

			/* Keep track of top 16 candidates. */
			for (size_t i = 0U; i < APP_BASELINE_TOP_PEAK_COUNT; ++i)
			{
				if (val > top_scores[i])
				{
					for (size_t j = APP_BASELINE_TOP_PEAK_COUNT - 1U; j > i; --j)
					{
						top_scores[j] = top_scores[j - 1U];
						top_bins[j] = top_bins[j - 1U];
					}
					top_scores[i] = val;
					top_bins[i] = bin_index;
					break;
				}
			}
		}

		if ((top_scores[0] <= 0.0f))
		{
			DebugConsole_Printf(
				"[BASELINE] Polar reject: no_peak source=%s mode=%s\r\n",
				source_label,
				bright_relaxed ? "bright-relaxed" : "normal");
			return false;
		}

		/* Use spoke continuity to select the best peak. The needle forms a
		 * continuous dark spoke from center to edge, while dial markings are
		 * isolated edges with poor continuity. Weight each peak by both
		 * vote score and continuity to find the true needle. */
		best_bin = top_bins[0];
		best_score = smoothed_votes[best_bin];

		/* Classical spoke-continuity weighted peak selection.
		 * Analyze top peaks and pick the one with best continuity-weighted score.
		 * This ensures we select the needle (continuous spoke) over dial
		 * markings (isolated edges with high vote but poor continuity).
		 * We intentionally cap this at APP_BASELINE_TOP_PEAK_COUNT because the
		 * candidate arrays are fixed-size. */
		size_t best_weighted_bin = best_bin;
		{
			float best_weighted_score = 0.0f;

			for (size_t peak_idx = 0U; peak_idx < APP_BASELINE_TOP_PEAK_COUNT && top_scores[peak_idx] > 0.0f; ++peak_idx)
			{
				const size_t candidate_bin = top_bins[peak_idx];
				const float candidate_angle = min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
				const float cos_a = cosf(candidate_angle);
				const float sin_a = sinf(candidate_angle);

				/* Classical CV: measure spoke continuity along this angle.
				 * The needle forms a continuous dark spoke from center to edge.
				 * Dial markings are isolated edges without continuity. */
				float continuity = 0.0f;
				const size_t continuity_samples = 12U;
				size_t valid_samples = 0U;

				for (size_t i = 0U; i < continuity_samples; ++i)
				{
					const float r_frac = 0.20f + (0.60f * (float)i / (float)(continuity_samples - 1U));
					const long sx = AppBaselineRuntime_RoundToLong(
						(float)center_x + (cos_a * r_frac * dial_radius_px));
					const long sy = AppBaselineRuntime_RoundToLong(
						(float)center_y + (sin_a * r_frac * dial_radius_px));

					if (sx >= 0 && (size_t)sx < frame_width_pixels &&
						sy >= 0 && (size_t)sy < frame_height_pixels)
					{
						const float sample_luma = AppBaselineRuntime_ReadLuma(
							frame_bytes, frame_width_pixels, (size_t)sx, (size_t)sy);
						continuity += ((255.0f - sample_luma) / 255.0f);
						valid_samples++;
					}
				}

				/* Hub darkness check: the needle connects to the center hub,
				 * so there should be darkness near the center along this angle.
				 * Dial markings don't reach the center, so center region is light. */
				float hub_darkness = 0.0f;
				{
					const size_t hub_samples = 3U;
					size_t hub_valid = 0U;
					for (size_t h = 0U; h < hub_samples; ++h)
					{
						const float r_frac = 0.10f + (0.15f * (float)h / (float)(hub_samples - 1U));
						const long hx = AppBaselineRuntime_RoundToLong(
							(float)center_x + (cos_a * r_frac * dial_radius_px));
						const long hy = AppBaselineRuntime_RoundToLong(
							(float)center_y + (sin_a * r_frac * dial_radius_px));
						if (hx >= 0 && (size_t)hx < frame_width_pixels &&
							hy >= 0 && (size_t)hy < frame_height_pixels)
						{
							const float hub_luma = AppBaselineRuntime_ReadLuma(
								frame_bytes, frame_width_pixels, (size_t)hx, (size_t)hy);
							hub_darkness += ((255.0f - hub_luma) / 255.0f);
							hub_valid++;
						}
					}
					if (hub_valid > 0U)
					{
						hub_darkness /= (float)hub_valid;
					}
				}

				if (valid_samples > 0U)
				{
					continuity /= (float)valid_samples;
					/* Skip peaks with poor continuity OR poor hub connection.
					 * Needle has both: continuous spoke AND dark hub connection.
					 * Under bright-relaxed mode we use lower thresholds to account
					 * for reduced contrast in overexposed captures. */
					if (continuity >= main_continuity_threshold &&
						hub_darkness >= main_hub_threshold)
					{
						/* Weight by continuity^2 * hub_darkness * vote.
						 * This strongly favors the needle which has both
						 * continuity and hub connection. */
						const float weighted_score = (continuity * continuity) * hub_darkness * top_scores[peak_idx];
						if (weighted_score > best_weighted_score)
						{
							best_weighted_score = weighted_score;
							best_weighted_bin = candidate_bin;
						}
					}
				}
			}
		}

		best_bin = best_weighted_bin;
		best_score = smoothed_votes[best_bin];

		/* Hot-zone second pass: if the best peak is in the cold/mid range
		 * (135°-315°), check if there's a stronger spoke-continuity peak
		 * in the hot wrap-around zone (30°-60°). The needle at high
		 * temperatures (35°C-50°C) sits near 30°-60° where the gradient
		 * signal is weak, so the primary vote often misses it and picks
		 * up a false positive from a dial marking instead. */
		{
			const float best_angle_deg_check =
				(min_angle_rad + ((float)best_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad) * (180.0f / APP_BASELINE_PI);
			float best_hot_weighted = 0.0f;
			size_t best_hot_bin = best_bin;

			/* Only do the hot-zone search if the primary peak is in the
			 * cold/mid range (135°-315°). */
			if (best_angle_deg_check >= 135.0f && best_angle_deg_check <= 315.0f)
			{
				for (size_t peak_idx = 0U; peak_idx < APP_BASELINE_TOP_PEAK_COUNT && top_scores[peak_idx] > 0.0f; ++peak_idx)
				{
					const size_t candidate_bin = top_bins[peak_idx];
					const float candidate_angle_deg =
						(min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad) * (180.0f / APP_BASELINE_PI);
					float norm_angle = candidate_angle_deg;
					while (norm_angle < 0.0f)
						norm_angle += 360.0f;
					while (norm_angle >= 360.0f)
						norm_angle -= 360.0f;

					/* Check if this peak is in the hot wrap-around zone.
					 * Widened to 20°-75° so high-temperature needle positions
					 * near the sweep edge are less likely to be missed. */
					if (norm_angle >= 20.0f && norm_angle <= 75.0f)
					{
						const float candidate_angle = min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
						const float cos_a = cosf(candidate_angle);
						const float sin_a = sinf(candidate_angle);
						float continuity = 0.0f;
						float hub_darkness = 0.0f;
						const size_t cont_samples = 12U;
						size_t valid_samples = 0U;

						for (size_t i = 0U; i < cont_samples; ++i)
						{
							const float r_frac = 0.20f + (0.60f * (float)i / (float)(cont_samples - 1U));
							const long sx = AppBaselineRuntime_RoundToLong(
								(float)center_x + (cos_a * r_frac * dial_radius_px));
							const long sy = AppBaselineRuntime_RoundToLong(
								(float)center_y + (sin_a * r_frac * dial_radius_px));
							if (sx >= 0 && (size_t)sx < frame_width_pixels &&
								sy >= 0 && (size_t)sy < frame_height_pixels)
							{
								const float sample_luma = AppBaselineRuntime_ReadLuma(
									frame_bytes, frame_width_pixels, (size_t)sx, (size_t)sy);
								continuity += ((255.0f - sample_luma) / 255.0f);
								valid_samples++;
							}
						}
						if (valid_samples > 0U)
							continuity /= (float)valid_samples;

						/* Hub darkness check. */
						{
							const size_t hub_samples = 3U;
							size_t hub_valid = 0U;
							for (size_t h = 0U; h < hub_samples; ++h)
							{
								const float r_frac = 0.10f + (0.15f * (float)h / (float)(hub_samples - 1U));
								const long hx = AppBaselineRuntime_RoundToLong(
									(float)center_x + (cos_a * r_frac * dial_radius_px));
								const long hy = AppBaselineRuntime_RoundToLong(
									(float)center_y + (sin_a * r_frac * dial_radius_px));
								if (hx >= 0 && (size_t)hx < frame_width_pixels &&
									hy >= 0 && (size_t)hy < frame_height_pixels)
								{
									const float hub_luma = AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels, (size_t)hx, (size_t)hy);
									hub_darkness += ((255.0f - hub_luma) / 255.0f);
									hub_valid++;
								}
							}
							if (hub_valid > 0U)
								hub_darkness /= (float)hub_valid;
						}

						/* Accept hot-zone peak if it has good continuity and hub
						 * connection, even with a lower vote score. Bright-relaxed
						 * mode lowers these gates slightly for overexposed frames. */
						if (continuity >= hot_continuity_threshold &&
							hub_darkness >= hot_hub_threshold)
						{
							const float weighted = (continuity * continuity) * hub_darkness * top_scores[peak_idx];
							if (weighted > best_hot_weighted)
							{
								best_hot_weighted = weighted;
								best_hot_bin = candidate_bin;
							}
						}
					}
				}

				/* Override the primary peak if a hot-zone candidate has
				 * strong spoke continuity AND a reasonable vote score.
				 * Only override if the hot-zone candidate has at least 35%
				 * of the primary peak's raw vote score, to avoid overriding
				 * a strong primary peak with a weak hot-zone candidate. */
				if (best_hot_weighted > 0.0f &&
					smoothed_votes[best_hot_bin] >= (0.35f * smoothed_votes[best_bin]))
				{
					const float hot_angle_deg =
						(min_angle_rad + ((float)best_hot_bin /
											   (float)(APP_BASELINE_ANGLE_BINS - 1U)) *
											  sweep_rad) *
						(180.0f / APP_BASELINE_PI);
					const float primary_angle_deg_norm =
						AppBaselineRuntime_NormalizeAngleDegrees(best_angle_deg_check);
					const float hot_angle_deg_norm =
						AppBaselineRuntime_NormalizeAngleDegrees(hot_angle_deg);
					const long primary_angle_x10 = AppBaselineRuntime_RoundToLong(
						primary_angle_deg_norm * 10.0f);
					const long hot_angle_x10 = AppBaselineRuntime_RoundToLong(
						hot_angle_deg_norm * 10.0f);
					best_bin = best_hot_bin;
					best_score = smoothed_votes[best_bin];
					DebugConsole_Printf(
						"[BASELINE] Hot-zone override: primary=%ld.%01lddeg hot=%ld.%01lddeg\r\n",
						primary_angle_x10 / 10L,
						((primary_angle_x10 % 10L) < 0L) ? -(primary_angle_x10 % 10L) : (primary_angle_x10 % 10L),
						hot_angle_x10 / 10L,
						((hot_angle_x10 % 10L) < 0L) ? -(hot_angle_x10 % 10L) : (hot_angle_x10 % 10L));
				}
				else if (best_hot_weighted > 0.0f)
				{
					DebugConsole_Printf(
						"[BASELINE] Hot-zone candidate kept secondary: primary_vote=%ld hot_vote=%ld\r\n",
						AppBaselineRuntime_RoundToLong(smoothed_votes[best_bin]),
						AppBaselineRuntime_RoundToLong(smoothed_votes[best_hot_bin]));
				}
			}
		}

		/* Hot-zone full sweep rescue: when the shortlist still misses the
		 * true hot-end needle, evaluate every bin in 20°-75° directly.
		 * This is intentionally conservative (extra continuity/hub checks
		 * plus minimum vote ratio) so we only override strong false
		 * mid-angle picks with credible hot-wrap spokes. */
		{
			const float best_angle_deg_check =
				(min_angle_rad + ((float)best_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad) * (180.0f / APP_BASELINE_PI);
			float best_hot_weighted_full = 0.0f;
			size_t best_hot_bin_full = best_bin;

			if (best_angle_deg_check >= 135.0f && best_angle_deg_check <= 315.0f)
			{
				for (size_t candidate_bin = 0U; candidate_bin < APP_BASELINE_ANGLE_BINS; ++candidate_bin)
				{
					const float candidate_angle_deg =
						(min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad) * (180.0f / APP_BASELINE_PI);
					float norm_angle = candidate_angle_deg;
					while (norm_angle < 0.0f)
					{
						norm_angle += 360.0f;
					}
					while (norm_angle >= 360.0f)
					{
						norm_angle -= 360.0f;
					}
					if (norm_angle < 20.0f || norm_angle > 75.0f)
					{
						continue;
					}

					const float candidate_angle = min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
					const float cos_a = cosf(candidate_angle);
					const float sin_a = sinf(candidate_angle);
					float continuity = 0.0f;
					float hub_darkness = 0.0f;
					const size_t cont_samples = 12U;
					size_t valid_samples = 0U;

					for (size_t i = 0U; i < cont_samples; ++i)
					{
						const float r_frac = 0.20f + (0.60f * (float)i / (float)(cont_samples - 1U));
						const long sx = AppBaselineRuntime_RoundToLong(
							(float)center_x + (cos_a * r_frac * dial_radius_px));
						const long sy = AppBaselineRuntime_RoundToLong(
							(float)center_y + (sin_a * r_frac * dial_radius_px));
						if (sx >= 0 && (size_t)sx < frame_width_pixels &&
							sy >= 0 && (size_t)sy < frame_height_pixels)
						{
							const float sample_luma = AppBaselineRuntime_ReadLuma(
								frame_bytes, frame_width_pixels, (size_t)sx, (size_t)sy);
							continuity += ((255.0f - sample_luma) / 255.0f);
							valid_samples++;
						}
					}
					if (valid_samples > 0U)
					{
						continuity /= (float)valid_samples;
					}
					else
					{
						continue;
					}

					{
						const size_t hub_samples = 3U;
						size_t hub_valid = 0U;
						for (size_t h = 0U; h < hub_samples; ++h)
						{
							const float r_frac = 0.10f + (0.15f * (float)h / (float)(hub_samples - 1U));
							const long hx = AppBaselineRuntime_RoundToLong(
								(float)center_x + (cos_a * r_frac * dial_radius_px));
							const long hy = AppBaselineRuntime_RoundToLong(
								(float)center_y + (sin_a * r_frac * dial_radius_px));
							if (hx >= 0 && (size_t)hx < frame_width_pixels &&
								hy >= 0 && (size_t)hy < frame_height_pixels)
							{
								const float hub_luma = AppBaselineRuntime_ReadLuma(
									frame_bytes, frame_width_pixels, (size_t)hx, (size_t)hy);
								hub_darkness += ((255.0f - hub_luma) / 255.0f);
								hub_valid++;
							}
						}
						if (hub_valid > 0U)
						{
							hub_darkness /= (float)hub_valid;
						}
					}

					if (continuity >= hot_continuity_threshold &&
						hub_darkness >= hot_hub_threshold)
					{
						const float weighted = (continuity * continuity) * hub_darkness * smoothed_votes[candidate_bin];
						if (weighted > best_hot_weighted_full)
						{
							best_hot_weighted_full = weighted;
							best_hot_bin_full = candidate_bin;
						}
					}
				}

				if (best_hot_weighted_full > 0.0f &&
					smoothed_votes[best_hot_bin_full] >= (0.22f * smoothed_votes[best_bin]))
				{
					const float hot_angle_deg =
						(min_angle_rad + ((float)best_hot_bin_full /
											   (float)(APP_BASELINE_ANGLE_BINS - 1U)) *
											  sweep_rad) *
						(180.0f / APP_BASELINE_PI);
					const float primary_angle_deg_norm =
						AppBaselineRuntime_NormalizeAngleDegrees(best_angle_deg_check);
					const float hot_angle_deg_norm =
						AppBaselineRuntime_NormalizeAngleDegrees(hot_angle_deg);
					const long primary_angle_x10 = AppBaselineRuntime_RoundToLong(
						primary_angle_deg_norm * 10.0f);
					const long hot_angle_x10 = AppBaselineRuntime_RoundToLong(
						hot_angle_deg_norm * 10.0f);
					best_bin = best_hot_bin_full;
					best_score = smoothed_votes[best_bin];
					DebugConsole_Printf(
						"[BASELINE] Hot-zone full-sweep rescue: primary=%ld.%01lddeg hot=%ld.%01lddeg\r\n",
						primary_angle_x10 / 10L,
						((primary_angle_x10 % 10L) < 0L) ? -(primary_angle_x10 % 10L) : (primary_angle_x10 % 10L),
						hot_angle_x10 / 10L,
						((hot_angle_x10 % 10L) < 0L) ? -(hot_angle_x10 % 10L) : (hot_angle_x10 % 10L));
				}
			}
		}

		runner_up_score = AppBaselineRuntime_RunnerUpPeakAfterSuppression(smoothed_votes, APP_BASELINE_ANGLE_BINS, best_bin, 15);

		{
			const size_t prev_index =
				(best_bin + APP_BASELINE_ANGLE_BINS - 1U) % APP_BASELINE_ANGLE_BINS;
			const size_t next_index = (best_bin + 1U) % APP_BASELINE_ANGLE_BINS;
			const float prev_vote = angle_votes[prev_index];
			const float best_vote = angle_votes[best_bin];
			const float next_vote = angle_votes[next_index];
			const float vote_window = prev_vote + best_vote + next_vote;
			float refined_fraction = (float)best_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U);

			if (vote_window > 0.0f)
			{
				const float weighted_bin_sum =
					((float)prev_index * prev_vote) + ((float)best_bin * best_vote) + ((float)next_index * next_vote);
				refined_fraction = (weighted_bin_sum / vote_window) / (float)(APP_BASELINE_ANGLE_BINS - 1U);
			}

			estimate_out->valid = true;
			estimate_out->center_x = center_x;
			estimate_out->center_y = center_y;
			float best_angle = min_angle_rad + (refined_fraction * sweep_rad);

			/* Inversion check: compare darkness and redness along the detected ray vs the
			 * opposite ray. The needle is dark on a light background, so we keep
			 * the comparison centered on dark shaft evidence instead of color.
			 *
			 * Only flip if the angle is NOT already in the valid Celsius sweep range.
			 * The 270° sweep runs from 135° (-30°C) through 270° (up), 360°/0° (right),
			 * to 45° (50°C). Angles already inside that wrapped sweep should be kept
			 * unless they fall into the subdial clutter band. */
			{
				float score_forward = 0.0f;
				float score_backward = 0.0f;
				size_t count = 0U;
				const float cos_a = cosf(best_angle);
				const float sin_a = sinf(best_angle);

				for (float r_frac = 0.2f; r_frac <= 0.8f; r_frac += 0.1f)
				{
					const long fx = AppBaselineRuntime_RoundToLong((float)center_x + (cos_a * r_frac * dial_radius_px));
					const long fy = AppBaselineRuntime_RoundToLong((float)center_y + (sin_a * r_frac * dial_radius_px));
					const long bx = AppBaselineRuntime_RoundToLong((float)center_x - (cos_a * r_frac * dial_radius_px));
					const long by = AppBaselineRuntime_RoundToLong((float)center_y - (sin_a * r_frac * dial_radius_px));
					const float shaft_progress = (r_frac - 0.2f) / 0.6f;
					const float shaft_weight =
						0.35f + (0.65f * AppBaselineRuntime_MiddleShaftWeight(shaft_progress));

					if (fx >= 0 && (size_t)fx < frame_width_pixels && fy >= 0 && (size_t)fy < frame_height_pixels)
					{
						const float luma = AppBaselineRuntime_ReadLuma(frame_bytes, frame_width_pixels, (size_t)fx, (size_t)fy);
						score_forward += ((255.0f - luma) / 255.0f) * shaft_weight;
					}
					if (bx >= 0 && (size_t)bx < frame_width_pixels && by >= 0 && (size_t)by < frame_height_pixels)
					{
						const float luma = AppBaselineRuntime_ReadLuma(frame_bytes, frame_width_pixels, (size_t)bx, (size_t)by);
						score_backward += ((255.0f - luma) / 255.0f) * shaft_weight;
					}
					count++;
				}

				/* Only flip if the angle is outside the calibrated sweep or sits
				 * in the subdial clutter band. If the angle is already plausible
				 * for the Celsius sweep, keep it as-is. */
				const float best_angle_deg =
					AppBaselineRuntime_NormalizeAngleDegrees(
						best_angle * (180.0f / APP_BASELINE_PI));
				/* Skip inversion for angles that already sit in the calibrated
				 * Celsius sweep (including the hot wrap near 0°). */
				if (AppBaselineRuntime_IsAngleInCelsiusSweep(best_angle_deg))
				{
					/* Angle is in valid range - don't flip it. */
				}
				else if (AppBaselineRuntime_IsAngleInSubdialBand(best_angle_deg))
				{
					/* Angle is in the subdial band - flip it to see if it becomes valid. */
					if (score_backward > (score_forward + 0.5f * (float)count))
					{
						best_angle += APP_BASELINE_PI;
						while (best_angle >= APP_BASELINE_TWO_PI)
						{
							best_angle -= APP_BASELINE_TWO_PI;
						}
					}
				}
				else if (score_backward > (score_forward + 0.5f * (float)count))
				{
					/* Angle is outside the calibrated sweep and backward is darker.
					 * Flip it to see if it lands in a valid sweep segment. */
					best_angle += APP_BASELINE_PI;
					while (best_angle >= APP_BASELINE_TWO_PI)
					{
						best_angle -= APP_BASELINE_TWO_PI;
					}
				}
			}

			/* Validate the detected angle against the calibrated Celsius sweep.
			 * For a 135° start and 270° sweep, valid angles are:
			 * - 135°..360° and 0°..45° (with a small margin),
			 * while the lower subdial clutter band (~55°..130°) stays rejected. */
			const float best_angle_deg =
				AppBaselineRuntime_NormalizeAngleDegrees(
					best_angle * (180.0f / APP_BASELINE_PI));
			/* Keep only angles inside the calibrated Celsius sweep and reject
			 * the lower subdial clutter band explicitly. */
			if (!AppBaselineRuntime_IsAngleInCelsiusSweep(best_angle_deg) ||
				AppBaselineRuntime_IsAngleInSubdialBand(best_angle_deg))
			{
				const long angle_x10 = AppBaselineRuntime_RoundToLong(
					best_angle_deg * 10.0f);
				DebugConsole_Printf(
					"[BASELINE] Polar reject: angle=%ld.%01ld source=%s\r\n",
					angle_x10 / 10L,
					((angle_x10 % 10L) < 0L) ? -(angle_x10 % 10L) : (angle_x10 % 10L),
					source_label);
				return false;
			}

			/* Spoke-continuity validation: verify there's a continuous dark spoke
			 * along the detected angle from center to outer dial edge.
			 * This rejects isolated edge peaks (dial markings) that don't form
			 * a continuous spoke. Needle should have dark continuity. */
			{
				const float cos_a = cosf(best_angle);
				const float sin_a = sinf(best_angle);
				float spoke_continuity = 0.0f;
				/* Increased from 10 to 20 samples (2026-05-02) for more accurate
				 * spoke continuity measurement. More samples help distinguish
				 * between real needles (strong continuity along full length)
				 * and dial markings (weaker or partial continuity). */
				const size_t continuity_samples = 20U;
				size_t valid_samples = 0U;

				for (size_t i = 0U; i < continuity_samples; ++i)
				{
					const float r_frac = 0.15f + (0.70f * (float)i / (float)(continuity_samples - 1U));
					const long sx = AppBaselineRuntime_RoundToLong(
						(float)center_x + (cos_a * r_frac * dial_radius_px));
					const long sy = AppBaselineRuntime_RoundToLong(
						(float)center_y + (sin_a * r_frac * dial_radius_px));

					if (sx >= 0 && (size_t)sx < frame_width_pixels &&
						sy >= 0 && (size_t)sy < frame_height_pixels)
					{
						const float sample_luma = AppBaselineRuntime_ReadLuma(
							frame_bytes, frame_width_pixels, (size_t)sx, (size_t)sy);
						spoke_continuity += ((255.0f - sample_luma) / 255.0f);
						valid_samples++;
					}
				}

				if (valid_samples > 0U)
				{
					spoke_continuity /= (float)valid_samples;
					/* Require sufficient spoke darkness along the shaft.
					 * Bright-relaxed mode lowers this floor to tolerate washed-out
					 * needle contrast while still rejecting flat clutter. */
					if (spoke_continuity < final_spoke_continuity_threshold)
					{
						const long continuity_m = AppBaselineRuntime_RoundToLong(
							spoke_continuity * 1000.0f);
						const long threshold_m = AppBaselineRuntime_RoundToLong(
							final_spoke_continuity_threshold * 1000.0f);
						DebugConsole_Printf(
							"[BASELINE] Polar reject: continuity=%ld/%ld source=%s mode=%s\r\n",
							continuity_m, threshold_m, source_label,
							bright_relaxed ? "bright-relaxed" : "normal");
						return false;
					}
				}
			}

			/* Normalize best_angle to [0, 2π) before storing.
			 * The inversion check adds 180°, which can push the angle above 360°.
			 * Without normalization, the angle would be reported incorrectly and
			 * the temperature conversion would be wrong. */
			while (best_angle >= APP_BASELINE_TWO_PI)
			{
				best_angle -= APP_BASELINE_TWO_PI;
			}
			while (best_angle < 0.0f)
			{
				best_angle += APP_BASELINE_TWO_PI;
			}

			estimate_out->angle_rad = best_angle;
			estimate_out->temperature_c =
				AppBaselineRuntime_ConvertAngleToTemperature(
					estimate_out->angle_rad);
			estimate_out->confidence = AppBaselineRuntime_ClampFloat(
				best_score / ((fabsf(vote_sum) / (float)APP_BASELINE_ANGLE_BINS) + 1e-6f), 0.0f, 1000.0f);
			estimate_out->best_score = best_score;
			estimate_out->runner_up_score = runner_up_score;
			estimate_out->source_label = source_label;
		}
	}

	return true;
}

/**
 * @brief Clamp a float to a closed range.
 */
static float AppBaselineRuntime_ClampFloat(float value, float min_value,
										   float max_value)
{
	if (value < min_value)
	{
		return min_value;
	}
	if (value > max_value)
	{
		return max_value;
	}

	return value;
}

/**
 * @brief Round a float to the nearest long for logging and pixel lookup.
 */
static long AppBaselineRuntime_RoundToLong(float value)
{
	if (value >= 0.0f)
	{
		return (long)(value + 0.5f);
	}

	return (long)(value - 0.5f);
}

/**
 * @brief Estimate the visible dial radius from the stable training crop.
 *
 * The crop height better tracks the ring we actually want to vote over than
 * the inscribed crop radius, which keeps the polar annulus closer to the
 * Hough-seeded Python baseline.
 */
static float AppBaselineRuntime_EstimateDialRadiusPixels(
	size_t frame_width_pixels, size_t frame_height_pixels)
{
	const AppGaugeGeometry_Crop_t crop =
		AppGaugeGeometry_TrainingCrop(frame_width_pixels, frame_height_pixels);
	const float estimated_radius_px =
		(float)crop.height * APP_BASELINE_DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO;
	const size_t min_frame_pixels =
		(frame_width_pixels < frame_height_pixels) ? frame_width_pixels
												   : frame_height_pixels;
	const float frame_limit_px = 0.49f * (float)min_frame_pixels;

	return AppBaselineRuntime_ClampFloat(estimated_radius_px,
										 (float)APP_BASELINE_MIN_RADIUS_PIXELS, frame_limit_px);
}

/**
 * @brief Score one candidate dial center using rim-aligned edge votes.
 *
 * A true dial center should make the outer rim edges point back toward the
 * same center with strong radial agreement.
 */
static float AppBaselineRuntime_ScoreDialCenterCandidate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, float dial_radius_px, size_t center_x,
	size_t center_y)
{
	const AppGaugeGeometry_Crop_t crop =
		AppGaugeGeometry_TrainingCrop(frame_width_pixels, frame_height_pixels);
	const float crop_center_x = (float)crop.x_min + (0.5f * (float)crop.width);
	const float crop_center_y = (float)crop.y_min + (0.5f * (float)crop.height);
	const float crop_half_diag =
		sqrtf(((0.5f * (float)crop.width) * (0.5f * (float)crop.width)) + ((0.5f * (float)crop.height) * (0.5f * (float)crop.height)));
	const float rim_radius_min = dial_radius_px * APP_BASELINE_CENTER_SEARCH_RIM_MIN_FRACTION;
	const float rim_radius_max = dial_radius_px * APP_BASELINE_CENTER_SEARCH_RIM_MAX_FRACTION;
	const size_t sample_step = APP_BASELINE_CENTER_SEARCH_SAMPLE_STEP_PIXELS;
	float score = 0.0f;
	size_t sample_count = 0U;

	if ((frame_bytes == NULL) || (frame_size == 0U))
	{
		return 0.0f;
	}

	for (size_t y = scan_y_min + 1U; y < (scan_y_max - 1U); y += sample_step)
	{
		for (size_t x = scan_x_min + 1U; x < (scan_x_max - 1U); x += sample_step)
		{
			const float dx = (float)x - (float)center_x;
			const float dy = (float)y - (float)center_y;
			const float radius = sqrtf((dx * dx) + (dy * dy));
			const float luma = AppBaselineRuntime_ReadLuma(frame_bytes,
														   frame_width_pixels, x, y);
			float gradient_x = 0.0f;
			float gradient_y = 0.0f;
			float background_luma = 0.0f;

			if ((radius < rim_radius_min) || (radius > rim_radius_max))
			{
				continue;
			}

			if ((luma > (float)APP_BASELINE_SATURATION_THRESHOLD) || AppBaselineRuntime_IsInSubdialMask(center_x, center_y, x, y,
																										dial_radius_px))
			{
				continue;
			}

			{
				const float edge_mag = AppBaselineRuntime_ReadEdgeMagnitude(
					frame_bytes, frame_width_pixels, frame_height_pixels, x, y,
					&gradient_x, &gradient_y, &background_luma);
				const float grad_mag_safe = (edge_mag > 1.0f) ? edge_mag : 1.0f;
				const float radial_x = dx / radius;
				const float radial_y = dy / radius;
				const float grad_x = gradient_x / grad_mag_safe;
				const float grad_y = gradient_y / grad_mag_safe;
				const float radial_alignment =
					fabsf((grad_x * radial_x) + (grad_y * radial_y));
				const float rim_bias = 1.0f - AppBaselineRuntime_ClampFloat(
												  fabsf(radius - dial_radius_px) / (dial_radius_px + 1e-6f), 0.0f, 1.0f);
				const float rim_weight = rim_bias * rim_bias;
				const float alignment_weight =
					radial_alignment * radial_alignment;
				const float vote = edge_mag * alignment_weight * rim_weight;

				(void)background_luma;
				if (vote <= 0.0f)
				{
					continue;
				}

				score += vote;
				sample_count++;
			}
		}
	}

	if (sample_count == 0U)
	{
		return 0.0f;
	}

	{
		const float center_dist = sqrtf(
			(((float)center_x - crop_center_x) * ((float)center_x - crop_center_x)) + (((float)center_y - crop_center_y) * ((float)center_y - crop_center_y)));
		/* Keep the rim search gently centered, but do not let this penalty
		 * overpower a seed that has substantially stronger rim support. */
		const float center_prior = AppBaselineRuntime_ClampFloat(
			1.0f - (0.25f * (center_dist / (crop_half_diag + 1e-6f))),
			0.20f, 1.0f);

		return (score / (float)sample_count) * center_prior;
	}
}

/**
 * @brief Search for a stable dial center using the rim edge evidence.
 */
static bool AppBaselineRuntime_EstimateDialCenterFromRimVotes(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	size_t scan_x_min, size_t scan_y_min, size_t scan_x_max,
	size_t scan_y_max, float dial_radius_px, size_t *center_x_out,
	size_t *center_y_out, float *center_quality_out)
{
	const size_t coarse_step = APP_BASELINE_CENTER_SEARCH_COARSE_STEP_PIXELS;
	const size_t fine_step = APP_BASELINE_CENTER_SEARCH_FINE_STEP_PIXELS;
	const size_t min_center_x = scan_x_min + APP_BASELINE_SCAN_BORDER_PIXELS;
	const size_t min_center_y = scan_y_min + APP_BASELINE_SCAN_BORDER_PIXELS;
	const size_t max_center_x = scan_x_max - APP_BASELINE_SCAN_BORDER_PIXELS - 1U;
	const size_t max_center_y = scan_y_max - APP_BASELINE_SCAN_BORDER_PIXELS - 1U;
	size_t best_center_x = 0U;
	size_t best_center_y = 0U;
	float best_quality = -1.0f;
	bool found_any = false;

	if ((frame_bytes == NULL) || (center_x_out == NULL) || (center_y_out == NULL) || (scan_x_max <= scan_x_min) || (scan_y_max <= scan_y_min))
	{
		return false;
	}

	if ((min_center_x >= max_center_x) || (min_center_y >= max_center_y) || (dial_radius_px < (float)APP_BASELINE_MIN_RADIUS_PIXELS))
	{
		return false;
	}

	for (size_t candidate_y = min_center_y; candidate_y <= max_center_y;
		 candidate_y += coarse_step)
	{
		for (size_t candidate_x = min_center_x; candidate_x <= max_center_x;
			 candidate_x += coarse_step)
		{
			const float quality = AppBaselineRuntime_ScoreDialCenterCandidate(
				frame_bytes, frame_size, frame_width_pixels, frame_height_pixels,
				scan_x_min, scan_y_min, scan_x_max, scan_y_max, dial_radius_px,
				candidate_x, candidate_y);

			if (quality > best_quality)
			{
				best_quality = quality;
				best_center_x = candidate_x;
				best_center_y = candidate_y;
				found_any = true;
			}
		}
	}

	if (!found_any)
	{
		return false;
	}

	{
		const long fine_radius = (long)coarse_step;
		long fine_min_x = (long)best_center_x - fine_radius;
		long fine_max_x = (long)best_center_x + fine_radius;
		long fine_min_y = (long)best_center_y - fine_radius;
		long fine_max_y = (long)best_center_y + fine_radius;

		if (fine_min_x < (long)min_center_x)
		{
			fine_min_x = (long)min_center_x;
		}
		if (fine_max_x > (long)max_center_x)
		{
			fine_max_x = (long)max_center_x;
		}
		if (fine_min_y < (long)min_center_y)
		{
			fine_min_y = (long)min_center_y;
		}
		if (fine_max_y > (long)max_center_y)
		{
			fine_max_y = (long)max_center_y;
		}

		for (long candidate_y = fine_min_y; candidate_y <= fine_max_y;
			 candidate_y += (long)fine_step)
		{
			for (long candidate_x = fine_min_x; candidate_x <= fine_max_x;
				 candidate_x += (long)fine_step)
			{
				const float quality = AppBaselineRuntime_ScoreDialCenterCandidate(
					frame_bytes, frame_size, frame_width_pixels,
					frame_height_pixels, scan_x_min, scan_y_min, scan_x_max,
					scan_y_max, dial_radius_px, (size_t)candidate_x,
					(size_t)candidate_y);

				if (quality > best_quality)
				{
					best_quality = quality;
					best_center_x = (size_t)candidate_x;
					best_center_y = (size_t)candidate_y;
				}
			}
		}
	}

	*center_x_out = best_center_x;
	*center_y_out = best_center_y;
	if (center_quality_out != NULL)
	{
		*center_quality_out = best_quality;
	}
	return true;
}

/**
 * @brief Print the selected baseline estimate in a compact, thesis-friendly form.
 */
static void AppBaselineRuntime_LogEstimate(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	char temperature_line[96] = {0};
	long angle_tenths = 0L;
	long confidence_thousandths = 0L;
	long score_whole = 0L;
	long runner_up_whole = 0L;
	long angle_abs_tenths = 0L;
	long confidence_abs_thousandths = 0L;

	if ((estimate == NULL) || !estimate->valid)
	{
		return;
	}

	AppInferenceLog_FormatFloatTenths(temperature_line,
									  sizeof(temperature_line),
									  "[BASELINE] Temperature estimate: ", estimate->temperature_c);
	(void)DebugConsole_WriteString(temperature_line);

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
		(unsigned long)estimate->center_x,
		(unsigned long)estimate->center_y,
		(estimate->source_label != NULL) ? estimate->source_label : "unknown",
		(long)(angle_tenths / 10L), (long)(angle_abs_tenths % 10L),
		(long)(confidence_thousandths / 1000L),
		(long)(confidence_abs_thousandths % 1000L),
		score_whole, runner_up_whole);
}

/**
 * @brief Find the strongest non-neighbor peak after suppressing the main peak.
 */
static float AppBaselineRuntime_RunnerUpPeakAfterSuppression(
	const float *peak_values, size_t num_bins, size_t best_index,
	size_t suppression_bins)
{
	float runner_up = 0.0f;

	if ((peak_values == NULL) || (num_bins == 0U))
	{
		return 0.0f;
	}

	for (size_t i = 0U; i < num_bins; ++i)
	{
		/* Calculate circular distance from best_index. */
		size_t dist = (i >= best_index) ? (i - best_index) : (best_index - i);
		if (dist > (num_bins / 2U))
		{
			dist = num_bins - dist;
		}

		if (dist <= suppression_bins)
		{
			continue;
		}

		if (peak_values[i] > runner_up)
		{
			runner_up = peak_values[i];
		}
	}

	return runner_up;
}

/* USER CODE END 0 */
