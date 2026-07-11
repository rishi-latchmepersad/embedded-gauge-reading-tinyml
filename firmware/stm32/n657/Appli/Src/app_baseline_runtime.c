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
#include "app_ai_config.h"
#include "app_baseline_hough.h"
#include "app_baseline_template.h"
#include "app_gauge_geometry.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_threadx_config.h"
/* The baseline detector remains enabled, but its verbose UART diagnostics are
 * disabled in production so they do not drown out the AI and camera logs. */
#define DEBUG_CONSOLE_ENABLE_LOGS 0
#include "debug_console.h"
#include "ina219_power.h"
#include "inference_metrics.h"
#include "threadx_utils.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */


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
/* Temporary board-fit calibration shift (2026-07-05).
 * The live OBB -> UNet path is decoding a stable needle angle, but the
 * shared angle-to-temp mapping was still landing a few degrees too warm at
 * the cold end. Shift the anchor toward the cold side so a true -30 C needle
 * resolves closer to the expected temperature before we make any larger
 * calibration-table changes. */
#define APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG (9.72f)
/* Second-stage board-fit calibration gain.
 * The offset alone fixed the cold anchor, but the hot end still read low.
 * Preserve the -30 C anchor and stretch the scale so a 35 C readout maps
 * back to the expected 40 C point. */
#define APP_BASELINE_TEMPERATURE_CALIBRATION_PIVOT_C (-30.0f)
#define APP_BASELINE_TEMPERATURE_CALIBRATION_GAIN 1.0769231f
/* Endpoint calibration anchors for the current board profile.
 * The logged hot-end capture at raw 148.21 deg corresponds to a calibrated
 * 157.93 deg point once the board offset is applied. The cold-end capture at
 * raw 33.22 deg closes the physical sweep. Keeping only the endpoints makes
 * every intermediate temperature interpolate linearly and consistently. */
#define APP_BASELINE_PROFILE_BOARD_COLD_ANCHOR_RAW_DEG 33.22f
#define APP_BASELINE_PROFILE_BOARD_COLD_ANCHOR_CALIBRATED_DEG \
	(APP_BASELINE_PROFILE_BOARD_COLD_ANCHOR_RAW_DEG + \
	 APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG)
#define APP_BASELINE_PROFILE_BOARD_HOT_ANCHOR_RAW_DEG 148.21f
#define APP_BASELINE_PROFILE_BOARD_HOT_ANCHOR_CALIBRATED_DEG \
	(APP_BASELINE_PROFILE_BOARD_HOT_ANCHOR_RAW_DEG + \
	 APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG)
/* Default calibration profile for the current board-gauge pairing.
 * Other gauges can define their own profiles and install them at runtime. */
const AppBaselineRuntime_CalibrationProfile_t AppBaselineRuntime_DefaultCalibrationProfile = {
	.profile_name = "board_celsius_v1",
	.angle_offset_deg = APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG,
	.temperature_pivot_c = APP_BASELINE_TEMPERATURE_CALIBRATION_PIVOT_C,
	.temperature_gain = APP_BASELINE_TEMPERATURE_CALIBRATION_GAIN,
	.calibration_point_count = 2U,
	.calibration_points = {
		{
			.angle_deg = APP_BASELINE_PROFILE_BOARD_HOT_ANCHOR_CALIBRATED_DEG,
			.temperature_c = 50.0f,
		},
		{
			.angle_deg = APP_BASELINE_PROFILE_BOARD_COLD_ANCHOR_CALIBRATED_DEG,
			.temperature_c = -30.0f,
		},
	},
};
/* Profile registry used for named selection at boot.
 * Add one entry per gauge family so the conversion layer can scale without
 * re-compiling the shared needle-angle math. */
static const AppBaselineRuntime_CalibrationProfile_t
	*const camera_baseline_calibration_profiles[] = {
		&AppBaselineRuntime_DefaultCalibrationProfile,
};
#define APP_BASELINE_BRIGHT_THRESHOLD 150U
/* Bright centroid adapts to gauge position. Use a wide threshold so the
 * baseline works when the gauge moves in the frame. */
#define APP_BASELINE_BRIGHT_CENTER_MAX_DRIFT_PIXELS 50.0f
/* The classical contrast scorer is not trustworthy on the dark frames seen
 * after the camera's final exposure retry. Fail closed instead of publishing
 * a boundary peak or replaying stale history. */
#define APP_BASELINE_MIN_FRAME_MEAN_LUMA 100.0f
/* Pixels above this luma are considered saturated/glare and excluded from
 * the bright-centroid calculation and ray scoring.
 * Raised from 220 to 235 (2026-04-30) — the 220 threshold was too aggressive
 * on bright board captures where the dial face is well-lit but not blown out,
 * causing valid needle pixels to be skipped and the detector to miss entirely. */
#define APP_BASELINE_SATURATION_THRESHOLD 235U
/* Keep the minimum bright-pixel floor at roughly the same ~2% of frame area
 * that the 224x224 capture budget uses, so the gate scales with frame size
 * instead of hard-coding a pixel count. */
#define APP_BASELINE_MIN_BRIGHT_PIXELS \
	(((CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_HEIGHT_PIXELS) + 24U) / 49U)
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
/* The editable baseline diagram gates weak or ambiguous hypotheses before
 * they enter the short history. Keep the original 1.25 confidence floor so
 * a broad dial-artifact peak cannot become the new baseline state. */
#define APP_BASELINE_CONFIDENCE_THRESHOLD 1.25f
/* Even a strong peak is not trustworthy if the runner-up is almost tied.
 * Clean ideal captures often have broader but still correct peaks, so keep
 * this close to the Python baseline's permissive gate.
 * At extreme temperatures (-30C) the needle sits at the sweep edge where
 * the background gradient can out-vote the needle. Lower the ratio so the
 * best peak still wins even when runner_up > best_score. Board log at -30C
 * shows pr=582-617 being rejected, so we need this well below 0.58. */
#define APP_BASELINE_MIN_PEAK_RATIO 0.35f
/* Extra trace for classical selector debugging. Keep enabled while we chase
 * center drift so we can see which gate or comparison is steering the read. */
#define APP_BASELINE_DEBUG_SELECTION 1U
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
/* Angle agreement is a better signal than temperature agreement for a spoke
 * detector because two estimates can land near the same temperature while
 * still belonging to different angular families after calibration. */
#define APP_BASELINE_CONSENSUS_ANGLE_DELTA_DEG 8.0f
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
 * needle instead of freezing on the first plausible anchor.
 * DISABLED: the refinement finds false peaks at offset centers. */
#define APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP 0U
/* The independent all-angle darkness rescue uses a different score scale than
 * the polar vote. Mixing its winner with the polar runner-up made every
 * confidence/ratio gate reject the candidate, so keep one score family live. */
#define APP_BASELINE_ENABLE_GLOBAL_SCORE_RESCUE 0U
/* Require a minimum amount of absolute vote support before we let a brand-new
 * baseline estimate seed the history. The hard-case sweep showed that the
 * gradient-polar detector is already the best classical family, so this floor
 * only needs to reject obvious noise, not the normal hard-frame range. */
#define APP_BASELINE_MIN_ACCEPT_SCORE 2.0f
/* Template matches need a stronger margin than polar candidates. A score of
 * 3 versus 2 is technically above the generic gate but is visibly ambiguous
 * and was publishing a false 0 C result on the board. */
#define APP_BASELINE_TEMPLATE_MIN_SCORE 8.0f
#define APP_BASELINE_TEMPLATE_MIN_PEAK_RATIO 2.0f
/* Let a candidate win on center proximity when it is meaningfully closer to
 * the dial center and still has comparable quality. */
#define APP_BASELINE_CENTER_DISTANCE_PREFER_DELTA_PIXELS 10.0f
#define APP_BASELINE_CENTER_DISTANCE_MIN_QUALITY_RATIO 0.92f
/* Continuity/Hough refinement may resolve a near-tie, but it must not replace
 * the strongest polar spoke with a much weaker candidate. */
#define APP_BASELINE_MIN_REFINED_SUPPORT_RATIO 0.75f
/* The face-center ray rescue was selecting dial-artifact peaks in the live
 * trace (about 148°, 0°, 358°, and 26°) even while the AI stayed near 258°.
 * Keep the primary polar vote authoritative until a separate offline replay
 * proves that this second score family improves the same frames. */
#define APP_BASELINE_ENABLE_FACE_RAY_RESCUE 0U
/* Keep a tiny history so the baseline can report a stable rough reading
 * instead of jumping frame-to-frame on glare or digit clutter. */
#define APP_BASELINE_ESTIMATE_HISTORY_SIZE 3U
/* If the scene jumps a lot between captures, drop the history and re-lock to
 * the new setpoint quickly rather than averaging across two different temps. */
#define APP_BASELINE_HISTORY_RESET_DELTA_C 12.0f
/* Scale factor for integer-encoded confidence in log output (avoids %f). */
#define APP_BASELINE_CONFIDENCE_LOG_SCALE 1000L
/* Hot-zone override is useful under heavy overexposure, but in normal frames
 * it can hijack cold readings by repeatedly snapping to the hot wrap edge. */
#define APP_BASELINE_HOT_OVERRIDE_ENABLE_IN_NORMAL_MODE 0U
/* Keep the hot-zone rescue available for offline replay, but disable it on the
 * live board: the 20-75 degree artwork band repeatedly beats the true needle
 * near 260 degrees and produces false endpoint temperatures. */
#define APP_BASELINE_ENABLE_HOT_ZONE_RESCUE 0U
/* Hysteresis guard: when recent stable history is cold, require very strong
 * hot vote dominance before allowing a warm/hot override. */
#define APP_BASELINE_HOT_OVERRIDE_COLD_HISTORY_TEMP_C 0.0f
#define APP_BASELINE_HOT_OVERRIDE_WARM_TARGET_TEMP_C 25.0f
#define APP_BASELINE_HOT_OVERRIDE_COLD_HISTORY_MIN_RATIO 0.90f
/* When the primary peak maps to a cold temperature (≤ 15°C), require a
 * higher vote ratio from the hot-zone candidate before overriding. This
 * prevents hot-zone override from replacing a genuine cold needle (e.g.
 * 258° / 6°C) with a false peak from dial artwork near 20° / 40°C.
 * At cold primary temps the needle sits in the mid-range where gradient
 * signal is strong, so the primary vote is generally correct. */
#define APP_BASELINE_HOT_OVERRIDE_COLD_PRIMARY_THRESHOLD_C 15.0f
#define APP_BASELINE_HOT_OVERRIDE_COLD_VOTE_RATIO 0.50f
/* Recovery guard: allow cold estimates to break out of poisoned warm history
 * in normal frames when the jump direction is warm->cold and confidence is
 * still reasonable for this baseline. */
#define APP_BASELINE_COLD_RECOVERY_HISTORY_TEMP_C 20.0f
#define APP_BASELINE_COLD_RECOVERY_TARGET_TEMP_C 0.0f
#define APP_BASELINE_COLD_RECOVERY_MIN_CONFIDENCE 2.5f
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

static volatile uint64_t camera_baseline_request_capture_time_us = 0ULL;

static volatile ULONG camera_baseline_request_generation = 0U;
static AppBaselineRuntime_Estimate_t camera_baseline_estimate_history
	[APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {0};
static size_t camera_baseline_estimate_history_count = 0U;
static size_t camera_baseline_estimate_history_next_index = 0U;
/* Per-frame brightness profile used to adapt detector thresholds under
 * heavy overexposure without globally weakening the baseline gate. */
static bool camera_baseline_current_frame_is_bright = false;
static float camera_baseline_current_frame_mean_luma = 0.0f;
static float camera_baseline_current_frame_bright_ratio = 0.0f;
/* Last-estimate state shared between the worker thread and other modules. */
static bool camera_baseline_last_result_valid = false;
static float camera_baseline_last_temperature_c = 0.0f;
static float camera_baseline_last_angle_rad = 0.0f;
static float camera_baseline_last_confidence = 0.0f;
static volatile ULONG camera_baseline_last_result_generation = 0U;
/* Guard for one-time initialisation of the baseline subsystem. */
static bool app_baseline_runtime_initialized = false;
/* Active gauge calibration profile. Kept as a pointer so the board can swap
 * profiles at runtime without rebuilding the shared decode path. */
static const AppBaselineRuntime_CalibrationProfile_t
	*camera_baseline_active_calibration_profile =
	&AppBaselineRuntime_DefaultCalibrationProfile;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static VOID CameraBaselineThread_Entry(ULONG thread_input);
static void AppBaselineRuntime_WriteDirectStatus(const char *text);
static void AppBaselineRuntime_WriteDirectQueueStatus(const char *event,
										  ULONG generation,
										  ULONG frame_length);
static void AppBaselineRuntime_WriteDirectGateStatus(
	const char *reason, const AppBaselineRuntime_Estimate_t *estimate);
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
static float AppBaselineRuntime_ComputeCenterDistancePixels(
	const AppBaselineRuntime_Estimate_t *estimate);
static float AppBaselineRuntime_MinAngleDistanceDegrees(
	float angle_a_deg, float angle_b_deg);
static bool AppBaselineRuntime_IsBetterEstimate(
	const AppBaselineRuntime_Estimate_t *candidate,
	const AppBaselineRuntime_Estimate_t *incumbent);
static void AppBaselineRuntime_LogCandidateSummary(
	const char *slot_name, bool candidate_ok,
	const AppBaselineRuntime_Estimate_t *estimate);
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
static bool AppBaselineRuntime_PushEstimateHistory(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_IsStableEstimateForHistory(
	const AppBaselineRuntime_Estimate_t *estimate);
static bool AppBaselineRuntime_ConvertAnchoredAngleToTemperature(
	const AppBaselineRuntime_CalibrationProfile_t *profile,
	float calibrated_angle_deg, float *temperature_out);
static const AppBaselineRuntime_CalibrationProfile_t *
AppBaselineRuntime_FindCalibrationProfile(const char *profile_name);
static float AppBaselineRuntime_ComputeSweepCenterWeight(float angle_rad);
static float AppBaselineRuntime_ComputePeakPersistenceWeight(
	const float *peak_values, size_t num_bins, size_t peak_index,
	size_t neighborhood_bins);
static bool AppBaselineRuntime_SelectSmoothedEstimate(
	AppBaselineRuntime_Estimate_t *estimate_out);
float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad);
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
bool AppBaselineRuntime_EstimatePolarNeedle(
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
static void AppBaselineRuntime_StoreLastEstimate(
	const AppBaselineRuntime_Estimate_t *estimate);
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
bool AppBaselineRuntime_EstimateDialCenterFromRimVotes(
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
 * @brief Store the most recent accepted baseline estimate and bump its version.
 */
static void AppBaselineRuntime_StoreLastEstimate(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if ((estimate == NULL) || !estimate->valid)
	{
		return;
	}

	camera_baseline_last_result_valid = true;
	camera_baseline_last_temperature_c = estimate->temperature_c;
	camera_baseline_last_angle_rad = estimate->angle_rad;
	camera_baseline_last_confidence = estimate->confidence;
	camera_baseline_last_result_generation++;
}

/**
 * @brief Look up a calibration profile by its registered name.
 *
 * @param profile_name Name supplied by the board bootstrap code.
 * @retval Matching profile when the name is known.
 * @retval Default board profile when the name is NULL or unknown.
 */
static const AppBaselineRuntime_CalibrationProfile_t *
AppBaselineRuntime_FindCalibrationProfile(const char *profile_name)
{
	const size_t profile_count =
		sizeof(camera_baseline_calibration_profiles) /
		sizeof(camera_baseline_calibration_profiles[0]);

	if ((profile_name == NULL) || (profile_name[0] == '\0'))
	{
		return &AppBaselineRuntime_DefaultCalibrationProfile;
	}

	for (size_t index = 0U; index < profile_count; ++index)
	{
		const AppBaselineRuntime_CalibrationProfile_t *candidate =
			camera_baseline_calibration_profiles[index];

		if ((candidate != NULL) && (candidate->profile_name != NULL) &&
			(strcmp(candidate->profile_name, profile_name) == 0))
		{
			return candidate;
		}
	}

	return &AppBaselineRuntime_DefaultCalibrationProfile;
}

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
 * @brief Write a compact lifecycle marker without the verbose-log macro.
 * @param text Null-terminated marker text to send to the debug console.
 * @sideeffects Transmits one short marker if the console is initialized.
 */
static void AppBaselineRuntime_WriteDirectStatus(const char *text)
{
	if (text != NULL)
	{
		(void)DebugConsole_WriteBytes((const uint8_t *)text, strlen(text));
	}
}

/**
 * @brief Report a baseline queue transition with its frame metadata.
 * @param event Queue transition label, such as accepted or dequeued.
 * @param generation Monotonic baseline request generation.
 * @param frame_length Snapshot length associated with the request.
 * @sideeffects Formats and transmits one compact lifecycle marker.
 */
static void AppBaselineRuntime_WriteDirectQueueStatus(const char *event,
											  ULONG generation,
											  ULONG frame_length)
{
	char line[96];
	const int written = DebugConsole_Snprintf(
		line, sizeof(line), "[BASELINE][QUEUE] %s gen=%lu len=%lu\r\n",
		(event != NULL) ? event : "unknown", (unsigned long)generation,
		(unsigned long)frame_length);

	if (written > 0)
	{
		const size_t line_length = ((size_t)written < sizeof(line))
			? (size_t)written
			: (sizeof(line) - 1U);
		(void)DebugConsole_WriteBytes((const uint8_t *)line, line_length);
	}
}

/**
 * @brief Report the exact classical acceptance-gate rejection.
 * @param reason Gate condition that rejected the candidate.
 * @param estimate Candidate values used by the gate.
 * @sideeffects Transmits one compact diagnostic line.
 */
static void AppBaselineRuntime_WriteDirectGateStatus(
	const char *reason, const AppBaselineRuntime_Estimate_t *estimate)
{
	char line[192];
	const float peak_ratio =
		((estimate != NULL) && (estimate->runner_up_score > 0.0f))
			? (estimate->best_score / estimate->runner_up_score)
			: 0.0f;
	const int written = DebugConsole_Snprintf(
		line, sizeof(line),
		"[BASELINE][CV] gate=%s src=%s conf=%ld score=%ld ru=%ld ratio=%ld center=(%lu,%lu)\r\n",
		(reason != NULL) ? reason : "unknown",
		((estimate != NULL) && (estimate->source_label != NULL))
			? estimate->source_label : "unknown",
		(estimate != NULL) ? AppBaselineRuntime_RoundToLong(
			estimate->confidence * 1000.0f) : 0L,
		(estimate != NULL) ? AppBaselineRuntime_RoundToLong(
			estimate->best_score) : 0L,
		(estimate != NULL) ? AppBaselineRuntime_RoundToLong(
			estimate->runner_up_score) : 0L,
		AppBaselineRuntime_RoundToLong(peak_ratio * 1000.0f),
		(estimate != NULL) ? (unsigned long)estimate->center_x : 0UL,
		(estimate != NULL) ? (unsigned long)estimate->center_y : 0UL);

	if (written > 0)
	{
		const size_t line_length = ((size_t)written < sizeof(line))
			? (size_t)written : (sizeof(line) - 1U);
		(void)DebugConsole_WriteBytes((const uint8_t *)line, line_length);
	}
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
		AppBaselineRuntime_WriteDirectStatus(
			"[BASELINE][THREAD][BUILD] lifecycle-started v9\r\n");
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

	/* Start timing before the snapshot copy so the latency includes the full
	 * request-to-result path. */
	camera_baseline_request_capture_time_us = Metrics_GetMicros();
	Metrics_StartInference("BASELINE");
	(void)memcpy((void *)camera_inference_frame_snapshot, frame_ptr,
				 (size_t)frame_length);
	(void)memcpy(first8, camera_inference_frame_snapshot,
				 (size_t)((frame_length < 8U) ? frame_length : 8U));
	DebugConsole_Printf(
		"[BASELINE] Shared snapshot copied: src=%p dst=%p len=%lu first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
		(const void *)frame_ptr, (void *)camera_inference_frame_snapshot,
		(unsigned long)frame_length, first8[0], first8[1], first8[2],
		first8[3], first8[4], first8[5], first8[6], first8[7]);

	camera_baseline_request_frame_ptr = camera_inference_frame_snapshot;
	camera_baseline_request_frame_length = frame_length;
	camera_baseline_request_generation++;

	if (tx_semaphore_put(&camera_baseline_request_semaphore) != TX_SUCCESS)
	{
		Metrics_EndInference("BASELINE", NAN);
		DebugConsole_Printf(
			"[BASELINE] Failed to signal baseline request semaphore.\r\n");
		return false;
	}

	AppBaselineRuntime_WriteDirectQueueStatus(
		"accepted", camera_baseline_request_generation, frame_length);

	return true;
}

/* USER CODE END 0 */

/**
 * @brief Low-priority worker that turns each accepted frame into a temperature.
 */
static VOID CameraBaselineThread_Entry(ULONG thread_input)
{
	(void)thread_input;

	AppBaselineRuntime_WriteDirectStatus(
		"[BASELINE][THREAD] worker-entered\r\n");
	(void)DebugConsole_WriteString("[BASELINE] worker alive\r\n");

	while (1)
	{
		const UINT request_status = tx_semaphore_get(
			&camera_baseline_request_semaphore, TX_WAIT_FOREVER);
		const uint8_t *frame_ptr = NULL;
		ULONG frame_length = 0U;
		AppBaselineRuntime_Estimate_t estimate = {0};

		if (request_status != TX_SUCCESS)
		{
			continue;
		}

		AppBaselineRuntime_WriteDirectQueueStatus(
			"dequeued", camera_baseline_request_generation,
			camera_baseline_request_frame_length);

		frame_ptr = (const uint8_t *)camera_baseline_request_frame_ptr;
		frame_length = camera_baseline_request_frame_length;
		const ULONG request_generation = camera_baseline_request_generation;
		const uint64_t frame_capture_time_us = camera_baseline_request_capture_time_us;
		camera_baseline_request_frame_ptr = NULL;
		camera_baseline_request_frame_length = 0U;
		camera_baseline_request_capture_time_us = 0ULL;

		DebugConsole_Printf(
			"[BASELINE] worker dequeued frame gen=%lu len=%lu ptr=%p\r\n",
			(unsigned long)camera_baseline_request_generation,
			(unsigned long)frame_length, (const void *)frame_ptr);

		if ((frame_ptr == NULL) || (frame_length == 0U))
		{
			DebugConsole_Printf(
				"[BASELINE] Worker woke without a queued frame; ignoring.\r\n");
			continue;
		}

		/* Log pre-baseline power (idle/background) */
		(void)INA219_LogReading("BASELINE-PRE");

		/* Mark the start of actual compute so the metrics can separate queue
		 * wait from the baseline's processing time. */
		Metrics_MarkComputeStart("BASELINE");
		AppBaselineRuntime_WriteDirectQueueStatus(
			"compute-start", request_generation, frame_length);
		DebugConsole_Printf(
			"[BASELINE] estimate begin gen=%lu len=%lu\r\n",
			(unsigned long)request_generation,
			(unsigned long)frame_length);

		if (!AppBaselineRuntime_EstimateFromFrame(frame_ptr,
																  (size_t)frame_length, &estimate))
		{
			/* Fail closed: a stale value is still an inaccurate publication when
			 * the physical setpoint has moved or the previous geometry was wrong. */
			AppBaselineRuntime_WriteDirectQueueStatus(
				"estimate-failed", request_generation, frame_length);
			DebugConsole_Printf(
				"[BASELINE] Classical baseline failed to estimate a temperature.\r\n");
			Metrics_EndInference("BASELINE", NAN);
			continue;
		}

		/* Preserve the unsmoothed selector result in the direct UART trace so
		 * history labels cannot hide which geometry hypothesis actually won. */
		{
			char raw_geometry_line[192] = {0};
			const long raw_angle_tenths = AppBaselineRuntime_RoundToLong(
				(estimate.angle_rad * 180.0f / APP_BASELINE_PI) * 10.0f);
			const long raw_temperature_tenths = AppBaselineRuntime_RoundToLong(
				estimate.temperature_c * 10.0f);
			const long raw_angle_abs_tenths =
				(raw_angle_tenths < 0L) ? -raw_angle_tenths : raw_angle_tenths;
			const long raw_temperature_abs_tenths =
				(raw_temperature_tenths < 0L) ? -raw_temperature_tenths
													 : raw_temperature_tenths;
			const int raw_geometry_length = DebugConsole_Snprintf(
				raw_geometry_line, sizeof(raw_geometry_line),
				"[BASELINE][RAW] src=%s center=(%lu,%lu) angle=%ld.%01lddeg temp=%ld.%01ldC score=%ld ru=%ld\r\n",
				(estimate.source_label != NULL) ? estimate.source_label : "unknown",
				(unsigned long)estimate.center_x,
				(unsigned long)estimate.center_y,
				(long)(raw_angle_tenths / 10L),
				(long)(raw_angle_abs_tenths % 10L),
				(long)(raw_temperature_tenths / 10L),
				(long)(raw_temperature_abs_tenths % 10L),
				AppBaselineRuntime_RoundToLong(estimate.best_score),
				AppBaselineRuntime_RoundToLong(estimate.runner_up_score));
			if (raw_geometry_length > 0)
			{
				const size_t bytes_to_write =
					((size_t)raw_geometry_length < sizeof(raw_geometry_line))
						? (size_t)raw_geometry_length
						: (sizeof(raw_geometry_line) - 1U);
				(void)DebugConsole_WriteBytes(
					(const uint8_t *)raw_geometry_line, bytes_to_write);
			}
		}
		AppBaselineRuntime_WriteDirectQueueStatus(
			"estimate-ok", request_generation, frame_length);

		/* Push accepted geometry into the tiny median history so one-frame
		 * artwork/glare peaks do not become the published baseline. */
		if (!AppBaselineRuntime_PushEstimateHistory(&estimate))
		{
			/* Do not replace a weak current frame with an older history sample.
			 * That turns an uncertain frame into a confidently wrong reading. */
			AppBaselineRuntime_WriteDirectQueueStatus(
				"estimate-unstable", request_generation, frame_length);
			DebugConsole_WriteString(
				"[BASELINE] Current estimate was not stable; no history value published.\r\n");
			Metrics_EndInference("BASELINE", NAN);
			continue;
		}
		if (!AppBaselineRuntime_SelectSmoothedEstimate(&estimate))
		{
			AppBaselineRuntime_WriteDirectQueueStatus(
				"selection-failed", request_generation, frame_length);
			DebugConsole_Printf(
				"[BASELINE] Classical baseline produced an invalid raw estimate.\r\n");
			continue;
		}

		if (!estimate.valid)
		{
			AppBaselineRuntime_WriteDirectQueueStatus(
				"estimate-invalid", request_generation, frame_length);
			continue;
		}

		{
			const long angle_tenths = AppBaselineRuntime_RoundToLong(
				(estimate.angle_rad * 180.0f / APP_BASELINE_PI) * 10.0f);
			const long temperature_tenths = AppBaselineRuntime_RoundToLong(
				estimate.temperature_c * 10.0f);
			const long confidence_thousandths = AppBaselineRuntime_RoundToLong(
				estimate.confidence * 1000.0f);
			const long angle_abs_tenths =
				(angle_tenths < 0L) ? -angle_tenths : angle_tenths;
			const long temperature_abs_tenths =
				(temperature_tenths < 0L) ? -temperature_tenths
										 : temperature_tenths;
			const long confidence_abs_thousandths =
				(confidence_thousandths < 0L) ? -confidence_thousandths
											  : confidence_thousandths;
			DebugConsole_Printf(
				"[BASELINE] raw geometry: src=%s center=(%lu,%lu) needle=%ld.%01lddeg temp=%ld.%01ldC confidence=%ld.%03ld score=%ld runner_up=%ld\r\n",
				(estimate.source_label != NULL) ? estimate.source_label : "unknown",
				(unsigned long)estimate.center_x,
				(unsigned long)estimate.center_y,
				(long)(angle_tenths / 10L),
				(long)(angle_abs_tenths % 10L),
				(long)(temperature_tenths / 10L),
				(long)(temperature_abs_tenths % 10L),
				(long)(confidence_thousandths / 1000L),
				(long)(confidence_abs_thousandths % 1000L),
				AppBaselineRuntime_RoundToLong(estimate.best_score),
				AppBaselineRuntime_RoundToLong(estimate.runner_up_score));
		}

		AppBaselineRuntime_StoreLastEstimate(&estimate);
		Metrics_OverrideStartTime("BASELINE", frame_capture_time_us);
		AppBaselineRuntime_LogEstimate(&estimate);
		AppBaselineRuntime_WriteDirectQueueStatus(
			"published", request_generation, frame_length);
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

	/* The live path follows the editable process diagram: brightness profile,
	 * five center hypotheses, polar spoke voting, local refinement, consensus,
	 * and one final acceptance gate. Template labels are kept for offline
	 * evaluation only and are not allowed to publish a baseline reading. */
	if ((frame_bytes == NULL) || (estimate_out == NULL))
	{
		return false;
	}

	AppBaselineRuntime_UpdateFrameBrightnessProfile(frame_bytes, frame_size);

	if (camera_baseline_current_frame_mean_luma <
		APP_BASELINE_MIN_FRAME_MEAN_LUMA)
	{
		AppBaselineRuntime_WriteDirectStatus(
			"[BASELINE][CV] low-light-reject\r\n");
		return false;
	}

	if ((frame_bytes == NULL) || (estimate_out == NULL))
	{
		return false;
	}

	DebugConsole_Printf(
		"[BASELINE] estimate frame begin len=%lu\r\n",
		(unsigned long)frame_size);

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

	if (camera_baseline_current_frame_mean_luma <
		APP_BASELINE_MIN_FRAME_MEAN_LUMA)
	{
		AppBaselineRuntime_WriteDirectStatus(
			"[BASELINE][CV] low-light-reject\r\n");
		return false;
	}

	/*
	 * The board capture bank is a deterministic classical descriptor, not a
	 * learned model. Use it only when the nearest match is clearly separated;
	 * an ambiguous board frame continues through the polar path so its raw
	 * candidate can be diagnosed, while history refuses to hide instability.
	 * The grouped replay is 86.4% within 5 C, while the prior polar selector was
	 * repeatedly choosing the subdial or dial-artwork spoke on these frames.
	 */
	{
		AppBaselineRuntime_Estimate_t template_estimate = {0};
		const bool template_built = AppBaselineTemplate_Estimate(
				frame_bytes, frame_size, CAMERA_CAPTURE_WIDTH_PIXELS,
				CAMERA_CAPTURE_HEIGHT_PIXELS, &template_estimate);
		const float template_peak_ratio =
				(template_built && (template_estimate.runner_up_score > 0.0f))
					? (template_estimate.best_score /
						template_estimate.runner_up_score)
					: 0.0f;
		if (template_built
				&& (template_estimate.best_score >=
					APP_BASELINE_TEMPLATE_MIN_SCORE)
				&& (template_peak_ratio >= APP_BASELINE_TEMPLATE_MIN_PEAK_RATIO)
				&& AppBaselineRuntime_PassesAcceptanceGate(&template_estimate))
		{
			*estimate_out = template_estimate;
			AppBaselineRuntime_WriteDirectStatus(
					"[BASELINE][CV] template-match-accepted\r\n");
			return true;
		}
		if (template_built)
		{
			AppBaselineRuntime_WriteDirectStatus(
					"[BASELINE][CV] template-match-rejected-weak\r\n");
		}
	}

	/*
	 * The live capture is not always an exact bank member. Use the dynamic
	 * rim-center Hough detector as a FALLBACK only: the fixed-crop and
	 * bright-center hypotheses are more reliable for standard framing.
	 * The Hough detector is moved after the other hypotheses so it only
	 * wins when they all fail. */
	/* Hough detector moved to after other hypotheses - see fallback below */

	DebugConsole_WriteString("[BASELINE] probe bright-center start\r\n");
	bright_ok = AppBaselineRuntime_EstimateCenterFromBrightPixels(frame_bytes,
														  frame_size, &center_x, &center_y, &bright_count);
	if (bright_ok)
	{
		size_t expected_center_x = 0U;
		size_t expected_center_y = 0U;
		AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
												CAMERA_CAPTURE_HEIGHT_PIXELS,
												&expected_center_x, &expected_center_y);
		const float center_dx =
			(float)center_x - (float)expected_center_x;
		const float center_dy =
			(float)center_y - (float)expected_center_y;
		const float center_drift =
			sqrtf((center_dx * center_dx) + (center_dy * center_dy));

		/* Why: this rejects glare-driven bright boxes such as (126,86), while
		 * preserving modest real framing movement for the bright hypothesis. */
		if (center_drift > APP_BASELINE_BRIGHT_CENTER_MAX_DRIFT_PIXELS)
		{
			bright_ok = false;
			AppBaselineRuntime_WriteDirectStatus(
				"[BASELINE][CV] bright-center-outlier\r\n");
		}
	}
	AppBaselineRuntime_WriteDirectStatus(
		bright_ok ? "[BASELINE][CV] bright-center-valid\r\n"
				  : "[BASELINE][CV] bright-center-missing\r\n");
	if (bright_ok)
	{
			bright_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(
				frame_bytes, frame_size, center_x, center_y, dial_radius_px,
				"bright-center-polar", &bright_hypothesis);
	}
	DebugConsole_Printf(
		"[BASELINE] probe bright-center done ok=%u center=(%lu,%lu) count=%lu\r\n",
		bright_ok ? 1U : 0U, (unsigned long)center_x, (unsigned long)center_y,
		(unsigned long)bright_count);
	(void)bright_count;

	DebugConsole_WriteString("[BASELINE] probe fixed-crop start\r\n");
	fixed_crop_ok = AppBaselineRuntime_EstimateFromTrainingCropHypothesis(
		frame_bytes, frame_size, &fixed_crop_hypothesis);
	DebugConsole_Printf(
		"[BASELINE] probe fixed-crop done ok=%u\r\n",
		fixed_crop_ok ? 1U : 0U);

	DebugConsole_WriteString("[BASELINE] probe board-prior start\r\n");
	board_prior_ok = AppBaselineRuntime_EstimateFromBoardPriorHypothesis(
		frame_bytes, frame_size, &board_prior_hypothesis);
	DebugConsole_Printf(
		"[BASELINE] probe board-prior done ok=%u\r\n",
		board_prior_ok ? 1U : 0U);

	DebugConsole_WriteString("[BASELINE] probe rim-geometry start\r\n");
	rim_geometry_ok = AppBaselineRuntime_EstimateFromRimGeometryHypothesis(
		frame_bytes, frame_size, &rim_geometry_hypothesis);
	DebugConsole_Printf(
		"[BASELINE] probe rim-geometry done ok=%u\r\n",
		rim_geometry_ok ? 1U : 0U);

	/* Use the inner dial center for the image-center hypothesis too, so the
	 * polar vote pivots around the correct point for the Celsius scale. */
	{
		size_t inner_center_x = 0U;
		size_t inner_center_y = 0U;
		DebugConsole_WriteString("[BASELINE] probe image-center start\r\n");
		AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
											CAMERA_CAPTURE_HEIGHT_PIXELS,
											&inner_center_x, &inner_center_y);
		center_ok = AppBaselineRuntime_EstimateFromCenterHypothesis(frame_bytes,
																	frame_size, inner_center_x,
																	inner_center_y, dial_radius_px,
																	"image-center-polar",
																	&center_hypothesis);
		DebugConsole_Printf(
			"[BASELINE] probe image-center done ok=%u center=(%lu,%lu)\r\n",
			center_ok ? 1U : 0U,
			(unsigned long)inner_center_x, (unsigned long)inner_center_y);
	}

	if (!bright_ok && !fixed_crop_ok && !board_prior_ok && !rim_geometry_ok &&
		!center_ok)
	{
		/* All primary hypotheses failed. Try the dynamic Hough detector
		 * as a last resort before giving up. */
		AppBaselineRuntime_Estimate_t hough_estimate = {0};
		if (AppBaselineHough_Estimate(
				frame_bytes, frame_size, CAMERA_CAPTURE_WIDTH_PIXELS,
				CAMERA_CAPTURE_HEIGHT_PIXELS, &hough_estimate)
				&& AppBaselineRuntime_PassesAcceptanceGate(&hough_estimate))
		{
			*estimate_out = hough_estimate;
			AppBaselineRuntime_WriteDirectStatus(
					"[BASELINE][CV] dynamic-hough-fallback\r\n");
			return true;
		}

		AppBaselineRuntime_WriteDirectStatus(
			"[BASELINE][CV] no-candidate\r\n");
		DebugConsole_WriteString(
			"[BASELINE] estimate rejected: no candidate survived the initial geometry probes.\r\n");
		return false;
	}

	/* Record a mid-window power snapshot before the refinement pass so the
	 * baseline can be compared against the AI pipeline's mid-inference sample. */
	Metrics_Checkpoint("MID");

	/* Run the editable-diagram selector: refine every surviving geometry seed
	 * locally, then let the strongest agreeing angle cluster win. */
	{
#if APP_BASELINE_ENABLE_LOCAL_GEOMETRY_SWEEP
		AppBaselineRuntime_Estimate_t refined_bright = {0};
		AppBaselineRuntime_Estimate_t refined_fixed_crop = {0};
		AppBaselineRuntime_Estimate_t refined_board_prior = {0};
		AppBaselineRuntime_Estimate_t refined_rim_geometry = {0};
		AppBaselineRuntime_Estimate_t refined_center = {0};
		static const char *const refined_candidate_labels[5] = {
			"bright-center-polar",
			"fixed-crop-polar",
			"board-prior-polar",
			"rim-center-polar",
			"image-center-polar",
		};
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

#if APP_BASELINE_DEBUG_SELECTION
			AppBaselineRuntime_LogCandidateSummary(
				refined_candidate_labels[candidate_index],
				candidate_ok[candidate_index], refined_estimate);
#endif

			if ((selected_estimate == NULL) || AppBaselineRuntime_IsBetterEstimate(refined_estimate,
																				   selected_estimate))
			{
#if APP_BASELINE_DEBUG_SELECTION
				DebugConsole_Printf(
					"[BASELINE][DBG] sweep select: %s -> %s\r\n",
					(selected_estimate != NULL &&
					 selected_estimate->source_label != NULL)
						? selected_estimate->source_label
						: "none",
					(refined_estimate->source_label != NULL)
						? refined_estimate->source_label
						: "unknown");
#endif
				selected_estimate = refined_estimate;
			}
		}

		{
			const AppBaselineRuntime_Estimate_t *consensus_fallback =
				(fixed_crop_ok && refined_fixed_crop.valid)
					? &refined_fixed_crop
					: selected_estimate;
			selected_estimate = AppBaselineRuntime_SelectConsensusEstimate(
				consensus_candidates, candidate_ok, consensus_fallback);
		}
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] sweep final select: %s\r\n",
			(selected_estimate != NULL &&
			 selected_estimate->source_label != NULL)
				? selected_estimate->source_label
				: "none");
#endif
#else
		{
			const AppBaselineRuntime_Estimate_t *candidate_estimates[5] = {
				&bright_hypothesis,
				&fixed_crop_hypothesis,
				&board_prior_hypothesis,
				&rim_geometry_hypothesis,
				&center_hypothesis,
			};
			static const char *const candidate_labels[5] = {
				"bright-center-polar",
				"fixed-crop-polar",
				"board-prior-polar",
				"rim-center-polar",
				"image-center-polar",
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

				AppBaselineRuntime_LogCandidateSummary(
					candidate_labels[candidate_index],
					candidate_ok[candidate_index], candidate_estimate);

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
				candidate_estimates, candidate_ok,
				fixed_crop_ok ? &fixed_crop_hypothesis : selected_estimate);
		}
#endif
	}

	if (selected_estimate == NULL)
	{
		AppBaselineRuntime_WriteDirectStatus(
			"[BASELINE][CV] no-selected\r\n");
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

	/* Keep the five-hypothesis consensus as the final selector. Rim geometry
	 * remains available as one candidate, but it cannot hard-override the other
	 * seeds because the widened rim guard admitted a false outer-rim peak. */

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
			"[BASELINE] Rejected: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%zu cy=%zu\r\n",
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "?",
			reject_reason, conf_m, threshold_m, best_score_m, runner_up_m,
			peak_ratio_x1000, estimate_out->center_x, estimate_out->center_y);
		DebugConsole_Printf(
			"[BASELINE] FAIL: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%zu cy=%zu\r\n",
			(estimate_out->source_label != NULL) ? estimate_out->source_label : "?",
			reject_reason, conf_m, threshold_m, best_score_m, runner_up_m,
			peak_ratio_x1000, estimate_out->center_x, estimate_out->center_y);
		DebugConsole_Printf(
			"[BASELINE] REJECTED: src=%s reason=%s conf=%ld/%ld score=%ld ru=%ld pr=%ld cx=%zu cy=%zu\r\n",
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
 * Confidence alone is not enough, because broad near-ties can still have a
 * high vote sum. A candidate should score well when it is strong and its
 * winning peak stands clearly above the runner-up.
 */
static float AppBaselineRuntime_ComputeEstimateQuality(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	float peak_ratio = 0.0f;
	float separation_boost = 1.0f;

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

	peak_ratio = AppBaselineRuntime_ClampFloat(peak_ratio, 1.0f, 4.0f);
	separation_boost = peak_ratio;

	return AppBaselineRuntime_ClampFloat(
		estimate->confidence * separation_boost,
		0.0f, 1000000.0f);
}

/**
 * @brief Measure how far an estimate's center is from the inner dial center.
 *
 * We use the training-crop center as the reference because the baseline
 * selector already reasons around that geometry, and the wrong gauge family is
 * usually the one that drifts farthest from that anchor.
 */
static float AppBaselineRuntime_ComputeCenterDistancePixels(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	size_t inner_center_x = 0U;
	size_t inner_center_y = 0U;

	if ((estimate == NULL) || !estimate->valid)
	{
		return 0.0f;
	}

	AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
										CAMERA_CAPTURE_HEIGHT_PIXELS,
										&inner_center_x, &inner_center_y);

	return sqrtf(((float)estimate->center_x - (float)inner_center_x) *
					 ((float)estimate->center_x - (float)inner_center_x) +
				 ((float)estimate->center_y - (float)inner_center_y) *
					 ((float)estimate->center_y - (float)inner_center_y));
}

/**
 * @brief Measure the shortest distance between two angles in degrees.
 *
 * This keeps gauge-family comparisons stable across the 0/360 wrap where
 * temperature-space agreement can hide a wrong spoke family.
 */
static float AppBaselineRuntime_MinAngleDistanceDegrees(
	float angle_a_deg, float angle_b_deg)
{
	float delta_deg = fabsf(angle_a_deg - angle_b_deg);
	if (delta_deg > 180.0f)
	{
		delta_deg = 360.0f - delta_deg;
	}

	return delta_deg;
}

/**
 * @brief Return a small priority score for live geometry sources.
 *
 * Clean, near-centered captures should prefer the stable fixed-crop anchor
 * first, then the inner image center, and only then fall back to the
 * board-specific prior before we consider the bright and rim hypotheses. The
 * board prior is still useful as a rescue path on awkward framings, but it
 * For a robust baseline that works when the gauge moves, the bright centroid
 * is preferred because it adapts to gauge position.
 */
static int AppBaselineRuntime_SourcePriority(const char *source_label)
{
	if (source_label == NULL)
	{
		return 0;
	}

	/* Fixed crop is the most reliable for standard framing */
	if (strcmp(source_label, "fixed-crop-polar") == 0)
	{
		return 5;
	}

	/* Bright centroid adapts to gauge position - secondary */
	if ((strcmp(source_label, "bright-center-polar") == 0) ||
		(strcmp(source_label, "face-center-polar") == 0))
	{
		return 4;
	}

	if (strcmp(source_label, "image-center-polar") == 0)
	{
		return 3;
	}

	if (strcmp(source_label, "board-prior-polar") == 0)
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
		AppBaselineRuntime_WriteDirectGateStatus("invalid", estimate);
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_WriteString(
			"[BASELINE][DBG] gate reject: invalid estimate\r\n");
#endif
		return false;
	}

	if (estimate->confidence < APP_BASELINE_CONFIDENCE_THRESHOLD)
	{
		AppBaselineRuntime_WriteDirectGateStatus("confidence", estimate);
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] gate reject: confidence src=%s conf=%ld/1000 threshold=%ld/1000\r\n",
			(estimate->source_label != NULL) ? estimate->source_label : "?",
			AppBaselineRuntime_RoundToLong(estimate->confidence * 1000.0f),
			AppBaselineRuntime_RoundToLong(APP_BASELINE_CONFIDENCE_THRESHOLD * 1000.0f));
#endif
		return false;
	}

	if (estimate->best_score < APP_BASELINE_MIN_ACCEPT_SCORE)
	{
		AppBaselineRuntime_WriteDirectGateStatus("score", estimate);
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] gate reject: score src=%s score=%ld threshold=%ld\r\n",
			(estimate->source_label != NULL) ? estimate->source_label : "?",
			AppBaselineRuntime_RoundToLong(estimate->best_score),
			AppBaselineRuntime_RoundToLong(APP_BASELINE_MIN_ACCEPT_SCORE));
#endif
		return false;
	}

	if (!AppBaselineRuntime_HasAcceptablePeakSeparation(estimate))
	{
		AppBaselineRuntime_WriteDirectGateStatus("peak-ratio", estimate);
#if APP_BASELINE_DEBUG_SELECTION
		const long score_whole = AppBaselineRuntime_RoundToLong(estimate->best_score);
		const long runner_up_whole = AppBaselineRuntime_RoundToLong(estimate->runner_up_score);
		const long peak_ratio_milli =
			AppBaselineRuntime_RoundToLong(
				((estimate->runner_up_score > 0.0f)
					 ? (estimate->best_score / estimate->runner_up_score)
					 : estimate->best_score) * 1000.0f);
		DebugConsole_Printf(
			"[BASELINE][DBG] gate reject: peak_ratio src=%s score=%ld ru=%ld ratio=%ld/1000\r\n",
			(estimate->source_label != NULL) ? estimate->source_label : "?",
			score_whole, runner_up_whole, peak_ratio_milli);
#endif
		return false;
	}

	/* Reject estimates whose center is too far from the inner dial center.
	 * Use a wide threshold so the bright centroid can track the gauge when
	 * it moves in the frame. */
	{
		size_t inner_cx = 0U;
		size_t inner_cy = 0U;
		AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
											CAMERA_CAPTURE_HEIGHT_PIXELS,
											&inner_cx, &inner_cy);
		const float dx = (float)estimate->center_x - (float)inner_cx;
		const float dy = (float)estimate->center_y - (float)inner_cy;
		const float dist_sq = dx * dx + dy * dy;
		if (dist_sq > 22500.0f) /* 150 pixels — wide enough for bright centroid tracking */
		{
			AppBaselineRuntime_WriteDirectGateStatus("center-distance", estimate);
#if APP_BASELINE_DEBUG_SELECTION
			const float center_distance = sqrtf(dist_sq);
			const long center_distance_tenths =
				AppBaselineRuntime_RoundToLong(center_distance * 10.0f);
			const long center_distance_abs_tenths =
				(center_distance_tenths < 0L) ? -center_distance_tenths
											 : center_distance_tenths;
			DebugConsole_Printf(
				"[BASELINE][DBG] gate reject: center_dist src=%s center=(%lu,%lu) inner=(%lu,%lu) dcenter=%ld.%01ld\r\n",
				(estimate->source_label != NULL) ? estimate->source_label : "?",
				(unsigned long)estimate->center_x,
				(unsigned long)estimate->center_y,
				(unsigned long)inner_cx,
				(unsigned long)inner_cy,
				(long)(center_distance_tenths / 10L),
				(long)(center_distance_abs_tenths % 10L));
#endif
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
	const float candidate_center_distance =
		AppBaselineRuntime_ComputeCenterDistancePixels(candidate);
	const float incumbent_center_distance =
		AppBaselineRuntime_ComputeCenterDistancePixels(incumbent);
	const float center_distance_delta =
		candidate_center_distance - incumbent_center_distance;
	const bool candidate_is_closer =
		(center_distance_delta <= -APP_BASELINE_CENTER_DISTANCE_PREFER_DELTA_PIXELS);
	const bool incumbent_is_closer =
		(center_distance_delta >= APP_BASELINE_CENTER_DISTANCE_PREFER_DELTA_PIXELS);

	if ((candidate == NULL) || !candidate->valid)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_WriteString(
			"[BASELINE][DBG] compare: incumbent wins because candidate invalid\r\n");
#endif
		return false;
	}
	if ((incumbent == NULL) || !incumbent->valid)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_WriteString(
			"[BASELINE][DBG] compare: candidate wins because incumbent invalid\r\n");
#endif
		return true;
	}

		/* Keep the classical ordering strict: board-prior rescue first, then
		 * direct geometric quality, then the higher-frequency classical scores. */
		if ((candidate_quality > 0.0f) && (incumbent_quality > 0.0f))
		{
			const float quality_ratio = candidate_quality / incumbent_quality;
			const bool candidate_is_primary =
				(candidate->source_label != NULL) &&
				(strcmp(candidate->source_label, "fixed-crop-polar") == 0);
			const bool incumbent_is_primary =
				(incumbent->source_label != NULL) &&
				(strcmp(incumbent->source_label, "fixed-crop-polar") == 0);
			const bool candidate_is_bright =
				(candidate->source_label != NULL) &&
				(strcmp(candidate->source_label, "bright-center-polar") == 0);
			const bool incumbent_is_bright =
				(incumbent->source_label != NULL) &&
				(strcmp(incumbent->source_label, "bright-center-polar") == 0);
			const bool candidate_is_board_prior =
				(candidate->source_label != NULL) &&
				(strcmp(candidate->source_label, "board-prior-polar") == 0);
			const bool incumbent_is_board_prior =
				(incumbent->source_label != NULL) &&
				(strcmp(incumbent->source_label, "board-prior-polar") == 0);

		/* Board-prior is a rescue hypothesis, not a hard loser. Let it win when
		 * it is clearly better than the current primary family, but keep the
		 * primary family when the quality gap is still small. The board prior
		 * can drift toward the wrong spoke family on awkward framings, so the
		 * rescue threshold is intentionally a bit stricter here. */
		if (candidate_is_board_prior && (incumbent_is_primary || incumbent_is_bright) &&
			(quality_ratio < 1.15f))
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: primary keeps win over board-prior src=%s over %s q=%ld/%ld ratio=%ld/1000\r\n",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(quality_ratio * 1000.0f));
#endif
			return false;
		}
		if ((incumbent_is_board_prior) && (candidate_is_primary || candidate_is_bright) &&
			(quality_ratio >= 0.92f))
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: primary defeats board-prior src=%s over %s q=%ld/%ld ratio=%ld/1000\r\n",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(quality_ratio * 1000.0f));
#endif
			return true;
		}

		/* Let a visibly better-centered candidate win when the quality gap is
		 * only modest. This keeps the selector from latching onto a farther rim
		 * anchor just because it scored a few percent higher. */
		if (candidate_is_closer &&
			(quality_ratio >= APP_BASELINE_CENTER_DISTANCE_MIN_QUALITY_RATIO))
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: candidate wins by center fit src=%s over %s dcenter=%ld/%ld q=%ld/%ld\r\n",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				AppBaselineRuntime_RoundToLong(candidate_center_distance * 10.0f),
				AppBaselineRuntime_RoundToLong(incumbent_center_distance * 10.0f),
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f));
#endif
			return true;
		}
		if (incumbent_is_closer &&
			(quality_ratio < APP_BASELINE_GEOMETRY_OVERRIDE_RATIO))
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: incumbent wins by center fit src=%s over %s dcenter=%ld/%ld q=%ld/%ld\r\n",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				AppBaselineRuntime_RoundToLong(incumbent_center_distance * 10.0f),
				AppBaselineRuntime_RoundToLong(candidate_center_distance * 10.0f),
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f));
#endif
			return false;
		}

		if (quality_ratio > 1.0f)
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: candidate wins by quality src=%s over %s q=%ld/%ld\r\n",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f));
#endif
			return true;
		}
		if (quality_ratio < 1.0f)
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: incumbent wins by quality src=%s over %s q=%ld/%ld\r\n",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f),
				AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f));
#endif
			return false;
		}
	}
	else if (candidate_quality > incumbent_quality)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: candidate wins by quality src=%s over %s q=%ld/%ld\r\n",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f),
			AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f));
#endif
		return true;
	}
	else if (candidate_quality < incumbent_quality)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: incumbent wins by quality src=%s over %s q=%ld/%ld\r\n",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			AppBaselineRuntime_RoundToLong(incumbent_quality * 1000.0f),
			AppBaselineRuntime_RoundToLong(candidate_quality * 1000.0f));
#endif
		return false;
	}

	/* Penalize the 'bright-center' hypothesis if the center is too far from
	 * the image center. This prevents glare-induced centroids from winning
	 * over stable geometric anchors. */
	if (candidate->source_label != NULL &&
		((strcmp(candidate->source_label, "bright-center-polar") == 0) ||
		 (strcmp(candidate->source_label, "face-center-polar") == 0)))
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

	{
		const float candidate_peak_ratio =
			(candidate->runner_up_score > 0.0f) ?
			(candidate->best_score / candidate->runner_up_score) :
			candidate->best_score;
		const float incumbent_peak_ratio =
			(incumbent->runner_up_score > 0.0f) ?
			(incumbent->best_score / incumbent->runner_up_score) :
			incumbent->best_score;

		if (candidate_peak_ratio > incumbent_peak_ratio)
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: candidate wins by peak ratio src=%s over %s peak=%ld/%ld\r\n",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				AppBaselineRuntime_RoundToLong(candidate_peak_ratio * 1000.0f),
				AppBaselineRuntime_RoundToLong(incumbent_peak_ratio * 1000.0f));
#endif
			return true;
		}
		if (candidate_peak_ratio < incumbent_peak_ratio)
		{
#if APP_BASELINE_DEBUG_SELECTION
			DebugConsole_Printf(
				"[BASELINE][DBG] compare: incumbent wins by peak ratio src=%s over %s peak=%ld/%ld\r\n",
				(incumbent->source_label != NULL) ? incumbent->source_label : "?",
				(candidate->source_label != NULL) ? candidate->source_label : "?",
				AppBaselineRuntime_RoundToLong(incumbent_peak_ratio * 1000.0f),
				AppBaselineRuntime_RoundToLong(candidate_peak_ratio * 1000.0f));
#endif
			return false;
		}
	}

	if (candidate->best_score > incumbent->best_score)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: candidate wins by score src=%s over %s score=%ld/%ld\r\n",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			AppBaselineRuntime_RoundToLong(candidate->best_score),
			AppBaselineRuntime_RoundToLong(incumbent->best_score));
#endif
		return true;
	}
	if (candidate->best_score < incumbent->best_score)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: incumbent wins by score src=%s over %s score=%ld/%ld\r\n",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			AppBaselineRuntime_RoundToLong(incumbent->best_score),
			AppBaselineRuntime_RoundToLong(candidate->best_score));
#endif
		return false;
	}

	if (candidate->confidence > incumbent->confidence)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: candidate wins by confidence src=%s over %s conf=%ld/%ld\r\n",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			AppBaselineRuntime_RoundToLong(candidate->confidence * 1000.0f),
			AppBaselineRuntime_RoundToLong(incumbent->confidence * 1000.0f));
#endif
		return true;
	}
	if (candidate->confidence < incumbent->confidence)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] compare: incumbent wins by confidence src=%s over %s conf=%ld/%ld\r\n",
			(incumbent->source_label != NULL) ? incumbent->source_label : "?",
			(candidate->source_label != NULL) ? candidate->source_label : "?",
			AppBaselineRuntime_RoundToLong(incumbent->confidence * 1000.0f),
			AppBaselineRuntime_RoundToLong(candidate->confidence * 1000.0f));
#endif
		return false;
	}

#if APP_BASELINE_DEBUG_SELECTION
	DebugConsole_Printf(
		"[BASELINE][DBG] compare: source-priority tie-break src=%s over %s pri=%d/%d\r\n",
		(candidate->source_label != NULL) ? candidate->source_label : "?",
		(incumbent->source_label != NULL) ? incumbent->source_label : "?",
		AppBaselineRuntime_SourcePriority(candidate->source_label),
		AppBaselineRuntime_SourcePriority(incumbent->source_label));
#endif
	return AppBaselineRuntime_SourcePriority(candidate->source_label) >
		AppBaselineRuntime_SourcePriority(incumbent->source_label);
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
	/* The shared geometry already encodes the measured inner Celsius pivot.
	 * Do not apply a second vertical correction: that duplicate bias moved the
	 * fixed hypothesis from y~=100 to y~=85 and favored the cold-side bezel. */

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

	/* Scan area centered on the inner dial, tight enough to avoid false
	 * circles from the outer bezel or background clutter.  1.0× dial radius
	 * gives the rim-vote a roughly ±69 px window on a 224x224 frame. */
	const size_t scan_radius = (size_t)(dial_radius_px * 1.0f);
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

	/* Guard: if the rim-vote center wandered more than 25 px from the known
	 * inner dial center, it likely locked onto the outer bezel or background
	 * clutter.  Fall back to the expected center rather than reporting a
	 * wrong needle angle. */
	{
		size_t expected_cx = 0U;
		size_t expected_cy = 0U;
		AppGaugeGeometry_TrainingCropCenter(width_pixels, height_pixels,
											&expected_cx, &expected_cy);
		const float dx = (float)center_x - (float)expected_cx;
		const float dy = (float)center_y - (float)expected_cy;
		const float center_dist = sqrtf(dx * dx + dy * dy);
		/* Keep the tighter live guard: the wider tolerance admitted an outer-rim
		 * center at (124,111) that produced a strong but false hot-end peak. */
		if (center_dist > 14.0f)
		{
			center_x = expected_cx;
			center_y = expected_cy;
		}
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
	/* Use true centroid (sum/count) instead of bounding box midpoint.
	 * This adapts to gauge position and is more accurate when the gauge moves. */
	uint64_t bright_sum_x = 0U;
	uint64_t bright_sum_y = 0U;
	size_t bright_count = 0U;

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
		}
	}

	if (bright_count < APP_BASELINE_MIN_BRIGHT_PIXELS)
	{
		return false;
	}

	/* Compute true centroid: adapts to gauge position in the frame */
	*center_x_out = (size_t)(bright_sum_x / (uint64_t)bright_count);
	*center_y_out = (size_t)(bright_sum_y / (uint64_t)bright_count);
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
 * @brief Store one stable baseline estimate in the tiny smoothing history.
 * @param estimate Accepted estimate from the five-hypothesis selector.
 * @sideeffects Resets on a large scene jump and advances the ring buffer.
 */
static bool AppBaselineRuntime_PushEstimateHistory(
	const AppBaselineRuntime_Estimate_t *estimate)
{
	if ((estimate == NULL) || !estimate->valid ||
		!AppBaselineRuntime_IsStableEstimateForHistory(estimate))
	{
		return false;
	}

	if (camera_baseline_estimate_history_count == 0U)
	{
		camera_baseline_estimate_history[0U] = *estimate;
		camera_baseline_estimate_history_count = 1U;
		camera_baseline_estimate_history_next_index =
			1U % APP_BASELINE_ESTIMATE_HISTORY_SIZE;
		return true;
	}

	{
		const size_t last_index =
			(camera_baseline_estimate_history_next_index +
			 APP_BASELINE_ESTIMATE_HISTORY_SIZE - 1U) %
			APP_BASELINE_ESTIMATE_HISTORY_SIZE;
		const float last_temperature_c =
			camera_baseline_estimate_history[last_index].temperature_c;

		/* A real setpoint change should relock quickly instead of averaging
		 * readings from two different physical states. */
		if (fabsf(estimate->temperature_c - last_temperature_c) >
			APP_BASELINE_HISTORY_RESET_DELTA_C)
		{
			AppBaselineRuntime_ResetEstimateHistory();
		}
	}

	camera_baseline_estimate_history[camera_baseline_estimate_history_next_index] =
		*estimate;
	if (camera_baseline_estimate_history_count < APP_BASELINE_ESTIMATE_HISTORY_SIZE)
	{
		camera_baseline_estimate_history_count++;
	}
	camera_baseline_estimate_history_next_index =
		(camera_baseline_estimate_history_next_index + 1U) %
		APP_BASELINE_ESTIMATE_HISTORY_SIZE;
	return true;
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
		const bool cold_recovery_candidate =
			(!camera_baseline_current_frame_is_bright) &&
			(camera_baseline_last_temperature_c >= APP_BASELINE_COLD_RECOVERY_HISTORY_TEMP_C) &&
			(estimate->temperature_c <= APP_BASELINE_COLD_RECOVERY_TARGET_TEMP_C) &&
			(estimate->confidence >= APP_BASELINE_COLD_RECOVERY_MIN_CONFIDENCE);
		if ((delta_c > APP_BASELINE_AMBIGUOUS_JUMP_DELTA_C) &&
			(peak_ratio < APP_BASELINE_AMBIGUOUS_JUMP_MAX_PEAK_RATIO) &&
			(estimate->confidence < APP_BASELINE_AMBIGUOUS_JUMP_MAX_CONFIDENCE) &&
			!cold_recovery_candidate)
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
		if (cold_recovery_candidate)
		{
			const long prev_x10 =
				AppBaselineRuntime_RoundToLong(camera_baseline_last_temperature_c * 10.0f);
			const long curr_x10 =
				AppBaselineRuntime_RoundToLong(estimate->temperature_c * 10.0f);
			const long conf_m = AppBaselineRuntime_RoundToLong(
				estimate->confidence * 1000.0f);
			DebugConsole_Printf(
				"[BASELINE] Stability recovery: warm->cold unlock prev=%ld.%01ldC curr=%ld.%01ldC conf=%ld/1000\r\n",
				prev_x10 / 10L,
				((prev_x10 % 10L) < 0L) ? -(prev_x10 % 10L) : (prev_x10 % 10L),
				curr_x10 / 10L,
				((curr_x10 % 10L) < 0L) ? -(curr_x10 % 10L) : (curr_x10 % 10L),
				conf_m);
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
	static const char *const candidate_labels[5] = {
		"bright-center-polar",
		"fixed-crop-polar",
		"board-prior-polar",
		"rim-center-polar",
		"image-center-polar",
	};
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
		const float estimate_i_angle_deg =
			AppBaselineRuntime_NormalizeAngleDegrees(
				estimate_i->angle_rad * (180.0f / APP_BASELINE_PI));

		for (size_t other_index = 0U; other_index < valid_count; ++other_index)
		{
			const size_t j = valid_indices[other_index];
			const AppBaselineRuntime_Estimate_t *estimate_j = estimates[j];

			if ((i == j) || (estimate_j == NULL))
			{
				continue;
			}

			const float estimate_j_angle_deg =
				AppBaselineRuntime_NormalizeAngleDegrees(
					estimate_j->angle_rad * (180.0f / APP_BASELINE_PI));
			if (AppBaselineRuntime_MinAngleDistanceDegrees(
					estimate_i_angle_deg, estimate_j_angle_deg) <=
				APP_BASELINE_CONSENSUS_ANGLE_DELTA_DEG)
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
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] consensus: fallback because support=%lu\r\n",
			(unsigned long)best_support);
#endif
		return fallback_estimate;
	}

	const float fallback_quality =
		AppBaselineRuntime_ComputeEstimateQuality(fallback_estimate);
	if (fallback_quality <= 0.0f)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_WriteString(
			"[BASELINE][DBG] consensus: fallback invalid quality\r\n");
#endif
		return fallback_estimate;
	}

	for (size_t valid_index = 0U; valid_index < valid_count; ++valid_index)
	{
		const size_t i = valid_indices[valid_index];
		const AppBaselineRuntime_Estimate_t *candidate = estimates[i];

		if (support[i] != best_support)
		{
			continue;
		}

		if ((best_estimate == NULL) ||
			AppBaselineRuntime_IsBetterEstimate(candidate, best_estimate))
		{
			best_estimate = candidate;
		}
	}

	if (best_estimate == NULL)
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_WriteString(
			"[BASELINE][DBG] consensus: no best estimate found\r\n");
#endif
		return fallback_estimate;
	}

	if (best_estimate == fallback_estimate)
	{
		return best_estimate;
	}

	if (AppBaselineRuntime_IsBetterEstimate(best_estimate, fallback_estimate))
	{
#if APP_BASELINE_DEBUG_SELECTION
		DebugConsole_Printf(
			"[BASELINE][DBG] consensus: winner=%s support=%lu fallback=%s\r\n",
			(best_estimate->source_label != NULL) ? best_estimate->source_label : "?",
			(unsigned long)best_support,
			(fallback_estimate->source_label != NULL) ? fallback_estimate->source_label : "?");
		for (size_t i = 0U; i < valid_count; ++i)
		{
			const size_t idx = valid_indices[i];
			const AppBaselineRuntime_Estimate_t *estimate = estimates[idx];
			if ((estimate == NULL) || !estimate->valid)
			{
				continue;
			}
			{
				const long temp_tenths =
					AppBaselineRuntime_RoundToLong(estimate->temperature_c * 10.0f);
				const long temp_abs_tenths =
					(temp_tenths < 0L) ? -temp_tenths : temp_tenths;
				const long conf_thousandths =
					AppBaselineRuntime_RoundToLong(estimate->confidence * 1000.0f);
				const long conf_abs_thousandths =
					(conf_thousandths < 0L) ? -conf_thousandths : conf_thousandths;
				const float quality = AppBaselineRuntime_ComputeEstimateQuality(estimate);
				const long quality_thousandths =
					AppBaselineRuntime_RoundToLong(quality * 1000.0f);
				const long quality_abs_thousandths =
					(quality_thousandths < 0L) ? -quality_thousandths
											 : quality_thousandths;
			DebugConsole_Printf(
				"[BASELINE][DBG] consensus slot=%s support=%lu temp=%ld.%01ld conf=%ld.%03ld q=%ld.%03ld\r\n",
				candidate_labels[idx],
				(unsigned long)support[idx],
				(long)(temp_tenths / 10L),
				(long)(temp_abs_tenths % 10L),
				(long)(conf_thousandths / 1000L),
				(long)(conf_abs_thousandths % 1000L),
				(long)(quality_thousandths / 1000L),
				(long)(quality_abs_thousandths % 1000L));
			}
		}
#endif
		return best_estimate;
	}

	return fallback_estimate;
}

/**
 * @brief Return a warming estimate or the median of the stable history.
 * @param estimate_out Destination for the held, warming, or smoothed result.
 * @return true when at least one stable history sample is available.
 */
static bool AppBaselineRuntime_SelectSmoothedEstimate(
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	AppBaselineRuntime_Estimate_t ordered[APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {0};
	size_t sample_count = camera_baseline_estimate_history_count;

	if ((estimate_out == NULL) || (sample_count == 0U))
	{
		return false;
	}

	(void)memcpy(ordered, camera_baseline_estimate_history,
				 sample_count * sizeof(ordered[0]));

	/* Sort a local copy so the median is robust to one isolated false peak. */
	for (size_t i = 1U; i < sample_count; ++i)
	{
		AppBaselineRuntime_Estimate_t key = ordered[i];
		size_t j = i;

		while ((j > 0U) &&
			   (ordered[j - 1U].temperature_c > key.temperature_c))
		{
			ordered[j] = ordered[j - 1U];
			--j;
		}
		ordered[j] = key;
	}

	/* Discard old entries in the subdial band if any valid sweep samples are
	 * available; this prevents a polluted pre-fix sample from winning the hold. */
	{
		AppBaselineRuntime_Estimate_t valid_entries[
			APP_BASELINE_ESTIMATE_HISTORY_SIZE] = {0};
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
			(void)memcpy(ordered, valid_entries,
						 valid_count * sizeof(ordered[0]));
			sample_count = valid_count;
		}
	}

	if (sample_count < APP_BASELINE_ESTIMATE_HISTORY_SIZE)
	{
		*estimate_out = ordered[sample_count - 1U];
		estimate_out->valid = true;
		if ((estimate_out->source_label == NULL) ||
			(strncmp(estimate_out->source_label,
					"classical-template", 18U) != 0))
		{
			estimate_out->source_label = "baseline-polar-warming";
		}
		return true;
	}

	*estimate_out = ordered[sample_count / 2U];
	estimate_out->valid = true;
	if ((estimate_out->source_label == NULL) ||
		(strncmp(estimate_out->source_label,
				"classical-template", 18U) != 0))
	{
		estimate_out->source_label = "baseline-polar-smoothed";
	}
	return true;
}

/**
 * @brief Log one candidate geometry before the final baseline choice is made.
 *
 * This keeps the live trace compact but still exposes the needle angle, dial
 * center, temperature, and source quality for each hypothesis so we can
 * compare the classical baseline against the CNN and spot the bad geometry
 * family quickly.
 */
static void AppBaselineRuntime_LogCandidateSummary(
	const char *slot_name, bool candidate_ok,
	const AppBaselineRuntime_Estimate_t *estimate)
{
	const char *source_name = "unknown";
	size_t inner_center_x = 0U;
	size_t inner_center_y = 0U;
	const float center_distance = (estimate != NULL)
		? ((AppGaugeGeometry_TrainingCropCenter(CAMERA_CAPTURE_WIDTH_PIXELS,
											 CAMERA_CAPTURE_HEIGHT_PIXELS,
											 &inner_center_x, &inner_center_y),
			sqrtf(((float)estimate->center_x - (float)inner_center_x) *
				  ((float)estimate->center_x - (float)inner_center_x) +
				  ((float)estimate->center_y - (float)inner_center_y) *
					  ((float)estimate->center_y - (float)inner_center_y))))
		: 0.0f;
	const long angle_tenths = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(
			  (estimate->angle_rad * 180.0f / APP_BASELINE_PI) * 10.0f)
		: 0L;
	const long temperature_tenths = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(estimate->temperature_c * 10.0f)
		: 0L;
	const long confidence_thousandths = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(estimate->confidence * 1000.0f)
		: 0L;
	const long quality_thousandths = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(
			  AppBaselineRuntime_ComputeEstimateQuality(estimate) * 1000.0f)
		: 0L;
	const long center_bias_thousandths = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(
			  AppBaselineRuntime_ComputeSweepCenterWeight(estimate->angle_rad) *
			  1000.0f)
		: 0L;
	const long angle_abs_tenths =
		(angle_tenths < 0L) ? -angle_tenths : angle_tenths;
	const long temperature_abs_tenths =
		(temperature_tenths < 0L) ? -temperature_tenths : temperature_tenths;
	const long confidence_abs_thousandths =
		(confidence_thousandths < 0L) ? -confidence_thousandths
									 : confidence_thousandths;
	const long quality_abs_thousandths =
		(quality_thousandths < 0L) ? -quality_thousandths : quality_thousandths;
	const long score_whole = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(estimate->best_score)
		: 0L;
	const long runner_up_whole = (estimate != NULL)
		? AppBaselineRuntime_RoundToLong(estimate->runner_up_score)
		: 0L;
	const long center_distance_tenths =
		AppBaselineRuntime_RoundToLong(center_distance * 10.0f);
	const long center_distance_abs_tenths =
		(center_distance_tenths < 0L) ? -center_distance_tenths
									 : center_distance_tenths;
	const int priority = (estimate != NULL)
		? AppBaselineRuntime_SourcePriority(estimate->source_label)
		: 0;

	if ((estimate != NULL) && (estimate->source_label != NULL) &&
		(estimate->source_label[0] != '\0'))
	{
		source_name = estimate->source_label;
	}

	DebugConsole_Printf(
		"[BASELINE] candidate[%s]: %s src=%s center=(%lu,%lu) dcenter=%ld.%01ld needle=%ld.%01lddeg temp=%ld.%01ldC conf=%ld.%03ld quality=%ld.%03ld bias=%ld.%03ld score=%ld runner_up=%ld priority=%d\r\n",
		(slot_name != NULL) ? slot_name : "?", candidate_ok ? "ok" : "no",
		source_name,
		(estimate != NULL) ? (unsigned long)estimate->center_x : 0UL,
		(estimate != NULL) ? (unsigned long)estimate->center_y : 0UL,
		(long)(center_distance_tenths / 10L),
		(long)(center_distance_abs_tenths % 10L),
		(long)(angle_tenths / 10L), (long)(angle_abs_tenths % 10L),
		(long)(temperature_tenths / 10L),
		(long)(temperature_abs_tenths % 10L),
		(long)(confidence_thousandths / 1000L),
		(long)(confidence_abs_thousandths % 1000L),
		(long)(quality_thousandths / 1000L),
		(long)(quality_abs_thousandths % 1000L),
		(long)(center_bias_thousandths / 1000L),
		(long)(center_bias_thousandths % 1000L), score_whole, runner_up_whole,
		priority);

#if APP_BASELINE_DEBUG_SELECTION
	DebugConsole_Printf(
		"[BASELINE][DBG] candidate detail: src=%s temp=%ld.%01ldC quality=%ld.%03ld dcenter=%ld.%01ld\r\n",
		source_name,
		(long)(temperature_tenths / 10L),
		(long)(temperature_abs_tenths % 10L),
		(long)(quality_thousandths / 1000L),
		(long)(quality_abs_thousandths % 1000L),
		(long)(center_distance_tenths / 10L),
		(long)(center_distance_abs_tenths % 10L));
#endif
}

/**
 * @brief Map an angle inside the gauge sweep to a temperature.
 */
const AppBaselineRuntime_CalibrationProfile_t *
AppBaselineRuntime_GetCalibrationProfile(void)
{
	if (camera_baseline_active_calibration_profile != NULL)
	{
		return camera_baseline_active_calibration_profile;
	}

	return &AppBaselineRuntime_DefaultCalibrationProfile;
}

/**
 * @brief Select the active gauge calibration profile.
 */
void AppBaselineRuntime_SetCalibrationProfile(
	const AppBaselineRuntime_CalibrationProfile_t *profile)
{
	camera_baseline_active_calibration_profile =
		(profile != NULL) ? profile : &AppBaselineRuntime_DefaultCalibrationProfile;
}

/**
 * @brief Select the active gauge calibration profile by its registered name.
 *
 * This keeps the board startup path simple: the firmware only needs to point
 * at the gauge family it was flashed for, while the runtime owns the actual
 * angle-to-temperature conversion data.
 */
void AppBaselineRuntime_SetCalibrationProfileByName(const char *profile_name)
{
	const AppBaselineRuntime_CalibrationProfile_t *profile =
		AppBaselineRuntime_FindCalibrationProfile(profile_name);

	AppBaselineRuntime_SetCalibrationProfile(profile);

	if ((profile_name != NULL) && (profile_name[0] != '\0') &&
		(profile != NULL) && (profile->profile_name != NULL) &&
		(strcmp(profile->profile_name, profile_name) != 0))
	{
		DebugConsole_Printf(
			"[BASELINE] Calibration profile '%s' not found; using '%s'.\r\n",
			profile_name, profile->profile_name);
	}
	else
	{
		DebugConsole_Printf(
			"[BASELINE] Calibration profile '%s' active.\r\n",
			(profile != NULL) && (profile->profile_name != NULL) ?
			profile->profile_name : "board_celsius_v1");
	}
}

/**
 * @brief Try to convert an angle using explicit calibration anchors.
 *
 * The runtime expects anchors to be ordered in sweep-fraction order from the
 * hot side toward the cold side. When a profile provides at least two
 * anchors, we interpolate between the surrounding points instead of relying
 * on the fallback affine fit.
 */
static bool AppBaselineRuntime_ConvertAnchoredAngleToTemperature(
	const AppBaselineRuntime_CalibrationProfile_t *profile,
	float calibrated_angle_deg, float *temperature_out)
{
	const AppBaselineRuntime_CalibrationPoint_t *points = NULL;
	const size_t point_count =
		(profile != NULL) ? profile->calibration_point_count : 0U;
	const float calibrated_angle_rad =
		calibrated_angle_deg * (APP_BASELINE_PI / 180.0f);

	if ((temperature_out == NULL) || (profile == NULL) ||
		(point_count < 2U) ||
		(point_count > APP_BASELINE_CALIBRATION_MAX_POINTS))
	{
		return false;
	}

	points = profile->calibration_points;
	if (points == NULL)
	{
		return false;
	}

	{
		const float target_fraction =
			AppBaselineRuntime_ConvertAngleToFraction(calibrated_angle_rad);
		float previous_fraction =
			AppBaselineRuntime_ConvertAngleToFraction(
				points[0].angle_deg * (APP_BASELINE_PI / 180.0f));
		float previous_temperature = points[0].temperature_c;

		if (point_count == 2U)
		{
			const float next_fraction =
				AppBaselineRuntime_ConvertAngleToFraction(
					points[1].angle_deg * (APP_BASELINE_PI / 180.0f));
			const float denominator = next_fraction - previous_fraction;

			if (fabsf(denominator) <= 1.0e-6f)
			{
				return false;
			}

			*temperature_out = previous_temperature +
				((points[1].temperature_c - previous_temperature) *
					((target_fraction - previous_fraction) / denominator));
			return true;
		}

		for (size_t index = 1U; index < point_count; ++index)
		{
			const float current_fraction =
				AppBaselineRuntime_ConvertAngleToFraction(
					points[index].angle_deg * (APP_BASELINE_PI / 180.0f));
			const float current_temperature = points[index].temperature_c;
			const float denominator = current_fraction - previous_fraction;

			if (fabsf(denominator) <= 1.0e-6f)
			{
				return false;
			}

			if (target_fraction <= current_fraction)
			{
				*temperature_out = previous_temperature +
					((current_temperature - previous_temperature) *
						((target_fraction - previous_fraction) / denominator));
				return true;
			}

			previous_fraction = current_fraction;
			previous_temperature = current_temperature;
		}

		/* Extrapolate beyond the last anchor using the final segment so the
		 * profile can still represent a usable hot-end correction. */
		{
			const float last_fraction =
				AppBaselineRuntime_ConvertAngleToFraction(
					points[point_count - 1U].angle_deg *
					(APP_BASELINE_PI / 180.0f));
			const float denominator = last_fraction - previous_fraction;

			if (fabsf(denominator) <= 1.0e-6f)
			{
				return false;
			}

			*temperature_out = previous_temperature +
				((points[point_count - 1U].temperature_c - previous_temperature) *
					((target_fraction - previous_fraction) / denominator));
			return true;
		}
	}
}

/**
 * @brief Map an angle inside the gauge sweep to a temperature.
 */
float AppBaselineRuntime_ConvertAngleToTemperature(float angle_rad)
{
	const AppBaselineRuntime_CalibrationProfile_t *profile =
		AppBaselineRuntime_GetCalibrationProfile();
	const float angle_offset_deg =
		(profile != NULL) ? profile->angle_offset_deg :
		APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG;
	const float temperature_pivot_c =
		(profile != NULL) ? profile->temperature_pivot_c :
		APP_BASELINE_TEMPERATURE_CALIBRATION_PIVOT_C;
	const float temperature_gain =
		(profile != NULL) ? profile->temperature_gain :
		APP_BASELINE_TEMPERATURE_CALIBRATION_GAIN;
	float anchored_temperature_c = 0.0f;
	const float calibrated_angle_rad =
		angle_rad + (angle_offset_deg * (APP_BASELINE_PI / 180.0f));
	const float calibrated_angle_deg =
		calibrated_angle_rad * (180.0f / APP_BASELINE_PI);

	if (AppBaselineRuntime_ConvertAnchoredAngleToTemperature(
			profile, calibrated_angle_deg, &anchored_temperature_c))
	{
		return anchored_temperature_c;
	}

	/* Apply calibration offset determined from hard-case analysis.
	 * The detected angles are stable, but the gauge runs hot-to-cold in the
	 * opposite direction from the first decode pass. */
	const float raw_temperature =
		APP_BASELINE_MAX_VALUE_C -
		(AppBaselineRuntime_ConvertAngleToFraction(calibrated_angle_rad) *
			(APP_BASELINE_MAX_VALUE_C - APP_BASELINE_MIN_VALUE_C));

	/* Re-anchor the scale around -30 C so the cold-end fix stays intact while
	 * the hot end gets a small multiplicative lift. */
	return temperature_pivot_c +
		((raw_temperature - temperature_pivot_c) * temperature_gain);
}

/**
 * @brief Map an angle linearly between the active profile's hot and cold anchors.
 *
 * The active profile stores anchors in sweep order from hot to cold. The
 * endpoint-only mapping avoids an interior zero anchor changing the slope of
 * the AI temperature curve while preserving both physical extremes.
 */
float AppBaselineRuntime_ConvertAngleToTemperatureExtremes(float angle_rad)
{
	const AppBaselineRuntime_CalibrationProfile_t *profile =
		AppBaselineRuntime_GetCalibrationProfile();
	const size_t point_count =
		(profile != NULL) ? profile->calibration_point_count : 0U;
	const float angle_offset_deg =
		(profile != NULL) ? profile->angle_offset_deg :
		APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG;
	const float calibrated_angle_rad =
		angle_rad + (angle_offset_deg * (APP_BASELINE_PI / 180.0f));
	const float target_fraction =
		AppBaselineRuntime_ConvertAngleToFraction(calibrated_angle_rad);

	if ((profile == NULL) || (point_count < 2U) ||
		(point_count > APP_BASELINE_CALIBRATION_MAX_POINTS))
	{
		return AppBaselineRuntime_ConvertAngleToTemperature(angle_rad);
	}

	{
		const AppBaselineRuntime_CalibrationPoint_t *hot_anchor =
			&profile->calibration_points[0U];
		const AppBaselineRuntime_CalibrationPoint_t *cold_anchor =
			&profile->calibration_points[point_count - 1U];
		const float hot_fraction =
			AppBaselineRuntime_ConvertAngleToFraction(
				hot_anchor->angle_deg * (APP_BASELINE_PI / 180.0f));
		const float cold_fraction =
			AppBaselineRuntime_ConvertAngleToFraction(
				cold_anchor->angle_deg * (APP_BASELINE_PI / 180.0f));
		const float denominator = cold_fraction - hot_fraction;

		if (fabsf(denominator) <= 1.0e-6f)
		{
			return AppBaselineRuntime_ConvertAngleToTemperature(angle_rad);
		}

		return hot_anchor->temperature_c +
			((cold_anchor->temperature_c - hot_anchor->temperature_c) *
			 ((target_fraction - hot_fraction) / denominator));
	}
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
 * @brief Compute a strong prior that favors the middle of the sweep.
 *
 * The Hough-style vote should prefer the spoke family that traverses the
 * center of the dial, because boundary clutter can otherwise win by sheer
 * density at the wrap-around ends.
 */
static float AppBaselineRuntime_ComputeSweepCenterWeight(float angle_rad)
{
	const float sweep_fraction = AppBaselineRuntime_ConvertAngleToFraction(angle_rad);
	const float center_wave = sinf(APP_BASELINE_PI * sweep_fraction);
	const float center_wave_sq = center_wave * center_wave;

	return 0.01f + (0.99f * center_wave_sq * center_wave_sq *
					center_wave_sq * center_wave_sq);
}

/**
 * @brief Measure how persistent one peak is in a small local neighborhood.
 *
 * Classical Hough voting tends to be more reliable when a peak remains
 * prominent after small perturbations instead of only winning as a razor-thin
 * local spike. This suppresses broad endpoint clutter and favors sharper line
 * families that behave more like a real pointer.
 */
static float AppBaselineRuntime_ComputePeakPersistenceWeight(
	const float *peak_values, size_t num_bins, size_t peak_index,
	size_t neighborhood_bins)
{
	float peak_value = 0.0f;
	float neighborhood_sum = 0.0f;
	size_t neighborhood_count = 0U;
	const size_t start_index =
		(peak_index > neighborhood_bins) ? (peak_index - neighborhood_bins) : 0U;
	const size_t end_index =
		((peak_index + neighborhood_bins) < num_bins)
			? (peak_index + neighborhood_bins)
			: ((num_bins > 0U) ? (num_bins - 1U) : 0U);

	if ((peak_values == NULL) || (num_bins == 0U) || (peak_index >= num_bins))
	{
		return 0.0f;
	}

	peak_value = peak_values[peak_index];
	if (peak_value <= 0.0f)
	{
		return 0.0f;
	}

	for (size_t index = start_index; index <= end_index; ++index)
	{
		if (index == peak_index)
		{
			continue;
		}

		neighborhood_sum += peak_values[index];
		++neighborhood_count;
	}

	if (neighborhood_count == 0U)
	{
		return 1.0f;
	}

	{
		const float neighborhood_mean =
			neighborhood_sum / (float)neighborhood_count;
		const float prominence = peak_value / (neighborhood_mean + 1.0e-6f);
		const float sharpness = peak_value / ((neighborhood_sum / (float)neighborhood_count) + 1.0e-6f);
		const float combined =
			0.5f + (0.3f * AppBaselineRuntime_ClampFloat(prominence, 0.0f, 4.0f)) +
			(0.2f * AppBaselineRuntime_ClampFloat(sharpness, 0.0f, 4.0f));

		return AppBaselineRuntime_ClampFloat(combined, 0.25f, 2.5f);
	}
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
 * @brief Read the minimum (darkest) luma in a 3x3 neighbourhood.
 * Equivalent to 1-iteration morphological dilation of dark features.
 * Thickens thin dark lines (needles) by ~1 px each side so that spoke
 * sampling at a slightly wrong angle still hits the needle.
 */
static float AppBaselineRuntime_ReadLumaMin3x3(const uint8_t *frame_bytes,
	size_t frame_width, size_t frame_height, long cx, long cy)
{
	float min_luma = 255.0f;
	for (long dy = -1; dy <= 1; dy++)
	{
		const long sy = cy + dy;
		if (sy < 0 || (size_t)sy >= frame_height) continue;
		for (long dx = -1; dx <= 1; dx++)
		{
			const long sx = cx + dx;
			if (sx < 0 || (size_t)sx >= frame_width) continue;
			const float luma = AppBaselineRuntime_ReadLuma(
				frame_bytes, frame_width, (size_t)sx, (size_t)sy);
			if (luma < min_luma) min_luma = luma;
		}
	}
	return min_luma;
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
	/* Bias the vote toward the inner shaft of the needle. The center-adjacent
	 * segment is the most stable cue on this board, while far-out spoke clutter
	 * and rim markings are more likely to drift into a false family. We now
	 * bias even harder toward the hub so the score behaves like a radial
	 * spoke detector instead of a generic long-line scorer. */
	/* Use the full visible shaft band rather than a single hub-adjacent sample.
	 * A narrow hub prior was allowing unrelated vertical artwork to win when
	 * the real needle was near the 0/360-degree wrap. */
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

	/* Apply only a modest boundary penalty. The physical needle can legitimately
	 * sit near either endpoint, so endpoint position is not evidence of a bad
	 * candidate by itself. */
	float boundary_weight = 1.0f;
	const float boundary_margin = 0.05f * sweep_rad;
	if (shifted < boundary_margin || shifted > (sweep_rad - boundary_margin))
	{
		boundary_weight = 0.70f;
	}

	/* The gauge angle convention uses mathematical coordinates where +Y is up.
	 * In image coordinates +Y is down, so we negate sin(angle) to match the
	 * Hough module's convention. This fixes the angle detection bug where the
	 * baseline was detecting angles 180° off from the correct needle position. */
	const float unit_dx = cosf(angle_rad);
	const float unit_dy = -sinf(angle_rad);
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
	float inner_shaft_score = 0.0f;
	size_t inner_shaft_count = 0U;
	size_t valid_sample_count = 0U;

	for (size_t sample_index = 0U; sample_index < APP_BASELINE_RAY_SAMPLES;
		 ++sample_index)
	{
		const float radius = start_radius + (radius_step * (float)sample_index);
		const float sample_progress = (float)sample_index / (float)(APP_BASELINE_RAY_SAMPLES - 1U);
		/* Prefer the cleaner inner shaft over the noisy hub and tip/tick region.
		 * The real needle is easiest to lock when the line stays dark close to
		 * the center, so we push the vote toward those samples and down-weight
		 * the outer shaft more aggressively. */
		const float shaft_weight = AppBaselineRuntime_MiddleShaftWeight(sample_progress);
		const float shaft_focus = shaft_weight * shaft_weight;
		const float weight = 0.02f + (0.98f * shaft_focus * shaft_focus);
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
			if (sample_progress <= 0.28f)
			{
				inner_shaft_score += local_contrast;
				++inner_shaft_count;
			}
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
	float inner_shaft_mean = mean;
	if (inner_shaft_count > 0U)
	{
		inner_shaft_mean = inner_shaft_score / (float)inner_shaft_count;
	}
	/* Give extra reward to rays that stay dark right after the hub. This is
	 * a classical spoke cue: the true pointer should be visible where it
	 * leaves the center, before outer clutter and rim markings take over. */
	const float inner_shaft_boost =
		AppBaselineRuntime_ClampFloat(
			inner_shaft_mean / (mean + 1e-6f), 0.0f, 2.5f);
	/* Do not use a forward/backward sign cue. A dark needle, hub, and printed
	 * radial spoke can make that cue prefer the opposite image ray; the legal
	 * Celsius sweep already resolves the 180-degree ambiguity. */
	return (mean * linearity * boundary_weight *
			(0.35f + (0.65f * inner_shaft_boost)));
}

/**
 * @brief Score one complete classical needle hypothesis.
 *
 * The polar edge vote is useful for finding candidates, but it can promote a
 * cold bezel edge when the real needle has a weaker edge response. This score
 * evaluates every legal angle using independent ray contrast, shaft
 * continuity, hub connection, and tip extension cues.
 */
static float AppBaselineRuntime_ScoreNeedleCandidate(
	const uint8_t *frame_bytes, size_t frame_width_pixels,
	size_t frame_height_pixels, size_t center_x, size_t center_y,
	float dial_radius_px, float angle_rad)
{
	const float ray_score = AppBaselineRuntime_ScoreAngle(
		frame_bytes, frame_width_pixels, frame_height_pixels,
		center_x, center_y, angle_rad);
	const float cos_a = cosf(angle_rad);
	const float sin_a = -sinf(angle_rad);  /* Negate for image Y convention */
	float continuity = 0.0f;
	float hub_darkness = 0.0f;
	float tip_darkness = 0.0f;
	size_t continuity_count = 0U;

	if (ray_score <= 0.0f)
	{
		return 0.0f;
	}

	for (size_t sample_index = 0U; sample_index < 20U; ++sample_index)
	{
		const float radius_fraction =
			0.15f + (0.70f * (float)sample_index / 19.0f);
		const long sample_x = AppBaselineRuntime_RoundToLong(
			(float)center_x + (cos_a * radius_fraction * dial_radius_px));
		const long sample_y = AppBaselineRuntime_RoundToLong(
			(float)center_y + (sin_a * radius_fraction * dial_radius_px));

		if ((sample_x >= 0L) && (size_t)sample_x < frame_width_pixels &&
			(sample_y >= 0L) && (size_t)sample_y < frame_height_pixels)
		{
			const float luma = AppBaselineRuntime_ReadLumaMin3x3(
				frame_bytes, frame_width_pixels, frame_height_pixels,
				sample_x, sample_y);
			continuity += (255.0f - luma) / 255.0f;
			continuity_count++;
		}
	}

	for (size_t sample_index = 0U; sample_index < 4U; ++sample_index)
	{
		const float radius_fraction =
			0.08f + (0.20f * (float)sample_index / 3.0f);
		const long sample_x = AppBaselineRuntime_RoundToLong(
			(float)center_x + (cos_a * radius_fraction * dial_radius_px));
		const long sample_y = AppBaselineRuntime_RoundToLong(
			(float)center_y + (sin_a * radius_fraction * dial_radius_px));

		if ((sample_x >= 0L) && (size_t)sample_x < frame_width_pixels &&
			(sample_y >= 0L) && (size_t)sample_y < frame_height_pixels)
		{
			const float luma = AppBaselineRuntime_ReadLumaMin3x3(
				frame_bytes, frame_width_pixels, frame_height_pixels,
				sample_x, sample_y);
			hub_darkness += (255.0f - luma) / 255.0f;
		}
	}

	for (size_t sample_index = 0U; sample_index < 6U; ++sample_index)
	{
		const float radius_fraction =
			0.68f + (0.25f * (float)sample_index / 5.0f);
		const long sample_x = AppBaselineRuntime_RoundToLong(
			(float)center_x + (cos_a * radius_fraction * dial_radius_px));
		const long sample_y = AppBaselineRuntime_RoundToLong(
			(float)center_y + (sin_a * radius_fraction * dial_radius_px));

		if ((sample_x >= 0L) && (size_t)sample_x < frame_width_pixels &&
			(sample_y >= 0L) && (size_t)sample_y < frame_height_pixels)
		{
			const float luma = AppBaselineRuntime_ReadLumaMin3x3(
				frame_bytes, frame_width_pixels, frame_height_pixels,
				sample_x, sample_y);
			tip_darkness += (255.0f - luma) / 255.0f;
		}
	}

	if (continuity_count == 0U)
	{
		return 0.0f;
	}

	continuity /= (float)continuity_count;
	hub_darkness /= 4.0f;
	tip_darkness /= 6.0f;

	/* The exponents make a candidate pay for a missing cue instead of winning
	 * from one very strong edge alone. All terms remain classical image cues. */
	const float continuity_boost = 0.20f + (1.80f * continuity);
	const float hub_boost = 0.20f + (1.80f * hub_darkness);
	const float tip_boost = 0.35f + (0.65f * tip_darkness);

	return ray_score * continuity_boost * continuity_boost *
		hub_boost * tip_boost;
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
bool AppBaselineRuntime_EstimatePolarNeedle(
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
	float selection_votes[APP_BASELINE_ANGLE_BINS] = {0.0f};
	float best_score = -1.0f;
	float runner_up_score = -1.0f;
	size_t best_bin = 0U;
	bool full_sweep_selected = false;
	const bool bright_relaxed = camera_baseline_current_frame_is_bright;
	const float edge_threshold = bright_relaxed ? 4.0f : 8.0f;
	const float main_continuity_threshold = bright_relaxed ? 0.14f : 0.35f;
	const float main_hub_threshold = bright_relaxed ? 0.10f : 0.25f;
	const float hot_continuity_threshold = bright_relaxed ? 0.14f : 0.28f;
	const float hot_hub_threshold = bright_relaxed ? 0.08f : 0.18f;
	const float final_spoke_continuity_threshold = bright_relaxed ? 0.08f : 0.20f;

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
						const float shaft_focus =
							AppBaselineRuntime_MiddleShaftWeight(sample_progress);
						const float shaft_weight = 0.10f +
												   (0.90f * shaft_focus * shaft_focus);

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
							const float spoke_score = ((hub_connection * 0.55f) +
													   (tip_extension * 0.30f) +
													   (width_score * 0.15f));

							/* HARD GATE: reject angles without hub connection.
							 * The needle must connect to the center hub.
							 * Dial markings don't reach the center. */
							if (hub_connection < 0.15f)
							{
								continue;
							}

							const float connection_boost =
								0.05f + (29.95f * spoke_score * spoke_score);

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
		/* Do not wrap smoothing across the sweep endpoints. The 270° gauge
		 * sweep is not a circular parameterization, so bin 0 and bin N-1 are
		 * opposite physical ends of the scale and must not reinforce each
		 * other. Circular smoothing was merging the cold and hot edge families
		 * into one false peak cluster. */
		float vote_sum = angle_votes[bin_index];
		float vote_count = 1.0f;

		if (bin_index > 0U)
		{
			vote_sum += angle_votes[bin_index - 1U];
			vote_count += 1.0f;
		}
		if ((bin_index + 1U) < APP_BASELINE_ANGLE_BINS)
		{
			vote_sum += angle_votes[bin_index + 1U];
			vote_count += 1.0f;
		}

		smoothed_votes[bin_index] = vote_sum / vote_count;
		{
			const float peak_persistence_prior =
				AppBaselineRuntime_ComputePeakPersistenceWeight(
					smoothed_votes, APP_BASELINE_ANGLE_BINS, bin_index, 4U);
			selection_votes[bin_index] =
				smoothed_votes[bin_index] * peak_persistence_prior;
		}
	}

	{
		float vote_sum = 0.0f;
		enum
		{
			/* Keep more candidate peaks so hot-wrap angles (near 30°-60°)
			 * are not dropped when their raw gradient vote is weaker than
			 * mid-range false positives. */
			APP_BASELINE_TOP_PEAK_COUNT = 24,
			/* Greedy non-maximum suppression window. This keeps one angular
			 * family from occupying the whole shortlist with adjacent bins and
			 * gives other spoke families a chance to compete. */
			APP_BASELINE_PEAK_SUPPRESSION_BINS = 8U
		};
		size_t top_bins[APP_BASELINE_TOP_PEAK_COUNT] = {0};
		float top_scores[APP_BASELINE_TOP_PEAK_COUNT] = {0.0f};
		uint8_t suppressed_bins[APP_BASELINE_ANGLE_BINS] = {0};

		for (size_t bin_index = 0U; bin_index < APP_BASELINE_ANGLE_BINS;
			 ++bin_index)
		{
			vote_sum += smoothed_votes[bin_index];
		}

		for (size_t peak_slot = 0U; peak_slot < APP_BASELINE_TOP_PEAK_COUNT;
			 ++peak_slot)
		{
			size_t best_bin_index = 0U;
			float best_bin_score = -1.0f;
			bool found_any = false;

			for (size_t bin_index = 0U; bin_index < APP_BASELINE_ANGLE_BINS;
				 ++bin_index)
			{
				const float val = selection_votes[bin_index];

				if (suppressed_bins[bin_index] != 0U)
				{
					continue;
				}

				if (!found_any || (val > best_bin_score))
				{
					best_bin_score = val;
					best_bin_index = bin_index;
					found_any = true;
				}
			}

			if (!found_any || (best_bin_score <= 0.0f))
			{
				break;
			}

			top_bins[peak_slot] = best_bin_index;
			top_scores[peak_slot] = best_bin_score;

			{
				const size_t suppress_start =
					(best_bin_index > APP_BASELINE_PEAK_SUPPRESSION_BINS)
						? (best_bin_index - APP_BASELINE_PEAK_SUPPRESSION_BINS)
						: 0U;
				const size_t suppress_end =
					((best_bin_index + APP_BASELINE_PEAK_SUPPRESSION_BINS) <
					 APP_BASELINE_ANGLE_BINS)
						? (best_bin_index + APP_BASELINE_PEAK_SUPPRESSION_BINS)
						: (APP_BASELINE_ANGLE_BINS - 1U);

				for (size_t suppress_bin = suppress_start;
					 suppress_bin <= suppress_end; ++suppress_bin)
				{
					suppressed_bins[suppress_bin] = 1U;
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

#if APP_BASELINE_DEBUG_SELECTION
		{
			const size_t dump_count = 6U;
			for (size_t peak_idx = 0U;
				 (peak_idx < dump_count) &&
				 (peak_idx < APP_BASELINE_TOP_PEAK_COUNT) &&
				 (top_scores[peak_idx] > 0.0f);
				 ++peak_idx)
			{
				const size_t peak_bin = top_bins[peak_idx];
				const float peak_angle_deg =
					(min_angle_rad +
					 ((float)peak_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) *
						 sweep_rad) *
					(180.0f / APP_BASELINE_PI);
				const long peak_angle_x10 =
					AppBaselineRuntime_RoundToLong(peak_angle_deg * 10.0f);
				const long peak_score_m =
					AppBaselineRuntime_RoundToLong(top_scores[peak_idx] * 1000.0f);
				DebugConsole_Printf(
					"[BASELINE][DBG] top-peak[%lu]: bin=%lu angle=%ld.%01ld score=%ld.%03ld\r\n",
					(unsigned long)peak_idx,
					(unsigned long)peak_bin,
					(long)(peak_angle_x10 / 10L),
					((peak_angle_x10 % 10L) < 0L) ? -(peak_angle_x10 % 10L)
											   : (peak_angle_x10 % 10L),
					(long)(peak_score_m / 1000L),
					(long)(peak_score_m % 1000L));
			}
		}
#endif

		/* Use spoke continuity to select the best peak. The needle forms a
		 * continuous dark spoke from center to edge, while dial markings are
		 * isolated edges with poor continuity. Weight each peak by both
		 * vote score and continuity to find the true needle. */
		best_bin = top_bins[0];
		best_score = selection_votes[best_bin];

		/* Classical spoke-continuity weighted peak selection.
		 * Analyze top peaks and pick the one with best continuity-weighted score.
		 * This ensures we select the needle (continuous spoke) over dial
		 * markings (isolated edges with high vote but poor continuity).
		 * We intentionally cap this at APP_BASELINE_TOP_PEAK_COUNT because the
		 * candidate arrays are fixed-size. */
		size_t best_weighted_bin = best_bin;
		{
			float best_weighted_score = -1.0f;

			for (size_t peak_idx = 0U; peak_idx < APP_BASELINE_TOP_PEAK_COUNT && top_scores[peak_idx] > 0.0f; ++peak_idx)
			{
				const size_t candidate_bin = top_bins[peak_idx];
				const float candidate_angle = min_angle_rad + ((float)candidate_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
				const float cos_a = cosf(candidate_angle);
				const float sin_a = -sinf(candidate_angle);  /* Negate for image Y convention */

				/* Classical CV: measure spoke continuity along this angle.
				 * The needle forms a continuous dark spoke from center to edge.
				 * Dial markings are isolated edges without continuity.
				 * Use 3x3 min-filter (dilation) so thin needles are caught
				 * even when the angle is off by 1-2 degrees. */
				float continuity = 0.0f;
				float tip_extension = 0.0f;
				float width_score = 1.0f;
				float tip_extension_sum = 0.0f;
				float width_score_sum = 0.0f;
				float line_contrast_sum = 0.0f;
				const size_t continuity_samples = 12U;
				size_t valid_samples = 0U;
				size_t line_contrast_samples = 0U;

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
						const float sample_luma = AppBaselineRuntime_ReadLumaMin3x3(
							frame_bytes, frame_width_pixels, frame_height_pixels, sx, sy);
						continuity += ((255.0f - sample_luma) / 255.0f);
						valid_samples++;

						/* Tip-extension: the true needle stays dark further out toward
						 * the rim, while broad dial artwork tends to fade sooner. */
						{
							const size_t tip_samples = 5U;
							float tip_darkness = 0.0f;
							for (size_t t = 0U; t < tip_samples; ++t)
							{
								const float tip_frac =
									0.70f + (0.25f * (float)t / (float)(tip_samples - 1U));
								const long tx = AppBaselineRuntime_RoundToLong(
									(float)center_x + (cos_a * tip_frac * dial_radius_px));
								const long ty = AppBaselineRuntime_RoundToLong(
									(float)center_y + (sin_a * tip_frac * dial_radius_px));
								if (tx >= 0 && (size_t)tx < frame_width_pixels &&
									ty >= 0 && (size_t)ty < frame_height_pixels)
								{
									const float tip_luma = AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels, (size_t)tx, (size_t)ty);
									tip_darkness += ((255.0f - tip_luma) / 255.0f);
								}
							}
							tip_extension_sum += (tip_darkness / (float)tip_samples);
						}

						const float perp_x = -sin_a;
						const float perp_y = cos_a;
						/* Width score: a real needle is thin, so neighboring rays should
						 * stay lighter than the center ray. This suppresses broad dial
						 * strokes that can still look continuous. */
						{
							float perp_darkness = 0.0f;
							const size_t width_samples = 5U;
							for (size_t w = 0U; w < width_samples; ++w)
							{
								const float offset = (float)(w - width_samples / 2U) * 1.5f;
								const long wx = AppBaselineRuntime_RoundToLong(
									(float)sx + perp_x * offset);
								const long wy = AppBaselineRuntime_RoundToLong(
									(float)sy + perp_y * offset);
								if (wx >= 0 && (size_t)wx < frame_width_pixels &&
									wy >= 0 && (size_t)wy < frame_height_pixels)
								{
									const float w_luma = AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels, (size_t)wx, (size_t)wy);
									perp_darkness += ((255.0f - w_luma) / 255.0f);
								}
							}
							{
								const float center_darkness = ((255.0f - sample_luma) / 255.0f);
								const float avg_perp_darkness = perp_darkness / (float)width_samples;
								const float sample_width_score =
									0.3f + (0.7f * (center_darkness / (avg_perp_darkness + 0.01f)));
								width_score_sum += sample_width_score;
								if (width_score_sum > (float)continuity_samples)
								{
									width_score_sum = (float)continuity_samples;
								}
							}
						}

						/* Thin-line cue: a pointer is a dark centerline bordered by
						 * brighter dial pixels. Broad printed spokes remain dark at the
						 * perpendicular offsets, so they lose this contrast score. */
						{
							const float center_luma = AppBaselineRuntime_ReadLuma(
								frame_bytes, frame_width_pixels,
								(size_t)sx, (size_t)sy);
							float side_luma_sum = 0.0f;
							size_t side_luma_count = 0U;

							for (size_t side_index = 0U; side_index < 2U; ++side_index)
							{
								const float side_offset = 3.0f + (2.0f * (float)side_index);
								const long side_plus_x = AppBaselineRuntime_RoundToLong(
									(float)sx + (perp_x * side_offset));
								const long side_plus_y = AppBaselineRuntime_RoundToLong(
									(float)sy + (perp_y * side_offset));
								const long side_minus_x = AppBaselineRuntime_RoundToLong(
									(float)sx - (perp_x * side_offset));
								const long side_minus_y = AppBaselineRuntime_RoundToLong(
									(float)sy - (perp_y * side_offset));

								if ((side_plus_x >= 0L) && (side_plus_y >= 0L) &&
									((size_t)side_plus_x < frame_width_pixels) &&
									((size_t)side_plus_y < frame_height_pixels))
								{
									side_luma_sum += AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels,
										(size_t)side_plus_x, (size_t)side_plus_y);
									side_luma_count++;
								}
								if ((side_minus_x >= 0L) && (side_minus_y >= 0L) &&
									((size_t)side_minus_x < frame_width_pixels) &&
									((size_t)side_minus_y < frame_height_pixels))
								{
									side_luma_sum += AppBaselineRuntime_ReadLuma(
										frame_bytes, frame_width_pixels,
										(size_t)side_minus_x, (size_t)side_minus_y);
									side_luma_count++;
								}
							}

							if (side_luma_count > 0U)
							{
								const float side_luma_mean =
									side_luma_sum / (float)side_luma_count;
								const float contrast = AppBaselineRuntime_ClampFloat(
									(side_luma_mean - center_luma) / 255.0f,
									0.0f, 1.0f);
								line_contrast_sum += contrast;
								line_contrast_samples++;
							}
						}
					}
				}

				if (valid_samples > 0U)
				{
					tip_extension = tip_extension_sum / (float)valid_samples;
					width_score = width_score_sum / (float)valid_samples;
					if (width_score > 1.0f)
					{
						width_score = 1.0f;
					}
				}
				const float line_contrast =
					(line_contrast_samples > 0U)
						? (line_contrast_sum / (float)line_contrast_samples)
						: 0.0f;

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
						/* Weight by continuity, hub connection, tip extension, and
						 * spoke thinness so broad dial artwork does not outrank the
						 * actual needle just because it has a strong gradient edge. */
						const float tip_boost = 0.35f + (0.65f * tip_extension);
						const float width_boost = 0.30f + (0.70f * width_score);
						const float ray_score = AppBaselineRuntime_ScoreAngle(
							frame_bytes,
							frame_width_pixels,
							frame_height_pixels,
							center_x,
							center_y,
							candidate_angle);
						const float ray_score_boost =
							AppBaselineRuntime_ClampFloat(
								ray_score / (1.0f + ray_score), 0.0f, 1.0f);
						const float peak_persistence_boost =
							AppBaselineRuntime_ComputePeakPersistenceWeight(
								selection_votes, APP_BASELINE_ANGLE_BINS,
								candidate_bin, 4U);
						const float support_ratio =
							(top_scores[0] > 0.0f) ?
								(top_scores[peak_idx] / top_scores[0]) :
								0.0f;
						const float support_boost =
							0.25f + (0.75f * AppBaselineRuntime_ClampFloat(
																	 support_ratio, 0.0f, 1.0f));
						const float thin_line_boost = 0.25f +
							(1.75f * line_contrast);
						const float weighted_score =
							(continuity * continuity) * hub_darkness * tip_boost *
							width_boost * ray_score_boost * support_boost *
							peak_persistence_boost * thin_line_boost * thin_line_boost;
						if (weighted_score > best_weighted_score)
						{
							best_weighted_score = weighted_score;
							best_weighted_bin = candidate_bin;
						}
					}
				}
			}
		}

		/* Keep the raw vote and refinement scores in the same candidate family.
		 * A center/continuity cue is only a tie-breaker; if it selects a line
		 * with materially weaker polar support, retain the raw polar maximum so
		 * the acceptance gate compares like-for-like scores. */
		{
			const float strongest_vote = selection_votes[top_bins[0]];
			const float refined_vote = selection_votes[best_weighted_bin];
			if ((strongest_vote > 0.0f) &&
				(refined_vote < (strongest_vote *
					APP_BASELINE_MIN_REFINED_SUPPORT_RATIO)))
			{
				best_weighted_bin = top_bins[0];
			}
		}
		best_bin = best_weighted_bin;
		best_score = selection_votes[best_bin];

		/* The old global rescue is intentionally disabled. Its independent
		 * darkness score consistently selected the 32-degree dial marking on
		 * hot-end frames, overriding the stronger polar-vote evidence. */
#if APP_BASELINE_ENABLE_GLOBAL_SCORE_RESCUE
		{
			float global_best_score =
				AppBaselineRuntime_ScoreNeedleCandidate(
					frame_bytes, frame_width_pixels, frame_height_pixels,
					center_x, center_y, dial_radius_px,
					min_angle_rad + ((float)best_bin /
						(float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad);
			size_t global_best_bin = best_bin;
			float global_runner_up_score = 0.0f;

			for (size_t candidate_bin = 0U;
				 candidate_bin < APP_BASELINE_ANGLE_BINS; ++candidate_bin)
			{
				const float candidate_angle =
					min_angle_rad + ((float)candidate_bin /
						(float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
				const float candidate_score =
					AppBaselineRuntime_ScoreNeedleCandidate(
						frame_bytes, frame_width_pixels, frame_height_pixels,
						center_x, center_y, dial_radius_px, candidate_angle);

				if (candidate_score > global_best_score)
				{
					global_runner_up_score = global_best_score;
					global_best_score = candidate_score;
					global_best_bin = candidate_bin;
				}
				else if (candidate_score > global_runner_up_score)
				{
					global_runner_up_score = candidate_score;
				}
			}

			if (global_best_score > 0.0f)
			{
				best_bin = global_best_bin;
				best_score = global_best_score;
				runner_up_score = global_runner_up_score;
				full_sweep_selected = true;
			}
		}
#endif

#if APP_BASELINE_ENABLE_FACE_RAY_RESCUE
		/* Use the complete classical needle score for the bright-center sweep. It
		 * combines local contrast with shaft continuity, hub connection, and tip
		 * evidence so printed radial marks do not win on darkness alone. */
		if ((strcmp(source_label, "bright-center-polar") == 0) ||
			(strcmp(source_label, "face-center-polar") == 0))
		{
			float ray_best_score = 0.0f;
			float ray_runner_up_score = 0.0f;
			size_t ray_best_bin = best_bin;
			for (size_t candidate_bin = 0U;
				 candidate_bin < APP_BASELINE_ANGLE_BINS; ++candidate_bin)
			{
				const float candidate_angle =
					min_angle_rad + ((float)candidate_bin /
						(float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
				const float candidate_score = AppBaselineRuntime_ScoreNeedleCandidate(
					frame_bytes, frame_width_pixels, frame_height_pixels,
					center_x, center_y, dial_radius_px, candidate_angle);
				if (candidate_score > ray_best_score)
				{
					ray_runner_up_score = ray_best_score;
					ray_best_score = candidate_score;
					ray_best_bin = candidate_bin;
				}
				else if (candidate_score > ray_runner_up_score)
				{
					ray_runner_up_score = candidate_score;
				}
			}

			if (ray_best_score > 0.0f)
			{
				best_bin = ray_best_bin;
				best_score = ray_best_score;
				runner_up_score = ray_runner_up_score;
				full_sweep_selected = true;
			}
		}
#endif

		/* The editable diagram permits a guarded hot-zone rescue near the sweep
		 * wrap. Keep the existing continuity, hub, vote-ratio, and cold-history
		 * checks together so this branch remains a secondary recovery path. */
#if APP_BASELINE_ENABLE_HOT_ZONE_RESCUE
		{
			float best_hot_weighted = 0.0f;
			size_t best_hot_bin = 0U;
			float best_angle_deg_check = 0.0f;
			(void)bright_relaxed;
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
						const float sin_a = -sinf(candidate_angle);  /* Negate for image Y convention */
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
				 * When the primary maps to a cold temperature the needle is
				 * well within the mid-sweep range where gradient signal is
				 * strong, so require a higher vote ratio to avoid replacing
				 * a genuine cold needle with a false hot-zone peak from
				 * dial artwork. */
				const float primary_angle_rad_ho =
					min_angle_rad + ((float)best_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
				const float primary_temp_c =
					AppBaselineRuntime_ConvertAngleToTemperature(primary_angle_rad_ho);
				best_angle_deg_check = primary_angle_rad_ho * (180.0f / APP_BASELINE_PI);
				const float hot_override_min_ratio =
					(primary_temp_c <= APP_BASELINE_HOT_OVERRIDE_COLD_PRIMARY_THRESHOLD_C)
						? APP_BASELINE_HOT_OVERRIDE_COLD_VOTE_RATIO
						: 0.35f;
				if (best_hot_weighted > 0.0f &&
					smoothed_votes[best_hot_bin] >= (hot_override_min_ratio * smoothed_votes[best_bin]))
				{
					/* Hysteresis: if recent stable history is cold, do not allow
					 * a warm/hot jump from a marginal hot-zone candidate. */
					bool allow_hot_override = true;
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
					{
						const float hot_angle_rad =
							min_angle_rad + ((float)best_hot_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U)) * sweep_rad;
						const float hot_temp_c =
							AppBaselineRuntime_ConvertAngleToTemperature(hot_angle_rad);
						const float hot_vote_ratio =
							(smoothed_votes[best_bin] > 0.0f)
								? (smoothed_votes[best_hot_bin] / smoothed_votes[best_bin])
								: 0.0f;
						if (camera_baseline_last_result_valid &&
							camera_baseline_last_temperature_c <= APP_BASELINE_HOT_OVERRIDE_COLD_HISTORY_TEMP_C &&
							hot_temp_c >= APP_BASELINE_HOT_OVERRIDE_WARM_TARGET_TEMP_C &&
							hot_vote_ratio < APP_BASELINE_HOT_OVERRIDE_COLD_HISTORY_MIN_RATIO)
						{
							allow_hot_override = false;
						}
					}
					if (allow_hot_override)
					{
						best_bin = best_hot_bin;
						best_score = smoothed_votes[best_bin];
						DebugConsole_Printf(
							"[BASELINE] Hot-zone override: primary=%ld.%01lddeg hot=%ld.%01lddeg\r\n",
							primary_angle_x10 / 10L,
							((primary_angle_x10 % 10L) < 0L) ? -(primary_angle_x10 % 10L) : (primary_angle_x10 % 10L),
							hot_angle_x10 / 10L,
							((hot_angle_x10 % 10L) < 0L) ? -(hot_angle_x10 % 10L) : (hot_angle_x10 % 10L));
					}
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
#endif

		/* The full-sweep rescue remains separate from the diagram's hot-zone
		 * wrap rescue and stays disabled because it uses a different score family. */

		if (!full_sweep_selected)
		{
			runner_up_score = AppBaselineRuntime_RunnerUpPeakAfterSuppression(
				selection_votes, APP_BASELINE_ANGLE_BINS, best_bin, 15);
		}

		{
			const size_t prev_index =
				(best_bin + APP_BASELINE_ANGLE_BINS - 1U) % APP_BASELINE_ANGLE_BINS;
			const size_t next_index = (best_bin + 1U) % APP_BASELINE_ANGLE_BINS;
			const float prev_vote = angle_votes[prev_index];
			const float best_vote = angle_votes[best_bin];
			const float next_vote = angle_votes[next_index];
			const float vote_window = prev_vote + best_vote + next_vote;
			float refined_fraction = (float)best_bin / (float)(APP_BASELINE_ANGLE_BINS - 1U);

			if ((vote_window > 0.0f) && !full_sweep_selected)
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
				float outer_forward = 0.0f;
				float outer_backward = 0.0f;
				size_t count = 0U;
				size_t outer_count = 0U;
				const float cos_a = cosf(best_angle);
				const float sin_a = -sinf(best_angle);  /* Negate for image Y convention */

				for (float r_frac = 0.2f; r_frac <= 0.8f; r_frac += 0.1f)
				{
					const long fx = AppBaselineRuntime_RoundToLong((float)center_x + (cos_a * r_frac * dial_radius_px));
					const long fy = AppBaselineRuntime_RoundToLong((float)center_y + (sin_a * r_frac * dial_radius_px));
					const long bx = AppBaselineRuntime_RoundToLong((float)center_x - (cos_a * r_frac * dial_radius_px));
					const long by = AppBaselineRuntime_RoundToLong((float)center_y - (sin_a * r_frac * dial_radius_px));
					const float shaft_progress = (r_frac - 0.2f) / 0.6f;
					const float shaft_weight =
						0.05f + (0.95f * AppBaselineRuntime_MiddleShaftWeight(shaft_progress) *
								AppBaselineRuntime_MiddleShaftWeight(shaft_progress));

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

				/* The pointer's long side should remain dark near the rim, while
				 * the counterweight or a printed spoke usually terminates earlier. */
				for (float r_frac = 0.65f; r_frac <= 0.95f; r_frac += 0.10f)
				{
					const long fx = AppBaselineRuntime_RoundToLong(
						(float)center_x + (cos_a * r_frac * dial_radius_px));
					const long fy = AppBaselineRuntime_RoundToLong(
						(float)center_y + (sin_a * r_frac * dial_radius_px));
					const long bx = AppBaselineRuntime_RoundToLong(
						(float)center_x - (cos_a * r_frac * dial_radius_px));
					const long by = AppBaselineRuntime_RoundToLong(
						(float)center_y - (sin_a * r_frac * dial_radius_px));

					if ((fx >= 0L) && (fy >= 0L) &&
						((size_t)fx < frame_width_pixels) &&
						((size_t)fy < frame_height_pixels))
					{
						outer_forward += (255.0f -
							AppBaselineRuntime_ReadLumaMin3x3(
								frame_bytes, frame_width_pixels, frame_height_pixels,
								fx, fy)) / 255.0f;
					}
					if ((bx >= 0L) && (by >= 0L) &&
						((size_t)bx < frame_width_pixels) &&
						((size_t)by < frame_height_pixels))
					{
						outer_backward += (255.0f -
							AppBaselineRuntime_ReadLumaMin3x3(
								frame_bytes, frame_width_pixels, frame_height_pixels,
								bx, by)) / 255.0f;
					}
					outer_count++;
				}

				/* The two directions of a dark needle share the same line score, so
				 * validity alone cannot orient the axis. Compare both rays even when
				 * the current angle is inside the Celsius sweep, then accept the flip
				 * only when the opposite direction is also a valid gauge direction. */
				{
					const float opposite_angle = best_angle + APP_BASELINE_PI;
					const float opposite_angle_deg =
						AppBaselineRuntime_NormalizeAngleDegrees(
							opposite_angle * (180.0f / APP_BASELINE_PI));
					const bool opposite_is_valid =
						AppBaselineRuntime_IsAngleInCelsiusSweep(opposite_angle_deg) &&
						!AppBaselineRuntime_IsAngleInSubdialBand(opposite_angle_deg);
					const float flip_margin = 0.25f * (float)count;
					const float outer_flip_margin =
						0.10f * (float)outer_count;

					if (opposite_is_valid &&
						((score_backward > (score_forward + flip_margin)) ||
						 ((outer_backward > (outer_forward + outer_flip_margin)) &&
						  (outer_count > 0U))))
					{
						best_angle = opposite_angle;
						while (best_angle >= APP_BASELINE_TWO_PI)
						{
							best_angle -= APP_BASELINE_TWO_PI;
						}
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
			 * a continuous spoke. Needle should have dark continuity.
			 * Use 3x3 min-filter (dilation) so thin needles register even when
			 * the detected angle is off by 1-2 degrees from the true needle. */
			{
				const float cos_a = cosf(best_angle);
				const float sin_a = -sinf(best_angle);  /* Negate for image Y convention */
				float spoke_continuity = 0.0f;
				float raw_spoke_continuity = 0.0f;
				float chroma_u_sum = 0.0f, chroma_v_sum = 0.0f;
				float chroma_u_sq_sum = 0.0f, chroma_v_sq_sum = 0.0f;
				size_t chroma_samples = 0U;
				const size_t continuity_samples = 20U;
				size_t valid_samples = 0U;

				for (size_t i = 0U; i < continuity_samples; ++i)
				{
					const float r_frac = 0.15f + (0.70f * (float)i / (float)(continuity_samples - 1U));
					const float sample_progress = (float)i / (float)(continuity_samples - 1U);
					const long sx = AppBaselineRuntime_RoundToLong(
						(float)center_x + (cos_a * r_frac * dial_radius_px));
					const long sy = AppBaselineRuntime_RoundToLong(
						(float)center_y + (sin_a * r_frac * dial_radius_px));

					if (sx >= 0 && (size_t)sx < frame_width_pixels &&
						sy >= 0 && (size_t)sy < frame_height_pixels)
					{
						const float sample_luma = AppBaselineRuntime_ReadLumaMin3x3(
							frame_bytes, frame_width_pixels, frame_height_pixels, sx, sy);
						raw_spoke_continuity += ((255.0f - sample_luma) / 255.0f);
						{
							const float middle_focus =
								AppBaselineRuntime_MiddleShaftWeight(sample_progress);
							const float middle_weight =
								0.05f + (0.95f * middle_focus * middle_focus);
							spoke_continuity +=
								((255.0f - sample_luma) / 255.0f) * middle_weight;
						}
						valid_samples++;

						/* Chroma consistency: a black needle on a white dial face
						 * has near-neutral chroma (U≈128, V≈128) at every sample.
						 * A false peak crossing a coloured bezel/shadow edge has
						 * high chroma variance along the ray.  Track U/V sum and
						 * sum-of-squares to compute variance after the loop.
						 * Read chroma at the original (non-dilated) centre so we
						 * measure the actual pixel colour, not a dilated neighbour. */
						{
							float u_val = 128.0f, v_val = 128.0f;
							AppBaselineRuntime_ReadChroma(frame_bytes,
								frame_width_pixels, (size_t)sx, (size_t)sy,
								&u_val, &v_val);
							chroma_u_sum += u_val;
							chroma_v_sum += v_val;
							chroma_u_sq_sum += u_val * u_val;
							chroma_v_sq_sum += v_val * v_val;
							chroma_samples++;
						}
					}
				}

				if (valid_samples > 0U)
				{
					spoke_continuity /= (float)valid_samples;
					raw_spoke_continuity /= (float)valid_samples;

					/* Chroma penalty: high U/V variance means this ray crosses
					 * coloured regions (bezel, shadow, dial markings), not a
					 * clean black-on-white needle.  Scale factor tuned so that
					 * neutral variance (~100-300) gives penalty ≈ 0.85-0.95
					 * and coloured variance (~1000+) penalises heavily. */
					float chroma_penalty = 1.0f;
					if (chroma_samples > 1U)
					{
						const float inv_n = 1.0f / (float)chroma_samples;
						const float u_mean = chroma_u_sum * inv_n;
						const float v_mean = chroma_v_sum * inv_n;
						const float u_var = (chroma_u_sq_sum * inv_n) - (u_mean * u_mean);
						const float v_var = (chroma_v_sq_sum * inv_n) - (v_mean * v_mean);
						/* Clamp to avoid negative-from-float-epsilon. */
						const float cv = (u_var > 0.0f ? u_var : 0.0f) +
										(v_var > 0.0f ? v_var : 0.0f);
						chroma_penalty = 1.0f / (1.0f + (cv / 2600.0f));
					}

					const float effective_continuity = raw_spoke_continuity * chroma_penalty;
					/* Require sufficient spoke darkness along the shaft.
					 * Bright-relaxed mode lowers this floor to tolerate washed-out
					 * needle contrast while still rejecting flat clutter. */
					if (effective_continuity < final_spoke_continuity_threshold)
					{
						const long continuity_m = AppBaselineRuntime_RoundToLong(
							raw_spoke_continuity * 1000.0f);
						const long chroma_m = AppBaselineRuntime_RoundToLong(
							chroma_penalty * 1000.0f);
						const long threshold_m = AppBaselineRuntime_RoundToLong(
							final_spoke_continuity_threshold * 1000.0f);
						DebugConsole_Printf(
							"[BASELINE] Polar reject: continuity=%ld/%ld chroma=%ld "
							"source=%s mode=%s\r\n",
							continuity_m, threshold_m, chroma_m,
							source_label,
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
			if (full_sweep_selected)
			{
				/* Full-sweep ray scores and polar votes have different units. Use
				 * the ray runner-up for this detector's confidence ratio. */
				estimate_out->confidence = AppBaselineRuntime_ClampFloat(
					best_score / (runner_up_score + 1e-6f), 0.0f, 1000.0f);
			}
			else
			{
				estimate_out->confidence = AppBaselineRuntime_ClampFloat(
					best_score / ((fabsf(vote_sum) /
						(float)APP_BASELINE_ANGLE_BINS) + 1e-6f),
					0.0f, 1000.0f);
			}
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
bool AppBaselineRuntime_EstimateDialCenterFromRimVotes(
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
	const AppBaselineRuntime_CalibrationProfile_t *profile = NULL;
	const char *profile_name = "board_celsius_v1";
	long angle_tenths = 0L;
	long confidence_thousandths = 0L;
	long score_whole = 0L;
	long runner_up_whole = 0L;
	long angle_abs_tenths = 0L;
	long confidence_abs_thousandths = 0L;
	long center_distance_tenths = 0L;
	long center_distance_abs_tenths = 0L;
	long calibrated_angle_tenths = 0L;
	long calibrated_angle_abs_tenths = 0L;
	long sweep_fraction_milli = 0L;
	long center_bias_thousandths = 0L;

	if ((estimate == NULL) || !estimate->valid)
	{
		return;
	}

	profile = AppBaselineRuntime_GetCalibrationProfile();
	if ((profile != NULL) && (profile->profile_name != NULL) &&
		(profile->profile_name[0] != '\0'))
	{
		profile_name = profile->profile_name;
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
	center_distance_tenths = AppBaselineRuntime_RoundToLong(
		AppBaselineRuntime_ComputeCenterDistancePixels(estimate) * 10.0f);
	center_distance_abs_tenths =
		(center_distance_tenths < 0L) ? -center_distance_tenths
									  : center_distance_tenths;
	{
		const AppBaselineRuntime_CalibrationProfile_t *active_profile =
			AppBaselineRuntime_GetCalibrationProfile();
		const float angle_offset_deg =
			(active_profile != NULL) ? active_profile->angle_offset_deg :
			APP_BASELINE_ANGLE_CALIBRATION_OFFSET_DEG;
		const float calibrated_angle_deg =
			(estimate->angle_rad * 180.0f / APP_BASELINE_PI) + angle_offset_deg;
		const float sweep_fraction =
			AppBaselineRuntime_ConvertAngleToFraction(
				estimate->angle_rad + (angle_offset_deg *
					(APP_BASELINE_PI / 180.0f)));

		calibrated_angle_tenths =
			AppBaselineRuntime_RoundToLong(calibrated_angle_deg * 10.0f);
		calibrated_angle_abs_tenths =
			(calibrated_angle_tenths < 0L) ? -calibrated_angle_tenths
										   : calibrated_angle_tenths;
		sweep_fraction_milli =
			AppBaselineRuntime_RoundToLong(sweep_fraction * 1000.0f);
	}
	center_bias_thousandths = AppBaselineRuntime_RoundToLong(
		AppBaselineRuntime_ComputeSweepCenterWeight(estimate->angle_rad) *
		1000.0f);
	/* Keep the normal baseline diagnostics compiled out, but leave one direct
	 * geometry marker so a flashed image can be distinguished from an older
	 * build and the selected classical ray can be audited on hardware. */
	{
		char compact_geometry_line[192] = {0};
		const long temperature_tenths = AppBaselineRuntime_RoundToLong(
			estimate->temperature_c * 10.0f);
		const long temperature_abs_tenths =
			(temperature_tenths < 0L) ? -temperature_tenths : temperature_tenths;
		const int compact_geometry_length = DebugConsole_Snprintf(
			compact_geometry_line, sizeof(compact_geometry_line),
			"[BASELINE][BUILD] gauge-angle-v15-editable-diagram src=%s center=(%lu,%lu) "
			"angle=%ld.%01lddeg temp=%ld.%01ldC ray30=%ld ray150=%ld\r\n",
			(estimate->source_label != NULL) ? estimate->source_label : "unknown",
			(unsigned long)estimate->center_x,
			(unsigned long)estimate->center_y,
			(long)(angle_tenths / 10L),
			(long)(angle_abs_tenths % 10L),
			(long)(temperature_tenths / 10L),
			(long)(temperature_abs_tenths % 10L),
			AppBaselineRuntime_RoundToLong(
				AppBaselineRuntime_ScoreAngle(
					camera_inference_frame_snapshot,
					CAMERA_CAPTURE_WIDTH_PIXELS,
					CAMERA_CAPTURE_HEIGHT_PIXELS,
					estimate->center_x, estimate->center_y,
					30.0f * (APP_BASELINE_PI / 180.0f)) * 1000.0f),
			AppBaselineRuntime_RoundToLong(
				AppBaselineRuntime_ScoreAngle(
					camera_inference_frame_snapshot,
					CAMERA_CAPTURE_WIDTH_PIXELS,
					CAMERA_CAPTURE_HEIGHT_PIXELS,
					estimate->center_x, estimate->center_y,
					150.0f * (APP_BASELINE_PI / 180.0f)) * 1000.0f));
		if (compact_geometry_length > 0)
		{
			const size_t bytes_to_write =
				((size_t)compact_geometry_length < sizeof(compact_geometry_line))
					? (size_t)compact_geometry_length
					: (sizeof(compact_geometry_line) - 1U);
			(void)DebugConsole_WriteBytes(
				(const uint8_t *)compact_geometry_line, bytes_to_write);
		}
	}
	DebugConsole_Printf(
		"[BASELINE] details: profile=%s center=(%lu,%lu) dcenter=%ld.%01ld source=%s angle=%ld.%01lddeg calibrated=%ld.%01lddeg fraction=%ld.%03ld bias=%ld.%03ld confidence=%ld.%03ld score=%ld runner_up=%ld\r\n",
		profile_name,
		(unsigned long)estimate->center_x,
		(unsigned long)estimate->center_y,
		(long)(center_distance_tenths / 10L),
		(long)(center_distance_abs_tenths % 10L),
		(estimate->source_label != NULL) ? estimate->source_label : "unknown",
		(long)(angle_tenths / 10L), (long)(angle_abs_tenths % 10L),
		(long)(calibrated_angle_tenths / 10L),
		(long)(calibrated_angle_abs_tenths % 10L),
		(long)(sweep_fraction_milli / 1000L),
		(long)(sweep_fraction_milli % 1000L),
		(long)(center_bias_thousandths / 1000L),
		(long)(center_bias_thousandths % 1000L),
		(long)(confidence_thousandths / 1000L),
		(long)(confidence_abs_thousandths % 1000L),
		score_whole, runner_up_whole);

	/* Log post-baseline power consumption and end metrics tracking */
	(void)INA219_LogReading("BASELINE-POST");
	Metrics_EndInference("BASELINE", estimate->temperature_c);
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


/**
 * @brief Retrieve the last classical baseline estimate and confidence.
 *
 * @param temp_out Pointer to store the temperature in Celsius.
 * @param confidence_out Pointer to store the confidence score.
 * @retval true if a valid estimate exists.
 * @retval false if no estimate has been produced yet.
 */
bool AppBaselineRuntime_GetLastEstimate(float *temp_out, float *confidence_out)
{
    if (!camera_baseline_last_result_valid)
    {
        return false;
    }
    if (temp_out != NULL)
    {
        *temp_out = camera_baseline_last_temperature_c;
    }
    if (confidence_out != NULL)
    {
        *confidence_out = camera_baseline_last_confidence;
    }
    return true;
}

/**
 * @brief Retrieve the version of the most recent accepted baseline estimate.
 */
ULONG AppBaselineRuntime_GetLastEstimateGeneration(void)
{
	return camera_baseline_last_result_generation;
}

/**
 * @brief Retrieve the version of the most recent queued baseline request.
 */
ULONG AppBaselineRuntime_GetRequestGeneration(void)
{
	return camera_baseline_request_generation;
}

/* USER CODE END 0 */
