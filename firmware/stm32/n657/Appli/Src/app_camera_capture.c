/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_capture.c
 * @brief   High-level camera capture and save helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_camera_capture.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "app_threadx.h"
#include "app_threadx_config.h"
#include "app_camera_buffers.h"
#include "app_camera_config.h"
#include "app_camera_diagnostics.h"
#include "app_camera_platform.h"
#include "app_ai_config.h"
#include "app_gauge_geometry.h"
#include "app_filex.h"
#include "app_inference_runtime.h"
#include "app_baseline_runtime.h"
#include "app_storage.h"
#include "debug_console.h"
#include "threadx_utils.h"
#include "cmw_imx335.h"
#include "imx335.h"
#include "cmw_camera.h"
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
extern CMW_IMX335_t camera_sensor;
extern bool camera_capture_use_cmw_pipeline;
extern bool camera_cmw_initialized;
extern bool camera_stream_started;
extern volatile bool camera_capture_isp_loop_paused;
extern volatile uint32_t camera_capture_isp_run_count;
extern volatile bool camera_capture_failed;
extern volatile uint32_t camera_capture_error_code;
extern volatile uint32_t camera_capture_byte_count;
extern volatile bool camera_capture_sof_seen;
extern volatile bool camera_capture_eof_seen;
extern volatile bool camera_capture_frame_done;
extern volatile bool camera_capture_snapshot_armed;
extern volatile uint32_t camera_capture_done_event_count;
extern volatile uint32_t camera_capture_frame_event_count;
extern volatile uint32_t camera_capture_line_error_count;
extern volatile uint32_t camera_capture_line_error_mask;
extern volatile uint32_t camera_capture_csi_linebyte_event_count;
extern volatile bool camera_capture_csi_linebyte_event_logged;
extern volatile uint32_t camera_capture_vsync_event_count;
extern volatile uint32_t camera_capture_csi_irq_count;
extern volatile uint32_t camera_capture_dcmipp_irq_count;
extern volatile uint32_t camera_capture_reported_byte_count;
extern volatile uint32_t camera_capture_counter_status;
extern uint8_t *camera_capture_result_buffer;
extern uint32_t camera_capture_active_buffer_index;

/**
 * @brief Resume the camera ISP service after AI releases the stable snapshot.
 *
 * The capture path pauses the background CMW service while a frame is owned by
 * inference. Keeping the pause until the worker finishes prevents DMA reuse
 * during inference but must be cleared before the next exposure settles.
 */
void AppCameraCapture_ReleaseInferenceFrame(void) {
	camera_capture_isp_loop_paused = false;
}

/**
 * @brief Decide whether a DCMIPP error is worth retrying once.
 *
 * We treat the CSI sync plus DPHY control combo as a transient link issue when
 * the capture buffer already filled, because the frame itself usually made it
 * through before the late error surfaced.
 * @retval true when one more capture attempt is reasonable.
 */
static bool AppCameraCapture_ShouldRetryDcmippError(uint32_t error_code) {
	return (error_code == 0x00008100U)
			&& (camera_capture_reported_byte_count >= CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
}

/**
 * @brief Wait for both consumers to release the private snapshot.
 * @retval true when the next capture may safely reuse the snapshot.
 * @retval false when ownership remains stuck or the wait budget expires.
 * @sideeffect Delays cooperatively and emits rate-limited ownership state.
 * @note Never clears an in-flight flag here: the worker or NPU may still be
 * reading the snapshot, so forcibly clearing it could create a new DMA/NPU
 * race and corrupt the next inference.
 */
static bool AppCameraCapture_WaitForInferenceOwnershipRelease(void) {
	static bool ownership_timeout_latched = false;
	uint32_t elapsed_ms = 0U;
	uint32_t next_log_ms = 0U;

	if (ownership_timeout_latched) {
		if (!AppInferenceRuntime_IsInferenceInFlight()
#if APP_BASELINE_ENABLE_THREAD
				&& !AppBaselineRuntime_IsEstimateInFlight()
#endif
		) {
			ownership_timeout_latched = false;
		} else {
			(void)DebugConsole_Printf(
				"[CAMERA][CAPTURE] Inference ownership remains latched; capture paused.\r\n");
			return false;
		}
	}

	while (AppInferenceRuntime_IsInferenceInFlight()) {
		if (elapsed_ms >= next_log_ms) {
			(void)DebugConsole_Printf(
				"[CAMERA][CAPTURE] Waiting for AI: elapsed=%lu ms state=%s gen=%lu progress_tick=%lu.\r\n",
				(unsigned long)elapsed_ms,
				AppInferenceRuntime_WorkerStateName(
					AppInferenceRuntime_GetWorkerState()),
				(unsigned long)AppInferenceRuntime_GetRequestGeneration(),
				(unsigned long)AppInferenceRuntime_GetWorkerProgressTick());
			next_log_ms += CAMERA_CAPTURE_INFERENCE_WAIT_LOG_PERIOD_MS;
		}
		if (elapsed_ms >= CAMERA_CAPTURE_INFERENCE_WAIT_TIMEOUT_MS) {
			ownership_timeout_latched = true;
			(void)DebugConsole_Printf(
				"[CAMERA][CAPTURE] AI ownership timeout=%lu ms; capture paused without releasing snapshot.\r\n",
				(unsigned long)CAMERA_CAPTURE_INFERENCE_WAIT_TIMEOUT_MS);
			return false;
		}
		DelayMilliseconds_Cooperative(100U);
		elapsed_ms += 100U;
	}

	elapsed_ms = 0U;
	next_log_ms = 0U;
	while (AppBaselineRuntime_IsEstimateInFlight()) {
		if (elapsed_ms >= next_log_ms) {
			(void)DebugConsole_Printf(
				"[CAMERA][CAPTURE] Waiting for baseline: elapsed=%lu ms state=%s gen=%lu progress_tick=%lu.\r\n",
				(unsigned long)elapsed_ms,
				AppBaselineRuntime_WorkerStateName(
					AppBaselineRuntime_GetWorkerState()),
				(unsigned long)AppBaselineRuntime_GetRequestGeneration(),
				(unsigned long)AppBaselineRuntime_GetWorkerProgressTick());
			next_log_ms += CAMERA_CAPTURE_INFERENCE_WAIT_LOG_PERIOD_MS;
		}
		if (elapsed_ms >= CAMERA_CAPTURE_INFERENCE_WAIT_TIMEOUT_MS) {
			ownership_timeout_latched = true;
			(void)DebugConsole_Printf(
				"[CAMERA][CAPTURE] Baseline ownership timeout=%lu ms; capture paused without releasing snapshot.\r\n",
				(unsigned long)CAMERA_CAPTURE_INFERENCE_WAIT_TIMEOUT_MS);
			return false;
		}
		DelayMilliseconds_Cooperative(100U);
		elapsed_ms += 100U;
	}

	return true;
}

/**
 * @brief Brightness classification for the processed capture gate.
 */
typedef enum {
	APP_CAMERA_CAPTURE_BRIGHTNESS_OK = 0,
	APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK,
	APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT,
} AppCameraCapture_BrightnessGate_t;

/**
 * @brief Crop luma summary used to decide whether to accept a frame.
 */
typedef struct {
	uint32_t sample_count;
	uint32_t bright_sample_count;
	uint8_t min_y;
	uint8_t max_y;
	uint32_t mean_y;
} AppCameraCapture_BrightnessStats_t;

/**
 * @brief Clamp a brightness nudge step to a safe runtime range.
 */
static uint32_t AppCameraCapture_ClampBrightnessStepPercent(uint32_t step_percent) {
	if (step_percent < CAMERA_CAPTURE_BRIGHTNESS_STEP_MIN_PERCENT) {
		return CAMERA_CAPTURE_BRIGHTNESS_STEP_MIN_PERCENT;
	}
	if (step_percent > CAMERA_CAPTURE_BRIGHTNESS_STEP_MAX_PERCENT) {
		return CAMERA_CAPTURE_BRIGHTNESS_STEP_MAX_PERCENT;
	}
	return step_percent;
}

/**
 * @brief Pick a damped exposure/gain nudge size from the current gate stats.
 *
 * Frames far from the target mean can take a larger step, while near-target
 * frames take a smaller step so we do not bounce back and forth between two
 * adjacent exposure settings.
 */
static uint32_t AppCameraCapture_ComputeBrightnessStepPercent(
		const AppCameraCapture_BrightnessStats_t *stats,
		AppCameraCapture_BrightnessGate_t gate,
		AppCameraCapture_BrightnessGate_t previous_gate) {
	uint32_t mean_error = 0U;
	uint32_t step_percent = CAMERA_CAPTURE_BRIGHTNESS_STEP_MIN_PERCENT;
	const uint32_t target_mean = CAMERA_CAPTURE_BRIGHTNESS_TARGET_MEAN;

	if (stats == NULL) {
		return CAMERA_CAPTURE_BRIGHTNESS_STEP_MAX_PERCENT;
	}

	if (stats->mean_y > target_mean) {
		mean_error = stats->mean_y - target_mean;
	} else {
		mean_error = target_mean - stats->mean_y;
	}

	/* Use a coarse/fine ladder instead of a purely linear ramp.
	 * Very dark or very bright frames need a full-range step so the gate can
	 * cross the usable band in a handful of nudges rather than exhausting the
	 * retry budget while still far from the target. */
	if (mean_error >= 80U) {
		step_percent = CAMERA_CAPTURE_BRIGHTNESS_STEP_MAX_PERCENT;
	} else if (mean_error >= 60U) {
		step_percent = 9U;
	} else if (mean_error >= 40U) {
		step_percent = 8U;
	} else if (mean_error >= 20U) {
		step_percent = 6U;
	} else {
		step_percent = 4U;
	}
	step_percent = AppCameraCapture_ClampBrightnessStepPercent(step_percent);

	/* If the retry direction just flipped, damp the next nudge so we do not
	 * bounce back and forth between the same two exposure settings. */
	if ((previous_gate != APP_CAMERA_CAPTURE_BRIGHTNESS_OK)
			&& (previous_gate != gate)
			&& (step_percent > CAMERA_CAPTURE_BRIGHTNESS_STEP_FLIP_PENALTY_PERCENT)) {
		step_percent -= CAMERA_CAPTURE_BRIGHTNESS_STEP_FLIP_PENALTY_PERCENT;
	}

	return AppCameraCapture_ClampBrightnessStepPercent(step_percent);
}

/**
 * @brief Measure luma over the full training crop region of a MONO_Y8 frame.
 *
 * Sampling the entire DCMIPP frame avoids being fooled by specular reflections
 * on the gauge glass. The live ellipse model sees this complete resized
 * frame, so the mean directly predicts whether its input will be well-exposed.
 */
static bool AppCameraCapture_ComputeBrightnessStats(const uint8_t *buffer_ptr,
		uint32_t length_bytes, AppCameraCapture_BrightnessStats_t *stats) {
	const uint32_t frame_width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const uint32_t frame_height_lines = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const uint32_t bytes_per_pixel = CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const uint32_t stride_bytes = frame_width_pixels * bytes_per_pixel;
	uint64_t sum_y = 0U;
	uint32_t sample_count = 0U;
	uint32_t bright_sample_count = 0U;
	uint8_t min_y = 0xFFU;
	uint8_t max_y = 0U;

	if ((buffer_ptr == NULL) || (stats == NULL) || (length_bytes < stride_bytes)) {
		return false;
	}

	for (uint32_t row = 0U; row < frame_height_lines; row++) {
		const uint32_t row_base = row * stride_bytes;

		if ((row_base + (frame_width_pixels * bytes_per_pixel)) > length_bytes) {
			return false;
		}

		for (uint32_t col = 0U; col < frame_width_pixels; col++) {
			const uint8_t y_sample = buffer_ptr[row_base + (col * bytes_per_pixel)];

			if (y_sample < min_y) {
				min_y = y_sample;
			}
			if (y_sample > max_y) {
				max_y = y_sample;
			}
			if (y_sample >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_PIXEL_LEVEL_THRESHOLD) {
				bright_sample_count++;
			}
			sum_y += y_sample;
			sample_count++;
		}

	}

	if (sample_count == 0U) {
		return false;
	}

	stats->sample_count = sample_count;
	stats->bright_sample_count = bright_sample_count;
	stats->min_y = min_y;
	stats->max_y = max_y;
	stats->mean_y = (uint32_t) (sum_y / sample_count);
	return true;
}

/**
 * @brief Decide whether a processed frame is too dark, too bright, or usable.
 */
static AppCameraCapture_BrightnessGate_t AppCameraCapture_ClassifyBrightness(
		const AppCameraCapture_BrightnessStats_t *stats) {
	if (stats == NULL) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_OK;
	}

	/* Treat the exact threshold as acceptable so the retry loop can exit on
	 * a borderline-but-usable frame instead of failing the whole capture. */
	/* A single saturated pixel must not make an otherwise dark frame pass.
	 * The previous max-pixel condition accepted frames with mean luma near 80
	 * as soon as one highlight reached 255, which starved both detectors. */
	if ((stats->mean_y < CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD)
			&& ((stats->bright_sample_count * 100U)
					< (stats->sample_count *
						CAMERA_CAPTURE_BRIGHTNESS_DARK_BRIGHT_RATIO_MAX_PERCENT))) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK;
	}

	if ((stats->mean_y >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_SOLID_MEAN_THRESHOLD)
			&& (stats->min_y >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD)) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT;
	}

	if ((stats->mean_y >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD)
			&& ((stats->bright_sample_count * 100U)
					>= (stats->sample_count
							* CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_RATIO_PERCENT))) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT;
	}

	return APP_CAMERA_CAPTURE_BRIGHTNESS_OK;
}

/**
 * @brief Print the brightness gate result so we can see why a frame was retried.
 */
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
static void AppCameraCapture_LogBrightnessGateDecision(
		const AppCameraCapture_BrightnessStats_t *stats,
		AppCameraCapture_BrightnessGate_t decision) {
	const char *decision_label = "ok";

	switch (decision) {
	case APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK:
		decision_label = "too-dark";
		break;
	case APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT:
		decision_label = "too-bright";
		break;
	default:
		break;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Brightness gate (%s): samples=%lu mean=%lu min=%u max=%u bright=%lu (%lu%%) thresholds dark<=%u/%u bright_ratio>=%u%%@%u bright_solid>=%u/%u.\r\n",
			decision_label,
			(unsigned long) ((stats != NULL) ? stats->sample_count : 0U),
			(unsigned long) ((stats != NULL) ? stats->mean_y : 0U),
			(unsigned int) ((stats != NULL) ? stats->min_y : 0U),
			(unsigned int) ((stats != NULL) ? stats->max_y : 0U),
			(unsigned long) ((stats != NULL) ? stats->bright_sample_count : 0U),
			(unsigned long) (((stats != NULL) && (stats->sample_count != 0U))
					? ((stats->bright_sample_count * 100U) / stats->sample_count)
					: 0U),
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_DARK_MAX_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_RATIO_PERCENT,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_SOLID_MEAN_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD);
}
#endif
/* USER CODE END PV */

/**
 * @brief Service ST's IMX335 middleware background process for ISP state updates.
 * @retval true when the background step succeeded or is not used by this driver.
 */
bool AppCameraCapture_RunImx335Background(void) {
	/* The capture thread owns the CMW/ISP state from snapshot arm until the
	 * frame has been copied and released.  Do not let the separate camera ISP
	 * thread enter CMW_CAMERA_Run() during that interval: the IMX335 driver
	 * calls ISP_BackgroundProcess(), which mutates shared ISP/statistics state
	 * while DCMIPP is filling the same capture buffer.  That race can corrupt
	 * the frame; it is independent of the explicit DebugConsole UART path. */
	if (camera_capture_isp_loop_paused) {
		return true;
	}

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized
			&& camera_stream_started) {
		/* Run the ISP background process so AEC/AWB update per-frame.
		 * Without this call the ISP demosaiced output has neutral chroma
		 * because AWB gains are never applied.
		 *
		 * The processed snapshot path now starts the middleware before this
		 * function can run, so the ISP handle is valid by the time the
		 * background loop reaches here. */
		int32_t cmw_ret = CMW_CAMERA_Run();
		camera_capture_isp_run_count++;
		if (cmw_ret != CMW_ERROR_NONE) {
			DebugConsole_Printf(
				"[CAMERA] ISP background run failed: %ld\r\n",
				(long)cmw_ret);
			return false;
		}
		return true;
	}


	return true;
}

/**
 * @brief Dump the current camera, ISP, and DCMIPP state for black-frame diagnostics.
 * @param reason Short note describing what triggered the dump.
 */
void AppCameraCapture_LogCaptureState(const char *reason) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();
	ISP_SensorInfoTypeDef sensor_info = { 0 };
	uint32_t pipe_mode = 0U;
	uint32_t pipe_state = 0U;
	uint32_t pipe_counter = 0U;
	uint8_t mode_select = 0U;
	uint8_t lane_mode_reg_3050 = 0U;
	uint8_t lane_mode_reg_319d = 0U;
	uint8_t lane_mode_reg_341c = 0U;
	uint8_t lane_mode_reg_341d = 0U;
	uint8_t lane_mode_reg_3a01 = 0U;
	uint8_t hold_reg = 0U;
	uint8_t tpg_reg = 0U;
	uint16_t gain_reg = 0U;
	uint32_t shutter_reg = 0U;
	uint32_t vmax_reg = 0U;
	int32_t cmw_exposure_mode = 0;
	uint8_t cmw_aec_enabled = 0U;
	int32_t cmw_exposure = 0;
	int32_t cmw_gain = 0;
	int32_t cmw_test_pattern = 0;
	bool cmw_state_ok = false;
	bool sensor_regs_ok = true;
	AppCameraDiagnostics_CaptureState_t snapshot = { 0 };

	if ((capture_dcmipp != NULL) && (capture_dcmipp->Instance != NULL)) {
		pipe_mode = HAL_DCMIPP_GetMode(capture_dcmipp);
		pipe_state = HAL_DCMIPP_PIPE_GetState(capture_dcmipp,
		CAMERA_CAPTURE_PIPE);
		(void) HAL_DCMIPP_PIPE_GetDataCounter(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &pipe_counter);
	}

	if (camera_cmw_initialized) {
		cmw_state_ok = true;
		if (CMW_CAMERA_GetExposureMode(&cmw_exposure_mode) != CMW_ERROR_NONE) {
			cmw_state_ok = false;
		}
		if (CMW_CAMERA_GetExposure(&cmw_exposure) != CMW_ERROR_NONE) {
			cmw_state_ok = false;
		}
		if (CMW_CAMERA_GetGain(&cmw_gain) != CMW_ERROR_NONE) {
			cmw_state_ok = false;
		}
		if (ISP_GetAECState(&camera_sensor.hIsp, &cmw_aec_enabled)
				!= ISP_OK) {
			cmw_state_ok = false;
		}
		if (CMW_CAMERA_GetTestPattern(&cmw_test_pattern) != CMW_ERROR_NONE) {
			cmw_state_ok = false;
		}
		if (CMW_CAMERA_GetSensorInfo(&sensor_info) != CMW_ERROR_NONE) {
			cmw_state_ok = false;
		}
	}

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, 0x3050U,
			&lane_mode_reg_3050, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, 0x319DU,
			&lane_mode_reg_319d, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, 0x341CU,
			&lane_mode_reg_341c, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, 0x341DU,
			&lane_mode_reg_341d, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, 0x3A01U,
			&lane_mode_reg_3a01, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, IMX335_REG_HOLD,
			&hold_reg, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, IMX335_REG_TPG,
			&tpg_reg, 1U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, IMX335_REG_GAIN,
			(uint8_t*) &gain_reg, 2U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_SHUTTER, (uint8_t*) &shutter_reg, 3U) != IMX335_OK) {
		sensor_regs_ok = false;
	}
	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, IMX335_REG_VMAX,
			(uint8_t*) &vmax_reg, 4U) != IMX335_OK) {
		sensor_regs_ok = false;
	}

	snapshot.reason = reason;
	snapshot.capture_dcmipp = capture_dcmipp;
	snapshot.capture_pipe = CAMERA_CAPTURE_PIPE;
	snapshot.pipe_memory_address =
			(capture_dcmipp != NULL) ?
					(uintptr_t) HAL_DCMIPP_PIPE_GetMemoryAddress(
							capture_dcmipp, CAMERA_CAPTURE_PIPE,
							DCMIPP_MEMORY_ADDRESS_0) :
					0U;
	snapshot.pipe_mode = pipe_mode;
	snapshot.pipe_state = pipe_state;
	snapshot.pipe_counter = pipe_counter;
	snapshot.buffer0 = camera_capture_buffers[0];
#if CAMERA_CAPTURE_BUFFER_COUNT > 1U
	snapshot.buffer1 = camera_capture_buffers[1];
#else
	snapshot.buffer1 = NULL;
#endif
	snapshot.result_buffer = (const uint8_t*) camera_capture_result_buffer;
	snapshot.snapshot_armed = camera_capture_snapshot_armed;
	snapshot.stream_started = camera_stream_started;
	snapshot.use_cmw_pipeline = camera_capture_use_cmw_pipeline;
	snapshot.cmw_initialized = camera_cmw_initialized;
	snapshot.frame_event_count = camera_capture_frame_event_count;
	snapshot.vsync_event_count = camera_capture_vsync_event_count;
	snapshot.isp_run_count = camera_capture_isp_run_count;
	snapshot.csi_irq_count = camera_capture_csi_irq_count;
	snapshot.dcmipp_irq_count = camera_capture_dcmipp_irq_count;
	snapshot.reported_byte_count = camera_capture_reported_byte_count;
	snapshot.counter_status = camera_capture_counter_status;
	snapshot.sof_seen = camera_capture_sof_seen;
	snapshot.eof_seen = camera_capture_eof_seen;
	snapshot.failed = camera_capture_failed;
	snapshot.error_code = camera_capture_error_code;
	snapshot.line_error_count = camera_capture_line_error_count;
	snapshot.line_error_mask = camera_capture_line_error_mask;
	snapshot.active_buffer_index = camera_capture_active_buffer_index;
	snapshot.cmw_state_ok = cmw_state_ok;
	snapshot.cmw_exposure_mode = cmw_exposure_mode;
	snapshot.cmw_aec_enabled = cmw_aec_enabled;
	snapshot.cmw_exposure = cmw_exposure;
	snapshot.cmw_gain = cmw_gain;
	snapshot.cmw_test_pattern = cmw_test_pattern;
	snapshot.sensor_name = sensor_info.name;
	snapshot.sensor_width = sensor_info.width;
	snapshot.sensor_height = sensor_info.height;
	snapshot.sensor_gain_min = sensor_info.gain_min;
	snapshot.sensor_gain_max = sensor_info.gain_max;
	snapshot.sensor_again_max = sensor_info.again_max;
	snapshot.sensor_exposure_min = sensor_info.exposure_min;
	snapshot.sensor_exposure_max = sensor_info.exposure_max;
	snapshot.sensor_regs_ok = sensor_regs_ok;
	snapshot.mode_select = mode_select;
	snapshot.lane_mode_reg_3050 = lane_mode_reg_3050;
	snapshot.lane_mode_reg_319d = lane_mode_reg_319d;
	snapshot.lane_mode_reg_341c = lane_mode_reg_341c;
	snapshot.lane_mode_reg_341d = lane_mode_reg_341d;
	snapshot.lane_mode_reg_3a01 = lane_mode_reg_3a01;
	snapshot.hold_reg = hold_reg;
	snapshot.tpg_reg = tpg_reg;
	snapshot.gain_reg = gain_reg;
	snapshot.shutter_reg = shutter_reg;
	snapshot.vmax_reg = vmax_reg;
	snapshot.csi_linebyte_event_count =
			camera_capture_csi_linebyte_event_count;

	AppCameraDiagnostics_LogCaptureState(&snapshot);
}

#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
/**
 * @brief Log the save-path state for the frame that is about to be written.
 *
 * The goal is to keep the message small enough for normal bring-up while
 * still showing whether the frame came from the processed CMW/ISP path or a
 * raw fallback path.
 */
static void AppCameraCapture_LogSavePathState(const uint8_t *image_ptr,
		uint32_t image_length) {
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] save-state: pipeline=%s cmw_init=%u stream=%u active_buf=%lu result=%p bytes=%lu\r\n",
			camera_capture_use_cmw_pipeline ? "processed" : "raw",
			(unsigned int) (camera_cmw_initialized ? 1U : 0U),
			(unsigned int) (camera_stream_started ? 1U : 0U),
			(unsigned long) camera_capture_active_buffer_index,
			(void *) image_ptr, (unsigned long) image_length);

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		AppCameraCapture_LogCaptureState("save-path");
	}
}
#endif
/**
 * @brief Capture one frame from the configured DCMIPP pipeline.
 * @retval true when a valid frame reaches the caller's buffer.
 */
bool AppCameraCapture_CaptureSingleFrame(uint32_t *captured_bytes_ptr) {
	const ULONG wait_ticks = CameraPlatform_MillisecondsToTicks(
	CAMERA_CAPTURE_TIMEOUT_MS);
	ULONG next_wait_log_tick = 0U;
	ULONG deadline_tick = 0U;
	bool should_reset_sensor_stream = false;
	uint32_t completion_event_baseline = 0U;
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (captured_bytes_ptr == NULL) {
		return false;
	}

	camera_capture_isp_loop_paused = true;

	if (!App_ThreadX_LockCameraMiddleware(
			CameraPlatform_MillisecondsToTicks(
					CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS))) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to lock camera middleware for snapshot setup.\r\n");
		camera_capture_isp_loop_paused = false;
		return false;
	}

	/* Keep blue available for the save-success flash later in the flow. */
	BSP_LED_Off(LED_BLUE);
	if (!CameraPlatform_PrepareDcmippSnapshot()) {
		App_ThreadX_UnlockCameraMiddleware();
		camera_capture_isp_loop_paused = false;
		return false;
	}

	camera_capture_failed = false;
	camera_capture_error_code = 0U;
	camera_capture_byte_count = 0U;
	camera_capture_sof_seen = false;
	camera_capture_eof_seen = false;
	camera_capture_frame_done = false;
	camera_capture_snapshot_armed = false;
	camera_capture_frame_event_count = 0U;
	camera_capture_line_error_count = 0U;
	camera_capture_line_error_mask = 0U;
	camera_capture_csi_linebyte_event_count = 0U;
	camera_capture_csi_linebyte_event_logged = false;
	camera_capture_vsync_event_count = 0U;
	camera_capture_isp_run_count = 0U;
	camera_capture_csi_irq_count = 0U;
	camera_capture_dcmipp_irq_count = 0U;
	camera_capture_reported_byte_count = 0U;
	camera_capture_counter_status = (uint32_t) HAL_ERROR;
	camera_capture_active_buffer_index = 0U;
	camera_capture_result_buffer = camera_capture_buffers[0];
	AppCameraBuffers_PrepareForDma();

	/* Match ST's CMW_CAMERA_Start() ordering: arm the CSI/DCMIPP receiver first,
	 * then start the ISP + sensor stream. This avoids missing the first valid
	 * frame while the middleware is bringing the stream up. */
	/* Capture the counter before arming.  The ISR only increments this aligned
	 * word; the thread later treats any change as a completion/error event. */
	completion_event_baseline = camera_capture_done_event_count;
	if (!CameraPlatform_StartDcmippSnapshot()) {
		DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
		if (!CameraPlatform_StartDcmippSnapshot()) {
			App_ThreadX_UnlockCameraMiddleware();
			camera_capture_isp_loop_paused = false;
			return false;
		}
	}

	camera_capture_snapshot_armed = true;

	if (!camera_stream_started) {
		if (!CameraPlatform_StartImx335Stream()) {
			(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
			DCMIPP_VIRTUAL_CHANNEL0);
			camera_capture_snapshot_armed = false;
			App_ThreadX_UnlockCameraMiddleware();
			camera_capture_isp_loop_paused = false;
			return false;
		}
	} else {
		/* On later snapshots, give the already-running stream a brief moment to
		 * advance to the armed frame boundary before we block on completion. */
		DelayMilliseconds_ThreadX(CAMERA_STREAM_WARMUP_DELAY_MS);
	}
	/* CMW_CAMERA_Start()/stream startup can restore the ISP IQ defaults, which
	 * include AEC enabled.  Lock it again at the real capture boundary so the
	 * brightness-gate nudges below control the same exposure/gain that produces
	 * this frame. */
	if (camera_capture_use_cmw_pipeline
			&& !CameraPlatform_DisableImx335AutoExposure()) {
		DebugConsole_WriteString(
				"[CAMERA][CAPTURE] Warning: could not lock AEC at capture boundary; continuing with current ISP state.\r\n");
	}
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	(void) CameraPlatform_LogImx335AutoExposureState("capture-start");
#endif
	App_ThreadX_UnlockCameraMiddleware();

	deadline_tick = tx_time_get() + wait_ticks;
	next_wait_log_tick = tx_time_get()
			+ CameraPlatform_MillisecondsToTicks(1000U);
	while (true) {
		if (camera_capture_done_event_count != completion_event_baseline) {
			if (!camera_capture_failed) {
				const uint32_t completed_buffer_index =
						camera_capture_active_buffer_index;
				uint32_t completed_nonzero_bytes = 0U;
				uint8_t *completed_buffer_ptr = NULL;
				bool keep_waiting_for_convergence = false;

				completed_buffer_ptr =
						camera_capture_buffers[completed_buffer_index];

				completed_nonzero_bytes = AppCameraBuffers_CountNonZeroBytes(
						completed_buffer_ptr, CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
				if ((completed_nonzero_bytes == 0U)
						&& camera_capture_use_cmw_pipeline) {
					keep_waiting_for_convergence = true;
				}

				if (keep_waiting_for_convergence) {
					if ((tx_time_get() >= deadline_tick)
							|| (camera_capture_failed)) {
						DebugConsole_Printf(
								"[CAMERA][CAPTURE] Camera path never produced nonzero pixels before timeout.\r\n");
						camera_capture_isp_loop_paused = false;
						return false;
					}

					DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
					continue;
				}

				camera_capture_result_buffer = completed_buffer_ptr;
				(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp,
				CAMERA_CAPTURE_PIPE, DCMIPP_VIRTUAL_CHANNEL0);
				/* Freeze the completed DMA buffer at the source. The board has no
				 * HyperRAM, and the lower AXISRAM1 alias used by the former snapshot
				 * copy can stall the CPU. The next capture waits for AI completion. */
				if (camera_capture_use_cmw_pipeline && camera_stream_started
						&& !CameraPlatform_StopImx335Stream()) {
					DebugConsole_WriteString(
							"[CAMERA][CAPTURE] Could not stop IMX335 after frame completion; refusing live-buffer handoff.\r\n");
					camera_capture_snapshot_armed = false;
					camera_capture_isp_loop_paused = false;
					return false;
				}
				camera_capture_snapshot_armed = false;
				*captured_bytes_ptr = camera_capture_byte_count;
				if (camera_capture_use_cmw_pipeline) {
					(void) AppCameraBuffers_InvalidateCaptureRegion(
							camera_capture_byte_count);
				}
				return true;
			}

			DebugConsole_Printf(
					"[CAMERA][CAPTURE] DCMIPP reported capture error code 0x%08lX.\r\n",
					(unsigned long) camera_capture_error_code);
			AppCameraDiagnostics_LogDcmippErrorCode(camera_capture_error_code);
			AppCameraCapture_LogCaptureState("capture-error");
			should_reset_sensor_stream =
			AppCameraCapture_ShouldRetryDcmippError(camera_capture_error_code);
			break;
		}

		if ((tx_time_get() >= deadline_tick) || camera_capture_failed) {
			if (tx_time_get() >= deadline_tick) {
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Timed out waiting for frame completion.\r\n");
			}
			break;
		}

		if ((next_wait_log_tick == 0U) || (tx_time_get() >= next_wait_log_tick)) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Waiting for frame completion...\r\n");
			next_wait_log_tick = tx_time_get()
					+ CameraPlatform_MillisecondsToTicks(1000U);
		}

		/* No kernel object is touched by a DCMIPP ISR.  A short cooperative
		 * sleep preserves the old polling cadence without a semaphore wakeup. */
		DelayMilliseconds_ThreadX(20U);
	}

	(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
	DCMIPP_VIRTUAL_CHANNEL0);
	if (should_reset_sensor_stream) {
		if (!CameraPlatform_StopImx335Stream()) {
			DebugConsole_WriteString(
					"[CAMERA][CAPTURE] IMX335 stream stop failed during DCMIPP recovery.\r\n");
		}
	}
	camera_capture_snapshot_armed = false;
	camera_capture_isp_loop_paused = false;
	return false;
}

/**
 * @brief Capture a single frame, best-effort save it, and queue learned inference.
 * @retval true when the learned AI request is accepted. SD storage is a durable
 *         diagnostic side effect and must not prevent a live inference.
 */
bool AppCameraCapture_CaptureAndStoreSingleFrame(void) {
	/* This marker is intentionally on the capture path as well as boot: it
	 * proves the running application contains the synchronous AI handoff even
	 * when log collection starts after the boot banner. */
	static bool capture_build_marker_logged = false;
	uint32_t captured_bytes = 0U;
	UINT filex_status = FX_SUCCESS;
	CHAR capture_file_name[CAMERA_CAPTURE_FILE_NAME_LENGTH] = { 0 };
	uint8_t *image_ptr = NULL;
	ULONG image_length = captured_bytes;
	bool result = false;
	bool storage_ready = AppFileX_IsMediaReady();
	const CHAR *file_extension = camera_capture_use_cmw_pipeline ? "gray8"
			: "raw16";
	const uint32_t max_brightness_adjustments =
	CAMERA_CAPTURE_BRIGHTNESS_RETRY_LIMIT;
	const uint32_t max_dcmipp_retries = 1U;
	uint32_t capture_attempt = 0U;
	uint32_t brightness_adjustment_count = 0U;
	uint32_t dcmipp_retry_count = 0U;
	bool capture_ok = false;
	bool capture_saved = !camera_capture_use_cmw_pipeline;
	bool ai_handoff_accepted = !camera_capture_use_cmw_pipeline;
	bool discard_next_successful_frame = false;
	/* Keep one compact failure reason so a truncated UART line still identifies
	 * the transaction stage without dumping the frame or adding a log burst. */
	const char *failure_stage = "capture";
	AppCameraCapture_BrightnessGate_t previous_brightness_gate =
	APP_CAMERA_CAPTURE_BRIGHTNESS_OK;
	AppCameraCapture_BrightnessStats_t brightness_stats = { 0 };
	AppCameraCapture_BrightnessGate_t brightness_gate =
	APP_CAMERA_CAPTURE_BRIGHTNESS_OK;

	(void) DebugConsole_WriteString(
			"[CAMERA][CAPTURE] Begin capture-and-store request.\r\n");
	if (!capture_build_marker_logged) {
		capture_build_marker_logged = true;
		(void) DebugConsole_WriteString(
			"[CAMERA][BUILD] capture-owner-v17-isr-safe-capture-events\r\n");
	}
	/* Do not arm DCMIPP again until the queued inference has released the
	 * completed non-cacheable capture buffer. Never force-clear ownership while
	 * the worker may still be reading it. */
	if (!AppCameraCapture_WaitForInferenceOwnershipRelease()) {
		failure_stage = "ownership-wait";
		goto cleanup;
	}
	if (!storage_ready) {
		(void) DebugConsole_WriteString(
				"[CAMERA][CAPTURE] FileX media not ready yet; this capture will skip SD save.\r\n");
	}

	/* Before the first capture attempt, let AE hardware settle then lock
	 * so the manual brightness-gate nudges start from a stable baseline. */
	if ((capture_attempt == 0U) && camera_capture_use_cmw_pipeline) {
		(void)CameraPlatform_AeSettleAndLock();
	}

	for (capture_attempt = 0U;; capture_attempt++) {
		if (capture_attempt > 0U) {
			if (camera_capture_error_code != 0U) {
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Retrying capture after DCMIPP error 0x%08lX.\r\n",
						(unsigned long) camera_capture_error_code);
			}
			DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
		}

		if (AppCameraCapture_CaptureSingleFrame(&captured_bytes)) {
			if (discard_next_successful_frame) {
				/* A DCMIPP retry can recover a usable buffer, but the preceding
				 * transport error means this frame is less trustworthy than a clean
				 * first-pass capture. Skip it and wait for the next clean frame. */
				(void) DebugConsole_WriteString(
						"[CAMERA][CAPTURE] Discarding frame after DCMIPP retry; requesting another capture.\r\n");
				discard_next_successful_frame = false;
				capture_ok = false;
				continue;
			}

			capture_ok = true;
			image_ptr = camera_capture_result_buffer;
			if (camera_capture_use_cmw_pipeline) {
				if (!AppCameraCapture_ComputeBrightnessStats(image_ptr,
						captured_bytes, &brightness_stats)) {
					DebugConsole_Printf(
							"[CAMERA][CAPTURE] Brightness gate could not analyze processed frame; accepting the valid frame without a gate decision.\r\n");
					/* The DMA completion and byte count already prove that this is a
					 * usable frame.  Do not turn a diagnostic-statistics failure into an
					 * unbounded capture loop that starves the AI pipeline. */
					capture_ok = true;
					break;
				}
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
				/* One line per capture attempt; gated so a steady-state loop
				 * does not spam the console with per-cycle stats. */
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Brightness stats complete mean=%lu min=%u max=%u.\r\n",
						(unsigned long) brightness_stats.mean_y,
						(unsigned int) brightness_stats.min_y,
						(unsigned int) brightness_stats.max_y);
#endif

				brightness_gate =
				AppCameraCapture_ClassifyBrightness(&brightness_stats);
				if (brightness_gate != APP_CAMERA_CAPTURE_BRIGHTNESS_OK) {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
					AppCameraCapture_LogBrightnessGateDecision(&brightness_stats,
							brightness_gate);
#endif
					if (brightness_adjustment_count
							>= max_brightness_adjustments) {
						DebugConsole_Printf(
								"[CAMERA][CAPTURE] Brightness gate exhausted its %lu manual nudges; accepting the last valid frame.\r\n",
								(unsigned long) max_brightness_adjustments);
						/* A dark/bright frame is still a complete camera frame and is
						 * preferable to restarting the whole capture operation forever.
						 * The AI stage can report low confidence while the board remains
						 * responsive and the frame is preserved for diagnosis. */
						capture_ok = true;
						break;
					}
					const uint32_t brightness_step_percent =
						AppCameraCapture_ComputeBrightnessStepPercent(
								&brightness_stats, brightness_gate,
								previous_brightness_gate);
					/* Let the retry path move the fixed manual exposure/gain toward
					 * the gate target. A static scene should converge in a few nudges;
					 * if the sensor hits its limit, fail fast instead of looping on the
					 * same underexposed or overexposed settings. */
					if (!CameraPlatform_AdjustImx335ExposureGain(
							brightness_gate ==
							APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK,
							brightness_step_percent)) {
						DebugConsole_WriteString(
								"[CAMERA][CAPTURE] IMX335 exposure/gain reached its adjustment limit; accepting the last valid frame.\r\n");
						/* A completed frame must not be discarded solely because the
						 * sensor has no remaining exposure headroom.  Returning false here
						 * would make the caller restart the same capture indefinitely. */
						capture_ok = true;
						break;
					}
					previous_brightness_gate = brightness_gate;
					brightness_adjustment_count++;
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
					DebugConsole_Printf(
							"[CAMERA][CAPTURE] Brightness gate triggered; retrying capture after exposure/gain nudge (%lu/%lu).\r\n",
							(unsigned long) brightness_adjustment_count,
							(unsigned long) max_brightness_adjustments);
#endif
					capture_ok = false;
					continue;
				}
				CameraPlatform_CacheAcceptedExposureGain();
			}

			break;
		}

		if (!AppCameraCapture_ShouldRetryDcmippError(camera_capture_error_code)) {
			break;
		}

		if (dcmipp_retry_count >= max_dcmipp_retries) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] DCMIPP retry budget exhausted after %lu transport retry.\r\n",
					(unsigned long) max_dcmipp_retries);
			break;
		}

		dcmipp_retry_count++;
		discard_next_successful_frame = true;
	}

	if (!capture_ok) {
		failure_stage = "capture";
		goto cleanup;
	}
	if (brightness_adjustment_count > 0U) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Brightness settled after %lu adjustment(s); final mean=%lu.\r\n",
				(unsigned long)brightness_adjustment_count,
				(unsigned long)brightness_stats.mean_y);
	}

	image_length = captured_bytes;
	image_ptr = camera_capture_result_buffer;
	if (image_ptr == NULL) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Capture buffer pointer is NULL after frame completion.\r\n");
		failure_stage = "frame-pointer";
		goto cleanup;
	}
	/* Snapshot before FileX or any later camera activity. The SD file and AI
	 * request now have one immutable source frame, which is the firmware-side
	 * equivalent of the offline evaluator's single-image contract. */
#if CAMERA_CAPTURE_USE_PRIVATE_SNAPSHOT
	if (camera_capture_use_cmw_pipeline
			&& !AppCameraBuffers_CopyCaptureToSnapshot(image_ptr,
					(uint32_t)image_length)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Immutable AI snapshot failed; frame rejected.\r\n");
		failure_stage = "snapshot-copy";
		goto cleanup;
	}
#endif
	/* Frame signatures are an opt-in parity diagnostic. The saved gray8 frame
	 * remains the durable artifact when that diagnosis is needed. */
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	AppCameraBuffers_LogFrameSignature("capture-ready", image_ptr,
			(uint32_t)image_length);
#endif

#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: frame-ready\r\n");
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Frame ready for save: ptr=%p length=%lu pipeline=%s\r\n",
			(void *) image_ptr, (unsigned long) image_length,
			camera_capture_use_cmw_pipeline ? "processed" : "raw");
#endif

#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: preview\r\n");
	AppCameraDiagnostics_LogCaptureBufferPreview("ready-to-save", image_ptr,
			(uint32_t) image_length);
#endif
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	AppCameraCapture_LogSavePathState(image_ptr, (uint32_t) image_length);
#endif

	/* The completed sensor frame is saved before the ownership handoff. The AI
	 * worker receives the private snapshot below so both model stages see the
	 * same immutable 640x640 image. */
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Snapshot handoff source: ptr=%p bytes=%lu\r\n",
			(const void *) image_ptr,
			(unsigned long) image_length);
#endif

	if (camera_capture_use_cmw_pipeline) {
		/*
		 * FileX can finish mounting while AE settling and brightness retries are
		 * running.  The value sampled at function entry is therefore only an
		 * early diagnostic; make the save decision from the current media state.
		 */
		storage_ready = AppFileX_IsMediaReady();
		if (storage_ready) {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		(void) DebugConsole_WriteString(
				"[CAMERA][CAPTURE] step: build-name\r\n");
#endif
		if (!AppStorage_BuildCaptureFileName(capture_file_name,
				sizeof(capture_file_name), file_extension)) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to build capture filename.\r\n");
			failure_stage = "filename";
		} else {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		(void) DebugConsole_WriteString(
				"[CAMERA][CAPTURE] step: build-name-done\r\n");
#endif

		if (camera_capture_use_cmw_pipeline) {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
			AppCameraCapture_LogCaptureState("processed-capture");
			AppCameraDiagnostics_LogProcessedFrameDiagnostics("processed-capture",
					image_ptr, (uint32_t) image_length);
#endif
		}

		{
			filex_status = AppFileX_WriteCapturedImage(capture_file_name,
					image_ptr, image_length);
		}
		if (filex_status != FX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to write image to SD card, status=%lu.\r\n",
					(unsigned long) filex_status);
			failure_stage = "filex-save";
		} else {
			capture_saved = true;
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
			AppCameraBuffers_LogFrameSignature("after-filex-save", image_ptr,
					(uint32_t)image_length);
#endif
		}
		}
		} else {
			(void) DebugConsole_WriteString(
					"[CAMERA][CAPTURE] FileX media unavailable at save point; SD write skipped.\r\n");
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		(void) DebugConsole_WriteString(
				"[CAMERA][CAPTURE] step: build-name-skipped\r\n");
#endif
	}

	if (camera_capture_use_cmw_pipeline) {
#if APP_BASELINE_ENABLE_THREAD && APP_BASELINE_QUEUE_WITH_CAPTURE
		/* Queue the classical comparator BEFORE the learned pipeline so both
		 * workers read the same stopped DMA frame while it is pristine. The
		 * baseline is CPU-only (~30 ms) and finishes long before the AI's NPU
		 * runs clobber the buffer rows; the camera thread waits for BOTH
		 * workers before re-arming (2026-08-05). */
		if (!AppBaselineRuntime_RequestEstimate(image_ptr,
				(ULONG) image_length)) {
			DebugConsole_Printf(
				"[BASELINE] Failed to queue pre-AI snapshot estimate.\r\n");
		}
#endif
		/* Capture stopped both DCMIPP and IMX335 before returning, so this
		 * non-cacheable buffer is immutable until the AI worker releases it. */
#if CAMERA_CAPTURE_USE_PRIVATE_SNAPSHOT
	#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		AppCameraBuffers_LogFrameSignature("snapshot-ready",
				camera_inference_frame_snapshot, (uint32_t)image_length);
	#endif
		ai_handoff_accepted = AppInferenceRuntime_RequestDryInference(
				(const uint8_t *) camera_inference_frame_snapshot,
				(ULONG) image_length);
		if (!ai_handoff_accepted) {
				DebugConsole_Printf(
					"[AI] Failed to queue one-shot dry-run inference.\r\n");
				failure_stage = "ai-queue";
		} else if (!AppCameraCapture_WaitForInferenceOwnershipRelease()) {
			DebugConsole_Printf(
				"[AI] Inference ownership did not complete within the wait budget.\r\n");
			ai_handoff_accepted = false;
			failure_stage = "ai-ownership";
		} else if (!AppInferenceRuntime_WasLastRequestSuccessful()) {
			DebugConsole_Printf(
				"[AI] Learned pipeline completed but did not publish a valid result.\r\n");
			ai_handoff_accepted = false;
			failure_stage = "ai-result";
		}
#else
		/* The stopped DMA buffer is the intentional no-copy ownership handoff on
		 * this board. It is also the exact source passed to FileX and AI. */
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		(void) DebugConsole_WriteString(
				"[CAMERA][FRAME] handing off stopped non-cacheable DMA buffer.\r\n");
#endif
		/* The live path is asynchronous: a successful queue operation is the
		 * handoff result here, while the AI worker later reports the model result.
		 * Keep this boolean synchronized with the actual request status so the
		 * camera thread does not falsely report a failed capture after the worker
		 * has already dequeued and executed it. */
		ai_handoff_accepted = AppInferenceRuntime_RequestDryInference(
				(const uint8_t *) image_ptr, (ULONG) image_length);
		if (!ai_handoff_accepted) {
			DebugConsole_Printf(
					"[AI] Failed to queue one-shot dry-run inference.\r\n");
			failure_stage = "ai-queue";
		}
#endif
	}

	}
	/* FileX is best-effort: a slow/unavailable SD card must not suppress the AI
	 * request or make the camera thread restart a valid capture forever. */
	result = camera_capture_use_cmw_pipeline ? ai_handoff_accepted
			: capture_saved && ai_handoff_accepted;

cleanup:
	/* Keep the ISP background loop paused while the AI worker still owns the
	 * capture buffer. CMW_CAMERA_Run() keeps streaming processed frames into
	 * the shared buffer after the snapshot pipe stops, so re-enabling it here
	 * lets the ISP overwrite the frame the worker is reading mid-inference
	 * (2026-08-02: center/tip stage saw a half-written top-black frame while
	 * the ellipse stage still read the valid capture). The next capture
	 * re-arms the ISP via snapshot_armed, so the loop only runs while a
	 * snapshot is actually in flight. */
	if (!AppInferenceRuntime_IsInferenceInFlight()) {
		camera_capture_isp_loop_paused = false;
	}
	if (!result) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] transaction failed stage=%s saved=%u filex=%lu ai=%u media=%u.\r\n",
				failure_stage,
				(unsigned int)capture_saved,
				(unsigned long)filex_status,
				(unsigned int)ai_handoff_accepted,
				(unsigned int)AppFileX_IsMediaReady());
	}
	return result;
}
