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
#include "app_baseline_runtime.h"
#include "app_gauge_geometry.h"
#include "app_filex.h"
#include "app_inference_runtime.h"
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
extern TX_SEMAPHORE camera_capture_done_semaphore;
extern TX_SEMAPHORE camera_capture_isp_semaphore;
extern volatile bool camera_capture_failed;
extern volatile uint32_t camera_capture_error_code;
extern volatile uint32_t camera_capture_byte_count;
extern volatile bool camera_capture_sof_seen;
extern volatile bool camera_capture_eof_seen;
extern volatile bool camera_capture_frame_done;
extern volatile bool camera_capture_snapshot_armed;
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
 * @brief Brightness classification for the processed capture gate.
 */
typedef enum {
	APP_CAMERA_CAPTURE_BRIGHTNESS_OK = 0,
	APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK,
	APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT,
} AppCameraCapture_BrightnessGate_t;

/**
 * @brief Center-ROI luma summary used to decide whether to accept a frame.
 */
typedef struct {
	uint32_t sample_count;
	uint8_t min_y;
	uint8_t max_y;
	uint32_t mean_y;
} AppCameraCapture_BrightnessStats_t;

/**
 * @brief Measure luma over the full training crop region of a YUV422 frame.
 *
 * Sampling the entire training crop (rather than a small centre ROI) avoids
 * being fooled by specular reflections on the gauge glass, which can make a
 * small centre ROI read as "bright enough" while the rest of the dial face
 * is still underexposed.  The model sees exactly this region, so the mean
 * here directly predicts whether the model input will be well-exposed.
 */
static bool AppCameraCapture_ComputeBrightnessStats(const uint8_t *buffer_ptr,
		uint32_t length_bytes, AppCameraCapture_BrightnessStats_t *stats) {
	const uint32_t frame_width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const uint32_t frame_height_lines = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const uint32_t bytes_per_pixel = CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const uint32_t stride_bytes = frame_width_pixels * bytes_per_pixel;
	uint64_t sum_y = 0U;
	uint32_t sample_count = 0U;
	uint8_t min_y = 0xFFU;
	uint8_t max_y = 0U;

	if ((buffer_ptr == NULL) || (stats == NULL) || (length_bytes < stride_bytes)) {
		return false;
	}

	const AppGaugeGeometry_Crop_t crop = AppGaugeGeometry_TrainingCrop(
			(size_t) frame_width_pixels, (size_t) frame_height_lines);
	const uint32_t x_end = (uint32_t) (crop.x_min + crop.width);
	const uint32_t y_end = (uint32_t) (crop.y_min + crop.height);

	if ((x_end > frame_width_pixels) || (y_end > frame_height_lines)) {
		return false;
	}

	for (uint32_t row = (uint32_t) crop.y_min; row < y_end; row++) {
		const uint32_t row_base = row * stride_bytes;

		if ((row_base + (x_end * bytes_per_pixel)) > length_bytes) {
			return false;
		}

		for (uint32_t col = (uint32_t) crop.x_min; col < x_end; col++) {
			const uint8_t y_sample = buffer_ptr[row_base + (col * bytes_per_pixel)];

			if (y_sample < min_y) {
				min_y = y_sample;
			}
			if (y_sample > max_y) {
				max_y = y_sample;
			}
			sum_y += y_sample;
			sample_count++;
		}
	}

	if (sample_count == 0U) {
		return false;
	}

	stats->sample_count = sample_count;
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
	if ((stats->mean_y < CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD)
			&& (stats->max_y <= CAMERA_CAPTURE_BRIGHTNESS_DARK_MAX_THRESHOLD)) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK;
	}

	if ((stats->mean_y >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD)
			&& (stats->min_y >= CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD)) {
		return APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_BRIGHT;
	}

	return APP_CAMERA_CAPTURE_BRIGHTNESS_OK;
}

/**
 * @brief Print the brightness gate result so we can see why a frame was retried.
 */
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
			"[CAMERA][CAPTURE] Brightness gate (%s): samples=%lu mean=%lu min=%u max=%u thresholds dark<=%u/%u bright>=%u/%u.\r\n",
			decision_label,
			(unsigned long) ((stats != NULL) ? stats->sample_count : 0U),
			(unsigned long) ((stats != NULL) ? stats->mean_y : 0U),
			(unsigned int) ((stats != NULL) ? stats->min_y : 0U),
			(unsigned int) ((stats != NULL) ? stats->max_y : 0U),
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_DARK_MEAN_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_DARK_MAX_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MEAN_THRESHOLD,
			(unsigned int) CAMERA_CAPTURE_BRIGHTNESS_BRIGHT_MIN_THRESHOLD);
}
/* USER CODE END PV */

/**
 * @brief Service ST's IMX335 middleware background process for ISP state updates.
 * @retval true when the background step succeeded or is not used by this driver.
 */
bool AppCameraCapture_RunImx335Background(void) {
	/* The ISP background loop is only needed for the processed image path.
	 * Raw Pipe0 diagnostics bypass the ISP output path, so running AWB/AEC
	 * updates there can trip middleware code that expects the YUV pipeline. */
	if (camera_capture_isp_loop_paused) {
		return true;
	}

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		/* Keep the background path quiescent for now. The live AEC loop was
		 * freezing the board; the capture retry gate below is the safer
		 * steering mechanism for bright and dark scenes. */
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

/**
 * @brief Capture a single frame, save it to the SD card, and queue inference.
 * @retval true when the frame reaches storage successfully.
 */
bool AppCameraCapture_CaptureSingleFrame(uint32_t *captured_bytes_ptr) {
	const ULONG wait_ticks = CameraPlatform_MillisecondsToTicks(
	CAMERA_CAPTURE_TIMEOUT_MS);
	const ULONG poll_ticks = CameraPlatform_MillisecondsToTicks(20U);
	ULONG next_wait_log_tick = 0U;
	ULONG deadline_tick = 0U;
	UINT semaphore_status = TX_SUCCESS;
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

	/* Drain any stale semaphore token before arming the next snapshot. */
	while (tx_semaphore_get(&camera_capture_done_semaphore, TX_NO_WAIT)
			== TX_SUCCESS) {
	}

	/* Match ST's CMW_CAMERA_Start() ordering: arm the CSI/DCMIPP receiver first,
	 * then start the ISP + sensor stream. This avoids missing the first valid
	 * frame while the middleware is bringing the stream up. */
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
	(void) CameraPlatform_LogImx335AutoExposureState("capture-start");
	App_ThreadX_UnlockCameraMiddleware();

	deadline_tick = tx_time_get() + wait_ticks;
	next_wait_log_tick = tx_time_get()
			+ CameraPlatform_MillisecondsToTicks(1000U);
	while (true) {
		semaphore_status = tx_semaphore_get(&camera_capture_done_semaphore,
				poll_ticks);
		if (semaphore_status == TX_SUCCESS) {
			if (!camera_capture_failed) {
				const uint32_t completed_buffer_index =
						camera_capture_active_buffer_index;
				uint32_t completed_nonzero_bytes = 0U;
				uint8_t *completed_buffer_ptr = NULL;
				bool keep_waiting_for_convergence = false;

				completed_buffer_ptr =
						camera_capture_buffers[completed_buffer_index];
				#if 0
				/* Buffer is noncacheable â€“ no invalidate needed, reads go to SRAM. */
				/* Read first 8 bytes straight from SRAM after invalidate â€“ before
				 * the nonzero scan â€“ to show whether the cache or DMA is the issue. */
				{
					volatile uint8_t *vb =
							(volatile uint8_t*) completed_buffer_ptr;
					DebugConsole_Printf(
							"[CAMERA][CAPTURE] Buffer %lu after invalidate first8=[%02X %02X %02X %02X %02X %02X %02X %02X] addr=0x%08lX\r\n",
							(unsigned long) completed_buffer_index,
							(unsigned int) vb[0], (unsigned int) vb[1],
							(unsigned int) vb[2], (unsigned int) vb[3],
							(unsigned int) vb[4], (unsigned int) vb[5],
							(unsigned int) vb[6], (unsigned int) vb[7],
							(unsigned long) (uintptr_t) completed_buffer_ptr);
				}
				/* Scan the capture buffer to find where DMA data landed.
				 * We look for words that are neither the 0xAA fill nor plain zero. */
				{
					const uint32_t scan_words = CAMERA_CAPTURE_BUFFER_SIZE_BYTES
							/ 4U;
					volatile uint32_t *scan_base =
							(volatile uint32_t*) (uintptr_t) completed_buffer_ptr;
					uint32_t first_data = 0xFFFFFFFFU; /* first word != 0xAA and != 0x00 */
					uint32_t first_nonaa = 0xFFFFFFFFU; /* first word != 0xAA (includes zero) */
					uint32_t last_nonaa = 0U;
					uint32_t nonaa_count = 0U;
					uint32_t nonzero_nonaa_count = 0U;
					for (uint32_t wi = 0U; wi < scan_words; wi++) {
						uint32_t v = scan_base[wi];
						if (v != 0xAAAAAAAAU) {
							if (first_nonaa == 0xFFFFFFFFU) {
								first_nonaa = wi;
							}
							last_nonaa = wi;
							nonaa_count++;
							if (v != 0x00000000U) {
								if (first_data == 0xFFFFFFFFU) {
									first_data = wi;
								}
								nonzero_nonaa_count++;
							}
						}
					}
					if (first_data != 0xFFFFFFFFU) {
						DebugConsole_Printf(
								"[CAMERA][SCAN] REAL DATA found at word=0x%05lX addr=0x%08lX val=[0x%08lX 0x%08lX 0x%08lX 0x%08lX] nonzero_nonaa=%lu total_nonaa=%lu\r\n",
								(unsigned long) first_data,
								(unsigned long) ((uintptr_t) completed_buffer_ptr
										+ first_data * 4U),
								(unsigned long) scan_base[first_data],
								(unsigned long) scan_base[first_data + 1U],
								(unsigned long) scan_base[first_data + 2U],
								(unsigned long) scan_base[first_data + 3U],
								(unsigned long) nonzero_nonaa_count,
								(unsigned long) nonaa_count);

						/* Post-scan DCMIPP snapshot */
						{
							DCMIPP_HandleTypeDef *pdc =
									CameraPlatform_GetCaptureDcmippHandle();
							DebugConsole_Printf(
									"[CAMERA][SCAN] Post CMSR1=0x%08lX CMSR2=0x%08lX P0DCCNTR=0x%08lX P0DCLMTR=0x%08lX P0SCSZR=0x%08lX CSI_SR0=0x%08lX CSI_SR1=0x%08lX\r\n",
									(unsigned long) pdc->Instance->CMSR1,
									(unsigned long) pdc->Instance->CMSR2,
									(unsigned long) pdc->Instance->P0DCCNTR,
									(unsigned long) pdc->Instance->P0DCLMTR,
									(unsigned long) pdc->Instance->P0SCSZR,
									(unsigned long) CSI->SR0,
									(unsigned long) CSI->SR1);
						}
					} else if (first_nonaa != 0xFFFFFFFFU) {
						DebugConsole_Printf(
								"[CAMERA][SCAN] Only zeros past 0xAA fill (BSS?): first_word=0x%05lX last=0x%05lX count=%lu addr=0x%08lX â€” DMA may be writing zeros or not writing.\r\n",
								(unsigned long) first_nonaa,
								(unsigned long) last_nonaa,
								(unsigned long) nonaa_count,
								(unsigned long) ((uintptr_t) completed_buffer_ptr
										+ first_nonaa * 4U));
					} else {
						DebugConsole_Printf(
								"[CAMERA][SCAN] All 0xAA in buffer from 0x%08lX â€” DMA not writing to SRAM at all.\r\n",
								(unsigned long) (uintptr_t) completed_buffer_ptr);
					}
				}
				/* Check IAC and RISAF2 for illegal access flags. */
				{
					volatile uint32_t iac_isr0 = IAC->ISR[0];
					volatile uint32_t iac_isr4 = IAC->ISR[4];
					volatile uint32_t r2_iasr = RISAF2_NS->IASR;
					volatile uint32_t r2_iaesr = RISAF2_NS->IAR[0].IAESR;
					volatile uint32_t r2_iaddr = RISAF2_NS->IAR[0].IADDR;
					volatile uint32_t r2s_iasr = RISAF2_S->IASR;
					volatile uint32_t r2s_iaesr = RISAF2_S->IAR[0].IAESR;
					DebugConsole_Printf(
							"[RIF] IAC ISR0=0x%08lX ISR4=0x%08lX | RISAF2_NS IASR=0x%08lX IAESR=0x%08lX IADDR=0x%08lX | RISAF2_S IASR=0x%08lX IAESR=0x%08lX \r\n",
							(unsigned long) iac_isr0, (unsigned long) iac_isr4,
							(unsigned long) r2_iasr, (unsigned long) r2_iaesr,
							(unsigned long) r2_iaddr, (unsigned long) r2s_iasr,
							(unsigned long) r2s_iaesr);
				}
				#endif

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
	}

	(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
	DCMIPP_VIRTUAL_CHANNEL0);
	camera_capture_snapshot_armed = false;
	camera_capture_isp_loop_paused = false;
	return false;
}

/**
 * @brief Capture a single frame, save it to the SD card, and queue inference.
 * @retval true when the frame reaches storage successfully.
 */
bool AppCameraCapture_CaptureAndStoreSingleFrame(void) {
	uint32_t captured_bytes = 0U;
	UINT filex_status = FX_SUCCESS;
	CHAR capture_file_name[CAMERA_CAPTURE_FILE_NAME_LENGTH] = { 0 };
	uint8_t *image_ptr = NULL;
	ULONG image_length = captured_bytes;
	bool result = false;
	const CHAR *file_extension = camera_capture_use_cmw_pipeline ? "yuv422"
			: "raw16";
	const uint32_t max_capture_attempts =
	CAMERA_CAPTURE_BRIGHTNESS_RETRY_LIMIT;
	uint32_t capture_attempt = 0U;
	bool capture_ok = false;
	AppCameraCapture_BrightnessStats_t brightness_stats = { 0 };
	AppCameraCapture_BrightnessGate_t brightness_gate =
	APP_CAMERA_CAPTURE_BRIGHTNESS_OK;

	if (!AppStorage_WaitForMediaReady(CAMERA_STORAGE_WAIT_TIMEOUT_MS)) {
		return false;
	}

	for (capture_attempt = 0U; capture_attempt < max_capture_attempts;
			capture_attempt++) {
		if (capture_attempt > 0U) {
			if (camera_capture_error_code != 0U) {
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Retrying capture after DCMIPP error 0x%08lX.\r\n",
						(unsigned long) camera_capture_error_code);
			} else {
				(void) DebugConsole_WriteString(
						"[CAMERA][CAPTURE] Retrying capture after brightness gate.\r\n");
			}
			DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
		}

		if (AppCameraCapture_CaptureSingleFrame(&captured_bytes)) {
			capture_ok = true;
			image_ptr = camera_capture_result_buffer;
			if (camera_capture_use_cmw_pipeline) {
				if (!AppCameraCapture_ComputeBrightnessStats(image_ptr,
						captured_bytes, &brightness_stats)) {
					DebugConsole_Printf(
							"[CAMERA][CAPTURE] Brightness gate could not analyze processed frame; retrying capture.\r\n");
					capture_ok = false;
					if ((capture_attempt + 1U) < max_capture_attempts) {
						DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
						continue;
					}
					break;
				}

				brightness_gate =
				AppCameraCapture_ClassifyBrightness(&brightness_stats);
				if (brightness_gate != APP_CAMERA_CAPTURE_BRIGHTNESS_OK) {
					AppCameraCapture_LogBrightnessGateDecision(&brightness_stats,
							brightness_gate);
					capture_ok = false;
					if ((capture_attempt + 1U) >= max_capture_attempts) {
						DebugConsole_Printf(
								"[CAMERA][CAPTURE] Brightness gate rejected the frame after %lu attempts.\r\n",
						(unsigned long) max_capture_attempts);
						break;
					}

					if (!CameraPlatform_AdjustImx335ExposureGain(
							brightness_gate
									== APP_CAMERA_CAPTURE_BRIGHTNESS_TOO_DARK)) {
						DebugConsole_Printf(
								"[CAMERA][CAPTURE] Failed to nudge IMX335 exposure/gain after brightness gate rejection.\r\n");
						break;
					}

					DelayMilliseconds_ThreadX(
					CAMERA_CAPTURE_BRIGHTNESS_SETTLE_DELAY_MS);
					continue;
				}
				CameraPlatform_CacheAcceptedExposureGain();
			}

			break;
		}

		if (!AppCameraCapture_ShouldRetryDcmippError(camera_capture_error_code)) {
			break;
		}
	}

	if (!capture_ok) {
		return false;
	}

	image_length = captured_bytes;
	image_ptr = camera_capture_result_buffer;
	if (image_ptr == NULL) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Capture buffer pointer is NULL after frame completion.\r\n");
		goto cleanup;
	}

	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: frame-ready\r\n");
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Frame ready for save: ptr=%p length=%lu pipeline=%s\r\n",
			(void *) image_ptr, (unsigned long) image_length,
			camera_capture_use_cmw_pipeline ? "processed" : "raw");
#endif

	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: preview\r\n");
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	AppCameraDiagnostics_LogCaptureBufferPreview("ready-to-save", image_ptr,
			(uint32_t) image_length);
#endif

	if (camera_capture_use_cmw_pipeline) {
		if (!AppBaselineRuntime_RequestEstimate((const uint8_t *) image_ptr,
					(ULONG) image_length)) {
			DebugConsole_Printf(
					"[BASELINE] Failed to queue classical baseline estimate.\r\n");
		}
	}

	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: build-name\r\n");
	if (!AppStorage_BuildCaptureFileName(capture_file_name,
			sizeof(capture_file_name), file_extension)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to build capture filename.\r\n");
		goto cleanup;
	}
	(void) DebugConsole_WriteString("[CAMERA][CAPTURE] step: build-name-done\r\n");

	if (camera_capture_use_cmw_pipeline) {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		AppCameraCapture_LogCaptureState("processed-capture");
		AppCameraDiagnostics_LogProcessedFrameDiagnostics("processed-capture",
				image_ptr, (uint32_t) image_length);
#endif
	}

	filex_status = AppFileX_WriteCapturedImage(capture_file_name,
			image_ptr, image_length);
	if (filex_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to write image to SD card, status=%lu.\r\n",
				(unsigned long) filex_status);
		goto cleanup;
	}

	if (camera_capture_use_cmw_pipeline) {
		if (!AppInferenceRuntime_RequestDryInference(
					(const uint8_t *) image_ptr, (ULONG) image_length)) {
			DebugConsole_Printf(
					"[AI] Failed to queue one-shot dry-run inference.\r\n");
		}
	}

	result = true;

cleanup:
	camera_capture_isp_loop_paused = false;
	return result;
}
