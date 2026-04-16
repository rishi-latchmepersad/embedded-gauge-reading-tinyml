/*
 *******************************************************************************
 * @file    app_camera_platform.c
 * @brief   Low-level camera board support helpers.
 *******************************************************************************
 */

#include "app_camera_platform.h"

#include "app_camera_buffers.h"
#include "app_camera_config.h"
#include "cmw_camera.h"
#include "cmw_imx335.h"
#include "debug_console.h"
#include "imx335.h"
#include "isp_api.h"
#include "threadx_utils.h"

extern DCMIPP_HandleTypeDef hdcmipp;
extern I2C_HandleTypeDef hi2c2;
extern CMW_IMX335_t camera_sensor;
extern bool camera_cmw_initialized;
extern bool camera_capture_use_cmw_pipeline;
extern bool camera_stream_started;
extern uint8_t *camera_capture_result_buffer;

/**
 * @brief Read the official IMX335 chip-ID register.
 * @param[out] chip_id Receives the register contents on success.
 * @retval HAL status of the I2C memory read transaction.
 */
HAL_StatusTypeDef CameraPlatform_ReadImx335ChipId(uint8_t *chip_id) {
	if (chip_id == NULL) {
		return HAL_ERROR;
	}

	return HAL_I2C_Mem_Read(&hi2c2, BCAMS_IMX_I2C_ADDRESS_HAL,
			IMX335_CHIP_ID_REG, I2C_MEMADD_SIZE_16BIT, chip_id, 1U,
			BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);
}

/**
 * @brief Read back the CSI PFCR after the IMX335 middleware init path.
 * @note This is a read-only sanity check so we can confirm the HAL kept the
 *       expected lane-direction and frequency-range programming intact.
 */
static void CameraPlatform_LogCsiDphySettle(void) {
	DebugConsole_Printf(
			"[CAMERA][DPHY] PFCR after app-layer check: raw=0x%08lX upper=0x%02lX LMCFGR=0x%08lX.\r\n",
			(unsigned long) CSI->PFCR,
			(unsigned long) ((CSI->PFCR >> 16U) & 0xFFUL),
			(unsigned long) CSI->LMCFGR);
}

/**
 * @brief Read a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Read(&hi2c2, dev_addr, reg,
	I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief Write a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Write(&hi2c2, dev_addr, reg,
	I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief Arm a CSI line/byte counter on VC0 so we can confirm line progress.
 * @retval true when the counter was programmed successfully.
 */
bool CameraPlatform_ConfigureCsiLineByteProbe(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();
	DCMIPP_CSI_LineByteCounterConfTypeDef linebyte_config = { 0 };

	if ((capture_dcmipp == NULL) || (capture_dcmipp->Instance == NULL)) {
		return false;
	}

	linebyte_config.VirtualChannel = DCMIPP_VIRTUAL_CHANNEL0;
	linebyte_config.LineCounter =
	CAMERA_CAPTURE_CSI_LB_PROBE_LINE_COUNTER;
	linebyte_config.ByteCounter =
	CAMERA_CAPTURE_CSI_LB_PROBE_BYTE_COUNTER;

	(void) HAL_DCMIPP_CSI_DisableLineByteCounter(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER);
	CSI->FCR0 = (CSI_FCR0_CLB0F | CSI_FCR0_CLB1F | CSI_FCR0_CLB2F
			| CSI_FCR0_CLB3F);

	if (HAL_DCMIPP_CSI_SetLineByteCounterConfig(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER, &linebyte_config) != HAL_OK) {
		return false;
	}

	if (HAL_DCMIPP_CSI_EnableLineByteCounter(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER) != HAL_OK) {
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI line/byte probe armed on VC0 (counter=%lu line=%lu byte=%lu).\r\n",
			(unsigned long) CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER,
			(unsigned long) linebyte_config.LineCounter,
			(unsigned long) linebyte_config.ByteCounter);
	return true;
}

/**
 * @brief Initialize the IMX335 through ST's public camera middleware path.
 * @retval true when the middleware-owned camera stack accepts the sensor setup.
 */
static bool CameraPlatform_InitializeImx335Sensor(void) {
	CMW_CameraInit_t camera_init = { 0 };
	CMW_Advanced_Config_t camera_advanced_config = { 0 };
	int32_t cmw_status = CMW_ERROR_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-defaults\r\n");
	camera_advanced_config.selected_sensor = CMW_IMX335_Sensor;
	cmw_status = CMW_CAMERA_SetDefaultSensorValues(&camera_advanced_config);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to load default IMX335 middleware values, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	camera_init.width = IMX335_SENSOR_WIDTH_PIXELS;
	camera_init.height = IMX335_SENSOR_HEIGHT_LINES;
	camera_init.fps = IMX335_CAPTURE_FRAMERATE_FPS;
	camera_init.mirror_flip = CMW_MIRRORFLIP_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-init\r\n");
	cmw_status = CMW_CAMERA_Init(&camera_init, &camera_advanced_config);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - CMW_CAMERA_Init() failed, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	CameraPlatform_LogCsiDphySettle();

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-test-pattern\r\n");
	cmw_status = CMW_CAMERA_SetTestPattern(IMX335_TEST_PATTERN_MODE);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to configure IMX335 test pattern mode %d, status=%ld.\r\n",
				IMX335_TEST_PATTERN_MODE, (long) cmw_status);
		return false;
	}

#if IMX335_TEST_PATTERN_MODE >= 0
	DebugConsole_Printf("[CAMERA][PROBE] IMX335 test pattern enabled.\r\n");
#else
	DebugConsole_Printf("[CAMERA][PROBE] IMX335 live optical path enabled.\r\n");
#endif

	if (!CameraPlatform_SeedImx335ExposureGain()) {
		return false;
	}
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: seed-done\r\n");

#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
	camera_capture_use_cmw_pipeline = false;
	DebugConsole_Printf("[CAMERA][PROBE] RAW diagnostic capture enabled.\r\n");
#else
	camera_capture_use_cmw_pipeline = true;
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-start\r\n");
	if (!CameraPlatform_EnableImx335AutoExposure()) {
		return false;
	}
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-done\r\n");
	DebugConsole_Printf("[CAMERA][PROBE] Using CMW/ISP capture path.\r\n");
#endif

	camera_cmw_initialized = true;
	camera_stream_started = false;
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: sensor-ready\r\n");

	return true;
}

/**
 * @brief Seed IMX335 exposure and gain with a conservative starting point.
 *
 * ST's middleware initializes the sensor conservatively. We back off the
 * previous maxed-out seed so the live optical path does not clip immediately
 * on bright scenes.
 * @retval true when the middleware accepted the seed settings.
 */
bool CameraPlatform_SeedImx335ExposureGain(void) {
	ISP_SensorInfoTypeDef sensor_info = { 0 };
	uint32_t seed_exposure_us = 0U;
	int32_t seed_gain_mdb = 0;
	int32_t cmw_status = CMW_ERROR_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: seed-start\r\n");
	cmw_status = CMW_CAMERA_GetSensorInfo(&sensor_info);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to read IMX335 sensor info for exposure seeding, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	{
		const uint32_t exposure_range =
				(sensor_info.exposure_max > sensor_info.exposure_min) ?
						(sensor_info.exposure_max - sensor_info.exposure_min) : 0U;
		seed_exposure_us = sensor_info.exposure_min
				+ ((exposure_range
						* CAMERA_IMX335_SEED_EXPOSURE_FRACTION_NUMERATOR)
						/ CAMERA_IMX335_SEED_EXPOSURE_FRACTION_DENOMINATOR);
	}
	if (seed_exposure_us < sensor_info.exposure_min) {
		seed_exposure_us = sensor_info.exposure_min;
	}

	{
		const int32_t gain_range =
				(sensor_info.gain_max > sensor_info.gain_min) ?
						(sensor_info.gain_max - sensor_info.gain_min) : 0;
		seed_gain_mdb = sensor_info.gain_min
				+ ((gain_range
						* CAMERA_IMX335_SEED_GAIN_FRACTION_NUMERATOR)
						/ CAMERA_IMX335_SEED_GAIN_FRACTION_DENOMINATOR);
	}
	if (seed_gain_mdb < sensor_info.gain_min) {
		seed_gain_mdb = sensor_info.gain_min;
	}

	cmw_status = CMW_CAMERA_SetExposure((int32_t) seed_exposure_us);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to seed IMX335 exposure, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	cmw_status = CMW_CAMERA_SetGain(seed_gain_mdb);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to seed IMX335 gain, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Seeded IMX335 exposure to %lu us and gain to %ld mdB.\r\n",
			(unsigned long) seed_exposure_us, (long) seed_gain_mdb);

	return true;
}

/**
 * @brief Nudge the IMX335 exposure and gain toward a brighter or darker frame.
 *
 * This is a small corrective step used by the capture gate when a frame is
 * technically valid but still far too dark or too bright for the model.
 * @param brighten When true, move the sensor toward a brighter capture.
 *                 When false, move it toward a darker capture.
 * @retval true when the new settings were accepted by the middleware.
 */
bool CameraPlatform_AdjustImx335ExposureGain(bool brighten) {
	ISP_SensorInfoTypeDef sensor_info = { 0 };
	int32_t current_exposure_us = 0;
	int32_t current_gain_mdb = 0;
	int32_t exposure_step_us = 0;
	int32_t gain_step_mdb = 0;
	int32_t target_exposure_us = 0;
	int32_t target_gain_mdb = 0;
	bool apply_exposure = false;
	bool apply_gain = false;
	int32_t cmw_status = CMW_ERROR_NONE;

	cmw_status = CMW_CAMERA_GetSensorInfo(&sensor_info);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to read sensor info for brightness adjustment, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	cmw_status = CMW_CAMERA_GetExposure(&current_exposure_us);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to read current IMX335 exposure for brightness adjustment, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	cmw_status = CMW_CAMERA_GetGain(&current_gain_mdb);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to read current IMX335 gain for brightness adjustment, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	{
		const int32_t exposure_range =
				(sensor_info.exposure_max > sensor_info.exposure_min) ?
						((int32_t) sensor_info.exposure_max
								- (int32_t) sensor_info.exposure_min) :
						0;
		const int32_t gain_range =
				(sensor_info.gain_max > sensor_info.gain_min) ?
						(sensor_info.gain_max - sensor_info.gain_min) : 0;

		exposure_step_us = (exposure_range
				* (int32_t) CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_NUMERATOR)
				/ (int32_t) CAMERA_CAPTURE_BRIGHTNESS_EXPOSURE_STEP_DENOMINATOR;
		gain_step_mdb =
				(gain_range
						* (int32_t) CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_NUMERATOR)
						/ (int32_t) CAMERA_CAPTURE_BRIGHTNESS_GAIN_STEP_DENOMINATOR;
	}

	if (exposure_step_us < 1) {
		exposure_step_us = 1;
	}
	if (gain_step_mdb < 1) {
		gain_step_mdb = 1;
	}

	/* Walk exposure first so a dim frame does not jump straight from
	 * under-exposed to heavily over-exposed just because gain moved too. */
	target_exposure_us = current_exposure_us;
	target_gain_mdb = current_gain_mdb;

	if (brighten) {
		if (current_exposure_us < (int32_t) sensor_info.exposure_max) {
			target_exposure_us = current_exposure_us + exposure_step_us;
			if (target_exposure_us > (int32_t) sensor_info.exposure_max) {
				target_exposure_us = (int32_t) sensor_info.exposure_max;
			}
			apply_exposure = (target_exposure_us != current_exposure_us);
		} else {
			target_gain_mdb = current_gain_mdb + gain_step_mdb;
			if (target_gain_mdb > sensor_info.gain_max) {
				target_gain_mdb = sensor_info.gain_max;
			}
			apply_gain = (target_gain_mdb != current_gain_mdb);
		}
	} else {
		if (current_exposure_us > (int32_t) sensor_info.exposure_min) {
			target_exposure_us = current_exposure_us - exposure_step_us;
			if (target_exposure_us < (int32_t) sensor_info.exposure_min) {
				target_exposure_us = (int32_t) sensor_info.exposure_min;
			}
			apply_exposure = (target_exposure_us != current_exposure_us);
		} else {
			target_gain_mdb = current_gain_mdb - gain_step_mdb;
			if (target_gain_mdb < sensor_info.gain_min) {
				target_gain_mdb = sensor_info.gain_min;
			}
			apply_gain = (target_gain_mdb != current_gain_mdb);
		}
	}

	if (!apply_exposure && !apply_gain) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 brightness nudge reached sensor limit; no change applied.\r\n");
		return false;
	}

	if (apply_exposure) {
		cmw_status = CMW_CAMERA_SetExposure(target_exposure_us);
		if (cmw_status != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to update IMX335 exposure to %ld us, status=%ld.\r\n",
					(long) target_exposure_us, (long) cmw_status);
			return false;
		}
	}

	if (apply_gain) {
		cmw_status = CMW_CAMERA_SetGain(target_gain_mdb);
		if (cmw_status != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to update IMX335 gain to %ld mdB, status=%ld.\r\n",
					(long) target_gain_mdb, (long) cmw_status);
			return false;
		}
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Nudged IMX335 %s: exposure=%ld us gain=%ld mdB.%s%s\r\n",
			brighten ? "brighter" : "darker", (long) target_exposure_us,
			(long) target_gain_mdb, apply_exposure ? " exposure" : "",
			apply_gain ? " gain" : "");
	return true;
}

/**
 * @brief Force the IMX335 ISP path into auto-exposure mode.
 *
 * The IMX335 middleware bridge does not expose a sensor-level exposure-mode
 * setter, so the ISP AEC state is the control point for auto exposure here.
 * @retval true when the ISP accepted the AEC enable request.
 */
bool CameraPlatform_EnableImx335AutoExposure(void) {
	uint8_t aec_enabled = 0U;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-call\r\n");
	if (ISP_SetAECState(&camera_sensor.hIsp, 1U) != ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to enable IMX335 ISP auto exposure.\r\n");
		return false;
	}

	(void) CameraPlatform_LogImx335AutoExposureState("probe");

	return true;
}

/**
 * @brief Lock the IMX335 ISP path out of auto-exposure mode.
 *
 * We use AEC during probe to settle the sensor, then disable it before the
 * regular capture loop so the gauge reading is not fighting a moving target.
 * @retval true when the ISP accepted the AEC disable request.
 */
bool CameraPlatform_DisableImx335AutoExposure(void) {
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-lock\r\n");
	if (ISP_SetAECState(&camera_sensor.hIsp, 0U) != ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to disable IMX335 ISP auto exposure.\r\n");
		return false;
	}

	(void) CameraPlatform_LogImx335AutoExposureState("capture-lock");
	return true;
}

/**
 * @brief Log the current IMX335 ISP auto-exposure state.
 *
 * This is a cheap readback that helps us prove AEC is still enabled at the
 * capture boundary, not just during initial probe.
 * @param reason Short label that explains when the check ran.
 * @retval true when the readback succeeded and AEC reported enabled.
 */
bool CameraPlatform_LogImx335AutoExposureState(const char *reason) {
	uint8_t aec_enabled = 0U;
	const char *tag = (reason != NULL) ? reason : "capture";

	if (ISP_GetAECState(&camera_sensor.hIsp, &aec_enabled) == ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE] IMX335 ISP auto exposure state (%s): AEC=%u\r\n",
				tag, (unsigned int) aec_enabled);
		return (aec_enabled != 0U);
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE] IMX335 ISP auto exposure state (%s): readback failed.\r\n",
			tag);
	return false;
}

/**
 * @brief Re-apply the configured IMX335 test pattern after streaming starts.
 *
 * Some sensors latch the test-pattern generator more reliably once the stream
 * is already live, so we re-write the configured pattern as a low-risk
 * diagnostic nudge after start-up.
 */
void CameraPlatform_ReapplyImx335TestPattern(void) {
#if IMX335_TEST_PATTERN_MODE >= 0
	int32_t cmw_status = CMW_CAMERA_SetTestPattern(
	IMX335_TEST_PATTERN_MODE);
	uint8_t tpg_value = 0U;

	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Warning: failed to reapply IMX335 test pattern mode %d after stream start, status=%ld.\r\n",
				IMX335_TEST_PATTERN_MODE, (long) cmw_status);
		return;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] IMX335 test pattern re-applied after stream start (mode=%d).\r\n",
			IMX335_TEST_PATTERN_MODE);

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_TPG, &tpg_value, 1U) == IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 test-pattern register = 0x%02X after stream start.\r\n",
				(unsigned int) tpg_value);
	}
#endif
}

/**
 * @brief Configure the capture pipe using ST's camera middleware crop/downsize helpers.
 * @retval true when the output path is ready for a 224x224 YUV422 frame.
 */
bool CameraPlatform_PrepareDcmippSnapshot(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		CMW_DCMIPP_Conf_t pipe_request = { 0 };
		uint32_t pitch_bytes = 0U;

		pipe_request.output_width = CAMERA_CAPTURE_WIDTH_PIXELS;
		pipe_request.output_height = CAMERA_CAPTURE_HEIGHT_PIXELS;
		pipe_request.output_format =
		DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1;
		pipe_request.output_bpp = CAMERA_CAPTURE_BYTES_PER_PIXEL;
		pipe_request.enable_swap = 0;
		pipe_request.enable_gamma_conversion = 0;
		pipe_request.mode = CMW_Aspect_ratio_manual_roi;
		{
			const uint32_t sensor_square_side =
					(IMX335_SENSOR_WIDTH_PIXELS < IMX335_SENSOR_HEIGHT_LINES) ?
							IMX335_SENSOR_WIDTH_PIXELS :
							IMX335_SENSOR_HEIGHT_LINES;

			pipe_request.manual_conf.width = sensor_square_side;
			pipe_request.manual_conf.height = sensor_square_side;
			pipe_request.manual_conf.offset_x =
					(IMX335_SENSOR_WIDTH_PIXELS - sensor_square_side) / 2U;
			pipe_request.manual_conf.offset_y =
					(IMX335_SENSOR_HEIGHT_LINES - sensor_square_side) / 2U;
		}

		if (CMW_CAMERA_SetPipeConfig(CAMERA_CAPTURE_PIPE, &pipe_request,
				&pitch_bytes) != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW_CAMERA_SetPipeConfig() failed for PIPE1.\r\n");
			return false;
		}

		return true;
	}

	{
		DCMIPP_CSI_PIPE_ConfTypeDef csi_pipe_config = { 0 };
		DCMIPP_PipeConfTypeDef pipe_config = { 0 };
		DCMIPP_CropConfTypeDef crop_config = { 0 };

		csi_pipe_config.DataTypeMode = DCMIPP_DTMODE_DTIDA;
		csi_pipe_config.DataTypeIDA = DCMIPP_DT_RAW10;
		csi_pipe_config.DataTypeIDB = 0U;
		if (HAL_DCMIPP_CSI_PIPE_SetConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &csi_pipe_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 for RAW10 input.\r\n");
			return false;
		}

		pipe_config.FrameRate = DCMIPP_FRAME_RATE_ALL;
		pipe_config.PixelPipePitch = 0U;
		pipe_config.PixelPackerFormat = 0U;
		if (HAL_DCMIPP_PIPE_SetConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &pipe_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 snapshot settings.\r\n");
			return false;
		}

		if (HAL_DCMIPP_CSI_SetVCConfig(capture_dcmipp,
		DCMIPP_VIRTUAL_CHANNEL0,
		DCMIPP_CSI_DT_BPP10) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure CSI VC0 as RAW10.\r\n");
			return false;
		}

		if (!CameraPlatform_ConfigureCsiLineByteProbe()) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to arm CSI line/byte probe for VC0.\r\n");
			return false;
		}

		crop_config.VStart = CAMERA_CAPTURE_RAW_TOP_SKIP_LINES;
		crop_config.HStart = 0U;
		crop_config.VSize = CAMERA_CAPTURE_HEIGHT_PIXELS;
		crop_config.HSize = CAMERA_CAPTURE_WIDTH_PIXELS;
		crop_config.PipeArea = DCMIPP_POSITIVE_AREA;
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] RAW crop at HStart=%lu VStart=%lu size=%lux%lu (skip=%lu top lines).\r\n",
				(unsigned long) crop_config.HStart,
				(unsigned long) crop_config.VStart,
				(unsigned long) crop_config.HSize,
				(unsigned long) crop_config.VSize,
				(unsigned long) CAMERA_CAPTURE_RAW_TOP_SKIP_LINES);
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Applying PIPE0 crop config.\r\n");
		if (HAL_DCMIPP_PIPE_SetCropConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &crop_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 crop window.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 crop config applied.\r\n");
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Enabling PIPE0 crop window.\r\n");
		if (HAL_DCMIPP_PIPE_EnableCrop(capture_dcmipp,
		CAMERA_CAPTURE_PIPE) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to enable PIPE0 crop window.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 crop window enabled.\r\n");

		if (HAL_DCMIPP_PIPE_EnableLimitEvent(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, CAMERA_CAPTURE_BUFFER_SIZE_BYTES) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to arm PIPE0 dump limit.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 dump limit armed: P0DCLMTR=0x%08lX.\r\n",
				(unsigned long) capture_dcmipp->Instance->P0DCLMTR);
	}

	return true;
}

/**
 * @brief Print a staged diagnostic sequence for B-CAMS-IMX camera bring-up.
 * @return TX_SUCCESS when the sensor probe succeeds, TX_NOT_AVAILABLE otherwise.
 */
UINT CameraPlatform_ProbeBCamsImx(void) {
	HAL_StatusTypeDef probe_status = HAL_ERROR;
	uint8_t chip_id = 0U;

	DebugConsole_Printf("[CAMERA][PROBE] Probing camera stack...\r\n");
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: reset\r\n");
	CameraPlatform_ResetImx335Module();

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: i2c-ack\r\n");
	probe_status = HAL_I2C_IsDeviceReady(&hi2c2, BCAMS_IMX_I2C_ADDRESS_HAL,
			BCAMS_IMX_I2C_PROBE_TRIALS, BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);
	if (probe_status != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Sensor did not ACK on I2C2 at 7-bit address 0x%02X.\r\n",
				(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Sensor ACKed on I2C2 at 7-bit address 0x%02X.\r\n",
			(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: chip-id\r\n");
	if (CameraPlatform_ReadImx335ChipId(&chip_id) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to read IMX335 ID register.\r\n");
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - IMX335 ID register 0x3912 = 0x%02X.\r\n",
			(unsigned int) chip_id);
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: sensor-init\r\n");
	if (!CameraPlatform_InitializeImx335Sensor()) {
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf("[CAMERA][PROBE] Sensor probe OK.\r\n");
	DebugConsole_Printf("[CAMERA][PROBE] Camera stack ready.\r\n");
	return TX_SUCCESS;
}

/**
 * @brief Apply the MB1854 enable/reset sequence used by ST's camera middleware.
 */
void CameraPlatform_ResetImx335Module(void) {
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: power-on\r\n");
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_POWER_SETTLE_DELAY_MS);

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: reset-pulse\r\n");
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin, GPIO_PIN_RESET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_RESET_ASSERT_DELAY_MS);
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_RESET_RELEASE_DELAY_MS);

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Applied IMX335 reset pulse after module enable.\r\n");
}

/**
 * @brief Drive the camera module enable pin in the form expected by ST's middleware.
 */
void CameraPlatform_CmwEnablePin(int value) {
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin,
			(value != 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/**
 * @brief Drive the camera reset pin in the form expected by ST's middleware.
 */
void CameraPlatform_CmwShutdownPin(int value) {
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin,
			(value != 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/**
 * @brief Delay helper used by ST's camera middleware from the ThreadX camera thread.
 */
void CameraPlatform_CmwDelay(uint32_t delay_ms) {
	DelayMilliseconds_ThreadX(delay_ms);
}

/**
 * @brief Return the active DCMIPP handle used by the current capture path.
 */
DCMIPP_HandleTypeDef *CameraPlatform_GetCaptureDcmippHandle(void) {
	DCMIPP_HandleTypeDef *cmw_handle = CMW_CAMERA_GetDCMIPPHandle();

	if ((cmw_handle != NULL) && (cmw_handle->Instance != NULL)) {
		return cmw_handle;
	}

	return &hdcmipp;
}

/**
 * @brief Adapter from the HAL tick API to the ST IMX335 driver callback type.
 * @retval Current HAL tick in milliseconds.
 */
int32_t CameraPlatform_GetTickMs(void) {
	return ThreadxUtils_GetTickMs();
}

/**
 * @brief Convert milliseconds to ThreadX ticks, rounding up so waits do not underflow.
 * @param timeout_ms Timeout in milliseconds.
 * @retval Equivalent timeout in scheduler ticks.
 */
ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms) {
	return ThreadxUtils_MillisecondsToTicks(timeout_ms);
}

/**
 * @brief Arm the DCMIPP CSI pipe for the next single-frame snapshot.
 * @retval true when the receiver is ready to accept the next frame.
 */
bool CameraPlatform_StartDcmippSnapshot(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if ((capture_dcmipp == NULL) || (capture_dcmipp->Instance == NULL)
			|| (camera_capture_result_buffer == NULL)) {
		return false;
	}

	if (HAL_DCMIPP_CSI_PIPE_Start(capture_dcmipp, CAMERA_CAPTURE_PIPE,
	DCMIPP_VIRTUAL_CHANNEL0, (uint32_t) camera_capture_result_buffer,
	CMW_MODE_SNAPSHOT) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to start DCMIPP CSI pipe for snapshot mode.\r\n");
		return false;
	}

	return true;
}

/**
 * @brief Start the IMX335 sensor stream using the same register order as the
 *        previous inline coordinator logic.
 * @retval true when the sensor is streaming or has already been started.
 */
bool CameraPlatform_StartImx335Stream(void) {
	uint8_t mode_select = 0x00U;

	if (camera_stream_started) {
		return true;
	}

	if (CameraPlatform_I2cWriteReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to write MODE_SELECT=0x00 to start IMX335 streaming.\r\n");
		return false;
	}
	DelayMilliseconds_ThreadX(20U);

	{
		uint8_t xmsta_master_start_value = 0x00U;

		if (CameraPlatform_I2cWriteReg(BCAMS_IMX_I2C_ADDRESS_HAL,
		IMX335_REG_XMSTA, &xmsta_master_start_value, 1U) != IMX335_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Warning: XMSTA write failed after MODE_SELECT.\r\n");
		}
	}
	DelayMilliseconds_ThreadX(5U);

	if (!CameraPlatform_SeedImx335ExposureGain()) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Warning: failed to re-seed IMX335 exposure/gain after raw stream start.\r\n");
	}

	CameraPlatform_ReapplyImx335TestPattern();

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 entered streaming, but mode-select readback failed.\r\n");
	}

	camera_stream_started = true;
	return true;
}
