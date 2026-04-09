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
#include "debug_console.h"
#include "imx335.h"
#include "threadx_utils.h"

extern DCMIPP_HandleTypeDef hdcmipp;
extern I2C_HandleTypeDef hi2c2;
extern bool camera_stream_started;

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
