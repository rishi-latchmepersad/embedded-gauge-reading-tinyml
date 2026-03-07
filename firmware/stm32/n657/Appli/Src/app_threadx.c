/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_threadx.c
 * @author  MCD Application Team
 * @brief   ThreadX applicative file
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_threadx.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>
#include "main.h"
#include "debug_console.h"
#include "threadx_utils.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define CAMERA_INIT_THREAD_STACK_SIZE_BYTES 2048U
#define CAMERA_INIT_THREAD_PRIORITY         12U
#define CAMERA_INIT_STARTUP_DELAY_MS        200U
#define BCAMS_IMX_I2C_ADDRESS_7BIT          0x1AU
#define BCAMS_IMX_I2C_ADDRESS_HAL           (BCAMS_IMX_I2C_ADDRESS_7BIT << 1U)
#define BCAMS_IMX_I2C_PROBE_TRIALS          3U
#define BCAMS_IMX_I2C_PROBE_TIMEOUT_MS      20U
#define BCAMS_IMX_POWER_SETTLE_DELAY_MS     5U
#define IMX335_REG_ID                       0x3912U
#define IMX335_CHIP_ID                      0x00U

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* Dedicated ThreadX object and stack for camera connection diagnostics. */
static TX_THREAD camera_init_thread;
static ULONG camera_init_thread_stack[CAMERA_INIT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_init_thread_created = false;

/* Reuse the CubeMX-generated camera control I2C instance from main.c. */
extern I2C_HandleTypeDef hi2c2;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input);

/**
 * @brief Print a staged diagnostic sequence for B-CAMS-IMX camera bring-up.
 * @return true when a camera probe callback reports success, false otherwise.
 */
static bool Camera_ProbeBCamsImx(void);

/**
 * @brief Weak probe hook for board-specific camera/driver integration.
 * @return TX_SUCCESS on successful camera detection, an error code otherwise.
 */
__attribute__((weak))   UINT CameraPlatform_ProbeBCamsImx(void);

/* USER CODE END PFP */

/**
 * @brief  Application ThreadX Initialization.
 * @param memory_ptr: memory pointer
 * @retval int
 */
UINT App_ThreadX_Init(VOID *memory_ptr) {
	UINT ret = TX_SUCCESS;
	/* USER CODE BEGIN App_ThreadX_MEM_POOL */

	/* USER CODE END App_ThreadX_MEM_POOL */
	/* USER CODE BEGIN App_ThreadX_Init */
	(void) memory_ptr;

	/* Defer thread creation until App_ThreadX_Start() so startup ordering is explicit. */
	DebugConsole_Printf(
			"[CAMERA][THREAD] ThreadX app init complete. Waiting to start camera thread...\r\n");
	/* USER CODE END App_ThreadX_Init */

	return ret;
}

UINT App_ThreadX_Start(void) {
	/* Keep this function idempotent to protect against accidental double-start. */
	if (camera_init_thread_created) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Start skipped: camera init thread already created.\r\n");
		return TX_SUCCESS;
	}

	/* Create a dedicated thread so camera probing is isolated from other startup work. */
	const UINT create_status = tx_thread_create(&camera_init_thread,
			"camera_init", CameraInitThread_Entry, 0U, camera_init_thread_stack,
			sizeof(camera_init_thread_stack),
			CAMERA_INIT_THREAD_PRIORITY,
			CAMERA_INIT_THREAD_PRIORITY,
			TX_NO_TIME_SLICE,
			TX_AUTO_START);

	if (create_status != TX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Failed to create camera init thread, status=%lu\r\n",
				(unsigned long) create_status);
		return create_status;
	}

	camera_init_thread_created = true;
	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera init thread created and started.\r\n");
	return TX_SUCCESS;
}

/**
 * @brief  Function that implements the kernel's initialization.
 * @param  None
 * @retval None
 */
void MX_ThreadX_Init(void) {
	/* USER CODE BEGIN Before_Kernel_Start */

	/* USER CODE END Before_Kernel_Start */

	tx_kernel_enter();

	/* USER CODE BEGIN Kernel_Start_Error */

	/* USER CODE END Kernel_Start_Error */
}

/* USER CODE BEGIN 1 */

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input) {
	(void) thread_input;

	DebugConsole_Printf(
			"[CAMERA][THREAD] Initializing camera diagnostics thread...\r\n");

	/* Delay a little to let other startup logs complete before camera probing. */
	DelayMilliseconds_ThreadX(CAMERA_INIT_STARTUP_DELAY_MS);

	DebugConsole_Printf(
			"[CAMERA][THREAD] Starting B-CAMS-IMX MIPI connection attempt...\r\n");

	if (Camera_ProbeBCamsImx()) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe completed successfully.\r\n");
	} else {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe failed or is not configured yet.\r\n");
	}

	/* Keep the diagnostics thread alive so the stack/object remain valid forever. */
	while (1) {
		DelayMilliseconds_ThreadX(1000U);
	}
}

/**
 * @brief Print a staged diagnostic sequence for B-CAMS-IMX camera bring-up.
 * @return true when a camera probe callback reports success, false otherwise.
 */
static bool Camera_ProbeBCamsImx(void) {
	bool stage_ok = true;

	DebugConsole_Printf(
			"[CAMERA][PROBE] Stage 1/5: Validate firmware camera features.\r\n");
#ifdef HAL_DCMIPP_MODULE_ENABLED
	DebugConsole_Printf(
			"[CAMERA][PROBE]   - HAL_DCMIPP_MODULE_ENABLED = ON\r\n");
#else
  DebugConsole_Printf("[CAMERA][PROBE]   - HAL_DCMIPP_MODULE_ENABLED = OFF (enable for image pipeline).\r\n");
  stage_ok = false;
#endif

#ifdef HAL_I2C_MODULE_ENABLED
	DebugConsole_Printf("[CAMERA][PROBE]   - HAL_I2C_MODULE_ENABLED = ON\r\n");
#else
  DebugConsole_Printf("[CAMERA][PROBE]   - HAL_I2C_MODULE_ENABLED = OFF (sensor control bus usually needed).\r\n");
#endif

#ifdef HAL_I3C_MODULE_ENABLED
  DebugConsole_Printf("[CAMERA][PROBE]   - HAL_I3C_MODULE_ENABLED = ON\r\n");
#else
	DebugConsole_Printf("[CAMERA][PROBE]   - HAL_I3C_MODULE_ENABLED = OFF\r\n");
#endif

	DebugConsole_Printf(
			"[CAMERA][PROBE] Stage 2/5: Check MIPI CSI/DCMIPP peripheral clocks.\r\n");
	DebugConsole_Printf("[CAMERA][PROBE]   - CSI clock enabled: %lu\r\n",
			(unsigned long) __HAL_RCC_CSI_IS_CLK_ENABLED());
	DebugConsole_Printf("[CAMERA][PROBE]   - DCMIPP clock enabled: %lu\r\n",
			(unsigned long) __HAL_RCC_DCMIPP_IS_CLK_ENABLED());

	DebugConsole_Printf(
			"[CAMERA][PROBE] Stage 3/5: Check likely control bus clocks (I2C/I3C).\r\n");
	DebugConsole_Printf("[CAMERA][PROBE]   - I2C1 clock enabled: %lu\r\n",
			(unsigned long) __HAL_RCC_I2C1_IS_CLK_ENABLED());
	DebugConsole_Printf("[CAMERA][PROBE]   - I2C2 clock enabled: %lu\r\n",
			(unsigned long) __HAL_RCC_I2C2_IS_CLK_ENABLED());
	DebugConsole_Printf("[CAMERA][PROBE]   - I3C1 clock enabled: %lu\r\n",
			(unsigned long) __HAL_RCC_I3C1_IS_CLK_ENABLED());

	DebugConsole_Printf(
			"[CAMERA][PROBE] Stage 4/5: Run board-specific sensor probe callback.\r\n");
	const UINT probe_status = CameraPlatform_ProbeBCamsImx();
	if (probe_status == TX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - CameraPlatform_ProbeBCamsImx() returned TX_SUCCESS.\r\n");
	} else {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - CameraPlatform_ProbeBCamsImx() failed, status=%lu\r\n",
				(unsigned long) probe_status);
		stage_ok = false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE] Stage 5/5: Final diagnostic verdict.\r\n");
	if (stage_ok) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - PASS: camera stack appears connected/configured.\r\n");
	} else {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - FAIL: inspect earlier stages for missing config.\r\n");
	}

	return stage_ok;
}

/**
 * @brief Weak probe hook for board-specific camera/driver integration.
 * @return TX_NOT_AVAILABLE to indicate the probe is not wired yet.
 */
UINT CameraPlatform_ProbeBCamsImx(void) {
	HAL_StatusTypeDef probe_status = HAL_ERROR;
	uint8_t chip_id = 0U;

	/* Enable camera module power before attempting control-bus traffic. */
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_POWER_SETTLE_DELAY_MS);

	/* Probe the camera control address to confirm the sensor bus is alive. */
	probe_status = HAL_I2C_IsDeviceReady(&hi2c2,
	BCAMS_IMX_I2C_ADDRESS_HAL,
	BCAMS_IMX_I2C_PROBE_TRIALS,
	BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);

	if (probe_status == HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Sensor ACKed on I2C2 at 7-bit address 0x%02X.\r\n",
				(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);

		/* Read the IMX335 chip ID register to confirm the sensor identity. */
		probe_status = HAL_I2C_Mem_Read(&hi2c2,
				BCAMS_IMX_I2C_ADDRESS_HAL,
				IMX335_REG_ID,
				I2C_MEMADD_SIZE_16BIT,
				&chip_id,
				1U,
				BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);
		if (probe_status != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Failed to read IMX335 ID register 0x%04X.\r\n",
					(unsigned int) IMX335_REG_ID);
			return TX_NOT_AVAILABLE;
		}

		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ID register 0x%04X = 0x%02X.\r\n",
				(unsigned int) IMX335_REG_ID,
				(unsigned int) chip_id);
		if (chip_id != IMX335_CHIP_ID) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Unexpected chip ID, expected 0x%02X.\r\n",
					(unsigned int) IMX335_CHIP_ID);
			return TX_NOT_AVAILABLE;
		}

		return TX_SUCCESS;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - No ACK from sensor on I2C2 at 7-bit address 0x%02X.\r\n"
					"[CAMERA][PROBE]   - Verify camera power, reset, and sensor/BSP driver integration.\r\n",
			(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
	return TX_NOT_AVAILABLE;
}

/* USER CODE END 1 */
