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
#include <stdint.h>
#include <string.h>
#include "app_filex.h"
#include "main.h"
#include "debug_console.h"
#include "threadx_utils.h"
#include "../Middlewares/Third_Party/Camera_Middleware/sensors/imx335/imx335.h"

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
#define BCAMS_IMX_RESET_ASSERT_DELAY_MS     1U
#define BCAMS_IMX_RESET_RELEASE_DELAY_MS    1U
#define CAMERA_CAPTURE_WIDTH_PIXELS         640U
#define CAMERA_CAPTURE_HEIGHT_PIXELS        480U
#define CAMERA_CAPTURE_BYTES_PER_PIXEL      2U
#define CAMERA_SENSOR_WIDTH_PIXELS          2592U
#define CAMERA_SENSOR_HEIGHT_PIXELS         1944U
#define CAMERA_CAPTURE_CROP_HSTART          ((CAMERA_SENSOR_WIDTH_PIXELS - CAMERA_CAPTURE_WIDTH_PIXELS) / 2U)
#define CAMERA_CAPTURE_CROP_VSTART          ((CAMERA_SENSOR_HEIGHT_PIXELS - CAMERA_CAPTURE_HEIGHT_PIXELS) / 2U)
#define CAMERA_CAPTURE_BUFFER_SIZE_BYTES    (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_HEIGHT_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)
#define CAMERA_CAPTURE_TIMEOUT_MS           3000U
#define CAMERA_STORAGE_WAIT_TIMEOUT_MS      10000U
#define CAMERA_CAPTURE_RETRY_DELAY_MS       50U
#define IMX335_CAPTURE_FRAMERATE_FPS        10
#define CAMERA_CAPTURE_FILE_NAME            "capture_000.raw16"
/* Match ST's IMX335 middleware and upstream Linux driver ID check. */
#define IMX335_CHIP_ID_REG                 0x3912U
#define IMX335_CHIP_ID_VALUE               0x00U
/* Temporary diagnostic: use the sensor's internal vertical color bars.
 * Set to -1 to return to the normal optical image path. */
#define IMX335_TEST_PATTERN_MODE           11

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
static IMX335_Object_t camera_sensor;
static TX_SEMAPHORE camera_capture_done_semaphore;
static bool camera_capture_sync_created = false;
static bool camera_stream_started = false;
static volatile bool camera_capture_failed = false;
static volatile uint32_t camera_capture_error_code = 0U;
static volatile uint32_t camera_capture_byte_count = 0U;
static volatile bool camera_capture_sof_seen = false;
static volatile bool camera_capture_eof_seen = false;
static volatile bool camera_capture_frame_done = false;
static volatile uint32_t camera_capture_line_error_count = 0U;
static volatile uint32_t camera_capture_line_error_mask = 0U;
static uint8_t camera_capture_buffer[CAMERA_CAPTURE_BUFFER_SIZE_BYTES]
__attribute__((aligned(16)));

/* Reuse the CubeMX-generated camera control I2C instance from main.c. */
extern DCMIPP_HandleTypeDef hdcmipp;
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
 * @brief Read the IMX335 chip ID register over the camera control bus.
 * @param[out] chip_id Receives the 8-bit IMX335 identification value.
 * @retval HAL status from the I2C register read.
 */
static HAL_StatusTypeDef CameraPlatform_ReadImx335ChipId(uint8_t *chip_id);
static void CameraPlatform_ResetImx335Module(void);

/**
 * @brief Register bus callbacks with the ST IMX335 driver object.
 * @retval true when the driver object is ready for sensor commands.
 */
static bool CameraPlatform_InitImx335Driver(void);

/**
 * @brief Run the vendor IMX335 sensor initialization sequence without streaming.
 * @retval true when the sensor accepted the default frequency and init tables.
 */
static bool CameraPlatform_InitializeImx335Sensor(void);
static bool CameraPlatform_StartImx335Stream(void);
static bool CameraPlatform_WaitForStorageReady(uint32_t timeout_ms);
static bool CameraPlatform_CaptureAndStoreSingleFrame(void);
static bool CameraPlatform_CaptureSingleFrame(uint32_t *captured_bytes_ptr);
static bool CameraPlatform_PrepareDcmippSnapshot(void);
static bool CameraPlatform_StartDcmippSnapshot(void);
static ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms);
void HAL_DCMIPP_PIPE_FrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Pipe);
void HAL_DCMIPP_PIPE_ErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Pipe);
void HAL_DCMIPP_ErrorCallback(DCMIPP_HandleTypeDef *hdcmipp);
void HAL_DCMIPP_CSI_ClockChangerFifoFullEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp);
void HAL_DCMIPP_CSI_LineErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t DataLane);
void HAL_DCMIPP_CSI_ShortPacketDetectionEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp);
void HAL_DCMIPP_CSI_StartOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel);
void HAL_DCMIPP_CSI_EndOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel);

/**
 * @brief Adapter from HAL tick API to the ST IMX335 driver callback type.
 * @retval Current HAL tick in milliseconds.
 */
static int32_t CameraPlatform_GetTickMs(void);

/**
 * @brief Local I2C init hook matching the IMX335 vendor driver callback type.
 * @retval IMX335_OK-style status code.
 */
static int32_t CameraPlatform_I2cInit(void);

/**
 * @brief Local I2C deinit hook matching the IMX335 vendor driver callback type.
 * @retval IMX335_OK-style status code.
 */
static int32_t CameraPlatform_I2cDeInit(void);

/**
 * @brief Read a 16-bit IMX335 register over I2C2.
 * @retval IMX335_OK on success, IMX335_ERROR on failure.
 */
static int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);

/**
 * @brief Write a 16-bit IMX335 register over I2C2.
 * @retval IMX335_OK on success, IMX335_ERROR on failure.
 */
static int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);

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
UINT App_ThreadX_Init(VOID *memory_ptr)
{
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

/* USER CODE BEGIN App_ThreadX_Start */
UINT App_ThreadX_Start(void) {
	/* Keep this function idempotent to protect against accidental double-start. */
	if (camera_init_thread_created) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Start skipped: camera init thread already created.\r\n");
		return TX_SUCCESS;
	}

	if (!camera_capture_sync_created) {
		const UINT semaphore_status = tx_semaphore_create(
				&camera_capture_done_semaphore, "camera_capture_done", 0U);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create capture semaphore, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}

		camera_capture_sync_created = true;
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
/* USER CODE END App_ThreadX_Start */

  /**
  * @brief  Function that implements the kernel's initialization.
  * @param  None
  * @retval None
  */
void MX_ThreadX_Init(void)
{
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

		if (CameraPlatform_CaptureAndStoreSingleFrame()) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Captured and stored first image successfully.\r\n");
		} else {
			DebugConsole_Printf(
					"[CAMERA][THREAD] First image capture/save attempt failed.\r\n");
		}
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

	CameraPlatform_ResetImx335Module();

	/* Probe the camera control address to confirm the sensor bus is alive. */
	probe_status = HAL_I2C_IsDeviceReady(&hi2c2,
	BCAMS_IMX_I2C_ADDRESS_HAL,
	BCAMS_IMX_I2C_PROBE_TRIALS,
	BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);

	if (probe_status == HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Sensor ACKed on I2C2 at 7-bit address 0x%02X.\r\n",
				(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);

		/* Read the same chip-ID register used by ST's IMX335 middleware. */
		probe_status = CameraPlatform_ReadImx335ChipId(&chip_id);
		if (probe_status != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Failed to read IMX335 ID register 0x%04X.\r\n",
					(unsigned int) IMX335_CHIP_ID_REG);
			return TX_NOT_AVAILABLE;
		}

		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ID register 0x%04X = 0x%02X.\r\n",
				(unsigned int) IMX335_CHIP_ID_REG,
				(unsigned int) chip_id);
		if (chip_id != IMX335_CHIP_ID_VALUE) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Unexpected chip ID, expected 0x%02X.\r\n",
					(unsigned int) IMX335_CHIP_ID_VALUE);
			return TX_NOT_AVAILABLE;
		}

		if (!CameraPlatform_InitializeImx335Sensor()) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - IMX335 init sequence failed.\r\n");
			return TX_NOT_AVAILABLE;
		}

		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 init sequence completed.\r\n");

		return TX_SUCCESS;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - No ACK from sensor on I2C2 at 7-bit address 0x%02X.\r\n"
					"[CAMERA][PROBE]   - Verify camera power, reset, and sensor/BSP driver integration.\r\n",
			(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
	return TX_NOT_AVAILABLE;
}

/**
 * @brief Read the official IMX335 chip-ID register.
 * @param[out] chip_id Receives the register contents on success.
 * @retval HAL status of the I2C memory read transaction.
 */
static HAL_StatusTypeDef CameraPlatform_ReadImx335ChipId(uint8_t *chip_id) {
	if (chip_id == NULL) {
		return HAL_ERROR;
	}

	return HAL_I2C_Mem_Read(&hi2c2,
			BCAMS_IMX_I2C_ADDRESS_HAL,
			IMX335_CHIP_ID_REG,
			I2C_MEMADD_SIZE_16BIT,
			chip_id,
			1U,
			BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);
}

/**
 * @brief Apply the MB1854 enable/reset sequence used by ST's camera middleware.
 *
 * The IMX335 board expects module power and the 24 MHz camera clock to be
 * enabled before the reset pin is pulsed low then high. Mirroring ST's own
 * sequence here keeps the standalone app aligned with the reference BSP.
 */
static void CameraPlatform_ResetImx335Module(void) {
	/* Keep the module powered so the board can provide DVDD and CAM_CLK. */
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_POWER_SETTLE_DELAY_MS);

	/* Pulse the sensor reset line exactly as the vendor middleware does. */
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin, GPIO_PIN_RESET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_RESET_ASSERT_DELAY_MS);
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_RESET_RELEASE_DELAY_MS);

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Applied IMX335 reset pulse after module enable.\r\n");
}

/**
 * @brief Prepare the ST IMX335 driver with the project's existing I2C bus hooks.
 * @retval true when bus registration succeeds.
 */
static bool CameraPlatform_InitImx335Driver(void) {
	IMX335_IO_t io_ctx;

	io_ctx.Init = CameraPlatform_I2cInit;
	io_ctx.DeInit = CameraPlatform_I2cDeInit;
	io_ctx.Address = BCAMS_IMX_I2C_ADDRESS_HAL;
	io_ctx.WriteReg = CameraPlatform_I2cWriteReg;
	io_ctx.ReadReg = CameraPlatform_I2cReadReg;
	io_ctx.GetTick = CameraPlatform_GetTickMs;

	memset(&camera_sensor, 0, sizeof(camera_sensor));

	if (IMX335_RegisterBusIO(&camera_sensor, &io_ctx) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to register IMX335 bus callbacks.\r\n");
		return false;
	}

	return true;
}

/**
 * @brief Start IMX335 streaming once the init tables have already been applied.
 * @retval true when the sensor enters streaming mode.
 */
static bool CameraPlatform_StartImx335Stream(void) {
	uint8_t mode_select = IMX335_MODE_STANDBY;

	if (camera_stream_started) {
		return true;
	}

	if (IMX335_Start(&camera_sensor) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to start IMX335 streaming.\r\n");
		return false;
	}

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL, IMX335_REG_MODE_SELECT,
			&mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 entered streaming, but mode-select readback failed.\r\n");
	} else {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 mode-select register = 0x%02X after stream start.\r\n",
				(unsigned int) mode_select);
	}

	camera_stream_started = true;
	return true;
}

/**
 * @brief Wait for FileX to finish mounting the SD card and creating the image directory.
 * @param timeout_ms Maximum time to wait before giving up.
 * @retval true when storage is ready for image writes.
 */
static bool CameraPlatform_WaitForStorageReady(uint32_t timeout_ms) {
	const ULONG deadline_tick = tx_time_get()
			+ CameraPlatform_MillisecondsToTicks(timeout_ms);

	while (!AppFileX_IsMediaReady()) {
		if (tx_time_get() >= deadline_tick) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Timed out waiting for FileX media readiness.\r\n");
			return false;
		}

		DelayMilliseconds_ThreadX(50U);
	}

	return true;
}

/**
 * @brief Capture a single cropped RAW10 frame and save it to the SD card.
 * @retval true when the frame reaches the SD card successfully.
 */
static bool CameraPlatform_CaptureAndStoreSingleFrame(void) {
	uint32_t captured_bytes = 0U;
	UINT filex_status = FX_SUCCESS;

	if (!CameraPlatform_WaitForStorageReady(CAMERA_STORAGE_WAIT_TIMEOUT_MS)) {
		return false;
	}

	if (!CameraPlatform_CaptureSingleFrame(&captured_bytes)) {
		return false;
	}

	filex_status = AppFileX_WriteCapturedImage(CAMERA_CAPTURE_FILE_NAME,
			camera_capture_buffer, (ULONG) captured_bytes);
	if (filex_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to write image to SD card, status=%lu.\r\n",
				(unsigned long) filex_status);
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Stored %lu-byte 16-bit raw image at /captured_images/%s.\r\n",
			(unsigned long) captured_bytes, CAMERA_CAPTURE_FILE_NAME);
	return true;
}

/**
 * @brief Log the active DCMIPP receiver clocking once per boot.
 */
static void CameraPlatform_LogDcmippClocking(void) {
	static bool clocking_logged = false;
	uint32_t dcmipp_freq_hz = 0U;
	uint32_t csi_freq_hz = 0U;
	uint32_t pclk5_freq_hz = 0U;
	uint32_t dcmipp_source = 0U;
	uint32_t sysclk_freq_hz = 0U;
	uint32_t hclk_freq_hz = 0U;
	uint32_t pll1_freq_hz = 0U;
	uint32_t pll4_freq_hz = 0U;
	uint32_t ic17_source = 0U;
	uint32_t ic17_divider = 0U;
	uint32_t ic17_enabled = 0U;
	uint32_t ic18_source = 0U;
	uint32_t ic18_divider = 0U;
	uint32_t ic18_enabled = 0U;

	if (clocking_logged) {
		return;
	}

	dcmipp_freq_hz = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_DCMIPP);
	csi_freq_hz = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_CSI);
	pclk5_freq_hz = HAL_RCC_GetPCLK5Freq();
	sysclk_freq_hz = HAL_RCC_GetSysClockFreq();
	hclk_freq_hz = HAL_RCC_GetHCLKFreq();
	pll1_freq_hz = HAL_RCCEx_GetPLL1CLKFreq();
	pll4_freq_hz = HAL_RCCEx_GetPLL4CLKFreq();
	dcmipp_source = __HAL_RCC_GET_DCMIPP_SOURCE();
	ic17_source = LL_RCC_IC17_GetSource();
	ic17_divider = LL_RCC_IC17_GetDivider();
	ic17_enabled = LL_RCC_IC17_IsEnabled();
	ic18_source = LL_RCC_IC18_GetSource();
	ic18_divider = LL_RCC_IC18_GetDivider();
	ic18_enabled = LL_RCC_IC18_IsEnabled();

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP clock source=%lu freq=%lu Hz, CSI freq=%lu Hz (PCLK5=%lu Hz, HCLK=%lu Hz, SYSCLK=%lu Hz).\r\n",
			(unsigned long) dcmipp_source, (unsigned long) dcmipp_freq_hz,
			(unsigned long) csi_freq_hz, (unsigned long) pclk5_freq_hz,
			(unsigned long) hclk_freq_hz, (unsigned long) sysclk_freq_hz);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] IC17 enabled=%lu source=%lu divider=%lu, IC18 enabled=%lu source=%lu divider=%lu, PLL1=%lu Hz, PLL4=%lu Hz.\r\n",
			(unsigned long) ic17_enabled, (unsigned long) ic17_source,
			(unsigned long) ic17_divider, (unsigned long) ic18_enabled,
			(unsigned long) ic18_source, (unsigned long) ic18_divider,
			(unsigned long) pll1_freq_hz, (unsigned long) pll4_freq_hz);
	if (dcmipp_freq_hz < 200000000U) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Warning: DCMIPP kernel clock is unexpectedly low for IMX335 capture.\r\n");
	}
	clocking_logged = true;
}

/**
 * @brief Configure PIPE0 for a centered RAW10 crop and capture one snapshot.
 * @param[out] captured_bytes_ptr Receives the DCMIPP byte counter on success.
 * @retval true when a frame-complete interrupt arrives without a DCMIPP error.
 */
static bool CameraPlatform_CaptureSingleFrame(uint32_t *captured_bytes_ptr) {
	const ULONG wait_ticks = CameraPlatform_MillisecondsToTicks(
	CAMERA_CAPTURE_TIMEOUT_MS);
	UINT semaphore_status = TX_SUCCESS;

	if (captured_bytes_ptr == NULL) {
		return false;
	}

	CameraPlatform_LogDcmippClocking();

	if (!CameraPlatform_PrepareDcmippSnapshot()) {
		return false;
	}

	camera_capture_failed = false;
	camera_capture_error_code = 0U;
	camera_capture_byte_count = 0U;
	camera_capture_sof_seen = false;
	camera_capture_eof_seen = false;
	camera_capture_frame_done = false;
	camera_capture_line_error_count = 0U;
	camera_capture_line_error_mask = 0U;

	/* Drain any stale semaphore token before arming the next snapshot. */
	while (tx_semaphore_get(&camera_capture_done_semaphore, TX_NO_WAIT)
			== TX_SUCCESS) {
	}

	if (!CameraPlatform_StartDcmippSnapshot()) {
		if (!CameraPlatform_StartImx335Stream()) {
			return false;
		}

		DelayMilliseconds_ThreadX(CAMERA_CAPTURE_RETRY_DELAY_MS);
		if (!CameraPlatform_StartDcmippSnapshot()) {
			return false;
		}
	} else if (!CameraPlatform_StartImx335Stream()) {
		(void) HAL_DCMIPP_CSI_PIPE_Stop(&hdcmipp, DCMIPP_PIPE0,
				DCMIPP_VIRTUAL_CHANNEL0);
		return false;
	}

	semaphore_status = tx_semaphore_get(&camera_capture_done_semaphore,
			wait_ticks);
	if (semaphore_status != TX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Timed out waiting for frame event, status=%lu cmsr1=0x%08lX cmsr2=0x%08lX sr0=0x%08lX p0fscr=0x%08lX p0fctcr=0x%08lX sof=%u eof=%u line_err_count=%lu line_err_mask=0x%08lX.\r\n",
				(unsigned long) semaphore_status,
				(unsigned long) hdcmipp.Instance->CMSR1,
				(unsigned long) hdcmipp.Instance->CMSR2,
				(unsigned long) CSI->SR0,
				(unsigned long) hdcmipp.Instance->P0FSCR,
				(unsigned long) hdcmipp.Instance->P0FCTCR,
				camera_capture_sof_seen ? 1U : 0U,
				camera_capture_eof_seen ? 1U : 0U,
				(unsigned long) camera_capture_line_error_count,
				(unsigned long) camera_capture_line_error_mask);
		(void) HAL_DCMIPP_CSI_PIPE_Stop(&hdcmipp, DCMIPP_PIPE0,
				DCMIPP_VIRTUAL_CHANNEL0);
		return false;
	}

	if (camera_capture_failed) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] DCMIPP reported capture error code 0x%08lX.\r\n",
				(unsigned long) camera_capture_error_code);
		return false;
	}

	if ((camera_capture_byte_count == 0U) && camera_capture_eof_seen) {
		if (HAL_DCMIPP_PIPE_GetDataCounter(&hdcmipp, DCMIPP_PIPE0,
				(uint32_t*) &camera_capture_byte_count) != HAL_OK) {
			camera_capture_byte_count = 0U;
		}
	}

	if ((camera_capture_byte_count == 0U)
			|| (camera_capture_byte_count > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Invalid capture byte count %lu (sof=%u eof=%u frame=%u).\r\n",
				(unsigned long) camera_capture_byte_count,
				camera_capture_sof_seen ? 1U : 0U,
				camera_capture_eof_seen ? 1U : 0U,
				camera_capture_frame_done ? 1U : 0U);
		return false;
	}

	*captured_bytes_ptr = camera_capture_byte_count;
	return true;
}

/**
 * @brief Attempt to arm a single PIPE0 RAW10 snapshot on VC0.
 * @retval true when HAL accepted the snapshot start request.
 */
static bool CameraPlatform_StartDcmippSnapshot(void) {
	if (HAL_DCMIPP_CSI_PIPE_Start(&hdcmipp, DCMIPP_PIPE0,
			DCMIPP_VIRTUAL_CHANNEL0, (uint32_t) camera_capture_buffer,
			DCMIPP_MODE_SNAPSHOT) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] HAL_DCMIPP_CSI_PIPE_Start() failed. mode=%lu pipe_state=%lu buffer=0x%08lX sr0=0x%08lX\r\n",
				(unsigned long) HAL_DCMIPP_GetMode(&hdcmipp),
				(unsigned long) HAL_DCMIPP_PIPE_GetState(&hdcmipp, DCMIPP_PIPE0),
				(unsigned long) ((uint32_t) camera_capture_buffer),
				(unsigned long) CSI->SR0);
		return false;
	}

	return true;
}

/**
 * @brief Configure PIPE0 for a centered 640x480 RAW10 crop.
 * @retval true when the crop and RAW10 settings are accepted by DCMIPP.
 */
static bool CameraPlatform_PrepareDcmippSnapshot(void) {
	DCMIPP_CSI_PIPE_ConfTypeDef csi_pipe_config = { 0 };
	DCMIPP_PipeConfTypeDef pipe_config = { 0 };
	DCMIPP_CropConfTypeDef crop_config = { 0 };

	csi_pipe_config.DataTypeMode = DCMIPP_DTMODE_DTIDA;
	csi_pipe_config.DataTypeIDA = DCMIPP_DT_RAW10;
	csi_pipe_config.DataTypeIDB = 0U;
	if (HAL_DCMIPP_CSI_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE0,
			&csi_pipe_config) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to configure PIPE0 for RAW10 input.\r\n");
		return false;
	}

	pipe_config.FrameRate = DCMIPP_FRAME_RATE_ALL;
	pipe_config.PixelPipePitch = 0U;
	pipe_config.PixelPackerFormat = 0U;
	if (HAL_DCMIPP_PIPE_SetConfig(&hdcmipp, DCMIPP_PIPE0, &pipe_config)
			!= HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to configure PIPE0 snapshot settings.\r\n");
		return false;
	}

	if (HAL_DCMIPP_CSI_SetVCConfig(&hdcmipp, DCMIPP_VIRTUAL_CHANNEL0,
			DCMIPP_CSI_DT_BPP10) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to configure CSI VC0 as RAW10.\r\n");
		return false;
	}

	crop_config.VStart = CAMERA_CAPTURE_CROP_VSTART;
	crop_config.HStart = CAMERA_CAPTURE_CROP_HSTART;
	crop_config.VSize = CAMERA_CAPTURE_HEIGHT_PIXELS;
	crop_config.HSize = CAMERA_CAPTURE_WIDTH_PIXELS;
	crop_config.PipeArea = DCMIPP_POSITIVE_AREA;
	if (HAL_DCMIPP_PIPE_SetCropConfig(&hdcmipp, DCMIPP_PIPE0, &crop_config)
			!= HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to configure PIPE0 crop window.\r\n");
		return false;
	}

	if (HAL_DCMIPP_PIPE_EnableCrop(&hdcmipp, DCMIPP_PIPE0) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to enable PIPE0 crop window.\r\n");
		return false;
	}

	return true;
}

/**
 * @brief Apply the vendor IMX335 frequency and default sensor init tables.
 * @retval true when the sensor acknowledges the init sequence.
 */
static bool CameraPlatform_InitializeImx335Sensor(void) {
	uint32_t sensor_id = 0U;

	if (!CameraPlatform_InitImx335Driver()) {
		return false;
	}

	if (IMX335_ReadID(&camera_sensor, &sensor_id) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - ST IMX335 driver could not read sensor ID.\r\n");
		return false;
	}

	if ((uint8_t) sensor_id != IMX335_CHIP_ID_VALUE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - ST IMX335 driver read unexpected ID 0x%02X.\r\n",
				(unsigned int) sensor_id);
		return false;
	}

	if (IMX335_Init(&camera_sensor, IMX335_R2592_1944, IMX335_RAW_RGGB10)
			!= IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to apply IMX335 default init tables.\r\n");
		return false;
	}

	/* Match ST's camera middleware ordering: the clock-specific timing tables
	 * are applied after the base resolution/mode tables so they are not
	 * overwritten by IMX335_Init(). */
	if (IMX335_SetFrequency(&camera_sensor, IMX335_INCK_24MHZ) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to configure IMX335 for 24 MHz input clock.\r\n");
		return false;
	}

	if (IMX335_SetFramerate(&camera_sensor, IMX335_CAPTURE_FRAMERATE_FPS)
			!= IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to set IMX335 frame rate to %d fps.\r\n",
				IMX335_CAPTURE_FRAMERATE_FPS);
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - IMX335 frame rate set to %d fps.\r\n",
			IMX335_CAPTURE_FRAMERATE_FPS);

#if (IMX335_TEST_PATTERN_MODE >= 0)
	if (IMX335_SetTestPattern(&camera_sensor, IMX335_TEST_PATTERN_MODE)
			!= IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to enable IMX335 test pattern mode %d.\r\n",
				IMX335_TEST_PATTERN_MODE);
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - IMX335 test pattern mode %d enabled for capture diagnostics.\r\n",
			IMX335_TEST_PATTERN_MODE);
#endif

	return true;
}

/**
 * @brief Provide the current HAL tick count to the vendor IMX335 driver.
 * @retval Current system tick in milliseconds.
 */
static int32_t CameraPlatform_GetTickMs(void) {
	return (int32_t) HAL_GetTick();
}

/**
 * @brief Convert milliseconds to ThreadX ticks, rounding up to ensure waits do not underflow.
 * @param timeout_ms Timeout in milliseconds.
 * @retval Equivalent timeout in scheduler ticks.
 */
static ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms) {
	uint32_t ticks = 0U;

	ticks = (timeout_ms * (uint32_t) TX_TIMER_TICKS_PER_SECOND + 999U) / 1000U;
	if ((timeout_ms > 0U) && (ticks == 0U)) {
		ticks = 1U;
	}

	return (ULONG) ticks;
}

/**
 * @brief Validate that the shared CubeMX I2C2 instance is ready for sensor use.
 * @retval IMX335_OK when the camera control bus handle is initialized.
 */
static int32_t CameraPlatform_I2cInit(void) {
	return (hi2c2.Instance == I2C2) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief No-op deinit for the shared CubeMX I2C2 peripheral.
 * @retval Always returns IMX335_OK because the app owns the bus globally.
 */
static int32_t CameraPlatform_I2cDeInit(void) {
	return IMX335_OK;
}

/**
 * @brief Read a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
static int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Read(&hi2c2, dev_addr, reg,
			I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief Write a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
static int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Write(&hi2c2, dev_addr, reg,
			I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief DCMIPP frame-complete callback used to release the capture thread.
 * @param hdcmipp HAL DCMIPP handle.
 * @param Pipe DCMIPP pipe that completed a frame.
 */
void HAL_DCMIPP_PIPE_FrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Pipe) {
	uint32_t byte_count = 0U;

	if ((hdcmipp == NULL) || (Pipe != DCMIPP_PIPE0)) {
		return;
	}

	if (HAL_DCMIPP_PIPE_GetDataCounter(hdcmipp, Pipe, &byte_count) != HAL_OK) {
		byte_count = 0U;
	}

	camera_capture_byte_count = byte_count;
	camera_capture_frame_done = true;
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief DCMIPP pipe-level error callback for the snapshot path.
 * @param hdcmipp HAL DCMIPP handle.
 * @param Pipe Pipe that reported the error.
 */
void HAL_DCMIPP_PIPE_ErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Pipe) {
	if ((hdcmipp == NULL) || (Pipe != DCMIPP_PIPE0)) {
		return;
	}

	camera_capture_failed = true;
	camera_capture_error_code = hdcmipp->ErrorCode;
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief DCMIPP global error callback for CSI/common failures.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_ErrorCallback(DCMIPP_HandleTypeDef *hdcmipp) {
	if (hdcmipp == NULL) {
		return;
	}

	camera_capture_failed = true;
	camera_capture_error_code = hdcmipp->ErrorCode;
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief CSI callback for clock-domain FIFO overflow diagnostics.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_CSI_ClockChangerFifoFullEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp) {
	UNUSED(hdcmipp);
	camera_capture_failed = true;
	camera_capture_error_code = 0xCCF1F0U;
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI clock changer FIFO full event detected.\r\n");
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief CSI start-of-frame callback used to confirm VC0 traffic is arriving.
 * @param hdcmipp HAL DCMIPP handle.
 * @param VirtualChannel CSI virtual channel that asserted SOF.
 */
void HAL_DCMIPP_CSI_StartOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel) {
	UNUSED(hdcmipp);

	if (VirtualChannel != DCMIPP_VIRTUAL_CHANNEL0) {
		return;
	}

	if (!camera_capture_sof_seen) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] CSI start-of-frame detected on VC0.\r\n");
	}

	camera_capture_sof_seen = true;
}

/**
 * @brief CSI end-of-frame callback used as a fallback wakeup for RAW dump capture.
 * @param hdcmipp HAL DCMIPP handle.
 * @param VirtualChannel CSI virtual channel that asserted EOF.
 */
void HAL_DCMIPP_CSI_EndOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel) {
	UNUSED(hdcmipp);

	if (VirtualChannel != DCMIPP_VIRTUAL_CHANNEL0) {
		return;
	}

	camera_capture_eof_seen = true;
	DebugConsole_Printf("[CAMERA][CAPTURE] CSI end-of-frame detected on VC0.\r\n");

	if (!camera_capture_frame_done) {
		(void) tx_semaphore_put(&camera_capture_done_semaphore);
	}
}

/**
 * @brief CSI callback for data-lane line errors.
 * @param hdcmipp HAL DCMIPP handle.
 * @param DataLane Failing CSI data lane.
 */
void HAL_DCMIPP_CSI_LineErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t DataLane) {
	UNUSED(hdcmipp);
	camera_capture_line_error_count++;
	camera_capture_line_error_mask |= (1UL << (DataLane & 0x1FU));
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI line error detected on data lane %lu (count=%lu).\r\n",
			(unsigned long) DataLane,
			(unsigned long) camera_capture_line_error_count);
	if ((camera_capture_line_error_count >= 8U) && !camera_capture_sof_seen) {
		camera_capture_failed = true;
		camera_capture_error_code = 0x1E000000U | DataLane;
		(void) tx_semaphore_put(&camera_capture_done_semaphore);
	}
}

/**
 * @brief CSI callback for short-packet detection visibility.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_CSI_ShortPacketDetectionEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp) {
	UNUSED(hdcmipp);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI short packet detected while waiting for frame start.\r\n");
}

/* USER CODE END 1 */
