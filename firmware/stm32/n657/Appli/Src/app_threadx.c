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
#include <stdio.h>
#include <string.h>
#include "app_camera_diagnostics.h"
#include "app_camera_config.h"
#include "app_camera_buffers.h"
#include "app_inference_log_config.h"
#include "app_inference_log_utils.h"
#include "app_threadx_config.h"
#include "app_memory_budget.h"
#include "app_filex.h"
#include "app_ai.h"
#include "main.h"
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "cmw_camera.h"
#include "cmw_imx335.h"
#include "cmw_utils.h"
#include "imx335.h"
#include "imx335_reg.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#if 0
#define CAMERA_INIT_STARTUP_DELAY_MS        200U
#define BCAMS_IMX_I2C_ADDRESS_7BIT          0x1AU
#define BCAMS_IMX_I2C_ADDRESS_HAL           (BCAMS_IMX_I2C_ADDRESS_7BIT << 1U)
#define BCAMS_IMX_I2C_PROBE_TRIALS          5U
#define BCAMS_IMX_I2C_PROBE_TIMEOUT_MS      50U
#define BCAMS_IMX_POWER_SETTLE_DELAY_MS     10U
#define BCAMS_IMX_RESET_ASSERT_DELAY_MS     5U
#define BCAMS_IMX_RESET_RELEASE_DELAY_MS    10U
#define IMX335_SENSOR_WIDTH_PIXELS          2592U
#define IMX335_SENSOR_HEIGHT_LINES          1944U
/* Use the processed CMW/ISP path so AE/AWB and demosaicing can converge on a
 * usable live image. Set to 1 only if we need raw Pipe0 diagnostics. */
#define CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC 0
#define CAMERA_CAPTURE_TARGET_FRAME_COUNT   4U
/* Capture crop is expressed directly in pixels/lines. */
#define CAMERA_CAPTURE_CROP_HSTART_PIXELS   0U
#define CAMERA_CAPTURE_CROP_VSTART_LINES    0U
/* Arm one CSI line/byte counter on VC0 so we can tell whether the receiver
 * is observing line progress even when the captured payload stays all zeros. */
#define CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER      DCMIPP_CSI_COUNTER0
#define CAMERA_CAPTURE_CSI_LB_PROBE_LINE_COUNTER  1U
#define CAMERA_CAPTURE_CSI_LB_PROBE_BYTE_COUNTER  (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)
/* Use a centered ROI for the raw diagnostic path so we do not accidentally
 * sample a blank top-left margin from the sensor frame. */
#define CAMERA_CAPTURE_RAW_CROP_HSTART_PIXELS   ((IMX335_SENSOR_WIDTH_PIXELS - CAMERA_CAPTURE_WIDTH_PIXELS) / 2U)
#define CAMERA_CAPTURE_RAW_CROP_VSTART_LINES    ((IMX335_SENSOR_HEIGHT_LINES - CAMERA_CAPTURE_HEIGHT_PIXELS) / 2U)
/* Pipe0 raw-capture frames store one 16-bit padded pixel per sample, so the
 * preview code should read them as a 224x224 source image and upscale only the view. */
#define CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS    CAMERA_CAPTURE_WIDTH_PIXELS
#define CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES    CAMERA_CAPTURE_HEIGHT_PIXELS
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE      2U
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS   (CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES   (CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_PIXEL_COUNT    (CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES)
#define CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES  14U
#define CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES   40U
#define CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES      (256U * 4U)
#define CAMERA_CAPTURE_RAW_BMP_HEADER_BYTES       (CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES)
/* IMX335 color-bar bring-up consistently shows four blank top lines before the
 * active test-pattern data starts, so we skip them in the raw Pipe0 view. */
#define CAMERA_CAPTURE_RAW_TOP_SKIP_LINES       4U
/* Give the ISP/AEC loop time to move the sensor away from its black-frame
 * startup state before we give up on the first saved capture. */
#define CAMERA_CAPTURE_TIMEOUT_MS           8000U
#define CAMERA_STORAGE_WAIT_TIMEOUT_MS      70000U
#define CAMERA_CAPTURE_RETRY_DELAY_MS       50U
#define CAMERA_FIRST_FRAME_WARMUP_DELAY_MS  1500U
#define CAMERA_STREAM_WARMUP_DELAY_MS       250U
#define IMX335_CAPTURE_FRAMERATE_FPS        10
#define CAMERA_CAPTURE_FILE_NAME_LENGTH     64U
#define CAMERA_STORAGE_READY_EVENT_FLAG     0x00000001U
/* Match ST's IMX335 middleware and upstream Linux driver ID check. */
#define IMX335_CHIP_ID_REG                 0x3912U
#define IMX335_CHIP_ID_VALUE               0x00U
/* IMX335 test-pattern selection.
 * -1 = disabled (live image), 0 = disabled (same as -1 in driver),
 *  1 = solid color (default color regs = 0x000 = black Ã¢â‚¬â€ all-zero pixels, NOT useful),
 * 10 = color bars (non-zero pixel values Ã¢â‚¬â€ use this to verify DMA path). */
/* Return to live optical input so the raw capture reflects the real gauge
 * scene instead of a synthetic test pattern. */
#define IMX335_TEST_PATTERN_MODE           -1

/* ST treats PIPE0 as the raw dump pipe and PIPE1 as the processed/YUV pipe.
 * Use PIPE0 only while the raw diagnostic branch is enabled. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE0
#else
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE1
#endif

/* Prevent accidentally using mode 1 (solid black = all-zero pixels) during
 * raw diagnostic; it is indistinguishable from a broken DMA path. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC && (IMX335_TEST_PATTERN_MODE == 1)
#error "IMX335_TEST_PATTERN_MODE=1 produces all-zero pixels in raw diag mode. Use mode 10 (color bars)."
#endif
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* Inference logger thread state machine. */
typedef enum {
	INFER_LOG_STATE_INIT_DIR = 0,   /* Create /inference directory if needed. */
	INFER_LOG_STATE_CHECK_DATE,     /* Open/roll over to today's log file. */
	INFER_LOG_STATE_NO_RTC,         /* DS3231 absent Ã¢â‚¬â€ blink red, retry. */
	INFER_LOG_STATE_LOGGING,        /* Steady state: wait for value, append row. */
} InferLogState_t;

static TX_THREAD inference_log_thread;
static ULONG inference_log_thread_stack[INFERENCE_LOG_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool inference_log_thread_created = false;
static TX_QUEUE inference_log_queue;
/* TX_QUEUE stores ULONG words; we pass the float bits as a ULONG. */
static ULONG inference_log_queue_storage[INFERENCE_LOG_QUEUE_DEPTH];

/* Dedicated ThreadX object and stack for camera connection diagnostics. */
static TX_THREAD camera_init_thread;
static ULONG camera_init_thread_stack[CAMERA_INIT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_init_thread_created = false;
static TX_THREAD camera_isp_thread;
static ULONG camera_isp_thread_stack[CAMERA_ISP_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_isp_thread_created = false;
static TX_THREAD camera_heartbeat_thread;
static ULONG camera_heartbeat_thread_stack[CAMERA_HEARTBEAT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_heartbeat_thread_created = false;
static TX_THREAD camera_ai_thread;
static ULONG camera_ai_thread_stack[CAMERA_AI_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_ai_thread_created = false;
static CMW_IMX335_t camera_sensor;
static CMW_Sensor_if_t camera_sensor_driver;
static CMW_IMX335_config_t camera_sensor_config;
static IMX335_Object_t camera_raw_sensor;
static int32_t camera_sensor_gain_cache = 0;
static int32_t camera_sensor_exposure_cache = 0;
static bool camera_cmw_initialized = false;
/* Keep the middleware path active so the ISP/AEC pipeline can produce optical
 * frames instead of the raw sensor dump. */
static bool camera_capture_use_cmw_pipeline = false;
static TX_SEMAPHORE camera_capture_done_semaphore;
static TX_SEMAPHORE camera_capture_isp_semaphore;
static TX_SEMAPHORE camera_ai_request_semaphore;
static TX_EVENT_FLAGS_GROUP camera_storage_ready_flags;
static TX_MUTEX debug_uart_mutex;
static bool camera_heartbeat_gpio_initialized = false;
static bool camera_capture_sync_created = false;
static bool camera_storage_ready_sync_created = false;
static bool camera_ai_sync_created = false;
static bool camera_stream_started = false;
static volatile const uint8_t *camera_ai_request_frame_ptr = NULL;
static volatile ULONG camera_ai_request_frame_length = 0U;
static volatile bool camera_capture_failed = false;
static volatile uint32_t camera_capture_error_code = 0U;
static volatile uint32_t camera_capture_byte_count = 0U;
static volatile bool camera_capture_sof_seen = false;
static volatile bool camera_capture_eof_seen = false;
static volatile bool camera_capture_frame_done = false;
static volatile bool camera_capture_snapshot_armed = false;
static volatile uint32_t camera_capture_frame_event_count = 0U;
static volatile uint32_t camera_capture_line_error_count = 0U;
static volatile uint32_t camera_capture_line_error_mask = 0U;
static volatile uint32_t camera_capture_csi_linebyte_event_count = 0U;
static volatile bool camera_capture_csi_linebyte_event_logged = false;
static volatile uint32_t camera_capture_vsync_event_count = 0U;
static volatile uint32_t camera_capture_isp_run_count = 0U;
/* Count raw IRQ entry points so we can tell whether the interrupt chain is
 * alive even when the higher-level callbacks stay silent. */
volatile uint32_t camera_capture_csi_irq_count = 0U;
volatile uint32_t camera_capture_dcmipp_irq_count = 0U;
static volatile uint32_t camera_capture_reported_byte_count = 0U;
static volatile uint32_t camera_capture_counter_status = (uint32_t) HAL_ERROR;

/* Reuse the CubeMX-generated camera control I2C instance from main.c. */
extern DCMIPP_HandleTypeDef hdcmipp;
extern I2C_HandleTypeDef hi2c2;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static VOID CameraHeartbeatThread_Entry(ULONG thread_input);
static VOID CameraAIThread_Entry(ULONG thread_input);
static VOID InferenceLogThread_Entry(ULONG thread_input);

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input);
static VOID CameraIspThread_Entry(ULONG thread_input);

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
static void CameraPlatform_CmwEnablePin(int value);
static void CameraPlatform_CmwShutdownPin(int value);
static void CameraPlatform_CmwDelay(uint32_t delay_ms);

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
static bool CameraPlatform_SeedImx335ExposureGain(void);
static bool CameraPlatform_EnableImx335AutoExposure(void);
static void CameraPlatform_ReapplyImx335TestPattern(void);
static bool CameraPlatform_StartImx335Stream(void);
static bool CameraPlatform_WaitForStorageReady(uint32_t timeout_ms);
static bool CameraPlatform_BuildCaptureFileName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr);
static bool CameraPlatform_CaptureAndStoreSingleFrame(void);
static bool CameraPlatform_RequestDryInference(const uint8_t *frame_ptr,
		ULONG frame_length);
static bool CameraPlatform_CaptureSingleFrame(uint32_t *captured_bytes_ptr);
static void CameraPlatform_LogCaptureBufferSummary(uint32_t captured_bytes);
static void CameraPlatform_LogCaptureState(const char *reason);
static bool CameraPlatform_ConfigureCsiLineByteProbe(void);
static bool CameraPlatform_PrepareDcmippSnapshot(void);
static bool CameraPlatform_StartDcmippSnapshot(void);
static bool CameraPlatform_RunImx335Background(void);

static ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms);
static DCMIPP_HandleTypeDef* CameraPlatform_GetCaptureDcmippHandle(void);
int CMW_CAMERA_PIPE_VsyncEventCallback(uint32_t pipe);
int CMW_CAMERA_PIPE_FrameEventCallback(uint32_t pipe);
void CMW_CAMERA_PIPE_ErrorCallback(uint32_t pipe);
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
 * @brief Adapter from the HAL tick API to the ST IMX335 driver callback type.
 * @retval Current HAL tick in milliseconds.
 */
static int32_t CameraPlatform_GetTickMs(void);

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
static ISP_StatusTypeDef CameraPlatform_IspSetSensorGain(
		uint32_t camera_instance, int32_t gain);
static ISP_StatusTypeDef CameraPlatform_IspGetSensorGain(
		uint32_t camera_instance, int32_t *gain);
static ISP_StatusTypeDef CameraPlatform_IspSetSensorExposure(
		uint32_t camera_instance, int32_t exposure);
static ISP_StatusTypeDef CameraPlatform_IspGetSensorExposure(
		uint32_t camera_instance, int32_t *exposure);
static ISP_StatusTypeDef CameraPlatform_IspGetSensorInfo(
		uint32_t camera_instance, ISP_SensorInfoTypeDef *info);
static ISP_StatusTypeDef CameraPlatform_IspSetSensorTestPattern(
		uint32_t camera_instance, int32_t mode);

/**
 * @brief Weak probe hook for board-specific camera/driver integration.
 * @return TX_SUCCESS on successful camera detection, an error code otherwise.
 */
__attribute__((weak))        UINT CameraPlatform_ProbeBCamsImx(void);

/* USER CODE END PFP */

static void DebugUartMutex_Lock(void) {
	/* TX_NO_WAIT: never block Ã¢â‚¬â€ a missed lock means garbled output, not deadlock. */
	ThreadxUtils_LockMutex(&debug_uart_mutex);
}
static void DebugUartMutex_Unlock(void) {
	ThreadxUtils_UnlockMutex(&debug_uart_mutex);
}

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

/* USER CODE BEGIN App_ThreadX_Start */
UINT App_ThreadX_Start(void) {
	/* Keep this function idempotent to protect against accidental double-start. */
	/* Leave the heartbeat LED under the dedicated thread so it reflects liveness
	 * instead of startup state. */
	BSP_LED_On(LED_RED);
	BSP_LED_Off(LED_BLUE);
	BSP_LED_Off(LED_GREEN);
	if (camera_init_thread_created && camera_isp_thread_created
			&& camera_heartbeat_thread_created && camera_ai_thread_created
			&& inference_log_thread_created) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Start skipped: camera threads already created.\r\n");
		return TX_SUCCESS;
	}

	if (!camera_capture_sync_created) {
		UINT semaphore_status = tx_semaphore_create(
				&camera_capture_done_semaphore, "camera_capture_done", 0U);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create capture semaphore, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}

		semaphore_status = tx_semaphore_create(&camera_capture_isp_semaphore,
				"camera_capture_isp", 0U);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create ISP semaphore, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}

		/* No UART mutex Ã¢â‚¬â€ concurrent prints may interleave but won't deadlock.
		 * The UART HAL transmit is not re-entrant; a mutex with TX_WAIT_FOREVER
		 * deadlocks when the holder's HAL_UART_Transmit itself blocks, and
		 * TX_NO_WAIT corrupts the UART state when two threads collide mid-frame. */
		(void) tx_mutex_create(&debug_uart_mutex, "debug_uart", TX_NO_INHERIT);

		camera_capture_sync_created = true;
	}

	if (!camera_storage_ready_sync_created) {
		const UINT ready_flags_status = tx_event_flags_create(
				&camera_storage_ready_flags, "camera_storage_ready");
		if (ready_flags_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create storage-ready event flags, status=%lu\r\n",
					(unsigned long) ready_flags_status);
			return ready_flags_status;
		}

		camera_storage_ready_sync_created = true;
	}

	if (!camera_ai_sync_created) {
		const UINT ai_request_status = tx_semaphore_create(
				&camera_ai_request_semaphore, "camera_ai_request", 0U);
		if (ai_request_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create AI request semaphore, status=%lu\r\n",
					(unsigned long) ai_request_status);
			return ai_request_status;
		}

		camera_ai_sync_created = true;
	}

	if (!camera_isp_thread_created) {
		const UINT isp_create_status = tx_thread_create(&camera_isp_thread,
				"camera_isp", CameraIspThread_Entry, 0U,
				camera_isp_thread_stack, sizeof(camera_isp_thread_stack),
				CAMERA_ISP_THREAD_PRIORITY, CAMERA_ISP_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (isp_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera ISP thread, status=%lu\r\n",
					(unsigned long) isp_create_status);
			return isp_create_status;
		}

		camera_isp_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera ISP thread created and started.\r\n");
	}

	if (!camera_heartbeat_thread_created) {
		const UINT heartbeat_create_status = tx_thread_create(
				&camera_heartbeat_thread, "camera_heartbeat",
				CameraHeartbeatThread_Entry, 0U,
				camera_heartbeat_thread_stack,
				sizeof(camera_heartbeat_thread_stack),
				CAMERA_HEARTBEAT_THREAD_PRIORITY,
				CAMERA_HEARTBEAT_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (heartbeat_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create heartbeat thread, status=%lu\r\n",
					(unsigned long) heartbeat_create_status);
			return heartbeat_create_status;
		}

		camera_heartbeat_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Heartbeat thread created and started.\r\n");
	}

	if (!camera_ai_thread_created) {
		const UINT ai_create_status = tx_thread_create(&camera_ai_thread,
				"camera_ai", CameraAIThread_Entry, 0U,
				camera_ai_thread_stack, sizeof(camera_ai_thread_stack),
				CAMERA_AI_THREAD_PRIORITY, CAMERA_AI_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (ai_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera AI thread, status=%lu\r\n",
					(unsigned long) ai_create_status);
			return ai_create_status;
		}

		camera_ai_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera AI thread created and started.\r\n");
	}

	if (!camera_init_thread_created) {
		/* Create a dedicated thread so camera probing is isolated from other startup work. */
		const UINT create_status = tx_thread_create(&camera_init_thread,
				"camera_init", CameraInitThread_Entry, 0U,
				camera_init_thread_stack, sizeof(camera_init_thread_stack),
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
	}

	if (!inference_log_thread_created) {
		const UINT queue_status = tx_queue_create(&inference_log_queue,
				"inference_log_queue",
				TX_1_ULONG,
				inference_log_queue_storage,
				sizeof(inference_log_queue_storage));
		if (queue_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[INFER_LOG] Failed to create inference log queue, status=%lu\r\n",
					(unsigned long) queue_status);
			return queue_status;
		}

		const UINT log_create_status = tx_thread_create(&inference_log_thread,
				"inference_log", InferenceLogThread_Entry, 0U,
				inference_log_thread_stack, sizeof(inference_log_thread_stack),
				INFERENCE_LOG_THREAD_PRIORITY, INFERENCE_LOG_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);
		if (log_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[INFER_LOG] Failed to create inference log thread, status=%lu\r\n",
					(unsigned long) log_create_status);
			return log_create_status;
		}

		inference_log_thread_created = true;
		DebugConsole_Printf(
				"[INFER_LOG] Inference log thread created and started.\r\n");
	}

	return TX_SUCCESS;
}
/* USER CODE END App_ThreadX_Start */

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

	/* Use the blue LED as the camera-thread marker; the green LED is owned by
	 * the heartbeat thread so it can keep pulsing independently. */
	(void) DebugConsole_WriteString("[CAMERA] thread entry\r\n");

	/* Delay a little to let other startup logs complete before camera probing. */
	DelayMilliseconds_ThreadX(CAMERA_INIT_STARTUP_DELAY_MS);

	(void) DebugConsole_WriteString("[CAMERA] probe start\r\n");

	if (Camera_ProbeBCamsImx()) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe completed successfully.\r\n");

		if (!App_AI_Model_Init()) {
			DebugConsole_Printf(
					"[AI] Model runtime init failed; continuing without inference.\r\n");
		}

		/* Turn blue off right before capture so a stuck-on LED means the probe
		 * completed and the freeze moved into the capture path. */
		BSP_LED_Off(LED_BLUE);
		DebugConsole_Printf(
				"[CAMERA][THREAD] Entering capture/inference loop (period=60s)...\r\n");
		while (1) {
			if (CameraPlatform_CaptureAndStoreSingleFrame()) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture and inference completed successfully.\r\n");
			} else {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture/inference attempt failed.\r\n");
			}
			DelayMilliseconds_ThreadX(60000U);
		}
	} else {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe failed or is not configured yet.\r\n");
	}

}

/**
 * @brief Low-priority heartbeat thread that toggles the board's LED1
 * every 5 seconds.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraHeartbeatThread_Entry(ULONG thread_input) {
	(void) thread_input;

	if (!camera_heartbeat_gpio_initialized) {
		GPIO_InitTypeDef gpio_init = { 0 };

		/* Match the board's green LED pin: LED3 is PG0 on this board. */
		__HAL_RCC_GPIOG_CLK_ENABLE();
		HAL_GPIO_WritePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN, GPIO_PIN_SET);

		gpio_init.Pin = CAMERA_HEARTBEAT_LED_PIN;
		gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
		gpio_init.Pull = GPIO_NOPULL;
		gpio_init.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
		HAL_GPIO_Init(CAMERA_HEARTBEAT_LED_GPIO_PORT, &gpio_init);

		camera_heartbeat_gpio_initialized = true;
	}

	HAL_GPIO_WritePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
			CAMERA_HEARTBEAT_LED_PIN, GPIO_PIN_SET);
	DebugConsole_Printf("[WATCHDOG] heartbeat thread running.\r\n");

	while (1) {
		DebugConsole_Printf("[WATCHDOG] pulse\r\n");
		/* Toggle the green user LED directly. */
		HAL_GPIO_TogglePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN);
		DelayMilliseconds_ThreadX(CAMERA_HEARTBEAT_PULSE_MS);
		HAL_GPIO_TogglePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN);
		DelayMilliseconds_ThreadX(CAMERA_HEARTBEAT_PERIOD_MS
				- CAMERA_HEARTBEAT_PULSE_MS);
	}
}

/**
 * @brief Low-priority AI worker that runs one queued dry inference at a time.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraAIThread_Entry(ULONG thread_input) {
	(void) thread_input;

	(void) DebugConsole_WriteString("[AI] worker alive\r\n");

	while (1) {
		const UINT request_status = tx_semaphore_get(&camera_ai_request_semaphore,
				TX_WAIT_FOREVER);
		const uint8_t *frame_ptr = NULL;
		ULONG frame_length = 0U;

		if (request_status != TX_SUCCESS) {
			continue;
		}

		frame_ptr = (const uint8_t *) camera_ai_request_frame_ptr;
		frame_length = camera_ai_request_frame_length;
		camera_ai_request_frame_ptr = NULL;
		camera_ai_request_frame_length = 0U;

		DebugConsole_Printf(
				"[AI] Worker dequeued frame: ptr=%p length=%lu semaphore_status=%u\r\n",
				(void *) frame_ptr, (unsigned long) frame_length,
				(unsigned int) request_status);

		if ((frame_ptr == NULL) || (frame_length == 0U)) {
			DebugConsole_Printf(
					"[AI] Worker woke without a queued frame; ignoring.\r\n");
			continue;
		}

		if (!App_AI_RunDryInferenceFromYuv422(frame_ptr,
				(size_t) frame_length)) {
			DebugConsole_Printf(
					"[AI] One-shot dry-run inference failed; continuing.\r\n");
		} else {
			float result = 0.0f;
			if (App_AI_GetLastInferenceResult(&result)) {
				union { float f; ULONG u; } bits = { .f = result };
				char inference_line[48] = { 0 };

				AppInferenceLog_FormatFloatTenths(inference_line,
						sizeof(inference_line),
						"[AI] Inference value: ", result);
				(void) DebugConsole_WriteString(inference_line);
				if (inference_log_thread_created) {
					(void) tx_queue_send(&inference_log_queue, &bits.u, TX_NO_WAIT);
				}
			}
		}
	}
}

/**
 * @brief Inference value logger thread.
 *
 * State machine:
 *   INIT_DIR   - Ensure /inference directory exists on SD card.
 *   CHECK_DATE - Build today's filename; open/create the CSV log file.
 *   NO_RTC     - DS3231 unreachable; blink red LED and retry.
 *   LOGGING    - Wait for a new inference value on the queue, append a CSV row.
 *
 * Each row: "YYYY-MM-DD HH:MM:SS,<value>\n"
 * File rolls over daily: inference/YYYY-MM-DD.csv
 */
static VOID InferenceLogThread_Entry(ULONG thread_input) {
	(void) thread_input;

	InferLogState_t state = INFER_LOG_STATE_INIT_DIR;
	char today_date[12] = { 0 };   /* "YYYY-MM-DD\0" */
	char log_file_name[INFERENCE_LOG_FILE_NAME_LENGTH] = { 0 };
	FX_MEDIA *media = NULL;

	(void) DebugConsole_WriteString("[INFER_LOG] thread alive\r\n");

	while (1) {
		switch (state) {

		/* ------------------------------------------------------------------ */
		case INFER_LOG_STATE_INIT_DIR: {
			/* Wait until FileX media is ready. */
			if (!AppFileX_IsMediaReady()) {
				DelayMilliseconds_ThreadX(500U);
				break;
			}

			media = AppFileX_GetMediaHandle();

			/* Create /inference directory (ignore FX_ALREADY_CREATED). */
			UINT fx_status = AppFileX_AcquireMediaLock();
			if (fx_status != TX_SUCCESS) {
				DelayMilliseconds_ThreadX(500U);
				break;
			}

			fx_status = fx_directory_create(media,
					INFERENCE_LOG_DIRECTORY_NAME);
			AppFileX_ReleaseMediaLock();

			if ((fx_status == FX_SUCCESS)
					|| (fx_status == FX_ALREADY_CREATED)) {
				DebugConsole_Printf(
						"[INFER_LOG] /inference directory ready.\r\n");
				state = INFER_LOG_STATE_CHECK_DATE;
			} else {
				DebugConsole_Printf(
						"[INFER_LOG] Failed to create /inference dir, status=%lu. Retrying.\r\n",
						(unsigned long) fx_status);
				DelayMilliseconds_ThreadX(2000U);
			}
			break;
		}

		/* ------------------------------------------------------------------ */
		case INFER_LOG_STATE_CHECK_DATE: {
			/* Read today's date from DS3231 to determine the log filename. */
			char rtc_timestamp[32] = { 0 };
			const bool rtc_ok = App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp));

			if (!rtc_ok) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC not available; entering NO_RTC state.\r\n");
				state = INFER_LOG_STATE_NO_RTC;
				break;
			}

			/* rtc_timestamp is "YYYY-MM-DD_HH-MM-SS"; extract date portion. */
			char new_date[12] = { 0 };
			/* Copy first 10 chars: YYYY-MM-DD */
			(void) memcpy(new_date, rtc_timestamp, 10U);
			new_date[10] = '\0';
			/* Replace underscores back to dashes (App_Clock sanitizes spaces/colons
			 * only; the date part has no space so it comes through as-is). */

			if (strcmp(new_date, today_date) != 0) {
				/* New day (or first run): build the new filename. */
				(void) memcpy(today_date, new_date, sizeof(today_date));
				int written = snprintf(log_file_name, sizeof(log_file_name),
						"%s/%s.csv", INFERENCE_LOG_DIRECTORY_NAME, today_date);
				if ((written <= 0)
						|| ((size_t) written >= sizeof(log_file_name))) {
					DebugConsole_Printf(
							"[INFER_LOG] Log filename overflow; retrying.\r\n");
					DelayMilliseconds_ThreadX(5000U);
					break;
				}

				/* Ensure the file exists (create header row if brand new). */
				UINT lock_status = AppFileX_AcquireMediaLock();
				if (lock_status == TX_SUCCESS) {
					FX_FILE log_file = { 0 };
					UINT open_status = fx_file_open(media, &log_file,
							log_file_name, FX_OPEN_FOR_WRITE);
					if (open_status == FX_NOT_FOUND) {
						/* Create and write CSV header. */
						(void) fx_file_create(media, log_file_name);
						open_status = fx_file_open(media, &log_file,
								log_file_name, FX_OPEN_FOR_WRITE);
						if (open_status == FX_SUCCESS) {
							const char *header = "datetime,value_degC\n";
							(void) fx_file_write(&log_file, (VOID*) header,
									(ULONG) strlen(header));
						}
					}
					if (open_status == FX_SUCCESS) {
						(void) fx_file_close(&log_file);
					}
					(void) fx_media_flush(media);
					AppFileX_ReleaseMediaLock();
				}

				DebugConsole_Printf(
						"[INFER_LOG] Logging to %s.\r\n", log_file_name);
			}

			state = INFER_LOG_STATE_LOGGING;
			break;
		}

		/* ------------------------------------------------------------------ */
		case INFER_LOG_STATE_NO_RTC: {
			/* Blink red LED to signal RTC fault. */
			DebugConsole_Printf(
					"[INFER_LOG] ERROR: DS3231 RTC not detected. Cannot timestamp log entries.\r\n");
			DebugLed_BlinkRedBlocking(INFERENCE_LOG_NO_RTC_BLINK_ON_MS,
					INFERENCE_LOG_NO_RTC_BLINK_OFF_MS, 5U);

			/* Retry RTC after a short delay. */
			DelayMilliseconds_ThreadX(INFERENCE_LOG_NO_RTC_RETRY_DELAY_MS);

			char rtc_timestamp[32] = { 0 };
			if (App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp))) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC recovered; resuming logging.\r\n");
				state = INFER_LOG_STATE_CHECK_DATE;
			}
			break;
		}

		/* ------------------------------------------------------------------ */
		case INFER_LOG_STATE_LOGGING: {
			ULONG value_bits = 0U;
			/* Wait for next inference result (1-minute timeout to roll over daily). */
			const ULONG wait_ticks = CameraPlatform_MillisecondsToTicks(
					65000U);
			const UINT q_status = tx_queue_receive(&inference_log_queue,
					&value_bits, wait_ticks);

			if (q_status == TX_NO_INSTANCE) {
				/* Timeout Ã¢â‚¬â€ recheck date for daily rollover. */
				state = INFER_LOG_STATE_CHECK_DATE;
				break;
			}

			if (q_status != TX_SUCCESS) {
				break;
			}

			/* Decode float from ULONG bits. */
			union { float f; ULONG u; } bits = { .u = value_bits };
			const float inference_value = bits.f;
			char inference_line[48] = { 0 };

			AppInferenceLog_FormatFloatTenths(inference_line,
					sizeof(inference_line),
					"[INFER_LOG] Inference value: ", inference_value);
			(void) DebugConsole_WriteString(inference_line);

			/* Get current timestamp. */
			char rtc_timestamp[32] = { 0 };
			const bool rtc_ok = App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp));

			if (!rtc_ok) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC lost during logging; entering NO_RTC state.\r\n");
				state = INFER_LOG_STATE_NO_RTC;
				break;
			}

			/* Reconstruct human-readable datetime: rtc_timestamp is
			 * "YYYY-MM-DD_HH-MM-SS"; convert _ Ã¢â€ â€™ space, - in time Ã¢â€ â€™ : */
			char datetime_str[24] = { 0 };
			{
				const char *src = rtc_timestamp;
				char *dst = datetime_str;
				uint32_t i = 0U;
				while ((i < (sizeof(datetime_str) - 1U))
						&& (src[i] != '\0')) {
					char ch = src[i];
					if (ch == '_') {
						ch = ' ';
					} else if ((i > 10U) && (ch == '-')) {
						ch = ':';
					}
					*dst++ = ch;
					i++;
				}
				*dst = '\0';
			}

			/* Recheck date for daily rollover. */
			char new_date[12] = { 0 };
			(void) memcpy(new_date, rtc_timestamp, 10U);
			new_date[10] = '\0';
			if (strcmp(new_date, today_date) != 0) {
				state = INFER_LOG_STATE_CHECK_DATE;
				/* Re-queue the value so it gets written to the new file. */
				(void) tx_queue_send(&inference_log_queue, &value_bits,
						TX_NO_WAIT);
				break;
			}

			/* Format CSV row: "YYYY-MM-DD HH:MM:SS,<value>\n"
			 * Use integer + one decimal to avoid full float formatting. */
			char row[INFERENCE_LOG_ROW_MAX_LENGTH] = { 0 };
			int int_part = (int) inference_value;
			int frac_part = (int) ((inference_value - (float) int_part) * 10.0f);
			if (frac_part < 0) {
				frac_part = -frac_part;
			}
			int written = snprintf(row, sizeof(row), "%s,%d.%d\n",
					datetime_str, int_part, frac_part);
			if ((written <= 0)
					|| ((size_t) written >= sizeof(row))) {
				DebugConsole_Printf(
						"[INFER_LOG] Row format overflow; skipping.\r\n");
				break;
			}

			/* Append row to the log file. */
			UINT lock_status = AppFileX_AcquireMediaLock();
			if (lock_status != TX_SUCCESS) {
				DebugConsole_Printf(
						"[INFER_LOG] Could not acquire media lock; dropping row.\r\n");
				break;
			}

			UINT fx_status = FX_SUCCESS;
			FX_FILE log_file = { 0 };
			fx_status = fx_file_open(media, &log_file, log_file_name,
					FX_OPEN_FOR_WRITE);
			if (fx_status == FX_SUCCESS) {
				(void) fx_file_relative_seek(&log_file, 0U, FX_SEEK_END);
				(void) fx_file_write(&log_file, (VOID*) row, (ULONG) written);
				(void) fx_file_close(&log_file);
				(void) fx_media_flush(media);
				DebugConsole_Printf("[INFER_LOG] Logged: %.*s", written, row);
			} else {
				DebugConsole_Printf(
						"[INFER_LOG] Failed to open log file, status=%lu.\r\n",
						(unsigned long) fx_status);
			}
			AppFileX_ReleaseMediaLock();
			break;
		}

		default:
			state = INFER_LOG_STATE_INIT_DIR;
			break;
		}
	}
}

/**
 * @brief ThreadX entry point that services the camera middleware's ISP work.
 *
 * The reference x-cube-n6-camera-capture app keeps ISP/background processing
 * on a dedicated thread that wakes from VSYNC events. That separation makes
 * camera bring-up less timing-sensitive, so we mirror it here.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraIspThread_Entry(ULONG thread_input) {
	(void) thread_input;

	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera ISP service thread running.\r\n");

	while (1) {
		UINT semaphore_status = tx_semaphore_get(&camera_capture_isp_semaphore,
				CameraPlatform_MillisecondsToTicks(20U));

		/* Keep the ISP background process alive even if the first VSYNC is late.
		 * ST's BSP notes that the background process should be called regularly. */
		if ((semaphore_status == TX_SUCCESS)
				|| (camera_stream_started && camera_cmw_initialized)) {
			if (!CameraPlatform_RunImx335Background()) {
				camera_capture_failed = true;
				camera_capture_error_code = 0x49535052U; /* 'ISPR' */
				(void) tx_semaphore_put(&camera_capture_done_semaphore);
			}
		}
	}
}

/**
 * @brief Print a staged diagnostic sequence for B-CAMS-IMX camera bring-up.
 * @return true when a camera probe callback reports success, false otherwise.
 */
static bool Camera_ProbeBCamsImx(void) {
	bool stage_ok = true;

	DebugConsole_Printf("[CAMERA][PROBE] Probing camera stack...\r\n");
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: reset\r\n");
	const UINT probe_status = CameraPlatform_ProbeBCamsImx();
	if (probe_status == TX_SUCCESS) {
		DebugConsole_Printf("[CAMERA][PROBE] Sensor probe OK.\r\n");
	} else {
		DebugConsole_Printf(
				"[CAMERA][PROBE] Sensor probe failed, status=%lu\r\n",
				(unsigned long) probe_status);
		stage_ok = false;
	}

	if (stage_ok) {
		DebugConsole_Printf("[CAMERA][PROBE] Camera stack ready.\r\n");
	} else {
		DebugConsole_Printf("[CAMERA][PROBE] Camera stack not ready.\r\n");
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
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: i2c-ack\r\n");
	probe_status = HAL_I2C_IsDeviceReady(&hi2c2,
	BCAMS_IMX_I2C_ADDRESS_HAL,
	BCAMS_IMX_I2C_PROBE_TRIALS,
	BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);

	if (probe_status == HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Sensor ACKed on I2C2 at 7-bit address 0x%02X.\r\n",
				(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);

		/* Read the same chip-ID register used by ST's IMX335 middleware. */
		(void) DebugConsole_WriteString("[CAMERA][PROBE] step: chip-id\r\n");
		probe_status = CameraPlatform_ReadImx335ChipId(&chip_id);
		if (probe_status != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Failed to read IMX335 ID register 0x%04X.\r\n",
					(unsigned int) IMX335_CHIP_ID_REG);
			return TX_NOT_AVAILABLE;
		}

		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ID register 0x%04X = 0x%02X.\r\n",
				(unsigned int) IMX335_CHIP_ID_REG, (unsigned int) chip_id);
		if (chip_id != IMX335_CHIP_ID_VALUE) {
			DebugConsole_Printf(
					"[CAMERA][PROBE]   - Unexpected chip ID, expected 0x%02X.\r\n",
					(unsigned int) IMX335_CHIP_ID_VALUE);
			return TX_NOT_AVAILABLE;
		}

		(void) DebugConsole_WriteString("[CAMERA][PROBE] step: sensor-init\r\n");
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
	I2C_MEMADD_SIZE_16BIT, chip_id, 1U,
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
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: power-on\r\n");
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin, GPIO_PIN_SET);
	DelayMilliseconds_ThreadX(BCAMS_IMX_POWER_SETTLE_DELAY_MS);

	/* Pulse the sensor reset line exactly as the vendor middleware does. */
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
 * @param value Non-zero to enable the module, zero to disable it.
 */
static void CameraPlatform_CmwEnablePin(int value) {
	HAL_GPIO_WritePin(CAM1_GPIO_Port, CAM1_Pin,
			(value != 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/**
 * @brief Drive the camera reset pin in the form expected by ST's middleware.
 * @param value Non-zero to release reset, zero to assert reset.
 */
static void CameraPlatform_CmwShutdownPin(int value) {
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin,
			(value != 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/**
 * @brief Delay helper used by ST's camera middleware from the ThreadX camera thread.
 * @param delay_ms Delay duration in milliseconds.
 */
static void CameraPlatform_CmwDelay(uint32_t delay_ms) {
	DelayMilliseconds_ThreadX(delay_ms);
}

/**
 * @brief Prepare ST's IMX335 middleware bridge with the project's I2C and ISP hooks.
 * @retval true when the middleware probe succeeded and exposed the sensor callbacks.
 */
static bool __attribute__((unused)) CameraPlatform_InitImx335Driver(void) {
	memset(&camera_sensor, 0, sizeof(camera_sensor));
	memset(&camera_sensor_driver, 0, sizeof(camera_sensor_driver));
	memset(&camera_sensor_config, 0, sizeof(camera_sensor_config));

	camera_sensor.Address = BCAMS_IMX_I2C_ADDRESS_HAL;
	camera_sensor.ClockInHz = IMX335_INCK_24MHZ;
	camera_sensor.Init = CameraPlatform_I2cInit;
	camera_sensor.DeInit = CameraPlatform_I2cDeInit;
	camera_sensor.WriteReg = CameraPlatform_I2cWriteReg;
	camera_sensor.ReadReg = CameraPlatform_I2cReadReg;
	camera_sensor.GetTick = CameraPlatform_GetTickMs;
	camera_sensor.Delay = CameraPlatform_CmwDelay;
	camera_sensor.ShutdownPin = CameraPlatform_CmwShutdownPin;
	camera_sensor.EnablePin = CameraPlatform_CmwEnablePin;
	camera_sensor.hdcmipp = &hdcmipp;
	camera_sensor.appliHelpers.SetSensorGain = CameraPlatform_IspSetSensorGain;
	camera_sensor.appliHelpers.GetSensorGain = CameraPlatform_IspGetSensorGain;
	camera_sensor.appliHelpers.SetSensorExposure =
			CameraPlatform_IspSetSensorExposure;
	camera_sensor.appliHelpers.GetSensorExposure =
			CameraPlatform_IspGetSensorExposure;
	camera_sensor.appliHelpers.GetSensorInfo = CameraPlatform_IspGetSensorInfo;
	camera_sensor.appliHelpers.SetSensorTestPattern =
			CameraPlatform_IspSetSensorTestPattern;

	if (CMW_IMX335_Probe(&camera_sensor,
			&camera_sensor_driver) != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to probe IMX335 through ST camera middleware.\r\n");
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Local IMX335 bridge prepared: driver_start=%u driver_run=%u ctx_write=%u ctx_read=%u.\r\n",
			(camera_sensor_driver.Start != NULL) ? 1U : 0U,
			(camera_sensor_driver.Run != NULL) ? 1U : 0U,
			(camera_sensor.ctx_driver.Ctx.WriteReg != NULL) ? 1U : 0U,
			(camera_sensor.ctx_driver.Ctx.ReadReg != NULL) ? 1U : 0U);

	return true;
}

/**
 * @brief Start IMX335 streaming through ST's middleware so the ISP is enabled too.
 * @retval true when the sensor enters streaming mode.
 */
static bool CameraPlatform_StartImx335Stream(void) {
	uint8_t mode_select = IMX335_MODE_STANDBY;
	int32_t raw_start_status = IMX335_OK;
	bool started_via_cmw_wrapper = false;

	if (camera_stream_started) {
		return true;
	}

#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
	int32_t cmw_start_status = CMW_ERROR_NONE;

	if (camera_sensor_driver.Start != NULL) {
		cmw_start_status = camera_sensor_driver.Start(&camera_sensor);
		if (cmw_start_status == CMW_ERROR_NONE) {
			started_via_cmw_wrapper = true;
			if (!CameraPlatform_EnableImx335AutoExposure()) {
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Warning: IMX335 auto exposure could not be confirmed after stream start.\r\n");
			}
		} else {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW wrapper sensor start failed in raw diagnostic mode, status=%ld; falling back to raw driver.\r\n",
					(long) cmw_start_status);
		}
	}
#else
	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		camera_stream_started = true;
		/* The first processed frame often lands before AEC has climbed out of
		 * the startup black level, so give the ISP a short head start here. */
		DelayMilliseconds_ThreadX(CAMERA_FIRST_FRAME_WARMUP_DELAY_MS);
		return true;
	}
#endif

	if (!started_via_cmw_wrapper) {
		/* camera_raw_sensor is only valid when CameraPlatform_InitImx335Driver()
		 * has been called.  In the CMW-init path we skip that function to avoid
		 * resetting the sensor, so fall back to a direct I2C write instead.
		 * Per the IMX335 datasheet the correct streaming start sequence is:
		 *   1. Write MODE_SELECT=0x00  (starts internal master clock oscillator)
		 *   2. Wait >=19 ms            (oscillator stabilisation)
		 *   3. Write XMSTA=0x00        (releases master start, begins pixel output)
		 * Writing XMSTA before MODE_SELECT was the previous (wrong) order and
		 * caused the sensor to output blank frames because the pixel array never
		 * actually started clocking. */
		uint8_t streaming_value = IMX335_MODE_STREAMING; /* 0x00 */
		uint8_t xmsta_master_start_value = 0x00U;

		if (CameraPlatform_I2cWriteReg(BCAMS_IMX_I2C_ADDRESS_HAL,
		IMX335_REG_MODE_SELECT, &streaming_value, 1U) != IMX335_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to write MODE_SELECT=0x00 to start IMX335 streaming.\r\n");
			return false;
		}
		DelayMilliseconds_ThreadX(20U);

		if (CameraPlatform_I2cWriteReg(BCAMS_IMX_I2C_ADDRESS_HAL,
		IMX335_REG_XMSTA, &xmsta_master_start_value, 1U) != IMX335_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Warning: XMSTA write failed after MODE_SELECT.\r\n");
		}
		DelayMilliseconds_ThreadX(5U);

		if (!CameraPlatform_SeedImx335ExposureGain()) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Warning: failed to re-seed IMX335 exposure/gain after raw stream start.\r\n");
		}

		CameraPlatform_ReapplyImx335TestPattern();

		raw_start_status = IMX335_OK;
	}

	if (!started_via_cmw_wrapper && (raw_start_status != IMX335_OK)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to start IMX335 streaming through vendor driver.\r\n");
		return false;
	}

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 entered streaming, but mode-select readback failed.\r\n");
	}

	camera_stream_started = true;
	return true;
}

/**
 * @brief Service ST's IMX335 middleware background process for ISP state updates.
 * @retval true when the background step succeeded or is not used by this driver.
 */
static bool CameraPlatform_RunImx335Background(void) {
	/* The ISP background loop is only needed for the processed image path.
	 * Raw Pipe0 diagnostics bypass the ISP output path, so running AWB/AEC
	 * updates there can trip middleware code that expects the YUV pipeline. */
	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		camera_capture_isp_run_count++;

		if (CMW_CAMERA_Run() != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW_CAMERA_Run() failed on ISP wake %lu.\r\n",
					(unsigned long) camera_capture_isp_run_count);
			return false;
		}

		return true;
	}

	return true;
}

/**
 * @brief Return the active DCMIPP handle used by the current capture path.
 * @retval Pointer to the middleware-owned DCMIPP handle when available, else the
 *         CubeMX-generated application handle.
 */
static DCMIPP_HandleTypeDef* CameraPlatform_GetCaptureDcmippHandle(void) {
	DCMIPP_HandleTypeDef *cmw_handle = CMW_CAMERA_GetDCMIPPHandle();

	if ((cmw_handle != NULL) && (cmw_handle->Instance != NULL)) {
		return cmw_handle;
	}

	return &hdcmipp;
}

/**
 * @brief Clock hook used by ST's camera middleware before HAL_DCMIPP_Init().
 * @param hdcmipp Unused DCMIPP handle supplied by the middleware.
 * @retval HAL status from the peripheral clock configuration.
 */
HAL_StatusTypeDef MX_DCMIPP_ClockConfig(DCMIPP_HandleTypeDef *hdcmipp) {
	RCC_PeriphCLKInitTypeDef periph_clk_init = { 0 };

	UNUSED(hdcmipp);

	periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_DCMIPP
			| RCC_PERIPHCLK_CSI;
	periph_clk_init.DcmippClockSelection = RCC_DCMIPPCLKSOURCE_IC17;
	periph_clk_init.ICSelection[RCC_IC17].ClockSelection = RCC_ICCLKSOURCE_PLL1;
	periph_clk_init.ICSelection[RCC_IC17].ClockDivider = 4;
	periph_clk_init.ICSelection[RCC_IC18].ClockSelection = RCC_ICCLKSOURCE_PLL1;
	/* Keep the camera kernel clock at 24 MHz so the IMX335 timing tables line up. */
	periph_clk_init.ICSelection[RCC_IC18].ClockDivider = 50;

	return HAL_RCCEx_PeriphCLKConfig(&periph_clk_init);
}

/**
 * @brief Wait for FileX to finish mounting the SD card and creating the image directory.
 * @param timeout_ms Maximum time to wait before giving up.
 * @retval true when storage is ready for image writes.
 */
static bool CameraPlatform_WaitForStorageReady(uint32_t timeout_ms) {
	const ULONG deadline_tick = tx_time_get()
			+ CameraPlatform_MillisecondsToTicks(timeout_ms);
	ULONG actual_flags = 0U;

	while (!AppFileX_IsMediaReady()) {
		if (tx_time_get() >= deadline_tick) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Timed out waiting for FileX media readiness.\r\n");
			return false;
		}

		if (camera_storage_ready_sync_created) {
			const UINT flag_status = tx_event_flags_get(
					&camera_storage_ready_flags,
					CAMERA_STORAGE_READY_EVENT_FLAG, TX_OR_CLEAR, &actual_flags,
					CameraPlatform_MillisecondsToTicks(50U));
			if ((flag_status == TX_SUCCESS)
					&& ((actual_flags & CAMERA_STORAGE_READY_EVENT_FLAG) != 0U)) {
				break;
			}
		} else {
			DelayMilliseconds_ThreadX(50U);
		}
	}

	return true;
}

void App_ThreadX_NotifyStorageReady(void) {
	if (!camera_storage_ready_sync_created) {
		return;
	}

	(void) tx_event_flags_set(&camera_storage_ready_flags,
			CAMERA_STORAGE_READY_EVENT_FLAG, TX_OR);
}

/**
 * @brief Build a capture filename from the DS3231 time if available.
 *
 * We prefer timestamped names so saved images are easy to correlate with the
 * RTC and the boot log. If the RTC cannot be read, we fall back to the legacy
 * numbered naming helper.
 */
static bool CameraPlatform_BuildCaptureFileName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr) {
	CHAR rtc_stamp[32] = { 0 };
	int written = 0;
	const bool rtc_ready = App_Clock_GetCaptureTimestamp(rtc_stamp,
			sizeof(rtc_stamp));

	if ((file_name_ptr == NULL) || (file_name_length == 0U)
			|| (file_extension_ptr == NULL) || (file_extension_ptr[0] == '\0')) {
		return false;
	}

	if (rtc_ready) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Using RTC timestamp %s for capture name.\r\n",
				rtc_stamp);
		written = snprintf(file_name_ptr, (size_t) file_name_length,
				"capture_%s.%s", rtc_stamp, file_extension_ptr);
		return (written > 0)
				&& ((ULONG) written < file_name_length);
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] RTC timestamp unavailable; using numbered fallback.\r\n");
	return (AppFileX_GetNextCapturedImageName(file_name_ptr, file_name_length,
			file_extension_ptr) == FX_SUCCESS);
}

/**
 * @brief Capture a single frame and save it to the SD card.
 * @retval true when the frame reaches the SD card successfully.
 */
static bool CameraPlatform_CaptureAndStoreSingleFrame(void) {
	uint32_t captured_bytes = 0U;
	UINT filex_status = FX_SUCCESS;
	CHAR capture_file_name[CAMERA_CAPTURE_FILE_NAME_LENGTH] = { 0 };
	uint8_t *image_ptr = NULL;
	ULONG image_length = captured_bytes;
	const CHAR *file_extension = camera_capture_use_cmw_pipeline ? "yuv422"
			: "raw16";
	if (!CameraPlatform_WaitForStorageReady(CAMERA_STORAGE_WAIT_TIMEOUT_MS)) {
		return false;
	}

	if (!CameraPlatform_CaptureSingleFrame(&captured_bytes)) {
		return false;
	}

	image_length = captured_bytes;

	if (camera_capture_use_cmw_pipeline) {
		image_ptr = camera_capture_result_buffer;
		if (image_ptr == NULL) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Capture buffer pointer is NULL after frame completion.\r\n");
			return false;
		}

	} else {
		image_ptr = camera_capture_result_buffer;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Frame ready for save: ptr=%p length=%lu pipeline=%s\r\n",
			(void *) image_ptr, (unsigned long) image_length,
			camera_capture_use_cmw_pipeline ? "processed" : "raw");

	AppCameraDiagnostics_LogCaptureBufferPreview("ready-to-save", image_ptr,
			(uint32_t) image_length);

	if (!CameraPlatform_BuildCaptureFileName(capture_file_name,
			sizeof(capture_file_name), file_extension)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to build capture filename.\r\n");
		return false;
	}

	if (camera_capture_use_cmw_pipeline) {
		CameraPlatform_LogCaptureState("processed-capture");
		AppCameraDiagnostics_LogProcessedFrameDiagnostics("processed-capture",
				image_ptr, (uint32_t) image_length);
	}

	DebugConsole_Printf(
			camera_capture_use_cmw_pipeline ?
					"[CAMERA][CAPTURE] Saving YUV422 capture to /captured_images/%s (%lu bytes)...\r\n"
					: "[CAMERA][CAPTURE] Saving raw Pipe0 capture to /captured_images/%s (%lu bytes)...\r\n",
			capture_file_name, (unsigned long) image_length);

	filex_status = AppFileX_WriteCapturedImage(capture_file_name,
			image_ptr, image_length);
	if (filex_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Failed to write image to SD card, status=%lu.\r\n",
				(unsigned long) filex_status);
		return false;
	}

	DebugConsole_Printf(
			camera_capture_use_cmw_pipeline ?
					"[CAMERA][CAPTURE] Stored %lu-byte YUV422 image at /captured_images/%s.\r\n"
					: "[CAMERA][CAPTURE] Stored %lu-byte raw Pipe0 frame at /captured_images/%s.\r\n",
			(unsigned long) image_length, capture_file_name);

	if (camera_capture_use_cmw_pipeline) {
		if (!CameraPlatform_RequestDryInference((const uint8_t *) image_ptr,
				(ULONG) image_length)) {
			DebugConsole_Printf(
					"[AI] Failed to queue one-shot dry-run inference.\r\n");
		}
	}

	return true;
}

/**
 * @brief Queue a dry-run inference request for the dedicated AI worker thread.
 * @param frame_ptr Pointer to the captured frame bytes.
 * @param frame_length Length of the captured frame in bytes.
 * @retval true when the request was queued successfully.
 */
static bool CameraPlatform_RequestDryInference(const uint8_t *frame_ptr,
		ULONG frame_length) {
	if (!camera_ai_sync_created) {
		DebugConsole_Printf(
				"[AI] Dry-run request dropped; AI queue not initialized.\r\n");
		return false;
	}

	if ((frame_ptr == NULL) || (frame_length == 0U)) {
		DebugConsole_Printf(
				"[AI] Dry-run request dropped; empty frame ptr=%p len=%lu.\r\n",
				(const void *) frame_ptr, (unsigned long) frame_length);
		return false;
	}

	if (frame_length > CAMERA_CAPTURE_BUFFER_SIZE_BYTES) {
		DebugConsole_Printf(
				"[AI] Dry-run request dropped; frame too large len=%lu max=%lu.\r\n",
				(unsigned long) frame_length,
				(unsigned long) CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		return false;
	}

	(void) memcpy(camera_ai_frame_snapshot, frame_ptr, (size_t) frame_length);
	DebugConsole_Printf(
			"[AI] Dry-run snapshot copied: src=%p dst=%p len=%lu first8=[%02X %02X %02X %02X %02X %02X %02X %02X]\r\n",
			(const void *) frame_ptr, (void *) camera_ai_frame_snapshot,
			(unsigned long) frame_length, camera_ai_frame_snapshot[0],
			camera_ai_frame_snapshot[1], camera_ai_frame_snapshot[2],
			camera_ai_frame_snapshot[3], camera_ai_frame_snapshot[4],
			camera_ai_frame_snapshot[5], camera_ai_frame_snapshot[6],
			camera_ai_frame_snapshot[7]);

	DebugConsole_Printf(
			"[AI] Queueing dry-run request: ptr=%p length=%lu\r\n",
			(void *) camera_ai_frame_snapshot, (unsigned long) frame_length);

	camera_ai_request_frame_ptr = camera_ai_frame_snapshot;
	camera_ai_request_frame_length = frame_length;
	if (tx_semaphore_put(&camera_ai_request_semaphore) != TX_SUCCESS) {
		DebugConsole_Printf(
				"[AI] Failed to signal dry-run request semaphore.\r\n");
		return false;
	}

	return true;
}

static void CameraPlatform_LogCaptureBufferSummary(uint32_t captured_bytes) {
	AppCameraDiagnostics_LogCaptureBufferSummary(
			(const uint8_t*) camera_capture_result_buffer, captured_bytes,
			camera_capture_use_cmw_pipeline);
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

	/* Log the noncacheable MPU region bounds so we can verify it covers the
	 * full 1.2 MB capture buffer.  __snoncacheable / __enoncacheable are linker
	 * symbols exported from the LD script. */
	{
		extern uint32_t __snoncacheable;
		extern uint32_t __enoncacheable;
		uint32_t nc_base = (uint32_t) (uintptr_t) &__snoncacheable;
		uint32_t nc_end = (uint32_t) (uintptr_t) &__enoncacheable;
		uint32_t nc_size = nc_end - nc_base;
		uint32_t buf_end = (uint32_t) (uintptr_t) camera_capture_buffers[0]
				+ CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
		DebugConsole_Printf(
				"[MPU] noncacheable: base=0x%08lX end=0x%08lX size=%lu KB; "
						"buf[0]=0x%08lX buf_end=0x%08lX %s\r\n",
				(unsigned long) nc_base, (unsigned long) nc_end,
				(unsigned long) (nc_size / 1024U),
				(unsigned long) (uintptr_t) camera_capture_buffers[0],
				(unsigned long) buf_end,
				(buf_end <= nc_end) ? "COVERED" : "OVERFLOW-NOT-COVERED");
	}

	/* Read RISAF2/RISAF3 and RIMC registers live Ã¢â‚¬â€ SystemIsolation_Config runs
	 * before UART init so we can only verify them here.
	 * Expected: RISAF2/3 CFGR=0x00000101 CIDCFGR=0x00020002
	 *           RIMC_ATTRx[9] (DCMIPP): MCID=1 MSEC=1 MPRIV=1 -> 0x00000310
	 *           RISC_SECCFGRx[2] bit29=1 and RISC_PRIVCFGRx[2] bit29=1. */
	{
		volatile uint32_t r2_cr = RISAF2_NS->CR;
		volatile uint32_t r2_cfgr = RISAF2_NS->REG[0].CFGR;
		volatile uint32_t r2_cidcfgr = RISAF2_NS->REG[0].CIDCFGR;
		volatile uint32_t r2_startr = RISAF2_NS->REG[0].STARTR;
		volatile uint32_t r2_endr = RISAF2_NS->REG[0].ENDR;
		volatile uint32_t r3_cfgr = RISAF3_NS->REG[0].CFGR;
		volatile uint32_t r3_cidcfgr = RISAF3_NS->REG[0].CIDCFGR;
		volatile uint32_t rimc_cr = RIFSC->RIMC_CR;
		volatile uint32_t rimc_dcmipp =
		RIFSC->RIMC_ATTRx[RIF_MASTER_INDEX_DCMIPP];
		volatile uint32_t risc_sec2 = RIFSC->RISC_SECCFGRx[2];
		volatile uint32_t risc_priv2 = RIFSC->RISC_PRIVCFGRx[2];
		/* RISC_SECCFGRx[4] bit12=AXISRAM1, bit13=AXISRAM2.
		 * RISC_PRIVCFGRx[4] same positions.
		 * Expected: SEC bits set, PRIV bits clear (DCMIPP is MPRIV=0). */
		volatile uint32_t risc_sec4 = RIFSC->RISC_SECCFGRx[4];
		volatile uint32_t risc_priv4 = RIFSC->RISC_PRIVCFGRx[4];
		DebugConsole_Printf(
				"[RIF] RISAF2 CR=0x%08lX REG0: CFGR=0x%08lX CIDCFGR=0x%08lX STARTR=0x%08lX ENDR=0x%08lX\r\n",
				(unsigned long) r2_cr, (unsigned long) r2_cfgr,
				(unsigned long) r2_cidcfgr, (unsigned long) r2_startr,
				(unsigned long) r2_endr);
		DebugConsole_Printf(
				"[RIF] RISAF3 REG0: CFGR=0x%08lX CIDCFGR=0x%08lX\r\n",
				(unsigned long) r3_cfgr, (unsigned long) r3_cidcfgr);
		DebugConsole_Printf(
				"[RIF] RIMC: CR=0x%08lX DCMIPP_ATTR=0x%08lX (MCID=%lu MSEC=%lu MPRIV=%lu)\r\n",
				(unsigned long) rimc_cr, (unsigned long) rimc_dcmipp,
				(unsigned long) ((rimc_dcmipp >> 4U) & 0x7U),
				(unsigned long) ((rimc_dcmipp >> 8U) & 0x1U),
				(unsigned long) ((rimc_dcmipp >> 9U) & 0x1U));
		DebugConsole_Printf(
				"[RIF] RISC REG2: SECCFGR=0x%08lX PRIVCFGR=0x%08lX "
				"(DCMIPP SEC=%lu PRIV=%lu, CSI SEC=%lu PRIV=%lu)\r\n",
				(unsigned long) risc_sec2, (unsigned long) risc_priv2,
				(unsigned long) ((risc_sec2 >> 29U) & 1U),
				(unsigned long) ((risc_priv2 >> 29U) & 1U),
				(unsigned long) ((risc_sec2 >> 28U) & 1U),
				(unsigned long) ((risc_priv2 >> 28U) & 1U));
		/* AXISRAM1=bit12, AXISRAM2=bit13. Want: SEC=1, PRIV=0 for both. */
		DebugConsole_Printf("[RIF] RISC REG4: SECCFGR=0x%08lX PRIVCFGR=0x%08lX "
				"(AXISRAM1 SEC=%lu PRIV=%lu, AXISRAM2 SEC=%lu PRIV=%lu)\r\n",
				(unsigned long) risc_sec4, (unsigned long) risc_priv4,
				(unsigned long) ((risc_sec4 >> 12U) & 1U),
				(unsigned long) ((risc_priv4 >> 12U) & 1U),
				(unsigned long) ((risc_sec4 >> 13U) & 1U),
				(unsigned long) ((risc_priv4 >> 13U) & 1U));
	}
}

#if 0
/**
 * @brief Dump a dense CSI fault snapshot whenever a lane error occurs.
 * @param reason Short note describing what triggered the dump.
 * @param data_lane Failing CSI data lane from the HAL callback.
 * @param capture_dcmipp Active DCMIPP handle, when available.
 */
static void CameraPlatform_LogCsiFaultSnapshot(const char *reason,
		uint32_t data_lane, DCMIPP_HandleTypeDef *capture_dcmipp) {
	const uint32_t csi_sr0 = CSI->SR0;
	const uint32_t csi_sr1 = CSI->SR1;
	const uint32_t csi_pcr = CSI->PCR;
	const uint32_t csi_pfcr = CSI->PFCR;
	const uint32_t csi_lmcfgr = CSI->LMCFGR;
	const uint32_t csi_ier0 = CSI->IER0;
	const uint32_t csi_ier1 = CSI->IER1;
	const uint32_t csi_fcr0 = CSI->FCR0;
	const uint32_t csi_fcr1 = CSI->FCR1;
	const uint32_t csi_err1 = CSI->ERR1;
	const uint32_t csi_err2 = CSI->ERR2;
	const uint32_t csi_spdfr = CSI->SPDFR;
	const uint32_t csi_lb0cfgr = CSI->LB0CFGR;
	const uint32_t csi_lb1cfgr = CSI->LB1CFGR;
	const uint32_t csi_lb2cfgr = CSI->LB2CFGR;
	const uint32_t csi_lb3cfgr = CSI->LB3CFGR;
	const uint32_t csi_prgitr = CSI->PRGITR;
	const uint32_t hdcmipp_error =
			(capture_dcmipp != NULL) ? capture_dcmipp->ErrorCode : 0U;

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault snapshot (%s): lane=%lu hdcmipp_err=0x%08lX armed=%u stream_started=%u failed=%u sof=%u eof=%u frame_events=%lu vsync_events=%lu isp_runs=%lu csi_irqs=%lu dcmipp_irqs=%lu line_errs=%lu mask=0x%08lX linebyte_events=%lu.\r\n",
			(reason != NULL) ? reason : "capture", (unsigned long) data_lane,
			(unsigned long) hdcmipp_error,
			camera_capture_snapshot_armed ? 1U : 0U,
			camera_stream_started ? 1U : 0U, camera_capture_failed ? 1U : 0U,
			camera_capture_sof_seen ? 1U : 0U,
			camera_capture_eof_seen ? 1U : 0U,
			(unsigned long) camera_capture_frame_event_count,
			(unsigned long) camera_capture_vsync_event_count,
			(unsigned long) camera_capture_isp_run_count,
			(unsigned long) camera_capture_csi_irq_count,
			(unsigned long) camera_capture_dcmipp_irq_count,
			(unsigned long) camera_capture_line_error_count,
			(unsigned long) camera_capture_line_error_mask,
			(unsigned long) camera_capture_csi_linebyte_event_count);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault regs (%s): SR0=0x%08lX SR1=0x%08lX PCR=0x%08lX PFCR=0x%08lX LMCFGR=0x%08lX IER0=0x%08lX IER1=0x%08lX FCR0=0x%08lX FCR1=0x%08lX ERR1=0x%08lX ERR2=0x%08lX SPDFR=0x%08lX LB0CFGR=0x%08lX LB1CFGR=0x%08lX LB2CFGR=0x%08lX LB3CFGR=0x%08lX PRGITR=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture", (unsigned long) csi_sr0,
			(unsigned long) csi_sr1, (unsigned long) csi_pcr,
			(unsigned long) csi_pfcr, (unsigned long) csi_lmcfgr,
			(unsigned long) csi_ier0, (unsigned long) csi_ier1,
			(unsigned long) csi_fcr0, (unsigned long) csi_fcr1,
			(unsigned long) csi_err1, (unsigned long) csi_err2,
			(unsigned long) csi_spdfr, (unsigned long) csi_lb0cfgr,
			(unsigned long) csi_lb1cfgr, (unsigned long) csi_lb2cfgr,
			(unsigned long) csi_lb3cfgr, (unsigned long) csi_prgitr);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault decode (%s): clk(active=%u stop=%u) dl0(active=%u sync=%u stop=%u esot=%u esotsync=%u esc=%u esyncesc=%u ectrl=%u) dl1(active=%u sync=%u stop=%u esot=%u esotsync=%u esc=%u esyncesc=%u ectrl=%u) sr0(short_pkt=%u vc0state=%u crc=%u ecc=%u cecc=%u id=%u spkterr=%u wd=%u syncerr=%u lb0=%u lb1=%u lb2=%u lb3=%u).\r\n",
			(reason != NULL) ? reason : "capture",
			(csi_sr1 & CSI_SR1_ACTCLF) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPCLF) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ACTDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_SYNCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTSYNCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_EESCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESYNCESCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ECTRLDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ACTDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_SYNCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTSYNCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_EESCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESYNCESCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ECTRLDL1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC0STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CRCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_ECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_IDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_WDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SYNCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB2F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB3F) ? 1U : 0U);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault error bits (%s): dcmipp(axi=%u parallel_sync=%u p0_limit=%u p0_ovr=%u p1_ovr=%u p2_ovr=%u) csi(sync=%u wdg=%u spkt=%u id=%u cecc=%u ecc=%u crc=%u dphy_ctrl=%u dphy_lp_sync=%u dphy_escape=%u sot_sync=%u sot=%u).\r\n",
			(reason != NULL) ? reason : "capture",
			((hdcmipp_error & HAL_DCMIPP_ERROR_AXI_TRANSFER) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PARALLEL_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE0_LIMIT) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE0_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE1_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE2_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_WDG) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SPKT) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DATA_ID) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_CECC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_ECC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_CRC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_CTRL) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_LP_SYNC) != 0U) ?
					1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_ESCAPE) != 0U) ?
					1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SOT_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SOT) != 0U) ? 1U : 0U);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault SR0 decode (%s): sof0=%u eof0=%u spkt=%u vc0state=%u vc1state=%u vc2state=%u vc3state=%u ccfifo=%u crc=%u ecc=%u cecc=%u id=%u spkterr=%u wd=%u syncerr=%u lb0=%u lb1=%u lb2=%u lb3=%u.\r\n",
			(reason != NULL) ? reason : "capture",
			(csi_sr0 & CSI_SR0_SOF0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_EOF0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC0STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC1STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC2STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC3STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CCFIFOFF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CRCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_ECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_IDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_WDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SYNCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB2F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB3F) ? 1U : 0U);

	AppCameraDiagnostics_LogDcmippPipeRegisters(reason, capture_dcmipp);
	AppCameraDiagnostics_LogCsiLineByteCounters(reason,
			camera_capture_csi_linebyte_event_count);
}
#endif


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
 * @brief Dump the current camera, ISP, and DCMIPP state for black-frame
 *        diagnostics.
 * @param reason Short note describing what triggered the dump.
 */
static void CameraPlatform_LogCaptureState(const char *reason) {
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
 * @brief Configure the capture pipe for a 224x224 YUV422 capture sourced from RAW10 CSI input.
 * @param[out] captured_bytes_ptr Receives the final image byte count on success.
 * @retval true when a frame-complete interrupt arrives without a DCMIPP error.
 */
static bool CameraPlatform_CaptureSingleFrame(uint32_t *captured_bytes_ptr) {
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

	/* Keep blue available for the save-success flash later in the flow. */
	BSP_LED_Off(LED_BLUE);
	if (!CameraPlatform_PrepareDcmippSnapshot()) {
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
			return false;
		}
	}

	camera_capture_snapshot_armed = true;

	if (!camera_stream_started) {
		if (!CameraPlatform_StartImx335Stream()) {
			(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
			DCMIPP_VIRTUAL_CHANNEL0);
			camera_capture_snapshot_armed = false;
			return false;
		}
	} else {
		/* On later snapshots, give the already-running stream a brief moment to
		 * advance to the armed frame boundary before we block on completion. */
		DelayMilliseconds_ThreadX(CAMERA_STREAM_WARMUP_DELAY_MS);
	}

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
				/* Buffer is noncacheable Ã¢â‚¬â€ no invalidate needed, reads go to SRAM. */
				/* Read first 8 bytes straight from SRAM after invalidate Ã¢â‚¬â€ before
				 * the nonzero scan Ã¢â‚¬â€ to show whether the cache or DMA is the issue. */
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
								"[CAMERA][SCAN] Only zeros past 0xAA fill (BSS?): first_word=0x%05lX last=0x%05lX count=%lu addr=0x%08lX Ã¢â‚¬â€ DMA may be writing zeros or not writing.\r\n",
								(unsigned long) first_nonaa,
								(unsigned long) last_nonaa,
								(unsigned long) nonaa_count,
								(unsigned long) ((uintptr_t) completed_buffer_ptr
										+ first_nonaa * 4U));
					} else {
						DebugConsole_Printf(
								"[CAMERA][SCAN] All 0xAA in buffer from 0x%08lX Ã¢â‚¬â€ DMA not writing to SRAM at all.\r\n",
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
				completed_nonzero_bytes =
						AppCameraBuffers_CountNonZeroBytes(
						completed_buffer_ptr,
						CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
				camera_capture_result_buffer = completed_buffer_ptr;
				CameraPlatform_LogCaptureBufferSummary(camera_capture_byte_count);
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Completed frame handed off: idx=%lu ptr=%p bytes=%lu nonzero=%lu snapshot_armed=%u pipe_state=%lu.\r\n",
						(unsigned long) completed_buffer_index,
						(void *) completed_buffer_ptr,
						(unsigned long) camera_capture_byte_count,
						(unsigned long) completed_nonzero_bytes,
						camera_capture_snapshot_armed ? 1U : 0U,
						(unsigned long) HAL_DCMIPP_PIPE_GetState(capture_dcmipp,
						CAMERA_CAPTURE_PIPE));

				if (camera_capture_use_cmw_pipeline) {
					const uint32_t next_buffer_index = (completed_buffer_index
							+ 1U) % CAMERA_CAPTURE_BUFFER_COUNT;

					if (HAL_DCMIPP_PIPE_SetMemoryAddress(capture_dcmipp,
					CAMERA_CAPTURE_PIPE, DCMIPP_MEMORY_ADDRESS_0,
							(uint32_t) camera_capture_buffers[next_buffer_index])
							!= HAL_OK) {
						camera_capture_failed = true;
						camera_capture_error_code = 0x50495045U; /* 'PIPE' */
						camera_capture_snapshot_armed = false;
						(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp,
						CAMERA_CAPTURE_PIPE,
						DCMIPP_VIRTUAL_CHANNEL0);
						return false;
					}

					camera_capture_active_buffer_index = next_buffer_index;
					camera_capture_frame_done = false;

					if (completed_nonzero_bytes == 0U) {
						/* Keep the stream alive until the ISP/AEC pipeline produces a
						 * nonzero frame or the overall capture timeout expires. */
						keep_waiting_for_convergence = true;
					}
				}

				/* The raw diagnostic path also needs a few warm-up frames when the
				 * sensor starts in a black state. Keep waiting until we either see a
				 * nonzero payload or run out of the overall capture timeout. */
				if (!camera_capture_use_cmw_pipeline
						&& (completed_nonzero_bytes == 0U)) {
					keep_waiting_for_convergence = true;
				}

				if (keep_waiting_for_convergence) {
					next_wait_log_tick = tx_time_get()
							+ CameraPlatform_MillisecondsToTicks(1000U);
					continue;
				}

				camera_capture_snapshot_armed = false;
			}
			break;
		}

		if ((LONG) (tx_time_get() - next_wait_log_tick) >= 0) {
			next_wait_log_tick = tx_time_get()
					+ CameraPlatform_MillisecondsToTicks(1000U);
		}

		if ((LONG) (deadline_tick - tx_time_get()) <= 0) {
			break;
		}
	}

	if (semaphore_status != TX_SUCCESS) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Timed out waiting for frame event, status=%lu cmsr1=0x%08lX cmsr2=0x%08lX sr0=0x%08lX pipe=%lu state=%lu sof=%u eof=%u line_err_count=%lu line_err_mask=0x%08lX.\r\n",
				(unsigned long) semaphore_status,
				(unsigned long) capture_dcmipp->Instance->CMSR1,
				(unsigned long) capture_dcmipp->Instance->CMSR2,
				(unsigned long) CSI->SR0, (unsigned long) CAMERA_CAPTURE_PIPE,
				(unsigned long) HAL_DCMIPP_PIPE_GetState(capture_dcmipp,
				CAMERA_CAPTURE_PIPE), camera_capture_sof_seen ? 1U : 0U,
				camera_capture_eof_seen ? 1U : 0U,
				(unsigned long) camera_capture_line_error_count,
				(unsigned long) camera_capture_line_error_mask);
		(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp,
		CAMERA_CAPTURE_PIPE,
		DCMIPP_VIRTUAL_CHANNEL0);
		return false;
	}

	if (camera_capture_failed) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] DCMIPP reported capture error code 0x%08lX.\r\n",
				(unsigned long) camera_capture_error_code);
		if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
			(void) CMW_CAMERA_Suspend(CAMERA_CAPTURE_PIPE);
			camera_stream_started = false;
		}
		(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
		DCMIPP_VIRTUAL_CHANNEL0);
		return false;
	}

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		(void) CMW_CAMERA_Suspend(CAMERA_CAPTURE_PIPE);
		camera_stream_started = false;
	}
	(void) HAL_DCMIPP_CSI_PIPE_Stop(capture_dcmipp, CAMERA_CAPTURE_PIPE,
	DCMIPP_VIRTUAL_CHANNEL0);

	AppCameraBuffers_InvalidateCaptureRegion(camera_capture_byte_count);

	if ((camera_capture_byte_count == 0U)
			|| (camera_capture_byte_count > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Invalid capture byte count %lu (sof=%u eof=%u frame=%u expected=%lu).\r\n",
				(unsigned long) camera_capture_byte_count,
				camera_capture_sof_seen ? 1U : 0U,
				camera_capture_eof_seen ? 1U : 0U,
				camera_capture_frame_done ? 1U : 0U,
				(unsigned long) CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		return false;
	}

	*captured_bytes_ptr = camera_capture_byte_count;
	return true;
}

/**
 * @brief Attempt to arm a single capture on VC0.
 * @retval true when HAL accepted the snapshot start request.
 */
static bool CameraPlatform_StartDcmippSnapshot(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();
	uint8_t mode_select = IMX335_MODE_STANDBY;

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		if (CMW_CAMERA_Start(CAMERA_CAPTURE_PIPE,
				camera_capture_buffers[camera_capture_active_buffer_index],
				CMW_MODE_CONTINUOUS) != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW_CAMERA_Start() failed for capture pipe continuous capture.\r\n");
			return false;
		}

		/* Some IMX335 bring-up notes expect XMSTA to be asserted after the sensor
		 * leaves standby, not only during the initial table load. Re-apply it here
		 * so we can see whether the timing matters on this board. */
		{
			uint8_t xmsta_master_start_value = 0x00U;

			if (CameraPlatform_I2cWriteReg(BCAMS_IMX_I2C_ADDRESS_HAL,
			IMX335_REG_XMSTA, &xmsta_master_start_value, 1U) != IMX335_OK) {
				DebugConsole_Printf(
						"[CAMERA][CAPTURE] Warning: XMSTA post-start write failed.\r\n");
			}

			DelayMilliseconds_ThreadX(5U);
		}

	if (!CameraPlatform_SeedImx335ExposureGain()) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Warning: failed to re-seed IMX335 exposure/gain after CMW stream start.\r\n");
	}

	CameraPlatform_ReapplyImx335TestPattern();

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 stream started through CMW, but mode-select readback failed.\r\n");
	}

		return true;
	}

	/* Keep the raw diagnostic pipe in continuous mode so it matches the
	 * reference capture start sequence, while the app still stops after the
	 * first useful frame. */
	if (HAL_DCMIPP_CSI_PIPE_Start(capture_dcmipp, CAMERA_CAPTURE_PIPE,
	DCMIPP_VIRTUAL_CHANNEL0,
			(uint32_t) camera_capture_buffers[camera_capture_active_buffer_index],
			DCMIPP_MODE_CONTINUOUS) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] HAL_DCMIPP_CSI_PIPE_Start() failed. mode=%lu pipe_state=%lu buffer=0x%08lX sr0=0x%08lX\r\n",
				(unsigned long) HAL_DCMIPP_GetMode(capture_dcmipp),
				(unsigned long) HAL_DCMIPP_PIPE_GetState(capture_dcmipp,
				CAMERA_CAPTURE_PIPE),
				(unsigned long) ((uint32_t) camera_capture_buffers[camera_capture_active_buffer_index]),
				(unsigned long) CSI->SR0);
		return false;
	}

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_MODE_SELECT, &mode_select, 1U) != IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] DCMIPP pipe started through HAL, but IMX335 mode-select readback failed.\r\n");
	}

	return true;
}

/**
 * @brief Configure the capture pipe using ST's camera middleware crop/downsize helpers.
 * @retval true when the output path is ready for a 224x224 YUV422 frame.
 */
static bool CameraPlatform_PrepareDcmippSnapshot(void) {
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
		/* Crop a centered square ROI first so the 4:3 sensor frame does not get
		 * stretched into a square output and turn circular dials into ellipses. */
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

		/* CMW_CAMERA_SetPipeConfig already configures YUV conversion internally
		 * when output_format == DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1. Calling
		 * the HAL YUV helpers again here resets the pipe state and causes
		 * HAL_DCMIPP_CSI_PIPE_Start to fail with HAL_ERROR. */

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

		/* Skip the confirmed blank/embedded prefix so the raw buffer starts on
		 * active pixels instead of the four top black lines. */
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

		/* Arm the raw dump length explicitly so PIPE0 has a bounded AXI write
		 * window before the sensor starts streaming. The dump counter reports a
		 * byte count on this path, so we cap the pipe to the full capture buffer. */
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
	/* Let the ISP/AEC loop settle quickly on the first processed frame. */
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
 * @brief Arm a CSI line/byte counter on VC0 so we can confirm line progress.
 * @retval true when the counter was programmed successfully.
 */
static bool CameraPlatform_ConfigureCsiLineByteProbe(void) {
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
 * @brief Seed IMX335 exposure and gain with a conservative starting point.
 *
 * ST's middleware initializes the sensor conservatively. We back off the
 * previous maxed-out seed so the live optical path does not clip immediately
 * on bright scenes.
 * @retval true when the middleware accepted the seed settings.
 */
static bool CameraPlatform_SeedImx335ExposureGain(void) {
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

	/* Start conservatively but not at the absolute floor, so the first live
	 * frame stays out of clipping while still preserving some scene detail. */
	/* Start a little brighter than the previous seed so the first usable frame
	 * lands closer to the scene instead of hugging the dark end. */
	/* Bias the very first frame a little brighter so the processed pipeline
	 * has a better starting point before AEC takes over. */
	seed_exposure_us = sensor_info.exposure_min
			+ ((sensor_info.exposure_max - sensor_info.exposure_min) / 3U);
	if (seed_exposure_us < sensor_info.exposure_min) {
		seed_exposure_us = sensor_info.exposure_min;
	}

	seed_gain_mdb = sensor_info.gain_min;
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
 * @brief Force the IMX335 ISP path into auto-exposure mode.
 *
 * The IMX335 middleware bridge does not expose a sensor-level exposure-mode
 * setter, so the ISP AEC state is the control point for auto exposure here.
 * @retval true when the ISP accepted the AEC enable request.
 */
static bool CameraPlatform_EnableImx335AutoExposure(void) {
	uint8_t aec_enabled = 0U;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-call\r\n");
	if (ISP_SetAECState(&camera_sensor.hIsp, 1U) != ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to enable IMX335 ISP auto exposure.\r\n");
		return false;
	}

	if (ISP_GetAECState(&camera_sensor.hIsp, &aec_enabled) == ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ISP auto exposure enabled (AEC=%u).\r\n",
				(unsigned int) aec_enabled);
	} else {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ISP auto exposure requested, but readback failed.\r\n");
	}

	return true;
}

/**
 * @brief Re-apply the configured IMX335 test pattern after streaming starts.
 *
 * Some sensors latch the test-pattern generator more reliably once the stream
 * is already live, so we re-write the configured pattern as a low-risk
 * diagnostic nudge after start-up.
 */
static void CameraPlatform_ReapplyImx335TestPattern(void) {
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
 * @brief Provide the current HAL tick count to the vendor IMX335 driver.
 * @retval Current system tick in milliseconds.
 */
static int32_t CameraPlatform_GetTickMs(void) {
	return ThreadxUtils_GetTickMs();
}

/**
 * @brief Convert milliseconds to ThreadX ticks, rounding up to ensure waits do not underflow.
 * @param timeout_ms Timeout in milliseconds.
 * @retval Equivalent timeout in scheduler ticks.
 */
static ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms) {
	return ThreadxUtils_MillisecondsToTicks(timeout_ms);
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
 * @brief Let the ST ISP layer update sensor gain through the IMX335 middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param gain Target sensor gain.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspSetSensorGain(
		uint32_t camera_instance, int32_t gain) {
	UNUSED(camera_instance);
	if ((camera_sensor_driver.SetGain == NULL)
			|| (camera_sensor_driver.SetGain(&camera_sensor, gain)
					!= CMW_ERROR_NONE)) {
		return ISP_ERR_SENSORGAIN;
	}

	camera_sensor_gain_cache = gain;
	return ISP_OK;
}

/**
 * @brief Let the ST ISP layer read sensor gain through the IMX335 middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param gain Receives the current sensor gain.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspGetSensorGain(
		uint32_t camera_instance, int32_t *gain) {
	UNUSED(camera_instance);
	if (gain == NULL) {
		return ISP_ERR_SENSORGAIN;
	}

	*gain = camera_sensor_gain_cache;
	return ISP_OK;
}

/**
 * @brief Let the ST ISP layer update sensor exposure through the IMX335 middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param exposure Target sensor exposure.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspSetSensorExposure(
		uint32_t camera_instance, int32_t exposure) {
	UNUSED(camera_instance);
	if ((camera_sensor_driver.SetExposure == NULL)
			|| (camera_sensor_driver.SetExposure(&camera_sensor, exposure)
					!= CMW_ERROR_NONE)) {
		return ISP_ERR_SENSOREXPOSURE;
	}

	camera_sensor_exposure_cache = exposure;
	return ISP_OK;
}

/**
 * @brief Let the ST ISP layer read sensor exposure through the IMX335 middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param exposure Receives the current sensor exposure.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspGetSensorExposure(
		uint32_t camera_instance, int32_t *exposure) {
	UNUSED(camera_instance);
	if (exposure == NULL) {
		return ISP_ERR_SENSOREXPOSURE;
	}

	*exposure = camera_sensor_exposure_cache;
	return ISP_OK;
}

/**
 * @brief Let the ST ISP layer query IMX335 capabilities through the middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param info Receives the sensor description.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspGetSensorInfo(
		uint32_t camera_instance, ISP_SensorInfoTypeDef *info) {
	UNUSED(camera_instance);
	if ((info == NULL) || (camera_sensor_driver.GetSensorInfo == NULL)
			|| (camera_sensor_driver.GetSensorInfo(&camera_sensor, info)
					!= CMW_ERROR_NONE)) {
		return ISP_ERR_SENSORINFO;
	}

	return ISP_OK;
}

/**
 * @brief Let the ST ISP layer control the IMX335 test pattern through the middleware bridge.
 * @param camera_instance Unused single-camera instance selector.
 * @param mode Requested IMX335 test-pattern mode.
 * @retval ISP status for the requested operation.
 */
static ISP_StatusTypeDef CameraPlatform_IspSetSensorTestPattern(
		uint32_t camera_instance, int32_t mode) {
	UNUSED(camera_instance);
	if ((camera_sensor_driver.SetTestPattern == NULL)
			|| (camera_sensor_driver.SetTestPattern(&camera_sensor, mode)
					!= CMW_ERROR_NONE)) {
		return ISP_ERR_SENSORTESTPATTERN;
	}

	return ISP_OK;
}

/**
 * @brief Camera middleware pipe VSYNC callback used for app-side diagnostics.
 * @param pipe DCMIPP pipe that asserted VSYNC.
 * @retval CMW_ERROR_NONE always.
 */
int CMW_CAMERA_PIPE_VsyncEventCallback(uint32_t pipe) {
	if (pipe != CAMERA_CAPTURE_PIPE) {
		return CMW_ERROR_NONE;
	}

	(void) tx_semaphore_put(&camera_capture_isp_semaphore);
	camera_capture_vsync_event_count++;

	/* No DebugConsole_Printf from ISR Ã¢â‚¬â€ mutex is illegal in interrupt context. */

	return CMW_ERROR_NONE;
}

/**
 * @brief Camera middleware pipe frame callback used to release the capture thread.
 * @param pipe DCMIPP pipe that completed a frame.
 * @retval CMW_ERROR_NONE always.
 */
int CMW_CAMERA_PIPE_FrameEventCallback(uint32_t pipe) {
	uint32_t byte_count = 0U;
	HAL_StatusTypeDef counter_status = HAL_ERROR;
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (pipe != CAMERA_CAPTURE_PIPE) {
		return CMW_ERROR_NONE;
	}

	camera_capture_frame_event_count++;

	if (camera_capture_use_cmw_pipeline) {
		counter_status = HAL_OK;
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	} else if ((capture_dcmipp != NULL) && (capture_dcmipp->Instance != NULL)) {
		counter_status = HAL_DCMIPP_PIPE_GetDataCounter(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &byte_count);
	}

	camera_capture_counter_status = (uint32_t) counter_status;
	camera_capture_reported_byte_count = byte_count;

	if ((counter_status != HAL_OK) || (byte_count == 0U)) {
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	} else if (!camera_capture_use_cmw_pipeline
			&& (byte_count > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Raw pipe counter %lu exceeds the %lux%lu capture buffer; normalizing to %lu bytes for save.\r\n",
				(unsigned long) byte_count,
				(unsigned long) CAMERA_CAPTURE_WIDTH_PIXELS,
				(unsigned long) CAMERA_CAPTURE_HEIGHT_PIXELS,
				(unsigned long) CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	}

	camera_capture_byte_count = byte_count;
	camera_capture_frame_done = true;

	/* No DebugConsole_Printf from ISR Ã¢â‚¬â€ tx_mutex_get is illegal in interrupt
	 * context.  The main capture thread logs first8 after the semaphore fires. */

	(void) tx_semaphore_put(&camera_capture_done_semaphore);

	return CMW_ERROR_NONE;
}

/**
 * @brief Camera middleware pipe error callback for the snapshot path.
 * @param pipe Pipe that reported the error.
 */
void CMW_CAMERA_PIPE_ErrorCallback(uint32_t pipe) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (pipe != CAMERA_CAPTURE_PIPE) {
		return;
	}

	camera_capture_failed = true;
	camera_capture_error_code = capture_dcmipp->ErrorCode;
	camera_capture_snapshot_armed = false;
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
	camera_capture_snapshot_armed = false;
	/* Log from main thread after semaphore fires Ã¢â‚¬â€ no Printf from ISR. */
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

	if (!camera_capture_snapshot_armed) {
		return;
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

	if (!camera_capture_snapshot_armed) {
		return;
	}

	camera_capture_eof_seen = true;
	/* Ignore VC-level EOF as a wake source. In continuous sensor streaming it can
	 * arrive for frames that are not the armed PIPE0 snapshot yet, which would
	 * release the waiting thread with a zero byte count. */
}

/**
 * @brief CSI callback for data-lane line errors.
 * @param hdcmipp HAL DCMIPP handle.
 * @param DataLane Failing CSI data lane.
 */
void HAL_DCMIPP_CSI_LineErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t DataLane) {
	if (hdcmipp == NULL) {
		return;
	}

	camera_capture_line_error_count++;
	camera_capture_line_error_mask |= (1UL << (DataLane & 0x1FU));
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
	if (hdcmipp == NULL) {
		return;
	}

	/* No Printf from ISR Ã¢â‚¬â€ state is read by main thread after semaphore fires. */
}

/**
 * @brief CSI callback for line/byte counter diagnostics.
 * @param hdcmipp HAL DCMIPP handle.
 * @param Counter Counter that asserted the line/byte event.
 */
void HAL_DCMIPP_CSI_LineByteEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Counter) {
	UNUSED(hdcmipp);
	camera_capture_csi_linebyte_event_count++;

	camera_capture_csi_linebyte_event_logged = true; /* flag for main thread */
}

/* USER CODE END 1 */


