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
#include "app_camera_capture.h"
#include "app_camera_platform.h"
#include "app_inference_runtime.h"
#include "app_storage.h"
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
static TX_THREAD camera_isp_thread;
static ULONG camera_isp_thread_stack[CAMERA_ISP_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_isp_thread_created = false;
static TX_THREAD camera_heartbeat_thread;
static ULONG camera_heartbeat_thread_stack[CAMERA_HEARTBEAT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_heartbeat_thread_created = false;
static TX_MUTEX camera_capture_cmw_mutex;
static bool camera_capture_cmw_mutex_created = false;
CMW_IMX335_t camera_sensor;
bool camera_cmw_initialized = false;
/* Keep the middleware path active so the ISP/AEC pipeline can produce optical
 * frames instead of the raw sensor dump. */
bool camera_capture_use_cmw_pipeline = false;
TX_SEMAPHORE camera_capture_done_semaphore;
TX_SEMAPHORE camera_capture_isp_semaphore;
static bool camera_heartbeat_gpio_initialized = false;
static bool camera_capture_sync_created = false;
bool camera_stream_started = false;
volatile bool camera_capture_failed = false;
volatile uint32_t camera_capture_error_code = 0U;
volatile uint32_t camera_capture_byte_count = 0U;
volatile bool camera_capture_sof_seen = false;
volatile bool camera_capture_eof_seen = false;
volatile bool camera_capture_frame_done = false;
volatile bool camera_capture_snapshot_armed = false;
volatile uint32_t camera_capture_frame_event_count = 0U;
volatile uint32_t camera_capture_line_error_count = 0U;
volatile uint32_t camera_capture_line_error_mask = 0U;
volatile uint32_t camera_capture_csi_linebyte_event_count = 0U;
volatile bool camera_capture_csi_linebyte_event_logged = false;
volatile uint32_t camera_capture_vsync_event_count = 0U;
volatile uint32_t camera_capture_isp_run_count = 0U;
volatile bool camera_capture_isp_loop_paused = false;
/* Count raw IRQ entry points so we can tell whether the interrupt chain is
 * alive even when the higher-level callbacks stay silent. */
volatile uint32_t camera_capture_csi_irq_count = 0U;
volatile uint32_t camera_capture_dcmipp_irq_count = 0U;
volatile uint32_t camera_capture_reported_byte_count = 0U;
volatile uint32_t camera_capture_counter_status = (uint32_t) HAL_ERROR;

/* Reuse the CubeMX-generated camera control I2C instance from main.c. */
extern DCMIPP_HandleTypeDef hdcmipp;
extern I2C_HandleTypeDef hi2c2;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static VOID CameraHeartbeatThread_Entry(ULONG thread_input);

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input);
static VOID CameraIspThread_Entry(ULONG thread_input);

/**
 * @brief ThreadX app initialization hook.
 * @param memory_ptr ThreadX memory pool pointer.
 * @retval TX_SUCCESS on success.
 */
UINT App_ThreadX_Init(VOID *memory_ptr) {
	UINT ret = TX_SUCCESS;

	(void) memory_ptr;

	/* Defer thread creation until App_ThreadX_Start() so startup ordering is explicit. */
	DebugConsole_Printf(
			"[CAMERA][THREAD] ThreadX app init complete. Waiting to start camera thread...\r\n");
	return ret;
}

/**
 * @brief ThreadX startup hook that creates the camera and runtime threads.
 * @retval TX_SUCCESS on success.
 */
UINT App_ThreadX_Start(void) {
	/* Keep this function idempotent to protect against accidental double-start. */
	/* Leave the heartbeat LED under the dedicated thread so it reflects liveness
	 * instead of startup state. */
	BSP_LED_On(LED_RED);
	BSP_LED_Off(LED_BLUE);
	BSP_LED_Off(LED_GREEN);
	if (camera_init_thread_created && camera_isp_thread_created
			&& camera_heartbeat_thread_created) {
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

		semaphore_status = tx_mutex_create(&camera_capture_cmw_mutex,
				"camera_capture_cmw", TX_INHERIT);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera middleware mutex, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}
		camera_capture_cmw_mutex_created = true;

		camera_capture_sync_created = true;
	}

	{
		const UINT storage_init_status = AppStorage_Init();
		if (storage_init_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create storage-ready event flags.\r\n");
			return storage_init_status;
		}
	}

	{
		const UINT runtime_init_status = AppInferenceRuntime_Init();
		if (runtime_init_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[AI] Failed to initialize inference runtime, status=%lu\r\n",
					(unsigned long) runtime_init_status);
			return runtime_init_status;
		}
	}

	{
		const UINT runtime_start_status = AppInferenceRuntime_Start();
		if (runtime_start_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[AI] Failed to start inference runtime, status=%lu\r\n",
					(unsigned long) runtime_start_status);
			return runtime_start_status;
		}
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

	if (!camera_init_thread_created) {
		/* Create a dedicated thread so camera probing is isolated from other startup work. */
		const UINT create_status = tx_thread_create(&camera_init_thread,
				"camera_init", CameraInitThread_Entry, 0U,
				camera_init_thread_stack, sizeof(camera_init_thread_stack),
				CAMERA_INIT_THREAD_PRIORITY, CAMERA_INIT_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

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

	return TX_SUCCESS;
}

/**
 * @brief Notify the storage module that FileX media is ready.
 */
void App_ThreadX_NotifyStorageReady(void) {
	AppStorage_NotifyMediaReady();
}

/**
 * @brief Lock the shared camera middleware so only one thread touches CMW/ISP.
 * @param timeout_ticks Maximum time to wait for the mutex.
 * @retval true when the caller owns the camera middleware lock.
 */
bool App_ThreadX_LockCameraMiddleware(ULONG timeout_ticks) {
	if (!camera_capture_cmw_mutex_created) {
		return false;
	}

	return (tx_mutex_get(&camera_capture_cmw_mutex, timeout_ticks) == TX_SUCCESS);
}

/**
 * @brief Release the shared camera middleware lock.
 */
void App_ThreadX_UnlockCameraMiddleware(void) {
	if (!camera_capture_cmw_mutex_created) {
		return;
	}

	(void) tx_mutex_put(&camera_capture_cmw_mutex);
}

/**
 * @brief Kernel initialization hook used by CubeMX.
 */
void MX_ThreadX_Init(void) {
	/* USER CODE BEGIN Before_Kernel_Start */

	/* USER CODE END Before_Kernel_Start */

	tx_kernel_enter();

	/* USER CODE BEGIN Kernel_Start_Error */

	/* USER CODE END Kernel_Start_Error */
}

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input) {
	(void) thread_input;

	(void) DebugConsole_WriteString("[CAMERA] thread entry\r\n");
	DelayMilliseconds_ThreadX(CAMERA_INIT_STARTUP_DELAY_MS);
	(void) DebugConsole_WriteString("[CAMERA] probe start\r\n");
	camera_capture_isp_loop_paused = true;

	if (!App_ThreadX_LockCameraMiddleware(
			CameraPlatform_MillisecondsToTicks(
					CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS))) {
		camera_capture_isp_loop_paused = false;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Failed to lock camera middleware for probe.\r\n");
		return;
	}

	if (CameraPlatform_ProbeBCamsImx() == TX_SUCCESS) {
		App_ThreadX_UnlockCameraMiddleware();
		camera_capture_isp_loop_paused = false;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe completed successfully.\r\n");

		if (!App_AI_Model_Init()) {
			DebugConsole_Printf(
					"[AI] Model runtime init failed; continuing without inference.\r\n");
		}

		BSP_LED_Off(LED_BLUE);
		DebugConsole_Printf(
				"[CAMERA][THREAD] Entering capture/inference loop (period=60s)...\r\n");
		while (1) {
			if (AppCameraCapture_CaptureAndStoreSingleFrame()) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture and inference completed successfully.\r\n");
			} else {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture/inference attempt failed.\r\n");
			}
			DelayMilliseconds_ThreadX(60000U);
		}
	}

	App_ThreadX_UnlockCameraMiddleware();
	camera_capture_isp_loop_paused = false;
	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera probe failed or is not configured yet.\r\n");
}

/**
 * @brief Low-priority heartbeat thread that toggles the board LED.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraHeartbeatThread_Entry(ULONG thread_input) {
	(void) thread_input;

	if (!camera_heartbeat_gpio_initialized) {
		GPIO_InitTypeDef gpio_init = { 0 };

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
 * @brief Low-priority camera ISP thread that keeps the middleware running.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraIspThread_Entry(ULONG thread_input) {
	(void) thread_input;

	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera ISP service thread running.\r\n");

	while (1) {
		UINT semaphore_status = tx_semaphore_get(&camera_capture_isp_semaphore,
				CameraPlatform_MillisecondsToTicks(20U));

		if ((semaphore_status == TX_SUCCESS)
				|| (camera_stream_started && camera_cmw_initialized)) {
			if (!AppCameraCapture_RunImx335Background()) {
				camera_capture_failed = true;
				camera_capture_error_code = 0x49535052U; /* 'ISPR' */
				(void) tx_semaphore_put(&camera_capture_done_semaphore);
			}
		}
	}
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


