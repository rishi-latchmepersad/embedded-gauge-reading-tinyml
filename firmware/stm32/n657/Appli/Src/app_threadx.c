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
#include "app_baseline_runtime.h"
#include "app_ai_config.h"
#include "app_inference_runtime.h"
#include "app_image_cleanup.h"
#include "app_storage.h"
#include "app_threadx_config.h"
#include "app_memory_budget.h"
#include "app_filex.h"
#include "app_ai.h"
#include "ds3231_clock.h"
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
#define CAMERA_STARTUP_MEDIA_READY_TIMEOUT_MS 90000U

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
/* The live product contract is CMW/ISP MONO_Y8.  The platform probe may set
 * this again after sensor initialization, but the pre-probe default must not
 * accidentally arm the raw diagnostic pipe. */
bool camera_capture_use_cmw_pipeline = true;
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
/* DCMIPP callbacks run in interrupt context.  This monotonically increasing
 * event count is the only completion notification they publish; the capture
 * thread consumes it without calling ThreadX from the ISR. */
volatile uint32_t camera_capture_done_event_count = 0U;
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
static VOID AppThreadX_StackErrorHandler(TX_THREAD *thread_ptr);

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
	/* Keep the visible LED off until the heartbeat thread takes ownership so a
	 * solid red LED means fault handling, not normal startup state. */
	BSP_LED_Off(LED_RED);
	BSP_LED_Off(LED_BLUE);
	BSP_LED_Off(LED_GREEN);
	(void) AppImageCleanup_SetBootTick(tx_time_get());
	if (camera_init_thread_created && camera_isp_thread_created
			&& camera_heartbeat_thread_created) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Start skipped: camera threads already created.\r\n");
		return TX_SUCCESS;
	}

	if (!camera_capture_sync_created) {
		const UINT mutex_status = tx_mutex_create(&camera_capture_cmw_mutex,
				"camera_capture_cmw", TX_INHERIT);
		if (mutex_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera middleware mutex, status=%lu\r\n",
					(unsigned long) mutex_status);
			return mutex_status;
		}
		camera_capture_cmw_mutex_created = true;

		camera_capture_sync_created = true;
	}

	{
		const UINT stack_notify_status =
				tx_thread_stack_error_notify(AppThreadX_StackErrorHandler);
		if (stack_notify_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to register stack error callback, status=%lu\r\n",
					(unsigned long) stack_notify_status);
		}
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

	#if APP_BASELINE_ENABLE_THREAD
	{
		AppBaselineRuntime_SetCalibrationProfileByName(
			APP_BASELINE_CALIBRATION_PROFILE_NAME);

		const UINT baseline_runtime_status = AppBaselineRuntime_Start();
		if (baseline_runtime_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[BASELINE] Failed to start baseline runtime, status=%lu\r\n",
					(unsigned long) baseline_runtime_status);
			return baseline_runtime_status;
		}
	}
	#endif

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
 * @brief ThreadX stack overflow/underflow callback.
 * @param thread_ptr Thread that tripped stack checking.
 */
static VOID AppThreadX_StackErrorHandler(TX_THREAD *thread_ptr) {
	const CHAR *thread_name = "unknown";

	if ((thread_ptr != TX_NULL) && (thread_ptr->tx_thread_name != TX_NULL)) {
		thread_name = thread_ptr->tx_thread_name;
	}

	DebugConsole_Printf(
			"[FAULT] ThreadX stack error: thread=%s stack_start=%p stack_end=%p stack_ptr=%p\r\n",
			thread_name,
			(thread_ptr != TX_NULL) ? thread_ptr->tx_thread_stack_start : TX_NULL,
			(thread_ptr != TX_NULL) ? thread_ptr->tx_thread_stack_end : TX_NULL,
			(thread_ptr != TX_NULL) ? thread_ptr->tx_thread_stack_ptr : TX_NULL);
	BSP_LED_On(LED_RED);
	while (1) {
		__NOP();
	}
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
	(void) DebugConsole_WriteString(
			"[CAMERA][THREAD] Camera init startup delay skipped; probing immediately.\r\n");
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
			if (!CameraPlatform_DisableImx335AutoExposure()) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Warning: failed to lock IMX335 exposure after probe.\r\n");
			}
			App_ThreadX_UnlockCameraMiddleware();
			camera_capture_isp_loop_paused = false;
			DebugConsole_Printf(
					"[CAMERA][THREAD] Camera probe completed successfully.\r\n");

			/* FileX mounts asynchronously. Complete that one-time startup work
			 * before model initialization so the boot dry-run cannot overlap SD
			 * ACMD41 polling and consume its inference time budget. */
			if (!AppFileX_IsMediaReady()) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Waiting for FileX media before AI startup.\r\n");
				if (!AppStorage_WaitForMediaReady(
						CAMERA_STARTUP_MEDIA_READY_TIMEOUT_MS)) {
					DebugConsole_Printf(
							"[CAMERA][THREAD] FileX startup wait timed out; continuing without SD.\r\n");
				}
			}

			/* Start cleanup only after the camera has proven it can probe and
			 * capture, so the background sweeper cannot interfere with startup. */
			{
				const UINT image_cleanup_start_status = AppImageCleanup_Start();
				if (image_cleanup_start_status != TX_SUCCESS) {
					DebugConsole_Printf(
							"[IMAGE][CLEANUP] Deferred start failed, status=%lu.\r\n",
							(unsigned long) image_cleanup_start_status);
				}
			}

	if (!App_AI_Model_Init()) {
				DebugConsole_Printf(
					"[AI] Model runtime init failed; continuing without inference.\r\n");
		}

#if APP_AI_ENABLE_TIP_FOCUS_GEOMETRY_STAGE && APP_AI_ENABLE_TIP_FOCUS_BOOT_DRY_RUN
		(void)AppAI_TipFocus_DryRun();
#endif

		BSP_LED_Off(LED_BLUE);
		DebugConsole_Printf(
				"[CAMERA][THREAD] Entering capture/inference loop (period=60s)...\r\n");
		while (1) {
			bool storage_ready = AppFileX_IsMediaReady();
			uint32_t next_delay_ms = CAMERA_CAPTURE_PERIOD_MS;

			/* Do not consume the first frame while FileX is still mounting the
			 * card. The previous flow captured immediately, skipped the SD write,
			 * and then allowed AI and SD bring-up to contend for the same window. */
			if (!storage_ready) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Waiting for FileX media before capture.\r\n");
				if (!AppStorage_WaitForMediaReady(
						CAMERA_STARTUP_MEDIA_READY_TIMEOUT_MS)) {
					DebugConsole_Printf(
							"[CAMERA][THREAD] FileX media wait timed out; retrying capture later.\r\n");
					DelayMilliseconds_Cooperative(5000U);
					continue;
				}
				storage_ready = true;
				DebugConsole_Printf(
						"[CAMERA][THREAD] FileX media ready; starting capture.\r\n");
			}

			if (AppCameraCapture_CaptureAndStoreSingleFrame()) {
				DebugConsole_Printf(
					"[CAMERA][THREAD] Capture saved and AI handoff accepted.\r\n");
			} else {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture/inference attempt failed.\r\n");
			}

			DelayMilliseconds_Cooperative(next_delay_ms);
		}
	}

	App_ThreadX_UnlockCameraMiddleware();
	camera_capture_isp_loop_paused = false;
	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera probe failed or is not configured yet.\r\n");
}

/**
 * @brief Low-priority heartbeat thread that toggles a visible board LED.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraHeartbeatThread_Entry(ULONG thread_input) {
	(void) thread_input;

	/* Drive the user-visible green LED so the board shows liveness without
	 * relying on UART traffic or a hidden GPIO. */
	BSP_LED_Off(LED_GREEN);
	DebugConsole_Printf("[WATCHDOG] heartbeat thread running.\r\n");

	while (1) {
		BSP_LED_Toggle(LED_GREEN);
		/* The LED remains the liveness indicator. UART pulses are opt-in because
		 * a five-second heartbeat obscures capture and inference failures. */
#if CAMERA_HEARTBEAT_ENABLE_UART_PULSES
		/* Read the external RTC only for this low-rate diagnostic line.  Keeping
		 * the capture and inference logs free of per-line I2C reads avoids adding
		 * blocking RTC retries to the hot path. */
		char watchdog_timestamp[32] = { 0 };
		if (App_Clock_GetCurrentTimestamp(watchdog_timestamp,
				sizeof(watchdog_timestamp))) {
			DebugConsole_Printf("[%s] [WATCHDOG] pulse\r\n",
					watchdog_timestamp);
		} else {
			/* Preserve the heartbeat even if the DS3231 is temporarily offline. */
			DebugConsole_WriteString("[WATCHDOG] pulse (RTC unavailable)\r\n");
		}
#endif
		DelayMilliseconds_ThreadX(CAMERA_HEARTBEAT_PULSE_MS);
		BSP_LED_Toggle(LED_GREEN);
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
		/* The ISP service owns CMW only when the capture transaction is not
		 * paused.  The old timeout-based semaphore path could wake here during
		 * snapshot setup and race the capture thread inside the same middleware. */
		if (camera_stream_started && camera_cmw_initialized
				&& !camera_capture_isp_loop_paused) {
			if (!AppCameraCapture_RunImx335Background()) {
				camera_capture_failed = true;
				camera_capture_error_code = 0x49535052U; /* 'ISPR' */
			}
		}

		DelayMilliseconds_ThreadX(20U);
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
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	}

	camera_capture_byte_count = byte_count;
	camera_capture_frame_done = true;
	camera_capture_done_event_count++;

	/* No DebugConsole_Printf or ThreadX call from ISR. The capture thread
	 * observes the event counter and performs all follow-up work in thread
	 * context. */

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
	camera_capture_done_event_count++;
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
	camera_capture_done_event_count++;
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
	camera_capture_done_event_count++;
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
		camera_capture_done_event_count++;
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
