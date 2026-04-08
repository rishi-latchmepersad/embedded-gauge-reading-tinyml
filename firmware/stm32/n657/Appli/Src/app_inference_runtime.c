/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_inference_runtime.c
 * @brief   AI worker and inference log thread runtime.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_inference_runtime.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "app_ai.h"
#include "app_camera_buffers.h"
#include "app_camera_platform.h"
#include "app_filex.h"
#include "app_inference_log_config.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_threadx_config.h"
#include "debug_console.h"
#include "debug_led.h"
#include "ds3231_clock.h"
#include "threadx_utils.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef enum {
	INFER_LOG_STATE_INIT_DIR = 0,
	INFER_LOG_STATE_CHECK_DATE,
	INFER_LOG_STATE_NO_RTC,
	INFER_LOG_STATE_LOGGING,
} InferLogState_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

static TX_THREAD inference_log_thread;
static ULONG inference_log_thread_stack[INFERENCE_LOG_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool inference_log_thread_created = false;
static TX_QUEUE inference_log_queue;
static ULONG inference_log_queue_storage[INFERENCE_LOG_QUEUE_DEPTH];

static TX_THREAD camera_ai_thread;
static ULONG camera_ai_thread_stack[CAMERA_AI_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_ai_thread_created = false;
static TX_SEMAPHORE camera_ai_request_semaphore;
static bool camera_ai_sync_created = false;
static volatile const uint8_t *camera_ai_request_frame_ptr = NULL;
static volatile ULONG camera_ai_request_frame_length = 0U;
static bool app_inference_runtime_initialized = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static VOID CameraAIThread_Entry(ULONG thread_input);
static VOID InferenceLogThread_Entry(ULONG thread_input);

/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

/**
 * @brief Create the runtime synchronization objects used by the AI workers.
 */
UINT AppInferenceRuntime_Init(void) {
	UINT status = TX_SUCCESS;

	if (app_inference_runtime_initialized) {
		return TX_SUCCESS;
	}

	status = tx_semaphore_create(&camera_ai_request_semaphore,
			"camera_ai_request", 0U);
	if (status != TX_SUCCESS) {
		return status;
	}

	camera_ai_sync_created = true;

	status = tx_queue_create(&inference_log_queue, "inference_log_queue",
			TX_1_ULONG, inference_log_queue_storage,
			sizeof(inference_log_queue_storage));
	if (status != TX_SUCCESS) {
		camera_ai_sync_created = false;
		return status;
	}

	app_inference_runtime_initialized = true;
	return TX_SUCCESS;
}

/**
 * @brief Start the AI worker and inference logger threads.
 */
UINT AppInferenceRuntime_Start(void) {
	if (!app_inference_runtime_initialized) {
		const UINT init_status = AppInferenceRuntime_Init();
		if (init_status != TX_SUCCESS) {
			return init_status;
		}
	}

	if (!camera_ai_thread_created) {
		const UINT create_status = tx_thread_create(&camera_ai_thread,
				"camera_ai", CameraAIThread_Entry, 0U, camera_ai_thread_stack,
				sizeof(camera_ai_thread_stack), CAMERA_AI_THREAD_PRIORITY,
				CAMERA_AI_THREAD_PRIORITY, TX_NO_TIME_SLICE, TX_AUTO_START);
		if (create_status != TX_SUCCESS) {
			return create_status;
		}

		camera_ai_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera AI thread created and started.\r\n");
	}

	if (!inference_log_thread_created) {
		const UINT create_status = tx_thread_create(&inference_log_thread,
				"inference_log", InferenceLogThread_Entry, 0U,
				inference_log_thread_stack, sizeof(inference_log_thread_stack),
				INFERENCE_LOG_THREAD_PRIORITY, INFERENCE_LOG_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);
		if (create_status != TX_SUCCESS) {
			return create_status;
		}

		inference_log_thread_created = true;
		DebugConsole_Printf(
				"[INFER_LOG] Inference log thread created and started.\r\n");
	}

	return TX_SUCCESS;
}

/**
 * @brief Queue a dry inference request for the AI worker thread.
 */
bool AppInferenceRuntime_RequestDryInference(const uint8_t *frame_ptr,
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

/* USER CODE END 0 */

/**
 * @brief Low-priority AI worker that runs one queued dry inference at a time.
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
				union {
					float f;
					ULONG u;
				} bits = { .f = result };
				char inference_line[48] = { 0 };

				AppInferenceLog_FormatFloatTenths(inference_line,
						sizeof(inference_line), "[AI] Inference value: ",
						result);
				(void) DebugConsole_WriteString(inference_line);
				if (inference_log_thread_created) {
					(void) tx_queue_send(&inference_log_queue, &bits.u,
							TX_NO_WAIT);
				}
			}
		}
	}
}

/**
 * @brief Inference value logger thread.
 */
static VOID InferenceLogThread_Entry(ULONG thread_input) {
	(void) thread_input;

	InferLogState_t state = INFER_LOG_STATE_INIT_DIR;
	char today_date[12] = { 0 };
	char log_file_name[INFERENCE_LOG_FILE_NAME_LENGTH] = { 0 };
	FX_MEDIA *media = NULL;

	(void) DebugConsole_WriteString("[INFER_LOG] thread alive\r\n");

	while (1) {
		switch (state) {

		case INFER_LOG_STATE_INIT_DIR: {
			if (!AppFileX_IsMediaReady()) {
				DelayMilliseconds_ThreadX(500U);
				break;
			}

			media = AppFileX_GetMediaHandle();

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

		case INFER_LOG_STATE_CHECK_DATE: {
			char rtc_timestamp[32] = { 0 };
			const bool rtc_ok = App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp));

			if (!rtc_ok) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC not available; entering NO_RTC state.\r\n");
				state = INFER_LOG_STATE_NO_RTC;
				break;
			}

			char new_date[12] = { 0 };
			(void) memcpy(new_date, rtc_timestamp, 10U);
			new_date[10] = '\0';

			if (strcmp(new_date, today_date) != 0) {
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

				UINT lock_status = AppFileX_AcquireMediaLock();
				if (lock_status == TX_SUCCESS) {
					FX_FILE log_file = { 0 };
					UINT open_status = fx_file_open(media, &log_file,
							log_file_name, FX_OPEN_FOR_WRITE);
					if (open_status == FX_NOT_FOUND) {
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

		case INFER_LOG_STATE_NO_RTC: {
			DebugConsole_Printf(
					"[INFER_LOG] ERROR: DS3231 RTC not detected. Cannot timestamp log entries.\r\n");
			DebugLed_BlinkRedBlocking(INFERENCE_LOG_NO_RTC_BLINK_ON_MS,
					INFERENCE_LOG_NO_RTC_BLINK_OFF_MS, 5U);

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

		case INFER_LOG_STATE_LOGGING: {
			ULONG value_bits = 0U;
			const ULONG wait_ticks = CameraPlatform_MillisecondsToTicks(65000U);
			const UINT q_status = tx_queue_receive(&inference_log_queue,
					&value_bits, wait_ticks);
			if (q_status != TX_SUCCESS) {
				break;
			}

			union {
				ULONG u;
				float f;
			} bits = { .u = value_bits };
			float inference_value = bits.f;
			char inference_line[48] = { 0 };
			char row[INFERENCE_LOG_ROW_MAX_LENGTH] = { 0 };
			char rtc_timestamp[32] = { 0 };

			AppInferenceLog_FormatFloatTenths(inference_line,
					sizeof(inference_line), "[INFER_LOG] Inference value: ",
					inference_value);
			(void) DebugConsole_WriteString(inference_line);

			if (!App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp))) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC unavailable while logging inference row.\r\n");
				break;
			}

			int written = snprintf(row, sizeof(row), "%s,%.1f\r\n",
					rtc_timestamp, (double) inference_value);
			if ((written <= 0) || ((size_t) written >= sizeof(row))) {
				DebugConsole_Printf(
						"[INFER_LOG] Failed to format CSV row.\r\n");
				break;
			}

			if (!AppFileX_IsMediaReady()) {
				DebugConsole_Printf(
						"[INFER_LOG] FileX media not ready; dropping row.\r\n");
				break;
			}

			UINT lock_status = AppFileX_AcquireMediaLock();
			if (lock_status == TX_SUCCESS) {
				FX_FILE log_file = { 0 };
				UINT open_status = fx_file_open(media, &log_file, log_file_name,
						FX_OPEN_FOR_WRITE);
				if (open_status == FX_SUCCESS) {
					(void) fx_file_relative_seek(&log_file, 0U, FX_SEEK_END);
					(void) fx_file_write(&log_file, row, (ULONG) written);
					(void) fx_file_close(&log_file);
					(void) fx_media_flush(media);
				}
				AppFileX_ReleaseMediaLock();
			}

			DebugConsole_Printf("[INFER_LOG] Logged: %s", row);
			break;
		}

		default:
			state = INFER_LOG_STATE_INIT_DIR;
			break;
		}
	}
}
