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
#include "app_ai_config.h"
#include "app_baseline_runtime.h"
#include "app_camera_buffers.h"
#include "app_camera_capture.h"
#include "app_camera_platform.h"
#include "app_filex.h"
#include "app_inference_log_config.h"
#include "app_inference_log_utils.h"
#include "app_memory_budget.h"
#include "app_threadx_config.h"
#include "debug_console.h"
#include "debug_led.h"
#include "ds3231_clock.h"
#include "inference_metrics.h"
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
/* The CNN is now the sole inference authority. The baseline is no longer
 * allowed to override the CNN output. The classical path may still run
 * for diagnostic logging, but the CNN value is always the final answer. */
/* #define APP_HYBRID_BASELINE_WAIT_MS 3000U  -- removed, no hybrid wait */
/* #define APP_HYBRID_BASELINE_POLL_MS 10U   -- removed, no hybrid poll */
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
/* Keep the AI worker stack out of the OBB reloc runtime window. The OBB
 * package claims a large AXISRAM span starting at 0x34100000, so placing the
 * worker stack in .npusram6 can collide with the live reloc image and corrupt
 * the thread context mid-inference. Let the linker place this in the normal
 * BSS/stack RAM instead. */
static ULONG camera_ai_thread_stack[CAMERA_AI_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_ai_thread_created = false;
static TX_SEMAPHORE camera_ai_request_semaphore;
static bool camera_ai_sync_created = false;
static volatile const uint8_t *camera_ai_request_frame_ptr = NULL;
static volatile ULONG camera_ai_request_frame_length = 0U;
static volatile uint64_t camera_ai_request_capture_time_us = 0ULL;
static volatile bool camera_ai_request_in_flight = false;
static volatile AppInferenceRuntime_WorkerState_t camera_ai_worker_state =
	APP_INFERENCE_WORKER_UNINITIALIZED;
static volatile ULONG camera_ai_request_generation = 0U;
static volatile ULONG camera_ai_worker_progress_tick = 0U;
static volatile bool camera_ai_last_request_succeeded = false;
static bool app_inference_runtime_initialized = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static VOID CameraAIThread_Entry(ULONG thread_input);
static VOID InferenceLogThread_Entry(ULONG thread_input);

static void AppInferenceRuntime_SetWorkerState(
		AppInferenceRuntime_WorkerState_t state) {
	camera_ai_worker_state = state;
	camera_ai_worker_progress_tick = tx_time_get();
}
/* AppInferenceRuntime_GetFreshBaselineEstimate removed: no hybrid override */



/* USER CODE END PFP */

/* USER CODE BEGIN 0 */
/* AppInferenceRuntime_GetFreshBaselineEstimate() removed -- no hybrid override */


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

	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	camera_ai_request_frame_ptr = NULL;
	camera_ai_request_frame_length = 0U;
	camera_ai_request_capture_time_us = 0ULL;
	camera_ai_request_in_flight = false;
	camera_ai_request_generation = 0U;
	camera_ai_last_request_succeeded = false;
	TX_RESTORE
	AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_WAITING);

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
 * @brief Report whether the AI worker still owns the submitted camera frame.
 * @retval true while the queued request or model execution is active.
 */
bool AppInferenceRuntime_IsInferenceInFlight(void) {
	bool in_flight = false;

	TX_INTERRUPT_SAVE_AREA
	TX_DISABLE
	in_flight = camera_ai_request_in_flight;
	TX_RESTORE
	return in_flight;
}

/**
 * @brief Return the result of the most recently completed AI request.
 * @retval true when the learned ellipse/keypoint pipeline published a value.
 * @retval false when the request failed or no request has completed yet.
 */
bool AppInferenceRuntime_WasLastRequestSuccessful(void) {
	return camera_ai_last_request_succeeded;
}

/**
 * @brief Queue a dry inference request for the AI worker thread.
 */
bool AppInferenceRuntime_RequestDryInference(const uint8_t *frame_ptr,
		ULONG frame_length) {
	TX_INTERRUPT_SAVE_AREA
	bool in_flight = false;
	AppInferenceRuntime_WorkerState_t worker_state =
		APP_INFERENCE_WORKER_UNINITIALIZED;
	const volatile uint8_t *queued_frame_ptr = NULL;
	ULONG queued_frame_length = 0U;
	bool repaired_idle_flag = false;

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

	/* Read the ownership tuple as one critical section. A stale in-flight byte
	 * must not block the first live capture, but a real queued/executing request
	 * must never be cleared because that would permit two readers of the same
	 * snapshot. */
	TX_DISABLE
	in_flight = camera_ai_request_in_flight;
	worker_state = camera_ai_worker_state;
	queued_frame_ptr = camera_ai_request_frame_ptr;
	queued_frame_length = camera_ai_request_frame_length;
	if (in_flight && (worker_state == APP_INFERENCE_WORKER_WAITING)
			&& (queued_frame_ptr == NULL) && (queued_frame_length == 0U)) {
		/* The worker is idle and owns no frame; repair only the inconsistent
		 * flag. This is safe even if a prior camera transaction was interrupted. */
		camera_ai_request_in_flight = false;
		in_flight = false;
		repaired_idle_flag = true;
	}
	TX_RESTORE
	if (repaired_idle_flag) {
		DebugConsole_Printf(
			"[AI] Repaired stale idle ownership flag gen=%lu.\r\n",
			(unsigned long)camera_ai_request_generation);
	}

	if (in_flight) {
		DebugConsole_Printf(
			"[AI] Dry-run request dropped; worker busy state=%s gen=%lu frame=%p len=%lu.\r\n",
			AppInferenceRuntime_WorkerStateName(worker_state),
			(unsigned long)camera_ai_request_generation,
			(const void *)(uintptr_t)queued_frame_ptr,
			(unsigned long)queued_frame_length);
		return false;
	}

	/* Anchor AI timing before queueing so the latency includes the full
	 * request-to-result path, not just worker execution. */

	TX_DISABLE
	camera_ai_request_capture_time_us = Metrics_GetMicros();
	Metrics_StartInference("AI");
	camera_ai_last_request_succeeded = false;
	/* The camera stops the sensor before handing this buffer to the worker, so
	 * the worker owns a stable frame without paying for a second 400 KiB copy.
	 * A later capture is not allowed until this request clears. Do not log while
	 * interrupts are disabled: HAL UART polling can consume the whole timeout
	 * and prevents the ThreadX/ATON progress machinery from running. */
	camera_ai_request_in_flight = true;
	camera_ai_request_generation++;
	camera_ai_request_frame_ptr = frame_ptr;
	camera_ai_request_frame_length = frame_length;
	AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_QUEUED);
	TX_RESTORE
	(void) DebugConsole_WriteString(
		"[AI] Queueing stable stopped-sensor frame.\r\n");

	if (tx_semaphore_put(&camera_ai_request_semaphore) != TX_SUCCESS) {
		TX_INTERRUPT_SAVE_AREA
		TX_DISABLE
		camera_ai_request_in_flight = false;
		camera_ai_request_frame_ptr = NULL;
		camera_ai_request_frame_length = 0U;
		AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_FAILED);
		TX_RESTORE
		Metrics_EndInference("AI", NAN);
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
	AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_WAITING);

	while (1) {
		const UINT request_status = tx_semaphore_get(&camera_ai_request_semaphore,
				TX_WAIT_FOREVER);
		const uint8_t *frame_ptr = NULL;
		ULONG frame_length = 0U;

		if (request_status != TX_SUCCESS) {
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_FAILED);
			continue;
		}
		AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_EXECUTING);

		frame_ptr = (const uint8_t *) camera_ai_request_frame_ptr;
		frame_length = camera_ai_request_frame_length;
		const uint64_t frame_capture_time_us = camera_ai_request_capture_time_us;
		camera_ai_request_frame_ptr = NULL;
		camera_ai_request_frame_length = 0U;
		camera_ai_request_capture_time_us = 0ULL;

		/* Queue failures and published values are the operational events. A
		 * successful dequeue is intentionally silent in the normal console. */

		if ((frame_ptr == NULL) || (frame_length == 0U)) {
			DebugConsole_Printf(
					"[AI] Worker woke without a queued frame; ignoring.\r\n");
			AppCameraCapture_ReleaseInferenceFrame();
			camera_ai_request_in_flight = false;
			camera_ai_last_request_succeeded = false;
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_FAILED);
			continue;
		}

		/* Keep the AI start time pinned to the queued frame capture moment. */
		if (frame_capture_time_us != 0ULL) {
			Metrics_OverrideStartTime("AI", frame_capture_time_us);
		}

		/* Mark the start of worker-side compute so queue wait is visible in the
		 * metrics while the AI model time stays comparable to the baseline. */
		Metrics_MarkComputeStart("AI");
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		AppCameraBuffers_LogFrameSignature("ai-worker-entry", frame_ptr,
				(uint32_t)frame_length);
#endif

		const bool inference_succeeded = App_AI_RunDryInferenceFromGray640(
				frame_ptr, (size_t) frame_length);
		camera_ai_last_request_succeeded = inference_succeeded;
		if (!inference_succeeded) {
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_FAILED);
			DebugConsole_Printf(
					"[AI] One-shot dry-run inference failed; continuing.\r\n");
		} else {
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_PUBLISHING);
			float result = 0.0f;
			if (App_AI_GetLastInferenceResult(&result)) {
				float final_value = result;

				union {
					float f;
					ULONG u;
				} bits = { .f = final_value };
				char inference_line[64] = { 0 };

				/* Log the final value that was published by the AI worker. */
				AppInferenceLog_FormatFloatTenths(inference_line,
						sizeof(inference_line), "[AI] Final AI value logged: ", final_value);
				(void) DebugConsole_WriteString(inference_line);

				/* The exact value is retained in the SD inference log. UART only
				 * needs the human-readable published value once. */
				(void) bits;
				if (inference_log_thread_created) {
					(void) tx_queue_send(&inference_log_queue, &bits.u,
							TX_NO_WAIT);
				}
			} else {
				(void) DebugConsole_WriteString(
						"[AI] Final AI value not published (held or invalid).\r\n");
			}
		}

		TX_INTERRUPT_SAVE_AREA
		TX_DISABLE
		AppCameraCapture_ReleaseInferenceFrame();
		camera_ai_request_in_flight = false;
		AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_WAITING);
		TX_RESTORE
	}
}

/**
 * @brief Return the current AI worker state.
 */
AppInferenceRuntime_WorkerState_t AppInferenceRuntime_GetWorkerState(void) {
	return camera_ai_worker_state;
}

/**
 * @brief Return the generation of the current or most recent AI request.
 */
ULONG AppInferenceRuntime_GetRequestGeneration(void) {
	return camera_ai_request_generation;
}

/**
 * @brief Return the last AI worker state-transition tick.
 */
ULONG AppInferenceRuntime_GetWorkerProgressTick(void) {
	return camera_ai_worker_progress_tick;
}

/**
 * @brief Convert an AI worker state to a compact diagnostic label.
 */
const char *AppInferenceRuntime_WorkerStateName(
		AppInferenceRuntime_WorkerState_t state) {
	switch (state) {
	case APP_INFERENCE_WORKER_WAITING:
		return "waiting";
	case APP_INFERENCE_WORKER_QUEUED:
		return "queued";
	case APP_INFERENCE_WORKER_EXECUTING:
		return "executing";
	case APP_INFERENCE_WORKER_PUBLISHING:
		return "publishing";
	case APP_INFERENCE_WORKER_FAILED:
		return "failed";
	case APP_INFERENCE_WORKER_UNINITIALIZED:
	default:
		return "uninitialized";
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
				int written = DebugConsole_Snprintf(log_file_name, sizeof(log_file_name),
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
			char inference_line[64] = { 0 };
			char row[INFERENCE_LOG_ROW_MAX_LENGTH] = { 0 };
			char rtc_timestamp[32] = { 0 };

			AppInferenceLog_FormatFloatTenths(inference_line,
					sizeof(inference_line), "[INFER_LOG] Inference value: ",
					inference_value);
			(void) DebugConsole_WriteString(inference_line);
			AppInferenceLog_FormatFloatMicros(inference_line,
					sizeof(inference_line), "[INFER_LOG] Inference exact: ",
					inference_value);
			(void) DebugConsole_WriteString(inference_line);

			if (!App_Clock_GetCaptureTimestamp(rtc_timestamp,
					sizeof(rtc_timestamp))) {
				DebugConsole_Printf(
						"[INFER_LOG] RTC unavailable while logging inference row.\r\n");
				break;
			}

			int written = DebugConsole_Snprintf(row, sizeof(row), "%s,",
					rtc_timestamp);
			if ((written <= 0) || ((size_t) written >= sizeof(row))) {
				DebugConsole_Printf(
						"[INFER_LOG] Failed to format CSV row.\r\n");
				break;
			}

			AppInferenceLog_FormatFloatTenths(row + written,
					sizeof(row) - (size_t) written, "", inference_value);
			written = (int) strlen(row);

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
