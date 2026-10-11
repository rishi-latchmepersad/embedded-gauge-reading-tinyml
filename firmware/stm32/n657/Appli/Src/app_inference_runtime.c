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
#include "sd_debug_log_service.h"
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
/* Wake the logger only for queued records; avoid a periodic polling wakeup. */
static TX_SEMAPHORE inference_log_ready_semaphore;
static ULONG inference_log_queue_storage[INFERENCE_LOG_QUEUE_DEPTH];
/* Queue depth can reach zero as soon as the logger dequeues a record. Keep a
 * separate state bit so Stop-mode entry also waits for the SD write itself. */
static volatile bool inference_log_write_in_progress = false;
static volatile ULONG inference_log_dropped_rows = 0U;

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
/* The capture thread marks only the last request in a burst as loggable.  The
 * marker travels with the queued request so an earlier AI completion cannot
 * be mistaken for the burst result. */
static volatile bool camera_ai_next_request_is_burst_final = true;
static volatile bool camera_ai_request_is_burst_final = true;
/* Retain a valid result from this burst so a failed final frame can still
 * publish the best value produced by an earlier frame. */
static volatile float camera_ai_burst_last_valid_value = 0.0f;
static volatile bool camera_ai_burst_last_valid_value_valid = false;
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

/**
 * @brief Emit one published inference value to UART and the SD log queue.
 * @param final_value Value selected by the live smoothing/filter pipeline.
 * @return None.
 * @sideeffects Writes UART diagnostics and queues one SD inference record.
 */
static void AppInferenceRuntime_LogPublishedResult(float final_value) {
	union {
		float f;
		ULONG u;
	} bits = { .f = final_value };
	char inference_line[64] = { 0 };

	AppInferenceLog_FormatFloatTenths(inference_line,
			sizeof(inference_line), "[AI] Final AI value published: ", final_value);
	(void) DebugConsole_WriteString(inference_line);

	/* The exact value is retained in the SD inference log. UART only needs the
	 * human-readable published value once, after the burst is complete. */
	if (inference_log_thread_created) {
        const UINT status = tx_queue_send(&inference_log_queue, &bits.u, TX_NO_WAIT);
        if (status == TX_SUCCESS) {
            (void) tx_semaphore_put(&inference_log_ready_semaphore);
        } else {
            TX_INTERRUPT_SAVE_AREA
            TX_DISABLE
            inference_log_dropped_rows++;
            TX_RESTORE
            DebugConsole_Printf("[INFER_LOG][LOSS] domain=ThreadX op=enqueue status=%lu dropped=%lu\r\n",
                    (unsigned long) status, (unsigned long) inference_log_dropped_rows);
        }
	}
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
	status = tx_semaphore_create(&inference_log_ready_semaphore, "inference_log_ready", 0U);
	if (status != TX_SUCCESS) { return status; }

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
 * @brief Wait until the asynchronous inference log queue is empty.
 * @param timeout_ms Maximum wait duration in milliseconds.
 * @retval true when no inference log rows are queued or being written,
 *         false on timeout.
 * @sideeffects Yields the camera thread while the FileX-backed logger drains.
 */
bool AppInferenceRuntime_WaitForLogQueueDrain(uint32_t timeout_ms) {
	uint32_t elapsed_ms = 0U;

	if (!inference_log_thread_created) {
		return true;
	}

	while (elapsed_ms <= timeout_ms) {
		ULONG enqueued = 0U;
		if ((tx_queue_info_get(&inference_log_queue, NULL, &enqueued, NULL,
					NULL, NULL, NULL) == TX_SUCCESS) && (enqueued == 0U)
				&& !inference_log_write_in_progress) {
			return true;
		}
		DelayMilliseconds_Cooperative(10U);
		elapsed_ms += 10U;
	}

	return false;
}

/**
 * @brief Start a new capture burst and clear its result fallback state.
 */
void AppInferenceRuntime_BeginBurst(void) {
	TX_INTERRUPT_SAVE_AREA

	TX_DISABLE
	camera_ai_burst_last_valid_value = 0.0f;
	camera_ai_burst_last_valid_value_valid = false;
	/* A burst caller explicitly marks each request below. Defaulting this to an
	 * intermediate request prevents a partially started burst from logging its
	 * first frame as the final average. */
	camera_ai_next_request_is_burst_final = false;
	TX_RESTORE
}

/**
 * @brief Mark whether the next accepted AI request is the burst final frame.
 */
void AppInferenceRuntime_SetNextRequestFinal(bool final_request) {
	TX_INTERRUPT_SAVE_AREA

	TX_DISABLE
	camera_ai_next_request_is_burst_final = final_request;
	TX_RESTORE
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
	/* Consume the marker atomically with the request ownership tuple. Restore
	 * the single-frame default so callers outside a burst remain loggable. */
	camera_ai_request_is_burst_final = camera_ai_next_request_is_burst_final;
	camera_ai_next_request_is_burst_final = true;
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
		bool request_is_burst_final = true;

		if (request_status != TX_SUCCESS) {
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_FAILED);
			continue;
		}
		AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_EXECUTING);

		frame_ptr = (const uint8_t *) camera_ai_request_frame_ptr;
		frame_length = camera_ai_request_frame_length;
		request_is_burst_final = camera_ai_request_is_burst_final;
		const uint64_t frame_capture_time_us = camera_ai_request_capture_time_us;
		camera_ai_request_frame_ptr = NULL;
		camera_ai_request_frame_length = 0U;
		camera_ai_request_capture_time_us = 0ULL;

		/* Queue failures and published values are the operational events. A
		 * successful dequeue is intentionally silent in the normal console. */

#if APP_BASELINE_ENABLE_THREAD && APP_BASELINE_QUEUE_WITH_CAPTURE
		/* The classical comparator was queued with the capture (before this
		 * worker) and reads the same stopped DMA frame. Its priority is lower
		 * than this worker, so without this wait it only gets CPU during our
		 * WFE sleeps - by which time the NPU has clobbered the buffer rows and
		 * the baseline reads garbage (2026-08-05). Yield until it finishes
		 * (bounded) before starting any model work. */
		{
			uint32_t baseline_wait_ms = 0U;
			while (AppBaselineRuntime_IsEstimateInFlight() &&
					(baseline_wait_ms < 3000U)) {
				DelayMilliseconds_Cooperative(10U);
				baseline_wait_ms += 10U;
			}
		}
#endif

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
			if (request_is_burst_final
					&& camera_ai_burst_last_valid_value_valid) {
				/* A failed final frame should not erase a valid result collected by
				 * the earlier burst frames. */
				AppInferenceRuntime_LogPublishedResult(
						camera_ai_burst_last_valid_value);
			}
		} else {
			AppInferenceRuntime_SetWorkerState(APP_INFERENCE_WORKER_PUBLISHING);
			float result = 0.0f;
			if (App_AI_GetLastInferenceResult(&result)) {
				camera_ai_burst_last_valid_value = result;
				camera_ai_burst_last_valid_value_valid = true;

				/* Intermediate frames still update smoothing and metrics, but only
				 * the burst-final value enters UART/SD inference logging. */
				if (request_is_burst_final) {
					AppInferenceRuntime_LogPublishedResult(result);
				}
			} else {
				if (request_is_burst_final
						&& camera_ai_burst_last_valid_value_valid) {
					/* Preserve one result when the final exposure fails geometry
					 * validation after an earlier frame succeeded. */
					AppInferenceRuntime_LogPublishedResult(
							camera_ai_burst_last_valid_value);
				} else if (request_is_burst_final) {
					(void) DebugConsole_WriteString(
							"[AI] Final AI value not published (held or invalid).\r\n");
				}
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
 * @brief Return the number of inference records that could not be saved this boot.
 * @return Queue, RTC, format and final write failures; never a success estimate.
 */
ULONG AppInferenceRuntime_GetDroppedLogRows(void) {
    return inference_log_dropped_rows;
}

/**
 * @brief Append one CSV row, checking every FileX operation and preserving its cause.
 * @param media Mounted card; name absolute CSV path; row complete text record.
 * @return FileX status or ThreadX lock status, with an explicit domain in errors.
 * @sideeffects Creates header for new/empty files; closes and flushes before success.
 */
static UINT AppInferenceRuntime_AppendCsv(FX_MEDIA *media, CHAR *name, CHAR *row) {
    FX_FILE file = {0};
    const char *operation = "create_directory";
    UINT status = AppFileX_AcquireMediaLock();
    bool opened = false;
    if (status != TX_SUCCESS) {
        DebugConsole_Printf("[INFER_LOG][ERROR] domain=ThreadX op=lock status=%lu\r\n", (unsigned long) status);
        return status;
    }
    status = fx_directory_create(media, "/inference");
    if (status == FX_ALREADY_CREATED) { status = FX_SUCCESS; }
    if (status == FX_SUCCESS) {
        operation = "open";
        status = fx_file_open(media, &file, name, FX_OPEN_FOR_WRITE);
        if (status == FX_NOT_FOUND) {
            operation = "create";
            status = fx_file_create(media, name);
            if ((status == FX_SUCCESS) || (status == FX_ALREADY_CREATED)) {
                operation = "open_created";
                status = fx_file_open(media, &file, name, FX_OPEN_FOR_WRITE);
            }
        }
        opened = (status == FX_SUCCESS);
    }
    if (opened && (file.fx_file_current_file_size == 0U)) {
        static const CHAR header[] = "datetime,value_degC\n";
        operation = "write_header";
        status = fx_file_write(&file, (VOID *) header, sizeof(header) - 1U);
    }
    if (status == FX_SUCCESS) {
        operation = "seek_end";
        status = fx_file_relative_seek(&file, 0U, FX_SEEK_END);
    }
    if (status == FX_SUCCESS) {
        operation = "write_row";
        status = fx_file_write(&file, row, (ULONG) strlen(row));
    }
    /* Close even after a partial write, but retain the first failing operation.
     * Flush after close: closing changes directory metadata that also needs committing. */
    if (opened) {
        const UINT close_status = fx_file_close(&file);
        if (status == FX_SUCCESS) { operation = "close"; status = close_status; }
    }
    const UINT flush_status = fx_media_flush(media);
    if (status == FX_SUCCESS) { operation = "flush"; status = flush_status; }
    if (status != FX_SUCCESS) {
        char error[160];
        (void) snprintf(error, sizeof(error), "[INFER_LOG][ERROR] domain=FileX op=%s path=%s status=%lu (0x%02lX)",
                operation, name, (unsigned long) status, (unsigned long) status);
        (void) DebugConsole_WriteString(error);
        (void) DebugConsole_WriteString("\r\n");
        (void) SdDebugLogService_EnqueueLine(error);
    }
    AppFileX_ReleaseMediaLock();
    return status;
}

/**
 * @brief Consume inference records and acknowledge only committed CSV writes.
 * @param thread_input Unused ThreadX input.
 * @sideeffects Reports dropped records; a failed/partial write is never retried
 * automatically because repeating it could create a duplicate or malformed row.
 */
static VOID InferenceLogThread_Entry(ULONG thread_input) {
    (void) thread_input;
    while (1) {
        ULONG value_bits = 0U;
        UINT receive_status;
        (void) tx_semaphore_get(&inference_log_ready_semaphore, TX_WAIT_FOREVER);
        /* Atomic dequeue + busy flag closes the race with the pre-sleep barrier.
         * TX_NO_WAIT ensures interrupts are never disabled across a wait or SD IO. */
        TX_INTERRUPT_SAVE_AREA
        TX_DISABLE
        receive_status = tx_queue_receive(&inference_log_queue, &value_bits, TX_NO_WAIT);
        inference_log_write_in_progress = (receive_status == TX_SUCCESS);
        TX_RESTORE
        if (receive_status != TX_SUCCESS) {
            continue;
        }
        union { ULONG u; float f; } bits = {.u = value_bits};
        CHAR timestamp[32] = {0};
        CHAR name[INFERENCE_LOG_FILE_NAME_LENGTH] = {0};
        CHAR row[INFERENCE_LOG_ROW_MAX_LENGTH] = {0};
        UINT status = FX_MEDIA_NOT_OPEN;
        if (AppFileX_IsMediaReady()
                && App_Clock_GetCaptureTimestamp(timestamp, sizeof(timestamp))) {
            /* Resolve the date for each dequeued row; midnight cannot keep the
             * previous day's filename, and empty files left by ENOSPC get a header. */
            (void) snprintf(name, sizeof(name), "/inference/%.10s.csv", timestamp);
            const int written = snprintf(row, sizeof(row), "%s,", timestamp);
            if ((written > 0) && ((size_t) written < sizeof(row))) {
                AppInferenceLog_FormatFloatTenths(row + written,
                        sizeof(row) - (size_t) written, "", bits.f);
                status = AppInferenceRuntime_AppendCsv(AppFileX_GetMediaHandle(), name, row);
            } else { status = FX_INVALID_NAME; }
        }
        if (status == FX_SUCCESS) {
            DebugConsole_Printf("[INFER_LOG] Saved and flushed: %s", row);
        } else {
            TX_DISABLE
            inference_log_dropped_rows++;
            TX_RESTORE
            char loss[128];
            (void) snprintf(loss, sizeof(loss), "[INFER_LOG][LOSS] row-not-saved dropped=%lu status=%lu media=%u rtc=%u",
                    (unsigned long) inference_log_dropped_rows, (unsigned long) status,
                    AppFileX_IsMediaReady() ? 1U : 0U, timestamp[0] ? 1U : 0U);
            (void) DebugConsole_WriteString(loss);
            (void) DebugConsole_WriteString("\r\n");
            (void) SdDebugLogService_EnqueueLine(loss);
        }
        inference_log_write_in_progress = false;
    }
}
