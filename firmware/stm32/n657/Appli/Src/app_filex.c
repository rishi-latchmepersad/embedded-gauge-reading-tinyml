/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_filex.c
 * @author  MCD Application Team
 * @brief   FileX applicative file
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
#include "app_filex.h"
#include "app_storage.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "app_azure_rtos_config.h"
#include "app_memory_budget.h"
#include "sd_spi_ll.h"
#include "main.h"
#include "app_threadx.h"
#include "tx_api.h" /* ThreadX services like tx_thread_sleep. */
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "ds3231_clock.h"
#include "sd_debug_log_service.h"

/*
 * Keep FileX console noise low by default. The thread now emits a small number
 * of explicit status/error lines, while the detailed state breadcrumbs stay
 * behind a local opt-in switch for bring-up sessions.
 */
#ifndef APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
#define APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS 1
#endif
#ifndef APP_FILEX_ENABLE_STATE_BREADCRUMBS
#define APP_FILEX_ENABLE_STATE_BREADCRUMBS 0
#endif
#ifndef APP_FILEX_ENABLE_CAPTURE_FILE_TIMESTAMP
/*
 * Capture files are timestamped by name, so keep the FAT directory metadata
 * aligned with the same DS3231 value in the capture transaction. The media
 * flush remains on its normal two-minute cadence.
 */
#define APP_FILEX_ENABLE_CAPTURE_FILE_TIMESTAMP 1
#endif
#if !APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
#undef DebugConsole_Printf
#define DebugConsole_Printf(...) ((void) 0)
#endif

/**
 * Emit a FileX timing line only when verbose console logging is enabled.
 *
 * The normal capture path stays quiet now that the SD timing issue has been
 * understood, but the helper remains available for future bring-up sessions.
 */
static void AppFileX_WriteTimingLine(const char *format_string_pointer, ...)
{
#if APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
	char message_buffer[192];
	va_list argument_list;
	int written_length;

	va_start(argument_list, format_string_pointer);
	written_length = vsnprintf(message_buffer, sizeof(message_buffer),
			format_string_pointer, argument_list);
	va_end(argument_list);

	if ((written_length < 0) || ((size_t) written_length >= sizeof(message_buffer))) {
		(void) DebugConsole_WriteString(
				"[FILEX][CAPTURE][TIMING] timing-line-format-error\r\n");
		return;
	}

	(void) DebugConsole_WriteString(message_buffer);
#else
	(void)format_string_pointer;
#endif
}

/**
 * @brief Emit the capture save result even when verbose FileX timing is off.
 * @param format_string_pointer printf-style status format.
 * @param ... Values consumed by the format string.
 *
 * The normal FileX printf macro is intentionally compiled out. Capture
 * success/failure is operationally important, however, so keep this small
 * status channel enabled and separate from the timing flood.
 */
static void AppFileX_WriteCaptureStatusLine(const char *format_string_pointer,
		...) {
	char message_buffer[192];
	va_list argument_list;
	int written_length;

	va_start(argument_list, format_string_pointer);
	written_length = vsnprintf(message_buffer, sizeof(message_buffer),
			format_string_pointer, argument_list);
	va_end(argument_list);

	if ((written_length > 0)
			&& ((size_t) written_length < sizeof(message_buffer))) {
		(void) DebugConsole_WriteString(message_buffer);
	}
}
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef enum {
	APP_FILEX_STATE_UNINITIALIZED = 0,
	APP_FILEX_STATE_SD_SEND_CMD0,
	APP_FILEX_STATE_SD_SEND_CMD8,
	APP_FILEX_STATE_SD_WAIT_READY_ACMD41,
	APP_FILEX_STATE_SD_READ_OCR_CMD58,
	APP_FILEX_STATE_SD_READ_PARTITION0_INFO,
	APP_FILEX_STATE_FILEX_MEDIA_OPEN,
	APP_FILEX_STATE_LOG_SERVICE_INITIALIZE,
	APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE,
	APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE,
	APP_FILEX_STATE_RUNNING,
	APP_FILEX_STATE_ERROR
} AppFileX_State;

typedef struct {
	AppFileX_State state;

	UINT filex_status;
	UINT threadx_status;

	uint8_t sd_cmd0_r1;
	uint8_t sd_cmd8_r1;
	uint8_t sd_cmd58_r1;
	uint8_t sd_acmd41_r1;

	uint8_t r7_response[4];
	uint8_t ocr_response[4];

	uint32_t partition_start_lba;
	uint32_t partition_sector_count;

	uint8_t log_service_is_initialized;
	uint8_t filex_media_is_open;

	ULONG last_progress_tick;
	ULONG state_entry_tick;

	AppFileX_State last_error_state;
	UINT last_error_code;
} AppFileX_StateMachineContext;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* Main thread stack size.
 * The FileX state machine now does enough directory/media work that 2 KB was
 * too tight on the N6 board. Keep extra headroom here so SD bring-up and file
 * maintenance do not trip a silent stack overflow. */
#define FX_APP_THREAD_STACK_SIZE         8192
/* Main thread priority.
 * Keep FileX above the compute-heavy baseline/AI workers so the blue capture
 * cue can turn off on time, while still staying below the highest bring-up
 * thread. */
#define FX_APP_THREAD_PRIO               11U
/* USER CODE BEGIN PD */
#define CAPTURED_IMAGES_DIRECTORY_NAME   "captured_images"
#define CAPTURED_IMAGE_MAX_PATH_LENGTH   96U
#define CAPTURED_IMAGE_FILE_NAME_LENGTH  32U
#define CAPTURED_IMAGE_MAX_INDEX         9999U
#define CAPTURED_IMAGE_READBACK_BYTES    32U
#define APP_FILEX_CAPTURE_LOCK_TIMEOUT_MS    1000U
#define APP_FILEX_CAPTURE_FLUSH_INTERVAL_MS  120000U
/* Use timestamped files by default; rotating slots are an opt-in benchmark
 * mode because they overwrite captures and hide the one-file-per-minute log. */
#define APP_FILEX_PREPARE_CAPTURE_SLOTS      0U
#define APP_FILEX_PREALLOCATE_CAPTURE_SLOTS   0U
#define APP_FILEX_CAPTURE_FILE_FORMAT_COUNT   2U
#define APP_FILEX_CAPTURE_RING_SLOT_COUNT     4U
#define FILEX_SD_INIT_TIMEOUT_MS         60000U
#define FILEX_SD_INIT_RETRY_DELAY_MS       250U
#define FILEX_PARTITION_READ_RETRY_DELAY_MS   50U
#define FILEX_PARTITION_READ_RETRY_TIMEOUT_MS 2000U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* Main thread global data structures.  */
TX_THREAD       fx_app_thread;

/* USER CODE BEGIN PV */
static UCHAR *g_filex_media_cache_buffer = NULL; /* FileX cache buffer allocated from ThreadX byte pool. */

static FX_MEDIA g_sd_fx_media; /* FileX media control block for the SD card. */
static FX_FILE g_sd_fx_file; /* Simple test file object. */

static Sd_FileX_DriverContext g_sd_filex_driver_context; /* Global driver context used by the media driver. */
static TX_BYTE_POOL *g_filex_byte_pool_ptr = NULL; /* Byte pool used for queue + cache allocations. */
static TX_MUTEX g_filex_media_mutex;
static volatile bool g_filex_media_ready = false;
static bool g_capture_blue_gpio_initialized = false;
static volatile bool g_capture_media_flush_pending = false;
static volatile bool g_capture_media_timestamp_pending = false;
static ULONG g_captured_image_fallback_next_index = 0U;
static bool g_captured_image_fallback_index_seeded = false;
static ULONG g_capture_slot_next_index = 0U;
static ULONG g_capture_media_last_flush_tick = 0U;

typedef struct {
	bool valid;
	CHAR file_name[CAPTURED_IMAGE_FILE_NAME_LENGTH];
	CHAR timestamp[32];
} AppFileX_CaptureTimestampEntry;

#define APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH 8U
static AppFileX_CaptureTimestampEntry g_capture_timestamp_queue[APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH];
static ULONG g_capture_timestamp_queue_head = 0U;
static ULONG g_capture_timestamp_queue_tail = 0U;
static ULONG g_capture_timestamp_queue_count = 0U;
typedef enum {
	APP_FILEX_CAPTURE_FORMAT_YUV422 = 0U,
	APP_FILEX_CAPTURE_FORMAT_RAW16 = 1U
} AppFileX_CaptureFileFormat;

typedef struct {
	FX_FILE file;
	CHAR file_name[CAPTURED_IMAGE_FILE_NAME_LENGTH];
	bool open;
} AppFileX_CaptureSlot;

static AppFileX_CaptureSlot g_capture_slots[APP_FILEX_CAPTURE_FILE_FORMAT_COUNT]
		[APP_FILEX_CAPTURE_RING_SLOT_COUNT];
static bool g_capture_slots_ready = false;
static ULONG g_capture_slot_count = APP_FILEX_CAPTURE_RING_SLOT_COUNT;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/

/* Main thread entry function.  */
void fx_app_thread_entry(ULONG thread_input);

/* USER CODE BEGIN PFP */
static void AppFileX_StateMachine_Initialize(
		AppFileX_StateMachineContext *context_ptr);
static void AppFileX_StateMachine_EnterError(
		AppFileX_StateMachineContext *context_ptr, AppFileX_State error_state,
		UINT error_code);
static void AppFileX_StateMachine_Step(
		AppFileX_StateMachineContext *context_ptr);
static UINT AppFileX_LockMedia(void);
static void AppFileX_UnlockMedia(void);
static UINT AppFileX_CreateCapturedImagesDirectoryLocked(void);
static UINT AppFileX_SeedCapturedImageFallbackIndexLocked(ULONG *next_index_ptr,
		const CHAR *file_extension_ptr);
static UINT AppFileX_PrepareCaptureSlotsLocked(void);
static UINT AppFileX_CloseCaptureSlotsLocked(void);
static bool AppFileX_ParseCaptureSlotName(const CHAR *file_name_ptr,
		ULONG *slot_index_ptr, AppFileX_CaptureFileFormat *format_ptr);
static bool AppFileX_StampCapturedImageTimestampLocked(
		const CHAR *file_name_ptr, const CHAR *timestamp_string_ptr);
static UINT AppFileX_QueueCapturedImageTimestampLocked(
		const CHAR *file_name_ptr);
static bool AppFileX_ProcessQueuedCapturedImageTimestampsLocked(void);
static bool AppFileX_ShouldFlushCaptureMediaLocked(void);
static void AppFileX_FlashCaptureBlue(uint32_t hold_ms);
static void AppFileX_LogTimingStep(const CHAR *label, ULONG start_tick);
static const CHAR *AppFileX_GetCaptureFileExtension(
		AppFileX_CaptureFileFormat format);
static ULONG AppFileX_MillisecondsToTicks(uint32_t timeout_ms);
static void AppFileX_LogStateMessage(const char *message);
/* USER CODE END PFP */

/**
  * @brief  Application FileX Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
*/
UINT MX_FileX_Init(VOID *memory_ptr)
{
  UINT ret = FX_SUCCESS;
  TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*)memory_ptr;
  VOID *pointer;

/* USER CODE BEGIN MX_FileX_MEM_POOL */

/* USER CODE END MX_FileX_MEM_POOL */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/*Allocate memory for the main thread's stack*/
  ret = tx_byte_allocate(byte_pool, &pointer, FX_APP_THREAD_STACK_SIZE, TX_NO_WAIT);

/* Check FX_APP_THREAD_STACK_SIZE allocation*/
  if (ret != FX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

/* Create the main thread.  */
  ret = tx_thread_create(&fx_app_thread, FX_APP_THREAD_NAME, fx_app_thread_entry, 0, pointer, FX_APP_THREAD_STACK_SIZE,
                         FX_APP_THREAD_PRIO, FX_APP_PREEMPTION_THRESHOLD, FX_APP_THREAD_TIME_SLICE, FX_APP_THREAD_AUTO_START);

/* Check main thread creation */
  if (ret != FX_SUCCESS)
  {
    return TX_THREAD_ERROR;
  }

/* USER CODE BEGIN MX_FileX_Init */
	g_filex_byte_pool_ptr = byte_pool;

	/* Allocate FileX media cache buffer from the same byte pool used for thread stacks. */
	ret = tx_byte_allocate(byte_pool, (VOID**) &g_filex_media_cache_buffer,
	FILEX_MEDIA_CACHE_BUFFER_SIZE,
	TX_NO_WAIT);
	if (ret != TX_SUCCESS) {
		return TX_POOL_ERROR;
	}

	ret = tx_mutex_create(&g_filex_media_mutex, "filex_media_mutex",
	TX_INHERIT);
	if (ret != TX_SUCCESS) {
		return TX_MUTEX_ERROR;
	}

	g_filex_media_ready = false;
/* USER CODE END MX_FileX_Init */

/* Initialize FileX.  */
  fx_system_initialize();

/* USER CODE BEGIN MX_FileX_Init 1*/

/* USER CODE END MX_FileX_Init 1*/

  return ret;
}

/**
 * @brief  Main thread entry.
 * @param thread_input: ULONG user argument used by the thread entry
 * @retval none
*/
void fx_app_thread_entry(ULONG thread_input)
 {

/* USER CODE BEGIN fx_app_thread_entry 0*/
	(void) DebugConsole_WriteString("[FILEX] thread alive\r\n");
/* USER CODE END fx_app_thread_entry 0*/

/* USER CODE BEGIN fx_app_thread_entry 1*/
	static AppFileX_StateMachineContext app_filex_context;
	(void) thread_input;

	AppFileX_StateMachine_Initialize(&app_filex_context);

	while (1) {
		AppFileX_StateMachine_Step(&app_filex_context);

		/* Always yield so we do not hog the CPU. */
		tx_thread_sleep(1U);
	}

/* USER CODE END fx_app_thread_entry 1*/
  }

/* USER CODE BEGIN 1 */

/*==============================================================================
 * Function: AppFileX_StateMachine_Initialize
 *
 * Purpose:
 *   Initialize the FileX state machine context and reset runtime state so the
 *   module can run from the beginning.
 *
 * Parameters:
 *   context_ptr - Pointer to the state machine context.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Resets context fields and sets the first state to CMD0.
 *
 * Preconditions:
 *   context_ptr is not NULL.
 *
 * Concurrency:
 *   FileX thread only.
 *==============================================================================*/
static void AppFileX_StateMachine_Initialize(
		AppFileX_StateMachineContext *context_ptr) {
	ULONG now_tick = 0U;

	if (context_ptr == NULL) {
		return;
	}

	now_tick = tx_time_get();

	(void) memset(context_ptr, 0, sizeof(*context_ptr));

	context_ptr->state = APP_FILEX_STATE_SD_SEND_CMD0;

	context_ptr->filex_status = FX_SUCCESS;
	context_ptr->threadx_status = TX_SUCCESS;

	context_ptr->sd_cmd0_r1 = 0xFFU;
	context_ptr->sd_cmd8_r1 = 0xFFU;
	context_ptr->sd_cmd58_r1 = 0xFFU;
	context_ptr->sd_acmd41_r1 = 0xFFU;

	context_ptr->partition_start_lba = 0U;
	context_ptr->partition_sector_count = 0U;

	context_ptr->log_service_is_initialized = 0U;
	context_ptr->filex_media_is_open = 0U;

	context_ptr->last_progress_tick = now_tick;
	context_ptr->state_entry_tick = now_tick;

	context_ptr->last_error_state = APP_FILEX_STATE_UNINITIALIZED;
	context_ptr->last_error_code = 0U;
	g_filex_media_ready = false;

	/* Keep driver context visibility consistent for debugger. */
	g_sd_filex_driver_context.is_initialized = 0U;
}

/*==============================================================================
 * Function: AppFileX_LogStateMessage
 *
 * Purpose:
 *   Emit a short FileX state breadcrumb when verbose state logging is enabled.
 *==============================================================================*/
static void AppFileX_LogStateMessage(const char *message) {
#if APP_FILEX_ENABLE_STATE_BREADCRUMBS
	if (message != NULL) {
		(void) DebugConsole_WriteString(message);
	}
#else
	(void) message;
#endif
}

/*==============================================================================
 * Function: AppFileX_LogErrorMessage
 *
 * Purpose:
 *   Emit a concise FileX error line even when the verbose printf path is off.
 *==============================================================================*/
static void AppFileX_LogErrorMessage(AppFileX_State error_state,
		UINT error_code) {
	char error_line[96];
	const int written = snprintf(error_line, sizeof(error_line),
			"[FILEX][ERROR] state=%lu code=%lu\r\n",
			(unsigned long) error_state, (unsigned long) error_code);

	if ((written > 0) && ((size_t) written < sizeof(error_line))) {
		(void) DebugConsole_WriteString(error_line);
	}
}

/*==============================================================================
 * Function: AppFileX_StateMachine_EnterError
 *
 * Purpose:
 *   Transition into ERROR state and record the error cause for diagnostics.
 *
 * Parameters:
 *   context_ptr - Pointer to the state machine context.
 *   error_state - State that encountered the error.
 *   error_code  - Error code, typically FileX/ThreadX status or an SPI R1 code.
 *
 * Returns:
 *   None.
 *
 * Side Effects:
 *   Sets state to ERROR and records diagnostic fields.
 *
 * Concurrency:
 *   FileX thread only.
 *==============================================================================*/
static void AppFileX_StateMachine_EnterError(
		AppFileX_StateMachineContext *context_ptr, AppFileX_State error_state,
		UINT error_code) {
	if (context_ptr == NULL) {
		return;
	}

	context_ptr->last_error_state = error_state;
	context_ptr->last_error_code = error_code;
	context_ptr->state = APP_FILEX_STATE_ERROR;
	context_ptr->state_entry_tick = tx_time_get();

	AppFileX_LogErrorMessage(error_state, error_code);
}

/*==============================================================================
 * Function: AppFileX_StateMachine_Step
 *
 * Purpose:
 *   Execute one state machine step. Each step performs a coarse action and
 *   either advances to the next state or enters ERROR.
 *
 * Parameters:
 *   context_ptr - Pointer to the state machine context.
 *
 * Returns:
 *   None.
 *
 * Notes:
 *   ERROR behaviour:
 *     - Blink red LED 1s on, 1s off,
 *     - then restart from the beginning (CMD0).
 *
 *   Cleanup strategy:
 *     - If media was opened, flush and close it before restart.
 *     - This is good enough for bringup and fault recovery during development.
 *==============================================================================*/
static void AppFileX_StateMachine_Step(
		AppFileX_StateMachineContext *context_ptr) {
	if (context_ptr == NULL) {
		return;
	}

	switch (context_ptr->state) {
	case APP_FILEX_STATE_SD_SEND_CMD0: {
		AppFileX_LogStateMessage("[FILEX][STATE] CMD0\r\n");
		/**
		 * Send each of the required SPI SD commands to initialize the SD card
		 * ie. CMD0->CMD8->ACMD41->CMD58
		 * And move onto the next subsequent state once the correct
		 * responses are received
		 */
		context_ptr->sd_cmd0_r1 = SPI_SendCMD0_GetR1();
		if (context_ptr->sd_cmd0_r1 != 0x01U) {
			const ULONG retry_timeout_ticks = AppFileX_MillisecondsToTicks(
					FILEX_SD_INIT_TIMEOUT_MS);

			if ((tx_time_get() - context_ptr->state_entry_tick)
					< retry_timeout_ticks) {
				DelayMilliseconds_ThreadX(FILEX_SD_INIT_RETRY_DELAY_MS);
				break;
			}

			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_SEND_CMD0,
					(UINT) context_ptr->sd_cmd0_r1);
			break;
		}

		AppFileX_LogStateMessage("[FILEX][STATE] CMD0 OK\r\n");
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_SEND_CMD8;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_SEND_CMD8: {
		AppFileX_LogStateMessage("[FILEX][STATE] CMD8\r\n");
		context_ptr->sd_cmd8_r1 = SPI_SendCMD8_ReadR7(context_ptr->r7_response);
		if ((context_ptr->sd_cmd8_r1 != 0x01U)
				&& (context_ptr->sd_cmd8_r1 != 0x05U)) {
			const ULONG retry_timeout_ticks = AppFileX_MillisecondsToTicks(
					FILEX_SD_INIT_TIMEOUT_MS);

			if ((tx_time_get() - context_ptr->state_entry_tick)
					< retry_timeout_ticks) {
				DelayMilliseconds_ThreadX(FILEX_SD_INIT_RETRY_DELAY_MS);
				break;
			}

			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_SEND_CMD8,
					(UINT) context_ptr->sd_cmd8_r1);
			break;
		}

		AppFileX_LogStateMessage("[FILEX][STATE] CMD8 OK\r\n");
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_WAIT_READY_ACMD41;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_WAIT_READY_ACMD41: {
		AppFileX_LogStateMessage("[FILEX][STATE] ACMD41\r\n");
		/* Blocking helper, acceptable for now. Can be made incremental later. */
		context_ptr->sd_acmd41_r1 = SPI_SendACMD41_UntilReady(NULL);

		if (context_ptr->sd_acmd41_r1 != 0x00U) {
			const ULONG retry_timeout_ticks = AppFileX_MillisecondsToTicks(
					FILEX_SD_INIT_TIMEOUT_MS);

			if ((tx_time_get() - context_ptr->state_entry_tick)
					< retry_timeout_ticks) {
				DelayMilliseconds_ThreadX(FILEX_SD_INIT_RETRY_DELAY_MS);
				break;
			}

			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_WAIT_READY_ACMD41,
					(UINT) context_ptr->sd_acmd41_r1);
			break;
		}

		AppFileX_LogStateMessage("[FILEX][STATE] ACMD41 OK\r\n");
		SPI_SD_SetHighSpeed();
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_READ_OCR_CMD58;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_READ_OCR_CMD58: {
		AppFileX_LogStateMessage("[FILEX][STATE] CMD58\r\n");
		context_ptr->sd_cmd58_r1 = SPI_SendCMD58_ReadOCR(
				context_ptr->ocr_response);
		DebugConsole_Printf(
				"All SPI commands sent and received successfully.\r\n");
		AppFileX_LogStateMessage("[FILEX][STATE] CMD58 OK\r\n");
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_READ_PARTITION0_INFO;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_READ_PARTITION0_INFO: {
		AppFileX_LogStateMessage("[FILEX][STATE] PARTITION\r\n");
		const ULONG retry_timeout_ticks = AppFileX_MillisecondsToTicks(
				FILEX_PARTITION_READ_RETRY_TIMEOUT_MS);
		const UINT partition_status = (UINT) SPI_ReadPartition0Info(
				&context_ptr->partition_start_lba,
				&context_ptr->partition_sector_count);
		/**
		 * Read the info on partition 0 on the sd card once we have sent all the commands
		 * and received the correct responses in return.
		 */
		if (partition_status != 0x00U) {
			if ((tx_time_get() - context_ptr->state_entry_tick)
					< retry_timeout_ticks) {
				/* Some cards need extra time after OCR before LBA0 becomes readable. */
				DelayMilliseconds_ThreadX(FILEX_PARTITION_READ_RETRY_DELAY_MS);
				break;
			}

			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_READ_PARTITION0_INFO, partition_status);
			break;
		}

		AppFileX_LogStateMessage("[FILEX][STATE] PARTITION OK\r\n");
		g_sd_filex_driver_context.partition_start_lba =
				context_ptr->partition_start_lba;
		g_sd_filex_driver_context.partition_sector_count =
				context_ptr->partition_sector_count;
		g_sd_filex_driver_context.is_initialized = 1U;

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_FILEX_MEDIA_OPEN;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_FILEX_MEDIA_OPEN: {
		const ULONG media_open_tick = tx_time_get();
		AppFileX_LogStateMessage("[FILEX][STATE] MEDIA OPEN\r\n");
		/**
		 * Open the SD card for writing
		 */
		context_ptr->filex_status = fx_media_open(&g_sd_fx_media,
				"SD_SPI_MEDIA", SPI_FileX_SdSpiMediaDriver,
				&g_sd_filex_driver_context, g_filex_media_cache_buffer,
				FILEX_MEDIA_CACHE_BUFFER_SIZE);

		if (context_ptr->filex_status != FX_SUCCESS) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_FILEX_MEDIA_OPEN,
					context_ptr->filex_status);
			break;
		}

		context_ptr->filex_media_is_open = 1U;
		AppFileX_LogStateMessage("[FILEX][STATE] MEDIA OPEN OK\r\n");
		AppFileX_LogTimingStep("media_open", media_open_tick);

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_LOG_SERVICE_INITIALIZE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_LOG_SERVICE_INITIALIZE: {
		const ULONG log_service_tick = tx_time_get();
		AppFileX_LogStateMessage("[FILEX][STATE] LOG SERVICE\r\n");
		/**
		 * Start the debug log service
		 */
		if (g_filex_byte_pool_ptr == NULL) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_LOG_SERVICE_INITIALIZE,
					(UINT) TX_POOL_ERROR);
			break;
		}

		context_ptr->threadx_status = SdDebugLogService_Initialize(
				g_filex_byte_pool_ptr, &g_sd_fx_media);
		if (context_ptr->threadx_status != TX_SUCCESS) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_LOG_SERVICE_INITIALIZE,
					context_ptr->threadx_status);
			break;
		}

		(void) SdDebugLogService_EnqueueLine("debug log service initialized");
		DebugConsole_Printf(
				"Initialized debug log service in FileX thread.\r\n");

		context_ptr->log_service_is_initialized = 1U;
		AppFileX_LogStateMessage("[FILEX][STATE] LOG SERVICE OK\r\n");
		AppFileX_LogTimingStep("log_service_init", log_service_tick);

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE: {
		const ULONG test_file_tick = tx_time_get();
		AppFileX_LogStateMessage("[FILEX][STATE] TEST FILE\r\n");
		/**
		 * Delete and rewrite a test file on the sd card to ensure proper operation
		 */
		(void) fx_file_delete(&g_sd_fx_media, "test.txt");

		context_ptr->filex_status = fx_file_create(&g_sd_fx_media, "test.txt");
		if ((context_ptr->filex_status != FX_SUCCESS)
				&& (context_ptr->filex_status != FX_ALREADY_CREATED)) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE,
					context_ptr->filex_status);
			break;
		}

		context_ptr->filex_status = fx_file_open(&g_sd_fx_media, &g_sd_fx_file,
				"test.txt",
				FX_OPEN_FOR_WRITE);
		if (context_ptr->filex_status != FX_SUCCESS) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE,
					context_ptr->filex_status);
			break;
		}

		{
			static const CHAR message[] =
					"Hello from STM32N6 + ThreadX + FileX\r\n";

			context_ptr->filex_status = fx_file_write(&g_sd_fx_file,
					(VOID*) message, (ULONG) (sizeof(message) - 1U));
			if (context_ptr->filex_status != FX_SUCCESS) {
				(void) fx_file_close(&g_sd_fx_file);
				AppFileX_StateMachine_EnterError(context_ptr,
						APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE,
						context_ptr->filex_status);
				break;
			}
		}

		(void) fx_file_close(&g_sd_fx_file);
		(void) fx_media_flush(&g_sd_fx_media);

		DebugConsole_Printf(
				"Successfully wrote test.txt to root of SD card.\r\n");
		AppFileX_LogStateMessage("[FILEX][STATE] TEST FILE OK\r\n");
		AppFileX_LogTimingStep("test_file_cycle", test_file_tick);

		context_ptr->last_progress_tick = tx_time_get();
		DebugConsole_Printf(
				"Now listening to debug messages and writing to .log files.\r\n");
		context_ptr->state = APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE: {
		const ULONG capture_dir_tick = tx_time_get();
		AppFileX_LogStateMessage("[FILEX][STATE] READY CHECK\r\n");
		/**
		 * Create the capture directory. The camera thread opens capture files
		 * on demand, so we avoid pre-creating a placeholder file here.
		 */
		context_ptr->threadx_status = AppFileX_LockMedia();
		if (context_ptr->threadx_status != TX_SUCCESS) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE,
					context_ptr->threadx_status);
			break;
		}

		context_ptr->filex_status = AppFileX_CreateCapturedImagesDirectoryLocked();
		AppFileX_LogTimingStep("capture_directory_create",
				capture_dir_tick);
		if ((context_ptr->filex_status == FX_SUCCESS)
				|| (context_ptr->filex_status == FX_ALREADY_CREATED)) {
			context_ptr->filex_status = fx_directory_default_set(&g_sd_fx_media,
					CAPTURED_IMAGES_DIRECTORY_NAME);
			if (context_ptr->filex_status != FX_SUCCESS) {
				DebugConsole_Printf(
						"[FILEX][CAPTURE] Failed to select capture directory for capture priming, status=%lu.\r\n",
						(unsigned long) context_ptr->filex_status);
			}
		}
		(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
		AppFileX_UnlockMedia();

		if ((context_ptr->filex_status == FX_SUCCESS)
				|| (context_ptr->filex_status == FX_ALREADY_CREATED)) {
#if APP_FILEX_PREPARE_CAPTURE_SLOTS
			const ULONG capture_slot_prep_tick = tx_time_get();
			context_ptr->filex_status = AppFileX_PrepareCaptureSlotsLocked();
			AppFileX_LogTimingStep("capture_slot_prep",
					capture_slot_prep_tick);
			if (context_ptr->filex_status != TX_SUCCESS) {
				DebugConsole_Printf(
						"[FILEX][CAPTURE] Capture slot preparation failed, status=%lu; falling back to on-demand opens.\r\n",
						(unsigned long) context_ptr->filex_status);
				context_ptr->filex_status = FX_SUCCESS;
			}
#else
			/* Keep one timestamped file per capture; do not pre-create the
			 * four rotating slot files used by the earlier bring-up path. */
			g_capture_slots_ready = false;
			g_capture_slot_next_index = 0U;
#endif
		}

		if ((context_ptr->filex_status != FX_SUCCESS)
				&& (context_ptr->filex_status != FX_ALREADY_CREATED)) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE,
					context_ptr->filex_status);
			break;
		}

		g_filex_media_ready = true;
		AppStorage_NotifyMediaReady();
		(void) DebugConsole_WriteString("[FILEX] media ready\r\n");

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_RUNNING;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_RUNNING: {
		/* Drain a bounded number of messages each cycle so we do not starve other work. */

		if (context_ptr->log_service_is_initialized != 0U) {
			SdDebugLogService_ServiceQueue(32U);
		}
		break;
	}

	case APP_FILEX_STATE_ERROR: {
		/* Best effort drain, then blink red 1s on, 1s off, then restart state machine. */
		AppFileX_LogStateMessage("[FILEX][STATE] ERROR\r\n");
		if (context_ptr->log_service_is_initialized != 0U) {
			SdDebugLogService_ServiceQueue(64U);
		}

		// print a message about the error state
		DebugConsole_Printf("FileX App thread ran into an error.\r\n");

		(void) DebugLed_BlinkRedBlocking(1000U, 1000U, 1U);

		/* Best effort cleanup before restart. */
		if (context_ptr->filex_media_is_open != 0U) {
			if (AppFileX_LockMedia() == TX_SUCCESS) {
				AppFileX_CloseCaptureSlotsLocked();
				(void) fx_media_flush(&g_sd_fx_media);
				(void) fx_media_close(&g_sd_fx_media);
				AppFileX_UnlockMedia();
			}
			context_ptr->filex_media_is_open = 0U;
		}

		g_sd_filex_driver_context.is_initialized = 0U;
		g_filex_media_ready = false;

		/* Restart the module from the beginning. */
		AppFileX_StateMachine_Initialize(context_ptr);
		break;
	}

	case APP_FILEX_STATE_UNINITIALIZED:
	default: {
		AppFileX_StateMachine_EnterError(context_ptr, context_ptr->state,
				(UINT) 0U);
		break;
	}
	}
}

/*==============================================================================
 * Function: AppFileX_IsMediaReady
 *
 * Purpose:
 *   Report whether the SD card media is mounted and the capture directory is
 *   ready for image storage.
 *==============================================================================*/
bool AppFileX_IsMediaReady(void) {
	return g_filex_media_ready;
}

/*==============================================================================
 * Function: AppFileX_GetCapturedImagesDirectoryName
 *
 * Purpose:
 *   Expose the capture directory name so other modules can scan or clean it
 *   without duplicating the storage layout string.
 *==============================================================================*/
const CHAR *AppFileX_GetCapturedImagesDirectoryName(void) {
	return CAPTURED_IMAGES_DIRECTORY_NAME;
}

/*==============================================================================
 * Function: AppFileX_GetMediaHandle
 *
 * Purpose:
 *   Expose the mounted FileX media handle so other boot-time loaders can read
 *   assets from the SD card without duplicating the mount state.
 *==============================================================================*/
FX_MEDIA *AppFileX_GetMediaHandle(void) {
	if (!g_filex_media_ready) {
		return NULL;
	}

	return &g_sd_fx_media;
}

/*==============================================================================
 * Function: AppFileX_AcquireMediaLock
 *
 * Purpose:
 *   Export the media mutex for other boot-time readers that need serialized
 *   access to the mounted SD card.
 *==============================================================================*/
UINT AppFileX_AcquireMediaLock(void) {
	return AppFileX_LockMedia();
}

/*==============================================================================
 * Function: AppFileX_ReleaseMediaLock
 *
 * Purpose:
 *   Release the exported media mutex after a serialized SD card access.
 *==============================================================================*/
void AppFileX_ReleaseMediaLock(void) {
	AppFileX_UnlockMedia();
}

/*==============================================================================*/
UINT AppFileX_PrepareCaptureSlots(void) {
	UINT status = TX_SUCCESS;

	if (!g_filex_media_ready) {
		return FX_MEDIA_NOT_OPEN;
	}

	status = AppFileX_LockMedia();
	if (status != TX_SUCCESS) {
		return status;
	}

	status = AppFileX_PrepareCaptureSlotsLocked();
	AppFileX_UnlockMedia();

	return status;
}

/*==============================================================================*/
static const CHAR *AppFileX_GetCaptureFileExtension(
		AppFileX_CaptureFileFormat format) {
	switch (format) {
	case APP_FILEX_CAPTURE_FORMAT_YUV422:
		return "yuv422";
	case APP_FILEX_CAPTURE_FORMAT_RAW16:
		return "raw16";
	default:
		return NULL;
	}
}

/*==============================================================================*/
static bool AppFileX_ParseCaptureSlotName(const CHAR *file_name_ptr,
		ULONG *slot_index_ptr, AppFileX_CaptureFileFormat *format_ptr) {
	const char prefix[] = "capture_slot_";
	const CHAR *extension_ptr = NULL;
	ULONG slot_index = 0U;
	size_t prefix_length = sizeof(prefix) - 1U;
	size_t digit_offset = 0U;

	if ((file_name_ptr == NULL) || (slot_index_ptr == NULL)
			|| (format_ptr == NULL)) {
		return false;
	}

	if (strncmp(file_name_ptr, prefix, prefix_length) != 0) {
		return false;
	}

	if ((file_name_ptr[prefix_length] < '0')
			|| (file_name_ptr[prefix_length] > '9')
			|| (file_name_ptr[prefix_length + 1U] < '0')
			|| (file_name_ptr[prefix_length + 1U] > '9')) {
		return false;
	}

	slot_index = (ULONG) ((file_name_ptr[prefix_length] - '0') * 10U
			+ (file_name_ptr[prefix_length + 1U] - '0'));
	digit_offset = prefix_length + 2U;
	if (file_name_ptr[digit_offset] != '.') {
		return false;
	}

	extension_ptr = &file_name_ptr[digit_offset + 1U];
	if (strcmp(extension_ptr, "yuv422") == 0) {
		*format_ptr = APP_FILEX_CAPTURE_FORMAT_YUV422;
	} else if (strcmp(extension_ptr, "raw16") == 0) {
		*format_ptr = APP_FILEX_CAPTURE_FORMAT_RAW16;
	} else {
		return false;
	}

	if (slot_index >= g_capture_slot_count) {
		return false;
	}

	*slot_index_ptr = slot_index;
	return true;
}

/*==============================================================================*/
static bool AppFileX_StampCapturedImageTimestampLocked(
		const CHAR *file_name_ptr, const CHAR *timestamp_string_ptr) {
	char timestamp[32] = { 0 };
	unsigned int year = 0U;
	unsigned int month = 0U;
	unsigned int day = 0U;
	unsigned int hour = 0U;
	unsigned int minute = 0U;
	unsigned int second = 0U;
	int parsed_count = 0;

	if (file_name_ptr == NULL) {
		return false;
	}

	if ((timestamp_string_ptr == NULL) || (timestamp_string_ptr[0] == '\0')) {
		/*
		 * Keep the FileX clock aligned with the DS3231 so any later FAT writes and
		 * Windows directory stamps stay consistent with the actual capture time.
		 */
		if (!App_Clock_GetCurrentTimestamp(timestamp, sizeof(timestamp))) {
			return false;
		}
		timestamp_string_ptr = timestamp;
	}

	parsed_count = sscanf(timestamp_string_ptr,
			"%4u-%2u-%2u_%2u-%2u-%2u", &year, &month,
			&day, &hour, &minute, &second);
	if (parsed_count != 6) {
		return false;
	}

	(void) fx_system_date_set((UINT) year, (UINT) month, (UINT) day);
	(void) fx_system_time_set((UINT) hour, (UINT) minute, (UINT) second);

	/* Keep the directory context explicit so we can measure whether the stamp
	 * update is slowed down by path switching or by the FAT metadata write. */
	const ULONG dir_set_start_tick = tx_time_get();
	UINT fx_status = FX_SUCCESS;
	fx_status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (fx_status != FX_SUCCESS) {
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] stamp-dir=%lu ms path=%s status=%lu\r\n",
				(unsigned long)(((tx_time_get() - dir_set_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				file_name_ptr, (unsigned long) fx_status);
		return false;
	}

	/* Navigate to daily subfolder if filename contains a date prefix */
	{
		const char *slash = strchr(file_name_ptr, '/');
		if (slash != NULL) {
			char date_folder[16] = { 0 };
			const size_t folder_len = (size_t)(slash - file_name_ptr);
			if (folder_len < sizeof(date_folder)) {
				memcpy(date_folder, file_name_ptr, folder_len);
				date_folder[folder_len] = '\0';
				fx_status = fx_directory_default_set(&g_sd_fx_media, date_folder);
				if (fx_status != FX_SUCCESS) {
					return false;
				}
				file_name_ptr = slash + 1;
			}
		}
	}

	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] stamp-dir=%lu ms path=%s\r\n",
			(unsigned long)(((tx_time_get() - dir_set_start_tick) *
				1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
			file_name_ptr, (unsigned long) fx_status);

	const ULONG file_time_start_tick = tx_time_get();
	fx_status = fx_file_date_time_set(&g_sd_fx_media, (CHAR*) file_name_ptr,
			(UINT) year, (UINT) month, (UINT) day, (UINT) hour,
			(UINT) minute, (UINT) second);
	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] stamp-write-time=%lu ms path=%s status=%lu\r\n",
			(unsigned long)(((tx_time_get() - file_time_start_tick) *
				1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
			file_name_ptr, (unsigned long) fx_status);
	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	if (fx_status != FX_SUCCESS) {
		return false;
	}

	return true;
}

/*==============================================================================*/
static UINT AppFileX_QueueCapturedImageTimestampLocked(
		const CHAR *file_name_ptr) {
	CHAR timestamp[32] = { 0 };
	AppFileX_CaptureTimestampEntry *entry_ptr = NULL;
	int written = 0;

	if (file_name_ptr == NULL) {
		return TX_PTR_ERROR;
	}

	if (!App_Clock_GetCurrentTimestamp(timestamp, sizeof(timestamp))) {
		return TX_NOT_AVAILABLE;
	}

	if (g_capture_timestamp_queue_count
			>= APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH) {
		DebugConsole_WriteString(
				"[FILEX][CAPTURE] Timestamp queue full; dropping oldest pending stamp.\r\n");
		g_capture_timestamp_queue_head =
				(g_capture_timestamp_queue_head + 1U)
				% APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH;
		g_capture_timestamp_queue_count--;
	}

	entry_ptr = &g_capture_timestamp_queue[g_capture_timestamp_queue_tail];
	(void) memset(entry_ptr, 0, sizeof(*entry_ptr));

	written = snprintf(entry_ptr->file_name, sizeof(entry_ptr->file_name), "%s",
			file_name_ptr);
	if ((written < 0) || ((size_t) written >= sizeof(entry_ptr->file_name))) {
		return TX_SIZE_ERROR;
	}

	written = snprintf(entry_ptr->timestamp, sizeof(entry_ptr->timestamp), "%s",
			timestamp);
	if ((written < 0) || ((size_t) written >= sizeof(entry_ptr->timestamp))) {
		return TX_SIZE_ERROR;
	}

	entry_ptr->valid = true;
	g_capture_timestamp_queue_tail =
			(g_capture_timestamp_queue_tail + 1U)
			% APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH;
	g_capture_timestamp_queue_count++;
	g_capture_media_timestamp_pending = true;
	return TX_SUCCESS;
}

/*==============================================================================*/
static bool AppFileX_ProcessQueuedCapturedImageTimestampsLocked(void) {
	bool stamped_any = false;

	while (g_capture_timestamp_queue_count > 0U) {
		AppFileX_CaptureTimestampEntry *entry_ptr =
				&g_capture_timestamp_queue[g_capture_timestamp_queue_head];

		if (!entry_ptr->valid) {
			g_capture_timestamp_queue_head =
					(g_capture_timestamp_queue_head + 1U)
					% APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH;
			g_capture_timestamp_queue_count--;
			continue;
		}

		if (!AppFileX_StampCapturedImageTimestampLocked(entry_ptr->file_name,
				entry_ptr->timestamp)) {
			return stamped_any;
		}

		(void) memset(entry_ptr, 0, sizeof(*entry_ptr));
		g_capture_timestamp_queue_head =
				(g_capture_timestamp_queue_head + 1U)
				% APP_FILEX_CAPTURE_TIMESTAMP_QUEUE_LENGTH;
		g_capture_timestamp_queue_count--;
		stamped_any = true;
	}

	if (g_capture_timestamp_queue_count == 0U) {
		g_capture_media_timestamp_pending = false;
	}

	return stamped_any;
}

/*==============================================================================*/
static bool AppFileX_ShouldFlushCaptureMediaLocked(void) {
	const ULONG now_tick = tx_time_get();
	const ULONG interval_ticks = AppFileX_MillisecondsToTicks(
			APP_FILEX_CAPTURE_FLUSH_INTERVAL_MS);

	if (g_capture_media_last_flush_tick == 0U) {
		g_capture_media_last_flush_tick = now_tick;
		return false;
	}

	if ((ULONG) (now_tick - g_capture_media_last_flush_tick) < interval_ticks) {
		return false;
	}

	g_capture_media_last_flush_tick = now_tick;
	return true;
}

UINT AppFileX_ServiceCaptureMediaFlush(void) {
	UINT flush_status = FX_SUCCESS;
	const ULONG flush_start_tick = tx_time_get();
	bool stamped_any = false;
	bool need_flush = false;

	if (!g_filex_media_ready && !g_capture_media_flush_pending
			&& !g_capture_media_timestamp_pending) {
		return FX_SUCCESS;
	}

	if (AppFileX_LockMedia() != TX_SUCCESS) {
		return TX_MUTEX_ERROR;
	}

	if (!g_filex_media_ready) {
		AppFileX_UnlockMedia();
		return FX_SUCCESS;
	}

	stamped_any = AppFileX_ProcessQueuedCapturedImageTimestampsLocked();
	need_flush = stamped_any;

	if (g_capture_media_flush_pending
			&& AppFileX_ShouldFlushCaptureMediaLocked()) {
		need_flush = true;
	}

	if (!need_flush) {
		AppFileX_UnlockMedia();
		return FX_SUCCESS;
	}

	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] flush-start pending=%u stamp=%u\r\n",
			(unsigned int) (g_capture_media_flush_pending ? 1U : 0U),
			(unsigned int) (stamped_any ? 1U : 0U));
	flush_status = fx_media_flush(&g_sd_fx_media);
	if (flush_status == FX_SUCCESS) {
		g_capture_media_flush_pending = false;
		g_capture_media_last_flush_tick = 0U;
	}

	AppFileX_UnlockMedia();

	if (flush_status == FX_SUCCESS) {
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] flush=%lu ms status=%lu\r\n",
				(unsigned long)(((tx_time_get() - flush_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				(unsigned long) flush_status);
		DebugConsole_WriteString(
				"[FILEX][CAPTURE] Scheduled media flush completed.\r\n");
		AppFileX_FlashCaptureBlue(3000U);
	} else {
		AppFileX_WriteCaptureStatusLine(
				"[FILEX][CAPTURE] media-flush-failed status=%lu; will retry.\r\n",
				(unsigned long) flush_status);
	}

	return flush_status;
}

/*==============================================================================*/
static UINT AppFileX_CloseCaptureSlotsLocked(void) {
	for (ULONG format_index = 0U;
			format_index < APP_FILEX_CAPTURE_FILE_FORMAT_COUNT; format_index++) {
		for (ULONG slot_index = 0U; slot_index < g_capture_slot_count;
				slot_index++) {
			AppFileX_CaptureSlot *slot_ptr =
					&g_capture_slots[format_index][slot_index];

			if (!slot_ptr->open) {
				continue;
			}

			(void) fx_file_close(&slot_ptr->file);
			slot_ptr->open = false;
		}
	}

	g_capture_slots_ready = false;
	g_capture_media_flush_pending = false;
	g_capture_media_last_flush_tick = 0U;
	g_capture_slot_next_index = 0U;
	return TX_SUCCESS;
}

/*==============================================================================*/
static UINT AppFileX_PrepareCaptureSlotsLocked(void) {
	static const AppFileX_CaptureFileFormat capture_formats[] = {
			APP_FILEX_CAPTURE_FORMAT_YUV422, APP_FILEX_CAPTURE_FORMAT_RAW16 };
	static const size_t capture_format_count =
			sizeof(capture_formats) / sizeof(capture_formats[0]);
	UINT status = FX_SUCCESS;

	if (g_capture_slots_ready) {
		return TX_SUCCESS;
	}

	status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (status != FX_SUCCESS) {
		return status;
	}

	(void) AppFileX_CloseCaptureSlotsLocked();

	for (size_t format_index = 0U; format_index < capture_format_count;
			format_index++) {
		const AppFileX_CaptureFileFormat format = capture_formats[format_index];
		const CHAR *extension_ptr = AppFileX_GetCaptureFileExtension(format);

		if ((extension_ptr == NULL) || (extension_ptr[0] == '\0')) {
			status = FX_INVALID_NAME;
			break;
		}

		for (ULONG slot_index = 0U; slot_index < g_capture_slot_count;
				slot_index++) {
			AppFileX_CaptureSlot *slot_ptr =
					&g_capture_slots[format][slot_index];
			FX_FILE *capture_file_ptr = &slot_ptr->file;
			bool file_open = false;

			(void) memset(slot_ptr, 0, sizeof(*slot_ptr));
			(void) snprintf(slot_ptr->file_name, sizeof(slot_ptr->file_name),
					"capture_slot_%02lu.%s", (unsigned long) slot_index,
					extension_ptr);

			status = fx_file_create(&g_sd_fx_media, slot_ptr->file_name);
			if ((status != FX_SUCCESS) && (status != FX_ALREADY_CREATED)) {
				goto prepare_fail;
			}

			status = fx_file_open(&g_sd_fx_media, capture_file_ptr,
					slot_ptr->file_name, FX_OPEN_FOR_WRITE);
			if (status != FX_SUCCESS) {
				goto prepare_fail;
			}

			file_open = true;

			if (APP_FILEX_PREALLOCATE_CAPTURE_SLOTS != 0U) {
				status = fx_file_allocate(capture_file_ptr,
						CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
				if (status != FX_SUCCESS) {
					DebugConsole_Printf(
							"[FILEX][CAPTURE] Preallocate %s returned status=%lu; continuing with open handle.\r\n",
							slot_ptr->file_name, (unsigned long) status);
					status = FX_SUCCESS;
				}
			}

			status = fx_file_seek(capture_file_ptr, 0U);
			if (status != FX_SUCCESS) {
				if (file_open) {
					(void) fx_file_close(capture_file_ptr);
					file_open = false;
				}
				goto prepare_fail;
			}

			slot_ptr->open = true;
		}
	}

prepare_fail:
	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	if (status != FX_SUCCESS) {
		AppFileX_CloseCaptureSlotsLocked();
		return status;
	}

	g_capture_slots_ready = true;
	g_capture_media_last_flush_tick = tx_time_get();
	g_capture_slot_next_index = 0U;
	return TX_SUCCESS;
}

/*==============================================================================
 * Function: AppFileX_GetNextCapturedImageName
 *
 * Purpose:
 *   Return the next capture file name.
 *
 *   When rotating capture slots are ready, we hand out the pre-opened
 *   capture_slot_##.<extension> names so the writer can overwrite the slot in
 *   place. That keeps the camera path on the fast open-handle path instead of
 *   forcing a fresh create/open on every frame.
 *
 *   If the slots are not ready yet, we fall back to the timestamped name or
 *   the numbered scan-based fallback used during bring-up.
 *==============================================================================*/
UINT AppFileX_GetNextCapturedImageName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr) {
	UINT tx_status = TX_SUCCESS;
	UINT fx_status = FX_SUCCESS;
	UINT file_attributes = 0U;
	ULONG capture_index = 0U;
	int file_name_chars = 0;
	CHAR rtc_stamp[32] = { 0 };
	int written = 0;
	const bool rtc_ready = App_Clock_GetCaptureTimestamp(rtc_stamp,
			sizeof(rtc_stamp));

	if ((file_name_ptr == NULL) || (file_name_length == 0U)
			|| (file_extension_ptr == NULL) || (file_extension_ptr[0] == '\0')) {
		return FX_PTR_ERROR;
	}

	if (!g_filex_media_ready) {
		return FX_MEDIA_NOT_OPEN;
	}

	tx_status = AppFileX_LockMedia();
	if (tx_status != TX_SUCCESS) {
		return tx_status;
	}

	if (g_capture_slots_ready && (g_capture_slot_count > 0U)) {
		const ULONG slot_index = g_capture_slot_next_index;

		written = snprintf(file_name_ptr, (size_t) file_name_length,
				"capture_slot_%02lu.%s", (unsigned long) slot_index,
				file_extension_ptr);
		if ((written > 0) && ((ULONG) written < file_name_length)) {
			g_capture_slot_next_index = (slot_index + 1U) % g_capture_slot_count;
			AppFileX_UnlockMedia();
			return FX_SUCCESS;
		}

		AppFileX_UnlockMedia();
		return FX_INVALID_NAME;
	}

	if (rtc_ready) {
#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Using RTC timestamp %s for capture name.\r\n",
				rtc_stamp);
#endif
		/* Use hourly subfolders: captured_images/2026-07-11_15/capture_23-23.yuv422
		 * Each folder has ~60 files, so FAT directory scans are fast. */
		{
			/* Extract date and hour: "2026-07-11_15" from "2026-07-11_15-23-23" */
			char hour_folder[14] = { 0 };
			if (strlen(rtc_stamp) >= 13U) {
				memcpy(hour_folder, rtc_stamp, 13U);
				hour_folder[13] = '\0';
			} else {
				strcpy(hour_folder, "unknown");
			}
			/* Time portion starts after the hour prefix */
			const char *time_part = rtc_stamp + 14U;  /* skip "YYYY-MM-DD_HH-" */
			written = snprintf(file_name_ptr, (size_t) file_name_length,
					"%s/capture_%s.%s", hour_folder, time_part, file_extension_ptr);
		}
		AppFileX_UnlockMedia();
		return ((written > 0)
				&& ((ULONG) written < file_name_length)) ? FX_SUCCESS
				: FX_INVALID_NAME;
	}

#if CAMERA_CAPTURE_ENABLE_VERBOSE_DIAGNOSTICS
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] RTC timestamp unavailable; using numbered fallback.\r\n");
#endif

	if (!g_captured_image_fallback_index_seeded) {
		fx_status = AppFileX_SeedCapturedImageFallbackIndexLocked(
				&g_captured_image_fallback_next_index, file_extension_ptr);
		if (fx_status != FX_SUCCESS) {
			AppFileX_UnlockMedia();
			return fx_status;
		}
		g_captured_image_fallback_index_seeded = true;
	}

	for (capture_index = g_captured_image_fallback_next_index;
			capture_index <= CAPTURED_IMAGE_MAX_INDEX; capture_index++) {
		file_name_chars = snprintf(file_name_ptr, (size_t) file_name_length,
				"capture_%04lu.%s", (unsigned long) capture_index,
				file_extension_ptr);
		if ((file_name_chars < 0)
				|| ((ULONG) file_name_chars >= file_name_length)) {
			fx_status = FX_INVALID_NAME;
			break;
		}

		fx_status = fx_file_attributes_read(&g_sd_fx_media, file_name_ptr,
				&file_attributes);
		if (fx_status == FX_NOT_FOUND) {
			g_captured_image_fallback_next_index = capture_index + 1U;
			fx_status = FX_SUCCESS;
			break;
		}

		if (fx_status != FX_SUCCESS) {
			break;
		}
	}

	if ((capture_index > CAPTURED_IMAGE_MAX_INDEX) && (fx_status == FX_SUCCESS)) {
		fx_status = FX_INVALID_NAME;
	}

	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	AppFileX_UnlockMedia();
	return fx_status;
}

/*==============================================================================
 * Function: AppFileX_WriteCapturedImage
 *
 * Purpose:
 *   Write a binary image buffer to /captured_images/<file_name>.
 *==============================================================================*/
UINT AppFileX_WriteCapturedImage(const CHAR *file_name_ptr,
		const VOID *data_ptr, ULONG data_length) {
	CHAR path[CAPTURED_IMAGE_MAX_PATH_LENGTH];
	int path_length = 0;
	UINT tx_status = TX_SUCCESS;
	UINT fx_status = FX_SUCCESS;
	const ULONG save_start_tick = tx_time_get();
	ULONG slot_index = 0U;
	AppFileX_CaptureFileFormat format = APP_FILEX_CAPTURE_FORMAT_YUV422;
	bool flash_save_success = false;

	if ((file_name_ptr == NULL) || (data_ptr == NULL) || (data_length == 0U)) {
		return FX_PTR_ERROR;
	}

	if (!g_filex_media_ready) {
		return FX_MEDIA_NOT_OPEN;
	}

	path_length = snprintf(path, sizeof(path), "%s/%s",
	CAPTURED_IMAGES_DIRECTORY_NAME,
			file_name_ptr);
	if ((path_length < 0) || ((ULONG) path_length >= sizeof(path))) {
		return FX_INVALID_NAME;
	}

	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] save-start path=%s len=%lu slots=%lu\r\n",
			path, (unsigned long) data_length,
			(unsigned long)(g_capture_slots_ready ? 1U : 0U));

	if (g_capture_slots_ready
			&& AppFileX_ParseCaptureSlotName(file_name_ptr, &slot_index,
					&format)) {
		AppFileX_CaptureSlot *slot_ptr = &g_capture_slots[format][slot_index];
		const ULONG lock_wait_start_tick = tx_time_get();

		tx_status = AppFileX_LockMedia();
		if (tx_status == TX_SUCCESS) {
			AppFileX_WriteTimingLine(
					"[FILEX][CAPTURE][TIMING] fast-lock=%lu ms slot=%lu path=%s\r\n",
					(unsigned long)(((tx_time_get() - lock_wait_start_tick) *
						1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
					(unsigned long) slot_index, path);
			if (slot_ptr->open) {
				const ULONG fast_seek_start_tick = tx_time_get();
				fx_status = fx_file_seek(&slot_ptr->file, 0U);
				if (fx_status == FX_SUCCESS) {
					const ULONG fast_write_start_tick = tx_time_get();
					fx_status = fx_file_write(&slot_ptr->file, (VOID*) data_ptr,
							data_length);
					AppFileX_WriteTimingLine(
							"[FILEX][CAPTURE][TIMING] fast-write=%lu ms slot=%lu path=%s status=%lu\r\n",
							(unsigned long)(((tx_time_get() - fast_write_start_tick) *
								1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
							(unsigned long) slot_index, path,
							(unsigned long) fx_status);
					if (fx_status == FX_SUCCESS) {
						/* Reusing a slot must not leave bytes from a longer prior
						 * capture at the end of the new file. */
						fx_status = fx_file_truncate(&slot_ptr->file,
								data_length);
					}
					if (fx_status == FX_SUCCESS) {
#if APP_FILEX_ENABLE_CAPTURE_FILE_TIMESTAMP
						(void) AppFileX_StampCapturedImageTimestampLocked(
								file_name_ptr, NULL);
#else
						UINT stamp_status = AppFileX_QueueCapturedImageTimestampLocked(
								file_name_ptr);
						if (stamp_status != TX_SUCCESS) {
							DebugConsole_Printf(
									"[FILEX][CAPTURE] Failed to queue timestamp for %s, status=%lu.\r\n",
									path, (unsigned long) stamp_status);
						}
#endif
						flash_save_success = true;
						g_capture_media_flush_pending = true;
						if (g_capture_media_last_flush_tick == 0U) {
							g_capture_media_last_flush_tick = tx_time_get();
						}
					}
				}
				AppFileX_WriteTimingLine(
						"[FILEX][CAPTURE][TIMING] fast-seek=%lu ms slot=%lu path=%s status=%lu\r\n",
						(unsigned long)(((tx_time_get() - fast_seek_start_tick) *
							1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
						(unsigned long) slot_index, path,
						(unsigned long) fx_status);
			} else {
				fx_status = FX_NOT_OPEN;
			}
			AppFileX_UnlockMedia();
		} else {
			fx_status = tx_status;
		}

		if (fx_status == FX_SUCCESS) {
			AppFileX_WriteCaptureStatusLine(
					"[FILEX][CAPTURE] write-ok bytes=%lu slot=%lu path=%s flush-pending=1\r\n",
					(unsigned long) data_length, (unsigned long) slot_index,
					path);
			AppFileX_WriteTimingLine(
					"[FILEX][CAPTURE][TIMING] save-total=%lu ms path=%s slot=%lu fast=1\r\n",
					(unsigned long)(((tx_time_get() - save_start_tick) *
						1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
					path, (unsigned long) slot_index);
			goto capture_save_finish;
		}

		DebugConsole_Printf(
				"[FILEX][CAPTURE] Fast capture slot write failed for %s, status=%lu; falling back to open/create path.\r\n",
				path, (unsigned long) fx_status);
	}

	const ULONG fallback_lock_wait_start_tick = tx_time_get();
	tx_status = tx_mutex_get(&g_filex_media_mutex,
			AppFileX_MillisecondsToTicks(APP_FILEX_CAPTURE_LOCK_TIMEOUT_MS));
	if (tx_status != TX_SUCCESS) {
		DebugConsole_Printf(
				"[FILEX][CAPTURE] Capture save waited too long for media lock (status=%lu).\r\n",
				(unsigned long) tx_status);
		return tx_status;
	}
	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] fallback-lock=%lu ms path=%s\r\n",
			(unsigned long)(((tx_time_get() - fallback_lock_wait_start_tick) *
				1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
			path);

	const ULONG fallback_dir_start_tick = tx_time_get();
	fx_status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (fx_status != FX_SUCCESS) {
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] fallback-dir=%lu ms path=%s status=%lu\r\n",
				(unsigned long)(((tx_time_get() - fallback_dir_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				path, (unsigned long) fx_status);
		AppFileX_UnlockMedia();
		return fx_status;
	}

	/* Create daily subfolder if the filename contains a date prefix.
	 * Filename format: "2026-07-11/capture_15-23-23.yuv422" */
	{
		const char *slash = strchr(file_name_ptr, '/');
		if (slash != NULL) {
			char date_folder[16] = { 0 };
			const size_t folder_len = (size_t)(slash - file_name_ptr);
			if (folder_len < sizeof(date_folder)) {
				memcpy(date_folder, file_name_ptr, folder_len);
				date_folder[folder_len] = '\0';

				/* Try to create the daily folder (ignore if already exists) */
				const ULONG create_dir_tick = tx_time_get();
				UINT dir_status = fx_directory_create(&g_sd_fx_media, date_folder);
				AppFileX_WriteTimingLine(
						"[FILEX][CAPTURE][TIMING] create-day-dir=%lu ms dir=%s status=%lu\r\n",
						(unsigned long)(((tx_time_get() - create_dir_tick) *
							1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
						date_folder, (unsigned long) dir_status);

				/* Navigate into the daily folder */
				const ULONG day_dir_tick = tx_time_get();
				dir_status = fx_directory_default_set(&g_sd_fx_media, date_folder);
				AppFileX_WriteTimingLine(
						"[FILEX][CAPTURE][TIMING] day-dir=%lu ms dir=%s status=%lu\r\n",
						(unsigned long)(((tx_time_get() - day_dir_tick) *
							1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
						date_folder, (unsigned long) dir_status);

				if (dir_status != FX_SUCCESS) {
					AppFileX_UnlockMedia();
					return dir_status;
				}

				/* Use just the filename part (after the slash) for the file operations */
				file_name_ptr = slash + 1;
			}
		}
	}

	AppFileX_WriteTimingLine(
			"[FILEX][CAPTURE][TIMING] fallback-dir=%lu ms path=%s\r\n",
			(unsigned long)(((tx_time_get() - fallback_dir_start_tick) *
				1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
			path, (unsigned long) fx_status);

	{
		FX_FILE capture_fx_file;
		UINT file_status = FX_SUCCESS;
		bool capture_file_open = false;
		const ULONG fallback_open_start_tick = tx_time_get();

		fx_status = fx_file_open(&g_sd_fx_media, &capture_fx_file,
				(CHAR*) file_name_ptr, FX_OPEN_FOR_WRITE);
		if (fx_status == FX_NOT_FOUND) {
			const ULONG fallback_create_start_tick = tx_time_get();
			fx_status = fx_file_create(&g_sd_fx_media, (CHAR*) file_name_ptr);
			AppFileX_WriteTimingLine(
					"[FILEX][CAPTURE][TIMING] fallback-create=%lu ms path=%s status=%lu\r\n",
					(unsigned long)(((tx_time_get() - fallback_create_start_tick) *
						1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
					path, (unsigned long) fx_status);
			if ((fx_status == FX_SUCCESS) || (fx_status == FX_ALREADY_CREATED)) {
				const ULONG fallback_reopen_start_tick = tx_time_get();
				fx_status = fx_file_open(&g_sd_fx_media, &capture_fx_file,
						(CHAR*) file_name_ptr, FX_OPEN_FOR_WRITE);
				AppFileX_WriteTimingLine(
						"[FILEX][CAPTURE][TIMING] fallback-reopen=%lu ms path=%s status=%lu\r\n",
						(unsigned long)(((tx_time_get() - fallback_reopen_start_tick) *
							1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
						path, (unsigned long) fx_status);
			}
		}

		if (fx_status != FX_SUCCESS) {
			(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
			AppFileX_UnlockMedia();
			return fx_status;
		}
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] fallback-open=%lu ms path=%s\r\n",
				(unsigned long)(((tx_time_get() - fallback_open_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				path);

		capture_file_open = true;

		const ULONG fallback_seek_start_tick = tx_time_get();
		file_status = fx_file_seek(&capture_fx_file, 0U);
		if (file_status != FX_SUCCESS) {
			(void) fx_file_close(&capture_fx_file);
			(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
			AppFileX_UnlockMedia();
			return file_status;
		}
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] fallback-seek=%lu ms path=%s\r\n",
				(unsigned long)(((tx_time_get() - fallback_seek_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				path);

		const ULONG fallback_write_start_tick = tx_time_get();
		file_status = fx_file_write(&capture_fx_file, (VOID*) data_ptr,
				data_length);
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] fallback-write=%lu ms path=%s status=%lu\r\n",
				(unsigned long)(((tx_time_get() - fallback_write_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				path, (unsigned long) file_status);
		if (file_status == FX_SUCCESS) {
			/* Keep timestamped fallback files just as tightly sized as slots. */
			file_status = fx_file_truncate(&capture_fx_file, data_length);
		}
		if (file_status == FX_SUCCESS) {
#if APP_FILEX_ENABLE_CAPTURE_FILE_TIMESTAMP
			(void) AppFileX_StampCapturedImageTimestampLocked(file_name_ptr,
					NULL);
#else
			UINT stamp_status = AppFileX_QueueCapturedImageTimestampLocked(
					file_name_ptr);
			if (stamp_status != TX_SUCCESS) {
				DebugConsole_Printf(
						"[FILEX][CAPTURE] Failed to queue timestamp for %s, status=%lu.\r\n",
						path, (unsigned long) stamp_status);
			}
#endif
			flash_save_success = true;
			g_capture_media_flush_pending = true;
			if (g_capture_media_last_flush_tick == 0U) {
				g_capture_media_last_flush_tick = tx_time_get();
			}
		}

		if (capture_file_open) {
			const ULONG fallback_close_start_tick = tx_time_get();
			(void) fx_file_close(&capture_fx_file);
			AppFileX_WriteTimingLine(
					"[FILEX][CAPTURE][TIMING] fallback-close=%lu ms path=%s\r\n",
					(unsigned long)(((tx_time_get() - fallback_close_start_tick) *
						1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
					path);
		}

		fx_status = file_status;
	}

capture_save_finish:
	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	AppFileX_UnlockMedia();

	if (flash_save_success) {
		AppFileX_FlashCaptureBlue(500U);
	}

	if (fx_status == FX_SUCCESS) {
		AppFileX_WriteCaptureStatusLine(
				"[FILEX][CAPTURE] save-ok path=/%s bytes=%lu flush-pending=1\r\n",
			path, (unsigned long) data_length);
		AppFileX_WriteTimingLine(
				"[FILEX][CAPTURE][TIMING] save-total=%lu ms path=%s fast=0\r\n",
				(unsigned long)(((tx_time_get() - save_start_tick) *
					1000U) / (ULONG)TX_TIMER_TICKS_PER_SECOND),
				path);
	}

	return fx_status;
}

/*==============================================================================
 * Function: AppFileX_LockMedia
 *
 * Purpose:
 *   Serialize access to the mounted FileX media across the FileX thread and
 *   camera thread.
 *==============================================================================*/
static UINT AppFileX_LockMedia(void) {
	return tx_mutex_get(&g_filex_media_mutex, TX_WAIT_FOREVER);
}

/*==============================================================================
 * Function: AppFileX_UnlockMedia
 *==============================================================================*/
static void AppFileX_UnlockMedia(void) {
	(void) tx_mutex_put(&g_filex_media_mutex);
}

/*==============================================================================
 * Function: AppFileX_CreateCapturedImagesDirectoryLocked
 *==============================================================================*/
static UINT AppFileX_CreateCapturedImagesDirectoryLocked(void) {
	return fx_directory_create(&g_sd_fx_media, CAPTURED_IMAGES_DIRECTORY_NAME);
}

/*==============================================================================*/
static UINT AppFileX_SeedCapturedImageFallbackIndexLocked(ULONG *next_index_ptr,
		const CHAR *file_extension_ptr) {
	UINT fx_status = FX_SUCCESS;
	UINT file_attributes = 0U;
	ULONG capture_index = 0U;
	int file_name_chars = 0;
	CHAR candidate_name[CAPTURED_IMAGE_FILE_NAME_LENGTH] = { 0 };

	if ((next_index_ptr == NULL) || (file_extension_ptr == NULL)
			|| (file_extension_ptr[0] == '\0')) {
		return FX_PTR_ERROR;
	}

	fx_status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (fx_status != FX_SUCCESS) {
		return fx_status;
	}

	for (capture_index = 0U; capture_index <= CAPTURED_IMAGE_MAX_INDEX;
			capture_index++) {
		file_name_chars = snprintf(candidate_name, sizeof(candidate_name),
				"capture_%04lu.%s", (unsigned long) capture_index,
				file_extension_ptr);
		if ((file_name_chars < 0)
				|| ((ULONG) file_name_chars >= sizeof(candidate_name))) {
			fx_status = FX_INVALID_NAME;
			break;
		}

		fx_status = fx_file_attributes_read(&g_sd_fx_media, candidate_name,
				&file_attributes);
		if (fx_status == FX_NOT_FOUND) {
			*next_index_ptr = capture_index;
			fx_status = FX_SUCCESS;
			break;
		}

		if (fx_status != FX_SUCCESS) {
			break;
		}
	}

	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	return fx_status;
}

/*==============================================================================
 * Function: AppFileX_MillisecondsToTicks
 *
 * Purpose:
 *   Convert a millisecond timeout into ThreadX scheduler ticks, rounding up so
 *   short delays still wait for at least one tick.
 *==============================================================================*/
static ULONG AppFileX_MillisecondsToTicks(uint32_t timeout_ms) {
	uint32_t ticks = 0U;

	ticks = (timeout_ms * (uint32_t) TX_TIMER_TICKS_PER_SECOND + 999U) / 1000U;
	if ((timeout_ms > 0U) && (ticks == 0U)) {
		ticks = 1U;
	}

	return (ULONG) ticks;
}

/**
 * @brief Flash the board's blue LED for a configurable duration.
 *
 * This uses the raw GPIO pin directly so the cue matches the ST example path
 * and stays independent of the BSP LED wrapper.
 */
static void AppFileX_FlashCaptureBlue(uint32_t hold_ms) {
	GPIO_InitTypeDef gpio_init = { 0 };

	if (!g_capture_blue_gpio_initialized) {
		__HAL_RCC_GPIOG_CLK_ENABLE();
		HAL_GPIO_WritePin(GPIOG, GPIO_PIN_8, GPIO_PIN_SET);

		gpio_init.Pin = GPIO_PIN_8;
		gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
		gpio_init.Pull = GPIO_NOPULL;
		gpio_init.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
		HAL_GPIO_Init(GPIOG, &gpio_init);

		g_capture_blue_gpio_initialized = true;
	}

	HAL_GPIO_WritePin(GPIOG, GPIO_PIN_8, GPIO_PIN_RESET);
	/* Hold the blue capture cue for the requested duration. */
	DelayMilliseconds_ThreadX(hold_ms);
	HAL_GPIO_WritePin(GPIOG, GPIO_PIN_8, GPIO_PIN_SET);
}

/*==============================================================================*/
static void AppFileX_LogTimingStep(const CHAR *label, ULONG start_tick) {
#if APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
	char line[128];
	const ULONG end_tick = tx_time_get();
	const ULONG elapsed_ticks = (ULONG) (end_tick - start_tick);
	const ULONG elapsed_ms = (elapsed_ticks * 1000U)
			/ (ULONG) TX_TIMER_TICKS_PER_SECOND;
	int written = 0;

	if (label == NULL) {
		label = "?";
	}

	written = snprintf(line, sizeof(line),
			"[FILEX][TIMING] %s +%lu ms\r\n", label,
			(unsigned long) elapsed_ms);
	if (written > 0) {
		(void) DebugConsole_WriteString(line);
	}
#else
	(void)label;
	(void)start_tick;
#endif
}
/* USER CODE END 1 */
