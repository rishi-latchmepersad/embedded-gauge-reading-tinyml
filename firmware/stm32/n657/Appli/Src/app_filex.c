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
#include <string.h>

#include "app_azure_rtos_config.h"
#include "sd_spi_ll.h"
#include "main.h"
#include "app_threadx.h"
#include "tx_api.h" /* ThreadX services like tx_thread_sleep. */
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "sd_debug_log_service.h"

/*
 * Keep FileX console noise low by default. The thread now emits a small number
 * of explicit status/error lines, while the detailed state breadcrumbs stay
 * behind a local opt-in switch for bring-up sessions.
 */
#ifndef APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
#define APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS 0
#endif
#ifndef APP_FILEX_ENABLE_STATE_BREADCRUMBS
#define APP_FILEX_ENABLE_STATE_BREADCRUMBS 0
#endif
#if !APP_FILEX_ENABLE_VERBOSE_CONSOLE_LOGS
#undef DebugConsole_Printf
#define DebugConsole_Printf(...) ((void) 0)
#endif
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
/* Main thread priority */
#define FX_APP_THREAD_PRIO               14
/* USER CODE BEGIN PD */
#define CAPTURED_IMAGES_DIRECTORY_NAME   "captured_images"
#define CAPTURED_IMAGE_MAX_PATH_LENGTH   96U
#define CAPTURED_IMAGE_FILE_NAME_LENGTH  32U
#define CAPTURED_IMAGE_MAX_INDEX         9999U
#define CAPTURED_IMAGE_READBACK_BYTES    32U
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
static ULONG AppFileX_MillisecondsToTicks(uint32_t timeout_ms);
static void AppFileX_FlashCaptureSuccessBlue(void);
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

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_LOG_SERVICE_INITIALIZE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_LOG_SERVICE_INITIALIZE: {
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

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE: {
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

		context_ptr->last_progress_tick = tx_time_get();
		DebugConsole_Printf(
				"Now listening to debug messages and writing to .log files.\r\n");
		context_ptr->state = APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_CAPTURE_DIRECTORY_CREATE: {
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
#if 0
			/* Temporarily disable SD debug-log draining while isolating capture/save deadlock behavior. */
			SdDebugLogService_ServiceQueue(32U);
#endif
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

/*==============================================================================
 * Function: AppFileX_GetNextCapturedImageName
 *
 * Purpose:
 *   Find the next unused capture_<index>.<extension> name inside /captured_images.
 *==============================================================================*/
UINT AppFileX_GetNextCapturedImageName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr) {
	UINT tx_status = TX_SUCCESS;
	UINT fx_status = FX_SUCCESS;
	UINT file_attributes = 0U;
	ULONG capture_index = 0U;
	int file_name_chars = 0;

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

	fx_status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (fx_status != FX_SUCCESS) {
		AppFileX_UnlockMedia();
		return fx_status;
	}

	for (capture_index = 0U; capture_index <= CAPTURED_IMAGE_MAX_INDEX;
			capture_index++) {
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

	DebugConsole_Printf("[FILEX][CAPTURE] Preparing to save capture /%s.\r\n", path);
	for (ULONG waited_ms = 0U; waited_ms < 15000U; waited_ms += 1000U) {
		tx_status = tx_mutex_get(&g_filex_media_mutex,
				AppFileX_MillisecondsToTicks(1000U));
		if (tx_status == TX_SUCCESS) {
			break;
		}
	}

	if (tx_status != TX_SUCCESS) {
		return tx_status;
	}

	DebugConsole_Printf(
			"[FILEX][CAPTURE] Setting default directory to /%s before opening.\r\n",
			CAPTURED_IMAGES_DIRECTORY_NAME);
	fx_status = fx_directory_default_set(&g_sd_fx_media,
			CAPTURED_IMAGES_DIRECTORY_NAME);
	if (fx_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[FILEX][CAPTURE] Failed to select capture directory before write, status=%lu.\r\n",
				(unsigned long) fx_status);
		AppFileX_UnlockMedia();
		return fx_status;
	}
	DebugConsole_Printf(
			"[FILEX][CAPTURE] Default directory set for /%s.\r\n",
			CAPTURED_IMAGES_DIRECTORY_NAME);

	{
		FX_FILE capture_fx_file;
		UINT file_status = FX_SUCCESS;
		bool capture_file_open = false;

		DebugConsole_Printf("[FILEX][CAPTURE] Opening capture file /%s.\r\n",
				path);
		fx_status = fx_file_open(&g_sd_fx_media, &capture_fx_file,
				(CHAR*) file_name_ptr, FX_OPEN_FOR_WRITE);
		if (fx_status == FX_NOT_FOUND) {
			DebugConsole_Printf(
					"[FILEX][CAPTURE] Capture file not found; creating /%s.\r\n",
					path);
			fx_status = fx_file_create(&g_sd_fx_media, (CHAR*) file_name_ptr);
			if ((fx_status == FX_SUCCESS) || (fx_status == FX_ALREADY_CREATED)) {
				fx_status = fx_file_open(&g_sd_fx_media, &capture_fx_file,
						(CHAR*) file_name_ptr, FX_OPEN_FOR_WRITE);
			}
		}

		if (fx_status != FX_SUCCESS) {
			DebugConsole_Printf(
					"[FILEX][CAPTURE] Failed to open capture file /%s, status=%lu.\r\n",
					path, (unsigned long) fx_status);
			(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
			AppFileX_UnlockMedia();
			return fx_status;
		}

		capture_file_open = true;

		file_status = fx_file_seek(&capture_fx_file, 0U);
		if (file_status != FX_SUCCESS) {
			DebugConsole_Printf(
					"[FILEX][CAPTURE] Failed to seek capture file /%s, status=%lu.\r\n",
					path, (unsigned long) file_status);
			(void) fx_file_close(&capture_fx_file);
			(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
			AppFileX_UnlockMedia();
			return file_status;
		}

		DebugConsole_Printf("[FILEX][CAPTURE] Writing %lu bytes to /%s.\r\n",
				(unsigned long) data_length, path);
		file_status = fx_file_write(&capture_fx_file, (VOID*) data_ptr,
				data_length);
		if (file_status == FX_SUCCESS) {
			DebugConsole_Printf("[FILEX][CAPTURE] Flushing /%s after write.\r\n",
					path);
			file_status = fx_media_flush(&g_sd_fx_media);
			if (file_status != FX_SUCCESS) {
				DebugConsole_Printf("[FILEX][CAPTURE] Failed to flush capture file /%s, status=%lu.\r\n",
						path, (unsigned long) file_status);
			}
		}

		if (capture_file_open) {
			(void) fx_file_close(&capture_fx_file);
		}

		fx_status = file_status;
	}
	if (fx_status == FX_SUCCESS) {
		DebugConsole_Printf("[FILEX][CAPTURE] Flushed /%s (%lu bytes).\r\n",
				path, (unsigned long) data_length);
	}

	(void) fx_directory_default_set(&g_sd_fx_media, FX_NULL);
	AppFileX_UnlockMedia();

	if (fx_status == FX_SUCCESS) {
		DebugConsole_Printf(
				"Saved captured image to /%s (%lu bytes).\r\n",
				path, (unsigned long) data_length);
		AppFileX_FlashCaptureSuccessBlue();
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
 * @brief Flash the board's blue LED after a successful capture save.
 *
 * This uses the raw GPIO pin directly so the cue matches the ST example path
 * and stays independent of the BSP LED wrapper.
 */
static void AppFileX_FlashCaptureSuccessBlue(void) {
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
	DelayMilliseconds_ThreadX(3000U);
	HAL_GPIO_WritePin(GPIOG, GPIO_PIN_8, GPIO_PIN_SET);
}
/* USER CODE END 1 */
