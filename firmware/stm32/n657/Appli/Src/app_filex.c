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

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <string.h>

#include "sd_spi_ll.h"
#include "main.h"
#include "tx_api.h" /* ThreadX services like tx_thread_sleep. */
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "sd_debug_log_service.h"
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
/* Main thread stack size */
#define FX_APP_THREAD_STACK_SIZE         16000
/* Main thread priority */
#define FX_APP_THREAD_PRIO               10
/* USER CODE BEGIN PD */
#define FILEX_MEDIA_CACHE_BUFFER_SIZE    (4U * 512U) /* 4 sectors cache, 2048 bytes. */
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
	DebugLed_BlinkBlueBlocking(100, 100, 1);
	DebugLed_BlinkGreenBlocking(100, 100, 1);
	DebugLed_BlinkRedBlocking(100, 100, 1);
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

	/* Keep driver context visibility consistent for debugger. */
	g_sd_filex_driver_context.is_initialized = 0U;
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

	DebugConsole_Printf("FileX state machine ERROR. state=%lu code=%lu\r\n",
			(ULONG) error_state, (ULONG) error_code);
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
		/**
		 * Send each of the required SPI SD commands to initialize the SD card
		 * ie. CMD0->CMD8->ACMD41->CMD58
		 * And move onto the next subsequent state once the correct
		 * responses are received
		 */
		context_ptr->sd_cmd0_r1 = SPI_SendCMD0_GetR1();
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_SEND_CMD8;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_SEND_CMD8: {
		context_ptr->sd_cmd8_r1 = SPI_SendCMD8_ReadR7(context_ptr->r7_response);
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_WAIT_READY_ACMD41;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_WAIT_READY_ACMD41: {
		/* Blocking helper, acceptable for now. Can be made incremental later. */
		context_ptr->sd_acmd41_r1 = SPI_SendACMD41_UntilReady(NULL);

		if (context_ptr->sd_acmd41_r1 != 0x00U) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_WAIT_READY_ACMD41,
					(UINT) context_ptr->sd_acmd41_r1);
			break;
		}

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_READ_OCR_CMD58;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_READ_OCR_CMD58: {
		context_ptr->sd_cmd58_r1 = SPI_SendCMD58_ReadOCR(
				context_ptr->ocr_response);
		DebugConsole_Printf(
				"All SPI commands sent and received successfully.\r\n");
		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_SD_READ_PARTITION0_INFO;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_SD_READ_PARTITION0_INFO: {
		/**
		 * Read the info on partition 0 on the sd card once we have sent all the commands
		 * and received the correct responses in return.
		 */
		if (SPI_ReadPartition0Info(&context_ptr->partition_start_lba,
				&context_ptr->partition_sector_count) != 0x00U) {
			AppFileX_StateMachine_EnterError(context_ptr,
					APP_FILEX_STATE_SD_READ_PARTITION0_INFO, (UINT) 1U);
			break;
		}

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

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_LOG_SERVICE_INITIALIZE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_LOG_SERVICE_INITIALIZE: {
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

		context_ptr->last_progress_tick = tx_time_get();
		context_ptr->state = APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE;
		context_ptr->state_entry_tick = context_ptr->last_progress_tick;
		break;
	}

	case APP_FILEX_STATE_TEST_FILE_CREATE_OPEN_WRITE_CLOSE: {
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

		DebugLed_BlinkBlueBlocking(500U, 500U, 1U);
		break;
	}

	case APP_FILEX_STATE_ERROR: {
		/* Best effort drain, then blink red 1s on, 1s off, then restart state machine. */
		if (context_ptr->log_service_is_initialized != 0U) {
			SdDebugLogService_ServiceQueue(64U);
		}

		// print a message about the error state
		DebugConsole_Printf("FileX App thread ran into an error.\r\n");

		(void) DebugLed_BlinkRedBlocking(1000U, 1000U, 1U);

		/* Best effort cleanup before restart. */
		if (context_ptr->filex_media_is_open != 0U) {
			(void) fx_media_flush(&g_sd_fx_media);
			(void) fx_media_close(&g_sd_fx_media);
			context_ptr->filex_media_is_open = 0U;
		}

		g_sd_filex_driver_context.is_initialized = 0U;

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
/* USER CODE END 1 */
