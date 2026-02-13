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
#include "sd_spi_ll.h"
#include "main.h"
#include "tx_api.h"                                         /* ThreadX services like tx_thread_sleep. */
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "sd_debug_log_service.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* Main thread stack size */
#define FX_APP_THREAD_STACK_SIZE         8192
/* Main thread priority */
#define FX_APP_THREAD_PRIO               10
/* USER CODE BEGIN PD */
#define FILEX_MEDIA_CACHE_BUFFER_SIZE    (8U * 512U)                         /* 8 sectors cache, 4096 bytes. */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* Main thread global data structures.  */
TX_THREAD fx_app_thread;

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

/* USER CODE END PFP */

/**
 * @brief  Application FileX Initialization.
 * @param memory_ptr: memory pointer
 * @retval int
 */
UINT MX_FileX_Init(VOID *memory_ptr) {
	UINT ret = FX_SUCCESS;
	TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*) memory_ptr;
	g_filex_byte_pool_ptr = byte_pool; /* Save for FileX thread so it can init logging service. */

	VOID *pointer;

	/* USER CODE BEGIN MX_FileX_MEM_POOL */

	/* USER CODE END MX_FileX_MEM_POOL */

	/* USER CODE BEGIN 0 */

	/* USER CODE END 0 */

	/*Allocate memory for the main thread's stack*/
	ret = tx_byte_allocate(byte_pool, &pointer, FX_APP_THREAD_STACK_SIZE,
	TX_NO_WAIT);

	/* Check FX_APP_THREAD_STACK_SIZE allocation*/
	if (ret != FX_SUCCESS) {
		return TX_POOL_ERROR;
	}

	/* Create the main thread.  */
	ret = tx_thread_create(&fx_app_thread, FX_APP_THREAD_NAME,
			fx_app_thread_entry, 0, pointer, FX_APP_THREAD_STACK_SIZE,
			FX_APP_THREAD_PRIO, FX_APP_PREEMPTION_THRESHOLD,
			FX_APP_THREAD_TIME_SLICE, FX_APP_THREAD_AUTO_START);

	/* Check main thread creation */
	if (ret != FX_SUCCESS) {
		return TX_THREAD_ERROR;
	}

	/* USER CODE BEGIN MX_FileX_Init */
	/* Allocate FileX media cache buffer from the same byte pool used for thread stacks. */
	ret = tx_byte_allocate(byte_pool, (VOID**) &g_filex_media_cache_buffer,
	FILEX_MEDIA_CACHE_BUFFER_SIZE, TX_NO_WAIT); /* Cache memory for FileX. */
	if (ret != TX_SUCCESS) /* If cache cannot be allocated, FileX mount will fail later. */
	{
		return TX_POOL_ERROR; /* Return pool error so you see it immediately. */
	}

	/* USER CODE END MX_FileX_Init */

	/* Initialize FileX.  */
	fx_system_initialize();

	/* USER CODE BEGIN MX_FileX_Init 1*/
	DebugConsole_Printf("MX FileX init 1 success.\r\n");
	/* USER CODE END MX_FileX_Init 1*/

	return ret;
}

/**
 * @brief  Main thread entry.
 * @param thread_input: ULONG user argument used by the thread entry
 * @retval none
 */
void fx_app_thread_entry(ULONG thread_input) {

	/* USER CODE BEGIN fx_app_thread_entry 0*/
	DebugConsole_Printf("FileX App thread started.\r\n");
	(void) thread_input; /* Thread argument unused for now. */

	UINT filex_status = FX_SUCCESS; /* Capture FileX API return codes for debugging. */

	uint8_t sd_cmd0_r1 = 0xFFU; /* CMD0 response. */
	uint8_t sd_cmd8_r1 = 0xFFU; /* CMD8 response. */
	uint8_t sd_cmd58_r1 = 0xFFU; /* CMD58 response. */
	uint8_t sd_acmd41_r1 = 0xFFU; /* ACMD41 final response. */

	uint8_t r7_response[4] = { 0U }; /* CMD8 R7 payload. */
	uint8_t ocr_response[4] = { 0U }; /* CMD58 OCR payload. */

	uint32_t partition_start_lba = 0U; /* Parsed partition start sector. */
	uint32_t partition_sector_count = 0U; /* Parsed partition length in sectors. */

	g_sd_filex_driver_context.is_initialized = 0U; /* Clear init flag for debugger visibility. */

	/* -------------------- SD SPI bringup sequence -------------------- */

	sd_cmd0_r1 = SPI_SendCMD0_GetR1(); /* Put card into SPI mode, request IDLE state. */
	sd_cmd8_r1 = SPI_SendCMD8_ReadR7(r7_response); /* Check SD version and voltage range support. */

	/* Bring the card out of IDLE using the robust looped ACMD41 helper. */
	sd_acmd41_r1 = SPI_SendACMD41_UntilReady(NULL); /* Repeats CMD55 + ACMD41 until ready or timeout. */

	sd_cmd58_r1 = SPI_SendCMD58_ReadOCR(ocr_response); /* Read OCR, confirms CCS (SDHC) and power status. */

	(void) sd_cmd0_r1; /* Keep variables for debugger inspection. */
	(void) sd_cmd8_r1; /* Keep variables for debugger inspection. */
	(void) sd_acmd41_r1; /* Keep variables for debugger inspection. */
	(void) sd_cmd58_r1; /* Keep variables for debugger inspection. */

	/* If the card is not ready, stop here so you can see the failure state. */
	if (sd_acmd41_r1 != 0x00U) /* ACMD41 must return 0x00 when ready. */
	{
		for (;;) /* Trap here for debugging. */
		{
			tx_thread_sleep(50U); /* Sleep so the system stays responsive under RTOS. */
		}
	}

	/* -------------------- Parse partition info (MBR entry 0) -------------------- */

	if (SPI_ReadPartition0Info(&partition_start_lba, &partition_sector_count)
			!= 0x00U) /* Read MBR and extract start and length. */
			{
		for (;;) /* Trap if partition parsing fails. */
		{
			tx_thread_sleep(50U); /* Sleep, do not spin hot. */
		}
	}

	g_sd_filex_driver_context.partition_start_lba = partition_start_lba; /* Save start LBA for the media driver offset. */
	g_sd_filex_driver_context.partition_sector_count = partition_sector_count;/* Save total sectors for FileX geometry. */
	g_sd_filex_driver_context.is_initialized = 1U; /* Mark init successful. */

	/* -------------------- FileX mount (FAT on FileX) -------------------- */

	filex_status = fx_media_open(&g_sd_fx_media, /* FileX media control block. */
	"SD_SPI_MEDIA", /* Media name shown in debugger. */
	SPI_FileX_SdSpiMediaDriver, /* Media driver function that does sector I/O. */
	&g_sd_filex_driver_context, /* Driver context pointer for offsets and size. */
	g_filex_media_cache_buffer, /* FileX cache buffer (allocated from byte pool). */
	FILEX_MEDIA_CACHE_BUFFER_SIZE); /* Cache buffer size in bytes. */

	if (filex_status != FX_SUCCESS) /* If mount fails, stop so you can inspect filex_status. */
	{
		for (;;) /* Trap for debugging. */
		{
			tx_thread_sleep(50U); /* Sleep to avoid CPU hogging. */
		}
	}

	/* -------------------- Debug log service init (queue + core + FileX bindings) -------------------- */

	if (g_filex_byte_pool_ptr == NULL) {
		for (;;) {
			tx_thread_sleep(50U);
		}
	}

	/* Initialize the logging queue and core, and bind it to the mounted media. */
	{
		UINT log_init_status = SdDebugLogService_Initialize(
				g_filex_byte_pool_ptr, &g_sd_fx_media);
		if (log_init_status != TX_SUCCESS) {
			for (;;) {
				tx_thread_sleep(50U);
			}
		}
	}

	/* Optional: write a startup marker. */
	(void) SdDebugLogService_EnqueueLine("debug log service initialized");
	DebugConsole_Printf("Initialized debug log service in FileX thread.");

	/* -------------------- Simple create + write test -------------------- */

	(void) fx_file_delete(&g_sd_fx_media, "test.txt"); /* Delete if it exists, ignore status for convenience. */

	filex_status = fx_file_create(&g_sd_fx_media, "test.txt"); /* Create a fresh test file on the SD card. */
	if ((filex_status != FX_SUCCESS) && (filex_status != FX_ALREADY_CREATED))/* Accept already created, otherwise treat as failure. */
	{
		for (;;) {
			tx_thread_sleep(50U);
		} /* Trap on failure. */
	}

	filex_status = fx_file_open(&g_sd_fx_media, &g_sd_fx_file, "test.txt",
	FX_OPEN_FOR_WRITE); /* Open for writing. */
	if (filex_status != FX_SUCCESS) /* If open fails, stop. */
	{
		for (;;) {
			tx_thread_sleep(50U);
		} /* Trap on failure. */
	}

	{
		static const CHAR message[] = "Hello from STM32N6 + ThreadX + FileX\r\n"; /* Static string avoids stack usage. */
		ULONG bytes_written = 0U; /* FileX returns number of bytes written. */

		filex_status = fx_file_write(&g_sd_fx_file, /* File handle. */
		(VOID*) message, /* Data pointer. */
		(ULONG) (sizeof(message) - 1U)); /* Write length, exclude null terminator. */

		(void) bytes_written; /* Keep variable pattern consistent for debugger. */

		if (filex_status != FX_SUCCESS) /* If write fails, stop. */
		{
			for (;;) {
				tx_thread_sleep(50U);
			} /* Trap on failure. */
		}
	}

	(void) fx_file_close(&g_sd_fx_file); /* Close file to flush metadata. */
	(void) fx_media_flush(&g_sd_fx_media); /* Force flush to ensure data hits the card. */

	DebugConsole_Printf("Successfully wrote text.txt to root of SD card.");
	/* USER CODE END fx_app_thread_entry 0*/

	/* USER CODE BEGIN fx_app_thread_entry 1*/
	while (1) {

		/* Drain a bounded number of messages each cycle so we do not starve other work. */
		SdDebugLogService_ServiceQueue(32U);

		DebugLed_BlinkBlueBlocking(1000U, 1000U, 1U);

	}
	DebugConsole_Printf("FileX thread closing.\r\n");
	/* USER CODE END fx_app_thread_entry 1*/
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */
