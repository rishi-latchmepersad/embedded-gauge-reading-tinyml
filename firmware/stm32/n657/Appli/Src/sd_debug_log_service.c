/*
 * sd_debug_log_service.c
 *
 *  Created on: 12 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "sd_debug_log_service.h"
#include <stdio.h>

#include <stdbool.h>
#include <string.h>

#include "app_filex.h"
#include "main.h"
#include "debug_console.h"
#include "threadx_utils.h"

/*==============================================================================
 * Type: SdDebugLogService_LogBuffer
 *
 * Purpose:
 *   Fixed-size log buffer allocated from a ThreadX block pool.
 *
 * Notes:
 *   The queue carries a pointer to this buffer (one ULONG message).
 *==============================================================================*/
typedef struct {
	ULONG length_bytes;
	UCHAR payload[SD_DEBUG_LOG_SERVICE_MAX_LINE_LENGTH_BYTES];

} SdDebugLogService_LogBuffer;

/* ThreadX queue message size in ULONGs.
 We enqueue one pointer, so message size is 1 ULONG. */
#define SD_DEBUG_LOG_SERVICE_QUEUE_MESSAGE_WORDS            (1U)

/* We keep at most QUEUE_DEPTH outstanding log buffers. */
#define SD_DEBUG_LOG_SERVICE_BLOCK_POOL_BLOCK_COUNT         (SD_DEBUG_LOG_SERVICE_QUEUE_DEPTH)

/* ThreadX block pool requires a fixed block size.
 Ensure it is ULONG-aligned. */
#define SD_DEBUG_LOG_SERVICE_ALIGN_UP_TO_ULONG(bytes) \
    (((bytes) + (sizeof(ULONG) - 1U)) & ~(sizeof(ULONG) - 1U))

#define SD_DEBUG_LOG_SERVICE_BLOCK_SIZE_BYTES \
    (SD_DEBUG_LOG_SERVICE_ALIGN_UP_TO_ULONG(sizeof(SdDebugLogService_LogBuffer)))

/*==============================================================================
 * Private state
 *==============================================================================*/
static TX_QUEUE g_sd_debug_log_queue;
static ULONG *g_sd_debug_log_queue_storage_ptr = NULL;

static TX_BLOCK_POOL g_sd_debug_log_block_pool;
static UCHAR *g_sd_debug_log_block_pool_storage_ptr = NULL;

static FX_MEDIA *g_sd_debug_log_media_ptr = NULL;
static FX_FILE g_sd_debug_log_fx_file;

static SdDebugLogCore_Context g_sd_debug_log_core_context;
static SdDebugLogCore_FileOps g_sd_debug_log_file_ops;
static volatile uint32_t g_sd_debug_log_enqueue_failures = 0U;
static uint32_t g_sd_debug_log_write_failures = 0U;
static uint32_t g_sd_debug_log_flush_failures = 0U;
static uint32_t g_sd_debug_log_adapter_failures = 0U;
static TX_MUTEX g_sd_debug_log_core_mutex;
static bool g_sd_debug_log_core_mutex_created = false;
static uint32_t g_sd_debug_log_reported_enqueue_failures = 0U;
static uint32_t g_sd_debug_log_reported_write_failures = 0U;
static UINT g_sd_debug_log_last_filex_status = FX_SUCCESS;
static char g_sd_debug_log_last_filex_operation[24] = "none";
static bool g_sd_debug_log_fault_persisted = false;
static bool g_sd_debug_log_low_power_persisted = false;

static uint8_t g_sd_debug_log_file_is_open = 0U;
static bool g_sd_debug_log_queue_created = false;
/* Queue depth becomes zero when this worker receives a record, so expose the
 * active FileX write separately for the Stop-mode drain barrier. */
static volatile bool g_sd_debug_log_write_in_progress = false;

/**
 * @brief Report the original FileX error before the adapter maps it to -2.
 * @param operation FileX operation that failed.
 * @param status Original FileX status, not the logger core status.
 * @param name File path involved, or a short handle description.
 * @sideeffects Writes the first and every sixteenth adapter failure to UART.
 */
static void SdDebugLogService_ReportFileXError(const char *operation,
		UINT status, const char *name) {
	g_sd_debug_log_adapter_failures++;
    g_sd_debug_log_last_filex_status = status;
    (void) snprintf(g_sd_debug_log_last_filex_operation,
            sizeof(g_sd_debug_log_last_filex_operation), "%.23s", operation);
	/* Repeated failures must retain their cause without flooding the console. */
	if ((g_sd_debug_log_adapter_failures == 1U)
			|| ((g_sd_debug_log_adapter_failures % 16U) == 0U)) {
		DebugConsole_Printf(
				"[SDLOG][FILEX] op=%s path=%s status=%lu (0x%02lX) cwd=%s\r\n",
				operation, name, (unsigned long) status, (unsigned long) status,
				g_sd_debug_log_media_ptr->fx_media_default_path.fx_path_string);
	}
}

/*==============================================================================
 * Function: SdDebugLogService_LockMedia
 *
 * Purpose:
 *   Acquire the shared FileX media mutex before touching the SD card.
 *============================================================================*/
static int32_t SdDebugLogService_LockMedia(void) {
	UINT status = AppFileX_AcquireMediaLock();

	if (status != TX_SUCCESS) {
		return -1;
	}

	return 0;
}

/*==============================================================================
 * Function: SdDebugLogService_UnlockMedia
 *
 * Purpose:
 *   Release the shared FileX media mutex after touching the SD card.
 *============================================================================*/
static void SdDebugLogService_UnlockMedia(void) {
	AppFileX_ReleaseMediaLock();
}

/*==============================================================================
 * Function: SdDebugLogService_StrnlenBounded
 *
 * Purpose:
 *   Bounded strlen for safety on embedded targets.
 *
 * Parameters:
 *   string_ptr  - Input string pointer.
 *   max_length  - Maximum number of characters to scan.
 *
 * Returns:
 *   Length up to max_length.
 *==============================================================================*/
static uint32_t SdDebugLogService_StrnlenBounded(const char *string_ptr,
		uint32_t max_length) {
	uint32_t index = 0U;

	/* NULL string means length 0. */
	if (string_ptr == NULL) {
		return 0U;
	}

	/* Scan until '\0' or max_length. */
	for (index = 0U; index < max_length; index++) {
		if (string_ptr[index] == '\0') {
			break;
		}
	}

	return index;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_OpenAppend
 *
 * Purpose:
 *   Open the active log file for append using FileX.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *
 * Notes:
 *   FileX does not have a single "append mode" flag; we open for write and seek
 *   to end.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_OpenAppend(void *user_context_ptr,
		const char *file_name_ptr) {
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* We must have a valid mounted media pointer. */
	if (g_sd_debug_log_media_ptr == NULL) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* If already open, nothing to do. */
	if (g_sd_debug_log_file_is_open != 0U) {
		SdDebugLogService_UnlockMedia();
		return 0;
	}

	/* Open file for write. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &g_sd_debug_log_fx_file,
			(CHAR*) file_name_ptr,
			FX_OPEN_FOR_WRITE);
	if (status != FX_SUCCESS) {
		SdDebugLogService_ReportFileXError("open_append", status, file_name_ptr);
		SdDebugLogService_UnlockMedia();
		return -2;
	}

	/* Seek to end so we append rather than overwrite. */
	status = fx_file_seek(&g_sd_debug_log_fx_file,
			g_sd_debug_log_fx_file.fx_file_current_file_size);
	if (status != FX_SUCCESS) {
		SdDebugLogService_ReportFileXError("seek_append", status, file_name_ptr);
		/* If seek fails, close to avoid leaked handle. */
		(void) fx_file_close(&g_sd_debug_log_fx_file);
		SdDebugLogService_UnlockMedia();
		return -3;
	}

	/* Mark file handle open. */
	g_sd_debug_log_file_is_open = 1U;
	SdDebugLogService_UnlockMedia();

	return 0;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_CreateNew
 *
 * Purpose:
 *   Create a new file if it does not exist.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_CreateNew(void *user_context_ptr,
		const char *file_name_ptr) {
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* Media must be mounted. */
	if (g_sd_debug_log_media_ptr == NULL) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* Create the file. */
	status = fx_file_create(g_sd_debug_log_media_ptr, (CHAR*) file_name_ptr);

	/* If it already exists, FileX returns FX_ALREADY_CREATED, which is okay. */
	if ((status == FX_SUCCESS) || (status == FX_ALREADY_CREATED)) {
		SdDebugLogService_UnlockMedia();
		return 0;
	}

	SdDebugLogService_ReportFileXError("create", status, file_name_ptr);
	SdDebugLogService_UnlockMedia();
	return -2;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_Close
 *
 * Purpose:
 *   Close the active file if open.
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_Close(void *user_context_ptr) {
	UINT status;
	(void) user_context_ptr;

	/* If not open, nothing to do. */
	if (g_sd_debug_log_file_is_open == 0U) {
		return 0;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -1;
	}

	/* Close file handle. */
	status = fx_file_close(&g_sd_debug_log_fx_file);

    /* FileX removes a valid handle from its open list before writing its
     * directory entry. Even an I/O error from that final write leaves it closed. */
    g_sd_debug_log_file_is_open = 0U;
    if (status != FX_SUCCESS) { SdDebugLogService_ReportFileXError("close", status, "/debug.log"); }
	SdDebugLogService_UnlockMedia();

	return (status == FX_SUCCESS) ? 0 : -2;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_Write
 *
 * Purpose:
 *   Write bytes to the active file.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_Write(void *user_context_ptr,
		const void *data_ptr, uint32_t data_length_bytes) {
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* Must have open file handle. */
	if (g_sd_debug_log_file_is_open == 0U) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* Write to file. */
	status = fx_file_write(&g_sd_debug_log_fx_file, (VOID*) data_ptr,
			(ULONG) data_length_bytes);
	if (status != FX_SUCCESS) {
		SdDebugLogService_ReportFileXError("write", status, "/debug.log");
		SdDebugLogService_UnlockMedia();
		return -2;
	}

	SdDebugLogService_UnlockMedia();

	return 0;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_Flush
 *
 * Purpose:
 *   Flush file and media.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *
 * Notes:
 *   Flushing media too often can reduce performance. For now, this is safe and
 *   simple. We can optimize once everything is stable.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_Flush(void *user_context_ptr) {
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* Media must be mounted. */
	if (g_sd_debug_log_media_ptr == NULL) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

    /* One media flush is sufficient; FileX has no separate file-flush API. */
    status = fx_media_flush(g_sd_debug_log_media_ptr);
    if (status != FX_SUCCESS) { SdDebugLogService_ReportFileXError("flush", status, "/debug.log"); }
	SdDebugLogService_UnlockMedia();

	return (status == FX_SUCCESS) ? 0 : -3;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_Rename
 *
 * Purpose:
 *   Rename a file on the media.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_Rename(void *user_context_ptr,
		const char *old_name_ptr, const char *new_name_ptr) {
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* Media must be mounted. */
	if (g_sd_debug_log_media_ptr == NULL) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* Perform rename. */
	status = fx_file_rename(g_sd_debug_log_media_ptr, (CHAR*) old_name_ptr,
			(CHAR*) new_name_ptr);
	if (status != FX_SUCCESS) {
        SdDebugLogService_ReportFileXError("rename", status, old_name_ptr);
		SdDebugLogService_UnlockMedia();
		return -2;
	}

	SdDebugLogService_UnlockMedia();

	return 0;
}

/**
 * @brief Remove an expired debug archive under the shared media mutex.
 * @return Zero for deleted/missing file; original FileX failure is reported to UART.
 */
static int32_t SdDebugLogService_FileX_Remove(void *context, const char *name) {
    (void) context;
    if (SdDebugLogService_LockMedia() != 0) { return -1; }
    const UINT status = fx_file_delete(g_sd_debug_log_media_ptr, (CHAR *) name);
    if ((status != FX_SUCCESS) && (status != FX_NOT_FOUND)) {
        SdDebugLogService_ReportFileXError("delete_archive", status, name);
    }
    SdDebugLogService_UnlockMedia();
    return ((status == FX_SUCCESS) || (status == FX_NOT_FOUND)) ? 0 : -2;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_Exists
 *
 * Purpose:
 *   Check for file existence using fx_file_open.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *
 * Notes:
 *   FileX has file attributes APIs, but open-for-read is a simple existence check.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_Exists(void *user_context_ptr,
		const char *file_name_ptr, uint8_t *exists_out_ptr) {
	FX_FILE temp_file;
	UINT status = FX_SUCCESS;
	(void) user_context_ptr;

	/* Validate required pointers. */
	if ((g_sd_debug_log_media_ptr == NULL) || (exists_out_ptr == NULL)) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* Try opening the file for read. If it opens, it exists. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &temp_file,
			(CHAR*) file_name_ptr, FX_OPEN_FOR_READ);
	if (status == FX_SUCCESS) {
		/* Close immediately, we only wanted existence. */
		(void) fx_file_close(&temp_file);
		SdDebugLogService_UnlockMedia();

		*exists_out_ptr = 1U;
		return 0;
	}

	/* If not found, return exists = 0. */
	if (status == FX_NOT_FOUND) {
		SdDebugLogService_UnlockMedia();
		*exists_out_ptr = 0U;
		return 0;
	}

	/* Any other error is a real failure. */
	SdDebugLogService_ReportFileXError("exists_open", status, file_name_ptr);
	SdDebugLogService_UnlockMedia();
	return -2;
}

/*==============================================================================
 * Function: SdDebugLogService_FileX_GetSize
 *
 * Purpose:
 *   Get the size of a file on the mounted FileX media.
 *
 * Parameters:
 *   user_context_ptr   - Unused (reserved for future use).
 *   file_name_ptr      - Null terminated file name to query.
 *   size_out_bytes_ptr - Output: file size in bytes.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *==============================================================================*/
static int32_t SdDebugLogService_FileX_GetSize(void *user_context_ptr,
		const char *file_name_ptr, uint32_t *size_out_bytes_ptr) {
	FX_FILE temp_file;
	ULONG file_size_bytes = 0U;
	UINT status = FX_SUCCESS;

	(void) user_context_ptr;

	/* Validate required pointers. */
	if ((g_sd_debug_log_media_ptr == NULL) || (file_name_ptr == NULL)
			|| (size_out_bytes_ptr == NULL)) {
		return -1;
	}

	if (SdDebugLogService_LockMedia() != 0) {
		return -2;
	}

	/* Default output to 0 so callers never read an uninitialized value. */
	*size_out_bytes_ptr = 0U;
    if ((g_sd_debug_log_file_is_open != 0U) && (strcmp(file_name_ptr, "/debug.log") == 0)) {
        *size_out_bytes_ptr = (uint32_t) g_sd_debug_log_fx_file.fx_file_current_file_size;
        SdDebugLogService_UnlockMedia();
        return 0;
    }

	/* Open the file for read. This allows FileX to populate temp_file metadata. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &temp_file,
			(CHAR*) file_name_ptr,
			FX_OPEN_FOR_READ);

	/* If the file does not exist, treat it as size 0 and succeed. */
	if (status == FX_NOT_FOUND) {
		SdDebugLogService_UnlockMedia();
		return 0;
	}

	/* Any other open failure is a real error. */
	if (status != FX_SUCCESS) {
        SdDebugLogService_ReportFileXError("get_size_open", status, file_name_ptr);
		SdDebugLogService_UnlockMedia();
		return -2;
	}

	/* FileX exposes the file size in the FX_FILE control block after open. */
	file_size_bytes = temp_file.fx_file_current_file_size;

	/* Close the temporary file handle. */
	status = fx_file_close(&temp_file);
    if (status != FX_SUCCESS) { SdDebugLogService_ReportFileXError("get_size_close", status, file_name_ptr); }

	/* Return size back as uint32_t for our core. */
	*size_out_bytes_ptr = (uint32_t) file_size_bytes;
	SdDebugLogService_UnlockMedia();

	return (status == FX_SUCCESS) ? 0 : -3;
}

UINT SdDebugLogService_Initialize(TX_BYTE_POOL *byte_pool_ptr,
		FX_MEDIA *media_ptr) {
	UINT status = TX_SUCCESS;
	ULONG queue_storage_bytes = 0U;
	ULONG block_pool_storage_bytes = 0U;

	/* Validate pointers. */
	if ((byte_pool_ptr == NULL) || (media_ptr == NULL)) {
		return TX_PTR_ERROR;
	}

    if (!g_sd_debug_log_core_mutex_created) {
        status = tx_mutex_create(&g_sd_debug_log_core_mutex, "sd_log_core", TX_INHERIT);
        if (status != TX_SUCCESS) { return status; }
        g_sd_debug_log_core_mutex_created = true;
    }
    /* Rebinding after a media restart must reuse the already-created queue/pool. */
	/* Store media pointer so FileX adapter functions can use it. */
	g_sd_debug_log_media_ptr = media_ptr;

	/* Initialize rollover core with chosen names and threshold. */
	SdDebugLogCore_Initialize(&g_sd_debug_log_core_context,
			(uint32_t) SD_DEBUG_LOG_SERVICE_ROLLOVER_THRESHOLD_BYTES,
			"/debug.log", "/debug_");

	/* Clear ops struct before binding function pointers. */
	(void) memset(&g_sd_debug_log_file_ops, 0, sizeof(g_sd_debug_log_file_ops));

	/* Bind FileX adapter functions into the core file ops interface. */
	g_sd_debug_log_file_ops.user_context_ptr = NULL;
	g_sd_debug_log_file_ops.open_append = SdDebugLogService_FileX_OpenAppend;
	g_sd_debug_log_file_ops.create_new = SdDebugLogService_FileX_CreateNew;
	g_sd_debug_log_file_ops.close = SdDebugLogService_FileX_Close;
	g_sd_debug_log_file_ops.write = SdDebugLogService_FileX_Write;
	g_sd_debug_log_file_ops.flush = SdDebugLogService_FileX_Flush; /* Now Option A: media flush only. */
	g_sd_debug_log_file_ops.rename = SdDebugLogService_FileX_Rename;
    g_sd_debug_log_file_ops.remove = SdDebugLogService_FileX_Remove;
	g_sd_debug_log_file_ops.exists = SdDebugLogService_FileX_Exists;
	g_sd_debug_log_file_ops.get_size = SdDebugLogService_FileX_GetSize;

    g_sd_debug_log_file_is_open = 0U;
    if (g_sd_debug_log_queue_created) { return TX_SUCCESS; }
	/* -------------------- Create block pool for log buffers -------------------- */

	block_pool_storage_bytes =
			(ULONG) (SD_DEBUG_LOG_SERVICE_BLOCK_POOL_BLOCK_COUNT
					* SD_DEBUG_LOG_SERVICE_BLOCK_SIZE_BYTES);

	status = tx_byte_allocate(byte_pool_ptr,
			(VOID**) &g_sd_debug_log_block_pool_storage_ptr,
			block_pool_storage_bytes,
			TX_NO_WAIT);
	if (status != TX_SUCCESS) {
		return status;
	}

	status = tx_block_pool_create(&g_sd_debug_log_block_pool,
			(CHAR*) "sd_debug_log_blocks",
			(ULONG) SD_DEBUG_LOG_SERVICE_BLOCK_SIZE_BYTES,
			(VOID*) g_sd_debug_log_block_pool_storage_ptr,
			block_pool_storage_bytes);
	if (status != TX_SUCCESS) {
		return status;
	}

	/* -------------------- Create queue of pointers (1 ULONG messages) -------------------- */

	queue_storage_bytes = (ULONG) (SD_DEBUG_LOG_SERVICE_QUEUE_DEPTH
			* sizeof(ULONG));

	status = tx_byte_allocate(byte_pool_ptr,
			(VOID**) &g_sd_debug_log_queue_storage_ptr, queue_storage_bytes,
			TX_NO_WAIT);
	if (status != TX_SUCCESS) {
		return status;
	}

	status = tx_queue_create(&g_sd_debug_log_queue,
			(CHAR*) "sd_debug_log_queue",
			(UINT) SD_DEBUG_LOG_SERVICE_QUEUE_MESSAGE_WORDS,
			(VOID*) g_sd_debug_log_queue_storage_ptr, queue_storage_bytes);
	if (status != TX_SUCCESS) {
		return status;
	}
	g_sd_debug_log_queue_created = true;

	return TX_SUCCESS;
}

UINT SdDebugLogService_EnqueueLine(const CHAR *line_ptr) {
	SdDebugLogService_LogBuffer *log_buffer_ptr = NULL;
	uint32_t input_length = 0U;
	UINT status = TX_SUCCESS;
	ULONG message_word = 0U;

	/* Validate string pointer. */
	if (line_ptr == NULL) {
		return TX_PTR_ERROR;
	}

    if (!g_sd_debug_log_queue_created) { return TX_NOT_AVAILABLE; }
	/* Allocate a log buffer from the block pool.
	 If the pool is empty, we drop the log line by returning the ThreadX error. */
	status = tx_block_allocate(&g_sd_debug_log_block_pool,
			(VOID**) &log_buffer_ptr, TX_NO_WAIT);
	if (status != TX_SUCCESS) {
        TX_INTERRUPT_SAVE_AREA
        TX_DISABLE
        g_sd_debug_log_enqueue_failures++;
        TX_RESTORE
		if ((g_sd_debug_log_enqueue_failures == 1U)
				|| ((g_sd_debug_log_enqueue_failures % 16U) == 0U)) {
			DebugConsole_Printf(
					"[SDLOG][ERROR] dropped line: no queue buffer; count=%lu status=%lu\r\n",
					(unsigned long) g_sd_debug_log_enqueue_failures,
					(unsigned long) status);
		}
		return status;
	}

	/* Clear buffer so the payload is predictable for debugging. */
	(void) memset(log_buffer_ptr, 0, sizeof(*log_buffer_ptr));

	/* Compute bounded length.
	 Reserve 2 bytes so we can add "\r\n" if needed. */
	input_length = SdDebugLogService_StrnlenBounded((const char*) line_ptr,
			(uint32_t) (SD_DEBUG_LOG_SERVICE_MAX_LINE_LENGTH_BYTES - 2U));

	/* Copy message into payload. */
	(void) memcpy(log_buffer_ptr->payload, line_ptr, input_length);

	/* Ensure line ends with newline.
	 If it does not end with '\n', append "\r\n". */
	if ((input_length == 0U)
			|| (log_buffer_ptr->payload[input_length - 1U] != (UCHAR) '\n')) {
		log_buffer_ptr->payload[input_length] = (UCHAR) '\r';
		log_buffer_ptr->payload[input_length + 1U] = (UCHAR) '\n';
		log_buffer_ptr->length_bytes = (ULONG) (input_length + 2U);
	} else {
		log_buffer_ptr->length_bytes = (ULONG) input_length;
	}

	/* Enqueue pointer as one ULONG message. */
	message_word = (ULONG) log_buffer_ptr;

	status = tx_queue_send(&g_sd_debug_log_queue, &message_word, TX_NO_WAIT);
	if (status != TX_SUCCESS) {
		/* If queue is full, release the block so we do not leak memory. */
		(void) tx_block_release((VOID*) log_buffer_ptr);
        TX_INTERRUPT_SAVE_AREA
        TX_DISABLE
        g_sd_debug_log_enqueue_failures++;
        TX_RESTORE
		if ((g_sd_debug_log_enqueue_failures == 1U)
				|| ((g_sd_debug_log_enqueue_failures % 16U) == 0U)) {
			DebugConsole_Printf(
					"[SDLOG][ERROR] dropped line: queue full; count=%lu status=%lu\r\n",
					(unsigned long) g_sd_debug_log_enqueue_failures,
					(unsigned long) status);
		}
		return status;
	}

	return TX_SUCCESS;
}

void SdDebugLogService_ServiceQueue(ULONG max_messages_to_process) {
	ULONG processed_count = 0U;

	for (processed_count = 0U; processed_count < max_messages_to_process;
			processed_count++) {
		ULONG message_word = 0U;
		UINT queue_status = TX_SUCCESS;
		SdDebugLogService_LogBuffer *log_buffer_ptr = NULL;

		/* Receive one pointer from the queue. */
        TX_INTERRUPT_SAVE_AREA
        TX_DISABLE
		queue_status = tx_queue_receive(&g_sd_debug_log_queue, &message_word,
				TX_NO_WAIT);
        if (queue_status == TX_SUCCESS) { g_sd_debug_log_write_in_progress = true; }
        TX_RESTORE
		if (queue_status != TX_SUCCESS) {
			/* Queue empty or error, stop this cycle. */
			break;
		}

		/* Convert ULONG back into pointer. */
		log_buffer_ptr = (SdDebugLogService_LogBuffer*) message_word;
		g_sd_debug_log_write_in_progress = true;
		if (log_buffer_ptr == NULL) {
			g_sd_debug_log_write_in_progress = false;
			continue;
		}

        (void) tx_mutex_get(&g_sd_debug_log_core_mutex, TX_WAIT_FOREVER);
		/* Write record to debug.log (rollover handled in core). */
		{
			const int32_t write_status = SdDebugLogCore_WriteRecord(
					&g_sd_debug_log_core_context,
				&g_sd_debug_log_file_ops, log_buffer_ptr->payload,
				(uint32_t) log_buffer_ptr->length_bytes);
			if (write_status != 0) {
				g_sd_debug_log_write_failures++;
				if ((g_sd_debug_log_write_failures == 1U)
						|| ((g_sd_debug_log_write_failures % 16U) == 0U)) {
					DebugConsole_Printf(
							"[SDLOG][ERROR] FileX write failed; count=%lu status=%ld\r\n",
							(unsigned long) g_sd_debug_log_write_failures,
							(long) write_status);
				}
			}
		}

        (void) tx_mutex_put(&g_sd_debug_log_core_mutex);
		/* Release buffer back to pool. */
		(void) tx_block_release((VOID*) log_buffer_ptr);
		g_sd_debug_log_write_in_progress = false;
	}
}

/**
 * @brief Wait until the queued SD debug/metrics records have been consumed.
 * @param timeout_ms Maximum wait duration in milliseconds.
 * @retval true when the queue is empty and no record is being written,
 *         false on timeout.
 * @sideeffects Yields so the FileX service thread can perform the writes.
 */
bool SdDebugLogService_WaitForQueueDrain(uint32_t timeout_ms) {
	uint32_t elapsed_ms = 0U;

	if (!g_sd_debug_log_queue_created) {
		return true;
	}

	while (elapsed_ms <= timeout_ms) {
		ULONG enqueued = 0U;
		if ((tx_queue_info_get(&g_sd_debug_log_queue, NULL, &enqueued, NULL,
					NULL, NULL, NULL) == TX_SUCCESS) && (enqueued == 0U)
				&& !g_sd_debug_log_write_in_progress) {
			return true;
		}
		DelayMilliseconds_Cooperative(10U);
		elapsed_ms += 10U;
	}

	return false;
}

/**
 * @brief Persist loss counters, close the debug handle, then commit directory metadata.
 * @return Zero only when the health record and close/flush succeeded.
 * @sideeffects Serializes with the FileX worker; emits UART evidence even on a full card.
 */
int32_t SdDebugLogService_ForceFlush(void) {
    if (!g_sd_debug_log_queue_created) { return -1; }
    if (tx_mutex_get(&g_sd_debug_log_core_mutex, TX_WAIT_FOREVER) != TX_SUCCESS) { return -1; }
    const uint32_t enqueue_losses = g_sd_debug_log_enqueue_failures;
    const uint32_t write_losses = g_sd_debug_log_write_failures;
    int32_t health_status = 0;
    const char *fault = App_GetPreviousFaultLogLine();
    const bool persist_fault = !g_sd_debug_log_fault_persisted && (fault[0] != '\0');
    const char *low_power = App_GetPreviousLowPowerLogLine();
    const bool persist_low_power = !g_sd_debug_log_low_power_persisted
            && (low_power[0] != '\0');
    if (persist_fault) {
        char record[256];
        const int length = snprintf(record, sizeof(record), "%s\r\n", fault);
        if ((length > 0) && ((size_t) length < sizeof(record))) {
            health_status = SdDebugLogCore_WriteRecord(&g_sd_debug_log_core_context,
                    &g_sd_debug_log_file_ops, record, (uint32_t) length);
        } else { health_status = -1; }
    }
    if (persist_low_power) {
        char record[256];
        const int length = snprintf(record, sizeof(record), "%s\r\n", low_power);
        if ((length > 0) && ((size_t) length < sizeof(record))) {
            const int32_t low_power_status = SdDebugLogCore_WriteRecord(
                    &g_sd_debug_log_core_context, &g_sd_debug_log_file_ops,
                    record, (uint32_t) length);
            if ((health_status == 0) && (low_power_status != 0)) {
                health_status = low_power_status;
            }
        } else if (health_status == 0) {
            health_status = -1;
        }
    }
    if ((enqueue_losses != g_sd_debug_log_reported_enqueue_failures)
            || (write_losses != g_sd_debug_log_reported_write_failures)) {
        char health[176];
        const int length = snprintf(health, sizeof(health),
                "[SDLOG][LOSS] dropped_enqueue=%lu dropped_write=%lu last_op=%s FileX=%lu\r\n",
                (unsigned long) enqueue_losses, (unsigned long) write_losses,
                g_sd_debug_log_last_filex_operation, (unsigned long) g_sd_debug_log_last_filex_status);
        (void) DebugConsole_WriteString(health);
        const int32_t loss_status = SdDebugLogCore_WriteRecord(&g_sd_debug_log_core_context,
                &g_sd_debug_log_file_ops, health, (uint32_t) length);
        if (health_status == 0) { health_status = loss_status; }
    }
    const int32_t close_status = SdDebugLogCore_ForceFlushAndClose(
            &g_sd_debug_log_core_context, &g_sd_debug_log_file_ops);
    const int32_t status = (health_status != 0) ? health_status : close_status;
    if (status != 0) {
        g_sd_debug_log_flush_failures++;
        DebugConsole_Printf("[SDLOG][ERROR] final health/flush/close failed count=%lu status=%ld\r\n",
                (unsigned long) g_sd_debug_log_flush_failures, (long) status);
    } else {
        g_sd_debug_log_reported_enqueue_failures = enqueue_losses;
        g_sd_debug_log_reported_write_failures = write_losses;
        if (persist_fault) {
            AppDiagnostics_AcknowledgePersistedFault();
            g_sd_debug_log_fault_persisted = true;
        }
        if (persist_low_power) {
            AppDiagnostics_AcknowledgePersistedLowPower();
            g_sd_debug_log_low_power_persisted = true;
        }
    }
    (void) tx_mutex_put(&g_sd_debug_log_core_mutex);
    return status;
}
