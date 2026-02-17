/*
 * sd_debug_log_service.c
 *
 *  Created on: 12 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "sd_debug_log_service.h"

#include <string.h>

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

static uint8_t g_sd_debug_log_file_is_open = 0U;

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

	/* If already open, nothing to do. */
	if (g_sd_debug_log_file_is_open != 0U) {
		return 0;
	}

	/* Open file for write. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &g_sd_debug_log_fx_file,
			(CHAR*) file_name_ptr,
			FX_OPEN_FOR_WRITE);
	if (status != FX_SUCCESS) {
		return -2;
	}

	/* Seek to end so we append rather than overwrite. */
	status = fx_file_seek(&g_sd_debug_log_fx_file,
			g_sd_debug_log_fx_file.fx_file_current_file_size);
	if (status != FX_SUCCESS) {
		/* If seek fails, close to avoid leaked handle. */
		(void) fx_file_close(&g_sd_debug_log_fx_file);
		return -3;
	}

	/* Mark file handle open. */
	g_sd_debug_log_file_is_open = 1U;

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

	/* Create the file. */
	status = fx_file_create(g_sd_debug_log_media_ptr, (CHAR*) file_name_ptr);

	/* If it already exists, FileX returns FX_ALREADY_CREATED, which is okay. */
	if ((status == FX_SUCCESS) || (status == FX_ALREADY_CREATED)) {
		return 0;
	}

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
	(void) user_context_ptr;

	/* If not open, nothing to do. */
	if (g_sd_debug_log_file_is_open == 0U) {
		return 0;
	}

	/* Close file handle. */
	(void) fx_file_close(&g_sd_debug_log_fx_file);

	/* Update state. */
	g_sd_debug_log_file_is_open = 0U;

	return 0;
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

	/* Write to file. */
	status = fx_file_write(&g_sd_debug_log_fx_file, (VOID*) data_ptr,
			(ULONG) data_length_bytes);
	if (status != FX_SUCCESS) {
		return -2;
	}

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
	(void) user_context_ptr;

	/* Media must be mounted. */
	if (g_sd_debug_log_media_ptr == NULL) {
		return -1;
	}

	/* Flush file if open. */
	if (g_sd_debug_log_file_is_open != 0U) {
		(void) fx_media_flush(g_sd_debug_log_media_ptr);
	}

	/* Flush media to commit cached FAT changes, etc. */
	(void) fx_media_flush(g_sd_debug_log_media_ptr);

	return 0;
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

	/* Perform rename. */
	status = fx_file_rename(g_sd_debug_log_media_ptr, (CHAR*) old_name_ptr,
			(CHAR*) new_name_ptr);
	if (status != FX_SUCCESS) {
		return -2;
	}

	return 0;
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

	/* Try opening the file for read. If it opens, it exists. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &temp_file,
			(CHAR*) file_name_ptr, FX_OPEN_FOR_READ);
	if (status == FX_SUCCESS) {
		/* Close immediately, we only wanted existence. */
		(void) fx_file_close(&temp_file);

		*exists_out_ptr = 1U;
		return 0;
	}

	/* If not found, return exists = 0. */
	if (status == FX_NOT_FOUND) {
		*exists_out_ptr = 0U;
		return 0;
	}

	/* Any other error is a real failure. */
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

	/* Default output to 0 so callers never read an uninitialized value. */
	*size_out_bytes_ptr = 0U;

	/* Open the file for read. This allows FileX to populate temp_file metadata. */
	status = fx_file_open(g_sd_debug_log_media_ptr, &temp_file,
			(CHAR*) file_name_ptr,
			FX_OPEN_FOR_READ);

	/* If the file does not exist, treat it as size 0 and succeed. */
	if (status == FX_NOT_FOUND) {
		return 0;
	}

	/* Any other open failure is a real error. */
	if (status != FX_SUCCESS) {
		return -2;
	}

	/* FileX exposes the file size in the FX_FILE control block after open. */
	file_size_bytes = temp_file.fx_file_current_file_size;

	/* Close the temporary file handle. */
	(void) fx_file_close(&temp_file);

	/* Return size back as uint32_t for our core. */
	*size_out_bytes_ptr = (uint32_t) file_size_bytes;

	return 0;
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

	/* Store media pointer so FileX adapter functions can use it. */
	g_sd_debug_log_media_ptr = media_ptr;

	/* Initialize rollover core with chosen names and threshold. */
	SdDebugLogCore_Initialize(&g_sd_debug_log_core_context,
			(uint32_t) SD_DEBUG_LOG_SERVICE_ROLLOVER_THRESHOLD_BYTES,
			"debug.log", "debug_");

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
	g_sd_debug_log_file_ops.exists = SdDebugLogService_FileX_Exists;
	g_sd_debug_log_file_ops.get_size = SdDebugLogService_FileX_GetSize;

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

	/* Allocate a log buffer from the block pool.
	 If the pool is empty, we drop the log line by returning the ThreadX error. */
	status = tx_block_allocate(&g_sd_debug_log_block_pool,
			(VOID**) &log_buffer_ptr, TX_NO_WAIT);
	if (status != TX_SUCCESS) {
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
		queue_status = tx_queue_receive(&g_sd_debug_log_queue, &message_word,
				TX_NO_WAIT);
		if (queue_status != TX_SUCCESS) {
			/* Queue empty or error, stop this cycle. */
			break;
		}

		/* Convert ULONG back into pointer. */
		log_buffer_ptr = (SdDebugLogService_LogBuffer*) message_word;
		if (log_buffer_ptr == NULL) {
			continue;
		}

		/* Write record to debug.log (rollover handled in core). */
		(void) SdDebugLogCore_WriteRecord(&g_sd_debug_log_core_context,
				&g_sd_debug_log_file_ops, log_buffer_ptr->payload,
				(uint32_t) log_buffer_ptr->length_bytes);

		/* Release buffer back to pool. */
		(void) tx_block_release((VOID*) log_buffer_ptr);
	}

	/* Keep Option A durability behavior. */
	(void) SdDebugLogCore_ForceFlushAndClose(&g_sd_debug_log_core_context,
			&g_sd_debug_log_file_ops);

	/* Optional: reopen immediately. */
	(void) SdDebugLogCore_OpenIfNeeded(&g_sd_debug_log_core_context,
			&g_sd_debug_log_file_ops);
}

void SdDebugLogService_ForceFlush(void) {
	/* Force flush and close of the active file. */
	(void) SdDebugLogCore_ForceFlushAndClose(&g_sd_debug_log_core_context,
			&g_sd_debug_log_file_ops);
}
