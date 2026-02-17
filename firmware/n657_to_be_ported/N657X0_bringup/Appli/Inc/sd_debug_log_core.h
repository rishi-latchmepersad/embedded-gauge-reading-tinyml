#ifndef SD_DEBUG_LOG_CORE_H
#define SD_DEBUG_LOG_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*==============================================================================
 * Type: SdDebugLogCore_FileOps
 *
 * Purpose:
 *   Dependency injection interface for filesystem operations so the core logic
 *   can be unit tested without FileX.
 *
 * Notes:
 *   - On target, the adapter binds these to FileX calls.
 *   - In Unity tests, you bind these to fake functions.
 *==============================================================================*/
typedef struct {
	void *user_context_ptr;

	int32_t (*open_append)(void *user_context_ptr, const char *file_name_ptr);
	int32_t (*create_new)(void *user_context_ptr, const char *file_name_ptr);
	int32_t (*close)(void *user_context_ptr);

	int32_t (*write)(void *user_context_ptr, const void *data_ptr,
			uint32_t data_length_bytes);
	int32_t (*flush)(void *user_context_ptr);

	int32_t (*rename)(void *user_context_ptr, const char *old_name_ptr,
			const char *new_name_ptr);
	int32_t (*exists)(void *user_context_ptr, const char *file_name_ptr,
			uint8_t *exists_out_ptr);
	int32_t (*get_size)(void *user_context_ptr, const char *file_name_ptr,
			uint32_t *size_out_bytes_ptr);

} SdDebugLogCore_FileOps;

/*==============================================================================
 * Type: SdDebugLogCore_Context
 *
 * Purpose:
 *   Holds rollover state for the logging system.
 *==============================================================================*/
typedef struct {
	uint32_t rollover_threshold_bytes;
	uint32_t current_file_size_bytes;

	uint16_t next_archive_index;
	uint8_t active_file_is_open;

	char active_file_name[32];
	char archive_file_prefix[16];

} SdDebugLogCore_Context;

/*==============================================================================
 * Function: SdDebugLogCore_Initialize
 *
 * Purpose:
 *   Initialize the debug log core state.
 *
 * Parameters:
 *   context_ptr              - Core context to initialize.
 *   rollover_threshold_bytes - Rollover threshold in bytes.
 *   active_file_name_ptr     - Active file name, for example "debug.log".
 *   archive_prefix_ptr       - Prefix for archived logs, for example "debug_".
 *
 * Returns:
 *   None.
 *
 * Notes:
 *   Does not touch filesystem, only initializes state.
 *==============================================================================*/
void SdDebugLogCore_Initialize(SdDebugLogCore_Context *context_ptr,
		uint32_t rollover_threshold_bytes, const char *active_file_name_ptr,
		const char *archive_prefix_ptr);

/*==============================================================================
 * Function: SdDebugLogCore_OpenIfNeeded
 *
 * Purpose:
 *   Ensure the active log file is open for append, creating it if needed.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *==============================================================================*/
int32_t SdDebugLogCore_OpenIfNeeded(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr);

/*==============================================================================
 * Function: SdDebugLogCore_WriteRecord
 *
 * Purpose:
 *   Write a record to the active log and roll over if threshold would be exceeded.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *==============================================================================*/
int32_t SdDebugLogCore_WriteRecord(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr, const void *record_ptr,
		uint32_t record_length_bytes);

/*==============================================================================
 * Function: SdDebugLogCore_ForceFlushAndClose
 *
 * Purpose:
 *   Flush and close the active file if it is open.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *==============================================================================*/
int32_t SdDebugLogCore_ForceFlushAndClose(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr);

#ifdef __cplusplus
}
#endif

#endif /* SD_DEBUG_LOG_CORE_H */
