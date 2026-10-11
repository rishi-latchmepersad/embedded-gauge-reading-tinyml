/*
 * sd_debug_log_core.c
 *
 *  Created on: 12 Feb 2026
 *      Author: rishi_latchmepersad
 */

#include "sd_debug_log_core.h"

#include <string.h>

/*==============================================================================
 * Function: SdDebugLogCore_CopyStringBounded
 *
 * Purpose:
 *   Copy a C string into a fixed buffer with guaranteed null termination.
 *
 * Parameters:
 *   destination_ptr        - Destination buffer.
 *   destination_capacity   - Destination capacity in bytes.
 *   source_ptr             - Null terminated source string.
 *
 * Returns:
 *   None.
 *
 * Notes:
 *   This is used to safely copy names into fixed arrays in the context struct.
 *==============================================================================*/
static void SdDebugLogCore_CopyStringBounded(char *destination_ptr,
		uint32_t destination_capacity, const char *source_ptr) {
	uint32_t index = 0U;

	/* Validate destination buffer. */
	if ((destination_ptr == NULL) || (destination_capacity == 0U)) {
		return;
	}

	/* If source is NULL, treat it as empty. */
	if (source_ptr == NULL) {
		destination_ptr[0] = '\0';
		return;
	}

	/* Copy until either we hit '\0' or we run out of room (leave 1 for '\0'). */
	for (index = 0U; index < (destination_capacity - 1U); index++) {
		if (source_ptr[index] == '\0') {
			/* End of source string. */
			break;
		}

		destination_ptr[index] = source_ptr[index];
	}

	/* Always null terminate. */
	destination_ptr[index] = '\0';
}

/*==============================================================================
 * Function: SdDebugLogCore_FormatArchiveName
 *
 * Purpose:
 *   Format archive file name: "<prefix><4-digit index>.log".
 *   Example: "debug_0001.log".
 *
 * Parameters:
 *   prefix_ptr        - File prefix, for example "debug_".
 *   archive_index     - Archive index, 1..9999.
 *   output_ptr        - Output buffer.
 *   output_capacity   - Output capacity in bytes.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *
 * Notes:
 *   Implemented without sprintf to avoid pulling heavy libc into embedded builds.
 *==============================================================================*/
static int32_t SdDebugLogCore_FormatArchiveName(const char *prefix_ptr,
		uint16_t archive_index, char *output_ptr, uint32_t output_capacity) {
	char digits[5];
	uint16_t value = archive_index;

	/* Basic validation. We require a sane output buffer size. */
	if ((prefix_ptr == NULL) || (output_ptr == NULL)
			|| (output_capacity < 16U)) {
		return -1;
	}

	/* Convert index into 4 ASCII digits, zero padded. */
	digits[4] = '\0';
	digits[3] = (char) ('0' + (value % 10U));
	value = (uint16_t) (value / 10U);
	digits[2] = (char) ('0' + (value % 10U));
	value = (uint16_t) (value / 10U);
	digits[1] = (char) ('0' + (value % 10U));
	value = (uint16_t) (value / 10U);
	digits[0] = (char) ('0' + (value % 10U));

    const size_t prefix_length = strlen(prefix_ptr);
    if ((prefix_length + 9U) > output_capacity) { return -1; }
    memcpy(output_ptr, prefix_ptr, prefix_length);
    memcpy(output_ptr + prefix_length, digits, 4U);
    memcpy(output_ptr + prefix_length + 4U, ".log", 5U);

	return 0;
}

/**
 * @brief Shift bounded archives newest-first and return the new archive path.
 * @param context_ptr Log naming configuration; file_ops_ptr storage adapter.
 * @param output_ptr/output_capacity Destination for archive 0001.
 * @return Zero on success; preserve active log unchanged on any rotation error.
 * @sideeffects Removes only archive 0032, then shifts 0031..0001 one position.
 */
static int32_t SdDebugLogCore_FindNextAvailableArchiveName(
        SdDebugLogCore_Context *context_ptr,
        const SdDebugLogCore_FileOps *file_ops_ptr, char *output_ptr,
        uint32_t output_capacity) {
    char source[32], target[32];
    if ((context_ptr == NULL) || (file_ops_ptr == NULL) || (output_ptr == NULL)
            || (file_ops_ptr->remove == NULL)) { return -1; }
    if (SdDebugLogCore_FormatArchiveName(context_ptr->archive_file_prefix,
            SD_DEBUG_LOG_CORE_MAX_ARCHIVES, target, sizeof(target)) != 0) { return -2; }
    if (file_ops_ptr->remove(file_ops_ptr->user_context_ptr, target) != 0) { return -3; }
    for (uint16_t index = SD_DEBUG_LOG_CORE_MAX_ARCHIVES - 1U; index > 0U; index--) {
        uint8_t exists = 0U;
        if ((SdDebugLogCore_FormatArchiveName(context_ptr->archive_file_prefix,
                    index, source, sizeof(source)) != 0)
                || (file_ops_ptr->exists(file_ops_ptr->user_context_ptr, source, &exists) != 0)) { return -4; }
        if (exists != 0U) {
            if ((SdDebugLogCore_FormatArchiveName(context_ptr->archive_file_prefix,
                        index + 1U, target, sizeof(target)) != 0)
                    || (file_ops_ptr->rename(file_ops_ptr->user_context_ptr, source, target) != 0)) { return -5; }
        }
    }
    return SdDebugLogCore_FormatArchiveName(context_ptr->archive_file_prefix,
            1U, output_ptr, output_capacity);
}

/*==============================================================================
 * Function: SdDebugLogCore_RollOverActiveFile
 *
 * Purpose:
 *   Close active file, rename it to an archive name, then create a new active file.
 *
 * Returns:
 *   0 on success, nonzero on error.
 *
 * Notes:
 *   - This assumes the caller has already decided rollover is necessary.
 *   - Active file name remains constant, for example "debug.log".
 *   - Archive file name becomes something like "debug_0001.log".
 *==============================================================================*/
static int32_t SdDebugLogCore_RollOverActiveFile(
		SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr) {
	char archive_name[32];

	/* Validate inputs. */
	if ((context_ptr == NULL) || (file_ops_ptr == NULL)) {
		return -1;
	}

	/* Close current active file if it is open. */
	if (context_ptr->active_file_is_open != 0U) {
		int32_t flush_status;
		int32_t close_status;

        close_status = file_ops_ptr->close(file_ops_ptr->user_context_ptr);
        flush_status = file_ops_ptr->flush(file_ops_ptr->user_context_ptr);

		/* Update state to reflect closed file. */
		context_ptr->active_file_is_open = 0U;
		if ((flush_status != 0) || (close_status != 0)) {
			return -6;
		}
	}

	/* Find an archive name that does not collide with existing archives. */
	if (SdDebugLogCore_FindNextAvailableArchiveName(context_ptr, file_ops_ptr,
			archive_name, (uint32_t) sizeof(archive_name)) != 0) {
		return -2;
	}

	/* Rename active file to archive name. */
	if (file_ops_ptr->rename(file_ops_ptr->user_context_ptr,
			context_ptr->active_file_name, archive_name) != 0) {
		return -3;
	}

	/* Create a brand new active file. */
	if (file_ops_ptr->create_new(file_ops_ptr->user_context_ptr,
			context_ptr->active_file_name) != 0) {
		return -4;
	}

	/* Open the new active file for append. */
	if (file_ops_ptr->open_append(file_ops_ptr->user_context_ptr,
			context_ptr->active_file_name) != 0) {
		return -5;
	}

	/* Update internal state to match fresh file. */
	context_ptr->active_file_is_open = 1U;
	context_ptr->current_file_size_bytes = 0U;

	return 0;
}

void SdDebugLogCore_Initialize(SdDebugLogCore_Context *context_ptr,
		uint32_t rollover_threshold_bytes, const char *active_file_name_ptr,
		const char *archive_prefix_ptr) {
	/* Validate input pointer. */
	if (context_ptr == NULL) {
		return;
	}

	/* Clear everything so we start in a known state. */
	(void) memset(context_ptr, 0, sizeof(*context_ptr));

	/* Store configuration values. */
	context_ptr->rollover_threshold_bytes = rollover_threshold_bytes;

	/* First archive will be debug_0001.log by default. */
	context_ptr->next_archive_index = 1U;

	/* Copy names into fixed buffers with bounded copy. */
	SdDebugLogCore_CopyStringBounded(context_ptr->active_file_name,
			(uint32_t) sizeof(context_ptr->active_file_name),
			active_file_name_ptr);

	SdDebugLogCore_CopyStringBounded(context_ptr->archive_file_prefix,
			(uint32_t) sizeof(context_ptr->archive_file_prefix),
			archive_prefix_ptr);
}

int32_t SdDebugLogCore_OpenIfNeeded(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr) {
	uint32_t existing_size_bytes = 0U;

	/* Validate pointers. */
	if ((context_ptr == NULL) || (file_ops_ptr == NULL)) {
		return -1;
	}

	/* If already open, nothing to do. */
	if (context_ptr->active_file_is_open != 0U) {
		return 0;
	}

	/* Check if active file exists, create it if missing. */
	{
		uint8_t file_exists = 0U;

		/* Query filesystem for existence. */
		if (file_ops_ptr->exists(file_ops_ptr->user_context_ptr,
				context_ptr->active_file_name, &file_exists) != 0) {
			return -2;
		}

		/* If missing, create it. */
		if (file_exists == 0U) {
			if (file_ops_ptr->create_new(file_ops_ptr->user_context_ptr,
					context_ptr->active_file_name) != 0) {
				return -3;
			}
		}
	}

	/* Open the file for append. */
	if (file_ops_ptr->open_append(file_ops_ptr->user_context_ptr,
			context_ptr->active_file_name) != 0) {
		return -4;
	}

	/* Mark file open in our state. */
	context_ptr->active_file_is_open = 1U;

	/* Discover current size so rollover decisions are correct after reboot. */
	if (file_ops_ptr->get_size(file_ops_ptr->user_context_ptr,
			context_ptr->active_file_name, &existing_size_bytes) == 0) {
		context_ptr->current_file_size_bytes = existing_size_bytes;
	} else {
		/* An unknown size must not silently disable bounded rollover. */
        (void) SdDebugLogCore_ForceFlushAndClose(context_ptr, file_ops_ptr);
        return -6;
	}

    /* A power loss may leave a partial final row. Separate the new session so
     * its boot/fault record cannot be joined to that damaged historical line. */
    if (existing_size_bytes > 0U) {
        if (file_ops_ptr->write(file_ops_ptr->user_context_ptr, "\r\n", 2U) != 0) {
            (void) SdDebugLogCore_ForceFlushAndClose(context_ptr, file_ops_ptr);
            return -5;
        }
        context_ptr->current_file_size_bytes += 2U;
    }

	return 0;
}

int32_t SdDebugLogCore_WriteRecord(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr, const void *record_ptr,
		uint32_t record_length_bytes) {
	int32_t status = 0;

	/* Validate pointers. */
	if ((context_ptr == NULL) || (file_ops_ptr == NULL)
			|| (record_ptr == NULL)) {
		return -1;
	}

	/* Writing 0 bytes is a no-op. */
	if (record_length_bytes == 0U) {
		return 0;
	}

	/* Ensure active file is ready. */
	status = SdDebugLogCore_OpenIfNeeded(context_ptr, file_ops_ptr);
	if (status != 0) {
		return status;
	}

	/* If this record would push us over threshold, roll over first. */
	if (((uint64_t) context_ptr->current_file_size_bytes + record_length_bytes)
			> context_ptr->rollover_threshold_bytes) {
		status = SdDebugLogCore_RollOverActiveFile(context_ptr, file_ops_ptr);
		if (status != 0) {
			return status;
		}
	}

	/* Write record bytes to file. */
	if (file_ops_ptr->write(file_ops_ptr->user_context_ptr, record_ptr,
			record_length_bytes) != 0) {
        (void) SdDebugLogCore_ForceFlushAndClose(context_ptr, file_ops_ptr);
		return -5;
	}

	/* Update our file size tracker. */
	context_ptr->current_file_size_bytes += record_length_bytes;

	return 0;
}

int32_t SdDebugLogCore_ForceFlushAndClose(SdDebugLogCore_Context *context_ptr,
		const SdDebugLogCore_FileOps *file_ops_ptr) {
	int32_t flush_status;
	int32_t close_status;

	/* Validate pointers. */
	if ((context_ptr == NULL) || (file_ops_ptr == NULL)) {
		return -1;
	}

	/* A previous close may have failed while committing metadata. Flush cached
     * media state even without an open file so a retry cannot falsely succeed. */
	if (context_ptr->active_file_is_open == 0U) {
		return (file_ops_ptr->flush(file_ops_ptr->user_context_ptr) == 0) ? 0 : -2;
	}

    close_status = file_ops_ptr->close(file_ops_ptr->user_context_ptr);
    flush_status = file_ops_ptr->flush(file_ops_ptr->user_context_ptr);

	/* Update state. */
	context_ptr->active_file_is_open = 0U;

	return ((flush_status == 0) && (close_status == 0)) ? 0 : -2;
}

