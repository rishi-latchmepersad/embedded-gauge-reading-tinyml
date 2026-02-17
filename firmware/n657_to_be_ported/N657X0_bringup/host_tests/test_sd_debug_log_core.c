/*==============================================================================
 * File: test_sd_debug_log_core.c
 *
 * Purpose:
 *   Unity unit tests for the SdDebugLogCore module.
 *
 * Approach:
 *   - We do NOT use FileX or ThreadX here.
 *   - We inject a fake filesystem using SdDebugLogCore_FileOps.
 *   - We verify rollover behavior, file naming, and size tracking.
 *
 * Notes:
 *   This test file is intended to compile and run on a host PC as part of your
 *   Unity test target.
 *==============================================================================*/

#include "unity.h"
#include "sd_debug_log_core.h"

#include <string.h>
#include <stdint.h>

/*==============================================================================
 * Type: FakeFs_Context
 *
 * Purpose:
 *   Minimal fake filesystem state used by the injected file operations.
 *
 * Design:
 *   We track:
 *   - Whether "debug.log" exists and its size.
 *   - Whether "debug_0001.log" exists and its size.
 *   - Whether a file handle is "open" (simulated).
 *
 * Notes:
 *   This is intentionally small; add more archive files if you expand tests.
 *==============================================================================*/
typedef struct {
	uint8_t debug_log_exists;
	uint32_t debug_log_size_bytes;

	uint8_t debug_0001_exists;
	uint32_t debug_0001_size_bytes;

	uint8_t file_is_open;

} FakeFs_Context;

/*==============================================================================
 * Function: FakeFs_Reset
 *
 * Purpose:
 *   Reset fake filesystem to a known baseline state.
 *
 * Parameters:
 *   fake_fs_ptr - Fake filesystem context to clear.
 *
 * Returns:
 *   None.
 *==============================================================================*/
static void FakeFs_Reset(FakeFs_Context *fake_fs_ptr) {
	if (fake_fs_ptr == NULL) {
		return;
	}

	(void) memset(fake_fs_ptr, 0, sizeof(*fake_fs_ptr));
}

/*==============================================================================
 * Function: FakeFs_OpenAppend
 *
 * Purpose:
 *   Fake implementation of "open active file for append".
 *
 * Behavior:
 *   - Marks file handle open.
 *   - Does not modify sizes.
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t FakeFs_OpenAppend(void *user_context_ptr,
		const char *file_name_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	/* In tests we do not require file_name_ptr, but validate it anyway. */
	if ((fake_fs_ptr == NULL) || (file_name_ptr == NULL)) {
		return -1;
	}

	/* Simulate a successful open. */
	fake_fs_ptr->file_is_open = 1U;

	return 0;
}

/*==============================================================================
 * Function: FakeFs_CreateNew
 *
 * Purpose:
 *   Fake implementation of "create file if missing".
 *
 * Behavior:
 *   - If "debug.log": mark exists and set size to 0.
 *   - Otherwise: treat as unsupported in this simplified fake.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t FakeFs_CreateNew(void *user_context_ptr,
		const char *file_name_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if ((fake_fs_ptr == NULL) || (file_name_ptr == NULL)) {
		return -1;
	}

	/* Only support creating the active log in this minimal fake. */
	if (strcmp(file_name_ptr, "debug.log") == 0) {
		fake_fs_ptr->debug_log_exists = 1U;
		fake_fs_ptr->debug_log_size_bytes = 0U;
		return 0;
	}

	return -2;
}

/*==============================================================================
 * Function: FakeFs_Close
 *
 * Purpose:
 *   Fake implementation of "close active file".
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t FakeFs_Close(void *user_context_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if (fake_fs_ptr == NULL) {
		return -1;
	}

	/* Mark file handle closed. */
	fake_fs_ptr->file_is_open = 0U;

	return 0;
}

/*==============================================================================
 * Function: FakeFs_Write
 *
 * Purpose:
 *   Fake implementation of "write bytes to active file".
 *
 * Behavior:
 *   - Requires debug.log to exist (in this fake).
 *   - Adds data_length_bytes to debug.log size.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t FakeFs_Write(void *user_context_ptr, const void *data_ptr,
		uint32_t data_length_bytes) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if ((fake_fs_ptr == NULL) || (data_ptr == NULL)) {
		return -1;
	}

	/* In real life, you could allow writes without existence, but we keep it strict. */
	if (fake_fs_ptr->debug_log_exists == 0U) {
		return -2;
	}

	/* Increase the active file size by the write size. */
	fake_fs_ptr->debug_log_size_bytes += data_length_bytes;

	return 0;
}

/*==============================================================================
 * Function: FakeFs_Flush
 *
 * Purpose:
 *   Fake implementation of "flush".
 *
 * Notes:
 *   For this test we do not model caching, so flush is always OK.
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t FakeFs_Flush(void *user_context_ptr) {
	(void) user_context_ptr;

	return 0;
}

/*==============================================================================
 * Function: FakeFs_Rename
 *
 * Purpose:
 *   Fake implementation of "rename file".
 *
 * Behavior:
 *   - Supports only "debug.log" -> "debug_0001.log" in this minimal fake.
 *   - Moves existence and size from active to archive.
 *
 * Returns:
 *   0 on success, nonzero on failure.
 *==============================================================================*/
static int32_t FakeFs_Rename(void *user_context_ptr, const char *old_name_ptr,
		const char *new_name_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if ((fake_fs_ptr == NULL) || (old_name_ptr == NULL)
			|| (new_name_ptr == NULL)) {
		return -1;
	}

	/* Only handle one rename path in this minimal fake. */
	if ((strcmp(old_name_ptr, "debug.log") == 0)
			&& (strcmp(new_name_ptr, "debug_0001.log") == 0)) {
		/* Move active file into archive slot. */
		fake_fs_ptr->debug_0001_exists = fake_fs_ptr->debug_log_exists;
		fake_fs_ptr->debug_0001_size_bytes = fake_fs_ptr->debug_log_size_bytes;

		/* Clear active file state as if it no longer exists after rename. */
		fake_fs_ptr->debug_log_exists = 0U;
		fake_fs_ptr->debug_log_size_bytes = 0U;

		return 0;
	}

	return -2;
}

/*==============================================================================
 * Function: FakeFs_Exists
 *
 * Purpose:
 *   Fake implementation of "does this file exist".
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t FakeFs_Exists(void *user_context_ptr, const char *file_name_ptr,
		uint8_t *exists_out_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if ((fake_fs_ptr == NULL) || (file_name_ptr == NULL)
			|| (exists_out_ptr == NULL)) {
		return -1;
	}

	/* Report existence for active log. */
	if (strcmp(file_name_ptr, "debug.log") == 0) {
		*exists_out_ptr = fake_fs_ptr->debug_log_exists;
		return 0;
	}

	/* Report existence for archive log #1. */
	if (strcmp(file_name_ptr, "debug_0001.log") == 0) {
		*exists_out_ptr = fake_fs_ptr->debug_0001_exists;
		return 0;
	}

	/* Unknown file names do not exist in this fake. */
	*exists_out_ptr = 0U;

	return 0;
}

/*==============================================================================
 * Function: FakeFs_GetSize
 *
 * Purpose:
 *   Fake implementation of "get file size".
 *
 * Returns:
 *   0 on success.
 *==============================================================================*/
static int32_t FakeFs_GetSize(void *user_context_ptr, const char *file_name_ptr,
		uint32_t *size_out_bytes_ptr) {
	FakeFs_Context *fake_fs_ptr = (FakeFs_Context*) user_context_ptr;

	if ((fake_fs_ptr == NULL) || (file_name_ptr == NULL)
			|| (size_out_bytes_ptr == NULL)) {
		return -1;
	}

	/* Size of active log. */
	if (strcmp(file_name_ptr, "debug.log") == 0) {
		*size_out_bytes_ptr = fake_fs_ptr->debug_log_size_bytes;
		return 0;
	}

	/* Size of archive log #1. */
	if (strcmp(file_name_ptr, "debug_0001.log") == 0) {
		*size_out_bytes_ptr = fake_fs_ptr->debug_0001_size_bytes;
		return 0;
	}

	/* Unknown files have size 0 in this fake. */
	*size_out_bytes_ptr = 0U;

	return 0;
}

/*==============================================================================
 * Function: Test_BindFileOps
 *
 * Purpose:
 *   Helper to bind fake filesystem operations into the FileOps struct.
 *
 * Parameters:
 *   ops_ptr     - FileOps struct to fill.
 *   fake_fs_ptr - Fake FS context to pass through user_context_ptr.
 *
 * Returns:
 *   None.
 *==============================================================================*/
static void Test_BindFileOps(SdDebugLogCore_FileOps *ops_ptr,
		FakeFs_Context *fake_fs_ptr) {
	if ((ops_ptr == NULL) || (fake_fs_ptr == NULL)) {
		return;
	}

	/* Clear ops to avoid uninitialized function pointers. */
	(void) memset(ops_ptr, 0, sizeof(*ops_ptr));

	/* Provide context to all fake functions. */
	ops_ptr->user_context_ptr = (void*) fake_fs_ptr;

	/* Bind fake functions. */
	ops_ptr->open_append = FakeFs_OpenAppend;
	ops_ptr->create_new = FakeFs_CreateNew;
	ops_ptr->close = FakeFs_Close;
	ops_ptr->write = FakeFs_Write;
	ops_ptr->flush = FakeFs_Flush;
	ops_ptr->rename = FakeFs_Rename;
	ops_ptr->exists = FakeFs_Exists;
	ops_ptr->get_size = FakeFs_GetSize;
}

/*==============================================================================
 * Test: test_rollover_occurs_when_record_would_exceed_threshold
 *
 * Purpose:
 *   Verify the rollover behavior:
 *   - Write some bytes below threshold, no rollover.
 *   - Write next chunk that would exceed threshold, rollover should occur first.
 *
 * Expected:
 *   - The pre-roll data ends up in debug_0001.log.
 *   - The post-roll data ends up in a fresh debug.log.
 *==============================================================================*/
void test_rollover_occurs_when_record_would_exceed_threshold(void) {
	FakeFs_Context fake_fs;
	SdDebugLogCore_FileOps ops;
	SdDebugLogCore_Context core;

	const char record_a[] = "AAAA"; /* 4 bytes */
	const char record_b[] = "BBBB"; /* 4 bytes */

	/* Reset fake FS to known state: no files exist. */
	FakeFs_Reset(&fake_fs);

	/* Bind fake FS operations. */
	Test_BindFileOps(&ops, &fake_fs);

	/* Initialize core with a small threshold for easy testing (6 bytes). */
	SdDebugLogCore_Initialize(&core, 6U, "debug.log", "debug_");

	/* First write: 4 bytes, should create/open debug.log and write into it. */
	TEST_ASSERT_EQUAL_INT32(0,
			SdDebugLogCore_WriteRecord(&core, &ops, record_a, 4U));

	/* Validate active file exists and contains 4 bytes. */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.debug_log_exists);
	TEST_ASSERT_EQUAL_UINT32(4U, fake_fs.debug_log_size_bytes);

	/* Second write: another 4 bytes.
	 4 + 4 = 8 which exceeds threshold 6, so core should rollover FIRST. */
	TEST_ASSERT_EQUAL_INT32(0,
			SdDebugLogCore_WriteRecord(&core, &ops, record_b, 4U));

	/* After rollover, the first 4 bytes should be in archive debug_0001.log. */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.debug_0001_exists);
	TEST_ASSERT_EQUAL_UINT32(4U, fake_fs.debug_0001_size_bytes);

	/* Active log should exist again (fresh), and contain record_b (4 bytes). */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.debug_log_exists);
	TEST_ASSERT_EQUAL_UINT32(4U, fake_fs.debug_log_size_bytes);
}

/*==============================================================================
 * Test: test_open_if_needed_creates_file_if_missing
 *
 * Purpose:
 *   Verify that OpenIfNeeded creates the active file if it does not exist.
 *==============================================================================*/
void test_open_if_needed_creates_file_if_missing(void) {
	FakeFs_Context fake_fs;
	SdDebugLogCore_FileOps ops;
	SdDebugLogCore_Context core;

	FakeFs_Reset(&fake_fs);
	Test_BindFileOps(&ops, &fake_fs);

	SdDebugLogCore_Initialize(&core, 100U, "debug.log", "debug_");

	/* At start, file does not exist. */
	TEST_ASSERT_EQUAL_UINT8(0U, fake_fs.debug_log_exists);

	/* OpenIfNeeded should create file (via create_new) and open it. */
	TEST_ASSERT_EQUAL_INT32(0, SdDebugLogCore_OpenIfNeeded(&core, &ops));

	/* Validate it now exists. */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.debug_log_exists);

	/* Our fake open just marks file_is_open. */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.file_is_open);
}

/*==============================================================================
 * Test: test_force_flush_and_close_closes_when_open
 *
 * Purpose:
 *   Verify ForceFlushAndClose closes the file if open, and is safe if called twice.
 *==============================================================================*/
void test_force_flush_and_close_closes_when_open(void) {
	FakeFs_Context fake_fs;
	SdDebugLogCore_FileOps ops;
	SdDebugLogCore_Context core;

	const char record_a[] = "AAAA";

	FakeFs_Reset(&fake_fs);
	Test_BindFileOps(&ops, &fake_fs);

	SdDebugLogCore_Initialize(&core, 100U, "debug.log", "debug_");

	/* Write a record to force open. */
	TEST_ASSERT_EQUAL_INT32(0,
			SdDebugLogCore_WriteRecord(&core, &ops, record_a, 4U));

	/* Ensure our fake shows open. */
	TEST_ASSERT_EQUAL_UINT8(1U, fake_fs.file_is_open);

	/* Force close should close it. */
	TEST_ASSERT_EQUAL_INT32(0, SdDebugLogCore_ForceFlushAndClose(&core, &ops));
	TEST_ASSERT_EQUAL_UINT8(0U, fake_fs.file_is_open);

	/* Calling again should be safe and still succeed. */
	TEST_ASSERT_EQUAL_INT32(0, SdDebugLogCore_ForceFlushAndClose(&core, &ops));
	TEST_ASSERT_EQUAL_UINT8(0U, fake_fs.file_is_open);
}
