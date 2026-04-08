/*
 *******************************************************************************
 * @file    app_storage.c
 * @brief   Storage coordination helpers for FileX/RTC file naming.
 *******************************************************************************
 */

#include "app_storage.h"

#include <stdio.h>
#include <string.h>

#include "main.h"
#include "app_camera_config.h"
#include "app_camera_platform.h"
#include "app_filex.h"
#include "debug_console.h"
#include "threadx_utils.h"

static TX_EVENT_FLAGS_GROUP camera_storage_ready_flags;
static bool camera_storage_ready_sync_created = false;

/**
 * @brief Create the storage coordination event flags used by the capture path.
 * @retval TX_SUCCESS when the event flags are ready, otherwise a ThreadX status.
 */
UINT AppStorage_Init(void) {
	if (camera_storage_ready_sync_created) {
		return TX_SUCCESS;
	}

	const UINT ready_flags_status = tx_event_flags_create(
			&camera_storage_ready_flags, "camera_storage_ready");
	if (ready_flags_status != TX_SUCCESS) {
		return ready_flags_status;
	}

	camera_storage_ready_sync_created = true;
	return TX_SUCCESS;
}

/**
 * @brief Wait for FileX to finish mounting the SD card and creating the image directory.
 * @param timeout_ms Maximum time to wait before giving up.
 * @retval true when storage is ready for image writes.
 */
bool AppStorage_WaitForMediaReady(uint32_t timeout_ms) {
	const ULONG deadline_tick = tx_time_get()
			+ CameraPlatform_MillisecondsToTicks(timeout_ms);
	ULONG actual_flags = 0U;

	while (!AppFileX_IsMediaReady()) {
		if (tx_time_get() >= deadline_tick) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Timed out waiting for FileX media readiness.\r\n");
			return false;
		}

		if (camera_storage_ready_sync_created) {
			const UINT flag_status = tx_event_flags_get(
					&camera_storage_ready_flags,
					CAMERA_STORAGE_READY_EVENT_FLAG, TX_OR_CLEAR, &actual_flags,
					CameraPlatform_MillisecondsToTicks(50U));
			if ((flag_status == TX_SUCCESS)
					&& ((actual_flags & CAMERA_STORAGE_READY_EVENT_FLAG) != 0U)) {
				break;
			}
		} else {
			DelayMilliseconds_ThreadX(50U);
		}
	}

	return true;
}

/**
 * @brief Notify the storage thread that FileX media is ready.
 */
void AppStorage_NotifyMediaReady(void) {
	if (!camera_storage_ready_sync_created) {
		return;
	}

	(void) tx_event_flags_set(&camera_storage_ready_flags,
			CAMERA_STORAGE_READY_EVENT_FLAG, TX_OR);
}

/**
 * @brief Build a capture filename from the DS3231 time if available.
 */
bool AppStorage_BuildCaptureFileName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr) {
	CHAR rtc_stamp[32] = { 0 };
	int written = 0;
	const bool rtc_ready = App_Clock_GetCaptureTimestamp(rtc_stamp,
			sizeof(rtc_stamp));

	if ((file_name_ptr == NULL) || (file_name_length == 0U)
			|| (file_extension_ptr == NULL) || (file_extension_ptr[0] == '\0')) {
		return false;
	}

	if (rtc_ready) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Using RTC timestamp %s for capture name.\r\n",
				rtc_stamp);
		written = snprintf(file_name_ptr, (size_t) file_name_length,
				"capture_%s.%s", rtc_stamp, file_extension_ptr);
		return (written > 0)
				&& ((ULONG) written < file_name_length);
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] RTC timestamp unavailable; using numbered fallback.\r\n");
	return (AppFileX_GetNextCapturedImageName(file_name_ptr, file_name_length,
			file_extension_ptr) == FX_SUCCESS);
}
