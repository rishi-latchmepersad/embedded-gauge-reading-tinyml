/*
 *******************************************************************************
 * @file    app_storage.c
 * @brief   Storage coordination helpers for FileX/RTC file naming.
 *******************************************************************************
 */

#include "app_storage.h"

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

	DebugConsole_Printf(
			"[STORAGE] Waiting for FileX media readiness (timeout=%lu ms).\r\n",
			(unsigned long) timeout_ms);

	while (!AppFileX_IsMediaReady()) {
		if (tx_time_get() >= deadline_tick) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Timed out waiting for FileX media readiness.\r\n");
			return false;
		}

		if (camera_storage_ready_sync_created) {
			const UINT flag_status = tx_event_flags_get(
					&camera_storage_ready_flags,
					CAMERA_STORAGE_READY_EVENT_FLAG, TX_NO_WAIT, &actual_flags,
					TX_NO_WAIT);
			if ((flag_status == TX_SUCCESS)
					&& ((actual_flags & CAMERA_STORAGE_READY_EVENT_FLAG) != 0U)) {
				break;
			}
		}

		DelayMilliseconds_Cooperative(50U);
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
 * @brief Build a capture filename from the RTC time if available.
 *
 * The FileX helper keeps the timestamped naming logic in one place and falls
 * back to a numbered name only when the RTC is unavailable.
 */
bool AppStorage_BuildCaptureFileName(CHAR *file_name_ptr,
		ULONG file_name_length, const CHAR *file_extension_ptr) {
	if ((file_name_ptr == NULL) || (file_name_length == 0U)
			|| (file_extension_ptr == NULL) || (file_extension_ptr[0] == '\0')) {
		return false;
	}

	return (AppFileX_GetNextCapturedImageName(file_name_ptr, file_name_length,
			file_extension_ptr) == FX_SUCCESS);
}
