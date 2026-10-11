/* USER CODE BEGIN Header */
/**
 *******************************************************************************
 * @file    app_image_cleanup.c
 * @brief   Background cleanup helpers for captured gauge images.
 *******************************************************************************
 */
/* USER CODE END Header */

#include "app_image_cleanup.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_memory_budget.h"
#include "app_filex.h"
#include "main.h"
#include "sd_debug_log_service.h"
#include "ds3231_clock.h"
#include "debug_console.h"
#include "threadx_utils.h"

/*
 * Keep the cleanup worker lean and bounded. The goal is to prune duplicate
 * captures without taking much RAM or time away from the camera and AI path.
 */
#define APP_IMAGE_CLEANUP_MAX_BUCKETS            1024U
#define APP_IMAGE_CLEANUP_CAPTURE_NAME_LENGTH      64U
#define APP_IMAGE_CLEANUP_THREAD_PRIORITY          16U
#define APP_IMAGE_CLEANUP_PERIOD_MS           1800000U
#define APP_IMAGE_CLEANUP_SERVICE_POLL_MS         1000U
#define APP_IMAGE_CLEANUP_RECENT_WINDOW_SECONDS    86400U
#define APP_IMAGE_CLEANUP_FILE_NAME_BUFFER_LENGTH FX_MAX_LONG_NAME_LEN
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef struct {
	bool used;
	uint64_t bucket_key;
	uint64_t timestamp_key;
	CHAR file_name[APP_IMAGE_CLEANUP_CAPTURE_NAME_LENGTH];
} AppImageCleanup_BucketRecord;

typedef struct {
	UINT year;
	UINT month;
	UINT day;
	UINT hour;
	UINT minute;
	UINT second;
} AppImageCleanup_Timestamp;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static TX_THREAD app_image_cleanup_thread;
static ULONG app_image_cleanup_thread_stack[IMAGE_CLEANUP_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool app_image_cleanup_thread_created = false;
/* A fixed batch bounds RAM while still visiting arbitrarily many hour folders. */
#define APP_IMAGE_CLEANUP_DIRECTORY_BATCH 64U
#define APP_IMAGE_CLEANUP_SCHEDULE_MAGIC 0x434C4E33U
static CHAR app_image_cleanup_directory_batch[APP_IMAGE_CLEANUP_DIRECTORY_BATCH][14];
static CHAR app_image_cleanup_hour_prefix[14];
static AppImageCleanup_BucketRecord app_image_cleanup_bucket_records
		[APP_IMAGE_CLEANUP_MAX_BUCKETS];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
UINT AppImageCleanup_SetBootTick(ULONG boot_tick);
static VOID AppImageCleanupThread_Entry(ULONG thread_input);
static UINT AppImageCleanup_RunDirectory(const CHAR *directory_path);
static UINT AppImageCleanup_RunSweep(void);
static bool AppImageCleanup_ParseHour(const CHAR *name,
        AppImageCleanup_Timestamp *timestamp_ptr);
static bool AppImageCleanup_IsDigit(char ch);
static bool AppImageCleanup_ParseTwoDigits(const CHAR *text_ptr,
		UINT *value_out);
static bool AppImageCleanup_ParseFourDigits(const CHAR *text_ptr,
		UINT *value_out);
static bool AppImageCleanup_IsPlausibleTimestamp(UINT year, UINT month,
		UINT day, UINT hour, UINT minute, UINT second);
static bool AppImageCleanup_ParseTimestampFromName(const CHAR *file_name_ptr,
		AppImageCleanup_Timestamp *timestamp_ptr);
static bool AppImageCleanup_ParseTimestampText(const CHAR *text_ptr,
		AppImageCleanup_Timestamp *timestamp_ptr);
static bool AppImageCleanup_ResolveTimestamp(
		const CHAR *file_name_ptr, UINT year, UINT month, UINT day, UINT hour,
		UINT minute, UINT second, AppImageCleanup_Timestamp *timestamp_ptr);
static bool AppImageCleanup_ReadCurrentTimestamp(
		AppImageCleanup_Timestamp *timestamp_ptr);
static bool AppImageCleanup_IsLeapYear(UINT year);
static uint32_t AppImageCleanup_DaysBeforeMonth(UINT year, UINT month);
static bool AppImageCleanup_ToSecondsSinceEpoch(
		const AppImageCleanup_Timestamp *timestamp_ptr, uint64_t *seconds_out);
static bool AppImageCleanup_IsWithinLast24Hours(
		const AppImageCleanup_Timestamp *file_timestamp_ptr,
		const AppImageCleanup_Timestamp *current_timestamp_ptr);
static uint64_t AppImageCleanup_BuildTimestampKey(
		const AppImageCleanup_Timestamp *timestamp_ptr);
static uint64_t AppImageCleanup_BuildBucketKey(
		const AppImageCleanup_Timestamp *timestamp_ptr);
static AppImageCleanup_BucketRecord *AppImageCleanup_FindBucket(
		uint64_t bucket_key);
static AppImageCleanup_BucketRecord *AppImageCleanup_AllocateBucket(
		uint64_t bucket_key);
static void AppImageCleanup_ResetBuckets(void);
/* USER CODE END PFP */

/**
 * @brief Start the low-priority image cleanup thread.
 * @retval TX_SUCCESS when the thread already exists or is created.
 */
UINT AppImageCleanup_Start(void) {
	if (app_image_cleanup_thread_created) {
		return TX_SUCCESS;
	}

	{
		const UINT create_status = tx_thread_create(&app_image_cleanup_thread,
				"image_cleanup", AppImageCleanupThread_Entry, 0U,
				app_image_cleanup_thread_stack,
				sizeof(app_image_cleanup_thread_stack),
				APP_IMAGE_CLEANUP_THREAD_PRIORITY,
				APP_IMAGE_CLEANUP_THREAD_PRIORITY, TX_NO_TIME_SLICE,
				TX_AUTO_START);

		if (create_status != TX_SUCCESS) {
			return create_status;
		}
	}

	app_image_cleanup_thread_created = true;
	return TX_SUCCESS;
}

/**
 * @brief Retain boot-call compatibility; cleanup scheduling uses retained RTC time.
 * @param boot_tick System tick captured near boot.
 * @retval TX_SUCCESS always.
 */
UINT AppImageCleanup_SetBootTick(ULONG boot_tick) {
    /* The schedule now uses retained wall time; keep this API for boot callers. */
    (void) boot_tick;
    return TX_SUCCESS;
}

/**
 * @brief Service deferred capture flushes; pruning runs at the stopped-camera boundary.
 * @param thread_input Unused ThreadX input. Never deletes files on the camera path.
 */
static VOID AppImageCleanupThread_Entry(ULONG thread_input) {
    (void) thread_input;
    while (1) {
        (void) AppFileX_ServiceCaptureMediaFlush();
        DelayMilliseconds_Cooperative(APP_IMAGE_CLEANUP_SERVICE_POLL_MS);
    }
}

/**
 * @brief Run due maintenance after camera/AI ownership has ended, before sleep.
 * @return FileX/ThreadX status; an invalid RTC preserves every capture.
 * @sideeffects Prunes duplicates and retains a completed RTC deadline in BKP20..22.
 */
UINT AppImageCleanup_RunDueBeforeSleep(void) {
    AppImageCleanup_Timestamp now = {0};
    uint64_t seconds = 0U;
    if (!AppFileX_IsMediaReady() || !AppImageCleanup_ReadCurrentTimestamp(&now)
            || !AppImageCleanup_IsPlausibleTimestamp(now.year, now.month,
                    now.day, now.hour, now.minute, now.second)
            || !AppImageCleanup_ToSecondsSinceEpoch(&now, &seconds)) {
        (void) SdDebugLogService_EnqueueLine("[IMAGE][CLEANUP] skipped: media/RTC unavailable");
        return FX_SUCCESS;
    }
    /* These three registers are separate from the fault snapshot in BKP23..31.
     * Complement + magic reject a partial write or uninitialized backup domain. */
    HAL_PWR_EnableBkUpAccess();
    const uint32_t previous = TAMP->BKP21R;
    if ((TAMP->BKP20R == APP_IMAGE_CLEANUP_SCHEDULE_MAGIC)
            && (TAMP->BKP22R == ~previous) && (seconds >= previous)
            && ((seconds - previous) < (APP_IMAGE_CLEANUP_PERIOD_MS / 1000U))) {
        return FX_SUCCESS;
    }
    (void) SdDebugLogService_EnqueueLine("[IMAGE][CLEANUP] sweep-start camera-stopped=1");
    const UINT status = AppImageCleanup_RunSweep();
    char line[112];
    (void) snprintf(line, sizeof(line), "[IMAGE][CLEANUP] sweep-end status=%lu",
            (unsigned long) status);
    (void) DebugConsole_WriteString(line);
    (void) DebugConsole_WriteString("\r\n");
    (void) SdDebugLogService_EnqueueLine(line);
    if (status == FX_SUCCESS) {
        /* Advance only after all deletes and the media flush succeeded. A reset
         * during cleanup retries the idempotent sweep on the next safe boundary. */
        TAMP->BKP20R = 0U;
        TAMP->BKP21R = (uint32_t) seconds;
        TAMP->BKP22R = ~(uint32_t) seconds;
        __DSB();
        TAMP->BKP20R = APP_IMAGE_CLEANUP_SCHEDULE_MAGIC;
    }
    return status;
}

/**
 * @brief Parse exactly YYYY-MM-DD_HH; reject unknown folders and impossible dates.
 * @return true with minute/second zero when this is a supported hour directory.
 */
static bool AppImageCleanup_ParseHour(const CHAR *name,
        AppImageCleanup_Timestamp *timestamp_ptr) {
    CHAR text[20];
    if ((name == NULL) || (strlen(name) != 13U)) {
        return false;
    }
    (void) snprintf(text, sizeof(text), "%.13s-00-00", name);
    return AppImageCleanup_ParseTimestampText(text, timestamp_ptr)
            && AppImageCleanup_IsPlausibleTimestamp(timestamp_ptr->year,
                    timestamp_ptr->month, timestamp_ptr->day,
                    timestamp_ptr->hour, 0U, 0U);
}

/**
 * @brief Traverse root captures and hourly folders in bounded sorted batches.
 * @return First traversal/pruning failure. Unknown and recent folders are preserved.
 * @sideeffects Locks media for enumeration only; subdirectories use fresh cursors.
 */
static UINT AppImageCleanup_RunSweep(void) {
    FX_MEDIA *media = AppFileX_GetMediaHandle();
    AppImageCleanup_Timestamp now = {0};
    CHAR cursor[14] = {0};
    uint64_t now_seconds = 0U;
    if (!AppImageCleanup_ReadCurrentTimestamp(&now)
            || !AppImageCleanup_ToSecondsSinceEpoch(&now, &now_seconds)) {
        return FX_SUCCESS;
    }
    app_image_cleanup_hour_prefix[0] = '\0';
    UINT status = AppImageCleanup_RunDirectory("/captured_images");
    if ((status != FX_SUCCESS) && (status != FX_NOT_FOUND)
            && (status != FX_INVALID_PATH)) {
        return status;
    }
    for (;;) {
        CHAR name[FX_MAX_LONG_NAME_LEN];
        UINT attributes, year, month, day, hour, minute, second;
        ULONG size;
        UINT count = 0U;
        status = AppFileX_AcquireMediaLock();
        if (status != TX_SUCCESS) { return status; }
        status = fx_directory_default_set(media, "/captured_images");
        if (status == FX_SUCCESS) {
            status = fx_directory_first_full_entry_find(media, name, &attributes,
                    &size, &year, &month, &day, &hour, &minute, &second);
            while (status == FX_SUCCESS) {
                AppImageCleanup_Timestamp folder_time = {0};
                uint64_t folder_seconds = 0U;
                if ((attributes & FX_DIRECTORY) && (strcmp(name, cursor) > 0)
                        && AppImageCleanup_ParseHour(name, &folder_time)
                        && AppImageCleanup_ToSecondsSinceEpoch(&folder_time, &folder_seconds)
                        && (folder_seconds < now_seconds)
                        && ((now_seconds - folder_seconds) > APP_IMAGE_CLEANUP_RECENT_WINDOW_SECONDS)) {
                    /* Keep the lexically first 64 eligible folders after cursor.
                     * The next batch rescans the parent, avoiding FileX's shared
                     * enumeration cursor being overwritten by a child scan. */
                    UINT pos = 0U;
                    while ((pos < count) && (strcmp(app_image_cleanup_directory_batch[pos], name) < 0)) { pos++; }
                    if (pos < APP_IMAGE_CLEANUP_DIRECTORY_BATCH) {
                        UINT last = (count < APP_IMAGE_CLEANUP_DIRECTORY_BATCH) ? count++ : (count - 1U);
                        while (last > pos) {
                            memcpy(app_image_cleanup_directory_batch[last], app_image_cleanup_directory_batch[last - 1U], 14U);
                            last--;
                        }
                        memcpy(app_image_cleanup_directory_batch[pos], name, 14U);
                    }
                }
                status = fx_directory_next_full_entry_find(media, name, &attributes,
                        &size, &year, &month, &day, &hour, &minute, &second);
            }
        }
        (void) fx_directory_default_set(media, FX_NULL);
        AppFileX_ReleaseMediaLock();
        if ((status == FX_NOT_FOUND) || (status == FX_INVALID_PATH)) { return FX_SUCCESS; }
        if (status != FX_NO_MORE_ENTRIES) { return status; }
        if (count == 0U) { return FX_SUCCESS; }
        for (UINT i = 0U; i < count; i++) {
            CHAR path[40];
            memcpy(app_image_cleanup_hour_prefix, app_image_cleanup_directory_batch[i], 14U);
            (void) snprintf(path, sizeof(path), "/captured_images/%s", app_image_cleanup_hour_prefix);
            status = AppImageCleanup_RunDirectory(path);
            if (status != FX_SUCCESS) { return status; }
        }
        memcpy(cursor, app_image_cleanup_directory_batch[count - 1U], 14U);
    }
}

/**
 * @brief Prune one directory; reset its bucket table and restore root on exit.
 * @param directory_path Absolute legacy-root or hourly capture directory.
 * @retval FX_SUCCESS when the sweep completed or found nothing to prune.
 */
static UINT AppImageCleanup_RunDirectory(const CHAR *directory_path) {
	FX_MEDIA *media_ptr = AppFileX_GetMediaHandle();
	AppImageCleanup_Timestamp current_timestamp = { 0 };
	CHAR file_name[APP_IMAGE_CLEANUP_FILE_NAME_BUFFER_LENGTH] = { 0 };
	UINT attributes = 0U;
	ULONG file_size = 0U;
	UINT year = 0U;
	UINT month = 0U;
	UINT day = 0U;
	UINT hour = 0U;
	UINT minute = 0U;
	UINT second = 0U;
	UINT iter_status = FX_SUCCESS;
	UINT flush_status = FX_SUCCESS;
	ULONG scanned_files = 0U;
	ULONG recent_files = 0U;
	ULONG deleted_files = 0U;
	ULONG kept_files = 0U;
	ULONG skipped_files = 0U;
	bool delete_failed = false;
	bool sweep_overflow = false;

	if (media_ptr == NULL) {
		return FX_MEDIA_NOT_OPEN;
	}

	if (!AppImageCleanup_ReadCurrentTimestamp(&current_timestamp)) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] RTC unavailable; skipping pruning for safety.\r\n");
		return FX_SUCCESS;
	}

	if (current_timestamp.year == 2000U) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] RTC year=2000 (power-on default); pruning disabled.\r\n");
		return FX_SUCCESS;
	}

	if (!AppImageCleanup_IsPlausibleTimestamp(current_timestamp.year,
			current_timestamp.month, current_timestamp.day,
			current_timestamp.hour, current_timestamp.minute,
			current_timestamp.second)) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] RTC year=%lu is not in the pruning window; skipping pruning.\r\n",
				(unsigned long) current_timestamp.year);
		return FX_SUCCESS;
	}

	if (AppFileX_AcquireMediaLock() != TX_SUCCESS) {
		return TX_MUTEX_ERROR;
	}

	AppImageCleanup_ResetBuckets();

	iter_status = fx_directory_default_set(media_ptr,
			(CHAR *) directory_path);
	if (iter_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] Failed to select %s, status=%lu.\r\n",
				directory_path,
				(unsigned long) iter_status);
		goto cleanup;
	}


	/*
	 * Pass 1: discover the newest file in each 10-minute bucket.
	 * The capture filenames are RTC stamped, and FileX also reports entry
	 * timestamps, so we use whichever is valid for the file.
	 */
	iter_status = fx_directory_first_full_entry_find(media_ptr, file_name,
			&attributes, &file_size, &year, &month, &day, &hour, &minute,
			&second);
	while (iter_status == FX_SUCCESS) {
		AppImageCleanup_Timestamp timestamp = { 0 };

		scanned_files++;

		if (((attributes & FX_DIRECTORY) == 0U)
                && (file_size > 0U)
				&& AppImageCleanup_ResolveTimestamp(file_name, year, month, day,
						hour, minute, second, &timestamp)) {
			if (AppImageCleanup_IsWithinLast24Hours(&timestamp,
					&current_timestamp)) {
				iter_status = fx_directory_next_full_entry_find(media_ptr,
						file_name, &attributes, &file_size, &year, &month, &day,
						&hour, &minute, &second);
				continue;
			}

			const uint64_t bucket_key = AppImageCleanup_BuildBucketKey(
					&timestamp);
			const uint64_t timestamp_key = AppImageCleanup_BuildTimestampKey(
					&timestamp);
			AppImageCleanup_BucketRecord *record =
					AppImageCleanup_FindBucket(bucket_key);

			if (record == NULL) {
				record = AppImageCleanup_AllocateBucket(bucket_key);
				if (record == NULL) {
					sweep_overflow = true;
					skipped_files++;
				} else {
					record->timestamp_key = timestamp_key;
					(void) snprintf(record->file_name,
							sizeof(record->file_name), "%.*s",
							(int)(sizeof(record->file_name) - 1U), file_name);
				}
			} else if (timestamp_key >= record->timestamp_key) {
				record->timestamp_key = timestamp_key;
				(void) snprintf(record->file_name, sizeof(record->file_name),
						"%.*s", (int)(sizeof(record->file_name) - 1U), file_name);
			}
		} else {
			skipped_files++;
		}

		iter_status = fx_directory_next_full_entry_find(media_ptr, file_name,
				&attributes, &file_size, &year, &month, &day, &hour, &minute,
				&second);
	}

	if ((iter_status != FX_NO_MORE_ENTRIES) && (iter_status != FX_SUCCESS)) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] Directory scan failed during discovery, status=%lu.\r\n",
				(unsigned long) iter_status);
		goto cleanup;
	}

	/*
	 * Pass 2: delete everything in the bucket except the recorded winner.
	 * We keep the newest capture per 10-minute slice.
	 */
	iter_status = fx_directory_first_full_entry_find(media_ptr, file_name,
			&attributes, &file_size, &year, &month, &day, &hour, &minute,
			&second);
	while (iter_status == FX_SUCCESS) {
		AppImageCleanup_Timestamp timestamp = { 0 };

		if (((attributes & FX_DIRECTORY) == 0U)
				&& AppImageCleanup_ResolveTimestamp(file_name, year, month, day,
						hour, minute, second, &timestamp)) {
			if (AppImageCleanup_IsWithinLast24Hours(&timestamp,
					&current_timestamp)) {
				recent_files++;
				iter_status = fx_directory_next_full_entry_find(media_ptr,
						file_name, &attributes, &file_size, &year, &month, &day,
						&hour, &minute, &second);
				continue;
			}

			const uint64_t bucket_key = AppImageCleanup_BuildBucketKey(
					&timestamp);
			const AppImageCleanup_BucketRecord *record =
					AppImageCleanup_FindBucket(bucket_key);

			if ((file_size == 0U) || ((record != NULL)
					&& (strcmp(file_name, record->file_name) != 0))) {
				flush_status = fx_file_delete(media_ptr, file_name);
				if (flush_status == FX_SUCCESS) {
					deleted_files++;
				} else if (flush_status != FX_NOT_FOUND) {
                    delete_failed = true;
					DebugConsole_Printf(
							"[IMAGE][CLEANUP] Failed to delete %s, status=%lu.\r\n",
							file_name, (unsigned long) flush_status);
				}
			} else if (record != NULL) {
				kept_files++;
			}
		}

		iter_status = fx_directory_next_full_entry_find(media_ptr, file_name,
				&attributes, &file_size, &year, &month, &day, &hour, &minute,
				&second);
	}

	if ((iter_status != FX_NO_MORE_ENTRIES) && (iter_status != FX_SUCCESS)) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] Directory scan failed during prune, status=%lu.\r\n",
				(unsigned long) iter_status);
		goto cleanup;
	}

	if (deleted_files > 0U) {
		flush_status = fx_media_flush(media_ptr);
		if (flush_status != FX_SUCCESS) {
			DebugConsole_Printf(
					"[IMAGE][CLEANUP] Media flush failed after pruning, status=%lu.\r\n",
					(unsigned long) flush_status);
			iter_status = flush_status;
			goto cleanup;
		}
	}

    if (deleted_files > 0U) {
        char summary[128];
        (void) snprintf(summary, sizeof(summary),
                "[IMAGE][CLEANUP] path=%s scanned=%lu deleted=%lu kept=%lu recent=%lu",
                directory_path, (unsigned long) scanned_files,
                (unsigned long) deleted_files, (unsigned long) kept_files,
                (unsigned long) recent_files);
        (void) SdDebugLogService_EnqueueLine(summary);
    }
	DebugConsole_Printf(
			"[IMAGE][CLEANUP] Directory complete: scanned=%lu kept=%lu recent=%lu deleted=%lu skipped=%lu%s\r\n",
			(unsigned long) scanned_files, (unsigned long) kept_files,
			(unsigned long) recent_files, (unsigned long) deleted_files,
			(unsigned long) skipped_files,
			sweep_overflow ? " (bucket table full)" : "");

	iter_status = delete_failed ? FX_IO_ERROR : FX_SUCCESS;

cleanup:
	(void) fx_directory_default_set(media_ptr, FX_NULL);

	AppFileX_ReleaseMediaLock();
	return iter_status;
}

/**
 * @brief Reset the bucket table before a new sweep.
 */
static void AppImageCleanup_ResetBuckets(void) {
	(void) memset(app_image_cleanup_bucket_records, 0,
			sizeof(app_image_cleanup_bucket_records));
}

/**
 * @brief Determine whether a character is an ASCII decimal digit.
 */
static bool AppImageCleanup_IsDigit(char ch) {
	return (ch >= '0') && (ch <= '9');
}

/**
 * @brief Parse a two-digit decimal field.
 */
static bool AppImageCleanup_ParseTwoDigits(const CHAR *text_ptr,
		UINT *value_out) {
	if ((text_ptr == NULL) || (value_out == NULL)) {
		return false;
	}

	if (!AppImageCleanup_IsDigit((char) text_ptr[0])
			|| !AppImageCleanup_IsDigit((char) text_ptr[1])) {
		return false;
	}

	*value_out = ((UINT) (text_ptr[0] - '0') * 10U)
			+ (UINT) (text_ptr[1] - '0');
	return true;
}

/**
 * @brief Parse a four-digit decimal field.
 */
static bool AppImageCleanup_ParseFourDigits(const CHAR *text_ptr,
		UINT *value_out) {
	if ((text_ptr == NULL) || (value_out == NULL)) {
		return false;
	}

	if (!AppImageCleanup_IsDigit((char) text_ptr[0])
			|| !AppImageCleanup_IsDigit((char) text_ptr[1])
			|| !AppImageCleanup_IsDigit((char) text_ptr[2])
			|| !AppImageCleanup_IsDigit((char) text_ptr[3])) {
		return false;
	}

	*value_out = ((UINT) (text_ptr[0] - '0') * 1000U)
			+ ((UINT) (text_ptr[1] - '0') * 100U)
			+ ((UINT) (text_ptr[2] - '0') * 10U)
			+ (UINT) (text_ptr[3] - '0');
	return true;
}

/**
 * @brief Check whether a timestamp looks like a real RTC value.
 */
static bool AppImageCleanup_IsPlausibleTimestamp(UINT year, UINT month,
		UINT day, UINT hour, UINT minute, UINT second) {
    static const UINT days[] = {31U,28U,31U,30U,31U,30U,31U,31U,30U,31U,30U,31U};
    if ((year < 2020U) || (year > 2099U) || (month < 1U) || (month > 12U)
            || (day < 1U) || (hour > 23U) || (minute > 59U) || (second > 59U)) {
        return false;
    }
    const UINT maximum_day = days[month - 1U]
            + (((month == 2U) && AppImageCleanup_IsLeapYear(year)) ? 1U : 0U);
    return day <= maximum_day;
}

/**
 * @brief Parse the capture timestamp embedded in a filename.
 */
static bool AppImageCleanup_ParseTimestampFromName(const CHAR *file_name_ptr,
		AppImageCleanup_Timestamp *timestamp_ptr) {
	const CHAR prefix[] = "capture_";
	const CHAR *cursor = NULL;
	UINT year = 0U;
	UINT month = 0U;
	UINT day = 0U;
	UINT hour = 0U;
	UINT minute = 0U;
	UINT second = 0U;

	if ((file_name_ptr == NULL) || (timestamp_ptr == NULL)) {
		return false;
	}

	if ((strlen(file_name_ptr) < 27U)
            || (strncmp(file_name_ptr, prefix, sizeof(prefix) - 1U) != 0)) {
		return false;
	}

	cursor = file_name_ptr + (sizeof(prefix) - 1U);
	if (!AppImageCleanup_ParseFourDigits(cursor, &year)) {
		return false;
	}
	cursor += 4U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &month)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &day)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '_') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &hour)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &minute)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &second)) {
		return false;
	}
	cursor += 2U;

	if ((*cursor != '\0') && (*cursor != '.')) {
		return false;
	}

	if (!AppImageCleanup_IsPlausibleTimestamp(year, month, day, hour, minute,
			second)) {
		return false;
	}

	timestamp_ptr->year = year;
	timestamp_ptr->month = month;
	timestamp_ptr->day = day;
	timestamp_ptr->hour = hour;
	timestamp_ptr->minute = minute;
	timestamp_ptr->second = second;
	return true;
}

/**
 * @brief Parse a bare timestamp string in the RTC capture format.
 */
static bool AppImageCleanup_ParseTimestampText(const CHAR *text_ptr,
		AppImageCleanup_Timestamp *timestamp_ptr) {
	const CHAR *cursor = NULL;
	UINT year = 0U;
	UINT month = 0U;
	UINT day = 0U;
	UINT hour = 0U;
	UINT minute = 0U;
	UINT second = 0U;

	if ((text_ptr == NULL) || (timestamp_ptr == NULL) || (strlen(text_ptr) != 19U)) {
		return false;
	}

	cursor = text_ptr;
	if (!AppImageCleanup_ParseFourDigits(cursor, &year)) {
		return false;
	}
	cursor += 4U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &month)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &day)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '_') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &hour)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &minute)) {
		return false;
	}
	cursor += 2U;
	if (*cursor != '-') {
		return false;
	}
	cursor++;
	if (!AppImageCleanup_ParseTwoDigits(cursor, &second)) {
		return false;
	}
	cursor += 2U;

	if (*cursor != '\0') {
		return false;
	}

	if ((year < 2000U) || (year > 2099U) || (month < 1U) || (month > 12U)
			|| (day < 1U) || (day > 31U) || (hour > 23U) || (minute > 59U)
			|| (second > 59U)) {
		return false;
	}

	timestamp_ptr->year = year;
	timestamp_ptr->month = month;
	timestamp_ptr->day = day;
	timestamp_ptr->hour = hour;
	timestamp_ptr->minute = minute;
	timestamp_ptr->second = second;
	return true;
}

/**
 * @brief Resolve the best timestamp source for one capture file.
 */
static bool AppImageCleanup_ResolveTimestamp(
		const CHAR *file_name_ptr, UINT year, UINT month, UINT day, UINT hour,
		UINT minute, UINT second, AppImageCleanup_Timestamp *timestamp_ptr) {
	if (timestamp_ptr == NULL) {
		return false;
	}

    const CHAR *extension = strrchr(file_name_ptr, '.');
    if ((strncmp(file_name_ptr, "capture_", 8U) != 0) || (extension == NULL)
            || ((strcmp(extension, ".gray8") != 0) && (strcmp(extension, ".raw16") != 0)
                && (strcmp(extension, ".yuv422") != 0))) {
        return false;  /* Never prune unrelated files based on their FAT timestamp. */
    }
    if ((app_image_cleanup_hour_prefix[0] != '\0')
            && ((size_t)(extension - file_name_ptr) == 13U)
            && (file_name_ptr[10] == '-')) {
        CHAR text[20];
        (void) snprintf(text, sizeof(text), "%s-%.5s", app_image_cleanup_hour_prefix, file_name_ptr + 8U);
        return AppImageCleanup_ParseTimestampText(text, timestamp_ptr)
                && AppImageCleanup_IsPlausibleTimestamp(timestamp_ptr->year,
                        timestamp_ptr->month, timestamp_ptr->day, timestamp_ptr->hour,
                        timestamp_ptr->minute, timestamp_ptr->second);
    }
	if (AppImageCleanup_ParseTimestampFromName(file_name_ptr,
			timestamp_ptr)) {
		return true;
	}

	if (AppImageCleanup_IsPlausibleTimestamp(year, month, day, hour, minute,
			second)) {
		timestamp_ptr->year = year;
		timestamp_ptr->month = month;
		timestamp_ptr->day = day;
		timestamp_ptr->hour = hour;
		timestamp_ptr->minute = minute;
		timestamp_ptr->second = second;
		return true;
	}

	return false;
}

/**
 * @brief Read the live RTC value without falling back to a cached timestamp.
 */
static bool AppImageCleanup_ReadCurrentTimestamp(
		AppImageCleanup_Timestamp *timestamp_ptr) {
	CHAR rtc_text[32] = { 0 };

	if (timestamp_ptr == NULL) {
		return false;
	}

	if (!App_Clock_GetCurrentTimestamp(rtc_text, sizeof(rtc_text))) {
		return false;
	}

	return AppImageCleanup_ParseTimestampText(rtc_text, timestamp_ptr);
}

/**
 * @brief Return whether a year is a leap year in the Gregorian calendar.
 */
static bool AppImageCleanup_IsLeapYear(UINT year) {
	return ((year % 4U) == 0U) && (((year % 100U) != 0U)
			|| ((year % 400U) == 0U));
}

/**
 * @brief Return the number of days before the given month in the given year.
 */
static uint32_t AppImageCleanup_DaysBeforeMonth(UINT year, UINT month) {
	static const uint32_t days_before_month[] = { 0U, 31U, 59U, 90U, 120U,
			151U, 181U, 212U, 243U, 273U, 304U, 334U };
	uint32_t days = 0U;

	if (month <= 1U) {
		return 0U;
	}

	if (month > 12U) {
		return 0U;
	}

	days = days_before_month[month - 1U];
	if ((month > 2U) && AppImageCleanup_IsLeapYear(year)) {
		days++;
	}

	return days;
}

/**
 * @brief Convert a timestamp to a monotonic second counter since 2000-01-01.
 */
static bool AppImageCleanup_ToSecondsSinceEpoch(
		const AppImageCleanup_Timestamp *timestamp_ptr, uint64_t *seconds_out) {
	uint64_t days = 0U;
	UINT year = 0U;

	if ((timestamp_ptr == NULL) || (seconds_out == NULL)) {
		return false;
	}

	if ((timestamp_ptr->year < 2000U) || (timestamp_ptr->year > 2099U)
			|| (timestamp_ptr->month < 1U) || (timestamp_ptr->month > 12U)
			|| (timestamp_ptr->day < 1U) || (timestamp_ptr->day > 31U)
			|| (timestamp_ptr->hour > 23U) || (timestamp_ptr->minute > 59U)
			|| (timestamp_ptr->second > 59U)) {
		return false;
	}

	for (year = 2000U; year < timestamp_ptr->year; year++) {
		days += AppImageCleanup_IsLeapYear(year) ? 366U : 365U;
	}

	days += AppImageCleanup_DaysBeforeMonth(timestamp_ptr->year,
			timestamp_ptr->month);
	days += (uint64_t) (timestamp_ptr->day - 1U);

	*seconds_out = (days * 86400U)
			+ ((uint64_t) timestamp_ptr->hour * 3600U)
			+ ((uint64_t) timestamp_ptr->minute * 60U)
			+ (uint64_t) timestamp_ptr->second;
	return true;
}

/**
 * @brief Return true when a file is within the last 24 hours of the current RTC time.
 */
static bool AppImageCleanup_IsWithinLast24Hours(
		const AppImageCleanup_Timestamp *file_timestamp_ptr,
		const AppImageCleanup_Timestamp *current_timestamp_ptr) {
	uint64_t file_seconds = 0U;
	uint64_t current_seconds = 0U;

	if (!AppImageCleanup_ToSecondsSinceEpoch(file_timestamp_ptr,
			&file_seconds)) {
		return false;
	}

	if (!AppImageCleanup_ToSecondsSinceEpoch(current_timestamp_ptr,
			&current_seconds)) {
		return false;
	}

	if (file_seconds >= current_seconds) {
		return true;
	}

	return ((current_seconds - file_seconds)
			<= (uint64_t) APP_IMAGE_CLEANUP_RECENT_WINDOW_SECONDS);
}

/**
 * @brief Pack a timestamp into a sortable 64-bit key.
 */
static uint64_t AppImageCleanup_BuildTimestampKey(
		const AppImageCleanup_Timestamp *timestamp_ptr) {
	if (timestamp_ptr == NULL) {
		return 0U;
	}

	return (((uint64_t) timestamp_ptr->year) << 40U)
			| (((uint64_t) timestamp_ptr->month) << 32U)
			| (((uint64_t) timestamp_ptr->day) << 24U)
			| (((uint64_t) timestamp_ptr->hour) << 16U)
			| (((uint64_t) timestamp_ptr->minute) << 8U)
			| (uint64_t) timestamp_ptr->second;
}

/**
 * @brief Pack a 10-minute bucket into a sortable 64-bit key.
 */
static uint64_t AppImageCleanup_BuildBucketKey(
		const AppImageCleanup_Timestamp *timestamp_ptr) {
	if (timestamp_ptr == NULL) {
		return 0U;
	}

	return (((uint64_t) timestamp_ptr->year) << 40U)
			| (((uint64_t) timestamp_ptr->month) << 32U)
			| (((uint64_t) timestamp_ptr->day) << 24U)
			| (((uint64_t) timestamp_ptr->hour) << 16U)
			| (((uint64_t) (timestamp_ptr->minute / 10U)) << 8U);
}

/**
 * @brief Find an existing bucket record.
 */
static AppImageCleanup_BucketRecord *AppImageCleanup_FindBucket(
		uint64_t bucket_key) {
	ULONG index = 0U;

	for (index = 0U; index < APP_IMAGE_CLEANUP_MAX_BUCKETS; index++) {
		if (app_image_cleanup_bucket_records[index].used
				&& (app_image_cleanup_bucket_records[index].bucket_key
						== bucket_key)) {
			return &app_image_cleanup_bucket_records[index];
		}
	}

	return NULL;
}

/**
 * @brief Allocate a new bucket record for a newly seen bucket.
 */
static AppImageCleanup_BucketRecord *AppImageCleanup_AllocateBucket(
		uint64_t bucket_key) {
	ULONG index = 0U;

	for (index = 0U; index < APP_IMAGE_CLEANUP_MAX_BUCKETS; index++) {
		if (!app_image_cleanup_bucket_records[index].used) {
			app_image_cleanup_bucket_records[index].used = true;
			app_image_cleanup_bucket_records[index].bucket_key = bucket_key;
			app_image_cleanup_bucket_records[index].timestamp_key = 0U;
			app_image_cleanup_bucket_records[index].file_name[0] = '\0';
			return &app_image_cleanup_bucket_records[index];
		}
	}

	return NULL;
}

/* USER CODE END 1 */
