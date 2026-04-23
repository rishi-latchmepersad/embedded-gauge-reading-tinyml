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
#include "app_storage.h"
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
#define APP_IMAGE_CLEANUP_PERIOD_MS            600000U
#define APP_IMAGE_CLEANUP_MEDIA_READY_TIMEOUT_MS  60000U
#define APP_IMAGE_CLEANUP_RETRY_DELAY_MS          30000U
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
static AppImageCleanup_BucketRecord app_image_cleanup_bucket_records
		[APP_IMAGE_CLEANUP_MAX_BUCKETS];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static VOID AppImageCleanupThread_Entry(ULONG thread_input);
static UINT AppImageCleanup_RunSweep(void);
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
 * @brief Background worker that prunes duplicate captures from /captured_images.
 * @param thread_input Unused ThreadX input value.
 */
static VOID AppImageCleanupThread_Entry(ULONG thread_input) {
	(void) thread_input;

	DebugConsole_Printf(
			"[IMAGE][CLEANUP] Thread started; pruning every %lu ms.\r\n",
			(unsigned long) APP_IMAGE_CLEANUP_PERIOD_MS);

	while (1) {
		UINT sweep_status = TX_SUCCESS;

		if (!AppStorage_WaitForMediaReady(APP_IMAGE_CLEANUP_MEDIA_READY_TIMEOUT_MS)) {
			DebugConsole_Printf(
					"[IMAGE][CLEANUP] Media not ready yet; retrying later.\r\n");
			DelayMilliseconds_Cooperative(APP_IMAGE_CLEANUP_RETRY_DELAY_MS);
			continue;
		}

		sweep_status = AppImageCleanup_RunSweep();
		if (sweep_status != FX_SUCCESS) {
			DebugConsole_Printf(
					"[IMAGE][CLEANUP] Sweep failed with status=%lu; retrying later.\r\n",
					(unsigned long) sweep_status);
			DelayMilliseconds_Cooperative(APP_IMAGE_CLEANUP_RETRY_DELAY_MS);
			continue;
		}

		DelayMilliseconds_Cooperative(APP_IMAGE_CLEANUP_PERIOD_MS);
	}
}

/**
 * @brief Perform one cleanup sweep over the captured image directory.
 * @retval FX_SUCCESS when the sweep completed or found nothing to prune.
 */
static UINT AppImageCleanup_RunSweep(void) {
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
	bool directory_selected = false;
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
			(CHAR *) AppFileX_GetCapturedImagesDirectoryName());
	if (iter_status != FX_SUCCESS) {
		DebugConsole_Printf(
				"[IMAGE][CLEANUP] Failed to select /%s, status=%lu.\r\n",
				AppFileX_GetCapturedImagesDirectoryName(),
				(unsigned long) iter_status);
		goto cleanup;
	}

	directory_selected = true;

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
							sizeof(record->file_name), "%s", file_name);
				}
			} else if (timestamp_key >= record->timestamp_key) {
				record->timestamp_key = timestamp_key;
				(void) snprintf(record->file_name, sizeof(record->file_name),
						"%s", file_name);
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

			if ((record != NULL)
					&& (strcmp(file_name, record->file_name) != 0)) {
				flush_status = fx_file_delete(media_ptr, file_name);
				if (flush_status == FX_SUCCESS) {
					deleted_files++;
				} else if (flush_status != FX_NOT_FOUND) {
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

	DebugConsole_Printf(
			"[IMAGE][CLEANUP] Sweep complete: scanned=%lu kept=%lu recent=%lu deleted=%lu skipped=%lu%s\r\n",
			(unsigned long) scanned_files, (unsigned long) kept_files,
			(unsigned long) recent_files, (unsigned long) deleted_files,
			(unsigned long) skipped_files,
			sweep_overflow ? " (bucket table full)" : "");

	iter_status = FX_SUCCESS;

cleanup:
	if (directory_selected) {
		(void) fx_directory_default_set(media_ptr, FX_NULL);
	}

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
	return (year >= 2020U) && (year <= 2099U) && (month >= 1U)
			&& (month <= 12U) && (day >= 1U) && (day <= 31U) && (hour <= 23U)
			&& (minute <= 59U) && (second <= 59U);
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

	if (strncmp(file_name_ptr, prefix, sizeof(prefix) - 1U) != 0) {
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

	if ((text_ptr == NULL) || (timestamp_ptr == NULL)) {
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
