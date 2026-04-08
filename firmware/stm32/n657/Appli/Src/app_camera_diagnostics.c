/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_camera_diagnostics.c
 * @brief   Camera capture summary and RAW10 diagnostics.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_camera_diagnostics.h"

#include <string.h>

#include "app_camera_buffers.h"
#include "debug_console.h"

/* Read the low 10 bits from a padded raw sample. */
static uint16_t AppCameraDiagnostics_ReadRaw10Level(uint16_t raw_word) {
	return (uint16_t) (raw_word & 0x03FFU);
}

/* Report the two most common RAW10 sample levels in a live capture. */
static void AppCameraDiagnostics_LogRawDominantLevels(
		const uint8_t *buffer_ptr, uint32_t length_bytes) {
	const uint16_t *samples = (const uint16_t*) buffer_ptr;
	const uint32_t sample_count = length_bytes / sizeof(uint16_t);
	uint32_t sum_levels = 0U;
	uint32_t bright_count = 0U;
	uint32_t top1_level = 0U;
	uint32_t top2_level = 0U;
	uint32_t top1_count = 0U;
	uint32_t top2_count = 0U;

	if ((buffer_ptr == NULL) || (length_bytes < sizeof(uint16_t))
			|| ((length_bytes % sizeof(uint16_t)) != 0U) || (sample_count == 0U)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] RAW10 dominant-level summary skipped for empty buffer.\r\n");
		return;
	}

	(void) memset(camera_capture_raw_level_histogram, 0,
			sizeof(camera_capture_raw_level_histogram));

	for (uint32_t sample_index = 0U; sample_index < sample_count;
			sample_index++) {
		const uint32_t level =
				AppCameraDiagnostics_ReadRaw10Level(samples[sample_index]);

		camera_capture_raw_level_histogram[level]++;
		sum_levels += level;
		if (level >= 900U) {
			bright_count++;
		}
	}

	for (uint32_t level = 0U; level < 1024U; level++) {
		const uint32_t count = camera_capture_raw_level_histogram[level];

		if ((count > top1_count) || ((count == top1_count)
				&& (level > top1_level))) {
			top2_level = top1_level;
			top2_count = top1_count;
			top1_level = level;
			top1_count = count;
		} else if ((count > top2_count) || ((count == top2_count)
				&& (level > top2_level))) {
			top2_level = level;
			top2_count = count;
		}
	}

	const uint32_t mean_level = sum_levels / sample_count;
	const uint32_t top1_pct = ((top1_count * 100U) + (sample_count / 2U))
			/ sample_count;
	const uint32_t top2_pct = ((top2_count * 100U) + (sample_count / 2U))
			/ sample_count;
	const uint32_t bright_pct = ((bright_count * 100U) + (sample_count / 2U))
			/ sample_count;

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] RAW10 dominant levels (not RGB): mean=%lu top1=%03lu (%lu px, %lu%%) top2=%03lu (%lu px, %lu%%) bright>=900=%lu px (%lu%%).\r\n",
			(unsigned long) mean_level, (unsigned long) top1_level,
			(unsigned long) top1_count, (unsigned long) top1_pct,
			(unsigned long) top2_level, (unsigned long) top2_count,
			(unsigned long) top2_pct, (unsigned long) bright_count,
			(unsigned long) bright_pct);
}

/* Summarize a raw Pipe0 buffer using padded 16-bit raw pixels. */
static void AppCameraDiagnostics_LogCaptureBufferSummaryRaw(
		const uint8_t *buffer_ptr, uint32_t captured_bytes) {
	const uint32_t halfword_count = captured_bytes / sizeof(uint16_t);
	const uint16_t *halfwords = (const uint16_t*) buffer_ptr;
	uint16_t minimum_halfword = 0xFFFFU;
	uint16_t maximum_halfword = 0U;
	uint32_t nonzero_halfword_count = 0U;
	uint32_t first_nonzero_index = halfword_count;
	uint32_t last_nonzero_index = 0U;

	if ((captured_bytes == 0U) || ((captured_bytes % sizeof(uint16_t)) != 0U)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Raw buffer summary skipped for odd/empty byte count %lu.\r\n",
				(unsigned long) captured_bytes);
		return;
	}

	for (uint32_t halfword_index = 0U; halfword_index < halfword_count;
			halfword_index++) {
		const uint16_t halfword = halfwords[halfword_index];

		if (halfword < minimum_halfword) {
			minimum_halfword = halfword;
		}

		if (halfword > maximum_halfword) {
			maximum_halfword = halfword;
		}

		if (halfword != 0U) {
			nonzero_halfword_count++;
			if (first_nonzero_index == halfword_count) {
				first_nonzero_index = halfword_index;
			}
			last_nonzero_index = halfword_index;
		}
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Buffer summary (raw halfwords): samples=%lu nonzero=%lu min=0x%04X max=0x%04X first_nonzero=%lu last_nonzero=%lu.\r\n",
			(unsigned long) halfword_count, (unsigned long) nonzero_halfword_count,
			(unsigned int) minimum_halfword, (unsigned int) maximum_halfword,
			(unsigned long) (
					(first_nonzero_index == halfword_count) ? 0U : first_nonzero_index),
			(unsigned long) last_nonzero_index);

	AppCameraDiagnostics_LogRawDominantLevels(buffer_ptr, captured_bytes);
}

/* Summarize the captured raw buffer so bring-up can distinguish real image
 * data from all-zero/all-flat frames before writing to SD. */
static void AppCameraDiagnostics_LogCaptureBufferSummaryProcessed(
		const uint8_t *buffer_ptr, uint32_t captured_bytes) {
	const uint32_t diagnostic_window_byte_count = 16U;
	const uint32_t sample_count = captured_bytes / sizeof(uint16_t);
	const uint8_t *bytes = (const uint8_t*) buffer_ptr;
	const uint16_t *samples = (const uint16_t*) buffer_ptr;
	const uint32_t diagnostic_window_sample_count = 8U;
	uint16_t minimum_sample = 0xFFFFU;
	uint16_t maximum_sample = 0U;
	uint32_t nonzero_sample_count = 0U;
	uint32_t first_nonzero_index = sample_count;
	uint32_t last_nonzero_index = 0U;
	uint32_t nonzero_byte_count = 0U;
	uint32_t first_nonzero_byte_index = captured_bytes;
	uint32_t last_nonzero_byte_index = 0U;

	if ((captured_bytes == 0U) || ((captured_bytes % sizeof(uint16_t)) != 0U)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Buffer summary skipped for odd/empty byte count %lu.\r\n",
				(unsigned long) captured_bytes);
		return;
	}

	for (uint32_t sample_index = 0U; sample_index < sample_count;
			sample_index++) {
		const uint16_t sample = samples[sample_index];

		if (sample < minimum_sample) {
			minimum_sample = sample;
		}

		if (sample > maximum_sample) {
			maximum_sample = sample;
		}

		if (sample != 0U) {
			nonzero_sample_count++;
			if (first_nonzero_index == sample_count) {
				first_nonzero_index = sample_index;
			}
			last_nonzero_index = sample_index;
		}
	}

	for (uint32_t byte_index = 0U; byte_index < captured_bytes; byte_index++) {
		if (bytes[byte_index] != 0U) {
			nonzero_byte_count++;
			if (first_nonzero_byte_index == captured_bytes) {
				first_nonzero_byte_index = byte_index;
			}
			last_nonzero_byte_index = byte_index;
		}
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Buffer summary: samples=%lu nonzero=%lu min=%u max=%u first_nonzero=%lu last_nonzero=%lu nonzero_bytes=%lu first_nonzero_byte=%lu last_nonzero_byte=%lu first8=[%u,%u,%u,%u,%u,%u,%u,%u].\r\n",
			(unsigned long) sample_count, (unsigned long) nonzero_sample_count,
			(unsigned int) minimum_sample, (unsigned int) maximum_sample,
			(unsigned long) (
					(first_nonzero_index == sample_count) ? 0U : first_nonzero_index),
			(unsigned long) last_nonzero_index,
			(unsigned long) nonzero_byte_count,
			(unsigned long) (
					(first_nonzero_byte_index == captured_bytes) ? 0U :
							first_nonzero_byte_index),
			(unsigned long) last_nonzero_byte_index, (unsigned int) samples[0],
			(unsigned int) samples[1], (unsigned int) samples[2],
			(unsigned int) samples[3], (unsigned int) samples[4],
			(unsigned int) samples[5], (unsigned int) samples[6],
			(unsigned int) samples[7]);

	if (first_nonzero_index < sample_count) {
		const uint32_t window_start =
				(first_nonzero_index >= diagnostic_window_sample_count) ?
						(first_nonzero_index - diagnostic_window_sample_count) :
						0U;
		const uint32_t window_end =
				((first_nonzero_index + diagnostic_window_sample_count)
						< sample_count) ?
						(first_nonzero_index + diagnostic_window_sample_count) :
						(sample_count - 1U);
		const uint32_t sample0_index = window_start;
		const uint32_t sample1_index =
				(sample0_index < window_end) ?
						(sample0_index + 1U) : window_end;
		const uint32_t sample2_index =
				(sample1_index < window_end) ?
						(sample1_index + 1U) : window_end;
		const uint32_t sample3_index =
				(sample2_index < window_end) ?
						(sample2_index + 1U) : window_end;
		const uint32_t sample4_index =
				(sample3_index < window_end) ?
						(sample3_index + 1U) : window_end;
		const uint32_t sample5_index =
				(sample4_index < window_end) ?
						(sample4_index + 1U) : window_end;
		const uint32_t sample6_index =
				(sample5_index < window_end) ?
						(sample5_index + 1U) : window_end;
		const uint32_t sample7_index =
				(sample6_index < window_end) ?
						(sample6_index + 1U) : window_end;
		const uint32_t sample8_index =
				(sample7_index < window_end) ?
						(sample7_index + 1U) : window_end;
		const uint32_t sample9_index =
				(sample8_index < window_end) ?
						(sample8_index + 1U) : window_end;
		const uint32_t sample10_index =
				(sample9_index < window_end) ?
						(sample9_index + 1U) : window_end;
		const uint32_t sample11_index =
				(sample10_index < window_end) ?
						(sample10_index + 1U) : window_end;
		const uint32_t sample12_index =
				(sample11_index < window_end) ?
						(sample11_index + 1U) : window_end;
		const uint32_t sample13_index =
				(sample12_index < window_end) ?
						(sample12_index + 1U) : window_end;
		const uint32_t sample14_index =
				(sample13_index < window_end) ?
						(sample13_index + 1U) : window_end;
		const uint32_t sample15_index =
				(sample14_index < window_end) ?
						(sample14_index + 1U) : window_end;

		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Samples around first_nonzero=%lu: [%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u].\r\n",
				(unsigned long) first_nonzero_index,
				(unsigned int) samples[sample0_index],
				(unsigned int) samples[sample1_index],
				(unsigned int) samples[sample2_index],
				(unsigned int) samples[sample3_index],
				(unsigned int) samples[sample4_index],
				(unsigned int) samples[sample5_index],
				(unsigned int) samples[sample6_index],
				(unsigned int) samples[sample7_index],
				(unsigned int) samples[sample8_index],
				(unsigned int) samples[sample9_index],
				(unsigned int) samples[sample10_index],
				(unsigned int) samples[sample11_index],
				(unsigned int) samples[sample12_index],
				(unsigned int) samples[sample13_index],
				(unsigned int) samples[sample14_index],
				(unsigned int) samples[sample15_index],
				(unsigned int) samples[window_end]);
	}

	if (first_nonzero_byte_index < captured_bytes) {
		const uint32_t byte_window_start =
				(first_nonzero_byte_index >= diagnostic_window_byte_count) ?
						(first_nonzero_byte_index - diagnostic_window_byte_count) :
						0U;
		const uint32_t byte_window_end =
				((first_nonzero_byte_index + diagnostic_window_byte_count)
						< captured_bytes) ?
						(first_nonzero_byte_index + diagnostic_window_byte_count) :
						(captured_bytes - 1U);
		uint32_t byte_indices[17U] = { 0U };

		byte_indices[0] = byte_window_start;
		for (uint32_t index = 1U; index < 17U; index++) {
			byte_indices[index] =
					(byte_indices[index - 1U] < byte_window_end) ?
							(byte_indices[index - 1U] + 1U) : byte_window_end;
		}

		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Bytes around first_nonzero_byte=%lu: [%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u].\r\n",
				(unsigned long) first_nonzero_byte_index,
				(unsigned int) bytes[byte_indices[0]],
				(unsigned int) bytes[byte_indices[1]],
				(unsigned int) bytes[byte_indices[2]],
				(unsigned int) bytes[byte_indices[3]],
				(unsigned int) bytes[byte_indices[4]],
				(unsigned int) bytes[byte_indices[5]],
				(unsigned int) bytes[byte_indices[6]],
				(unsigned int) bytes[byte_indices[7]],
				(unsigned int) bytes[byte_indices[8]],
				(unsigned int) bytes[byte_indices[9]],
				(unsigned int) bytes[byte_indices[10]],
				(unsigned int) bytes[byte_indices[11]],
				(unsigned int) bytes[byte_indices[12]],
				(unsigned int) bytes[byte_indices[13]],
				(unsigned int) bytes[byte_indices[14]],
				(unsigned int) bytes[byte_indices[15]],
				(unsigned int) bytes[byte_indices[16]]);
	}
}

void AppCameraDiagnostics_LogCaptureBufferSummary(const uint8_t *buffer_ptr,
		uint32_t captured_bytes, bool use_cmw_pipeline) {
	if (!use_cmw_pipeline) {
		AppCameraDiagnostics_LogCaptureBufferSummaryRaw(buffer_ptr,
				captured_bytes);
		return;
	}

	AppCameraDiagnostics_LogCaptureBufferSummaryProcessed(buffer_ptr,
			captured_bytes);
}
