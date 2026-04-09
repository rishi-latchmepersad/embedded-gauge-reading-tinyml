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
#include "main.h"

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

/**
 * @brief Print a small byte/word preview of the current frame buffer.
 *
 * This keeps the preview logic isolated from the ThreadX capture flow so the
 * thread file only owns orchestration.
 * @param reason Human-readable reason that triggered the preview.
 * @param buffer_ptr Buffer to inspect.
 * @param length_bytes Number of bytes to preview.
 */
void AppCameraDiagnostics_LogCaptureBufferPreview(const char *reason,
		const uint8_t *buffer_ptr, uint32_t length_bytes) {
	const uint32_t preview_byte_count =
			(length_bytes < 16U) ? length_bytes : 16U;
	uint32_t preview_bytes[16U] = { 0U };
	const uint32_t preview_word_count =
			(length_bytes / sizeof(uint32_t) < 4U) ?
					(length_bytes / sizeof(uint32_t)) : 4U;
	uint32_t preview_words[4U] = { 0U };

	if ((buffer_ptr == NULL) || (length_bytes == 0U)) {
		return;
	}

	for (uint32_t index = 0U; index < preview_byte_count; index++) {
		preview_bytes[index] = (uint32_t) buffer_ptr[index];
	}

	for (uint32_t index = 0U; index < preview_word_count; index++) {
		const uint32_t byte_index = index * sizeof(uint32_t);
		uint32_t word_value = 0U;

		(void) memcpy(&word_value, &buffer_ptr[byte_index], sizeof(word_value));
		preview_words[index] = word_value;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Raw preview (%s): bytes=[%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX,%02lX] words=[0x%08lX,0x%08lX,0x%08lX,0x%08lX].\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) preview_bytes[0], (unsigned long) preview_bytes[1],
			(unsigned long) preview_bytes[2], (unsigned long) preview_bytes[3],
			(unsigned long) preview_bytes[4], (unsigned long) preview_bytes[5],
			(unsigned long) preview_bytes[6], (unsigned long) preview_bytes[7],
			(unsigned long) preview_bytes[8], (unsigned long) preview_bytes[9],
			(unsigned long) preview_bytes[10],
			(unsigned long) preview_bytes[11],
			(unsigned long) preview_bytes[12],
			(unsigned long) preview_bytes[13],
			(unsigned long) preview_bytes[14],
			(unsigned long) preview_bytes[15], (unsigned long) preview_words[0],
			(unsigned long) preview_words[1], (unsigned long) preview_words[2],
			(unsigned long) preview_words[3]);
}

/**
 * @brief Log a compact diagnostic window from the processed YUV frame.
 *
 * This helps distinguish a truly dark frame from a frame that just needs the
 * center ROI or exposure state to settle.
 * @param reason Human-readable reason that triggered the diagnostics.
 * @param buffer_ptr Processed YUV422 frame bytes.
 * @param length_bytes Number of bytes available in the frame buffer.
 */
void AppCameraDiagnostics_LogProcessedFrameDiagnostics(const char *reason,
		const uint8_t *buffer_ptr, uint32_t length_bytes) {
	const uint32_t roi_size_pixels = 32U;
	const uint32_t frame_width_pixels = CAMERA_CAPTURE_WIDTH_PIXELS;
	const uint32_t frame_height_lines = CAMERA_CAPTURE_HEIGHT_PIXELS;
	const uint32_t bytes_per_pixel = CAMERA_CAPTURE_BYTES_PER_PIXEL;
	const uint32_t stride_bytes = frame_width_pixels * bytes_per_pixel;
	uint32_t roi_start_x = 0U;
	uint32_t roi_start_y = 0U;
	uint32_t roi_min = 0xFFU;
	uint32_t roi_max = 0U;
	uint64_t roi_sum = 0U;
	uint32_t roi_samples = 0U;
	uint32_t center_line = 0U;
	uint32_t center_preview[8U] = { 0U };

	if ((buffer_ptr == NULL) || (length_bytes < stride_bytes)) {
		return;
	}

	if ((frame_width_pixels < roi_size_pixels)
			|| (frame_height_lines < roi_size_pixels)) {
		return;
	}

	roi_start_x = (frame_width_pixels - roi_size_pixels) / 2U;
	roi_start_y = (frame_height_lines - roi_size_pixels) / 2U;
	center_line = roi_start_y + (roi_size_pixels / 2U);

	for (uint32_t row = 0U; row < roi_size_pixels; row++) {
		const uint32_t src_y = roi_start_y + row;
		const uint32_t row_base = src_y * stride_bytes;

		if ((row_base + ((roi_start_x + roi_size_pixels) * bytes_per_pixel))
				> length_bytes) {
			break;
		}

		for (uint32_t col = 0U; col < roi_size_pixels; col++) {
			const uint32_t sample_index = row_base
					+ ((roi_start_x + col) * bytes_per_pixel);
			const uint8_t y_sample = buffer_ptr[sample_index];

			if (y_sample < roi_min) {
				roi_min = y_sample;
			}
			if (y_sample > roi_max) {
				roi_max = y_sample;
			}
			roi_sum += y_sample;
			roi_samples++;
		}
	}

	for (uint32_t index = 0U; index < 8U; index++) {
		const uint32_t sample_x = roi_start_x + index;
		const uint32_t sample_index = (center_line * stride_bytes)
				+ (sample_x * bytes_per_pixel);

		if (sample_index < length_bytes) {
			center_preview[index] = buffer_ptr[sample_index];
		}
	}

	if (roi_samples == 0U) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Processed ROI stats skipped (%s): no valid samples.\r\n",
				(reason != NULL) ? reason : "capture");
		return;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Processed ROI stats (%s): roi=%lux%lu start=(%lu,%lu) y_min=%lu y_max=%lu y_mean=%lu.%02lu center8=[%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu].\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) roi_size_pixels, (unsigned long) roi_size_pixels,
			(unsigned long) roi_start_x, (unsigned long) roi_start_y,
			(unsigned long) roi_min, (unsigned long) roi_max,
			(unsigned long) (roi_sum / roi_samples),
			(unsigned long) (((roi_sum * 100U) / roi_samples) % 100U),
			(unsigned long) center_preview[0], (unsigned long) center_preview[1],
			(unsigned long) center_preview[2], (unsigned long) center_preview[3],
			(unsigned long) center_preview[4], (unsigned long) center_preview[5],
			(unsigned long) center_preview[6], (unsigned long) center_preview[7]);
}

/**
 * @brief Dump the CSI line/byte counter configuration and current event count.
 * @param reason Short note describing what triggered the dump.
 * @param event_count Number of line/byte events observed so far.
 */
void AppCameraDiagnostics_LogCsiLineByteCounters(const char *reason,
		uint32_t event_count) {
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI line/byte counters (%s): events=%lu SR0=0x%08lX SR1=0x%08lX LB0CFGR=0x%08lX LB1CFGR=0x%08lX LB2CFGR=0x%08lX LB3CFGR=0x%08lX PRGITR=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) event_count, (unsigned long) CSI->SR0,
			(unsigned long) CSI->SR1, (unsigned long) CSI->LB0CFGR,
			(unsigned long) CSI->LB1CFGR, (unsigned long) CSI->LB2CFGR,
			(unsigned long) CSI->LB3CFGR, (unsigned long) CSI->PRGITR);
}

/**
 * @brief Dump the active DCMIPP pipe registers for capture bring-up.
 * @param reason Short note describing what triggered the dump.
 * @param capture_dcmipp Active DCMIPP handle, when available.
 */
void AppCameraDiagnostics_LogDcmippPipeRegisters(const char *reason,
		DCMIPP_HandleTypeDef *capture_dcmipp) {
	uint32_t dcmipp_irq_enabled = 0U;
	uint32_t dcmipp_irq_pending = 0U;
	uint32_t csi_irq_enabled = 0U;
	uint32_t csi_irq_pending = 0U;

	if ((capture_dcmipp == NULL) || (capture_dcmipp->Instance == NULL)) {
		return;
	}

	dcmipp_irq_enabled = (NVIC_GetEnableIRQ(DCMIPP_IRQn) != 0) ? 1U : 0U;
	dcmipp_irq_pending = (NVIC_GetPendingIRQ(DCMIPP_IRQn) != 0) ? 1U : 0U;
	csi_irq_enabled = (NVIC_GetEnableIRQ(CSI_IRQn) != 0) ? 1U : 0U;
	csi_irq_pending = (NVIC_GetPendingIRQ(CSI_IRQn) != 0) ? 1U : 0U;

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP pipe regs (%s): CMCR=0x%08lX CMSR1=0x%08lX CMSR2=0x%08lX P0FSCR=0x%08lX P0FCTCR=0x%08lX P0PPCR=0x%08lX P0PPM0AR1=0x%08lX P0DCCNTR=0x%08lX P0DCLMTR=0x%08lX P0SCSTR=0x%08lX P0SCSZR=0x%08lX P0CFSCR=0x%08lX P0CFCTCR=0x%08lX P1FSCR=0x%08lX P1FCTCR=0x%08lX P1PPCR=0x%08lX P1PPM0AR1=0x%08lX P1CFSCR=0x%08lX P1CFCTCR=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) capture_dcmipp->Instance->CMCR,
			(unsigned long) capture_dcmipp->Instance->CMSR1,
			(unsigned long) capture_dcmipp->Instance->CMSR2,
			(unsigned long) capture_dcmipp->Instance->P0FSCR,
			(unsigned long) capture_dcmipp->Instance->P0FCTCR,
			(unsigned long) capture_dcmipp->Instance->P0PPCR,
			(unsigned long) capture_dcmipp->Instance->P0PPM0AR1,
			(unsigned long) capture_dcmipp->Instance->P0DCCNTR,
			(unsigned long) capture_dcmipp->Instance->P0DCLMTR,
			(unsigned long) capture_dcmipp->Instance->P0SCSTR,
			(unsigned long) capture_dcmipp->Instance->P0SCSZR,
			(unsigned long) capture_dcmipp->Instance->P0CFSCR,
			(unsigned long) capture_dcmipp->Instance->P0CFCTCR,
			(unsigned long) capture_dcmipp->Instance->P1FSCR,
			(unsigned long) capture_dcmipp->Instance->P1FCTCR,
			(unsigned long) capture_dcmipp->Instance->P1PPCR,
			(unsigned long) capture_dcmipp->Instance->P1PPM0AR1,
			(unsigned long) capture_dcmipp->Instance->P1CFSCR,
			(unsigned long) capture_dcmipp->Instance->P1CFCTCR);
	/* IPPlug AXI master FIFO partition - if DPREGSTART==DPREGEND the FIFO has
	 * zero words and the DMA client cannot write anything to memory. */
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP IPPlug (%s): IPGR1=0x%08lX IPGR2=0x%08lX IPGR3=0x%08lX IPC1R1=0x%08lX IPC1R3=0x%08lX IPC2R1=0x%08lX IPC2R3=0x%08lX IPC3R1=0x%08lX IPC3R3=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) capture_dcmipp->Instance->IPGR1,
			(unsigned long) capture_dcmipp->Instance->IPGR2,
			(unsigned long) capture_dcmipp->Instance->IPGR3,
			(unsigned long) capture_dcmipp->Instance->IPC1R1,
			(unsigned long) capture_dcmipp->Instance->IPC1R3,
			(unsigned long) capture_dcmipp->Instance->IPC2R1,
			(unsigned long) capture_dcmipp->Instance->IPC2R3,
			(unsigned long) capture_dcmipp->Instance->IPC3R1,
			(unsigned long) capture_dcmipp->Instance->IPC3R3);
	/* Shadow/current registers - reflect what was active during the last
	 * captured frame. */
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP PIPE0 current-frame regs (%s): P0CPPCR=0x%08lX P0CPPM0AR1=0x%08lX P0CPPM0AR2=0x%08lX P0CSCSTR=0x%08lX P0CSCSZR=0x%08lX P0CFCTCR=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) capture_dcmipp->Instance->P0CPPCR,
			(unsigned long) capture_dcmipp->Instance->P0CPPM0AR1,
			(unsigned long) capture_dcmipp->Instance->P0CPPM0AR2,
			(unsigned long) capture_dcmipp->Instance->P0CSCSTR,
			(unsigned long) capture_dcmipp->Instance->P0CSCSZR,
			(unsigned long) capture_dcmipp->Instance->P0CFCTCR);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP interrupt state (%s): CMIER=0x%08lX CSI_IER0=0x%08lX CSI_IER1=0x%08lX NVIC(DCMIPP enabled=%lu pending=%lu, CSI enabled=%lu pending=%lu).\r\n",
			(reason != NULL) ? reason : "capture",
			(unsigned long) capture_dcmipp->Instance->CMIER,
			(unsigned long) CSI->IER0, (unsigned long) CSI->IER1,
			(unsigned long) dcmipp_irq_enabled,
			(unsigned long) dcmipp_irq_pending,
			(unsigned long) csi_irq_enabled, (unsigned long) csi_irq_pending);
}

/**
 * @brief Decode the DCMIPP error bitmask into a human-readable summary.
 * @param error_code Raw DCMIPP ErrorCode value from the HAL handle.
 */
void AppCameraDiagnostics_LogDcmippErrorCode(uint32_t error_code) {
	const struct {
		uint32_t bit;
		const char *label;
	} error_map[] = {
		{ HAL_DCMIPP_ERROR_AXI_TRANSFER, "AXI_TRANSFER" },
		{ HAL_DCMIPP_ERROR_PARALLEL_SYNC, "PARALLEL_SYNC" },
		{ HAL_DCMIPP_ERROR_PIPE0_LIMIT, "PIPE0_LIMIT" },
		{ HAL_DCMIPP_ERROR_PIPE0_OVR, "PIPE0_OVR" },
		{ HAL_DCMIPP_ERROR_PIPE1_OVR, "PIPE1_OVR" },
		{ HAL_DCMIPP_ERROR_PIPE2_OVR, "PIPE2_OVR" },
		{ HAL_DCMIPP_CSI_ERROR_SYNC, "CSI_SYNC" },
		{ HAL_DCMIPP_CSI_ERROR_WDG, "CSI_WDG" },
		{ HAL_DCMIPP_CSI_ERROR_SPKT, "CSI_SHORT_PACKET" },
		{ HAL_DCMIPP_CSI_ERROR_DATA_ID, "CSI_DATA_ID" },
		{ HAL_DCMIPP_CSI_ERROR_CECC, "CSI_CECC" },
		{ HAL_DCMIPP_CSI_ERROR_ECC, "CSI_ECC" },
		{ HAL_DCMIPP_CSI_ERROR_CRC, "CSI_CRC" },
		{ HAL_DCMIPP_CSI_ERROR_DPHY_CTRL, "CSI_DPHY_CTRL" },
		{ HAL_DCMIPP_CSI_ERROR_DPHY_LP_SYNC, "CSI_DPHY_LP_SYNC" },
		{ HAL_DCMIPP_CSI_ERROR_DPHY_ESCAPE, "CSI_DPHY_ESCAPE" },
		{ HAL_DCMIPP_CSI_ERROR_SOT_SYNC, "CSI_SOT_SYNC" },
		{ HAL_DCMIPP_CSI_ERROR_SOT, "CSI_SOT" },
	};
	bool any = false;

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] DCMIPP error decode: raw=0x%08lX bits=",
			(unsigned long) error_code);

	for (uint32_t index = 0U; index < (sizeof(error_map) / sizeof(error_map[0]));
			index++) {
		if ((error_code & error_map[index].bit) == 0U) {
			continue;
		}

		if (any) {
			DebugConsole_WriteString("|");
		}
		DebugConsole_WriteString(error_map[index].label);
		any = true;
	}

	if (!any) {
		DebugConsole_WriteString("none");
	}

	DebugConsole_WriteString("\r\n");
}

/**
 * @brief Dump the camera, ISP, and pipe state snapshot used during capture bring-up.
 * @param snapshot Snapshot data collected by the camera thread.
 */
void AppCameraDiagnostics_LogCaptureState(
		const AppCameraDiagnostics_CaptureState_t *snapshot) {
	const char *reason = NULL;

	if (snapshot == NULL) {
		return;
	}

	reason = (snapshot->reason != NULL) ? snapshot->reason : "capture";

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] State snapshot (%s): armed=%u stream_started=%u use_cmw=%u cmw_init=%u frame_events=%lu vsync_events=%lu isp_runs=%lu csi_irqs=%lu dcmipp_irqs=%lu pipe=%lu mode=%lu pipe_state=%lu data_counter=%lu reported_bytes=%lu counter_status=%lu sof=%u eof=%u failed=%u err=0x%08lX line_errs=%lu mask=0x%08lX buffer_index=%lu.\r\n",
			reason, snapshot->snapshot_armed ? 1U : 0U,
			snapshot->stream_started ? 1U : 0U,
			snapshot->use_cmw_pipeline ? 1U : 0U,
			snapshot->cmw_initialized ? 1U : 0U,
			(unsigned long) snapshot->frame_event_count,
			(unsigned long) snapshot->vsync_event_count,
			(unsigned long) snapshot->isp_run_count,
			(unsigned long) snapshot->csi_irq_count,
			(unsigned long) snapshot->dcmipp_irq_count,
			(unsigned long) snapshot->capture_pipe,
			(unsigned long) snapshot->pipe_mode,
			(unsigned long) snapshot->pipe_state,
			(unsigned long) snapshot->pipe_counter,
			(unsigned long) snapshot->reported_byte_count,
			(unsigned long) snapshot->counter_status,
			snapshot->sof_seen ? 1U : 0U,
			snapshot->eof_seen ? 1U : 0U, snapshot->failed ? 1U : 0U,
			(unsigned long) snapshot->error_code,
			(unsigned long) snapshot->line_error_count,
			(unsigned long) snapshot->line_error_mask,
			(unsigned long) snapshot->active_buffer_index);

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] Buffer addresses: buf0=0x%08lX buf1=0x%08lX result=0x%08lX pipe_mem=0x%08lX.\r\n",
			(unsigned long) (uintptr_t) snapshot->buffer0,
			(unsigned long) (uintptr_t) snapshot->buffer1,
			(unsigned long) (uintptr_t) snapshot->result_buffer,
			(unsigned long) snapshot->pipe_memory_address);

	AppCameraDiagnostics_LogDcmippPipeRegisters(reason, snapshot->capture_dcmipp);
	AppCameraDiagnostics_LogCsiLineByteCounters(reason,
			snapshot->csi_linebyte_event_count);

	if (snapshot->cmw_state_ok) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] CMW state: exposure_mode=%ld aec=%u exposure=%ld us gain=%ld mdB test_pattern=%ld sensor=%s %lux%lu gain=[%lu,%lu] again_max=%lu exposure=[%lu,%lu].\r\n",
				(long) snapshot->cmw_exposure_mode,
				(unsigned int) snapshot->cmw_aec_enabled,
				(long) snapshot->cmw_exposure, (long) snapshot->cmw_gain,
				(long) snapshot->cmw_test_pattern,
				(snapshot->sensor_name != NULL) ? snapshot->sensor_name : "?",
				(unsigned long) snapshot->sensor_width,
				(unsigned long) snapshot->sensor_height,
				(unsigned long) snapshot->sensor_gain_min,
				(unsigned long) snapshot->sensor_gain_max,
				(unsigned long) snapshot->sensor_again_max,
				(unsigned long) snapshot->sensor_exposure_min,
				(unsigned long) snapshot->sensor_exposure_max);
	} else if (snapshot->cmw_initialized) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] CMW state readback failed while dumping camera state.\r\n");
	}

	if (snapshot->sensor_regs_ok) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 lane-mode regs: 0x3050=0x%02X 0x319D=0x%02X 0x341C=0x%02X 0x341D=0x%02X 0x3A01=0x%02X.\r\n",
				(unsigned int) snapshot->lane_mode_reg_3050,
				(unsigned int) snapshot->lane_mode_reg_319d,
				(unsigned int) snapshot->lane_mode_reg_341c,
				(unsigned int) snapshot->lane_mode_reg_341d,
				(unsigned int) snapshot->lane_mode_reg_3a01);
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 registers: mode_select=0x%02X hold=0x%02X tpg=0x%02X gain_reg=0x%04X shutter=0x%06lX vmax=0x%08lX.\r\n",
				(unsigned int) snapshot->mode_select,
				(unsigned int) snapshot->hold_reg,
				(unsigned int) snapshot->tpg_reg,
				(unsigned int) snapshot->gain_reg,
				(unsigned long) snapshot->shutter_reg,
				(unsigned long) snapshot->vmax_reg);
	} else {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 register readback failed while dumping camera state.\r\n");
	}
}
