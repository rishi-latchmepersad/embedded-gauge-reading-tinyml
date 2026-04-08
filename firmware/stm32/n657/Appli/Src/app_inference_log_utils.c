/*
 *******************************************************************************
 * @file    app_inference_log_utils.c
 * @brief   Small helpers shared by the inference logging path.
 *******************************************************************************
 */

#include "app_inference_log_utils.h"

#include <stdbool.h>
#include <stdio.h>

/**
 * @brief Format a floating-point value with one decimal place.
 * @param dst Output buffer.
 * @param dst_len Output buffer length in bytes.
 * @param prefix Text to prepend before the formatted value.
 * @param value Floating-point value to format.
 */
void AppInferenceLog_FormatFloatTenths(char *dst, size_t dst_len,
		const char *prefix, float value) {
	if ((dst == NULL) || (dst_len == 0U) || (prefix == NULL)) {
		return;
	}

	const bool negative = (value < 0.0f);
	const float magnitude = negative ? -value : value;
	long whole = (long) magnitude;
	float fraction = magnitude - (float) whole;

	unsigned tenths = (unsigned) (fraction * 10.0f + 0.5f);
	if (tenths >= 10U) {
		tenths = 0U;
		whole += 1L;
	}

	(void) snprintf(dst, dst_len, "%s%s%ld.%01u\r\n", prefix,
			negative ? "-" : "", whole, tenths);
}
