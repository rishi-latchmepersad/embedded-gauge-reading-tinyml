/*
 *******************************************************************************
 * @file    app_inference_log_utils.c
 * @brief   Small helpers shared by the inference logging path.
 *******************************************************************************
 */

#include "app_inference_log_utils.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* Detect IEEE-754 infinities and NaNs so the UART formatter can preserve
 * their special textual form instead of flattening them into a fake zero. */
static bool AppInferenceLog_IsSpecialFloat(float value, const char **suffix_out) {
	union {
		float f;
		uint32_t u;
	} bits = {
		.f = value
	};

	if ((bits.u & 0x7F800000U) != 0x7F800000U) {
		return false;
	}

	if ((bits.u & 0x007FFFFFU) != 0U) {
		*suffix_out = "NaN";
	} else if ((bits.u & 0x80000000U) != 0U) {
		*suffix_out = "-Inf";
	} else {
		*suffix_out = "+Inf";
	}

	return true;
}

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

	/* Keep non-finite values visible in the UART log instead of collapsing them
	 * into a misleading numeric zero. */
	{
		const char *special = NULL;
		if (AppInferenceLog_IsSpecialFloat(value, &special)) {
			(void) snprintf(dst, dst_len, "%s%s\r\n", prefix, special);
			return;
		}
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

/**
 * @brief Format a floating-point value with six decimal places.
 *
 * The worker thread uses this when we need to see whether a value is really
 * pinned, or just rounded into the same tenths bucket by the compact logger.
 */
void AppInferenceLog_FormatFloatMicros(char *dst, size_t dst_len,
		const char *prefix, float value) {
	if ((dst == NULL) || (dst_len == 0U) || (prefix == NULL)) {
		return;
	}

	/* Keep non-finite values visible in the UART log instead of collapsing them
	 * into a misleading numeric zero. */
	{
		const char *special = NULL;
		if (AppInferenceLog_IsSpecialFloat(value, &special)) {
			(void) snprintf(dst, dst_len, "%s%s\r\n", prefix, special);
			return;
		}
	}

	const bool negative = (value < 0.0f);
	const double magnitude = negative ? -(double) value : (double) value;
	unsigned long whole = (unsigned long) magnitude;
	double fraction = magnitude - (double) whole;
	unsigned long micros = (unsigned long) (fraction * 1000000.0 + 0.5);

	if (micros >= 1000000UL) {
		micros = 0UL;
		whole += 1UL;
	}

	(void) snprintf(dst, dst_len, "%s%s%lu.%06lu\r\n", prefix,
			negative ? "-" : "", whole, micros);
}
