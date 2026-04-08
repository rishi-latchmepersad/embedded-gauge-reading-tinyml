/*
 *******************************************************************************
 * @file    app_inference_log_utils.h
 * @brief   Small helpers shared by the inference logging path.
 *******************************************************************************
 */

#ifndef __APP_INFERENCE_LOG_UTILS_H
#define __APP_INFERENCE_LOG_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Format a floating-point value with one decimal place for embedded logs. */
void AppInferenceLog_FormatFloatTenths(char *dst, size_t dst_len,
		const char *prefix, float value);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_LOG_UTILS_H */
