/*
 *******************************************************************************
 * @file    app_inference_log_config.h
 * @brief   Inference logging constants used by the ThreadX app.
 *******************************************************************************
 */

#ifndef __APP_INFERENCE_LOG_CONFIG_H
#define __APP_INFERENCE_LOG_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_LOG_THREAD_PRIORITY          14U
#define INFERENCE_LOG_DIRECTORY_NAME           "inference"
#define INFERENCE_LOG_FILE_NAME_LENGTH         32U
#define INFERENCE_LOG_ROW_MAX_LENGTH           48U
#define INFERENCE_LOG_NO_RTC_BLINK_ON_MS       200U
#define INFERENCE_LOG_NO_RTC_BLINK_OFF_MS      200U
#define INFERENCE_LOG_NO_RTC_RETRY_DELAY_MS    2000U

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_LOG_CONFIG_H */
