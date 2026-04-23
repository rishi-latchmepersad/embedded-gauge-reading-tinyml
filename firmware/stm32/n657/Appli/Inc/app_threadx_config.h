/*
 *******************************************************************************
 * @file    app_threadx_config.h
 * @brief   Shared ThreadX scheduling and heartbeat constants.
 *******************************************************************************
 */

#ifndef __APP_THREADX_CONFIG_H
#define __APP_THREADX_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal_gpio.h"

/* Thread priorities -------------------------------------------------------- */
#define CAMERA_INIT_THREAD_PRIORITY          9U
#define CAMERA_ISP_THREAD_PRIORITY          11U
#define CAMERA_HEARTBEAT_THREAD_PRIORITY    10U
#define CAMERA_AI_THREAD_PRIORITY           13U
#define BASELINE_RUNTIME_THREAD_PRIORITY    15U
#define IMAGE_CLEANUP_THREAD_PRIORITY       16U

/* Heartbeat timing --------------------------------------------------------- */
#define CAMERA_HEARTBEAT_PERIOD_MS          5000U
#define CAMERA_HEARTBEAT_PULSE_MS           1000U
#define CAMERA_HEARTBEAT_LED_GPIO_PORT      GPIOG
#define CAMERA_HEARTBEAT_LED_PIN            GPIO_PIN_0

/* Storage maintenance timing ---------------------------------------------- */
#define IMAGE_CLEANUP_PERIOD_MS            600000U

/* Camera middleware coordination ------------------------------------------ */
#define CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS    5000U

#ifdef __cplusplus
}
#endif

#endif /* __APP_THREADX_CONFIG_H */
