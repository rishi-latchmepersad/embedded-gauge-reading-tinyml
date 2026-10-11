/* USER CODE BEGIN Header */
/**
 *******************************************************************************
 * @file    app_image_cleanup.h
 * @brief   Background cleanup helpers for captured gauge images.
 *******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_IMAGE_CLEANUP_H
#define __APP_IMAGE_CLEANUP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tx_api.h"

/**
 * @brief Start the low-priority captured-image maintenance worker.
 *
 * The worker services pending FileX media flushes. Pruning is called only at
 * the stopped-camera boundary, using a schedule retained across Standby resets.
 *
 * @retval TX_SUCCESS when the thread already exists or is created.
 * @retval ThreadX status code on creation failure.
 */
UINT AppImageCleanup_Start(void);

/**
 * @brief Retain compatibility with boot callers; scheduling now uses retained RTC time.
 *
 * @param boot_tick System tick captured near boot.
 * @retval TX_SUCCESS always.
 */
UINT AppImageCleanup_SetBootTick(ULONG boot_tick);

/**
 * @brief Prune due captures after camera/AI ownership ends, before storage flush.
 * @retval FX_SUCCESS when completed/not due; otherwise original storage status.
 * @sideeffects Preserves last 24 hours and newest older capture per 10 minutes.
 */
UINT AppImageCleanup_RunDueBeforeSleep(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_IMAGE_CLEANUP_H */
