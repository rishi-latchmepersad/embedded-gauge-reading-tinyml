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
 * @brief Start the low-priority captured-image cleanup worker.
 *
 * The worker keeps the newest capture in each 10-minute bucket and deletes
 * older images from the same bucket.
 *
 * @retval TX_SUCCESS when the thread already exists or is created.
 * @retval ThreadX status code on creation failure.
 */
UINT AppImageCleanup_Start(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_IMAGE_CLEANUP_H */
