/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_ai.h
 * @brief   Minimal AI runtime bootstrap helpers.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_AI_H
#define __APP_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/**
 * @brief Initialize the generated AI runtime package.
 *
 * This is step 1 only: we bring the runtime up and validate the generated
 * network package links correctly, but we do not execute inference yet.
 *
 * @retval true when the runtime init calls succeed.
 * @retval false when the model package fails to initialize.
 */
bool App_AI_Model_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
