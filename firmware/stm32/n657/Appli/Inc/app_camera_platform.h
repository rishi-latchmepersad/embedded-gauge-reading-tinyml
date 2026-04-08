/*
 *******************************************************************************
 * @file    app_camera_platform.h
 * @brief   Low-level camera board support helpers.
 *******************************************************************************
 */

#ifndef __APP_CAMERA_PLATFORM_H
#define __APP_CAMERA_PLATFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "main.h"
#include "tx_api.h"

HAL_StatusTypeDef CameraPlatform_ReadImx335ChipId(uint8_t *chip_id);
void CameraPlatform_ResetImx335Module(void);
void CameraPlatform_CmwEnablePin(int value);
void CameraPlatform_CmwShutdownPin(int value);
void CameraPlatform_CmwDelay(uint32_t delay_ms);
DCMIPP_HandleTypeDef *CameraPlatform_GetCaptureDcmippHandle(void);
int32_t CameraPlatform_GetTickMs(void);
ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_PLATFORM_H */
