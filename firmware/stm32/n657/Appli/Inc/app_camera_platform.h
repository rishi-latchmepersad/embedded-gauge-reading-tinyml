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

#include <stdbool.h>
#include <stdint.h>
#include "main.h"
#include "tx_api.h"

HAL_StatusTypeDef CameraPlatform_ReadImx335ChipId(uint8_t *chip_id);
UINT CameraPlatform_ProbeBCamsImx(void);
void CameraPlatform_ResetImx335Module(void);
void CameraPlatform_CmwEnablePin(int value);
void CameraPlatform_CmwShutdownPin(int value);
void CameraPlatform_CmwDelay(uint32_t delay_ms);
DCMIPP_HandleTypeDef *CameraPlatform_GetCaptureDcmippHandle(void);
int32_t CameraPlatform_GetTickMs(void);
ULONG CameraPlatform_MillisecondsToTicks(uint32_t timeout_ms);
bool CameraPlatform_SeedImx335ExposureGain(void);
bool CameraPlatform_AdjustImx335ExposureGain(bool brighten);
bool CameraPlatform_EnableImx335AutoExposure(void);
bool CameraPlatform_DisableImx335AutoExposure(void);
bool CameraPlatform_LogImx335AutoExposureState(const char *reason);
void CameraPlatform_ReapplyImx335TestPattern(void);
bool CameraPlatform_StartImx335Stream(void);
bool CameraPlatform_PrepareDcmippSnapshot(void);
bool CameraPlatform_StartDcmippSnapshot(void);
bool CameraPlatform_ConfigureCsiLineByteProbe(void);
int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);
int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);

#ifdef __cplusplus
}
#endif

#endif /* __APP_CAMERA_PLATFORM_H */
