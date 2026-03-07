/**
  ******************************************************************************
  * @file    stm32n6xx_nucleo_bus.h
  * @brief   Minimal Nucleo I2C2 bus wrapper for the ST camera middleware.
  ******************************************************************************
  */

#ifndef STM32N6XX_NUCLEO_BUS_H
#define STM32N6XX_NUCLEO_BUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal.h"
#include "stm32n6xx_nucleo_errno.h"

int32_t BSP_I2C2_Init(void);
int32_t BSP_I2C2_DeInit(void);
int32_t BSP_I2C2_ReadReg16(uint16_t dev_addr, uint16_t reg, uint8_t *pdata,
		uint16_t length);
int32_t BSP_I2C2_WriteReg16(uint16_t dev_addr, uint16_t reg, uint8_t *pdata,
		uint16_t length);

#ifdef __cplusplus
}
#endif

#endif /* STM32N6XX_NUCLEO_BUS_H */
