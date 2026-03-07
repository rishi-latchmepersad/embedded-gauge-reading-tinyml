/**
  ******************************************************************************
  * @file    stm32n6xx_nucleo_bus.c
  * @brief   Minimal Nucleo I2C2 bus wrapper for the ST camera middleware.
  ******************************************************************************
  */

#include "stm32n6xx_nucleo_bus.h"
#include "main.h"

extern I2C_HandleTypeDef hi2c2;

static int32_t BSP_I2C2_StatusToError(HAL_StatusTypeDef status) {
	switch (status) {
	case HAL_OK:
		return BSP_ERROR_NONE;
	case HAL_BUSY:
		return BSP_ERROR_BUSY;
	case HAL_TIMEOUT:
		return BSP_ERROR_BUS_TRANSACTION_FAILURE;
	case HAL_ERROR:
	default:
		return BSP_ERROR_PERIPH_FAILURE;
	}
}

int32_t BSP_I2C2_Init(void) {
	return (hi2c2.Instance == I2C2) ? BSP_ERROR_NONE : BSP_ERROR_NO_INIT;
}

int32_t BSP_I2C2_DeInit(void) {
	return BSP_ERROR_NONE;
}

int32_t BSP_I2C2_ReadReg16(uint16_t dev_addr, uint16_t reg, uint8_t *pdata,
		uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Read(&hi2c2, dev_addr, reg,
			I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return BSP_I2C2_StatusToError(status);
}

int32_t BSP_I2C2_WriteReg16(uint16_t dev_addr, uint16_t reg, uint8_t *pdata,
		uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Write(&hi2c2, dev_addr, reg,
			I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return BSP_I2C2_StatusToError(status);
}
