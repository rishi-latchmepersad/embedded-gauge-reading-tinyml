/**
  ******************************************************************************
  * @file    stm32n6xx_hal_cacheaxi.c
  * @brief   Minimal CACHEAXI HAL support used by the NPU cache helper.
  ******************************************************************************
  */

#include "stm32n6xx_hal.h"

#if defined(CACHEAXI)

/* Keep the implementation intentionally small:
 * the application brings the peripheral clocks out of reset itself, and we only
 * need enough HAL surface area for the NPU cache helper to link and run.
 */

HAL_StatusTypeDef HAL_CACHEAXI_Init(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return HAL_ERROR;
  }

  hcacheaxi->ErrorCode = HAL_CACHEAXI_ERROR_NONE;
  hcacheaxi->State = HAL_CACHEAXI_STATE_READY;
  return HAL_CACHEAXI_Enable(hcacheaxi);
}

HAL_StatusTypeDef HAL_CACHEAXI_DeInit(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return HAL_ERROR;
  }

  (void)HAL_CACHEAXI_Disable(hcacheaxi);
  hcacheaxi->ErrorCode = HAL_CACHEAXI_ERROR_NONE;
  hcacheaxi->State = HAL_CACHEAXI_STATE_RESET;
  return HAL_OK;
}

__weak void HAL_CACHEAXI_MspInit(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

__weak void HAL_CACHEAXI_MspDeInit(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

HAL_StatusTypeDef HAL_CACHEAXI_Enable(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return HAL_ERROR;
  }

  SET_BIT(hcacheaxi->Instance->CR1, CACHEAXI_CR1_EN);
  hcacheaxi->State = HAL_CACHEAXI_STATE_READY;
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_Disable(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return HAL_ERROR;
  }

  CLEAR_BIT(hcacheaxi->Instance->CR1, CACHEAXI_CR1_EN);
  hcacheaxi->State = HAL_CACHEAXI_STATE_READY;
  return HAL_OK;
}

uint32_t HAL_CACHEAXI_IsEnabled(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return 0U;
  }

  return ((READ_BIT(hcacheaxi->Instance->CR1, CACHEAXI_CR1_EN) != 0U) ? 1U : 0U);
}

HAL_StatusTypeDef HAL_CACHEAXI_Invalidate(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if ((hcacheaxi == NULL) || (hcacheaxi->Instance == NULL))
  {
    return HAL_ERROR;
  }

  SET_BIT(hcacheaxi->Instance->CR1, CACHEAXI_CR1_CACHEINV);
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_CleanByAddr(CACHEAXI_HandleTypeDef *hcacheaxi,
                                           const uint32_t *const pAddr,
                                           uint32_t dSize)
{
  (void)hcacheaxi;
  (void)pAddr;
  (void)dSize;
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_CleanInvalidByAddr(CACHEAXI_HandleTypeDef *hcacheaxi,
                                                  const uint32_t *const pAddr,
                                                  uint32_t dSize)
{
  (void)hcacheaxi;
  (void)pAddr;
  (void)dSize;
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_Invalidate_IT(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  return HAL_CACHEAXI_Invalidate(hcacheaxi);
}

HAL_StatusTypeDef HAL_CACHEAXI_CleanByAddr_IT(CACHEAXI_HandleTypeDef *hcacheaxi,
                                              const uint32_t *const pAddr,
                                              uint32_t dSize)
{
  return HAL_CACHEAXI_CleanByAddr(hcacheaxi, pAddr, dSize);
}

HAL_StatusTypeDef HAL_CACHEAXI_CleanInvalidByAddr_IT(CACHEAXI_HandleTypeDef *hcacheaxi,
                                                     const uint32_t *const pAddr,
                                                     uint32_t dSize)
{
  return HAL_CACHEAXI_CleanInvalidByAddr(hcacheaxi, pAddr, dSize);
}

void HAL_CACHEAXI_IRQHandler(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

void HAL_CACHEAXI_ErrorCallback(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

void HAL_CACHEAXI_CleanByAddrCallback(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

void HAL_CACHEAXI_InvalidateCompleteCallback(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

void HAL_CACHEAXI_CleanAndInvalidateByAddrCallback(CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
}

uint32_t HAL_CACHEAXI_Monitor_GetReadHitValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetReadMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetWriteHitValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetWriteMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetReadAllocMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetWriteAllocMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetWriteThroughValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

uint32_t HAL_CACHEAXI_Monitor_GetEvictionValue(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  UNUSED(hcacheaxi);
  return 0U;
}

HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Reset(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType)
{
  (void)hcacheaxi;
  (void)MonitorType;
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Start(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType)
{
  (void)hcacheaxi;
  (void)MonitorType;
  return HAL_OK;
}

HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Stop(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType)
{
  (void)hcacheaxi;
  (void)MonitorType;
  return HAL_OK;
}

HAL_CACHEAXI_StateTypeDef HAL_CACHEAXI_GetState(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if (hcacheaxi == NULL)
  {
    return HAL_CACHEAXI_STATE_RESET;
  }

  return hcacheaxi->State;
}

uint32_t HAL_CACHEAXI_GetError(const CACHEAXI_HandleTypeDef *hcacheaxi)
{
  if (hcacheaxi == NULL)
  {
    return HAL_CACHEAXI_ERROR_INVALID_OPERATION;
  }

  return hcacheaxi->ErrorCode;
}

#endif /* CACHEAXI */
