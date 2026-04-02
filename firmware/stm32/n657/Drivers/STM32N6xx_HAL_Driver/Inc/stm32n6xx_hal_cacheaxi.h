/**
  ******************************************************************************
  * @file    stm32n6xx_hal_cacheaxi.h
  * @author  MCD Application Team
  * @brief   Header file of CACHEAXI HAL module.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#ifndef STM32N6xx_HAL_CACHEAXI_H
#define STM32N6xx_HAL_CACHEAXI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal_def.h"

/** @addtogroup STM32N6xx_HAL_Driver
  * @{
  */

#if defined (CACHEAXI)
/** @addtogroup CACHEAXI
  * @{
  */

/* Exported types ------------------------------------------------------------*/

/** @defgroup CACHEAXI_Exported_Types CACHEAXI Exported Types
  * @{
  */

/**
  * @brief  HAL State structures definition
  */
typedef enum
{
  HAL_CACHEAXI_STATE_RESET   = 0x00U,
  HAL_CACHEAXI_STATE_READY   = 0x01U,
  HAL_CACHEAXI_STATE_BUSY    = 0x02U,
  HAL_CACHEAXI_STATE_TIMEOUT = 0x05U,
  HAL_CACHEAXI_STATE_ERROR   = 0x06U,
} HAL_CACHEAXI_StateTypeDef;

/** @defgroup CACHEAXI_Configuration_Structure_definition CACHEAXI Configuration Structure definition
  * @brief  CACHEAXI Configuration Structure definition
  * @{
  */
#if (USE_HAL_CACHEAXI_REGISTER_CALLBACKS == 1)
typedef struct __CACHEAXI_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_CACHEAXI_REGISTER_CALLBACKS */
{
  CACHEAXI_TypeDef               *Instance;
  __IO HAL_CACHEAXI_StateTypeDef State;
  __IO uint32_t                  ErrorCode;

#if (USE_HAL_CACHEAXI_REGISTER_CALLBACKS == 1)
  void (* ErrorCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
  void (* CleanByAddrCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
  void (* InvalidateCompleteCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
  void (* CleanAndInvalidateByAddrCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
  void (* MspInitCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
  void (* MspDeInitCallback)(struct __CACHEAXI_HandleTypeDef *hcacheaxi);
#endif /* USE_HAL_CACHEAXI_REGISTER_CALLBACKS */
} CACHEAXI_HandleTypeDef;

#if (USE_HAL_CACHEAXI_REGISTER_CALLBACKS == 1)
typedef enum
{
  HAL_CACHEAXI_CLEAN_BY_ADDRESS_CB_ID                 = 0x00U,
  HAL_CACHEAXI_CLEAN_AND_INVALIDATE_BY_ADDRESS_CB_ID  = 0x01U,
  HAL_CACHEAXI_INVALIDATE_COMPLETE_CB_ID              = 0x02U,
  HAL_CACHEAXI_ERROR_CB_ID                            = 0x03U,
  HAL_CACHEAXI_MSPINIT_CB_ID                          = 0x04U,
  HAL_CACHEAXI_MSPDEINIT_CB_ID                        = 0x05U
} HAL_CACHEAXI_CallbackIDTypeDef;

typedef void (*pCACHEAXI_CallbackTypeDef)(CACHEAXI_HandleTypeDef *hcacheaxi);
#endif /* USE_HAL_CACHEAXI_REGISTER_CALLBACKS */

/* Exported constants -------------------------------------------------------*/

/** @defgroup CACHEAXI_Error_Code CACHEAXI Error Code
  * @{
  */
#define HAL_CACHEAXI_ERROR_NONE              0x00000000U
#define HAL_CACHEAXI_ERROR_TIMEOUT           0x00000010U
#define HAL_CACHEAXI_ERROR_EVICTION_CLEAN    0x00000020U
#define HAL_CACHEAXI_ERROR_INVALID_OPERATION 0x00000040U
#if (USE_HAL_CACHEAXI_REGISTER_CALLBACKS == 1)
#define HAL_CACHEAXI_ERROR_INVALID_CALLBACK  0x00000080U
#endif /* USE_HAL_CACHEAXI_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @defgroup CACHEAXI_Monitor_Type Monitor type
  * @{
  */
#define CACHEAXI_MONITOR_READ_HIT         CACHEAXI_CR1_RHITMEN
#define CACHEAXI_MONITOR_READ_MISS        CACHEAXI_CR1_RMISSMEN
#define CACHEAXI_MONITOR_WRITE_HIT        CACHEAXI_CR1_WHITMEN
#define CACHEAXI_MONITOR_WRITE_MISS       CACHEAXI_CR1_WMISSMEN
#define CACHEAXI_MONITOR_READALLOC_MISS   CACHEAXI_CR1_RAMMEN
#define CACHEAXI_MONITOR_WRITEALLOC_MISS  CACHEAXI_CR1_WAMMEN
#define CACHEAXI_MONITOR_WRITETHROUGH     CACHEAXI_CR1_WTMEN
#define CACHEAXI_MONITOR_EVICTION         CACHEAXI_CR1_EVIMEN
#define CACHEAXI_MONITOR_ALL              (CACHEAXI_CR1_RHITMEN | CACHEAXI_CR1_RMISSMEN | \
                                           CACHEAXI_CR1_WHITMEN | CACHEAXI_CR1_WMISSMEN | \
                                           CACHEAXI_CR1_RAMMEN  | CACHEAXI_CR1_WAMMEN   | \
                                           CACHEAXI_CR1_WTMEN   | CACHEAXI_CR1_EVIMEN   )
/**
  * @}
  */

/** @defgroup CACHEAXI_Interrupts Interrupts
  * @{
  */
#define CACHEAXI_IT_BUSYEND              CACHEAXI_IER_BSYENDIE
#define CACHEAXI_IT_ERROR                CACHEAXI_IER_ERRIE
#define CACHEAXI_IT_CMDEND               CACHEAXI_IER_CMDENDIE
/**
  * @}
  */

/** @defgroup CACHEAXI_Flags Flags
  * @{
  */
#define CACHEAXI_FLAG_BUSY               CACHEAXI_SR_BUSYF
#define CACHEAXI_FLAG_BUSYEND            CACHEAXI_SR_BSYENDF
#define CACHEAXI_FLAG_ERROR              CACHEAXI_SR_ERRF
#define CACHEAXI_FLAG_BUSYCMD            CACHEAXI_SR_BUSYCMDF
#define CACHEAXI_FLAG_CMDEND             CACHEAXI_SR_CMDENDF
/**
  * @}
  */

/* Exported macros ----------------------------------------------------------*/
#define __HAL_CACHEAXI_ENABLE_IT(__HANDLE__, __INTERRUPT__) SET_BIT((__HANDLE__)->Instance->IER, (__INTERRUPT__))
#define __HAL_CACHEAXI_DISABLE_IT(__HANDLE__, __INTERRUPT__) CLEAR_BIT((__HANDLE__)->Instance->IER, (__INTERRUPT__))
#define __HAL_CACHEAXI_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) \
  ((READ_BIT((__HANDLE__)->Instance->IER, (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)
#define __HAL_CACHEAXI_GET_FLAG(__HANDLE__, __FLAG__) \
  ((READ_BIT((__HANDLE__)->Instance->SR, (__FLAG__)) != 0U) ? 1U : 0U)
#define __HAL_CACHEAXI_CLEAR_FLAG(__HANDLE__, __FLAG__) WRITE_REG((__HANDLE__)->Instance->FCR, (__FLAG__))

/* Exported functions -------------------------------------------------------*/
HAL_StatusTypeDef HAL_CACHEAXI_Init(CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_DeInit(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_MspInit(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_MspDeInit(CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_Enable(CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_Disable(CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_IsEnabled(const CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_Invalidate(CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_CleanByAddr(CACHEAXI_HandleTypeDef *hcacheaxi,
                                           const uint32_t *const pAddr, uint32_t Length);
HAL_StatusTypeDef HAL_CACHEAXI_CleanInvalidByAddr(CACHEAXI_HandleTypeDef *hcacheaxi,
                                                  const uint32_t *const pAddr, uint32_t Length);
HAL_StatusTypeDef HAL_CACHEAXI_Invalidate_IT(CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_CleanByAddr_IT(CACHEAXI_HandleTypeDef *hcacheaxi,
                                              const uint32_t *const pAddr, uint32_t Length);
HAL_StatusTypeDef HAL_CACHEAXI_CleanInvalidByAddr_IT(CACHEAXI_HandleTypeDef *hcacheaxi,
                                                     const uint32_t *const pAddr, uint32_t Length);
void              HAL_CACHEAXI_IRQHandler(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_ErrorCallback(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_CleanByAddrCallback(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_InvalidateCompleteCallback(CACHEAXI_HandleTypeDef *hcacheaxi);
void              HAL_CACHEAXI_CleanAndInvalidateByAddrCallback(CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetReadHitValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetReadMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetWriteHitValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetWriteMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetReadAllocMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetWriteAllocMissValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetWriteThroughValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_Monitor_GetEvictionValue(const CACHEAXI_HandleTypeDef *hcacheaxi);
HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Reset(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType);
HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Start(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType);
HAL_StatusTypeDef HAL_CACHEAXI_Monitor_Stop(CACHEAXI_HandleTypeDef *hcacheaxi, uint32_t MonitorType);
HAL_CACHEAXI_StateTypeDef HAL_CACHEAXI_GetState(const CACHEAXI_HandleTypeDef *hcacheaxi);
uint32_t          HAL_CACHEAXI_GetError(const CACHEAXI_HandleTypeDef *hcacheaxi);
/**
  * @}
  */
#endif /* CACHEAXI */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* STM32N6xx_HAL_CACHEAXI_H */
