/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32n6xx_it.c
  * @brief   Interrupt Service Routines.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32n6xx_it.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "cmw_camera.h"
#include "debug_console.h"
#include "stm32n6xx_ll_lpuart.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <cmsis_compiler.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

extern volatile size_t app_ai_scalar_preprocess_last_row;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
extern DCMIPP_HandleTypeDef hdcmipp;
extern volatile uint32_t camera_capture_csi_irq_count;
extern volatile uint32_t camera_capture_dcmipp_irq_count;
extern void CDNN0_IRQHandler(void);
extern UART_HandleTypeDef hlpuart1;

static void IT_RawUartWrite(const char *line)
{
  if (line == NULL)
  {
    return;
  }

  while (*line != '\0')
  {
    while (!LL_LPUART_IsActiveFlag_TXE_TXFNF(LPUART1))
    {
      __NOP();
    }

    LL_LPUART_TransmitData8(LPUART1, (uint8_t)*line);
    line++;
  }
}

static void IT_RawUartWriteHex32(uint32_t value)
{
  static const char hex_digits[] = "0123456789ABCDEF";
  char text[11] = {'0', 'x', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};

  for (size_t i = 0U; i < 8U; i++)
  {
    const size_t shift = (7U - i) * 4U;
    text[2U + i] = hex_digits[(value >> shift) & 0xFU];
  }

  IT_RawUartWrite(text);
}

static void IT_EnterFaultLedState(void)
{
  /* Use the visible board LEDs to make fault state obvious even when the UART
   * console is wedged or the fault happens inside the logger path. */
  BSP_LED_On(LED_RED);
  BSP_LED_Off(LED_BLUE);
  BSP_LED_Off(LED_GREEN);
}

static void IT_LogFaultSnapshot(const char *fault_name, uint32_t stacked_pc, uint32_t stacked_lr)
{
  static volatile uint32_t fault_cfsr = 0U;
  static volatile uint32_t fault_hfsr = 0U;
  static volatile uint32_t fault_dfsr = 0U;
  static volatile uint32_t fault_afsr = 0U;
  static volatile uint32_t fault_mmar = 0U;
  static volatile uint32_t fault_bfar = 0U;
  static volatile uint32_t fault_sfsr = 0U;
  static volatile uint32_t fault_sfar = 0U;

  __disable_irq();

  fault_cfsr = SCB->CFSR;
  fault_hfsr = SCB->HFSR;
  fault_dfsr = SCB->DFSR;
  fault_afsr = SCB->AFSR;
  fault_mmar = SCB->MMFAR;
  fault_bfar = SCB->BFAR;
#if defined(SCB_SFSR_SFSR_Msk)
  fault_sfsr = SCB->SFSR;
  fault_sfar = SCB->SFAR;
#endif

  IT_RawUartWrite("[FAULT] ");
  IT_RawUartWrite(fault_name);
  IT_RawUartWrite(" PC=");
  IT_RawUartWriteHex32(stacked_pc);
  IT_RawUartWrite(" LR=");
  IT_RawUartWriteHex32(stacked_lr);
  IT_RawUartWrite(" CFSR=");
  IT_RawUartWriteHex32(fault_cfsr);
  IT_RawUartWrite(" HFSR=");
  IT_RawUartWriteHex32(fault_hfsr);
  IT_RawUartWrite(" DFSR=");
  IT_RawUartWriteHex32(fault_dfsr);
  IT_RawUartWrite(" AFSR=");
  IT_RawUartWriteHex32(fault_afsr);
  IT_RawUartWrite(" MMFAR=");
  IT_RawUartWriteHex32(fault_mmar);
  IT_RawUartWrite(" BFAR=");
  IT_RawUartWriteHex32(fault_bfar);
  IT_RawUartWrite(" SFSR=");
  IT_RawUartWriteHex32(fault_sfsr);
  IT_RawUartWrite(" SFAR=");
  IT_RawUartWriteHex32(fault_sfar);
  IT_RawUartWrite("\r\n");
}

void HardFault_Handler_C(uint32_t *stacked_regs, uint32_t exc_lr)
{
  const uint32_t stacked_r0 = stacked_regs[0];
  const uint32_t stacked_r1 = stacked_regs[1];
  const uint32_t stacked_r2 = stacked_regs[2];
  const uint32_t stacked_r3 = stacked_regs[3];
  const uint32_t stacked_r12 = stacked_regs[4];
  const uint32_t stacked_lr = stacked_regs[5];
  const uint32_t stacked_pc = stacked_regs[6];
  const uint32_t stacked_psr = stacked_regs[7];

  /* Log EXC_LR to determine MSP vs PSP context at fault time.
   * Bit 2 of EXC_LR: 0 = MSP was used (handler mode), 1 = PSP was used (thread mode).
   * Also log MSPLIM/PSPLIM so we can see which stack limit was violated. */
  const uint32_t msp_val   = __get_MSP();
  const uint32_t msplim_val = __get_MSPLIM();
  const uint32_t psp_val   = __get_PSP();
  const uint32_t psplim_val = __get_PSPLIM();

  (void)exc_lr;

  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("HardFault", stacked_pc, stacked_lr);
  IT_RawUartWrite("[FAULT] HardFault regs SP=");
  IT_RawUartWriteHex32((uint32_t)stacked_regs);
  IT_RawUartWrite(" R0=");
  IT_RawUartWriteHex32(stacked_r0);
  IT_RawUartWrite(" R1=");
  IT_RawUartWriteHex32(stacked_r1);
  IT_RawUartWrite(" R2=");
  IT_RawUartWriteHex32(stacked_r2);
  IT_RawUartWrite(" R3=");
  IT_RawUartWriteHex32(stacked_r3);
  IT_RawUartWrite(" R12=");
  IT_RawUartWriteHex32(stacked_r12);
  IT_RawUartWrite(" PSR=");
  IT_RawUartWriteHex32(stacked_psr);
  IT_RawUartWrite(" EXC_LR=");
  IT_RawUartWriteHex32(exc_lr);
  IT_RawUartWrite(" MSP=");
  IT_RawUartWriteHex32(msp_val);
  IT_RawUartWrite(" MSPLIM=");
  IT_RawUartWriteHex32(msplim_val);
  IT_RawUartWrite(" PSP=");
  IT_RawUartWriteHex32(psp_val);
  IT_RawUartWrite(" PSPLIM=");
  IT_RawUartWriteHex32(psplim_val);
  IT_RawUartWrite(" last_scalar_row=");
  IT_RawUartWriteHex32((uint32_t)app_ai_scalar_preprocess_last_row);
  IT_RawUartWrite("\r\n");
  IT_RawUartWrite("[FAULT] HardFault latched; staying in fault loop.\r\n");
  while (1)
  {
    __NOP();
  }
}

static void IT_LogContextFault(const char *fault_name, uint32_t *stacked_regs, uint32_t exc_lr)
{
  const uint32_t stacked_r0 = stacked_regs[0];
  const uint32_t stacked_r1 = stacked_regs[1];
  const uint32_t stacked_r2 = stacked_regs[2];
  const uint32_t stacked_r3 = stacked_regs[3];
  const uint32_t stacked_r12 = stacked_regs[4];
  const uint32_t stacked_lr = stacked_regs[5];
  const uint32_t stacked_pc = stacked_regs[6];
  const uint32_t stacked_psr = stacked_regs[7];

  (void)exc_lr;

  IT_EnterFaultLedState();
  IT_LogFaultSnapshot(fault_name, stacked_pc, stacked_lr);
  IT_RawUartWrite("[FAULT] ");
  IT_RawUartWrite(fault_name);
  IT_RawUartWrite(" regs SP=");
  IT_RawUartWriteHex32((uint32_t)stacked_regs);
  IT_RawUartWrite(" R0=");
  IT_RawUartWriteHex32(stacked_r0);
  IT_RawUartWrite(" R1=");
  IT_RawUartWriteHex32(stacked_r1);
  IT_RawUartWrite(" R2=");
  IT_RawUartWriteHex32(stacked_r2);
  IT_RawUartWrite(" R3=");
  IT_RawUartWriteHex32(stacked_r3);
  IT_RawUartWrite(" R12=");
  IT_RawUartWriteHex32(stacked_r12);
  IT_RawUartWrite(" PSR=");
  IT_RawUartWriteHex32(stacked_psr);
  IT_RawUartWrite(" EXC_LR=");
  IT_RawUartWriteHex32(exc_lr);
  IT_RawUartWrite(" last_scalar_row=");
  IT_RawUartWriteHex32((uint32_t)app_ai_scalar_preprocess_last_row);
  IT_RawUartWrite("\r\n");
  IT_RawUartWrite("[FAULT] ");
  IT_RawUartWrite(fault_name);
  IT_RawUartWrite(" latched; staying in fault loop.\r\n");
  while (1)
  {
    __NOP();
  }
}

void MemManage_Handler_C(uint32_t *stacked_regs, uint32_t exc_lr)
{
  IT_LogContextFault("MemManage", stacked_regs, exc_lr);
}

void BusFault_Handler_C(uint32_t *stacked_regs, uint32_t exc_lr)
{
  IT_LogContextFault("BusFault", stacked_regs, exc_lr);
}

void UsageFault_Handler_C(uint32_t *stacked_regs, uint32_t exc_lr)
{
  IT_LogContextFault("UsageFault", stacked_regs, exc_lr);
}

void SecureFault_Handler_C(uint32_t *stacked_regs, uint32_t exc_lr)
{
  IT_LogContextFault("SecureFault", stacked_regs, exc_lr);
}

/**
  * @brief Return the DCMIPP handle currently owning the camera pipeline.
  * @retval Middleware handle when initialized, otherwise CubeMX global handle.
  */
static DCMIPP_HandleTypeDef *IT_GetActiveDcmippHandle(void)
{
  DCMIPP_HandleTypeDef *cmw_handle = CMW_CAMERA_GetDCMIPPHandle();

  if ((cmw_handle != NULL) && (cmw_handle->Instance == DCMIPP))
  {
    return cmw_handle;
  }

  return &hdcmipp;
}

/* USER CODE END 0 */

/* External variables --------------------------------------------------------*/
extern TIM_HandleTypeDef htim5;

/* USER CODE BEGIN EV */
extern DCMIPP_HandleTypeDef hdcmipp;
/* Count how often the TIM5 timebase handler has to self-heal around a
 * corrupted handle. If this ever increments, we know we still have an upstream
 * overwrite to chase. */
static volatile uint32_t tim5_irq_guard_drop_count = 0U;

/* USER CODE END EV */

/******************************************************************************/
/*           Cortex Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void NMI_Handler(void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */

  /* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */
   while (1)
  {
  }
  /* USER CODE END NonMaskableInt_IRQn 1 */
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void HardFault_Handler(void) __attribute__((naked));
void HardFault_Handler(void)
{
  __asm volatile(
      "tst lr, #4                        \n"
      "ite eq                            \n"
      "mrseq r0, msp                     \n"
      "mrsne r0, psp                     \n"
      "mov r1, lr                        \n"
      "b HardFault_Handler_C             \n");
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void) __attribute__((naked));
void MemManage_Handler(void)
{
  __asm volatile(
      "tst lr, #4                        \n"
      "ite eq                            \n"
      "mrseq r0, msp                     \n"
      "mrsne r0, psp                     \n"
      "mov r1, lr                        \n"
      "b MemManage_Handler_C             \n");
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void BusFault_Handler(void) __attribute__((naked));
void BusFault_Handler(void)
{
  __asm volatile(
      "tst lr, #4                        \n"
      "ite eq                            \n"
      "mrseq r0, msp                     \n"
      "mrsne r0, psp                     \n"
      "mov r1, lr                        \n"
      "b BusFault_Handler_C              \n");
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void) __attribute__((naked));
void UsageFault_Handler(void)
{
  __asm volatile(
      "tst lr, #4                        \n"
      "ite eq                            \n"
      "mrseq r0, msp                     \n"
      "mrsne r0, psp                     \n"
      "mov r1, lr                        \n"
      "b UsageFault_Handler_C            \n");
}

/**
  * @brief This function handles Secure fault.
  */
void SecureFault_Handler(void) __attribute__((naked));
void SecureFault_Handler(void)
{
  __asm volatile(
      "tst lr, #4                        \n"
      "ite eq                            \n"
      "mrseq r0, msp                     \n"
      "mrsne r0, psp                     \n"
      "mov r1, lr                        \n"
      "b SecureFault_Handler_C           \n");
}

/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */

  /* USER CODE END DebugMonitor_IRQn 0 */
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}

/******************************************************************************/
/* STM32N6xx Peripheral Interrupt Handlers                                    */
/* Add here the Interrupt Handlers for the used peripherals.                  */
/* For the available peripheral interrupt handler names,                      */
/* please refer to the startup file (startup_stm32n6xx.s).                    */
/******************************************************************************/

/**
  * @brief This function handles EXTI Line13 interrupt.
  */
void EXTI13_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI13_IRQn 0 */

  /* USER CODE END EXTI13_IRQn 0 */
  BSP_PB_IRQHandler(BUTTON_USER);
  /* USER CODE BEGIN EXTI13_IRQn 1 */

  /* USER CODE END EXTI13_IRQn 1 */
}

/**
  * @brief This function handles TIM5 global interrupt.
  */
void TIM5_IRQHandler(void)
{
  /* USER CODE BEGIN TIM5_IRQn 0 */

  /* USER CODE END TIM5_IRQn 0 */
  /* Do not let a corrupted handle turn the tick interrupt into a hard fault.
   * A null or bogus Instance pointer will fault inside HAL_TIM_IRQHandler(),
   * so clear the update flag directly and keep the board alive for diagnosis. */
  if ((htim5.Instance == NULL) || (htim5.Instance != TIM5))
  {
    tim5_irq_guard_drop_count++;
    TIM5->SR &= ~TIM_SR_UIF;
    HAL_IncTick();
    return;
  }
  HAL_TIM_IRQHandler(&htim5);
  /* USER CODE BEGIN TIM5_IRQn 1 */

  /* USER CODE END TIM5_IRQn 1 */
}

/* USER CODE BEGIN 1 */

/**
  * @brief This function handles CSI global interrupt.
  */
void CSI_IRQHandler(void)
{
  camera_capture_csi_irq_count++;
  HAL_DCMIPP_CSI_IRQHandler(IT_GetActiveDcmippHandle());
}

/**
  * @brief This function handles DCMIPP global interrupt.
  */
void DCMIPP_IRQHandler(void)
{
  camera_capture_dcmipp_irq_count++;
  HAL_DCMIPP_IRQHandler(IT_GetActiveDcmippHandle());
}

/* USER CODE END 1 */
