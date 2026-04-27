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
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

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

  DebugConsole_Printf(
      "[FAULT] %s PC=0x%08lX LR=0x%08lX CFSR=0x%08lX HFSR=0x%08lX DFSR=0x%08lX AFSR=0x%08lX MMFAR=0x%08lX BFAR=0x%08lX SFSR=0x%08lX SFAR=0x%08lX\r\n",
      fault_name,
      (unsigned long)stacked_pc,
      (unsigned long)stacked_lr,
      (unsigned long)fault_cfsr,
      (unsigned long)fault_hfsr,
      (unsigned long)fault_dfsr,
      (unsigned long)fault_afsr,
      (unsigned long)fault_mmar,
      (unsigned long)fault_bfar,
      (unsigned long)fault_sfsr,
      (unsigned long)fault_sfar);
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

  (void)stacked_r0;
  (void)stacked_r1;
  (void)stacked_r2;
  (void)stacked_r3;
  (void)stacked_r12;
  (void)stacked_psr;
  (void)exc_lr;

  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("HardFault", stacked_pc, stacked_lr);
  DebugConsole_Printf(
      "[FAULT] HardFault regs SP=0x%08lX R0=0x%08lX R1=0x%08lX R2=0x%08lX R3=0x%08lX R12=0x%08lX PSR=0x%08lX\r\n",
      (unsigned long)stacked_regs,
      (unsigned long)stacked_r0,
      (unsigned long)stacked_r1,
      (unsigned long)stacked_r2,
      (unsigned long)stacked_r3,
      (unsigned long)stacked_r12,
      (unsigned long)stacked_psr);
  __BKPT(0);
  while (1)
  {
    __NOP();
  }
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
void MemManage_Handler(void)
{
  /* USER CODE BEGIN MemoryManagement_IRQn 0 */
  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("MemManage", 0U, 0U);
  /* USER CODE END MemoryManagement_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_MemoryManagement_IRQn 0 */
    /* USER CODE END W1_MemoryManagement_IRQn 0 */
  }
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
  /* USER CODE BEGIN BusFault_IRQn 0 */
  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("BusFault", 0U, 0U);
  /* USER CODE END BusFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_BusFault_IRQn 0 */
    /* USER CODE END W1_BusFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
  /* USER CODE BEGIN UsageFault_IRQn 0 */
  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("UsageFault", 0U, 0U);
  /* USER CODE END UsageFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_UsageFault_IRQn 0 */
    /* USER CODE END W1_UsageFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Secure fault.
  */
void SecureFault_Handler(void)
{
  /* USER CODE BEGIN SecureFault_IRQn 0 */
  IT_EnterFaultLedState();
  IT_LogFaultSnapshot("SecureFault", 0U, 0U);
  /* USER CODE END SecureFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_SecureFault_IRQn 0 */
    /* USER CODE END W1_SecureFault_IRQn 0 */
  }
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

/**
  * @brief Forward the NPU interrupt lines into the ATON runtime handler.
  *
  * Cube's startup file exposes four NPU vectors. The runtime installs the
  * ATON-standard handler, so we forward each vector to keep the completion
  * path alive even if the active line changes.
  */
void NPU0_IRQHandler(void)
{
  CDNN0_IRQHandler();
}

void NPU1_IRQHandler(void)
{
  CDNN0_IRQHandler();
}

void NPU2_IRQHandler(void)
{
  CDNN0_IRQHandler();
}

void NPU3_IRQHandler(void)
{
  CDNN0_IRQHandler();
}

/* USER CODE END 1 */
