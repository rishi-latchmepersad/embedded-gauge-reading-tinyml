/**
  ******************************************************************************
  * @file    main.c
  * @brief   Retained STM32N6 Standby wake loader.
  *
  * This image is copied by the cold-boot FSBL into retained SRAM1. Standby
  * wake starts here directly, so the loader restores xSPI2, copies the large
  * application into its normal SRAM1 destination, and branches to its reset
  * handler without using external RAM, UART, or an RTOS.
  ******************************************************************************
  */

#include "main.h"
#include <string.h>
#include "stm32n6xx_ll_exti.h"

/* The signed application payload begins after its 0x400-byte STM2 header. */
#define WAKE_APP_FLASH_BASE       (0x70100400UL)
#define WAKE_APP_RAM_BASE         (0x34020400UL)
#define WAKE_APP_COPY_SIZE        (0x00060000UL)

XSPI_HandleTypeDef hxspi2;
UART_HandleTypeDef hlpuart1;

static void WakeLoader_ClockConfig(void);
static void WakeLoader_DebugLedInit(void);
static void WakeLoader_DebugLedMark(uint16_t pin);
static void WakeLoader_UartInit(void);
static void WakeLoader_Log(const char *message);
static void WakeLoader_Xspi2Init(void);
static void WakeLoader_LeaveXspi2DeepPowerDown(void);
static void WakeLoader_CopyApplication(void);
static void WakeLoader_JumpToApplication(void);

/**
  * @brief Clear an RTC wake-up event if it is delivered while the loader is
  *        restoring the application.
  * @retval None.
  * @sideeffect Clears the RTC wake-up flag and prevents the weak startup
  *              handler from trapping the retained loader in an infinite loop.
  */
void RTC_S_IRQHandler(void)
{
  /* Keep the secure wake source quiescent while the retained loader restores
   * clocks and xSPI2.  SMISR is the secure RTC status view on STM32N6. */
  if (((RTC->SMISR & RTC_SMISR_WUTMF) != 0U)
      || ((RTC->SR & RTC_SR_WUTF) != 0U))
  {
    RTC->SCR = RTC_SCR_CWUTF;
  }
  LL_EXTI_ClearRisingFlag_0_31(LL_EXTI_LINE_17);
  LL_EXTI_ClearFallingFlag_0_31(LL_EXTI_LINE_17);
}

/**
  * @brief  Enter a permanent failure loop for an unrecoverable wake error.
  * @retval None.
  * @sideeffect Stops execution while leaving the retained wake image intact.
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
    __NOP();
  }
}

/**
  * @brief  Restore the application image after Standby wake.
  * @retval None; control transfers to the application reset handler.
  * @sideeffect Reconfigures clocks and xSPI2, copies the application into
  *              AXISRAM1, cleans caches, and changes VTOR and MSP.
  */
int main(void)
{
  /* The loader stack, data, and code all reside in retained SRAM1. */
  WakeLoader_DebugLedInit();
  WakeLoader_DebugLedMark(GPIO_PIN_8); /* blue: execution reached loader */

  if (HAL_Init() != HAL_OK)
  {
    Error_Handler();
  }
  WakeLoader_DebugLedMark(GPIO_PIN_10); /* red: HAL initialization completed */

  WakeLoader_ClockConfig();
  WakeLoader_DebugLedMark(GPIO_PIN_0); /* green: clock tree restored */

  /* Reinitialize the same LPUART1 pins used by the application so the loader
   * can identify a wake failure before the application image is copied. */
  WakeLoader_UartInit();
  WakeLoader_Log("[WAKE] retained loader entered; clocks restored.\r\n");

  WakeLoader_Xspi2Init();
  WakeLoader_Log("[WAKE] xSPI2 mapped; application copy starting.\r\n");
  WakeLoader_CopyApplication();
  WakeLoader_Log("[WAKE] application copy complete; jumping.\r\n");
  WakeLoader_JumpToApplication();

  /* The jump is expected to be non-returning. */
  Error_Handler();
  return 0;
}

/**
  * @brief  Configure the Nucleo RGB LED pins for wake-loader diagnostics.
  * @retval None.
  * @sideeffect Enables GPIOG and drives all three active-low LEDs off.
  *              These markers are intentionally independent of UART and xSPI2.
  */
static void WakeLoader_DebugLedInit(void)
{
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOG_CLK_ENABLE();
  gpio.Pin = GPIO_PIN_0 | GPIO_PIN_8 | GPIO_PIN_10;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &gpio);
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_0 | GPIO_PIN_8 | GPIO_PIN_10,
                   GPIO_PIN_SET);
}

/**
  * @brief  Turn on one active-low diagnostic LED marker.
  * @param  pin GPIOG pin carrying the marker.
  * @retval None.
  * @sideeffect Drives the selected Nucleo LED low and leaves earlier markers
  *              unchanged so the final LED combination identifies the stage.
  */
static void WakeLoader_DebugLedMark(uint16_t pin)
{
  HAL_GPIO_WritePin(GPIOG, pin, GPIO_PIN_RESET);
}

/**
  * @brief  Initialize LPUART1 for the retained-loader diagnostic messages.
  * @retval None; unrecoverable UART errors enter Error_Handler().
  * @sideeffect Configures PE5/PE6 and the LPUART1 peripheral at 115200 baud.
  */
static void WakeLoader_UartInit(void)
{
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 115200U;
  hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
  hlpuart1.Init.StopBits = UART_STOPBITS_1;
  hlpuart1.Init.Parity = UART_PARITY_NONE;
  hlpuart1.Init.Mode = UART_MODE_TX_RX;
  hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;

  if (HAL_UART_Init(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  Emit a short loader-stage message without libc or an RTOS.
  * @param  message NUL-terminated diagnostic string.
  * @retval None; transmit failures enter Error_Handler().
  * @sideeffect Blocks until the short message has left LPUART1.
  */
static void WakeLoader_Log(const char *message)
{
  const uint16_t length = (uint16_t)strlen(message);

  if (HAL_UART_Transmit(&hlpuart1, (uint8_t *)message, length,
                        HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  Restore the clock tree needed by the xSPI2 boot read.
  * @retval None.
  * @sideeffect Starts HSI, PLL1/PLL4, and restores the CPU and peripheral
  *              clock dividers used by the existing FSBL.
  */
static void WakeLoader_ClockConfig(void)
{
  RCC_OscInitTypeDef osc = {0};
  RCC_ClkInitTypeDef clk = {0};
  RCC_PeriphCLKInitTypeDef periph = {0};

  if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK ||
      HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  osc.HSIState = RCC_HSI_ON;
  osc.HSIDiv = RCC_HSI_DIV1;
  osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  osc.PLL1.PLLState = RCC_PLL_NONE;
  osc.PLL2.PLLState = RCC_PLL_NONE;
  osc.PLL3.PLLState = RCC_PLL_NONE;
  osc.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&osc) != HAL_OK)
  {
    Error_Handler();
  }

  periph.PeriphClockSelection = RCC_PERIPHCLK_TIM;
  periph.TIMPresSelection = RCC_TIMPRES_DIV1;
  if (HAL_RCCEx_PeriphCLKConfig(&periph) != HAL_OK)
  {
    Error_Handler();
  }

  osc.OscillatorType = RCC_OSCILLATORTYPE_NONE;
  osc.PLL1.PLLState = RCC_PLL_ON;
  osc.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
  osc.PLL1.PLLM = 4;
  osc.PLL1.PLLN = 75;
  osc.PLL1.PLLFractional = 0;
  osc.PLL1.PLLP1 = 1;
  osc.PLL1.PLLP2 = 1;
  osc.PLL4.PLLState = RCC_PLL_ON;
  osc.PLL4.PLLSource = RCC_PLLSOURCE_HSI;
  osc.PLL4.PLLM = 1;
  osc.PLL4.PLLN = 25;
  osc.PLL4.PLLFractional = 0;
  osc.PLL4.PLLP1 = 1;
  osc.PLL4.PLLP2 = 1;
  if (HAL_RCC_OscConfig(&osc) != HAL_OK)
  {
    Error_Handler();
  }

  clk.ClockType = RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_HCLK |
                  RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 |
                  RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_PCLK5 |
                  RCC_CLOCKTYPE_PCLK4;
  clk.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  clk.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  clk.AHBCLKDivider = RCC_HCLK_DIV4;
  clk.APB1CLKDivider = RCC_APB1_DIV1;
  clk.APB2CLKDivider = RCC_APB2_DIV1;
  clk.APB4CLKDivider = RCC_APB4_DIV1;
  clk.APB5CLKDivider = RCC_APB5_DIV1;
  clk.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  clk.IC1Selection.ClockDivider = 2;
  clk.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  clk.IC2Selection.ClockDivider = 3;
  clk.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  clk.IC6Selection.ClockDivider = 4;
  clk.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  clk.IC11Selection.ClockDivider = 3;
  if (HAL_RCC_ClockConfig(&clk) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  Reinitialize the MX25UM51245G in Octal STR memory-mapped mode.
  * @retval None.
  * @sideeffect Enables xSPI2/XSPIM, exits NOR deep-power-down, and maps the
  *              external flash at 0x70000000.
  */
static void WakeLoader_Xspi2Init(void)
{
  XSPIM_CfgTypeDef manager = {0};
  XSPI_MemoryMappedTypeDef mapped = {0};
  XSPI_RegularCmdTypeDef cmd = {0};
  uint8_t sopi = 0x01U;

  hxspi2.Instance = XSPI2;
  hxspi2.Init.FifoThresholdByte = 4;
  hxspi2.Init.MemoryMode = HAL_XSPI_SINGLE_MEM;
  hxspi2.Init.MemoryType = HAL_XSPI_MEMTYPE_MACRONIX;
  hxspi2.Init.MemorySize = HAL_XSPI_SIZE_512MB;
  hxspi2.Init.ChipSelectHighTimeCycle = 2;
  hxspi2.Init.FreeRunningClock = HAL_XSPI_FREERUNCLK_DISABLE;
  hxspi2.Init.ClockMode = HAL_XSPI_CLOCK_MODE_0;
  hxspi2.Init.WrapSize = HAL_XSPI_WRAP_NOT_SUPPORTED;
  hxspi2.Init.ClockPrescaler = 0;
  hxspi2.Init.SampleShifting = HAL_XSPI_SAMPLE_SHIFT_NONE;
  hxspi2.Init.DelayHoldQuarterCycle = HAL_XSPI_DHQC_ENABLE;
  hxspi2.Init.ChipSelectBoundary = HAL_XSPI_BONDARYOF_NONE;
  hxspi2.Init.MaxTran = 0;
  hxspi2.Init.Refresh = 0;
  hxspi2.Init.MemorySelect = HAL_XSPI_CSSEL_NCS1;
  if (HAL_XSPI_Init(&hxspi2) != HAL_OK)
  {
    Error_Handler();
  }

  manager.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
  manager.IOPort = HAL_XSPIM_IOPORT_2;
  manager.Req2AckTime = 1;
  if (HAL_XSPIM_Config(&hxspi2, &manager,
                      HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }

  /* The application deliberately puts the MX25UM51245G into deep power-down
   * before Standby.  Macronix requires a one-line NOP to release that state;
   * WREN and the configuration-register command below are ignored until the
   * release cycle has completed. */
  WakeLoader_LeaveXspi2DeepPowerDown();

  cmd.OperationType = HAL_XSPI_OPTYPE_COMMON_CFG;
  cmd.IOSelect = HAL_XSPI_SELECT_IO_7_0;
  cmd.Instruction = 0x06U;
  cmd.InstructionMode = HAL_XSPI_INSTRUCTION_1_LINE;
  cmd.InstructionWidth = HAL_XSPI_INSTRUCTION_8_BITS;
  cmd.InstructionDTRMode = HAL_XSPI_INSTRUCTION_DTR_DISABLE;
  cmd.AddressMode = HAL_XSPI_ADDRESS_NONE;
  cmd.AlternateBytesMode = HAL_XSPI_ALT_BYTES_NONE;
  cmd.DataMode = HAL_XSPI_DATA_NONE;
  cmd.DummyCycles = 0;
  cmd.DQSMode = HAL_XSPI_DQS_DISABLE;
  if (HAL_XSPI_Command(&hxspi2, &cmd,
                       HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }

  cmd.Instruction = 0x72U;
  cmd.AddressMode = HAL_XSPI_ADDRESS_1_LINE;
  cmd.AddressWidth = HAL_XSPI_ADDRESS_32_BITS;
  cmd.AddressDTRMode = HAL_XSPI_ADDRESS_DTR_DISABLE;
  cmd.Address = 0;
  cmd.DataMode = HAL_XSPI_DATA_1_LINE;
  cmd.DataDTRMode = HAL_XSPI_DATA_DTR_DISABLE;
  cmd.DataLength = 1;
  if (HAL_XSPI_Command(&hxspi2, &cmd,
                       HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK ||
      HAL_XSPI_Transmit(&hxspi2, &sopi,
                        HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }
  HAL_Delay(1);

  cmd.OperationType = HAL_XSPI_OPTYPE_READ_CFG;
  cmd.Instruction = 0xEC13U;
  cmd.InstructionMode = HAL_XSPI_INSTRUCTION_8_LINES;
  cmd.InstructionWidth = HAL_XSPI_INSTRUCTION_16_BITS;
  cmd.InstructionDTRMode = HAL_XSPI_INSTRUCTION_DTR_DISABLE;
  cmd.AddressMode = HAL_XSPI_ADDRESS_8_LINES;
  cmd.AddressWidth = HAL_XSPI_ADDRESS_32_BITS;
  cmd.AddressDTRMode = HAL_XSPI_ADDRESS_DTR_DISABLE;
  cmd.Address = 0;
  cmd.AlternateBytesMode = HAL_XSPI_ALT_BYTES_NONE;
  cmd.DataMode = HAL_XSPI_DATA_8_LINES;
  cmd.DataDTRMode = HAL_XSPI_DATA_DTR_DISABLE;
  cmd.DummyCycles = 20U;
  cmd.DQSMode = HAL_XSPI_DQS_DISABLE;
  cmd.DataLength = 0;
  if (HAL_XSPI_Command(&hxspi2, &cmd,
                       HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }

  cmd.OperationType = HAL_XSPI_OPTYPE_WRITE_CFG;
  cmd.Instruction = 0x12EDU;
  cmd.DummyCycles = 0;
  if (HAL_XSPI_Command(&hxspi2, &cmd,
                       HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }

  mapped.TimeOutActivation = HAL_XSPI_TIMEOUT_COUNTER_DISABLE;
  if (HAL_XSPI_MemoryMapped(&hxspi2, &mapped) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  Release the external NOR from deep power-down before reconfiguration.
  * @retval None; unrecoverable xSPI errors enter Error_Handler().
  * @sideeffect Sends the Macronix one-line NOP wake command and waits for the
  *              memory's minimum wake-up time.
  */
static void WakeLoader_LeaveXspi2DeepPowerDown(void)
{
  XSPI_RegularCmdTypeDef cmd = {0};

  cmd.OperationType = HAL_XSPI_OPTYPE_COMMON_CFG;
  cmd.IOSelect = HAL_XSPI_SELECT_IO_7_0;
  cmd.Instruction = 0x00U; /* MX25UM51245G SPI NOP/release command. */
  cmd.InstructionMode = HAL_XSPI_INSTRUCTION_1_LINE;
  cmd.InstructionWidth = HAL_XSPI_INSTRUCTION_8_BITS;
  cmd.InstructionDTRMode = HAL_XSPI_INSTRUCTION_DTR_DISABLE;
  cmd.AddressMode = HAL_XSPI_ADDRESS_NONE;
  cmd.AlternateBytesMode = HAL_XSPI_ALT_BYTES_NONE;
  cmd.DataMode = HAL_XSPI_DATA_NONE;
  cmd.DummyCycles = 0U;
  cmd.DQSMode = HAL_XSPI_DQS_DISABLE;

  if (HAL_XSPI_Command(&hxspi2, &cmd,
                       HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }

  /* The component driver specifies 30 us minimum; 1 ms leaves margin for
   * oscillator and regulator restart variation after Standby. */
  HAL_Delay(1U);
}

/**
  * @brief  Copy the application payload from mapped xSPI2 into AXISRAM1.
  * @retval None.
  * @sideeffect Writes the full-application destination and cleans stale CPU
  *              cache lines before execution is transferred.
  */
static void WakeLoader_CopyApplication(void)
{
  const uint32_t *src = (const uint32_t *)WAKE_APP_FLASH_BASE;
  uint32_t *dst = (uint32_t *)WAKE_APP_RAM_BASE;

  for (uint32_t i = 0U; i < (WAKE_APP_COPY_SIZE / sizeof(uint32_t)); i++)
  {
    dst[i] = src[i];
  }
  SCB_CleanInvalidateDCache();
  SCB_InvalidateICache();
}

/**
  * @brief  Start the restored application's normal reset sequence.
  * @retval None; this function does not return on valid application vectors.
  * @sideeffect Repoints VTOR, MSP, and MSPLIM, then branches to the restored
  *              application's Reset_Handler.
  */
static void WakeLoader_JumpToApplication(void)
{
  const uint32_t *vectors = (const uint32_t *)WAKE_APP_RAM_BASE;
  void (*reset_handler)(void) = (void (*)(void))vectors[1];

  if ((vectors[0] < 0x34020400UL) || (vectors[0] > 0x34100000UL) ||
      ((vectors[1] & 1U) == 0U))
  {
    Error_Handler();
  }

  __disable_irq();
  SysTick->CTRL = 0;
  SCB->VTOR = WAKE_APP_RAM_BASE;
  __set_MSP(vectors[0]);
  __set_MSPLIM(0);
  __DSB();
  __ISB();
  reset_handler();
}
