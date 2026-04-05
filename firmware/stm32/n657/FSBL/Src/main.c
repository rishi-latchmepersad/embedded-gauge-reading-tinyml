/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include <stdio.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define FSBL_LED_ONLY_SMOKE_TEST 0

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

__IO uint32_t BspButtonState = BUTTON_RELEASED;

I2C_HandleTypeDef hi2c2;

UART_HandleTypeDef hlpuart1;

XSPI_HandleTypeDef hxspi2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_LPUART1_UART_Init(void);
static void MX_I2C2_Init(void);
static void MX_XSPI2_Init(void);
static void FSBL_BlinkLED(Led_TypeDef led, uint32_t n, uint32_t period_ms);
static void FSBL_LogAppImageState(void);
static void FSBL_TryBootApplication(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* Application image in external flash (after the 0x400-byte STM2 header) */
#define FSBL_APP_FLASH_BASE   (0x70100400UL)  /* raw binary in xSPI2 flash */
#define FSBL_APP_RAM_BASE     (0x34000400UL)  /* AXISRAM1 — matches app linker ROM origin */
#define FSBL_APP_MAX_SIZE     (0x80000UL)     /* 512 KB copy limit (safe upper bound) */

/* Init XSPI2 and put the MX25UM51245G into OctoSPI STR memory-mapped mode.
 *
 * Sequence (mirrors the ST Template_FSBL_LRUN / BSP component approach):
 *   1. Init XSPI2 in 1-line SPI mode (chip power-on default)
 *   2. Configure XSPIM port 2
 *   3. Send WriteEnable (SPI, 1-line)
 *   4. Send WriteCfg2Register addr=0 val=0x01 (SOPI=1) to enter OctoSPI STR
 *   5. Reconfigure XSPI2 HAL handle for 8-line STR mode (no re-init needed,
 *      just reconfigure the command width via TCR/DCR registers, done implicitly
 *      by the command structs below)
 *   6. Set READ_CFG command (8-line, 16-bit instruction 0xEC13, 6 dummy cycles)
 *   7. Set WRITE_CFG command (dummy — we are read-only for boot)
 *   8. Enter memory-mapped mode
 */
static void MX_XSPI2_Init(void)
{
  XSPIM_CfgTypeDef         sXspiManagerCfg = {0};
  XSPI_MemoryMappedTypeDef sMemMappedCfg   = {0};
  XSPI_RegularCmdTypeDef   sCommand        = {0};
  uint8_t                  reg             = 0x01U; /* SOPI bit */

  /* ---- Step 1: Init XSPI2 (1-line SPI mode, 50 MHz) ---- */
  hxspi2.Instance                    = XSPI2;
  hxspi2.Init.FifoThresholdByte      = 4;
  hxspi2.Init.MemoryMode             = HAL_XSPI_SINGLE_MEM;
  hxspi2.Init.MemoryType             = HAL_XSPI_MEMTYPE_MACRONIX;
  hxspi2.Init.MemorySize             = HAL_XSPI_SIZE_512MB;
  hxspi2.Init.ChipSelectHighTimeCycle= 2;
  hxspi2.Init.FreeRunningClock       = HAL_XSPI_FREERUNCLK_DISABLE;
  hxspi2.Init.ClockMode              = HAL_XSPI_CLOCK_MODE_0;
  hxspi2.Init.WrapSize               = HAL_XSPI_WRAP_NOT_SUPPORTED;
  hxspi2.Init.ClockPrescaler         = 0;
  hxspi2.Init.SampleShifting         = HAL_XSPI_SAMPLE_SHIFT_NONE;
  hxspi2.Init.DelayHoldQuarterCycle  = HAL_XSPI_DHQC_ENABLE;
  hxspi2.Init.ChipSelectBoundary     = HAL_XSPI_BONDARYOF_NONE;
  hxspi2.Init.MaxTran                = 0;
  hxspi2.Init.Refresh                = 0;
  hxspi2.Init.MemorySelect           = HAL_XSPI_CSSEL_NCS1;

  FSBL_BlinkLED(LED_GREEN, 1, 200);
  if (HAL_XSPI_Init(&hxspi2) != HAL_OK)
  {
    printf("[FSBL] XSPI2 HAL_XSPI_Init failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] XSPI2 HAL_XSPI_Init OK\r\n");

  /* ---- Step 2: Configure XSPIM port 2 for XSPI2 ---- */
  sXspiManagerCfg.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
  sXspiManagerCfg.IOPort      = HAL_XSPIM_IOPORT_2;
  sXspiManagerCfg.Req2AckTime = 1;
  if (HAL_XSPIM_Config(&hxspi2, &sXspiManagerCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 HAL_XSPIM_Config failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] XSPI2 HAL_XSPIM_Config OK\r\n");
  FSBL_BlinkLED(LED_GREEN, 2, 200);

  /* ---- Step 3: WriteEnable over 1-line SPI ---- */
  sCommand.OperationType     = HAL_XSPI_OPTYPE_COMMON_CFG;
  sCommand.IOSelect          = HAL_XSPI_SELECT_IO_7_0;
  sCommand.Instruction       = 0x06U;              /* WREN */
  sCommand.InstructionMode   = HAL_XSPI_INSTRUCTION_1_LINE;
  sCommand.InstructionWidth  = HAL_XSPI_INSTRUCTION_8_BITS;
  sCommand.InstructionDTRMode= HAL_XSPI_INSTRUCTION_DTR_DISABLE;
  sCommand.AddressMode       = HAL_XSPI_ADDRESS_NONE;
  sCommand.AlternateBytesMode= HAL_XSPI_ALT_BYTES_NONE;
  sCommand.DataMode          = HAL_XSPI_DATA_NONE;
  sCommand.DummyCycles       = 0;
  sCommand.DQSMode           = HAL_XSPI_DQS_DISABLE;
  if (HAL_XSPI_Command(&hxspi2, &sCommand, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 WriteEnable failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] WriteEnable OK\r\n");

  /* ---- Step 4: Write CR2 register addr=0, value=0x01 (SOPI) over 1-line SPI ---- */
  sCommand.Instruction  = 0x72U;              /* WRCR2 */
  sCommand.AddressMode  = HAL_XSPI_ADDRESS_1_LINE;
  sCommand.AddressWidth = HAL_XSPI_ADDRESS_32_BITS;
  sCommand.AddressDTRMode = HAL_XSPI_ADDRESS_DTR_DISABLE;
  sCommand.Address      = 0x00000000U;
  sCommand.DataMode     = HAL_XSPI_DATA_1_LINE;
  sCommand.DataDTRMode  = HAL_XSPI_DATA_DTR_DISABLE;
  sCommand.DataLength   = 1;
  if (HAL_XSPI_Command(&hxspi2, &sCommand, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 WriteCR2 cmd failed\r\n");
    Error_Handler();
  }
  if (HAL_XSPI_Transmit(&hxspi2, &reg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 WriteCR2 transmit failed\r\n");
    Error_Handler();
  }
  /* Small delay for the OPI mode switch to take effect */
  HAL_Delay(1);
  printf("[FSBL] SOPI enabled on MX25UM51245G\r\n");
  FSBL_BlinkLED(LED_GREEN, 3, 200);

  /* ---- Step 5+6: Set READ_CFG for memory-mapped mode (8-line STR OctoSPI) ----
   * Command: 0xEC13 (16-bit), 8 data lines, 4-byte address, 6 dummy cycles.
   * The BSP component uses DUMMY_CYCLES_READ_OCTAL = 6. */
  sCommand.OperationType     = HAL_XSPI_OPTYPE_READ_CFG;
  sCommand.Instruction       = 0xEC13U;           /* OCTA_READ_CMD */
  sCommand.InstructionMode   = HAL_XSPI_INSTRUCTION_8_LINES;
  sCommand.InstructionWidth  = HAL_XSPI_INSTRUCTION_16_BITS;
  sCommand.InstructionDTRMode= HAL_XSPI_INSTRUCTION_DTR_DISABLE;
  sCommand.AddressMode       = HAL_XSPI_ADDRESS_8_LINES;
  sCommand.AddressWidth      = HAL_XSPI_ADDRESS_32_BITS;
  sCommand.AddressDTRMode    = HAL_XSPI_ADDRESS_DTR_DISABLE;
  sCommand.Address           = 0;
  sCommand.AlternateBytesMode= HAL_XSPI_ALT_BYTES_NONE;
  sCommand.DataMode          = HAL_XSPI_DATA_8_LINES;
  sCommand.DataDTRMode       = HAL_XSPI_DATA_DTR_DISABLE;
  sCommand.DummyCycles       = 20U;               /* chip default (CR2 reg3 = 0x00 = 20 cycles) */
  sCommand.DQSMode           = HAL_XSPI_DQS_DISABLE;
  sCommand.DataLength        = 0;                 /* not used for config */
  if (HAL_XSPI_Command(&hxspi2, &sCommand, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 READ_CFG command failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] READ_CFG OK\r\n");

  /* ---- Step 7: WRITE_CFG (dummy — read-only boot) ---- */
  sCommand.OperationType = HAL_XSPI_OPTYPE_WRITE_CFG;
  sCommand.Instruction   = 0x12EDU;           /* OCTA_PAGE_PROG_CMD */
  sCommand.DummyCycles   = 0;
  if (HAL_XSPI_Command(&hxspi2, &sCommand, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    printf("[FSBL] XSPI2 WRITE_CFG command failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] WRITE_CFG OK\r\n");

  /* ---- Step 8: Enter memory-mapped mode ---- */
  sMemMappedCfg.TimeOutActivation = HAL_XSPI_TIMEOUT_COUNTER_DISABLE;
  if (HAL_XSPI_MemoryMapped(&hxspi2, &sMemMappedCfg) != HAL_OK)
  {
    printf("[FSBL] XSPI2 HAL_XSPI_MemoryMapped failed\r\n");
    Error_Handler();
  }
  printf("[FSBL] XSPI2 memory-mapped mode OK — flash visible at 0x70000000\r\n");
  FSBL_BlinkLED(LED_GREEN, 4, 200);
}

/* Route printf to LPUART1. */
int __io_putchar(int ch)
{
  HAL_UART_Transmit(&hlpuart1, (uint8_t *)&ch, 1, 10);
  return ch;
}

/* Blink a LED N times with on/off period ms, then leave it off. */
static void FSBL_BlinkLED(Led_TypeDef led, uint32_t n, uint32_t period_ms)
{
  BSP_LED_Init(led);
  for (uint32_t i = 0; i < n; i++)
  {
    BSP_LED_On(led);
    HAL_Delay(period_ms);
    BSP_LED_Off(led);
    HAL_Delay(period_ms);
  }
}

/* Print one short line so we can tell how far the FSBL got. */
static void FSBL_LogStage(const char *stage)
{
  printf("[FSBL] %s\r\n", stage);
}

/* Dump the app image header words and vector table from flash. */
static void FSBL_LogAppImageState(void)
{
  const uint32_t *const hdr  = (const uint32_t *)(FSBL_APP_FLASH_BASE - 0x400U); /* STM2 header */
  const uint32_t *const vecs = (const uint32_t *)FSBL_APP_FLASH_BASE;

  printf("[FSBL] STM2 header  @0x%08lX: %08lX %08lX %08lX %08lX\r\n",
         (unsigned long)(FSBL_APP_FLASH_BASE - 0x400U),
         (unsigned long)hdr[0], (unsigned long)hdr[1],
         (unsigned long)hdr[2], (unsigned long)hdr[3]);

  printf("[FSBL] App vectors  @0x%08lX: SP=%08lX  Reset=%08lX\r\n",
         (unsigned long)FSBL_APP_FLASH_BASE,
         (unsigned long)vecs[0], (unsigned long)vecs[1]);
}

/* Copy app from xSPI2 flash to AXISRAM1, then jump to it. */
static void FSBL_TryBootApplication(void)
{
  const uint32_t *const flash_vecs = (const uint32_t *)FSBL_APP_FLASH_BASE;
  const uint32_t app_sp    = flash_vecs[0];
  const uint32_t app_reset = flash_vecs[1];

  printf("[FSBL] app_sp=0x%08lX  app_reset=0x%08lX\r\n",
         (unsigned long)app_sp, (unsigned long)app_reset);

  /* Blink the top nibble of SP so we can tell what we read without UART.
   * 0x34xxxxxx -> 3 slow green blinks then 4 slow green blinks.
   * 0x00000000 -> 0 blinks (just a pause). 0xFFFFFFFF -> 15+15. */
  {
    uint8_t hi = (app_sp >> 28) & 0xF;
    uint8_t lo = (app_sp >> 24) & 0xF;
    FSBL_BlinkLED(LED_GREEN, hi ? hi : 1, 300);
    HAL_Delay(500);
    FSBL_BlinkLED(LED_GREEN, lo ? lo : 1, 300);
    HAL_Delay(500);
  }

  /* Sanity-check the vector table in flash */
  if ((app_sp == 0x00000000UL) || (app_sp == 0xFFFFFFFFUL))
  {
    printf("[FSBL] ERROR: SP invalid.\r\n");
    FSBL_BlinkLED(LED_RED, 10, 50);  /* fast red: bad SP */
    return;
  }
  if ((app_reset == 0x00000000UL) || (app_reset == 0xFFFFFFFFUL) ||
      ((app_reset & 0x1UL) == 0UL))
  {
    printf("[FSBL] ERROR: Reset vector invalid (not Thumb or erased).\r\n");
    FSBL_BlinkLED(LED_RED, 10, 50);  /* fast red: bad reset vector */
    return;
  }

  /* Vectors look sane — signal we are about to copy */
  FSBL_BlinkLED(LED_GREEN, 3, 150);  /* 3x green: starting LRUN copy */
  printf("[FSBL] Copying %lu bytes from flash 0x%08lX to RAM 0x%08lX...\r\n",
         (unsigned long)FSBL_APP_MAX_SIZE,
         (unsigned long)FSBL_APP_FLASH_BASE,
         (unsigned long)FSBL_APP_RAM_BASE);

  const uint32_t *src = (const uint32_t *)FSBL_APP_FLASH_BASE;
  uint32_t       *dst = (uint32_t *)FSBL_APP_RAM_BASE;
  for (uint32_t i = 0; i < FSBL_APP_MAX_SIZE / 4U; i++)
  {
    dst[i] = src[i];
  }

  printf("[FSBL] Copy done. Flushing caches.\r\n");
  SCB_CleanInvalidateDCache();
  SCB_InvalidateICache();

  /* Verify: first two words in RAM should match flash */
  const uint32_t *ram_vecs = (const uint32_t *)FSBL_APP_RAM_BASE;
  printf("[FSBL] RAM verify   @0x%08lX: SP=%08lX  Reset=%08lX\r\n",
         (unsigned long)FSBL_APP_RAM_BASE,
         (unsigned long)ram_vecs[0], (unsigned long)ram_vecs[1]);

  if (ram_vecs[0] != app_sp || ram_vecs[1] != app_reset)
  {
    printf("[FSBL] ERROR: RAM verify mismatch — copy failed!\r\n");
    FSBL_BlinkLED(LED_RED, 20, 50);  /* fast red: copy verify failed */
    return;
  }

  FSBL_BlinkLED(LED_GREEN, 1, 500);  /* 1x long green: jumping now */
  printf("[FSBL] Jumping to app: SP=0x%08lX  Reset=0x%08lX\r\n",
         (unsigned long)app_sp, (unsigned long)app_reset);

  /* Disable SysTick before handing off — the FSBL's SysTick keeps running
   * after the jump and can fire before the app's .data/.bss are initialised. */
  SysTick->CTRL = 0;

  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  SCB->VTOR = FSBL_APP_RAM_BASE;
  __set_MSP(app_sp);
  __set_MSPLIM(0);
  __set_PRIMASK(primask);   /* restore interrupt state for the app */

  ((void (*)(void))app_reset)();

  /* Should never reach here */
  printf("[FSBL] ERROR: App returned to FSBL!\r\n");
  FSBL_BlinkLED(LED_RED, 30, 100);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_LPUART1_UART_Init();

#if FSBL_LED_ONLY_SMOKE_TEST
  /* Minimal smoke test: prove flash boot chain executes at all. */
  BSP_LED_Init(LED_RED);
  FSBL_LogStage("LED smoke test — red blink 250ms");
  while (1)
  {
    BSP_LED_Toggle(LED_RED);
    HAL_Delay(250U);
  }
#else
  FSBL_LogStage("=== FSBL started ===");

  /* 2x blue blink = FSBL alive, GPIO + UART init OK */
  FSBL_BlinkLED(LED_BLUE, 2, 200);
  FSBL_LogStage("GPIO + UART init OK");

  /* Re-initialise XSPI2: SPI→OPI enable sequence + memory-mapped mode.
   * SystemInit resets XSPI2/XSPIM (as per ST reference template), so we must
   * send the OPI-enable sequence over 1-line SPI before entering 8-line mode. */
  MX_XSPI2_Init();
  FSBL_LogStage("xSPI2 OctoSPI STR memory-mapped mode OK");

  FSBL_LogAppImageState();
  FSBL_TryBootApplication();

  /* If we get here the handoff failed — fast red blink forever */
  FSBL_LogStage("ERROR: handoff failed, stuck in FSBL error loop");
  BSP_LED_Init(LED_RED);
  while (1)
  {
    BSP_LED_Toggle(LED_RED);
    HAL_Delay(100U);
  }
#endif
}
/* USER CODE BEGIN CLK 1 */
/* USER CODE END CLK 1 */

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the System Power Supply
  */
  if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /* Enable HSI */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes TIMPRE when TIM is used as Systick Clock Source
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_TIM;
  PeriphClkInitStruct.TIMPresSelection = RCC_TIMPRES_DIV1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Get current CPU/System buses clocks configuration and if necessary switch
 to intermediate HSI clock to ensure target clock can be set
  */
  HAL_RCC_GetClockConfig(&RCC_ClkInitStruct);
  if ((RCC_ClkInitStruct.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
     (RCC_ClkInitStruct.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11))
  {
    RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK);
    RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
    {
      /* Initialization Error */
      Error_Handler();
    }
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL1.PLLM = 4;
  RCC_OscInitStruct.PLL1.PLLN = 75;
  RCC_OscInitStruct.PLL1.PLLFractional = 0;
  RCC_OscInitStruct.PLL1.PLLP1 = 1;
  RCC_OscInitStruct.PLL1.PLLP2 = 1;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL4.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL4.PLLM = 1;
  RCC_OscInitStruct.PLL4.PLLN = 25;
  RCC_OscInitStruct.PLL4.PLLFractional = 0;
  RCC_OscInitStruct.PLL4.PLLP1 = 1;
  RCC_OscInitStruct.PLL4.PLLP2 = 1;

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_HCLK
                              |RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1
                              |RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_PCLK5
                              |RCC_CLOCKTYPE_PCLK4;
  RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;
  RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC1Selection.ClockDivider = 2;
  RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC2Selection.ClockDivider = 3;
  RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC6Selection.ClockDivider = 4;
  RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC11Selection.ClockDivider = 3;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.Timing = 0x009034B6;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c2, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c2, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{

  /* USER CODE BEGIN LPUART1_Init 0 */

  /* USER CODE END LPUART1_Init 0 */

  /* USER CODE BEGIN LPUART1_Init 1 */

  /* USER CODE END LPUART1_Init 1 */
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 115200;
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
  if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LPUART1_Init 2 */

  /* USER CODE END LPUART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPION_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(PWR_EN_GPIO_Port, PWR_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, UCPD1_ISENSE_Pin|UCPD_PWR_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB1_OCP_GPIO_Port, USB1_OCP_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : I2C1_SDA_Pin */
  GPIO_InitStruct.Pin = I2C1_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(I2C1_SDA_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PWR_EN_Pin */
  GPIO_InitStruct.Pin = PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(PWR_EN_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : UCPD1_INT_Pin */
  GPIO_InitStruct.Pin = UCPD1_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(UCPD1_INT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : I2CA_SCL_Pin */
  GPIO_InitStruct.Pin = I2CA_SCL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(I2CA_SCL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PA10 UCPD1_VSENSE_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_10|UCPD1_VSENSE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : UCPD1_ISENSE_Pin UCPD_PWR_EN_Pin */
  GPIO_InitStruct.Pin = UCPD1_ISENSE_Pin|UCPD_PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : USB1_OCP_Pin */
  GPIO_InitStruct.Pin = USB1_OCP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB1_OCP_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  printf("[FSBL] Error_Handler entered!\r\n");
  BSP_LED_Init(LED_RED);
  /* Alternate red on/off so it's visible even without UART */
  while (1)
  {
    BSP_LED_Toggle(LED_RED);
    /* Busy-wait ~200ms without HAL_Delay (SysTick may be broken) */
    for (volatile uint32_t i = 0; i < 2000000UL; i++) { __NOP(); }
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
