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
#include "app_threadx.h"
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "debug_console.h"
#include "debug_led.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

__IO uint32_t BspButtonState = BUTTON_RELEASED;

UART_HandleTypeDef hlpuart1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_LPUART1_UART_Init(void);
static void SystemIsolation_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_LPUART1_UART_Init();
	SystemIsolation_Config();
	/* USER CODE BEGIN 2 */
	// set up the debug console
	DebugConsole_Configuration_t debug_console_configuration = { 0 };
	debug_console_configuration.uart_handle_pointer = &hlpuart1;
	debug_console_configuration.uart_transmit_timeout_milliseconds = 100U;
	debug_console_configuration.lock_callback = NULL;
	debug_console_configuration.unlock_callback = NULL;

	(void) DebugConsole_Init(&debug_console_configuration);

	DebugConsole_Printf(
			"Welcome to STM32 world!\r\nApplication project is running...\r\n");
	// set up the debug leds
	DebugLed_Configuration_t debug_led_configuration = {
			.bsp_led_for_color_array = { LED_BLUE, LED_RED, LED_GREEN },
			.delay_milliseconds_callback = NULL };

	(void) DebugLed_Initialize(&debug_led_configuration);
	/* USER CODE END 2 */

	/* Initialize leds */
	BSP_LED_Init(LED_BLUE);
	BSP_LED_Init(LED_RED);
	BSP_LED_Init(LED_GREEN);

	/* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
	BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

	MX_ThreadX_Init();

	/* USER CODE BEGIN BSP */

	/* USER CODE END BSP */

	/* We should never get here as control is now taken by the scheduler */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {

		/* -- Sample board code for User push-button in interrupt mode ---- */
		if (BspButtonState == BUTTON_PRESSED) {
			/* Update button state */
			BspButtonState = BUTTON_RELEASED;
			/* -- Sample board code to toggle leds ---- */
			BSP_LED_Toggle(LED_BLUE);
			BSP_LED_Toggle(LED_RED);
			BSP_LED_Toggle(LED_GREEN);

			/* ..... Perform your action ..... */
		}
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief LPUART1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_LPUART1_UART_Init(void) {

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
	if (HAL_UART_Init(&hlpuart1) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8)
			!= HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8)
			!= HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN LPUART1_Init 2 */

	/* USER CODE END LPUART1_Init 2 */

}

/**
 * @brief RIF Initialization Function
 * @param None
 * @retval None
 */
static void SystemIsolation_Config(void) {

	/* USER CODE BEGIN RIF_Init 0 */

	/* USER CODE END RIF_Init 0 */

	/* set all required IPs as secure privileged */
	__HAL_RCC_RIFSC_CLK_ENABLE();

	/*RIMC configuration*/
	RIMC_MasterConfig_t RIMC_master = { 0 };
	RIMC_master.MasterCID = RIF_CID_1;
	RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV;
	HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_ETH1, &RIMC_master);

	/* RIF-Aware IPs Config */

	/* set up GPIO configuration */
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_5,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_7,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_10,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_11,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_0,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_3,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_6,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_7,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_10,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_11,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOC, GPIO_PIN_1,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOD, GPIO_PIN_2,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOD, GPIO_PIN_10,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_3,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_5,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_6,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOH, GPIO_PIN_9,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPION, GPIO_PIN_7,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOO, GPIO_PIN_5,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);

	/* USER CODE BEGIN RIF_Init 1 */

	/* USER CODE END RIF_Init 1 */
	/* USER CODE BEGIN RIF_Init 2 */

	/* USER CODE END RIF_Init 2 */

}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
	/* USER CODE BEGIN MX_GPIO_Init_1 */

	/* USER CODE END MX_GPIO_Init_1 */

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOE_CLK_ENABLE();

	/* USER CODE BEGIN MX_GPIO_Init_2 */

	/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
 * @brief  Period elapsed callback in non blocking mode
 * @note   This function is called  when TIM5 interrupt took place, inside
 * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
 * a global variable "uwTick" used as application time base.
 * @param  htim : TIM handle
 * @retval None
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
	/* USER CODE BEGIN Callback 0 */

	/* USER CODE END Callback 0 */
	if (htim->Instance == TIM5) {
		HAL_IncTick();
	}
	/* USER CODE BEGIN Callback 1 */

	/* USER CODE END Callback 1 */
}

/**
 * @brief BSP Push Button callback
 * @param Button Specifies the pressed button
 * @retval None
 */
void BSP_PB_Callback(Button_TypeDef Button) {
	if (Button == BUTTON_USER) {
		BspButtonState = BUTTON_PRESSED;
	}
}

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
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
