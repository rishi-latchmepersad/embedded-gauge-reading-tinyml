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
#include "threadx_utils.h"
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

SPI_HandleTypeDef hspi5;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_SPI5_Init(void);
static void MX_USART1_UART_Init(void);
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
	MX_SPI5_Init();
	MX_USART1_UART_Init();
	SystemIsolation_Config();
	/* USER CODE BEGIN 2 */
	DebugConsole_Configuration_t debug_console_configuration = { 0 };
	debug_console_configuration.uart_handle_pointer = &huart1;
	debug_console_configuration.uart_transmit_timeout_milliseconds = 100U;
	debug_console_configuration.lock_callback = NULL;
	debug_console_configuration.unlock_callback = NULL;

	(void) DebugConsole_Init(&debug_console_configuration);

	DebugConsole_Printf(
			"Welcome to STM32 world!\r\nApplication project is running...\r\n");
	/* USER CODE END 2 */

	/* Initialize leds */
	BSP_LED_Init(LED_BLUE);
	BSP_LED_Init(LED_RED);
	BSP_LED_Init(LED_GREEN);
	DebugLed_Configuration_t debug_led_configuration = {
			.bsp_led_for_color_array = { LED_BLUE, LED_RED, LED_GREEN },
			.delay_milliseconds_callback = DelayMilliseconds_ThreadX };

	(void) DebugLed_Initialize(&debug_led_configuration);
	MX_ThreadX_Init();

	/* USER CODE BEGIN BSP */

	/* USER CODE END BSP */

	/* We should never get here as control is now taken by the scheduler */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {

		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
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

	/* RIF-Aware IPs Config */

	/* set up GPIO configuration */
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_3,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_5,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOA, GPIO_PIN_7,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_0,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_3,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_6,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOB, GPIO_PIN_7,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOC, GPIO_PIN_0,
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
	HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_15,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOG, GPIO_PIN_1,
	GPIO_PIN_SEC | GPIO_PIN_NPRIV);
	HAL_GPIO_ConfigPinAttributes(GPIOG, GPIO_PIN_2,
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
 * @brief SPI5 Initialization Function
 * @param None
 * @retval None
 */
static void MX_SPI5_Init(void) {

	/* USER CODE BEGIN SPI5_Init 0 */

	/* USER CODE END SPI5_Init 0 */

	/* USER CODE BEGIN SPI5_Init 1 */

	/* USER CODE END SPI5_Init 1 */
	/* SPI5 parameter configuration*/
	hspi5.Instance = SPI5;
	hspi5.Init.Mode = SPI_MODE_MASTER;
	hspi5.Init.Direction = SPI_DIRECTION_2LINES;
	hspi5.Init.DataSize = SPI_DATASIZE_8BIT;
	hspi5.Init.CLKPolarity = SPI_POLARITY_LOW;
	hspi5.Init.CLKPhase = SPI_PHASE_1EDGE;
	hspi5.Init.NSS = SPI_NSS_SOFT;
	hspi5.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_256;
	hspi5.Init.FirstBit = SPI_FIRSTBIT_MSB;
	hspi5.Init.TIMode = SPI_TIMODE_DISABLE;
	hspi5.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	hspi5.Init.CRCPolynomial = 0x7;
	hspi5.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
	hspi5.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
	hspi5.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
	hspi5.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
	hspi5.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
	hspi5.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
	hspi5.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
	hspi5.Init.IOSwap = SPI_IO_SWAP_DISABLE;
	hspi5.Init.ReadyMasterManagement = SPI_RDY_MASTER_MANAGEMENT_INTERNALLY;
	hspi5.Init.ReadyPolarity = SPI_RDY_POLARITY_HIGH;
	if (HAL_SPI_Init(&hspi5) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN SPI5_Init 2 */

	/* USER CODE END SPI5_Init 2 */

}

/**
 * @brief USART1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART1_UART_Init(void) {

	/* USER CODE BEGIN USART1_Init 0 */

	/* USER CODE END USART1_Init 0 */

	/* USER CODE BEGIN USART1_Init 1 */

	/* USER CODE END USART1_Init 1 */
	huart1.Instance = USART1;
	huart1.Init.BaudRate = 115200;
	huart1.Init.WordLength = UART_WORDLENGTH_8B;
	huart1.Init.StopBits = UART_STOPBITS_1;
	huart1.Init.Parity = UART_PARITY_NONE;
	huart1.Init.Mode = UART_MODE_TX_RX;
	huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart1.Init.OverSampling = UART_OVERSAMPLING_16;
	huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
	huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
	huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
	if (HAL_UART_Init(&huart1) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8)
			!= HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8)
			!= HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN USART1_Init 2 */

	/* USER CODE END USART1_Init 2 */

}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
	GPIO_InitTypeDef GPIO_InitStruct = { 0 };
	/* USER CODE BEGIN MX_GPIO_Init_1 */

	/* USER CODE END MX_GPIO_Init_1 */

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOE_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOO_CLK_ENABLE();
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOG_CLK_ENABLE();
	__HAL_RCC_GPION_CLK_ENABLE();

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(PWR_EN_GPIO_Port, PWR_EN_Pin, GPIO_PIN_RESET);

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(CAM_NRST_GPIO_Port, CAM_NRST_Pin, GPIO_PIN_RESET);

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(GPIOA, UCPD1_ISENSE_Pin | UCPD_PWR_EN_Pin,
			GPIO_PIN_RESET);

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(SPI5_CS_GPIO_Port, SPI5_CS_Pin, GPIO_PIN_SET);

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(USB1_OCP_GPIO_Port, USB1_OCP_Pin, GPIO_PIN_RESET);

	/*Configure GPIO pin : PWR_EN_Pin */
	GPIO_InitStruct.Pin = PWR_EN_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(PWR_EN_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pin : PC0 */
	GPIO_InitStruct.Pin = GPIO_PIN_0;
	GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

	/*Configure GPIO pin : UCPD1_INT_Pin */
	GPIO_InitStruct.Pin = UCPD1_INT_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	HAL_GPIO_Init(UCPD1_INT_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pin : CAM_NRST_Pin */
	GPIO_InitStruct.Pin = CAM_NRST_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(CAM_NRST_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pins : UCPD1_ISENSE_Pin UCPD_PWR_EN_Pin SPI5_CS_Pin */
	GPIO_InitStruct.Pin = UCPD1_ISENSE_Pin | UCPD_PWR_EN_Pin | SPI5_CS_Pin;
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
