/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_inference_calibration.c
 * @brief   Affine scalar correction for gauge inference outputs.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_inference_calibration.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifndef APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 1
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
/* Crop-domain affine calibration derived from the board-crop probe set
 * captured in `captured_images/_live_rectified_probe`. This is the simplest
 * correction that stayed stable on the close-up board domain and it avoids the
 * overfit piecewise curve that was amplifying the live board error. */
static const float kCalibrationScale = 1.1953182220458984f;
static const float kCalibrationBias = -1.0408254861831665f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static float AppInferenceCalibration_EvaluateAffine(float raw_value);
/* USER CODE END PFP */

/* Private function definitions ---------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
 * @brief Apply the scalar correction fit to a raw inference value.
 */
float AppInferenceCalibration_Apply(float raw_value) {
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
	return AppInferenceCalibration_EvaluateAffine(raw_value);
#else
	return raw_value;
#endif
}

/**
 * @brief Evaluate the saved affine scalar calibration.
 */
static float AppInferenceCalibration_EvaluateAffine(float raw_value) {
	return kCalibrationBias + (kCalibrationScale * raw_value);
}

/* USER CODE END 0 */
