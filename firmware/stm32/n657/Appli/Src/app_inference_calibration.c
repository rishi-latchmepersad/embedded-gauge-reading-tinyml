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
/* Hard-case affine calibration from scalar_full_finetune_from_best_affine_calibrated_p5.
 * This is the lighter postprocess that keeps the original hard-case manifest
 * much closer than the older board30 piecewise tail.
 */
static const float kCalibrationAffineScale = 1.1630995273590088f;
static const float kCalibrationAffineBias = 0.7423046231269836f;
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
	/* Keep the board correction mild so the hard-case manifest does not over-shoot. */
	return kCalibrationAffineBias + (kCalibrationAffineScale * raw_value);
}

/* USER CODE END 0 */
