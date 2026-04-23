/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_inference_calibration.c
 * @brief   Piecewise scalar correction for gauge inference outputs.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_inference_calibration.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stddef.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifndef APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 1
#endif
#ifndef APP_INFERENCE_USE_PIECEWISE_CALIBRATION
#define APP_INFERENCE_USE_PIECEWISE_CALIBRATION 1
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
/* Piecewise calibration fit against the mid-band board mix, then stress-tested
 * on hard-case manifests. The learned OBB crop is already doing the geometry
 * work; this tail just nudges the scalar prediction back onto the board scale.
 */
static const float kCalibrationPiecewiseBias = 38.98446273803711f;
static const float kCalibrationPiecewiseWeights[] = {
	1.5801066160202026f,
	8.961633682250977f,
	-11.134612083435059f,
	1.1817320585250854f,
	-1.3768948316574097f,
	0.6362906694412231f,
	-0.11085506528615952f,
};
static const float kCalibrationPiecewiseKnots[] = {
	-6.861651420593262f,
	-6.208160877227783f,
	-4.901179313659668f,
	-2.7539942264556885f,
	-0.32674530148506165f,
	15.35702896118164f,
};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
#if !APP_INFERENCE_USE_PIECEWISE_CALIBRATION
static float AppInferenceCalibration_EvaluateAffine(float raw_value);
#endif
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value);
/* USER CODE END PFP */

/* Private function definitions ---------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
 * @brief Apply the scalar correction fit to a raw inference value.
 */
float AppInferenceCalibration_Apply(float raw_value) {
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
#if APP_INFERENCE_USE_PIECEWISE_CALIBRATION
	return AppInferenceCalibration_EvaluatePiecewise(raw_value);
#else
	return AppInferenceCalibration_EvaluateAffine(raw_value);
#endif
#else
	return raw_value;
#endif
}

/**
 * @brief Evaluate the saved affine scalar calibration.
 */
#if !APP_INFERENCE_USE_PIECEWISE_CALIBRATION
static float AppInferenceCalibration_EvaluateAffine(float raw_value) {
	/* Keep the affine form available for quick fallback comparisons. */
	return 32.26465606689453f + (-0.11213056743144989f * raw_value);
}
#endif

/**
 * @brief Evaluate the fitted piecewise scalar calibration.
 */
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value) {
	float output = kCalibrationPiecewiseBias;

	/* The first weight multiplies the raw prediction directly. */
	output += kCalibrationPiecewiseWeights[0] * raw_value;

	/* Add each hinge term so the tail can bend around the hard cases. */
	for (size_t i = 0U; i < (sizeof(kCalibrationPiecewiseKnots) / sizeof(kCalibrationPiecewiseKnots[0])); ++i) {
		float hinge = raw_value - kCalibrationPiecewiseKnots[i];
		if (hinge > 0.0f) {
			output += kCalibrationPiecewiseWeights[i + 1U] * hinge;
		}
	}

	return output;
}

/* USER CODE END 0 */
