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
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifndef APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
/* Prod v0.8 uses the external scalar calibration / postprocess path to keep
 * the live board aligned with the offline rectified-scalar recipe. */
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 1
#endif

#define APP_INFERENCE_CALIBRATION_PIECEWISE_KNOT_COUNT 6U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
/* Piecewise hinge-basis calibration saved from prodv0_3_obb_scalar_calibration.json.
 * Keep these coefficients in sync with the offline replay helper so the board
 * and Python validation paths apply the same postprocess. */
static const float kCalibrationPiecewiseBias = 11.042082786560059f;
static const float kCalibrationPiecewiseWeights[APP_INFERENCE_CALIBRATION_PIECEWISE_KNOT_COUNT + 1U] = {
	2.030168294906616f,
	-8.992179870605469f,
	4.932034492492676f,
	4.263505458831787f,
	-4.158827304840088f,
	6.242796421051025f,
	-3.0523934364318848f,
};
static const float kCalibrationPiecewiseKnots[APP_INFERENCE_CALIBRATION_PIECEWISE_KNOT_COUNT] = {
	5.026547431945801f,
	6.800623416900635f,
	8.279019355773926f,
	10.348774909973145f,
	14.48828411102295f,
	18.332115173339844f,
};
#endif
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value);
#endif
/* USER CODE END PFP */

/* Private function definitions ---------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
 * @brief Apply the scalar correction fit to a raw inference value.
 */
float AppInferenceCalibration_Apply(float raw_value)
{
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
	return AppInferenceCalibration_EvaluatePiecewise(raw_value);
#else
	return raw_value;
#endif
}

/**
 * @brief Evaluate the saved piecewise hinge calibration.
 *
 * The offline replay path uses the same basis:
 *   bias + weights[0] * x + sum_i weights[i + 1] * max(x - knot_i, 0)
 *
 * Keep this in lockstep with the Python board replay helper so the firmware
 * and the offline graph stay numerically aligned.
 */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value)
{
	float calibrated_value =
		kCalibrationPiecewiseBias +
		(kCalibrationPiecewiseWeights[0] * raw_value);

	for (unsigned int knot_index = 0U;
		 knot_index < APP_INFERENCE_CALIBRATION_PIECEWISE_KNOT_COUNT;
		 ++knot_index)
	{
		const float hinge = raw_value - kCalibrationPiecewiseKnots[knot_index];
		if (hinge > 0.0f)
		{
			calibrated_value +=
				kCalibrationPiecewiseWeights[knot_index + 1U] * hinge;
		}
	}

	return calibrated_value;
}
#endif

/* USER CODE END 0 */
