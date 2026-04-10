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
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static const float kCalibrationBias = -0.033284228295087814f;
static const float kCalibrationWeights[26] = {
	1.0021905899047852f,
	0.006945972330868244f,
	-0.017404647544026375f,
	-0.03176581859588623f,
	0.06929904222488403f,
	-0.19600756466388702f,
	0.42863717675209045f,
	-0.24360869824886322f,
	0.05921878293156624f,
	-1.0775048732757568f,
	0.9772740602493286f,
	0.025004656985402107f,
	25.137296676635742f,
	-26.139575958251953f,
	-2.1726652903453214e-06f,
	1.9660687939904165e-06f,
	5.6629475278990524e-14f,
	-6.714916708609742e-14f,
	-3.3866941369364856e-13f,
	1.3028973626205698e-05f,
	1.114143967628479f,
	-0.1483132541179657f,
	-0.9658437967300415f,
	1.1698728799819946f,
	-0.18366287648677826f,
	0.0f,
};
static const float kCalibrationKnots[25] = {
	-29.901214599609375f,
	-19.00080680847168f,
	-17.992469787597656f,
	-9.658842086791992f,
	0.056827545166015625f,
	6.057373046875f,
	10.019668579101562f,
	17.876007080078125f,
	18.8040771484375f,
	18.816268920898438f,
	19.839523315429688f,
	21.834976196289062f,
	22.14102554321289f,
	22.780811309814453f,
	22.881500244140625f,
	29.78125f,
	29.79193115234375f,
	29.809921264648438f,
	29.837753295898438f,
	29.852947235107422f,
	34.34064483642578f,
	44.69428634643555f,
	44.76510238647461f,
	45.619895935058594f,
	49.67582702636719f,
};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value);
/* USER CODE END PFP */

/* Private function definitions ---------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
 * @brief Apply the scalar correction fit to a raw inference value.
 */
float AppInferenceCalibration_Apply(float raw_value) {
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
	return AppInferenceCalibration_EvaluatePiecewise(raw_value);
#else
	return raw_value;
#endif
}

/**
 * @brief Evaluate the saved piecewise-linear scalar calibration.
 */
static float AppInferenceCalibration_EvaluatePiecewise(float raw_value) {
	float calibrated = kCalibrationBias + (kCalibrationWeights[0] * raw_value);

	for (size_t knot_index = 0U; knot_index < (sizeof(kCalibrationKnots) /
			sizeof(kCalibrationKnots[0])); ++knot_index) {
		const float knot = kCalibrationKnots[knot_index];
		const float relu = (raw_value > knot) ? (raw_value - knot) : 0.0f;
		calibrated += kCalibrationWeights[knot_index + 1U] * relu;
	}

	return calibrated;
}

/* USER CODE END 0 */
