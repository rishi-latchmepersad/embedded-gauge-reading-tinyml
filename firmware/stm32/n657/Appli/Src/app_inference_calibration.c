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
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
static const float kCalibrationBias = -0.024120653048157692f;
static const float kCalibrationWeights[30] = {
	1.2266254425048828f,
	0.6375268697738647f,
	-0.5811766982078552f,
	0.7542758584022522f,
	-2.0372517108917236f,
	4.793534278869629f,
	2.6630752086639404f,
	-42.53657150268555f,
	49.477996826171875f,
	-25.13555145263672f,
	11.856005668640137f,
	-1.9573670625686646f,
	3.0759220123291016f,
	-2.382946729660034f,
	5.418795585632324f,
	-2.876126527786255f,
	-24.20728302001953f,
	23.823801040649414f,
	-0.14913296699523926f,
	-1.8641489744186401f,
	-3.065540340685402e-06f,
	-2.054593323919107e-06f,
	2.0545930965454318e-06f,
	3.355468988418579f,
	-0.7456538081169128f,
	-1.9386872053146362f,
	-1.3422569036483765f,
	0.6711322665214539f,
	5.59247350692749f,
	0.0f,
};
static const float kCalibrationKnots[29] = {
	-24.437679290771484f,
	-21.755495071411133f,
	-16.68914794921875f,
	-12.516860008239746f,
	-1.1920819282531738f,
	0.8940614461898804f,
	3.5762457847595215f,
	3.8742661476135254f,
	5.364368438720703f,
	6.854471206665039f,
	10.430716514587402f,
	11.622798919677734f,
	12.516860008239746f,
	19.3713321685791f,
	21.457473754882812f,
	23.543617248535156f,
	24.139659881591797f,
	25.6297607421875f,
	28.31194496154785f,
	28.907987594604492f,
	29.20600700378418f,
	30.696109771728516f,
	30.994129180908203f,
	32.48423385620117f,
	35.16641616821289f,
	41.126827239990234f,
	42.61692810058594f,
	44.107032775878906f,
	45.00109100341797f,
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
