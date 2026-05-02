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
/* Re-enabled 2026-05-02 as an incremental board-side correction while we
 * continue long-term model retraining for full-range performance. */
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 1
#endif

/* Temperature range for calibration. The affine fit was trained on warmer
 * data and makes cold readings worse (raw ~-20 at true -30C, affine pushes
 * to -22 to -24). Below -10C raw output, skip the affine correction entirely
 * since the model is already under-reading and the affine makes it worse. */
#ifndef APP_INFERENCE_CALIBRATION_COLD_THRESHOLD
#define APP_INFERENCE_CALIBRATION_COLD_THRESHOLD -10.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_HOT_THRESHOLD
#define APP_INFERENCE_CALIBRATION_HOT_THRESHOLD 43.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_COLD_BLEND
#define APP_INFERENCE_CALIBRATION_COLD_BLEND 0.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_COLD_TAIL_START
/* Start adding extra cold correction once raw predictions go below this
 * temperature. */
#define APP_INFERENCE_CALIBRATION_COLD_TAIL_START -12.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_COLD_TAIL_GAIN
/* Convert low-end "too warm" bias into a stronger negative correction.
 * Example: raw=-17.3, depth=5.3 -> about -5.6C extra correction. */
#define APP_INFERENCE_CALIBRATION_COLD_TAIL_GAIN 1.05f
#endif
#ifndef APP_INFERENCE_CALIBRATION_COLD_TAIL_MAX_DELTA
/* Cap the extra cold correction so extreme tails are bounded. */
#define APP_INFERENCE_CALIBRATION_COLD_TAIL_MAX_DELTA 8.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_LOW_BAND_MAX
/* In the low band, the scalar model already tends to over-read on board.
 * Keep calibration neutral here so we do not push 10C-class readings higher. */
#define APP_INFERENCE_CALIBRATION_LOW_BAND_MAX 20.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_LOW_BAND_BLEND
#define APP_INFERENCE_CALIBRATION_LOW_BAND_BLEND 0.0f
#endif
#ifndef APP_INFERENCE_CALIBRATION_HOT_BLEND
/* Above the hot threshold, apply only a partial correction so bright
 * close-up hard cases do not overshoot into the 50C range. */
#define APP_INFERENCE_CALIBRATION_HOT_BLEND 0.35f
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
/* Milder hard-case affine calibration from scalar_hardcase_boost_v8.
 * Compared to the older p5 fit, this slope is less aggressive and tracks
 * current board hard cases better after re-enabling affine crop fill. */
static const float kCalibrationAffineScale = 1.0502802133560180f;
static const float kCalibrationAffineBias = 0.6553916335105896f;
#endif
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
static float AppInferenceCalibration_EvaluateAffine(float raw_value);
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
	return AppInferenceCalibration_EvaluateAffine(raw_value);
#else
	return raw_value;
#endif
}

/**
 * @brief Evaluate the saved affine scalar calibration.
 */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
static float AppInferenceCalibration_EvaluateAffine(float raw_value)
{
	const float full_calibrated =
		kCalibrationAffineBias + (kCalibrationAffineScale * raw_value);

	/* Apply the full affine correction for the core hard-case range. */
	if ((raw_value >= APP_INFERENCE_CALIBRATION_LOW_BAND_MAX) &&
		(raw_value <= APP_INFERENCE_CALIBRATION_HOT_THRESHOLD))
	{
		return full_calibrated;
	}

	/* Outside the core range, blend toward identity:
	 * - cold side: no extra correction
	 * - low band: no extra correction
	 * - hot side: partial correction only (to avoid overshoot)
	 */
	{
		const float blend =
			(raw_value < APP_INFERENCE_CALIBRATION_COLD_THRESHOLD)
				? APP_INFERENCE_CALIBRATION_COLD_BLEND
				: ((raw_value < APP_INFERENCE_CALIBRATION_LOW_BAND_MAX)
					   ? APP_INFERENCE_CALIBRATION_LOW_BAND_BLEND
					   : APP_INFERENCE_CALIBRATION_HOT_BLEND);
		float corrected = raw_value + (blend * (full_calibrated - raw_value));

		/* Add a separate cold-tail correction for deep cold readings where the
		 * model remains conservative (too warm / not negative enough). */
		if (raw_value < APP_INFERENCE_CALIBRATION_COLD_TAIL_START)
		{
			float cold_depth =
				APP_INFERENCE_CALIBRATION_COLD_TAIL_START - raw_value;
			float extra_cold_delta =
				APP_INFERENCE_CALIBRATION_COLD_TAIL_GAIN * cold_depth;
			if (extra_cold_delta > APP_INFERENCE_CALIBRATION_COLD_TAIL_MAX_DELTA)
			{
				extra_cold_delta =
					APP_INFERENCE_CALIBRATION_COLD_TAIL_MAX_DELTA;
			}
			corrected -= extra_cold_delta;
		}

		return corrected;
	}
}
#endif

/* USER CODE END 0 */
