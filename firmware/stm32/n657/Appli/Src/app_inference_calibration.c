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
/* Keep the deployed polar-vote path aligned with the training script: the
 * board should report the raw model decode, not a post-hoc affine fit. */
#define APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION 0
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
#ifndef APP_INFERENCE_CALIBRATION_HOT_BOOST_GAIN
/* Linear boost per degree above LOW_BAND_MAX to compensate for the model's
 * progressive under-reading at high temperatures. At raw=34 this adds
 * 0.25*14 = 3.5C, bringing the worst-case error from -8.6C to -5.1C. */
#define APP_INFERENCE_CALIBRATION_HOT_BOOST_GAIN 0.25f
#endif
#ifndef APP_INFERENCE_CALIBRATION_HOT_BOOST_MAX
/* Cap the total hot boost so extreme raw values do not overshoot. */
#define APP_INFERENCE_CALIBRATION_HOT_BOOST_MAX 12.0f
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
 * @brief Evaluate the saved affine scalar calibration with hot-temperature
 *        boost.
 *
 * The model progressively under-reads above ~20C raw. A linear boost of
 * 0.25C per degree above 20C (capped at 12C) is added to the affine
 * correction in the core band. Outside the core band, the boost is blended
 * proportionally with the affine blend to avoid overshoot at the edges.
 *
 * Expected improvement at true 45C (raw ~34-40):
 *   raw=34: 36.4 -> 39.9 (error -5.1C, was -8.6C)
 *   raw=40: 42.7 -> 47.7 (error +2.7C, was -2.3C)
 */
#if APP_INFERENCE_ENABLE_OUTPUT_CALIBRATION
static float AppInferenceCalibration_EvaluateAffine(float raw_value)
{
	const float full_calibrated =
		kCalibrationAffineBias + (kCalibrationAffineScale * raw_value);

	/* Compute the hot-temperature boost for progressive under-reading.
	 * The model is accurate at ~12C (raw=12.8) but under-reads by 5-11C at
	 * true 45C. The boost grows linearly above LOW_BAND_MAX and is capped. */
	float hot_boost = 0.0f;
	if (raw_value > APP_INFERENCE_CALIBRATION_LOW_BAND_MAX)
	{
		hot_boost = APP_INFERENCE_CALIBRATION_HOT_BOOST_GAIN *
			(raw_value - APP_INFERENCE_CALIBRATION_LOW_BAND_MAX);
		if (hot_boost > APP_INFERENCE_CALIBRATION_HOT_BOOST_MAX)
		{
			hot_boost = APP_INFERENCE_CALIBRATION_HOT_BOOST_MAX;
		}
	}

	/* Apply the full affine correction plus hot boost in the core range. */
	if ((raw_value >= APP_INFERENCE_CALIBRATION_LOW_BAND_MAX) &&
		(raw_value <= APP_INFERENCE_CALIBRATION_HOT_THRESHOLD))
	{
		return full_calibrated + hot_boost;
	}

	/* Outside the core range, blend toward identity:
	 * - cold side: no affine correction, no boost
	 * - low band: no affine correction, no boost
	 * - hot side: partial affine + partial boost (to avoid overshoot)
	 */
	{
		const float blend =
			(raw_value < APP_INFERENCE_CALIBRATION_COLD_THRESHOLD)
				? APP_INFERENCE_CALIBRATION_COLD_BLEND
				: ((raw_value < APP_INFERENCE_CALIBRATION_LOW_BAND_MAX)
					   ? APP_INFERENCE_CALIBRATION_LOW_BAND_BLEND
					   : APP_INFERENCE_CALIBRATION_HOT_BLEND);
		float corrected = raw_value + (blend * (full_calibrated - raw_value));

		/* Blend the hot boost proportionally with the affine blend so the
		 * boost tapers off at the band edges rather than stepping. */
		corrected += blend * hot_boost;

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
