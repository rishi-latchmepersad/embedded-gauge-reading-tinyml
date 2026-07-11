/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_baseline_hough.c
 * @brief   Simple calibrated-center radial Hough baseline for the board.
 ******************************************************************************
 */
/* USER CODE END Header */

#include "app_baseline_hough.h"

#include <math.h>
#include <string.h>

#include "app_gauge_geometry.h"

/* Keep the implementation intentionally small and board-readable. */
#define APP_BASELINE_HOUGH_PI 3.14159265358979323846f
#define APP_BASELINE_HOUGH_MIN_ANGLE_DEG 135.0f
#define APP_BASELINE_HOUGH_SWEEP_DEG 270.0f
#define APP_BASELINE_HOUGH_ANGLE_BINS 360U
#define APP_BASELINE_HOUGH_RAY_SAMPLES 24U
#define APP_BASELINE_HOUGH_RAY_START_FRACTION 0.15f
#define APP_BASELINE_HOUGH_RAY_END_FRACTION 0.83f
#define APP_BASELINE_HOUGH_BACKGROUND_OFFSETS 2U
#define APP_BASELINE_HOUGH_MIN_CONTRAST 6.0f
#define APP_BASELINE_HOUGH_MIN_SCORE 8.0f
/* The saved board examples have valid dynamic-Hough ratios down to about
 * 1.19. Keep this detector-specific gate modestly permissive; the runtime
 * history gate still rejects unstable frame-to-frame jumps. */
#define APP_BASELINE_HOUGH_MIN_PEAK_RATIO 1.15f
#define APP_BASELINE_HOUGH_PEAK_SUPPRESSION_BINS 12U
#define APP_BASELINE_HOUGH_SOURCE_LABEL "simple-hough-polar"
#define APP_BASELINE_HOUGH_CENTER_ANGLE_BINS 24U
#define APP_BASELINE_HOUGH_CENTER_X_MARGIN 64U
#define APP_BASELINE_HOUGH_CENTER_Y_MARGIN 64U
#define APP_BASELINE_HOUGH_RADIUS_MIN_PIXELS 44U
#define APP_BASELINE_HOUGH_RADIUS_MAX_PIXELS 84U
#define APP_BASELINE_HOUGH_CENTER_COARSE_STEP 8U
#define APP_BASELINE_HOUGH_CENTER_FINE_STEP 2U
#define APP_BASELINE_HOUGH_RADIUS_COARSE_STEP 4U
#define APP_BASELINE_HOUGH_RADIUS_FINE_STEP 2U

/**
 * @brief Read the luma byte from one packed YUV422 pixel.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_width_pixels Frame width in pixels.
 * @param x Pixel x coordinate.
 * @param y Pixel y coordinate.
 * @return Eight-bit luma value.
 */
static float AppBaselineHough_ReadLuma(
	const uint8_t *frame_bytes, size_t frame_width_pixels, size_t x, size_t y)
{
	const size_t row_stride_bytes = frame_width_pixels * 2U;
	const size_t pair_offset =
		(y * row_stride_bytes) + ((x & ~1U) * 2U);
	const size_t y_offset = pair_offset + (((x & 1U) != 0U) ? 2U : 0U);

	return (float)frame_bytes[y_offset];
}

/**
 * @brief Score a candidate circular dial face.
 *
 * The white dial face should be brighter just inside its boundary than the
 * desk/background just outside it. This deliberately small radial score gives
 * the Hough stage a frame-local center and radius without a learned localizer.
 */
static float AppBaselineHough_ScoreFace(
	const uint8_t *frame_bytes, size_t frame_width_pixels,
	size_t frame_height_pixels, size_t center_x, size_t center_y,
	float radius, size_t expected_center_x, size_t expected_center_y)
{
	float support = 0.0f;
	const float two_pi = 2.0f * APP_BASELINE_HOUGH_PI;

	for (size_t angle_index = 0U;
		 angle_index < APP_BASELINE_HOUGH_CENTER_ANGLE_BINS;
		 ++angle_index)
	{
		const float angle = two_pi *
			((float)angle_index /
			 (float)APP_BASELINE_HOUGH_CENTER_ANGLE_BINS);
		const float cosine = cosf(angle);
		const float sine = sinf(angle);
		const long inside_x = lroundf(
			(float)center_x + (radius * 0.82f * cosine));
		const long inside_y = lroundf(
			(float)center_y + (radius * 0.82f * sine));
		const long edge_x = lroundf(
			(float)center_x + (radius * cosine));
		const long edge_y = lroundf(
			(float)center_y + (radius * sine));
		const long outside_x = lroundf(
			(float)center_x + (radius * 1.18f * cosine));
		const long outside_y = lroundf(
			(float)center_y + (radius * 1.18f * sine));

		if ((inside_x < 0L) || (inside_y < 0L) ||
			(edge_x < 0L) || (edge_y < 0L) ||
			(outside_x < 0L) || (outside_y < 0L) ||
			((size_t)inside_x >= frame_width_pixels) ||
			((size_t)inside_y >= frame_height_pixels) ||
			((size_t)edge_x >= frame_width_pixels) ||
			((size_t)edge_y >= frame_height_pixels) ||
			((size_t)outside_x >= frame_width_pixels) ||
			((size_t)outside_y >= frame_height_pixels))
		{
			continue;
		}

		{
			const float inside = AppBaselineHough_ReadLuma(
				frame_bytes, frame_width_pixels, (size_t)inside_x,
				(size_t)inside_y);
			const float edge = AppBaselineHough_ReadLuma(
				frame_bytes, frame_width_pixels, (size_t)edge_x,
				(size_t)edge_y);
			const float outside = AppBaselineHough_ReadLuma(
				frame_bytes, frame_width_pixels, (size_t)outside_x,
				(size_t)outside_y);

			/* The edge term keeps a flat bright background from winning just
			 * because it has a large bright interior. */
			support += fmaxf(0.0f, inside - outside) +
				(0.35f * fabsf(edge - outside));
		}
	}

	{
		const float dx = (float)center_x - (float)expected_center_x;
		const float dy = (float)center_y - (float)expected_center_y;
		/* Why: the framing prior breaks ties against laptop-screen and desk
		 * circles without forcing the old fixed center. */
		return support - (0.15f * sqrtf((dx * dx) + (dy * dy)));
	}
}

/**
 * @brief Find a frame-local dial center and radius.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_width_pixels Frame width.
 * @param frame_height_pixels Frame height.
 * @param center_x_out Detected center x destination.
 * @param center_y_out Detected center y destination.
 * @param radius_out Detected dial radius destination.
 */
static void AppBaselineHough_FindFaceGeometry(
	const uint8_t *frame_bytes, size_t frame_width_pixels,
	size_t frame_height_pixels, size_t *center_x_out, size_t *center_y_out,
	float *radius_out)
{
	size_t expected_center_x = frame_width_pixels / 2U;
	size_t expected_center_y = frame_height_pixels / 2U;
	size_t best_center_x = expected_center_x;
	size_t best_center_y = expected_center_y;
	float best_radius = 0.35f * (float)((frame_width_pixels < frame_height_pixels)
		? frame_width_pixels : frame_height_pixels);
	float best_score = -1.0e30f;
	const size_t min_dimension =
		(frame_width_pixels < frame_height_pixels) ? frame_width_pixels :
		frame_height_pixels;
	const size_t radius_min =
		(APP_BASELINE_HOUGH_RADIUS_MIN_PIXELS < (min_dimension / 3U))
			? APP_BASELINE_HOUGH_RADIUS_MIN_PIXELS : (min_dimension / 5U);
	const size_t radius_max =
		(APP_BASELINE_HOUGH_RADIUS_MAX_PIXELS < (min_dimension / 2U))
			? APP_BASELINE_HOUGH_RADIUS_MAX_PIXELS : (min_dimension / 2U);
	const size_t center_min_x =
		(expected_center_x > APP_BASELINE_HOUGH_CENTER_X_MARGIN)
			? (expected_center_x - APP_BASELINE_HOUGH_CENTER_X_MARGIN) : 1U;
	const size_t center_max_x =
		((expected_center_x + APP_BASELINE_HOUGH_CENTER_X_MARGIN) <
		 frame_width_pixels - 1U)
			? (expected_center_x + APP_BASELINE_HOUGH_CENTER_X_MARGIN) :
			(frame_width_pixels - 2U);
	const size_t center_min_y =
		(expected_center_y > APP_BASELINE_HOUGH_CENTER_Y_MARGIN)
			? (expected_center_y - APP_BASELINE_HOUGH_CENTER_Y_MARGIN) : 1U;
	const size_t center_max_y =
		((expected_center_y + APP_BASELINE_HOUGH_CENTER_Y_MARGIN) <
		 frame_height_pixels - 1U)
			? (expected_center_y + APP_BASELINE_HOUGH_CENTER_Y_MARGIN) :
			(frame_height_pixels - 2U);

	if ((frame_bytes == NULL) || (center_x_out == NULL) ||
		(center_y_out == NULL) || (radius_out == NULL) ||
		(frame_width_pixels < 16U) || (frame_height_pixels < 16U))
	{
		return;
	}

	AppGaugeGeometry_TrainingCropCenter(
		frame_width_pixels, frame_height_pixels,
		&expected_center_x, &expected_center_y);

	for (size_t radius = radius_min;
		 radius <= radius_max;
		 radius += APP_BASELINE_HOUGH_RADIUS_COARSE_STEP)
	{
		for (size_t candidate_y = center_min_y;
			 candidate_y <= center_max_y;
			 candidate_y += APP_BASELINE_HOUGH_CENTER_COARSE_STEP)
		{
			for (size_t candidate_x = center_min_x;
				 candidate_x <= center_max_x;
				 candidate_x += APP_BASELINE_HOUGH_CENTER_COARSE_STEP)
			{
				const float score = AppBaselineHough_ScoreFace(
					frame_bytes, frame_width_pixels, frame_height_pixels,
					candidate_x, candidate_y, (float)radius,
					expected_center_x, expected_center_y);
				if (score > best_score)
				{
					best_score = score;
					best_center_x = candidate_x;
					best_center_y = candidate_y;
					best_radius = (float)radius;
				}
			}
		}
	}

	/* A small refinement recovers the fractional center movement that matters
	 * more than another expensive full-frame search. */
	for (long radius_offset = -4L;
		 radius_offset <= 4L;
		 radius_offset += APP_BASELINE_HOUGH_RADIUS_FINE_STEP)
	{
		const long candidate_radius =
			(long)lroundf(best_radius) + radius_offset;
		if ((candidate_radius < (long)radius_min) ||
			(candidate_radius > (long)radius_max))
		{
			continue;
		}
		for (long y_offset = -8L; y_offset <= 8L;
			 y_offset += APP_BASELINE_HOUGH_CENTER_FINE_STEP)
		{
			for (long x_offset = -8L; x_offset <= 8L;
				 x_offset += APP_BASELINE_HOUGH_CENTER_FINE_STEP)
			{
				const long candidate_x = (long)best_center_x + x_offset;
				const long candidate_y = (long)best_center_y + y_offset;
				if ((candidate_x < (long)center_min_x) ||
					(candidate_x > (long)center_max_x) ||
					(candidate_y < (long)center_min_y) ||
					(candidate_y > (long)center_max_y))
				{
					continue;
				}
				const float score = AppBaselineHough_ScoreFace(
					frame_bytes, frame_width_pixels, frame_height_pixels,
					(size_t)candidate_x, (size_t)candidate_y,
					(float)candidate_radius, expected_center_x, expected_center_y);
				if (score > best_score)
				{
					best_score = score;
					best_center_x = (size_t)candidate_x;
					best_center_y = (size_t)candidate_y;
					best_radius = (float)candidate_radius;
				}
			}
		}
	}

	*center_x_out = best_center_x;
	*center_y_out = best_center_y;
	*radius_out = best_radius;
}

/**
 * @brief Return one radial line score from local dark-line contrast.
 * @param frame_bytes Packed YUV422 frame.
 * @param frame_width_pixels Frame width in pixels.
 * @param frame_height_pixels Frame height in pixels.
 * @param center_x Calibrated center x coordinate.
 * @param center_y Calibrated center y coordinate.
 * @param dial_radius_px Approximate inner-dial radius.
 * @param angle_rad Candidate angle in the live gauge convention.
 * @return Continuity-weighted dark-line score.
 */
static float AppBaselineHough_ScoreRay(
	const uint8_t *frame_bytes, size_t frame_width_pixels,
	size_t frame_height_pixels, size_t center_x, size_t center_y,
	float dial_radius_px, float angle_rad)
{
	const float unit_x = cosf(angle_rad);
	/* The board/AI angle convention negates image-space Y. */
	const float unit_y = -sinf(angle_rad);
	const float perpendicular_x = -unit_y;
	const float perpendicular_y = unit_x;
	const float center_x_f = (float)center_x;
	const float center_y_f = (float)center_y;
	float contrast_sum = 0.0f;
	size_t positive_samples = 0U;
	size_t longest_run = 0U;
	size_t current_run = 0U;

	/* HARD GATE: check hub darkness first. The needle must connect to a
	 * dark center hub. Dial markings don't reach the center. */
	float hub_darkness_sum = 0.0f;
	size_t hub_count = 0U;
	for (size_t hub_idx = 0U; hub_idx < 6U; ++hub_idx)
	{
		const float hub_r_frac = 0.08f + (0.17f * (float)hub_idx / 5.0f);
		const float hub_r = dial_radius_px * hub_r_frac;
		const long hx = (long)lroundf(center_x_f + (unit_x * hub_r));
		const long hy = (long)lroundf(center_y_f + (unit_y * hub_r));
		if (hx >= 0L && (size_t)hx < frame_width_pixels &&
			hy >= 0L && (size_t)hy < frame_height_pixels)
		{
			const float luma = AppBaselineHough_ReadLuma(
				frame_bytes, frame_width_pixels, (size_t)hx, (size_t)hy);
			hub_darkness_sum += (255.0f - luma) / 255.0f;
			hub_count++;
		}
	}
	if (hub_count > 0U && (hub_darkness_sum / (float)hub_count) < 0.15f)
	{
		return 0.0f;  /* No hub connection - not the needle */
	}

	for (size_t sample_index = 0U;
		 sample_index < APP_BASELINE_HOUGH_RAY_SAMPLES;
		 ++sample_index)
	{
		const float sample_fraction =
			(float)sample_index /
			(float)(APP_BASELINE_HOUGH_RAY_SAMPLES - 1U);
		const float radius = dial_radius_px *
			(APP_BASELINE_HOUGH_RAY_START_FRACTION +
			 ((APP_BASELINE_HOUGH_RAY_END_FRACTION -
			   APP_BASELINE_HOUGH_RAY_START_FRACTION) * sample_fraction));
		const long sample_x = (long)lroundf(center_x_f + (unit_x * radius));
		const long sample_y = (long)lroundf(center_y_f + (unit_y * radius));
		float background_sum = 0.0f;
		size_t background_count = 0U;

		if ((sample_x < 0L) || (sample_y < 0L) ||
			((size_t)sample_x >= frame_width_pixels) ||
			((size_t)sample_y >= frame_height_pixels))
		{
			current_run = 0U;
			continue;
		}

		const float line_luma = AppBaselineHough_ReadLuma(
			frame_bytes, frame_width_pixels, (size_t)sample_x, (size_t)sample_y);

		for (size_t offset_index = 0U;
			 offset_index < APP_BASELINE_HOUGH_BACKGROUND_OFFSETS;
			 ++offset_index)
		{
			const float offset = 2.0f + (2.0f * (float)offset_index);
			const long plus_x = (long)lroundf(
				(float)sample_x + (perpendicular_x * offset));
			const long plus_y = (long)lroundf(
				(float)sample_y + (perpendicular_y * offset));
			const long minus_x = (long)lroundf(
				(float)sample_x - (perpendicular_x * offset));
			const long minus_y = (long)lroundf(
				(float)sample_y - (perpendicular_y * offset));

			if ((plus_x >= 0L) && (plus_y >= 0L) &&
				((size_t)plus_x < frame_width_pixels) &&
				((size_t)plus_y < frame_height_pixels))
			{
				background_sum += AppBaselineHough_ReadLuma(
					frame_bytes, frame_width_pixels, (size_t)plus_x,
					(size_t)plus_y);
				background_count++;
			}

			if ((minus_x >= 0L) && (minus_y >= 0L) &&
				((size_t)minus_x < frame_width_pixels) &&
				((size_t)minus_y < frame_height_pixels))
			{
				background_sum += AppBaselineHough_ReadLuma(
					frame_bytes, frame_width_pixels, (size_t)minus_x,
					(size_t)minus_y);
				background_count++;
			}
		}

		if (background_count == 0U)
		{
			current_run = 0U;
			continue;
		}

		const float background_luma =
			background_sum / (float)background_count;
		const float contrast = background_luma - line_luma;

		if (contrast > APP_BASELINE_HOUGH_MIN_CONTRAST)
		{
			const float darkness =
				(255.0f - line_luma) / 255.0f;
			contrast_sum += contrast * (0.50f + (0.50f * darkness));
			positive_samples++;
			current_run++;
			if (current_run > longest_run)
			{
				longest_run = current_run;
			}
		}
		else
		{
			current_run = 0U;
		}
	}

	if (positive_samples == 0U)
	{
		return 0.0f;
	}

	{
		const float positive_fraction =
			(float)positive_samples / (float)APP_BASELINE_HOUGH_RAY_SAMPLES;
		const float continuity_fraction =
			(float)longest_run / (float)APP_BASELINE_HOUGH_RAY_SAMPLES;
		const float mean_contrast =
			contrast_sum / (float)positive_samples;

		return mean_contrast *
			(0.35f + (0.65f * positive_fraction)) *
			(0.30f + (0.70f * continuity_fraction));
	}
}

/**
 * @brief Find the strongest and second-best separated Hough bins.
 * @param scores Angle-bin scores.
 * @param best_index_out Winning bin destination.
 * @param best_score_out Winning score destination.
 * @param runner_up_score_out Separated runner-up score destination.
 */
static void AppBaselineHough_FindPeaks(
	const float scores[APP_BASELINE_HOUGH_ANGLE_BINS],
	size_t *best_index_out, float *best_score_out,
	float *runner_up_score_out)
{
	size_t best_index = 0U;
	float best_score = 0.0f;
	float runner_up_score = 0.0f;

	for (size_t index = 0U; index < APP_BASELINE_HOUGH_ANGLE_BINS; ++index)
	{
		if (scores[index] > best_score)
		{
			best_score = scores[index];
			best_index = index;
		}
	}

	for (size_t index = 0U; index < APP_BASELINE_HOUGH_ANGLE_BINS; ++index)
	{
		const size_t distance =
			(index > best_index) ? (index - best_index) : (best_index - index);
		if ((distance > APP_BASELINE_HOUGH_PEAK_SUPPRESSION_BINS) &&
			(scores[index] > runner_up_score))
		{
			runner_up_score = scores[index];
		}
	}

	*best_index_out = best_index;
	*best_score_out = best_score;
	*runner_up_score_out = runner_up_score;
}

/**
 * @brief Run the fresh center-plus-radial-Hough baseline.
 */
bool AppBaselineHough_Estimate(
	const uint8_t *frame_bytes, size_t frame_size,
	size_t frame_width_pixels, size_t frame_height_pixels,
	AppBaselineRuntime_Estimate_t *estimate_out)
{
	const float min_angle_rad =
		APP_BASELINE_HOUGH_MIN_ANGLE_DEG *
		(APP_BASELINE_HOUGH_PI / 180.0f);
	const float sweep_rad =
		APP_BASELINE_HOUGH_SWEEP_DEG *
		(APP_BASELINE_HOUGH_PI / 180.0f);
	float dial_radius_px = 0.0f;
	float scores[APP_BASELINE_HOUGH_ANGLE_BINS] = {0.0f};
	size_t center_x = 0U;
	size_t center_y = 0U;
	size_t best_index = 0U;
	float best_score = 0.0f;
	float runner_up_score = 0.0f;

	if ((frame_bytes == NULL) || (estimate_out == NULL) ||
		(frame_width_pixels == 0U) || (frame_height_pixels == 0U) ||
		(frame_size < (frame_width_pixels * frame_height_pixels * 2U)))
	{
		return false;
	}

	/* The training geometry is the trivial, reproducible center detector. */
	AppGaugeGeometry_TrainingCropCenter(
		frame_width_pixels, frame_height_pixels, &center_x, &center_y);
	AppBaselineHough_FindFaceGeometry(
		frame_bytes, frame_width_pixels, frame_height_pixels,
		&center_x, &center_y, &dial_radius_px);

	for (size_t bin_index = 0U;
		 bin_index < APP_BASELINE_HOUGH_ANGLE_BINS; ++bin_index)
	{
		const float fraction =
			(float)bin_index /
			(float)(APP_BASELINE_HOUGH_ANGLE_BINS - 1U);
		const float angle_rad = min_angle_rad + (fraction * sweep_rad);
		scores[bin_index] = AppBaselineHough_ScoreRay(
			frame_bytes, frame_width_pixels, frame_height_pixels,
			center_x, center_y, dial_radius_px, angle_rad);
	}

	AppBaselineHough_FindPeaks(
		scores, &best_index, &best_score, &runner_up_score);

	if ((best_score < APP_BASELINE_HOUGH_MIN_SCORE) ||
		(runner_up_score <= 0.0f) ||
		((best_score / runner_up_score) < APP_BASELINE_HOUGH_MIN_PEAK_RATIO))
	{
		return false;
	}

	{
		float bin_offset = 0.0f;
		if ((best_index > 0U) &&
			((best_index + 1U) < APP_BASELINE_HOUGH_ANGLE_BINS))
		{
			const float previous_score = scores[best_index - 1U];
			const float next_score = scores[best_index + 1U];
			const float denominator =
				previous_score - (2.0f * best_score) + next_score;
			if (fabsf(denominator) > 1.0e-6f)
			{
				bin_offset = 0.5f * (previous_score - next_score) / denominator;
				if (bin_offset > 0.5f)
				{
					bin_offset = 0.5f;
				}
				else if (bin_offset < -0.5f)
				{
					bin_offset = -0.5f;
				}
			}
		}

		const float refined_fraction =
			((float)best_index + bin_offset) /
			(float)(APP_BASELINE_HOUGH_ANGLE_BINS - 1U);
		const float angle_rad = min_angle_rad +
			(refined_fraction * sweep_rad);
		const float peak_ratio = best_score / runner_up_score;

		(void)memset(estimate_out, 0, sizeof(*estimate_out));
		estimate_out->valid = true;
		estimate_out->center_x = center_x;
		estimate_out->center_y = center_y;
		estimate_out->angle_rad = angle_rad;
		estimate_out->temperature_c =
			AppBaselineRuntime_ConvertAngleToTemperature(angle_rad);
		estimate_out->confidence = 1.0f + (0.5f * peak_ratio);
		estimate_out->best_score = best_score;
		estimate_out->runner_up_score = runner_up_score;
		estimate_out->source_label = APP_BASELINE_HOUGH_SOURCE_LABEL;
	}

	return true;
}
