/**
 * @file    app_inner_celsius_mask.h
 * @brief   Inner-Celsius-only mask for the tip-focus preprocessing path.
 *
 * Applies a circular keep region centered at the inner Celsius dial centre
 * plus a lower exclusion zone.  The mask blanks pixels that correspond to
 * outer-Fahrenheit labels, the lower subdial, and the taped blob/distractor
 * below the main dial.
 *
 * Geometry (224x224 image space):
 *   - Keep circle: centre (112, 100), radius 62 px
 *   - Lower exclusion: rows >= 150 are blanked (even inside the circle)
 *
 * Masked pixels are set to -128 (the int8 zero-point), so the model sees
 * zero-valued float32 input at those locations.
 *
 * This replaces the older AppAI_BlankTipFocusLowerInset elliptical mask.
 * The same geometry is mirrored in:
 *   ml/src/embedded_gauge_reading_tinyml/inner_celsius_mask.py
 */

#ifndef __APP_INNER_CELSIUS_MASK_H
#define __APP_INNER_CELSIUS_MASK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Mask geometry constants                                           */
/* ------------------------------------------------------------------ */

/** X coordinate of the inner Celsius dial centre on a 224-wide frame. */
#define APP_INNER_CELSIUS_MASK_CENTER_X  112U

/** Y coordinate of the inner Celsius dial centre on a 224-high frame. */
#define APP_INNER_CELSIUS_MASK_CENTER_Y  100U

/** Radius in pixels of the inner-Celsius keep circle. */
#define APP_INNER_CELSIUS_MASK_KEEP_RADIUS_PX  62U

/** Row index (exclusive) at which the lower exclusion zone starts.
 *  Pixels with row >= LOWER_EXCLUDE_Y are blanked even if they lie
 *  inside the keep circle. */
#define APP_INNER_CELSIUS_MASK_LOWER_EXCLUDE_Y  150U

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Apply the inner-Celsius-only mask to an int8 RGB image.
 *
 * For every pixel that falls outside the keep circle OR inside the
 * lower exclusion zone, all three colour channels are written to
 * -128 (the int8 quantisation zero-point, which decodes to 0.0 f32).
 *
 * @param input   Pointer to the int8 RGB NHWC image (row-major,
 *                height * width * 3 bytes).  Modified in place.
 * @param width   Image width in pixels (expected 224).
 * @param height  Image height in pixels (expected 224).
 */
void AppInnerCelsiusMask_Apply(int8_t *input, size_t width, size_t height);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INNER_CELSIUS_MASK_H */
