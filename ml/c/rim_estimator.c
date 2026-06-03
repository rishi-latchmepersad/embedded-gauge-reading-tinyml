/*
 * Standalone firmware rim-centre estimator for offline label generation.
 *
 * Extracted from app_baseline_runtime.c — pure math operating on a YUV422
 * byte array, no STM32 dependencies.  Compiled into a shared library
 * (rim_estimator.so) for use by the Python training pipeline via ctypes.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---------- Firmware constants (from app_baseline_runtime.c) ---------- */

#define RIM_SATURATION_THRESHOLD        235.0f
#define RIM_SCAN_BORDER                 8
#define RIM_MIN_RADIUS                  16.0f
#define RIM_COARSE_STEP                 8
#define RIM_FINE_STEP                   4
#define RIM_SAMPLE_STEP                 4
#define RIM_RIM_MIN_FRACTION            0.84f
#define RIM_RIM_MAX_FRACTION            1.04f
#define RIM_SUBDIAL_X_FRACTION          0.35f
#define RIM_SUBDIAL_Y_MIN_FRACTION      0.10f
#define RIM_SUBDIAL_Y_MAX_FRACTION      0.58f

/* Training-crop ratios (from app_gauge_geometry.h) */
#define TRAINING_CROP_X_MIN_RATIO       0.1027f
#define TRAINING_CROP_Y_MIN_RATIO       0.2573f
#define TRAINING_CROP_X_MAX_RATIO       0.7987f
#define TRAINING_CROP_Y_MAX_RATIO       0.8071f

/* ---------- Helper functions (matching app_baseline_runtime.c) ---------- */

static inline float rim_clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/**
 * Read the Y (luma) component from a packed YUV422 pixel.
 * YUV422 layout: Y0 U0 Y1 V0 — each pair is 4 bytes.
 */
static float read_luma(const uint8_t *frame, size_t row_stride_bytes,
                        size_t x, size_t y)
{
    size_t pair_offset = y * row_stride_bytes + (x & ~1U) * 2U;
    return (float)frame[pair_offset];
}

/**
 * 3×3 Sobel edge magnitude and direction, matching
 * AppBaselineRuntime_ReadEdgeMagnitude.
 */
static float read_edge_magnitude(const uint8_t *frame, size_t row_stride,
                                  size_t width, size_t height,
                                  size_t x, size_t y,
                                  float *gx_out, float *gy_out)
{
    if (x < 1 || y < 1 || x + 1 >= width || y + 1 >= height) {
        if (gx_out) *gx_out = 0.0f;
        if (gy_out) *gy_out = 0.0f;
        return 0.0f;
    }
    float tl = read_luma(frame, row_stride, x - 1, y - 1);
    float tc = read_luma(frame, row_stride, x,     y - 1);
    float tr = read_luma(frame, row_stride, x + 1, y - 1);
    float ml = read_luma(frame, row_stride, x - 1, y);
    float mr = read_luma(frame, row_stride, x + 1, y);
    float bl = read_luma(frame, row_stride, x - 1, y + 1);
    float bc = read_luma(frame, row_stride, x,     y + 1);
    float br = read_luma(frame, row_stride, x + 1, y + 1);

    float gx = (tr + 2.0f*mr + br) - (tl + 2.0f*ml + bl);
    float gy = (bl + 2.0f*bc + br) - (tl + 2.0f*tc + tr);

    if (gx_out) *gx_out = gx;
    if (gy_out) *gy_out = gy;
    return sqrtf(gx * gx + gy * gy);
}

/**
 * Subdial clutter mask — matches AppBaselineRuntime_IsInSubdialMask.
 */
static bool is_in_subdial_mask(float cx, float cy, int x, int y,
                                float dial_radius)
{
    float dx = fabsf((float)x - cx);
    float dy = fabsf((float)y - cy);
    return (dx < RIM_SUBDIAL_X_FRACTION * dial_radius) &&
           ((float)y > cy + RIM_SUBDIAL_Y_MIN_FRACTION * dial_radius) &&
           ((float)y < cy + RIM_SUBDIAL_Y_MAX_FRACTION * dial_radius) &&
           (dy > RIM_SUBDIAL_Y_MIN_FRACTION * dial_radius);
}

/**
 * Training-crop geometry, matching AppGaugeGeometry_TrainingCrop.
 */
static void training_crop(size_t fw, size_t fh,
                           float *x0, float *y0, float *cw, float *ch)
{
    *x0 = fw * TRAINING_CROP_X_MIN_RATIO;
    *y0 = fh * TRAINING_CROP_Y_MIN_RATIO;
    *cw = fw * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO);
    *ch = fh * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO);
}

/* ---------- Core scoring and search (from app_baseline_runtime.c) ---------- */

/**
 * Score one candidate dial centre — matches
 * AppBaselineRuntime_ScoreDialCenterCandidate.
 */
static float score_rim_candidate(const uint8_t *frame, size_t row_stride,
                                  size_t fw, size_t fh,
                                  size_t sx0, size_t sy0,
                                  size_t sx1, size_t sy1,
                                  float dial_radius,
                                  size_t cx, size_t cy)
{
    float tcx, tcy, tcw, tch;
    training_crop(fw, fh, &tcx, &tcy, &tcw, &tch);
    float crop_centre_x = tcx + 0.5f * tcw;
    float crop_centre_y = tcy + 0.5f * tch;
    float crop_half_diag = sqrtf((0.5f * tcw) * (0.5f * tcw) +
                                  (0.5f * tch) * (0.5f * tch));
    float rim_min = dial_radius * RIM_RIM_MIN_FRACTION;
    float rim_max = dial_radius * RIM_RIM_MAX_FRACTION;
    float score = 0.0f;
    size_t count = 0;

    for (size_t y = sy0 + 1; y < sy1 - 1; y += RIM_SAMPLE_STEP) {
        for (size_t x = sx0 + 1; x < sx1 - 1; x += RIM_SAMPLE_STEP) {
            float dx = (float)x - (float)cx;
            float dy = (float)y - (float)cy;
            float r = sqrtf(dx*dx + dy*dy);
            float luma = read_luma(frame, row_stride, x, y);

            if (r < rim_min || r > rim_max) continue;
            if (luma > RIM_SATURATION_THRESHOLD) continue;
            if (is_in_subdial_mask((float)cx, (float)cy, (int)x, (int)y, dial_radius)) continue;

            float gx, gy;
            float mag = read_edge_magnitude(frame, row_stride, fw, fh, x, y, &gx, &gy);
            float gsafe = (mag > 1.0f) ? mag : 1.0f;
            float rx = dx / r;
            float ry = dy / r;
            float alignment = fabsf(gx / gsafe * rx + gy / gsafe * ry);
            float rim_bias = 1.0f - rim_clamp(
                fabsf(r - dial_radius) / (dial_radius + 1e-6f), 0.0f, 1.0f);
            float vote = mag * alignment * alignment * rim_bias * rim_bias;
            if (vote > 0.0f) {
                score += vote;
                count++;
            }
        }
    }
    if (count == 0) return 0.0f;

    float cdx = (float)cx - crop_centre_x;
    float cdy = (float)cy - crop_centre_y;
    float cdist = sqrtf(cdx * cdx + cdy * cdy);
    float center_prior = rim_clamp(
        1.0f - 0.25f * cdist / (crop_half_diag + 1e-6f), 0.20f, 1.0f);
    return (score / (float)count) * center_prior;
}

/**
 * Coarse-to-fine rim centre search — matches
 * AppBaselineRuntime_EstimateDialCenterFromRimVotes.
 *
 * @param frame       Pointer to YUV422 bytes.
 * @param row_stride  Bytes per row (= width * 2 for YUV422).
 * @param fw, fh      Frame width and height in pixels.
 * @param dial_radius Dial radius to search around.
 * @param[out] cx_out Best centre X (pixels).
 * @param[out] cy_out Best centre Y (pixels).
 * @param[out] q_out  Best centre quality score.
 * @return true if a centre was found.
 */
bool rim_estimator_find_center(const uint8_t *frame, size_t frame_size,
                                size_t row_stride, size_t fw, size_t fh,
                                float dial_radius,
                                float *cx_out, float *cy_out, float *q_out)
{
    size_t sx0 = RIM_SCAN_BORDER;
    size_t sy0 = RIM_SCAN_BORDER;
    size_t sx1 = fw - RIM_SCAN_BORDER;
    size_t sy1 = fh - RIM_SCAN_BORDER;
    float best_q = -1.0f;
    float best_cx = 0.0f, best_cy = 0.0f;

    if (frame == NULL || frame_size < fw * fh * 2 || dial_radius < RIM_MIN_RADIUS)
        return false;

    /* Coarse pass */
    for (size_t cy = sy0; cy < sy1; cy += RIM_COARSE_STEP) {
        for (size_t cx = sx0; cx < sx1; cx += RIM_COARSE_STEP) {
            float q = score_rim_candidate(frame, row_stride, fw, fh,
                                           sx0, sy0, sx1, sy1,
                                           dial_radius, cx, cy);
            if (q > best_q) {
                best_q = q;
                best_cx = (float)cx;
                best_cy = (float)cy;
            }
        }
    }
    if (best_q < 0.0f) return false;

    /* Fine pass around coarse winner */
    float fine_radius = (float)RIM_COARSE_STEP;
    float fmin_x = best_cx - fine_radius;
    float fmax_x = best_cx + fine_radius;
    float fmin_y = best_cy - fine_radius;
    float fmax_y = best_cy + fine_radius;
    if (fmin_x < (float)(int)sx0) fmin_x = (float)(int)sx0;
    if (fmax_x > (float)(int)(sx1 - 1)) fmax_x = (float)(int)(sx1 - 1);
    if (fmin_y < (float)(int)sy0) fmin_y = (float)(int)sy0;
    if (fmax_y > (float)(int)(sy1 - 1)) fmax_y = (float)(int)(sy1 - 1);

    for (int fy = (int)fmin_y; fy <= (int)fmax_y; fy += RIM_FINE_STEP) {
        for (int fx = (int)fmin_x; fx <= (int)fmax_x; fx += RIM_FINE_STEP) {
            float q = score_rim_candidate(frame, row_stride, fw, fh,
                                           sx0, sy0, sx1, sy1,
                                           dial_radius, (size_t)fx, (size_t)fy);
            if (q > best_q) {
                best_q = q;
                best_cx = (float)fx;
                best_cy = (float)fy;
            }
        }
    }

    *cx_out = best_cx;
    *cy_out = best_cy;
    *q_out = best_q;
    return true;
}
