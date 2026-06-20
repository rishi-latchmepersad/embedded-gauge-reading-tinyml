/**
 * @file    app_qarepvgg_decode.inc
 * @brief   QARepVGG-Pro heatmap decode — replaces scalar OBB+centre regressor.
 *
 * Included by app_ai_helpers.inc.  Provides AppAI_DecodeQarepvggOutput() which
 * reads the NPU's 3-tensor output (heatmap, box_size, angle) and extracts:
 *   - gauge centre (cx, cy) via parabolic sub-pixel peak refinement
 *   - OBB box (w, h) at the peak cell
 *   - rotation (cos2θ, sin2θ) at the peak cell
 *
 * The model outputs int8 tensors in ch-first layout (c_info.json: channel_first=true):
 *   heatmap  [1, 1, HEATMAP_H, HEATMAP_W]   — logits (pre-sigmoid)
 *   box_size [1, 2, HEATMAP_H, HEATMAP_W]   — (w, h) in [0,1] norm space
 *   angle    [1, 2, HEATMAP_H, HEATMAP_W]   — tanh → (sin2θ, cos2θ)
 * where HEATMAP_H = HEATMAP_W = 40 (320 / 8 downsampling).
 * Memory layout per 2-channel tensor: ch0[0..N-1] then ch1[N..2N-1],
 * where N = H*W = 1600, ch0[y*W + x] and ch1[N + y*W + x].
 *
 * Decoding flow:
 *   1. Dequantise int8 → float for each tensor.
 *   2. Apply sigmoid to heatmap logits.
 *   3. Find argmax of sigmoid heatmap.
 *   4. Parabolic 1-D refinement along x and y for sub-pixel centre.
 *   5. Read box_size[w,h] and angle[sin2θ,cos2θ] at the peak cell.
 *   6. Normalise pixel coords [0,39] → [0,1] for cx, cy.
 */

#ifndef APP_QAREPVGG_DECODE_INC
#define APP_QAREPVGG_DECODE_INC

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <float.h>

/* Heatmap grid dimensions — must match the model's output shape. */
#ifndef QAREPVGG_HEATMAP_SIZE
#define QAREPVGG_HEATMAP_SIZE      40U
#endif
#ifndef QAREPVGG_HEATMAP_PIXELS
#define QAREPVGG_HEATMAP_PIXELS    (QAREPVGG_HEATMAP_SIZE * QAREPVGG_HEATMAP_SIZE)
#endif

/* Output tensor indices as reported by Cube.AI / interpreter. */
#ifndef QAREPVGG_OUTPUT_IDX_HEATMAP
#define QAREPVGG_OUTPUT_IDX_HEATMAP  2U   /* StatefulPartitionedCall:2 */
#endif
#ifndef QAREPVGG_OUTPUT_IDX_BOX
#define QAREPVGG_OUTPUT_IDX_BOX      1U   /* StatefulPartitionedCall:1 */
#endif
#ifndef QAREPVGG_OUTPUT_IDX_ANGLE
#define QAREPVGG_OUTPUT_IDX_ANGLE    0U   /* StatefulPartitionedCall:0 */
#endif

/**
 * @brief Decoded QARepVGG-Pro output — gauge centre + OBB box + rotation.
 */
typedef struct {
    float center_x;      /**< Normalised [0,1] gauge centre x in the legacy 320×320 frame */
    float center_y;      /**< Normalised [0,1] gauge centre y in the legacy 320×320 frame */
    float box_w;         /**< Normalised [0,1] OBB width in the legacy 320×320 frame */
    float box_h;         /**< Normalised [0,1] OBB height in the legacy 320×320 frame */
    float cos2theta;     /**< cos(2θ) ∈ [-1, 1] — gauge rotation */
    float sin2theta;     /**< sin(2θ) ∈ [-1, 1] */
    float peak_value;    /**< Sigmoid value at detected peak — confidence proxy */
    bool  valid;         /**< True if a valid peak was found */
} QarepvggOutput;

/**
 * @brief 1-D parabolic sub-pixel refinement around an integer argmax.
 *
 * Given three consecutive samples f(x0-1), f(x0), f(x0+1), fits a parabola
 * f(x) = a(x-x0)^2 + b(x-x0) + c and returns the vertex offset Δx:
 *   Δx = (f(x0-1) - f(x0+1)) / (2 * (2*f(x0) - f(x0-1) - f(x0+1)))
 *
 * @param left   f(x0-1)
 * @param center f(x0)
 * @param right  f(x0+1)
 * @return       Sub-pixel offset in [-0.5, 0.5] or 0.0f if denominator ≈ 0.
 */
static inline float _parabolic_offset(float left, float center, float right)
{
    float denom = 2.0f * center - left - right;          /* 2·f(x₀) - f(x₀−1) - f(x₀+1) */
    if (fabsf(denom) < 1e-7f) {
        return 0.0f;                                     /* flat peak → no offset */
    }
    return (left - right) / (2.0f * denom);              /* Δx = (L−R) / (2·denom) */
}

/**
 * @brief Fast sigmoid: σ(x) = 1 / (1 + exp(-x)).
 */
static inline float _sigmoid_f(float x)
{
    if (x >= 20.0f)  return 1.0f;        /* clamp to avoid overflow */
    if (x <= -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Dequantise an int8 array to float32.
 *
 * @param dst       Output float buffer (caller-allocated, len elements).
 * @param src       Input int8 buffer.
 * @param len       Number of elements.
 * @param scale     Dequantisation scale.
 * @param zero_point Dequantisation zero point.
 */
static void _dequantise_int8(float *dst, const int8_t *src, size_t len,
                              float scale, int32_t zero_point)
{
    for (size_t i = 0U; i < len; ++i) {
        dst[i] = ((float)(int32_t)src[i] - (float)zero_point) * scale;
    }
}

/**
 * @brief Decode the QARepVGG-Pro NPU output into a QarepvggOutput struct.
 *
 * @param out             [out]  Filled with decoded results.
 * @param output_buffers  Array of pointer+size pairs from the NPU stage:
 *                        [0]=heatmap, [1]=box_size, [2]=angle (int8 pointers).
 * @param hmap_scale      Heatmap tensor dequantisation scale (float).
 * @param hmap_zp         Heatmap tensor dequantisation zero point (int32).
 * @param box_scale       Box_size tensor dequantisation scale.
 * @param box_zp          Box_size tensor dequantisation zero point.
 * @param ang_scale       Angle tensor dequantisation scale.
 * @param ang_zp          Angle tensor dequantisation zero point.
 */
static void AppAI_DecodeQarepvggOutput(
    QarepvggOutput            *out,
    const int8_t              *heatmap_raw,     /* [HEATMAP_PIXELS] int8 */
    const int8_t              *box_raw,         /* [HEATMAP_PIXELS * 2] int8 */
    const int8_t              *angle_raw,       /* [HEATMAP_PIXELS * 2] int8 */
    float                      hmap_scale,
    int32_t                    hmap_zp,
    float                      box_scale,
    int32_t                    box_zp,
    float                      ang_scale,
    int32_t                    ang_zp)
{
    const uint32_t H  = QAREPVGG_HEATMAP_SIZE;
    const uint32_t W  = QAREPVGG_HEATMAP_SIZE;
    const uint32_t N  = QAREPVGG_HEATMAP_PIXELS;

    /* ── 1. Dequantise heatmap (logits) → sigmoid ───────────────────────── */
    float hmap_f[N];  /* stack buffer for 40×40 = 1600 floats → 6.4 KB */
    _dequantise_int8(hmap_f, heatmap_raw, N, hmap_scale, hmap_zp);

    /* Apply sigmoid in-place so we have a proper probability map. */
    for (uint32_t i = 0U; i < N; ++i) {
        hmap_f[i] = _sigmoid_f(hmap_f[i]);
    }

    /* ── 2. Argmax of the sigmoid heatmap ───────────────────────────────── */
    uint32_t best_idx = 0U;
    float    best_val = -FLT_MAX;
    for (uint32_t i = 0U; i < N; ++i) {
        if (hmap_f[i] > best_val) {
            best_val = hmap_f[i];
            best_idx = i;
        }
    }

    if (best_val < 0.02f) {   /* dead heatmap → no valid centre */
        memset(out, 0, sizeof(*out));
        out->valid = false;
        return;
    }
    out->peak_value = best_val;

    uint32_t y0 = best_idx / W;
    uint32_t x0 = best_idx % W;

    /* ── 3. Parabolic sub-pixel refinement in x and y ──────────────────── */
    float px = (float)x0;
    float py = (float)y0;

    /* Refine x */
    if (x0 >= 1U && x0 <= W - 2U) {
        uint32_t idx = y0 * W;
        px += _parabolic_offset(hmap_f[idx + x0 - 1U],
                                 hmap_f[idx + x0],
                                 hmap_f[idx + x0 + 1U]);
    }
    /* Refine y */
    if (y0 >= 1U && y0 <= H - 2U) {
        py += _parabolic_offset(hmap_f[(y0 - 1U) * W + x0],
                                 hmap_f[y0 * W + x0],
                                 hmap_f[(y0 + 1U) * W + x0]);
    }

    /* Clamp to grid bounds */
    if (px < 0.0f) px = 0.0f;
    if (py < 0.0f) py = 0.0f;
    if (px > (float)(W - 1U)) px = (float)(W - 1U);
    if (py > (float)(H - 1U)) py = (float)(H - 1U);

    /* Normalise: heatmap coords [0, H-1] → frame coords [0, 1] */
    out->center_x = px / (float)(W - 1U);
    out->center_y = py / (float)(H - 1U);

    /* ── 4. Decode box_size and angle at the integer peak cell ──────────── */
    /* The cell index is clamped back to integer for box/angle lookup since
     * the per-cell regression target applies to the grid cell, not the sub-pixel
     * position.  Using the integer argmax cell is standard practice. */
    /* Cube.AI output tensors are ch-first (c_info.json: channel_first=true).
     * For [H, W, 2] ch-first: channel 0 occupies elements 0..H*W-1,
     * channel 1 occupies elements H*W..2*H*W-1.
     * cell_i = y0*W + x0 gives the spatial offset within each channel. */
    uint32_t cell_i = y0 * W + x0;

    /* Dequantise box (w, h) from ch-first layout.
     * Row 0: box_w for all cells, row 1 (=H*W offset): box_h for all cells. */
    out->box_w = ((float)(int32_t)box_raw[cell_i] - (float)box_zp) * box_scale;
    out->box_h = ((float)(int32_t)box_raw[N + cell_i] - (float)box_zp) * box_scale;
    if (out->box_w < 0.0f) { out->box_w = 0.0f; }
    if (out->box_h < 0.0f) { out->box_h = 0.0f; }

    /* Dequantise angle (sin2θ, cos2θ) from ch-first layout. */
    out->sin2theta = ((float)(int32_t)angle_raw[cell_i] - (float)ang_zp) * ang_scale;
    out->cos2theta = ((float)(int32_t)angle_raw[N + cell_i] - (float)ang_zp) * ang_scale;
    /* Clamp to valid range (tanh output ∈ [-1, 1]). */
    if (out->sin2theta < -1.0f) { out->sin2theta = -1.0f; }
    if (out->sin2theta >  1.0f) { out->sin2theta =  1.0f; }
    if (out->cos2theta < -1.0f) { out->cos2theta = -1.0f; }
    if (out->cos2theta >  1.0f) { out->cos2theta =  1.0f; }

    out->valid = true;
}

#endif /* APP_QAREPVGG_DECODE_INC */
