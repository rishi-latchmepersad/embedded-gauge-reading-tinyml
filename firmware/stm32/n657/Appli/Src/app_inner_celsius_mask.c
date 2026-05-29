/**
 * @file    app_inner_celsius_mask.c
 * @brief   Inner-Celsius-only mask implementation.
 */

#include "app_inner_celsius_mask.h"

#include <stddef.h>
#include <stdint.h>

void AppInnerCelsiusMask_Apply(int8_t *input, size_t width, size_t height)
{
    if ((input == NULL) || (width == 0U) || (height == 0U))
    {
        return;
    }

    /* Mask geometry constants are defined for 224x224 image space.
     * Scale them to the actual input dimensions so the keep circle
     * and exclusion zone align correctly regardless of model size. */
    const float scale = (float)width / 224.0f;

    const int32_t cx = (int32_t)((float)APP_INNER_CELSIUS_MASK_CENTER_X * scale);
    const int32_t cy = (int32_t)((float)APP_INNER_CELSIUS_MASK_CENTER_Y * scale);
    const int32_t r2 = (int32_t)((float)APP_INNER_CELSIUS_MASK_KEEP_RADIUS_PX *
                                  APP_INNER_CELSIUS_MASK_KEEP_RADIUS_PX * scale * scale);
    const int32_t exclude_y = (int32_t)((float)APP_INNER_CELSIUS_MASK_LOWER_EXCLUDE_Y * scale);

    for (size_t py = 0U; py < height; ++py)
    {
        const int32_t dy = (int32_t)(int32_t)py - cy;
        const int32_t dy2 = dy * dy;

        for (size_t px = 0U; px < width; ++px)
        {
            int32_t blank = 0;
            const int32_t dx = (int32_t)(int32_t)px - cx;
            const int32_t dist2 = dx * dx + dy2;

            if (dist2 > r2)
            {
                /* Outside the keep circle: blank */
                blank = 1;
            }
            if ((int32_t)py >= exclude_y)
            {
                /* In the lower exclusion zone: blank */
                blank = 1;
            }

            if (blank)
            {
                const size_t idx = (py * width + px) * 3U;
                input[idx]     = -128;
                input[idx + 1U] = -128;
                input[idx + 2U] = -128;
            }
        }
    }
}
