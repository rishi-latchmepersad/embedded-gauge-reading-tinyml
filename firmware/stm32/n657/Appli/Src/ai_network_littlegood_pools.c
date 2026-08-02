/**
 * @file ai_network_littlegood_pools.c
 * @brief Absolute xSPI2 symbols for the LittleGood relocatable models.
 *
 * The generated relocatable sources require model-pool symbols at the flash
 * addresses used when the networks were compiled. Keeping these symbols
 * absolute avoids putting dummy data in the application image or consuming
 * activation SRAM; the actual bytes are provisioned by flash_boot.ps1.
 */

#if defined(__GNUC__)
__asm__(
    ".global _mem_pool_xSPI2_ocdet_ellipse_320_v2\n"
    ".set _mem_pool_xSPI2_ocdet_ellipse_320_v2, 0x70C00000\n"
    ".global _mem_pool_xSPI2_gauge_ellipse_littlegood_v2_gamma070\n"
    ".set _mem_pool_xSPI2_gauge_ellipse_littlegood_v2_gamma070, 0x70400000\n"
    ".global _mem_pool_xSPI2_gauge_center_tip_littlegood_unet_v1\n"
    ".set _mem_pool_xSPI2_gauge_center_tip_littlegood_unet_v1, 0x70800000\n"
    ".global _mem_pool_xSPI2_ellipse_iter8_universal_wide_deep_int8\n"
    ".set _mem_pool_xSPI2_ellipse_iter8_universal_wide_deep_int8, 0x70400000\n"
    ".global _mem_pool_xSPI2_keypoint_unet_224g_wide_aug_int8\n"
    ".set _mem_pool_xSPI2_keypoint_unet_224g_wide_aug_int8, 0x70800000\n"
);
#endif
