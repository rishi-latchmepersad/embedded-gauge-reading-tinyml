/**
 ******************************************************************************
 * @file    isp_conf.h
 * @author  AIS Application Team
 * @brief   Header configuration file - template
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __ISP_CONF_H
#define __ISP_CONF_H


/* Includes ------------------------------------------------------------------*/
#define ISP_MW_SW_AEC_ALGO_SUPPORT
#define ISP_MW_SW_AWB_ALGO_SUPPORT

/*
 * Product-build safety boundary.
 *
 * ST's optional ISP tuning transport can answer a host dump command with the
 * complete camera frame through USBX.  That is useful on a tuning bench but
 * it is never part of the gauge firmware console contract: it can appear as
 * binary/ASCII garbage on the same host serial session and it can expose the
 * live DMA buffer while capture is running.  Keep the symbol explicitly
 * undefined here even if a CubeIDE configuration or inherited command line
 * accidentally defines it.
 */
#ifdef ISP_MW_TUNING_TOOL_SUPPORT
#undef ISP_MW_TUNING_TOOL_SUPPORT
#endif

/* The vendor ISP algorithms use raw printf() for diagnostic chatter. That
 * path bypasses DebugConsole's UART serialization and was interleaving with
 * application records during capture. Keep the product UART owned by
 * DebugConsole; a tuning image can omit this header override. stdio is
 * included first so the macro cannot interfere with its declarations. */
#ifndef APP_ISP_TUNING_IMAGE
#include <stdio.h>
#define printf(...) ((void)0)
#endif

#endif /* __ISP_CONF_H */
