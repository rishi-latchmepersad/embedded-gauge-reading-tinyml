/**
 * @file ai_network_littlegood_models.h
 * @brief Public interfaces for the exact-int8 LittleGood deployment models.
 */

#ifndef AI_NETWORK_LITTLEGOOD_MODELS_H
#define AI_NETWORK_LITTLEGOOD_MODELS_H

#include "ll_aton_NN_interface.h"

/** Full-frame 320x320x1 ellipse proposer API. */
bool AppAI_LittleGoodProposer_Init(void);
bool AppAI_LittleGoodProposer_Run(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodProposer_InputInfo(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodProposer_OutputInfo(void);

/** Local 256x256x1 ellipse refiner API. */
bool AppAI_LittleGoodRefiner_Init(void);
bool AppAI_LittleGoodRefiner_Run(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodRefiner_InputInfo(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodRefiner_OutputInfo(void);

/** Ellipse-conditioned 160x160x2 center/tip U-Net API. */
bool AppAI_LittleGoodCenterTip_New_Init(void);
bool AppAI_LittleGoodCenterTip_New_Run(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodCenterTip_New_InputInfo(void);
const LL_Buffer_InfoTypeDef *AppAI_LittleGoodCenterTip_New_OutputInfo(void);

#endif /* AI_NETWORK_LITTLEGOOD_MODELS_H */
