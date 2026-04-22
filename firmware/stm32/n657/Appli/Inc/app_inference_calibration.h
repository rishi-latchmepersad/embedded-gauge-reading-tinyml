/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_inference_calibration.h
 * @brief   Scalar output calibration helpers for AI inference results.
 ******************************************************************************
 */
/* USER CODE END Header */

#ifndef __APP_INFERENCE_CALIBRATION_H
#define __APP_INFERENCE_CALIBRATION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply the deploy-time scalar calibration to the model output.
 *
 * The deployed scalar model uses a lightweight affine post-processing
 * correction derived from the board-crop probe fit.
 *
 * @param raw_value Model output in Celsius before calibration.
 * @return Calibrated output in Celsius.
 */
float AppInferenceCalibration_Apply(float raw_value);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_CALIBRATION_H */
