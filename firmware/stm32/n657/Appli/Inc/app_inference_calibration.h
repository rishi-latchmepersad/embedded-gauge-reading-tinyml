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
 * The current prod v0.4 board path leaves this as a pass-through so the
 * firmware reports the raw model output directly. Older deployments can
 * still enable the affine postprocess in app_inference_calibration.c.
 *
 * @param raw_value Model output in Celsius before calibration.
 * @return Calibrated output in Celsius, or the raw value when disabled.
 */
float AppInferenceCalibration_Apply(float raw_value);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_CALIBRATION_H */
