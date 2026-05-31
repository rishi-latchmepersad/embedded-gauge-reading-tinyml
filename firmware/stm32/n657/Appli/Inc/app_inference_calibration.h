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
 * The board currently uses the active piecewise postprocess in
 * app_inference_calibration.c so we can compare calibrated and raw outputs
 * during live capture work. If the correction needs to be disabled again,
 * that file owns the switch.
 *
 * @param raw_value Model output in Celsius before calibration.
 * @return Calibrated output in Celsius, or the raw value when calibration is
 *         compiled out.
 */
float AppInferenceCalibration_Apply(float raw_value);

#ifdef __cplusplus
}
#endif

#endif /* __APP_INFERENCE_CALIBRATION_H */
