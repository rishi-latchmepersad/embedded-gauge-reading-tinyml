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
 * The live ellipse/center-tip board path leaves this as a no-op by default so the
 * model output is published directly. The legacy piecewise correction still
 * exists in app_inference_calibration.c for replay or experimentation, but
 * it is compiled out unless explicitly re-enabled there.
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
