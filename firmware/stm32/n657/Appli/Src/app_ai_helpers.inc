/* This file is included by app_ai.c. */

/* USER CODE BEGIN PFP */
static const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void);
static const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
	const AppAI_ModelStageSpec *stage);
static const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
	const AppAI_ModelStageSpec *stage);
static const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *name);
static const LL_Buffer_InfoTypeDef *AppAI_FindFirstBufferInfoByNames(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *const *names,
	size_t name_count);
static void AppAI_LogInitFailure(const char *step);
static void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
									  int32_t bsp_status);
static void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
											   ULONG flash_offset, ULONG chunk_size);
static void AppAI_LogXspi2FlashStatus(const char *label);
static void AppAI_LogXspi2PrefixBytes(const char *label,
									  const uint8_t *bytes);
static void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
									size_t frame_size);
static void AppAI_LogInputSignature(const float *input_buffer,
									size_t input_float_count);
static void AppAI_LogInputTensorWindow(const float *input_buffer,
									   size_t input_float_count);
static void AppAI_LogInputProbeSummary(const float *input_buffer,
									   size_t input_float_count);
static void AppAI_LogTensorRowSamples(const char *label,
									  const float *input_buffer, size_t tensor_width, size_t row_y,
									  size_t x_min, size_t x_max);
static void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
								 size_t frame_width_pixels, size_t center_x, size_t center_y,
								 size_t radius_pixels);
static void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
								 size_t tensor_width, size_t center_x, size_t center_y,
								 size_t radius_pixels);
static void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
									  size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
									  size_t crop_x_min, size_t crop_y_min, size_t crop_width,
									  size_t crop_height);
static void AppAI_LogInt8BufferSignature(const char *label,
										 const int8_t *buffer_ptr, size_t buffer_len_bytes);
static void AppAI_LogRawBufferSignature(const char *label,
										const uint8_t *buffer_ptr, size_t buffer_len_bytes);
static const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info);
static void AppAI_LogBufferInfoAndSignature(const char *label,
											const LL_Buffer_InfoTypeDef *buffer_info);
static void AppAI_LogBufferPreview(const char *label,
								   const LL_Buffer_InfoTypeDef *buffer_info);
static void AppAI_LogScalarInternalOutputProbe(
	const AppAI_ModelStageSpec *stage,
	const LL_Buffer_InfoTypeDef *stage_output_info);
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
static bool AppAI_EstimateGaugeCropBoxFromYuv422(const uint8_t *frame_bytes,
												 size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
												 size_t *crop_x_min, size_t *crop_y_min, size_t *crop_width,
												 size_t *crop_height);
#endif
static bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr);
static bool AppAI_ReadXspi2ModelSourceProbes(FX_FILE *model_file_ptr,
											 ULONG file_size, uint8_t *source_prefix, uint8_t *source_tail,
											 bool *has_tail_out);
static void AppAI_LogXspi2FlashPrefix(void);
static void AppAI_LogXspi2MappedScaleBytes(void);
static void AppAI_LogXspi2IndirectAndMappedPrefix(void);
static void AppAI_LogFloatApprox(const char *label, float value);
static float AppAI_TraceAndApplyInferenceCalibration(float raw_value);
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
static void AppAI_ResetInferenceBurstHistory(void);
#endif
static bool AppAI_IsFiniteFloat(float value);
static bool AppAI_IsPlausibleInferenceValue(float value);
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
static float AppAI_FilterInferenceValue(float value);
#endif
static void AppAI_LogInferenceResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info);
static void AppAI_LogRectifierResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_RectifierBox *rectifier_box);
static void AppAI_LogObbResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_ObbBox *obb_box);
static int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
								 bool clean, bool invalidate);
static uint32_t AppAI_GrayToFloatBits(uint8_t gray);
static bool AppAI_SoftArgmaxGlobal(const int8_t *heatmap, int32_t size,
	float *x_out, float *y_out);
static void AppAI_TipFocus_ResetMedianHistory(float seed_value);
static float AppAI_TipFocus_MedianFilter(float new_value);
static void AppAI_BlankTipFocusLowerInset(int8_t *input, size_t width,
	size_t height);
static void AppAI_FormatFixedFloat(char *dst, size_t dst_len, float value,
	unsigned decimals);

static bool __attribute__((noinline)) AppAI_PreprocessScalarRow(
	const uint8_t *frame_bytes, size_t frame_size, size_t source_width,
	size_t source_height, size_t crop_x_min, size_t crop_y_min,
	size_t crop_width, size_t crop_height, size_t output_width,
	size_t output_height, float resize_scale, size_t resized_width,
	size_t resized_height, size_t resize_pad_x, size_t resize_pad_y,
	size_t out_y, float *input_ptr, size_t input_len_bytes);
static void AppAI_EnableNpuMemoryAndCaches(void);
static void AppAI_ConfigureNpuAccessControl(void);
static void AppAI_ConfigureNpuRisafDefaults(void);
static bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage);
static bool AppAI_EnsureXspi2ModelImageReadyForStage(
	const AppAI_ModelStageSpec *stage);
static bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms);
static bool AppAI_RuntimeInitStepwise(void);
static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToFloatInput(
	const uint8_t *frame_bytes, size_t frame_size, float *input_buffer,
	size_t input_float_count, size_t input_len_bytes);
static bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToInt8Input(
	const uint8_t *frame_bytes, size_t frame_size, uint8_t *input_ptr,
	size_t input_len_bytes, const LL_Buffer_InfoTypeDef *input_info);
static float AppAI_ClampNormalizedFloat(float value);

/* Split the helper implementation into themed include fragments to keep this file manageable. */
#include "app_ai_helpers_core.inc"
#include "app_ai_helpers_logging.inc"
#include "app_ai_helpers_model.inc"
#include "app_ai_helpers_decode.inc"
#include "app_ai_helpers_runtime.inc"
