/* This file is included by app_ai.c. */

/* USER CODE BEGIN PFP */
const LL_Buffer_InfoTypeDef *AppAI_GetInputBufferInfo(void);
const LL_Buffer_InfoTypeDef *AppAI_GetOutputBufferInfo(void);
const LL_Buffer_InfoTypeDef *AppAI_GetStageInputBufferInfo(
	const AppAI_ModelStageSpec *stage);
const LL_Buffer_InfoTypeDef *AppAI_GetStageOutputBufferInfo(
	const AppAI_ModelStageSpec *stage);
const LL_Buffer_InfoTypeDef *AppAI_FindBufferInfoByName(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *name);
const LL_Buffer_InfoTypeDef *AppAI_FindFirstBufferInfoByNames(
	const LL_Buffer_InfoTypeDef *buffer_list, const char *const *names,
	size_t name_count);
void AppAI_LogInitFailure(const char *step);
void AppAI_LogXspi2LoadFailure(const char *step, UINT fx_status,
									  int32_t bsp_status);
void AppAI_LogXspi2ProgramChunkProgress(ULONG chunk_index,
											   ULONG flash_offset, ULONG chunk_size);
void AppAI_LogXspi2FlashStatus(const char *label);
void AppAI_LogXspi2PrefixBytes(const char *label,
									  const uint8_t *bytes);
void AppAI_LogFrameSignature(const uint8_t *frame_bytes,
									size_t frame_size);
void AppAI_LogInputSignature(const float *input_buffer,
									size_t input_float_count);
void AppAI_LogInputTensorWindow(const float *input_buffer,
									   size_t input_float_count);
void AppAI_LogInputProbeSummary(const float *input_buffer,
									   size_t input_float_count);
void AppAI_LogTensorRowSamples(const char *label,
									  const float *input_buffer, size_t tensor_width, size_t row_y,
									  size_t x_min, size_t x_max);
void AppAI_LogSourcePatch(const char *label, const uint8_t *frame_bytes,
								 size_t frame_width_pixels, size_t center_x, size_t center_y,
								 size_t radius_pixels);
void AppAI_LogTensorPatch(const char *label, const float *input_buffer,
								 size_t tensor_width, size_t center_x, size_t center_y,
								 size_t radius_pixels);
void AppAI_LogSourceCropWindow(const uint8_t *frame_bytes,
									  size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
									  size_t crop_x_min, size_t crop_y_min, size_t crop_width,
									  size_t crop_height);
void AppAI_LogInt8BufferSignature(const char *label,
										 const int8_t *buffer_ptr, size_t buffer_len_bytes);
void AppAI_LogRawBufferSignature(const char *label,
										const uint8_t *buffer_ptr, size_t buffer_len_bytes);
const char *AppAI_BufferTypeName(const LL_Buffer_InfoTypeDef *buffer_info);
void AppAI_LogBufferInfoAndSignature(const char *label,
												const LL_Buffer_InfoTypeDef *buffer_info);
void AppAI_LogBufferPreview(const char *label,
									   const LL_Buffer_InfoTypeDef *buffer_info);
void AppAI_LogScalarInternalOutputProbe(
	const AppAI_ModelStageSpec *stage,
	const LL_Buffer_InfoTypeDef *stage_output_info);
#if APP_AI_USE_ADAPTIVE_GAUGE_CROP
bool AppAI_EstimateGaugeCropBoxFromYuv422(const uint8_t *frame_bytes,
												 size_t frame_size, size_t frame_width_pixels, size_t frame_height_pixels,
												 size_t *crop_x_min, size_t *crop_y_min, size_t *crop_width,
												 size_t *crop_height);
#endif
bool AppAI_LogXspi2ModelFilePrefix(FX_FILE *model_file_ptr);
bool AppAI_ReadXspi2ModelSourceProbes(FX_FILE *model_file_ptr,
											 ULONG file_size, uint8_t *source_prefix, uint8_t *source_tail,
											 bool *has_tail_out);
void AppAI_LogXspi2FlashPrefix(void);
void AppAI_LogXspi2MappedScaleBytes(void);
void AppAI_LogXspi2IndirectAndMappedPrefix(void);
void AppAI_LogFloatApprox(const char *label, float value);
float AppAI_TraceAndApplyInferenceCalibration(float raw_value);
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
void AppAI_ResetInferenceBurstHistory(void);
#endif
bool AppAI_IsFiniteFloat(float value);
bool AppAI_IsPlausibleInferenceValue(float value);
#if APP_AI_ENABLE_INFERENCE_BURST_SMOOTHING
float AppAI_FilterInferenceValue(float value);
#endif
void AppAI_LogInferenceResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info);
void AppAI_LogRectifierResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_RectifierBox *rectifier_box);
void AppAI_LogObbResult(
	const LL_Buffer_InfoTypeDef *output_buffer_info,
	const AppAI_ObbBox *obb_box);
int AppAI_ApplyCacheRange(uint32_t start_addr, uint32_t end_addr,
								 bool clean, bool invalidate);
uint32_t AppAI_GrayToFloatBits(uint8_t gray);
bool AppAI_TipFocus_DecodeSimccAxis(const int8_t *axis, int32_t size,
	float *coord_out, float *peak_out, float *spread_px_out);
void AppAI_TipFocus_ResetMedianHistory(float seed_value);
float AppAI_TipFocus_MedianFilter(float new_value);
void AppAI_BlankTipFocusLowerInset(int8_t *input, size_t width,
	size_t height);
void AppAI_FormatFixedFloat(char *dst, size_t dst_len, float value,
	unsigned decimals);

bool __attribute__((noinline)) AppAI_PreprocessScalarRow(
	const uint8_t *frame_bytes, size_t frame_size, size_t source_width,
	size_t source_height, size_t crop_x_min, size_t crop_y_min,
	size_t crop_width, size_t crop_height, size_t output_width,
	size_t output_height, float resize_scale, size_t resized_width,
	size_t resized_height, size_t resize_pad_x, size_t resize_pad_y,
	size_t out_y, float *input_ptr, size_t input_len_bytes);
void AppAI_EnableNpuMemoryAndCaches(void);
void AppAI_ConfigureNpuAccessControl(void);
void AppAI_ConfigureNpuRisafDefaults(void);
bool AppAI_EnsureStageRuntimeReady(const AppAI_ModelStageSpec *stage);
bool AppAI_EnsureXspi2ModelImageReadyForStage(
	const AppAI_ModelStageSpec *stage);
bool AppAI_WaitForFileXMediaReady(uint32_t timeout_ms);
bool AppAI_RuntimeInitStepwise(void);
bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToFloatInput(
	const uint8_t *frame_bytes, size_t frame_size, float *input_buffer,
	size_t input_float_count, size_t input_len_bytes,
	size_t output_width, size_t output_height);
bool __attribute__((noinline)) AppAI_PreprocessYuv422FrameToInt8Input(
	const uint8_t *frame_bytes, size_t frame_size, uint8_t *input_ptr,
	size_t input_len_bytes, const LL_Buffer_InfoTypeDef *input_info);
void AppAI_ReadRgbFromYuv422Bilinear(const uint8_t *frame_bytes,
	size_t frame_size_bytes, size_t frame_width_pixels, size_t frame_height_pixels,
	float source_x, float source_y, float *r_out, float *g_out, float *b_out);
void AppAI_ReadRgbFromRgbBilinear(const uint8_t *rgb_bytes,
	size_t frame_width_pixels, size_t frame_height_pixels,
	float source_x, float source_y, float *r_out, float *g_out, float *b_out);
float AppAI_ClampNormalizedFloat(float value);


