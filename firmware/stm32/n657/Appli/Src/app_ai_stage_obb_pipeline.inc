/* Live OBB dispatch helper for the tip-focus pipeline.
 *
 * This file is included by app_ai_runtime_tail.inc so we can keep the OBB
 * stage orchestration separate from the rest of the dry-run path without
 * changing the CubeIDE project file list.
 */

static bool AppAI_RunObbStageForTipFocusLive(
	const uint8_t *safe_frame_bytes,
	size_t frame_size,
	const AppAI_SourceCrop *full_frame_crop,
	const AppAI_SourceCrop *fixed_training_crop,
	AppAI_SourceCrop *obb_crop_out,
	AppAI_ObbBox *obb_box_out)
{
	const LL_Buffer_InfoTypeDef *obb_output_info = NULL;
	float obb_output_value = 0.0f;
	bool obb_crop_valid = false;

	if ((safe_frame_bytes == NULL) || (full_frame_crop == NULL) ||
		(fixed_training_crop == NULL) || (obb_crop_out == NULL) ||
		(obb_box_out == NULL))
	{
		return false;
	}

	*obb_crop_out = *fixed_training_crop;

#if APP_AI_ENABLE_OBB_STAGE
	{
		const bool obb_ok = AppAI_RunStageInference(
			&app_ai_obb_stage, safe_frame_bytes, frame_size,
			full_frame_crop, &obb_output_info, &obb_output_value);

		if (obb_ok &&
			AppAI_DecodeQarepvggObb(obb_output_info, obb_crop_out, obb_box_out))
		{
			obb_crop_valid = true;
			AppAI_LogObbResult(obb_output_info, obb_box_out);
			DebugConsole_Printf(
				"[AI] OBB crop: x=%lu y=%lu w=%lu h=%lu\r\n",
				(unsigned long)obb_crop_out->x_min,
				(unsigned long)obb_crop_out->y_min,
				(unsigned long)obb_crop_out->width,
				(unsigned long)obb_crop_out->height);
		}
		else
		{
			DebugConsole_WriteString(
				"[AI] OBB stage failed; using fixed training crop.\r\n");
		}
	}
#else
	(void)obb_output_info;
	(void)obb_output_value;
	DebugConsole_WriteString(
		"[AI] OBB stage disabled; using fixed training crop.\r\n");
#endif

	return obb_crop_valid;
}
