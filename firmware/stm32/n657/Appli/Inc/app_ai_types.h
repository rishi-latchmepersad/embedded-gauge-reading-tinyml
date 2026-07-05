/**
 * @file    app_ai_types.h
 * @brief   Shared type definitions for the AI pipeline stage system.
 *
 * Every model stage (OBB localizer, tip-focus UNet, centre-detector, etc.)
 * uses these types so that the pipeline orchestrator can wire them together
 * without knowing compile-time model details.
 *
 * To swap a model out: update the generated wrapper for that stage, point
 * its stage spec at the new nn_instance / init functions, and the pipeline
 * picks it up automatically.
 */

#ifndef __APP_AI_TYPES_H
#define __APP_AI_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* ATON / ST types — include order matters.                           */
/* ll_aton_rt_user_api.h must be included BEFORE this header in every */
/* .c file so NN_Instance_TypeDef and LL_Buffer_InfoTypeDef resolve.  */
/* ------------------------------------------------------------------ */
#include "ll_aton_rt_user_api.h"
#include "ll_aton.h"

/* ------------------------------------------------------------------ */
/* Stage specification                                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Descriptor for one model stage in the pipeline.
 *
 * Each compiled-in model gets one static instance of this struct.
 * The pipeline init code walks the configured stage list and calls
 * network_init_fn / inference_init_fn / prepares xSPI2 for each.
 */
typedef struct AppAI_ModelStageSpec AppAI_ModelStageSpec;

struct AppAI_ModelStageSpec
{
	const char *stage_label;         /* Human-readable name for logging */
	const char *model_image_path;    /* SD card path (xSPI2 flash source) */
	NN_Instance_TypeDef *nn_instance;/* LL_ATON NN instance handle */
	bool (*network_init_fn)(void);   /* Platform-level network init */
	bool (*inference_init_fn)(void); /* Per-stage inference init */
	bool uses_rectifier_box;         /* Set true for rectifier stages */
	uint32_t xspi2_chip_offset;      /* Byte offset from chip base 0x70000000 */
	uint32_t xspi2_base_addr;        /* Mapped window address for weight blob */
};

/* ------------------------------------------------------------------ */
/* Geometry / crop types                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Oriented bounding box output from the OBB localizer model.
 *
 * centre_x / centre_y are normalised [0,1] within the source frame.
 * gauge_centre_x / gauge_centre_y hold the needle-pivot estimate
 * from the QARepVGG-Pro heatmap (normalised [0,1]).  Set to -1.0
 * when the model does not provide a valid centre.
 */
typedef struct
{
	float center_x;
	float center_y;
	float box_w;
	float box_h;
	float angle_rad;         /* box rotation (radians) */
	float confidence;        /* model confidence [0,1] */
	float gauge_center_x;    /* needle-pivot x (normalised [0,1]) */
	float gauge_center_y;    /* needle-pivot y (normalised [0,1]) */
} AppAI_ObbBox;

/**
 * @brief Axis-aligned crop rectangle in source pixel coordinates.
 */
typedef struct
{
	size_t x_min;
	size_t y_min;
	size_t width;
	size_t height;
} AppAI_SourceCrop;

/* ------------------------------------------------------------------ */
/* OBB decode internals                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Intermediate OBB decode result before geometry validation.
 *
 * One candidate is built from corner-style bbox output; another from
 * the centre/size-style output.  The decoder picks the first plausible
 * result and discards outliers.
 */
typedef struct
{
	bool valid;              /* whether fields below are finite / in-range */
	float x_min;
	float y_min;
	float x_max;
	float y_max;
	float center_x;
	float center_y;
	float box_w;
	float box_h;
	AppAI_SourceCrop crop;   /* quantised integer crop for downstream stages */
} AppAI_ObbDecodeCandidate;

/* ------------------------------------------------------------------ */
/* Legacy rectifier box                                               */
/* ------------------------------------------------------------------ */

typedef struct
{
	float center_x;
	float center_y;
	float box_w;
	float box_h;
} AppAI_RectifierBox;

#endif /* __APP_AI_TYPES_H */
