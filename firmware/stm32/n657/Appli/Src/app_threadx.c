/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    app_threadx.c
 * @author  MCD Application Team
 * @brief   ThreadX applicative file
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_threadx.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "app_camera_diagnostics.h"
#include "app_camera_config.h"
#include "app_camera_buffers.h"
#include "app_camera_capture.h"
#include "app_camera_platform.h"
#include "app_inference_runtime.h"
#include "app_storage.h"
#include "app_threadx_config.h"
#include "app_memory_budget.h"
#include "app_filex.h"
#include "app_ai.h"
#include "main.h"
#include "debug_console.h"
#include "debug_led.h"
#include "threadx_utils.h"
#include "cmw_camera.h"
#include "cmw_imx335.h"
#include "cmw_utils.h"
#include "imx335.h"
#include "imx335_reg.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#if 0
#define CAMERA_INIT_STARTUP_DELAY_MS        200U
#define BCAMS_IMX_I2C_ADDRESS_7BIT          0x1AU
#define BCAMS_IMX_I2C_ADDRESS_HAL           (BCAMS_IMX_I2C_ADDRESS_7BIT << 1U)
#define BCAMS_IMX_I2C_PROBE_TRIALS          5U
#define BCAMS_IMX_I2C_PROBE_TIMEOUT_MS      50U
#define BCAMS_IMX_POWER_SETTLE_DELAY_MS     10U
#define BCAMS_IMX_RESET_ASSERT_DELAY_MS     5U
#define BCAMS_IMX_RESET_RELEASE_DELAY_MS    10U
#define IMX335_SENSOR_WIDTH_PIXELS          2592U
#define IMX335_SENSOR_HEIGHT_LINES          1944U
/* Use the processed CMW/ISP path so AE/AWB and demosaicing can converge on a
 * usable live image. Set to 1 only if we need raw Pipe0 diagnostics. */
#define CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC 0
#define CAMERA_CAPTURE_TARGET_FRAME_COUNT   4U
/* Capture crop is expressed directly in pixels/lines. */
#define CAMERA_CAPTURE_CROP_HSTART_PIXELS   0U
#define CAMERA_CAPTURE_CROP_VSTART_LINES    0U
/* Arm one CSI line/byte counter on VC0 so we can tell whether the receiver
 * is observing line progress even when the captured payload stays all zeros. */
#define CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER      DCMIPP_CSI_COUNTER0
#define CAMERA_CAPTURE_CSI_LB_PROBE_LINE_COUNTER  1U
#define CAMERA_CAPTURE_CSI_LB_PROBE_BYTE_COUNTER  (CAMERA_CAPTURE_WIDTH_PIXELS * CAMERA_CAPTURE_BYTES_PER_PIXEL)
/* Use a centered ROI for the raw diagnostic path so we do not accidentally
 * sample a blank top-left margin from the sensor frame. */
#define CAMERA_CAPTURE_RAW_CROP_HSTART_PIXELS   ((IMX335_SENSOR_WIDTH_PIXELS - CAMERA_CAPTURE_WIDTH_PIXELS) / 2U)
#define CAMERA_CAPTURE_RAW_CROP_VSTART_LINES    ((IMX335_SENSOR_HEIGHT_LINES - CAMERA_CAPTURE_HEIGHT_PIXELS) / 2U)
/* Pipe0 raw-capture frames store one 16-bit padded pixel per sample, so the
 * preview code should read them as a 224x224 source image and upscale only the view. */
#define CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS    CAMERA_CAPTURE_WIDTH_PIXELS
#define CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES    CAMERA_CAPTURE_HEIGHT_PIXELS
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE      2U
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS   (CAMERA_CAPTURE_RAW_SOURCE_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES   (CAMERA_CAPTURE_RAW_SOURCE_HEIGHT_LINES * CAMERA_CAPTURE_RAW_BMP_PREVIEW_SCALE)
#define CAMERA_CAPTURE_RAW_BMP_PREVIEW_PIXEL_COUNT    (CAMERA_CAPTURE_RAW_BMP_PREVIEW_WIDTH_PIXELS * CAMERA_CAPTURE_RAW_BMP_PREVIEW_HEIGHT_LINES)
#define CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES  14U
#define CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES   40U
#define CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES      (256U * 4U)
#define CAMERA_CAPTURE_RAW_BMP_HEADER_BYTES       (CAMERA_CAPTURE_RAW_BMP_FILE_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_INFO_HEADER_BYTES + CAMERA_CAPTURE_RAW_BMP_PALETTE_BYTES)
/* IMX335 color-bar bring-up consistently shows four blank top lines before the
 * active test-pattern data starts, so we skip them in the raw Pipe0 view. */
#define CAMERA_CAPTURE_RAW_TOP_SKIP_LINES       4U
/* Give the ISP/AEC loop time to move the sensor away from its black-frame
 * startup state before we give up on the first saved capture. */
#define CAMERA_CAPTURE_TIMEOUT_MS           8000U
#define CAMERA_STORAGE_WAIT_TIMEOUT_MS      70000U
#define CAMERA_CAPTURE_RETRY_DELAY_MS       50U
#define CAMERA_FIRST_FRAME_WARMUP_DELAY_MS  1500U
#define CAMERA_STREAM_WARMUP_DELAY_MS       250U
#define IMX335_CAPTURE_FRAMERATE_FPS        10
#define CAMERA_CAPTURE_FILE_NAME_LENGTH     64U
#define CAMERA_STORAGE_READY_EVENT_FLAG     0x00000001U
/* Match ST's IMX335 middleware and upstream Linux driver ID check. */
#define IMX335_CHIP_ID_REG                 0x3912U
#define IMX335_CHIP_ID_VALUE               0x00U
/* IMX335 test-pattern selection.
 * -1 = disabled (live image), 0 = disabled (same as -1 in driver),
 *  1 = solid color (default color regs = 0x000 = black Ã¢â‚¬â€ all-zero pixels, NOT useful),
 * 10 = color bars (non-zero pixel values Ã¢â‚¬â€ use this to verify DMA path). */
/* Return to live optical input so the raw capture reflects the real gauge
 * scene instead of a synthetic test pattern. */
#define IMX335_TEST_PATTERN_MODE           -1

/* ST treats PIPE0 as the raw dump pipe and PIPE1 as the processed/YUV pipe.
 * Use PIPE0 only while the raw diagnostic branch is enabled. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE0
#else
#define CAMERA_CAPTURE_PIPE                 DCMIPP_PIPE1
#endif

/* Prevent accidentally using mode 1 (solid black = all-zero pixels) during
 * raw diagnostic; it is indistinguishable from a broken DMA path. */
#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC && (IMX335_TEST_PATTERN_MODE == 1)
#error "IMX335_TEST_PATTERN_MODE=1 produces all-zero pixels in raw diag mode. Use mode 10 (color bars)."
#endif
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* Dedicated ThreadX object and stack for camera connection diagnostics. */
static TX_THREAD camera_init_thread;
static ULONG camera_init_thread_stack[CAMERA_INIT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_init_thread_created = false;
static TX_THREAD camera_isp_thread;
static ULONG camera_isp_thread_stack[CAMERA_ISP_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_isp_thread_created = false;
static TX_THREAD camera_heartbeat_thread;
static ULONG camera_heartbeat_thread_stack[CAMERA_HEARTBEAT_THREAD_STACK_SIZE_BYTES
		/ sizeof(ULONG)];
static bool camera_heartbeat_thread_created = false;
static TX_MUTEX camera_capture_cmw_mutex;
static bool camera_capture_cmw_mutex_created = false;
CMW_IMX335_t camera_sensor;
bool camera_cmw_initialized = false;
/* Keep the middleware path active so the ISP/AEC pipeline can produce optical
 * frames instead of the raw sensor dump. */
bool camera_capture_use_cmw_pipeline = false;
TX_SEMAPHORE camera_capture_done_semaphore;
TX_SEMAPHORE camera_capture_isp_semaphore;
static TX_MUTEX debug_uart_mutex;
static bool camera_heartbeat_gpio_initialized = false;
static bool camera_capture_sync_created = false;
bool camera_stream_started = false;
volatile bool camera_capture_failed = false;
volatile uint32_t camera_capture_error_code = 0U;
volatile uint32_t camera_capture_byte_count = 0U;
volatile bool camera_capture_sof_seen = false;
volatile bool camera_capture_eof_seen = false;
volatile bool camera_capture_frame_done = false;
volatile bool camera_capture_snapshot_armed = false;
volatile uint32_t camera_capture_frame_event_count = 0U;
volatile uint32_t camera_capture_line_error_count = 0U;
volatile uint32_t camera_capture_line_error_mask = 0U;
volatile uint32_t camera_capture_csi_linebyte_event_count = 0U;
volatile bool camera_capture_csi_linebyte_event_logged = false;
volatile uint32_t camera_capture_vsync_event_count = 0U;
volatile uint32_t camera_capture_isp_run_count = 0U;
volatile bool camera_capture_isp_loop_paused = false;
/* Count raw IRQ entry points so we can tell whether the interrupt chain is
 * alive even when the higher-level callbacks stay silent. */
volatile uint32_t camera_capture_csi_irq_count = 0U;
volatile uint32_t camera_capture_dcmipp_irq_count = 0U;
volatile uint32_t camera_capture_reported_byte_count = 0U;
volatile uint32_t camera_capture_counter_status = (uint32_t) HAL_ERROR;

/* Reuse the CubeMX-generated camera control I2C instance from main.c. */
extern DCMIPP_HandleTypeDef hdcmipp;
extern I2C_HandleTypeDef hi2c2;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

static VOID CameraHeartbeatThread_Entry(ULONG thread_input);

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input);
static VOID CameraIspThread_Entry(ULONG thread_input);

UINT CameraPlatform_ProbeBCamsImx(void);

/**
 * @brief Register bus callbacks with the ST IMX335 driver object.
 * @retval true when the driver object is ready for sensor commands.
 */
static bool CameraPlatform_InitializeImx335Sensor(void);
bool CameraPlatform_SeedImx335ExposureGain(void);
bool CameraPlatform_EnableImx335AutoExposure(void);
void CameraPlatform_ReapplyImx335TestPattern(void);

bool CameraPlatform_PrepareDcmippSnapshot(void);
bool CameraPlatform_ConfigureCsiLineByteProbe(void);
int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);
int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length);

#if 0
/**
 * @brief Dump a dense CSI fault snapshot whenever a lane error occurs.
 * @param reason Short note describing what triggered the dump.
 * @param data_lane Failing CSI data lane from the HAL callback.
 * @param capture_dcmipp Active DCMIPP handle, when available.
 */
static void CameraPlatform_LogCsiFaultSnapshot(const char *reason,
		uint32_t data_lane, DCMIPP_HandleTypeDef *capture_dcmipp) {
	const uint32_t csi_sr0 = CSI->SR0;
	const uint32_t csi_sr1 = CSI->SR1;
	const uint32_t csi_pcr = CSI->PCR;
	const uint32_t csi_pfcr = CSI->PFCR;
	const uint32_t csi_lmcfgr = CSI->LMCFGR;
	const uint32_t csi_ier0 = CSI->IER0;
	const uint32_t csi_ier1 = CSI->IER1;
	const uint32_t csi_fcr0 = CSI->FCR0;
	const uint32_t csi_fcr1 = CSI->FCR1;
	const uint32_t csi_err1 = CSI->ERR1;
	const uint32_t csi_err2 = CSI->ERR2;
	const uint32_t csi_spdfr = CSI->SPDFR;
	const uint32_t csi_lb0cfgr = CSI->LB0CFGR;
	const uint32_t csi_lb1cfgr = CSI->LB1CFGR;
	const uint32_t csi_lb2cfgr = CSI->LB2CFGR;
	const uint32_t csi_lb3cfgr = CSI->LB3CFGR;
	const uint32_t csi_prgitr = CSI->PRGITR;
	const uint32_t hdcmipp_error =
			(capture_dcmipp != NULL) ? capture_dcmipp->ErrorCode : 0U;

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault snapshot (%s): lane=%lu hdcmipp_err=0x%08lX armed=%u stream_started=%u failed=%u sof=%u eof=%u frame_events=%lu vsync_events=%lu isp_runs=%lu csi_irqs=%lu dcmipp_irqs=%lu line_errs=%lu mask=0x%08lX linebyte_events=%lu.\r\n",
			(reason != NULL) ? reason : "capture", (unsigned long) data_lane,
			(unsigned long) hdcmipp_error,
			camera_capture_snapshot_armed ? 1U : 0U,
			camera_stream_started ? 1U : 0U, camera_capture_failed ? 1U : 0U,
			camera_capture_sof_seen ? 1U : 0U,
			camera_capture_eof_seen ? 1U : 0U,
			(unsigned long) camera_capture_frame_event_count,
			(unsigned long) camera_capture_vsync_event_count,
			(unsigned long) camera_capture_isp_run_count,
			(unsigned long) camera_capture_csi_irq_count,
			(unsigned long) camera_capture_dcmipp_irq_count,
			(unsigned long) camera_capture_line_error_count,
			(unsigned long) camera_capture_line_error_mask,
			(unsigned long) camera_capture_csi_linebyte_event_count);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault regs (%s): SR0=0x%08lX SR1=0x%08lX PCR=0x%08lX PFCR=0x%08lX LMCFGR=0x%08lX IER0=0x%08lX IER1=0x%08lX FCR0=0x%08lX FCR1=0x%08lX ERR1=0x%08lX ERR2=0x%08lX SPDFR=0x%08lX LB0CFGR=0x%08lX LB1CFGR=0x%08lX LB2CFGR=0x%08lX LB3CFGR=0x%08lX PRGITR=0x%08lX.\r\n",
			(reason != NULL) ? reason : "capture", (unsigned long) csi_sr0,
			(unsigned long) csi_sr1, (unsigned long) csi_pcr,
			(unsigned long) csi_pfcr, (unsigned long) csi_lmcfgr,
			(unsigned long) csi_ier0, (unsigned long) csi_ier1,
			(unsigned long) csi_fcr0, (unsigned long) csi_fcr1,
			(unsigned long) csi_err1, (unsigned long) csi_err2,
			(unsigned long) csi_spdfr, (unsigned long) csi_lb0cfgr,
			(unsigned long) csi_lb1cfgr, (unsigned long) csi_lb2cfgr,
			(unsigned long) csi_lb3cfgr, (unsigned long) csi_prgitr);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault decode (%s): clk(active=%u stop=%u) dl0(active=%u sync=%u stop=%u esot=%u esotsync=%u esc=%u esyncesc=%u ectrl=%u) dl1(active=%u sync=%u stop=%u esot=%u esotsync=%u esc=%u esyncesc=%u ectrl=%u) sr0(short_pkt=%u vc0state=%u crc=%u ecc=%u cecc=%u id=%u spkterr=%u wd=%u syncerr=%u lb0=%u lb1=%u lb2=%u lb3=%u).\r\n",
			(reason != NULL) ? reason : "capture",
			(csi_sr1 & CSI_SR1_ACTCLF) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPCLF) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ACTDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_SYNCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTSYNCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_EESCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESYNCESCDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ECTRLDL0F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ACTDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_SYNCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_STOPDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESOTSYNCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_EESCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ESYNCESCDL1F) ? 1U : 0U,
			(csi_sr1 & CSI_SR1_ECTRLDL1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC0STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CRCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_ECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_IDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_WDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SYNCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB2F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB3F) ? 1U : 0U);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault error bits (%s): dcmipp(axi=%u parallel_sync=%u p0_limit=%u p0_ovr=%u p1_ovr=%u p2_ovr=%u) csi(sync=%u wdg=%u spkt=%u id=%u cecc=%u ecc=%u crc=%u dphy_ctrl=%u dphy_lp_sync=%u dphy_escape=%u sot_sync=%u sot=%u).\r\n",
			(reason != NULL) ? reason : "capture",
			((hdcmipp_error & HAL_DCMIPP_ERROR_AXI_TRANSFER) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PARALLEL_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE0_LIMIT) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE0_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE1_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_ERROR_PIPE2_OVR) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_WDG) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SPKT) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DATA_ID) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_CECC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_ECC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_CRC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_CTRL) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_LP_SYNC) != 0U) ?
					1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_DPHY_ESCAPE) != 0U) ?
					1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SOT_SYNC) != 0U) ? 1U : 0U,
			((hdcmipp_error & HAL_DCMIPP_CSI_ERROR_SOT) != 0U) ? 1U : 0U);
	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI lane fault SR0 decode (%s): sof0=%u eof0=%u spkt=%u vc0state=%u vc1state=%u vc2state=%u vc3state=%u ccfifo=%u crc=%u ecc=%u cecc=%u id=%u spkterr=%u wd=%u syncerr=%u lb0=%u lb1=%u lb2=%u lb3=%u.\r\n",
			(reason != NULL) ? reason : "capture",
			(csi_sr0 & CSI_SR0_SOF0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_EOF0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC0STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC1STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC2STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_VC3STATEF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CCFIFOFF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CRCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_ECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_CECCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_IDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SPKTERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_WDERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_SYNCERRF) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB0F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB1F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB2F) ? 1U : 0U,
			(csi_sr0 & CSI_SR0_LB3F) ? 1U : 0U);

	AppCameraDiagnostics_LogDcmippPipeRegisters(reason, capture_dcmipp);
	AppCameraDiagnostics_LogCsiLineByteCounters(reason,
			camera_capture_csi_linebyte_event_count);
}
#endif


/**
 * @brief ThreadX app initialization hook.
 * @param memory_ptr ThreadX memory pool pointer.
 * @retval TX_SUCCESS on success.
 */
UINT App_ThreadX_Init(VOID *memory_ptr) {
	UINT ret = TX_SUCCESS;

	(void) memory_ptr;

	/* Defer thread creation until App_ThreadX_Start() so startup ordering is explicit. */
	DebugConsole_Printf(
			"[CAMERA][THREAD] ThreadX app init complete. Waiting to start camera thread...\r\n");
	return ret;
}

/**
 * @brief ThreadX startup hook that creates the camera and runtime threads.
 * @retval TX_SUCCESS on success.
 */
UINT App_ThreadX_Start(void) {
	/* Keep this function idempotent to protect against accidental double-start. */
	/* Leave the heartbeat LED under the dedicated thread so it reflects liveness
	 * instead of startup state. */
	BSP_LED_On(LED_RED);
	BSP_LED_Off(LED_BLUE);
	BSP_LED_Off(LED_GREEN);
	if (camera_init_thread_created && camera_isp_thread_created
			&& camera_heartbeat_thread_created) {
		DebugConsole_Printf(
				"[CAMERA][THREAD] Start skipped: camera threads already created.\r\n");
		return TX_SUCCESS;
	}

	if (!camera_capture_sync_created) {
		UINT semaphore_status = tx_semaphore_create(
				&camera_capture_done_semaphore, "camera_capture_done", 0U);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create capture semaphore, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}

		semaphore_status = tx_semaphore_create(&camera_capture_isp_semaphore,
				"camera_capture_isp", 0U);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create ISP semaphore, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}

		/* No UART mutex - concurrent prints may interleave but won't deadlock.
		 * The UART HAL transmit is not re-entrant; a mutex with TX_WAIT_FOREVER
		 * deadlocks when the holder's HAL_UART_Transmit itself blocks, and
		 * TX_NO_WAIT corrupts the UART state when two threads collide mid-frame. */
		(void) tx_mutex_create(&debug_uart_mutex, "debug_uart", TX_NO_INHERIT);

		semaphore_status = tx_mutex_create(&camera_capture_cmw_mutex,
				"camera_capture_cmw", TX_INHERIT);
		if (semaphore_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera middleware mutex, status=%lu\r\n",
					(unsigned long) semaphore_status);
			return semaphore_status;
		}
		camera_capture_cmw_mutex_created = true;

		camera_capture_sync_created = true;
	}

	{
		const UINT storage_init_status = AppStorage_Init();
		if (storage_init_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create storage-ready event flags.\r\n");
			return storage_init_status;
		}
	}

	{
		const UINT runtime_init_status = AppInferenceRuntime_Init();
		if (runtime_init_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[AI] Failed to initialize inference runtime, status=%lu\r\n",
					(unsigned long) runtime_init_status);
			return runtime_init_status;
		}
	}

	{
		const UINT runtime_start_status = AppInferenceRuntime_Start();
		if (runtime_start_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[AI] Failed to start inference runtime, status=%lu\r\n",
					(unsigned long) runtime_start_status);
			return runtime_start_status;
		}
	}

	if (!camera_isp_thread_created) {
		const UINT isp_create_status = tx_thread_create(&camera_isp_thread,
				"camera_isp", CameraIspThread_Entry, 0U,
				camera_isp_thread_stack, sizeof(camera_isp_thread_stack),
				CAMERA_ISP_THREAD_PRIORITY, CAMERA_ISP_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (isp_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera ISP thread, status=%lu\r\n",
					(unsigned long) isp_create_status);
			return isp_create_status;
		}

		camera_isp_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera ISP thread created and started.\r\n");
	}

	if (!camera_heartbeat_thread_created) {
		const UINT heartbeat_create_status = tx_thread_create(
				&camera_heartbeat_thread, "camera_heartbeat",
				CameraHeartbeatThread_Entry, 0U,
				camera_heartbeat_thread_stack,
				sizeof(camera_heartbeat_thread_stack),
				CAMERA_HEARTBEAT_THREAD_PRIORITY,
				CAMERA_HEARTBEAT_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (heartbeat_create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create heartbeat thread, status=%lu\r\n",
					(unsigned long) heartbeat_create_status);
			return heartbeat_create_status;
		}

		camera_heartbeat_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Heartbeat thread created and started.\r\n");
	}

	if (!camera_init_thread_created) {
		/* Create a dedicated thread so camera probing is isolated from other startup work. */
		const UINT create_status = tx_thread_create(&camera_init_thread,
				"camera_init", CameraInitThread_Entry, 0U,
				camera_init_thread_stack, sizeof(camera_init_thread_stack),
				CAMERA_INIT_THREAD_PRIORITY, CAMERA_INIT_THREAD_PRIORITY,
				TX_NO_TIME_SLICE, TX_AUTO_START);

		if (create_status != TX_SUCCESS) {
			DebugConsole_Printf(
					"[CAMERA][THREAD] Failed to create camera init thread, status=%lu\r\n",
					(unsigned long) create_status);
			return create_status;
		}

		camera_init_thread_created = true;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera init thread created and started.\r\n");
	}

	return TX_SUCCESS;
}

/**
 * @brief Notify the storage module that FileX media is ready.
 */
void App_ThreadX_NotifyStorageReady(void) {
	AppStorage_NotifyMediaReady();
}

/**
 * @brief Lock the shared camera middleware so only one thread touches CMW/ISP.
 * @param timeout_ticks Maximum time to wait for the mutex.
 * @retval true when the caller owns the camera middleware lock.
 */
bool App_ThreadX_LockCameraMiddleware(ULONG timeout_ticks) {
	if (!camera_capture_cmw_mutex_created) {
		return false;
	}

	return (tx_mutex_get(&camera_capture_cmw_mutex, timeout_ticks) == TX_SUCCESS);
}

/**
 * @brief Release the shared camera middleware lock.
 */
void App_ThreadX_UnlockCameraMiddleware(void) {
	if (!camera_capture_cmw_mutex_created) {
		return;
	}

	(void) tx_mutex_put(&camera_capture_cmw_mutex);
}

/**
 * @brief Kernel initialization hook used by CubeMX.
 */
void MX_ThreadX_Init(void) {
	/* USER CODE BEGIN Before_Kernel_Start */

	/* USER CODE END Before_Kernel_Start */

	tx_kernel_enter();

	/* USER CODE BEGIN Kernel_Start_Error */

	/* USER CODE END Kernel_Start_Error */
}

/**
 * @brief Print a staged diagnostic sequence for B-CAMS-IMX camera bring-up.
 * @return TX_SUCCESS when the sensor probe succeeds, TX_NOT_AVAILABLE otherwise.
 */
UINT CameraPlatform_ProbeBCamsImx(void) {
	HAL_StatusTypeDef probe_status = HAL_ERROR;
	uint8_t chip_id = 0U;

	DebugConsole_Printf("[CAMERA][PROBE] Probing camera stack...\r\n");
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: reset\r\n");
	CameraPlatform_ResetImx335Module();

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: i2c-ack\r\n");
	probe_status = HAL_I2C_IsDeviceReady(&hi2c2, BCAMS_IMX_I2C_ADDRESS_HAL,
			BCAMS_IMX_I2C_PROBE_TRIALS, BCAMS_IMX_I2C_PROBE_TIMEOUT_MS);
	if (probe_status != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Sensor did not ACK on I2C2 at 7-bit address 0x%02X.\r\n",
				(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Sensor ACKed on I2C2 at 7-bit address 0x%02X.\r\n",
			(unsigned int) BCAMS_IMX_I2C_ADDRESS_7BIT);
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: chip-id\r\n");
	if (CameraPlatform_ReadImx335ChipId(&chip_id) != HAL_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to read IMX335 ID register.\r\n");
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - IMX335 ID register 0x3912 = 0x%02X.\r\n",
			(unsigned int) chip_id);
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: sensor-init\r\n");
	if (!CameraPlatform_InitializeImx335Sensor()) {
		return TX_NOT_AVAILABLE;
	}

	DebugConsole_Printf("[CAMERA][PROBE] Sensor probe OK.\r\n");
	DebugConsole_Printf("[CAMERA][PROBE] Camera stack ready.\r\n");
	return TX_SUCCESS;
}

/**
 * @brief ThreadX entry point used to run camera bring-up diagnostics.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraInitThread_Entry(ULONG thread_input) {
	(void) thread_input;

	(void) DebugConsole_WriteString("[CAMERA] thread entry\r\n");
	DelayMilliseconds_ThreadX(CAMERA_INIT_STARTUP_DELAY_MS);
	(void) DebugConsole_WriteString("[CAMERA] probe start\r\n");
	camera_capture_isp_loop_paused = true;

	if (!App_ThreadX_LockCameraMiddleware(
			CameraPlatform_MillisecondsToTicks(
					CAMERA_MIDDLEWARE_LOCK_TIMEOUT_MS))) {
		camera_capture_isp_loop_paused = false;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Failed to lock camera middleware for probe.\r\n");
		return;
	}

	if (CameraPlatform_ProbeBCamsImx() == TX_SUCCESS) {
		App_ThreadX_UnlockCameraMiddleware();
		camera_capture_isp_loop_paused = false;
		DebugConsole_Printf(
				"[CAMERA][THREAD] Camera probe completed successfully.\r\n");

		if (!App_AI_Model_Init()) {
			DebugConsole_Printf(
					"[AI] Model runtime init failed; continuing without inference.\r\n");
		}

		BSP_LED_Off(LED_BLUE);
		DebugConsole_Printf(
				"[CAMERA][THREAD] Entering capture/inference loop (period=60s)...\r\n");
		while (1) {
			if (AppCameraCapture_CaptureAndStoreSingleFrame()) {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture and inference completed successfully.\r\n");
			} else {
				DebugConsole_Printf(
						"[CAMERA][THREAD] Capture/inference attempt failed.\r\n");
			}
			DelayMilliseconds_ThreadX(60000U);
		}
	}

	App_ThreadX_UnlockCameraMiddleware();
	camera_capture_isp_loop_paused = false;
	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera probe failed or is not configured yet.\r\n");
}

/**
 * @brief Low-priority heartbeat thread that toggles the board LED.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraHeartbeatThread_Entry(ULONG thread_input) {
	(void) thread_input;

	if (!camera_heartbeat_gpio_initialized) {
		GPIO_InitTypeDef gpio_init = { 0 };

		__HAL_RCC_GPIOG_CLK_ENABLE();
		HAL_GPIO_WritePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN, GPIO_PIN_SET);

		gpio_init.Pin = CAMERA_HEARTBEAT_LED_PIN;
		gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
		gpio_init.Pull = GPIO_NOPULL;
		gpio_init.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
		HAL_GPIO_Init(CAMERA_HEARTBEAT_LED_GPIO_PORT, &gpio_init);

		camera_heartbeat_gpio_initialized = true;
	}

	HAL_GPIO_WritePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
			CAMERA_HEARTBEAT_LED_PIN, GPIO_PIN_SET);
	DebugConsole_Printf("[WATCHDOG] heartbeat thread running.\r\n");

	while (1) {
		DebugConsole_Printf("[WATCHDOG] pulse\r\n");
		HAL_GPIO_TogglePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN);
		DelayMilliseconds_ThreadX(CAMERA_HEARTBEAT_PULSE_MS);
		HAL_GPIO_TogglePin(CAMERA_HEARTBEAT_LED_GPIO_PORT,
				CAMERA_HEARTBEAT_LED_PIN);
		DelayMilliseconds_ThreadX(CAMERA_HEARTBEAT_PERIOD_MS
				- CAMERA_HEARTBEAT_PULSE_MS);
	}
}

/**
 * @brief Low-priority camera ISP thread that keeps the middleware running.
 * @param thread_input Unused ThreadX input value.
 */
static VOID CameraIspThread_Entry(ULONG thread_input) {
	(void) thread_input;

	DebugConsole_Printf(
			"[CAMERA][THREAD] Camera ISP service thread running.\r\n");

	while (1) {
		UINT semaphore_status = tx_semaphore_get(&camera_capture_isp_semaphore,
				CameraPlatform_MillisecondsToTicks(20U));

		if ((semaphore_status == TX_SUCCESS)
				|| (camera_stream_started && camera_cmw_initialized)) {
			if (!AppCameraCapture_RunImx335Background()) {
				camera_capture_failed = true;
				camera_capture_error_code = 0x49535052U; /* 'ISPR' */
				(void) tx_semaphore_put(&camera_capture_done_semaphore);
			}
		}
	}
}

/**
 * @brief Read back the CSI PFCR after the IMX335 middleware init path.
 * @note This is a read-only sanity check so we can confirm the HAL kept the
 *       expected lane-direction and frequency-range programming intact.
 */
static void CameraPlatform_LogCsiDphySettle(void) {
	DebugConsole_Printf(
			"[CAMERA][DPHY] PFCR after app-layer check: raw=0x%08lX upper=0x%02lX LMCFGR=0x%08lX.\r\n",
			(unsigned long) CSI->PFCR,
			(unsigned long) ((CSI->PFCR >> 16U) & 0xFFUL),
			(unsigned long) CSI->LMCFGR);
}

/**
 * @brief Dump the current camera, ISP, and DCMIPP state for black-frame
 *        diagnostics.
 * @param reason Short note describing what triggered the dump.
 */

/**
 * @brief Configure the capture pipe for a 224x224 YUV422 capture sourced from RAW10 CSI input.
 * @param[out] captured_bytes_ptr Receives the final image byte count on success.
 * @retval true when a frame-complete interrupt arrives without a DCMIPP error.
 */
/**
 * @brief Service ST's IMX335 middleware background process for ISP state updates.
 * @retval true when the background step succeeded or is not used by this driver.
 */
/**
 * @brief Configure the capture pipe using ST's camera middleware crop/downsize helpers.
 * @retval true when the output path is ready for a 224x224 YUV422 frame.
 */
bool CameraPlatform_PrepareDcmippSnapshot(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (camera_capture_use_cmw_pipeline && camera_cmw_initialized) {
		CMW_DCMIPP_Conf_t pipe_request = { 0 };
		uint32_t pitch_bytes = 0U;

		pipe_request.output_width = CAMERA_CAPTURE_WIDTH_PIXELS;
		pipe_request.output_height = CAMERA_CAPTURE_HEIGHT_PIXELS;
		pipe_request.output_format =
		DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1;
		pipe_request.output_bpp = CAMERA_CAPTURE_BYTES_PER_PIXEL;
		pipe_request.enable_swap = 0;
		pipe_request.enable_gamma_conversion = 0;
		pipe_request.mode = CMW_Aspect_ratio_manual_roi;
		/* Crop a centered square ROI first so the 4:3 sensor frame does not get
		 * stretched into a square output and turn circular dials into ellipses. */
		{
			const uint32_t sensor_square_side =
					(IMX335_SENSOR_WIDTH_PIXELS < IMX335_SENSOR_HEIGHT_LINES) ?
							IMX335_SENSOR_WIDTH_PIXELS :
							IMX335_SENSOR_HEIGHT_LINES;

			pipe_request.manual_conf.width = sensor_square_side;
			pipe_request.manual_conf.height = sensor_square_side;
			pipe_request.manual_conf.offset_x =
					(IMX335_SENSOR_WIDTH_PIXELS - sensor_square_side) / 2U;
			pipe_request.manual_conf.offset_y =
					(IMX335_SENSOR_HEIGHT_LINES - sensor_square_side) / 2U;
		}

		if (CMW_CAMERA_SetPipeConfig(CAMERA_CAPTURE_PIPE, &pipe_request,
				&pitch_bytes) != CMW_ERROR_NONE) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] CMW_CAMERA_SetPipeConfig() failed for PIPE1.\r\n");
			return false;
		}

		/* CMW_CAMERA_SetPipeConfig already configures YUV conversion internally
		 * when output_format == DCMIPP_PIXEL_PACKER_FORMAT_YUV422_1. Calling
		 * the HAL YUV helpers again here resets the pipe state and causes
		 * HAL_DCMIPP_CSI_PIPE_Start to fail with HAL_ERROR. */

		return true;
	}

	{
		DCMIPP_CSI_PIPE_ConfTypeDef csi_pipe_config = { 0 };
		DCMIPP_PipeConfTypeDef pipe_config = { 0 };
		DCMIPP_CropConfTypeDef crop_config = { 0 };

		csi_pipe_config.DataTypeMode = DCMIPP_DTMODE_DTIDA;
		csi_pipe_config.DataTypeIDA = DCMIPP_DT_RAW10;
		csi_pipe_config.DataTypeIDB = 0U;
		if (HAL_DCMIPP_CSI_PIPE_SetConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &csi_pipe_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 for RAW10 input.\r\n");
			return false;
		}

		pipe_config.FrameRate = DCMIPP_FRAME_RATE_ALL;
		pipe_config.PixelPipePitch = 0U;
		pipe_config.PixelPackerFormat = 0U;
		if (HAL_DCMIPP_PIPE_SetConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &pipe_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 snapshot settings.\r\n");
			return false;
		}

		if (HAL_DCMIPP_CSI_SetVCConfig(capture_dcmipp,
		DCMIPP_VIRTUAL_CHANNEL0,
		DCMIPP_CSI_DT_BPP10) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure CSI VC0 as RAW10.\r\n");
			return false;
		}

		if (!CameraPlatform_ConfigureCsiLineByteProbe()) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to arm CSI line/byte probe for VC0.\r\n");
			return false;
		}

		/* Skip the confirmed blank/embedded prefix so the raw buffer starts on
		 * active pixels instead of the four top black lines. */
		crop_config.VStart = CAMERA_CAPTURE_RAW_TOP_SKIP_LINES;
		crop_config.HStart = 0U;
		crop_config.VSize = CAMERA_CAPTURE_HEIGHT_PIXELS;
		crop_config.HSize = CAMERA_CAPTURE_WIDTH_PIXELS;
		crop_config.PipeArea = DCMIPP_POSITIVE_AREA;
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] RAW crop at HStart=%lu VStart=%lu size=%lux%lu (skip=%lu top lines).\r\n",
				(unsigned long) crop_config.HStart,
				(unsigned long) crop_config.VStart,
				(unsigned long) crop_config.HSize,
				(unsigned long) crop_config.VSize,
				(unsigned long) CAMERA_CAPTURE_RAW_TOP_SKIP_LINES);
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Applying PIPE0 crop config.\r\n");
		if (HAL_DCMIPP_PIPE_SetCropConfig(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &crop_config) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to configure PIPE0 crop window.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 crop config applied.\r\n");
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Enabling PIPE0 crop window.\r\n");
		if (HAL_DCMIPP_PIPE_EnableCrop(capture_dcmipp,
		CAMERA_CAPTURE_PIPE) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to enable PIPE0 crop window.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 crop window enabled.\r\n");

		/* Arm the raw dump length explicitly so PIPE0 has a bounded AXI write
		 * window before the sensor starts streaming. The dump counter reports a
		 * byte count on this path, so we cap the pipe to the full capture buffer. */
		if (HAL_DCMIPP_PIPE_EnableLimitEvent(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, CAMERA_CAPTURE_BUFFER_SIZE_BYTES) != HAL_OK) {
			DebugConsole_Printf(
					"[CAMERA][CAPTURE] Failed to arm PIPE0 dump limit.\r\n");
			return false;
		}
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] PIPE0 dump limit armed: P0DCLMTR=0x%08lX.\r\n",
				(unsigned long) capture_dcmipp->Instance->P0DCLMTR);
	}

	return true;
}

/**
 * @brief Initialize the IMX335 through ST's public camera middleware path.
 * @retval true when the middleware-owned camera stack accepts the sensor setup.
 */
static bool CameraPlatform_InitializeImx335Sensor(void) {
	CMW_CameraInit_t camera_init = { 0 };
	CMW_Advanced_Config_t camera_advanced_config = { 0 };
	int32_t cmw_status = CMW_ERROR_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-defaults\r\n");
	camera_advanced_config.selected_sensor = CMW_IMX335_Sensor;
	cmw_status = CMW_CAMERA_SetDefaultSensorValues(&camera_advanced_config);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to load default IMX335 middleware values, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	camera_init.width = IMX335_SENSOR_WIDTH_PIXELS;
	camera_init.height = IMX335_SENSOR_HEIGHT_LINES;
	camera_init.fps = IMX335_CAPTURE_FRAMERATE_FPS;
	camera_init.mirror_flip = CMW_MIRRORFLIP_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-init\r\n");
	cmw_status = CMW_CAMERA_Init(&camera_init, &camera_advanced_config);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - CMW_CAMERA_Init() failed, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	CameraPlatform_LogCsiDphySettle();

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: mw-test-pattern\r\n");
	cmw_status = CMW_CAMERA_SetTestPattern(IMX335_TEST_PATTERN_MODE);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to configure IMX335 test pattern mode %d, status=%ld.\r\n",
				IMX335_TEST_PATTERN_MODE, (long) cmw_status);
		return false;
	}

#if IMX335_TEST_PATTERN_MODE >= 0
	DebugConsole_Printf("[CAMERA][PROBE] IMX335 test pattern enabled.\r\n");
#else
	DebugConsole_Printf("[CAMERA][PROBE] IMX335 live optical path enabled.\r\n");
#endif

	if (!CameraPlatform_SeedImx335ExposureGain()) {
		return false;
	}
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: seed-done\r\n");

#if CAMERA_CAPTURE_FORCE_RAW_DIAGNOSTIC
	camera_capture_use_cmw_pipeline = false;
	DebugConsole_Printf("[CAMERA][PROBE] RAW diagnostic capture enabled.\r\n");
#else
	camera_capture_use_cmw_pipeline = true;
	/* Let the ISP/AEC loop settle quickly on the first processed frame. */
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-start\r\n");
	if (!CameraPlatform_EnableImx335AutoExposure()) {
		return false;
	}
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-done\r\n");
	DebugConsole_Printf("[CAMERA][PROBE] Using CMW/ISP capture path.\r\n");
#endif

	camera_cmw_initialized = true;
	camera_stream_started = false;
	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: sensor-ready\r\n");

	return true;
}

/**
 * @brief Arm a CSI line/byte counter on VC0 so we can confirm line progress.
 * @retval true when the counter was programmed successfully.
 */
bool CameraPlatform_ConfigureCsiLineByteProbe(void) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();
	DCMIPP_CSI_LineByteCounterConfTypeDef linebyte_config = { 0 };

	if ((capture_dcmipp == NULL) || (capture_dcmipp->Instance == NULL)) {
		return false;
	}

	linebyte_config.VirtualChannel = DCMIPP_VIRTUAL_CHANNEL0;
	linebyte_config.LineCounter =
	CAMERA_CAPTURE_CSI_LB_PROBE_LINE_COUNTER;
	linebyte_config.ByteCounter =
	CAMERA_CAPTURE_CSI_LB_PROBE_BYTE_COUNTER;

	(void) HAL_DCMIPP_CSI_DisableLineByteCounter(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER);
	CSI->FCR0 = (CSI_FCR0_CLB0F | CSI_FCR0_CLB1F | CSI_FCR0_CLB2F
			| CSI_FCR0_CLB3F);

	if (HAL_DCMIPP_CSI_SetLineByteCounterConfig(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER, &linebyte_config) != HAL_OK) {
		return false;
	}

	if (HAL_DCMIPP_CSI_EnableLineByteCounter(capture_dcmipp,
	CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER) != HAL_OK) {
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] CSI line/byte probe armed on VC0 (counter=%lu line=%lu byte=%lu).\r\n",
			(unsigned long) CAMERA_CAPTURE_CSI_LB_PROBE_COUNTER,
			(unsigned long) linebyte_config.LineCounter,
			(unsigned long) linebyte_config.ByteCounter);
	return true;
}

/**
 * @brief Seed IMX335 exposure and gain with a conservative starting point.
 *
 * ST's middleware initializes the sensor conservatively. We back off the
 * previous maxed-out seed so the live optical path does not clip immediately
 * on bright scenes.
 * @retval true when the middleware accepted the seed settings.
 */
bool CameraPlatform_SeedImx335ExposureGain(void) {
	ISP_SensorInfoTypeDef sensor_info = { 0 };
	uint32_t seed_exposure_us = 0U;
	int32_t seed_gain_mdb = 0;
	int32_t cmw_status = CMW_ERROR_NONE;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: seed-start\r\n");
	cmw_status = CMW_CAMERA_GetSensorInfo(&sensor_info);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to read IMX335 sensor info for exposure seeding, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	/* Start conservatively but not at the absolute floor, so the first live
	 * frame stays out of clipping while still preserving some scene detail. */
	/* Start a little brighter than the previous seed so the first usable frame
	 * lands closer to the scene instead of hugging the dark end. */
	/* Bias the very first frame a little brighter so the processed pipeline
	 * has a better starting point before AEC takes over. */
	seed_exposure_us = sensor_info.exposure_min
			+ ((sensor_info.exposure_max - sensor_info.exposure_min) / 3U);
	if (seed_exposure_us < sensor_info.exposure_min) {
		seed_exposure_us = sensor_info.exposure_min;
	}

	seed_gain_mdb = sensor_info.gain_min;
	if (seed_gain_mdb < sensor_info.gain_min) {
		seed_gain_mdb = sensor_info.gain_min;
	}

	cmw_status = CMW_CAMERA_SetExposure((int32_t) seed_exposure_us);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to seed IMX335 exposure, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	cmw_status = CMW_CAMERA_SetGain(seed_gain_mdb);
	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to seed IMX335 gain, status=%ld.\r\n",
				(long) cmw_status);
		return false;
	}

	DebugConsole_Printf(
			"[CAMERA][PROBE]   - Seeded IMX335 exposure to %lu us and gain to %ld mdB.\r\n",
			(unsigned long) seed_exposure_us, (long) seed_gain_mdb);

	return true;
}

/**
 * @brief Force the IMX335 ISP path into auto-exposure mode.
 *
 * The IMX335 middleware bridge does not expose a sensor-level exposure-mode
 * setter, so the ISP AEC state is the control point for auto exposure here.
 * @retval true when the ISP accepted the AEC enable request.
 */
bool CameraPlatform_EnableImx335AutoExposure(void) {
	uint8_t aec_enabled = 0U;

	(void) DebugConsole_WriteString("[CAMERA][PROBE] step: ae-call\r\n");
	if (ISP_SetAECState(&camera_sensor.hIsp, 1U) != ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - Failed to enable IMX335 ISP auto exposure.\r\n");
		return false;
	}

	if (ISP_GetAECState(&camera_sensor.hIsp, &aec_enabled) == ISP_OK) {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ISP auto exposure enabled (AEC=%u).\r\n",
				(unsigned int) aec_enabled);
	} else {
		DebugConsole_Printf(
				"[CAMERA][PROBE]   - IMX335 ISP auto exposure requested, but readback failed.\r\n");
	}

	return true;
}

/**
 * @brief Re-apply the configured IMX335 test pattern after streaming starts.
 *
 * Some sensors latch the test-pattern generator more reliably once the stream
 * is already live, so we re-write the configured pattern as a low-risk
 * diagnostic nudge after start-up.
 */
void CameraPlatform_ReapplyImx335TestPattern(void) {
#if IMX335_TEST_PATTERN_MODE >= 0
	int32_t cmw_status = CMW_CAMERA_SetTestPattern(
	IMX335_TEST_PATTERN_MODE);
	uint8_t tpg_value = 0U;

	if (cmw_status != CMW_ERROR_NONE) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Warning: failed to reapply IMX335 test pattern mode %d after stream start, status=%ld.\r\n",
				IMX335_TEST_PATTERN_MODE, (long) cmw_status);
		return;
	}

	DebugConsole_Printf(
			"[CAMERA][CAPTURE] IMX335 test pattern re-applied after stream start (mode=%d).\r\n",
			IMX335_TEST_PATTERN_MODE);

	if (CameraPlatform_I2cReadReg(BCAMS_IMX_I2C_ADDRESS_HAL,
	IMX335_REG_TPG, &tpg_value, 1U) == IMX335_OK) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] IMX335 test-pattern register = 0x%02X after stream start.\r\n",
				(unsigned int) tpg_value);
	}
#endif
}


/**
 * @brief Read a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
int32_t CameraPlatform_I2cReadReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Read(&hi2c2, dev_addr, reg,
	I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief Write a 16-bit IMX335 register using the existing HAL I2C2 handle.
 * @retval IMX335_OK on success, IMX335_ERROR otherwise.
 */
int32_t CameraPlatform_I2cWriteReg(uint16_t dev_addr, uint16_t reg,
		uint8_t *pdata, uint16_t length) {
	const HAL_StatusTypeDef status = HAL_I2C_Mem_Write(&hi2c2, dev_addr, reg,
	I2C_MEMADD_SIZE_16BIT, pdata, length, 100U);
	return (status == HAL_OK) ? IMX335_OK : IMX335_ERROR;
}

/**
 * @brief Camera middleware pipe VSYNC callback used for app-side diagnostics.
 * @param pipe DCMIPP pipe that asserted VSYNC.
 * @retval CMW_ERROR_NONE always.
 */
int CMW_CAMERA_PIPE_VsyncEventCallback(uint32_t pipe) {
	if (pipe != CAMERA_CAPTURE_PIPE) {
		return CMW_ERROR_NONE;
	}

	(void) tx_semaphore_put(&camera_capture_isp_semaphore);
	camera_capture_vsync_event_count++;

	/* No DebugConsole_Printf from ISR Ã¢â‚¬â€ mutex is illegal in interrupt context. */

	return CMW_ERROR_NONE;
}

/**
 * @brief Camera middleware pipe frame callback used to release the capture thread.
 * @param pipe DCMIPP pipe that completed a frame.
 * @retval CMW_ERROR_NONE always.
 */
int CMW_CAMERA_PIPE_FrameEventCallback(uint32_t pipe) {
	uint32_t byte_count = 0U;
	HAL_StatusTypeDef counter_status = HAL_ERROR;
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (pipe != CAMERA_CAPTURE_PIPE) {
		return CMW_ERROR_NONE;
	}

	camera_capture_frame_event_count++;

	if (camera_capture_use_cmw_pipeline) {
		counter_status = HAL_OK;
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	} else if ((capture_dcmipp != NULL) && (capture_dcmipp->Instance != NULL)) {
		counter_status = HAL_DCMIPP_PIPE_GetDataCounter(capture_dcmipp,
		CAMERA_CAPTURE_PIPE, &byte_count);
	}

	camera_capture_counter_status = (uint32_t) counter_status;
	camera_capture_reported_byte_count = byte_count;

	if ((counter_status != HAL_OK) || (byte_count == 0U)) {
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	} else if (!camera_capture_use_cmw_pipeline
			&& (byte_count > CAMERA_CAPTURE_BUFFER_SIZE_BYTES)) {
		DebugConsole_Printf(
				"[CAMERA][CAPTURE] Raw pipe counter %lu exceeds the %lux%lu capture buffer; normalizing to %lu bytes for save.\r\n",
				(unsigned long) byte_count,
				(unsigned long) CAMERA_CAPTURE_WIDTH_PIXELS,
				(unsigned long) CAMERA_CAPTURE_HEIGHT_PIXELS,
				(unsigned long) CAMERA_CAPTURE_BUFFER_SIZE_BYTES);
		byte_count = CAMERA_CAPTURE_BUFFER_SIZE_BYTES;
	}

	camera_capture_byte_count = byte_count;
	camera_capture_frame_done = true;

	/* No DebugConsole_Printf from ISR Ã¢â‚¬â€ tx_mutex_get is illegal in interrupt
	 * context.  The main capture thread logs first8 after the semaphore fires. */

	(void) tx_semaphore_put(&camera_capture_done_semaphore);

	return CMW_ERROR_NONE;
}

/**
 * @brief Camera middleware pipe error callback for the snapshot path.
 * @param pipe Pipe that reported the error.
 */
void CMW_CAMERA_PIPE_ErrorCallback(uint32_t pipe) {
	DCMIPP_HandleTypeDef *capture_dcmipp =
			CameraPlatform_GetCaptureDcmippHandle();

	if (pipe != CAMERA_CAPTURE_PIPE) {
		return;
	}

	camera_capture_failed = true;
	camera_capture_error_code = capture_dcmipp->ErrorCode;
	camera_capture_snapshot_armed = false;
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief DCMIPP global error callback for CSI/common failures.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_ErrorCallback(DCMIPP_HandleTypeDef *hdcmipp) {
	if (hdcmipp == NULL) {
		return;
	}

	camera_capture_failed = true;
	camera_capture_error_code = hdcmipp->ErrorCode;
	camera_capture_snapshot_armed = false;
	/* Log from main thread after semaphore fires Ã¢â‚¬â€ no Printf from ISR. */
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief CSI callback for clock-domain FIFO overflow diagnostics.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_CSI_ClockChangerFifoFullEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp) {
	UNUSED(hdcmipp);
	camera_capture_failed = true;
	camera_capture_error_code = 0xCCF1F0U;
	(void) tx_semaphore_put(&camera_capture_done_semaphore);
}

/**
 * @brief CSI start-of-frame callback used to confirm VC0 traffic is arriving.
 * @param hdcmipp HAL DCMIPP handle.
 * @param VirtualChannel CSI virtual channel that asserted SOF.
 */
void HAL_DCMIPP_CSI_StartOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel) {
	UNUSED(hdcmipp);

	if (VirtualChannel != DCMIPP_VIRTUAL_CHANNEL0) {
		return;
	}

	if (!camera_capture_snapshot_armed) {
		return;
	}

	camera_capture_sof_seen = true;
}

/**
 * @brief CSI end-of-frame callback used as a fallback wakeup for RAW dump capture.
 * @param hdcmipp HAL DCMIPP handle.
 * @param VirtualChannel CSI virtual channel that asserted EOF.
 */
void HAL_DCMIPP_CSI_EndOfFrameEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t VirtualChannel) {
	UNUSED(hdcmipp);

	if (VirtualChannel != DCMIPP_VIRTUAL_CHANNEL0) {
		return;
	}

	if (!camera_capture_snapshot_armed) {
		return;
	}

	camera_capture_eof_seen = true;
	/* Ignore VC-level EOF as a wake source. In continuous sensor streaming it can
	 * arrive for frames that are not the armed PIPE0 snapshot yet, which would
	 * release the waiting thread with a zero byte count. */
}

/**
 * @brief CSI callback for data-lane line errors.
 * @param hdcmipp HAL DCMIPP handle.
 * @param DataLane Failing CSI data lane.
 */
void HAL_DCMIPP_CSI_LineErrorCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t DataLane) {
	if (hdcmipp == NULL) {
		return;
	}

	camera_capture_line_error_count++;
	camera_capture_line_error_mask |= (1UL << (DataLane & 0x1FU));
	if ((camera_capture_line_error_count >= 8U) && !camera_capture_sof_seen) {
		camera_capture_failed = true;
		camera_capture_error_code = 0x1E000000U | DataLane;
		(void) tx_semaphore_put(&camera_capture_done_semaphore);
	}
}

/**
 * @brief CSI callback for short-packet detection visibility.
 * @param hdcmipp HAL DCMIPP handle.
 */
void HAL_DCMIPP_CSI_ShortPacketDetectionEventCallback(
		DCMIPP_HandleTypeDef *hdcmipp) {
	if (hdcmipp == NULL) {
		return;
	}

	/* No Printf from ISR Ã¢â‚¬â€ state is read by main thread after semaphore fires. */
}

/**
 * @brief CSI callback for line/byte counter diagnostics.
 * @param hdcmipp HAL DCMIPP handle.
 * @param Counter Counter that asserted the line/byte event.
 */
void HAL_DCMIPP_CSI_LineByteEventCallback(DCMIPP_HandleTypeDef *hdcmipp,
		uint32_t Counter) {
	UNUSED(hdcmipp);
	camera_capture_csi_linebyte_event_count++;

	camera_capture_csi_linebyte_event_logged = true; /* flag for main thread */
}

/* USER CODE END 1 */


