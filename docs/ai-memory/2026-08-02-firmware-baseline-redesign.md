# Firmware baseline redesign and console data-plane isolation

Date: 2026-08-02
Status: Source integration complete; Windows CubeIDE Debug build passes; board reflashing and live validation remain pending because the earlier FSBL flash verification failed.

## Scope

This note records the firmware baseline after the IMX335/AI pipeline cleanup. The product image has no HyperRAM and uses the deployment pair:

- Ellipse: `ellipse_iter8_universal_wide_deep_int8_n6_npu`, 384x384 int8 grayscale, xSPI2 slot `0x70400000`.
- Keypoint: `keypoint_unet_224g_wide_aug_int8_n6_npu`, 224x224 int8 grayscale, 56x56x2 int8 heatmap, xSPI2 slot `0x70800000`.

## Module map

- `app_threadx.c`: ThreadX startup and thread ownership. Starts the AI worker and the optional baseline diagnostic worker.
- `app_camera_platform.c`: IMX335/CMW/ISP setup, DCMIPP pipe selection, AE/AWB control, and stream lifecycle.
- `app_camera_capture.c`: One capture transaction. It waits for outstanding consumers, accepts a completed MONO_Y8 frame, saves it, and publishes one private snapshot.
- `app_camera_buffers.c`: DMA buffer ownership, cache/MPU boundary, snapshot copy, and compact frame signatures. The snapshot copy is UART-silent.
- `app_inference_runtime.c`: AI queue/worker ownership. The worker reads only the immutable private snapshot and releases the in-flight flag after both model stages finish.
- `app_ai_stage_tip_focus.c` plus `app_ai_runtime_tail.inc`: Generated model handles and the active ellipse-to-keypoint coordinator. The coordinator logs tensor signatures and writes optional exact tensors to FileX/SD, never to UART.
- `app_ai_aton_cache.c`: Shared MCU/NPU cache callbacks for every relocatable ATON network. The retired tip-focus wrapper no longer owns production runtime services.
- `app_baseline_runtime.c` and `app_baseline_hough.c`: Optional classical-CV baseline comparator. It consumes the same immutable MONO_Y8 snapshot after AI, never the live DMA buffer, and is diagnostic only.
- `debug_console.c`: Sole application UART transmit boundary. It rejects oversized payloads, camera-buffer ranges, and all writes while the snapshot copy is active.
- `isp_conf.h`, `isp_tool_com_vendor.c`, and `isp_cmd_parser_vendor.c`: Product safety boundary. ST's optional ISP tuning host channel and its preview/ISP/raw-frame dump parser are excluded from the product image.

## Frame ownership contract

`DCMIPP DMA -> completed capture buffer -> FileX save -> one private 640x640 snapshot -> AI and baseline read-only consumers -> release -> next capture`

No worker is allowed to retain or queue a live DMA pointer. The baseline request API accepts only `camera_inference_frame_snapshot`, and the camera thread waits for both AI and baseline consumers before starting another capture.

## ASCII leak finding

The ST ISP tuning source contains `ISP_ToolCom_SendData()`, which calls `usbx_write(buffer, buffer_size)`. The command parser routes preview, ISP, and raw-frame dump commands to that function. Those commands can send a complete camera frame as binary bytes, which appear as ASCII when viewed in the human UART terminal.

The product wrappers now compile that layer only when an explicit `APP_ISP_TUNING_IMAGE` build define is present, and `isp_conf.h` undefines the inherited tuning macro. The rebuilt application ELF has no `usbx_write`, `ISP_ToolCom`, or `ISP_CmdParser_SendDumpData` symbols. The boot log includes `firmware=2026-08-02-baseline-redesign-console-safe`; if that marker is absent, the board is running an older image and source changes cannot affect its UART output.

The rebuilt ELF also does not contain the former `snapshot-copy progress +64KiB` breadcrumb. Seeing that line followed by camera bytes on the board is therefore a stale-image signature, not a current snapshot-copy behavior. `flash_boot.ps1` now refuses to sign or flash an application unless the new boot marker is present and that removed marker is absent, and it prints the application SHA-256 before programming.

The retired v18/112 compatibility wrapper remains in the source tree only for
explicit replay builds and is disabled by default with
`APP_AI_ENABLE_LEGACY_TIP_FOCUS_COMPAT=0`. This keeps the board's production
model pair unambiguous while preserving a deliberate compatibility switch.

## Model parity contract

The active board path uses the same quantization convention as the packaged int8 artifacts: `q = round(gray * 255) - 128`, clamped to int8. The ellipse input is a bilinear 640-to-384 grayscale resize. The decoded ellipse produces a 1.35x square crop; its integer crop bounds are clamped before the 224x224 pixel-center bilinear resize. The keypoint decoder consumes interleaved 56x56x2 output, dequantizes with scale `1/256` and zero point `-128`, then applies the offline fourth-power heatmap weighting and north-zero angle conversion.

Each live stage logs first/middle/last/min/max/sum plus an FNV-1a tensor hash. The first eight exact keypoint input/output tensors can be written to FileX/SD for WSL replay. This makes an offline-vs-board discrepancy separable into: captured frame, ellipse tensor, ellipse result/crop, keypoint tensor, keypoint output, or final calibration.

The WSL replay script was corrected to match the board contract as well: it now uses `keypoint_unet_224g_wide_aug` with a 56x56 output, endpoint-mapped 640-to-384 ellipse preprocessing, the board's 1.35x integer crop, and signed north-zero `atan2(dx, -dy)` calibration. The previous script still described the retired stride-2/112x112 model and used letterbox/east-zero semantics, so its results were not a valid board comparison.

## Memory evidence

From `Appli/Debug/n657_Appli.map` after the final build:

- AI noncacheable input shadow: `0x24160000..0x2416c3ff`.
- Camera DMA buffers: `0x2416c400..0x241d03ff`.
- Ellipse/keypoint activation sections: `0x24000000..0x24024bff`.
- Snapshot: `0x24080000..0x240e3fff`.
- `NPU_SRAM6` is empty in the final image; no AI worker stack is placed in the model activation window.

These ranges do not overlap. The snapshot is aligned to its own MPU region and remains outside the NPU activation data.

## Verification

The final Windows STM32CubeIDE Debug build completed with `0 errors, 0 warnings` and produced `firmware/stm32/n657/Appli/Debug/n657_Appli.bin`. The flash script points to the ellipse and keypoint raw packages at `0x70400000` and `0x70800000`, checks slot ranges/overlap, signs the FSBL and application, and verifies each CubeProgrammer write. Live board validation must begin by confirming the new boot marker, then comparing the logged tensor hashes against the corresponding SD/WSL replay.
