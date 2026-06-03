#!/usr/bin/env python3
"""Offline board pipeline: OBB → centre CNN → polar vote, matching firmware math.

Saves debug visualisations alongside each capture so you can verify the
centre prediction is correct before flashing.

Usage:
  poetry run python scripts/run_ai_pipeline.py \\
      artifacts/adaptive_crop_batch/capture_*/capture_*/capture_preview.png \\
      --debug-dir tmp/ai_pipeline_debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    _quantize_input,
    decode_obb_crop_box,
)
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    estimate_rim_center,
    needle_angle_from_polar_vote,
    refine_center,
    score_polar_quality,
    polar_spoke_vote,
    rgb_to_luma,
)

ARTIFACTS = PROJECT_ROOT / "artifacts" / "deployment"
OBB_PATH = ARTIFACTS / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
CTR_PATH = ARTIFACTS / "center_model_v3_obbft_int8" / "model_int8.tflite"

INPUT_SIZE = 224
TC_W = 155
TC_H = 123
MIN_ANGLE_DEG = 135.0
SWEEP_DEG = 270.0
TEMP_MIN = -30.0
TEMP_MAX = 50.0


def _angle_to_temp(angle_deg: float) -> float:
    shifted = angle_deg - MIN_ANGLE_DEG
    while shifted < 0.0:
        shifted += 360.0
    while shifted >= 360.0:
        shifted -= 360.0
    fraction = np.clip(shifted / SWEEP_DEG, 0.0, 1.0)
    return TEMP_MIN + fraction * (TEMP_MAX - TEMP_MIN)


def _resize_with_pad_geometry(crop_w: int, crop_h: int):
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


def _crop_center_to_frame(
    cx_norm: float,
    cy_norm: float,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
):
    scale, pad_x, pad_y = _resize_with_pad_geometry(crop_w, crop_h)
    padded_cx = cx_norm * INPUT_SIZE
    padded_cy = cy_norm * INPUT_SIZE
    ff_cx = crop_x + (padded_cx - pad_x) / scale
    ff_cy = crop_y + (padded_cy - pad_y) / scale
    return ff_cx, ff_cy


def _run_tflite_raw(interpreter: tf.lite.Interpreter, batch_float: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    q_batch = _quantize_input(batch_float, inp)
    interpreter.set_tensor(int(inp["index"]), q_batch)
    interpreter.invoke()
    q_out = interpreter.get_tensor(int(out["index"]))[0]
    scale = float(out["quantization"][0])
    zp = int(out["quantization"][1])
    return scale * (np.asarray(q_out, dtype=np.float32) - zp)


def _draw_debug_img(
    frame: np.ndarray,
    cd_x: int, cd_y: int,
    ctr_cx_norm: float, ctr_cy_norm: float,
    ff_cx: float, ff_cy: float,
    rim_cx: float, rim_cy: float, rim_ok: bool,
    polar_canvas: np.ndarray | None,
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    # CD crop rectangle
    cv2.rectangle(vis, (cd_x, cd_y), (cd_x + TC_W, cd_y + TC_H), (0, 255, 255), 1)
    # CNN centre
    cv2.circle(vis, (int(round(ff_cx)), int(round(ff_cy))), 3, (0, 0, 255), -1)
    cv2.putText(vis, "CNN", (int(round(ff_cx)) + 4, int(round(ff_cy)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # Rim centre
    if rim_ok:
        cv2.circle(vis, (int(round(rim_cx)), int(round(rim_cy))), 3, (0, 255, 0), -1)
        cv2.putText(vis, "RIM", (int(round(rim_cx)) + 4, int(round(rim_cy)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--ctr-model", type=Path, default=CTR_PATH)
    parser.add_argument("--obb-model", type=Path, default=OBB_PATH)
    parser.add_argument("--debug-dir", type=Path)
    args = parser.parse_args()

    print(f"OBB:  {args.obb_model}")
    print(f"CTR:  {args.ctr_model}")
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug: {args.debug_dir}")

    obb_interp = tf.lite.Interpreter(model_path=str(args.obb_model))
    obb_interp.allocate_tensors()
    ctr_interp = tf.lite.Interpreter(model_path=str(args.ctr_model))
    ctr_interp.allocate_tensors()

    cnns = []
    rims = []
    for img_path in args.images:
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

        # -- OBB --
        obb_batch = (frame.astype(np.float32) / 255.0)[None, ...]
        obb_out = _run_tflite_raw(obb_interp, obb_batch)
        obb_params = obb_out.reshape(-1)
        dec = decode_obb_crop_box(
            obb_params.astype(np.float32),
            source_width=INPUT_SIZE,
            source_height=INPUT_SIZE,
        )
        if not dec.accepted:
            print(f"{img_path.name}: OBB rejected ({dec.fallback_reason})")
            continue

        # -- Centre-detector crop --
        obb_cx = float(obb_params[0]) * INPUT_SIZE
        obb_cy = float(obb_params[1]) * INPUT_SIZE
        cd_x = int(round(obb_cx - TC_W / 2.0))
        cd_y = int(round(obb_cy - TC_H / 2.0))
        cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
        cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))
        cd_crop = frame[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]

        scale, pad_x, pad_y = _resize_with_pad_geometry(TC_W, TC_H)
        cd_resized_w = int(round(TC_W * scale))
        cd_resized_h = int(round(TC_H * scale))
        cd_resized = cv2.resize(cd_crop, (cd_resized_w, cd_resized_h),
                                interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        x_off = int(round(pad_x))
        y_off = int(round(pad_y))
        canvas[y_off:y_off + cd_resized_h, x_off:x_off + cd_resized_w] = cd_resized

        # -- CNN centre --
        ctr_batch = (canvas.astype(np.float32))[None, ...]
        ctr_out = _run_tflite_raw(ctr_interp, ctr_batch)
        cx_norm, cy_norm = float(ctr_out[0]), float(ctr_out[1])
        ff_cx, ff_cy = _crop_center_to_frame(cx_norm, cy_norm, cd_x, cd_y, TC_W, TC_H)
        dial_radius = float(obb_params[2]) * INPUT_SIZE * 0.5

        # -- Polar vote: try multiple centre seeds, refine only the best --
        luma = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        rim_cx, rim_cy, rim_ok = estimate_rim_center(luma, INPUT_SIZE, INPUT_SIZE, dial_radius)
        rim_angle = rim_temp = float("nan")

        seeds = [(float(ff_cx), float(ff_cy), "CNN"),
                 (float(obb_cx), float(obb_cy), "OBB")]
        if rim_ok:
            seeds.append((rim_cx, rim_cy, "RIM"))

        # Quick quality check on raw seeds (no refinement)
        best_q = -1.0
        best_sx = best_sy = 0.0
        best_slabel = ""
        for sx, sy, slabel in seeds:
            votes = polar_spoke_vote(luma, sx, sy, dial_radius,
                                     edge_threshold=8.0, use_structural_boost=True)
            q = score_polar_quality(votes, luma, sx, sy, dial_radius)
            if q > best_q:
                best_q, best_sx, best_sy, best_slabel = q, sx, sy, slabel

        # Refine only the best seed
        angle = temp = 0.0
        if best_q > 0.0:
            _, _, angle, _ = refine_center(
                frame, best_sx, best_sy, dial_radius,
                edge_threshold=8.0, use_structural_boost=True,
            )
            temp = _angle_to_temp(angle)
        else:
            # Fallthrough: pick any seed and use simple peak
            sx, sy, slabel = seeds[0]
            votes = polar_spoke_vote(luma, sx, sy, dial_radius,
                                     edge_threshold=8.0, use_structural_boost=True)
            from embedded_gauge_reading_tinyml.hybrid_localizer import smooth_and_find_peak
            angle, _, _ = smooth_and_find_peak(votes)
            temp = _angle_to_temp(angle)

        if rim_ok:
            rim_angle = needle_angle_from_polar_vote(frame, rim_cx, rim_cy, dial_radius,
                                                     refine=True, use_structural_boost=True)
            rim_temp = _angle_to_temp(rim_angle)

        cnns.append((ff_cx, ff_cy, angle, temp))
        rims.append((rim_cx, rim_cy, rim_angle, rim_temp) if rim_ok else None)

        print(f"{img_path.name}:")
        print(f"  CD crop: ({cd_x},{cd_y}) {TC_W}x{TC_H}")
        print(f"  CNN centre: ({cx_norm:.3f},{cy_norm:.3f}) -> frame ({ff_cx:.1f},{ff_cy:.1f})")
        print(f"  AI:    angle={angle:.1f} deg temp={temp:.1f} C")
        if rim_ok:
            print(f"  Rim:   centre=({rim_cx:.0f},{rim_cy:.0f}) angle={rim_angle:.1f} deg temp={rim_temp:.1f} C")
        else:
            print(f"  Rim:   not detected")
        print()

        # Save debug image
        if args.debug_dir:
            vis = _draw_debug_img(frame, cd_x, cd_y, cx_norm, cy_norm, ff_cx, ff_cy,
                                  rim_cx, rim_cy, rim_ok, None)
            debug_path = args.debug_dir / f"{img_path.parent.parent.name}_debug.png"
            cv2.imwrite(str(debug_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            # Also save the canvas (model input)
            canvas_path = args.debug_dir / f"{img_path.parent.parent.name}_model_input.png"
            cv2.imwrite(str(canvas_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Summary
    print("=" * 60)
    if cnns:
        cx_arr = np.array([c[0] for c in cnns])
        cy_arr = np.array([c[1] for c in cnns])
        ang_arr = np.array([c[2] for c in cnns])
        tmp_arr = np.array([c[3] for c in cnns])
        print(f"CNN centre: cx={cx_arr.mean():.1f}+-{cx_arr.std():.1f}  "
              f"cy={cy_arr.mean():.1f}+-{cy_arr.std():.1f}")
        print(f"CNN angle:  {ang_arr.mean():.1f}+-{ang_arr.std():.1f} deg")
        print(f"CNN temp:   {tmp_arr.mean():.1f}+-{tmp_arr.std():.1f} C")
    valid_rims = [r for r in rims if r is not None]
    if valid_rims:
        rcx = np.array([r[0] for r in valid_rims])
        rcy = np.array([r[1] for r in valid_rims])
        rang = np.array([r[2] for r in valid_rims])
        rtmp = np.array([r[3] for r in valid_rims])
        print(f"Rim centre: cx={rcx.mean():.1f}+-{rcx.std():.1f}  "
              f"cy={rcy.mean():.1f}+-{rcy.std():.1f}")
        print(f"Rim angle:  {rang.mean():.1f}+-{rang.std():.1f} deg")
        print(f"Rim temp:   {rtmp.mean():.1f}+-{rtmp.std():.1f} C")


if __name__ == "__main__":
    main()
