"""Replay the current STM32 board inference pipeline on laptop captures.

This module mirrors the firmware path closely enough for parity debugging:
- load a raw board capture or regular RGB image,
- run the OBB localizer on the full frame,
- decode the OBB crop with the same training-window guard as firmware,
- fall back to the rectifier when the OBB crop is implausible,
- run the scalar reader on the selected crop, and
- apply the same affine calibration plus 3-frame burst smoothing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import math
from typing import Any, Callable, Final, Literal, cast
import zlib

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import tensorflow as tf

from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    load_yuv422_capture_as_rgb,
)
from embedded_gauge_reading_tinyml.geometry_cascade import source_xy_from_resized_xy

RGBImage = NDArray[np.uint8]
ModelKind = Literal["auto", "keras", "tflite"]

ML_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
REPO_ROOT: Final[Path] = ML_ROOT.parent

DEFAULT_IMAGE_SIZE: Final[int] = 224
CAPTURE_YUV422_SUFFIX: Final[str] = ".yuv422"

TRAINING_CROP_X_MIN_RATIO: Final[float] = 0.1027
TRAINING_CROP_Y_MIN_RATIO: Final[float] = 0.2573
TRAINING_CROP_X_MAX_RATIO: Final[float] = 0.7987
TRAINING_CROP_Y_MAX_RATIO: Final[float] = 0.8071

OBB_CROP_SCALE: Final[float] = 1.20
OBB_MIN_BOX_RATIO: Final[float] = 0.05
OBB_TRAINING_CROP_MIN_RATIO: Final[float] = 0.60
OBB_TRAINING_CROP_MAX_RATIO: Final[float] = 1.40
OBB_MIN_CROP_SIZE_PIXELS: Final[float] = 48.0

RECTIFIER_MIN_BOX_RATIO: Final[float] = 0.05
RECTIFIER_FIXED_SCALE_CROP: Final[bool] = True
RECTIFIER_CENTER_BLEND_NUMERATOR: Final[int] = 1
RECTIFIER_CENTER_BLEND_DENOMINATOR: Final[int] = 5
RECTIFIER_CENTER_MIN_RATIO: Final[float] = 0.10
RECTIFIER_CENTER_MAX_RATIO: Final[float] = 0.90
RECTIFIER_CROP_SCALE: Final[float] = 1.80

INFERENCE_BURST_HISTORY_SIZE: Final[int] = 3
INFERENCE_BURST_RESET_DELTA_C: Final[float] = 12.0

CALIBRATION_AFFINE_SCALE: Final[float] = 1.1630995273590088
CALIBRATION_AFFINE_BIAS: Final[float] = 0.7423046231269836

DEFAULT_OBB_MODEL: Final[Path] = (
    ML_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
)
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    ML_ROOT
    / "artifacts"
    / "deployment"
    / "mobilenetv2_rectifier_hardcase_finetune_v3_int8"
    / "model_int8.tflite"
)
DEFAULT_SCALAR_MODEL: Final[Path] = (
    ML_ROOT
    / "artifacts"
    / "deployment"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "model_int8.tflite"
)


@dataclass(frozen=True, slots=True)
class TensorProbe:
    """A compact, firmware-style summary of a tensor buffer."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    crc32_hex: str
    first8: tuple[int, ...]
    mid8: tuple[int, ...]
    last8: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelSession:
    """Loaded model backend plus the tensor metadata needed to run it."""

    kind: Literal["keras", "tflite"]
    model: tf.keras.Model | tf.lite.Interpreter
    input_details: dict[str, Any] | None = None
    output_details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CropDecodeDecision:
    """One crop decode decision, including whether the firmware would fallback."""

    stage: Literal["obb", "rectifier"]
    accepted: bool
    crop_box_xyxy: tuple[float, float, float, float]
    fallback_reason: str | None
    details: dict[str, float]


@dataclass(frozen=True, slots=True)
class BoardPipelineResult:
    """Complete replay of the board pipeline for one capture."""

    capture_path: str
    source_kind: str
    source_shape: tuple[int, int, int]
    full_frame_probe: TensorProbe
    obb_output_probe: TensorProbe
    obb_decision: CropDecodeDecision
    rectifier_output_probe: TensorProbe | None
    rectifier_decision: CropDecodeDecision | None
    selected_stage: Literal["obb", "rectifier"]
    selected_crop_box_xyxy: tuple[float, float, float, float]
    scalar_input_probe: TensorProbe
    scalar_output_probe: TensorProbe
    raw_prediction: float
    calibrated_prediction: float
    reported_prediction: float
    burst_history_count: int
    burst_history_reset: bool


def _jsonable(result: BoardPipelineResult) -> dict[str, Any]:
    """Convert a pipeline result into JSON-friendly data."""
    return asdict(result)


def _normalize_output_map(outputs: Any) -> dict[str, NDArray[np.float32]]:
    """Convert Keras outputs into a stable name -> tensor mapping."""
    if isinstance(outputs, dict):
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in outputs.items()
        }
    if isinstance(outputs, (list, tuple)):
        return {
            f"output_{index}": np.asarray(value, dtype=np.float32)
            for index, value in enumerate(outputs)
        }
    return {"output": np.asarray(outputs, dtype=np.float32)}


def _extract_tensor(
    outputs: Any,
    *,
    preferred_keys: tuple[str, ...] = (),
) -> tuple[np.ndarray, str]:
    """Select the first usable tensor from a model output bundle."""
    output_map = _normalize_output_map(outputs)
    for key in preferred_keys:
        tensor = output_map.get(key)
        if tensor is not None:
            return tensor, key
    first_key = next(iter(output_map))
    return output_map[first_key], first_key


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to a TFLite int8 input tensor."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_tensor(tensor: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    """Convert a quantized tensor back into float32 values."""
    scale = float(details["quantization"][0])
    zero_point = int(details["quantization"][1])
    return scale * (np.asarray(tensor, dtype=np.float32) - zero_point)


def _run_session(
    session: ModelSession,
    batch: np.ndarray,
    *,
    preferred_output_keys: tuple[str, ...] = (),
) -> tuple[np.ndarray, str]:
    """Run one model backend and return the first usable output tensor."""
    if session.kind == "keras":
        model = cast(tf.keras.Model, session.model)
        outputs = model.predict(batch, verbose=0)
        return _extract_tensor(outputs, preferred_keys=preferred_output_keys)

    interpreter = cast(tf.lite.Interpreter, session.model)
    if session.input_details is None or session.output_details is None:
        raise ValueError("TFLite session is missing tensor metadata.")
    quantized_batch = _quantize_input(batch, session.input_details)
    interpreter.set_tensor(int(session.input_details["index"]), quantized_batch)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(int(session.output_details["index"]))[0]
    return _dequantize_tensor(output_tensor, session.output_details), str(
        session.output_details.get("name", "output")
    )


def _probe_tensor(name: str, tensor: np.ndarray) -> TensorProbe:
    """Create a small firmware-style fingerprint for a tensor buffer."""
    array = np.ascontiguousarray(np.asarray(tensor))
    raw = array.tobytes()
    raw_bytes = np.frombuffer(raw, dtype=np.uint8)
    byte_length = int(raw_bytes.size)
    first8 = tuple(int(x) for x in raw_bytes[:8].tolist())
    mid_start = max(0, (byte_length // 2) - 4)
    mid8 = tuple(int(x) for x in raw_bytes[mid_start : mid_start + 8].tolist())
    last8 = tuple(int(x) for x in raw_bytes[-8:].tolist()) if byte_length >= 8 else first8
    crc32_hex = f"0x{zlib.crc32(raw) & 0xFFFFFFFF:08X}"
    return TensorProbe(
        name=name,
        dtype=str(array.dtype),
        shape=tuple(int(dim) for dim in array.shape),
        byte_length=byte_length,
        crc32_hex=crc32_hex,
        first8=first8,
        mid8=mid8,
        last8=last8,
    )


def _clamp_norm(value: float) -> float:
    """Clamp a normalized model output to the [0, 1] interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _training_crop_box(width_pixels: int, height_pixels: int) -> tuple[float, float, float, float]:
    """Return the stable training crop box in source-image coordinates."""
    x_min = int(float(width_pixels) * TRAINING_CROP_X_MIN_RATIO)
    y_min = int(float(height_pixels) * TRAINING_CROP_Y_MIN_RATIO)
    width = max(
        1,
        int(float(width_pixels) * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO)),
    )
    height = max(
        1,
        int(float(height_pixels) * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO)),
    )
    return (
        float(x_min),
        float(y_min),
        float(x_min + width),
        float(y_min + height),
    )


def _training_crop_center(width_pixels: int, height_pixels: int) -> tuple[float, float]:
    """Return the integer center used by the firmware training crop helper."""
    x_min = int(float(width_pixels) * TRAINING_CROP_X_MIN_RATIO)
    y_min = int(float(height_pixels) * TRAINING_CROP_Y_MIN_RATIO)
    width = max(
        1,
        int(float(width_pixels) * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO)),
    )
    height = max(
        1,
        int(float(height_pixels) * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO)),
    )
    return (float(x_min + (width // 2)), float(y_min + (height // 2)))


def _expand_axis_aligned_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    image_width: int,
    image_height: int,
    min_size: float,
) -> tuple[float, float, float, float]:
    """Expand a box so it stays usable and stays inside the source image."""
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    target_width = max(width, min_size)
    target_height = max(height, min_size)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    new_x_min = center_x - 0.5 * target_width
    new_y_min = center_y - 0.5 * target_height
    new_x_max = new_x_min + target_width
    new_y_max = new_y_min + target_height

    if new_x_min < 0.0:
        new_x_max -= new_x_min
        new_x_min = 0.0
    if new_y_min < 0.0:
        new_y_max -= new_y_min
        new_y_min = 0.0
    if new_x_max > float(image_width):
        shift = new_x_max - float(image_width)
        new_x_min = max(0.0, new_x_min - shift)
        new_x_max = float(image_width)
    if new_y_max > float(image_height):
        shift = new_y_max - float(image_height)
        new_y_min = max(0.0, new_y_min - shift)
        new_y_max = float(image_height)

    if new_x_max <= new_x_min + 1.0:
        new_x_max = min(float(image_width), new_x_min + 1.0)
    if new_y_max <= new_y_min + 1.0:
        new_y_max = min(float(image_height), new_y_min + 1.0)
    return (new_x_min, new_y_min, new_x_max, new_y_max)


def load_capture_image(
    capture_path: Path,
    *,
    image_width: int = DEFAULT_IMAGE_SIZE,
    image_height: int = DEFAULT_IMAGE_SIZE,
) -> tuple[RGBImage, str]:
    """Load a raw board capture or standard RGB image for replay."""
    suffix = capture_path.suffix.lower()
    if suffix == CAPTURE_YUV422_SUFFIX:
        return (
            load_yuv422_capture_as_rgb(
                capture_path,
                image_width=image_width,
                image_height=image_height,
            ),
            "yuv422",
        )
    return (load_rgb_image(capture_path), "rgb")


def prepare_full_frame(image: RGBImage, *, image_size: int = DEFAULT_IMAGE_SIZE) -> RGBImage:
    """Resize the full source frame into the model's square input canvas."""
    height, width = image.shape[:2]
    if height == image_size and width == image_size:
        return np.ascontiguousarray(image)
    return _resize_with_pad_rgb_pil(
        image,
        (0.0, 0.0, float(width), float(height)),
        image_size=image_size,
    )


def _resize_with_pad_rgb_pil(
    image: RGBImage,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> RGBImage:
    """Crop and resize with padding using Pillow instead of TensorFlow."""
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = crop_box_xyxy
    x_min_i = max(0, int(math.floor(x_min)))
    y_min_i = max(0, int(math.floor(y_min)))
    x_max_i = min(image_width, int(math.ceil(x_max)))
    y_max_i = min(image_height, int(math.ceil(y_max)))
    if x_max_i <= x_min_i:
        x_max_i = min(image_width, x_min_i + 1)
    if y_max_i <= y_min_i:
        y_max_i = min(image_height, y_min_i + 1)

    crop = image[y_min_i:y_max_i, x_min_i:x_max_i]
    crop_height, crop_width = crop.shape[:2]
    if crop_height == image_size and crop_width == image_size:
        return np.ascontiguousarray(crop)

    scale = min(float(image_size) / float(crop_width), float(image_size) / float(crop_height))
    resized_width = max(1, int(round(float(crop_width) * scale)))
    resized_height = max(1, int(round(float(crop_height) * scale)))
    resized = Image.fromarray(crop, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    offset_x = max(0, (image_size - resized_width) // 2)
    offset_y = max(0, (image_size - resized_height) // 2)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = np.asarray(resized, dtype=np.uint8)
    return canvas


def load_model_session(model_path: Path, model_kind: ModelKind = "auto") -> ModelSession:
    """Load either a Keras model or a TFLite interpreter for replay."""
    resolved_kind: Literal["keras", "tflite"]
    if model_kind == "auto":
        resolved_kind = "tflite" if model_path.suffix.lower() == ".tflite" else "keras"
    else:
        resolved_kind = cast(Literal["keras", "tflite"], model_kind)

    if resolved_kind == "keras":
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            },
            compile=False,
            safe_mode=False,
        )
        return ModelSession(kind="keras", model=model)

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return ModelSession(
        kind="tflite",
        model=interpreter,
        input_details=input_details,
        output_details=output_details,
    )


def decode_obb_crop_box(
    obb_params: NDArray[np.float32],
    *,
    source_width: int,
    source_height: int,
    input_size: int = DEFAULT_IMAGE_SIZE,
    obb_crop_scale: float = OBB_CROP_SCALE,
    min_crop_size: float = OBB_MIN_CROP_SIZE_PIXELS,
) -> CropDecodeDecision:
    """Mirror the firmware's OBB-to-scalar crop decoder."""
    if obb_params.size < 6:
        raise ValueError("OBB prediction did not contain six parameters.")

    center_x_norm = _clamp_norm(float(obb_params[0]))
    center_y_norm = _clamp_norm(float(obb_params[1]))
    box_w_norm = max(OBB_MIN_BOX_RATIO, min(1.0, float(obb_params[2])))
    box_h_norm = max(OBB_MIN_BOX_RATIO, min(1.0, float(obb_params[3])))
    angle_cos = float(obb_params[4])
    angle_sin = float(obb_params[5])
    theta_rad = 0.5 * math.atan2(angle_sin, angle_cos)

    canvas_center_x = center_x_norm * float(input_size)
    canvas_center_y = center_y_norm * float(input_size)
    half_width = 0.5 * box_w_norm * float(input_size) * obb_crop_scale
    half_height = 0.5 * box_h_norm * float(input_size) * obb_crop_scale

    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    source_points: list[tuple[float, float]] = []
    for dx, dy in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        canvas_x = canvas_center_x + (dx * cos_theta - dy * sin_theta)
        canvas_y = canvas_center_y + (dx * sin_theta + dy * cos_theta)
        source_point = source_xy_from_resized_xy(
            (canvas_x, canvas_y),
            crop_box_xyxy=(0.0, 0.0, float(source_width), float(source_height)),
            image_height=source_height,
            image_width=source_width,
        )
        source_points.append(source_point)

    x_values = [point[0] for point in source_points]
    y_values = [point[1] for point in source_points]
    crop_x_min_f, crop_y_min_f, crop_x_max_f, crop_y_max_f = _expand_axis_aligned_box(
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
        image_width=source_width,
        image_height=source_height,
        min_size=min_crop_size,
    )

    crop_width = crop_x_max_f - crop_x_min_f
    crop_height = crop_y_max_f - crop_y_min_f
    training_crop_x0, training_crop_y0, training_crop_x1, training_crop_y1 = (
        _training_crop_box(source_width, source_height)
    )
    training_crop_width = max(training_crop_x1 - training_crop_x0, 1.0)
    training_crop_height = max(training_crop_y1 - training_crop_y0, 1.0)
    crop_width_ratio = crop_width / training_crop_width
    crop_height_ratio = crop_height / training_crop_height

    accepted = (
        OBB_TRAINING_CROP_MIN_RATIO <= crop_width_ratio <= OBB_TRAINING_CROP_MAX_RATIO
        and OBB_TRAINING_CROP_MIN_RATIO <= crop_height_ratio <= OBB_TRAINING_CROP_MAX_RATIO
    )
    fallback_reason = None if accepted else "crop outside training window"
    details = {
        "center_x": center_x_norm,
        "center_y": center_y_norm,
        "box_w": box_w_norm,
        "box_h": box_h_norm,
        "angle_cos": angle_cos,
        "angle_sin": angle_sin,
        "theta_deg": math.degrees(theta_rad),
        "training_crop_width": training_crop_width,
        "training_crop_height": training_crop_height,
        "crop_width_ratio": crop_width_ratio,
        "crop_height_ratio": crop_height_ratio,
    }
    return CropDecodeDecision(
        stage="obb",
        accepted=accepted,
        crop_box_xyxy=(crop_x_min_f, crop_y_min_f, crop_x_max_f, crop_y_max_f),
        fallback_reason=fallback_reason,
        details=details,
    )


def decode_rectifier_crop_box(
    rectifier_params: NDArray[np.float32],
    *,
    source_width: int,
    source_height: int,
    input_size: int = DEFAULT_IMAGE_SIZE,
    rectifier_crop_scale: float = RECTIFIER_CROP_SCALE,
) -> CropDecodeDecision:
    """Mirror the firmware's rectifier-to-scalar crop decoder."""
    if rectifier_params.size < 4:
        raise ValueError("Rectifier prediction did not contain four parameters.")

    center_x = _clamp_norm(float(rectifier_params[0]))
    center_y = _clamp_norm(float(rectifier_params[1]))
    box_w = max(RECTIFIER_MIN_BOX_RATIO, min(1.0, float(rectifier_params[2])))
    box_h = max(RECTIFIER_MIN_BOX_RATIO, min(1.0, float(rectifier_params[3])))

    training_x0, training_y0, training_x1, training_y1 = _training_crop_box(
        source_width,
        source_height,
    )
    training_width = max(training_x1 - training_x0, 1.0)
    training_height = max(training_y1 - training_y0, 1.0)
    training_center_x, training_center_y = _training_crop_center(
        source_width,
        source_height,
    )
    blend_factor = (
        float(RECTIFIER_CENTER_BLEND_NUMERATOR)
        / float(RECTIFIER_CENTER_BLEND_DENOMINATOR)
    )

    use_fixed_training_crop = False
    if RECTIFIER_FIXED_SCALE_CROP:
        if (
            center_x < RECTIFIER_CENTER_MIN_RATIO
            or center_x > RECTIFIER_CENTER_MAX_RATIO
            or center_y < RECTIFIER_CENTER_MIN_RATIO
            or center_y > RECTIFIER_CENTER_MAX_RATIO
        ):
            use_fixed_training_crop = True
    else:
        if (
            box_w < RECTIFIER_MIN_BOX_RATIO
            or box_h < RECTIFIER_MIN_BOX_RATIO
            or box_w > 1.5
            or box_h > 1.5
        ):
            use_fixed_training_crop = True

    if use_fixed_training_crop:
        crop_box_xyxy = training_x0, training_y0, training_x1, training_y1
        details = {
            "center_x": center_x,
            "center_y": center_y,
            "box_w": box_w,
            "box_h": box_h,
            "training_center_x": training_center_x,
            "training_center_y": training_center_y,
            "training_width": training_width,
            "training_height": training_height,
            "blend_factor": blend_factor,
            "use_fixed_training_crop": 1.0,
        }
        return CropDecodeDecision(
            stage="rectifier",
            accepted=True,
            crop_box_xyxy=crop_box_xyxy,
            fallback_reason="centre out of range",
            details=details,
        )

    if RECTIFIER_FIXED_SCALE_CROP:
        crop_width_f = training_width
        crop_height_f = training_height
        crop_x_min_f = (
            float(training_center_x)
            + (((float(source_width) * center_x) - float(training_center_x)) * blend_factor)
            - (crop_width_f * 0.5)
        )
        crop_y_min_f = (
            float(training_center_y)
            + (((float(source_height) * center_y) - float(training_center_y)) * blend_factor)
            - (crop_height_f * 0.5)
        )
    else:
        crop_width_f = float(source_width) * box_w * rectifier_crop_scale
        crop_height_f = float(source_height) * box_h * rectifier_crop_scale
        crop_x_min_f = (float(source_width) * center_x) - (crop_width_f * 0.5)
        crop_y_min_f = (float(source_height) * center_y) - (crop_height_f * 0.5)

    if crop_width_f < 1.0:
        crop_width_f = 1.0
    if crop_height_f < 1.0:
        crop_height_f = 1.0

    if crop_x_min_f < 0.0:
        crop_x_min_f = 0.0
    if crop_y_min_f < 0.0:
        crop_y_min_f = 0.0
    if (crop_x_min_f + crop_width_f) > float(source_width):
        crop_x_min_f = float(source_width) - crop_width_f
    if (crop_y_min_f + crop_height_f) > float(source_height):
        crop_y_min_f = float(source_height) - crop_height_f
    if crop_x_min_f < 0.0:
        crop_x_min_f = 0.0
    if crop_y_min_f < 0.0:
        crop_y_min_f = 0.0

    crop_x_min = float(int(crop_x_min_f + 0.5))
    crop_y_min = float(int(crop_y_min_f + 0.5))
    crop_width = float(int(crop_width_f + 0.5)) if RECTIFIER_FIXED_SCALE_CROP else crop_width_f
    crop_height = float(int(crop_height_f + 0.5)) if RECTIFIER_FIXED_SCALE_CROP else crop_height_f
    crop_x_max = crop_x_min + crop_width
    crop_y_max = crop_y_min + crop_height
    if crop_x_max > float(source_width):
        crop_x_max = float(source_width)
        crop_x_min = max(0.0, crop_x_max - crop_width)
    if crop_y_max > float(source_height):
        crop_y_max = float(source_height)
        crop_y_min = max(0.0, crop_y_max - crop_height)

    details = {
        "center_x": center_x,
        "center_y": center_y,
        "box_w": box_w,
        "box_h": box_h,
        "training_center_x": training_center_x,
        "training_center_y": training_center_y,
        "training_width": training_width,
        "training_height": training_height,
        "blend_factor": blend_factor,
        "use_fixed_training_crop": 0.0,
    }
    return CropDecodeDecision(
        stage="rectifier",
        accepted=True,
        crop_box_xyxy=(crop_x_min, crop_y_min, crop_x_max, crop_y_max),
        fallback_reason=None,
        details=details,
    )


class InferenceBurstHistory:
    """Small 3-frame median history that matches the firmware smoothing."""

    def __init__(
        self,
        *,
        size: int = INFERENCE_BURST_HISTORY_SIZE,
        reset_delta_c: float = INFERENCE_BURST_RESET_DELTA_C,
    ) -> None:
        self._size = size
        self._reset_delta_c = reset_delta_c
        self._values: deque[float] = deque(maxlen=size)

    def reset(self) -> None:
        """Drop the current history buffer."""
        self._values.clear()

    def update(self, value: float) -> tuple[float, bool, int]:
        """Add one value and return the firmware-style smoothed output."""
        if not math.isfinite(value):
            if self._values:
                return (self._values[-1], False, len(self._values))
            return (value, False, 0)

        reset = False
        if self._values and abs(value - self._values[-1]) > self._reset_delta_c:
            self.reset()
            reset = True

        self._values.append(value)
        sample_count = len(self._values)
        ordered = sorted(self._values)
        if sample_count == 1:
            return (ordered[0], reset, sample_count)
        if sample_count == 2:
            return (0.5 * (ordered[0] + ordered[1]), reset, sample_count)
        return (ordered[sample_count // 2], reset, sample_count)


def apply_affine_calibration(
    raw_value: float,
    *,
    scale: float = CALIBRATION_AFFINE_SCALE,
    bias: float = CALIBRATION_AFFINE_BIAS,
) -> float:
    """Apply the firmware's fixed scalar output calibration."""
    return bias + (scale * raw_value)


def _predict_full_frame_batch(image: RGBImage, *, image_size: int) -> np.ndarray:
    """Build the full-frame batch used by the OBB and rectifier stages."""
    full_frame = prepare_full_frame(image, image_size=image_size)
    return np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)


def _predict_scalar_from_crop(
    scalar_session: ModelSession,
    source_image: RGBImage,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_size: int,
) -> tuple[np.ndarray, TensorProbe, TensorProbe, float]:
    """Run the scalar reader on one crop and return probes plus raw value."""
    crop = _resize_with_pad_rgb_pil(source_image, crop_box_xyxy, image_size=image_size)
    scalar_batch = np.expand_dims(crop.astype(np.float32) / 255.0, axis=0)
    scalar_input_probe = _probe_tensor("scalar_input", scalar_batch)
    scalar_output, output_name = _run_session(scalar_session, scalar_batch)
    scalar_output_probe = _probe_tensor(output_name, scalar_output)
    raw_value = float(np.asarray(scalar_output, dtype=np.float32).reshape(-1)[0])
    return scalar_output, scalar_input_probe, scalar_output_probe, raw_value


def predict_board_pipeline_on_capture(
    capture_path: Path,
    *,
    obb_session: ModelSession,
    rectifier_session: ModelSession,
    scalar_session: ModelSession,
    history: InferenceBurstHistory | None = None,
    progress: Callable[[str], None] | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    obb_crop_scale: float = OBB_CROP_SCALE,
    min_crop_size: float = OBB_MIN_CROP_SIZE_PIXELS,
    use_calibration: bool = True,
    calibration_scale: float = CALIBRATION_AFFINE_SCALE,
    calibration_bias: float = CALIBRATION_AFFINE_BIAS,
) -> BoardPipelineResult:
    """Replay the full board pipeline on one capture."""
    def emit(message: str) -> None:
        """Forward a progress message when a callback is available."""
        if progress is not None:
            progress(message)

    emit(f"load capture start path={capture_path.name}")
    source_image, source_kind = load_capture_image(
        capture_path,
        image_width=image_size,
        image_height=image_size,
    )
    source_height, source_width = source_image.shape[:2]
    emit(f"load capture done kind={source_kind} shape={source_image.shape}")
    emit("prepare full-frame batch start")
    full_frame_batch = _predict_full_frame_batch(source_image, image_size=image_size)
    full_frame_probe = _probe_tensor("full_frame", full_frame_batch)
    emit("prepare full-frame batch done")

    emit("obb invoke start")
    obb_output, obb_output_name = _run_session(
        obb_session,
        full_frame_batch,
        preferred_output_keys=("obb_params",),
    )
    emit("obb invoke done")
    obb_output_probe = _probe_tensor(obb_output_name, obb_output)
    obb_decision = decode_obb_crop_box(
        np.asarray(obb_output, dtype=np.float32).reshape(-1),
        source_width=source_width,
        source_height=source_height,
        input_size=image_size,
        obb_crop_scale=obb_crop_scale,
        min_crop_size=min_crop_size,
    )

    selected_stage: Literal["obb", "rectifier"] = "obb"
    rectifier_output_probe: TensorProbe | None = None
    rectifier_decision: CropDecodeDecision | None = None

    if obb_decision.accepted:
        selected_crop_box = obb_decision.crop_box_xyxy
        try:
            emit("scalar invoke start stage=obb")
            _scalar_output, scalar_input_probe, scalar_output_probe, raw_prediction = (
                _predict_scalar_from_crop(
                    scalar_session,
                    source_image,
                    selected_crop_box,
                    image_size=image_size,
                )
            )
            emit("scalar invoke done stage=obb")
        except Exception:
            selected_stage = "rectifier"
        else:
            calibrated_prediction = (
                apply_affine_calibration(
                    raw_prediction,
                    scale=calibration_scale,
                    bias=calibration_bias,
                )
                if use_calibration
                else raw_prediction
            )
            if history is None:
                reported_prediction = calibrated_prediction
                history_reset = False
                history_count = 0
            else:
                reported_prediction, history_reset, history_count = history.update(
                    calibrated_prediction
                )
            return BoardPipelineResult(
                capture_path=str(capture_path.resolve()),
                source_kind=source_kind,
                source_shape=(int(source_height), int(source_width), int(source_image.shape[2])),
                full_frame_probe=full_frame_probe,
                obb_output_probe=obb_output_probe,
                obb_decision=obb_decision,
                rectifier_output_probe=None,
                rectifier_decision=None,
                selected_stage="obb",
                selected_crop_box_xyxy=selected_crop_box,
                scalar_input_probe=scalar_input_probe,
                scalar_output_probe=scalar_output_probe,
                raw_prediction=raw_prediction,
                calibrated_prediction=calibrated_prediction,
                reported_prediction=reported_prediction,
                burst_history_count=history_count,
                burst_history_reset=history_reset,
            )

    emit("rectifier invoke start")
    rectifier_output, rectifier_output_name = _run_session(
        rectifier_session,
        full_frame_batch,
        preferred_output_keys=("rectifier_box",),
    )
    emit("rectifier invoke done")
    rectifier_output_probe = _probe_tensor(rectifier_output_name, rectifier_output)
    rectifier_decision = decode_rectifier_crop_box(
        np.asarray(rectifier_output, dtype=np.float32).reshape(-1),
        source_width=source_width,
        source_height=source_height,
        input_size=image_size,
    )
    selected_crop_box = rectifier_decision.crop_box_xyxy
    emit("scalar invoke start stage=rectifier")
    _scalar_output, scalar_input_probe, scalar_output_probe, raw_prediction = (
        _predict_scalar_from_crop(
            scalar_session,
            source_image,
            selected_crop_box,
            image_size=image_size,
        )
    )
    emit("scalar invoke done stage=rectifier")
    calibrated_prediction = (
        apply_affine_calibration(
            raw_prediction,
            scale=calibration_scale,
            bias=calibration_bias,
        )
        if use_calibration
        else raw_prediction
    )
    if history is None:
        reported_prediction = calibrated_prediction
        history_reset = False
        history_count = 0
    else:
        reported_prediction, history_reset, history_count = history.update(
            calibrated_prediction
        )
    return BoardPipelineResult(
        capture_path=str(capture_path.resolve()),
        source_kind=source_kind,
        source_shape=(int(source_height), int(source_width), int(source_image.shape[2])),
        full_frame_probe=full_frame_probe,
        obb_output_probe=obb_output_probe,
        obb_decision=obb_decision,
        rectifier_output_probe=rectifier_output_probe,
        rectifier_decision=rectifier_decision,
        selected_stage=selected_stage,
        selected_crop_box_xyxy=selected_crop_box,
        scalar_input_probe=scalar_input_probe,
        scalar_output_probe=scalar_output_probe,
        raw_prediction=raw_prediction,
        calibrated_prediction=calibrated_prediction,
        reported_prediction=reported_prediction,
        burst_history_count=history_count,
        burst_history_reset=history_reset,
    )
