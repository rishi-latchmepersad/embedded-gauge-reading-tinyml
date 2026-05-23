#!/usr/bin/env python3
"""Inspect the exported geometry heatmap TFLite tensor contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import summarize_tflite_contract


def _resolve_path(base_path: Path, path: Path) -> Path:
    """Resolve relative paths against the repository root."""

    return path if path.is_absolute() else base_path / path


def _write_report(report_path: Path, contract: dict[str, Any]) -> None:
    """Write a concise markdown report for the selected int8 contract."""

    int8_contract = contract["int8"]
    float32_contract = contract["float32"]

    lines = [
        "# Geometry Heatmap v2 TFLite Tensor Contract",
        "",
        "## Input Tensor",
        "",
        "| model | tensor name | shape | dtype | scale | zero point | quantized |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
        f"| float32 | {float32_contract['input']['name']} | {float32_contract['input']['shape']} | {float32_contract['input']['dtype']} | {float32_contract['input']['quantization']['scale']} | {float32_contract['input']['quantization']['zero_point']} | {float32_contract['input']['quantized']} |",
        f"| int8 | {int8_contract['input']['name']} | {int8_contract['input']['shape']} | {int8_contract['input']['dtype']} | {int8_contract['input']['quantization']['scale']} | {int8_contract['input']['quantization']['zero_point']} | {int8_contract['input']['quantized']} |",
        "",
        "## Output Tensors",
        "",
        "| model | output name | shape | dtype | scale | zero point | quantized |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]

    for model_name, model_contract in (("float32", float32_contract), ("int8", int8_contract)):
        for output in model_contract["outputs"]:
            lines.append(
                f"| {model_name} | {output['name']} | {output['shape']} | {output['dtype']} | {output['quantization']['scale']} | {output['quantization']['zero_point']} | {output['quantized']} |"
            )

    lines.extend(
        [
            "",
            "## Dequantization Rules",
            "",
            f"- float32 model requires no output dequantization: {not float32_contract['requires_dequantization']}",
            f"- int8 model requires output dequantization: {int8_contract['requires_dequantization']}",
            "- For int8 tensors, decode values with `(tensor - zero_point) * scale` after reading the raw output buffer.",
            "- Semantic output order is center_heatmap, tip_heatmap, confidence, even though the raw TFLite tensors arrive as tip_heatmap, center_heatmap, confidence.",
            "- Keep the board input path as RGB + bilinear resize + 224x224 + float32 normalization to 0..1.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Inspect both exported TFLite variants and record the tensor contract."""

    parser = argparse.ArgumentParser(description="Inspect geometry heatmap TFLite contract")
    parser.add_argument(
        "--float32-model-path",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_float32.tflite"),
        help="Float32 TFLite model path.",
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/model_int8.tflite"),
        help="Int8 TFLite model path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_tflite_contract.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_tensor_contract.json"),
        help="JSON artifact path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    float32_model_path = _resolve_path(repo_root, args.float32_model_path)
    int8_model_path = _resolve_path(repo_root, args.int8_model_path)
    report_path = _resolve_path(repo_root, args.report_path)
    json_path = _resolve_path(repo_root, args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    contract = {
        "float32": summarize_tflite_contract(float32_model_path),
        "int8": summarize_tflite_contract(int8_model_path),
    }
    # The converter emits the two heatmap tensors in reverse semantic order.
    semantic_output_order_indices = [1, 0, 2]
    semantic_output_names = ["center_heatmap", "tip_heatmap", "confidence"]
    for model_name in ("float32", "int8"):
        contract[model_name]["semantic_output_order_indices"] = semantic_output_order_indices
        contract[model_name]["semantic_output_names"] = semantic_output_names
    json_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report_path, contract)

    print(f"[TFLITE] Wrote tensor contract JSON to {json_path}", flush=True)
    print(f"[TFLITE] Wrote markdown report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
