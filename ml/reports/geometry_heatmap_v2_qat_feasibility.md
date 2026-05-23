# Geometry Heatmap v2 QAT Feasibility

- TensorFlow: 2.20.0
- Keras: 3.13.2
- tensorflow_model_optimization available: False
- Model path: /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/geometry_heatmap_v2/model.keras
- Model name: mobilenetv2_geometry_heatmap_v1
- Input shape: [None, 224, 224, 3]
- Output names: center_heatmap, tip_heatmap, confidence
- Output shapes: [[None, 56, 56, 1], [None, 56, 56, 1], [None, 1]]
- Custom layer count: 0
- Custom layers: none
- Standard TFMOT QAT feasible: False
- Preferred training strategy: quantization_noise_fine_tuning

## Notes
- The model has three named outputs: center_heatmap, tip_heatmap, confidence.
- A clean TFMOT path requires the optional tensorflow_model_optimization dependency.
- tensorflow_model_optimization is not installed in the active Poetry environment.
- Prefer quantization-noise fine-tuning with fake int8 output round-trips.