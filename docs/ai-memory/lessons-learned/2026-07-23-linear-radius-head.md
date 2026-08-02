# Linear radius head solves int8 quantization collapse — 2026-07-23

Date: 2026-07-23
Status: validated
Scope: ellipse detector int8 deployment
Evidence: `ml/artifacts/gauge_ellipse_qat_linear_v1/`, `ml/scripts/eval_full_pipeline_test3.py`
Decision: Always use linear output (no activation) for radius prediction in
         ellipse detectors. Sigmoid wastes int8 precision on unused [0,1] range.

## The problem

Sigmoid output for radius prediction collapses to a constant after int8
quantization. The sigmoid maps [0,1] to int8 [-128,127] with step size
0.0039. The radius variation across images (e.g., rx from 0.1947 to 0.1963)
is only 0.0016 — less than half an int8 step. So the quantized model
predicts the same radius for all inputs.

## The solution

Use a **linear output** (no activation) for the radius head:
```python
radius_xy = keras.layers.Dense(2, activation=None, name="radius_xy")(shared)
```

The quantization grid is calibrated from the representative dataset. If
radius values cluster around 0.195, the grid covers that range with fine
precision. The same 0.0016 variation now spans ~4 int8 steps.

## Results

| Model | Center ≤8px | Tip ≤8px | Radius error | Radius varies? |
|-------|------------|----------|-------------|----------------|
| Multi-head sigmoid | 91% | N/A | 38px | NO (fixed 0.2539) |
| **Linear radius** | **100%** | **100%** | **0.4-2.9px** | **YES (0.194-0.200)** |

Full pipeline on test_3 (linear radius ellipse + new center/tip):
- Center: 100% ≤8px, mean 2.8px, max 5.5px
- Tip: 100% ≤8px, mean 2.8px, max 6.3px

## Why it works

The quantization grid is calibrated to the actual output range from the
representative dataset. For radius values concentrated around 0.195:
- Linear output: grid covers [0.15, 0.25] with step ~0.0004 → 4 int8 steps for 0.0016 variation
- Sigmoid output: grid covers [0, 1] with step 0.0039 → <1 int8 step for 0.0016 variation

## Architecture

```python
# Center head: sigmoid (bounded [0,1], quantization-friendly)
center_xy = Dense(2, activation="sigmoid", name="center_xy")(shared)

# Radius head: LINEAR (unbounded, quantization grid covers actual range)
radius_xy = Dense(2, activation=None, name="radius_xy")(shared)

# Confidence head: sigmoid
confidence = Dense(1, activation="sigmoid", name="confidence")(shared)
```

## Loss weights

- Center: Huber(delta=0.05), weight 1.0
- Radius: Huber(delta=0.05), weight 3.0 (higher to focus on radius accuracy)
- Confidence: Huber(delta=0.05), weight 0.1

## References

- `ml/scripts/train_gauge_ellipse_qat_linear.py` — training script
- `ml/artifacts/gauge_ellipse_qat_linear_v1/` — model artifacts
- `ml/scripts/prepare_full_student_data.py` — data generation with linear radius model
