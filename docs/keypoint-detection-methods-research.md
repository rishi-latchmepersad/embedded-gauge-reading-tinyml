# Keypoint Detection Methods for Analog Gauge Reading
## Research Summary - Deep Learning Approaches for Needle Center + Tip Detection

---

## 1. Problem Statement

**Goal:** Detect two keypoints (needle center and needle tip) on analog gauge faces with sub-5px accuracy on 224x224 images.

**Constraints:**
- Model must be small enough for embedded deployment (under 2.5MB peak activation for int8)
- ~8000 training images available
- 224x224 input resolution
- Must work with INT8 quantization on STM32 N6 NPU

**Current Issue:** The model predicts tip position poorly (20px error) despite getting center right (9px error) — it underestimates the center-to-tip distance.

---

## 2. Methods Overview

### 2.1 Heatmap-Based Methods

**How it works:**
- Generate 2D Gaussian heatmaps for each keypoint
- Model predicts heatmap per keypoint
- Post-processing: argmax + sub-pixel refinement (e.g., Taylor expansion)
- Most common approach in pose estimation (COCO benchmark leader)

**Accuracy:**
- State-of-the-art on COCO: 73-79 AP (varies by backbone)
- Sub-pixel refinement typically achieves < 1px improvement over raw argmax
- Heatmap resolution directly affects quantization error (typically stride 4)

**Architecture:**
- UNet-like encoder-decoder with upsampling
- HRNet: maintains high-resolution throughout (multi-scale parallel branches)
- Output: K heatmaps (one per keypoint)

**Pros:**
- Well-studied, robust
- Handles occlusion well (multiple peaks)
- Good at localizing keypoints with clear visual features

**Cons:**
- Quantization error from downsampling (stride 4-8 pixels)
- Requires upsampling layers (memory intensive)
- Heatmaps are 2D — doesn't exploit geometric constraints
- Underestimates center-to-tip distance (exactly your problem!)

**Why it fails on tip:**
- Heatmaps treat each keypoint independently
- No geometric relationship between center and tip
- Tip is visually subtle (thin needle end) vs center (pivot point)
- Network learns to predict "safe" average positions, compressing the distribution

**For your 224x224 input:**
- With stride 4: 56x56 heatmap → 4px quantization error
- With stride 2: 112x112 heatmap → 2px quantization error (but 4x more memory)
- Sub-pixel refinement can recover ~0.5-1px, but only if the heatmap is well-formed

---

### 2.2 SimCC (Simple Coordinate Classification)

**Paper:** "SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation" (Li et al., 2021, arXiv:2107.03332)

**How it works:**
- Reformulates keypoint detection as TWO 1D classification tasks
- For each keypoint, predict:
  - X coordinate: classify among W bins (e.g., 224 bins for 224px image)
  - Y coordinate: classify among H bins (e.g., 224 bins)
- Output: two vectors of length 224 (x distribution + y distribution)
- Can subdivide each pixel into bins for sub-pixel accuracy (e.g., 3 bins per pixel → 672 bins)

**Accuracy:**
- Outperforms heatmap methods on COCO, especially at low resolution
- Achieves 73.6 AP with ResNet-50 (vs ~72 AP for heatmap with same backbone)
- At low resolution (e.g., 56x56 input), SimCC outperforms heatmaps by large margin

**Model Size:**
- Very small output: 2 × K × num_bins (e.g., 2 × 2 × 224 = 896 values)
- No upsampling layers needed
- Can use stride-32 backbone and still achieve sub-pixel accuracy

**INT8 Compatibility:**
- Excellent: classification outputs are softmax probabilities
- Easy to quantize (no extreme value ranges)
- Can reduce to top-K bins for even smaller output

**Pros:**
- Eliminates quantization error entirely
- No upsampling → smaller memory footprint
- Sub-pixel accuracy without post-processing
- Very fast inference
- Natural fit for 2-keypoint problem

**Cons:**
- Assumes X and Y are independent (may miss correlations)
- Doesn't encode geometric relationships between keypoints
- Classification head adds parameters (224 × 224 × feature_dim)

**Adaptation for gauge reading:**
- Output: 2 × 2 × 224 (x,y for center + x,y for tip)
- Or with sub-pixel bins: 2 × 2 × 672 (3 bins per pixel)
- Total output: ~2,688 values (tiny!)

---

### 2.3 CenterNet (Anchor-Free Detection)

**Paper:** "Objects as Points" (Zhou et al., 2019, arXiv:1904.07850)

**How it works:**
- Detect object center as a keypoint heatmap
- Regress ALL other properties from the center:
  - Size (w, h)
  - Offset (dx, dy)
  - Depth, 3D orientation, pose, etc.
- Key insight: anchor-free, single-point representation

**For Gauge Reading:**
- Detect needle CENTER as primary keypoint
- Regress TIP as offset from center:
  - tip_x = center_x + dx
  - tip_y = center_y + dy
- Or regress (angle, length) from center:
  - angle = atan2(dy, dx)
  - length = sqrt(dx² + dy²)

**Accuracy:**
- COCO: 45.1 AP (multi-scale testing)
- At 142 FPS: 28.1 AP (real-time)
- Single keypoint detection + regression is very accurate for center

**Model Size:**
- Backbone: ResNet-18/34/50 (configurable)
- Head: Very small (just regression convolutions)
- Total: 10-50M params depending on backbone

**INT8 Compatibility:**
- Excellent: regression outputs are continuous values
- Heatmap branch: easy to quantize
- Offset branch: can clamp to reasonable ranges

**Pros:**
- Center is easy to detect (high contrast, circular shape)
- Offset regression is simpler than predicting two independent keypoints
- Geometric relationship is explicitly modeled
- Very fast

**Cons:**
- Offset regression can be noisy
- Requires good center detection first
- If center is wrong, tip is wrong (error propagation)
- Length regression is particularly challenging (needle can vary)

**For your 224x224 input:**
- Detect center at 56x56 heatmap (stride 4)
- Regress (dx, dy) as 2 values per pixel
- Total output: 56×56×3 (heatmap + 2 offsets) = 9,408 values

---

### 2.4 Center + Offset Regression (Direct Offset)

**Variation of CenterNet, specifically for your problem:**

**Architecture:**
1. Detect needle CENTER via heatmap (or SimCC)
2. At center location, regress:
   - dx, dy: offset to tip
   - OR: angle θ and length L (polar coordinates)

**Why this helps your problem:**
- Current model predicts center and tip INDEPENDENTLY
- No constraint that tip = center + direction × length
- Model can predict center at (100, 100) and tip at (108, 108) when true center is (100, 100) and tip is (108, 108)
- With offset: model learns center is (100, 100) and offset is (8, 8)
- The offset is much easier to learn (smaller range, more consistent)

**Loss Functions:**
- L1/L2 on offset (dx, dy)
- Angular loss: L_angle = 1 - cos(θ_pred - θ_true)
- Combined: L = α × L_center + β × L_offset + γ × L_angle

**Expected Improvement:**
- Center error: ~9px (already good)
- Tip error with independent prediction: 20px
- Tip error with offset: likely 5-10px (offset is simpler to learn)
- Combined: sqrt(9² + 5²) ≈ 10px (if offsets are independent errors)

**INT8 Considerations:**
- Offset values are small (e.g., -50 to +50 pixels)
- Easy to quantize with fixed-point arithmetic
- Can predict normalized offset (offset / image_size) for [-1, 1] range

---

### 2.5 SoftArgmax (Differentiable Argmax)

**How it works:**
- Instead of argmax (hard), use weighted average (soft)
- Formula: x = Σ(i × softmax(heatmap_i))
- Differentiable, end-to-end trainable
- Used in SimpleBaseline, HRNet

**Accuracy:**
- Slightly better than hard argmax (+0.5-1px)
- Smooths out quantization errors
- Standard in modern pose estimation

**For your problem:**
- Apply to center heatmap → get smooth center coordinates
- Apply to tip heatmap → get smooth tip coordinates
- Doesn't solve the independent prediction problem

---

### 2.6 HRNet (High-Resolution Network)

**Paper:** "High-Resolution Representations for Labeling Pixels and Regions" (Sun et al., 2019, arXiv:1904.04514)

**Architecture:**
- Maintains high-resolution representations throughout
- Multi-scale parallel branches (1/4, 1/8, 1/16, 1/32)
- Repeated fusion across branches
- No upsampling needed (unlike UNet)

**Accuracy:**
- Best on COCO pose estimation (74-79 AP)
- Especially good for fine-grained keypoint localization

**Model Size:**
- HRNet-W16: 13M params, 5.1 GFLOPs
- HRNet-W32: 28.5M params, 7.1 GFLOPs
- HRNet-W48: 63.6M params, 14.6 GFLOPs

**INT8 Compatibility:**
- Good, but large models may exceed 2.5MB activation limit
- Can use smaller variants (W16) for embedded

**For your 224x224 input:**
- HRNet-W16 at 224x224: ~5.1 GFLOPs
- Activation memory: depends on batch size, but typically 10-50MB
- May need to reduce to 160x160 or 128x128 input

**Key Insight:**
- High-resolution features are critical for fine localization
- Center and tip both need high-resolution features
- Tip is especially challenging (thin, low contrast)

---

### 2.7 Polar Coordinate Regression

**How it works:**
- From detected center, predict (θ, r) where:
  - θ = angle of needle (0-360°)
  - r = length of needle (0-max_length)
- Tip = center + (r × cos(θ), r × sin(θ))

**Advantages for Gauge Reading:**
- Exploits needle geometry explicitly
- Angle is the primary output (what we actually need!)
- Length is constrained (needle can't be longer than gauge diameter)
- More natural parameterization than (dx, dy)

**Implementation:**
- Output: 3 values (center_x, center_y, θ) or 4 values (center_x, center_y, θ, r)
- Can use heatmap for center, regression head for (θ, r)
- Loss: L1 for center, L1 or angular loss for θ, L1 for r

**Expected Performance:**
- Angle error: < 1° achievable
- Center error: ~5-10px
- Tip error: constrained by angle + length
- Much better than independent tip prediction

---

## 3. Gauge Reading Papers

### 3.1 "Learning to Read Analog Gauges from Synthetic Data" (WACV 2024)

**Paper:** Leon-Alcazar et al., arXiv:2308.14583

**Method:**
- Two-stage CNN pipeline
- Stage 1: Detect gauge components (circle, needle, markers)
- Stage 2: Predict angular reading
- Four keypoints: min marker, max marker, needle center, needle tip
- Uses synthetic data for training + real validation set (4,813 images)

**Results:**
- 52% improvement over previous state-of-the-art (4.55 average error reduction)
- Real-world dataset collected and manually annotated

**Key Insights:**
- Four keypoints provide sufficient information for angle calculation
- Synthetic data is effective for training
- Two-stage approach: detect → reason about geometry

**Relevance to Your Problem:**
- Confirms that center + tip are the critical keypoints
- Their method achieves high accuracy with explicit keypoint detection
- May be using similar heatmap approach

### 3.2 "Intelligent Meter Reading Technology Based on YOLOv8n" (ACM 2025)

**Method:**
- Dual-stream processing architecture
- Detects dial center and needle tip keypoints
- Uses YOLOv8n (lightweight) for real-time detection

**Relevance:**
- Uses keypoint detection for gauge reading
- Lightweight model suitable for embedded deployment

### 3.3 "Pointer Meter Recognition Method Based on YOLOv7 and Hough Transform" (2023)

**Method:**
- YOLOv7 for object detection
- Hough transform for line detection
- Combined approach for needle detection

**Relevance:**
- Traditional CV (Hough) can complement deep learning
- Line detection gives direction + length naturally

---

## 4. Recommended Architecture for Your Problem

Based on the research, here's what I recommend:

### 4.1 Option A: Center + Offset Regression (Best for Embedded)

**Architecture:**
```
Input (224×224×3)
    ↓
Lightweight Backbone (MobileNetV3-Small, EfficientNet-B0, or Custom)
    ↓
Feature Map (28×28×128 or 14×14×256)
    ↓
┌─────────────────────────────────────────┐
│ Center Head: 1×1 conv → 56×56×1        │
│   (center heatmap, stride 4)            │
│                                         │
│ Offset Head: 1×1 conv → 28×28×2        │
│   (dx, dy offset to tip)                │
│                                         │
│ Length Head: 1×1 conv → 28×28×1         │
│   (needle length, optional)             │
│                                         │
│ Angle Head: 1×1 conv → 28×28×2         │
│   (sin(θ), cos(θ) for angle)            │
└─────────────────────────────────────────┘
    ↓
Inference:
  1. Find center via argmax on center heatmap
  2. Sample offset at center location → (dx, dy)
  3. tip = center + (dx, dy)
  4. OR: angle = atan2(sin, cos), length = sqrt(dx²+dy²)
```

**Output:**
- Center heatmap: 56×56×1 = 3,136 values
- Offset: 28×28×2 = 1,568 values
- Angle: 28×28×2 = 1,568 values (optional)
- Total: ~6,272 values (tiny!)

**Loss Function:**
```python
L = α × BCE(center_heatmap, gt_center_heatmap)  # heatmap loss
  + β × L1(offset, gt_offset)                     # offset loss
  + γ × (1 - cos(angle - gt_angle))              # angular loss
```

**Why This Works:**
- Center is easy to detect (high accuracy already)
- Offset is constrained (needle is thin, direction is clear)
- Geometric relationship is explicit
- Very small output (fits in 2.5MB easily)

### 4.2 Option B: SimCC (Best for Accuracy)

**Architecture:**
```
Input (224×224×3)
    ↓
Backbone (ResNet-18 or EfficientNet-B0)
    ↓
Feature Map (7×7×512)
    ↓
┌─────────────────────────────────────────┐
│ Center X: FC → 224 (or 672 with 3 bins)│
│ Center Y: FC → 224 (or 672 with 3 bins)│
│ Tip X: FC → 224 (or 672 with 3 bins)   │
│ Tip Y: FC → 224 (or 672 with 3 bins)   │
└─────────────────────────────────────────┘
    ↓
Inference:
  1. center_x = argmax(center_x_logits)
  2. center_y = argmax(center_y_logits)
  3. tip_x = argmax(tip_x_logits)
  4. tip_y = argmax(tip_y_logits)
```

**Output:**
- 4 × 224 = 896 values (without sub-pixel bins)
- 4 × 672 = 2,688 values (with 3 bins per pixel)
- Very small!

**Pros:**
- No quantization error
- Sub-pixel accuracy out of the box
- Very fast inference

**Cons:**
- Doesn't model center-tip relationship
- May still underestimate distance

### 4.3 Option C: Hybrid (Best Balance)

**Architecture:**
```
Input (224×224×3)
    ↓
Backbone (MobileNetV3-Small)
    ↓
Feature Map (14×14×256)
    ↓
┌─────────────────────────────────────────┐
│ Center SimCC: FC → 224×2 (x, y)        │
│ Offset Regression: FC → 2 (dx, dy)     │
│ Angle Regression: FC → 2 (sin, cos)    │
└─────────────────────────────────────────┘
    ↓
Inference:
  1. center = (argmax(cx), argmax(cy))
  2. tip = center + (dx, dy)
  3. angle = atan2(sin, cos)
```

**Output:**
- Center: 224×2 = 448 values
- Offset: 2 values
- Angle: 2 values
- Total: 452 values (extremely small!)

**Why This Is Best:**
- SimCC gives accurate center (no quantization)
- Offset gives explicit center→tip relationship
- Angle is directly useful for gauge reading
- Total output is tiny (fits in registers!)

---

## 5. Loss Functions for Center + Offset

### 5.1 Standard L1/L2 Loss
```python
L_offset = MSE(predicted_offset, ground_truth_offset)
```
- Simple, stable
- Doesn't penalize angular errors

### 5.2 Angular Loss
```python
L_angle = 1 - cos(θ_pred - θ_true)  # range [0, 2]
# OR
L_angle = atan2(|sin(θ_pred - θ_true)|, cos(θ_pred - θ_true))
```
- Directly optimizes angle
- Better than L1 on (dx, dy)

### 5.3 Composite Loss
```python
# Decompose offset into angle + length
dx = length * cos(angle)
dy = length * sin(angle)

L = α * L_heatmap(center)
  + β * L1(length_pred, length_true)
  + γ * L_angle(angle_pred, angle_true)
```

### 5.4 Keypoint-Specific Loss (for heatmap + offset)
```python
# Gaussian heatmap generation
def generate_heatmap(center, sigma=2):
    # Create 2D Gaussian at center location
    pass

L = BCE(heatmap_pred, heatmap_gt)
  + λ * L1(offset_pred, offset_gt) * center_mask  # only apply offset loss near center
```

---

## 6. Training Strategy for Small Dataset (~8000 images)

### 6.1 Data Augmentation
- Random rotation (±30°) — preserves needle geometry
- Random scale (0.8-1.2)
- Random brightness/contrast
- Gaussian noise injection
- **Critical:** preserve needle visibility (don't over-darken)

### 6.2 Transfer Learning
- Pretrain backbone on ImageNet (standard)
- Or pretrain on COCO keypoints (if using pose estimation backbone)
- Fine-tune on gauge dataset

### 6.3 Curriculum Learning
- Stage 1: Train center detection only (easy)
- Stage 2: Add offset regression (harder)
- Stage 3: Add angle regression (hardest)

### 6.4 Synthetic Data Generation
- Render synthetic gauges with known needle positions
- Vary: background, lighting, needle color, gauge style
- Use 10k-100k synthetic images for pretraining
- Fine-tune on 8k real images

---

## 7. INT8 Quantization Considerations

### 7.1 Output Layer Quantization
- **Heatmap:** Softmax output → easy to quantize (bounded [0, 1])
- **Offset:** Small range (e.g., -50 to +50) → fixed-point Q7.8 or Q8.8
- **Angle:** (sin, cos) ∈ [-1, 1] → Q1.14 or similar
- **SimCC:** Logits → argmax doesn't need quantization

### 7.2 Activation Memory
- Target: < 2.5MB peak activation
- Calculate: feature_maps × height × width × channels × 4 bytes (float32)
- For 224×224 input with stride-16 backbone:
  - 14×14×256 = 50,176 values × 4 = 200KB (OK)
  - 28×28×128 = 100,352 values × 4 = 400KB (OK)
  - 56×56×64 = 200,704 values × 4 = 800KB (OK)
- For INT8: divide by 4 → 50-200KB (very safe)

### 7.3 Quantization-Aware Training (QAT)
- Use TFLite QAT during training
- Simulate quantization in forward pass
- Ensures accuracy is maintained after quantization
- Required for deployment (per your AGENTS.md instructions)

---

## 8. Key Insights for Your Problem

### 8.1 Why Current Model Fails on Tip
- Two independent heatmaps don't model center→tip relationship
- Network learns to predict "average" tip position
- Tip has less visual features (thin line end) vs center (bright pivot)
- No geometric constraint enforces tip = center + offset

### 8.2 Why Center + Offset Will Help
- Offset is easier to learn (smaller range, more consistent)
- Explicit geometric constraint: tip = center + offset
- Network only needs to learn direction + length from center
- Center is already accurate (9px) → offset is the missing piece

### 8.3 Why Angle + Length Is Better Than dx, dy
- Angle is the actual output we need (for gauge reading)
- Length is constrained (needle has finite size)
- Polar coordinates are more natural for this problem
- Easier to regularize (angle ∈ [0, 360°], length ∈ [0, max])

### 8.4 Recommended Minimum Changes
1. Keep center heatmap (already working well)
2. Add offset head (2 channels: dx, dy)
3. Change loss: L_total = L_center + 0.5 × L_offset
4. Train for 50 epochs, evaluate tip error
5. If tip error < 10px, add angle head for direct angle output

---

## 9. References

1. Zhou et al., "Objects as Points" (CenterNet), arXiv:1904.07850, 2019
2. Li et al., "SimCC: a Simple Coordinate Classification Perspective", arXiv:2107.03332, 2021
3. Sun et al., "High-Resolution Representations for Labeling Pixels and Regions" (HRNet), arXiv:1904.04514, 2019
4. Leon-Alcazar et al., "Learning to Read Analog Gauges from Synthetic Data", WACV 2024, arXiv:2308.14583
5. Yuan et al., "HRFormer: High-Resolution Transformer", arXiv:2110.09408, 2021
6. Cheng et al., "HigherHRNet: Scale-Aware Representation Learning", arXiv:1908.10357, 2019
7. Keles et al., "RSPose: Ranking Based Losses for Human Pose Estimation", arXiv:2511.13857, 2025

---

## 10. Next Steps

1. **Immediate:** Add offset regression head to existing UNet
2. **Experiment:** Compare (center + offset) vs (center + tip) training
3. **Validate:** Run Keras vs TFLite parity check
4. **Deploy:** Package for STM32 N6 with INT8 QAT
5. **Measure:** Track center error, tip error, angle error separately

---

*Document generated: 2026-07-30*
*Purpose: Research summary for keypoint detection methods suitable for embedded gauge reading*
