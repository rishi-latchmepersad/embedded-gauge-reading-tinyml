# YOLOv11m teacher model for knowledge distillation — 2026-07-23

Date: 2026-07-23
Status: critical finding
Scope: ellipse detector training strategy

## Teacher models available

Located in `D:\Projects\cvat`:

1. **YOLOv11m gauge face fine-tune** — `training/gaugeface_segmentation/runs/gaugeface_yolo11m_v2_finetune/weights/best.pt`
   - 161MB
   - 99.7% mAP@0.5 on 4,562 images
   - Trained on 4 gauge types
   - Best teacher for our 8K-image ellipse detector

2. **YOLOv11 general gauge face** — `training/gaugeface_segmentation/runs/gaugeface_yolo11m/weights/best.pt`
   - 99.8% mAP@0.5

3. **ETHZ two-stage model** — `models/analog-gauge-reader/`
   - gauge_detection_model.pt + segmentation_model.pt
   - 99.8% on test set

## How to use for knowledge distillation

The YOLOv11m teacher (99.7% mAP) is much more accurate than our current student (25.9% on test_1). We can:

1. **Generate soft pseudo-labels**: Run YOLOv11m on all 8K training images, get high-quality ellipse predictions with confidence scores
2. **Train student with KD loss**: Student model matches teacher's predictions (soft labels) in addition to hard ground truth
3. **Transfer learning**: Student inherits the teacher's feature extraction capabilities

## Why this solves the convergence risk

The Rank-1 recommendation (RepVGG from scratch) had a major risk: convergence from scratch on 8K images. Knowledge distillation from a strong teacher (99.7% mAP) solves this:

- Student doesn't need to discover optimal solution from scratch
- Soft labels from teacher encode more information than hard ground truth
- Teacher's confidence scores provide weighted training signal
- Teacher's predictions are more accurate than ground truth annotations in some cases

## Pipeline for tomorrow

1. **Teacher inference**: Run YOLOv11m on all 8K training images, save ellipse predictions (cx, cy, rx, ry, confidence)
2. **Soft label generation**: Use teacher predictions as soft targets alongside hard ground truth
3. **Distillation loss**: L_total = α * L_hard + (1-α) * L_soft where:
   - L_hard = Huber loss against ground truth
   - L_soft = Huber loss against teacher predictions
   - α = 0.5 (balanced)
4. **Student training**: Same QAT-safe architecture (Conv+BN+ReLU) but with KD loss
5. **Evaluation**: Test on test_1 (primary), test_2, test_3

## Key advantage

The YOLOv11m teacher's soft labels are much more informative than hard ground truth. The student learns the "dark knowledge" of how to predict ellipses accurately, not just the final answer.

## Files to use

- **Teacher model**: `/mnt/d/Projects/cvat/training/gaugeface_segmentation/runs/gaugeface_yolo11m_v2_finetune/weights/best.pt`
- **YOLOv11 inference**: via ultralytics package (`pip install ultralytics`)
- **Output format**: Convert YOLO boxes to ellipse parameters (cx, cy, w/2, h/2)

## What to use the ETHZ model for

The ETHZ model (`models/analog-gauge-reader/`) provides a two-stage pipeline: detect → segment → fit ellipse. This can be used as a second teacher or for hard-negative mining (find images where the student fails but the teacher succeeds).

## Key insight

The student doesn't need to be better than the teacher. It needs to be:
- **Compact enough** for embedded deployment (1-2MB int8)
- **Accurate enough** for the target use case (80%+ center/tip accuracy)
- **QAT-compatible** for int8 deployment

KD from a 99.7% teacher to a compact student should achieve this if the student architecture is appropriate.
