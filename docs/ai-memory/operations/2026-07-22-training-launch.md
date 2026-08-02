# Launch background training jobs in WSL — 2026-07-22

Date: 2026-07-22
Status: current
Scope: WSL ML training launch pattern

## Launch command

From the repo root, always `cd` into `ml/` first because `pyproject.toml` lives there:

```bash
cd /home/rishi_latchmepersad/Projects/embedded-gauge-reading-tinyml/ml
setsid poetry run python scripts/<script>.py > /tmp/<logname>.log 2>&1 &
disown
```

## Monitor

```bash
tail -f /tmp/<logname>.log
```

## One-at-a-time rule

Only run ONE GPU training job at a time. Two concurrent TF processes
share the GPU and each slows to ~3x worse throughput. Kill with:

```bash
kill $(ps aux | grep '<script_basename>' | grep -v grep | awk '{print $2}')
```

## Current training recipes

### Center/tip UNet (v9, best so far)
```bash
cd ~/Projects/embedded-gauge-reading-tinyml/ml
setsid poetry run python scripts/train_gauge_center_tip_v8_improved.py \
  > /tmp/center_tip_v9.log 2>&1 &
disown
```
Output: `artifacts/gauge_center_tip_littlegood_v9/`

### Ellipse regressor (v9, pending)
```bash
cd ~/Projects/embedded-gauge-reading-tinyml/ml
setsid poetry run python scripts/train_gauge_ellipse_v9.py \
  --epochs 80 --qat-epochs 20 --batch-size 8 --temp-weight 12.0 \
  > /tmp/ellipse_v9.log 2>&1 &
disown
```
Output: `artifacts/gauge_ellipse_littlegood_v9/`

### 224x224 center/tip (needs 640²→224² reprocess, not 160² upscale)
```bash
cd ~/Projects/embedded-gauge-reading-tinyml/ml
setsid poetry run python scripts/train_gauge_center_tip_224_v1.py \
  > /tmp/center_tip_224.log 2>&1 &
disown
```
Output: `artifacts/gauge_center_tip_224_littlegood_v2/`

## Training time estimates (RTX A5500 Laptop GPU, 15 GB)

| Script | Epoch time | Total time |
|--------|-----------|------------|
| center/tip 160² v9 (30+10 epochs) | 13s FP32, 33s QAT | ~17 min |
| center/tip 224² (40+15 epochs) | 16s FP32, 42s QAT | ~24 min |
| ellipse v9 (80+20 epochs, 972 steps) | ~5s FP32 | ~10 min |
