"""
Training script for PyTorch OBB regression models targeting STM32N6.

Usage:
    cd ml && poetry run python scripts/train_torch_obb.py --model pplcnet_linear

Supports:
    pplcnet_linear  — linearised PP-LCNet-0.5x
    edgevgg_linear  — EdgeVGG-Linear (structural re-param)
    qarepvgg_mini   — QARepVGG-Mini (QAT-friendly re-param)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import (
    QConfig,
    default_qconfig,
    default_qat_qconfig,
    prepare_qat,
    convert,
)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.torch_models import (
    MODEL_REGISTRY,
    build_model,
    model_summary,
)
from embedded_gauge_reading_tinyml.torch_losses import JointOBBLoss


# ── config ───────────────────────────────────────────────────────────────────

def default_config() -> dict:
    return {
        "model": "qarepvgg_mini",
        "image_size": 256,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 4e-5,
        "epochs_warmup": 80,
        "epochs_qat": 20,
        "qat_lr": 1e-4,
        "qat_start_epoch": 80,           # epoch to enable fake-quant
        "huber_beta": 1.0 / 9.0,
        "heatmap_focal_alpha": 2.0,
        "heatmap_focal_beta": 4.0,
        "w_heat": 1.0,
        "w_box": 1.0,
        "w_angle": 1.0,
        "clip_grad_norm": 10.0,
    }


# ── data helpers (fill in with your dataset) ─────────────────────────────────

def make_dummy_batch(batch_size: int = 4, img_size: int = 256,
                     device: torch.device = torch.device("cpu")):
    """Placeholder — replace with your actual data loader."""
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    target_heatmap = torch.zeros(batch_size, 1, img_size // 8, img_size // 8,
                                  device=device)
    target_box_size = torch.randn(batch_size, 2, img_size // 8, img_size // 8,
                                   device=device)
    target_angle = torch.randn(batch_size, 2, img_size // 8, img_size // 8,
                                device=device)

    # set a single centre peak in each sample
    h, w = target_heatmap.shape[-2:]
    for i in range(batch_size):
        cy, cx = h // 2, w // 2
        target_heatmap[i, 0, cy, cx] = 1.0

    return images, target_heatmap, target_box_size, target_angle


def build_dataloaders(cfg: dict):
    """Replace this with your train/val Dataset + DataLoader."""
    train_loader = [
        make_dummy_batch(cfg["batch_size"])
        for _ in range(50)
    ]
    val_loader = [
        make_dummy_batch(cfg["batch_size"])
        for _ in range(10)
    ]
    return train_loader, val_loader


# ── training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: list,
    optimizer: optim.Optimizer,
    criterion: JointOBBLoss,
    device: torch.device,
    cfg: dict,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_heat = 0.0
    total_box = 0.0
    total_angle = 0.0
    steps = 0

    for images, heat, box, angle in loader:
        images = images.to(device)
        heat = heat.to(device)
        box = box.to(device)
        angle = angle.to(device)

        outputs = model(images)
        losses = criterion(outputs, heat, box, angle)

        optimizer.zero_grad()
        losses["loss"].backward()
        if cfg["clip_grad_norm"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
        optimizer.step()

        total_loss += losses["loss"].item()
        total_heat += losses["heatmap_loss"].item()
        total_box += losses["box_loss"].item()
        total_angle += losses["angle_loss"].item()
        steps += 1

    n = max(steps, 1)
    return {
        "loss": total_loss / n,
        "heatmap_loss": total_heat / n,
        "box_loss": total_box / n,
        "angle_loss": total_angle / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: list,
    criterion: JointOBBLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_heat = 0.0
    total_box = 0.0
    total_angle = 0.0
    steps = 0

    for images, heat, box, angle in loader:
        images = images.to(device)
        heat = heat.to(device)
        box = box.to(device)
        angle = angle.to(device)

        outputs = model(images)
        losses = criterion(outputs, heat, box, angle)

        total_loss += losses["loss"].item()
        total_heat += losses["heatmap_loss"].item()
        total_box += losses["box_loss"].item()
        total_angle += losses["angle_loss"].item()
        steps += 1

    n = max(steps, 1)
    return {
        "val_loss": total_loss / n,
        "val_heatmap_loss": total_heat / n,
        "val_box_loss": total_box / n,
        "val_angle_loss": total_angle / n,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qarepvgg_mini",
                        choices=list(MODEL_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-warmup", type=int, default=80)
    parser.add_argument("--epochs-qat", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = default_config()
    cfg.update(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── build model ──────────────────────────────────────────────────────
    model = build_model(cfg["model"]).to(device)
    model_summary(model)
    criterion = JointOBBLoss(
        huber_beta=cfg["huber_beta"],
        heatmap_alpha=cfg["heatmap_focal_alpha"],
        heatmap_beta=cfg["heatmap_focal_beta"],
        w_heat=cfg["w_heat"],
        w_box=cfg["w_box"],
        w_angle=cfg["w_angle"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs_warmup"], eta_min=1e-6,
    )

    train_loader, val_loader = build_dataloaders(cfg)

    run_dir = Path("artifacts/training") / f"torch_{cfg['model']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── phase 1: FP32 warmup ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: FP32 warmup ({cfg['epochs_warmup']} epochs)")
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, cfg["epochs_warmup"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{cfg['epochs_warmup']}  "
            f"loss {train_metrics['loss']:.4f}  "
            f"heat {train_metrics['heatmap_loss']:.4f}  "
            f"box {train_metrics['box_loss']:.4f}  "
            f"angle {train_metrics['angle_loss']:.4f}  |  "
            f"val_loss {val_metrics['val_loss']:.4f}  "
            f"lr {lr:.2e}"
        )

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_warmup.pt")

    # ── phase 2: QAT fine-tune ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: QAT fine-tune ({cfg['epochs_qat']} epochs)")

    # Reload best warmup weights
    model.load_state_dict(torch.load(run_dir / "best_warmup.pt",
                                      map_location=device))

    # Reparameterise if applicable (EdgeVGG / QARepVGG)
    if hasattr(model, "switch_to_deploy"):
        print("  Switching to deploy mode (fusing multi-branch convs)...")
        model = model.switch_to_deploy().to(device)
        model_summary(model)

    # ── Attach fake-quantisation wrappers ────────────────────────────────
    #   Only for QARepVGG-Mini; for other models, skip or customise.
    if "qarepvgg" in cfg["model"]:
        model.qconfig = default_qat_qconfig  # FakeQuantise with QAT
        model = prepare_qat(model, inplace=False)
        print("  QAT fake-quantisation enabled.")
    else:
        print("  QAT not implemented for this model (FP32 export only).")

    model = model.to(device)
    qat_optimizer = optim.AdamW(model.parameters(), lr=cfg["qat_lr"],
                                 weight_decay=cfg["weight_decay"])

    for epoch in range(1, cfg["epochs_qat"] + 1):
        # Flag QAT observers for training (some frameworks require this)
        if hasattr(model, "apply"):
            model.apply(torch.ao.quantization.enable_observer)
            model.apply(torch.ao.quantization.enable_fake_quant)

        train_metrics = train_one_epoch(
            model, train_loader, qat_optimizer, criterion, device, cfg,
            epoch + cfg["epochs_warmup"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"QAT Epoch {epoch:2d}/{cfg['epochs_qat']}  "
            f"loss {train_metrics['loss']:.4f}  "
            f"heat {train_metrics['heatmap_loss']:.4f}  "
            f"box {train_metrics['box_loss']:.4f}  "
            f"angle {train_metrics['angle_loss']:.4f}  |  "
            f"val_loss {val_metrics['val_loss']:.4f}"
        )

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(model.state_dict(), run_dir / "best_qat.pt")

    # ── Export to ONNX (for ST Edge AI Core) ─────────────────────────────
    print(f"\n{'='*60}")
    print("Exporting to ONNX...")
    model.eval()
    if hasattr(model, "apply"):
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.ao.quantization.disable_fake_quant)

    dummy = torch.randn(1, 3, cfg["image_size"], cfg["image_size"]).to(device)
    onnx_path = run_dir / f"{cfg['model']}_int8.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["heatmap", "box_size", "angle"],
        opset_version=16,
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "model": cfg["model"],
        "params_fp32": sum(p.numel() for p in model.parameters()),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "config": cfg,
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {run_dir / 'training_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
