"""
Joint loss for anchor-free OBB regression (PyTorch).

Loss terms:
  1. Heatmap: modified focal loss on 2D Gaussian centre peak
  2. Box size: Smooth-L1 (Huber) at peak locations only
  3. Angle (sin2θ, cos2θ): Smooth-L1 at peak locations only
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_radius(bbox_size: torch.Tensor, min_overlap: float = 0.7) -> torch.Tensor:
    """Compute Gaussian radius from bounding-box size.  (Simplified)
    
    For a target heatmap we can pre-compute this offline;
    this function is a placeholder for per-batch computation if needed.
    """
    return torch.max(bbox_size[..., 0], bbox_size[..., 1]) * 0.15  # heuristic


def _smooth_l1(pred: torch.Tensor, target: torch.Tensor,
               beta: float = 1.0) -> torch.Tensor:
    """Huber / Smooth-L1 element-wise."""
    diff = (pred - target).abs()
    loss = torch.where(diff < beta,
                       0.5 * diff.square() / beta,
                       diff - 0.5 * beta)
    return loss


class HeatmapFocalLoss(nn.Module):
    """Modified focal loss for 2D Gaussian centre heatmap.

    L_heat = -1/N * Σ (1 - p̂)^γ * log(p̂)   where target = 1  (peak)
             -1/N * Σ     p̂^γ    * log(1-p̂)  where target = 0  (background)

    With a modulating factor that down-weights easy negatives.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha      # focal power for positives
        self.beta = beta        # focal power for negatives
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, 1, H, W)  — logits, not sigmoid-activated."""
        pred = torch.sigmoid(pred)
        target = target.float()

        # Positive locations
        pos_mask = target > 0.5
        pos_loss = -torch.where(
            pos_mask,
            (1 - pred).pow(self.alpha) * target.log().clamp(min=self.eps),
            torch.zeros_like(pred),
        ).sum()

        # Negative locations with hard-mining
        neg_mask = target < 0.5
        neg_weight = (1 - target).pow(self.beta)
        neg_loss = -torch.where(
            neg_mask,
            neg_weight * pred.pow(self.alpha) * (1 - pred + self.eps).log(),
            torch.zeros_like(pred),
        ).sum()

        N = max(pos_mask.sum().item(), 1)
        return (pos_loss + neg_loss) / N


class PeakHuberLoss(nn.Module):
    """Smooth-L1 regression loss evaluated only at peak locations.

    For box size (w, h) and angle (sin2θ, cos2θ) — only compute
    where the ground-truth heatmap has its centre peak.
    """
    def __init__(self, beta: float = 1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                heatmap_target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, C, H, W)
        heatmap_target: (B, 1, H, W) — ground-truth centre map
        """
        peak_mask = (heatmap_target > 0.5).float()
        N = max(peak_mask.sum().item(), 1)

        loss = _smooth_l1(pred, target, beta=self.beta)
        loss = (loss * peak_mask).sum() / N
        return loss


class JointOBBLoss(nn.Module):
    """Combined loss: heatmap focal + box-size Huber + angle Huber.

    Returns dict with individual losses and total.
    """
    def __init__(self, heatmap_alpha: float = 2.0, heatmap_beta: float = 4.0,
                 huber_beta: float = 1.0 / 9.0,
                 w_heat: float = 1.0, w_box: float = 1.0, w_angle: float = 1.0):
        super().__init__()
        self.heatmap_loss = HeatmapFocalLoss(alpha=heatmap_alpha, beta=heatmap_beta)
        self.box_loss = PeakHuberLoss(beta=huber_beta)
        self.angle_loss = PeakHuberLoss(beta=huber_beta)
        self.w_heat = w_heat
        self.w_box = w_box
        self.w_angle = w_angle

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target_heatmap: torch.Tensor,
        target_box_size: torch.Tensor,
        target_angle: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        pred:             {"heatmap": (B,1,H,W), "box_size": (B,2,H,W), "angle": (B,2,H,W)}
        target_heatmap:   (B,1,H,W)  — ground-truth centre Gaussian
        target_box_size:  (B,2,H,W)  — ground-truth w, h (normalised)
        target_angle:     (B,2,H,W)  — ground-truth sin(2θ), cos(2θ)
        """
        l_heat = self.heatmap_loss(pred["heatmap"], target_heatmap)
        l_box = self.box_loss(pred["box_size"], target_box_size, target_heatmap)
        l_angle = self.angle_loss(pred["angle"], target_angle, target_heatmap)

        total = self.w_heat * l_heat + self.w_box * l_box + self.w_angle * l_angle
        return {
            "loss": total,
            "heatmap_loss": l_heat,
            "box_loss": l_box,
            "angle_loss": l_angle,
        }
