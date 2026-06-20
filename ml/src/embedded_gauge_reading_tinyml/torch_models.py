"""
PyTorch backbones for STM32N6 Neural-ART NPU deployment.

All three models:
  - Accept (3, 256, 256) input.
  - Output three 32×32 regression maps (8× downsampling).
  - Use nn.ReLU only (no SiLU, Hard-Swish, ReLU6).
  - Fit ≤ 2.5 MB INT8 (~700k–900k FP32 params).

Architectures:
  1. PP-LCNet-0.5x (linearised – dense convs early, depthwise only at final block)
  2. EdgeVGG-Linear  (multi-branch reparam with expansion-contraction)
  3. QARepVGG-Mini   (QAT-friendly variant with residual scaling)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def _conv3x3(in_c: int, out_c: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, groups=groups, bias=False)


def _conv1x1(in_c: int, out_c: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 1, stride=stride, groups=groups, bias=False)


def _conv5x5(in_c: int, out_c: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 5, stride=stride, padding=2, groups=groups, bias=False)


class _BNReLU(nn.Sequential):
    def __init__(self, c: int):
        super().__init__(nn.BatchNorm2d(c), nn.ReLU(inplace=True))


class _ConvBlock(nn.Sequential):
    """Conv (dense or depthwise) → BN → ReLU."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                 stride: int = 1, depthwise: bool = False):
        groups = in_c if depthwise else 1
        if kernel_size == 1:
            conv = _conv1x1(in_c, out_c, stride=stride, groups=groups)
        elif kernel_size == 3:
            conv = _conv3x3(in_c, out_c, stride=stride, groups=groups)
        elif kernel_size == 5:
            conv = _conv5x5(in_c, out_c, stride=stride, groups=groups)
        else:
            raise ValueError(f"Unsupported kernel_size: {kernel_size}")
        super().__init__(OrderedDict([
            ("conv", conv),
            ("bn", nn.BatchNorm2d(out_c)),
            ("relu", nn.ReLU(inplace=True)),
        ]))


# ── Shared decoder + heads ───────────────────────────────────────────────────

class _DecoderHeads(nn.Module):
    """2× upsample (nearest) → 1×1 proj → parallel heads.

    Input:  16×16 feature map
    Output: 32×32 regression maps for heatmap (1), box_size (2), angle (2).
    """
    def __init__(self, feat_c: int, proj_c: int = 64):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.proj = _ConvBlock(feat_c, proj_c, kernel_size=1)
        self.heatmap = nn.Conv2d(proj_c, 1, 1)               # logits → sigmoid later
        self.box_size = nn.Conv2d(proj_c, 2, 1)               # (w, h)  spatial reg
        self.angle = nn.Conv2d(proj_c, 2, 1)                  # (sin2θ, cos2θ)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.up(x)
        x = self.proj(x)
        return {
            "heatmap": self.heatmap(x),
            "box_size": self.box_size(x),
            "angle": torch.tanh(self.angle(x)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: PP-LCNet-0.5x (Linearised)
# ═══════════════════════════════════════════════════════════════════════════════
# Channel plan:  Stem:16  S1:32  S2:64  S3:128  S4:128(dw) → 16×16 → 32×32
# Params: ~820k

class PPLCNetLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = _ConvBlock(3, 16, kernel_size=3, stride=2)                    # 128

        # Stage 1 – dense 3×3
        self.stage1 = nn.Sequential(
            _ConvBlock(16, 32, kernel_size=3, stride=2),                           # 64
            _ConvBlock(32, 32, kernel_size=3, stride=1),
        )

        # Stage 2 – dense 3×3
        self.stage2 = nn.Sequential(
            _ConvBlock(32, 64, kernel_size=3, stride=2),                           # 32
            _ConvBlock(64, 64, kernel_size=3, stride=1),
            _ConvBlock(64, 64, kernel_size=3, stride=1),
        )

        # Stage 3 – dense 3×3
        self.stage3 = nn.Sequential(
            _ConvBlock(64, 160, kernel_size=3, stride=2),                          # 16
            _ConvBlock(160, 160, kernel_size=3, stride=1),
            _ConvBlock(160, 160, kernel_size=3, stride=1),
        )

        # Stage 4 – depthwise 5×5 (deep channels only)
        self.stage4 = nn.Sequential(
            _ConvBlock(160, 160, kernel_size=5, stride=1, depthwise=True),         # 16
            _ConvBlock(160, 160, kernel_size=5, stride=1, depthwise=True),
        )

        self.head = _DecoderHeads(160, proj_c=64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: EdgeVGG-Linear (Structural Re-parameterisation)
# ═══════════════════════════════════════════════════════════════════════════════

class _RepVGGBlock(nn.Module):
    """
    Training: y = Conv3×3(x) + Conv1×1(x) + identity(x)   [if stride==1]
    Deploy:  single 3×3 Conv2d via switch_to_deploy().
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1,
                 use_identity: bool = True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.use_identity = use_identity and (in_c == out_c) and (stride == 1)

        self.conv3x3 = nn.Sequential(
            _conv3x3(in_c, out_c, stride=stride), nn.BatchNorm2d(out_c),
        )
        self.conv1x1 = nn.Sequential(
            _conv1x1(in_c, out_c, stride=stride), nn.BatchNorm2d(out_c),
        )
        if self.use_identity:
            self.bn_id = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3x3(x) + self.conv1x1(x)
        if self.use_identity:
            y = y + self.bn_id(x)
        return self.relu(y)

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        eps = bn.eps
        std = (var + eps).sqrt()
        w_fused = w * (gamma / std).view(-1, 1, 1, 1)
        b_fused = beta - (gamma * mean / std)
        return w_fused, b_fused

    def _pad_1x1_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        return F.pad(w, [1, 1, 1, 1])

    def _build_identity_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_id = torch.zeros(self.out_c, self.in_c, 3, 3,
                           device=self.conv3x3[0].weight.device)
        for i in range(min(self.in_c, self.out_c)):
            w_id[i, i, 1, 1] = 1.0
        mean, var = self.bn_id.running_mean, self.bn_id.running_var
        gamma, beta = self.bn_id.weight, self.bn_id.bias
        eps = self.bn_id.eps
        std = (var + eps).sqrt()
        w_fused = w_id * (gamma / std).view(-1, 1, 1, 1)
        b_fused = beta - (gamma * mean / std)
        return w_fused, b_fused

    def switch_to_deploy(self) -> nn.Conv2d:
        w3, b3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])
        w1, b1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        w1 = self._pad_1x1_to_3x3(w1)
        w_fused = w3 + w1
        b_fused = b3 + b1
        if self.use_identity:
            w_id, b_id = self._build_identity_kernel()
            w_fused = w_fused + w_id
            b_fused = b_fused + b_id
        conv = nn.Conv2d(self.in_c, self.out_c, 3,
                         stride=self.stride, padding=1, bias=True)
        conv.weight.data = w_fused
        conv.bias.data = b_fused
        return conv


class _EdgeVGGStage(nn.Module):
    """
    Expansion-contraction stage:
      1×1 expand (×mid_ratio) → N× RepVGG(expand_c) → 1×1 contract (→ out_c)
    """
    def __init__(self, in_c: int, out_c: int, n_blocks: int,
                 stride: int, mid_ratio: float = 1.5):
        super().__init__()
        mid_c = max(int(in_c * mid_ratio), out_c)
        self.expand = _ConvBlock(in_c, mid_c, kernel_size=1, stride=stride)
        blocks = []
        for i in range(n_blocks):
            blocks.append(_RepVGGBlock(mid_c, mid_c, stride=1,
                                        use_identity=True))
        self.blocks = nn.Sequential(*blocks)
        self.contract = _ConvBlock(mid_c, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.blocks(x)
        return self.contract(x)


class EdgeVGGLinar(nn.Module):
    """
    Channel plan:  Stem:24  S1:48(→72)  S2:64(→96)  S3:96(→144)  S4:128(→192→128, 2 blocks)
    Decoder: upsample stage-4 from 16×16 → 32×32.
    Params: ~830k FP32 / ~750k deploy.
    """
    def __init__(self):
        super().__init__()
        self.stem = _ConvBlock(3, 24, kernel_size=3, stride=2)                     # 128
        self.stage1 = _EdgeVGGStage(24, 48, n_blocks=2, stride=2, mid_ratio=1.5)    # 64
        self.stage2 = _EdgeVGGStage(48, 64, n_blocks=2, stride=2, mid_ratio=1.5)    # 32
        self.stage3 = _EdgeVGGStage(64, 96, n_blocks=2, stride=2, mid_ratio=1.5)    # 16
        self.stage4 = _EdgeVGGStage(96, 128, n_blocks=2, stride=1, mid_ratio=1.5)   # 16
        self.head = _DecoderHeads(128, proj_c=64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

    def switch_to_deploy(self) -> EdgeVGGLinarDeploy:
        """Convert all RepVGG blocks into single 3×3 convs."""
        return EdgeVGGLinarDeploy(self)


class EdgeVGGLinarDeploy(nn.Module):
    """Inference-only – all RepVGG blocks fused to single 3×3 convs."""
    def __init__(self, train_model: EdgeVGGLinar):
        super().__init__()
        self.head = train_model.head
        layers: list[nn.Module] = []
        for name, mod in train_model.named_children():
            if name == "head":
                continue
            if isinstance(mod, _EdgeVGGStage):
                layers.append(mod.expand)
                for block in mod.blocks:
                    layers.append(block.switch_to_deploy())
                    layers.append(nn.ReLU(inplace=True))
                layers.append(mod.contract)
            else:
                layers.append(mod)
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: QARepVGG-Mini (Quantization-Aware Re-Param)
# ═══════════════════════════════════════════════════════════════════════════════

class _QARepVGGBlock(nn.Module):
    """
    QAT-friendly RepVGG block with learnable residual scale α for
    stable INT8 weight distributions.

    Training:  y = Conv3×3(x) + Conv1×1(x) + α · BN_id(x)
    Deploy:    y = single fused 3×3 Conv2d(x)
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1,
                 use_identity: bool = True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.use_identity = use_identity and (in_c == out_c) and (stride == 1)

        self.conv3x3 = nn.Sequential(
            _conv3x3(in_c, out_c, stride=stride), nn.BatchNorm2d(out_c),
        )
        self.conv1x1 = nn.Sequential(
            _conv1x1(in_c, out_c, stride=stride), nn.BatchNorm2d(out_c),
        )
        if self.use_identity:
            self.bn_id = nn.BatchNorm2d(out_c)
            self.res_scale = nn.Parameter(torch.full((out_c,), 0.1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3x3(x) + self.conv1x1(x)
        if self.use_identity:
            y = y + self.res_scale.view(1, -1, 1, 1) * self.bn_id(x)
        return self.relu(y)

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        eps = bn.eps
        std = (var + eps).sqrt()
        w_fused = w * (gamma / std).view(-1, 1, 1, 1)
        b_fused = beta - (gamma * mean / std)
        return w_fused, b_fused

    def _pad_1x1_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        return F.pad(w, [1, 1, 1, 1])

    def _build_identity_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_id = torch.zeros(self.out_c, self.in_c, 3, 3,
                           device=self.conv3x3[0].weight.device)
        for i in range(min(self.in_c, self.out_c)):
            w_id[i, i, 1, 1] = 1.0
        mean, var = self.bn_id.running_mean, self.bn_id.running_var
        gamma, beta = self.bn_id.weight, self.bn_id.bias
        eps = self.bn_id.eps
        std = (var + eps).sqrt()
        w_fused = w_id * ((self.res_scale * gamma) / std).view(-1, 1, 1, 1)
        b_fused = beta - (gamma * mean / std)
        return w_fused, b_fused

    def switch_to_deploy(self) -> nn.Conv2d:
        w3, b3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])
        w1, b1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        w1 = self._pad_1x1_to_3x3(w1)
        w_fused = w3 + w1
        b_fused = b3 + b1
        if self.use_identity:
            w_id, b_id = self._build_identity_kernel()
            w_fused = w_fused + w_id
            b_fused = b_fused + b_id
        conv = nn.Conv2d(self.in_c, self.out_c, 3,
                         stride=self.stride, padding=1, bias=True)
        conv.weight.data = w_fused
        conv.bias.data = b_fused
        return conv


class _QAStage(nn.Module):
    """Sequential stage of QARepVGG blocks."""
    def __init__(self, in_c: int, out_c: int, n_blocks: int, stride: int):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            use_id = (s == 1)
            blocks.append(_QARepVGGBlock(
                in_c if i == 0 else out_c, out_c,
                stride=s, use_identity=use_id,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class QARepVGGAmin(nn.Module):
    """
    Channel plan:  Stem:32  S1:64  S2:96  S3:128  S4:160
    Decoder: upsample S4 (16×16) → 32×32.
    Params: ~800k.
    """
    def __init__(self):
        super().__init__()
        self.stem = _ConvBlock(3, 32, kernel_size=3, stride=2)                     # 128
        self.stage1 = _QAStage(32, 64, n_blocks=2, stride=2)                       # 64
        self.stage2 = _QAStage(64, 96, n_blocks=2, stride=2)                       # 32
        self.stage3 = _QAStage(96, 128, n_blocks=2, stride=2)                      # 16
        self.stage4 = _QAStage(128, 160, n_blocks=1, stride=1)                     # 16
        self.head = _DecoderHeads(160, proj_c=64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

    def switch_to_deploy(self) -> QARepVGGAminDeploy:
        return QARepVGGAminDeploy(self)


class QARepVGGAminDeploy(nn.Module):
    """Inference-only QARepVGG with fused convolutions."""
    def __init__(self, train_model: QARepVGGAmin):
        super().__init__()
        self.head = train_model.head
        layers: list[nn.Module] = []
        for name, mod in train_model.named_children():
            if name == "head":
                continue
            if isinstance(mod, _QAStage):
                for block in mod.blocks:
                    layers.append(block.switch_to_deploy())
                    layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(mod)
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone(x)
        return self.head(x)


# ── factory + helpers ─────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "pplcnet_linear": PPLCNetLinear,
    "edgevgg_linear": EdgeVGGLinar,
    "qarepvgg_mini":  QARepVGGAmin,
}


def build_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def model_summary(m: nn.Module) -> None:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Total params:     {total:>8,d}")
    print(f"  Trainable params: {trainable:>8,d}")
    print(f"  Est. INT8 size:   {total * 1.0 / 1e6:.2f} MB ({total} bytes)")
