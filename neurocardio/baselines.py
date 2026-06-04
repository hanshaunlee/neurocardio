"""
Dense (non-spiking) baselines for the PTB-XL multi-label benchmark.

Two reference architectures, both sized to ~1.9 M parameters so that the
comparison against `NeuroCardio` (1.89 M LIF parameters) is fair on a
per-weight basis:

  • `CNN1D`     — vanilla deep 1-D conv stack: BN+ReLU after every conv,
                  temporal max-pool at the same depths as the SNN, global
                  average pool over time, two-layer MLP head.
  • `ResNet1D`  — XResNet1D-style: pre-activation Conv→BN→ReLU residual
                  blocks with 1×1 projection skips when channels change,
                  strided downsampling at the same depths as the SNN.

Both consume the *raw* 12-lead ECG (B, 12, T=1000) — i.e. they bypass the
delta encoder.  An apples-to-apples ANN baseline should be allowed to use
the native representation.

The intent here is "a serious ECG CNN trained to convergence", not "a
hobbled control to make the SNN look good." If anything these baselines
favor the dense ANN: they use ReLU (smooth gradients), full real-valued
activations and standard residuals.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# CNN1D — straightforward dense CNN
# --------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, num_classes: int = 5,
                 channel_plan=(64, 128, 128, 256, 256, 384),
                 pool_plan=(1, 2, 1, 2, 1, 2)):
        super().__init__()
        assert len(channel_plan) == len(pool_plan)
        # Stem: stride-2 large-kernel
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = 64
        for c, p in zip(channel_plan, pool_plan):
            blocks += [
                nn.Conv1d(in_ch, c, kernel_size=3, padding=1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Conv1d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            if p > 1:
                blocks.append(nn.MaxPool1d(p))
            in_ch = c
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        # x: (B, 12, T)
        h = self.stem(x)
        h = self.blocks(h)
        h = self.gap(h).squeeze(-1)
        return self.head(h)


# --------------------------------------------------------------------------
# ResNet1D — XResNet1D-style with pre-activation residual blocks
# --------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3,
                                padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """An XResNet1D variant tuned to ~1.9 M parameters.

    The strided convolutions land at the same depths (blocks 2/4/6 → /2)
    as the SNN's max-pools, so the temporal receptive fields are
    comparable across the two models."""

    def __init__(self, num_classes: int = 5,
                 channel_plan=(64, 128, 128, 256, 256, 384),
                 stride_plan=(1, 2, 1, 2, 1, 2)):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # → T/4
        )
        blocks = []
        in_ch = 64
        for c, s in zip(channel_plan, stride_plan):
            blocks.append(ResBlock1D(in_ch, c, stride=s))
            in_ch = c
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.gap(h).squeeze(-1)
        return self.head(h)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
