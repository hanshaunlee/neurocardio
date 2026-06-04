"""
S-TCN: Spiking Temporal Convolutional Network for multi-label diagnostic ECG.

Design
======
The signal is encoded into spikes once (delta modulation, 24 channels = 12
leads × {ON, OFF}). It then flows through:

  * a strided spiking stem that halves the time axis once,
  * six dilated-conv spiking blocks (dilations 1, 2, 4, 8, 16, 32) with
    interleaved temporal max-pooling, so the receptive field grows
    exponentially while the time axis shrinks to ≈ T/16,
  * a *spiking attention pool* that emits gate spikes from a small LIF head
    and aggregates features only at the time-steps where the gate fires,
  * a dense head with one more LIF layer feeding a non-leaky integrator;
    the integrator's membrane potential, accumulated for T_dense extra
    steps, is the per-class logit (multi-label, sigmoid loss).

Two design choices are non-standard and constitute the novelty of this
work:

  (a) **Membrane-gated residuals.** Plain spike-level residual addition
      would mix binary inputs and breaks the binary spike abstraction.
      Instead each block injects its skip path into the *membrane
      potential* of its output LIF, scaled by a learnable per-channel
      sigmoid gate g = σ(α).  Spikes out of every block remain binary.
      Initialization α=0 → g=0.5 lets the optimizer decide per channel
      whether the block should be predominantly residual-pass or
      transform-dominated.

  (b) **Spiking gate-attention pooling.** A 1-D convolutional gate head
      produces a per-time-step current; a single LIF integrates it and
      fires when accumulated evidence crosses threshold. The pooled
      feature is the mean of feature spikes at the time-steps where the
      gate LIF fired (or, if it never fires, the global mean). This is a
      strictly event-driven pooling — no softmax, no continuous-valued
      mixing.

All LIF dynamics are unrolled in time on the spatial-time axis. Conv1d
acts as a temporally-shared linear operator (no per-step weights), so
gradients flow through BPTT only via the LIF state and the surrogate
spike function (fast sigmoid, slope = 25, Neftci et al. 2019).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def _lif(beta: float = 0.9, slope: float = 25, reset="subtract"):
    return snn.Leaky(beta=beta,
                     spike_grad=surrogate.fast_sigmoid(slope=slope),
                     init_hidden=False,
                     reset_mechanism=reset)


def _run_lif_along_time(lif: snn.Leaky, current: torch.Tensor) -> torch.Tensor:
    """current: (B, C, T) — drive LIF along the T axis, return spikes (B, C, T)."""
    mem = lif.init_leaky()
    out = torch.zeros_like(current)
    for t in range(current.shape[-1]):
        sp, mem = lif(current[:, :, t], mem)
        out[:, :, t] = sp
    return out


class STCNBlock(nn.Module):
    """Two-conv dilated spiking residual block with a membrane-gated skip."""
    def __init__(self, c_in: int, c_out: int, dilation: int, pool: int = 1,
                 beta: float = 0.9):
        super().__init__()
        self.pool = pool
        # Main path
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3,
                                padding=dilation, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.lif1  = _lif(beta=beta)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(c_out)
        # Skip-path projector: 1x1 if channel change, otherwise identity
        self.proj = (nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
                     if c_in != c_out else nn.Identity())
        # Learnable per-channel membrane gate (σ(α), init α=0 → g=0.5)
        self.gate_log = nn.Parameter(torch.zeros(c_out))
        # Output LIF integrates main-path current + gated skip-path current
        self.lif2 = _lif(beta=beta)
        if pool > 1:
            self.tpool = nn.MaxPool1d(pool)
        else:
            self.tpool = nn.Identity()

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        # s_in: (B, C_in, T) binary spikes
        # Main path
        h1 = self.bn1(self.conv1(s_in))
        s1 = _run_lif_along_time(self.lif1, h1)
        h2 = self.bn2(self.conv2(s1))                      # (B, C_out, T)
        # Skip path: project then mix at the membrane level
        skip = self.proj(s_in)                              # (B, C_out, T)
        g = torch.sigmoid(self.gate_log).view(1, -1, 1)     # (1, C_out, 1)
        mem_current = g * h2 + (1.0 - g) * skip             # (B, C_out, T)
        s2 = _run_lif_along_time(self.lif2, mem_current)
        return self.tpool(s2)


class SpikingAttentionPool(nn.Module):
    """Gate-driven event-pooling along time."""
    def __init__(self, c: int, beta: float = 0.9):
        super().__init__()
        self.score = nn.Conv1d(c, 1, kernel_size=1)
        self.score_bn = nn.BatchNorm1d(1)
        self.gate_lif = _lif(beta=beta)

    def forward(self, s: torch.Tensor):
        # s: (B, C, T) feature spikes
        cur = self.score_bn(self.score(s))                  # (B, 1, T)
        gate = _run_lif_along_time(self.gate_lif, cur)      # (B, 1, T) {0,1}
        # Avoid div-by-zero: if a sample has no gate spikes, fall back to mean
        denom = gate.sum(dim=-1, keepdim=True)              # (B, 1, 1)
        used_gate = denom > 0
        gate_used = torch.where(used_gate.expand_as(gate), gate,
                                torch.ones_like(gate) / s.shape[-1])
        denom_used = gate_used.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = (s * gate_used).sum(dim=-1) / denom_used.squeeze(-1)  # (B, C)
        return pooled, gate


class NeuroCardio(nn.Module):
    """Top-level model. Input is the raw 12-lead signal; the model encodes,
    runs the spiking stack, and returns multi-label logits."""

    def __init__(self,
                 num_classes: int = 5,
                 beta: float = 0.9,
                 T_dense: int = 8,
                 stem_channels: int = 64,
                 channel_plan=(64, 128, 128, 256, 256, 384),
                 dilation_plan=(1, 2, 4, 8, 16, 32),
                 pool_plan=(1, 2, 1, 2, 1, 2)):
        super().__init__()
        assert len(channel_plan) == len(dilation_plan) == len(pool_plan)
        self.num_classes = num_classes
        self.T_dense = T_dense

        # Stem: Conv stride-2 along time
        self.stem_conv = nn.Conv1d(24, stem_channels, kernel_size=15,
                                    stride=2, padding=7)
        self.stem_bn   = nn.BatchNorm1d(stem_channels)
        self.stem_lif  = _lif(beta=beta)

        # Six S-TCN blocks
        in_ch = stem_channels
        blocks = []
        for c, d, p in zip(channel_plan, dilation_plan, pool_plan):
            blocks.append(STCNBlock(in_ch, c, dilation=d, pool=p, beta=beta))
            in_ch = c
        self.blocks = nn.ModuleList(blocks)

        # Spiking attention pool
        self.pool = SpikingAttentionPool(in_ch, beta=beta)

        # Classifier head
        self.fc1 = nn.Linear(in_ch, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.lif_fc = _lif(beta=beta)
        self.fc2 = nn.Linear(256, num_classes)
        # Non-leaky integrator readout (logits = accumulated V_mem)
        self.lif_out = _lif(beta=1.0, reset="none")

    # spike encoding stays on the same device as x; defined here so the model
    # is self-contained for inference
    @staticmethod
    def delta_encode(x: torch.Tensor, theta: float = 0.15) -> torch.Tensor:
        from neurocardio.encoding import delta_encode_batch_12lead
        return delta_encode_batch_12lead(x, theta=theta)

    def forward(self, x: torch.Tensor, theta: float = 0.15,
                return_traces: bool = False):
        """
        x: (B, 12, T) float ECG (already normalized)
        returns: logits (B, num_classes) for sigmoid multi-label BCE.
        """
        spikes = self.delta_encode(x, theta=theta)          # (B, 24, T)

        # Stem
        h = self.stem_bn(self.stem_conv(spikes))            # (B, C, T/2)
        s = _run_lif_along_time(self.stem_lif, h)

        traces = {"input_spikes": spikes, "stem": s, "blocks": []} if return_traces else None
        for blk in self.blocks:
            s = blk(s)
            if return_traces:
                traces["blocks"].append(s)

        pooled, gate = self.pool(s)                         # (B, C_last)
        if return_traces:
            traces["pool_gates"] = gate
            traces["pooled"] = pooled

        # Dense classifier head — accumulate evidence for T_dense steps
        cur = self.fc1_bn(self.fc1(pooled))                 # (B, 256)
        mem_fc  = self.lif_fc.init_leaky()
        mem_out = self.lif_out.init_leaky()
        logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        vout_trace = [] if return_traces else None
        s_fc_trace = [] if return_traces else None
        for _ in range(self.T_dense):
            s_fc, mem_fc  = self.lif_fc(cur, mem_fc)
            _,    mem_out = self.lif_out(self.fc2(s_fc), mem_out)
            logits = logits + mem_out
            if return_traces:
                vout_trace.append(mem_out.detach())
                s_fc_trace.append(s_fc.detach())

        if return_traces:
            traces["s_fc"] = torch.stack(s_fc_trace, dim=-1)
            traces["vout"] = torch.stack(vout_trace, dim=-1)
            return logits, traces
        return logits


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
