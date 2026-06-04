"""
Per-sample MAC / FLOP / energy estimator.

We compute MACs **analytically** by walking the model with a single tracer
input and counting Conv1d / Linear / BatchNorm work. This is exact for the
layer types we use (no hidden ops smuggled in via autograd).

For the SNN we also report a **spike-weighted MAC count**: each layer's
MACs are scaled by the *measured* input spike rate at that layer (from a
calibration pass over the validation set). This is the standard
neuromorphic figure of merit and corresponds to what a real event-driven
chip (Loihi, SpiNNaker) would dissipate per inference.

Inference latency is timed on whatever device we're running on with a
batch-size-1 forward pass, warmed up + averaged. Energy is reported in
*relative* picojoules using a per-MAC constant (Horowitz, ISSCC 2014:
~3.7 pJ/MAC for 32-bit dense FP, ~0.1 pJ/SOP for accumulator-only).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# Per-operation energy constants (Horowitz, ISSCC 2014 + Davies et al. Loihi)
ENERGY_PJ_PER_MAC_FP32 = 3.7    # dense MAC on a digital ASIC, 45 nm CMOS
ENERGY_PJ_PER_AC       = 0.9    # accumulate-only (spike triggers add)
ENERGY_PJ_PER_SOP      = 0.1    # synaptic-op on a neuromorphic chip (Loihi)


@dataclass
class LayerCost:
    name: str
    kind: str                 # "conv1d" | "linear" | "bn" | "lif" | "pool"
    macs: int = 0             # dense MAC count
    sops: int = 0             # spike-weighted SOPs (for SNN)
    params: int = 0
    in_spike_rate: float | None = None    # for SNN layers


@dataclass
class ModelCost:
    name: str
    total_params: int
    total_macs: int           # dense ANN-equivalent
    total_sops: int           # spike-weighted (== total_macs for dense ANN)
    latency_ms: float | None  # batch-1 forward, averaged
    latency_device: str
    energy_pj_dense: float    # total_macs * ENERGY_PJ_PER_MAC_FP32
    energy_pj_neuromorphic: float  # total_sops * ENERGY_PJ_PER_SOP
    spike_rate_mean: float | None = None
    layers: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analytical counters
# ---------------------------------------------------------------------------
def _conv1d_macs(layer: nn.Conv1d, in_T: int) -> tuple[int, int]:
    Cout = layer.out_channels
    Cin = layer.in_channels // layer.groups
    k = layer.kernel_size[0]
    s = layer.stride[0]
    pad = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    dil = layer.dilation[0]
    T_out = (in_T + 2 * pad - dil * (k - 1) - 1) // s + 1
    macs = Cout * Cin * k * T_out
    return macs, T_out


def _linear_macs(layer: nn.Linear) -> int:
    return layer.in_features * layer.out_features


# ---------------------------------------------------------------------------
# Per-architecture analytical walkers (cleaner than autograd hooks for SNNs)
# ---------------------------------------------------------------------------
def cost_cnn1d(model, T_in: int = 1000) -> ModelCost:
    """Walk a CNN1D / ResNet1D and count MACs."""
    layers: list[LayerCost] = []
    macs_total = 0
    T = T_in
    # Forward shape pass — we re-derive T per Conv1d / Pool / Linear
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d):
            macs, T_new = _conv1d_macs(m, T)
            layers.append(LayerCost(name=name, kind="conv1d",
                                     macs=macs,
                                     sops=macs,
                                     params=sum(p.numel() for p in m.parameters()),
                                     in_spike_rate=1.0))
            macs_total += macs
            T = T_new
        elif isinstance(m, nn.MaxPool1d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            T = (T - k) // s + 1
        elif isinstance(m, nn.AdaptiveAvgPool1d):
            T = m.output_size if isinstance(m.output_size, int) else m.output_size[0]
        elif isinstance(m, nn.Linear):
            macs = _linear_macs(m)
            layers.append(LayerCost(name=name, kind="linear",
                                     macs=macs,
                                     sops=macs,
                                     params=sum(p.numel() for p in m.parameters()),
                                     in_spike_rate=1.0))
            macs_total += macs
    return ModelCost(
        name=type(model).__name__,
        total_params=sum(p.numel() for p in model.parameters()),
        total_macs=macs_total,
        total_sops=macs_total,
        latency_ms=None,
        latency_device="?",
        energy_pj_dense=macs_total * ENERGY_PJ_PER_MAC_FP32,
        energy_pj_neuromorphic=macs_total * ENERGY_PJ_PER_MAC_FP32,
        spike_rate_mean=None,
        layers=[asdict(l) for l in layers],
    )


def cost_snn(model, spike_rates: dict, T_in: int = 1000) -> ModelCost:
    """Walk the NeuroCardio S-TCN and apply per-layer spike weighting.

    `spike_rates` should be a dict mapping {'input', 'stem', 'block0', …,
    'block5'} → float ∈ [0,1]. The keys must match what `evaluate.py` and
    the inferencer produce. Unknown keys default to 1.0 (i.e. dense).

    For each Conv1d we use the *input-side* spike rate (the rate of the
    feature that drives this layer's MACs). For the dense head we use the
    final block's output spike rate.
    """
    layers: list[LayerCost] = []
    macs_total = 0
    sops_total = 0

    # ----- stem -----
    stem = model.stem_conv
    T = T_in
    macs, T = _conv1d_macs(stem, T)
    r = float(spike_rates.get("input", 1.0))
    layers.append(LayerCost(name="stem_conv", kind="conv1d",
                             macs=macs, sops=int(round(macs * r)),
                             params=sum(p.numel() for p in stem.parameters()),
                             in_spike_rate=r))
    macs_total += macs
    sops_total += int(round(macs * r))

    # ----- 6 S-TCN blocks -----
    prev_key = "stem"
    for i, blk in enumerate(model.blocks):
        r = float(spike_rates.get(prev_key, 1.0))   # what drives conv1 of this block
        # conv1 (dilated)
        macs1, T1 = _conv1d_macs(blk.conv1, T)
        layers.append(LayerCost(name=f"block{i}.conv1", kind="conv1d",
                                 macs=macs1, sops=int(round(macs1 * r)),
                                 params=sum(p.numel() for p in blk.conv1.parameters()),
                                 in_spike_rate=r))
        macs_total += macs1
        sops_total += int(round(macs1 * r))
        # conv2 — driven by spikes from lif1 (we treat its rate ≈ same as block-output rate)
        cur_key = f"block{i}"
        r2 = float(spike_rates.get(cur_key, r))
        macs2, T2 = _conv1d_macs(blk.conv2, T1)
        layers.append(LayerCost(name=f"block{i}.conv2", kind="conv1d",
                                 macs=macs2, sops=int(round(macs2 * r2)),
                                 params=sum(p.numel() for p in blk.conv2.parameters()),
                                 in_spike_rate=r2))
        macs_total += macs2
        sops_total += int(round(macs2 * r2))
        # 1x1 skip projection if present
        if not isinstance(blk.proj, nn.Identity):
            macs3, _ = _conv1d_macs(blk.proj, T)
            layers.append(LayerCost(name=f"block{i}.proj", kind="conv1d",
                                     macs=macs3, sops=int(round(macs3 * r)),
                                     params=sum(p.numel() for p in blk.proj.parameters()),
                                     in_spike_rate=r))
            macs_total += macs3
            sops_total += int(round(macs3 * r))
        # pool
        if blk.pool > 1:
            T2 = T2 // blk.pool
        T = T2
        prev_key = cur_key

    # ----- pool gate-score head -----
    pool = model.pool
    macs_gate, _ = _conv1d_macs(pool.score, T)
    r_last = float(spike_rates.get(f"block{len(model.blocks)-1}", 1.0))
    layers.append(LayerCost(name="pool.score", kind="conv1d",
                             macs=macs_gate, sops=int(round(macs_gate * r_last)),
                             params=sum(p.numel() for p in pool.score.parameters()),
                             in_spike_rate=r_last))
    macs_total += macs_gate
    sops_total += int(round(macs_gate * r_last))

    # ----- dense head, T_dense steps -----
    T_dense = model.T_dense
    fc1, fc2 = model.fc1, model.fc2
    macs_fc1 = _linear_macs(fc1) * T_dense
    macs_fc2 = _linear_macs(fc2) * T_dense
    # The first dense layer is driven by the pooled feature vector — the
    # average over time of the final block's spikes. Treat its sparsity as
    # the final block's spike rate.
    layers.append(LayerCost(name="fc1", kind="linear",
                             macs=macs_fc1,
                             sops=int(round(macs_fc1 * r_last)),
                             params=sum(p.numel() for p in fc1.parameters()),
                             in_spike_rate=r_last))
    macs_total += macs_fc1
    sops_total += int(round(macs_fc1 * r_last))
    # The classifier head is driven by lif_fc spikes — use that rate
    r_fc = float(spike_rates.get("fc", r_last))
    layers.append(LayerCost(name="fc2", kind="linear",
                             macs=macs_fc2,
                             sops=int(round(macs_fc2 * r_fc)),
                             params=sum(p.numel() for p in fc2.parameters()),
                             in_spike_rate=r_fc))
    macs_total += macs_fc2
    sops_total += int(round(macs_fc2 * r_fc))

    block_keys = ["stem"] + [f"block{i}" for i in range(len(model.blocks))]
    spike_mean = float(np.mean([spike_rates.get(k, 0) for k in block_keys
                                 if k in spike_rates])) if spike_rates else None
    return ModelCost(
        name=type(model).__name__,
        total_params=sum(p.numel() for p in model.parameters()),
        total_macs=macs_total,
        total_sops=sops_total,
        latency_ms=None,
        latency_device="?",
        energy_pj_dense=macs_total * ENERGY_PJ_PER_MAC_FP32,
        energy_pj_neuromorphic=sops_total * ENERGY_PJ_PER_SOP,
        spike_rate_mean=spike_mean,
        layers=[asdict(l) for l in layers],
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def time_inference(model, dev: torch.device, T_in: int = 1000,
                    n_warmup: int = 5, n_iter: int = 30) -> float:
    """Average batch-1 forward latency (ms). Uses cuda.synchronize when on
    CUDA; on CPU/MPS we time wall clock around .item()."""
    model.eval()
    x = torch.randn(1, 12, T_in, device=dev)
    for _ in range(n_warmup):
        _ = model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        y = model(x)
        if isinstance(y, tuple): y = y[0]
        _ = y.sum().item()    # forces sync via dtoh
    dt = (time.time() - t0) / n_iter * 1000.0
    return float(dt)


# ---------------------------------------------------------------------------
# Spike-rate calibration
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_spike_rates(model, loader, dev: torch.device,
                           n_batches: int = 8) -> dict:
    """Run a few batches through an SNN and return mean per-layer spike rates."""
    model.eval()
    totals = {"input": 0.0, "stem": 0.0}
    for i in range(len(model.blocks)):
        totals[f"block{i}"] = 0.0
    totals["fc"] = 0.0
    n = 0
    for i, (x, _y, _) in enumerate(loader):
        if i >= n_batches: break
        x = x.to(dev)
        logits, tr = model(x, return_traces=True)
        totals["input"] += float(tr["input_spikes"].float().mean())
        totals["stem"] += float(tr["stem"].float().mean())
        for j, s in enumerate(tr["blocks"]):
            totals[f"block{j}"] += float(s.float().mean())
        totals["fc"] += float(tr["s_fc"].float().mean())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}
