"""
Delta-modulation spike encoder for multi-lead ECG.

For each of the 12 leads we maintain a moving reference voltage `ref`. We emit
an ON spike whenever the lead has risen by ≥ θ since the last event, an OFF
spike whenever it has dropped by ≥ θ. The reference is then snapped to the
new value (one quantum at a time). The result is a 24-channel binary tensor
(ON_lead0 … ON_lead11, OFF_lead0 … OFF_lead11) that fully reconstructs the
analog signal up to ±θ quantization — i.e. it is information-preserving in
the same way a silicon-retina (DVS) encoding is for vision.
"""
from __future__ import annotations

import numpy as np
import torch


def delta_encode_batch_12lead(x: torch.Tensor, theta: float = 0.15) -> torch.Tensor:
    """
    x: (B, 12, T) float tensor
    returns: (B, 24, T) binary spike tensor:
       channels [0..11]  = ON  spikes for leads 0..11
       channels [12..23] = OFF spikes for leads 0..11
    Run on whatever device x lives on (MPS / CUDA / CPU).
    """
    B, C, T = x.shape
    assert C == 12, f"expected 12 leads, got {C}"
    out = torch.zeros(B, 2 * C, T, device=x.device, dtype=x.dtype)
    ref = x[:, :, 0].clone()                              # (B, 12)
    for t in range(T):
        diff = x[:, :, t] - ref
        on = (diff >= theta).float()
        off = (diff <= -theta).float()
        out[:, :C, t] = on
        out[:, C:, t] = off
        ref = ref + on * theta - off * theta
    return out


def delta_encode_numpy_12lead(x: np.ndarray, theta: float = 0.15) -> np.ndarray:
    """Numpy version for offline/web export.  x: (12, T) → (24, T)."""
    C, T = x.shape
    out = np.zeros((2 * C, T), dtype=np.float32)
    ref = x[:, 0].copy().astype(np.float32)
    for t in range(T):
        diff = x[:, t] - ref
        on = (diff >= theta).astype(np.float32)
        off = (diff <= -theta).astype(np.float32)
        out[:C, t] = on
        out[C:, t] = off
        ref = ref + on * theta - off * theta
    return out
