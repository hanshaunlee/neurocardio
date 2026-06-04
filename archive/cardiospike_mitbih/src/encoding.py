"""
Delta-modulation spike encoding.

This is the neuromorphic-native way to turn a continuous analog signal into
spikes: emit a +1 event when the signal has risen by `theta` since the last
event, -1 when it has fallen by `theta`, otherwise 0. The encoder is itself
event-driven and stateful — exactly what a silicon retina / cochlea does.

We split into two non-negative channels (ON, OFF) because LIF neurons spike
non-negatively. This is the standard DVS-style polarity convention.
"""
import numpy as np
import torch


def delta_encode_batch(x: torch.Tensor, theta: float = 0.15) -> torch.Tensor:
    """
    x: (B, T) float tensor of normalized signal
    returns: (T, B, 2) binary spike tensor (channels = [ON, OFF])
    """
    B, T = x.shape
    out = torch.zeros(T, B, 2, device=x.device, dtype=x.dtype)
    ref = x[:, 0].clone()
    for t in range(T):
        diff = x[:, t] - ref
        on = (diff >= theta).float()
        off = (diff <= -theta).float()
        out[t, :, 0] = on
        out[t, :, 1] = off
        # Reset reference on event by theta in the event direction
        ref = ref + on * theta - off * theta
    return out


def delta_encode_numpy(x: np.ndarray, theta: float = 0.15) -> np.ndarray:
    """Numpy version for offline export (visualization)."""
    T = x.shape[-1]
    out = np.zeros((T, 2), dtype=np.float32)
    ref = float(x[0])
    for t in range(T):
        diff = float(x[t]) - ref
        if diff >= theta:
            out[t, 0] = 1.0
            ref += theta
        elif diff <= -theta:
            out[t, 1] = 1.0
            ref -= theta
    return out
