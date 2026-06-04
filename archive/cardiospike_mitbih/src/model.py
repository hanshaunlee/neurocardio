"""
Convolutional Spiking Neural Network for ECG beat classification.

All hidden units are leaky-integrate-and-fire neurons with surrogate-gradient
backprop through time. Conv1d acts as a temporally-shared linear operator;
the LIF neurons supply the time dynamics via their membrane potential.

  delta-encoded ECG (T=260, C=2)
    Conv1d(2→16, k=7) → LIF → MaxPool(2)
    Conv1d(16→32, k=5) → LIF → MaxPool(2)
    Conv1d(32→64, k=3) → LIF → MaxPool(2)
    Flatten → Linear(2048→128) → LIF
    Linear(128→5) → non-leaky readout (membrane potential = logits)

The classifier head runs for T_dense extra SNN timesteps so evidence can
accumulate in the output membrane.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class CardioSpikeSNN(nn.Module):
    def __init__(self, num_classes: int = 5, beta: float = 0.9, T_dense: int = 12,
                 rr_dim: int = 4):
        super().__init__()
        self.T_dense = T_dense
        self.rr_dim = rr_dim
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, padding=3)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.pool3 = nn.MaxPool1d(2)

        # 260 → 130 → 65 → 32 with 64 channels, + RR features
        self.flat_dim = 64 * 32 + rr_dim
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=1.0, spike_grad=spike_grad,
                                  init_hidden=False, reset_mechanism="none")

    def forward(self, spikes_TBC, rr=None, return_traces: bool = False):
        T, B, _ = spikes_TBC.shape
        device = spikes_TBC.device

        x = spikes_TBC.permute(1, 2, 0)                # (B, 2, T)

        h1 = self.pool1(self.conv1(x))                 # (B, 16, T1)
        mem1 = self.lif1.init_leaky()
        s1 = torch.zeros_like(h1)
        for t in range(h1.shape[-1]):
            sp, mem1 = self.lif1(h1[:, :, t], mem1)
            s1[:, :, t] = sp

        h2 = self.pool2(self.conv2(s1))                # (B, 32, T2)
        mem2 = self.lif2.init_leaky()
        s2 = torch.zeros_like(h2)
        for t in range(h2.shape[-1]):
            sp, mem2 = self.lif2(h2[:, :, t], mem2)
            s2[:, :, t] = sp

        h3 = self.pool3(self.conv3(s2))                # (B, 64, T3)
        mem3 = self.lif3.init_leaky()
        s3 = torch.zeros_like(h3)
        for t in range(h3.shape[-1]):
            sp, mem3 = self.lif3(h3[:, :, t], mem3)
            s3[:, :, t] = sp

        flat_conv = s3.reshape(B, -1)                  # (B, 64*T3)
        if rr is None:
            rr = torch.zeros(B, self.rr_dim, device=device)
        flat = torch.cat([flat_conv, rr], dim=1)       # (B, flat_dim)

        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()
        logits = torch.zeros(B, self.fc2.out_features, device=device)
        traces = {"s1": s1, "s2": s2, "s3": s3, "s4": [], "vout": []} if return_traces else None

        for _ in range(self.T_dense):
            sp4, mem4 = self.lif4(self.fc1(flat), mem4)
            sp_out, mem_out = self.lif_out(self.fc2(sp4), mem_out)
            logits = logits + mem_out
            if return_traces:
                traces["s4"].append(sp4.detach())
                traces["vout"].append(mem_out.detach())

        if return_traces:
            traces["s4"] = torch.stack(traces["s4"], dim=-1)
            traces["vout"] = torch.stack(traces["vout"], dim=-1)
            return logits, traces
        return logits


def spike_rates(traces) -> dict:
    """Fraction of LIF units firing per timestep, per layer (sparsity probe)."""
    out = {}
    for k in ("s1", "s2", "s3", "s4"):
        s = traces[k]
        out[k] = float(s.float().mean().item())
    return out
