"""Inference utilities: load a checkpoint, run an ECG, return rich traces for
the web demo."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurocardio.data import SUPERCLASSES, SUPERCLASS_NAMES
from neurocardio.encoding import delta_encode_numpy_12lead
from neurocardio.model import NeuroCardio


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CardioInferencer:
    """Single-purpose inference handle. Cache the model + labels in memory."""
    def __init__(self, ckpt_path: str | Path, label_set: str = "diagnostic_superclass"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        # Prefer labels persisted in the checkpoint; fall back to superclass.
        self.labels = ckpt.get("label_columns") or SUPERCLASSES
        self.label_names = SUPERCLASS_NAMES
        self.model = NeuroCardio(num_classes=len(self.labels))
        # EMA weights if available — they generalize better for inference
        state = ckpt.get("ema") or ckpt.get("state_dict")
        self.model.load_state_dict(state)
        self.dev = device()
        self.model.to(self.dev).eval()
        self.theta = cfg.get("theta", 0.15)

    @torch.no_grad()
    def infer(self, ecg_12_T: np.ndarray) -> dict[str, Any]:
        """ecg_12_T: (12, T) normalized float32 ECG. Returns probs + traces."""
        assert ecg_12_T.ndim == 2 and ecg_12_T.shape[0] == 12
        x = torch.from_numpy(ecg_12_T).float().unsqueeze(0).to(self.dev)
        logits, traces = self.model(x, theta=self.theta, return_traces=True)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Spike events per layer, in a JSON-friendly form
        out: dict[str, Any] = {
            "labels": self.labels,
            "label_names": self.label_names,
            "probs": {c: float(probs[i]) for i, c in enumerate(self.labels)},
            "spike_rates": {},
            "rasters": {},
        }
        # Input spike events (24 channels × T)
        in_sp = traces["input_spikes"][0].cpu().numpy()              # (24, T)
        out["input_spikes"] = self._raster_events(in_sp)
        out["spike_rates"]["input"] = float(in_sp.mean())
        # Per-block rasters: we keep the *last* block's spikes (most abstract).
        for i, s in enumerate(traces["blocks"]):
            arr = s[0].cpu().numpy()  # (C, T')
            out["spike_rates"][f"block{i}"] = float(arr.mean())
            if i == len(traces["blocks"]) - 1:
                out["rasters"]["block_last"] = self._raster_events(arr)
                out["rasters"]["block_last_shape"] = list(arr.shape)
        # Pool gate spikes (1, T')
        gate = traces["pool_gates"][0].cpu().numpy()                # (1, T')
        out["pool_gates"] = np.argwhere(gate > 0).astype(int).tolist()
        out["pool_gates_T"] = int(gate.shape[1])
        # Output integrator membrane potential over T_dense steps
        vout = traces["vout"][0].cpu().numpy()                       # (K, Td)
        out["vout"] = vout.tolist()
        return out

    @staticmethod
    def _raster_events(arr: np.ndarray) -> list[list[int]]:
        events = np.argwhere(arr > 0).astype(int)  # rows: (channel, time)
        return events.tolist()
