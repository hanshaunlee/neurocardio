"""
ECG data augmentations.

All operations preserve the (12, T) lead/time layout and the temporal
calibration — no horizontal flip (would reverse the QRS), no large time
warps. The augmentations chosen are conservative, physiologically plausible
perturbations that simulate measurement nuisance (sensor noise, brief lead
disconnects, baseline wander, sampling jitter).
"""
from __future__ import annotations

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class GaussianNoise:
    def __init__(self, sigma: float = 0.05, p: float = 0.5):
        self.sigma = sigma
        self.p = p
    def __call__(self, x):
        if np.random.rand() > self.p: return x
        return x + np.random.randn(*x.shape).astype(x.dtype) * self.sigma


class LeadDropout:
    """Zero a random subset of the 12 leads — simulates electrode disconnect.
    Forces the network to use redundancy across leads."""
    def __init__(self, max_n: int = 3, p: float = 0.3):
        self.max_n = max_n
        self.p = p
    def __call__(self, x):
        if np.random.rand() > self.p: return x
        n = np.random.randint(1, self.max_n + 1)
        idx = np.random.choice(x.shape[0], size=n, replace=False)
        x = x.copy()
        x[idx] = 0
        return x


class BaselineWander:
    """Add a low-frequency sinusoid to mimic respiratory baseline drift."""
    def __init__(self, amp: float = 0.2, freq_hz_range=(0.15, 0.35),
                 fs: int = 100, p: float = 0.4):
        self.amp = amp
        self.freq_range = freq_hz_range
        self.fs = fs
        self.p = p
    def __call__(self, x):
        if np.random.rand() > self.p: return x
        T = x.shape[1]
        f = np.random.uniform(*self.freq_range)
        phi = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0, self.amp)
        t = np.arange(T) / self.fs
        return x + (amp * np.sin(2 * np.pi * f * t + phi)).astype(x.dtype)


class RandomCrop:
    """Crop a random sub-window of length `target_len` and pad back to T."""
    def __init__(self, target_len: int = 900, fs: int = 100, p: float = 0.5):
        self.target_len = target_len
        self.fs = fs
        self.p = p
    def __call__(self, x):
        if np.random.rand() > self.p: return x
        T = x.shape[1]
        if self.target_len >= T: return x
        start = np.random.randint(0, T - self.target_len)
        crop = x[:, start:start + self.target_len]
        pad = T - self.target_len
        left = np.random.randint(0, pad + 1)
        right = pad - left
        return np.pad(crop, ((0, 0), (left, right)), mode="constant")


def standard_train_augment(fs: int = 100):
    return Compose([
        GaussianNoise(sigma=0.05, p=0.6),
        BaselineWander(amp=0.2, fs=fs, p=0.4),
        LeadDropout(max_n=2, p=0.25),
        RandomCrop(target_len=int(fs * 9), fs=fs, p=0.4),
    ])
