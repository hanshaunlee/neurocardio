"""PyTorch Dataset wrappers around the PTB-XL cache produced by data.build_cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PTBXLCache(Dataset):
    """Reads the memory-mapped signals + label matrix produced by
    `neurocardio.data.build_cache`. Returns (signal, labels, fold)."""

    def __init__(self, cache_dir: str | Path, fold_subset: list[int] | None = None,
                 sampling_rate: int = 100, augment=None):
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.augment = augment
        self.signals = np.load(cache_dir / f"signals_{sampling_rate}.npy",
                               mmap_mode="r")
        meta = pd.read_csv(cache_dir / "labels.csv")
        with open(cache_dir / "label_columns.txt") as f:
            self.label_columns = [ln.strip() for ln in f if ln.strip()]

        if fold_subset is not None:
            mask = meta["strat_fold"].isin(fold_subset).to_numpy()
            self.indices = np.where(mask)[0]
        else:
            self.indices = np.arange(len(meta))
        self.meta = meta.iloc[self.indices].reset_index(drop=True)
        self.labels_np = self.meta[self.label_columns].to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        src = self.indices[i]
        sig = np.asarray(self.signals[src]).copy()   # (12, T)
        y = self.labels_np[i]
        if self.augment is not None:
            sig = self.augment(sig)
        return (torch.from_numpy(sig).float(),
                torch.from_numpy(y).float(),
                int(self.meta.iloc[i]["strat_fold"]))

    @property
    def num_classes(self) -> int:
        return len(self.label_columns)

    def positive_counts(self) -> np.ndarray:
        return self.labels_np.sum(axis=0)
