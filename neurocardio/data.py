"""
PTB-XL data loading.

PTB-XL is the gold-standard benchmark for ECG deep learning: 21,837 12-lead
recordings of 10 s at 100 Hz (and 500 Hz), each annotated with one or more
SCP-ECG diagnostic statements that were either machine-generated and manually
validated by a cardiologist (~70%) or written from scratch by a cardiologist.

Wagner P. et al. PTB-XL, a large publicly available electrocardiography
dataset. Sci. Data 7, 154 (2020). https://doi.org/10.13026/x4td-x982

The recommended evaluation protocol stratifies recordings into 10 folds. We
follow the literature convention: folds 1-8 for training, fold 9 for
validation, fold 10 for test. This is the same split used by Strodthoff et al.
(2020) and most subsequent PTB-XL papers, making our numbers directly
comparable to the rest of the field.

Two label spaces are supported:
- `diagnostic_superclass`: 5 labels {NORM, MI, STTC, CD, HYP} — the standard
  PTB-XL benchmark, with macro-AUROC scored across the 5 disease categories.
- `diagnostic_subclass`: 24 labels — finer-grained diagnostic groups.

Each recording can carry multiple labels (multi-label, not multi-class).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import wfdb

# ----- constants --------------------------------------------------------
FS = 100  # we use the 100 Hz version (1.7 GB total)
WIN_LEN = 1000  # 10 seconds at 100 Hz

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}


# ----- preprocessing ---------------------------------------------------
@dataclass
class PTBXLConfig:
    root: Path
    label_set: str = "diagnostic_superclass"  # or "diagnostic_subclass"
    sampling_rate: int = 100
    win_len: int = WIN_LEN

    @property
    def metadata_csv(self) -> Path:
        return self.root / "ptbxl_database.csv"

    @property
    def scp_csv(self) -> Path:
        return self.root / "scp_statements.csv"

    @property
    def signal_root(self) -> Path:
        return self.root / ("records100" if self.sampling_rate == 100 else "records500")


def load_metadata(cfg: PTBXLConfig) -> tuple[pd.DataFrame, list[str]]:
    """Return (df, label_columns). df has one row per recording with one binary
    column per label and `strat_fold` for the split."""
    df = pd.read_csv(cfg.metadata_csv, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(cfg.scp_csv, index_col=0)

    if cfg.label_set == "diagnostic_superclass":
        scp = scp[scp.diagnostic == 1]
        mapping = scp.diagnostic_class.to_dict()
        labels = SUPERCLASSES
    elif cfg.label_set == "diagnostic_subclass":
        scp = scp[scp.diagnostic == 1]
        mapping = scp.diagnostic_subclass.to_dict()
        labels = sorted(scp.diagnostic_subclass.dropna().unique().tolist())
    else:
        raise ValueError(cfg.label_set)

    def project(codes: dict) -> set[str]:
        out = set()
        for k in codes:
            if k in mapping and isinstance(mapping[k], str):
                out.add(mapping[k])
        return out

    df["labels"] = df.scp_codes.apply(project)
    for c in labels:
        df[c] = df.labels.apply(lambda s: int(c in s))

    # Drop recordings with no diagnostic label at all (rare).
    df = df[df[labels].sum(axis=1) > 0].copy()
    return df, labels


def read_signal(cfg: PTBXLConfig, row: pd.Series) -> np.ndarray:
    """Returns float32 array of shape (12, win_len)."""
    rel = row["filename_lr"] if cfg.sampling_rate == 100 else row["filename_hr"]
    path = cfg.root / rel
    rec = wfdb.rdrecord(str(path))
    sig = rec.p_signal.astype(np.float32).T  # (C, T)
    # PTB-XL recordings are already 10 s at the listed rate, but we still pad/
    # truncate defensively to win_len in case of edge cases.
    C, T = sig.shape
    if T < cfg.win_len:
        sig = np.concatenate(
            [sig, np.zeros((C, cfg.win_len - T), dtype=np.float32)], axis=1)
    else:
        sig = sig[:, : cfg.win_len]
    return sig


def normalize(sig: np.ndarray) -> np.ndarray:
    """Per-lead z-score (robust: median / 1.4826*MAD)."""
    med = np.median(sig, axis=1, keepdims=True)
    mad = np.median(np.abs(sig - med), axis=1, keepdims=True) + 1e-6
    return (sig - med) / (1.4826 * mad)


def build_cache(cfg: PTBXLConfig, out_dir: Path, indices: Iterable[int] | None = None,
                verbose: bool = True) -> None:
    """
    Read all (or a subset of) PTB-XL recordings, normalize per-lead, stack
    into a single memory-mapped .npy plus a labels.csv. This is the format
    the training loop consumes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, labels = load_metadata(cfg)
    if indices is not None:
        df = df.loc[df.index.intersection(list(indices))]

    n = len(df)
    if verbose:
        print(f"Building cache for {n} recordings → {out_dir}")
    sig_path = out_dir / f"signals_{cfg.sampling_rate}.npy"
    sig = np.lib.format.open_memmap(
        sig_path, mode="w+", dtype=np.float32,
        shape=(n, 12, cfg.win_len))

    ecg_ids = []
    folds = []
    label_matrix = np.zeros((n, len(labels)), dtype=np.float32)
    for i, (ecg_id, row) in enumerate(df.iterrows()):
        raw = read_signal(cfg, row)
        sig[i] = normalize(raw)
        ecg_ids.append(int(ecg_id))
        folds.append(int(row["strat_fold"]))
        for j, c in enumerate(labels):
            label_matrix[i, j] = float(row[c])
        if verbose and (i + 1) % 1000 == 0:
            print(f"  cached {i+1}/{n}")
    sig.flush()

    meta = pd.DataFrame({
        "ecg_id": ecg_ids,
        "strat_fold": folds,
    })
    for j, c in enumerate(labels):
        meta[c] = label_matrix[:, j].astype(int)
    meta.to_csv(out_dir / "labels.csv", index=False)
    with open(out_dir / "label_columns.txt", "w") as f:
        f.write("\n".join(labels))
    if verbose:
        print(f"Done. label counts:")
        for c in labels:
            print(f"  {c:6s}  {int(meta[c].sum()):5d}")
