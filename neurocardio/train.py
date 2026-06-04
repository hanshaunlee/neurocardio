"""
Training loop for NeuroCardio.

- Multi-label BCEWithLogits with optional pos_weight (inverse-prevalence).
- Cosine LR schedule.
- EMA of weights for evaluation (Polyak averaging).
- Periodic checkpointing of (best AUROC) and (most recent).
- JSON history with per-epoch metrics for plotting.
- Designed to be safely resumable from a Modal Volume checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from neurocardio.augment import standard_train_augment
from neurocardio.dataset import PTBXLCache
from neurocardio.factory import make_model


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----- config -----------------------------------------------------------
@dataclass
class TrainConfig:
    cache_dir: str = "data/ptbxl_cache"
    output_dir: str = "runs/local"
    label_set: str = "diagnostic_superclass"
    model_name: str = "snn"   # snn | cnn | resnet
    sampling_rate: int = 100
    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple = (9,)
    test_folds: tuple = (10,)
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    warmup_epochs: int = 3
    pos_weight_mode: str = "sqrt"      # "none" | "sqrt" | "inv"
    pos_weight_cap: float = 8.0
    theta: float = 0.15
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 0
    save_every: int = 1                # save every N epochs (latest)
    eval_use_ema: bool = True
    device: str = "auto"
    log_every: int = 50
    # 14-hour wallclock budget for the Modal run; falls back to `epochs`.
    wallclock_hours: float = 0.0
    # Cap steps per epoch (0 = unlimited) — useful for smoke tests.
    max_steps_per_epoch: int = 0
    # Resume from a saved (model+opt+ema+epoch) bundle if it exists in output_dir.
    resume: bool = True


# ----- utilities --------------------------------------------------------
def device_of(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if cfg.device == "mps" or (cfg.device == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def pos_weight(counts: np.ndarray, n: int, mode: str, cap: float) -> torch.Tensor:
    neg = n - counts
    if mode == "none":
        w = np.ones_like(counts)
    elif mode == "inv":
        w = neg / np.maximum(counts, 1.0)
    elif mode == "sqrt":
        w = np.sqrt(neg / np.maximum(counts, 1.0))
    else:
        raise ValueError(mode)
    return torch.from_numpy(np.minimum(w, cap)).float()


class EMA:
    """Polyak averaging of model parameters.

    BatchNorm running statistics are excluded from EMA (averaging two already-
    averaged stats is unsound). They are copied from the live model directly,
    which is the convention used by timm and most other EMA implementations.
    """
    BN_KEYS = ("running_mean", "running_var", "num_batches_tracked")

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if any(k.endswith(suffix) for suffix in self.BN_KEYS):
                self.shadow[k].copy_(v.detach())
            elif v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_to(self, model: nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict):
        model.load_state_dict(backup, strict=True)


# ----- metrics ----------------------------------------------------------
def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> dict:
    yt = y_true.astype(np.int64)
    per = {}
    aurocs, auprcs = [], []
    for j, c in enumerate(labels):
        col = yt[:, j]
        if col.sum() == 0 or col.sum() == len(col):
            per[c] = {"auroc": float("nan"), "auprc": float("nan"),
                       "support": int(col.sum())}
            continue
        a = roc_auc_score(col, y_score[:, j])
        p = average_precision_score(col, y_score[:, j])
        per[c] = {"auroc": float(a), "auprc": float(p),
                   "support": int(col.sum())}
        aurocs.append(a); auprcs.append(p)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "per_label": per,
    }


# ----- core -------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device,
             labels: list[str]) -> dict:
    model.eval()
    all_y, all_p = [], []
    for x, y, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        logits = model(x)
        all_y.append(y.cpu().numpy())
        all_p.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(all_y, axis=0)
    y_score = np.concatenate(all_p, axis=0)
    return multilabel_metrics(y_true, y_score, labels)


def save_ckpt(path: Path, *, model: nn.Module, ema: EMA, opt, sched, epoch: int,
              cfg: TrainConfig, best_macro_auroc: float, history: list,
              label_columns: list[str]):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "ema": ema.shadow,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "cfg": asdict(cfg),
        "label_columns": label_columns,
        "best_macro_auroc": best_macro_auroc,
        "history": history,
    }, path)


def load_ckpt_if_any(path: Path, *, model: nn.Module, ema: EMA, opt, sched) -> dict:
    if not path.exists():
        return {"epoch_start": 0, "best_macro_auroc": -1.0, "history": []}
    # Load directly on the model's device — otherwise EMA shadow + optimizer
    # moments end up on CPU and crash the first ema.update() / opt.step().
    dev = next(model.parameters()).device
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ema.shadow = {k: v.to(dev) if torch.is_tensor(v) else v
                   for k, v in ckpt["ema"].items()}
    opt.load_state_dict(ckpt["opt"])
    # opt.load_state_dict respects the device of params now, but state may
    # still hold cpu tensors from older checkpoints — move them explicitly.
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(dev)
    sched.load_state_dict(ckpt["sched"])
    return {
        "epoch_start": int(ckpt["epoch"]) + 1,
        "best_macro_auroc": float(ckpt["best_macro_auroc"]),
        "history": list(ckpt.get("history", [])),
    }


def main(cfg: TrainConfig):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    seed_all(cfg.seed)
    dev = device_of(cfg)
    print(f"device: {dev}")

    # ----- data -----
    train_ds = PTBXLCache(cfg.cache_dir, fold_subset=list(cfg.train_folds),
                          sampling_rate=cfg.sampling_rate,
                          augment=standard_train_augment(fs=cfg.sampling_rate))
    val_ds   = PTBXLCache(cfg.cache_dir, fold_subset=list(cfg.val_folds),
                          sampling_rate=cfg.sampling_rate, augment=None)
    test_ds  = PTBXLCache(cfg.cache_dir, fold_subset=list(cfg.test_folds),
                          sampling_rate=cfg.sampling_rate, augment=None)
    labels = train_ds.label_columns
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"labels ({len(labels)}): {labels}")
    print(f"positive counts (train): {dict(zip(labels, train_ds.positive_counts().astype(int).tolist()))}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True,
                              pin_memory=(dev.type == "cuda"),
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=(dev.type == "cuda"),
                            persistent_workers=cfg.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers,
                             pin_memory=(dev.type == "cuda"),
                             persistent_workers=cfg.num_workers > 0)

    # ----- model -----
    model = make_model(cfg.model_name, num_classes=len(labels)).to(dev)
    print(f"model: {cfg.model_name}  params: {count_parameters(model):,}")
    ema = EMA(model, decay=cfg.ema_decay)

    pw = pos_weight(train_ds.positive_counts(), len(train_ds),
                    cfg.pos_weight_mode, cfg.pos_weight_cap).to(dev)
    print(f"pos_weight ({cfg.pos_weight_mode}, cap={cfg.pos_weight_cap}): "
          f"{dict(zip(labels, pw.cpu().tolist()))}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    state = (load_ckpt_if_any(out_dir / "latest.pt",
                              model=model, ema=ema, opt=opt, sched=sched)
             if cfg.resume else {"epoch_start": 0, "best_macro_auroc": -1.0,
                                  "history": []})
    epoch_start = state["epoch_start"]
    best = state["best_macro_auroc"]
    history = state["history"]
    if epoch_start > 0:
        print(f"resumed from epoch {epoch_start}, best macro-AUROC {best:.4f}")

    t_run_start = time.time()
    wallclock_seconds = cfg.wallclock_hours * 3600.0 if cfg.wallclock_hours > 0 else float("inf")
    print(f"wallclock budget: {wallclock_seconds:.0f}s "
          f"(unbounded if 0)")

    # ----- training loop -----
    for epoch in range(epoch_start, cfg.epochs):
        elapsed = time.time() - t_run_start
        if elapsed > wallclock_seconds:
            print(f"wallclock budget exhausted at epoch {epoch} "
                  f"({elapsed:.0f}s ≥ {wallclock_seconds:.0f}s) — stopping.")
            break

        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        max_steps = cfg.max_steps_per_epoch or len(train_loader)
        for step, (x, y, _) in enumerate(train_loader):
            if step >= max_steps:
                break
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            if not torch.isfinite(loss):
                print(f"  skipping non-finite loss at ep {epoch} step {step}",
                      flush=True)
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            loss_sum += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
            if step % cfg.log_every == 0:
                lr_now = opt.param_groups[0]["lr"]
                print(f"  ep {epoch:03d} step {step:04d}/{max_steps}  "
                      f"loss {loss.item():.4f}  lr {lr_now:.2e}", flush=True)
        sched.step()
        train_loss = loss_sum / max(n_seen, 1)

        # ---- evaluate (optionally on EMA weights) ----
        if cfg.eval_use_ema:
            backup = ema.apply_to(model)
        val = evaluate(model, val_loader, dev, labels)
        if cfg.eval_use_ema:
            EMA.restore(model, backup)

        dt = time.time() - t0
        per_str = " ".join(f"{c}:{val['per_label'][c]['auroc']:.3f}" for c in labels)
        print(f"ep {epoch:03d}  loss {train_loss:.4f}  val_macro_auroc {val['macro_auroc']:.4f}  "
              f"val_macro_auprc {val['macro_auprc']:.4f}  per[{per_str}]  ({dt:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_auroc": val["macro_auroc"],
            "val_macro_auprc": val["macro_auprc"],
            "val_per_label": val["per_label"],
            "lr": float(opt.param_groups[0]["lr"]),
            "wallclock_seconds": time.time() - t_run_start,
        })
        with open(out_dir / "history.json", "w") as f:
            json.dump({"history": history, "best_macro_auroc": best,
                       "config": asdict(cfg)}, f, indent=2)

        # latest checkpoint (resume-safe)
        save_ckpt(out_dir / "latest.pt", model=model, ema=ema, opt=opt,
                  sched=sched, epoch=epoch, cfg=cfg,
                  best_macro_auroc=best, history=history,
                  label_columns=labels)
        # best checkpoint (eval-only)
        if val["macro_auroc"] > best:
            best = val["macro_auroc"]
            save_ckpt(out_dir / "best.pt", model=model, ema=ema, opt=opt,
                      sched=sched, epoch=epoch, cfg=cfg,
                      best_macro_auroc=best, history=history,
                      label_columns=labels)
            print(f"  ★ new best macro-AUROC: {best:.4f}")

    # ----- final test on test fold -----
    print("\n=== TEST (fold 10) ===")
    # Load best model
    best_path = out_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=dev, weights_only=False)
        if cfg.eval_use_ema:
            model.load_state_dict(ckpt["ema"])
        else:
            model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, dev, labels)
    print(f"test macro-AUROC: {test['macro_auroc']:.4f}  "
          f"macro-AUPRC: {test['macro_auprc']:.4f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/ptbxl_cache")
    p.add_argument("--output-dir", default="runs/local")
    p.add_argument("--label-set", default="diagnostic_superclass")
    p.add_argument("--model", default="snn", choices=["snn", "cnn", "resnet"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wallclock-hours", type=float, default=0.0)
    p.add_argument("--max-steps-per-epoch", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        label_set=args.label_set,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        wallclock_hours=args.wallclock_hours,
        max_steps_per_epoch=args.max_steps_per_epoch,
        resume=not args.no_resume,
        device=args.device,
    )
    main(cfg)
